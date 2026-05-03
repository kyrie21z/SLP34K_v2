#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tools.mdiff_corrector_utils import (
    align_pred_gt,
    build_confusion_knowledge,
    load_confusion_table,
    load_feature_shards,
    load_manifest,
    load_pair_thresholds,
    record_arrays,
    token_id_to_char,
    token_ids_to_text,
)


DEPLOYABLE_POLICIES = {"low_conf", "confusion_pair", "low_conf_or_confusion"}
ORACLE_POLICIES = {"oracle_replace", "all_replace", "topk_oracle"}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def format_pair(pred_id: int, gt_id: int, tokenizer) -> str:
    return f"{token_id_to_char(int(pred_id), tokenizer)}->{token_id_to_char(int(gt_id), tokenizer)}"


def compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(payload)
    if "results" in compact:
        compact["results"] = [
            {
                key: row[key]
                for key in [
                    "policy",
                    "tau_low",
                    "tau_corr",
                    "tau_keep",
                    "delta_gain",
                    "correction_rate",
                    "preservation_rate",
                    "harmful_change_rate",
                    "corrected_accuracy",
                    "changed_token_count",
                    "selected_positions",
                ]
                if key in row
            }
            for row in compact["results"][:10]
        ]
    for key in ["pair_stats", "subset_stats", "position_diagnostics", "per_sample_results", "case_studies"]:
        if key in compact:
            compact[key] = f"<{key}:{len(compact[key])}>"
    return compact


def make_decoder_hidden_input(sample: Dict[str, Any], contract_type: str) -> torch.Tensor:
    decoder_hidden = sample["decoder_hidden"].unsqueeze(0)
    if contract_type == "token_only":
        return torch.zeros_like(decoder_hidden)
    return decoder_hidden


def run_model_outputs(
    model,
    sample: Dict[str, Any],
    device: torch.device,
    selected_mask_cpu: torch.Tensor,
    topk_size: int,
) -> Dict[str, torch.Tensor]:
    pred_ids = sample["pred_token_ids"].to(device).unsqueeze(0)
    pred_conf = sample["pred_token_conf"].to(device).unsqueeze(0)
    decoder_hidden = make_decoder_hidden_input(sample, getattr(model, "contract_type", "token_decoder_hidden")).to(device)
    selected_mask = selected_mask_cpu.to(device).unsqueeze(0)
    with torch.inference_mode():
        logits = model(
            pred_token_ids=pred_ids,
            pred_token_conf=pred_conf,
            correction_mask=selected_mask,
            decoder_hidden=decoder_hidden,
        )
        probs = logits.softmax(-1)
        corr_conf, corr_id = probs.max(-1)
        topk_conf, topk_ids = probs.topk(k=min(topk_size, probs.shape[-1]), dim=-1)
    return {
        "selected_mask": selected_mask_cpu.clone(),
        "corr_conf": corr_conf[0].detach().cpu(),
        "corr_id": corr_id[0].detach().cpu(),
        "topk_ids": topk_ids[0].detach().cpu(),
        "topk_conf": topk_conf[0].detach().cpu(),
    }


def build_deployable_confusion_mask(
    pred_ids_cpu: torch.Tensor,
    valid_mask_cpu: torch.Tensor,
    probe_topk_ids_cpu: torch.Tensor,
    confusion_info: Dict[str, Any],
    confusion_topk: int,
) -> torch.Tensor:
    mask = torch.zeros_like(valid_mask_cpu, dtype=torch.bool)
    if not confusion_info:
        return mask
    pair_ids = confusion_info["pair_ids"]
    pred_ids_known = confusion_info["pred_ids"]
    topk_limit = min(confusion_topk, probe_topk_ids_cpu.shape[-1])
    for pos in valid_mask_cpu.nonzero(as_tuple=False).flatten().tolist():
        pred_id = int(pred_ids_cpu[pos].item())
        if pred_id in pred_ids_known:
            mask[pos] = True
            continue
        for cand_id in probe_topk_ids_cpu[pos, :topk_limit].tolist():
            if (pred_id, int(cand_id)) in pair_ids:
                mask[pos] = True
                break
    return mask


def build_position_maps(
    alignment: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    replace_pos_to_gt = {}
    replace_pos_to_pair = {}
    correct_pos_to_gt = {}
    for step in alignment["steps"]:
        if step["op"] == "replace":
            pred_pos = int(step["pred_pos"])
            gt_id = int(step["gt_id"])
            pred_id = int(step["pred_id"])
            replace_pos_to_gt[pred_pos] = gt_id
            replace_pos_to_pair[pred_pos] = format_pair(pred_id, gt_id, tokenizer)
        elif step["op"] == "correct":
            pred_pos = int(step["pred_pos"])
            correct_pos_to_gt[pred_pos] = int(step["gt_id"])
    return {
        "replace_pos_to_gt": replace_pos_to_gt,
        "replace_pos_to_pair": replace_pos_to_pair,
        "correct_pos_to_gt": correct_pos_to_gt,
    }


def prepare_samples(
    model,
    records: List[Dict[str, Any]],
    shards: Dict[str, Dict[str, Any]],
    device: torch.device,
    confusion_info: Dict[str, Any],
    confusion_topk: int,
    oracle_topk: int,
) -> List[Dict[str, Any]]:
    samples = []
    probe_topk_size = max(5, oracle_topk, confusion_topk)
    for record in records:
        arrays = record_arrays(record, shards)
        pred_token_ids = torch.as_tensor(arrays["pred_token_ids"], dtype=torch.long)
        pred_token_conf = torch.as_tensor(arrays["pred_token_conf"], dtype=torch.float32)
        gt_token_ids = torch.as_tensor(arrays["gt_token_ids"], dtype=torch.long)
        decoder_hidden = torch.as_tensor(arrays["decoder_hidden"], dtype=torch.float32)
        valid_mask = pred_token_ids.ne(model.pad_id) & pred_token_ids.ne(model.eos_id)
        low_conf_base = pred_token_conf.clone()
        alignment = align_pred_gt(
            pred_token_ids.tolist(),
            gt_token_ids.tolist(),
            eos_id=model.eos_id,
            pad_id=model.pad_id,
        )
        position_maps = build_position_maps(alignment, model.tokenizer)
        oracle_replace_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        oracle_confusion_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for pred_pos, gt_id in position_maps["replace_pos_to_gt"].items():
            oracle_replace_mask[pred_pos] = True
            if confusion_info and (int(pred_token_ids[pred_pos].item()), int(gt_id)) in confusion_info["pair_ids"]:
                oracle_confusion_mask[pred_pos] = True
        all_valid_mask = valid_mask.clone()
        probe_outputs = run_model_outputs(
            model=model,
            sample={
                "pred_token_ids": pred_token_ids,
                "pred_token_conf": pred_token_conf,
                "decoder_hidden": decoder_hidden,
            },
            device=device,
            selected_mask_cpu=all_valid_mask,
            topk_size=probe_topk_size,
        )
        deployable_confusion_mask = build_deployable_confusion_mask(
            pred_ids_cpu=pred_token_ids,
            valid_mask_cpu=valid_mask,
            probe_topk_ids_cpu=probe_outputs["topk_ids"],
            confusion_info=confusion_info,
            confusion_topk=confusion_topk,
        )
        topk_oracle_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for pred_pos, gt_id in position_maps["replace_pos_to_gt"].items():
            if bool((probe_outputs["topk_ids"][pred_pos, : min(oracle_topk, probe_outputs["topk_ids"].shape[-1])] == gt_id).any().item()):
                topk_oracle_mask[pred_pos] = True
        samples.append(
            {
                "sample_id": record["sample_id"],
                "subset": record.get("subset"),
                "gt_text": record["gt_text"],
                "baseline_pred": record["pred_text"],
                "metadata": record.get("metadata"),
                "low_conf_sample": bool(record.get("low_conf_sample", False)),
                "pred_token_ids": pred_token_ids,
                "pred_token_conf": pred_token_conf,
                "gt_token_ids": gt_token_ids,
                "decoder_hidden": decoder_hidden,
                "valid_mask": valid_mask,
                "low_conf_base": low_conf_base,
                "alignment": alignment,
                "replace_pos_to_gt": position_maps["replace_pos_to_gt"],
                "replace_pos_to_pair": position_maps["replace_pos_to_pair"],
                "correct_pos_to_gt": position_maps["correct_pos_to_gt"],
                "oracle_replace_mask": oracle_replace_mask,
                "oracle_confusion_mask": oracle_confusion_mask,
                "deployable_confusion_mask": deployable_confusion_mask,
                "topk_oracle_mask": topk_oracle_mask,
                "probe_outputs": probe_outputs,
                "mask_cache": {},
                "final_output_cache": {},
            }
        )
    return samples


def resolve_selected_mask(sample: Dict[str, Any], policy: str, tau_low: float) -> torch.Tensor:
    cache_key = f"{policy}:{tau_low:.4f}"
    if cache_key in sample["mask_cache"]:
        return sample["mask_cache"][cache_key]
    low_conf_mask = sample["valid_mask"] & sample["low_conf_base"].lt(tau_low)
    if policy == "low_conf":
        mask = low_conf_mask
    elif policy == "confusion_pair":
        mask = sample["deployable_confusion_mask"]
    elif policy == "low_conf_or_confusion":
        mask = low_conf_mask | sample["deployable_confusion_mask"]
    elif policy in {"oracle_replace", "all_replace"}:
        mask = sample["oracle_replace_mask"]
    elif policy == "topk_oracle":
        mask = sample["topk_oracle_mask"]
    else:
        raise ValueError(f"Unsupported selected-mask policy: {policy}")
    sample["mask_cache"][cache_key] = mask.clone()
    return mask


def get_final_outputs(
    model,
    sample: Dict[str, Any],
    device: torch.device,
    policy: str,
    tau_low: float,
) -> Dict[str, torch.Tensor]:
    cache_key = f"{policy}:{tau_low:.4f}"
    if cache_key not in sample["final_output_cache"]:
        selected_mask = resolve_selected_mask(sample, policy, tau_low)
        sample["final_output_cache"][cache_key] = run_model_outputs(
            model=model,
            sample=sample,
            device=device,
            selected_mask_cpu=selected_mask,
            topk_size=5,
        )
    return sample["final_output_cache"][cache_key]


def resolve_base_thresholds(
    model,
    args: argparse.Namespace,
    pair_threshold_config: Dict[str, Any],
) -> Dict[str, float]:
    file_defaults = pair_threshold_config.get("default_thresholds", {})
    tau_low = getattr(model, "tau_low", 0.70) if args.tau_low is None else args.tau_low
    tau_corr = file_defaults.get("tau_corr", getattr(model, "tau_corr", 0.80)) if args.tau_corr is None else args.tau_corr
    tau_keep = file_defaults.get("tau_keep", getattr(model, "tau_keep", 0.90)) if args.tau_keep is None else args.tau_keep
    delta_gain = file_defaults.get("delta_gain", getattr(model, "delta_gain", 0.05)) if args.delta_gain is None else args.delta_gain
    return {
        "tau_low": float(tau_low),
        "tau_corr": float(tau_corr),
        "tau_keep": float(tau_keep),
        "delta_gain": float(delta_gain),
    }


def apply_gating(
    sample: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    tokenizer,
    tau_corr: float,
    tau_keep: float,
    delta_gain: float,
    pair_threshold_config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    pred_ids = sample["pred_token_ids"]
    pred_conf = sample["pred_token_conf"]
    valid_mask = sample["valid_mask"]
    selected_mask = outputs["selected_mask"]
    corr_id = outputs["corr_id"]
    corr_conf = outputs["corr_conf"]
    valid_change_positions = selected_mask & valid_mask
    tau_corr_used = torch.full(pred_conf.shape, float(tau_corr), dtype=torch.float32)
    tau_keep_used = torch.full(pred_conf.shape, float(tau_keep), dtype=torch.float32)
    delta_gain_used = torch.full(pred_conf.shape, float(delta_gain), dtype=torch.float32)
    for pos in valid_change_positions.nonzero(as_tuple=False).flatten().tolist():
        pair_key = format_pair(int(pred_ids[pos].item()), int(corr_id[pos].item()), tokenizer)
        override = pair_threshold_config["pairs"].get(pair_key)
        if override:
            tau_corr_used[pos] = float(override.get("tau_corr", tau_corr))
            tau_keep_used[pos] = float(override.get("tau_keep", tau_keep))
            delta_gain_used[pos] = float(override.get("delta_gain", delta_gain))
    keep_mask = ~valid_change_positions
    keep_mask = keep_mask | corr_id.eq(pred_ids)
    keep_mask = keep_mask | corr_conf.lt(tau_corr_used)
    keep_mask = keep_mask | (pred_conf.ge(tau_keep_used) & (corr_conf - pred_conf).lt(delta_gain_used))
    changed_mask = ~keep_mask
    corrected_ids = pred_ids.clone()
    corrected_conf = pred_conf.clone()
    corrected_ids[changed_mask] = corr_id[changed_mask]
    corrected_conf[changed_mask] = corr_conf[changed_mask]
    return {
        "selected_mask": selected_mask,
        "corr_id": corr_id,
        "corr_conf": corr_conf,
        "topk_ids": outputs["topk_ids"],
        "topk_conf": outputs["topk_conf"],
        "tau_corr_used": tau_corr_used,
        "tau_keep_used": tau_keep_used,
        "delta_gain_used": delta_gain_used,
        "keep_mask": keep_mask,
        "changed_mask": changed_mask,
        "corrected_ids": corrected_ids,
        "corrected_conf": corrected_conf,
    }


def topk_tokens_as_text(topk_ids: torch.Tensor, tokenizer, k: int) -> str:
    return ",".join(token_id_to_char(int(token_id), tokenizer) or "?" for token_id in topk_ids[:k].tolist())


def evaluate_policy(
    model,
    samples: List[Dict[str, Any]],
    device: torch.device,
    policy: str,
    tau_low: float,
    tau_corr: float,
    tau_keep: float,
    delta_gain: float,
    pair_threshold_config: Dict[str, Any],
    pair_stats: bool,
    subset_stats: bool,
    include_case_studies: bool,
    include_position_diagnostics: bool,
    max_case_studies: int,
) -> Dict[str, Any]:
    baseline_correct_samples = 0
    corrected_correct_samples = 0
    originally_correct_samples = 0
    originally_correct_samples_broken = 0
    originally_wrong_positions = 0
    corrected_wrong_positions = 0
    corrected_to_gt_positions = 0
    originally_correct_positions = 0
    preserved_positions = 0
    oracle1_hits = 0
    oracle3_hits = 0
    oracle5_hits = 0
    selected_positions = 0
    selected_wrong_positions = 0
    selected_correct_positions = 0
    selected_oracle1_positions = 0
    changed_token_count = 0
    changed_correct_positions = 0
    changed_wrong_positions = 0
    changed_sample_count = 0
    case_studies = []
    position_diagnostics = []
    per_sample_results = []
    pair_rows = defaultdict(lambda: {"support": 0, "default_corrected": 0, "corrected": 0, "oracle1": 0, "oracle5": 0})
    subset_rows = defaultdict(
        lambda: {
            "sample_count": 0,
            "baseline_correct_samples": 0,
            "corrected_correct_samples": 0,
            "originally_correct_samples": 0,
            "originally_correct_samples_broken": 0,
            "originally_wrong_positions": 0,
            "corrected_to_gt_positions": 0,
            "originally_correct_positions": 0,
            "preserved_positions": 0,
            "oracle1_hits": 0,
            "oracle5_hits": 0,
            "changed_token_count": 0,
        }
    )

    for sample in samples:
        outputs = get_final_outputs(model, sample, device, policy, tau_low)
        gated = apply_gating(
            sample=sample,
            outputs=outputs,
            tokenizer=model.tokenizer,
            tau_corr=tau_corr,
            tau_keep=tau_keep,
            delta_gain=delta_gain,
            pair_threshold_config=pair_threshold_config,
        )
        corrected_text = token_ids_to_text(gated["corrected_ids"].tolist(), model.tokenizer)
        metadata = sample.get("metadata") or {}
        max_seq_len = max(sample["alignment"]["gt_length"], sample["alignment"]["pred_length"])
        subset_keys = [
            ("subset", sample.get("subset", "unknown")),
            ("length_bucket", "long_21plus" if max_seq_len >= 21 else "short_lt21"),
            ("low_conf_sample", str(sample.get("low_conf_sample", False)).lower()),
            ("vocabulary_type", metadata.get("vocabulary_type", "missing")),
            ("quality", metadata.get("quality", "missing")),
            ("resolution_type", metadata.get("resolution_type", "missing")),
            ("structure_type", metadata.get("structure_type", "missing")),
        ]
        if subset_stats:
            for subset_name, subset_value in subset_keys:
                subset_rows[(subset_name, subset_value)]["sample_count"] += 1
        if sample["baseline_pred"] == sample["gt_text"]:
            baseline_correct_samples += 1
            originally_correct_samples += 1
            if corrected_text != sample["gt_text"]:
                originally_correct_samples_broken += 1
            if subset_stats:
                for subset_name, subset_value in subset_keys:
                    subset_rows[(subset_name, subset_value)]["baseline_correct_samples"] += 1
                    subset_rows[(subset_name, subset_value)]["originally_correct_samples"] += 1
                    if corrected_text != sample["gt_text"]:
                        subset_rows[(subset_name, subset_value)]["originally_correct_samples_broken"] += 1
        if corrected_text == sample["gt_text"]:
            corrected_correct_samples += 1
            if subset_stats:
                for subset_name, subset_value in subset_keys:
                    subset_rows[(subset_name, subset_value)]["corrected_correct_samples"] += 1

        changed_positions = gated["changed_mask"].nonzero(as_tuple=False).flatten().tolist()
        if changed_positions:
            changed_sample_count += 1
        changed_token_count += len(changed_positions)
        if subset_stats:
            for subset_name, subset_value in subset_keys:
                subset_rows[(subset_name, subset_value)]["changed_token_count"] += len(changed_positions)
        selected_positions += int(gated["selected_mask"].sum().item())

        selected_low_conf = sample["valid_mask"] & sample["low_conf_base"].lt(tau_low)
        selected_confusion = sample["deployable_confusion_mask"]
        selected_oracle_confusion = sample["oracle_confusion_mask"]

        sample_case_reasons = []
        sample_success_count = 0
        sample_harmful_count = 0
        sample_oracle_no_change_count = 0

        for pos, gt_id in sample["correct_pos_to_gt"].items():
            originally_correct_positions += 1
            if bool(gated["selected_mask"][pos].item()):
                selected_correct_positions += 1
            preserved = int(gated["corrected_ids"][pos].item() == gt_id)
            preserved_positions += preserved
            harmful = int(bool(gated["changed_mask"][pos].item()) and gated["corrected_ids"][pos].item() != gt_id)
            changed_correct_positions += harmful
            sample_harmful_count += harmful
            if subset_stats:
                for subset_name, subset_value in subset_keys:
                    subset_rows[(subset_name, subset_value)]["originally_correct_positions"] += 1
                    subset_rows[(subset_name, subset_value)]["preserved_positions"] += preserved
            if include_position_diagnostics:
                position_diagnostics.append(
                    {
                        "sample_id": sample["sample_id"],
                        "position": pos,
                        "gt_token": token_id_to_char(gt_id, model.tokenizer),
                        "base_token": token_id_to_char(int(sample["pred_token_ids"][pos].item()), model.tokenizer),
                        "base_conf": float(sample["pred_token_conf"][pos].item()),
                        "corr_top1": token_id_to_char(int(gated["corr_id"][pos].item()), model.tokenizer),
                        "corr_top1_conf": float(gated["corr_conf"][pos].item()),
                        "corr_top3": topk_tokens_as_text(gated["topk_ids"][pos], model.tokenizer, 3),
                        "corr_top5": topk_tokens_as_text(gated["topk_ids"][pos], model.tokenizer, 5),
                        "selected_by_low_conf": int(bool(selected_low_conf[pos].item())),
                        "selected_by_confusion": int(bool(selected_confusion[pos].item())),
                        "selected_by_oracle_confusion": int(bool(selected_oracle_confusion[pos].item())),
                        "selected_by_policy": int(bool(gated["selected_mask"][pos].item())),
                        "changed": int(bool(gated["changed_mask"][pos].item())),
                        "success": 0,
                        "harmful": harmful,
                        "pair": "",
                    }
                )

        for pos, gt_id in sample["replace_pos_to_gt"].items():
            originally_wrong_positions += 1
            if bool(gated["selected_mask"][pos].item()):
                selected_wrong_positions += 1
            pair_key = sample["replace_pos_to_pair"][pos]
            topk_ids = gated["topk_ids"][pos]
            oracle1 = int((topk_ids[:1] == gt_id).any().item())
            oracle3 = int((topk_ids[:3] == gt_id).any().item())
            oracle5 = int((topk_ids[:5] == gt_id).any().item())
            oracle1_hits += oracle1
            oracle3_hits += oracle3
            oracle5_hits += oracle5
            success = int(gated["corrected_ids"][pos].item() == gt_id)
            if subset_stats:
                for subset_name, subset_value in subset_keys:
                    subset_rows[(subset_name, subset_value)]["originally_wrong_positions"] += 1
                    subset_rows[(subset_name, subset_value)]["corrected_to_gt_positions"] += success
                    subset_rows[(subset_name, subset_value)]["oracle1_hits"] += oracle1
                    subset_rows[(subset_name, subset_value)]["oracle5_hits"] += oracle5
            if bool(gated["selected_mask"][pos].item()) and int(gated["corr_id"][pos].item()) == gt_id:
                selected_oracle1_positions += 1
            corrected_to_gt_positions += success
            corrected_wrong_positions += 1 - success
            changed_here = int(bool(gated["changed_mask"][pos].item()))
            if changed_here:
                changed_wrong_positions += 1
            if changed_here and success:
                sample_success_count += 1
            if oracle5 and not changed_here:
                sample_oracle_no_change_count += 1
            if pair_stats:
                pair_rows[pair_key]["support"] += 1
                pair_rows[pair_key]["corrected"] += success
                pair_rows[pair_key]["oracle1"] += oracle1
                pair_rows[pair_key]["oracle5"] += oracle5
            if include_position_diagnostics:
                position_diagnostics.append(
                    {
                        "sample_id": sample["sample_id"],
                        "position": pos,
                        "gt_token": token_id_to_char(gt_id, model.tokenizer),
                        "base_token": token_id_to_char(int(sample["pred_token_ids"][pos].item()), model.tokenizer),
                        "base_conf": float(sample["pred_token_conf"][pos].item()),
                        "corr_top1": token_id_to_char(int(gated["corr_id"][pos].item()), model.tokenizer),
                        "corr_top1_conf": float(gated["corr_conf"][pos].item()),
                        "corr_top3": topk_tokens_as_text(topk_ids, model.tokenizer, 3),
                        "corr_top5": topk_tokens_as_text(topk_ids, model.tokenizer, 5),
                        "selected_by_low_conf": int(bool(selected_low_conf[pos].item())),
                        "selected_by_confusion": int(bool(selected_confusion[pos].item())),
                        "selected_by_oracle_confusion": int(bool(selected_oracle_confusion[pos].item())),
                        "selected_by_policy": int(bool(gated["selected_mask"][pos].item())),
                        "changed": changed_here,
                        "success": success,
                        "harmful": 0,
                        "pair": pair_key,
                    }
                )

        if sample_success_count:
            sample_case_reasons.append("corrected_success")
        if sample_harmful_count:
            sample_case_reasons.append("harmful_change")
        if sample_oracle_no_change_count:
            sample_case_reasons.append("oracle_contains_gt_but_no_change")
        if include_case_studies and sample_case_reasons and len(case_studies) < max_case_studies:
            case_studies.append(
                {
                    "sample_id": sample["sample_id"],
                    "gt_text": sample["gt_text"],
                    "baseline_pred": sample["baseline_pred"],
                    "corrected_pred": corrected_text,
                    "changed_positions": changed_positions,
                    "replace_positions": sorted(sample["replace_pos_to_gt"])[:10],
                    "base_conf": [float(sample["pred_token_conf"][pos].item()) for pos in changed_positions[:10]],
                    "corr_conf": [float(gated["corr_conf"][pos].item()) for pos in changed_positions[:10]],
                    "selected_mask_policy": policy,
                    "reason": sample_case_reasons,
                }
            )
        if include_case_studies:
            per_sample_results.append(
                {
                    "sample_id": sample["sample_id"],
                    "gt_text": sample["gt_text"],
                    "baseline_pred": sample["baseline_pred"],
                    "corrected_pred": corrected_text,
                    "changed_positions": changed_positions,
                    "replace_positions": sorted(sample["replace_pos_to_gt"])[:10],
                    "selected_positions": gated["selected_mask"].nonzero(as_tuple=False).flatten().tolist(),
                    "alignment_replace_count": sample["alignment"]["replace_count"],
                    "alignment_correct_count": sample["alignment"]["correct_count"],
                    "base_conf": [float(x) for x in sample["pred_token_conf"].tolist()],
                    "corr_conf": [float(x) for x in gated["corr_conf"].tolist()],
                }
            )

    baseline_accuracy = safe_div(baseline_correct_samples, len(samples))
    corrected_accuracy = safe_div(corrected_correct_samples, len(samples))
    change_precision = safe_div(corrected_to_gt_positions, changed_token_count)
    summary: Dict[str, Any] = {
        "policy": policy,
        "policy_type": "deployable" if policy in DEPLOYABLE_POLICIES else "oracle_analysis",
        "tau_low": tau_low,
        "tau_corr": tau_corr,
        "tau_keep": tau_keep,
        "delta_gain": delta_gain,
        "num_samples": len(samples),
        "baseline_accuracy": baseline_accuracy,
        "corrected_accuracy": corrected_accuracy,
        "gain_accuracy": corrected_accuracy - baseline_accuracy,
        "correction_rate": safe_div(corrected_to_gt_positions, originally_wrong_positions),
        "preservation_rate": safe_div(preserved_positions, originally_correct_positions),
        "harmful_change_rate": safe_div(originally_correct_samples_broken, originally_correct_samples),
        "replace_error_reduction": safe_div(originally_wrong_positions - corrected_wrong_positions, originally_wrong_positions),
        "oracle@1": safe_div(oracle1_hits, originally_wrong_positions),
        "oracle@3": safe_div(oracle3_hits, originally_wrong_positions),
        "oracle@5": safe_div(oracle5_hits, originally_wrong_positions),
        "oracle_gap": safe_div(oracle1_hits, originally_wrong_positions) - safe_div(corrected_to_gt_positions, originally_wrong_positions),
        "gating_gap": selected_oracle1_positions - corrected_to_gt_positions,
        "changed_token_count": changed_token_count,
        "changed_sample_rate": safe_div(changed_sample_count, len(samples)),
        "selected_positions": selected_positions,
        "selected_wrong_coverage": safe_div(selected_wrong_positions, originally_wrong_positions),
        "selected_correct_coverage": safe_div(selected_correct_positions, originally_correct_positions),
        "change_precision": change_precision,
        "change_recall": safe_div(corrected_to_gt_positions, originally_wrong_positions),
        "originally_wrong_positions": originally_wrong_positions,
        "originally_correct_positions": originally_correct_positions,
        "contract_type": getattr(model, "contract_type", "token_decoder_hidden"),
    }
    if pair_stats:
        summary["pair_stats"] = [
            {
                "pair": pair,
                "support": stats["support"],
                "corrected": stats["corrected"],
                "correction_rate": safe_div(stats["corrected"], stats["support"]),
                "oracle@1": safe_div(stats["oracle1"], stats["support"]),
                "oracle@5": safe_div(stats["oracle5"], stats["support"]),
            }
            for pair, stats in sorted(pair_rows.items(), key=lambda item: (-item[1]["support"], item[0]))
        ]
    if subset_stats:
        summary["subset_stats"] = [
            {
                "subset_name": subset_name,
                "subset_value": subset_value,
                "sample_count": stats["sample_count"],
                "baseline_accuracy": safe_div(stats["baseline_correct_samples"], stats["sample_count"]),
                "corrected_accuracy": safe_div(stats["corrected_correct_samples"], stats["sample_count"]),
                "gain_accuracy": safe_div(stats["corrected_correct_samples"], stats["sample_count"]) - safe_div(stats["baseline_correct_samples"], stats["sample_count"]),
                "correction_rate": safe_div(stats["corrected_to_gt_positions"], stats["originally_wrong_positions"]),
                "preservation_rate": safe_div(stats["preserved_positions"], stats["originally_correct_positions"]),
                "harmful_change_rate": safe_div(stats["originally_correct_samples_broken"], stats["originally_correct_samples"]),
                "oracle@1": safe_div(stats["oracle1_hits"], stats["originally_wrong_positions"]),
                "oracle@5": safe_div(stats["oracle5_hits"], stats["originally_wrong_positions"]),
                "changed_token_count": stats["changed_token_count"],
                "originally_wrong_positions": stats["originally_wrong_positions"],
                "originally_correct_positions": stats["originally_correct_positions"],
            }
            for (subset_name, subset_value), stats in sorted(subset_rows.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
    if include_case_studies:
        summary["case_studies"] = case_studies
        summary["per_sample_results"] = per_sample_results
    if include_position_diagnostics:
        summary["position_diagnostics"] = position_diagnostics
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_auxiliary_outputs(output_json: Path, payload: Dict[str, Any]) -> None:
    output_dir = output_json.parent
    stem = output_json.stem
    if "position_diagnostics" in payload:
        diagnostics_path = output_dir / f"{stem}_position_diagnostics.csv"
        write_csv(diagnostics_path, payload["position_diagnostics"])
        write_csv(output_dir / "position_diagnostics.csv", payload["position_diagnostics"])
    if "case_studies" in payload:
        case_path = output_dir / f"{stem}_case_study.json"
        case_path.write_text(json.dumps(payload["case_studies"], ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "case_study.json").write_text(json.dumps(payload["case_studies"], ensure_ascii=False, indent=2), encoding="utf-8")
    if "pair_stats" in payload:
        write_csv(output_dir / f"{stem}_pair_stats.csv", payload["pair_stats"])
        (output_dir / f"{stem}_pair_stats.json").write_text(json.dumps(payload["pair_stats"], ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(output_dir / "pair_stats.csv", payload["pair_stats"])
        (output_dir / "pair_stats.json").write_text(json.dumps(payload["pair_stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    if "subset_stats" in payload:
        write_csv(output_dir / f"{stem}_subset_stats.csv", payload["subset_stats"])
        (output_dir / f"{stem}_subset_stats.json").write_text(json.dumps(payload["subset_stats"], ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(output_dir / "subset_stats.csv", payload["subset_stats"])
        (output_dir / "subset_stats.json").write_text(json.dumps(payload["subset_stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    if payload.get("sweep_thresholds"):
        rows = []
        for row in payload["results"]:
            rows.append(
                {
                    "policy": row["policy"],
                    "tau_low": row["tau_low"],
                    "tau_corr": row["tau_corr"],
                    "tau_keep": row["tau_keep"],
                    "delta_gain": row["delta_gain"],
                    "selected_positions": row["selected_positions"],
                    "changed_token_count": row["changed_token_count"],
                    "correction_rate": row["correction_rate"],
                    "preservation_rate": row["preservation_rate"],
                    "harmful_change_rate": row["harmful_change_rate"],
                    "replace_error_reduction": row["replace_error_reduction"],
                    "corrected_accuracy": row["corrected_accuracy"],
                    "gain_accuracy": row["gain_accuracy"],
                    "oracle@1": row["oracle@1"],
                    "oracle@3": row["oracle@3"],
                    "oracle@5": row["oracle@5"],
                }
            )
        write_csv(output_dir / "calibration_grid.csv", rows)
        eligible = [
            row
            for row in payload["results"]
            if row["harmful_change_rate"] <= payload["max_harmful_change_rate"]
            and row["preservation_rate"] >= payload["min_preservation_rate"]
        ]
        best = None
        if eligible:
            best = max(
                eligible,
                key=lambda row: (
                    row["correction_rate"],
                    row["corrected_accuracy"],
                    -row["changed_token_count"],
                ),
            )
        best_payload = {
            "filters": {
                "max_harmful_change_rate": payload["max_harmful_change_rate"],
                "min_preservation_rate": payload["min_preservation_rate"],
            },
            "best_result": best,
        }
        (output_dir / "calibration_best.json").write_text(
            json.dumps(best_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--corrector_ckpt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--tau_low", type=float, default=None)
    parser.add_argument("--tau_corr", type=float, default=None)
    parser.add_argument("--tau_keep", type=float, default=None)
    parser.add_argument("--delta_gain", type=float, default=None)
    parser.add_argument("--tau-low-list", default="")
    parser.add_argument("--tau-corr-list", default="")
    parser.add_argument("--tau-keep-list", default="")
    parser.add_argument("--delta-gain-list", default="")
    parser.add_argument("--sweep-thresholds", type=parse_bool, default=False)
    parser.add_argument("--selected-mask-policy", default="low_conf")
    parser.add_argument("--compare-mask-policies", default="")
    parser.add_argument("--confusion-table", default=None)
    parser.add_argument("--confusion-topk", type=int, default=20)
    parser.add_argument("--pair-thresholds", default=None)
    parser.add_argument("--pair-stats", type=parse_bool, default=False)
    parser.add_argument("--subset-stats", type=parse_bool, default=False)
    parser.add_argument("--save-case-study", type=parse_bool, default=False)
    parser.add_argument("--save-position-diagnostics", type=parse_bool, default=False)
    parser.add_argument("--oracle-topk", type=int, default=5)
    parser.add_argument("--max-harmful-change-rate", type=float, default=0.02)
    parser.add_argument("--min-preservation-rate", type=float, default=0.98)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max_case_studies", type=int, default=10)
    args = parser.parse_args()

    ckpt = torch.load(args.corrector_ckpt, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["model_config"])
    model = instantiate(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    confusion_rows = load_confusion_table(args.confusion_table) if args.confusion_table else []
    confusion_info = build_confusion_knowledge(confusion_rows) if confusion_rows else {}
    pair_threshold_config = load_pair_thresholds(args.pair_thresholds)
    base_thresholds = resolve_base_thresholds(model, args, pair_threshold_config)
    records = load_manifest(args.cache_dir)
    shards = load_feature_shards(args.cache_dir)
    samples = prepare_samples(
        model=model,
        records=records,
        shards=shards,
        device=device,
        confusion_info=confusion_info,
        confusion_topk=args.confusion_topk,
        oracle_topk=args.oracle_topk,
    )

    output_json = Path(args.output_json).absolute()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    policy_list = parse_str_list(args.compare_mask_policies)
    if policy_list:
        results = []
        for policy in policy_list:
            results.append(
                evaluate_policy(
                    model=model,
                    samples=samples,
                    device=device,
                    policy=policy,
                    tau_low=base_thresholds["tau_low"],
                    tau_corr=base_thresholds["tau_corr"],
                    tau_keep=base_thresholds["tau_keep"],
                    delta_gain=base_thresholds["delta_gain"],
                    pair_threshold_config=pair_threshold_config,
                    pair_stats=args.pair_stats,
                    subset_stats=False,
                    include_case_studies=False,
                    include_position_diagnostics=False,
                    max_case_studies=args.max_case_studies,
                )
            )
        payload: Dict[str, Any] = {
            "cache_dir": str(Path(args.cache_dir).absolute()),
            "corrector_ckpt": str(Path(args.corrector_ckpt).absolute()),
            "device": str(device),
            "compare_mask_policies": policy_list,
            "results": results,
        }
    elif args.sweep_thresholds:
        tau_low_list = parse_float_list(args.tau_low_list) or [base_thresholds["tau_low"]]
        tau_corr_list = parse_float_list(args.tau_corr_list) or [base_thresholds["tau_corr"]]
        tau_keep_list = parse_float_list(args.tau_keep_list) or [base_thresholds["tau_keep"]]
        delta_gain_list = parse_float_list(args.delta_gain_list) or [base_thresholds["delta_gain"]]
        results = []
        for tau_low in tau_low_list:
            for tau_corr in tau_corr_list:
                for tau_keep in tau_keep_list:
                    for delta_gain in delta_gain_list:
                        results.append(
                            evaluate_policy(
                                model=model,
                                samples=samples,
                                device=device,
                                policy=args.selected_mask_policy,
                                tau_low=tau_low,
                                tau_corr=tau_corr,
                                tau_keep=tau_keep,
                                delta_gain=delta_gain,
                                pair_threshold_config=pair_threshold_config,
                                pair_stats=False,
                                subset_stats=False,
                                include_case_studies=False,
                                include_position_diagnostics=False,
                                max_case_studies=args.max_case_studies,
                            )
                        )
        payload = {
            "cache_dir": str(Path(args.cache_dir).absolute()),
            "corrector_ckpt": str(Path(args.corrector_ckpt).absolute()),
            "device": str(device),
            "sweep_thresholds": True,
            "selected_mask_policy": args.selected_mask_policy,
            "max_harmful_change_rate": args.max_harmful_change_rate,
            "min_preservation_rate": args.min_preservation_rate,
            "results": results,
        }
    else:
        payload = evaluate_policy(
            model=model,
            samples=samples,
            device=device,
            policy=args.selected_mask_policy,
            tau_low=base_thresholds["tau_low"],
            tau_corr=base_thresholds["tau_corr"],
            tau_keep=base_thresholds["tau_keep"],
            delta_gain=base_thresholds["delta_gain"],
            pair_threshold_config=pair_threshold_config,
            pair_stats=args.pair_stats,
            subset_stats=args.subset_stats,
            include_case_studies=args.save_case_study,
            include_position_diagnostics=args.save_position_diagnostics,
            max_case_studies=args.max_case_studies,
        )
        payload["cache_dir"] = str(Path(args.cache_dir).absolute())
        payload["corrector_ckpt"] = str(Path(args.corrector_ckpt).absolute())
        payload["device"] = str(device)
        if args.confusion_table:
            payload["confusion_table"] = str(Path(args.confusion_table).absolute())
        if args.pair_thresholds:
            payload["pair_thresholds"] = str(Path(args.pair_thresholds).absolute())
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_auxiliary_outputs(output_json, payload)
    print(json.dumps(compact_payload(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
