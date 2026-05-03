#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset

from tools.mdiff_corrector_utils import (
    classify_token_id,
    load_confusion_table,
    load_pair_difficulty_table,
    load_feature_shards,
    load_manifest,
    record_arrays,
)


class CorrectorSmokeDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        shards: Dict[str, Dict[str, np.ndarray]],
        output_num_classes: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        contract_type: str,
        tokenizer,
        synthetic_mode: str,
        synthetic_ratio: float,
        synthetic_per_sample: int,
        max_noise_positions: int,
        segment_preserving: bool,
        confusion_table: List[Dict[str, object]],
        confusion_topk: int,
        pair_difficulty_rows: List[Dict[str, object]],
        pair_synthetic_mode: str,
        pair_synthetic_topk: int,
        pair_synthetic_multiplier: float,
        pair_min_count: int,
        pair_include_list: List[str],
        pair_exclude_list: List[str],
        pair_weight_alpha: float,
        pair_weight_min: float,
        pair_weight_max: float,
        noise_seed: int,
    ) -> None:
        self.samples = []
        self.output_num_classes = output_num_classes
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.contract_type = contract_type
        self.synthetic_mode = synthetic_mode
        self.synthetic_ratio = synthetic_ratio
        self.synthetic_per_sample = synthetic_per_sample
        self.max_noise_positions = max_noise_positions
        self.segment_preserving = segment_preserving
        self.random_state = random.Random(noise_seed)
        self.np_random = np.random.default_rng(noise_seed)
        self.synthetic_generated_count = 0
        self.synthetic_confusion_generated_count = 0
        self.synthetic_random_generated_count = 0
        self.synthetic_pair_specific_generated_count = 0
        self.synthetic_fallback_count = 0
        self.synthetic_skip_count = 0
        self.synthetic_target_count = 0
        self.tokenizer = tokenizer
        self.segment_token_ids = self._build_segment_token_ids()
        self.all_token_ids = list(range(1, self.output_num_classes))
        self.pair_synthetic_mode = pair_synthetic_mode
        self.pair_synthetic_multiplier = pair_synthetic_multiplier
        self.pair_min_count = pair_min_count
        self.pair_include_set = set(pair_include_list)
        self.pair_exclude_set = set(pair_exclude_list)
        self.pair_weight_alpha = pair_weight_alpha
        self.pair_weight_min = pair_weight_min
        self.pair_weight_max = pair_weight_max
        self.pair_difficulty_by_pair = self._build_pair_difficulty_lookup(pair_difficulty_rows)
        self.pair_weight_by_pair = self._build_pair_weight_map(pair_difficulty_rows)
        self.confusion_candidates = self._build_confusion_candidates(confusion_table, confusion_topk)
        self.pair_specific_candidates = self._build_pair_specific_candidates(pair_difficulty_rows, pair_synthetic_topk)
        self.real_records: List[Dict[str, Any]] = []
        for record in records:
            arrays = record_arrays(record, shards)
            gt_ids = arrays["gt_token_ids"].astype(np.int64)
            pred_ids = arrays["pred_token_ids"].astype(np.int64)
            pred_conf = arrays["pred_token_conf"].astype(np.float32)
            decoder_hidden = arrays["decoder_hidden"].astype(np.float32)
            valid = gt_ids != pad_id
            non_eos = gt_ids != eos_id
            baseline_selected = valid & non_eos & (pred_ids != gt_ids)
            baseline_preserve = valid & ~baseline_selected
            pair_weights = self._make_pair_weights(pred_ids, gt_ids, baseline_selected)
            if record["is_correct"]:
                self.samples.append(
                    {
                        "source": "correct_context",
                        "pred_token_ids": pred_ids,
                        "pred_token_conf": pred_conf,
                        "gt_token_ids": gt_ids,
                        "decoder_hidden": decoder_hidden,
                        "selected_mask": baseline_selected.astype(np.bool_),
                        "preserve_mask": baseline_preserve.astype(np.bool_),
                        "pair_weights": pair_weights,
                    }
                )
                self.real_records.append(
                    {
                        "gt_token_ids": gt_ids,
                        "decoder_hidden": decoder_hidden,
                    }
                )
            elif record["alignment_summary"]["replace_only_candidate"] and baseline_selected.any():
                self.samples.append(
                    {
                        "source": "baseline",
                        "pred_token_ids": pred_ids,
                        "pred_token_conf": pred_conf,
                        "gt_token_ids": gt_ids,
                        "decoder_hidden": decoder_hidden,
                        "selected_mask": baseline_selected.astype(np.bool_),
                        "preserve_mask": baseline_preserve.astype(np.bool_),
                        "pair_weights": pair_weights,
                    }
                )
                self.real_records.append(
                    {
                        "gt_token_ids": gt_ids,
                        "decoder_hidden": decoder_hidden,
                    }
                )
        self._append_synthetic_samples()

    def _build_segment_token_ids(self) -> Dict[str, List[int]]:
        buckets = {"digit": [], "alphabet": [], "chinese": [], "other": []}
        for token_id in range(1, self.output_num_classes):
            buckets[classify_token_id(token_id, self.tokenizer)].append(token_id)
        return buckets

    def _format_pair(self, pred_id: int, gt_id: int) -> str:
        pred_token = self.tokenizer._ids2tok([int(pred_id)], join=True)
        gt_token = self.tokenizer._ids2tok([int(gt_id)], join=True)
        return f"{pred_token}->{gt_token}"

    def _build_pair_difficulty_lookup(self, pair_difficulty_rows: List[Dict[str, object]]) -> Dict[tuple[int, int], Dict[str, object]]:
        lookup: Dict[tuple[int, int], Dict[str, object]] = {}
        for row in pair_difficulty_rows:
            lookup[(int(row["pred_token_id"]), int(row["gt_token_id"]))] = row
        return lookup

    def _build_pair_weight_map(self, pair_difficulty_rows: List[Dict[str, object]]) -> Dict[tuple[int, int], float]:
        pair_weights: Dict[tuple[int, int], float] = {}
        for row in pair_difficulty_rows:
            key = (int(row["pred_token_id"]), int(row["gt_token_id"]))
            difficulty_score = float(row.get("difficulty_score", 0.0))
            weight = 1.0 + self.pair_weight_alpha * difficulty_score
            weight = max(self.pair_weight_min, min(self.pair_weight_max, weight))
            pair_weights[key] = float(weight)
        return pair_weights

    def _make_pair_weights(self, pred_ids: np.ndarray, gt_ids: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
        pair_weights = np.ones(gt_ids.shape, dtype=np.float32)
        for pos in np.where(selected_mask)[0]:
            key = (int(pred_ids[pos]), int(gt_ids[pos]))
            pair_weights[pos] = float(self.pair_weight_by_pair.get(key, 1.0))
        return pair_weights

    def _build_confusion_candidates(
        self,
        confusion_table: List[Dict[str, object]],
        confusion_topk: int,
    ) -> Dict[int, List[Dict[str, object]]]:
        by_gt: Dict[int, List[Dict[str, object]]] = {}
        for row in confusion_table:
            gt_id = int(row["gt_token_id"])
            pred_id = int(row["pred_token_id"])
            if gt_id == pred_id:
                continue
            if self.segment_preserving:
                gt_segment = classify_token_id(gt_id, self.tokenizer)
                pred_segment = classify_token_id(pred_id, self.tokenizer)
                if gt_segment != pred_segment:
                    continue
            by_gt.setdefault(gt_id, []).append(row)
        for gt_id, rows in by_gt.items():
            rows.sort(key=lambda row: (-int(row["count"]), int(row["pred_token_id"])))
            by_gt[gt_id] = rows[:confusion_topk]
        return by_gt

    def _build_pair_specific_candidates(
        self,
        pair_difficulty_rows: List[Dict[str, object]],
        pair_synthetic_topk: int,
    ) -> Dict[int, List[Dict[str, object]]]:
        by_gt: Dict[int, List[Dict[str, object]]] = {}
        for row in pair_difficulty_rows:
            train_count = int(row.get("train_count", row.get("count", 0)))
            if train_count < self.pair_min_count:
                continue
            pred_id = int(row["pred_token_id"])
            gt_id = int(row["gt_token_id"])
            pair_name = row.get("pair") or self._format_pair(pred_id, gt_id)
            if self.pair_include_set and pair_name not in self.pair_include_set:
                continue
            if pair_name in self.pair_exclude_set:
                continue
            if self.segment_preserving:
                gt_segment = classify_token_id(gt_id, self.tokenizer)
                pred_segment = classify_token_id(pred_id, self.tokenizer)
                if gt_segment != pred_segment:
                    continue
            candidate = dict(row)
            oracle_gap = max(0.0, float(candidate.get("eval_oracle@5", candidate.get("oracle@5", 0.0))) - float(candidate.get("eval_correction_rate", candidate.get("correction_rate", 0.0))))
            difficulty_score = float(candidate.get("difficulty_score", 0.0))
            recommended_multiplier = float(candidate.get("recommended_synthetic_multiplier", 1.0))
            if self.pair_synthetic_mode == "balanced":
                sample_weight = float(train_count)
            elif self.pair_synthetic_mode == "inverse_success":
                sample_weight = float(train_count) * (1.0 + self.pair_synthetic_multiplier * oracle_gap)
            else:
                sample_weight = float(train_count) * max(
                    1.0,
                    recommended_multiplier * self.pair_synthetic_multiplier,
                    1.0 + self.pair_synthetic_multiplier * difficulty_score,
                )
            candidate["_sample_weight"] = float(sample_weight)
            candidate["pair"] = pair_name
            by_gt.setdefault(gt_id, []).append(candidate)
        for gt_id, rows in by_gt.items():
            rows.sort(
                key=lambda row: (
                    -float(row["_sample_weight"]),
                    -float(row.get("difficulty_score", 0.0)),
                    -int(row.get("train_count", row.get("count", 0))),
                    int(row["pred_token_id"]),
                )
            )
            by_gt[gt_id] = rows[:pair_synthetic_topk]
        return by_gt

    def _append_synthetic_samples(self) -> None:
        if self.synthetic_mode == "none" or self.synthetic_ratio <= 0 or not self.real_records:
            return
        target_count = int(round(len(self.real_records) * self.synthetic_ratio))
        self.synthetic_target_count = target_count
        if target_count <= 0:
            return
        produced = 0
        rounds = 0
        max_rounds = max(1, target_count * 4)
        while produced < target_count and rounds < max_rounds:
            rounds += 1
            indices = list(range(len(self.real_records)))
            self.random_state.shuffle(indices)
            any_progress = False
            for record_index in indices:
                if produced >= target_count:
                    break
                base_record = self.real_records[record_index]
                for _ in range(max(1, self.synthetic_per_sample)):
                    if produced >= target_count:
                        break
                    synthetic = self._build_synthetic_sample(
                        gt_ids=base_record["gt_token_ids"],
                        decoder_hidden=base_record["decoder_hidden"],
                    )
                    if synthetic is None:
                        continue
                    self.samples.append(synthetic)
                    produced += 1
                    any_progress = True
            if not any_progress:
                break

    def _sample_noise_positions(self, eligible_positions: np.ndarray) -> np.ndarray:
        max_choices = min(len(eligible_positions), max(1, self.max_noise_positions))
        num_noised = self.random_state.randint(1, max_choices)
        return self.np_random.choice(eligible_positions, size=num_noised, replace=False)

    def _noise_valid_positions(self, token_ids: np.ndarray) -> np.ndarray:
        valid = (token_ids != self.pad_id) & (token_ids != self.eos_id)
        if self.bos_id >= 0:
            valid = valid & (token_ids != self.bos_id)
        return np.where(valid)[0]

    def _random_replacement_candidates(self, token_id: int) -> List[int]:
        if self.segment_preserving:
            segment = classify_token_id(token_id, self.tokenizer)
            candidates = [idx for idx in self.segment_token_ids.get(segment, []) if idx != token_id]
            if candidates:
                return candidates
        return [idx for idx in self.all_token_ids if idx != token_id]

    def _build_random_synthetic(self, gt_ids: np.ndarray, decoder_hidden: np.ndarray, source: str) -> Dict[str, Any] | None:
        pred_ids = gt_ids.copy()
        pred_conf = np.full(gt_ids.shape, 0.99, dtype=np.float32)
        valid_positions = self._noise_valid_positions(gt_ids)
        eligible_positions = np.array(
            [pos for pos in valid_positions if self._random_replacement_candidates(int(gt_ids[pos]))],
            dtype=np.int64,
        )
        selected_mask = np.zeros(gt_ids.shape, dtype=np.bool_)
        if not eligible_positions.size:
            return None
        chosen = self._sample_noise_positions(eligible_positions)
        for pos in chosen:
            pos = int(pos)
            candidates = self._random_replacement_candidates(int(gt_ids[pos]))
            replacement = int(self.random_state.choice(candidates))
            pred_ids[pos] = replacement
            pred_conf[pos] = 0.10
            selected_mask[pos] = True
        preserve_mask = (gt_ids != self.pad_id) & ~selected_mask
        pair_weights = self._make_pair_weights(pred_ids, gt_ids, selected_mask)
        return {
            "source": source,
            "pred_token_ids": pred_ids,
            "pred_token_conf": pred_conf,
            "gt_token_ids": gt_ids,
            "decoder_hidden": decoder_hidden,
            "selected_mask": selected_mask,
            "preserve_mask": preserve_mask.astype(np.bool_),
            "pair_weights": pair_weights,
        }

    def _build_confusion_synthetic(
        self,
        gt_ids: np.ndarray,
        decoder_hidden: np.ndarray,
        source: str = "synthetic_confusion",
    ) -> Dict[str, Any] | None:
        pred_ids = gt_ids.copy()
        pred_conf = np.full(gt_ids.shape, 0.99, dtype=np.float32)
        eligible_positions = []
        for pos in self._noise_valid_positions(gt_ids):
            gt_id = int(gt_ids[pos])
            candidates = self.confusion_candidates.get(gt_id, [])
            if candidates:
                eligible_positions.append(int(pos))
        if not eligible_positions:
            self.synthetic_fallback_count += 1
            return self._build_random_synthetic(gt_ids, decoder_hidden, source="synthetic_confusion_fallback")
        selected_mask = np.zeros(gt_ids.shape, dtype=np.bool_)
        chosen = self._sample_noise_positions(np.asarray(eligible_positions, dtype=np.int64))
        for pos in chosen:
            pos = int(pos)
            gt_id = int(gt_ids[pos])
            candidates = self.confusion_candidates[gt_id]
            weights = [max(1, int(row["count"])) for row in candidates]
            picked = self.random_state.choices(candidates, weights=weights, k=1)[0]
            pred_ids[pos] = int(picked["pred_token_id"])
            pred_conf[pos] = 0.10
            selected_mask[pos] = True
        preserve_mask = (gt_ids != self.pad_id) & ~selected_mask
        pair_weights = self._make_pair_weights(pred_ids, gt_ids, selected_mask)
        return {
            "source": source,
            "pred_token_ids": pred_ids,
            "pred_token_conf": pred_conf,
            "gt_token_ids": gt_ids,
            "decoder_hidden": decoder_hidden,
            "selected_mask": selected_mask,
            "preserve_mask": preserve_mask.astype(np.bool_),
            "pair_weights": pair_weights,
        }

    def _build_pair_specific_synthetic(self, gt_ids: np.ndarray, decoder_hidden: np.ndarray) -> Dict[str, Any] | None:
        pred_ids = gt_ids.copy()
        pred_conf = np.full(gt_ids.shape, 0.99, dtype=np.float32)
        eligible_positions = []
        for pos in self._noise_valid_positions(gt_ids):
            gt_id = int(gt_ids[pos])
            candidates = self.pair_specific_candidates.get(gt_id, [])
            if candidates:
                eligible_positions.append(int(pos))
        if not eligible_positions:
            self.synthetic_fallback_count += 1
            return self._build_confusion_synthetic(
                gt_ids,
                decoder_hidden,
                source="synthetic_pair_specific_fallback",
            )
        selected_mask = np.zeros(gt_ids.shape, dtype=np.bool_)
        chosen = self._sample_noise_positions(np.asarray(eligible_positions, dtype=np.int64))
        for pos in chosen:
            pos = int(pos)
            gt_id = int(gt_ids[pos])
            candidates = self.pair_specific_candidates[gt_id]
            weights = [max(1e-6, float(row["_sample_weight"])) for row in candidates]
            picked = self.random_state.choices(candidates, weights=weights, k=1)[0]
            pred_ids[pos] = int(picked["pred_token_id"])
            pred_conf[pos] = 0.10
            selected_mask[pos] = True
        preserve_mask = (gt_ids != self.pad_id) & ~selected_mask
        pair_weights = self._make_pair_weights(pred_ids, gt_ids, selected_mask)
        return {
            "source": "synthetic_pair_specific",
            "pred_token_ids": pred_ids,
            "pred_token_conf": pred_conf,
            "gt_token_ids": gt_ids,
            "decoder_hidden": decoder_hidden,
            "selected_mask": selected_mask,
            "preserve_mask": preserve_mask.astype(np.bool_),
            "pair_weights": pair_weights,
        }

    def _build_synthetic_sample(self, gt_ids: np.ndarray, decoder_hidden: np.ndarray) -> Dict[str, Any] | None:
        synthetic = None
        if self.synthetic_mode == "random":
            synthetic = self._build_random_synthetic(gt_ids, decoder_hidden, source="synthetic_random")
        elif self.synthetic_mode == "confusion":
            synthetic = self._build_confusion_synthetic(gt_ids, decoder_hidden)
        elif self.synthetic_mode == "pair_specific":
            synthetic = self._build_pair_specific_synthetic(gt_ids, decoder_hidden)
        if synthetic is None:
            self.synthetic_skip_count += 1
            return None
        self.synthetic_generated_count += 1
        if synthetic["source"] == "synthetic_random":
            self.synthetic_random_generated_count += 1
        elif synthetic["source"] == "synthetic_confusion":
            self.synthetic_confusion_generated_count += 1
        elif synthetic["source"] == "synthetic_confusion_fallback":
            self.synthetic_random_generated_count += 1
        elif synthetic["source"] == "synthetic_pair_specific":
            self.synthetic_pair_specific_generated_count += 1
        elif synthetic["source"] == "synthetic_pair_specific_fallback":
            self.synthetic_confusion_generated_count += 1
        return synthetic

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]
        return {
            "source": row["source"],
            "pred_token_ids": torch.as_tensor(row["pred_token_ids"], dtype=torch.long),
            "pred_token_conf": torch.as_tensor(row["pred_token_conf"], dtype=torch.float32),
            "gt_token_ids": torch.as_tensor(row["gt_token_ids"], dtype=torch.long),
            "decoder_hidden": torch.as_tensor(row["decoder_hidden"], dtype=torch.float32),
            "selected_mask": torch.as_tensor(row["selected_mask"], dtype=torch.bool),
            "preserve_mask": torch.as_tensor(row["preserve_mask"], dtype=torch.bool),
            "pair_weights": torch.as_tensor(row["pair_weights"], dtype=torch.float32),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source": [row["source"] for row in batch],
        "pred_token_ids": torch.stack([row["pred_token_ids"] for row in batch], dim=0),
        "pred_token_conf": torch.stack([row["pred_token_conf"] for row in batch], dim=0),
        "gt_token_ids": torch.stack([row["gt_token_ids"] for row in batch], dim=0),
        "decoder_hidden": torch.stack([row["decoder_hidden"] for row in batch], dim=0),
        "selected_mask": torch.stack([row["selected_mask"] for row in batch], dim=0),
        "preserve_mask": torch.stack([row["preserve_mask"] for row in batch], dim=0),
        "pair_weights": torch.stack([row["pair_weights"] for row in batch], dim=0),
    }


def make_model(args: argparse.Namespace):
    overrides = [
        "model=slp_mdiff_corrector",
        f"model.contract_type={args.contract_type}",
        f"model.use_encoder_memory=false",
        f"model.replace_only={str(args.replace_only).lower()}",
        f"model.loss_type={args.loss_type}",
        f"model.lambda_preservation={args.lambda_preservation}",
        f"model.batch_size={args.batch_size}",
        f"model.lr={args.lr}",
    ]
    with initialize(config_path="../configs", version_base="1.2"):
        cfg = compose(config_name="main", overrides=overrides)
    with open_dict(cfg):
        cfg.data.root_dir = os.path.abspath(cfg.data.root_dir)
    return cfg, instantiate(cfg.model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contract_type", default="token_decoder_hidden")
    parser.add_argument("--replace_only", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--loss_type", default="selected_plus_preservation")
    parser.add_argument("--lambda_preservation", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--include_synthetic", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--synthetic-mode", choices=["none", "random", "confusion", "pair_specific"], default="none")
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    parser.add_argument("--synthetic-per-sample", type=int, default=1)
    parser.add_argument("--confusion-table", default=None)
    parser.add_argument("--confusion-topk", type=int, default=20)
    parser.add_argument("--max-noise-positions", type=int, default=2)
    parser.add_argument("--segment-preserving", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--pair-difficulty-table", default=None)
    parser.add_argument("--pair-synthetic-mode", choices=["balanced", "hard_focus", "inverse_success"], default="hard_focus")
    parser.add_argument("--pair-synthetic-topk", type=int, default=50)
    parser.add_argument("--pair-synthetic-multiplier", type=float, default=1.0)
    parser.add_argument("--pair-min-count", type=int, default=1)
    parser.add_argument("--pair-include-list", default="")
    parser.add_argument("--pair-exclude-list", default="")
    parser.add_argument("--pair-weight-table", default=None)
    parser.add_argument("--pair-weight-alpha", type=float, default=1.0)
    parser.add_argument("--pair-weight-max", type=float, default=3.0)
    parser.add_argument("--pair-weight-min", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    torch.manual_seed(args.noise_seed)
    output_dir = Path(args.output_dir).absolute()
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg, model = make_model(args)
    records = load_manifest(args.cache_dir)
    shards = load_feature_shards(args.cache_dir)
    baseline_replace_only_count = sum(
        (not record["is_correct"]) and record["alignment_summary"]["replace_only_candidate"]
        for record in records
    )
    synthetic_mode = args.synthetic_mode
    synthetic_ratio = args.synthetic_ratio
    if args.include_synthetic is not None and args.synthetic_mode == "none":
        synthetic_mode = "random" if args.include_synthetic else "none"
        if args.include_synthetic and synthetic_ratio == 0.0:
            synthetic_ratio = 1.0
    if args.include_synthetic is None and synthetic_mode == "none" and baseline_replace_only_count == 0:
        synthetic_mode = "random"
        if synthetic_ratio == 0.0:
            synthetic_ratio = 1.0
    confusion_table = []
    if synthetic_mode in {"confusion", "pair_specific"}:
        if not args.confusion_table:
            raise ValueError("--confusion-table is required when synthetic noise uses confusion priors")
        confusion_table = load_confusion_table(args.confusion_table)
    pair_difficulty_rows = []
    pair_weight_rows = []
    if synthetic_mode == "pair_specific":
        if not args.pair_difficulty_table:
            raise ValueError("--pair-difficulty-table is required when --synthetic-mode=pair_specific")
        pair_difficulty_rows = load_pair_difficulty_table(args.pair_difficulty_table)
    if args.loss_type == "pair_weighted_selected_plus_preservation":
        if not args.pair_weight_table:
            raise ValueError("--pair-weight-table is required when --loss_type=pair_weighted_selected_plus_preservation")
        pair_weight_rows = load_pair_difficulty_table(args.pair_weight_table)
    merged_pair_rows = pair_difficulty_rows or pair_weight_rows
    dataset = CorrectorSmokeDataset(
        records,
        shards,
        model.output_num_classes,
        model.bos_id,
        model.eos_id,
        model.pad_id,
        contract_type=args.contract_type,
        tokenizer=model.tokenizer,
        synthetic_mode=synthetic_mode,
        synthetic_ratio=synthetic_ratio,
        synthetic_per_sample=args.synthetic_per_sample,
        max_noise_positions=args.max_noise_positions,
        segment_preserving=args.segment_preserving,
        confusion_table=confusion_table,
        confusion_topk=args.confusion_topk,
        pair_difficulty_rows=merged_pair_rows,
        pair_synthetic_mode=args.pair_synthetic_mode,
        pair_synthetic_topk=args.pair_synthetic_topk,
        pair_synthetic_multiplier=args.pair_synthetic_multiplier,
        pair_min_count=args.pair_min_count,
        pair_include_list=[item.strip() for item in args.pair_include_list.split(",") if item.strip()],
        pair_exclude_list=[item.strip() for item in args.pair_exclude_list.split(",") if item.strip()],
        pair_weight_alpha=args.pair_weight_alpha,
        pair_weight_min=args.pair_weight_min,
        pair_weight_max=args.pair_weight_max,
        noise_seed=args.noise_seed,
    )
    if not len(dataset):
        raise RuntimeError("No training samples constructed from cache")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch, drop_last=False)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    data_iter = iter(loader)
    initial_loss = None
    initial_selected = None
    initial_preservation = None
    last_metrics = None
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        optimizer.zero_grad(set_to_none=True)
        pred_token_ids = batch["pred_token_ids"].to(device)
        pred_token_conf = batch["pred_token_conf"].to(device)
        gt_token_ids = batch["gt_token_ids"].to(device)
        decoder_hidden = batch["decoder_hidden"].to(device)
        selected_mask = batch["selected_mask"].to(device)
        preserve_mask = batch["preserve_mask"].to(device)
        pair_weights = batch["pair_weights"].to(device)
        decoder_hidden_input = decoder_hidden
        if args.contract_type == "token_only":
            decoder_hidden_input = torch.zeros_like(decoder_hidden)
        logits = model(
            pred_token_ids=pred_token_ids,
            pred_token_conf=pred_token_conf,
            correction_mask=selected_mask,
            decoder_hidden=decoder_hidden_input,
        )
        metrics = model.compute_loss(logits, gt_token_ids, selected_mask, preserve_mask, pair_weights=pair_weights)
        metrics["loss"].backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = float(metrics["loss"].detach().cpu())
            initial_selected = float(metrics["selected_ce"].cpu())
            initial_preservation = float(metrics["preservation_ce"].cpu())
        last_metrics = {
            "step": step,
            "loss": float(metrics["loss"].detach().cpu()),
            "selected_ce": float(metrics["selected_ce"].cpu()),
            "preservation_ce": float(metrics["preservation_ce"].cpu()),
            "selected_count": int(metrics["selected_count"].cpu()),
            "preserve_count": int(metrics["preserve_count"].cpu()),
        }
        if step <= 3 or step == args.max_steps or step % 20 == 0:
            print(json.dumps(last_metrics, ensure_ascii=False), flush=True)

    runtime_sec = time.time() - start
    ckpt_path = ckpt_dir / "last.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": dict(cfg.model),
            "max_steps": args.max_steps,
        },
        ckpt_path,
    )
    summary = {
        "cache_dir": str(Path(args.cache_dir).absolute()),
        "output_dir": str(output_dir),
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "dataset_size": len(dataset),
        "contract_type": args.contract_type,
        "loss_type": args.loss_type,
        "include_synthetic": synthetic_mode != "none",
        "synthetic_mode": synthetic_mode,
        "synthetic_ratio": synthetic_ratio,
        "synthetic_per_sample": args.synthetic_per_sample,
        "max_noise_positions": args.max_noise_positions,
        "segment_preserving": args.segment_preserving,
        "confusion_table": args.confusion_table,
        "confusion_topk": args.confusion_topk,
        "pair_difficulty_table": args.pair_difficulty_table,
        "pair_synthetic_mode": args.pair_synthetic_mode,
        "pair_synthetic_topk": args.pair_synthetic_topk,
        "pair_synthetic_multiplier": args.pair_synthetic_multiplier,
        "pair_min_count": args.pair_min_count,
        "pair_include_list": [item.strip() for item in args.pair_include_list.split(",") if item.strip()],
        "pair_exclude_list": [item.strip() for item in args.pair_exclude_list.split(",") if item.strip()],
        "pair_weight_table": args.pair_weight_table,
        "pair_weight_alpha": args.pair_weight_alpha,
        "pair_weight_min": args.pair_weight_min,
        "pair_weight_max": args.pair_weight_max,
        "baseline_replace_only_count": baseline_replace_only_count,
        "baseline_sample_count": sum(sample["source"] == "baseline" for sample in dataset.samples),
        "correct_context_sample_count": sum(sample["source"] == "correct_context" for sample in dataset.samples),
        "synthetic_sample_count": sum(sample["source"].startswith("synthetic") for sample in dataset.samples),
        "synthetic_target_count": dataset.synthetic_target_count,
        "synthetic_confusion_sample_count": sum(sample["source"] == "synthetic_confusion" for sample in dataset.samples),
        "synthetic_random_sample_count": sum(sample["source"] == "synthetic_random" for sample in dataset.samples),
        "synthetic_pair_specific_sample_count": sum(sample["source"] == "synthetic_pair_specific" for sample in dataset.samples),
        "synthetic_confusion_fallback_count": sum(sample["source"] == "synthetic_confusion_fallback" for sample in dataset.samples),
        "synthetic_pair_specific_fallback_count": sum(sample["source"] == "synthetic_pair_specific_fallback" for sample in dataset.samples),
        "synthetic_generated_count": dataset.synthetic_generated_count,
        "synthetic_confusion_generated_count": dataset.synthetic_confusion_generated_count,
        "synthetic_random_generated_count": dataset.synthetic_random_generated_count,
        "synthetic_pair_specific_generated_count": dataset.synthetic_pair_specific_generated_count,
        "synthetic_fallback_count": dataset.synthetic_fallback_count,
        "synthetic_skip_count": dataset.synthetic_skip_count,
        "selected_positions_count": int(sum(sample["selected_mask"].sum() for sample in dataset.samples)),
        "preserve_positions_count": int(sum(sample["preserve_mask"].sum() for sample in dataset.samples)),
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "initial_loss": initial_loss,
        "final_loss": last_metrics["loss"],
        "initial_selected_ce": initial_selected,
        "final_selected_ce": last_metrics["selected_ce"],
        "initial_preservation_ce": initial_preservation,
        "final_preservation_ce": last_metrics["preservation_ce"],
        "runtime_sec": runtime_sec,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
        ),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
