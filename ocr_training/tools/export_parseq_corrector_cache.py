#!/usr/bin/env python
import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from tools.mdiff_corrector_utils import (
    align_pred_gt,
    build_confusion_knowledge,
    load_confusion_table,
    normalize_prediction,
    read_charset,
    token_id_to_char,
    token_ids_to_text,
)


class ExportLmdbDataset(LmdbDataset):
    def __getitem__(self, index):
        if self.unlabelled:
            label = index
            lmdb_index = index + 1
        else:
            label = self.labels[index]
            lmdb_index = self.filtered_index_list[index]
        img_key = f"image-{lmdb_index:09d}".encode()
        meta_key = f"meta-{lmdb_index:09d}".encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
            meta_buf = txn.get(meta_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        meta = json.loads(meta_buf.decode()) if meta_buf is not None else None
        return {
            "image": img,
            "label": label,
            "lmdb_index": lmdb_index,
            "lmdb_root": self.root,
            "meta": meta,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([row["image"] for row in batch], dim=0),
        "labels": [row["label"] for row in batch],
        "lmdb_index": [row["lmdb_index"] for row in batch],
        "lmdb_root": [row["lmdb_root"] for row in batch],
        "meta": [row["meta"] for row in batch],
    }


def build_dataset(split: str, subset: Optional[str], charset: str, max_label_length: int):
    root_dir = OCR_ROOT / "data"
    transform = SceneTextDataModule.get_transform((224, 224), augment=False)
    if split == "train":
        root = root_dir / "train" / "SLP34K_lmdb_train"
        resolved_subset = root.name
    elif split == "val":
        root = root_dir / "val" / "SLP34K_lmdb_test"
        resolved_subset = root.name
    elif split == "test":
        if not subset:
            raise ValueError("--subset is required for split=test")
        root = root_dir / "test" / "SLP34K_lmdb_benchmark" / subset
        resolved_subset = subset
    else:
        raise ValueError(f"Unsupported split: {split}")
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    dataset = ExportLmdbDataset(root=str(root), charset=charset, max_label_len=max_label_length, transform=transform)
    return dataset, resolved_subset, str(root)


def close_env(dataset: ExportLmdbDataset) -> None:
    env = getattr(dataset, "_env", None)
    if env is not None:
        env.close()
        dataset._env = None


def summarize_array(name: str, value: np.ndarray) -> Dict[str, Any]:
    return {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}


def bool_arg(value: str) -> bool:
    return value.lower() == "true"


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def is_low_conf_sample(pred_ids: np.ndarray, pred_conf: np.ndarray, pad_id: int, eos_id: int, tau_low: float) -> bool:
    valid = (pred_ids != pad_id) & (pred_ids != eos_id)
    return bool(np.any(valid & (pred_conf < tau_low)))


def hard_slice_match(meta: Optional[Dict[str, Any]], gt_length: int, low_conf_sample: bool) -> bool:
    if gt_length >= 21 or low_conf_sample:
        return True
    if not meta:
        return False
    if meta.get("vocabulary_type") == "OOV":
        return True
    if meta.get("resolution_type") == "low":
        return True
    if meta.get("structure_type") in {"vertical", "multi-lines"}:
        return True
    if meta.get("quality") in {"low", "hard"}:
        return True
    return False


def metadata_matches(
    meta: Optional[Dict[str, Any]],
    vocabulary_type: Optional[str],
    quality: Optional[str],
    structure_type: Optional[str],
) -> bool:
    if not any([vocabulary_type, quality, structure_type]):
        return True
    if not meta:
        return False
    if vocabulary_type and meta.get("vocabulary_type") != vocabulary_type:
        return False
    if quality and meta.get("quality") != quality:
        return False
    if structure_type and meta.get("structure_type") != structure_type:
        return False
    return True


def length_matches(gt_length: int, pred_length: int, min_length: Optional[int], max_length: Optional[int]) -> bool:
    if min_length is not None and max(gt_length, pred_length) < min_length:
        return False
    if max_length is not None and max(gt_length, pred_length) > max_length:
        return False
    return True


def confusion_pair_match(
    alignment: Dict[str, Any],
    confusion_info: Dict[str, Any],
    tokenizer,
    confusion_topk: int,
) -> bool:
    if not confusion_info:
        return False
    allowed_pairs = confusion_info["top_pairs"]
    if confusion_topk > 0:
        allowed_pairs = set(list(allowed_pairs)[:confusion_topk])
    for step in alignment["steps"]:
        if step["op"] != "replace":
            continue
        pair = f"{token_id_to_char(step['pred_id'], tokenizer)}->{token_id_to_char(step['gt_id'], tokenizer)}"
        if pair in allowed_pairs:
            return True
    return False


def main_filter_match(
    filter_mode: str,
    is_correct: bool,
    alignment: Dict[str, Any],
    low_conf_sample: bool,
    meta: Optional[Dict[str, Any]],
    gt_length: int,
    pred_length: int,
    min_replace_count: int,
    max_insert_delete_count: int,
    min_length: Optional[int],
    max_length: Optional[int],
    vocabulary_type: Optional[str],
    quality: Optional[str],
    structure_type: Optional[str],
    confusion_info: Dict[str, Any],
    tokenizer,
    confusion_topk: int,
) -> bool:
    replace_count = alignment["replace_count"]
    insert_delete_count = alignment["insert_count"] + alignment["delete_count"]
    if not metadata_matches(meta, vocabulary_type, quality, structure_type):
        return False
    if not length_matches(gt_length, pred_length, min_length, max_length):
        return False
    if filter_mode == "all":
        return True
    if filter_mode == "incorrect":
        return not is_correct
    if filter_mode == "replace_only":
        return (not is_correct) and alignment["replace_only_candidate"] and replace_count >= min_replace_count
    if filter_mode == "replace_dominant":
        return (
            not is_correct
            and replace_count >= min_replace_count
            and replace_count > insert_delete_count
            and insert_delete_count <= max_insert_delete_count
        )
    if filter_mode == "low_conf":
        return low_conf_sample
    if filter_mode == "hard_slice":
        return hard_slice_match(meta, gt_length, low_conf_sample)
    if filter_mode == "confusion_pair":
        return (not is_correct) and confusion_pair_match(alignment, confusion_info, tokenizer, confusion_topk)
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def build_record(
    resolved_subset: str,
    batch: Dict[str, Any],
    row_idx: int,
    gt_text: str,
    pred_text: str,
    valid_length: int,
    eos_position: int,
    alignment: Dict[str, Any],
    feature_shard: str,
    feature_index: int,
    low_conf_sample: bool,
) -> Dict[str, Any]:
    return {
        "sample_id": f"{resolved_subset}:{int(batch['lmdb_index'][row_idx]):09d}",
        "subset": resolved_subset,
        "lmdb_index": int(batch["lmdb_index"][row_idx]),
        "lmdb_root": batch["lmdb_root"][row_idx],
        "gt_text": gt_text,
        "pred_text": pred_text,
        "is_correct": pred_text == gt_text,
        "valid_length": int(valid_length),
        "eos_position": int(eos_position),
        "alignment_ops": alignment["ops"],
        "alignment_summary": {
            "replace_count": alignment["replace_count"],
            "correct_count": alignment["correct_count"],
            "insert_count": alignment["insert_count"],
            "delete_count": alignment["delete_count"],
            "replace_only_candidate": alignment["replace_only_candidate"],
        },
        "metadata": batch["meta"][row_idx],
        "feature_shard": feature_shard,
        "feature_index": feature_index,
        "low_conf_sample": low_conf_sample,
    }


def append_feature_rows(
    feature_rows: Dict[str, List[np.ndarray]],
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    pred_conf: np.ndarray,
    topk_indices: np.ndarray,
    topk_values: np.ndarray,
    valid_length: int,
    eos_position: int,
    decoder_hidden_row: Optional[np.ndarray],
    encoder_memory_row: Optional[np.ndarray],
) -> None:
    feature_rows["pred_token_ids"].append(pred_ids.astype(np.int16))
    feature_rows["gt_token_ids"].append(gt_ids.astype(np.int16))
    feature_rows["pred_token_conf"].append(pred_conf.astype(np.float16))
    feature_rows["topk_indices"].append(topk_indices.astype(np.int16))
    feature_rows["topk_values"].append(topk_values.astype(np.float16))
    feature_rows["valid_length"].append(np.asarray(valid_length, dtype=np.int16))
    feature_rows["eos_position"].append(np.asarray(eos_position, dtype=np.int16))
    if decoder_hidden_row is not None:
        feature_rows["decoder_hidden"].append(decoder_hidden_row.astype(np.float16))
    if encoder_memory_row is not None:
        feature_rows["encoder_memory"].append(encoder_memory_row.astype(np.float16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--max-export", type=int, default=512)
    parser.add_argument("--max-wrong", type=int, default=None)
    parser.add_argument("--min-replace-count", type=int, default=1)
    parser.add_argument("--max-insert-delete-count", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--vocabulary-type", default=None)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--structure-type", default=None)
    parser.add_argument("--slice-name", default=None)
    parser.add_argument("--confusion-table", default=None)
    parser.add_argument("--confusion-topk", type=int, default=20)
    parser.add_argument("--tau-low", type=float, default=0.70)
    parser.add_argument("--filter-mode", choices=["all", "incorrect", "replace_only", "replace_dominant", "low_conf", "hard_slice", "confusion_pair"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--export_decoder_hidden", type=bool_arg, default=True)
    parser.add_argument("--export_encoder_memory", type=bool_arg, default=False)
    parser.add_argument("--save-correct-context", type=bool_arg, default=True)
    parser.add_argument("--include-correct-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.limit is not None and args.max_export == 512:
        args.max_export = args.limit
    if not 0.0 <= args.include_correct_ratio < 1.0:
        raise ValueError("--include-correct-ratio must be in [0, 1)")

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_rows = load_confusion_table(args.confusion_table) if args.confusion_table else []
    confusion_info = build_confusion_knowledge(confusion_rows) if confusion_rows else {}
    if confusion_rows:
        confusion_info["top_pairs"] = [f"{row['pred_token']}->{row['gt_token']}" for row in confusion_rows[: args.confusion_topk]]
    else:
        confusion_info["top_pairs"] = []
    charset = read_charset(str(OCR_ROOT / "configs" / "charset" / "SLP34K_568.yaml"))
    model = load_from_checkpoint(
        args.checkpoint,
        charset_test=charset,
        mae_pretrained_path="pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth",
    ).eval()
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset, resolved_subset, dataset_root = build_dataset(args.split, args.subset, charset, model.max_label_length)
    scan_limit = min(args.scan_limit, len(dataset))
    subset_dataset = torch.utils.data.Subset(dataset, list(range(scan_limit)))
    loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    manifest_path = output_dir / "manifest.jsonl"
    feature_shard = output_dir / "features_0000.npz"
    manifest_rows: List[Dict[str, Any]] = []
    feature_rows: Dict[str, List[np.ndarray]] = {
        "pred_token_ids": [],
        "gt_token_ids": [],
        "pred_token_conf": [],
        "topk_indices": [],
        "topk_values": [],
        "valid_length": [],
        "eos_position": [],
        "decoder_hidden": [],
        "encoder_memory": [],
    }
    op_counter = Counter()
    replace_pair_counter = Counter()
    wrong_token_confs: List[float] = []
    correct_token_confs: List[float] = []
    low_conf_token_count = 0
    gt_lengths: List[int] = []
    pred_lengths: List[int] = []
    baseline_wrong_positions = 0
    baseline_correct_positions = 0
    metadata_available_count = 0
    metadata_missing_fields = Counter()

    total_scanned = 0
    baseline_correct_samples = 0
    baseline_incorrect_samples = 0
    replace_only_scanned = 0
    replace_dominant_scanned = 0
    insert_delete_scanned = 0
    low_conf_scanned = 0
    long_21plus_count = 0

    main_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    feature_index = 0
    target_correct_context = int(round(args.max_export * args.include_correct_ratio))
    target_main_export = args.max_export - target_correct_context
    target_wrong_export = args.max_wrong if args.max_wrong is not None else target_main_export
    exported_wrong_count = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)
            outputs = model.forward_with_aux(
                images,
                max_length=model.max_label_length,
                return_memory=args.export_encoder_memory,
                return_hidden=args.export_decoder_hidden,
                return_token_ids=True,
            )
            logits = outputs["logits"].detach().cpu()
            raw_pred_ids = outputs["pred_token_ids"].detach().cpu().numpy()
            raw_pred_conf = outputs["pred_token_conf"].detach().cpu().numpy()
            topk_values, topk_indices = logits.topk(args.topk, dim=-1)
            gt_token_ids_batch = model.tokenizer.encode(batch["labels"], torch.device("cpu"))[:, 1:].numpy()
            gt_token_ids = np.full((len(batch["labels"]), model.max_label_length + 1), model.tokenizer.pad_id, dtype=np.int16)
            gt_token_ids[:, : gt_token_ids_batch.shape[1]] = gt_token_ids_batch
            decoder_hidden = outputs.get("decoder_hidden")
            encoder_memory = outputs.get("encoder_memory")
            if decoder_hidden is not None:
                decoder_hidden = decoder_hidden.detach().cpu().numpy().astype(np.float16)
            if encoder_memory is not None:
                encoder_memory = encoder_memory.detach().cpu().numpy().astype(np.float16)

            for row_idx, gt_text in enumerate(batch["labels"]):
                total_scanned += 1
                pred_ids, pred_conf, eos_position, valid_length = normalize_prediction(
                    raw_pred_ids[row_idx],
                    raw_pred_conf[row_idx],
                    eos_id=model.tokenizer.eos_id,
                    pad_id=model.tokenizer.pad_id,
                )
                pred_text = token_ids_to_text(pred_ids, model.tokenizer)
                is_correct = pred_text == gt_text
                gt_length = int(np.sum((gt_token_ids[row_idx] != model.tokenizer.pad_id) & (gt_token_ids[row_idx] != model.tokenizer.eos_id)))
                low_conf_sample = is_low_conf_sample(pred_ids, pred_conf, model.tokenizer.pad_id, model.tokenizer.eos_id, args.tau_low)
                alignment = align_pred_gt(pred_ids, gt_token_ids[row_idx], model.tokenizer.eos_id, model.tokenizer.pad_id)

                op_counter.update(alignment["ops"])
                gt_lengths.append(gt_length)
                pred_lengths.append(alignment["pred_length"])
                if batch["meta"][row_idx] is not None:
                    metadata_available_count += 1
                    for field in ["quality", "vocabulary_type", "resolution_type", "structure_type"]:
                        if field not in batch["meta"][row_idx]:
                            metadata_missing_fields[field] += 1
                if gt_length >= 21:
                    long_21plus_count += 1
                if is_correct:
                    baseline_correct_samples += 1
                else:
                    baseline_incorrect_samples += 1
                if alignment["replace_only_candidate"] and alignment["replace_count"] > 0:
                    replace_only_scanned += 1
                if alignment["replace_count"] > alignment["insert_count"] + alignment["delete_count"] and alignment["replace_count"] > 0:
                    replace_dominant_scanned += 1
                if alignment["insert_count"] + alignment["delete_count"] > 0:
                    insert_delete_scanned += 1
                if low_conf_sample:
                    low_conf_scanned += 1

                for step in alignment["steps"]:
                    if step["op"] == "replace":
                        pair = f"{token_id_to_char(step['pred_id'], model.tokenizer)}->{token_id_to_char(step['gt_id'], model.tokenizer)}"
                        replace_pair_counter[pair] += 1
                    if step["pred_pos"] is None:
                        continue
                    pred_pos = int(step["pred_pos"])
                    conf = float(pred_conf[pred_pos])
                    if conf < args.tau_low:
                        low_conf_token_count += 1
                    if step["op"] == "correct":
                        correct_token_confs.append(conf)
                        baseline_correct_positions += 1
                    elif step["op"] in {"replace", "delete"}:
                        wrong_token_confs.append(conf)
                        baseline_wrong_positions += 1

                main_match = main_filter_match(
                    filter_mode=args.filter_mode,
                    is_correct=is_correct,
                    alignment=alignment,
                    low_conf_sample=low_conf_sample,
                    meta=batch["meta"][row_idx],
                    gt_length=gt_length,
                    pred_length=alignment["pred_length"],
                    min_replace_count=args.min_replace_count,
                    max_insert_delete_count=args.max_insert_delete_count,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    vocabulary_type=args.vocabulary_type,
                    quality=args.quality,
                    structure_type=args.structure_type,
                    confusion_info=confusion_info,
                    tokenizer=model.tokenizer,
                    confusion_topk=args.confusion_topk,
                )
                record = build_record(
                    resolved_subset=resolved_subset,
                    batch=batch,
                    row_idx=row_idx,
                    gt_text=gt_text,
                    pred_text=pred_text,
                    valid_length=valid_length,
                    eos_position=eos_position,
                    alignment=alignment,
                    feature_shard=feature_shard.name,
                    feature_index=feature_index,
                    low_conf_sample=low_conf_sample,
                )
                feature_bundle = {
                    "pred_ids": pred_ids,
                    "gt_ids": gt_token_ids[row_idx],
                    "pred_conf": pred_conf,
                    "topk_indices": topk_indices[row_idx].numpy(),
                    "topk_values": topk_values[row_idx].numpy(),
                    "decoder_hidden": decoder_hidden[row_idx] if decoder_hidden is not None else None,
                    "encoder_memory": encoder_memory[row_idx] if encoder_memory is not None else None,
                    "valid_length": valid_length,
                    "eos_position": eos_position,
                }

                can_take_main = len(main_rows) < target_main_export
                if not is_correct:
                    can_take_main = can_take_main and exported_wrong_count < target_wrong_export
                if main_match and can_take_main:
                    main_rows.append((record, feature_bundle))
                    if not is_correct:
                        exported_wrong_count += 1
                    feature_index += 1
                elif (
                    args.save_correct_context
                    and is_correct
                    and args.include_correct_ratio > 0
                    and len(context_rows) < target_correct_context
                ):
                    context_rows.append((record, feature_bundle))
                    feature_index += 1

                enough_main = len(main_rows) >= target_main_export
                enough_context = len(context_rows) >= target_correct_context
                if enough_main and enough_context:
                    break
            if total_scanned >= scan_limit or (len(main_rows) >= target_main_export and len(context_rows) >= target_correct_context):
                break

    selected_rows = main_rows + context_rows
    selected_rows = selected_rows[: args.max_export]
    if not selected_rows:
        raise RuntimeError("Exporter did not collect any samples. Increase scan-limit or relax filter-mode.")
    for export_index, (record, feature_bundle) in enumerate(selected_rows):
        record["feature_index"] = export_index
        manifest_rows.append(record)
        append_feature_rows(
            feature_rows=feature_rows,
            pred_ids=feature_bundle["pred_ids"],
            gt_ids=feature_bundle["gt_ids"],
            pred_conf=feature_bundle["pred_conf"],
            topk_indices=feature_bundle["topk_indices"],
            topk_values=feature_bundle["topk_values"],
            valid_length=feature_bundle["valid_length"],
            eos_position=feature_bundle["eos_position"],
            decoder_hidden_row=feature_bundle["decoder_hidden"],
            encoder_memory_row=feature_bundle["encoder_memory"],
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    feature_payload: Dict[str, Any] = {
        "pred_token_ids": np.stack(feature_rows["pred_token_ids"], axis=0).astype(np.int16),
        "gt_token_ids": np.stack(feature_rows["gt_token_ids"], axis=0).astype(np.int16),
        "pred_token_conf": np.stack(feature_rows["pred_token_conf"], axis=0).astype(np.float16),
        "topk_indices": np.stack(feature_rows["topk_indices"], axis=0).astype(np.int16),
        "topk_values": np.stack(feature_rows["topk_values"], axis=0).astype(np.float16),
        "valid_length": np.asarray(feature_rows["valid_length"], dtype=np.int16),
        "eos_position": np.asarray(feature_rows["eos_position"], dtype=np.int16),
    }
    if feature_rows["decoder_hidden"]:
        feature_payload["decoder_hidden"] = np.stack(feature_rows["decoder_hidden"], axis=0).astype(np.float16)
    if feature_rows["encoder_memory"]:
        feature_payload["encoder_memory"] = np.stack(feature_rows["encoder_memory"], axis=0).astype(np.float16)
    np.savez_compressed(feature_shard, **feature_payload)

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "requested_subset": args.subset,
        "resolved_subset": resolved_subset,
        "dataset_root": dataset_root,
        "device": str(device),
        "filter_mode": args.filter_mode,
        "slice_name": args.slice_name,
        "scan_limit": args.scan_limit,
        "max_export": args.max_export,
        "max_wrong": args.max_wrong,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "vocabulary_type": args.vocabulary_type,
        "quality": args.quality,
        "structure_type": args.structure_type,
        "confusion_table": args.confusion_table,
        "confusion_topk": args.confusion_topk,
        "include_correct_ratio": args.include_correct_ratio,
        "total_scanned": total_scanned,
        "total_exported": len(manifest_rows),
        "baseline_correct_samples": baseline_correct_samples,
        "baseline_incorrect_samples": baseline_incorrect_samples,
        "replace_only_samples": replace_only_scanned,
        "replace_dominant_samples": replace_dominant_scanned,
        "insert_delete_samples": insert_delete_scanned,
        "low_conf_samples": low_conf_scanned,
        "correct_context_samples": sum(row["is_correct"] for row in manifest_rows),
        "manifest_path": str(manifest_path),
        "feature_shard_path": str(feature_shard),
        "export_decoder_hidden": args.export_decoder_hidden,
        "export_encoder_memory": args.export_encoder_memory,
        "feature_summaries": [summarize_array(name, value) for name, value in feature_payload.items()],
        "op_counts": dict(op_counter),
        "top_replace_pairs": [{"pair": pair, "count": count} for pair, count in replace_pair_counter.most_common(10)],
        "confidence_stats": {
            "correct_token_conf_mean": safe_mean(correct_token_confs),
            "wrong_token_conf_mean": safe_mean(wrong_token_confs),
            "low_conf_token_count": int(low_conf_token_count),
        },
        "length_stats": {
            "gt_len_mean": safe_mean(gt_lengths),
            "pred_len_mean": safe_mean(pred_lengths),
            "long_21plus_count": long_21plus_count,
        },
        "metadata_available": metadata_available_count > 0,
        "metadata_available_count": metadata_available_count,
        "missing_fields": sorted([field for field, count in metadata_missing_fields.items() if count > 0]),
        "originally_wrong_positions": int(baseline_wrong_positions),
        "originally_correct_positions": int(baseline_correct_positions),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    close_env(dataset)


if __name__ == "__main__":
    main()
