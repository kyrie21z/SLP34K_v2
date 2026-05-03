#!/usr/bin/env python
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import numpy as np

from tools.mdiff_corrector_utils import (
    build_confusion_knowledge,
    load_confusion_table,
    load_feature_shards,
    load_manifest,
    record_arrays,
    token_id_to_char,
)


def bool_arg(value: str) -> bool:
    return value.lower() == "true"


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def metadata_matches(meta: Optional[Dict[str, Any]], vocabulary_type: Optional[str], quality: Optional[str], structure_type: Optional[str]) -> bool:
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


def length_matches(record: Dict[str, Any], min_length: Optional[int], max_length: Optional[int]) -> bool:
    seq_len = max(int(record.get("valid_length", 0)), len(record["gt_text"]), len(record["pred_text"]))
    if min_length is not None and seq_len < min_length:
        return False
    if max_length is not None and seq_len > max_length:
        return False
    return True


def hard_slice_match(record: Dict[str, Any]) -> bool:
    meta = record.get("metadata") or {}
    seq_len = max(int(record.get("valid_length", 0)), len(record["gt_text"]), len(record["pred_text"]))
    if seq_len >= 21 or record.get("low_conf_sample", False):
        return True
    if meta.get("vocabulary_type") == "OOV":
        return True
    if meta.get("resolution_type") == "low":
        return True
    if meta.get("structure_type") in {"vertical", "multi-lines"}:
        return True
    if meta.get("quality") in {"low", "hard"}:
        return True
    return False


def confusion_pair_match(record: Dict[str, Any], confusion_pairs: set[Tuple[str, str]]) -> bool:
    if record.get("is_correct", False):
        return False
    if not confusion_pairs:
        return False
    steps = record.get("alignment_steps")
    if steps:
        for step in steps:
            if step.get("op") == "replace" and (step.get("pred_token"), step.get("gt_token")) in confusion_pairs:
                return True
    ops = record.get("alignment_ops", [])
    if "replace" not in ops:
        return False
    summary = record.get("alignment_summary", {})
    return int(summary.get("replace_count", 0)) > 0


def filter_match(record: Dict[str, Any], filter_mode: str, confusion_pairs: set[Tuple[str, str]]) -> bool:
    summary = record.get("alignment_summary", {})
    replace_count = int(summary.get("replace_count", 0))
    insert_delete_count = int(summary.get("insert_count", 0)) + int(summary.get("delete_count", 0))
    is_correct = bool(record.get("is_correct", False))
    if filter_mode == "all":
        return True
    if filter_mode == "incorrect":
        return not is_correct
    if filter_mode == "replace_only":
        return (not is_correct) and bool(summary.get("replace_only_candidate", False)) and replace_count > 0
    if filter_mode == "replace_dominant":
        return (not is_correct) and replace_count > 0 and replace_count > insert_delete_count
    if filter_mode == "low_conf":
        return bool(record.get("low_conf_sample", False))
    if filter_mode == "hard_slice":
        return hard_slice_match(record)
    if filter_mode == "confusion_pair":
        return confusion_pair_match(record, confusion_pairs)
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def sample_context_rows(rows: List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]], include_correct_ratio: float, seed: int) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    if include_correct_ratio <= 0:
        return [row for row in rows if not row[0]["is_correct"]]
    wrong_rows = [row for row in rows if not row[0]["is_correct"]]
    correct_rows = [row for row in rows if row[0]["is_correct"]]
    if not correct_rows:
        return wrong_rows
    target_correct = int(round(len(wrong_rows) * include_correct_ratio / max(1e-8, 1.0 - include_correct_ratio)))
    rng = random.Random(seed)
    rng.shuffle(correct_rows)
    return wrong_rows + correct_rows[:target_correct]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--filter-mode", choices=["all", "incorrect", "replace_only", "replace_dominant", "low_conf", "hard_slice", "confusion_pair"], default="incorrect")
    parser.add_argument("--slice-name", default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--vocabulary-type", default=None)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--structure-type", default=None)
    parser.add_argument("--confusion-table", default=None)
    parser.add_argument("--confusion-topk", type=int, default=20)
    parser.add_argument("--max-export", type=int, default=None)
    parser.add_argument("--include-correct-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = load_manifest(args.cache_dir)
    shards = load_feature_shards(args.cache_dir)
    confusion_pairs: set[Tuple[str, str]] = set()
    if args.confusion_table:
        rows = load_confusion_table(args.confusion_table)[: args.confusion_topk]
        confusion_pairs = {(str(row["pred_token"]), str(row["gt_token"])) for row in rows}
        _ = build_confusion_knowledge(rows)

    selected_rows: List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]] = []
    metadata_available_count = 0
    missing_fields = Counter()
    replace_pair_counter = Counter()
    gt_lengths: List[int] = []
    pred_lengths: List[int] = []

    for record in records:
        meta = record.get("metadata")
        if meta is not None:
            metadata_available_count += 1
            for field in ["quality", "vocabulary_type", "resolution_type", "structure_type"]:
                if field not in meta:
                    missing_fields[field] += 1
        if not metadata_matches(meta, args.vocabulary_type, args.quality, args.structure_type):
            continue
        if not length_matches(record, args.min_length, args.max_length):
            continue
        if not filter_match(record, args.filter_mode, confusion_pairs):
            continue
        arrays = record_arrays(record, shards)
        selected_rows.append((record, arrays))
        gt_lengths.append(len(record["gt_text"]))
        pred_lengths.append(len(record["pred_text"]))
        if not record["is_correct"]:
            pred_text = record["pred_text"]
            gt_text = record["gt_text"]
            for idx, (pred_char, gt_char) in enumerate(zip(pred_text, gt_text)):
                if pred_char != gt_char:
                    replace_pair_counter[f"{pred_char}->{gt_char}"] += 1

    selected_rows = sample_context_rows(selected_rows, args.include_correct_ratio, args.seed)
    if args.max_export is not None:
        selected_rows = selected_rows[: args.max_export]
    if not selected_rows:
        raise RuntimeError("No rows matched the requested slice filters")

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    feature_path = output_dir / "features_0000.npz"
    feature_rows: Dict[str, List[np.ndarray]] = defaultdict(list)
    manifest_rows: List[Dict[str, Any]] = []
    wrong_positions = 0
    correct_context_count = 0
    for new_index, (record, arrays) in enumerate(selected_rows):
        copied = dict(record)
        copied["feature_shard"] = feature_path.name
        copied["feature_index"] = new_index
        manifest_rows.append(copied)
        if copied["is_correct"]:
            correct_context_count += 1
        summary = copied.get("alignment_summary", {})
        wrong_positions += int(summary.get("replace_count", 0))
        for key, value in arrays.items():
            feature_rows[key].append(value)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(feature_path, **{key: np.stack(values, axis=0) for key, values in feature_rows.items()})

    summary = {
        "source_cache_dir": str(Path(args.cache_dir).absolute()),
        "output_dir": str(output_dir),
        "slice_name": args.slice_name,
        "filter_mode": args.filter_mode,
        "total_scanned": len(records),
        "total_exported": len(manifest_rows),
        "correct_context_samples": correct_context_count,
        "replace_positions": wrong_positions,
        "metadata_available": metadata_available_count > 0,
        "metadata_available_count": metadata_available_count,
        "missing_fields": sorted([field for field, count in missing_fields.items() if count > 0]),
        "length_stats": {
            "gt_len_mean": safe_mean(gt_lengths),
            "pred_len_mean": safe_mean(pred_lengths),
            "long_21plus_count": sum(max(len(row["gt_text"]), len(row["pred_text"]), int(row.get("valid_length", 0))) >= 21 for row in manifest_rows),
        },
        "top_replace_pairs": [{"pair": pair, "count": count} for pair, count in replace_pair_counter.most_common(10)],
        "feature_summaries": {key: list(np.stack(values, axis=0).shape) for key, values in feature_rows.items()},
        "vocabulary_type": args.vocabulary_type,
        "quality": args.quality,
        "structure_type": args.structure_type,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "confusion_table": args.confusion_table,
        "confusion_topk": args.confusion_topk,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
