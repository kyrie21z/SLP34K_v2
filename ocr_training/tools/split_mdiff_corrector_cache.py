#!/usr/bin/env python
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import numpy as np

from tools.mdiff_corrector_utils import load_feature_shards, load_manifest, record_arrays


def sample_type(record: Dict[str, Any]) -> str:
    return "correct_context" if record["is_correct"] else "incorrect"


def wrong_positions(record: Dict[str, Any]) -> int:
    summary = record.get("alignment_summary", {})
    replace_count = int(summary.get("replace_count", 0))
    insert_count = int(summary.get("insert_count", 0))
    delete_count = int(summary.get("delete_count", 0))
    return replace_count + insert_count + delete_count


def stratify_key(record: Dict[str, Any], mode: str) -> str:
    if mode == "sample_type":
        return sample_type(record)
    if mode == "replace_count":
        if record["is_correct"]:
            return "correct_context"
        replace_count = int(record["alignment_summary"]["replace_count"])
        if replace_count <= 1:
            return "replace_1"
        if replace_count == 2:
            return "replace_2"
        return "replace_3plus"
    if mode == "length":
        valid_length = int(record["valid_length"])
        if valid_length < 10:
            return "len_lt10"
        if valid_length < 21:
            return "len_10_20"
        return "len_21plus"
    raise ValueError(f"Unsupported stratify mode: {mode}")


def build_split_records(records: List[Dict[str, Any]], train_ratio: float, seed: int, stratify_mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[stratify_key(record, stratify_mode)].append(record)
    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    for bucket_records in buckets.values():
        bucket_records = list(bucket_records)
        rng.shuffle(bucket_records)
        if len(bucket_records) == 1:
            target_train = 1 if not train_records else 0
        else:
            target_train = int(round(len(bucket_records) * train_ratio))
            target_train = min(max(target_train, 1), len(bucket_records) - 1)
        train_records.extend(bucket_records[:target_train])
        eval_records.extend(bucket_records[target_train:])
    return train_records, eval_records


def export_split(split_dir: Path, records: List[Dict[str, Any]], shards: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    feature_path = split_dir / "features_0000.npz"
    sorted_records = sorted(records, key=lambda row: row["sample_id"])
    feature_rows: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_ids = set()
    wrong_position_count = 0
    correct_context_count = 0
    for new_index, record in enumerate(sorted_records):
        copied = dict(record)
        copied["feature_shard"] = feature_path.name
        copied["feature_index"] = new_index
        sample_ids.add(copied["sample_id"])
        if copied["is_correct"]:
            correct_context_count += 1
        wrong_position_count += wrong_positions(copied)
        arrays = record_arrays(record, shards)
        for key, value in arrays.items():
            feature_rows[key].append(value)
        sorted_records[new_index] = copied
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in sorted_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(feature_path, **{key: np.stack(values, axis=0) for key, values in feature_rows.items()})
    return {
        "manifest_path": str(manifest_path),
        "feature_path": str(feature_path),
        "sample_count": len(sorted_records),
        "wrong_positions": wrong_position_count,
        "correct_context_count": correct_context_count,
        "sample_ids": sample_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stratify", choices=["sample_type", "replace_count", "length"], default="sample_type")
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be in (0, 1)")
    records = load_manifest(args.cache_dir)
    if len(records) < 2:
        raise RuntimeError("Need at least 2 samples to split cache")
    shards = load_feature_shards(args.cache_dir)
    train_records, eval_records = build_split_records(records, args.train_ratio, args.seed, args.stratify)
    if not train_records or not eval_records:
        raise RuntimeError("Split produced empty train or eval set")
    train_dir = Path(args.output_dir).absolute() / "train"
    eval_dir = Path(args.output_dir).absolute() / "eval"
    train_summary = export_split(train_dir, train_records, shards)
    eval_summary = export_split(eval_dir, eval_records, shards)
    overlap = sorted(train_summary["sample_ids"].intersection(eval_summary["sample_ids"]))
    summary = {
        "cache_dir": str(Path(args.cache_dir).absolute()),
        "output_dir": str(Path(args.output_dir).absolute()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "stratify": args.stratify,
        "train_sample_count": train_summary["sample_count"],
        "eval_sample_count": eval_summary["sample_count"],
        "train_wrong_positions": train_summary["wrong_positions"],
        "eval_wrong_positions": eval_summary["wrong_positions"],
        "train_correct_context_count": train_summary["correct_context_count"],
        "eval_correct_context_count": eval_summary["correct_context_count"],
        "sample_id_overlap": len(overlap),
        "overlap_examples": overlap[:5],
        "train_manifest_path": train_summary["manifest_path"],
        "eval_manifest_path": eval_summary["manifest_path"],
        "train_feature_path": train_summary["feature_path"],
        "eval_feature_path": eval_summary["feature_path"],
    }
    (Path(args.output_dir).absolute() / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
