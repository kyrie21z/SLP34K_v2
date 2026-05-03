#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from hydra import compose, initialize
from hydra.utils import instantiate

from tools.mdiff_corrector_utils import (
    align_pred_gt,
    classify_token_id,
    load_feature_shards,
    load_manifest,
    record_arrays,
    token_id_to_char,
)


def load_tokenizer():
    with initialize(config_path="../configs", version_base="1.2"):
        cfg = compose(config_name="main", overrides=["model=slp_mdiff_corrector"])
    model = instantiate(cfg.model)
    return model.tokenizer, model.eos_id, model.pad_id


def pair_key(pred_id: int, gt_id: int) -> Tuple[int, int]:
    return int(pred_id), int(gt_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    tokenizer, eos_id, pad_id = load_tokenizer()
    records = load_manifest(args.cache_dir)
    shards = load_feature_shards(args.cache_dir)

    stats: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "conf_sum": 0.0, "pos_sum": 0.0}
    )

    for record in records:
        if record.get("is_correct", False):
            continue
        arrays = record_arrays(record, shards)
        pred_ids = arrays["pred_token_ids"]
        gt_ids = arrays["gt_token_ids"]
        pred_conf = arrays["pred_token_conf"]
        alignment = align_pred_gt(pred_ids, gt_ids, eos_id=eos_id, pad_id=pad_id)
        for step in alignment["steps"]:
            if step["op"] != "replace":
                continue
            pred_id = int(step["pred_id"])
            gt_id = int(step["gt_id"])
            pred_pos = int(step["pred_pos"])
            entry = stats[pair_key(pred_id, gt_id)]
            entry["count"] += 1
            entry["conf_sum"] += float(pred_conf[pred_pos])
            entry["pos_sum"] += float(pred_pos)

    rows: List[Dict[str, object]] = []
    for (pred_id, gt_id), entry in stats.items():
        count = int(entry["count"])
        if count < args.min_count:
            continue
        rows.append(
            {
                "pred_token_id": pred_id,
                "pred_token": token_id_to_char(pred_id, tokenizer),
                "gt_token_id": gt_id,
                "gt_token": token_id_to_char(gt_id, tokenizer),
                "count": count,
                "segment_type": classify_token_id(gt_id, tokenizer),
                "avg_base_conf": entry["conf_sum"] / count,
                "avg_position": entry["pos_sum"] / count,
            }
        )

    rows.sort(key=lambda row: (-int(row["count"]), row["pred_token_id"], row["gt_token_id"]))
    top_rows = rows[: args.top_k]

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "confusion_table.json"
    csv_path = output_dir / "confusion_table.csv"

    payload = {
        "cache_dir": str(Path(args.cache_dir).absolute()),
        "min_count": args.min_count,
        "top_k": args.top_k,
        "num_pairs_total": len(rows),
        "num_pairs_exported": len(top_rows),
        "pairs": rows,
        "top_pairs": top_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pred_token_id",
                "pred_token",
                "gt_token_id",
                "gt_token",
                "count",
                "segment_type",
                "avg_base_conf",
                "avg_position",
            ],
        )
        writer.writeheader()
        writer.writerows(top_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
