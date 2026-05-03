#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from tools.mdiff_corrector_utils import load_confusion_table, load_manifest


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache_dir", required=True)
    parser.add_argument("--confusion_table", required=True)
    parser.add_argument("--pair_stats_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_support", type=int, default=5)
    args = parser.parse_args()

    train_records = load_manifest(args.train_cache_dir)
    confusion_rows = load_confusion_table(args.confusion_table)
    pair_stats_rows = json.loads(Path(args.pair_stats_json).read_text(encoding="utf-8"))

    pair_stats_by_pair: Dict[str, Dict[str, object]] = {
        str(row["pair"]): row for row in pair_stats_rows if int(row.get("support", 0)) >= args.min_support
    }
    max_train_count = max((int(row.get("count", 0)) for row in confusion_rows), default=1)

    rows: List[Dict[str, object]] = []
    for row in confusion_rows:
        pred_token = str(row["pred_token"])
        gt_token = str(row["gt_token"])
        pair_name = f"{pred_token}->{gt_token}"
        pair_stats = pair_stats_by_pair.get(pair_name, {})
        train_count = int(row.get("count", 0))
        eval_support = int(pair_stats.get("support", 0))
        eval_correction_rate = float(pair_stats.get("correction_rate", 0.0))
        oracle1 = float(pair_stats.get("oracle@1", 0.0))
        oracle5 = float(pair_stats.get("oracle@5", 0.0))
        train_count_norm = float(train_count / max_train_count) if max_train_count else 0.0
        oracle_gap = max(0.0, oracle5 - eval_correction_rate)
        difficulty_score = train_count_norm * oracle_gap * (1.0 - eval_correction_rate)
        recommended_weight = clamp(1.0 + 2.0 * difficulty_score, 1.0, 3.0)
        recommended_synthetic_multiplier = clamp(1.0 + 4.0 * difficulty_score, 1.0, 5.0)
        rows.append(
            {
                "pair": pair_name,
                "pred_token_id": int(row["pred_token_id"]),
                "pred_token": pred_token,
                "gt_token_id": int(row["gt_token_id"]),
                "gt_token": gt_token,
                "train_count": train_count,
                "eval_support": eval_support,
                "eval_correction_rate": eval_correction_rate,
                "eval_oracle@1": oracle1,
                "eval_oracle@5": oracle5,
                "segment_type": row.get("segment_type", "other"),
                "difficulty_score": difficulty_score,
                "recommended_weight": recommended_weight,
                "recommended_synthetic_multiplier": recommended_synthetic_multiplier,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["difficulty_score"]),
            -int(row["eval_support"]),
            -int(row["train_count"]),
            str(row["pair"]),
        )
    )

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pair_difficulty.json"
    csv_path = output_dir / "pair_difficulty.csv"
    payload = {
        "analysis_only": True,
        "note": "analysis split only; not official benchmark conclusion",
        "train_cache_dir": str(Path(args.train_cache_dir).absolute()),
        "confusion_table": str(Path(args.confusion_table).absolute()),
        "pair_stats_json": str(Path(args.pair_stats_json).absolute()),
        "min_support": args.min_support,
        "train_sample_count": len(train_records),
        "pairs": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "pred_token_id",
                "pred_token",
                "gt_token_id",
                "gt_token",
                "train_count",
                "eval_support",
                "eval_correction_rate",
                "eval_oracle@1",
                "eval_oracle@5",
                "segment_type",
                "difficulty_score",
                "recommended_weight",
                "recommended_synthetic_multiplier",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output_json": str(json_path),
        "output_csv": str(csv_path),
        "pair_count": len(rows),
        "top_pairs": rows[:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
