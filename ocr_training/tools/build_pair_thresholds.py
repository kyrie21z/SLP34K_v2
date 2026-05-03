#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from tools.mdiff_corrector_utils import load_confusion_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion-table", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--default_tau_corr", type=float, default=0.60)
    parser.add_argument("--default_tau_keep", type=float, default=0.80)
    parser.add_argument("--default_delta_gain", type=float, default=0.00)
    args = parser.parse_args()

    rows = load_confusion_table(args.confusion_table)
    pair_rules = {}
    for row in rows:
        pair_key = f"{row['pred_token']}->{row['gt_token']}"
        count = int(row.get("count", 0))
        segment_type = row.get("segment_type", "other")
        tau_corr = args.default_tau_corr
        tau_keep = args.default_tau_keep
        delta_gain = args.default_delta_gain
        if segment_type == "digit" and count >= 10:
            tau_corr = 0.50
        elif segment_type == "alphabet" and count >= 8:
            tau_corr = 0.55
        pair_rules[pair_key] = {
            "tau_corr": tau_corr,
            "tau_keep": tau_keep,
            "delta_gain": delta_gain,
            "count": count,
            "segment_type": segment_type,
        }

    payload = {
        "source": str(Path(args.confusion_table).absolute()),
        "rule": "train confusion-table counts only; digit count>=10 => tau_corr=0.50, alphabet count>=8 => tau_corr=0.55, else defaults",
        "default_thresholds": {
            "tau_corr": args.default_tau_corr,
            "tau_keep": args.default_tau_keep,
            "delta_gain": args.default_delta_gain,
        },
        "pairs": pair_rules,
    }
    output_json = Path(args.output_json).absolute()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "num_pairs": len(pair_rules)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
