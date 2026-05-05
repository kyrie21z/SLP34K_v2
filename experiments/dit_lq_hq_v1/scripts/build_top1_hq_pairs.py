#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb


EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
QUALITY_PRIORITY = {"easy": 3, "middle": 2, "hard": 1}

SAMPLES_CSV_FIELDS = [
    "lmdb_root",
    "lmdb_index",
    "label",
    "quality",
    "quality_priority",
    "structure",
    "structure_type",
    "source_path",
    "raw_label",
    "ocr_pred",
    "ocr_correct",
    "confidence",
    "avg_conf",
    "min_conf",
    "pred_length",
    "label_length",
]

PAIR_CSV_FIELDS = [
    "pair_id",
    "label",
    "group_size",
    "lq_lmdb_root",
    "lq_lmdb_index",
    "lq_quality",
    "lq_quality_priority",
    "lq_structure",
    "lq_structure_type",
    "lq_ocr_pred",
    "lq_ocr_correct",
    "lq_confidence",
    "lq_avg_conf",
    "lq_min_conf",
    "lq_source_path",
    "hq_lmdb_root",
    "hq_lmdb_index",
    "hq_quality",
    "hq_quality_priority",
    "hq_structure",
    "hq_structure_type",
    "hq_ocr_pred",
    "hq_ocr_correct",
    "hq_confidence",
    "hq_avg_conf",
    "hq_min_conf",
    "hq_source_path",
    "quality_relation",
    "structure_relation",
    "pair_type",
]


def parse_args():
    parser = argparse.ArgumentParser(description="构造 train sample manifest 与 same-label top1-HQ pair manifest")
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--ocr-csv", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--pair-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def validate_args(args):
    lmdb_root = args.lmdb_root.resolve()
    ocr_csv = args.ocr_csv.resolve()
    samples_csv = args.samples_csv.resolve()
    samples_jsonl = args.samples_jsonl.resolve()
    pair_csv = args.pair_csv.resolve()
    report = args.report.resolve()

    if not lmdb_root.exists():
        raise FileNotFoundError(f"LMDB 不存在: {lmdb_root}")
    if not ocr_csv.exists():
        raise FileNotFoundError(f"OCR CSV 不存在: {ocr_csv}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")

    for output_path in [samples_csv, samples_jsonl, pair_csv, report]:
        if not is_relative_to(output_path, SAFE_OUTPUT_ROOT):
            raise ValueError(f"输出路径必须位于 {SAFE_OUTPUT_ROOT} 下: {output_path}")
        if is_relative_to(output_path, FORBIDDEN_OUTPUT_ROOT):
            raise ValueError(f"输出路径不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {output_path}")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"输出文件已存在，未提供 --overwrite: {output_path}")
    return lmdb_root, ocr_csv, samples_csv, samples_jsonl, pair_csv, report


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def as_int(value: Any) -> int:
    return int(str(value).strip())


def as_float(value: Any) -> float:
    return float(str(value).strip())


def resolve_ocr_fields(fieldnames: List[str]) -> Dict[str, str]:
    candidates = {
        "lmdb_index": ["lmdb_index"],
        "label": ["label", "gt", "raw_label"],
        "pred": ["pred", "ocr_pred"],
        "correct": ["correct", "ocr_correct"],
        "confidence": ["confidence", "avg_conf"],
        "avg_conf": ["avg_conf", "confidence"],
        "min_conf": ["min_conf"],
        "pred_length": ["pred_length"],
        "label_length": ["label_length"],
    }
    resolved = {}
    for key, names in candidates.items():
        for name in names:
            if name in fieldnames:
                resolved[key] = name
                break
        if key not in resolved:
            raise ValueError(f"OCR CSV 缺少字段 {key}，现有字段: {fieldnames}")
    return resolved


def load_ocr_rows(ocr_csv: Path):
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    sample_count = 0
    with ocr_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("OCR CSV 没有表头")
        resolved = resolve_ocr_fields(reader.fieldnames)
        for row in reader:
            sample_count += 1
            lmdb_index = as_int(row[resolved["lmdb_index"]])
            if lmdb_index in rows_by_index:
                raise ValueError(f"OCR CSV 中存在重复 lmdb_index: {lmdb_index}")
            rows_by_index[lmdb_index] = {
                "lmdb_index": lmdb_index,
                "label": row[resolved["label"]],
                "pred": row[resolved["pred"]],
                "correct": parse_bool(row[resolved["correct"]]),
                "confidence": as_float(row[resolved["confidence"]]),
                "avg_conf": as_float(row[resolved["avg_conf"]]),
                "min_conf": as_float(row[resolved["min_conf"]]),
                "pred_length": as_int(row[resolved["pred_length"]]),
                "label_length": as_int(row[resolved["label_length"]]),
            }
    return rows_by_index, sample_count, resolved


def read_lmdb_rows(lmdb_root: Path, limit: Optional[int]):
    env = lmdb.open(str(lmdb_root), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = txn.get(b"num-samples")
            if raw is None:
                raise ValueError("LMDB 缺少 num-samples")
            num_samples = int(raw.decode("utf-8"))
            max_index = num_samples if limit is None else min(limit, num_samples)
            rows = []
            for idx in range(1, max_index + 1):
                label_raw = txn.get(f"label-{idx:09d}".encode("utf-8"))
                meta_raw = txn.get(f"meta-{idx:09d}".encode("utf-8"))
                if label_raw is None or meta_raw is None:
                    raise ValueError(f"LMDB 缺少 label/meta 键: {idx}")
                label = label_raw.decode("utf-8")
                meta = json.loads(meta_raw.decode("utf-8"))
                rows.append({"lmdb_index": idx, "label": label, "metadata": meta})
    finally:
        env.close()
    return num_samples, rows


def merge_rows(
    lmdb_root: Path,
    lmdb_rows: List[Dict[str, Any]],
    ocr_rows_by_index: Dict[int, Dict[str, Any]],
):
    merged = []
    consistency = {
        "metadata_id_mismatch": 0,
        "metadata_raw_label_mismatch": 0,
        "metadata_split_mismatch": 0,
        "missing_quality": 0,
        "missing_structure": 0,
        "missing_structure_type": 0,
        "ocr_missing_index": 0,
        "ocr_label_mismatch": 0,
    }
    for row in lmdb_rows:
        idx = row["lmdb_index"]
        label = row["label"]
        meta = row["metadata"]
        if meta.get("id") != idx:
            consistency["metadata_id_mismatch"] += 1
            raise ValueError(f"metadata.id 与 lmdb_index 不一致: {idx} vs {meta.get('id')}")
        if meta.get("raw_label") != label:
            consistency["metadata_raw_label_mismatch"] += 1
            raise ValueError(f"metadata.raw_label 与 label 不一致: {idx}")
        if meta.get("split") != "train":
            consistency["metadata_split_mismatch"] += 1
            raise ValueError(f"metadata.split 不是 train: {idx} -> {meta.get('split')}")
        if not meta.get("quality"):
            consistency["missing_quality"] += 1
            raise ValueError(f"metadata 缺少 quality: {idx}")
        if not meta.get("structure"):
            consistency["missing_structure"] += 1
            raise ValueError(f"metadata 缺少 structure: {idx}")
        if not meta.get("structure_type"):
            consistency["missing_structure_type"] += 1
            raise ValueError(f"metadata 缺少 structure_type: {idx}")
        if idx not in ocr_rows_by_index:
            consistency["ocr_missing_index"] += 1
            raise ValueError(f"OCR CSV 中缺少 lmdb_index={idx}")
        ocr = ocr_rows_by_index[idx]
        if ocr["label"] != label:
            consistency["ocr_label_mismatch"] += 1
            raise ValueError(f"OCR CSV 的 label 与 LMDB 不一致: {idx}")
        quality = meta["quality"]
        if quality not in QUALITY_PRIORITY:
            raise ValueError(f"未知 quality: {quality} (lmdb_index={idx})")
        merged.append(
            {
                "lmdb_root": str(lmdb_root),
                "lmdb_index": idx,
                "label": label,
                "quality": quality,
                "quality_priority": QUALITY_PRIORITY[quality],
                "structure": meta["structure"],
                "structure_type": meta["structure_type"],
                "source_path": meta["source_path"],
                "raw_label": meta["raw_label"],
                "ocr_pred": ocr["pred"],
                "ocr_correct": ocr["correct"],
                "confidence": ocr["confidence"],
                "avg_conf": ocr["avg_conf"],
                "min_conf": ocr["min_conf"],
                "pred_length": ocr["pred_length"],
                "label_length": ocr["label_length"],
                "metadata": meta,
            }
        )
    return merged, consistency


def sort_group(rows: List[Dict[str, Any]]):
    return sorted(
        rows,
        key=lambda row: (
            -row["quality_priority"],
            -int(row["ocr_correct"]),
            -row["confidence"],
            -row["min_conf"],
            row["lmdb_index"],
        ),
    )


def build_pairs(samples: List[Dict[str, Any]]):
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups[row["label"]].append(row)

    sorted_groups = {label: sort_group(rows) for label, rows in groups.items()}
    pairs = []
    hq_rows = []
    pair_id = 1
    for label, rows in sorted_groups.items():
        if not rows:
            continue
        hq = rows[0]
        hq_rows.append(hq)
        if len(rows) == 1:
            continue
        for lq in rows[1:]:
            if lq["label"] != hq["label"]:
                raise ValueError(f"pair label 不一致: {lq['label']} vs {hq['label']}")
            pair = {
                "pair_id": pair_id,
                "label": label,
                "group_size": len(rows),
                "lq_lmdb_root": lq["lmdb_root"],
                "lq_lmdb_index": lq["lmdb_index"],
                "lq_quality": lq["quality"],
                "lq_quality_priority": lq["quality_priority"],
                "lq_structure": lq["structure"],
                "lq_structure_type": lq["structure_type"],
                "lq_ocr_pred": lq["ocr_pred"],
                "lq_ocr_correct": lq["ocr_correct"],
                "lq_confidence": lq["confidence"],
                "lq_avg_conf": lq["avg_conf"],
                "lq_min_conf": lq["min_conf"],
                "lq_source_path": lq["source_path"],
                "hq_lmdb_root": hq["lmdb_root"],
                "hq_lmdb_index": hq["lmdb_index"],
                "hq_quality": hq["quality"],
                "hq_quality_priority": hq["quality_priority"],
                "hq_structure": hq["structure"],
                "hq_structure_type": hq["structure_type"],
                "hq_ocr_pred": hq["ocr_pred"],
                "hq_ocr_correct": hq["ocr_correct"],
                "hq_confidence": hq["confidence"],
                "hq_avg_conf": hq["avg_conf"],
                "hq_min_conf": hq["min_conf"],
                "hq_source_path": hq["source_path"],
                "quality_relation": f"{lq['quality']}_to_{hq['quality']}",
                "structure_relation": "same_structure" if lq["structure"] == hq["structure"] else "cross_structure",
                "pair_type": "same_label_top1_hq",
            }
            pairs.append(pair)
            pair_id += 1
    return groups, sorted_groups, hq_rows, pairs


def write_samples(samples: List[Dict[str, Any]], samples_csv: Path, samples_jsonl: Path):
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    with samples_csv.open("w", encoding="utf-8", newline="") as csv_fp, samples_jsonl.open("w", encoding="utf-8") as jsonl_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=SAMPLES_CSV_FIELDS)
        writer.writeheader()
        for row in samples:
            writer.writerow({key: row[key] for key in SAMPLES_CSV_FIELDS})
            jsonl_fp.write(
                json.dumps(
                    {
                        "lmdb_root": row["lmdb_root"],
                        "lmdb_index": row["lmdb_index"],
                        "label": row["label"],
                        "quality": row["quality"],
                        "quality_priority": row["quality_priority"],
                        "structure": row["structure"],
                        "structure_type": row["structure_type"],
                        "source_path": row["source_path"],
                        "raw_label": row["raw_label"],
                        "ocr": {
                            "pred": row["ocr_pred"],
                            "correct": row["ocr_correct"],
                            "confidence": row["confidence"],
                            "avg_conf": row["avg_conf"],
                            "min_conf": row["min_conf"],
                            "pred_length": row["pred_length"],
                            "label_length": row["label_length"],
                        },
                        "metadata": row["metadata"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_pairs(pairs: List[Dict[str, Any]], pair_csv: Path):
    pair_csv.parent.mkdir(parents=True, exist_ok=True)
    with pair_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=PAIR_CSV_FIELDS)
        writer.writeheader()
        for row in pairs:
            writer.writerow({key: row[key] for key in PAIR_CSV_FIELDS})


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], headers: Optional[Dict[str, str]] = None):
    if not rows:
        return "无"
    headers = headers or {}
    aligns = []
    for col in columns:
        if col in {"label", "quality_relation", "structure_relation", "matrix_key", "group_size", "quality", "structure", "structure_type"}:
            aligns.append("---")
        else:
            aligns.append("---:")
    lines = [
        "| " + " | ".join(headers.get(col, col) for col in columns) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def counter_rows(counter: Counter, key_name: str):
    return [{key_name: key, "count": value} for key, value in counter.items()]


def summarize_groups(groups: Dict[str, List[Dict[str, Any]]]):
    sizes = [len(rows) for rows in groups.values()]
    size_counter = Counter(sizes)
    return {
        "unique_labels": len(groups),
        "size_eq_1": sum(1 for x in sizes if x == 1),
        "size_ge_2": sum(1 for x in sizes if x >= 2),
        "size_ge_3": sum(1 for x in sizes if x >= 3),
        "max_group_size": max(sizes) if sizes else 0,
        "mean_group_size": statistics.fmean(sizes) if sizes else 0.0,
        "median_group_size": statistics.median(sizes) if sizes else 0.0,
        "pairable_count": sum(size - 1 for size in sizes if size >= 2),
        "size_counter": size_counter,
    }


def build_structure_matrix(pairs: List[Dict[str, Any]]):
    counter = Counter()
    for row in pairs:
        counter[f"{row['lq_structure']}->{row['hq_structure']}"] += 1
    return counter


def sample_pair_rows(rows: List[Dict[str, Any]], limit: int):
    sampled = []
    for row in rows[:limit]:
        sampled.append(
            {
                "label": row["label"],
                "lq_index": row["lq_lmdb_index"],
                "hq_index": row["hq_lmdb_index"],
                "lq_quality": row["lq_quality"],
                "hq_quality": row["hq_quality"],
                "lq_structure": row["lq_structure"],
                "hq_structure": row["hq_structure"],
                "lq_confidence": row["lq_confidence"],
                "hq_confidence": row["hq_confidence"],
            }
        )
    return sampled


def write_report(
    report: Path,
    inputs_outputs: Dict[str, str],
    consistency: Dict[str, Any],
    samples: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    sorted_groups: Dict[str, List[Dict[str, Any]]],
    hq_rows: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    ocr_csv_fields: Dict[str, str],
    limit: Optional[int],
):
    quality_counter = Counter(row["quality"] for row in samples)
    structure_counter = Counter(row["structure"] for row in samples)
    structure_type_counter = Counter(row["structure_type"] for row in samples)
    ocr_correct_counter = Counter(bool(row["ocr_correct"]) for row in samples)
    confidence_values = [row["confidence"] for row in samples]
    group_summary = summarize_groups(groups)
    quality_relation_counter = Counter(row["quality_relation"] for row in pairs)
    structure_relation_counter = Counter(row["structure_relation"] for row in pairs)
    structure_matrix_counter = build_structure_matrix(pairs)
    lq_quality_counter = Counter(row["lq_quality"] for row in pairs)
    hq_quality_counter = Counter(row["hq_quality"] for row in pairs)
    lq_structure_counter = Counter(row["lq_structure"] for row in pairs)
    hq_structure_counter = Counter(row["hq_structure"] for row in pairs)
    hq_wrong = [row for row in hq_rows if not row["ocr_correct"]]
    hq_conf_values = [row["confidence"] for row in hq_rows]
    hard_to_easy_pairs = [row for row in pairs if row["quality_relation"] == "hard_to_easy"]
    cross_structure_pairs = [row for row in pairs if row["structure_relation"] == "cross_structure"]
    single_instance_labels = [label for label, rows in sorted_groups.items() if len(rows) == 1]

    report_text = f"""# Same-label Top1-HQ Pair 构造报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建脚本、manifest 和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `{inputs_outputs["lmdb_root"]}`
- 输入 OCR CSV: `{inputs_outputs["ocr_csv"]}`
- 输出 `train_samples_meta.csv`: `{inputs_outputs["samples_csv"]}`
- 输出 `train_samples_meta.jsonl`: `{inputs_outputs["samples_jsonl"]}`
- 输出 `pair_manifest_top1_hq.csv`: `{inputs_outputs["pair_csv"]}`
- 输出报告: `{inputs_outputs["report"]}`
- `--limit`: `{limit}`

## 3. 数据读取与一致性检查

- LMDB `num-samples`: `{consistency["lmdb_num_samples"]}`
- OCR CSV 样本数: `{consistency["ocr_csv_num_rows"]}`
- 实际处理样本数: `{len(samples)}`
- OCR CSV `lmdb_index` 唯一性: `{consistency["ocr_index_unique"]}`
- LMDB `label` 与 OCR CSV `label` 一致性: `{consistency["ocr_label_match"]}`
- `metadata.id == lmdb_index`: `{consistency["meta_id_match"]}`
- `metadata.raw_label == label`: `{consistency["meta_raw_label_match"]}`
- `metadata.split == train`: `{consistency["meta_split_train"]}`
- `quality/structure/structure_type` 非空: `{consistency["meta_required_fields_ok"]}`
- OCR CSV 实际字段映射: `{json.dumps(ocr_csv_fields, ensure_ascii=False)}`

## 4. HQ 选择规则

同一 `label` group 内排序规则：

1. `quality_priority` 降序：`easy=3 > middle=2 > hard=1`
2. `ocr_correct` 降序：`True > False`
3. `confidence` 降序
4. `min_conf` 降序
5. `lmdb_index` 升序

排序后每个 label group 的第一条样本作为 HQ，其余样本作为 LQ。

## 5. Train Sample Manifest 统计

- `num_samples`: `{len(samples)}`
- OCR `correct`: `{ocr_correct_counter.get(True, 0)}`
- OCR `wrong`: `{ocr_correct_counter.get(False, 0)}`
- OCR train accuracy: `{(ocr_correct_counter.get(True, 0) / len(samples)) if samples else 0.0:.6f}`
- confidence 均值: `{statistics.fmean(confidence_values) if confidence_values else 0.0:.6f}`
- confidence 中位数: `{statistics.median(confidence_values) if confidence_values else 0.0:.6f}`

### quality 分布

{markdown_table(counter_rows(quality_counter, "quality"), ["quality", "count"])}

### structure 分布

{markdown_table(counter_rows(structure_counter, "structure"), ["structure", "count"])}

### structure_type 分布

{markdown_table(counter_rows(structure_type_counter, "structure_type"), ["structure_type", "count"])}

## 6. Label Group 统计

- unique label 数: `{group_summary["unique_labels"]}`
- group_size = 1 的 label 数: `{group_summary["size_eq_1"]}`
- group_size >= 2 的 label 数: `{group_summary["size_ge_2"]}`
- group_size >= 3 的 label 数: `{group_summary["size_ge_3"]}`
- 最大 group_size: `{group_summary["max_group_size"]}`
- 平均 group_size: `{group_summary["mean_group_size"]:.6f}`
- 中位数 group_size: `{group_summary["median_group_size"]:.6f}`
- 可构造 pair 数: `{group_summary["pairable_count"]}`
- single-instance label 数: `{len(single_instance_labels)}`

### group_size 分布表

{markdown_table(
    [{"group_size": size, "count": count} for size, count in sorted(group_summary["size_counter"].items())],
    ["group_size", "count"],
)}

## 7. Pair Manifest 统计

- `num_pairs`: `{len(pairs)}`
- `same_structure` pair 数: `{structure_relation_counter.get("same_structure", 0)}`
- `cross_structure` pair 数: `{structure_relation_counter.get("cross_structure", 0)}`
- 每个 label group 只有一个 HQ: `True`
- `sum(group_size - 1)` 校验: `{len(pairs) == group_summary["pairable_count"]}`

### quality_relation 分布

{markdown_table(counter_rows(quality_relation_counter, "quality_relation"), ["quality_relation", "count"])}

### structure pair matrix

{markdown_table(counter_rows(structure_matrix_counter, "matrix_key"), ["matrix_key", "count"], headers={"matrix_key": "structure_pair"})}

### LQ quality 分布

{markdown_table(counter_rows(lq_quality_counter, "quality"), ["quality", "count"])}

### HQ quality 分布

{markdown_table(counter_rows(hq_quality_counter, "quality"), ["quality", "count"])}

### LQ structure 分布

{markdown_table(counter_rows(lq_structure_counter, "structure"), ["structure", "count"])}

### HQ structure 分布

{markdown_table(counter_rows(hq_structure_counter, "structure"), ["structure", "count"])}

## 8. Quality Relation 分布

{markdown_table(counter_rows(quality_relation_counter, "quality_relation"), ["quality_relation", "count"])}

## 9. Structure Relation 分布

{markdown_table(counter_rows(structure_relation_counter, "structure_relation"), ["structure_relation", "count"])}

## 10. HQ 样本分布

- HQ 样本数: `{len(hq_rows)}`
- HQ `ocr_correct=False` 数量: `{len(hq_wrong)}`
- HQ confidence 均值: `{statistics.fmean(hq_conf_values) if hq_conf_values else 0.0:.6f}`
- HQ confidence 最小值: `{min(hq_conf_values) if hq_conf_values else 0.0:.6f}`
- HQ confidence 中位数: `{statistics.median(hq_conf_values) if hq_conf_values else 0.0:.6f}`

### HQ quality 分布

{markdown_table(counter_rows(Counter(row["quality"] for row in hq_rows), "quality"), ["quality", "count"])}

### HQ structure 分布

{markdown_table(counter_rows(Counter(row["structure"] for row in hq_rows), "structure"), ["structure", "count"])}

### 被选为 HQ 的错误 OCR 样本

{markdown_table(
    [
        {
            "label": row["label"],
            "lmdb_index": row["lmdb_index"],
            "quality": row["quality"],
            "structure": row["structure"],
            "confidence": row["confidence"],
            "source_path": row["source_path"],
        }
        for row in hq_wrong
    ],
    ["label", "lmdb_index", "quality", "structure", "confidence", "source_path"],
) if hq_wrong else "无"}

## 11. OCR 置信度统计

- 全样本 confidence 均值: `{statistics.fmean(confidence_values) if confidence_values else 0.0:.6f}`
- 全样本 confidence 中位数: `{statistics.median(confidence_values) if confidence_values else 0.0:.6f}`
- HQ confidence 均值: `{statistics.fmean(hq_conf_values) if hq_conf_values else 0.0:.6f}`
- LQ confidence 均值: `{statistics.fmean([row['lq_confidence'] for row in pairs]) if pairs else 0.0:.6f}`

## 12. 样本示例

### 前 5 个 pair 示例

{markdown_table(sample_pair_rows(pairs, 5), ["label", "lq_index", "hq_index", "lq_quality", "hq_quality", "lq_structure", "hq_structure", "lq_confidence", "hq_confidence"])}

### hard_to_easy pair 示例

{markdown_table(sample_pair_rows(hard_to_easy_pairs, 5), ["label", "lq_index", "hq_index", "lq_quality", "hq_quality", "lq_structure", "hq_structure", "lq_confidence", "hq_confidence"]) if hard_to_easy_pairs else "无"}

### cross_structure pair 示例

{markdown_table(sample_pair_rows(cross_structure_pairs, 5), ["label", "lq_index", "hq_index", "lq_quality", "hq_quality", "lq_structure", "hq_structure", "lq_confidence", "hq_confidence"]) if cross_structure_pairs else "无"}

## 13. 警告与限制

- 当前 top1-HQ 选择使用 `quality priority + OCR confidence`。
- OCR confidence 来自 baseline OCR，会引入 baseline 偏置。
- 同 label pair 不一定是像素对齐 pair。
- cross_structure pair 可能包含布局差异，后续 Diffusion Loss 训练需谨慎分析。
{"- 本次运行使用了 --limit，仅反映子集统计。\n" if limit is not None else ""}

## 14. 下一步建议

下一步可基于 `pair_manifest_top1_hq.csv` 做 pair 质量可视化抽样与统计确认，然后再进入 conditional latent DiT 的 Stage 1 diffusion-loss 预训练准备。
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text, encoding="utf-8")


def write_failure_report(report: Path, error: str):
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Same-label Top1-HQ Pair 构造报告\n\n"
        "## 1. 范围与隔离声明\n\n"
        "本轮只在 `experiments/dit_lq_hq_v1/` 下尝试创建脚本、manifest 和报告。\n"
        "未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。\n\n"
        "## 13. 警告与限制\n\n"
        f"- 任务失败：{error}\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    try:
        lmdb_root, ocr_csv, samples_csv, samples_jsonl, pair_csv, report = validate_args(args)
        ocr_rows_by_index, ocr_csv_num_rows, ocr_csv_fields = load_ocr_rows(ocr_csv)
        lmdb_num_samples, lmdb_rows = read_lmdb_rows(lmdb_root, args.limit)
        samples, merge_consistency = merge_rows(lmdb_root, lmdb_rows, ocr_rows_by_index)
        groups, sorted_groups, hq_rows, pairs = build_pairs(samples)

        write_samples(samples, samples_csv, samples_jsonl)
        write_pairs(pairs, pair_csv)

        consistency = {
            "lmdb_num_samples": lmdb_num_samples,
            "ocr_csv_num_rows": ocr_csv_num_rows,
            "ocr_index_unique": True,
            "ocr_label_match": merge_consistency["ocr_label_mismatch"] == 0,
            "meta_id_match": merge_consistency["metadata_id_mismatch"] == 0,
            "meta_raw_label_match": merge_consistency["metadata_raw_label_mismatch"] == 0,
            "meta_split_train": merge_consistency["metadata_split_mismatch"] == 0,
            "meta_required_fields_ok": (
                merge_consistency["missing_quality"] == 0
                and merge_consistency["missing_structure"] == 0
                and merge_consistency["missing_structure_type"] == 0
            ),
        }
        if args.limit is None and ocr_csv_num_rows != lmdb_num_samples:
            raise ValueError(f"OCR CSV 样本数与 LMDB num-samples 不一致: {ocr_csv_num_rows} != {lmdb_num_samples}")

        write_report(
            report=report,
            inputs_outputs={
                "lmdb_root": str(lmdb_root),
                "ocr_csv": str(ocr_csv),
                "samples_csv": str(samples_csv),
                "samples_jsonl": str(samples_jsonl),
                "pair_csv": str(pair_csv),
                "report": str(report),
            },
            consistency=consistency,
            samples=samples,
            groups=groups,
            sorted_groups=sorted_groups,
            hq_rows=hq_rows,
            pairs=pairs,
            ocr_csv_fields=ocr_csv_fields,
            limit=args.limit,
        )
    except Exception as exc:
        write_failure_report(args.report.resolve(), str(exc))
        raise


if __name__ == "__main__":
    main()
