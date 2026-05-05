#!/usr/bin/env python3
import argparse
import csv
import io
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
BUCKETS = [
    "same_structure",
    "cross_structure",
    "quality_improved",
    "same_structure_quality_improved",
    "hard_to_easy",
    "hard_to_middle",
    "middle_to_easy",
    "hard_to_hard",
    "hq_ocr_wrong",
    "large_group_examples",
    "random_all",
]


def parse_args():
    parser = argparse.ArgumentParser(description="pair 可视化抽样与训练子集 manifest 筛选")
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--pair-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-preview-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


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


def validate_args(args):
    lmdb_root = args.lmdb_root.resolve()
    pair_csv = args.pair_csv.resolve()
    out_dir = args.out_dir.resolve()
    if not lmdb_root.exists():
        raise FileNotFoundError(f"LMDB 不存在: {lmdb_root}")
    if not pair_csv.exists():
        raise FileNotFoundError(f"pair CSV 不存在: {pair_csv}")
    if not is_relative_to(out_dir, SAFE_OUTPUT_ROOT):
        raise ValueError(f"out-dir 必须位于 {SAFE_OUTPUT_ROOT} 下: {out_dir}")
    if is_relative_to(out_dir, FORBIDDEN_OUTPUT_ROOT):
        raise ValueError(f"out-dir 不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {out_dir}")
    if args.num_preview_per_bucket <= 0:
        raise ValueError("--num-preview-per-bucket 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")

    manifests_dir = out_dir / "data/manifests"
    report_path = out_dir / "reports/pair_visual_inspection_report.md"
    preview_root = out_dir / "previews/pair_samples"
    output_paths = [
        manifests_dir / "pair_manifest_top1_hq_same_structure.csv",
        manifests_dir / "pair_manifest_top1_hq_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_same_structure_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_no_wrong_hq.csv",
        report_path,
        preview_root,
    ]
    for output_path in output_paths:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"输出路径已存在，未提供 --overwrite: {output_path}")
    return lmdb_root, pair_csv, manifests_dir, report_path, preview_root


def resolve_pair_fields(fieldnames: List[str]) -> Dict[str, str]:
    required = [
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
    mapping = {}
    for key in required:
        if key not in fieldnames:
            raise ValueError(f"pair CSV 缺少字段 {key}，现有字段: {fieldnames}")
        mapping[key] = key
    return mapping


def load_pair_rows(pair_csv: Path, limit: Optional[int]):
    rows = []
    with pair_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("pair CSV 没有表头")
        fields = resolve_pair_fields(reader.fieldnames)
        for i, row in enumerate(reader, start=1):
            if limit is not None and i > limit:
                break
            parsed = {
                "pair_id": as_int(row[fields["pair_id"]]),
                "label": row[fields["label"]],
                "group_size": as_int(row[fields["group_size"]]),
                "lq_lmdb_root": row[fields["lq_lmdb_root"]],
                "lq_lmdb_index": as_int(row[fields["lq_lmdb_index"]]),
                "lq_quality": row[fields["lq_quality"]],
                "lq_quality_priority": as_int(row[fields["lq_quality_priority"]]),
                "lq_structure": row[fields["lq_structure"]],
                "lq_structure_type": row[fields["lq_structure_type"]],
                "lq_ocr_pred": row[fields["lq_ocr_pred"]],
                "lq_ocr_correct": parse_bool(row[fields["lq_ocr_correct"]]),
                "lq_confidence": as_float(row[fields["lq_confidence"]]),
                "lq_avg_conf": as_float(row[fields["lq_avg_conf"]]),
                "lq_min_conf": as_float(row[fields["lq_min_conf"]]),
                "lq_source_path": row[fields["lq_source_path"]],
                "hq_lmdb_root": row[fields["hq_lmdb_root"]],
                "hq_lmdb_index": as_int(row[fields["hq_lmdb_index"]]),
                "hq_quality": row[fields["hq_quality"]],
                "hq_quality_priority": as_int(row[fields["hq_quality_priority"]]),
                "hq_structure": row[fields["hq_structure"]],
                "hq_structure_type": row[fields["hq_structure_type"]],
                "hq_ocr_pred": row[fields["hq_ocr_pred"]],
                "hq_ocr_correct": parse_bool(row[fields["hq_ocr_correct"]]),
                "hq_confidence": as_float(row[fields["hq_confidence"]]),
                "hq_avg_conf": as_float(row[fields["hq_avg_conf"]]),
                "hq_min_conf": as_float(row[fields["hq_min_conf"]]),
                "hq_source_path": row[fields["hq_source_path"]],
                "quality_relation": row[fields["quality_relation"]],
                "structure_relation": row[fields["structure_relation"]],
                "pair_type": row[fields["pair_type"]],
            }
            rows.append(parsed)
    return rows, fields


class LmdbAccessor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False)
        self.label_cache: Dict[int, str] = {}
        self.image_cache: Dict[int, bytes] = {}

    def close(self):
        self.env.close()

    def get_label(self, idx: int) -> str:
        if idx not in self.label_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"label-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 label-{idx:09d}")
            self.label_cache[idx] = raw.decode("utf-8")
        return self.label_cache[idx]

    def get_image(self, idx: int) -> Image.Image:
        if idx not in self.image_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"image-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 image-{idx:09d}")
            self.image_cache[idx] = raw
        return Image.open(io.BytesIO(self.image_cache[idx])).convert("RGB")


def check_consistency(rows: List[Dict[str, Any]], accessor: LmdbAccessor):
    checks = {
        "pair_id_unique": True,
        "pair_type_ok": True,
        "lq_hq_label_same": True,
        "lq_hq_index_diff": True,
        "structure_relation_ok": True,
        "quality_relation_ok": True,
    }
    pair_ids = set()
    seen_hq_by_label: Dict[str, int] = {}
    structure_matrix = Counter()
    quality_relation_counter = Counter()
    hq_wrong_labels = set()

    for row in rows:
        if row["pair_id"] in pair_ids:
            checks["pair_id_unique"] = False
            raise ValueError(f"存在重复 pair_id: {row['pair_id']}")
        pair_ids.add(row["pair_id"])

        if row["pair_type"] != "same_label_top1_hq":
            checks["pair_type_ok"] = False
            raise ValueError(f"pair_type 非 same_label_top1_hq: pair_id={row['pair_id']}")

        if row["lq_lmdb_index"] == row["hq_lmdb_index"]:
            checks["lq_hq_index_diff"] = False
            raise ValueError(f"LQ/HQ 索引相同: pair_id={row['pair_id']}")

        lq_label = accessor.get_label(row["lq_lmdb_index"])
        hq_label = accessor.get_label(row["hq_lmdb_index"])
        if not (lq_label == hq_label == row["label"]):
            checks["lq_hq_label_same"] = False
            raise ValueError(f"pair label 不一致: pair_id={row['pair_id']}")

        expected_structure_relation = "same_structure" if row["lq_structure"] == row["hq_structure"] else "cross_structure"
        if row["structure_relation"] != expected_structure_relation:
            checks["structure_relation_ok"] = False
            raise ValueError(f"structure_relation 不一致: pair_id={row['pair_id']}")

        expected_quality_relation = f"{row['lq_quality']}_to_{row['hq_quality']}"
        if row["quality_relation"] != expected_quality_relation:
            checks["quality_relation_ok"] = False
            raise ValueError(f"quality_relation 不一致: pair_id={row['pair_id']}")

        if row["label"] in seen_hq_by_label and seen_hq_by_label[row["label"]] != row["hq_lmdb_index"]:
            raise ValueError(f"同一 label 存在多个 HQ: {row['label']}")
        seen_hq_by_label[row["label"]] = row["hq_lmdb_index"]

        structure_matrix[f"{row['lq_structure']}->{row['hq_structure']}"] += 1
        quality_relation_counter[row["quality_relation"]] += 1
        if not row["hq_ocr_correct"]:
            hq_wrong_labels.add(row["label"])

    checks["hq_wrong_count"] = len(hq_wrong_labels)
    checks["structure_matrix"] = structure_matrix
    checks["quality_relation_counter"] = quality_relation_counter
    return checks


def filter_rows(rows: List[Dict[str, Any]]):
    same_structure = [row for row in rows if row["structure_relation"] == "same_structure"]
    quality_improved = [row for row in rows if row["lq_quality_priority"] < row["hq_quality_priority"]]
    same_structure_quality_improved = [
        row for row in rows
        if row["structure_relation"] == "same_structure" and row["lq_quality_priority"] < row["hq_quality_priority"]
    ]
    no_wrong_hq = [row for row in rows if row["hq_ocr_correct"]]
    return {
        "pair_manifest_top1_hq_same_structure.csv": same_structure,
        "pair_manifest_top1_hq_quality_improved.csv": quality_improved,
        "pair_manifest_top1_hq_same_structure_quality_improved.csv": same_structure_quality_improved,
        "pair_manifest_top1_hq_no_wrong_hq.csv": no_wrong_hq,
    }


def write_manifest(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def counter_rows(counter: Counter, key_name: str):
    return [{key_name: k, "count": v} for k, v in counter.items()]


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], headers: Optional[Dict[str, str]] = None):
    if not rows:
        return "无"
    headers = headers or {}
    lines = [
        "| " + " | ".join(headers.get(col, col) for col in columns) + " |",
        "| " + " | ".join("---" if col in {"filename", "quality_relation", "structure_relation", "structure_pair", "quality", "structure", "bucket", "sample_file"} else "---:" for col in columns) + " |",
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


def summarize_subset(name: str, rows: List[Dict[str, Any]], total: int):
    quality_relation = Counter(row["quality_relation"] for row in rows)
    structure_relation = Counter(row["structure_relation"] for row in rows)
    lq_quality = Counter(row["lq_quality"] for row in rows)
    hq_quality = Counter(row["hq_quality"] for row in rows)
    lq_structure = Counter(row["lq_structure"] for row in rows)
    hq_structure = Counter(row["hq_structure"] for row in rows)
    wrong_hq_count = sum(1 for row in rows if not row["hq_ocr_correct"])
    return {
        "filename": name,
        "num_pairs": len(rows),
        "ratio": (len(rows) / total) if total else 0.0,
        "quality_relation": quality_relation,
        "structure_relation": structure_relation,
        "lq_quality": lq_quality,
        "hq_quality": hq_quality,
        "lq_structure": lq_structure,
        "hq_structure": hq_structure,
        "wrong_hq_count": wrong_hq_count,
    }


def fit_image(img: Image.Image, target_height: int = 220, target_width: int = 320) -> Image.Image:
    image = img.copy()
    image.thumbnail((target_width, target_height))
    canvas = Image.new("RGB", (target_width, target_height), "white")
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_preview(row: Dict[str, Any], accessor: LmdbAccessor, output_path: Path):
    font = ImageFont.load_default()
    lq_img = fit_image(accessor.get_image(row["lq_lmdb_index"]))
    hq_img = fit_image(accessor.get_image(row["hq_lmdb_index"]))
    panel_w = lq_img.width + hq_img.width
    panel_h = max(lq_img.height, hq_img.height)
    text_lines = [
        f"label: {row['label']}",
        f"pair_id: {row['pair_id']}",
        f"{row['lq_lmdb_index']} -> {row['hq_lmdb_index']}",
        f"{row['lq_quality']} -> {row['hq_quality']}",
        f"{row['lq_structure']} -> {row['hq_structure']}",
        f"{row['lq_confidence']:.6f} / {row['hq_confidence']:.6f}",
        f"{row['quality_relation']} | {row['structure_relation']}",
    ]
    line_height = 14
    text_h = 10 + line_height * len(text_lines)
    canvas = Image.new("RGB", (panel_w, panel_h + text_h), "white")
    canvas.paste(lq_img, (0, 0))
    canvas.paste(hq_img, (lq_img.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((lq_img.width, 0, lq_img.width, panel_h), fill="gray", width=2)
    draw.text((8, panel_h + 5), "\n".join(text_lines), font=font, fill="black", spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def choose_sample(rows: List[Dict[str, Any]], rng: random.Random, n: int):
    if len(rows) <= n:
        return list(rows)
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def build_preview_buckets(rows: List[Dict[str, Any]], n: int, seed: int):
    rng = random.Random(seed)
    by_bucket = {
        "same_structure": [row for row in rows if row["structure_relation"] == "same_structure"],
        "cross_structure": [row for row in rows if row["structure_relation"] == "cross_structure"],
        "quality_improved": [row for row in rows if row["lq_quality_priority"] < row["hq_quality_priority"]],
        "same_structure_quality_improved": [
            row for row in rows
            if row["structure_relation"] == "same_structure" and row["lq_quality_priority"] < row["hq_quality_priority"]
        ],
        "hard_to_easy": [row for row in rows if row["quality_relation"] == "hard_to_easy"],
        "hard_to_middle": [row for row in rows if row["quality_relation"] == "hard_to_middle"],
        "middle_to_easy": [row for row in rows if row["quality_relation"] == "middle_to_easy"],
        "hard_to_hard": [row for row in rows if row["quality_relation"] == "hard_to_hard"],
        "hq_ocr_wrong": [row for row in rows if not row["hq_ocr_correct"]],
        "random_all": rows,
    }
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[row["label"]].append(row)
    largest_labels = [label for label, _ in sorted(group_rows.items(), key=lambda item: (-item[1][0]["group_size"], item[0]))[:20]]
    large_group_examples = [row for label in largest_labels for row in group_rows[label]]
    by_bucket["large_group_examples"] = large_group_examples

    selected = {}
    for bucket, bucket_rows in by_bucket.items():
        if bucket == "hq_ocr_wrong":
            selected[bucket] = list(bucket_rows)
        else:
            selected[bucket] = choose_sample(bucket_rows, rng, n)
    return selected


def remove_existing_outputs(manifests_dir: Path, report_path: Path, preview_root: Path):
    for path in [
        manifests_dir / "pair_manifest_top1_hq_same_structure.csv",
        manifests_dir / "pair_manifest_top1_hq_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_same_structure_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_no_wrong_hq.csv",
        report_path,
    ]:
        if path.exists():
            path.unlink()
    if preview_root.exists():
        shutil.rmtree(preview_root)


def write_report(
    report_path: Path,
    inputs_outputs: Dict[str, str],
    checks: Dict[str, Any],
    total_rows: List[Dict[str, Any]],
    subset_stats: List[Dict[str, Any]],
    preview_index: Dict[str, List[str]],
    limit: Optional[int],
):
    total = len(total_rows)
    same_structure = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_same_structure.csv")
    quality_improved = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_quality_improved.csv")
    clean = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_same_structure_quality_improved.csv")
    no_wrong_hq = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_no_wrong_hq.csv")
    removed_wrong_hq = total - no_wrong_hq["num_pairs"]

    subset_overview_rows = [
        {
            "filename": stat["filename"],
            "num_pairs": stat["num_pairs"],
            "ratio": stat["ratio"],
            "wrong_hq_count": stat["wrong_hq_count"],
        }
        for stat in subset_stats
    ]
    preview_rows = [{"bucket": bucket, "count": len(files), "sample_file": files[0] if files else "无"} for bucket, files in preview_index.items()]
    wrong_hq_rows = [row for row in total_rows if not row["hq_ocr_correct"]]
    structure_pair_rows = counter_rows(checks["structure_matrix"], "structure_pair")
    quality_relation_rows = counter_rows(checks["quality_relation_counter"], "quality_relation")

    report_text = f"""# Pair 可视化抽样与训练子集筛选报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建脚本、过滤 manifest、预览图和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `{inputs_outputs['lmdb_root']}`
- 输入 pair CSV: `{inputs_outputs['pair_csv']}`
- 输出 same-structure manifest: `{inputs_outputs['same_structure']}`
- 输出 quality-improved manifest: `{inputs_outputs['quality_improved']}`
- 输出 same-structure+quality-improved manifest: `{inputs_outputs['same_structure_quality_improved']}`
- 输出 no-wrong-HQ manifest: `{inputs_outputs['no_wrong_hq']}`
- 预览图目录: `{inputs_outputs['preview_root']}`
- 输出报告: `{inputs_outputs['report']}`
- `--limit`: `{limit}`

## 3. 一致性检查

- pair_id 唯一: `{checks['pair_id_unique']}`
- pair_type 正确: `{checks['pair_type_ok']}`
- 每个 pair 的 lq/hq label 相同: `{checks['lq_hq_label_same']}`
- `lq_lmdb_index != hq_lmdb_index`: `{checks['lq_hq_index_diff']}`
- structure_relation 校验: `{checks['structure_relation_ok']}`
- quality_relation 校验: `{checks['quality_relation_ok']}`

## 4. 原始 Pair Manifest 总览

- `num_pairs`: `{total}`
- `same_structure`: `{same_structure['num_pairs']}` ({same_structure['ratio']:.6f})
- `cross_structure`: `{total - same_structure['num_pairs']}` ({1 - same_structure['ratio'] if total else 0.0:.6f})
- `HQ ocr_correct=False` 数量: `{checks['hq_wrong_count']}`

### quality_relation 分布

{markdown_table(quality_relation_rows, ['quality_relation', 'count'])}

### structure_pair matrix

{markdown_table(structure_pair_rows, ['structure_pair', 'count'])}

## 5. 过滤后 Manifest 统计

{markdown_table(subset_overview_rows, ['filename', 'num_pairs', 'ratio', 'wrong_hq_count'])}

### 各子集详细统计

{"".join(
    f"#### {stat['filename']}\n\n"
    + f"- num_pairs: `{stat['num_pairs']}`\n"
    + f"- 占原始 pair 比例: `{stat['ratio']:.6f}`\n"
    + f"- wrong HQ 数量: `{stat['wrong_hq_count']}`\n\n"
    + "quality_relation 分布：\n\n"
    + markdown_table(counter_rows(stat['quality_relation'], 'quality_relation'), ['quality_relation', 'count'])
    + "\n\nstructure_relation 分布：\n\n"
    + markdown_table(counter_rows(stat['structure_relation'], 'structure_relation'), ['structure_relation', 'count'])
    + "\n\nLQ quality 分布：\n\n"
    + markdown_table(counter_rows(stat['lq_quality'], 'quality'), ['quality', 'count'])
    + "\n\nHQ quality 分布：\n\n"
    + markdown_table(counter_rows(stat['hq_quality'], 'quality'), ['quality', 'count'])
    + "\n\nLQ structure 分布：\n\n"
    + markdown_table(counter_rows(stat['lq_structure'], 'structure'), ['structure', 'count'])
    + "\n\nHQ structure 分布：\n\n"
    + markdown_table(counter_rows(stat['hq_structure'], 'structure'), ['structure', 'count'])
    + "\n\n"
    for stat in subset_stats
)}

## 6. Same-structure 子集分析

- 文件: `pair_manifest_top1_hq_same_structure.csv`
- pair 数: `{same_structure['num_pairs']}`
- 占原始比例: `{same_structure['ratio']:.6f}`
- 说明: 保留相同结构 pair，减少布局迁移干扰，但仍包含 `easy_to_easy`、`middle_to_middle`、`hard_to_hard` 等非质量提升样本。

## 7. Quality-improved 子集分析

- 文件: `pair_manifest_top1_hq_quality_improved.csv`
- pair 数: `{quality_improved['num_pairs']}`
- 占原始比例: `{quality_improved['ratio']:.6f}`
- 包含关系: `hard_to_middle`, `hard_to_easy`, `middle_to_easy`
- 说明: 保证 HQ 质量优于 LQ，但仍包含 cross-structure pair。

## 8. Same-structure + Quality-improved 子集分析

- 文件: `pair_manifest_top1_hq_same_structure_quality_improved.csv`
- pair 数: `{clean['num_pairs']}`
- 占原始比例: `{clean['ratio']:.6f}`
- 说明: 同结构且 HQ 质量优于 LQ，是第一版 diffusion loss 训练最干净子集。

## 9. Wrong-HQ 分析

- wrong-HQ pair 数量（原始）: `{checks['hq_wrong_count']}`
- `pair_manifest_top1_hq_no_wrong_hq.csv` 移除 pair 数量: `{removed_wrong_hq}`

{markdown_table(
    [
        {
            'pair_id': row['pair_id'],
            'label': row['label'],
            'hq_lmdb_index': row['hq_lmdb_index'],
            'hq_quality': row['hq_quality'],
            'hq_structure': row['hq_structure'],
            'hq_confidence': row['hq_confidence'],
        }
        for row in wrong_hq_rows
    ],
    ['pair_id', 'label', 'hq_lmdb_index', 'hq_quality', 'hq_structure', 'hq_confidence'],
) if wrong_hq_rows else "无"}

## 10. Cross-structure 分析

- cross-structure pair 数量: `{total - same_structure['num_pairs']}`
- cross-structure 比例: `{1 - same_structure['ratio'] if total else 0.0:.6f}`
- 说明: 这些样本可能让 diffusion loss 学到布局迁移，而不是单纯质量恢复。

## 11. 可视化抽样说明

- 每个 bucket 默认抽样数量: `{inputs_outputs['num_preview_per_bucket']}`
- 随机种子: `{inputs_outputs['seed']}`
- `hq_ocr_wrong` bucket: 若样本很少则全部导出
- `large_group_examples` bucket: 从 `group_size` 最大的若干 label group 中抽样

## 12. 预览图目录索引

{markdown_table(preview_rows, ['bucket', 'count', 'sample_file'])}

## 13. 推荐用于 Stage 1 Diffusion 训练的 Manifest

第一版最干净训练集：

- `data/manifests/pair_manifest_top1_hq_same_structure_quality_improved.csv`
- 原因：
  - 同结构，减少布局迁移干扰
  - HQ 质量优于 LQ
  - 更符合 Diffusion loss 的“低质量到高质量”恢复设定

备选训练集：

- `data/manifests/pair_manifest_top1_hq_same_structure.csv`
- 原因：
  - 样本更多
  - 但包含 `easy_to_easy`、`middle_to_middle`、`hard_to_hard` 等非质量提升 pair

扩展训练集：

- `data/manifests/pair_manifest_top1_hq_no_wrong_hq.csv`
- 原因：
  - 排除被 OCR 判错的 HQ
  - 但仍包含 cross-structure pair

## 14. 警告与限制

- 同 label pair 不一定像素对齐。
- cross_structure pair 可能导致 diffusion loss 学习布局迁移。
- OCR confidence 接近饱和，不能完全代表图像质量。
- quality_improved 只基于 raw quality 标注 `easy/middle/hard`。
- 可视化抽样只能辅助人工判断，不能替代训练验证。
{"- 本次运行使用了 --limit，仅反映子集统计。\n" if limit is not None else ""}

## 15. 下一步建议

人工检查 `previews/pair_samples/` 中的样例；若 `same_structure_quality_improved` 视觉质量可接受，则用该 manifest 作为 Stage 1 conditional latent DiT diffusion-loss 预训练的第一版训练集；随后再设计 DiT 数据读取与训练脚本。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def write_failure_report(report_path: Path, error: str):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Pair 可视化抽样与训练子集筛选报告\n\n"
        "## 1. 范围与隔离声明\n\n"
        "本轮只在 `experiments/dit_lq_hq_v1/` 下尝试创建脚本、过滤 manifest、预览图和报告。\n"
        "未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。\n\n"
        "## 14. 警告与限制\n\n"
        f"- 任务失败：{error}\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    try:
        lmdb_root, pair_csv, manifests_dir, report_path, preview_root = validate_args(args)
        if args.overwrite:
            remove_existing_outputs(manifests_dir, report_path, preview_root)

        rows, fields = load_pair_rows(pair_csv, args.limit)
        accessor = LmdbAccessor(lmdb_root)
        try:
            checks = check_consistency(rows, accessor)
            filtered = filter_rows(rows)

            output_paths = {
                name: manifests_dir / name
                for name in filtered
            }
            for name, manifest_rows in filtered.items():
                write_manifest(output_paths[name], manifest_rows)

            preview_root.mkdir(parents=True, exist_ok=True)
            selected = build_preview_buckets(rows, args.num_preview_per_bucket, args.seed)
            preview_index: Dict[str, List[str]] = {}
            for bucket in BUCKETS:
                bucket_dir = preview_root / bucket
                bucket_dir.mkdir(parents=True, exist_ok=True)
                preview_index[bucket] = []
                for row in selected.get(bucket, []):
                    filename = f"pair_{row['pair_id']}_lq_{row['lq_lmdb_index']}_hq_{row['hq_lmdb_index']}_{row['quality_relation']}_{row['structure_relation']}.jpg"
                    output_path = bucket_dir / filename
                    make_preview(row, accessor, output_path)
                    preview_index[bucket].append(str(output_path.relative_to(preview_root)))

            subset_stats = [summarize_subset(name, manifest_rows, len(rows)) for name, manifest_rows in filtered.items()]
            write_report(
                report_path=report_path,
                inputs_outputs={
                    "lmdb_root": str(lmdb_root),
                    "pair_csv": str(pair_csv),
                    "same_structure": str(output_paths["pair_manifest_top1_hq_same_structure.csv"]),
                    "quality_improved": str(output_paths["pair_manifest_top1_hq_quality_improved.csv"]),
                    "same_structure_quality_improved": str(output_paths["pair_manifest_top1_hq_same_structure_quality_improved.csv"]),
                    "no_wrong_hq": str(output_paths["pair_manifest_top1_hq_no_wrong_hq.csv"]),
                    "preview_root": str(preview_root),
                    "report": str(report_path),
                    "num_preview_per_bucket": str(args.num_preview_per_bucket),
                    "seed": str(args.seed),
                },
                checks=checks,
                total_rows=rows,
                subset_stats=subset_stats,
                preview_index=preview_index,
                limit=args.limit,
            )
        finally:
            accessor.close()
    except Exception as exc:
        write_failure_report((args.out_dir.resolve() / "reports/pair_visual_inspection_report.md"), str(exc))
        raise


if __name__ == "__main__":
    main()
