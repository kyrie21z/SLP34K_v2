#!/usr/bin/env python3
import argparse
import csv
import io
import json
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import lmdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
QUALITY_PRIORITY = {"easy": 3, "middle": 2, "hard": 1}
PAIR_TYPE = "same_label_top1_hq"
HQ_SELECTION_VERSION = "visual_v2"

TRAIN_SAMPLE_FIELDS = [
    "lmdb_root",
    "lmdb_index",
    "label",
    "quality",
    "quality_priority",
    "structure",
    "structure_type",
    "source_path",
    "ocr_pred",
    "ocr_correct",
    "confidence",
    "avg_conf",
    "min_conf",
    "sharpness",
    "sharpness_norm",
    "contrast",
    "contrast_norm",
    "brightness",
    "brightness_score",
    "resolution",
    "resolution_norm",
    "visual_quality_score",
]

PAIR_FIELDS = [
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
    "lq_visual_quality_score",
    "lq_sharpness_norm",
    "lq_contrast_norm",
    "lq_brightness_score",
    "lq_resolution_norm",
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
    "hq_visual_quality_score",
    "hq_sharpness_norm",
    "hq_contrast_norm",
    "hq_brightness_score",
    "hq_resolution_norm",
    "hq_source_path",
    "quality_relation",
    "structure_relation",
    "pair_type",
    "hq_selection_version",
]

PREVIEW_BUCKETS = [
    "same_structure_quality_improved",
    "quality_improved",
    "same_structure",
    "hard_to_easy",
    "hard_to_middle",
    "middle_to_easy",
    "hard_to_hard",
    "cross_structure",
    "hq_changed_examples",
    "random_all",
]


def parse_args():
    parser = argparse.ArgumentParser(description="基于视觉质量分数重构 Top1-HQ Pair Manifest v2")
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--ocr-csv", type=Path, required=True)
    parser.add_argument("--v1-pair-csv", type=Path, required=True)
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
    ocr_csv = args.ocr_csv.resolve()
    v1_pair_csv = args.v1_pair_csv.resolve()
    out_dir = args.out_dir.resolve()
    if not lmdb_root.exists():
        raise FileNotFoundError(f"LMDB 不存在: {lmdb_root}")
    if not ocr_csv.exists():
        raise FileNotFoundError(f"OCR CSV 不存在: {ocr_csv}")
    if not v1_pair_csv.exists():
        raise FileNotFoundError(f"v1 pair CSV 不存在: {v1_pair_csv}")
    if not is_relative_to(out_dir, SAFE_OUTPUT_ROOT):
        raise ValueError(f"out-dir 必须位于 {SAFE_OUTPUT_ROOT} 下: {out_dir}")
    if is_relative_to(out_dir, FORBIDDEN_OUTPUT_ROOT):
        raise ValueError(f"out-dir 不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {out_dir}")
    if args.num_preview_per_bucket <= 0:
        raise ValueError("--num-preview-per-bucket 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")

    manifests_dir = out_dir / "data/manifests"
    preview_root = out_dir / "previews/pair_samples_visual_v2"
    report_path = out_dir / "reports/pair_stats_top1_hq_visual_v2_report.md"
    output_paths = [
        manifests_dir / "train_samples_visual_quality_v2.csv",
        manifests_dir / "train_samples_visual_quality_v2.jsonl",
        manifests_dir / "pair_manifest_top1_hq_visual_v2.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_same_structure.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv",
        preview_root,
        report_path,
    ]
    for output_path in output_paths:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"输出路径已存在，未提供 --overwrite: {output_path}")
    return lmdb_root, ocr_csv, v1_pair_csv, manifests_dir, preview_root, report_path


def resolve_ocr_fields(fieldnames: List[str]) -> Dict[str, str]:
    candidates = {
        "lmdb_index": ["lmdb_index"],
        "label": ["label", "raw_label"],
        "pred": ["pred", "ocr_pred"],
        "correct": ["correct", "ocr_correct"],
        "confidence": ["confidence", "avg_conf"],
        "avg_conf": ["avg_conf", "confidence"],
        "min_conf": ["min_conf"],
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


def resolve_v1_pair_fields(fieldnames: List[str]) -> Dict[str, str]:
    required = ["label", "hq_lmdb_index", "hq_quality", "hq_structure", "hq_confidence"]
    resolved = {}
    for key in required:
        if key not in fieldnames:
            raise ValueError(f"v1 pair CSV 缺少字段 {key}，现有字段: {fieldnames}")
        resolved[key] = key
    return resolved


class LmdbAccessor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False)
        self.meta_cache: Dict[int, Dict[str, Any]] = {}
        self.label_cache: Dict[int, str] = {}
        self.image_cache: Dict[int, bytes] = {}

    def close(self):
        self.env.close()

    def num_samples(self) -> int:
        with self.env.begin(write=False) as txn:
            raw = txn.get(b"num-samples")
        if raw is None:
            raise ValueError("LMDB 缺少 num-samples")
        return int(raw.decode("utf-8"))

    def get_label(self, idx: int) -> str:
        if idx not in self.label_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"label-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 label-{idx:09d}")
            self.label_cache[idx] = raw.decode("utf-8")
        return self.label_cache[idx]

    def get_meta(self, idx: int) -> Dict[str, Any]:
        if idx not in self.meta_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"meta-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 meta-{idx:09d}")
            self.meta_cache[idx] = json.loads(raw.decode("utf-8"))
        return self.meta_cache[idx]

    def get_image_bytes(self, idx: int) -> bytes:
        if idx not in self.image_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"image-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 image-{idx:09d}")
            self.image_cache[idx] = raw
        return self.image_cache[idx]

    def get_image(self, idx: int) -> Image.Image:
        return Image.open(io.BytesIO(self.get_image_bytes(idx))).convert("RGB")

    def get_gray_array(self, idx: int) -> np.ndarray:
        img = self.get_image(idx).convert("L")
        return np.array(img, dtype=np.uint8)


def compute_visual_metrics(gray: np.ndarray) -> Dict[str, float]:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    brightness_score = max(0.0, min(1.0, 1.0 - abs(brightness - 128.0) / 128.0))
    h, w = gray.shape
    resolution = float(w * h)
    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "brightness_score": brightness_score,
        "resolution": resolution,
    }


def normalize_metric(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.5 for _ in values]
    return [(value - vmin) / (vmax - vmin) for value in values]


def load_ocr_rows(ocr_csv: Path):
    rows = {}
    count = 0
    with ocr_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("OCR CSV 没有表头")
        fields = resolve_ocr_fields(reader.fieldnames)
        for row in reader:
            count += 1
            idx = as_int(row[fields["lmdb_index"]])
            if idx in rows:
                raise ValueError(f"OCR CSV 存在重复 lmdb_index: {idx}")
            rows[idx] = {
                "lmdb_index": idx,
                "label": row[fields["label"]],
                "pred": row[fields["pred"]],
                "correct": parse_bool(row[fields["correct"]]),
                "confidence": as_float(row[fields["confidence"]]),
                "avg_conf": as_float(row[fields["avg_conf"]]),
                "min_conf": as_float(row[fields["min_conf"]]),
            }
    return rows, count, fields


def build_sample_rows(accessor: LmdbAccessor, ocr_rows: Dict[int, Dict[str, Any]], limit: Optional[int]):
    total = accessor.num_samples()
    max_index = total if limit is None else min(limit, total)
    raw_rows = []
    sharpness_values = []
    contrast_values = []
    resolution_values = []
    for idx in range(1, max_index + 1):
        label = accessor.get_label(idx)
        meta = accessor.get_meta(idx)
        if meta.get("id") != idx:
            raise ValueError(f"metadata.id 与 lmdb_index 不一致: {idx}")
        if meta.get("raw_label") != label:
            raise ValueError(f"metadata.raw_label 与 label 不一致: {idx}")
        if meta.get("split") != "train":
            raise ValueError(f"metadata.split 不是 train: {idx}")
        quality = meta.get("quality")
        structure = meta.get("structure")
        structure_type = meta.get("structure_type")
        if quality not in QUALITY_PRIORITY:
            raise ValueError(f"未知 quality: {quality} (lmdb_index={idx})")
        if not structure or not structure_type:
            raise ValueError(f"metadata 缺少 structure/structure_type: {idx}")
        if idx not in ocr_rows:
            raise ValueError(f"OCR CSV 缺少 lmdb_index={idx}")
        ocr = ocr_rows[idx]
        if ocr["label"] != label:
            raise ValueError(f"OCR CSV label 与 LMDB label 不一致: {idx}")

        gray = accessor.get_gray_array(idx)
        metrics = compute_visual_metrics(gray)
        sharpness_values.append(metrics["sharpness"])
        contrast_values.append(metrics["contrast"])
        resolution_values.append(metrics["resolution"])
        raw_rows.append(
            {
                "lmdb_root": str(accessor.root),
                "lmdb_index": idx,
                "label": label,
                "quality": quality,
                "quality_priority": QUALITY_PRIORITY[quality],
                "structure": structure,
                "structure_type": structure_type,
                "source_path": meta.get("source_path"),
                "ocr_pred": ocr["pred"],
                "ocr_correct": ocr["correct"],
                "confidence": ocr["confidence"],
                "avg_conf": ocr["avg_conf"],
                "min_conf": ocr["min_conf"],
                "metadata": meta,
                **metrics,
            }
        )

    sharpness_norm = normalize_metric(sharpness_values)
    contrast_norm = normalize_metric(contrast_values)
    resolution_norm = normalize_metric(resolution_values)
    final_rows = []
    for i, row in enumerate(raw_rows):
        row["sharpness_norm"] = sharpness_norm[i]
        row["contrast_norm"] = contrast_norm[i]
        row["resolution_norm"] = resolution_norm[i]
        row["visual_quality_score"] = (
            0.50 * row["sharpness_norm"]
            + 0.20 * row["contrast_norm"]
            + 0.20 * row["brightness_score"]
            + 0.10 * row["resolution_norm"]
        )
        final_rows.append(row)
    return total, final_rows


def write_train_samples(rows: List[Dict[str, Any]], csv_path: Path, jsonl_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_fp, jsonl_path.open("w", encoding="utf-8") as jsonl_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=TRAIN_SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in TRAIN_SAMPLE_FIELDS})
            jsonl_fp.write(
                json.dumps(
                    {
                        **{key: row[key] for key in TRAIN_SAMPLE_FIELDS},
                        "metadata": row["metadata"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def select_v2_hq(groups: Dict[str, List[Dict[str, Any]]]):
    hq_by_label = {}
    all_ocr_wrong_groups = []
    for label, rows in groups.items():
        true_rows = [row for row in rows if row["ocr_correct"]]
        candidates = true_rows if true_rows else rows
        if not true_rows:
            all_ocr_wrong_groups.append(label)
        ordered = sorted(
            candidates,
            key=lambda row: (
                -row["quality_priority"],
                -row["visual_quality_score"],
                -row["sharpness_norm"],
                -row["contrast_norm"],
                -row["brightness_score"],
                -row["resolution_norm"],
                row["lmdb_index"],
            ),
        )
        hq_by_label[label] = ordered[0]
    return hq_by_label, all_ocr_wrong_groups


def build_v2_pairs(sample_rows: List[Dict[str, Any]], hq_by_label: Dict[str, Dict[str, Any]]):
    groups = defaultdict(list)
    for row in sample_rows:
        groups[row["label"]].append(row)

    pairs = []
    pair_id = 1
    for label, rows in groups.items():
        if len(rows) < 2:
            continue
        hq = hq_by_label[label]
        for row in rows:
            if row["lmdb_index"] == hq["lmdb_index"]:
                continue
            pair = {
                "pair_id": pair_id,
                "label": label,
                "group_size": len(rows),
                "lq_lmdb_root": row["lmdb_root"],
                "lq_lmdb_index": row["lmdb_index"],
                "lq_quality": row["quality"],
                "lq_quality_priority": row["quality_priority"],
                "lq_structure": row["structure"],
                "lq_structure_type": row["structure_type"],
                "lq_ocr_pred": row["ocr_pred"],
                "lq_ocr_correct": row["ocr_correct"],
                "lq_confidence": row["confidence"],
                "lq_avg_conf": row["avg_conf"],
                "lq_min_conf": row["min_conf"],
                "lq_visual_quality_score": row["visual_quality_score"],
                "lq_sharpness_norm": row["sharpness_norm"],
                "lq_contrast_norm": row["contrast_norm"],
                "lq_brightness_score": row["brightness_score"],
                "lq_resolution_norm": row["resolution_norm"],
                "lq_source_path": row["source_path"],
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
                "hq_visual_quality_score": hq["visual_quality_score"],
                "hq_sharpness_norm": hq["sharpness_norm"],
                "hq_contrast_norm": hq["contrast_norm"],
                "hq_brightness_score": hq["brightness_score"],
                "hq_resolution_norm": hq["resolution_norm"],
                "hq_source_path": hq["source_path"],
                "quality_relation": f"{row['quality']}_to_{hq['quality']}",
                "structure_relation": "same_structure" if row["structure"] == hq["structure"] else "cross_structure",
                "pair_type": PAIR_TYPE,
                "hq_selection_version": HQ_SELECTION_VERSION,
            }
            pairs.append(pair)
            pair_id += 1
    return groups, pairs


def write_manifest(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def build_filtered_subsets(rows: List[Dict[str, Any]]):
    same_structure = [row for row in rows if row["structure_relation"] == "same_structure"]
    quality_improved = [row for row in rows if row["lq_quality_priority"] < row["hq_quality_priority"]]
    same_structure_quality_improved = [
        row for row in rows
        if row["structure_relation"] == "same_structure" and row["lq_quality_priority"] < row["hq_quality_priority"]
    ]
    return {
        "pair_manifest_top1_hq_visual_v2_same_structure.csv": same_structure,
        "pair_manifest_top1_hq_visual_v2_quality_improved.csv": quality_improved,
        "pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv": same_structure_quality_improved,
    }


def load_v1_hq_by_label(v1_pair_csv: Path):
    result = {}
    with v1_pair_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("v1 pair CSV 没有表头")
        fields = resolve_v1_pair_fields(reader.fieldnames)
        for row in reader:
            label = row[fields["label"]]
            hq_idx = as_int(row[fields["hq_lmdb_index"]])
            if label in result:
                if result[label]["hq_lmdb_index"] != hq_idx:
                    raise ValueError(f"v1 pair CSV 中同一 label 存在多个 HQ: {label}")
                continue
            result[label] = {
                "label": label,
                "hq_lmdb_index": hq_idx,
                "hq_quality": row[fields["hq_quality"]],
                "hq_structure": row[fields["hq_structure"]],
                "hq_confidence": as_float(row[fields["hq_confidence"]]),
            }
    return result


def fit_image(img: Image.Image, target_height: int = 220, target_width: int = 320) -> Image.Image:
    image = img.copy()
    image.thumbnail((target_width, target_height))
    canvas = Image.new("RGB", (target_width, target_height), "white")
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_pair_preview(row: Dict[str, Any], accessor: LmdbAccessor, output_path: Path):
    font = ImageFont.load_default()
    lq_img = fit_image(accessor.get_image(row["lq_lmdb_index"]))
    hq_img = fit_image(accessor.get_image(row["hq_lmdb_index"]))
    panel_w = lq_img.width + hq_img.width
    panel_h = max(lq_img.height, hq_img.height)
    lines = [
        f"label: {row['label']}",
        f"pair_id: {row['pair_id']}",
        f"{row['lq_lmdb_index']} -> {row['hq_lmdb_index']}",
        f"{row['lq_quality']} -> {row['hq_quality']}",
        f"{row['lq_structure']} -> {row['hq_structure']}",
        f"{row['lq_visual_quality_score']:.6f} / {row['hq_visual_quality_score']:.6f}",
        f"{row['lq_sharpness_norm']:.6f} / {row['hq_sharpness_norm']:.6f}",
        f"{row['quality_relation']} | {row['structure_relation']}",
    ]
    line_height = 14
    text_h = 10 + line_height * len(lines)
    canvas = Image.new("RGB", (panel_w, panel_h + text_h), "white")
    canvas.paste(lq_img, (0, 0))
    canvas.paste(hq_img, (lq_img.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((lq_img.width, 0, lq_img.width, panel_h), fill="gray", width=2)
    draw.text((8, panel_h + 5), "\n".join(lines), font=font, fill="black", spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def make_hq_change_preview(label: str, v1_row: Dict[str, Any], v2_row: Dict[str, Any], accessor: LmdbAccessor, output_path: Path):
    font = ImageFont.load_default()
    v1_img = fit_image(accessor.get_image(v1_row["lmdb_index"]))
    v2_img = fit_image(accessor.get_image(v2_row["lmdb_index"]))
    panel_w = v1_img.width + v2_img.width
    panel_h = max(v1_img.height, v2_img.height)
    lines = [
        f"label: {label}",
        f"v1: {v1_row['lmdb_index']} | v2: {v2_row['lmdb_index']}",
        f"quality: {v1_row['quality']} | {v2_row['quality']}",
        f"visual_score: {v1_row['visual_quality_score']:.6f} | {v2_row['visual_quality_score']:.6f}",
        f"confidence: {v1_row['confidence']:.6f} | {v2_row['confidence']:.6f}",
    ]
    line_height = 14
    text_h = 10 + line_height * len(lines)
    canvas = Image.new("RGB", (panel_w, panel_h + text_h), "white")
    canvas.paste(v1_img, (0, 0))
    canvas.paste(v2_img, (v1_img.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((v1_img.width, 0, v1_img.width, panel_h), fill="gray", width=2)
    draw.text((8, panel_h + 5), "\n".join(lines), font=font, fill="black", spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def choose_sample(rows: List[Any], rng: random.Random, n: int):
    if len(rows) <= n:
        return list(rows)
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def build_preview_samples(v2_pairs: List[Dict[str, Any]], changed_labels: List[str], groups: Dict[str, List[Dict[str, Any]]], n: int, seed: int):
    rng = random.Random(seed)
    buckets = {
        "same_structure_quality_improved": [
            row for row in v2_pairs
            if row["structure_relation"] == "same_structure" and row["lq_quality_priority"] < row["hq_quality_priority"]
        ],
        "quality_improved": [row for row in v2_pairs if row["lq_quality_priority"] < row["hq_quality_priority"]],
        "same_structure": [row for row in v2_pairs if row["structure_relation"] == "same_structure"],
        "hard_to_easy": [row for row in v2_pairs if row["quality_relation"] == "hard_to_easy"],
        "hard_to_middle": [row for row in v2_pairs if row["quality_relation"] == "hard_to_middle"],
        "middle_to_easy": [row for row in v2_pairs if row["quality_relation"] == "middle_to_easy"],
        "hard_to_hard": [row for row in v2_pairs if row["quality_relation"] == "hard_to_hard"],
        "cross_structure": [row for row in v2_pairs if row["structure_relation"] == "cross_structure"],
        "random_all": v2_pairs,
    }
    selected = {bucket: choose_sample(rows, rng, n) for bucket, rows in buckets.items()}
    selected["hq_changed_examples"] = choose_sample(changed_labels, rng, n)
    return selected


def counter_rows(counter: Counter, key_name: str):
    return [{key_name: k, "count": v} for k, v in counter.items()]


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], headers: Optional[Dict[str, str]] = None):
    if not rows:
        return "无"
    headers = headers or {}
    lines = [
        "| " + " | ".join(headers.get(col, col) for col in columns) + " |",
        "| " + " | ".join("---" if col in {"metric", "quality_relation", "structure_relation", "structure_pair", "quality", "structure", "bucket", "sample_file", "label"} else "---:" for col in columns) + " |",
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
    return {
        "filename": name,
        "num_pairs": len(rows),
        "ratio": (len(rows) / total) if total else 0.0,
        "quality_relation": Counter(row["quality_relation"] for row in rows),
        "structure_relation": Counter(row["structure_relation"] for row in rows),
        "lq_quality": Counter(row["lq_quality"] for row in rows),
        "hq_quality": Counter(row["hq_quality"] for row in rows),
        "lq_structure": Counter(row["lq_structure"] for row in rows),
        "hq_structure": Counter(row["hq_structure"] for row in rows),
        "wrong_hq_count": sum(1 for row in rows if not row["hq_ocr_correct"]),
    }


def remove_existing_outputs(manifests_dir: Path, preview_root: Path, report_path: Path):
    for path in [
        manifests_dir / "train_samples_visual_quality_v2.csv",
        manifests_dir / "train_samples_visual_quality_v2.jsonl",
        manifests_dir / "pair_manifest_top1_hq_visual_v2.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_same_structure.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_quality_improved.csv",
        manifests_dir / "pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv",
        report_path,
    ]:
        if path.exists():
            path.unlink()
    if preview_root.exists():
        shutil.rmtree(preview_root)


def write_report(
    report_path: Path,
    inputs_outputs: Dict[str, str],
    sample_rows: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    all_ocr_wrong_groups: List[str],
    v2_pairs: List[Dict[str, Any]],
    subset_stats: List[Dict[str, Any]],
    v1_hq_by_label: Dict[str, Dict[str, Any]],
    v2_hq_by_label: Dict[str, Dict[str, Any]],
    changed_labels: List[str],
    preview_index: Dict[str, List[str]],
    limit: Optional[int],
):
    sharpness = [row["sharpness"] for row in sample_rows]
    contrast = [row["contrast"] for row in sample_rows]
    brightness = [row["brightness"] for row in sample_rows]
    resolution = [row["resolution"] for row in sample_rows]
    visual_scores = [row["visual_quality_score"] for row in sample_rows]
    same_structure = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_visual_v2_same_structure.csv")
    quality_improved = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_visual_v2_quality_improved.csv")
    clean = next(item for item in subset_stats if item["filename"] == "pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv")
    hq_rows = list(v2_hq_by_label.values())
    structure_matrix = Counter(f"{row['lq_structure']}->{row['hq_structure']}" for row in v2_pairs)
    quality_relation_counter = Counter(row["quality_relation"] for row in v2_pairs)
    structure_relation_counter = Counter(row["structure_relation"] for row in v2_pairs)
    hq_wrong_count = sum(1 for row in hq_rows if not row["ocr_correct"])
    num_label_groups = len(v1_hq_by_label)
    v1_clean_pairs = None
    for stat in subset_stats:
        pass
    preview_rows = [{"bucket": bucket, "count": len(files), "sample_file": files[0] if files else "无"} for bucket, files in preview_index.items()]
    changed_examples = []
    for label in changed_labels[:10]:
        v1 = v1_hq_by_label[label]
        v2 = v2_hq_by_label[label]
        changed_examples.append(
            {
                "label": label,
                "v1_hq": v1["lmdb_index"],
                "v2_hq": v2["lmdb_index"],
                "v1_quality": v1["quality"],
                "v2_quality": v2["quality"],
                "v1_visual_score": v1["visual_quality_score"],
                "v2_visual_score": v2["visual_quality_score"],
                "v1_confidence": v1["confidence"],
                "v2_confidence": v2["confidence"],
            }
        )

    report_text = f"""# Top1-HQ Visual-v2 Pair 构造与对比报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建 v2 脚本、v2 manifest、v2 预览图和 v2 报告。
未修改 v1 manifest、原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `{inputs_outputs['lmdb_root']}`
- 输入 OCR CSV: `{inputs_outputs['ocr_csv']}`
- 输入 v1 pair CSV: `{inputs_outputs['v1_pair_csv']}`
- 输出 train samples v2 CSV: `{inputs_outputs['train_samples_csv']}`
- 输出 train samples v2 JSONL: `{inputs_outputs['train_samples_jsonl']}`
- 输出 v2 pair manifest: `{inputs_outputs['v2_pair_csv']}`
- 输出 v2 same-structure manifest: `{inputs_outputs['v2_same_structure_csv']}`
- 输出 v2 quality-improved manifest: `{inputs_outputs['v2_quality_improved_csv']}`
- 输出 v2 same-structure+quality-improved manifest: `{inputs_outputs['v2_clean_csv']}`
- 输出预览图目录: `{inputs_outputs['preview_root']}`
- 输出报告: `{inputs_outputs['report']}`
- `--limit`: `{limit}`

## 3. 视觉质量指标定义

- `sharpness = Laplacian variance`
- `contrast = grayscale std`
- `brightness = grayscale mean`
- `brightness_score = 1 - abs(brightness - 128) / 128`
- `resolution = width * height`
- `visual_quality_score = 0.5 * sharpness_norm + 0.2 * contrast_norm + 0.2 * brightness_score + 0.1 * resolution_norm`

## 4. 样本级视觉质量统计

- 样本数: `{len(sample_rows)}`

{markdown_table(
    [
        {"metric": "sharpness", "mean": statistics.fmean(sharpness), "median": statistics.median(sharpness), "min": min(sharpness), "max": max(sharpness)},
        {"metric": "contrast", "mean": statistics.fmean(contrast), "median": statistics.median(contrast), "min": min(contrast), "max": max(contrast)},
        {"metric": "brightness", "mean": statistics.fmean(brightness), "median": statistics.median(brightness), "min": min(brightness), "max": max(brightness)},
        {"metric": "resolution", "mean": statistics.fmean(resolution), "median": statistics.median(resolution), "min": min(resolution), "max": max(resolution)},
        {"metric": "visual_quality_score", "mean": statistics.fmean(visual_scores), "median": statistics.median(visual_scores), "min": min(visual_scores), "max": max(visual_scores)},
    ],
    ["metric", "mean", "median", "min", "max"],
)}

## 5. v2 HQ 选择规则

每个 `label` group 的 HQ 选择逻辑：

1. 如果 group 内存在 `ocr_correct=True` 样本，则候选集合只保留这些样本
2. 如果 group 内全是 `ocr_correct=False`，则允许从全组中选 HQ，并记录为 `all_ocr_wrong_group`
3. 在候选集合内按以下顺序排序：
   - `quality_priority` 降序
   - `visual_quality_score` 降序
   - `sharpness_norm` 降序
   - `contrast_norm` 降序
   - `brightness_score` 降序
   - `resolution_norm` 降序
   - `lmdb_index` 升序

## 6. v2 Pair Manifest 总览

- `num_pairs`: `{len(v2_pairs)}`
- `same_structure`: `{same_structure['num_pairs']}` ({same_structure['ratio']:.6f})
- `cross_structure`: `{len(v2_pairs) - same_structure['num_pairs']}` ({1 - same_structure['ratio'] if v2_pairs else 0.0:.6f})
- HQ quality 分布:

{markdown_table(counter_rows(Counter(row['quality'] for row in hq_rows), 'quality'), ['quality', 'count'])}

- HQ structure 分布:

{markdown_table(counter_rows(Counter(row['structure'] for row in hq_rows), 'structure'), ['structure', 'count'])}

- HQ `ocr_correct=False` 数量: `{hq_wrong_count}`

### quality_relation 分布

{markdown_table(counter_rows(quality_relation_counter, 'quality_relation'), ['quality_relation', 'count'])}

### structure_pair matrix

{markdown_table(counter_rows(structure_matrix, 'structure_pair'), ['structure_pair', 'count'])}

## 7. v2 过滤子集统计

{markdown_table(
    [{"filename": stat["filename"], "num_pairs": stat["num_pairs"], "ratio": stat["ratio"], "wrong_hq_count": stat["wrong_hq_count"]} for stat in subset_stats],
    ["filename", "num_pairs", "ratio", "wrong_hq_count"],
)}

{"".join(
    f"### {stat['filename']}\n\n"
    + markdown_table(counter_rows(stat['quality_relation'], 'quality_relation'), ['quality_relation', 'count'])
    + "\n\n"
    + markdown_table(counter_rows(stat['structure_relation'], 'structure_relation'), ['structure_relation', 'count'])
    + "\n\n"
    + markdown_table(counter_rows(stat['lq_quality'], 'quality'), ['quality', 'count'])
    + "\n\n"
    + markdown_table(counter_rows(stat['hq_quality'], 'quality'), ['quality', 'count'])
    + "\n\n"
    + markdown_table(counter_rows(stat['lq_structure'], 'structure'), ['structure', 'count'])
    + "\n\n"
    + markdown_table(counter_rows(stat['hq_structure'], 'structure'), ['structure', 'count'])
    + "\n\n"
    for stat in subset_stats
)}

## 8. v1 vs v2 HQ 选择对比

- `num_label_groups`: `{num_label_groups}`
- `num_hq_changed`: `{len(changed_labels)}`
- `hq_changed_ratio`: `{(len(changed_labels) / num_label_groups) if num_label_groups else 0.0:.6f}`
- `v1 clean subset pair 数`: `{inputs_outputs['v1_clean_pairs']}`
- `v2 clean subset pair 数`: `{clean['num_pairs']}`
- `v1 wrong-HQ 数`: `{inputs_outputs['v1_wrong_hq_count']}`
- `v2 wrong-HQ 数`: `{hq_wrong_count}`

### HQ changed 示例

{markdown_table(changed_examples, ['label', 'v1_hq', 'v2_hq', 'v1_quality', 'v2_quality', 'v1_visual_score', 'v2_visual_score', 'v1_confidence', 'v2_confidence']) if changed_examples else "无"}

## 9. Quality Relation 分布

{markdown_table(counter_rows(quality_relation_counter, 'quality_relation'), ['quality_relation', 'count'])}

## 10. Structure Relation 分布

{markdown_table(counter_rows(structure_relation_counter, 'structure_relation'), ['structure_relation', 'count'])}

## 11. Wrong-HQ 与 all-OCR-wrong group 分析

- `HQ ocr_correct=False` 数量: `{hq_wrong_count}`
- `all_ocr_wrong_group` 数量: `{len(all_ocr_wrong_groups)}`

{markdown_table(
    [{"label": label} for label in all_ocr_wrong_groups[:20]],
    ['label'],
) if all_ocr_wrong_groups else "无"}

## 12. 可视化抽样说明

- 每个 bucket 默认抽样数: `{inputs_outputs['num_preview_per_bucket']}`
- 随机种子: `{inputs_outputs['seed']}`
- 预览图 buckets:

{markdown_table(preview_rows, ['bucket', 'count', 'sample_file'])}

## 13. 推荐用于 Stage 1 Diffusion 训练的 Manifest

优先推荐：

- `data/manifests/pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv`
- 理由：
  - 同结构，减少布局迁移
  - HQ 质量高于 LQ
  - HQ 由人工 quality 档 + 图像视觉质量指标选出
  - 不再依赖饱和的 OCR confidence

备选：

- `data/manifests/pair_manifest_top1_hq_visual_v2_same_structure.csv`

扩展：

- `data/manifests/pair_manifest_top1_hq_visual_v2.csv`

## 14. 警告与限制

- 同 label pair 不一定像素对齐。
- cross_structure pair 可能导致 diffusion loss 学习布局迁移。
- OCR confidence 接近饱和，不能完全代表图像质量。
- visual_quality_score 只是基于 sharpness/contrast/brightness/resolution 的启发式评分，不等同于人工主观质量。
- quality_improved 仍然依赖 raw quality 标注 `easy/middle/hard`。
{"- 本次运行使用了 --limit，仅反映子集统计。\n" if limit is not None else ""}

## 15. 下一步建议

人工检查 `previews/pair_samples_visual_v2/` 中的样例；若 `same_structure_quality_improved` 视觉质量可接受，则用该 manifest 作为 Stage 1 conditional latent DiT diffusion-loss 预训练的第一版训练集；随后再设计 DiT 数据读取与训练脚本。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def write_failure_report(report_path: Path, error: str):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Top1-HQ Visual-v2 Pair 构造与对比报告\n\n"
        "## 1. 范围与隔离声明\n\n"
        "本轮只在 `experiments/dit_lq_hq_v1/` 下尝试创建 v2 脚本、v2 manifest、v2 预览图和 v2 报告。\n"
        "未修改 v1 manifest、原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。\n\n"
        "## 14. 警告与限制\n\n"
        f"- 任务失败：{error}\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    try:
        lmdb_root, ocr_csv, v1_pair_csv, manifests_dir, preview_root, report_path = validate_args(args)
        if args.overwrite:
            remove_existing_outputs(manifests_dir, preview_root, report_path)

        accessor = LmdbAccessor(lmdb_root)
        try:
            ocr_rows, ocr_count, ocr_fields = load_ocr_rows(ocr_csv)
            total_samples, sample_rows = build_sample_rows(accessor, ocr_rows, args.limit)
            if args.limit is None and ocr_count != total_samples:
                raise ValueError(f"OCR CSV 样本数与 LMDB 样本数不一致: {ocr_count} != {total_samples}")

            train_samples_csv = manifests_dir / "train_samples_visual_quality_v2.csv"
            train_samples_jsonl = manifests_dir / "train_samples_visual_quality_v2.jsonl"
            write_train_samples(sample_rows, train_samples_csv, train_samples_jsonl)

            groups = defaultdict(list)
            for row in sample_rows:
                groups[row["label"]].append(row)
            v2_hq_by_label, all_ocr_wrong_groups = select_v2_hq(groups)
            groups, v2_pairs = build_v2_pairs(sample_rows, v2_hq_by_label)

            v2_pair_csv = manifests_dir / "pair_manifest_top1_hq_visual_v2.csv"
            write_manifest(v2_pair_csv, v2_pairs, PAIR_FIELDS)

            filtered = build_filtered_subsets(v2_pairs)
            filtered_paths = {name: manifests_dir / name for name in filtered}
            for name, rows in filtered.items():
                write_manifest(filtered_paths[name], rows, PAIR_FIELDS)

            v1_hq_meta = load_v1_hq_by_label(v1_pair_csv)
            subset_stats = [summarize_subset(name, rows, len(v2_pairs)) for name, rows in filtered.items()]
            sample_by_index = {row["lmdb_index"]: row for row in sample_rows}
            v1_clean_csv = manifests_dir / "pair_manifest_top1_hq_same_structure_quality_improved.csv"
            with v1_clean_csv.open("r", encoding="utf-8", newline="") as fp:
                v1_clean_pairs = sum(1 for _ in csv.DictReader(fp))
            v1_hq_rows_for_compare = {
                label: {
                    **v1,
                    **sample_by_index[v1["hq_lmdb_index"]],
                }
                for label, v1 in v1_hq_meta.items()
                if v1["hq_lmdb_index"] in sample_by_index
            }
            v1_wrong_hq_count = sum(1 for row in v1_hq_rows_for_compare.values() if not row["ocr_correct"])
            changed_labels = []
            for label, v1 in v1_hq_rows_for_compare.items():
                if label not in v2_hq_by_label:
                    continue
                if v1["hq_lmdb_index"] != v2_hq_by_label[label]["lmdb_index"]:
                    changed_labels.append(label)

            rng_samples = build_preview_samples(v2_pairs, changed_labels, groups, args.num_preview_per_bucket, args.seed)
            preview_root.mkdir(parents=True, exist_ok=True)
            preview_index: Dict[str, List[str]] = {}
            for bucket in PREVIEW_BUCKETS:
                bucket_dir = preview_root / bucket
                bucket_dir.mkdir(parents=True, exist_ok=True)
                preview_index[bucket] = []
                if bucket == "hq_changed_examples":
                    for label in rng_samples[bucket]:
                        v1_row = v1_hq_rows_for_compare[label]
                        v2_row = v2_hq_by_label[label]
                        filename = f"label_{label}_v1_{v1_row['lmdb_index']}_v2_{v2_row['lmdb_index']}"
                        safe_name = "".join(ch if ch.isalnum() else "_" for ch in filename)[:180] + ".jpg"
                        output_path = bucket_dir / safe_name
                        make_hq_change_preview(label, v1_row, v2_row, accessor, output_path)
                        preview_index[bucket].append(str(output_path.relative_to(preview_root)))
                else:
                    for row in rng_samples[bucket]:
                        filename = f"pair_{row['pair_id']}_lq_{row['lq_lmdb_index']}_hq_{row['hq_lmdb_index']}_{row['quality_relation']}_{row['structure_relation']}.jpg"
                        output_path = bucket_dir / filename
                        make_pair_preview(row, accessor, output_path)
                        preview_index[bucket].append(str(output_path.relative_to(preview_root)))
            write_report(
                report_path=report_path,
                inputs_outputs={
                    "lmdb_root": str(lmdb_root),
                    "ocr_csv": str(ocr_csv),
                    "v1_pair_csv": str(v1_pair_csv),
                    "train_samples_csv": str(train_samples_csv),
                    "train_samples_jsonl": str(train_samples_jsonl),
                    "v2_pair_csv": str(v2_pair_csv),
                    "v2_same_structure_csv": str(filtered_paths["pair_manifest_top1_hq_visual_v2_same_structure.csv"]),
                    "v2_quality_improved_csv": str(filtered_paths["pair_manifest_top1_hq_visual_v2_quality_improved.csv"]),
                    "v2_clean_csv": str(filtered_paths["pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv"]),
                    "preview_root": str(preview_root),
                    "report": str(report_path),
                    "num_preview_per_bucket": str(args.num_preview_per_bucket),
                    "seed": str(args.seed),
                    "v1_clean_pairs": str(v1_clean_pairs),
                    "v1_wrong_hq_count": str(v1_wrong_hq_count),
                },
                sample_rows=sample_rows,
                groups=groups,
                all_ocr_wrong_groups=all_ocr_wrong_groups,
                v2_pairs=v2_pairs,
                subset_stats=subset_stats,
                v1_hq_by_label=v1_hq_rows_for_compare,
                v2_hq_by_label=v2_hq_by_label,
                changed_labels=changed_labels,
                preview_index=preview_index,
                limit=args.limit,
            )
        finally:
            accessor.close()
    except Exception as exc:
        write_failure_report((args.out_dir.resolve() / "reports/pair_stats_top1_hq_visual_v2_report.md"), str(exc))
        raise


if __name__ == "__main__":
    main()
