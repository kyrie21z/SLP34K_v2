#!/usr/bin/env python3
import argparse
import csv
import hashlib
import html
import io
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
from PIL import Image


EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
QUALITY_PRIORITY = {"easy": 3, "middle": 2, "hard": 1}


def parse_args():
    parser = argparse.ArgumentParser(description="构建人工 HQ 选择候选包")
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--ocr-csv", type=Path, required=True)
    parser.add_argument("--visual-csv", type=Path, required=True)
    parser.add_argument("--v1-pair-csv", type=Path, required=True)
    parser.add_argument("--v2-pair-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups-per-page", type=int, default=20)
    parser.add_argument("--max-samples-per-group", type=int, default=30)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260504)
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
    visual_csv = args.visual_csv.resolve()
    v1_pair_csv = args.v1_pair_csv.resolve()
    v2_pair_csv = args.v2_pair_csv.resolve()
    out_dir = args.out_dir.resolve()

    for path, label in [
        (lmdb_root, "LMDB"),
        (ocr_csv, "OCR CSV"),
        (visual_csv, "visual CSV"),
        (v1_pair_csv, "v1 pair CSV"),
        (v2_pair_csv, "v2 pair CSV"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    if not is_relative_to(out_dir, SAFE_OUTPUT_ROOT):
        raise ValueError(f"out-dir 必须位于 {SAFE_OUTPUT_ROOT} 下: {out_dir}")
    if is_relative_to(out_dir, FORBIDDEN_OUTPUT_ROOT):
        raise ValueError(f"out-dir 不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {out_dir}")
    if args.groups_per_page <= 0:
        raise ValueError("--groups-per-page 必须大于 0")
    if args.max_samples_per_group <= 0:
        raise ValueError("--max-samples-per-group 必须大于 0")
    if args.limit_groups is not None and args.limit_groups <= 0:
        raise ValueError("--limit-groups 必须大于 0")
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在，未提供 --overwrite: {out_dir}")
    return lmdb_root, ocr_csv, visual_csv, v1_pair_csv, v2_pair_csv, out_dir


def resolve_ocr_fields(fieldnames: List[str]) -> Dict[str, str]:
    mapping = {}
    candidates = {
        "lmdb_index": ["lmdb_index"],
        "label": ["label", "raw_label"],
        "pred": ["pred", "ocr_pred"],
        "correct": ["correct", "ocr_correct"],
        "confidence": ["confidence", "avg_conf"],
        "avg_conf": ["avg_conf", "confidence"],
        "min_conf": ["min_conf"],
        "pred_length": ["pred_length"],
        "label_length": ["label_length"],
    }
    for key, names in candidates.items():
        for name in names:
            if name in fieldnames:
                mapping[key] = name
                break
        if key not in mapping:
            raise ValueError(f"OCR CSV 缺少字段 {key}，现有字段: {fieldnames}")
    return mapping


def resolve_visual_fields(fieldnames: List[str]) -> Dict[str, str]:
    required = [
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
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"visual CSV 缺少字段: {missing}")
    return {key: key for key in required}


def resolve_pair_fields(fieldnames: List[str]) -> Dict[str, str]:
    required = ["label", "hq_lmdb_index"]
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"pair CSV 缺少字段: {missing}")
    return {key: key for key in required}


class LmdbAccessor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False)
        self.label_cache: Dict[int, str] = {}
        self.meta_cache: Dict[int, Dict[str, Any]] = {}
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

    def get_image(self, idx: int) -> Image.Image:
        if idx not in self.image_cache:
            with self.env.begin(write=False) as txn:
                raw = txn.get(f"image-{idx:09d}".encode("utf-8"))
            if raw is None:
                raise ValueError(f"LMDB 缺少 image-{idx:09d}")
            self.image_cache[idx] = raw
        return Image.open(io.BytesIO(self.image_cache[idx])).convert("RGB")


def load_ocr_rows(path: Path):
    rows = {}
    count = 0
    with path.open("r", encoding="utf-8", newline="") as fp:
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
                "ocr_pred": row[fields["pred"]],
                "ocr_correct": parse_bool(row[fields["correct"]]),
                "confidence": as_float(row[fields["confidence"]]),
                "avg_conf": as_float(row[fields["avg_conf"]]),
                "min_conf": as_float(row[fields["min_conf"]]),
                "pred_length": as_int(row[fields["pred_length"]]),
                "label_length": as_int(row[fields["label_length"]]),
            }
    return rows, count, fields


def load_visual_rows(path: Path):
    rows = {}
    count = 0
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("visual CSV 没有表头")
        fields = resolve_visual_fields(reader.fieldnames)
        for row in reader:
            count += 1
            idx = as_int(row[fields["lmdb_index"]])
            if idx in rows:
                raise ValueError(f"visual CSV 存在重复 lmdb_index: {idx}")
            rows[idx] = {
                "lmdb_root": row[fields["lmdb_root"]],
                "lmdb_index": idx,
                "label": row[fields["label"]],
                "quality": row[fields["quality"]],
                "quality_priority": as_int(row[fields["quality_priority"]]),
                "structure": row[fields["structure"]],
                "structure_type": row[fields["structure_type"]],
                "source_path": row[fields["source_path"]],
                "ocr_pred": row[fields["ocr_pred"]],
                "ocr_correct": parse_bool(row[fields["ocr_correct"]]),
                "confidence": as_float(row[fields["confidence"]]),
                "avg_conf": as_float(row[fields["avg_conf"]]),
                "min_conf": as_float(row[fields["min_conf"]]),
                "sharpness": as_float(row[fields["sharpness"]]),
                "sharpness_norm": as_float(row[fields["sharpness_norm"]]),
                "contrast": as_float(row[fields["contrast"]]),
                "contrast_norm": as_float(row[fields["contrast_norm"]]),
                "brightness": as_float(row[fields["brightness"]]),
                "brightness_score": as_float(row[fields["brightness_score"]]),
                "resolution": as_float(row[fields["resolution"]]),
                "resolution_norm": as_float(row[fields["resolution_norm"]]),
                "visual_quality_score": as_float(row[fields["visual_quality_score"]]),
            }
    return rows, count, fields


def load_pair_hq_map(path: Path, label_name: str):
    hq_map = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"{label_name} 没有表头")
        fields = resolve_pair_fields(reader.fieldnames)
        for row in reader:
            label = row[fields["label"]]
            hq_idx = as_int(row[fields["hq_lmdb_index"]])
            if label in hq_map and hq_map[label] != hq_idx:
                raise ValueError(f"{label_name} 中同一 label 存在多个 HQ: {label}")
            hq_map[label] = hq_idx
    return hq_map


def merge_sample_rows(
    accessor: LmdbAccessor,
    ocr_rows: Dict[int, Dict[str, Any]],
    visual_rows: Dict[int, Dict[str, Any]],
):
    total = accessor.num_samples()
    if len(ocr_rows) != total:
        raise ValueError(f"OCR CSV 样本数与 LMDB num-samples 不一致: {len(ocr_rows)} != {total}")
    if len(visual_rows) != total:
        raise ValueError(f"visual CSV 样本数与 LMDB num-samples 不一致: {len(visual_rows)} != {total}")

    merged = []
    for idx in range(1, total + 1):
        label = accessor.get_label(idx)
        meta = accessor.get_meta(idx)
        if meta.get("id") != idx:
            raise ValueError(f"metadata.id 与 lmdb_index 不一致: {idx}")
        if meta.get("raw_label") != label:
            raise ValueError(f"metadata.raw_label 与 label 不一致: {idx}")
        if meta.get("split") != "train":
            raise ValueError(f"metadata.split 非 train: {idx}")
        if idx not in ocr_rows or idx not in visual_rows:
            raise ValueError(f"样本缺失 OCR/visual 信息: {idx}")
        ocr = ocr_rows[idx]
        visual = visual_rows[idx]
        if ocr["label"] != label:
            raise ValueError(f"OCR CSV label 与 LMDB label 不一致: {idx}")
        if visual["label"] != label:
            raise ValueError(f"visual CSV label 与 LMDB label 不一致: {idx}")
        if visual["quality"] != meta.get("quality"):
            raise ValueError(f"visual CSV quality 与 metadata 不一致: {idx}")
        merged.append(
            {
                **visual,
                "ocr_pred": ocr["ocr_pred"],
                "ocr_correct": ocr["ocr_correct"],
                "confidence": ocr["confidence"],
                "avg_conf": ocr["avg_conf"],
                "min_conf": ocr["min_conf"],
                "pred_length": ocr["pred_length"],
                "label_length": ocr["label_length"],
                "raw_label": meta.get("raw_label"),
                "metadata": meta,
            }
        )
    return merged


def compute_group_priority(rows: List[Dict[str, Any]], v1_hq_index: int, v2_hq_index: int):
    qualities = {row["quality"] for row in rows}
    structures = {row["structure"] for row in rows}
    has_quality_improvement = ("easy" in qualities) and (("middle" in qualities) or ("hard" in qualities))
    has_multiple_structures = len(structures) >= 2
    multiple_quality = len(qualities) >= 2
    v1_v2_different = int(v1_hq_index != v2_hq_index)
    score = (
        3.0 * int(has_quality_improvement)
        + 2.0 * v1_v2_different
        + 1.0 * int(has_multiple_structures)
        + 0.5 * int(multiple_quality)
        + math.log(len(rows))
    )
    return {
        "priority_score": score,
        "has_quality_improvement_potential": has_quality_improvement,
        "has_multiple_structures": has_multiple_structures,
        "has_multiple_quality": multiple_quality,
        "v1_v2_different": bool(v1_v2_different),
    }


def rank_group_samples(rows: List[Dict[str, Any]]):
    structure_counter = Counter(row["structure"] for row in rows)
    majority_structure = max(sorted(structure_counter), key=lambda key: (structure_counter[key], key))
    return sorted(
        rows,
        key=lambda row: (
            -row["quality_priority"],
            -row["visual_quality_score"],
            -row["sharpness_norm"],
            0 if row["structure"] == majority_structure else 1,
            row["lmdb_index"],
        ),
    )


def build_review_groups(
    sample_rows: List[Dict[str, Any]],
    v1_hq_map: Dict[str, int],
    v2_hq_map: Dict[str, int],
    limit_groups: Optional[int],
):
    by_label = defaultdict(list)
    for row in sample_rows:
        by_label[row["label"]].append(row)

    candidate_groups = []
    groups_eq1 = 0
    for label, rows in by_label.items():
        if len(rows) == 1:
            groups_eq1 += 1
            continue
        if label not in v1_hq_map or label not in v2_hq_map:
            raise ValueError(f"label 缺少 v1/v2 HQ 信息: {label}")
        v1_hq = v1_hq_map[label]
        v2_hq = v2_hq_map[label]
        sample_index_set = {row["lmdb_index"] for row in rows}
        if v1_hq not in sample_index_set or v2_hq not in sample_index_set:
            raise ValueError(f"v1/v2 HQ index 不在 group 内: {label}")
        extra = compute_group_priority(rows, v1_hq, v2_hq)
        ranked_rows = rank_group_samples(rows)
        qualities = Counter(row["quality"] for row in rows)
        structures = Counter(row["structure"] for row in rows)
        group = {
            "label": label,
            "group_size": len(rows),
            "num_easy": qualities.get("easy", 0),
            "num_middle": qualities.get("middle", 0),
            "num_hard": qualities.get("hard", 0),
            "num_single": structures.get("single", 0),
            "num_multi": structures.get("multi", 0),
            "num_vertical": structures.get("vertical", 0),
            "has_ocr_wrong": any(not row["ocr_correct"] for row in rows),
            "v1_hq_index": v1_hq,
            "v2_hq_index": v2_hq,
            "v1_v2_same": v1_hq == v2_hq,
            **extra,
            "rows": ranked_rows,
        }
        candidate_groups.append(group)

    candidate_groups.sort(
        key=lambda group: (
            -group["priority_score"],
            -group["group_size"],
            group["label"],
        )
    )
    for rank, group in enumerate(candidate_groups, start=1):
        group["group_rank"] = rank
        group["label_hash"] = hashlib.md5(group["label"].encode("utf-8")).hexdigest()[:8]

    if limit_groups is not None:
        candidate_groups = candidate_groups[:limit_groups]
    return candidate_groups, len(by_label), groups_eq1


def export_group_images(
    accessor: LmdbAccessor,
    candidate_groups: List[Dict[str, Any]],
    images_root: Path,
):
    local_paths = {}
    for group in candidate_groups:
        group_dir = images_root / f"group_{group['group_rank']:04d}_{group['label_hash']}"
        group_dir.mkdir(parents=True, exist_ok=True)
        for row in group["rows"]:
            img = accessor.get_image(row["lmdb_index"])
            image = img.copy()
            image.thumbnail((320, 320))
            output_path = group_dir / f"idx_{row['lmdb_index']}.jpg"
            image.save(output_path, quality=90)
            local_paths[row["lmdb_index"]] = output_path
    return local_paths


def write_candidate_csvs(
    candidate_groups: List[Dict[str, Any]],
    local_paths: Dict[int, Path],
    groups_csv: Path,
    samples_csv: Path,
    template_csv: Path,
    groups_per_page: int,
):
    groups_csv.parent.mkdir(parents=True, exist_ok=True)
    review_page_lookup = {}
    for idx, group in enumerate(candidate_groups, start=1):
        page_index = (idx - 1) // groups_per_page + 1
        review_page_lookup[group["group_rank"]] = f"page_{page_index:04d}.html"

    with groups_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "group_rank", "label", "label_hash", "group_size",
            "num_easy", "num_middle", "num_hard",
            "num_single", "num_multi", "num_vertical",
            "has_ocr_wrong", "v1_hq_index", "v2_hq_index",
            "v1_v2_same", "priority_score", "review_page",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for group in candidate_groups:
            writer.writerow(
                {
                    "group_rank": group["group_rank"],
                    "label": group["label"],
                    "label_hash": group["label_hash"],
                    "group_size": group["group_size"],
                    "num_easy": group["num_easy"],
                    "num_middle": group["num_middle"],
                    "num_hard": group["num_hard"],
                    "num_single": group["num_single"],
                    "num_multi": group["num_multi"],
                    "num_vertical": group["num_vertical"],
                    "has_ocr_wrong": group["has_ocr_wrong"],
                    "v1_hq_index": group["v1_hq_index"],
                    "v2_hq_index": group["v2_hq_index"],
                    "v1_v2_same": group["v1_v2_same"],
                    "priority_score": group["priority_score"],
                    "review_page": review_page_lookup[group["group_rank"]],
                }
            )

    with samples_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "label", "label_hash", "lmdb_index", "quality", "quality_priority",
            "structure", "structure_type", "ocr_correct", "ocr_pred", "confidence",
            "visual_quality_score", "sharpness_norm", "contrast_norm", "brightness_score",
            "resolution_norm", "source_path", "is_v1_hq", "is_v2_hq", "local_image_path",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for group in candidate_groups:
            for row in group["rows"]:
                writer.writerow(
                    {
                        "label": group["label"],
                        "label_hash": group["label_hash"],
                        "lmdb_index": row["lmdb_index"],
                        "quality": row["quality"],
                        "quality_priority": row["quality_priority"],
                        "structure": row["structure"],
                        "structure_type": row["structure_type"],
                        "ocr_correct": row["ocr_correct"],
                        "ocr_pred": row["ocr_pred"],
                        "confidence": row["confidence"],
                        "visual_quality_score": row["visual_quality_score"],
                        "sharpness_norm": row["sharpness_norm"],
                        "contrast_norm": row["contrast_norm"],
                        "brightness_score": row["brightness_score"],
                        "resolution_norm": row["resolution_norm"],
                        "source_path": row["source_path"],
                        "is_v1_hq": row["lmdb_index"] == group["v1_hq_index"],
                        "is_v2_hq": row["lmdb_index"] == group["v2_hq_index"],
                        "local_image_path": str(local_paths[row["lmdb_index"]].relative_to(samples_csv.parent.parent)),
                    }
                )

    with template_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "label", "label_hash", "group_size", "v1_hq_index", "v2_hq_index",
            "suggested_hq_index", "suggested_source", "manual_hq_index",
            "review_status", "review_note",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for group in candidate_groups:
            writer.writerow(
                {
                    "label": group["label"],
                    "label_hash": group["label_hash"],
                    "group_size": group["group_size"],
                    "v1_hq_index": group["v1_hq_index"],
                    "v2_hq_index": group["v2_hq_index"],
                    "suggested_hq_index": group["v2_hq_index"],
                    "suggested_source": "visual_v2",
                    "manual_hq_index": "",
                    "review_status": "pending",
                    "review_note": "",
                }
            )


def make_group_html(group: Dict[str, Any], out_dir: Path, max_samples_per_group: int):
    dist_quality = f"easy={group['num_easy']} / middle={group['num_middle']} / hard={group['num_hard']}"
    dist_structure = f"single={group['num_single']} / multi={group['num_multi']} / vertical={group['num_vertical']}"
    cards = []
    for row in group["rows"][:max_samples_per_group]:
        image_rel = f"../images/group_{group['group_rank']:04d}_{group['label_hash']}/idx_{row['lmdb_index']}.jpg"
        classes = []
        badges = []
        if row["lmdb_index"] == group["v1_hq_index"]:
            classes.append("v1")
            badges.append("v1 HQ")
        if row["lmdb_index"] == group["v2_hq_index"]:
            classes.append("v2")
            badges.append("v2 HQ")
        if row["lmdb_index"] == group["v1_hq_index"] and row["lmdb_index"] == group["v2_hq_index"]:
            classes.append("both")
        if not row["ocr_correct"]:
            classes.append("ocr-wrong")
            badges.append("OCR Wrong")
        class_name = "candidate " + " ".join(classes)
        badges_html = " ".join(f"<span class='badge'>{html.escape(text)}</span>" for text in badges)
        cards.append(
            f"""
            <div class="{class_name}">
              <img src="{html.escape(image_rel)}" alt="{row['lmdb_index']}">
              <div class="meta">
                <div><strong>idx:</strong> {row['lmdb_index']}</div>
                <div><strong>quality:</strong> {html.escape(row['quality'])}</div>
                <div><strong>structure:</strong> {html.escape(row['structure'])}</div>
                <div><strong>visual:</strong> {row['visual_quality_score']:.6f}</div>
                <div><strong>sharpness:</strong> {row['sharpness_norm']:.6f}</div>
                <div><strong>contrast:</strong> {row['contrast_norm']:.6f}</div>
                <div><strong>brightness:</strong> {row['brightness_score']:.6f}</div>
                <div><strong>ocr_correct:</strong> {row['ocr_correct']}</div>
                <div><strong>source:</strong> {html.escape(row['source_path'])}</div>
                <div class="badges">{badges_html}</div>
              </div>
            </div>
            """
        )
    hidden_count = max(0, len(group["rows"]) - max_samples_per_group)
    hidden_note = f"<p class='hidden-note'>该 group 还有 {hidden_count} 张图未在 HTML 中展示，请参考 candidate_samples.csv。</p>" if hidden_count else ""
    return f"""
    <section class="group-card" id="group-{group['group_rank']:04d}">
      <h2>#{group['group_rank']:04d} {html.escape(group['label'])}</h2>
      <p><strong>group_size:</strong> {group['group_size']} | <strong>priority_score:</strong> {group['priority_score']:.4f}</p>
      <p><strong>quality 分布:</strong> {dist_quality}</p>
      <p><strong>structure 分布:</strong> {dist_structure}</p>
      <p><strong>v1_hq_index:</strong> {group['v1_hq_index']} | <strong>v2_hq_index:</strong> {group['v2_hq_index']} | <strong>人工填写:</strong> 请在 manual_hq_selection_template.csv 中填写 manual_hq_index</p>
      {hidden_note}
      <div class="candidate-grid">
        {''.join(cards)}
      </div>
    </section>
    """


def write_review_pages(candidate_groups: List[Dict[str, Any]], review_pages_dir: Path, groups_per_page: int, max_samples_per_group: int):
    review_pages_dir.mkdir(parents=True, exist_ok=True)
    page_index_rows = []
    total_pages = math.ceil(len(candidate_groups) / groups_per_page) if candidate_groups else 0
    for page_idx in range(total_pages):
        start = page_idx * groups_per_page
        end = min(len(candidate_groups), (page_idx + 1) * groups_per_page)
        groups = candidate_groups[start:end]
        group_html = "".join(make_group_html(group, review_pages_dir, max_samples_per_group) for group in groups)
        page_name = f"page_{page_idx + 1:04d}.html"
        page_path = review_pages_dir / page_name
        page_path.write_text(
            f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Manual HQ Review {page_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; }}
    h1 {{ margin-bottom: 8px; }}
    .group-card {{ background: white; border: 1px solid #ddd; padding: 16px; margin-bottom: 20px; border-radius: 10px; }}
    .candidate-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
    .candidate {{ border: 2px solid #ccc; border-radius: 8px; background: #fff; overflow: hidden; }}
    .candidate.v1 {{ border-color: #2b6cb0; }}
    .candidate.v2 {{ border-color: #2f855a; }}
    .candidate.both {{ border-color: #805ad5; }}
    .candidate.ocr-wrong {{ box-shadow: 0 0 0 3px rgba(220,38,38,0.2); }}
    .candidate img {{ width: 100%; display: block; background: #fff; }}
    .candidate .meta {{ padding: 10px; font-size: 12px; line-height: 1.45; }}
    .badge {{ display: inline-block; margin-right: 6px; padding: 2px 6px; border-radius: 12px; background: #edf2f7; }}
    .hidden-note {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Manual HQ Review - {page_name}</h1>
  <p>覆盖 group rank: {groups[0]['group_rank']:04d} - {groups[-1]['group_rank']:04d}</p>
  {group_html}
</body>
</html>
""",
            encoding="utf-8",
        )
        page_index_rows.append(
            {
                "page_file": page_name,
                "group_rank_range": f"{groups[0]['group_rank']:04d}-{groups[-1]['group_rank']:04d}",
                "num_groups": len(groups),
            }
        )
    return page_index_rows


def write_instructions(path: Path):
    path.write_text(
        """# Manual HQ Selection Instructions

## 1. 审核目标

为每个 same-label group 人工选择一张最适合作为 HQ reference 的图像。
后续会使用该 HQ 与 group 内其他样本构造 LQ-HQ pair。

## 2. 推荐 HQ 选择标准

1. 字符完整，无明显遮挡
2. 字符边缘清楚，中文笔画可辨
3. 结构稳定，布局代表该船常见形态
4. 不过曝、不过暗
5. 裁剪范围合理，船牌区域居中
6. 尽量与多数 LQ 的 structure 一致
7. 若多张都合适，优先 easy，其次 middle，再 hard
8. 不要只看 OCR confidence 或 visual_quality_score

## 3. 填写方式

请编辑 `manual_hq_selection_template.csv`，填写以下字段：

- `manual_hq_index`
- `review_status`
- `review_note`

示例：

- `manual_hq_index = 4718`
- `review_status = reviewed`
- `review_note = 清晰且结构代表性强`

## 4. 注意事项

如果没有合适 HQ，可将 `review_status` 写为 `skip`。
如果多个候选都很好，选最完整且结构最稳定的一张。
如果图像显示异常，请在 `review_note` 中说明。
""",
        encoding="utf-8",
    )


def write_report(
    out_dir: Path,
    num_total_samples: int,
    candidate_groups: List[Dict[str, Any]],
    unique_label_count: int,
    groups_eq1: int,
    page_index_rows: List[Dict[str, Any]],
):
    groups_ge2 = len(candidate_groups)
    group_sizes = [group["group_size"] for group in candidate_groups]
    report_path = out_dir / "manual_hq_review_report.md"
    num_groups_v1_v2_diff = sum(1 for group in candidate_groups if not group["v1_v2_same"])
    num_groups_multiple_quality = sum(1 for group in candidate_groups if group["has_multiple_quality"])
    num_groups_multiple_structure = sum(1 for group in candidate_groups if group["has_multiple_structures"])
    review_page_table = "| page file | covered group rank range | num groups |\n| --- | --- | ---: |\n" + "\n".join(
        f"| {row['page_file']} | {row['group_rank_range']} | {row['num_groups']} |"
        for row in page_index_rows
    ) if page_index_rows else "无"
    report_path.write_text(
        f"""# Manual HQ Review Package 构建报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/manual_hq_review/` 下创建人工审核页面、候选 CSV、模板和报告。
未修改 v1/v2 manifest、原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输出目录: `{out_dir}`
- review_pages: `{out_dir / 'review_pages'}`
- images: `{out_dir / 'images'}`
- candidate_groups.csv: `{out_dir / 'candidate_groups.csv'}`
- candidate_samples.csv: `{out_dir / 'candidate_samples.csv'}`
- manual_hq_selection_template.csv: `{out_dir / 'manual_hq_selection_template.csv'}`
- manual_hq_selection_instructions.md: `{out_dir / 'manual_hq_selection_instructions.md'}`
- manual_hq_review_report.md: `{report_path}`

## 3. 样本与 Group 统计

- `num_total_samples`: `{num_total_samples}`
- `num_unique_labels`: `{unique_label_count}`
- `num_groups_ge2`: `{groups_ge2}`
- `num_groups_eq1`: `{groups_eq1}`
- `max_group_size`: `{max(group_sizes) if group_sizes else 0}`
- `median_group_size`: `{statistics.median(group_sizes) if group_sizes else 0}`
- `num_groups_v1_v2_hq_different`: `{num_groups_v1_v2_diff}`
- `num_groups_has_multiple_quality`: `{num_groups_multiple_quality}`
- `num_groups_has_multiple_structure`: `{num_groups_multiple_structure}`

## 4. Group 优先级排序规则

当前排序使用如下优先级分数：

`priority_score = 3 * has_quality_improvement_potential + 2 * v1_v2_different + 1 * has_multiple_structures + 0.5 * has_multiple_quality + log(group_size)`

含义：

- 优先审核存在 `hard/middle` 与 `easy` 共存的 group
- 优先审核 v1/v2 HQ 不一致的 group
- 优先审核结构多样的 group
- group 越大，优先级越高

## 5. 候选图排序规则

每个 group 内候选图展示排序为：

1. `quality_priority` 降序：`easy=3 > middle=2 > hard=1`
2. `visual_quality_score` 降序
3. `sharpness_norm` 降序
4. `majority structure` 优先
5. `lmdb_index` 升序

注意：这只是展示排序，不是最终 HQ 决策。

## 6. v1/v2 HQ 参考信息

- `v1_hq_index` 与 `v2_hq_index` 都已写入 `candidate_groups.csv`
- `candidate_samples.csv` 中标记了 `is_v1_hq` 与 `is_v2_hq`
- review HTML 中：
  - v1 HQ: 蓝色边框
  - v2 HQ: 绿色边框
  - 同时是 v1/v2 HQ: 紫色边框
  - OCR wrong: 红色提示

## 7. Review Pages 索引

{review_page_table}

## 8. 输出文件说明

- `candidate_groups.csv`: group 级别统计与排序结果
- `candidate_samples.csv`: group 内候选图的完整样本表
- `manual_hq_selection_template.csv`: 人工填写模板，`manual_hq_index` 为空
- `manual_hq_selection_instructions.md`: 中文审核说明
- `review_pages/*.html`: 离线审核页面

## 9. 人工审核流程

1. 打开 `review_pages/` 下的 HTML 页面浏览 group
2. 对照 `candidate_samples.csv` 与页面中的候选图
3. 在 `manual_hq_selection_template.csv` 中填写：
   - `manual_hq_index`
   - `review_status`
   - `review_note`
4. 审核完成后，下一步再基于该模板生成 manual HQ pair manifest

## 10. 警告与限制

- 当前页面展示排序只是辅助，不是最终标准
- OCR confidence 仅作为参考信息，不能替代人工判断
- visual_quality_score 仅是启发式指标，不等同于主观视觉质量
- HTML 页面默认最多展示每组前若干张候选，完整 group 仍需参考 `candidate_samples.csv`
- 候选图片已缩放到较小尺寸，最终判断如有疑问应结合原图索引

## 11. 下一步建议

人工填写 `manual_hq_selection_template.csv` 后，下一步运行脚本检查 `manual_hq_index` 合法性，并生成 `pair_manifest_manual_hq.csv` 及其过滤子集。
""",
        encoding="utf-8",
    )


def write_failure_report(out_dir: Path, error: str):
    report_path = out_dir / "manual_hq_review_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Manual HQ Review Package 构建报告\n\n"
        "## 1. 范围与隔离声明\n\n"
        "本轮只在 `experiments/dit_lq_hq_v1/manual_hq_review/` 下尝试创建人工审核页面、候选 CSV、模板和报告。\n"
        "未修改 v1/v2 manifest、原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。\n\n"
        "## 10. 警告与限制\n\n"
        f"- 任务失败：{error}\n",
        encoding="utf-8",
    )


def remove_existing_out_dir(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)


def main():
    args = parse_args()
    try:
        lmdb_root, ocr_csv, visual_csv, v1_pair_csv, v2_pair_csv, out_dir = validate_args(args)
        if args.overwrite:
            remove_existing_out_dir(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        accessor = LmdbAccessor(lmdb_root)
        try:
            ocr_rows, _, _ = load_ocr_rows(ocr_csv)
            visual_rows, _, _ = load_visual_rows(visual_csv)
            v1_hq_map = load_pair_hq_map(v1_pair_csv, "v1 pair CSV")
            v2_hq_map = load_pair_hq_map(v2_pair_csv, "v2 pair CSV")
            sample_rows = merge_sample_rows(accessor, ocr_rows, visual_rows)
            candidate_groups, unique_label_count, groups_eq1 = build_review_groups(
                sample_rows, v1_hq_map, v2_hq_map, args.limit_groups
            )

            images_root = out_dir / "images"
            local_paths = export_group_images(accessor, candidate_groups, images_root)

            review_pages_dir = out_dir / "review_pages"
            groups_csv = out_dir / "candidate_groups.csv"
            samples_csv = out_dir / "candidate_samples.csv"
            template_csv = out_dir / "manual_hq_selection_template.csv"
            write_candidate_csvs(
                candidate_groups,
                local_paths,
                groups_csv,
                samples_csv,
                template_csv,
                args.groups_per_page,
            )

            page_index_rows = write_review_pages(candidate_groups, review_pages_dir, args.groups_per_page, args.max_samples_per_group)
            write_instructions(out_dir / "manual_hq_selection_instructions.md")
            write_report(out_dir, len(sample_rows), candidate_groups, unique_label_count, groups_eq1, page_index_rows)
        finally:
            accessor.close()
    except Exception as exc:
        write_failure_report(args.out_dir.resolve(), str(exc))
        raise

if __name__ == "__main__":
    main()
