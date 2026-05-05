#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image, ImageDraw, ImageFont

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


PROJECT_ROOT = Path("/mnt/data/zyx/SLP34K_v2").resolve()
EXPERIMENT_ROOT = (PROJECT_ROOT / "experiments" / "dit_lq_hq_v1").resolve()
ALLOWED_OUT_DIR = (EXPERIMENT_ROOT / "qwen_vl_hq_review_full").resolve()
RUNTIME_REPORT_NAME = "Stage00_step11_full_qwen_vl_hq_review_report.md"
CODE_REPORT_NAME = "Stage00_step11_full_qwen_vl_hq_review_code_report.md"
RUNTIME_ARTIFACTS = [
    "panels",
    "responses",
    "qwen_full_hq_selection_raw.jsonl",
    "qwen_full_hq_selection_parsed.csv",
    "qwen_full_need_human_review.csv",
    "qwen_full_failed_cases.csv",
    "progress_state.json",
]
PARSED_FIELDNAMES = [
    "label",
    "label_hash",
    "group_rank",
    "group_size",
    "selected_candidate_id",
    "selected_lmdb_index",
    "confidence",
    "need_human_review",
    "reason",
    "second_best_lmdb_index",
    "risk_flags",
    "json_parse_ok",
    "illegal_selection",
    "panel_path",
    "response_path",
    "selected_quality",
    "selected_structure",
    "selected_visual_quality_score",
    "manual_hq_index",
    "manual_exact_match_if_available",
]
FAILED_FIELDNAMES = [
    "label",
    "label_hash",
    "group_rank",
    "error_stage",
    "error_type",
    "error_message",
    "manual_hq_index",
    "panel_path",
    "response_path",
]
HUMAN_REVIEW_RISK_FLAGS = {"no_clear_hq", "ambiguous", "layout_conflict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全量 Qwen3-VL HQ 审核脚本")
    parser.add_argument(
        "--candidate-groups",
        type=Path,
        default=EXPERIMENT_ROOT / "manual_hq_review/candidate_groups.csv",
    )
    parser.add_argument(
        "--candidate-samples",
        type=Path,
        default=EXPERIMENT_ROOT / "manual_hq_review/candidate_samples.csv",
    )
    parser.add_argument(
        "--review-images-root",
        type=Path,
        default=EXPERIMENT_ROOT / "manual_hq_review/images",
    )
    parser.add_argument(
        "--manual-selection",
        type=Path,
        default=EXPERIMENT_ROOT / "manual_hq_review_site/data/manual_hq_selection_export.csv",
    )
    parser.add_argument(
        "--manual-selection-fallback",
        type=Path,
        default=EXPERIMENT_ROOT / "manual_hq_review_site/data/manual_hq_selection_live.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ALLOWED_OUT_DIR,
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:22002/v1")
    parser.add_argument("--model", default="/mnt/data/zyx/llm/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--start-rank", type=int, default=None)
    parser.add_argument("--end-rank", type=int, default=None)
    parser.add_argument("--max-candidates-per-panel", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def as_int(value: Any) -> int:
    return int(str(value).strip())


def as_float(value: Any) -> float:
    return float(str(value).strip())


def load_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头: {path}")
        rows = list(reader)
    return rows, reader.fieldnames


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir = out_dir.resolve()
    if out_dir != ALLOWED_OUT_DIR:
        raise ValueError(f"out-dir 仅允许为: {ALLOWED_OUT_DIR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def clear_runtime_artifacts(out_dir: Path) -> None:
    for name in RUNTIME_ARTIFACTS:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    runtime_report = out_dir / "reports" / RUNTIME_REPORT_NAME
    if runtime_report.exists():
        runtime_report.unlink()


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.candidate_groups = args.candidate_groups.resolve()
    args.candidate_samples = args.candidate_samples.resolve()
    args.review_images_root = args.review_images_root.resolve()
    args.manual_selection = args.manual_selection.resolve()
    args.manual_selection_fallback = args.manual_selection_fallback.resolve()
    args.out_dir = ensure_out_dir(args.out_dir)

    for path, label in [
        (args.candidate_groups, "candidate_groups.csv"),
        (args.candidate_samples, "candidate_samples.csv"),
        (args.review_images_root, "review images root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    if args.max_candidates_per_panel <= 0:
        raise ValueError("--max-candidates-per-panel 必须大于 0")
    if args.flush_every <= 0:
        raise ValueError("--flush-every 必须大于 0")
    if args.limit_groups is not None and args.limit_groups <= 0:
        raise ValueError("--limit-groups 必须大于 0")
    if args.start_rank is not None and args.start_rank <= 0:
        raise ValueError("--start-rank 必须大于 0")
    if args.end_rank is not None and args.end_rank <= 0:
        raise ValueError("--end-rank 必须大于 0")
    if args.start_rank is not None and args.end_rank is not None and args.start_rank > args.end_rank:
        raise ValueError("--start-rank 不能大于 --end-rank")
    if args.max_retries <= 0:
        raise ValueError("--max-retries 必须大于 0")
    if args.retry_sleep < 0:
        raise ValueError("--retry-sleep 不能小于 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout 必须大于 0")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite 与 --resume 不能同时使用")
    return args


def load_candidate_groups(path: Path) -> List[Dict[str, Any]]:
    rows, fields = load_csv_rows(path)
    required = [
        "group_rank",
        "label",
        "label_hash",
        "group_size",
        "num_easy",
        "num_middle",
        "num_hard",
        "num_single",
        "num_multi",
        "num_vertical",
        "has_ocr_wrong",
        "v1_hq_index",
        "v2_hq_index",
        "v1_v2_same",
        "priority_score",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"candidate_groups.csv 缺少字段: {missing}")
    groups = []
    for row in rows:
        groups.append(
            {
                "group_rank": as_int(row["group_rank"]),
                "label": row["label"],
                "label_hash": row["label_hash"],
                "group_size": as_int(row["group_size"]),
                "num_easy": as_int(row["num_easy"]),
                "num_middle": as_int(row["num_middle"]),
                "num_hard": as_int(row["num_hard"]),
                "num_single": as_int(row["num_single"]),
                "num_multi": as_int(row["num_multi"]),
                "num_vertical": as_int(row["num_vertical"]),
                "has_ocr_wrong": parse_bool(row["has_ocr_wrong"]),
                "v1_hq_index": as_int(row["v1_hq_index"]),
                "v2_hq_index": as_int(row["v2_hq_index"]),
                "v1_v2_same": parse_bool(row["v1_v2_same"]),
                "priority_score": as_float(row["priority_score"]),
            }
        )
    groups.sort(key=lambda item: item["group_rank"])
    return groups


def resolve_sample_image_path(review_images_root: Path, local_image_path: str) -> Path:
    rel = local_image_path.strip().replace("\\", "/")
    prefix = "manual_hq_review/images/"
    if rel.startswith(prefix):
        rel = rel[len(prefix) :]
    path = (review_images_root / rel).resolve()
    if path.exists():
        return path
    alt = (PROJECT_ROOT / local_image_path).resolve()
    if alt.exists():
        return alt
    return path


def load_candidate_samples(path: Path, review_images_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows, fields = load_csv_rows(path)
    required = [
        "label",
        "label_hash",
        "lmdb_index",
        "quality",
        "quality_priority",
        "structure",
        "structure_type",
        "ocr_correct",
        "ocr_pred",
        "confidence",
        "visual_quality_score",
        "sharpness_norm",
        "contrast_norm",
        "brightness_score",
        "resolution_norm",
        "source_path",
        "is_v1_hq",
        "is_v2_hq",
        "local_image_path",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"candidate_samples.csv 缺少字段: {missing}")
    samples_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = {
            "label": row["label"],
            "label_hash": row["label_hash"],
            "lmdb_index": as_int(row["lmdb_index"]),
            "quality": row["quality"],
            "quality_priority": as_int(row["quality_priority"]),
            "structure": row["structure"],
            "structure_type": row["structure_type"],
            "ocr_correct": parse_bool(row["ocr_correct"]),
            "ocr_pred": row["ocr_pred"],
            "confidence": as_float(row["confidence"]),
            "visual_quality_score": as_float(row["visual_quality_score"]),
            "sharpness_norm": as_float(row["sharpness_norm"]),
            "contrast_norm": as_float(row["contrast_norm"]),
            "brightness_score": as_float(row["brightness_score"]),
            "resolution_norm": as_float(row["resolution_norm"]),
            "source_path": row["source_path"],
            "is_v1_hq": parse_bool(row["is_v1_hq"]),
            "is_v2_hq": parse_bool(row["is_v2_hq"]),
            "local_image_path": row["local_image_path"],
            "resolved_image_path": resolve_sample_image_path(review_images_root, row["local_image_path"]),
        }
        samples_by_hash[sample["label_hash"]].append(sample)
    for label_hash in samples_by_hash:
        samples_by_hash[label_hash].sort(
            key=lambda item: (
                -item["quality_priority"],
                -item["visual_quality_score"],
                -item["sharpness_norm"],
                item["lmdb_index"],
            )
        )
    return samples_by_hash


def load_manual_selection_map(primary: Path, fallback: Path) -> Tuple[Dict[str, int], Dict[str, Any]]:
    manual_map: Dict[str, int] = {}
    source_note = {
        "primary_exists": primary.exists(),
        "fallback_exists": fallback.exists(),
        "primary_reviewed_count": 0,
        "fallback_reviewed_count": 0,
        "manual_available_count": 0,
        "manual_invalid_count": 0,
        "primary_used_count": 0,
        "fallback_used_count": 0,
    }

    def read_map(path: Path, key: str) -> Dict[str, int]:
        if not path.exists():
            return {}
        rows, fields = load_csv_rows(path)
        required = ["label_hash", "manual_hq_index", "review_status"]
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"人工选择 CSV 缺少字段 {missing}: {path}")
        local_map: Dict[str, int] = {}
        reviewed_count = 0
        invalid_count = 0
        for row in rows:
            if (row.get("review_status") or "").strip() != "reviewed":
                continue
            manual_raw = (row.get("manual_hq_index") or "").strip()
            if not manual_raw:
                continue
            reviewed_count += 1
            try:
                local_map[row["label_hash"]] = as_int(manual_raw)
            except Exception:
                invalid_count += 1
        source_note[f"{key}_reviewed_count"] = reviewed_count
        source_note["manual_invalid_count"] += invalid_count
        return local_map

    primary_map = read_map(primary, "primary")
    fallback_map = read_map(fallback, "fallback")
    for label_hash, manual_idx in primary_map.items():
        manual_map[label_hash] = manual_idx
        source_note["primary_used_count"] += 1
    for label_hash, manual_idx in fallback_map.items():
        if label_hash not in manual_map:
            manual_map[label_hash] = manual_idx
            source_note["fallback_used_count"] += 1
    source_note["manual_available_count"] = len(manual_map)
    return manual_map, source_note


def load_font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def choose_panel_candidates(
    group: Dict[str, Any],
    samples: List[Dict[str, Any]],
    manual_hq_index: Optional[int],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    if len(samples) <= max_candidates:
        return list(samples)

    selected: List[Dict[str, Any]] = []
    seen = set()
    majority_structure = Counter(sample["structure"] for sample in samples).most_common(1)[0][0]
    samples_by_idx = {sample["lmdb_index"]: sample for sample in samples}

    def add_sample(sample: Optional[Dict[str, Any]]) -> None:
        if sample is None:
            return
        idx = sample["lmdb_index"]
        if idx not in seen and len(selected) < max_candidates:
            selected.append(sample)
            seen.add(idx)

    for idx in [manual_hq_index, group["v1_hq_index"], group["v2_hq_index"]]:
        if idx is not None:
            add_sample(samples_by_idx.get(idx))

    for sample in sorted(
        [item for item in samples if item["quality"] == "easy"],
        key=lambda item: (-item["visual_quality_score"], -item["sharpness_norm"], item["lmdb_index"]),
    ):
        add_sample(sample)

    for sample in sorted(
        [item for item in samples if item["structure"] == majority_structure],
        key=lambda item: (-item["visual_quality_score"], -item["sharpness_norm"], item["lmdb_index"]),
    ):
        add_sample(sample)

    for sample in sorted(
        samples,
        key=lambda item: (
            -item["visual_quality_score"],
            -item["sharpness_norm"],
            -item["contrast_norm"],
            item["lmdb_index"],
        ),
    ):
        add_sample(sample)
    return selected


def make_panel(
    group: Dict[str, Any],
    panel_samples: List[Dict[str, Any]],
    manual_hq_index: Optional[int],
    output_path: Path,
) -> List[Dict[str, Any]]:
    missing_images = [sample["lmdb_index"] for sample in panel_samples if not sample["resolved_image_path"].exists()]
    if missing_images:
        raise FileNotFoundError(f"panel 候选样本缺图: {missing_images[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    thumb_w = 240
    thumb_h = 180
    text_h = 104
    card_w = thumb_w + 20
    card_h = thumb_h + text_h + 20
    margin = 20
    header_h = 126
    rows = math.ceil(len(panel_samples) / cols)
    panel_w = cols * card_w + (cols + 1) * margin
    panel_h = header_h + rows * card_h + (rows + 1) * margin

    image = Image.new("RGB", (panel_w, panel_h), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    meta_font = load_font(18)
    small_font = load_font(16)
    tiny_font = load_font(15)

    title = f"Group #{group['group_rank']:04d} | {group['label_hash']} | size={group['group_size']}"
    draw.text((margin, 14), title, fill=(15, 23, 42), font=title_font)
    draw.text((margin, 48), f"Label: {group['label']}", fill=(30, 41, 59), font=meta_font)
    manual_text = str(manual_hq_index) if manual_hq_index is not None else "none"
    draw.text(
        (margin, 76),
        f"manual={manual_text}  v1={group['v1_hq_index']}  v2={group['v2_hq_index']}  quality(e/m/h)={group['num_easy']}/{group['num_middle']}/{group['num_hard']}",
        fill=(51, 65, 85),
        font=small_font,
    )

    mappings: List[Dict[str, Any]] = []
    for candidate_id, sample in enumerate(panel_samples, start=1):
        row = (candidate_id - 1) // cols
        col = (candidate_id - 1) % cols
        x0 = margin + col * (card_w + margin)
        y0 = header_h + margin + row * (card_h + margin)
        x1 = x0 + card_w
        y1 = y0 + card_h
        border_color = (209, 213, 219)
        if manual_hq_index is not None and sample["lmdb_index"] == manual_hq_index:
            border_color = (22, 163, 74)
        elif sample["is_v1_hq"] and sample["is_v2_hq"]:
            border_color = (126, 34, 206)
        elif sample["is_v1_hq"]:
            border_color = (29, 78, 216)
        elif sample["is_v2_hq"]:
            border_color = (5, 150, 105)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(255, 255, 255), outline=border_color, width=4)

        thumb = Image.open(sample["resolved_image_path"]).convert("RGB")
        thumb.thumbnail((thumb_w, thumb_h))
        thumb_x = x0 + (card_w - thumb.width) // 2
        thumb_y = y0 + 10 + (thumb_h - thumb.height) // 2
        image.paste(thumb, (thumb_x, thumb_y))

        text_x = x0 + 10
        base_y = y0 + thumb_h + 16
        draw.text((text_x, base_y), f"cid={candidate_id}  idx={sample['lmdb_index']}", fill=(15, 23, 42), font=small_font)
        draw.text((text_x, base_y + 20), f"{sample['quality']} | {sample['structure']} | ocr={sample['ocr_correct']}", fill=(51, 65, 85), font=tiny_font)
        draw.text((text_x, base_y + 40), f"visual={sample['visual_quality_score']:.4f}", fill=(51, 65, 85), font=tiny_font)
        draw.text((text_x, base_y + 58), f"sharp={sample['sharpness_norm']:.4f}", fill=(51, 65, 85), font=tiny_font)
        tags = []
        if manual_hq_index is not None and sample["lmdb_index"] == manual_hq_index:
            tags.append("manual")
        if sample["is_v1_hq"]:
            tags.append("v1")
        if sample["is_v2_hq"]:
            tags.append("v2")
        if tags:
            draw.text((text_x, base_y + 76), " / ".join(tags), fill=(127, 29, 29), font=tiny_font)

        mappings.append(
            {
                "candidate_id": candidate_id,
                "lmdb_index": sample["lmdb_index"],
                "quality": sample["quality"],
                "structure": sample["structure"],
                "visual_quality_score": sample["visual_quality_score"],
                "sharpness_norm": sample["sharpness_norm"],
                "ocr_correct": sample["ocr_correct"],
                "is_v1_hq": sample["is_v1_hq"],
                "is_v2_hq": sample["is_v2_hq"],
                "is_manual_hq": manual_hq_index is not None and sample["lmdb_index"] == manual_hq_index,
                "source_path": sample["source_path"],
            }
        )
    image.save(output_path, quality=92)
    return mappings


def build_prompt(group: Dict[str, Any], candidate_mapping: List[Dict[str, Any]]) -> str:
    lines = []
    for item in candidate_mapping:
        tags = []
        if item["is_v1_hq"]:
            tags.append("v1")
        if item["is_v2_hq"]:
            tags.append("v2")
        if item["is_manual_hq"]:
            tags.append("manual")
        suffix = f" [{'|'.join(tags)}]" if tags else ""
        lines.append(
            f"- candidate_id={item['candidate_id']}, lmdb_index={item['lmdb_index']}, "
            f"quality={item['quality']}, structure={item['structure']}, "
            f"visual_quality_score={item['visual_quality_score']:.4f}, "
            f"sharpness_norm={item['sharpness_norm']:.4f}, ocr_correct={item['ocr_correct']}{suffix}"
        )
    return f"""
你将看到一个同一艘船、相同 label 的候选船牌图像面板。每个候选图都有 candidate_id 和 lmdb_index。

请只根据视觉质量选择最适合作为 HQ reference 的一张图。HQ reference 的标准如下：
1. 船牌字符完整，无明显遮挡；
2. 字符边缘清楚，中文笔画和数字/英文尽量可辨；
3. 船牌区域裁剪完整，位置合理；
4. 光照正常，不过曝、不过暗；
5. 不要因为背景清晰就选择，重点看船牌文字区域；
6. 尽量选择结构稳定、适合作为同船低质量样本参考的图；
7. 如果多张都合适，选择字符最完整、最清楚的一张。

不要做 OCR，不需要识别完整船牌文本。
不要输出多余文字。请严格输出 JSON：
{{
  "selected_candidate_id": 整数,
  "selected_lmdb_index": 整数,
  "confidence": 0到1之间的小数,
  "need_human_review": true或false,
  "reason": "一句中文理由",
  "second_best_lmdb_index": 整数或null,
  "risk_flags": ["all_blurry","occlusion","over_exposure","under_exposure","layout_conflict","no_clear_hq","tiny_text","ambiguous"] 中的若干项
}}

当前 group 信息：
- label: {group['label']}
- group_rank: {group['group_rank']}
- group_size: {group['group_size']}

当前 panel 中的候选列表：
{chr(10).join(lines)}
""".strip()


def image_to_data_url(path: Path) -> str:
    raw = base64.b64encode(path.read_bytes()).decode("utf-8")
    return "data:image/jpeg;base64," + raw


def call_qwen_once(
    api_base: str,
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = api_base.rstrip("/") + "/chat/completions"
    if requests is not None:
        response = requests.post(url, json=payload, timeout=request_timeout)
        response.raise_for_status()
        return {"request": payload, "response": response.json(), "backend": "requests"}

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=request_timeout) as response:
        text = response.read().decode("utf-8")
    return {"request": payload, "response": json.loads(text), "backend": "urllib"}


def call_qwen_with_retries(
    api_base: str,
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = call_qwen_once(
                api_base=api_base,
                model=model,
                prompt=prompt,
                image_path=image_path,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            result["attempt"] = attempt
            return result
        except Exception as exc:  # pragma: no cover - runtime path
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep)
    raise last_error


def extract_assistant_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("response 缺少 choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        if parts:
            return "".join(parts)
    raise ValueError("无法从 response 提取 assistant content")


def parse_json_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("输出中未找到 JSON 对象")
    return json.loads(stripped[start : end + 1])


def parse_risk_flags(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def parse_qwen_output(
    parsed_json: Dict[str, Any],
    group: Dict[str, Any],
    candidate_mapping: List[Dict[str, Any]],
    panel_path: Path,
    response_path: Path,
    manual_hq_index: Optional[int],
) -> Dict[str, Any]:
    selected_candidate_id = parsed_json.get("selected_candidate_id")
    selected_lmdb_index = parsed_json.get("selected_lmdb_index")
    confidence = parsed_json.get("confidence")
    need_human_review = parsed_json.get("need_human_review")
    reason = parsed_json.get("reason")
    second_best = parsed_json.get("second_best_lmdb_index")
    risk_flags = parsed_json.get("risk_flags", [])

    candidate_by_id = {item["candidate_id"]: item for item in candidate_mapping}
    candidate_by_idx = {item["lmdb_index"]: item for item in candidate_mapping}
    json_parse_ok = True
    illegal_selection = False

    if not isinstance(selected_candidate_id, int):
        raise ValueError("selected_candidate_id 不是整数")
    if not isinstance(selected_lmdb_index, int):
        raise ValueError("selected_lmdb_index 不是整数")
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence 不是数字")
    if not isinstance(need_human_review, bool):
        raise ValueError("need_human_review 不是 bool")
    if second_best is not None and not isinstance(second_best, int):
        raise ValueError("second_best_lmdb_index 非 int/null")
    if not isinstance(risk_flags, list):
        raise ValueError("risk_flags 不是 list")

    if selected_candidate_id not in candidate_by_id:
        illegal_selection = True
    if selected_lmdb_index not in candidate_by_idx:
        illegal_selection = True

    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        illegal_selection = True

    selected_candidate = candidate_by_id.get(selected_candidate_id)
    if selected_candidate and selected_candidate["lmdb_index"] != selected_lmdb_index:
        illegal_selection = True
    selected_meta = candidate_by_idx.get(selected_lmdb_index, {})

    return {
        "label": group["label"],
        "label_hash": group["label_hash"],
        "group_rank": group["group_rank"],
        "group_size": group["group_size"],
        "selected_candidate_id": selected_candidate_id,
        "selected_lmdb_index": selected_lmdb_index,
        "confidence": confidence,
        "need_human_review": need_human_review,
        "reason": str(reason or ""),
        "second_best_lmdb_index": second_best,
        "risk_flags": [str(item) for item in risk_flags],
        "json_parse_ok": json_parse_ok,
        "illegal_selection": illegal_selection,
        "panel_path": str(panel_path),
        "response_path": str(response_path),
        "selected_quality": selected_meta.get("quality", ""),
        "selected_structure": selected_meta.get("structure", ""),
        "selected_visual_quality_score": selected_meta.get("visual_quality_score", ""),
        "manual_hq_index": manual_hq_index if manual_hq_index is not None else "",
        "manual_exact_match_if_available": (
            selected_lmdb_index == manual_hq_index if manual_hq_index is not None else ""
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, list):
                    formatted[key] = json.dumps(value, ensure_ascii=False)
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_parsed(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows, _ = load_csv_rows(path)
    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                "label": row["label"],
                "label_hash": row["label_hash"],
                "group_rank": as_int(row["group_rank"]),
                "group_size": as_int(row["group_size"]),
                "selected_candidate_id": as_int(row["selected_candidate_id"]),
                "selected_lmdb_index": as_int(row["selected_lmdb_index"]),
                "confidence": as_float(row["confidence"]),
                "need_human_review": parse_bool(row["need_human_review"]),
                "reason": row["reason"],
                "second_best_lmdb_index": as_int(row["second_best_lmdb_index"]) if row["second_best_lmdb_index"] not in {"", "None"} else "",
                "risk_flags": parse_risk_flags(row["risk_flags"]),
                "json_parse_ok": parse_bool(row["json_parse_ok"]),
                "illegal_selection": parse_bool(row["illegal_selection"]),
                "panel_path": row["panel_path"],
                "response_path": row["response_path"],
                "selected_quality": row["selected_quality"],
                "selected_structure": row["selected_structure"],
                "selected_visual_quality_score": as_float(row["selected_visual_quality_score"]) if row["selected_visual_quality_score"] not in {"", "None"} else "",
                "manual_hq_index": as_int(row["manual_hq_index"]) if row["manual_hq_index"] not in {"", "None"} else "",
                "manual_exact_match_if_available": (
                    parse_bool(row["manual_exact_match_if_available"])
                    if row["manual_exact_match_if_available"] not in {"", "None"}
                    else ""
                ),
            }
        )
    parsed_rows.sort(key=lambda item: item["group_rank"])
    return parsed_rows


def load_existing_failed(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows, _ = load_csv_rows(path)
    failed_rows = []
    for row in rows:
        failed_rows.append(
            {
                "label": row["label"],
                "label_hash": row["label_hash"],
                "group_rank": as_int(row["group_rank"]) if row["group_rank"] else "",
                "error_stage": row["error_stage"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
                "manual_hq_index": row["manual_hq_index"],
                "panel_path": row["panel_path"],
                "response_path": row["response_path"],
            }
        )
    return failed_rows


def load_progress_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def needs_human_review(row: Dict[str, Any]) -> bool:
    risk_flags = set(parse_risk_flags(row.get("risk_flags", [])))
    return (
        parse_bool(row["need_human_review"])
        or float(row["confidence"]) < 0.80
        or parse_bool(row["illegal_selection"])
        or not parse_bool(row["json_parse_ok"])
        or bool(risk_flags & HUMAN_REVIEW_RISK_FLAGS)
    )


def make_failure_row(
    group: Dict[str, Any],
    error_stage: str,
    error_type: str,
    error_message: str,
    manual_hq_index: Optional[int],
    panel_path: Path,
    response_path: Path,
) -> Dict[str, Any]:
    return {
        "label": group["label"],
        "label_hash": group["label_hash"],
        "group_rank": group["group_rank"],
        "error_stage": error_stage,
        "error_type": error_type,
        "error_message": error_message,
        "manual_hq_index": manual_hq_index if manual_hq_index is not None else "",
        "panel_path": str(panel_path),
        "response_path": str(response_path),
    }


def parse_existing_response_file(
    response_path: Path,
    group: Dict[str, Any],
    panel_path: Path,
    manual_hq_index: Optional[int],
) -> Dict[str, Any]:
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    candidate_mapping = payload.get("candidate_mapping")
    if not isinstance(candidate_mapping, list) or not candidate_mapping:
        raise ValueError("response 文件缺少 candidate_mapping")
    response_json = payload.get("response")
    if not isinstance(response_json, dict):
        raise ValueError("response 文件缺少 response")
    assistant_text = extract_assistant_text(response_json)
    parsed_json = parse_json_from_text(assistant_text)
    return parse_qwen_output(
        parsed_json=parsed_json,
        group=group,
        candidate_mapping=candidate_mapping,
        panel_path=panel_path,
        response_path=response_path,
        manual_hq_index=manual_hq_index,
    )


def bucket_confidence(value: float) -> str:
    if value >= 0.95:
        return "0.95-1.00"
    if value >= 0.90:
        return "0.90-0.94"
    if value >= 0.80:
        return "0.80-0.89"
    return "<0.80"


def format_counter(counter: Counter) -> str:
    if not counter:
        return "- 无"
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))


def build_runtime_stats(
    args: argparse.Namespace,
    all_groups: List[Dict[str, Any]],
    target_groups: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    manual_source_note: Dict[str, Any],
    prepared_groups: int,
    skipped_completed: int,
    skipped_existing: int,
    interrupted: bool,
) -> Dict[str, Any]:
    api_success_count = len(
        [row for row in failed_rows if row["error_stage"] == "parse_or_validation"]
    ) + len(parsed_rows)
    api_failed_count = len([row for row in failed_rows if row["error_stage"] == "api_call"])
    json_parse_success_count = len(parsed_rows)
    json_parse_failed_count = len([row for row in failed_rows if row["error_stage"] == "parse_or_validation"])
    illegal_selection_count = sum(1 for row in parsed_rows if row["illegal_selection"])
    need_human_rows = [row for row in parsed_rows if needs_human_review(row)]
    selected_quality = Counter(row["selected_quality"] for row in parsed_rows if row["selected_quality"])
    selected_structure = Counter(row["selected_structure"] for row in parsed_rows if row["selected_structure"])
    risk_flags = Counter(flag for row in parsed_rows for flag in parse_risk_flags(row["risk_flags"]))
    confidence_buckets = Counter(bucket_confidence(float(row["confidence"])) for row in parsed_rows)
    manual_available_rows = [row for row in parsed_rows if row["manual_hq_index"] not in {"", None}]
    manual_exact_match_count = sum(1 for row in manual_available_rows if row["manual_exact_match_if_available"] is True)

    return {
        "mode": {
            "no_api": args.no_api,
            "dry_run": args.dry_run,
            "resume": args.resume,
            "skip_existing": args.skip_existing,
        },
        "num_total_groups": len(all_groups),
        "num_target_groups": len(target_groups),
        "num_processed": len(parsed_rows) + len(failed_rows),
        "prepared_groups": prepared_groups,
        "api_success_count": api_success_count,
        "api_failed_count": api_failed_count,
        "json_parse_success_count": json_parse_success_count,
        "json_parse_failed_count": json_parse_failed_count,
        "illegal_selection_count": illegal_selection_count,
        "need_human_review_count": len(need_human_rows),
        "need_human_review_rate": len(need_human_rows) / len(parsed_rows) if parsed_rows else 0.0,
        "selected_quality_distribution": selected_quality,
        "selected_structure_distribution": selected_structure,
        "risk_flags_distribution": risk_flags,
        "confidence_bucket_distribution": confidence_buckets,
        "manual_available_count": len(manual_available_rows),
        "manual_exact_match_count": manual_exact_match_count,
        "manual_exact_match_rate": manual_exact_match_count / len(manual_available_rows) if manual_available_rows else 0.0,
        "manual_source_note": manual_source_note,
        "skipped_completed": skipped_completed,
        "skipped_existing": skipped_existing,
        "interrupted": interrupted,
    }


def write_progress_state(
    path: Path,
    stats: Dict[str, Any],
    last_group_rank: Optional[int],
    success_count: int,
    failed_count: int,
) -> None:
    payload = {
        "total_groups": stats["num_target_groups"],
        "processed_groups": stats["num_processed"],
        "prepared_groups": stats["prepared_groups"],
        "success_count": success_count,
        "failed_count": failed_count,
        "last_group_rank": last_group_rank,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": stats["mode"],
        "skipped_completed": stats["skipped_completed"],
        "skipped_existing": stats["skipped_existing"],
        "interrupted": stats["interrupted"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_runtime_outputs(
    out_dir: Path,
    parsed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    args: argparse.Namespace,
    all_groups: List[Dict[str, Any]],
    target_groups: List[Dict[str, Any]],
) -> None:
    parsed_path = out_dir / "qwen_full_hq_selection_parsed.csv"
    failed_path = out_dir / "qwen_full_failed_cases.csv"
    need_human_path = out_dir / "qwen_full_need_human_review.csv"
    report_path = out_dir / "reports" / RUNTIME_REPORT_NAME

    parsed_rows_sorted = sorted(parsed_rows, key=lambda item: item["group_rank"])
    failed_rows_sorted = sorted(failed_rows, key=lambda item: (item["group_rank"] if item["group_rank"] != "" else 10**9))
    need_human_rows = [row for row in parsed_rows_sorted if needs_human_review(row)]

    write_csv(parsed_path, parsed_rows_sorted, PARSED_FIELDNAMES)
    write_csv(failed_path, failed_rows_sorted, FAILED_FIELDNAMES)
    write_csv(need_human_path, need_human_rows, PARSED_FIELDNAMES)
    write_runtime_report(report_path, args, all_groups, target_groups, parsed_rows_sorted, failed_rows_sorted, stats)


def write_runtime_report(
    report_path: Path,
    args: argparse.Namespace,
    all_groups: List[Dict[str, Any]],
    target_groups: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected_quality = stats["selected_quality_distribution"]
    selected_structure = stats["selected_structure_distribution"]
    risk_flags = stats["risk_flags_distribution"]
    confidence_buckets = stats["confidence_bucket_distribution"]
    manual_note = stats["manual_source_note"]

    report = [
        "# Stage00_step11 Full Qwen3-VL HQ Review Report",
        "",
        "## 1. 范围与隔离声明",
        "",
        "本轮只在 `experiments/dit_lq_hq_v1/qwen_vl_hq_review_full/` 下写入 panel、response、CSV、progress_state 和运行报告。",
        "未修改 manual_hq_review_site 的 SQLite/CSV，未修改 qwen_vl_hq_review_pilot、qwen_mismatch_review、v1/v2 manifest、原始数据、OCR LMDB、configs、checkpoints、outputs 或源码。",
        "",
        "## 2. 输入与输出",
        "",
        f"- candidate_groups.csv: `{args.candidate_groups}`",
        f"- candidate_samples.csv: `{args.candidate_samples}`",
        f"- review images root: `{args.review_images_root}`",
        f"- manual selection primary: `{args.manual_selection}`",
        f"- manual selection fallback: `{args.manual_selection_fallback}`",
        f"- output dir: `{args.out_dir}`",
        "",
        "## 3. 运行参数",
        "",
        f"- api_base: `{args.api_base}`",
        f"- model: `{args.model}`",
        f"- max_candidates_per_panel: `{args.max_candidates_per_panel}`",
        f"- temperature: `{args.temperature}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- flush_every: `{args.flush_every}`",
        f"- max_retries: `{args.max_retries}`",
        f"- retry_sleep: `{args.retry_sleep}`",
        f"- request_timeout: `{args.request_timeout}`",
        f"- start_rank: `{args.start_rank}`",
        f"- end_rank: `{args.end_rank}`",
        f"- limit_groups: `{args.limit_groups}`",
        f"- no_api: `{args.no_api}`",
        f"- dry_run: `{args.dry_run}`",
        f"- resume: `{args.resume}`",
        f"- skip_existing: `{args.skip_existing}`",
        "",
        "## 4. Group 处理范围",
        "",
        f"- num_total_groups: `{len(all_groups)}`",
        f"- num_target_groups: `{len(target_groups)}`",
        f"- num_processed: `{stats['num_processed']}`",
        f"- prepared_groups: `{stats['prepared_groups']}`",
        f"- skipped_completed: `{stats['skipped_completed']}`",
        f"- skipped_existing: `{stats['skipped_existing']}`",
        "",
        "## 5. Panel 生成策略",
        "",
        "- 默认处理 `group_size >= 2` 的全部 group，按 `group_rank` 升序。",
        "- 若 group 大于 `max_candidates_per_panel`，优先保留 manual_hq（若有）、v1_hq、v2_hq、easy 高 visual 候选、majority structure 高 visual 候选，再按 visual_quality_score 补足。",
        "- panel 显示 `candidate_id`、`lmdb_index`、quality、structure、visual_quality_score、sharpness_norm、ocr_correct 以及 manual/v1/v2 标记。",
        "",
        "## 6. API 调用统计",
        "",
        f"- api_success_count: `{stats['api_success_count']}`",
        f"- api_failed_count: `{stats['api_failed_count']}`",
        f"- interrupted: `{stats['interrupted']}`",
        "",
        "## 7. JSON 解析与合法性统计",
        "",
        f"- json_parse_success_count: `{stats['json_parse_success_count']}`",
        f"- json_parse_failed_count: `{stats['json_parse_failed_count']}`",
        f"- illegal_selection_count: `{stats['illegal_selection_count']}`",
        "",
        "## 8. Qwen 选择分布",
        "",
        "selected quality 分布：",
        "",
        format_counter(selected_quality),
        "",
        "selected structure 分布：",
        "",
        format_counter(selected_structure),
        "",
        "confidence bucket 分布：",
        "",
        format_counter(confidence_buckets),
        "",
        "risk_flags 分布：",
        "",
        format_counter(risk_flags),
        "",
        "## 9. Need-human-review 分析",
        "",
        f"- need_human_review_count: `{stats['need_human_review_count']}`",
        f"- need_human_review_rate: `{stats['need_human_review_rate']:.4f}`",
        "- 规则：`need_human_review=true` 或 `confidence<0.80` 或 `illegal_selection=true` 或 `json_parse_ok=false` 或包含 `no_clear_hq/ambiguous/layout_conflict`。",
        "",
        "## 10. 与人工 130 组对照统计",
        "",
        f"- manual_available_count: `{stats['manual_available_count']}`",
        f"- manual_exact_match_count: `{stats['manual_exact_match_count']}`",
        f"- manual_exact_match_rate: `{stats['manual_exact_match_rate']:.4f}`",
        f"- manual primary reviewed count: `{manual_note['primary_reviewed_count']}`",
        f"- manual fallback reviewed count: `{manual_note['fallback_reviewed_count']}`",
        f"- manual primary used count: `{manual_note['primary_used_count']}`",
        f"- manual fallback used count: `{manual_note['fallback_used_count']}`",
        "",
        "## 11. 失败样本分析",
        "",
    ]
    if failed_rows:
        report.extend(
            [
                "| label_hash | group_rank | error_stage | error_type | error_message |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for row in failed_rows[:30]:
            report.append(
                f"| {row['label_hash']} | {row['group_rank']} | {row['error_stage']} | {row['error_type']} | {row['error_message']} |"
            )
    else:
        report.append("- 无失败样本")

    report.extend(
        [
            "",
            "## 12. Resume / 中断状态",
            "",
            f"- resume: `{args.resume}`",
            f"- progress_state.json: `{args.out_dir / 'progress_state.json'}`",
            f"- last processed report time: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## 13. 警告与限制",
            "",
            "- 当前脚本默认使用 pilot prompt，以保持 pilot 与 full 结果可比。",
            "- panel 不是完整 group，只是按策略保留的候选子集。",
            "- no-api / dry-run 不会得到 Qwen 结果，只用于验证 panel、路径和清单。",
            "- confidence 仅作辅助，不代表视觉质量的充分条件。",
            "",
            "## 14. 下一步建议",
            "",
            "- 先用 `--no-api --limit-groups 10` 做面板检查。",
            "- 再用真实 API 对 10 个 group 做 smoke test。",
            "- 确认输出稳定后，再由用户自行执行全量审核或分段审核。",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def persist_all(
    out_dir: Path,
    args: argparse.Namespace,
    all_groups: List[Dict[str, Any]],
    target_groups: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    manual_source_note: Dict[str, Any],
    prepared_groups: int,
    skipped_completed: int,
    skipped_existing: int,
    last_group_rank: Optional[int],
    interrupted: bool,
) -> Dict[str, Any]:
    stats = build_runtime_stats(
        args=args,
        all_groups=all_groups,
        target_groups=target_groups,
        parsed_rows=parsed_rows,
        failed_rows=failed_rows,
        manual_source_note=manual_source_note,
        prepared_groups=prepared_groups,
        skipped_completed=skipped_completed,
        skipped_existing=skipped_existing,
        interrupted=interrupted,
    )
    write_runtime_outputs(out_dir, parsed_rows, failed_rows, stats, args, all_groups, target_groups)
    write_progress_state(
        out_dir / "progress_state.json",
        stats=stats,
        last_group_rank=last_group_rank,
        success_count=len(parsed_rows),
        failed_count=len(failed_rows),
    )
    return stats


def print_success(script_path: Path, out_dir: Path) -> None:
    print("Full Qwen3-VL HQ review code generation completed.")
    print(f"Script: {script_path.resolve()}")
    print(f"Code dir: {out_dir.resolve()}")
    print(f"README: {(out_dir / 'README.md').resolve()}")
    print(f"Examples: {(out_dir / 'run_examples.sh').resolve()}")
    print(f"Code report: {(out_dir / 'reports' / CODE_REPORT_NAME).resolve()}")
    print("No full Qwen review was executed.")
    print("No manual review site state, OCR source/data/config/checkpoint/output, or original dataset files were modified.")


def main() -> None:
    args = validate_args(parse_args())
    if args.overwrite:
        clear_runtime_artifacts(args.out_dir)

    panels_dir = args.out_dir / "panels"
    responses_dir = args.out_dir / "responses"
    reports_dir = args.out_dir / "reports"
    panels_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_groups = load_candidate_groups(args.candidate_groups)
    samples_by_hash = load_candidate_samples(args.candidate_samples, args.review_images_root)
    manual_map, manual_source_note = load_manual_selection_map(args.manual_selection, args.manual_selection_fallback)

    target_groups = []
    for group in all_groups:
        if group["group_size"] < 2:
            continue
        rank = group["group_rank"]
        if args.start_rank is not None and rank < args.start_rank:
            continue
        if args.end_rank is not None and rank > args.end_rank:
            continue
        target_groups.append(group)
    if args.limit_groups is not None:
        target_groups = target_groups[: args.limit_groups]

    parsed_rows = load_existing_parsed(args.out_dir / "qwen_full_hq_selection_parsed.csv") if args.resume else []
    failed_rows = load_existing_failed(args.out_dir / "qwen_full_failed_cases.csv") if args.resume else []
    progress_state = load_progress_state(args.out_dir / "progress_state.json") if args.resume else {}
    parsed_by_hash = {row["label_hash"]: row for row in parsed_rows}
    failed_by_hash = {row["label_hash"]: row for row in failed_rows}
    skipped_completed = 0
    skipped_existing = 0
    prepared_groups = progress_state.get("prepared_groups", 0) if isinstance(progress_state, dict) else 0
    last_group_rank = progress_state.get("last_group_rank") if isinstance(progress_state, dict) else None
    interrupted = False
    raw_buffer: List[Dict[str, Any]] = []
    touched_since_flush = 0

    try:
        for group in target_groups:
            label_hash = group["label_hash"]
            if args.resume and label_hash in parsed_by_hash:
                skipped_completed += 1
                continue

            samples = samples_by_hash.get(label_hash, [])
            manual_hq_index = manual_map.get(label_hash)
            if manual_hq_index is not None and manual_hq_index not in {sample["lmdb_index"] for sample in samples}:
                manual_hq_index = None

            panel_path = panels_dir / f"group_{group['group_rank']:04d}_{label_hash}_panel.jpg"
            response_path = responses_dir / f"group_{group['group_rank']:04d}_{label_hash}_response.json"
            last_group_rank = group["group_rank"]

            try:
                if not samples:
                    raise ValueError("group 在 candidate_samples.csv 中没有候选样本")
                panel_samples = choose_panel_candidates(group, samples, manual_hq_index, args.max_candidates_per_panel)
                candidate_mapping = make_panel(group, panel_samples, manual_hq_index, panel_path)
                prepared_groups += 1
            except Exception as exc:
                failed_row = make_failure_row(
                    group=group,
                    error_stage="panel_generation",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    manual_hq_index=manual_hq_index,
                    panel_path=panel_path,
                    response_path=response_path,
                )
                failed_by_hash[label_hash] = failed_row
                touched_since_flush += 1
                continue

            if args.skip_existing and response_path.exists():
                try:
                    parsed = parse_existing_response_file(
                        response_path=response_path,
                        group=group,
                        panel_path=panel_path,
                        manual_hq_index=manual_hq_index,
                    )
                    parsed_by_hash[label_hash] = parsed
                    failed_by_hash.pop(label_hash, None)
                    skipped_existing += 1
                except Exception as exc:
                    failed_by_hash[label_hash] = make_failure_row(
                        group=group,
                        error_stage="existing_response_parse",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        manual_hq_index=manual_hq_index,
                        panel_path=panel_path,
                        response_path=response_path,
                    )
                touched_since_flush += 1
                if touched_since_flush >= args.flush_every:
                    parsed_rows = sorted(parsed_by_hash.values(), key=lambda item: item["group_rank"])
                    failed_rows = list(failed_by_hash.values())
                    append_jsonl(args.out_dir / "qwen_full_hq_selection_raw.jsonl", raw_buffer)
                    raw_buffer = []
                    persist_all(
                        out_dir=args.out_dir,
                        args=args,
                        all_groups=all_groups,
                        target_groups=target_groups,
                        parsed_rows=parsed_rows,
                        failed_rows=failed_rows,
                        manual_source_note=manual_source_note,
                        prepared_groups=prepared_groups,
                        skipped_completed=skipped_completed,
                        skipped_existing=skipped_existing,
                        last_group_rank=last_group_rank,
                        interrupted=interrupted,
                    )
                    touched_since_flush = 0
                continue

            if args.no_api:
                touched_since_flush += 1
                if touched_since_flush >= args.flush_every:
                    parsed_rows = sorted(parsed_by_hash.values(), key=lambda item: item["group_rank"])
                    failed_rows = list(failed_by_hash.values())
                    if raw_buffer:
                        append_jsonl(args.out_dir / "qwen_full_hq_selection_raw.jsonl", raw_buffer)
                        raw_buffer = []
                    persist_all(
                        out_dir=args.out_dir,
                        args=args,
                        all_groups=all_groups,
                        target_groups=target_groups,
                        parsed_rows=parsed_rows,
                        failed_rows=failed_rows,
                        manual_source_note=manual_source_note,
                        prepared_groups=prepared_groups,
                        skipped_completed=skipped_completed,
                        skipped_existing=skipped_existing,
                        last_group_rank=last_group_rank,
                        interrupted=interrupted,
                    )
                    touched_since_flush = 0
                continue

            prompt = build_prompt(group, candidate_mapping)
            raw_record = {
                "label": group["label"],
                "label_hash": group["label_hash"],
                "group_rank": group["group_rank"],
                "group_size": group["group_size"],
                "manual_hq_index": manual_hq_index,
                "panel_path": str(panel_path),
                "candidate_mapping": candidate_mapping,
            }
            try:
                api_result = call_qwen_with_retries(
                    api_base=args.api_base,
                    model=args.model,
                    prompt=prompt,
                    image_path=panel_path,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
                response_payload = {
                    **raw_record,
                    "prompt": prompt,
                    "request": api_result["request"],
                    "response": api_result["response"],
                    "transport": api_result["backend"],
                    "attempt": api_result["attempt"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                response_path.write_text(json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                raw_buffer.append(response_payload)

                assistant_text = extract_assistant_text(api_result["response"])
                parsed_json = parse_json_from_text(assistant_text)
                parsed = parse_qwen_output(
                    parsed_json=parsed_json,
                    group=group,
                    candidate_mapping=candidate_mapping,
                    panel_path=panel_path,
                    response_path=response_path,
                    manual_hq_index=manual_hq_index,
                )
                parsed_by_hash[label_hash] = parsed
                failed_by_hash.pop(label_hash, None)
            except (HTTPError, URLError) as exc:
                failed_by_hash[label_hash] = make_failure_row(
                    group=group,
                    error_stage="api_call",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    manual_hq_index=manual_hq_index,
                    panel_path=panel_path,
                    response_path=response_path,
                )
            except Exception as exc:
                failed_by_hash[label_hash] = make_failure_row(
                    group=group,
                    error_stage="parse_or_validation",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    manual_hq_index=manual_hq_index,
                    panel_path=panel_path,
                    response_path=response_path,
                )

            touched_since_flush += 1
            if touched_since_flush >= args.flush_every:
                parsed_rows = sorted(parsed_by_hash.values(), key=lambda item: item["group_rank"])
                failed_rows = list(failed_by_hash.values())
                append_jsonl(args.out_dir / "qwen_full_hq_selection_raw.jsonl", raw_buffer)
                raw_buffer = []
                persist_all(
                    out_dir=args.out_dir,
                    args=args,
                    all_groups=all_groups,
                    target_groups=target_groups,
                    parsed_rows=parsed_rows,
                    failed_rows=failed_rows,
                    manual_source_note=manual_source_note,
                    prepared_groups=prepared_groups,
                    skipped_completed=skipped_completed,
                    skipped_existing=skipped_existing,
                    last_group_rank=last_group_rank,
                    interrupted=interrupted,
                )
                touched_since_flush = 0
    except KeyboardInterrupt:
        interrupted = True
    finally:
        parsed_rows = sorted(parsed_by_hash.values(), key=lambda item: item["group_rank"])
        failed_rows = list(failed_by_hash.values())
        if raw_buffer:
            append_jsonl(args.out_dir / "qwen_full_hq_selection_raw.jsonl", raw_buffer)
        persist_all(
            out_dir=args.out_dir,
            args=args,
            all_groups=all_groups,
            target_groups=target_groups,
            parsed_rows=parsed_rows,
            failed_rows=failed_rows,
            manual_source_note=manual_source_note,
            prepared_groups=prepared_groups,
            skipped_completed=skipped_completed,
            skipped_existing=skipped_existing,
            last_group_rank=last_group_rank,
            interrupted=interrupted,
        )
        if interrupted:
            raise KeyboardInterrupt("运行被中断，已写出当前 progress_state 和报告")


if __name__ == "__main__":
    main()
