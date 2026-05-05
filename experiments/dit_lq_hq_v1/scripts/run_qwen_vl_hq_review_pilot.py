#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import json
import math
import shutil
import statistics
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/mnt/data/zyx/SLP34K_v2").resolve()
EXPERIMENT_ROOT = (PROJECT_ROOT / "experiments" / "dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL HQ 审核 pilot")
    parser.add_argument("--candidate-groups", type=Path, required=True)
    parser.add_argument("--candidate-samples", type=Path, required=True)
    parser.add_argument("--manual-selection", type=Path, required=True)
    parser.add_argument("--manual-selection-fallback", type=Path, required=True)
    parser.add_argument("--review-images-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:22002/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit-groups", type=int, default=130)
    parser.add_argument("--max-candidates-per-panel", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def ensure_safe_output(out_dir: Path, overwrite: bool) -> Path:
    out_dir = out_dir.resolve()
    try:
        out_dir.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(f"out-dir 必须位于 {SAFE_OUTPUT_ROOT} 下: {out_dir}") from exc
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，未提供 --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头: {path}")
        rows = list(reader)
    return rows, reader.fieldnames


def load_candidate_groups(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
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
        "review_page",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"candidate_groups.csv 缺少字段: {missing}")
    groups = []
    groups_by_hash: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = {
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
            "review_page": row["review_page"],
        }
        groups.append(item)
        groups_by_hash[item["label_hash"]] = item
    groups.sort(key=lambda item: item["group_rank"])
    return groups, groups_by_hash


def resolve_sample_image_path(review_images_root: Path, local_image_path: str) -> Path:
    rel = local_image_path.strip().replace("\\", "/")
    prefix = "manual_hq_review/images/"
    if rel.startswith(prefix):
        rel = rel[len(prefix) :]
    path = (review_images_root / rel).resolve()
    if not path.exists():
        alt = (PROJECT_ROOT / local_image_path).resolve()
        if alt.exists():
            return alt
        raise FileNotFoundError(f"候选缩略图不存在: {local_image_path}")
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
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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
        by_hash[sample["label_hash"]].append(sample)
    for label_hash, samples in by_hash.items():
        samples.sort(
            key=lambda item: (
                -item["quality_priority"],
                -item["visual_quality_score"],
                -item["sharpness_norm"],
                item["lmdb_index"],
            )
        )
    return by_hash


def load_manual_selection(primary: Path, fallback: Path, required_count: int) -> Tuple[Path, List[Dict[str, Any]], str]:
    selected_path = None
    rows: List[Dict[str, Any]] = []
    note = ""
    for path in [primary, fallback]:
        if not path.exists():
            continue
        raw_rows, fields = load_csv_rows(path)
        if "review_status" not in fields or "manual_hq_index" not in fields or "label_hash" not in fields:
            raise ValueError(f"人工选择 CSV 缺少必要字段: {path}")
        reviewed = []
        for row in raw_rows:
            status = (row.get("review_status") or "").strip()
            manual = (row.get("manual_hq_index") or "").strip()
            if status == "reviewed" and manual:
                reviewed.append(row)
        if path == primary and len(reviewed) >= required_count:
            selected_path = path
            rows = reviewed
            note = "used_primary"
            break
        if path == fallback:
            selected_path = path
            rows = reviewed
            note = "used_fallback" if primary.exists() else "primary_missing_used_fallback"
            break
    if selected_path is None:
        raise FileNotFoundError("primary/fallback 人工选择文件都不存在")
    return selected_path, rows, note


def select_pilot_groups(
    reviewed_rows: List[Dict[str, Any]],
    groups_by_hash: Dict[str, Dict[str, Any]],
    samples_by_hash: Dict[str, List[Dict[str, Any]]],
    limit_groups: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    failed: List[Dict[str, Any]] = []
    validated: List[Dict[str, Any]] = []
    rows_by_hash = {row["label_hash"]: row for row in reviewed_rows}
    candidate_hashes = [group["label_hash"] for group in sorted(groups_by_hash.values(), key=lambda item: item["group_rank"]) if group["label_hash"] in rows_by_hash]
    for label_hash in candidate_hashes[:limit_groups]:
        row = rows_by_hash[label_hash]
        group = groups_by_hash.get(label_hash)
        if group is None:
            failed.append(
                {
                    "label": row.get("label", ""),
                    "label_hash": label_hash,
                    "group_rank": "",
                    "error_stage": "manual_selection_validation",
                    "error_type": "missing_group",
                    "error_message": "label_hash 不存在于 candidate_groups.csv",
                    "manual_hq_index": row.get("manual_hq_index", ""),
                    "panel_path": "",
                    "response_path": "",
                }
            )
            continue
        manual_raw = (row.get("manual_hq_index") or "").strip()
        try:
            manual_hq_index = as_int(manual_raw)
        except Exception:
            failed.append(
                {
                    "label": group["label"],
                    "label_hash": label_hash,
                    "group_rank": group["group_rank"],
                    "error_stage": "manual_selection_validation",
                    "error_type": "invalid_manual_hq_index",
                    "error_message": f"manual_hq_index 无法转为 int: {manual_raw}",
                    "manual_hq_index": manual_raw,
                    "panel_path": "",
                    "response_path": "",
                }
            )
            continue
        group_samples = samples_by_hash.get(label_hash, [])
        lmdb_indices = {sample["lmdb_index"] for sample in group_samples}
        if manual_hq_index not in lmdb_indices:
            failed.append(
                {
                    "label": group["label"],
                    "label_hash": label_hash,
                    "group_rank": group["group_rank"],
                    "error_stage": "manual_selection_validation",
                    "error_type": "manual_hq_not_in_group",
                    "error_message": "manual_hq_index 不属于该 label group",
                    "manual_hq_index": manual_hq_index,
                    "panel_path": "",
                    "response_path": "",
                }
            )
            continue
        validated.append(
            {
                "group": group,
                "manual_row": row,
                "manual_hq_index": manual_hq_index,
                "samples": group_samples,
            }
        )
    return validated, failed


def load_font(size: int):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: Tuple[int, int], font, fill, max_width: int, line_spacing: int = 2):
    x, y = xy
    current = ""
    for char in text:
        trial = current + char
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = trial
            continue
        draw.text((x, y), current, font=font, fill=fill)
        line_h = draw.textbbox((0, 0), current, font=font)[3]
        y += line_h + line_spacing
        current = char
    if current:
        draw.text((x, y), current, font=font, fill=fill)


def choose_panel_candidates(group: Dict[str, Any], samples: List[Dict[str, Any]], manual_hq_index: int, max_candidates: int) -> List[Dict[str, Any]]:
    if len(samples) <= max_candidates:
        return list(samples)
    majority_structure = Counter(sample["structure"] for sample in samples).most_common(1)[0][0]
    selected: List[Dict[str, Any]] = []
    seen = set()

    def add_sample(sample: Dict[str, Any]) -> None:
        idx = sample["lmdb_index"]
        if idx not in seen and len(selected) < max_candidates:
            selected.append(sample)
            seen.add(idx)

    by_idx = {sample["lmdb_index"]: sample for sample in samples}
    for idx in [manual_hq_index, group["v1_hq_index"], group["v2_hq_index"]]:
        sample = by_idx.get(idx)
        if sample:
            add_sample(sample)

    easy_samples = sorted(
        [sample for sample in samples if sample["quality"] == "easy"],
        key=lambda item: (-item["visual_quality_score"], -item["sharpness_norm"], item["lmdb_index"]),
    )
    for sample in easy_samples:
        add_sample(sample)

    majority_samples = sorted(
        [sample for sample in samples if sample["structure"] == majority_structure],
        key=lambda item: (-item["visual_quality_score"], -item["sharpness_norm"], item["lmdb_index"]),
    )
    for sample in majority_samples:
        add_sample(sample)

    remainder = sorted(
        samples,
        key=lambda item: (
            -item["visual_quality_score"],
            -item["sharpness_norm"],
            -item["contrast_norm"],
            item["lmdb_index"],
        ),
    )
    for sample in remainder:
        add_sample(sample)

    if manual_hq_index not in {sample["lmdb_index"] for sample in selected}:
        raise ValueError("manual_hq_index 未被保留进 panel")
    return selected


def make_panel(
    group: Dict[str, Any],
    panel_samples: List[Dict[str, Any]],
    manual_hq_index: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    thumb_w = 240
    thumb_h = 180
    text_h = 96
    card_w = thumb_w + 20
    card_h = thumb_h + text_h + 20
    margin = 20
    header_h = 120
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
    draw.text((margin, 16), title, fill=(15, 23, 42), font=title_font)
    draw.text((margin, 50), f"Label: {group['label']}", fill=(30, 41, 59), font=meta_font)
    draw.text(
        (margin, 76),
        f"manual={manual_hq_index}  v1={group['v1_hq_index']}  v2={group['v2_hq_index']}  quality(e/m/h)={group['num_easy']}/{group['num_middle']}/{group['num_hard']}",
        fill=(51, 65, 85),
        font=small_font,
    )

    mappings: List[Dict[str, Any]] = []
    for idx, sample in enumerate(panel_samples, start=1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        x0 = margin + col * (card_w + margin)
        y0 = header_h + margin + row * (card_h + margin)
        x1 = x0 + card_w
        y1 = y0 + card_h
        border_color = (209, 213, 219)
        if sample["lmdb_index"] == manual_hq_index:
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
        draw.text((text_x, base_y), f"cid={idx}  idx={sample['lmdb_index']}", fill=(15, 23, 42), font=small_font)
        draw.text((text_x, base_y + 20), f"{sample['quality']} | {sample['structure']} | ocr={sample['ocr_correct']}", fill=(51, 65, 85), font=tiny_font)
        draw.text((text_x, base_y + 40), f"visual={sample['visual_quality_score']:.4f}", fill=(51, 65, 85), font=tiny_font)
        draw.text((text_x, base_y + 58), f"sharp={sample['sharpness_norm']:.4f}", fill=(51, 65, 85), font=tiny_font)
        badges = []
        if sample["lmdb_index"] == manual_hq_index:
            badges.append("manual")
        if sample["is_v1_hq"]:
            badges.append("v1")
        if sample["is_v2_hq"]:
            badges.append("v2")
        if badges:
            draw.text((text_x, base_y + 76), " / ".join(badges), fill=(127, 29, 29), font=tiny_font)

        mappings.append(
            {
                "candidate_id": idx,
                "lmdb_index": sample["lmdb_index"],
                "quality": sample["quality"],
                "structure": sample["structure"],
                "visual_quality_score": sample["visual_quality_score"],
                "sharpness_norm": sample["sharpness_norm"],
                "ocr_correct": sample["ocr_correct"],
                "is_v1_hq": sample["is_v1_hq"],
                "is_v2_hq": sample["is_v2_hq"],
                "is_manual_hq": sample["lmdb_index"] == manual_hq_index,
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
        tag_text = f" [{'|'.join(tags)}]" if tags else ""
        lines.append(
            f"- candidate_id={item['candidate_id']}, lmdb_index={item['lmdb_index']}, "
            f"quality={item['quality']}, structure={item['structure']}, "
            f"visual_quality_score={item['visual_quality_score']:.4f}, "
            f"sharpness_norm={item['sharpness_norm']:.4f}, ocr_correct={item['ocr_correct']}{tag_text}"
        )
    candidates_text = "\n".join(lines)
    prompt = f"""
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
{candidates_text}
""".strip()
    return prompt


def image_to_data_url(path: Path) -> str:
    raw = base64.b64encode(path.read_bytes()).decode("utf-8")
    return "data:image/jpeg;base64," + raw


def call_qwen(
    api_base: str,
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
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
    request = Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        text = response.read().decode("utf-8")
    return {"request": payload, "response": json.loads(text)}


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
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("输出中未找到 JSON 对象")
    json_text = stripped[start : end + 1]
    return json.loads(json_text)


def parse_qwen_output(
    parsed_json: Dict[str, Any],
    group: Dict[str, Any],
    candidate_mapping: List[Dict[str, Any]],
    manual_hq_index: int,
    panel_path: Path,
    response_path: Path,
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
    if selected_candidate_id not in candidate_by_id:
        illegal_selection = True
    if selected_lmdb_index not in candidate_by_idx:
        illegal_selection = True
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence 不是数字")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        illegal_selection = True
    if not isinstance(need_human_review, bool):
        raise ValueError("need_human_review 不是 bool")
    if not (isinstance(second_best, int) or second_best is None):
        raise ValueError("second_best_lmdb_index 非 int/null")
    if not isinstance(risk_flags, list):
        raise ValueError("risk_flags 不是 list")

    candidate = candidate_by_id.get(selected_candidate_id)
    if candidate and candidate["lmdb_index"] != selected_lmdb_index:
        illegal_selection = True

    return {
        "label": group["label"],
        "label_hash": group["label_hash"],
        "group_rank": group["group_rank"],
        "group_size": group["group_size"],
        "manual_hq_index": manual_hq_index,
        "selected_candidate_id": selected_candidate_id,
        "selected_lmdb_index": selected_lmdb_index,
        "exact_match": selected_lmdb_index == manual_hq_index,
        "confidence": confidence,
        "need_human_review": need_human_review,
        "reason": str(reason or ""),
        "second_best_lmdb_index": second_best,
        "risk_flags": risk_flags,
        "json_parse_ok": json_parse_ok,
        "illegal_selection": illegal_selection,
        "manual_candidate_in_panel": manual_hq_index in candidate_by_idx,
        "panel_path": str(panel_path),
        "response_path": str(response_path),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in list(formatted.items()):
                if isinstance(value, list):
                    formatted[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(formatted)


def bucket_group_size(value: int) -> str:
    if value == 2:
        return "2"
    if 3 <= value <= 5:
        return "3-5"
    if 6 <= value <= 10:
        return "6-10"
    return ">10"


def bucket_confidence(value: float) -> str:
    if value >= 0.9:
        return ">=0.9"
    if value >= 0.8:
        return "0.8-0.9"
    return "<0.8"


def summarize_bucket(rows: List[Dict[str, Any]], key_name: str) -> List[Tuple[str, int, int, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key_name]].append(row)
    summary = []
    for key, items in groups.items():
        count = len(items)
        exact = sum(1 for item in items if item["exact_match"])
        summary.append((str(key), count, exact, exact / count if count else 0.0))
    summary.sort(key=lambda item: item[0])
    return summary


def make_report(
    out_path: Path,
    api_base: str,
    model: str,
    selection_source: Path,
    selection_note: str,
    limit_groups: int,
    panel_strategy_note: str,
    stats: Dict[str, Any],
    parsed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
) -> None:
    bucket_rows = []
    enriched = []
    for row in parsed_rows:
        enriched.append(
            {
                **row,
                "group_size_bucket": bucket_group_size(row["group_size"]),
                "confidence_bucket": bucket_confidence(row["confidence"]) if isinstance(row["confidence"], float) else "unknown",
            }
        )
    bucket_sections = [
        ("group_size_bucket", "group_size bucket"),
        ("manual_hq_quality", "manual_hq quality"),
        ("manual_hq_structure", "manual_hq structure"),
        ("v1_v2_same", "v1_v2_same"),
        ("confidence_bucket", "qwen confidence bucket"),
    ]
    tables = []
    for key, title in bucket_sections:
        table = "| bucket | count | exact_match | exact_match_rate |\n| --- | ---: | ---: | ---: |\n"
        for bucket, count, exact, rate in summarize_bucket(enriched, key):
            table += f"| {bucket} | {count} | {exact} | {rate:.4f} |\n"
        tables.append((title, table))

    mismatches = [row for row in parsed_rows if not row["exact_match"]]
    need_review = [row for row in parsed_rows if row["need_human_review"] or row["confidence"] < 0.80 or not row["exact_match"] or row["illegal_selection"] or not row["json_parse_ok"]]
    recommendation = []
    if (
        stats["json_parse_success_rate"] >= 0.98
        and stats["illegal_selection_rate"] <= 0.01
        and stats["exact_match_rate"] >= 0.60
        and stats["need_human_review_rate"] <= 0.50
    ):
        recommendation.append("自动指标达到扩大试验阈值，建议扩大到 500 group。")
    else:
        recommendation.append("自动指标未完全达到扩大试验阈值，建议先人工抽查 mismatch 与 need-human-review 样本后再决定是否扩大。")
    if stats["exact_match_rate"] >= 0.70:
        recommendation.append("如果抽查结果也稳定，可进一步扩大到 1000 group。")
    else:
        recommendation.append("在 exact match 仍偏低时，不建议直接扩大到全量。")

    report = [
        "# Stage00_step09 Qwen3-VL HQ Review Pilot Report",
        "",
        "## 1. 范围与隔离声明",
        "",
        "本轮只在 `experiments/dit_lq_hq_v1/qwen_vl_hq_review_pilot/` 下创建 panel、response、CSV 和报告。",
        "未修改 manual_hq_review_site 的 SQLite/CSV，未修改 v1/v2 manifest、原始数据、OCR LMDB、configs、checkpoints、outputs 或源码。",
        "",
        "## 2. 输入与输出",
        "",
        f"- candidate_groups.csv: `{EXPERIMENT_ROOT / 'manual_hq_review/candidate_groups.csv'}`",
        f"- candidate_samples.csv: `{EXPERIMENT_ROOT / 'manual_hq_review/candidate_samples.csv'}`",
        f"- review images root: `{EXPERIMENT_ROOT / 'manual_hq_review/images'}`",
        f"- manual selection source: `{selection_source}`",
        f"- selection source note: `{selection_note}`",
        f"- pilot limit_groups: `{limit_groups}`",
        f"- output dir: `{out_path.parent.parent}`",
        "",
        "## 3. Qwen3-VL 服务配置",
        "",
        f"- api_base: `{api_base}`",
        f"- model: `{model}`",
        "- request format: OpenAI-compatible `/v1/chat/completions` + `image_url` data URL",
        "- temperature: `0`",
        "- max_tokens: `512`",
        "",
        "## 4. Pilot Group 选择",
        "",
        f"- num_manual_reviewed_available: `{stats['num_manual_reviewed_available']}`",
        f"- num_pilot_groups: `{stats['num_pilot_groups']}`",
        f"- num_qwen_processed: `{stats['num_qwen_processed']}`",
        f"- skipped_invalid_groups: `{stats['skipped_invalid_groups']}`",
        "",
        "## 5. Panel 生成策略",
        "",
        panel_strategy_note,
        "",
        "## 6. Prompt 与输出 JSON Schema",
        "",
        "- Prompt 明确要求：不要做 OCR，只根据视觉质量选最适合作为 HQ reference 的候选图。",
        "- 输出 JSON 字段：`selected_candidate_id`, `selected_lmdb_index`, `confidence`, `need_human_review`, `reason`, `second_best_lmdb_index`, `risk_flags`。",
        "",
        "## 7. API 调用与解析结果",
        "",
        f"- api_success_count: `{stats['api_success_count']}`",
        f"- api_failed_count: `{stats['api_failed_count']}`",
        f"- json_parse_success_count: `{stats['json_parse_success_count']}`",
        f"- json_parse_failed_count: `{stats['json_parse_failed_count']}`",
        f"- illegal_selection_count: `{stats['illegal_selection_count']}`",
        f"- json_parse_success_rate: `{stats['json_parse_success_rate']:.4f}`",
        f"- illegal_selection_rate: `{stats['illegal_selection_rate']:.4f}`",
        "",
        "## 8. Qwen vs Manual 对比指标",
        "",
        f"- exact_match_count: `{stats['exact_match_count']}`",
        f"- exact_match_rate: `{stats['exact_match_rate']:.4f}`",
        f"- mismatch_count: `{stats['mismatch_count']}`",
        f"- need_human_review_count: `{stats['need_human_review_count']}`",
        f"- need_human_review_rate: `{stats['need_human_review_rate']:.4f}`",
        f"- mean_qwen_confidence: `{stats['mean_qwen_confidence']:.4f}`",
        f"- median_qwen_confidence: `{stats['median_qwen_confidence']:.4f}`",
        "",
    ]
    for title, table in tables:
        report.extend([f"### {title}", "", table])
    report.extend(
        [
            "",
            "## 9. Need-human-review 样本分析",
            "",
            f"- need-human-review 总数: `{len(need_review)}`",
            f"- 低置信度(<0.80)或 mismatch/illegal/json_parse_failed 都会进入该列表。",
        ]
    )
    if need_review:
        report.extend(["", "| group_rank | label_hash | manual_hq_index | qwen_hq_index | confidence | need_human_review | illegal_selection | reason |", "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"])
        for row in need_review[:20]:
            report.append(
                f"| {row['group_rank']} | {row['label_hash']} | {row['manual_hq_index']} | {row['selected_lmdb_index']} | {row['confidence']:.4f} | {row['need_human_review']} | {row['illegal_selection']} | {row['reason']} |"
            )

    report.extend(["", "## 10. 失败样本分析", ""])
    if failed_rows:
        report.append("| label_hash | group_rank | error_stage | error_type | error_message |")
        report.append("| --- | ---: | --- | --- | --- |")
        for row in failed_rows[:20]:
            report.append(f"| {row['label_hash']} | {row['group_rank']} | {row['error_stage']} | {row['error_type']} | {row['error_message']} |")
    else:
        report.append("- 无失败样本")

    report.extend(
        [
            "",
            "## 11. 是否建议扩大到 500/1000/全量",
            "",
            *[f"- {item}" for item in recommendation],
            "- 注意：exact match 不是唯一标准，因为同一 group 可能有多个可接受 HQ；最终是否全量跑仍需要人工抽查 mismatch 样本。",
            "",
            "## 12. 警告与限制",
            "",
            "- panel 最多只给 Qwen 看 `max_candidates_per_panel` 张候选图，不是完整 group。",
            "- exact match 只是与人工单一选择对齐，不等价于视觉上唯一正确答案。",
            "- Qwen 输出依赖当前 prompt 与 panel 排版，后续如果要扩大全量，应保持同一模板一致性。",
            "- 本轮没有把 Qwen 结果写回人工审核网站，也没有生成 manual pair manifest。",
            "",
            "## 13. 下一步建议",
            "",
            "- 先人工抽查 `qwen_need_human_review.csv` 与 mismatch 样本，确认 Qwen 选择是否至少“可接受”。",
            "- 如果抽查结果稳定，再扩大到 500 group 或 1000 group 做第二轮 pilot。",
            "- 只有在更大规模 pilot 仍稳定时，才考虑用 Qwen 作为全量预审助手。",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_safe_output(args.out_dir, args.overwrite)
    panels_dir = out_dir / "panels"
    responses_dir = out_dir / "responses"
    reports_dir = out_dir / "reports"
    panels_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    groups, groups_by_hash = load_candidate_groups(args.candidate_groups.resolve())
    samples_by_hash = load_candidate_samples(args.candidate_samples.resolve(), args.review_images_root.resolve())
    selection_source, reviewed_rows, selection_note = load_manual_selection(
        args.manual_selection.resolve(),
        args.manual_selection_fallback.resolve(),
        args.limit_groups,
    )
    validated_groups, failed_rows = select_pilot_groups(
        reviewed_rows=reviewed_rows,
        groups_by_hash=groups_by_hash,
        samples_by_hash=samples_by_hash,
        limit_groups=args.limit_groups,
    )

    raw_jsonl_path = out_dir / "qwen_hq_selection_raw.jsonl"
    parsed_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    need_review_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(validated_groups, start=1):
        group = item["group"]
        samples = item["samples"]
        manual_hq_index = item["manual_hq_index"]
        panel_samples = choose_panel_candidates(group, samples, manual_hq_index, args.max_candidates_per_panel)
        panel_path = panels_dir / f"group_{group['group_rank']:04d}_{group['label_hash']}_panel.jpg"
        response_path = responses_dir / f"group_{group['group_rank']:04d}_{group['label_hash']}_response.json"

        try:
            candidate_mapping = make_panel(group, panel_samples, manual_hq_index, panel_path)
        except Exception as exc:
            failed_rows.append(
                {
                    "label": group["label"],
                    "label_hash": group["label_hash"],
                    "group_rank": group["group_rank"],
                    "error_stage": "panel_generation",
                    "error_type": "panel_generation_failed",
                    "error_message": str(exc),
                    "manual_hq_index": manual_hq_index,
                    "panel_path": str(panel_path),
                    "response_path": str(response_path),
                }
            )
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
            api_result = call_qwen(
                api_base=args.api_base,
                model=args.model,
                prompt=prompt,
                image_path=panel_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            response_payload = {
                **raw_record,
                "prompt": prompt,
                "request": api_result["request"],
                "response": api_result["response"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            response_path.write_text(json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with raw_jsonl_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(response_payload, ensure_ascii=False) + "\n")

            assistant_text = extract_assistant_text(api_result["response"])
            parsed_json = parse_json_from_text(assistant_text)
            parsed = parse_qwen_output(parsed_json, group, candidate_mapping, manual_hq_index, panel_path, response_path)
            manual_sample = next(sample for sample in samples if sample["lmdb_index"] == manual_hq_index)
            parsed["manual_hq_quality"] = manual_sample["quality"]
            parsed["manual_hq_structure"] = manual_sample["structure"]
            parsed["v1_v2_same"] = group["v1_v2_same"]
            parsed_rows.append(parsed)
            comparison_rows.append(
                {
                    "label": parsed["label"],
                    "label_hash": parsed["label_hash"],
                    "group_rank": parsed["group_rank"],
                    "group_size": parsed["group_size"],
                    "manual_hq_index": parsed["manual_hq_index"],
                    "qwen_hq_index": parsed["selected_lmdb_index"],
                    "exact_match": parsed["exact_match"],
                    "qwen_confidence": parsed["confidence"],
                    "qwen_need_human_review": parsed["need_human_review"],
                    "qwen_reason": parsed["reason"],
                    "qwen_risk_flags": parsed["risk_flags"],
                    "manual_candidate_in_panel": parsed["manual_candidate_in_panel"],
                    "panel_path": parsed["panel_path"],
                }
            )
            if parsed["need_human_review"] or parsed["confidence"] < 0.80 or not parsed["exact_match"] or parsed["illegal_selection"] or not parsed["json_parse_ok"]:
                need_review_rows.append(parsed)
        except (HTTPError, URLError) as exc:
            failed_rows.append(
                {
                    "label": group["label"],
                    "label_hash": group["label_hash"],
                    "group_rank": group["group_rank"],
                    "error_stage": "api_call",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "manual_hq_index": manual_hq_index,
                    "panel_path": str(panel_path),
                    "response_path": str(response_path),
                }
            )
            continue
        except Exception as exc:
            failed_rows.append(
                {
                    "label": group["label"],
                    "label_hash": group["label_hash"],
                    "group_rank": group["group_rank"],
                    "error_stage": "parse_or_validation",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "manual_hq_index": manual_hq_index,
                    "panel_path": str(panel_path),
                    "response_path": str(response_path),
                }
            )
            continue

    parsed_csv_path = out_dir / "qwen_hq_selection_parsed.csv"
    comparison_csv_path = out_dir / "qwen_vs_manual_comparison.csv"
    need_review_csv_path = out_dir / "qwen_need_human_review.csv"
    failed_csv_path = out_dir / "qwen_failed_cases.csv"

    write_csv(
        parsed_csv_path,
        parsed_rows,
        [
            "label",
            "label_hash",
            "group_rank",
            "group_size",
            "manual_hq_index",
            "selected_candidate_id",
            "selected_lmdb_index",
            "exact_match",
            "confidence",
            "need_human_review",
            "reason",
            "second_best_lmdb_index",
            "risk_flags",
            "json_parse_ok",
            "illegal_selection",
            "panel_path",
            "response_path",
            "manual_hq_quality",
            "manual_hq_structure",
            "v1_v2_same",
            "manual_candidate_in_panel",
        ],
    )
    write_csv(
        comparison_csv_path,
        comparison_rows,
        [
            "label",
            "label_hash",
            "group_rank",
            "group_size",
            "manual_hq_index",
            "qwen_hq_index",
            "exact_match",
            "qwen_confidence",
            "qwen_need_human_review",
            "qwen_reason",
            "qwen_risk_flags",
            "manual_candidate_in_panel",
            "panel_path",
        ],
    )
    write_csv(
        need_review_csv_path,
        need_review_rows,
        [
            "label",
            "label_hash",
            "group_rank",
            "group_size",
            "manual_hq_index",
            "selected_candidate_id",
            "selected_lmdb_index",
            "exact_match",
            "confidence",
            "need_human_review",
            "reason",
            "second_best_lmdb_index",
            "risk_flags",
            "json_parse_ok",
            "illegal_selection",
            "panel_path",
            "response_path",
            "manual_hq_quality",
            "manual_hq_structure",
            "v1_v2_same",
            "manual_candidate_in_panel",
        ],
    )
    write_csv(
        failed_csv_path,
        failed_rows,
        [
            "label",
            "label_hash",
            "group_rank",
            "error_stage",
            "error_type",
            "error_message",
            "manual_hq_index",
            "panel_path",
            "response_path",
        ],
    )

    api_success_count = len(parsed_rows) + sum(1 for row in failed_rows if row["error_stage"] == "parse_or_validation")
    api_failed_count = sum(1 for row in failed_rows if row["error_stage"] == "api_call")
    json_parse_success_count = len(parsed_rows)
    json_parse_failed_count = sum(1 for row in failed_rows if row["error_stage"] == "parse_or_validation")
    illegal_selection_count = sum(1 for row in parsed_rows if row["illegal_selection"])
    exact_match_count = sum(1 for row in parsed_rows if row["exact_match"])
    need_human_review_count = len(need_review_rows)
    mean_conf = statistics.mean([row["confidence"] for row in parsed_rows]) if parsed_rows else 0.0
    median_conf = statistics.median([row["confidence"] for row in parsed_rows]) if parsed_rows else 0.0
    stats = {
        "num_manual_reviewed_available": len(reviewed_rows),
        "num_pilot_groups": min(args.limit_groups, len(reviewed_rows)),
        "num_qwen_processed": len(parsed_rows),
        "skipped_invalid_groups": len([row for row in failed_rows if row["error_stage"] == "manual_selection_validation"]),
        "api_success_count": api_success_count,
        "api_failed_count": api_failed_count,
        "json_parse_success_count": json_parse_success_count,
        "json_parse_failed_count": json_parse_failed_count,
        "illegal_selection_count": illegal_selection_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / len(parsed_rows) if parsed_rows else 0.0,
        "mismatch_count": len(parsed_rows) - exact_match_count,
        "need_human_review_count": need_human_review_count,
        "need_human_review_rate": need_human_review_count / len(parsed_rows) if parsed_rows else 0.0,
        "mean_qwen_confidence": mean_conf,
        "median_qwen_confidence": median_conf,
        "json_parse_success_rate": json_parse_success_count / api_success_count if api_success_count else 0.0,
        "illegal_selection_rate": illegal_selection_count / len(parsed_rows) if parsed_rows else 0.0,
    }

    panel_strategy_note = (
        f"每个 group 最多保留 `{args.max_candidates_per_panel}` 张候选。若 group_size 超限，优先保留 manual/v1/v2 HQ、"
        "easy 高 visual 样本、majority structure 高 visual 样本，再用其余高 visual 样本补足；manual_hq_index 必须在 panel 中。"
    )
    report_path = reports_dir / "Stage00_step09_qwen_vl_hq_review_pilot_report.md"
    make_report(
        out_path=report_path,
        api_base=args.api_base,
        model=args.model,
        selection_source=selection_source,
        selection_note=selection_note,
        limit_groups=args.limit_groups,
        panel_strategy_note=panel_strategy_note,
        stats=stats,
        parsed_rows=parsed_rows,
        failed_rows=failed_rows,
    )


if __name__ == "__main__":
    main()
