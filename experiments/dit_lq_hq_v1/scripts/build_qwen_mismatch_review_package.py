#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import os
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path("/mnt/data/zyx/SLP34K_v2").resolve()
EXPERIMENT_ROOT = (PROJECT_ROOT / "experiments" / "dit_lq_hq_v1").resolve()
ALLOWED_OUT_DIRS = {
    (EXPERIMENT_ROOT / "qwen_mismatch_review").resolve(),
    (EXPERIMENT_ROOT / "qwen_mismatch_review_dryrun").resolve(),
}
ALLOWED_REVIEW_DECISIONS = [
    "acceptable_different",
    "worse_but_usable",
    "wrong",
    "uncertain",
    "prefer_manual",
    "prefer_qwen",
    "skip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Qwen mismatch 人工复核包")
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--need-human-csv", type=Path, required=True)
    parser.add_argument("--candidate-samples", type=Path, required=True)
    parser.add_argument("--review-images-root", type=Path, required=True)
    parser.add_argument("--pilot-panels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--cases-per-page", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
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


def load_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头: {path}")
        rows = list(reader)
    return rows, reader.fieldnames


def ensure_output_dir(out_dir: Path, overwrite: bool) -> Path:
    out_dir = out_dir.resolve()
    if out_dir not in ALLOWED_OUT_DIRS:
        raise ValueError(
            "out-dir 仅允许为以下目录之一: "
            + ", ".join(str(path) for path in sorted(ALLOWED_OUT_DIRS))
        )
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，未提供 --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def load_candidate_samples(path: Path, review_images_root: Path) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], List[str]]:
    rows, fields = load_csv_rows(path)
    required = [
        "label",
        "label_hash",
        "lmdb_index",
        "quality",
        "structure",
        "ocr_correct",
        "ocr_pred",
        "confidence",
        "visual_quality_score",
        "sharpness_norm",
        "source_path",
        "local_image_path",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"candidate_samples.csv 缺少字段: {missing}")
    samples_by_hash: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        label_hash = row["label_hash"].strip()
        lmdb_index = as_int(row["lmdb_index"])
        sample = {
            "label": row["label"],
            "label_hash": label_hash,
            "lmdb_index": lmdb_index,
            "quality": row["quality"],
            "structure": row["structure"],
            "ocr_correct": parse_bool(row["ocr_correct"]),
            "ocr_pred": row["ocr_pred"],
            "confidence": as_float(row["confidence"]),
            "visual_quality_score": as_float(row["visual_quality_score"]),
            "sharpness_norm": as_float(row["sharpness_norm"]),
            "source_path": row["source_path"],
            "local_image_path": row["local_image_path"],
            "resolved_image_path": resolve_sample_image_path(review_images_root, row["local_image_path"]),
        }
        samples_by_hash.setdefault(label_hash, {})
        if lmdb_index in samples_by_hash[label_hash]:
            raise ValueError(f"candidate_samples.csv 存在重复样本: label_hash={label_hash}, lmdb_index={lmdb_index}")
        samples_by_hash[label_hash][lmdb_index] = sample
    return samples_by_hash, fields


def load_comparison_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, fields = load_csv_rows(path)
    required = [
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
        "panel_path",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"qwen_vs_manual_comparison.csv 缺少字段: {missing}")
    parsed = []
    for row in rows:
        parsed.append(
            {
                "label": row["label"],
                "label_hash": row["label_hash"],
                "group_rank": as_int(row["group_rank"]),
                "group_size": as_int(row["group_size"]),
                "manual_hq_index": row["manual_hq_index"].strip(),
                "qwen_hq_index": row["qwen_hq_index"].strip(),
                "exact_match": parse_bool(row["exact_match"]),
                "qwen_confidence": as_float(row["qwen_confidence"]),
                "qwen_need_human_review": parse_bool(row["qwen_need_human_review"]),
                "qwen_reason": row["qwen_reason"],
                "qwen_risk_flags": row["qwen_risk_flags"],
                "panel_path": Path(row["panel_path"]).resolve(),
            }
        )
    parsed.sort(key=lambda item: item["group_rank"])
    return parsed, fields


def load_need_human_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, fields = load_csv_rows(path)
    required = [
        "label_hash",
        "group_rank",
        "selected_lmdb_index",
        "exact_match",
        "need_human_review",
        "illegal_selection",
        "json_parse_ok",
    ]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"qwen_need_human_review.csv 缺少字段: {missing}")
    parsed = []
    for row in rows:
        parsed.append(
            {
                "label_hash": row["label_hash"],
                "group_rank": as_int(row["group_rank"]),
                "selected_lmdb_index": row["selected_lmdb_index"].strip(),
                "exact_match": parse_bool(row["exact_match"]),
                "need_human_review": parse_bool(row["need_human_review"]),
                "illegal_selection": parse_bool(row["illegal_selection"]),
                "json_parse_ok": parse_bool(row["json_parse_ok"]),
            }
        )
    parsed.sort(key=lambda item: item["group_rank"])
    return parsed, fields


def stringify_risk_flags(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed) if parsed else "[]"
    return str(parsed)


def make_relative_href(target: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(str(target), str(base_dir))).as_posix()


def build_review_cases(
    comparison_rows: List[Dict[str, Any]],
    need_human_rows: List[Dict[str, Any]],
    samples_by_hash: Dict[str, Dict[int, Dict[str, Any]]],
    pilot_panels_dir: Path,
    limit_cases: int = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_rows = [row for row in comparison_rows if (not row["exact_match"]) or row["qwen_need_human_review"]]
    need_human_hashes = {row["label_hash"] for row in need_human_rows}
    selected_hashes = {row["label_hash"] for row in selected_rows}
    invalid_cases: List[Dict[str, Any]] = []
    review_cases: List[Dict[str, Any]] = []
    for row in selected_rows:
        label_hash = row["label_hash"]
        manual_text = row["manual_hq_index"]
        qwen_text = row["qwen_hq_index"]
        issues: List[str] = []

        if not manual_text:
            issues.append("manual_hq_index 为空")
        if not qwen_text:
            issues.append("qwen_hq_index 为空")

        manual_idx = None
        qwen_idx = None
        if not issues:
            try:
                manual_idx = as_int(manual_text)
            except Exception:
                issues.append(f"manual_hq_index 非法: {manual_text}")
            try:
                qwen_idx = as_int(qwen_text)
            except Exception:
                issues.append(f"qwen_hq_index 非法: {qwen_text}")

        samples_for_hash = samples_by_hash.get(label_hash, {})
        if manual_idx is not None and manual_idx not in samples_for_hash:
            issues.append(f"manual_hq_index 不在 candidate_samples.csv: {manual_idx}")
        if qwen_idx is not None and qwen_idx not in samples_for_hash:
            issues.append(f"qwen_hq_index 不在 candidate_samples.csv: {qwen_idx}")
        if manual_idx is not None and qwen_idx is not None and manual_idx == qwen_idx:
            issues.append(f"manual_hq_index 与 qwen_hq_index 相同: {manual_idx}")

        if issues:
            invalid_cases.append(
                {
                    "label": row["label"],
                    "label_hash": label_hash,
                    "group_rank": row["group_rank"],
                    "group_size": row["group_size"],
                    "manual_hq_index": manual_text,
                    "qwen_hq_index": qwen_text,
                    "issues": "; ".join(issues),
                }
            )
            continue

        manual_sample = samples_for_hash[manual_idx]
        qwen_sample = samples_for_hash[qwen_idx]
        panel_path = row["panel_path"]
        panel_exists = panel_path.exists()
        if not panel_exists:
            candidate_panel = (pilot_panels_dir / panel_path.name).resolve()
            if candidate_panel.exists():
                panel_path = candidate_panel
                panel_exists = True
        manual_image_exists = manual_sample["resolved_image_path"].exists()
        qwen_image_exists = qwen_sample["resolved_image_path"].exists()

        if not manual_image_exists:
            num_missing_manual_images += 1
        if not qwen_image_exists:
            num_missing_qwen_images += 1
        if not panel_exists:
            num_missing_panels += 1

        review_cases.append(
            {
                "label": row["label"],
                "label_hash": label_hash,
                "group_rank": row["group_rank"],
                "group_size": row["group_size"],
                "manual_hq_index": manual_idx,
                "manual_quality": manual_sample["quality"],
                "manual_structure": manual_sample["structure"],
                "manual_visual_quality_score": manual_sample["visual_quality_score"],
                "manual_sharpness_norm": manual_sample["sharpness_norm"],
                "manual_ocr_correct": manual_sample["ocr_correct"],
                "manual_ocr_pred": manual_sample["ocr_pred"],
                "manual_source_path": manual_sample["source_path"],
                "qwen_hq_index": qwen_idx,
                "qwen_quality": qwen_sample["quality"],
                "qwen_structure": qwen_sample["structure"],
                "qwen_visual_quality_score": qwen_sample["visual_quality_score"],
                "qwen_sharpness_norm": qwen_sample["sharpness_norm"],
                "qwen_ocr_correct": qwen_sample["ocr_correct"],
                "qwen_ocr_pred": qwen_sample["ocr_pred"],
                "qwen_source_path": qwen_sample["source_path"],
                "qwen_confidence": row["qwen_confidence"],
                "qwen_need_human_review": row["qwen_need_human_review"],
                "qwen_reason": row["qwen_reason"],
                "qwen_risk_flags": row["qwen_risk_flags"],
                "panel_path": panel_path,
                "panel_exists": panel_exists,
                "manual_image_path": manual_sample["resolved_image_path"],
                "manual_image_exists": manual_image_exists,
                "qwen_image_path": qwen_sample["resolved_image_path"],
                "qwen_image_exists": qwen_image_exists,
            }
        )

    if limit_cases is not None:
        review_cases = review_cases[:limit_cases]

    num_missing_manual_images = sum(1 for case in review_cases if not case["manual_image_exists"])
    num_missing_qwen_images = sum(1 for case in review_cases if not case["qwen_image_exists"])
    num_missing_panels = sum(1 for case in review_cases if not case["panel_exists"])

    stats = {
        "num_comparison_rows": len(comparison_rows),
        "num_mismatch_cases": sum(1 for row in comparison_rows if not row["exact_match"]),
        "num_need_human_flag_true_in_comparison": sum(1 for row in comparison_rows if row["qwen_need_human_review"]),
        "num_need_human_cases": len(need_human_rows),
        "need_human_csv_flag_true_count": sum(1 for row in need_human_rows if row["need_human_review"]),
        "need_human_csv_exact_match_false_count": sum(1 for row in need_human_rows if not row["exact_match"]),
        "num_review_cases": len(review_cases),
        "num_invalid_cases": len(invalid_cases),
        "num_missing_manual_images": num_missing_manual_images,
        "num_missing_qwen_images": num_missing_qwen_images,
        "num_missing_panels": num_missing_panels,
        "comparison_only_hashes": sorted(selected_hashes - need_human_hashes),
        "need_human_only_hashes": sorted(need_human_hashes - selected_hashes),
    }
    return review_cases, {"stats": stats, "invalid_cases": invalid_cases}


def write_cases_csv(path: Path, review_cases: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "manual_hq_index",
        "manual_quality",
        "manual_structure",
        "manual_visual_quality_score",
        "manual_sharpness_norm",
        "manual_ocr_correct",
        "manual_ocr_pred",
        "manual_source_path",
        "qwen_hq_index",
        "qwen_quality",
        "qwen_structure",
        "qwen_visual_quality_score",
        "qwen_sharpness_norm",
        "qwen_ocr_correct",
        "qwen_ocr_pred",
        "qwen_source_path",
        "qwen_confidence",
        "qwen_need_human_review",
        "qwen_reason",
        "qwen_risk_flags",
        "panel_path",
        "manual_image_path",
        "qwen_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for case in review_cases:
            writer.writerow(
                {
                    "label": case["label"],
                    "label_hash": case["label_hash"],
                    "group_rank": case["group_rank"],
                    "group_size": case["group_size"],
                    "manual_hq_index": case["manual_hq_index"],
                    "manual_quality": case["manual_quality"],
                    "manual_structure": case["manual_structure"],
                    "manual_visual_quality_score": f"{case['manual_visual_quality_score']:.6f}",
                    "manual_sharpness_norm": f"{case['manual_sharpness_norm']:.6f}",
                    "manual_ocr_correct": case["manual_ocr_correct"],
                    "manual_ocr_pred": case["manual_ocr_pred"],
                    "manual_source_path": case["manual_source_path"],
                    "qwen_hq_index": case["qwen_hq_index"],
                    "qwen_quality": case["qwen_quality"],
                    "qwen_structure": case["qwen_structure"],
                    "qwen_visual_quality_score": f"{case['qwen_visual_quality_score']:.6f}",
                    "qwen_sharpness_norm": f"{case['qwen_sharpness_norm']:.6f}",
                    "qwen_ocr_correct": case["qwen_ocr_correct"],
                    "qwen_ocr_pred": case["qwen_ocr_pred"],
                    "qwen_source_path": case["qwen_source_path"],
                    "qwen_confidence": f"{case['qwen_confidence']:.4f}",
                    "qwen_need_human_review": case["qwen_need_human_review"],
                    "qwen_reason": case["qwen_reason"],
                    "qwen_risk_flags": case["qwen_risk_flags"],
                    "panel_path": str(case["panel_path"]),
                    "manual_image_path": str(case["manual_image_path"]),
                    "qwen_image_path": str(case["qwen_image_path"]),
                }
            )


def write_template_csv(path: Path, review_cases: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "manual_hq_index",
        "qwen_hq_index",
        "qwen_confidence",
        "qwen_reason",
        "qwen_risk_flags",
        "review_decision",
        "final_hq_index",
        "review_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for case in review_cases:
            writer.writerow(
                {
                    "label": case["label"],
                    "label_hash": case["label_hash"],
                    "group_rank": case["group_rank"],
                    "group_size": case["group_size"],
                    "manual_hq_index": case["manual_hq_index"],
                    "qwen_hq_index": case["qwen_hq_index"],
                    "qwen_confidence": f"{case['qwen_confidence']:.4f}",
                    "qwen_reason": case["qwen_reason"],
                    "qwen_risk_flags": case["qwen_risk_flags"],
                    "review_decision": "",
                    "final_hq_index": "",
                    "review_note": "",
                }
            )


def render_sample_card(title: str, border_class: str, sample: Dict[str, Any], image_href: str, image_exists: bool) -> str:
    ocr_badge = "<span class='badge badge-danger'>OCR wrong</span>" if not sample["ocr_correct"] else "<span class='badge badge-ok'>OCR correct</span>"
    missing_note = "<div class='missing'>Image missing</div>" if not image_exists else ""
    image_html = (
        f"<img src=\"{html.escape(image_href)}\" alt=\"{sample['lmdb_index']}\">"
        if image_exists
        else "<div class='image-placeholder'>image missing</div>"
    )
    return f"""
    <div class="sample-card {border_class}">
      <div class="sample-title">{html.escape(title)}</div>
      {missing_note}
      <div class="image-wrap">{image_html}</div>
      <div class="meta">
        <div><strong>lmdb_index:</strong> {sample['lmdb_index']}</div>
        <div><strong>quality:</strong> {html.escape(sample['quality'])}</div>
        <div><strong>structure:</strong> {html.escape(sample['structure'])}</div>
        <div><strong>visual_quality_score:</strong> {sample['visual_quality_score']:.6f}</div>
        <div><strong>sharpness_norm:</strong> {sample['sharpness_norm']:.6f}</div>
        <div><strong>ocr_correct:</strong> {sample['ocr_correct']}</div>
        <div><strong>ocr_pred:</strong> {html.escape(sample['ocr_pred'])}</div>
        <div><strong>source_path:</strong> {html.escape(sample['source_path'])}</div>
      </div>
      <div class="footer">{ocr_badge}</div>
    </div>
    """


def render_case_html(case: Dict[str, Any], review_pages_dir: Path) -> str:
    manual_sample = {
        "lmdb_index": case["manual_hq_index"],
        "quality": case["manual_quality"],
        "structure": case["manual_structure"],
        "visual_quality_score": case["manual_visual_quality_score"],
        "sharpness_norm": case["manual_sharpness_norm"],
        "ocr_correct": case["manual_ocr_correct"],
        "ocr_pred": case["manual_ocr_pred"],
        "source_path": case["manual_source_path"],
    }
    qwen_sample = {
        "lmdb_index": case["qwen_hq_index"],
        "quality": case["qwen_quality"],
        "structure": case["qwen_structure"],
        "visual_quality_score": case["qwen_visual_quality_score"],
        "sharpness_norm": case["qwen_sharpness_norm"],
        "ocr_correct": case["qwen_ocr_correct"],
        "ocr_pred": case["qwen_ocr_pred"],
        "source_path": case["qwen_source_path"],
    }
    manual_href = make_relative_href(case["manual_image_path"], review_pages_dir)
    qwen_href = make_relative_href(case["qwen_image_path"], review_pages_dir)
    panel_href = make_relative_href(case["panel_path"], review_pages_dir) if case["panel_exists"] else ""
    risk_flags = stringify_risk_flags(case["qwen_risk_flags"])
    panel_html = (
        f"<img src=\"{html.escape(panel_href)}\" alt=\"panel {case['group_rank']}\">"
        if case["panel_exists"]
        else "<div class='panel-placeholder'>panel missing</div>"
    )
    return f"""
    <section class="case-card" id="case-{case['group_rank']:04d}">
      <div class="case-head">
        <h2>#{case['group_rank']:04d} {html.escape(case['label'])}</h2>
        <div class="summary">
          <span><strong>label_hash:</strong> {html.escape(case['label_hash'])}</span>
          <span><strong>group_size:</strong> {case['group_size']}</span>
          <span><strong>manual_hq_index:</strong> {case['manual_hq_index']}</span>
          <span><strong>qwen_hq_index:</strong> {case['qwen_hq_index']}</span>
        </div>
        <div class="qwen-meta">
          <div><strong>Qwen confidence:</strong> {case['qwen_confidence']:.4f}</div>
          <div><strong>Qwen need_human_review flag:</strong> {case['qwen_need_human_review']}</div>
          <div><strong>Qwen reason:</strong> {html.escape(case['qwen_reason'])}</div>
          <div><strong>Qwen risk_flags:</strong> {html.escape(risk_flags)}</div>
        </div>
      </div>
      <div class="compare-grid">
        {render_sample_card("Manual HQ", "manual-card", manual_sample, manual_href, case['manual_image_exists'])}
        {render_sample_card("Qwen HQ", "qwen-card", qwen_sample, qwen_href, case['qwen_image_exists'])}
      </div>
      <div class="panel-card">
        <div class="panel-title">Pilot Panel</div>
        {panel_html}
      </div>
    </section>
    """


def write_review_pages(review_cases: List[Dict[str, Any]], review_pages_dir: Path, cases_per_page: int) -> List[Dict[str, Any]]:
    review_pages_dir.mkdir(parents=True, exist_ok=True)
    page_index_rows = []
    total_pages = math.ceil(len(review_cases) / cases_per_page) if review_cases else 0
    for page_idx in range(total_pages):
        start = page_idx * cases_per_page
        end = min(len(review_cases), (page_idx + 1) * cases_per_page)
        cases = review_cases[start:end]
        page_name = f"page_{page_idx + 1:04d}.html"
        page_path = review_pages_dir / page_name
        cases_html = "".join(render_case_html(case, review_pages_dir) for case in cases)
        page_path.write_text(
            f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Mismatch Review {page_name}</title>
  <style>
    :root {{
      --bg: #edf2f7;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #4b5563;
      --manual: #2563eb;
      --qwen: #16a34a;
      --danger: #dc2626;
      --line: #d1d5db;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .page-head {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }}
    .page-head p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .case-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 22px;
      box-shadow: 0 16px 32px rgba(15, 23, 42, 0.07);
    }}
    .case-head h2 {{
      margin: 0 0 10px;
      font-size: 24px;
      word-break: break-all;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-bottom: 12px;
      font-size: 14px;
      color: var(--muted);
    }}
    .qwen-meta {{
      padding: 14px 16px;
      border-radius: 12px;
      background: #f8fafc;
      border-left: 4px solid #0f766e;
      line-height: 1.55;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .sample-card {{
      border: 3px solid #cbd5e1;
      border-radius: 16px;
      overflow: hidden;
      background: #fff;
    }}
    .sample-card.manual-card {{ border-color: var(--manual); }}
    .sample-card.qwen-card {{ border-color: var(--qwen); }}
    .sample-title {{
      padding: 12px 14px;
      font-size: 18px;
      font-weight: 700;
      color: #111827;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
    }}
    .image-wrap {{
      background: #f8fafc;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 14px;
    }}
    .image-wrap img {{
      max-width: 100%;
      max-height: 420px;
      object-fit: contain;
      display: block;
      background: #fff;
    }}
    .image-placeholder, .panel-placeholder {{
      width: 100%;
      min-height: 220px;
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      background: #fff;
    }}
    .meta {{
      padding: 14px;
      font-size: 13px;
      line-height: 1.55;
      word-break: break-all;
    }}
    .footer {{
      padding: 0 14px 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge-ok {{
      background: #dcfce7;
      color: #166534;
    }}
    .badge-danger {{
      background: #fee2e2;
      color: var(--danger);
    }}
    .missing {{
      padding: 8px 14px;
      color: var(--danger);
      font-weight: 700;
    }}
    .panel-card {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #f8fafc;
    }}
    .panel-title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .panel-card img {{
      width: 100%;
      max-height: 720px;
      object-fit: contain;
      border-radius: 12px;
      background: #fff;
      display: block;
    }}
    @media (max-width: 960px) {{
      .compare-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 16px;
      }}
      .case-card {{
        padding: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="page-head">
      <h1>Qwen mismatch review - {page_name}</h1>
      <p>覆盖 case: #{cases[0]['group_rank']:04d} - #{cases[-1]['group_rank']:04d}，共 {len(cases)} 个。</p>
      <p>左侧为 Manual HQ，右侧为 Qwen HQ，下方为原始 pilot panel。</p>
    </section>
    {cases_html}
  </div>
</body>
</html>
""",
            encoding="utf-8",
        )
        page_index_rows.append(
            {
                "page_file": page_name,
                "group_rank_range": f"{cases[0]['group_rank']:04d}-{cases[-1]['group_rank']:04d}",
                "num_cases": len(cases),
            }
        )
    return page_index_rows


def write_instructions(path: Path) -> None:
    path.write_text(
        """# Qwen mismatch 人工复核说明

## 1. 复核目标

本包用于复核 Qwen 与人工 HQ 不一致的样本，判断 Qwen 选择是否仍然可接受。

## 2. 打开方式

请依次打开 `review_pages/page_0001.html`、`review_pages/page_0002.html` 等离线页面进行复核。

## 3. 每个 case 需要判断什么

对每个 case 判断 Qwen 选择是否可接受：

- `acceptable_different`: Qwen 与人工不同，但 Qwen 选择也可作为 HQ
- `worse_but_usable`: Qwen 略差但仍可用
- `wrong`: Qwen 明显不适合作为 HQ
- `uncertain`: 无法判断
- `prefer_manual`: 最终使用 `manual_hq_index`
- `prefer_qwen`: 最终使用 `qwen_hq_index`
- `skip`: 该 group 不用于训练

## 4. 填写位置

请在 `qwen_mismatch_review_template.csv` 中填写以下字段：

- `review_decision`
- `final_hq_index`
- `review_note`

## 5. final_hq_index 填写规则

- 如果最终保留人工 HQ，可填写 `manual_hq_index`
- 如果最终保留 Qwen HQ，可填写 `qwen_hq_index`
- 如果 Qwen 与人工都不合适，可填写同 group 内其他合法 `lmdb_index`
- 如果该 group 不用于训练，可填写 `skip` 并在 `review_note` 说明原因
""",
        encoding="utf-8",
    )


def counter_markdown(counter: Counter) -> str:
    if not counter:
        return "- 无"
    lines = []
    for key, value in sorted(counter.items(), key=lambda item: (str(item[0]))):
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def confidence_distribution(values: List[float]) -> List[Tuple[str, int]]:
    bins = [
        ("<0.80", lambda x: x < 0.80),
        ("0.80-0.89", lambda x: 0.80 <= x < 0.90),
        ("0.90-0.94", lambda x: 0.90 <= x < 0.95),
        ("0.95-0.99", lambda x: 0.95 <= x < 1.00),
        ("1.00", lambda x: x >= 1.00),
    ]
    counts = []
    for label, matcher in bins:
        counts.append((label, sum(1 for value in values if matcher(value))))
    return counts


def make_invalid_case_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "无"
    header = "| group_rank | label_hash | manual_hq_index | qwen_hq_index | issues |\n| --- | --- | ---: | ---: | --- |"
    body = "\n".join(
        f"| {row['group_rank']} | {row['label_hash']} | {row['manual_hq_index']} | {row['qwen_hq_index']} | {row['issues']} |"
        for row in rows
    )
    return header + "\n" + body


def write_report(
    report_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    review_cases: List[Dict[str, Any]],
    stats: Dict[str, Any],
    invalid_cases: List[Dict[str, Any]],
    page_index_rows: List[Dict[str, Any]],
) -> None:
    manual_quality = Counter(case["manual_quality"] for case in review_cases)
    qwen_quality = Counter(case["qwen_quality"] for case in review_cases)
    manual_structure = Counter(case["manual_structure"] for case in review_cases)
    qwen_structure = Counter(case["qwen_structure"] for case in review_cases)
    qwen_conf_values = [case["qwen_confidence"] for case in review_cases]
    conf_bins = confidence_distribution(qwen_conf_values)
    conf_lines = "\n".join(f"- `{label}`: `{count}`" for label, count in conf_bins) if conf_bins else "- 无"
    page_table = (
        "| page file | covered group rank range | num cases |\n| --- | --- | ---: |\n"
        + "\n".join(
            f"| {row['page_file']} | {row['group_rank_range']} | {row['num_cases']} |"
            for row in page_index_rows
        )
        if page_index_rows
        else "无"
    )
    comparison_only = ", ".join(stats["comparison_only_hashes"]) if stats["comparison_only_hashes"] else "无"
    need_only = ", ".join(stats["need_human_only_hashes"]) if stats["need_human_only_hashes"] else "无"
    conf_summary = (
        f"- `qwen_confidence_mean`: `{statistics.mean(qwen_conf_values):.4f}`\n"
        f"- `qwen_confidence_median`: `{statistics.median(qwen_conf_values):.4f}`\n"
        f"- `qwen_confidence_min`: `{min(qwen_conf_values):.4f}`\n"
        f"- `qwen_confidence_max`: `{max(qwen_conf_values):.4f}`\n"
        if qwen_conf_values
        else "- 无"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# Stage00_step10 Qwen mismatch Review Package Report

## 1. 范围与隔离声明

本轮只在 experiments/dit_lq_hq_v1/qwen_mismatch_review/ 下创建 mismatch 复核页面、CSV 模板、说明和报告。
未重新调用 Qwen，未修改 Qwen pilot 原始结果，未修改 manual_hq_review_site 状态，未修改 v1/v2 manifest、原始数据、OCR LMDB、configs、checkpoints、outputs 或源码。

补充说明：本次按用户要求额外执行了 dry-run 输出目录 `{EXPERIMENT_ROOT / 'qwen_mismatch_review_dryrun'}`，除此之外未写入其他路径。

## 2. 输入与输出

- comparison CSV: `{args.comparison_csv.resolve()}`
- need-human CSV: `{args.need_human_csv.resolve()}`
- candidate_samples.csv: `{args.candidate_samples.resolve()}`
- review images root: `{args.review_images_root.resolve()}`
- pilot panels dir: `{args.pilot_panels_dir.resolve()}`
- output dir: `{out_dir}`
- review_pages: `{out_dir / 'review_pages'}`
- qwen_mismatch_review_template.csv: `{out_dir / 'qwen_mismatch_review_template.csv'}`
- qwen_mismatch_cases.csv: `{out_dir / 'qwen_mismatch_cases.csv'}`
- qwen_mismatch_review_instructions.md: `{out_dir / 'qwen_mismatch_review_instructions.md'}`
- report: `{report_path}`

## 3. Mismatch Case 选择规则

选择规则：

- `exact_match == False`
- 或 `qwen_need_human_review == True`

每个 case 额外校验：

- `manual_hq_index` 非空
- `qwen_hq_index` 非空
- `manual_hq_index != qwen_hq_index`
- `manual_hq_index` 在 `candidate_samples.csv` 中存在
- `qwen_hq_index` 在 `candidate_samples.csv` 中存在

与 `qwen_need_human_review.csv` 的集合对照：

- `comparison_selected_but_not_in_need_human_csv`: {comparison_only}
- `need_human_csv_but_not_in_comparison_selected`: {need_only}

异常 case：

{make_invalid_case_table(invalid_cases)}

## 4. Case 统计

- `num_comparison_rows`: `{stats['num_comparison_rows']}`
- `num_mismatch_cases`: `{stats['num_mismatch_cases']}`
- `num_need_human_cases`: `{stats['num_need_human_cases']}`
- `num_review_cases`: `{stats['num_review_cases']}`
- `num_invalid_cases`: `{stats['num_invalid_cases']}`
- `num_missing_manual_images`: `{stats['num_missing_manual_images']}`
- `num_missing_qwen_images`: `{stats['num_missing_qwen_images']}`
- `num_missing_panels`: `{stats['num_missing_panels']}`
- `num_qwen_need_human_flag_true_in_comparison`: `{stats['num_need_human_flag_true_in_comparison']}`
- `need_human_csv_flag_true_count`: `{stats['need_human_csv_flag_true_count']}`
- `need_human_csv_exact_match_false_count`: `{stats['need_human_csv_exact_match_false_count']}`

manual_hq quality 分布：

{counter_markdown(manual_quality)}

qwen_hq quality 分布：

{counter_markdown(qwen_quality)}

manual_hq structure 分布：

{counter_markdown(manual_structure)}

qwen_hq structure 分布：

{counter_markdown(qwen_structure)}

qwen confidence 分布：

{conf_summary}
{conf_lines}

## 5. HTML Review Page 说明

- 每页展示 `--cases-per-page` 个 case
- 布局为：左侧 Manual HQ，右侧 Qwen HQ，下方 Pilot Panel
- Manual HQ 使用蓝色边框
- Qwen HQ 使用绿色边框
- `ocr_correct=False` 使用红色提示
- 页面不依赖 JS，离线打开即可浏览

## 6. Template CSV 说明

模板文件：`qwen_mismatch_review_template.csv`

字段：

- `label`
- `label_hash`
- `group_rank`
- `group_size`
- `manual_hq_index`
- `qwen_hq_index`
- `qwen_confidence`
- `qwen_reason`
- `qwen_risk_flags`
- `review_decision`
- `final_hq_index`
- `review_note`

允许的 `review_decision`：

- `acceptable_different`
- `worse_but_usable`
- `wrong`
- `uncertain`
- `prefer_manual`
- `prefer_qwen`
- `skip`

## 7. Manual vs Qwen 初步分布

quality 对照：

| type | easy | middle | hard |
| --- | ---: | ---: | ---: |
| manual_hq | {manual_quality.get('easy', 0)} | {manual_quality.get('middle', 0)} | {manual_quality.get('hard', 0)} |
| qwen_hq | {qwen_quality.get('easy', 0)} | {qwen_quality.get('middle', 0)} | {qwen_quality.get('hard', 0)} |

structure 对照：

| type | single | multi | vertical |
| --- | ---: | ---: | ---: |
| manual_hq | {manual_structure.get('single', 0)} | {manual_structure.get('multi', 0)} | {manual_structure.get('vertical', 0)} |
| qwen_hq | {qwen_structure.get('single', 0)} | {qwen_structure.get('multi', 0)} | {qwen_structure.get('vertical', 0)} |

补充观察：

- 当前 mismatch 主要来自 `exact_match=False`
- 当前 `qwen_need_human_review` 布尔字段在 comparison/need-human 文件中均未触发新增 case
- `qwen_need_human_review.csv` 当前更接近“复核集合”输出，而不是严格意义上的 `need_human_review=True` 子集

## 8. Review Pages 索引

{page_table}

## 9. 人工复核流程

1. 打开 `review_pages/page_0001.html` 等页面浏览 mismatch case。
2. 对每个 case 判断 Qwen 选择是否可接受。
3. 在 `qwen_mismatch_review_template.csv` 中填写 `review_decision`、`final_hq_index`、`review_note`。
4. 如果最终使用人工 HQ，可填写 `manual_hq_index`。
5. 如果最终使用 Qwen HQ，可填写 `qwen_hq_index`。
6. 如果 Qwen 与人工都不合适，可填写同 group 其他合法 `lmdb_index`，或标记 `skip`。

## 10. 警告与限制

- 本轮未重新调用 Qwen API，仅复用已有 CSV 与 panel/image 结果
- HTML 页面直接引用已有 `manual_hq_review/images/` 与 `qwen_vl_hq_review_pilot/panels/` 文件，不复制原图
- `visual_quality_score` 与 `sharpness_norm` 仅供参考，不能代替人工判断
- 若后续原始图片或 panel 文件被移动，HTML 页面中的对应图片会失效
- 本轮不生成最终 pair manifest

## 11. 下一步建议

1. 先人工填写 `qwen_mismatch_review_template.csv`。
2. 填写完成后，单独编写校验脚本检查 `review_decision` 与 `final_hq_index` 的合法性。
3. 仅在人工复核完成后，再决定是否构建最终训练用 pair manifest。
""",
        encoding="utf-8",
    )


def print_success(script_path: Path, out_dir: Path) -> None:
    print("Qwen mismatch review package completed.")
    print(f"Script: {script_path.resolve()}")
    print(f"Review package: {out_dir.resolve()}")
    print(f"Template CSV: {(out_dir / 'qwen_mismatch_review_template.csv').resolve()}")
    print(f"Cases CSV: {(out_dir / 'qwen_mismatch_cases.csv').resolve()}")
    print(f"Report: {(out_dir / 'reports' / 'Stage00_step10_qwen_mismatch_review_package_report.md').resolve()}")
    print(
        "No Qwen re-run, manual review site state, OCR source/data/config/checkpoint/output, "
        "or original dataset files were modified."
    )


def main() -> None:
    args = parse_args()
    if args.cases_per_page <= 0:
        raise ValueError("--cases-per-page 必须大于 0")
    if args.limit_cases is not None and args.limit_cases <= 0:
        raise ValueError("--limit-cases 必须大于 0")

    comparison_csv = args.comparison_csv.resolve()
    need_human_csv = args.need_human_csv.resolve()
    candidate_samples = args.candidate_samples.resolve()
    review_images_root = args.review_images_root.resolve()
    pilot_panels_dir = args.pilot_panels_dir.resolve()
    out_dir = ensure_output_dir(args.out_dir, args.overwrite)

    for path, label in [
        (comparison_csv, "comparison CSV"),
        (need_human_csv, "need-human CSV"),
        (candidate_samples, "candidate_samples.csv"),
        (review_images_root, "review images root"),
        (pilot_panels_dir, "pilot panels dir"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    samples_by_hash, _ = load_candidate_samples(candidate_samples, review_images_root)
    comparison_rows, _ = load_comparison_rows(comparison_csv)
    need_human_rows, _ = load_need_human_rows(need_human_csv)
    review_cases, diagnostics = build_review_cases(
        comparison_rows=comparison_rows,
        need_human_rows=need_human_rows,
        samples_by_hash=samples_by_hash,
        pilot_panels_dir=pilot_panels_dir,
        limit_cases=args.limit_cases,
    )

    cases_csv_path = out_dir / "qwen_mismatch_cases.csv"
    template_csv_path = out_dir / "qwen_mismatch_review_template.csv"
    review_pages_dir = out_dir / "review_pages"
    instructions_path = out_dir / "qwen_mismatch_review_instructions.md"
    report_path = out_dir / "reports" / "Stage00_step10_qwen_mismatch_review_package_report.md"

    write_cases_csv(cases_csv_path, review_cases)
    write_template_csv(template_csv_path, review_cases)
    page_index_rows = write_review_pages(review_cases, review_pages_dir, args.cases_per_page)
    write_instructions(instructions_path)
    write_report(
        report_path=report_path,
        out_dir=out_dir,
        args=args,
        review_cases=review_cases,
        stats=diagnostics["stats"],
        invalid_cases=diagnostics["invalid_cases"],
        page_index_rows=page_index_rows,
    )
    print_success(Path(__file__), out_dir)


if __name__ == "__main__":
    main()
