#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build final Qwen-assisted vote HQ selections and pair manifests."
    )
    parser.add_argument(
        "--candidate-groups",
        default=str(DEFAULT_ROOT / "manual_hq_review/candidate_groups.csv"),
    )
    parser.add_argument(
        "--candidate-samples",
        default=str(DEFAULT_ROOT / "manual_hq_review/candidate_samples.csv"),
    )
    parser.add_argument(
        "--qwen-parsed",
        default=str(DEFAULT_ROOT / "qwen_vl_hq_review_full/qwen_full_hq_selection_parsed.csv"),
    )
    parser.add_argument(
        "--audit-export",
        default=str(DEFAULT_ROOT / "qwen_full_audit_site/data/qwen_full_audit_export.csv"),
    )
    parser.add_argument(
        "--v1-pair-csv",
        default=str(DEFAULT_ROOT / "data/manifests/pair_manifest_top1_hq.csv"),
    )
    parser.add_argument(
        "--v2-pair-csv",
        default=str(DEFAULT_ROOT / "data/manifests/pair_manifest_top1_hq_visual_v2.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_ROOT / "final_qwen_assisted_vote"),
    )
    parser.add_argument("--treat-auto-accept-as-final", action="store_true")
    parser.add_argument("--exclude-all-hard", dest="exclude_all_hard", action="store_true")
    parser.add_argument("--no-exclude-all-hard", dest="exclude_all_hard", action="store_false")
    parser.set_defaults(exclude_all_hard=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-groups", type=int, default=None)
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_jsonl(path, rows):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def truthy(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def choose_priority(reasons):
    if any(reason.startswith("P0:") for reason in reasons):
        return "P0"
    if any(reason.startswith("P1:") for reason in reasons):
        return "P1"
    if any(reason.startswith("P2:") for reason in reasons):
        return "P2"
    return "P3"


def load_manifest_hq_by_label(path):
    label_to_hq = {}
    for row in read_csv(path):
        label = row.get("label", "").strip()
        hq_index = to_int(row.get("hq_lmdb_index"))
        if label and hq_index is not None:
            label_to_hq.setdefault(label, hq_index)
    return label_to_hq


def build_majority_structure(samples):
    counts = Counter(sample["structure"] for sample in samples if sample["structure"])
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def group_has_structure_diversity(samples):
    return len({sample["structure"] for sample in samples if sample["structure"]}) > 1


def make_audit_priority(group_row, samples, selected_sample, need_human_row):
    reasons = []
    selected_quality = selected_sample.get("quality", "")
    selected_structure = selected_sample.get("structure", "")
    selected_ocr_correct = truthy(selected_sample.get("ocr_correct"))
    risk_flags_text = (group_row.get("qwen_risk_flags") or "").strip()
    risk_flags = {
        item.strip()
        for item in risk_flags_text.replace(";", "|").split("|")
        if item.strip()
    }
    majority_structure = build_majority_structure(samples)
    has_easy = any(sample.get("quality") == "easy" for sample in samples)
    qwen_is_v1 = to_int(group_row.get("qwen_hq_index")) == to_int(group_row.get("v1_hq_index"))
    qwen_is_v2 = to_int(group_row.get("qwen_hq_index")) == to_int(group_row.get("v2_hq_index"))

    if need_human_row:
        reasons.append("P0:listed_in_need_human_review")
    if not truthy(group_row.get("json_parse_ok")):
        reasons.append("P0:json_parse_failed")
    if truthy(group_row.get("illegal_selection")):
        reasons.append("P0:illegal_selection")
    if not selected_ocr_correct:
        reasons.append("P0:selected_ocr_incorrect")
    if risk_flags & {"ambiguous", "no_clear_hq", "layout_conflict"}:
        reasons.append("P0:risk_flag_" + ",".join(sorted(risk_flags & {"ambiguous", "no_clear_hq", "layout_conflict"})))

    if has_easy and selected_quality in {"middle", "hard"}:
        reasons.append("P1:easy_exists_but_qwen_not_easy")
    if majority_structure == "vertical" and selected_structure and selected_structure != "vertical":
        reasons.append("P1:majority_vertical_but_qwen_not_vertical")
    if group_has_structure_diversity(samples) and majority_structure and selected_structure and selected_structure != majority_structure:
        reasons.append("P1:qwen_not_majority_structure")
    if to_int(group_row.get("group_size")) and to_int(group_row.get("group_size")) > 16 and not qwen_is_v1 and not qwen_is_v2:
        reasons.append("P1:panel_truncated_qwen_not_v1_v2")

    if not qwen_is_v1 and not qwen_is_v2:
        reasons.append("P2:qwen_not_v1_or_v2")
    confidence = to_float(group_row.get("qwen_confidence"))
    if confidence is not None and confidence < 0.90:
        reasons.append("P2:confidence_below_0_90")
    if to_int(group_row.get("group_size")) and to_int(group_row.get("group_size")) > 20:
        reasons.append("P2:group_size_gt_20")
    if to_int(group_row.get("group_size")) and to_int(group_row.get("group_size")) > 16:
        reasons.append("P2:panel_truncated")

    priority = choose_priority(reasons)
    if priority == "P3":
        reasons.append("P3:no_high_risk_rule_triggered")
    return priority, ";".join(reasons)


def make_selection_row(
    group_row,
    final_sample,
    selection_source,
    selection_rule,
    review_row,
    qwen_row,
):
    return {
        "label": group_row["label"],
        "label_hash": group_row["label_hash"],
        "group_rank": group_row["group_rank"],
        "group_size": group_row["group_size"],
        "final_hq_index": final_sample["lmdb_index"],
        "final_hq_quality": final_sample.get("quality", ""),
        "final_hq_structure": final_sample.get("structure", ""),
        "final_hq_structure_type": final_sample.get("structure_type", ""),
        "selection_source": selection_source,
        "selection_rule": selection_rule,
        "review_status": review_row.get("review_status", "") if review_row else "",
        "audit_priority": review_row.get("audit_priority", "") if review_row else "",
        "audit_reasons": review_row.get("audit_reasons", "") if review_row else "",
        "review_decision": review_row.get("review_decision", "") if review_row else "",
        "v1_hq_index": group_row.get("v1_hq_index", ""),
        "v2_hq_index": group_row.get("v2_hq_index", ""),
        "qwen_hq_index": qwen_row.get("selected_lmdb_index", "") if qwen_row else "",
        "v1_v2_same": str(to_int(group_row.get("v1_hq_index")) == to_int(group_row.get("v2_hq_index"))),
        "v1_qwen_same": str(to_int(group_row.get("v1_hq_index")) == to_int(qwen_row.get("selected_lmdb_index")) if qwen_row else False),
        "v2_qwen_same": str(to_int(group_row.get("v2_hq_index")) == to_int(qwen_row.get("selected_lmdb_index")) if qwen_row else False),
        "all_three_different": str(
            len(
                {
                    to_int(group_row.get("v1_hq_index")),
                    to_int(group_row.get("v2_hq_index")),
                    to_int(qwen_row.get("selected_lmdb_index")) if qwen_row else None,
                }
                - {None}
            )
            == 3
        ),
        "num_easy": group_row.get("num_easy", ""),
        "num_middle": group_row.get("num_middle", ""),
        "num_hard": group_row.get("num_hard", ""),
        "num_single": group_row.get("num_single", ""),
        "num_multi": group_row.get("num_multi", ""),
        "num_vertical": group_row.get("num_vertical", ""),
        "is_all_hard": str(to_int(group_row.get("num_hard")) == to_int(group_row.get("group_size"))),
        "qwen_confidence": qwen_row.get("confidence", "") if qwen_row else "",
        "qwen_risk_flags": qwen_row.get("risk_flags", "") if qwen_row else "",
        "final_hq_source_path": final_sample.get("source_path", ""),
    }


def make_pair_rows(selection_rows, samples_by_hash, sample_by_hash_and_index):
    pair_rows = []
    pair_id = 1
    for selection in selection_rows:
        label_hash = selection["label_hash"]
        hq_index = to_int(selection["final_hq_index"])
        hq_sample = sample_by_hash_and_index[label_hash][hq_index]
        for sample in samples_by_hash[label_hash]:
            if to_int(sample["lmdb_index"]) == hq_index:
                continue
            lq_priority = to_int(sample.get("quality_priority"))
            hq_priority = to_int(hq_sample.get("quality_priority"))
            structure_relation = (
                "same_structure"
                if sample.get("structure") == hq_sample.get("structure")
                else "cross_structure"
            )
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "label": selection["label"],
                    "label_hash": label_hash,
                    "group_rank": selection["group_rank"],
                    "group_size": selection["group_size"],
                    "lq_lmdb_index": sample.get("lmdb_index", ""),
                    "lq_quality": sample.get("quality", ""),
                    "lq_quality_priority": sample.get("quality_priority", ""),
                    "lq_structure": sample.get("structure", ""),
                    "lq_structure_type": sample.get("structure_type", ""),
                    "lq_ocr_correct": sample.get("ocr_correct", ""),
                    "lq_visual_quality_score": sample.get("visual_quality_score", ""),
                    "lq_source_path": sample.get("source_path", ""),
                    "hq_lmdb_index": hq_sample.get("lmdb_index", ""),
                    "hq_quality": hq_sample.get("quality", ""),
                    "hq_quality_priority": hq_sample.get("quality_priority", ""),
                    "hq_structure": hq_sample.get("structure", ""),
                    "hq_structure_type": hq_sample.get("structure_type", ""),
                    "hq_ocr_correct": hq_sample.get("ocr_correct", ""),
                    "hq_visual_quality_score": hq_sample.get("visual_quality_score", ""),
                    "hq_source_path": hq_sample.get("source_path", ""),
                    "selection_source": selection["selection_source"],
                    "selection_rule": selection["selection_rule"],
                    "quality_relation": f"{sample.get('quality', '')}_to_{hq_sample.get('quality', '')}",
                    "structure_relation": structure_relation,
                    "pair_type": "qwen_assisted_vote_hq",
                }
            )
            pair_id += 1
    return pair_rows


def main():
    args = parse_args()

    candidate_groups_path = Path(args.candidate_groups)
    candidate_samples_path = Path(args.candidate_samples)
    qwen_parsed_path = Path(args.qwen_parsed)
    audit_export_path = Path(args.audit_export)
    v1_pair_path = Path(args.v1_pair_csv)
    v2_pair_path = Path(args.v2_pair_csv)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    candidate_groups = read_csv(candidate_groups_path)
    candidate_samples = read_csv(candidate_samples_path)
    qwen_rows = read_csv(qwen_parsed_path)
    audit_rows = read_csv(audit_export_path)

    v1_by_label = load_manifest_hq_by_label(v1_pair_path)
    v2_by_label = load_manifest_hq_by_label(v2_pair_path)

    samples_by_hash = defaultdict(list)
    sample_by_hash_and_index = defaultdict(dict)
    for row in candidate_samples:
        label_hash = row["label_hash"]
        lmdb_index = to_int(row.get("lmdb_index"))
        if lmdb_index is None:
            continue
        normalized = dict(row)
        normalized["lmdb_index"] = lmdb_index
        samples_by_hash[label_hash].append(normalized)
        sample_by_hash_and_index[label_hash][lmdb_index] = normalized

    for label_hash in samples_by_hash:
        samples_by_hash[label_hash].sort(key=lambda item: item["lmdb_index"])

    qwen_by_hash = {row["label_hash"]: row for row in qwen_rows}
    audit_by_hash = {row["label_hash"]: row for row in audit_rows}
    need_human_hashes = {
        row["label_hash"]
        for row in qwen_rows
        if truthy(row.get("need_human_review"))
        or (to_float(row.get("confidence")) is not None and to_float(row.get("confidence")) < 0.80)
        or truthy(row.get("illegal_selection"))
        or not truthy(row.get("json_parse_ok"))
        or any(
            flag in (row.get("risk_flags") or "")
            for flag in ["no_clear_hq", "ambiguous", "layout_conflict"]
        )
    }

    group_rows = []
    manifest_mismatch_warnings = []
    for row in candidate_groups:
        group_size = to_int(row.get("group_size"))
        if group_size is None or group_size < 2:
            continue
        normalized = dict(row)
        label = row["label"]
        if not normalized.get("v1_hq_index"):
            normalized["v1_hq_index"] = v1_by_label.get(label, "")
        if not normalized.get("v2_hq_index"):
            normalized["v2_hq_index"] = v2_by_label.get(label, "")
        v1_manifest = v1_by_label.get(label)
        v2_manifest = v2_by_label.get(label)
        if v1_manifest is not None and to_int(row.get("v1_hq_index")) not in {None, v1_manifest}:
            manifest_mismatch_warnings.append(f"v1 mismatch label={label} group={row['group_rank']}")
        if v2_manifest is not None and to_int(row.get("v2_hq_index")) not in {None, v2_manifest}:
            manifest_mismatch_warnings.append(f"v2 mismatch label={label} group={row['group_rank']}")
        group_rows.append(normalized)

    group_rows.sort(key=lambda item: to_int(item["group_rank"]) or 0)
    if args.limit_groups is not None:
        group_rows = group_rows[: args.limit_groups]

    excluded_all_hard = []
    excluded_invalid = []
    final_selection_rows = []
    human_review_used = 0
    vote_groups = 0
    majority_vote_count = 0
    all_different_choose_v2_count = 0
    human_reviewed_groups = 0
    human_reviewed_valid_groups = 0

    for group_row in group_rows:
        label_hash = group_row["label_hash"]
        samples = samples_by_hash.get(label_hash, [])
        sample_map = sample_by_hash_and_index.get(label_hash, {})
        group_size = to_int(group_row.get("group_size")) or 0
        num_hard = to_int(group_row.get("num_hard"))
        if num_hard is None:
            num_hard = sum(sample.get("quality") == "hard" for sample in samples)
            group_row["num_hard"] = str(num_hard)
        if not group_row.get("num_easy"):
            group_row["num_easy"] = str(sum(sample.get("quality") == "easy" for sample in samples))
        if not group_row.get("num_middle"):
            group_row["num_middle"] = str(sum(sample.get("quality") == "middle" for sample in samples))
        if not group_row.get("num_single"):
            group_row["num_single"] = str(sum(sample.get("structure") == "single" for sample in samples))
        if not group_row.get("num_multi"):
            group_row["num_multi"] = str(sum(sample.get("structure") == "multi" for sample in samples))
        if not group_row.get("num_vertical"):
            group_row["num_vertical"] = str(sum(sample.get("structure") == "vertical" for sample in samples))

        if args.exclude_all_hard and group_size > 0 and num_hard == group_size:
            excluded_all_hard.append(
                {
                    "label": group_row["label"],
                    "label_hash": label_hash,
                    "group_rank": group_row["group_rank"],
                    "group_size": group_row["group_size"],
                    "num_easy": group_row.get("num_easy", ""),
                    "num_middle": group_row.get("num_middle", ""),
                    "num_hard": group_row.get("num_hard", ""),
                    "exclude_reason": "all_candidates_hard",
                }
            )
            continue

        review_row = audit_by_hash.get(label_hash)
        qwen_row = qwen_by_hash.get(label_hash)
        if review_row and review_row.get("review_status") == "reviewed" and to_int(review_row.get("final_hq_index")) is not None:
            human_reviewed_groups += 1
        elif args.treat_auto_accept_as_final and review_row and review_row.get("review_status") == "auto_accept" and to_int(review_row.get("final_hq_index")) is not None:
            human_reviewed_groups += 1

        def append_invalid(reason, final_candidate=None, selection_source="", selection_rule=""):
            excluded_invalid.append(
                {
                    "label": group_row["label"],
                    "label_hash": label_hash,
                    "group_rank": group_row["group_rank"],
                    "group_size": group_row["group_size"],
                    "v1_hq_index": group_row.get("v1_hq_index", ""),
                    "v2_hq_index": group_row.get("v2_hq_index", ""),
                    "qwen_hq_index": qwen_row.get("selected_lmdb_index", "") if qwen_row else "",
                    "review_status": review_row.get("review_status", "") if review_row else "",
                    "final_hq_index": final_candidate if final_candidate is not None else "",
                    "selection_source": selection_source,
                    "selection_rule": selection_rule,
                    "exclude_reason": reason,
                }
            )

        review_eligible = False
        if review_row and to_int(review_row.get("final_hq_index")) is not None:
            if review_row.get("review_status") == "reviewed":
                review_eligible = True
            elif args.treat_auto_accept_as_final and review_row.get("review_status") == "auto_accept":
                review_eligible = True

        selected_index = None
        selection_source = ""
        selection_rule = ""

        if review_eligible:
            candidate = to_int(review_row.get("final_hq_index"))
            if candidate not in sample_map:
                append_invalid("reviewed_final_hq_not_in_group", candidate, "human_review", "reviewed_final_hq")
                continue
            selected_index = candidate
            selection_source = "human_review"
            selection_rule = "reviewed_final_hq"
            human_review_used += 1
            human_reviewed_valid_groups += 1
        else:
            vote_groups += 1
            vote_candidates = {
                "v1": to_int(group_row.get("v1_hq_index")),
                "v2": to_int(group_row.get("v2_hq_index")),
                "qwen": to_int(qwen_row.get("selected_lmdb_index")) if qwen_row else None,
            }
            valid_votes = {key: value for key, value in vote_candidates.items() if value in sample_map}
            if len(valid_votes) >= 2:
                counts = Counter(valid_votes.values())
                winner, winner_count = counts.most_common(1)[0]
                if winner_count >= 2:
                    selected_index = winner
                    selection_source = "vote_majority"
                    selection_rule = "majority_vote"
                    majority_vote_count += 1
                else:
                    fallback_v2 = valid_votes.get("v2")
                    if fallback_v2 is None:
                        fallback_qwen = valid_votes.get("qwen")
                        if fallback_qwen is None:
                            append_invalid("all_vote_sources_different_and_v2_missing", "", "vote_tie_break_v2", "all_different_choose_v2")
                            continue
                        selected_index = fallback_qwen
                        selection_source = "vote_fallback_qwen"
                        selection_rule = "all_different_v2_missing_choose_qwen"
                    else:
                        selected_index = fallback_v2
                        selection_source = "vote_tie_break_v2"
                        selection_rule = "all_different_choose_v2"
                        all_different_choose_v2_count += 1
            else:
                fallback_order = [
                    ("vote_fallback_v2", "insufficient_valid_votes_choose_v2", vote_candidates.get("v2")),
                    ("vote_fallback_qwen", "insufficient_valid_votes_choose_qwen", vote_candidates.get("qwen")),
                    ("vote_fallback_v1", "insufficient_valid_votes_choose_v1", vote_candidates.get("v1")),
                ]
                for source_name, rule_name, candidate in fallback_order:
                    if candidate in sample_map:
                        selected_index = candidate
                        selection_source = source_name
                        selection_rule = rule_name
                        break
                if selected_index is None:
                    append_invalid("fewer_than_two_valid_votes_and_no_valid_fallback", "", "vote_fallback", "invalid_vote_sources")
                    continue

        final_sample = sample_map.get(selected_index)
        if final_sample is None:
            append_invalid("final_hq_missing_sample", selected_index, selection_source, selection_rule)
            continue
        if not (final_sample.get("quality") or "").strip():
            append_invalid("final_hq_missing_quality", selected_index, selection_source, selection_rule)
            continue
        if not (final_sample.get("structure") or "").strip():
            append_invalid("final_hq_missing_structure", selected_index, selection_source, selection_rule)
            continue

        final_selection_rows.append(
            make_selection_row(
                group_row=group_row,
                final_sample=final_sample,
                selection_source=selection_source,
                selection_rule=selection_rule,
                review_row=review_row,
                qwen_row=qwen_row,
            )
        )

    for row in final_selection_rows:
        review_row = audit_by_hash.get(row["label_hash"])
        qwen_row = qwen_by_hash.get(row["label_hash"])
        if review_row:
            row["review_status"] = review_row.get("review_status", "")
            row["audit_priority"] = review_row.get("audit_priority", "")
            row["audit_reasons"] = review_row.get("audit_reasons", "")
            row["review_decision"] = review_row.get("review_decision", "")
        elif qwen_row:
            synthetic_group = {
                "group_size": row["group_size"],
                "qwen_hq_index": qwen_row.get("selected_lmdb_index", ""),
                "v1_hq_index": row["v1_hq_index"],
                "v2_hq_index": row["v2_hq_index"],
                "json_parse_ok": qwen_row.get("json_parse_ok", ""),
                "illegal_selection": qwen_row.get("illegal_selection", ""),
                "qwen_confidence": qwen_row.get("confidence", ""),
                "qwen_risk_flags": qwen_row.get("risk_flags", ""),
            }
            selected_qwen_index = to_int(row["qwen_hq_index"])
            selected_sample = sample_by_hash_and_index[row["label_hash"]].get(selected_qwen_index)
            if selected_sample is not None:
                priority, reasons = make_audit_priority(
                    group_row=synthetic_group,
                    samples=samples_by_hash[row["label_hash"]],
                    selected_sample=selected_sample,
                    need_human_row=row["label_hash"] in need_human_hashes,
                )
                row["audit_priority"] = priority
                row["audit_reasons"] = reasons

    final_selection_rows.sort(key=lambda item: to_int(item["group_rank"]) or 0)
    pair_rows = make_pair_rows(final_selection_rows, samples_by_hash, sample_by_hash_and_index)
    pair_rows.sort(key=lambda item: int(item["pair_id"]))

    same_structure_pairs = [row for row in pair_rows if row["structure_relation"] == "same_structure"]
    quality_improved_pairs = [
        row
        for row in pair_rows
        if to_int(row.get("lq_quality_priority")) is not None
        and to_int(row.get("hq_quality_priority")) is not None
        and to_int(row.get("lq_quality_priority")) < to_int(row.get("hq_quality_priority"))
    ]
    same_structure_quality_improved_pairs = [
        row
        for row in quality_improved_pairs
        if row["structure_relation"] == "same_structure"
    ]

    selection_fieldnames = [
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "final_hq_index",
        "final_hq_quality",
        "final_hq_structure",
        "final_hq_structure_type",
        "selection_source",
        "selection_rule",
        "review_status",
        "audit_priority",
        "audit_reasons",
        "review_decision",
        "v1_hq_index",
        "v2_hq_index",
        "qwen_hq_index",
        "v1_v2_same",
        "v1_qwen_same",
        "v2_qwen_same",
        "all_three_different",
        "num_easy",
        "num_middle",
        "num_hard",
        "num_single",
        "num_multi",
        "num_vertical",
        "is_all_hard",
        "qwen_confidence",
        "qwen_risk_flags",
        "final_hq_source_path",
    ]
    pair_fieldnames = [
        "pair_id",
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "lq_lmdb_index",
        "lq_quality",
        "lq_quality_priority",
        "lq_structure",
        "lq_structure_type",
        "lq_ocr_correct",
        "lq_visual_quality_score",
        "lq_source_path",
        "hq_lmdb_index",
        "hq_quality",
        "hq_quality_priority",
        "hq_structure",
        "hq_structure_type",
        "hq_ocr_correct",
        "hq_visual_quality_score",
        "hq_source_path",
        "selection_source",
        "selection_rule",
        "quality_relation",
        "structure_relation",
        "pair_type",
    ]
    all_hard_fieldnames = [
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "num_easy",
        "num_middle",
        "num_hard",
        "exclude_reason",
    ]
    invalid_fieldnames = [
        "label",
        "label_hash",
        "group_rank",
        "group_size",
        "v1_hq_index",
        "v2_hq_index",
        "qwen_hq_index",
        "review_status",
        "final_hq_index",
        "selection_source",
        "selection_rule",
        "exclude_reason",
    ]

    write_csv(out_dir / "final_hq_selection_qwen_assisted_vote.csv", final_selection_rows, selection_fieldnames)
    write_jsonl(out_dir / "final_hq_selection_qwen_assisted_vote.jsonl", final_selection_rows)
    write_csv(out_dir / "excluded_all_hard_groups.csv", excluded_all_hard, all_hard_fieldnames)
    write_csv(out_dir / "excluded_invalid_groups.csv", excluded_invalid, invalid_fieldnames)
    write_csv(out_dir / "pair_manifest_qwen_assisted_vote_hq.csv", pair_rows, pair_fieldnames)
    write_csv(out_dir / "pair_manifest_qwen_assisted_vote_hq_same_structure.csv", same_structure_pairs, pair_fieldnames)
    write_csv(out_dir / "pair_manifest_qwen_assisted_vote_hq_quality_improved.csv", quality_improved_pairs, pair_fieldnames)
    write_csv(
        out_dir / "pair_manifest_qwen_assisted_vote_hq_same_structure_quality_improved.csv",
        same_structure_quality_improved_pairs,
        pair_fieldnames,
    )

    num_groups_ge2 = len(group_rows)
    final_quality_counts = Counter(row["final_hq_quality"] for row in final_selection_rows)
    final_structure_counts = Counter(row["final_hq_structure"] for row in final_selection_rows)
    selection_source_counts = Counter(row["selection_source"] for row in final_selection_rows)
    selection_rule_counts = Counter(row["selection_rule"] for row in final_selection_rows)
    pair_selection_source_counts = Counter(row["selection_source"] for row in pair_rows)
    quality_relation_counts = Counter(row["quality_relation"] for row in pair_rows)
    structure_relation_counts = Counter(row["structure_relation"] for row in pair_rows)
    v1_v2_same = sum(row["v1_v2_same"] == "True" for row in final_selection_rows)
    v1_qwen_same = sum(row["v1_qwen_same"] == "True" for row in final_selection_rows)
    v2_qwen_same = sum(row["v2_qwen_same"] == "True" for row in final_selection_rows)
    all_three_different = sum(row["all_three_different"] == "True" for row in final_selection_rows)

    report_lines = [
        "# Stage00_step13 Final Qwen-assisted Vote-HQ Pair Report",
        "",
        "## 1. 范围与隔离声明",
        "本轮只在 `experiments/dit_lq_hq_v1/final_qwen_assisted_vote/` 下生成最终 HQ selection、pair manifest、过滤子集和报告。",
        "未修改 `qwen_full_audit_site` 状态、Qwen 全量结果、v1/v2 manifest、原始数据、OCR LMDB、configs、checkpoints、outputs 或源码。",
        "",
        "## 2. 输入与输出",
        f"- candidate_groups: `{candidate_groups_path}`",
        f"- candidate_samples: `{candidate_samples_path}`",
        f"- qwen parsed: `{qwen_parsed_path}`",
        f"- audit export: `{audit_export_path}`",
        f"- v1 pair manifest: `{v1_pair_path}`",
        f"- v2 pair manifest: `{v2_pair_path}`",
        f"- out dir: `{out_dir}`",
        "",
        "## 3. 最终 HQ 选择规则",
        "- `review_status == reviewed` 且 `final_hq_index` 非空的 group，直接使用人工 reviewed 结果。",
        f"- `review_status == auto_accept` 默认不直接视为最终结果；本次运行 `treat_auto_accept_as_final={args.treat_auto_accept_as_final}`。",
        "- 未人工 reviewed 的 group 使用 `v1 / v2 / Qwen` 三方投票。",
        "- 若三者中任意两者相同，取多数票。",
        "- 若三者都不同，优先使用 `v2_hq_index`。",
        "- 若合法投票来源少于 2 个，按 `v2 -> qwen -> v1` 回退；仍无合法值则排除。",
        "",
        "## 4. Group 过滤统计",
        f"- num_groups_ge2: {num_groups_ge2}",
        f"- num_all_hard_excluded: {len(excluded_all_hard)}",
        f"- num_invalid_excluded: {len(excluded_invalid)}",
        f"- num_final_hq_groups: {len(final_selection_rows)}",
        "",
        "## 5. all-hard group 统计",
        f"- excluded_all_hard_enabled: {args.exclude_all_hard}",
        f"- num_all_hard_excluded: {len(excluded_all_hard)}",
        "",
        "## 6. 人工 reviewed 使用统计",
        f"- num_human_reviewed_groups: {human_reviewed_groups}",
        f"- num_human_reviewed_used: {human_review_used}",
        "",
        "## 7. v1/v2/Qwen 投票统计",
        f"- num_vote_groups: {vote_groups}",
        f"- num_majority_vote: {majority_vote_count}",
        f"- num_all_different_choose_v2: {all_different_choose_v2_count}",
        f"- num_v1_v2_same: {v1_v2_same}",
        f"- num_v1_qwen_same: {v1_qwen_same}",
        f"- num_v2_qwen_same: {v2_qwen_same}",
        f"- num_all_three_different: {all_three_different}",
        "",
        "## 8. Final HQ 分布",
        f"- final_hq_quality distribution: {dict(final_quality_counts)}",
        f"- final_hq_structure distribution: {dict(final_structure_counts)}",
        f"- selection_source distribution: {dict(selection_source_counts)}",
        f"- selection_rule distribution: {dict(selection_rule_counts)}",
        "",
        "## 9. Pair Manifest 统计",
        f"- num_pairs_total: {len(pair_rows)}",
        f"- structure_relation distribution: {dict(structure_relation_counts)}",
        f"- quality_relation distribution: {dict(quality_relation_counts)}",
        f"- selection_source pair counts: {dict(pair_selection_source_counts)}",
        "",
        "## 10. 过滤子集统计",
        f"- same_structure pairs: {len(same_structure_pairs)}",
        f"- quality_improved pairs: {len(quality_improved_pairs)}",
        f"- same_structure_quality_improved pairs: {len(same_structure_quality_improved_pairs)}",
        "",
        "## 11. 风险与限制",
        "- `auto_accept` 默认不等同于人工显式审核，仍进入投票规则。",
        "- v1/v2 HQ 以 `candidate_groups.csv` 为主，并用原 pair manifest 做存在性与一致性校验。",
        "- 若 group 的 reviewed final 或投票候选不属于该 group，会被写入 `excluded_invalid_groups.csv` 并排除。",
        f"- v1/v2 manifest mismatch warnings: {len(manifest_mismatch_warnings)}",
        "",
        "## 12. 下一步建议",
        "- 优先使用 `pair_manifest_qwen_assisted_vote_hq_same_structure_quality_improved.csv` 作为 Stage 1 的更干净训练子集。",
        "- 对 `excluded_invalid_groups.csv` 做人工回查，确认是否存在上游审核导出异常或候选表不一致。",
        "- 如需更激进的数据量，可再评估是否将 `auto_accept` 直接视为最终结果并重跑一次构造脚本。",
    ]
    (out_dir / "reports/Stage00_step13_final_qwen_assisted_vote_pair_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
