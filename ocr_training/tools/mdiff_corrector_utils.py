import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def read_charset(charset_yaml: str) -> str:
    with open(charset_yaml, "r", encoding="utf-8") as f:
        data = yaml.load(f, yaml.Loader)
    return data["model"]["charset_train"]


def extract_sequence(token_ids: Sequence[int], eos_id: int, pad_id: int) -> Tuple[List[int], bool, int]:
    seq = []
    has_eos = False
    eos_position = None
    for idx, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if token_id == pad_id:
            break
        if token_id == eos_id:
            has_eos = True
            eos_position = idx
            break
        seq.append(token_id)
    if eos_position is None:
        eos_position = len(seq)
    return seq, has_eos, eos_position


def normalize_prediction(
    pred_ids: Sequence[int],
    pred_conf: Sequence[float],
    eos_id: int,
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    pred_ids = np.asarray(pred_ids, dtype=np.int16)
    pred_conf = np.asarray(pred_conf, dtype=np.float32)
    fixed_ids = np.full_like(pred_ids, pad_id)
    fixed_conf = np.zeros_like(pred_conf)
    eos_matches = np.where(pred_ids == eos_id)[0]
    if eos_matches.size:
        stop = int(eos_matches[0])
        copy_len = stop + 1
        fixed_ids[:copy_len] = pred_ids[:copy_len]
        fixed_conf[:copy_len] = pred_conf[:copy_len]
        return fixed_ids, fixed_conf, stop, stop
    fixed_ids[:] = pred_ids
    fixed_conf[:] = pred_conf
    return fixed_ids, fixed_conf, int(len(pred_ids)), int(len(pred_ids))


def token_ids_to_text(token_ids: Sequence[int], tokenizer) -> str:
    seq, _, _ = extract_sequence(token_ids, tokenizer.eos_id, tokenizer.pad_id)
    return tokenizer._ids2tok(seq, join=True)


def align_pred_gt(pred_ids: Sequence[int], gt_ids: Sequence[int], eos_id: int, pad_id: int) -> Dict[str, object]:
    pred_seq, pred_has_eos, pred_eos_pos = extract_sequence(pred_ids, eos_id, pad_id)
    gt_seq, gt_has_eos, gt_eos_pos = extract_sequence(gt_ids, eos_id, pad_id)
    n = len(pred_seq)
    m = len(gt_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == gt_seq[j - 1]:
                cost = dp[i - 1][j - 1]
                op = "correct"
            else:
                cost = dp[i - 1][j - 1] + 1
                op = "replace"
            delete_cost = dp[i - 1][j] + 1
            insert_cost = dp[i][j - 1] + 1
            best_cost = cost
            best_op = op
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_op = "delete"
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_op = "insert"
            dp[i][j] = best_cost
            back[i][j] = best_op
    i, j = n, m
    ops = []
    steps = []
    while i > 0 or j > 0:
        op = back[i][j]
        ops.append(op)
        if op in {"correct", "replace"}:
            steps.append(
                {
                    "op": op,
                    "pred_pos": i - 1,
                    "gt_pos": j - 1,
                    "pred_id": int(pred_seq[i - 1]),
                    "gt_id": int(gt_seq[j - 1]),
                }
            )
            i -= 1
            j -= 1
        elif op == "delete":
            steps.append(
                {
                    "op": op,
                    "pred_pos": i - 1,
                    "gt_pos": None,
                    "pred_id": int(pred_seq[i - 1]),
                    "gt_id": None,
                }
            )
            i -= 1
        elif op == "insert":
            steps.append(
                {
                    "op": op,
                    "pred_pos": None,
                    "gt_pos": j - 1,
                    "pred_id": None,
                    "gt_id": int(gt_seq[j - 1]),
                }
            )
            j -= 1
        else:
            break
    ops.reverse()
    steps.reverse()
    if pred_has_eos and gt_has_eos:
        if pred_eos_pos < gt_eos_pos:
            ops.append("eos_early")
        elif pred_eos_pos > gt_eos_pos:
            ops.append("eos_late")
    replace_only_candidate = len(pred_seq) == len(gt_seq) and all(op in {"correct", "replace"} for op in ops)
    return {
        "ops": ops,
        "pred_length": len(pred_seq),
        "gt_length": len(gt_seq),
        "pred_eos_position": pred_eos_pos,
        "gt_eos_position": gt_eos_pos,
        "has_insert_delete": any(op in {"insert", "delete"} for op in ops),
        "replace_count": sum(op == "replace" for op in ops),
        "correct_count": sum(op == "correct" for op in ops),
        "insert_count": sum(op == "insert" for op in ops),
        "delete_count": sum(op == "delete" for op in ops),
        "replace_only_candidate": replace_only_candidate,
        "steps": steps,
    }


def token_id_to_char(token_id: Optional[int], tokenizer) -> Optional[str]:
    if token_id is None:
        return None
    return tokenizer._ids2tok([int(token_id)], join=True)


def classify_char(char: Optional[str]) -> str:
    if not char:
        return "other"
    if len(char) != 1:
        return "other"
    code = ord(char)
    if char.isdigit():
        return "digit"
    if ("A" <= char <= "Z") or ("a" <= char <= "z"):
        return "alphabet"
    if 0x4E00 <= code <= 0x9FFF:
        return "chinese"
    return "other"


def classify_token_id(token_id: Optional[int], tokenizer) -> str:
    return classify_char(token_id_to_char(token_id, tokenizer))


def load_confusion_table(confusion_table_path: str) -> List[Dict[str, object]]:
    payload = json.loads(Path(confusion_table_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pairs" in payload:
        return list(payload["pairs"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported confusion table format: {confusion_table_path}")


def build_confusion_knowledge(confusion_rows: List[Dict[str, object]]) -> Dict[str, object]:
    pair_ids = set()
    pred_ids = set()
    gt_ids = set()
    counts_by_pair = {}
    for row in confusion_rows:
        pred_id = int(row["pred_token_id"])
        gt_id = int(row["gt_token_id"])
        pair_ids.add((pred_id, gt_id))
        pred_ids.add(pred_id)
        gt_ids.add(gt_id)
        counts_by_pair[(pred_id, gt_id)] = int(row.get("count", 0))
    return {
        "pair_ids": pair_ids,
        "pred_ids": pred_ids,
        "gt_ids": gt_ids,
        "counts_by_pair": counts_by_pair,
    }


def load_pair_thresholds(pair_thresholds_path: Optional[str]) -> Dict[str, object]:
    if not pair_thresholds_path:
        return {"default_thresholds": {}, "pairs": {}}
    payload = json.loads(Path(pair_thresholds_path).read_text(encoding="utf-8"))
    if "pairs" in payload:
        return {
            "default_thresholds": dict(payload.get("default_thresholds", {})),
            "pairs": dict(payload["pairs"]),
            "source": payload.get("source"),
        }
    return {"default_thresholds": {}, "pairs": dict(payload)}


def load_pair_difficulty_table(pair_difficulty_path: Optional[str]) -> List[Dict[str, object]]:
    if not pair_difficulty_path:
        return []
    payload = json.loads(Path(pair_difficulty_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pairs" in payload:
        return list(payload["pairs"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported pair difficulty format: {pair_difficulty_path}")


def load_manifest(cache_dir: str) -> List[Dict[str, object]]:
    manifest_path = Path(cache_dir) / "manifest.jsonl"
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_feature_shards(cache_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    shards = {}
    for shard_path in sorted(Path(cache_dir).glob("features_*.npz")):
        with np.load(shard_path, allow_pickle=False) as npz:
            shards[shard_path.name] = {key: npz[key] for key in npz.files}
    return shards


def record_arrays(record: Dict[str, object], shards: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    shard = shards[record["feature_shard"]]
    index = int(record["feature_index"])
    row = {}
    for key, value in shard.items():
        row[key] = value[index]
    return row
