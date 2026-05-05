#!/usr/bin/env python3
import argparse
import csv
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = EXPERIMENT_ROOT.resolve()
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
OCR_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training").resolve()
PRETRAIN_MAE = (OCR_ROOT / "pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth").resolve()
CHARSET_CONFIG = (OCR_ROOT / "configs/charset/SLP34K_568.yaml").resolve()

if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import lmdb
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


CSV_FIELDS = [
    "lmdb_root",
    "lmdb_index",
    "label",
    "pred",
    "correct",
    "confidence",
    "avg_conf",
    "min_conf",
    "nll",
    "pred_length",
    "label_length",
    "quality",
    "structure",
    "structure_type",
    "source_path",
]


class ExportLmdbDataset(LmdbDataset):
    def __getitem__(self, index):
        if self.unlabelled:
            raise ValueError("ExportLmdbDataset 不支持 unlabelled 模式")
        label = self.labels[index]
        lmdb_index = self.filtered_index_list[index]
        img_key = f"image-{lmdb_index:09d}".encode()
        meta_key = f"meta-{lmdb_index:09d}".encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
            meta_buf = txn.get(meta_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        meta = json.loads(meta_buf.decode("utf-8")) if meta_buf is not None else None
        return {
            "image": img,
            "label": label,
            "lmdb_index": lmdb_index,
            "lmdb_root": self.root,
            "meta": meta,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([row["image"] for row in batch], dim=0),
        "labels": [row["label"] for row in batch],
        "lmdb_index": [row["lmdb_index"] for row in batch],
        "lmdb_root": [row["lmdb_root"] for row in batch],
        "meta": [row["meta"] for row in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="导出 metadata-rich train LMDB 的 OCR 预测与置信度")
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def validate_args(args):
    lmdb_root = args.lmdb_root.resolve()
    ckpt = args.ckpt.resolve()
    output_csv = args.output_csv.resolve()
    output_jsonl = args.output_jsonl.resolve()
    report = args.report.resolve()

    if not lmdb_root.exists():
        raise FileNotFoundError(f"lmdb_root 不存在: {lmdb_root}")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    if not PRETRAIN_MAE.exists():
        raise FileNotFoundError(f"MAE 预训练权重不存在: {PRETRAIN_MAE}")
    if not CHARSET_CONFIG.exists():
        raise FileNotFoundError(f"charset 配置不存在: {CHARSET_CONFIG}")
    for output_path in [output_csv, output_jsonl, report]:
        if not is_relative_to(output_path, SAFE_OUTPUT_ROOT):
            raise ValueError(f"输出路径必须位于 {SAFE_OUTPUT_ROOT} 下: {output_path}")
        if is_relative_to(output_path, FORBIDDEN_OUTPUT_ROOT):
            raise ValueError(f"输出路径不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {output_path}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0")
    for output_path in [output_csv, output_jsonl, report]:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"输出文件已存在，未提供 --overwrite: {output_path}")
    return lmdb_root, ckpt, output_csv, output_jsonl, report


def resolve_device(requested_device: str):
    note = None
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested_device), note
        note = f"请求设备 {requested_device} 不可用，已自动回退到 cpu"
        return torch.device("cpu"), note
    return torch.device(requested_device), note


def load_charset_test():
    config = yaml.load(CHARSET_CONFIG.read_text(encoding="utf-8"), yaml.Loader)
    return config["model"]["charset_test"]


def build_dataset(lmdb_root: Path, charset: str, max_label_length: int, img_size):
    transform = SceneTextDataModule.get_transform(tuple(img_size), augment=False)
    return ExportLmdbDataset(
        root=str(lmdb_root),
        charset=charset,
        max_label_len=max_label_length,
        transform=transform,
    )


def maybe_limit_dataset(dataset, limit: Optional[int]):
    if limit is None or limit >= len(dataset):
        return dataset
    from torch.utils.data import Subset

    return Subset(dataset, list(range(limit)))


def create_dataloader(dataset, batch_size: int, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )


def confidence_stats(seq_probs: torch.Tensor):
    probs = seq_probs.float().clamp_min(1e-12)
    avg_conf = float(probs.mean().item())
    min_conf = float(probs.min().item())
    nll = float((-probs.log().sum()).item())
    return avg_conf, min_conf, nll


def record_from_batch(batch: Dict[str, Any], row_idx: int, pred: str, seq_probs: torch.Tensor):
    meta = batch["meta"][row_idx] or {}
    avg_conf, min_conf, nll = confidence_stats(seq_probs)
    label = batch["labels"][row_idx]
    record = {
        "lmdb_root": batch["lmdb_root"][row_idx],
        "lmdb_index": int(batch["lmdb_index"][row_idx]),
        "label": label,
        "pred": pred,
        "correct": pred == label,
        "confidence": avg_conf,
        "avg_conf": avg_conf,
        "min_conf": min_conf,
        "nll": nll,
        "pred_length": len(pred),
        "label_length": len(label),
        "quality": meta.get("quality"),
        "structure": meta.get("structure"),
        "structure_type": meta.get("structure_type"),
        "source_path": meta.get("source_path"),
        "metadata": meta,
    }
    return record


def init_group_stats():
    return {"count": 0, "correct": 0, "confidence_sum": 0.0, "avg_conf_sum": 0.0, "min_conf_sum": 0.0}


def update_group_stats(stats: Dict[str, float], record: Dict[str, Any]):
    stats["count"] += 1
    stats["correct"] += int(record["correct"])
    stats["confidence_sum"] += record["confidence"]
    stats["avg_conf_sum"] += record["avg_conf"]
    stats["min_conf_sum"] += record["min_conf"]


def finalize_group_stats(grouped: Dict[str, Dict[str, float]]):
    rows = []
    for key, stats in grouped.items():
        count = stats["count"]
        rows.append(
            {
                "group": key,
                "count": count,
                "accuracy": stats["correct"] / count if count else 0.0,
                "mean_confidence": stats["confidence_sum"] / count if count else 0.0,
                "mean_avg_conf": stats["avg_conf_sum"] / count if count else 0.0,
                "mean_min_conf": stats["min_conf_sum"] / count if count else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["group"])


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], headers: Optional[Dict[str, str]] = None):
    if not rows:
        return "无"
    headers = headers or {}
    lines = [
        "| " + " | ".join(headers.get(col, col) for col in columns) + " |",
        "| " + " | ".join("---:" if col not in {"group", "label", "pred", "quality", "structure", "structure_type", "source_path"} else "---" for col in columns) + " |",
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


def write_report(
    report_path: Path,
    status: str,
    environment: Dict[str, Any],
    inputs_outputs: Dict[str, Any],
    model_loading: Dict[str, Any],
    dataset_reading: Dict[str, Any],
    export_summary: Dict[str, Any],
    overall_stats: Dict[str, Any],
    quality_rows: List[Dict[str, Any]],
    structure_rows: List[Dict[str, Any]],
    structure_type_rows: List[Dict[str, Any]],
    first_samples: List[Dict[str, Any]],
    wrong_samples: List[Dict[str, Any]],
    warnings: List[str],
):
    sample_rows = [
        {
            "lmdb_index": row["lmdb_index"],
            "label": row["label"],
            "pred": row["pred"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "quality": row["quality"],
            "structure": row["structure"],
            "source_path": row["source_path"],
        }
        for row in first_samples
    ]
    wrong_rows = [
        {
            "lmdb_index": row["lmdb_index"],
            "label": row["label"],
            "pred": row["pred"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "quality": row["quality"],
            "structure": row["structure"],
            "source_path": row["source_path"],
        }
        for row in wrong_samples
    ]

    report_text = f"""# Train OCR Confidence Export Report

## 1. Scope and Isolation Statement

本轮仅在 `experiments/dit_lq_hq_v1/` 下创建或覆盖脚本、manifest 和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。
当前执行状态：{status}

## 2. Environment

- Python: `{environment["python"]}`
- torch: `{environment["torch_version"]}`
- pytorch_lightning: `{environment["pl_version"]}`
- lmdb: `OK`
- PIL: `OK`
- yaml: `OK`
- hydra: `OK`
- CUDA 可用: `{environment["cuda_available"]}`
- 实际设备: `{environment["device"]}`
- 设备说明: `{environment["device_note"] or "无"}`

## 3. Inputs and Outputs

- 输入 LMDB: `{inputs_outputs["lmdb_root"]}`
- 输入 checkpoint: `{inputs_outputs["ckpt"]}`
- MAE 预训练权重: `{inputs_outputs["mae_pretrained_path"]}`
- 输出 CSV: `{inputs_outputs["output_csv"]}`
- 输出 JSONL: `{inputs_outputs["output_jsonl"]}`
- 输出报告: `{inputs_outputs["report"]}`

## 4. Model Loading

- 加载函数: `strhub.models.utils.load_from_checkpoint`
- checkpoint 类型: `maevit_infonce_plm`
- 图像尺寸: `{model_loading["img_size"]}`
- `max_label_length`: `{model_loading["max_label_length"]}`
- `charset_train` 长度: `{model_loading["charset_train_len"]}`
- `charset_test` 长度: `{model_loading["charset_test_len"]}`

## 5. Dataset Reading

- 数据集类: `ExportLmdbDataset(LmdbDataset)`
- 读取键: `image-%09d`, `label-%09d`, `meta-%09d`
- `lmdb_index`: 1-based
- 样本总数: `{dataset_reading["num_samples_total"]}`
- 实际处理样本数: `{dataset_reading["num_samples_processed"]}`
- `batch_size`: `{dataset_reading["batch_size"]}`
- `num_workers`: `{dataset_reading["num_workers"]}`

## 6. Decoding and Confidence Definition

- 图像预处理: `SceneTextDataModule.get_transform(model.hparams.img_size, augment=False)`
- 推理路径: `model(images) -> logits.softmax(-1) -> model.tokenizer.decode(probs)`
- `pred`: greedy decode 后的字符串
- `avg_conf`: `tokenizer.decode()` 返回的预测路径 token 概率序列均值
- `min_conf`: 同一概率序列最小值
- `confidence`: 当前版本直接等于 `avg_conf`
- `nll`: 当前定义为预测路径的 `-sum(log p)`，其中 `p` 为同一 greedy 路径上的 token 概率

## 7. Export Summary

- `num_samples_total`: `{export_summary["num_samples_total"]}`
- `num_samples_processed`: `{export_summary["num_samples_processed"]}`
- `num_correct`: `{export_summary["num_correct"]}`
- `accuracy`: `{export_summary["accuracy"]:.6f}`
- `output_csv`: `{export_summary["output_csv"]}`
- `output_jsonl`: `{export_summary["output_jsonl"]}`
- `runtime_seconds`: `{export_summary["runtime_seconds"]:.3f}`
- `device`: `{export_summary["device"]}`
- `batch_size`: `{export_summary["batch_size"]}`

## 8. Accuracy and Confidence Statistics

- `overall_accuracy`: `{overall_stats["accuracy"]:.6f}`
- `mean_confidence`: `{overall_stats["mean_confidence"]:.6f}`
- `median_confidence`: `{overall_stats["median_confidence"]:.6f}`
- `mean_avg_conf`: `{overall_stats["mean_avg_conf"]:.6f}`
- `mean_min_conf`: `{overall_stats["mean_min_conf"]:.6f}`
- `correct_mean_confidence`: `{overall_stats["correct_mean_confidence"]:.6f}`
- `wrong_mean_confidence`: `{overall_stats["wrong_mean_confidence"]:.6f}`

## 9. Quality / Structure Breakdown

### quality

{markdown_table(quality_rows, ["group", "count", "accuracy", "mean_confidence", "mean_avg_conf", "mean_min_conf"], headers={"group": "quality"})}

### structure

{markdown_table(structure_rows, ["group", "count", "accuracy", "mean_confidence", "mean_avg_conf", "mean_min_conf"], headers={"group": "structure"})}

### structure_type

{markdown_table(structure_type_rows, ["group", "count", "accuracy", "mean_confidence", "mean_avg_conf", "mean_min_conf"], headers={"group": "structure_type"})}

## 10. Sample Records

### 前 5 条样本

{markdown_table(sample_rows, ["lmdb_index", "label", "pred", "correct", "confidence", "quality", "structure", "source_path"])}

### 错误样本示例

{markdown_table(wrong_rows, ["lmdb_index", "label", "pred", "correct", "confidence", "quality", "structure", "source_path"])}

## 11. Warnings / Limitations

{"".join(f"- {warning}\n" for warning in warnings) if warnings else "- 无"}

## 12. Recommended Next Step

下一步建议基于 `train_ocr_predictions.csv` 和 `SLP34K_lmdb_train_meta` 构造 `train_samples_meta.csv/jsonl` 与 `same-label top1-HQ pair_manifest_top1_hq.csv`。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def remove_existing_outputs(paths: List[Path]):
    for path in paths:
        if path.exists():
            path.unlink()


@torch.inference_mode()
def run_export(args):
    lmdb_root, ckpt, output_csv, output_jsonl, report = validate_args(args)
    device, device_note = resolve_device(args.device)
    charset_test = load_charset_test()

    model = load_from_checkpoint(
        str(ckpt),
        mae_pretrained_path=str(PRETRAIN_MAE),
        charset_test=charset_test,
    ).eval().to(device)
    dataset = build_dataset(lmdb_root, model.hparams.charset_train, model.hparams.max_label_length, model.hparams.img_size)
    num_samples_total = len(dataset)
    dataset_for_run = maybe_limit_dataset(dataset, args.limit)
    num_samples_processed = len(dataset_for_run)
    dataloader = create_dataloader(dataset_for_run, args.batch_size, args.num_workers, device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_existing_outputs([output_csv, output_jsonl, report])

    confidences = []
    avg_confs = []
    min_confs = []
    correct_confs = []
    wrong_confs = []
    num_correct = 0
    first_samples = []
    wrong_samples = []
    grouped_quality = defaultdict(init_group_stats)
    grouped_structure = defaultdict(init_group_stats)
    grouped_structure_type = defaultdict(init_group_stats)

    start_time = time.perf_counter()
    with output_csv.open("w", encoding="utf-8", newline="") as csv_fp, output_jsonl.open("w", encoding="utf-8") as jsonl_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            probs = logits.softmax(-1)
            preds, seq_probs = model.tokenizer.decode(probs.detach().cpu())

            for row_idx, pred in enumerate(preds):
                record = record_from_batch(batch, row_idx, pred, seq_probs[row_idx])
                row = {key: record[key] for key in CSV_FIELDS}
                writer.writerow(row)
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                confidences.append(record["confidence"])
                avg_confs.append(record["avg_conf"])
                min_confs.append(record["min_conf"])
                if record["correct"]:
                    num_correct += 1
                    correct_confs.append(record["confidence"])
                else:
                    wrong_confs.append(record["confidence"])
                    if len(wrong_samples) < 5:
                        wrong_samples.append(record)

                if len(first_samples) < 5:
                    first_samples.append(record)

                update_group_stats(grouped_quality[record["quality"] or "null"], record)
                update_group_stats(grouped_structure[record["structure"] or "null"], record)
                update_group_stats(grouped_structure_type[record["structure_type"] or "null"], record)

    runtime_seconds = time.perf_counter() - start_time

    accuracy = num_correct / num_samples_processed if num_samples_processed else 0.0
    overall_stats = {
        "accuracy": accuracy,
        "mean_confidence": statistics.fmean(confidences) if confidences else 0.0,
        "median_confidence": statistics.median(confidences) if confidences else 0.0,
        "mean_avg_conf": statistics.fmean(avg_confs) if avg_confs else 0.0,
        "mean_min_conf": statistics.fmean(min_confs) if min_confs else 0.0,
        "correct_mean_confidence": statistics.fmean(correct_confs) if correct_confs else 0.0,
        "wrong_mean_confidence": statistics.fmean(wrong_confs) if wrong_confs else 0.0,
    }
    export_summary = {
        "num_samples_total": num_samples_total,
        "num_samples_processed": num_samples_processed,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "runtime_seconds": runtime_seconds,
        "device": str(device),
        "batch_size": args.batch_size,
    }
    environment = {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "pl_version": __import__("pytorch_lightning").__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_note": device_note,
    }
    inputs_outputs = {
        "lmdb_root": str(lmdb_root),
        "ckpt": str(ckpt),
        "mae_pretrained_path": str(PRETRAIN_MAE),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "report": str(report),
    }
    model_loading = {
        "img_size": model.hparams.img_size,
        "max_label_length": model.hparams.max_label_length,
        "charset_train_len": len(model.hparams.charset_train),
        "charset_test_len": len(charset_test),
    }
    dataset_reading = {
        "num_samples_total": num_samples_total,
        "num_samples_processed": num_samples_processed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    warnings = [
        "confidence 基于 greedy decode 路径上的 token 概率，不等于完整序列后验概率。",
        "nll 当前是预测路径上的 -sum(log p)，不是 GT teacher-forcing NLL。",
        "tokenizer.decode() 返回的概率序列会包含 EOS 概率，因此 pred_length 与概率序列长度不完全相同。",
    ]
    if args.limit is not None:
        warnings.append(f"本次运行使用了 --limit={args.limit}，结果不是全量统计。")
    if device_note:
        warnings.append(device_note)

    write_report(
        report,
        status="成功",
        environment=environment,
        inputs_outputs=inputs_outputs,
        model_loading=model_loading,
        dataset_reading=dataset_reading,
        export_summary=export_summary,
        overall_stats=overall_stats,
        quality_rows=finalize_group_stats(grouped_quality),
        structure_rows=finalize_group_stats(grouped_structure),
        structure_type_rows=finalize_group_stats(grouped_structure_type),
        first_samples=first_samples,
        wrong_samples=wrong_samples,
        warnings=warnings,
    )


def main():
    args = parse_args()
    try:
        run_export(args)
    except Exception as exc:
        report = args.report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Train OCR Confidence Export Report\n\n"
            "## 1. Scope and Isolation Statement\n\n"
            "本轮仅在 `experiments/dit_lq_hq_v1/` 下尝试创建脚本、manifest 和报告。\n"
            "未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。\n\n"
            "## 11. Warnings / Limitations\n\n"
            f"- 任务失败：{exc}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
