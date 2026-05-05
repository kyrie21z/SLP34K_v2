#!/usr/bin/env python3
import argparse
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import lmdb
from PIL import Image


ALLOWED_QUALITY = {"easy", "middle", "hard"}
ALLOWED_STRUCTURE = {"single", "multi", "vertical"}
STRUCTURE_TYPE_MAP = {
    "single": "single_line",
    "multi": "multi_lines",
    "vertical": "vertical",
}
EXPERIMENT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1").resolve()
SAFE_OUTPUT_ROOT = (EXPERIMENT_ROOT / "data").resolve()
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/data/zyx/SLP34K_v2/ocr_training/data").resolve()
DEFAULT_MAP_SIZE = 64 * 1024 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="构建带 metadata 的 SLP34K train LMDB")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    parser.add_argument("--output-lmdb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commit-interval", type=int, default=1000)
    return parser.parse_args()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_counter(counter: Counter):
    if not counter:
        return "无"
    lines = ["| 值 | 数量 |", "| --- | ---: |"]
    for key, value in counter.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def parse_filename_parts(relative_path: str):
    filename = Path(relative_path).name
    parts = filename.split("&", 7)
    if len(parts) != 8:
        raise ValueError(f"文件名字段数不是 8 段: {relative_path}")
    quality, structure, flag3, flag4, path_label, num_a_str, num_b_str, capture_filename = parts
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"未知 quality: {quality} ({relative_path})")
    if structure not in ALLOWED_STRUCTURE:
        raise ValueError(f"未知 structure: {structure} ({relative_path})")
    try:
        num_a = int(num_a_str)
        num_b = int(num_b_str)
    except ValueError as exc:
        raise ValueError(f"num_a/num_b 不是整数: {relative_path}") from exc
    return {
        "quality": quality,
        "structure": structure,
        "structure_type": STRUCTURE_TYPE_MAP[structure],
        "raw_flag3": flag3,
        "raw_flag4": flag4,
        "path_label": path_label,
        "raw_num_a": num_a,
        "raw_num_b": num_b,
        "capture_filename": capture_filename,
    }


def validate_paths(args):
    raw_root = args.raw_root.resolve()
    gt_file = args.gt_file.resolve()
    output_lmdb = args.output_lmdb.resolve()
    report = args.report.resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root 不存在: {raw_root}")
    if not gt_file.exists():
        raise FileNotFoundError(f"gt_file 不存在: {gt_file}")
    if not is_relative_to(output_lmdb, SAFE_OUTPUT_ROOT):
        raise ValueError(f"output_lmdb 必须位于 {SAFE_OUTPUT_ROOT} 下: {output_lmdb}")
    if is_relative_to(output_lmdb, FORBIDDEN_OUTPUT_ROOT):
        raise ValueError(f"output_lmdb 不能位于 {FORBIDDEN_OUTPUT_ROOT} 下: {output_lmdb}")
    if not is_relative_to(report, EXPERIMENT_ROOT):
        raise ValueError(f"report 必须位于 {EXPERIMENT_ROOT} 下: {report}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")
    if args.commit_interval <= 0:
        raise ValueError("--commit-interval 必须大于 0")
    if output_lmdb.exists() and not args.overwrite:
        raise FileExistsError(f"输出 LMDB 已存在，未提供 --overwrite: {output_lmdb}")
    return raw_root, gt_file, output_lmdb, report


def load_records(raw_root: Path, gt_file: Path, limit: int | None):
    records = []
    total_input_lines = 0
    quality_counter = Counter()
    structure_counter = Counter()
    structure_type_counter = Counter()
    stats = {
        "missing_images": 0,
        "label_mismatches": 0,
        "unknown_quality": 0,
        "unknown_structure": 0,
    }

    with gt_file.open("r", encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            total_input_lines += 1
            if limit is not None and len(records) >= limit:
                continue
            line = raw_line.rstrip("\n")
            if not line:
                raise ValueError(f"第 {line_no} 行为空")
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"第 {line_no} 行不是 <path>\\t<label> 格式: {line}")
            relative_path, label = parts
            image_path = (raw_root / relative_path).resolve()
            parsed = parse_filename_parts(relative_path)
            if parsed["path_label"] != label:
                stats["label_mismatches"] += 1
                raise ValueError(
                    f"第 {line_no} 行 path_label 与 label 不一致: "
                    f"{parsed['path_label']} != {label}"
                )
            if not image_path.exists():
                stats["missing_images"] += 1
                raise FileNotFoundError(f"第 {line_no} 行引用的图片不存在: {image_path}")
            quality_counter[parsed["quality"]] += 1
            structure_counter[parsed["structure"]] += 1
            structure_type_counter[parsed["structure_type"]] += 1
            records.append(
                {
                    "line_no": line_no,
                    "relative_path": relative_path,
                    "image_path": image_path,
                    "label": label,
                    **parsed,
                }
            )
    return records, total_input_lines, quality_counter, structure_counter, structure_type_counter, stats


def build_meta(record, index: int):
    return {
        "id": index,
        "quality": record["quality"],
        "structure": record["structure"],
        "structure_type": record["structure_type"],
        "vocabulary_type": None,
        "resolution_type": None,
        "source_path": record["relative_path"],
        "raw_label": record["label"],
        "split": "train",
        "raw_flag3": record["raw_flag3"],
        "raw_flag4": record["raw_flag4"],
        "raw_num_a": record["raw_num_a"],
        "raw_num_b": record["raw_num_b"],
        "capture_filename": record["capture_filename"],
    }


def remove_existing_output(output_lmdb: Path):
    if output_lmdb.exists():
        shutil.rmtree(output_lmdb)


def flush_cache(env, cache):
    if not cache:
        return
    with env.begin(write=True) as txn:
        for key, value in cache.items():
            txn.put(key, value)
    cache.clear()


def write_lmdb(records, output_lmdb: Path, commit_interval: int):
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_lmdb), map_size=DEFAULT_MAP_SIZE)
    cache = {}
    try:
        for index, record in enumerate(records, start=1):
            with record["image_path"].open("rb") as fp:
                image_bytes = fp.read()
            meta_bytes = json.dumps(build_meta(record, index), ensure_ascii=False).encode("utf-8")
            cache[f"image-{index:09d}".encode("utf-8")] = image_bytes
            cache[f"label-{index:09d}".encode("utf-8")] = record["label"].encode("utf-8")
            cache[f"meta-{index:09d}".encode("utf-8")] = meta_bytes
            if index % commit_interval == 0:
                flush_cache(env, cache)
        cache[b"num-samples"] = str(len(records)).encode("utf-8")
        flush_cache(env, cache)
    finally:
        env.close()


def validate_lmdb(records, raw_root: Path, output_lmdb: Path):
    env = lmdb.open(str(output_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
    validation = {
        "num_samples_readback": None,
        "first20_labels_match": True,
        "sampled_indexes": [],
        "sampled_records": [],
    }
    try:
        with env.begin(write=False) as txn:
            num_samples_raw = txn.get(b"num-samples")
            if num_samples_raw is None:
                raise ValueError("回读时缺少 num-samples")
            num_samples = int(num_samples_raw.decode("utf-8"))
            validation["num_samples_readback"] = num_samples
            if num_samples != len(records):
                raise ValueError(f"回读 num-samples={num_samples} 与写入数量 {len(records)} 不一致")

            for index, record in enumerate(records[:20], start=1):
                label = txn.get(f"label-{index:09d}".encode("utf-8"))
                if label is None or label.decode("utf-8") != record["label"]:
                    validation["first20_labels_match"] = False
                    raise ValueError(f"前 20 条 label 回读校验失败，索引 {index}")

            sample_count = min(10, len(records))
            rng = random.Random(20260504)
            sample_indexes = (
                list(range(1, sample_count + 1))
                if len(records) <= sample_count
                else sorted(rng.sample(range(1, len(records) + 1), sample_count))
            )
            validation["sampled_indexes"] = sample_indexes

            for index in sample_indexes:
                image_key = f"image-{index:09d}".encode("utf-8")
                label_key = f"label-{index:09d}".encode("utf-8")
                meta_key = f"meta-{index:09d}".encode("utf-8")
                image_bytes = txn.get(image_key)
                label_bytes = txn.get(label_key)
                meta_bytes = txn.get(meta_key)
                if not image_bytes or not label_bytes or not meta_bytes:
                    raise ValueError(f"采样索引 {index} 缺少 image/label/meta 键")

                meta = json.loads(meta_bytes.decode("utf-8"))
                for field in ["quality", "structure", "structure_type", "source_path", "raw_label"]:
                    if not meta.get(field):
                        raise ValueError(f"采样索引 {index} 的 metadata 缺少字段 {field}")

                source_path = (raw_root / meta["source_path"]).resolve()
                if not source_path.exists():
                    raise ValueError(f"采样索引 {index} 的 source_path 不存在: {source_path}")

                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.verify()

                validation["sampled_records"].append(
                    {
                        "lmdb_index": index,
                        "label": label_bytes.decode("utf-8"),
                        "metadata": meta,
                    }
                )
    finally:
        env.close()
    return validation


def make_report(args, raw_root, gt_file, output_lmdb, report, records, total_input_lines,
                quality_counter, structure_counter, structure_type_counter, stats,
                validation=None, error=None):
    sample_records = []
    for index, record in enumerate(records[:3], start=1):
        sample_records.append(
            {
                "lmdb_index": index,
                "label": record["label"],
                "metadata": build_meta(record, index),
            }
        )

    status_line = "构建成功。" if error is None else f"构建失败：{error}"
    output_files = []
    if output_lmdb.exists():
        output_files = sorted(p.name for p in output_lmdb.iterdir())

    report_text = f"""# Train Metadata-rich LMDB Build Report

## 1. Scope and Isolation Statement

本次任务仅在 `experiments/dit_lq_hq_v1/` 目录下创建或覆盖文件。
未修改现有 `SLP34K/` 原始数据、`ocr_training/data/` 既有 LMDB、`ocr_training/configs/`、`ocr_training/strhub/`、`ocr_training/outputs/`、`ocr_training/checkpoint/`、根目录 `reports/` 或根目录 `outputs/`。

## 2. Environment

- Python: `/mnt/data/zyx/miniconda3/envs/slpr_ocr/bin/python`
- `lmdb`: 可用
- `PIL`: 可用

## 3. Inputs

- 原始数据根目录: `{raw_root}`
- 标注文件: `{gt_file}`
- 标注总行数: `{total_input_lines}`
- 实际处理样本数: `{len(records)}`
- `--limit`: `{args.limit}`
- `--commit-interval`: `{args.commit_interval}`

## 4. Output LMDB

- 输出路径: `{output_lmdb}`
- 输出目录存在: `{"yes" if output_lmdb.exists() else "no"}`
- 输出文件: `{", ".join(output_files) if output_files else "无"}`

## 5. Metadata Schema

每个 `meta-%09d` 写入 UTF-8 JSON，字段如下：

```json
{safe_json({
    "id": 1,
    "quality": "middle",
    "structure": "single",
    "structure_type": "single_line",
    "vocabulary_type": None,
    "resolution_type": None,
    "source_path": "train/xxx.jpg",
    "raw_label": "示例车牌",
    "split": "train",
    "raw_flag3": "ng",
    "raw_flag4": "nd",
    "raw_num_a": 2,
    "raw_num_b": 0,
    "capture_filename": "T_xxx.jpg"
})}
```

## 6. Build Procedure

1. 校验 `raw_root`、`gt_file`、`output_lmdb`、`report` 的安全边界。
2. 读取 `train_gt.txt`，解析 `<relative_image_path>\\t<label>`。
3. 从文件名中解析 `quality`、`structure`、`flag3`、`flag4`、`path_label`、`num_a`、`num_b`、`capture_filename`。
4. 校验：
   - `quality` 属于 `easy/middle/hard`
   - `structure` 属于 `single/multi/vertical`
   - `path_label == label`
   - 图片文件存在
5. 写入 LMDB 键：
   - `image-%09d`
   - `label-%09d`
   - `meta-%09d`
   - `num-samples`
6. 回读验证 `num-samples`、前 20 条 label、一组随机采样样本和 JSON metadata。

## 7. Build Summary

- 状态: {status_line}
- 写入样本数: `{len(records) if error is None else 0}`
- 缺失图片数: `{stats["missing_images"]}`
- label/path_label 不一致数: `{stats["label_mismatches"]}`
- 未知 quality 数: `{stats["unknown_quality"]}`
- 未知 structure 数: `{stats["unknown_structure"]}`

## 8. Validation Results

- `num-samples` 回读值: `{validation["num_samples_readback"] if validation else "未执行"}`
- 前 20 条 label 回读一致: `{validation["first20_labels_match"] if validation else "未执行"}`
- 采样索引: `{validation["sampled_indexes"] if validation else "未执行"}`
- 采样样本数: `{len(validation["sampled_records"]) if validation else 0}`

## 9. Quality / Structure Distribution

### quality

{format_counter(quality_counter)}

### structure

{format_counter(structure_counter)}

### structure_type

{format_counter(structure_type_counter)}

## 10. Sample Records

```json
{safe_json(sample_records)}
```

## 11. Compatibility Notes

- 新 LMDB 键格式与现有 OCR LMDB 约定兼容：`num-samples`、`image-%09d`、`label-%09d`、`meta-%09d`。
- 现有基础 `LmdbDataset` 仍只读取 `image-%09d` 与 `label-%09d`；若要消费 metadata，需要使用显式读取 `meta-%09d` 的数据集实现。
- `structure_type` 按以下规则映射：
  - `single -> single_line`
  - `multi -> multi_lines`
  - `vertical -> vertical`

## 12. Warnings / Limitations

- `vocabulary_type` 统一写为 `null`，因为当前尚未确认 `raw_flag3` 的真实语义。
- `resolution_type` 统一写为 `null`，因为当前尚未确认 `raw_flag4` 的真实语义。
- `quality` 包含 `middle`，并非只有 `easy/hard` 二分类。
- 若后续需要与 `unified_lmdb` 完全对齐，还需要补清 `vocabulary_type` 与 `resolution_type` 的映射来源。

## 13. Recommended Next Step

下一步建议基于 `SLP34K_lmdb_train_meta` 生成 train sample manifest，并进一步构造 same-label top1-HQ pair manifest。
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text, encoding="utf-8")


def main():
    args = parse_args()
    records = []
    total_input_lines = 0
    quality_counter = Counter()
    structure_counter = Counter()
    structure_type_counter = Counter()
    stats = {
        "missing_images": 0,
        "label_mismatches": 0,
        "unknown_quality": 0,
        "unknown_structure": 0,
    }
    validation = None
    raw_root = gt_file = output_lmdb = report = None

    try:
        raw_root, gt_file, output_lmdb, report = validate_paths(args)
        records, total_input_lines, quality_counter, structure_counter, structure_type_counter, stats = load_records(
            raw_root, gt_file, args.limit
        )
        if args.overwrite:
            remove_existing_output(output_lmdb)
        write_lmdb(records, output_lmdb, args.commit_interval)
        validation = validate_lmdb(records, raw_root, output_lmdb)

        expected_quality = Counter({"middle": 17206, "easy": 6233, "hard": 4062})
        expected_structure = Counter({"multi": 15384, "single": 8112, "vertical": 4005})
        if args.limit is None:
            if quality_counter != expected_quality:
                raise ValueError(f"quality 分布与预期不一致: {quality_counter} != {expected_quality}")
            if structure_counter != expected_structure:
                raise ValueError(f"structure 分布与预期不一致: {structure_counter} != {expected_structure}")

        make_report(
            args,
            raw_root,
            gt_file,
            output_lmdb,
            report,
            records,
            total_input_lines,
            quality_counter,
            structure_counter,
            structure_type_counter,
            stats,
            validation=validation,
            error=None,
        )
    except Exception as exc:
        error = str(exc)
        if report is None:
            report = args.report.resolve()
        if raw_root is None:
            raw_root = args.raw_root.resolve()
        if gt_file is None:
            gt_file = args.gt_file.resolve()
        if output_lmdb is None:
            output_lmdb = args.output_lmdb.resolve()
        make_report(
            args,
            raw_root,
            gt_file,
            output_lmdb,
            report,
            records,
            total_input_lines,
            quality_counter,
            structure_counter,
            structure_type_counter,
            stats,
            validation=validation,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
