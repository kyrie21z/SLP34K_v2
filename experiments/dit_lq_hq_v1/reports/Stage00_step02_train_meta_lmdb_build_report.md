# Train Metadata-rich LMDB Build Report

## 1. Scope and Isolation Statement

本次任务仅在 `experiments/dit_lq_hq_v1/` 目录下创建或覆盖文件。
未修改现有 `SLP34K/` 原始数据、`ocr_training/data/` 既有 LMDB、`ocr_training/configs/`、`ocr_training/strhub/`、`ocr_training/outputs/`、`ocr_training/checkpoint/`、根目录 `reports/` 或根目录 `outputs/`。

## 2. Environment

- Python: `/mnt/data/zyx/miniconda3/envs/slpr_ocr/bin/python`
- `lmdb`: 可用
- `PIL`: 可用

## 3. Inputs

- 原始数据根目录: `/mnt/data/zyx/SLP34K_v2/SLP34K`
- 标注文件: `/mnt/data/zyx/SLP34K_v2/SLP34K/train_gt.txt`
- 标注总行数: `27501`
- 实际处理样本数: `27501`
- `--limit`: `None`
- `--commit-interval`: `1000`

## 4. Output LMDB

- 输出路径: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta`
- 输出目录存在: `yes`
- 输出文件: `data.mdb, lock.mdb`

## 5. Metadata Schema

每个 `meta-%09d` 写入 UTF-8 JSON，字段如下：

```json
{
  "id": 1,
  "quality": "middle",
  "structure": "single",
  "structure_type": "single_line",
  "vocabulary_type": null,
  "resolution_type": null,
  "source_path": "train/xxx.jpg",
  "raw_label": "示例车牌",
  "split": "train",
  "raw_flag3": "ng",
  "raw_flag4": "nd",
  "raw_num_a": 2,
  "raw_num_b": 0,
  "capture_filename": "T_xxx.jpg"
}
```

## 6. Build Procedure

1. 校验 `raw_root`、`gt_file`、`output_lmdb`、`report` 的安全边界。
2. 读取 `train_gt.txt`，解析 `<relative_image_path>\t<label>`。
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

- 状态: 构建成功。
- 写入样本数: `27501`
- 缺失图片数: `0`
- label/path_label 不一致数: `0`
- 未知 quality 数: `0`
- 未知 structure 数: `0`

## 8. Validation Results

- `num-samples` 回读值: `27501`
- 前 20 条 label 回读一致: `True`
- 采样索引: `[3436, 6952, 7031, 7682, 7714, 11256, 11775, 11924, 18901, 21409]`
- 采样样本数: `10`

## 9. Quality / Structure Distribution

### quality

| 值 | 数量 |
| --- | ---: |
| middle | 17206 |
| hard | 4062 |
| easy | 6233 |

### structure

| 值 | 数量 |
| --- | ---: |
| single | 8112 |
| vertical | 4005 |
| multi | 15384 |

### structure_type

| 值 | 数量 |
| --- | ---: |
| single_line | 8112 |
| vertical | 4005 |
| multi_lines | 15384 |

## 10. Sample Records

```json
[
  {
    "lmdb_index": 1,
    "label": "浙萧山货23765",
    "metadata": {
      "id": 1,
      "quality": "middle",
      "structure": "single",
      "structure_type": "single_line",
      "vocabulary_type": null,
      "resolution_type": null,
      "source_path": "train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg",
      "raw_label": "浙萧山货23765",
      "split": "train",
      "raw_flag3": "ng",
      "raw_flag4": "nd",
      "raw_num_a": 2,
      "raw_num_b": 0,
      "capture_filename": "T_20220125_07_20_05_935375.jpg"
    }
  },
  {
    "lmdb_index": 2,
    "label": "六安港LUANGANG",
    "metadata": {
      "id": 2,
      "quality": "middle",
      "structure": "vertical",
      "structure_type": "vertical",
      "vocabulary_type": null,
      "resolution_type": null,
      "source_path": "train/middle&vertical&ng&nd&六安港LUANGANG&2&2&T_20220626_15_49_06_292625.jpg",
      "raw_label": "六安港LUANGANG",
      "split": "train",
      "raw_flag3": "ng",
      "raw_flag4": "nd",
      "raw_num_a": 2,
      "raw_num_b": 2,
      "capture_filename": "T_20220626_15_49_06_292625.jpg"
    }
  },
  {
    "lmdb_index": 3,
    "label": "苏常州货068SUCHANGZHOUHUO",
    "metadata": {
      "id": 3,
      "quality": "hard",
      "structure": "multi",
      "structure_type": "multi_lines",
      "vocabulary_type": null,
      "resolution_type": null,
      "source_path": "train/hard&multi&ng&nd&苏常州货068SUCHANGZHOUHUO&4&2&T_20220312_10_19_05_545125.jpg",
      "raw_label": "苏常州货068SUCHANGZHOUHUO",
      "split": "train",
      "raw_flag3": "ng",
      "raw_flag4": "nd",
      "raw_num_a": 4,
      "raw_num_b": 2,
      "capture_filename": "T_20220312_10_19_05_545125.jpg"
    }
  }
]
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
