# SLP34K_v2 Train Metadata 复核报告

## 1. 范围与未修改声明

本轮为只读复核。未修改任何源码、数据文件、checkpoint、配置或 LMDB。

本轮唯一新增文件是：

- `/mnt/data/zyx/SLP34K_v2/train_metadata_recheck_report.md`

本轮目标：

- 直接验证 `ocr_training/data/train/SLP34K_lmdb_train` 是否实际包含 metadata
- 在 `train`、`val`、`test/unified_lmdb` 三类 LMDB 上做原始 key probing
- 结合 metadata 相关代码路径，判断已有 manifest 中的 `metadata: null` 应如何解释

## 2. 使用的 Python / Conda 环境

`conda env list` 中可见与本项目相关的环境包括：

- `slpr_ocr`
- `slk34k_rec`

本轮按你的要求，使用：

- `slpr_ocr`

实际 Python 可执行文件：

- `/mnt/data/zyx/miniconda3/envs/slpr_ocr/bin/python`

LMDB 依赖检查结果：

- `import lmdb` 在 `slpr_ocr` 环境中成功

补充说明：

- `README.md` 仍然记录了 `slk34k_rec`
- 但当前仓库中的脚本与历史报告也多次使用 `slpr_ocr`

## 3. 检查过的 LMDB 路径

本轮直接检查了以下 LMDB 根目录：

| Split / Subset | LMDB 路径 | 是否存在 |
| --- | --- | --- |
| train | `/mnt/data/zyx/SLP34K_v2/ocr_training/data/train/SLP34K_lmdb_train` | yes |
| val | `/mnt/data/zyx/SLP34K_v2/ocr_training/data/val/SLP34K_lmdb_test` | yes |
| test unified | `/mnt/data/zyx/SLP34K_v2/ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` | yes |

另外还补充检查了 benchmark 下这些子集是否存在 `meta-` 前缀：

- `IV_lmdb`
- `OOV_lmdb`
- `low_lmdb`
- `multi-lines_lmdb`
- `normal_lmdb`
- `single-line_lmdb`
- `vertical_lmdb`

这些 LMDB 目录内的文件形态统一为：

- `data.mdb`
- `lock.mdb`

## 4. 原始 Key Probe 结果

### 4.1 Train LMDB 直接探查

train LMDB：

- 路径：`ocr_training/data/train/SLP34K_lmdb_train`
- `num-samples = 27501`

前 50 个原始 key 中可以看到：

- `image-000000001`
- `image-000000002`
- ...

并且 `label-%09d` 可直接读取。

针对结构化 key 的探查结果：

| Key 家族 | 结果 |
| --- | --- |
| `label-%09d` | found |
| `image-%09d` | found |
| `meta-%09d` | not found |
| `metadata-%09d` | not found |
| `quality-%09d` | not found |
| `layout-%09d` | not found |
| `structure-%09d` | not found |
| `structure_type-%09d` | not found |
| `difficulty-%09d` | not found |
| `recognition_difficulty-%09d` | not found |
| `text_layout-%09d` | not found |

前缀级确认结果：

- `image-` 前缀存在
- `label-` 前缀存在
- `meta-` 前缀在整个 train LMDB 中不存在

### 4.2 Val LMDB 直接探查

val LMDB：

- 路径：`ocr_training/data/val/SLP34K_lmdb_test`
- `num-samples = 6884`

结构化 key 探查结果：

| Key 家族 | 结果 |
| --- | --- |
| `label-%09d` | found |
| `image-%09d` | found |
| `meta-%09d` | not found |
| `metadata-%09d` | not found |
| `quality-%09d` | not found |
| `layout-%09d` | not found |
| `structure-%09d` | not found |
| `structure_type-%09d` | not found |
| `difficulty-%09d` | not found |

前缀级确认：

- `meta-` 前缀在整个 val LMDB 中不存在

### 4.3 Unified benchmark LMDB 直接探查

unified benchmark LMDB：

- 路径：`ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb`
- `num-samples = 6884`

结构化 key 探查结果：

| Key 家族 | 结果 |
| --- | --- |
| `label-%09d` | found |
| `image-%09d` | found |
| `meta-%09d` | found |
| `metadata-%09d` | not found |
| `quality-%09d` | not found |
| `layout-%09d` | not found |
| `structure-%09d` | not found |
| `structure_type-%09d` | not found |
| `difficulty-%09d` | not found |

发现的第一个 metadata key：

- `meta-000000001`

metadata 值格式：

- JSON 字符串

观测到的 metadata 字段：

- `id`
- `quality`
- `structure`
- `vocabulary_type`
- `resolution_type`
- `structure_type`

样例：

```json
{"id": 1, "quality": "easy", "structure": "multi", "vocabulary_type": "IV", "resolution_type": "normal", "structure_type": "multi_lines"}
```

### 4.4 benchmark 其他子集的一致性检查

对 `data/test/SLP34K_lmdb_benchmark/` 下所有子集做了 `meta-` 前缀检查：

| Benchmark 子集 | 是否有 `meta-` 前缀 |
| --- | --- |
| `IV_lmdb` | no |
| `OOV_lmdb` | no |
| `low_lmdb` | no |
| `multi-lines_lmdb` | no |
| `normal_lmdb` | no |
| `single-line_lmdb` | no |
| `unified_lmdb` | yes |
| `vertical_lmdb` | no |

关键结论：

- 即使在 benchmark 目录内部，也不是所有子集 LMDB 都带 `meta-%09d`
- 当前本地数据树里，`meta-%09d` 只在 `unified_lmdb` 中明确存在

## 5. Train LMDB Metadata 结果

结论：**Train LMDB 不包含 metadata key。**

这是本轮直接验证得到的结果，不再是间接推断。

本轮做了两类验证：

- 用 `txn.get()` 直接探查 `meta-%09d` 及其等价 key
- 用 prefix search 检查整个 LMDB 中是否存在 `meta-`、`metadata-`、`quality-`、`layout-`、`structure-` 等前缀

结果表明：

- `ocr_training/data/train/SLP34K_lmdb_train` 中存在：
  - `image-%09d`
  - `label-%09d`
  - `num-samples`
- 不存在：
  - `meta-%09d`
  - `metadata-%09d`
  - `quality-%09d`
  - `layout-%09d`
  - `structure-%09d`
  - `structure_type-%09d`
  - `difficulty-%09d`

因此，对当前本地 train LMDB 而言：

- `quality`
- `layout`
- `structure`
- `structure_type`
- `difficulty`

都**不是**以逐样本 key 的方式存储在 train LMDB 中。

## 6. Val/Test Metadata 对比

| Split | LMDB Path | Has `meta-%09d`? | Has quality/layout-like fields? | Notes |
| --- | --- | ---: | ---: | --- |
| train | `ocr_training/data/train/SLP34K_lmdb_train` | no | no | 只看到 `image-*`、`label-*`、`num-samples` |
| val | `ocr_training/data/val/SLP34K_lmdb_test` | no | no | 与 train 一样 |
| test unified | `ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` | yes | yes | metadata 以 JSON 形式挂在 `meta-%09d` |

额外说明：

- benchmark 下的 `IV_lmdb`、`OOV_lmdb`、`low_lmdb`、`vertical_lmdb` 等子集，在当前本地数据副本中也不带 `meta-%09d`

因此 key 结构在以下对象之间**并不一致**：

- train
- val
- unified benchmark
- benchmark 其他子集

## 7. Metadata 相关代码路径

### 7.1 基础加载路径

基础 dataset reader：

- `ocr_training/strhub/data/dataset.py`

行为：

- 读取 `label-%09d`
- 读取 `image-%09d`
- 不尝试读取任何 metadata key
- `__getitem__()` 只返回 `(img, label)`

这意味着：

- 即便 LMDB 中存在 metadata，基础 OCR 训练链路目前也不会读取
- 而当前本地 train 数据副本里，train metadata 本身也不存在

### 7.2 metadata-aware 导出器

metadata-aware 读取路径：

- `ocr_training/tools/export_parseq_corrector_cache.py:34-58`

关键逻辑：

- 构造 `meta_key = f"meta-{lmdb_index:09d}".encode()`
- 读取 `meta_buf = txn.get(meta_key)`
- 若存在则 `json.loads(...)`
- 将 `meta` 放进 batch dict

manifest 写出路径：

- `ocr_training/tools/export_parseq_corrector_cache.py:227-262`

关键逻辑：

- 将 `"metadata": batch["meta"][row_idx]` 写入 manifest row

这说明：

- exporter 确实会读 `meta-%09d`
- manifest 中出现 `metadata: null`，不是因为 exporter 忘了读，而是因为 LMDB 里没读到

### 7.3 metadata 的消费者

metadata 在后续工具里会被消费：

- `ocr_training/tools/export_parseq_corrector_cache.py`
- `ocr_training/tools/filter_mdiff_corrector_cache.py`
- `ocr_training/tools/eval_mdiff_corrector_offline.py`

代码中实际使用过的 metadata 字段：

- `quality`
- `vocabulary_type`
- `resolution_type`
- `structure_type`

但基础 train/test 主流程并不会使用这些字段。

### 7.4 metadata writer / LMDB 生成脚本

本轮在仓库中搜索 LMDB 写入逻辑后发现：

- 找到了 `meta-%09d` 的读取逻辑
- 没有找到 in-repo 的 SLP34K LMDB 构建脚本去写 `meta-%09d`
- 也没有找到将 train annotation 转成带 metadata LMDB 的现成脚本

说明：

- 当前仓库保存了 reader 和后处理工具
- 但没有保存原始 SLP34K LMDB 构建流水线

### 7.5 原始 annotation 来源搜索

在本 repo 范围内，本轮没有找到明显的 train 原始 sidecar：

- 没有 train CSV
- 没有 train JSON
- 没有 train TXT annotation sidecar
- 没有 train JSONL annotation sidecar

这并不证明仓库外不存在额外 annotation，只能说明 repo 内没有找到。

## 8. 现有 Cache Manifest 的正确解释

之前观察到的 manifest 例如：

- `ocr_training/outputs/V2-M02u_corrector_cache_replace_only/manifest.jsonl`
- `ocr_training/outputs/V2-M02u_corrector_cache_replace_only_val/manifest.jsonl`

都包含：

- `"metadata": null`

本轮可以确认，这种现象应解释为：

1. exporter 确实会读取 `meta-%09d`
2. exporter 确实会把读到的内容写入 `metadata`
3. 因此 train / val manifest 中的 `metadata: null` 就是对 LMDB 真实内容的反映

也就是说：

- train manifest 的 `metadata: null` 不是工具缺陷造成的
- val manifest 的 `metadata: null` 也不是工具缺陷造成的
- unified manifest 中出现 `metadata: {...}` 与本轮直接 key probe 的结果完全一致

## 9. 样本示例

### 9.1 Train 样本

直接 LMDB probe 样例：

```text
sample[1] = {
  label: "浙萧山货23765",
  image: exists,
  metadata: missing,
  lmdb_index: 1
}

sample[2] = {
  label: "六安港LUANGANG",
  image: exists,
  metadata: missing,
  lmdb_index: 2
}

sample[3] = {
  label: "苏常州货068SUCHANGZHOUHUO",
  image: exists,
  metadata: missing,
  lmdb_index: 3
}
```

### 9.2 Val 样本

```text
sample[1] = {
  label: "浙富阳货00268",
  image: exists,
  metadata: missing,
  lmdb_index: 1
}

sample[2] = {
  label: "浙绍运集668ZHESHAOYUNJI",
  image: exists,
  metadata: missing,
  lmdb_index: 2
}
```

### 9.3 Unified benchmark 样本

```text
sample[1] = {
  label: "鲁菏泽货0846LUHEZEHUO",
  metadata: {
    id: 1,
    quality: "easy",
    structure: "multi",
    vocabulary_type: "IV",
    resolution_type: "normal",
    structure_type: "multi_lines"
  },
  quality: "easy",
  layout_or_structure: "structure=multi, structure_type=multi_lines",
  lmdb_index: 1
}

sample[2] = {
  label: "万顺668WANSHUN",
  metadata: {
    id: 2,
    quality: "easy",
    structure: "multi",
    vocabulary_type: "IV",
    resolution_type: "normal",
    structure_type: "multi_lines"
  },
  quality: "easy",
  layout_or_structure: "structure=multi, structure_type=multi_lines",
  lmdb_index: 2
}
```

## 10. 最终结论

**Case B：train LMDB 已确认不含 metadata。**

明确结论如下：

- `ocr_training/data/train/SLP34K_lmdb_train` **不包含** `meta-%09d`
- 也**不包含** `metadata-%09d`、`quality-%09d`、`layout-%09d`、`structure-%09d`、`structure_type-%09d`、`difficulty-%09d`
- 当前本地 train LMDB 仅包含基础样本内容：
  - `image-%09d`
  - `label-%09d`
  - `num-samples`

额外对比结果：

- `ocr_training/data/val/SLP34K_lmdb_test` 也没有 metadata key
- `ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` 确实有 `meta-%09d`
- benchmark 其他子集 LMDB 在当前本地副本中不含 `meta-%09d`

因此：

- base train split 只有 image 与 label
- 若后续 train pair stats 需要 `quality/layout/structure`，不能直接依赖当前 train LMDB
- 这些字段必须来自其他来源

## 11. 针对下一阶段 pair-stats 的建议

建议下一阶段默认把 train pair-stats 当成纯 image / label 驱动，而不是 metadata 驱动。

建议：

1. 以当前 train LMDB 为基础读取：
   - `label`
   - 用 `lmdb_root + lmdb_index` 做稳定样本定位
2. 不要假设 train 自带 `quality` / `layout`
3. 如果下一步确实强依赖 `quality/layout`，应先找到额外 annotation 源或另一份更完整的数据副本

对 pair stats 的直接影响：

- same-label grouping 可以直接从 train label 做
- top1-HQ pair 构造不能依赖当前 train LMDB 中的 metadata
- 若要定义 HQ，需要外部标准或其他启发式

## 12. 未解问题 / Blockers

1. 仓库外是否存在另一份原始 annotation、CSV/JSON 或替代 LMDB，里面包含 train metadata？

2. 原始 SLP34K 数据预处理流程中，是否曾经生成过另一份带 metadata 的 train LMDB，只是当前本地挂载的不是那一份？

3. 你所说的 train `quality/layout` 是否来自：
   - 论文描述
   - 另一份本地数据副本
   - 或 `unified_lmdb` 的 schema 记忆

4. 如果后续必须使用 `quality/layout`，是否需要在下一轮 discovery 中扩大搜索范围，到 repo 外寻找：
   - 其他 SLP34K 压缩包
   - 原始图片标注文件
   - 其他 LMDB 数据副本
