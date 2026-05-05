# 原始带标注 SLP34K 数据集 Schema 报告

## 1. 范围与未修改声明

本轮为只读调查。未修改任何源码、已有数据、checkpoint、配置或 LMDB。
本轮只在 `experiments/dit_lq_hq_v1/reports/` 下创建了报告文件。

本轮目标：

- 调查 `/mnt/data/zyx/SLP34K_v2/SLP34K/` 下原始带标注数据集的结构
- 确认图像位置、标注文件、split 表示方式与 metadata 字段
- 判断其与当前 OCR train/val LMDB 的兼容性
- 不执行 LMDB 重建、不生成 manifest、不做 pair 构造

## 2. 原始数据集位置

原始数据集根目录：

- `/mnt/data/zyx/SLP34K_v2/SLP34K/`

该目录下当前可见内容：

- `train/`
- `test/`
- `train_gt.txt`
- `test_gt.txt`

在 `SLP34K/` 内部未找到额外的 README、schema 文档或数据说明文件。

## 3. 原始数据顶层结构

观测到的目录结构：

| 路径 | 用途判断 |
| --- | --- |
| `SLP34K/train/` | 原始训练图像目录 |
| `SLP34K/test/` | 原始测试图像目录 |
| `SLP34K/train_gt.txt` | train 标注索引文件 |
| `SLP34K/test_gt.txt` | test 标注索引文件 |

顶层结构结论：

- `SLP34K/` 非常扁平
- 没有 `val/` 目录
- 没有独立 `annotations/` 目录
- 数据组织方式是“图像目录 + 一个 txt 索引文件”

因此当前原始 split 的表达方式非常直接：

- `train_gt.txt` + `train/`
- `test_gt.txt` + `test/`

## 4. 图像文件

图像数量：

| Split 目录 | JPG 数量 |
| --- | ---: |
| `SLP34K/train/` | 27501 |
| `SLP34K/test/` | 6884 |
| total | 34385 |

这与标注文件行数完全一致：

- `wc -l SLP34K/train_gt.txt = 27501`
- `wc -l SLP34K/test_gt.txt = 6884`

图像文件命名规则：

每张图的文件名都遵循 8 段结构：

```text
<quality>&<structure>&<flag3>&<flag4>&<path_label>&<num_a>&<num_b>&<capture_filename>.jpg
```

示例：

```text
middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg
```

图像路径的重要性质：

- 相对图像路径本身就存储在标注文件第一列
- 文件名内部已经编码了多个 metadata 字段
- 文件名中的 `path_label` 与标注第二列 `label` 在本轮抽样和全量对比中保持一致

图像路径与标注的对应关系：

- 可以直接对应
- 标注文件第一列就是原始相对图像路径

## 5. 标注文件

在 `SLP34K/` 下只找到两个标注类文件：

| 文件 | 类型 | 大小 | 行数 | 推测作用 |
| --- | --- | ---: | ---: | --- |
| `SLP34K/train_gt.txt` | 制表符分隔文本 | 2,887,512 bytes | 27,501 | train 标注索引 |
| `SLP34K/test_gt.txt` | 制表符分隔文本 | 714,634 bytes | 6,884 | test 标注索引 |

在 `SLP34K/` 下未发现这些常见标注格式：

- `.csv`
- `.json`
- `.jsonl`
- `.tsv`
- `.xml`
- `.yaml`

每行的格式为：

```text
<relative_image_path>\t<label>
```

示例：

```text
train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg	浙萧山货23765
```

## 6. 标注字段 Schema

### 从 TXT 行中直接可恢复的 schema

每一行直接提供：

1. 相对图像路径
2. label

相对图像路径本身又能拆成：

```text
<split_dir>/<quality>&<structure>&<flag3>&<flag4>&<path_label>&<num_a>&<num_b>&<capture_filename>.jpg
```

### 字段映射表

| Concept | Raw field | Available? | Example | Notes |
| --- | --- | ---: | --- | --- |
| image path | 第 1 列 | yes | `train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg` | 相对 `SLP34K/` 的路径 |
| label | 第 2 列 | yes | `浙萧山货23765` | 明确的 ground-truth label |
| quality | 文件名第 1 段 | yes | `middle` | 可观测取值：`easy` / `middle` / `hard` |
| layout / structure | 文件名第 2 段 | yes | `single` | 可观测取值：`single` / `multi` / `vertical` |
| split | 路径前缀与标注文件归属 | yes | `train` | 来自 `train/...` 或 `test/...` |
| sample id | 无单独字段 | partial | 行号或相对路径 | 没有独立数值 id 列 |
| bbox / crop info | 未显式给出 | no / unclear | n/a | 未发现显式 bbox 字段 |

### 文件名中的其他字段

原始图像文件名里还包含 4 个暂时无法可靠解释的字段：

| 原始字段 | 示例 | 当前状态 |
| --- | --- | --- |
| `flag3` | `g` / `ng` | 存在，但语义尚未可靠确认 |
| `flag4` | `d` / `nd` | 存在，但语义尚未可靠确认 |
| `num_a` | `2`、`-5`、`20` | 整数型，语义未确认 |
| `num_b` | `0`、`1`、`5` | 整数型，语义未确认 |

观测范围：

- `num_a`：train `[-41, 44]`，test `[-36, 45]`
- `num_b`：train `[0, 12]`，test `[0, 11]`

关键一致性验证：

- `path_label_vs_label_mismatches = 0`

也就是说：

- 文件名中的 `path_label` 与显式标注列 `label` 在当前数据副本中完全一致

## 7. Train/Test/Val Split 表示方式

原始数据中的 split 表示方式如下：

- train split 由以下两部分直接表示：
  - `SLP34K/train/`
  - `SLP34K/train_gt.txt`
- test split 由以下两部分直接表示：
  - `SLP34K/test/`
  - `SLP34K/test_gt.txt`
- 未发现独立 raw val split

样本数：

| Raw split | 样本数 |
| --- | ---: |
| train | 27501 |
| test | 6884 |

与当前 OCR LMDB 的对应关系：

| 当前 LMDB | 样本数 | 对应的 raw 来源 |
| --- | ---: | --- |
| `ocr_training/data/train/SLP34K_lmdb_train` | 27501 | raw `train` |
| `ocr_training/data/val/SLP34K_lmdb_test` | 6884 | raw `test` |
| `ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` | 6884 | raw `test` 的另一种重编码结果 |

本轮完成的验证：

- raw `train_gt.txt` 前 20 条 label 与 `SLP34K_lmdb_train` 前 20 条 label 完全一致
- raw `test_gt.txt` 前 20 条 label 与 `SLP34K_lmdb_test` 前 20 条 label 完全一致

因此可得：

- 原始数据直接包含 train split
- 原始数据直接包含 test split
- 原始数据不直接包含 val split
- 当前 OCR 的 `val` LMDB 实际上很可能是由 raw `test` 构建的

## 8. Quality 与 Layout Metadata 可用性

### train 中是否可用

可用。

raw train 中可以直接从文件名解析出：

- `quality`
- `structure`

train 统计：

- `quality`: `middle=17206`, `easy=6233`, `hard=4062`
- `structure`: `multi=15384`, `single=8112`, `vertical=4005`

### test 中是否可用

同样可用。

raw test 中可以直接解析出：

- `quality`
- `structure`

test 统计：

- `quality`: `middle=4303`, `easy=1572`, `hard=1009`
- `structure`: `multi=3847`, `single=2042`, `vertical=995`

### 原始字段名

在 raw 文件名中可直接确认的字段名 / 概念：

- `quality`
- `structure`

建议映射到 unified 风格时使用：

- `structure=single` -> `structure_type=single_line`
- `structure=multi` -> `structure_type=multi_lines`
- `structure=vertical` -> `structure_type=vertical`

### 哪些是明确的，哪些仍不明确

明确的：

- `quality` 字段存在，且 train 可直接使用
- `structure` 字段存在，且 train 可直接使用

仍不明确的：

- `flag3` 是否映射为 `vocabulary_type`
- `flag4` 是否映射为 `resolution_type`
- `num_a` 与 `num_b` 的语义

这意味着：

- raw 数据足以支持构建“带 `quality` / `structure` 的 metadata-rich train LMDB”
- 但还不足以无歧义地恢复 unified test 那套完整 metadata（尤其是 `vocabulary_type` 与 `resolution_type`）

## 9. 样本记录示例

### Train 示例

```text
sample[1] = {
  image_path: "train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg",
  label: "浙萧山货23765",
  quality: "middle",
  structure: "single",
  flag3: "ng",
  flag4: "nd",
  num_a: 2,
  num_b: 0,
  split: "train",
  sample_id: null
}

sample[2] = {
  image_path: "train/middle&vertical&ng&nd&六安港LUANGANG&2&2&T_20220626_15_49_06_292625.jpg",
  label: "六安港LUANGANG",
  quality: "middle",
  structure: "vertical",
  flag3: "ng",
  flag4: "nd",
  num_a: 2,
  num_b: 2,
  split: "train",
  sample_id: null
}

sample[3] = {
  image_path: "train/hard&multi&ng&nd&苏常州货068SUCHANGZHOUHUO&4&2&T_20220312_10_19_05_545125.jpg",
  label: "苏常州货068SUCHANGZHOUHUO",
  quality: "hard",
  structure: "multi",
  flag3: "ng",
  flag4: "nd",
  num_a: 4,
  num_b: 2,
  split: "train",
  sample_id: null
}
```

### Test 示例

```text
sample[1] = {
  image_path: "test/middle&single&ng&nd&浙富阳货00268&20&3&O_20180721_16_08_06_890625.jpg",
  label: "浙富阳货00268",
  quality: "middle",
  structure: "single",
  flag3: "ng",
  flag4: "nd",
  num_a: 20,
  num_b: 3,
  split: "test",
  sample_id: null
}

sample[2] = {
  image_path: "test/middle&multi&ng&nd&浙绍运集668ZHESHAOYUNJI&4&2&T_20220212_12_33_47_057250.jpg",
  label: "浙绍运集668ZHESHAOYUNJI",
  quality: "middle",
  structure: "multi",
  flag3: "ng",
  flag4: "nd",
  num_a: 4,
  num_b: 2,
  split: "test",
  sample_id: null
}
```

## 10. 与当前 OCR Train LMDB 的兼容性

### 与当前 train LMDB 的对应性

支持对应的证据：

- raw `train_gt.txt` 行数为 27,501
- raw `train/` JPG 数量为 27,501
- 当前 `SLP34K_lmdb_train` 的 `num-samples = 27501`
- 前 20 条 raw train label 与 LMDB label 完全一致

这强烈说明：

- 当前 OCR train LMDB 很可能直接由 raw `train_gt.txt` 与 `train/` 构建
- 至少在前缀样本上，顺序未被打乱

### 与当前 val LMDB 的对应性

支持对应的证据：

- raw `test_gt.txt` 行数为 6,884
- raw `test/` JPG 数量为 6,884
- 当前 `SLP34K_lmdb_test` 的 `num-samples = 6884`
- 前 20 条 raw test label 与 val LMDB label 完全一致

这强烈说明：

- 当前 OCR val LMDB 是由 raw `test_gt.txt` 构建的

### 关于 unified test LMDB

`ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` 同样有 6,884 条样本，但：

- 按 index 的样本顺序与 raw `test_gt.txt` 不一致
- 但其 metadata 分布与 raw test 的 `quality` / `structure` 统计完全对齐

这说明：

- unified benchmark metadata 很可能来自同一份 raw test schema
- 但 unified benchmark LMDB 构建时采用了不同顺序，并补充了更多派生 metadata

### 是否能据此重建 metadata-rich train LMDB

结论：**partial yes**

原因：

- `image bytes`: yes
- `label`: yes
- `metadata`: yes，至少可以拿到 `quality`、`structure`、`source_path` 与原始附加字段
- `train split`: yes

限制：

- 若要求与 unified test schema 完全等价，目前还不能保证，因为 `vocabulary_type` 与 `resolution_type` 尚无可靠映射

## 11. 建议的 `meta-%09d` JSON Schema

### 当前 raw train 可直接支持的 schema

建议的最小可用 train metadata JSON：

```json
{
  "id": 1,
  "quality": "middle",
  "structure": "single",
  "structure_type": "single_line",
  "source_path": "train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg",
  "raw_label": "浙萧山货23765",
  "split": "train",
  "raw_flag3": "ng",
  "raw_flag4": "nd",
  "raw_num_a": 2,
  "raw_num_b": 0
}
```

### 若希望尽量贴近 unified test schema

更保守的推荐写法：

```json
{
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
  "raw_num_b": 0
}
```

### 字段可用性建议

当前可以直接获取：

- `id`
  - 若原始数据没有显式 id，可用顺序行号
- `quality`
- `structure`
- `structure_type`
- `source_path`
- `raw_label`
- `split`
- `raw_flag3`
- `raw_flag4`
- `raw_num_a`
- `raw_num_b`

仍需后续确认或映射：

- `vocabulary_type`
- `resolution_type`

在映射未确认前，建议：

- 置为 `null`
- 或完全省略

## 12. 建议的下一步

当前 schema 已经足够清楚，可以进入下一步脚本设计。

建议下一步：

- 编写 `experiments/dit_lq_hq_v1/scripts/build_train_meta_lmdb.py`

目标输出目录：

- `experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta/`

建议该脚本未来完成的工作：

1. 读取 `SLP34K/train_gt.txt`
2. 从 `SLP34K/train/` 加载图像
3. 写入：
   - `image-%09d`
   - `label-%09d`
   - `meta-%09d`
4. `meta-%09d` 先采用保守 schema
5. `vocabulary_type` 与 `resolution_type` 在映射未确认前先置 `null`

## 13. 未解问题 / Blockers

1. raw 文件名中的 `g/ng` 到底表示什么？
   - 它与 unified 的 `IV/OOV` 计数并不能直接一一对上

2. raw 文件名中的 `d/nd` 到底表示什么？
   - 它与 unified 的 `normal/low` 计数也不能直接一一对上

3. `num_a` 与 `num_b` 的语义是什么？
   - 它们显然稳定存在，但当前 repo 内没有说明文档

4. 是否存在 repo 外部文档或旧预处理脚本，解释 raw test annotation 如何映射到 unified benchmark metadata：
   - `vocabulary_type`
   - `resolution_type`

5. 如果未来必须让 train metadata 与现有 unified test schema 完全对齐，那么在真正实现前，必须先解决 `vocabulary_type` 和 `resolution_type` 的映射问题
