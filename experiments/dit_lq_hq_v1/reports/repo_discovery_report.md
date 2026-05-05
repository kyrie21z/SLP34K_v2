# SLP34K_v2 仓库探查报告

## 1. 范围与未修改声明

本轮 discovery 为只读调查。未修改任何源码、数据文件、checkpoint 或配置文件。

本轮唯一新增文件是本报告：

- `/mnt/data/zyx/SLP34K_v2/repo_discovery_report.md`

本轮结论的边界说明：

- 本轮目标仅为仓库结构理解，不涉及数据导出、pair 构造、训练或评估执行。
- 当时尝试直接用 Python 探查 LMDB 原始 key，但当时 shell Python 环境缺少 `lmdb` 模块，因此没有完成从零开始的原始 key 验证。
- 为弥补这一点，报告结合了代码读取链路和 `ocr_training/outputs/` 中已有 manifest / summary 文件进行交叉确认。

## 2. 顶层目录结构

在 `/mnt/data/zyx/SLP34K_v2` 下观察到的主要顶层目录如下：

| 路径 | 用途判断 |
| --- | --- |
| `.git/` | Git 元数据 |
| `docs/` | 文档 |
| `image/README/` | README 插图资源 |
| `logs/` | 本地日志 |
| `mae/` | MAE 预训练代码 |
| `ocr_training/` | OCR 训练、评估、V2 corrector 主体代码 |
| `outputs/` | 顶层输出目录，与 `ocr_training/outputs/` 分开 |
| `reports/` | 历史分析与实验报告 |
| `scripts/` | 辅助脚本 |

需要特别说明的缺失项：

- 根目录没有独立 `configs/`，配置位于 `ocr_training/configs/`
- 根目录没有独立 `datasets/`，数据读取代码位于 `ocr_training/strhub/data/`
- 根目录没有独立 `evaluation/`，评估入口分散在 `ocr_training/test.py` 与 `ocr_training/tools/`
- 根目录没有独立 `data/`，数据位于 `ocr_training/data/`
- 根目录没有独立 `checkpoint/`，checkpoint 位于 `ocr_training/checkpoint/`
- 根目录没有独立 `tools/`，工具脚本位于 `ocr_training/tools/`

## 3. OCR Training 目录结构

`ocr_training/` 是后续分析与改造最关键的子树。

主要目录与文件：

| 路径 | 用途判断 |
| --- | --- |
| `ocr_training/train.py` | Hydra 训练入口 |
| `ocr_training/test.py` | benchmark 评估入口 |
| `ocr_training/configs/` | Hydra 源配置目录 |
| `ocr_training/config/` | 某次运行生成的 Hydra 配置输出，不是主要源配置 |
| `ocr_training/data/` | 本地 LMDB 数据 |
| `ocr_training/strhub/data/` | dataset / datamodule 实现 |
| `ocr_training/strhub/models/` | baseline 与 V2 模型实现 |
| `ocr_training/tools/` | cache 导出、split、confusion、pair 分析、offline eval 工具 |
| `ocr_training/checkpoint/` | baseline 识别模型 checkpoint |
| `ocr_training/pretrain_model/` | MAE 预训练权重 |
| `ocr_training/outputs/` | 已有分析缓存、summary、pair stats 与 corrector 实验输出 |

`strhub` 内与后续最相关的目录：

- `ocr_training/strhub/data/`
- `ocr_training/strhub/models/maevit_infonce_plm/`
- `ocr_training/strhub/models/maevit_plm/`
- `ocr_training/strhub/models/slp_mdiff/`
- `ocr_training/strhub/models/slp_mdiff_corrector/`

## 4. 数据集配置文件

当前 SLP34K 相关的核心配置文件：

| 文件 | 作用 |
| --- | --- |
| `ocr_training/configs/main.yaml` | 主 Hydra 配置，组织 model / charset / dataset / trainer |
| `ocr_training/configs/dataset/SLP34K.yaml` | 数据集目录名配置 |
| `ocr_training/configs/charset/SLP34K_568.yaml` | SLP34K 字符集 |
| `ocr_training/configs/model/maevit_infonce_plm.yaml` | baseline 识别模型配置 |
| `ocr_training/configs/model/slp_mdiff.yaml` | V2 MDiff 识别模型配置 |
| `ocr_training/configs/model/slp_mdiff_corrector.yaml` | V2 corrector 配置 |
| `ocr_training/configs/bench.yaml` | bench 用的 Hydra 辅助配置 |

`ocr_training/configs/main.yaml:1-6` 的默认组合：

- `model: maevit_infonce_plm`
- `charset: SLP34K_568`
- `dataset: SLP34K`

`ocr_training/configs/main.yaml:8-41` 中与数据最相关的关键字段：

- `model.img_size: [224, 224]`
- `model.max_label_length: 50`
- `model.batch_size: 32`
- `data.root_dir: data`
- `data.batch_size: ${model.batch_size}`
- `data.img_size: ${model.img_size}`
- `data.charset_train: ${model.charset_train}`
- `data.charset_test: ${model.charset_test}`
- `data.max_label_length: ${model.max_label_length}`
- `data.remove_whitespace: false`
- `data.normalize_unicode: false`
- `data.augment: true`
- `data.num_workers: 40`
- `trainer.val_check_interval: 400`
- `trainer.max_epochs: 100`
- `trainer.gpus: 4`

`ocr_training/configs/dataset/SLP34K.yaml:1-5` 中与 split 路径相关的字段：

- `data.train_dir: SLP34K_lmdb_train`
- `data.val_dir: SLP34K_lmdb_test`
- `data.test_dir: SLP34K_lmdb_benchmark`

字符集来源：

- `ocr_training/configs/charset/SLP34K_568.yaml` 将 `model.charset_train` 与 `model.charset_test` 定义为完整的 568 字符字符串

checkpoint 相关配置位置：

- baseline 识别模型初始化配置：`ocr_training/configs/model/maevit_infonce_plm.yaml`
- V2 MDiff 使用 baseline checkpoint 的配置：`ocr_training/configs/model/slp_mdiff.yaml:39-42`
  - `baseline_ckpt_path: checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`

## 5. Train/Test Split 与数据来源

### 基础训练 / 验证 / benchmark split

基础 OCR 训练链路读取的本地 LMDB 目录如下：

- train: `ocr_training/data/train/SLP34K_lmdb_train`
- val: `ocr_training/data/val/SLP34K_lmdb_test`
- test benchmark 根目录：`ocr_training/data/test/SLP34K_lmdb_benchmark`

在 benchmark 根目录下观察到的子集：

- `IV_lmdb`
- `OOV_lmdb`
- `low_lmdb`
- `multi-lines_lmdb`
- `normal_lmdb`
- `single-line_lmdb`
- `unified_lmdb`
- `vertical_lmdb`

LMDB 目录内实际看到的文件结构：

- `data.mdb`
- `lock.mdb`

根据 `ocr_training/strhub/data/dataset.py:85-115`，基础 dataset 至少依赖这些 key：

- `num-samples`
- `label-%09d`
- `image-%09d`

根据 `ocr_training/tools/export_parseq_corrector_cache.py:34-58`，某些 LMDB 还可能包含：

- `meta-%09d`

### train split 的读取路径

train split 的代码路径如下：

1. `ocr_training/train.py:58-108` 使用 Hydra 配置 `configs/main.yaml`
2. `ocr_training/configs/dataset/SLP34K.yaml` 填充 `data.train_dir`、`data.val_dir`、`data.test_dir`
3. `ocr_training/strhub/data/module.py:73-81` 将 train root 拼为 `PurePath(self.root_dir, 'train', self.train_dir)`
4. 在默认配置下，这会解析为 `data/train/SLP34K_lmdb_train`
5. `ocr_training/strhub/data/dataset.py:31-46` 递归搜索该目录下的 `**/data.mdb`，并构造 `LmdbDataset`

### README 与当前磁盘结构的差异

`README.md:123-137` 中给出的“Expected structure”与当前实际磁盘内容并不完全一致。当前代码和实际目录都使用：

- `ocr_training/data/test/SLP34K_lmdb_benchmark/...`

而不是 README 示例中的 `ocr_training/data/test/SLP34K_lmdb_train/`

## 6. 数据加载代码

### 基础 Dataset 类

file:

- `ocr_training/strhub/data/dataset.py`

class:

- `LmdbDataset`

key methods:

- `build_tree_dataset()`：`31-46`
- `LmdbDataset._preprocess_labels()`：`85-115`
- `LmdbDataset.__getitem__()`：`120-136`

returns:

- 基础 `__getitem__()` 只返回 `(img, label)`

关键行为：

- 从 LMDB 中读取 `label-%09d` 和 `image-%09d`
- 在预处理阶段将 label 通过 `CharsetAdapter` 转换后写入 `self.labels`
- 会过滤掉长度超过 `max_label_len` 的样本
- 支持 whitespace 去除和 unicode normalize，但当前配置都关闭
- 不返回 `image_path`、`layout`、`quality`、`split`、LMDB index 或 metadata

### label 归一化细节

file:

- `ocr_training/strhub/data/utils.py`

class:

- `CharsetAdapter`

key methods:

- `CharsetAdapter.__call__()`：`34-41`

行为：

- 若目标字符集全小写，则 label 转小写
- 若目标字符集全大写，则 label 转大写
- 移除不在目标字符集中的字符

对当前 SLP34K 配置的影响：

- `data.remove_whitespace = false`
- `data.normalize_unicode = false`
- 但 label 仍不是完全“原始标注文本”，因为不支持字符会被 `CharsetAdapter` 删除

### DataModule

file:

- `ocr_training/strhub/data/module.py`

class:

- `SceneTextDataModule`

key methods:

- `train_dataset`：`73-81`
- `val_dataset`：`83-91`
- `train_dataloader`：`93-96`
- `val_dataloader`：`98-101`
- `test_dataloaders`：`104-112`

returns:

- `train_dataset` / `val_dataset`：`ConcatDataset`
- `test_dataloaders`：以 benchmark 子目录名为 key 的 dataloader 字典

### 用于后续分析的 metadata-aware 导出器

file:

- `ocr_training/tools/export_parseq_corrector_cache.py`

class:

- `ExportLmdbDataset`

key methods:

- `ExportLmdbDataset.__getitem__()`：`34-58`
- `build_dataset()`：`71-90`
- `build_record()`：`227-262`

returns:

- 返回包含 `image`、`label`、`lmdb_index`、`lmdb_root`、`meta` 的 dict
- manifest row 中包含 `sample_id`、`subset`、`lmdb_index`、`lmdb_root`、`gt_text`、`pred_text`、`metadata`

这条 metadata-aware 路径不是基础 OCR 训练链路，而是 V2 分析 / 导出链路。

## 7. 可用样本字段

下表区分“基础 train dataset 链路”与“后续导出 cache 链路”：

| Field | Available? | Source | Notes |
| --- | ---: | --- | --- |
| `image_path` | no / unclear | 基础 train dataset | `LmdbDataset` 返回的是解码后的图像，不是路径字符串 |
| `label` | yes | LMDB `label-%09d` | 由 `LmdbDataset` 暴露，但经过 `CharsetAdapter` |
| `layout` | 基础 train 无；unified cache 有近似字段 | `metadata.structure` / `metadata.structure_type` | 仓库里没有字面量 `layout` 字段，最接近的是 `structure` 与 `structure_type` |
| `quality` | 基础 train 无；unified cache 有 | `metadata.quality` | unified cache manifest 中可见；train/val cache manifest 中为 `null` |
| `split` | 隐式有 | 目录路径 + 调用方 | 基础 dataset 不逐样本返回 split，但 root 已决定 split |

少量样本示例来自已有 cache manifest，而非本轮新导出：

```text
sample[train, existing cache] = {
  image_path: null,
  label: "浙萧山货23765",
  layout: null,
  quality: null,
  split: "train",
  source: "ocr_training/outputs/V2-M02u_corrector_cache_replace_only/manifest.jsonl",
  lmdb_root: "/mnt/data/zyx/SLP34K_v2/ocr_training/data/train/SLP34K_lmdb_train",
  lmdb_index: 1
}

sample[val, existing cache] = {
  image_path: null,
  label: "浙富阳货00268",
  layout: null,
  quality: null,
  split: "val",
  source: "ocr_training/outputs/V2-M02u_corrector_cache_replace_only_val/manifest.jsonl",
  lmdb_root: "/mnt/data/zyx/SLP34K_v2/ocr_training/data/val/SLP34K_lmdb_test",
  lmdb_index: 1
}

sample[test unified, existing cache] = {
  image_path: null,
  label: "浙余杭货02039ZHEYUHANGHUO",
  layout: "structure=multi, structure_type=multi_lines",
  quality: "easy",
  split: "test",
  source: "ocr_training/outputs/V2-M02w0_cache_test_unified_replace_dominant_bs64/manifest.jsonl",
  lmdb_root: "/mnt/data/zyx/SLP34K_v2/ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb",
  lmdb_index: 15
}
```

解释：

- train / val 的已有 cache row 中没有 metadata
- unified benchmark 的 cache row 中有 `quality`、`vocabulary_type`、`resolution_type`、`structure_type`

## 8. Annotation / LMDB / Metadata 文件

### 基础数据文件

当前在 `ocr_training/data/` 下找到的基础数据存储形式为：

- `ocr_training/data/train/SLP34K_lmdb_train/`
- `ocr_training/data/val/SLP34K_lmdb_test/`
- `ocr_training/data/test/SLP34K_lmdb_benchmark/*_lmdb/`

这些目录中可见的文件仅有：

- `data.mdb`
- `lock.mdb`

### 原始 annotation sidecar

在 `ocr_training/data/` 下，本轮没有找到原始 annotation sidecar：

- 没有 `.csv`
- 没有 `.json`
- 没有 `.txt`
- 没有 `.tsv`

### 在其他目录中发现的 metadata / analysis 文件

在 `ocr_training/outputs/` 下可以找到大量分析产物：

- `manifest.jsonl`
- `export_summary.json`
- `split_summary.json`
- `confusion_table.csv`
- `confusion_table.json`
- `pair_stats.csv`
- `pair_stats.json`
- `subset_stats.csv`
- `subset_stats.json`
- `position_diagnostics.csv`
- `case_study.json`

结论：

- 基础 OCR 数据存储是 LMDB 主导的
- 仓库内出现的 CSV / JSON / TXT 多数是后续实验分析产物，不是 train split 的原始 annotation

## 9. 训练与评估入口

### 训练入口

- `ocr_training/train.py`

行为：

- 使用 `@hydra.main(config_path='configs', config_name='main')`
- 实例化模型与 `SceneTextDataModule`
- 将 `data.root_dir` 解析为绝对路径
- 运行 `trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)`

### baseline 评估入口

- `ocr_training/test.py`

行为：

- 加载识别 checkpoint
- 构造 `SceneTextDataModule(..., test_dir='SLP34K_lmdb_benchmark', ...)`
- 遍历 `SceneTextDataModule.TEST_SLP34K`
- 对 benchmark 下每个 `*_lmdb` 子集分别评估

### Hydra 配置组织

`ocr_training/configs/` 下的配置组：

- `dataset/`
- `charset/`
- `model/`
- `experiment/`

主配置：

- `ocr_training/configs/main.yaml`

辅助 bench 配置：

- `ocr_training/configs/bench.yaml`

checkpoint 位置：

- baseline checkpoint：
  - `ocr_training/checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- V2 模型配置中也引用了该路径：
  - `ocr_training/configs/model/slp_mdiff.yaml`

## 10. Unified Evaluation 入口

本轮显式检查过以下文件：

- `ocr_training/evaluation/evaluate_unified.py`
- `ocr_training/evaluation/csv_to_lmdb.py`

结果：

- 这两个文件都不存在
- 当前仓库中也没有顶层 `ocr_training/evaluation/` 目录

当前仓库里实际承担“unified analysis / evaluation”职责的入口更接近于：

1. `ocr_training/tools/export_parseq_corrector_cache.py`
   - 从 `train` / `val` / `test` 读取样本，导出 metadata-aware cache
   - 对 unified benchmark 使用 `--split test --subset unified_lmdb`
   - 读取可选的 `meta-%09d`，并写出 `manifest.jsonl`

2. `ocr_training/tools/eval_mdiff_corrector_offline.py`
   - 针对导出的 cache manifest 做 offline 评估
   - 使用 `vocabulary_type`、`quality`、`resolution_type`、`structure_type` 等 metadata 分组

3. `ocr_training/tools/split_mdiff_corrector_cache.py`
   - 将导出的 cache 再 split 成 analysis `train/` 与 `eval/`

4. `ocr_training/tools/build_confusion_table.py`
   - 从 cache manifest 构建 confusion table

也就是说，当前 repo 里的 “unified evaluation / analysis” 实际是在 `tools/` 里，而不是 `ocr_training/evaluation/`

## 11. 与 same-label pair 构造相关的结论

### 是否可以直接按 label 分组

可以，但要注意一个细节。

- 基础 train label 来自 LMDB `label-%09d`
- 在进入 `LmdbDataset.labels` 之前，会先经过 `CharsetAdapter`
- 当前配置虽然关闭了 whitespace / unicode normalize，但不在字符集内的字符仍可能被删掉

这意味着：

- 后续 same-label 分组最好使用“dataset 实际吐出的 label”
- 如果使用已有 exporter 导出的 manifest，则用 `gt_text`
- 如果直接遍历 `LmdbDataset`，则用返回的 `label`

### label 字段来源

- 基础路径：LMDB `label-%09d`
- 导出 cache 路径：`gt_text`，来源仍是 dataset label

### same-label 是否可直接作为 group key

大概率可以。

原因：

- 当前训练链路本身就把这个 label 作为 canonical ground truth
- 现有 cache manifest 中也用 `gt_text` 作为主文本字段
- 在 dataset 读取之后，没有看到额外的 label 归一化层

### image path 是否可追溯

在当前基础 dataset 接口下，不能直接拿到原始图像路径字符串。

- 基础 dataset 只暴露图像 bytes 解码后的图像对象
- 导出 manifest 会保留 `lmdb_root` 与 `lmdb_index`
- 没有在基础 train/val cache row 或 unified cache row 中看到原始 JPG 路径

因此目前稳定的样本定位方式是：

- `lmdb_root + lmdb_index`

而不是：

- 原始 `image_path`

### layout / quality 是否可用于后续统计

对基础 train split 来说，不行。

- `LmdbDataset.__getitem__()` 只返回 `(img, label)`
- 已有 train / val cache manifest 也显示 `metadata: null`

对 unified benchmark analysis 来说，可以。

- unified cache manifest row 中可见：
  - `quality`
  - `vocabulary_type`
  - `resolution_type`
  - `structure_type`
  - `structure`

### “layout” 的术语问题

当前 repo 中并没有明确使用 `layout` 这个字段名，更接近的是：

- `structure`
- `structure_type`

如果后续分析里需要 `layout`，应优先理解为这些字段中的一种映射。

## 12. 针对 pair statistics 的下一步建议

建议下一步先保持只读，并从基础 train LMDB 出发，而不是复用现有的 unified test analysis split。

建议的入口：

1. 做 train split 分布统计
   - 从 `ocr_training/strhub/data/dataset.py` 与 `ocr_training/strhub/data/module.py` 入手
   - 读取 `data/train/SLP34K_lmdb_train`
   - 使用 dataset 实际返回的 `label` 做 group key

2. 如果需要更稳定的样本定位
   - 复用 `ocr_training/tools/export_parseq_corrector_cache.py` 的结构设计
   - 特别是 `ExportLmdbDataset` 暴露 `lmdb_index` / `lmdb_root` 的方式

3. same-label grouping
   - 直接使用 `label` 或导出 manifest 中的 `gt_text`

4. top1-HQ pair 构造
   - 图像定位建议使用 `lmdb_root + lmdb_index`
   - 不要假设当前 train split 中有原始 `image_path`

5. layout / quality 丰富化
   - 需要先确认 train LMDB 是否真的有 `meta-%09d`
   - 如果没有，则 train split 的 pair stats 不能直接依赖 `quality` / `structure_type`

更具体的建议：

下一步应先做一个只读 train sample exporter，导出一个轻量级 `train_samples.csv` 或 `manifest`，至少包含：

- `lmdb_root`
- `lmdb_index`
- `label`
- `split`
- 若可得则加 `metadata`

然后再做：

- label group 频次统计
- duplicate-label group 检查
- 在此基础上设计 same-label top1-HQ pair 构造

重要提醒：

- 不应把 `outputs/V2-M02w0_cache_main_split*` 当作 train split 使用，因为这些 split 来自 `test/unified_lmdb`，不是基础 train split

## 13. 未解问题 / Blockers

1. `ocr_training/data/train/SLP34K_lmdb_train` 是否真的包含 `meta-%09d`？
   - 当时尚未直接验证，只能从已有 train cache manifest 的 `metadata: null` 推测大概率没有

2. `ocr_training/data/val/SLP34K_lmdb_test` 是否包含 `meta-%09d`？
   - 同样未直接验证，只能从已有 val cache manifest 的 `metadata: null` 推测大概率没有

3. 是否存在额外的原始 JPG 路径来源？
   - 本轮未在基础 dataset 接口或已有 manifest 中看到

4. “layout” 到底应映射为 `structure`、`structure_type` 还是别的概念？
   - 当前仓库使用的是 structure 系字段

5. 仓库外是否还有另一个 SLP34K 数据版本？
   - 可能有，因为 README 的数据结构示意与当前磁盘内容并不完全一致

6. 后续 same-label 分组应使用 raw label 还是 post-charset-filter label？
   - 当前训练链路实际使用的是 post-`CharsetAdapter` label
