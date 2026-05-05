# Pair 可视化抽样与训练子集筛选报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建脚本、过滤 manifest、预览图和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta`
- 输入 pair CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq.csv`
- 输出 same-structure manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_same_structure.csv`
- 输出 quality-improved manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_quality_improved.csv`
- 输出 same-structure+quality-improved manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_same_structure_quality_improved.csv`
- 输出 no-wrong-HQ manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_no_wrong_hq.csv`
- 预览图目录: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/previews/pair_samples`
- 输出报告: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step05_pair_visual_inspection_report.md`
- `--limit`: `None`

## 3. 一致性检查

- pair_id 唯一: `True`
- pair_type 正确: `True`
- 每个 pair 的 lq/hq label 相同: `True`
- `lq_lmdb_index != hq_lmdb_index`: `True`
- structure_relation 校验: `True`
- quality_relation 校验: `True`

## 4. 原始 Pair Manifest 总览

- `num_pairs`: `15960`
- `same_structure`: `14072` (0.881704)
- `cross_structure`: `1888` (0.118296)
- `HQ ocr_correct=False` 数量: `0`

### quality_relation 分布

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| middle_to_middle | 3616 |
| hard_to_middle | 1289 |
| hard_to_hard | 237 |

### structure_pair matrix

| structure_pair | count |
| --- | ---: |
| single->single | 3814 |
| vertical->vertical | 3376 |
| single->vertical | 400 |
| multi->multi | 6882 |
| multi->single | 600 |
| vertical->single | 456 |
| single->multi | 420 |
| multi->vertical | 12 |

## 5. 过滤后 Manifest 统计

| filename | num_pairs | ratio | wrong_hq_count |
| --- | ---: | ---: | ---: |
| pair_manifest_top1_hq_same_structure.csv | 14072 | 0.881704 | 0 |
| pair_manifest_top1_hq_quality_improved.csv | 9399 | 0.588910 | 0 |
| pair_manifest_top1_hq_same_structure_quality_improved.csv | 8088 | 0.506767 | 0 |
| pair_manifest_top1_hq_no_wrong_hq.csv | 15960 | 1.000000 | 0 |

### 各子集详细统计

#### pair_manifest_top1_hq_same_structure.csv

- num_pairs: `14072`
- 占原始 pair 比例: `0.881704`
- wrong HQ 数量: `0`

quality_relation 分布：

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 5802 |
| easy_to_easy | 2390 |
| hard_to_easy | 1060 |
| middle_to_middle | 3368 |
| hard_to_middle | 1226 |
| hard_to_hard | 226 |

structure_relation 分布：

| structure_relation | count |
| --- | ---: |
| same_structure | 14072 |

LQ quality 分布：

| quality | count |
| --- | ---: |
| middle | 9170 |
| easy | 2390 |
| hard | 2512 |

HQ quality 分布：

| quality | count |
| --- | ---: |
| easy | 9252 |
| middle | 4594 |
| hard | 226 |

LQ structure 分布：

| structure | count |
| --- | ---: |
| single | 3814 |
| vertical | 3376 |
| multi | 6882 |

HQ structure 分布：

| structure | count |
| --- | ---: |
| single | 3814 |
| vertical | 3376 |
| multi | 6882 |

#### pair_manifest_top1_hq_quality_improved.csv

- num_pairs: `9399`
- 占原始 pair 比例: `0.588910`
- wrong HQ 数量: `0`

quality_relation 分布：

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| hard_to_easy | 1253 |
| hard_to_middle | 1289 |

structure_relation 分布：

| structure_relation | count |
| --- | ---: |
| same_structure | 8088 |
| cross_structure | 1311 |

LQ quality 分布：

| quality | count |
| --- | ---: |
| middle | 6857 |
| hard | 2542 |

HQ quality 分布：

| quality | count |
| --- | ---: |
| easy | 8110 |
| middle | 1289 |

LQ structure 分布：

| structure | count |
| --- | ---: |
| single | 2710 |
| vertical | 2703 |
| multi | 3986 |

HQ structure 分布：

| structure | count |
| --- | ---: |
| single | 2881 |
| vertical | 2759 |
| multi | 3759 |

#### pair_manifest_top1_hq_same_structure_quality_improved.csv

- num_pairs: `8088`
- 占原始 pair 比例: `0.506767`
- wrong HQ 数量: `0`

quality_relation 分布：

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 5802 |
| hard_to_easy | 1060 |
| hard_to_middle | 1226 |

structure_relation 分布：

| structure_relation | count |
| --- | ---: |
| same_structure | 8088 |

LQ quality 分布：

| quality | count |
| --- | ---: |
| middle | 5802 |
| hard | 2286 |

HQ quality 分布：

| quality | count |
| --- | ---: |
| easy | 6862 |
| middle | 1226 |

LQ structure 分布：

| structure | count |
| --- | ---: |
| single | 2146 |
| vertical | 2410 |
| multi | 3532 |

HQ structure 分布：

| structure | count |
| --- | ---: |
| single | 2146 |
| vertical | 2410 |
| multi | 3532 |

#### pair_manifest_top1_hq_no_wrong_hq.csv

- num_pairs: `15960`
- 占原始 pair 比例: `1.000000`
- wrong HQ 数量: `0`

quality_relation 分布：

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| middle_to_middle | 3616 |
| hard_to_middle | 1289 |
| hard_to_hard | 237 |

structure_relation 分布：

| structure_relation | count |
| --- | ---: |
| same_structure | 14072 |
| cross_structure | 1888 |

LQ quality 分布：

| quality | count |
| --- | ---: |
| middle | 10473 |
| easy | 2708 |
| hard | 2779 |

HQ quality 分布：

| quality | count |
| --- | ---: |
| easy | 10818 |
| middle | 4905 |
| hard | 237 |

LQ structure 分布：

| structure | count |
| --- | ---: |
| single | 4634 |
| vertical | 3832 |
| multi | 7494 |

HQ structure 分布：

| structure | count |
| --- | ---: |
| single | 4870 |
| vertical | 3788 |
| multi | 7302 |



## 6. Same-structure 子集分析

- 文件: `pair_manifest_top1_hq_same_structure.csv`
- pair 数: `14072`
- 占原始比例: `0.881704`
- 说明: 保留相同结构 pair，减少布局迁移干扰，但仍包含 `easy_to_easy`、`middle_to_middle`、`hard_to_hard` 等非质量提升样本。

## 7. Quality-improved 子集分析

- 文件: `pair_manifest_top1_hq_quality_improved.csv`
- pair 数: `9399`
- 占原始比例: `0.588910`
- 包含关系: `hard_to_middle`, `hard_to_easy`, `middle_to_easy`
- 说明: 保证 HQ 质量优于 LQ，但仍包含 cross-structure pair。

## 8. Same-structure + Quality-improved 子集分析

- 文件: `pair_manifest_top1_hq_same_structure_quality_improved.csv`
- pair 数: `8088`
- 占原始比例: `0.506767`
- 说明: 同结构且 HQ 质量优于 LQ，是第一版 diffusion loss 训练最干净子集。

## 9. Wrong-HQ 分析

- wrong-HQ pair 数量（原始）: `0`
- `pair_manifest_top1_hq_no_wrong_hq.csv` 移除 pair 数量: `0`

无

## 10. Cross-structure 分析

- cross-structure pair 数量: `1888`
- cross-structure 比例: `0.118296`
- 说明: 这些样本可能让 diffusion loss 学到布局迁移，而不是单纯质量恢复。

## 11. 可视化抽样说明

- 每个 bucket 默认抽样数量: `20`
- 随机种子: `20260504`
- `hq_ocr_wrong` bucket: 若样本很少则全部导出
- `large_group_examples` bucket: 从 `group_size` 最大的若干 label group 中抽样

## 12. 预览图目录索引

| bucket | count | sample_file |
| --- | ---: | --- |
| same_structure | 20 | same_structure/pair_1504_lq_22034_hq_2070_easy_to_easy_same_structure.jpg |
| cross_structure | 20 | cross_structure/pair_139_lq_13207_hq_6477_easy_to_easy_cross_structure.jpg |
| quality_improved | 20 | quality_improved/pair_389_lq_10179_hq_14_middle_to_easy_same_structure.jpg |
| same_structure_quality_improved | 20 | same_structure_quality_improved/pair_654_lq_4976_hq_642_middle_to_easy_same_structure.jpg |
| hard_to_easy | 20 | hard_to_easy/pair_573_lq_16807_hq_14_hard_to_easy_same_structure.jpg |
| hard_to_middle | 20 | hard_to_middle/pair_5010_lq_594_hq_12987_hard_to_middle_same_structure.jpg |
| middle_to_easy | 20 | middle_to_easy/pair_339_lq_2526_hq_14_middle_to_easy_same_structure.jpg |
| hard_to_hard | 20 | hard_to_hard/pair_4280_lq_17434_hq_451_hard_to_hard_same_structure.jpg |
| hq_ocr_wrong | 0 | 无 |
| large_group_examples | 20 | large_group_examples/pair_254_lq_7255_hq_14_easy_to_easy_same_structure.jpg |
| random_all | 20 | random_all/pair_1665_lq_826_hq_255_middle_to_easy_same_structure.jpg |

## 13. 推荐用于 Stage 1 Diffusion 训练的 Manifest

第一版最干净训练集：

- `data/manifests/pair_manifest_top1_hq_same_structure_quality_improved.csv`
- 原因：
  - 同结构，减少布局迁移干扰
  - HQ 质量优于 LQ
  - 更符合 Diffusion loss 的“低质量到高质量”恢复设定

备选训练集：

- `data/manifests/pair_manifest_top1_hq_same_structure.csv`
- 原因：
  - 样本更多
  - 但包含 `easy_to_easy`、`middle_to_middle`、`hard_to_hard` 等非质量提升 pair

扩展训练集：

- `data/manifests/pair_manifest_top1_hq_no_wrong_hq.csv`
- 原因：
  - 排除被 OCR 判错的 HQ
  - 但仍包含 cross-structure pair

## 14. 警告与限制

- 同 label pair 不一定像素对齐。
- cross_structure pair 可能导致 diffusion loss 学习布局迁移。
- OCR confidence 接近饱和，不能完全代表图像质量。
- quality_improved 只基于 raw quality 标注 `easy/middle/hard`。
- 可视化抽样只能辅助人工判断，不能替代训练验证。


## 15. 下一步建议

人工检查 `previews/pair_samples/` 中的样例；若 `same_structure_quality_improved` 视觉质量可接受，则用该 manifest 作为 Stage 1 conditional latent DiT diffusion-loss 预训练的第一版训练集；随后再设计 DiT 数据读取与训练脚本。
