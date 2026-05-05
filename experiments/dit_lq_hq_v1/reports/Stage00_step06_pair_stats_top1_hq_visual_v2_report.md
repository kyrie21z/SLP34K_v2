# Top1-HQ Visual-v2 Pair 构造与对比报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建 v2 脚本、v2 manifest、v2 预览图和 v2 报告。
未修改 v1 manifest、原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta`
- 输入 OCR CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.csv`
- 输入 v1 pair CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq.csv`
- 输出 train samples v2 CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_samples_visual_quality_v2.csv`
- 输出 train samples v2 JSONL: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_samples_visual_quality_v2.jsonl`
- 输出 v2 pair manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_visual_v2.csv`
- 输出 v2 same-structure manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_visual_v2_same_structure.csv`
- 输出 v2 quality-improved manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_visual_v2_quality_improved.csv`
- 输出 v2 same-structure+quality-improved manifest: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv`
- 输出预览图目录: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/previews/pair_samples_visual_v2`
- 输出报告: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step06_pair_stats_top1_hq_visual_v2_report.md`
- `--limit`: `None`

## 3. 视觉质量指标定义

- `sharpness = Laplacian variance`
- `contrast = grayscale std`
- `brightness = grayscale mean`
- `brightness_score = 1 - abs(brightness - 128) / 128`
- `resolution = width * height`
- `visual_quality_score = 0.5 * sharpness_norm + 0.2 * contrast_norm + 0.2 * brightness_score + 0.1 * resolution_norm`

## 4. 样本级视觉质量统计

- 样本数: `27501`

| metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| sharpness | 640.706343 | 352.250029 | 1.216625 | 10467.475127 |
| contrast | 32.484072 | 31.576789 | 1.197152 | 91.767790 |
| brightness | 106.412661 | 104.909818 | 10.813032 | 250.746975 |
| resolution | 13050.686593 | 8928.000000 | 462.000000 | 281088.000000 |
| visual_quality_score | 0.240201 | 0.239108 | 0.023446 | 0.846032 |

## 5. v2 HQ 选择规则

每个 `label` group 的 HQ 选择逻辑：

1. 如果 group 内存在 `ocr_correct=True` 样本，则候选集合只保留这些样本
2. 如果 group 内全是 `ocr_correct=False`，则允许从全组中选 HQ，并记录为 `all_ocr_wrong_group`
3. 在候选集合内按以下顺序排序：
   - `quality_priority` 降序
   - `visual_quality_score` 降序
   - `sharpness_norm` 降序
   - `contrast_norm` 降序
   - `brightness_score` 降序
   - `resolution_norm` 降序
   - `lmdb_index` 升序

## 6. v2 Pair Manifest 总览

- `num_pairs`: `15960`
- `same_structure`: `14315` (0.896930)
- `cross_structure`: `1645` (0.103070)
- HQ quality 分布:

| quality | count |
| --- | ---: |
| easy | 3525 |
| middle | 6733 |
| hard | 1283 |

- HQ structure 分布:

| structure | count |
| --- | ---: |
| single | 3467 |
| vertical | 173 |
| multi | 7901 |

- HQ `ocr_correct=False` 数量: `5`

### quality_relation 分布

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| hard_to_middle | 1289 |
| middle_to_middle | 3616 |
| hard_to_hard | 237 |

### structure_pair matrix

| structure_pair | count |
| --- | ---: |
| single->single | 3693 |
| vertical->vertical | 3729 |
| single->vertical | 458 |
| multi->multi | 6893 |
| multi->single | 581 |
| single->multi | 494 |
| vertical->single | 103 |
| multi->vertical | 9 |

## 7. v2 过滤子集统计

| filename | num_pairs | ratio | wrong_hq_count |
| ---: | ---: | ---: | ---: |
| pair_manifest_top1_hq_visual_v2_same_structure.csv | 14315 | 0.896930 | 0 |
| pair_manifest_top1_hq_visual_v2_quality_improved.csv | 9399 | 0.588910 | 0 |
| pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv | 8221 | 0.515100 | 0 |

### pair_manifest_top1_hq_visual_v2_same_structure.csv

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 5935 |
| easy_to_easy | 2518 |
| hard_to_easy | 1074 |
| hard_to_middle | 1212 |
| middle_to_middle | 3350 |
| hard_to_hard | 226 |

| structure_relation | count |
| --- | ---: |
| same_structure | 14315 |

| quality | count |
| --- | ---: |
| middle | 9285 |
| easy | 2518 |
| hard | 2512 |

| quality | count |
| --- | ---: |
| easy | 9527 |
| middle | 4562 |
| hard | 226 |

| structure | count |
| --- | ---: |
| single | 3693 |
| vertical | 3729 |
| multi | 6893 |

| structure | count |
| --- | ---: |
| single | 3693 |
| vertical | 3729 |
| multi | 6893 |

### pair_manifest_top1_hq_visual_v2_quality_improved.csv

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| hard_to_easy | 1253 |
| hard_to_middle | 1289 |

| structure_relation | count |
| --- | ---: |
| same_structure | 8221 |
| cross_structure | 1178 |

| quality | count |
| --- | ---: |
| middle | 6857 |
| hard | 2542 |

| quality | count |
| --- | ---: |
| easy | 8110 |
| middle | 1289 |

| structure | count |
| --- | ---: |
| single | 2710 |
| vertical | 2703 |
| multi | 3986 |

| structure | count |
| --- | ---: |
| single | 2565 |
| vertical | 3024 |
| multi | 3810 |

### pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 5935 |
| hard_to_easy | 1074 |
| hard_to_middle | 1212 |

| structure_relation | count |
| --- | ---: |
| same_structure | 8221 |

| quality | count |
| --- | ---: |
| middle | 5935 |
| hard | 2286 |

| quality | count |
| --- | ---: |
| easy | 7009 |
| middle | 1212 |

| structure | count |
| --- | ---: |
| single | 2053 |
| vertical | 2630 |
| multi | 3538 |

| structure | count |
| --- | ---: |
| single | 2053 |
| vertical | 2630 |
| multi | 3538 |



## 8. v1 vs v2 HQ 选择对比

- `num_label_groups`: `4964`
- `num_hq_changed`: `1797`
- `hq_changed_ratio`: `0.362006`
- `v1 clean subset pair 数`: `8088`
- `v2 clean subset pair 数`: `8221`
- `v1 wrong-HQ 数`: `0`
- `v2 wrong-HQ 数`: `5`

### HQ changed 示例

| label | v1_hq | v2_hq | v1_quality | v2_quality | v1_visual_score | v2_visual_score | v1_confidence | v2_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 六安港LUANGANG | 942 | 13311 | easy | easy | 0.175481 | 0.407393 | 1.000000 | 1.000000 |
| 志远08 | 20223 | 27082 | easy | easy | 0.312081 | 0.376607 | 1.000000 | 1.000000 |
| 翔运989XIANGYUN | 10789 | 5 | easy | easy | 0.261790 | 0.332388 | 1.000000 | 1.000000 |
| 浙余杭货00898杭州港 | 12063 | 7 | middle | middle | 0.274621 | 0.348265 | 1.000000 | 1.000000 |
| 驰源999CHIYUAN | 8 | 8880 | middle | middle | 0.349501 | 0.435733 | 1.000000 | 1.000000 |
| 芜湖WUHU | 531 | 18335 | easy | easy | 0.368227 | 0.580072 | 1.000000 | 1.000000 |
| 杭州港 | 6477 | 17544 | easy | easy | 0.429503 | 0.439853 | 1.000000 | 0.999989 |
| 绍兴港SHAOXINGGANG | 14 | 24024 | easy | easy | 0.281227 | 0.441939 | 1.000000 | 1.000000 |
| 浙湖州货2768ZHEHUZHOUHUO | 16 | 4429 | middle | middle | 0.258252 | 0.287614 | 1.000000 | 1.000000 |
| 浙余姚货003ZHEYUYAOHUO | 5904 | 4494 | middle | middle | 0.355062 | 0.400790 | 1.000000 | 1.000000 |

## 9. Quality Relation 分布

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| hard_to_middle | 1289 |
| middle_to_middle | 3616 |
| hard_to_hard | 237 |

## 10. Structure Relation 分布

| structure_relation | count |
| --- | ---: |
| same_structure | 14315 |
| cross_structure | 1645 |

## 11. Wrong-HQ 与 all-OCR-wrong group 分析

- `HQ ocr_correct=False` 数量: `5`
- `all_ocr_wrong_group` 数量: `5`

| label |
| --- |
| 绍航集168绍兴港SHAOHANGJISHAOXINGGANG |
| 皖太和货339阜阳港WANTAIHEHUOFUYANGGANG |
| 苏盐货11965盐城港SUYANHUO |
| 豫信货11756信阳港YUXINHUOXINYANGGANG |
| 浙富阳货00836杭州港ZHEFUYANGHUOHANGZHOUGANG |

## 12. 可视化抽样说明

- 每个 bucket 默认抽样数: `20`
- 随机种子: `20260504`
- 预览图 buckets:

| bucket | count | sample_file |
| --- | ---: | --- |
| same_structure_quality_improved | 20 | same_structure_quality_improved/pair_526_lq_21501_hq_24024_middle_to_easy_same_structure.jpg |
| quality_improved | 20 | quality_improved/pair_326_lq_7438_hq_24024_hard_to_easy_cross_structure.jpg |
| same_structure | 20 | same_structure/pair_858_lq_23401_hq_26226_middle_to_easy_same_structure.jpg |
| hard_to_easy | 20 | hard_to_easy/pair_420_lq_14605_hq_24024_hard_to_easy_same_structure.jpg |
| hard_to_middle | 20 | hard_to_middle/pair_3589_lq_331_hq_22127_hard_to_middle_same_structure.jpg |
| middle_to_easy | 20 | middle_to_easy/pair_269_lq_2334_hq_24024_middle_to_easy_same_structure.jpg |
| hard_to_hard | 20 | hard_to_hard/pair_4280_lq_17434_hq_9218_hard_to_hard_same_structure.jpg |
| cross_structure | 20 | cross_structure/pair_1855_lq_18285_hq_21500_middle_to_easy_cross_structure.jpg |
| hq_changed_examples | 20 | hq_changed_examples/label_浙长兴货6758ZHECHANGXINGHUO_v1_409_v2_24038.jpg |
| random_all | 20 | random_all/pair_89_lq_20850_hq_18335_easy_to_easy_same_structure.jpg |

## 13. 推荐用于 Stage 1 Diffusion 训练的 Manifest

优先推荐：

- `data/manifests/pair_manifest_top1_hq_visual_v2_same_structure_quality_improved.csv`
- 理由：
  - 同结构，减少布局迁移
  - HQ 质量高于 LQ
  - HQ 由人工 quality 档 + 图像视觉质量指标选出
  - 不再依赖饱和的 OCR confidence

备选：

- `data/manifests/pair_manifest_top1_hq_visual_v2_same_structure.csv`

扩展：

- `data/manifests/pair_manifest_top1_hq_visual_v2.csv`

## 14. 警告与限制

- 同 label pair 不一定像素对齐。
- cross_structure pair 可能导致 diffusion loss 学习布局迁移。
- OCR confidence 接近饱和，不能完全代表图像质量。
- visual_quality_score 只是基于 sharpness/contrast/brightness/resolution 的启发式评分，不等同于人工主观质量。
- quality_improved 仍然依赖 raw quality 标注 `easy/middle/hard`。


## 15. 下一步建议

人工检查 `previews/pair_samples_visual_v2/` 中的样例；若 `same_structure_quality_improved` 视觉质量可接受，则用该 manifest 作为 Stage 1 conditional latent DiT diffusion-loss 预训练的第一版训练集；随后再设计 DiT 数据读取与训练脚本。
