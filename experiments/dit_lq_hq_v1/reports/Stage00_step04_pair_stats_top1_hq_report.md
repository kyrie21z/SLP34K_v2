# Same-label Top1-HQ Pair 构造报告

## 1. 范围与隔离声明

本轮只在 `experiments/dit_lq_hq_v1/` 下创建脚本、manifest 和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- 输入 LMDB: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta`
- 输入 OCR CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.csv`
- 输出 `train_samples_meta.csv`: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_samples_meta.csv`
- 输出 `train_samples_meta.jsonl`: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_samples_meta.jsonl`
- 输出 `pair_manifest_top1_hq.csv`: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/pair_manifest_top1_hq.csv`
- 输出报告: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step04_pair_stats_top1_hq_report.md`
- `--limit`: `None`

## 3. 数据读取与一致性检查

- LMDB `num-samples`: `27501`
- OCR CSV 样本数: `27501`
- 实际处理样本数: `27501`
- OCR CSV `lmdb_index` 唯一性: `True`
- LMDB `label` 与 OCR CSV `label` 一致性: `True`
- `metadata.id == lmdb_index`: `True`
- `metadata.raw_label == label`: `True`
- `metadata.split == train`: `True`
- `quality/structure/structure_type` 非空: `True`
- OCR CSV 实际字段映射: `{"lmdb_index": "lmdb_index", "label": "label", "pred": "pred", "correct": "correct", "confidence": "confidence", "avg_conf": "avg_conf", "min_conf": "min_conf", "pred_length": "pred_length", "label_length": "label_length"}`

## 4. HQ 选择规则

同一 `label` group 内排序规则：

1. `quality_priority` 降序：`easy=3 > middle=2 > hard=1`
2. `ocr_correct` 降序：`True > False`
3. `confidence` 降序
4. `min_conf` 降序
5. `lmdb_index` 升序

排序后每个 label group 的第一条样本作为 HQ，其余样本作为 LQ。

## 5. Train Sample Manifest 统计

- `num_samples`: `27501`
- OCR `correct`: `27494`
- OCR `wrong`: `7`
- OCR train accuracy: `0.999745`
- confidence 均值: `0.999901`
- confidence 中位数: `0.999999`

### quality 分布

| quality | count |
| --- | ---: |
| middle | 17206 |
| hard | 4062 |
| easy | 6233 |

### structure 分布

| structure | count |
| --- | ---: |
| single | 8112 |
| vertical | 4005 |
| multi | 15384 |

### structure_type 分布

| structure_type | count |
| --- | ---: |
| single_line | 8112 |
| vertical | 4005 |
| multi_lines | 15384 |

## 6. Label Group 统计

- unique label 数: `11541`
- group_size = 1 的 label 数: `6577`
- group_size >= 2 的 label 数: `4964`
- group_size >= 3 的 label 数: `2625`
- 最大 group_size: `378`
- 平均 group_size: `2.382896`
- 中位数 group_size: `1.000000`
- 可构造 pair 数: `15960`
- single-instance label 数: `6577`

### group_size 分布表

| group_size | count |
| --- | ---: |
| 1 | 6577 |
| 2 | 2339 |
| 3 | 1103 |
| 4 | 553 |
| 5 | 336 |
| 6 | 170 |
| 7 | 127 |
| 8 | 74 |
| 9 | 51 |
| 10 | 37 |
| 11 | 23 |
| 12 | 22 |
| 13 | 12 |
| 14 | 14 |
| 15 | 9 |
| 16 | 8 |
| 17 | 6 |
| 18 | 8 |
| 19 | 6 |
| 20 | 6 |
| 21 | 2 |
| 22 | 2 |
| 23 | 2 |
| 24 | 1 |
| 25 | 2 |
| 26 | 1 |
| 27 | 4 |
| 28 | 3 |
| 29 | 2 |
| 31 | 2 |
| 33 | 2 |
| 34 | 1 |
| 36 | 1 |
| 37 | 1 |
| 38 | 1 |
| 39 | 1 |
| 40 | 2 |
| 41 | 1 |
| 43 | 1 |
| 45 | 1 |
| 46 | 2 |
| 48 | 1 |
| 49 | 1 |
| 52 | 1 |
| 55 | 1 |
| 64 | 1 |
| 68 | 1 |
| 71 | 1 |
| 72 | 1 |
| 80 | 1 |
| 84 | 1 |
| 90 | 1 |
| 95 | 1 |
| 98 | 1 |
| 100 | 1 |
| 101 | 1 |
| 116 | 1 |
| 122 | 1 |
| 130 | 1 |
| 132 | 1 |
| 136 | 1 |
| 141 | 1 |
| 165 | 1 |
| 190 | 1 |
| 371 | 1 |
| 378 | 1 |

## 7. Pair Manifest 统计

- `num_pairs`: `15960`
- `same_structure` pair 数: `14072`
- `cross_structure` pair 数: `1888`
- 每个 label group 只有一个 HQ: `True`
- `sum(group_size - 1)` 校验: `True`

### quality_relation 分布

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| middle_to_middle | 3616 |
| hard_to_middle | 1289 |
| hard_to_hard | 237 |

### structure pair matrix

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

### LQ quality 分布

| quality | count |
| --- | ---: |
| middle | 10473 |
| easy | 2708 |
| hard | 2779 |

### HQ quality 分布

| quality | count |
| --- | ---: |
| easy | 10818 |
| middle | 4905 |
| hard | 237 |

### LQ structure 分布

| structure | count |
| --- | ---: |
| single | 4634 |
| vertical | 3832 |
| multi | 7494 |

### HQ structure 分布

| structure | count |
| --- | ---: |
| single | 4870 |
| vertical | 3788 |
| multi | 7302 |

## 8. Quality Relation 分布

| quality_relation | count |
| --- | ---: |
| middle_to_easy | 6857 |
| easy_to_easy | 2708 |
| hard_to_easy | 1253 |
| middle_to_middle | 3616 |
| hard_to_middle | 1289 |
| hard_to_hard | 237 |

## 9. Structure Relation 分布

| structure_relation | count |
| --- | ---: |
| same_structure | 14072 |
| cross_structure | 1888 |

## 10. HQ 样本分布

- HQ 样本数: `11541`
- HQ `ocr_correct=False` 数量: `5`
- HQ confidence 均值: `0.999896`
- HQ confidence 最小值: `0.956074`
- HQ confidence 中位数: `0.999999`

### HQ quality 分布

| quality | count |
| --- | ---: |
| easy | 3525 |
| middle | 6733 |
| hard | 1283 |

### HQ structure 分布

| structure | count |
| --- | ---: |
| single | 3478 |
| vertical | 173 |
| multi | 7890 |

### 被选为 HQ 的错误 OCR 样本

| label | lmdb_index | quality | structure | confidence | source_path |
| --- | ---: | --- | --- | ---: | ---: |
| 绍航集168绍兴港SHAOHANGJISHAOXINGGANG | 2249 | middle | multi | 0.956074 | train/middle&multi&ng&nd&绍航集168绍兴港SHAOHANGJISHAOXINGGANG&-5&4&T_20220410_10_08_33_354250.jpg |
| 皖太和货339阜阳港WANTAIHEHUOFUYANGGANG | 7541 | middle | multi | 0.992906 | train/middle&multi&ng&nd&皖太和货339阜阳港WANTAIHEHUOFUYANGGANG&5&2&T_20220210_13_48_15_464368.jpg |
| 苏盐货11965盐城港SUYANHUO | 19828 | middle | multi | 0.987734 | train/middle&multi&ng&nd&苏盐货11965盐城港SUYANHUO&-1&1&T_20181207_12_32_14_209600.jpg |
| 豫信货11756信阳港YUXINHUOXINYANGGANG | 20291 | middle | multi | 0.988306 | train/middle&multi&ng&nd&豫信货11756信阳港YUXINHUOXINYANGGANG&11&2&O_20211031_14_44_07_693532.jpg |
| 浙富阳货00836杭州港ZHEFUYANGHUOHANGZHOUGANG | 26631 | middle | multi | 0.961937 | train/middle&multi&ng&nd&浙富阳货00836杭州港ZHEFUYANGHUOHANGZHOUGANG&0&2&T_20190605_13_20_03_000000.jpg |

## 11. OCR 置信度统计

- 全样本 confidence 均值: `0.999901`
- 全样本 confidence 中位数: `0.999999`
- HQ confidence 均值: `0.999896`
- LQ confidence 均值: `0.999905`

## 12. 样本示例

### 前 5 个 pair 示例

| label | lq_index | hq_index | lq_quality | hq_quality | lq_structure | hq_structure | lq_confidence | hq_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 浙萧山货23765 | 1 | 4718 | middle | easy | single | single | 0.999999 | 1.000000 |
| 浙萧山货23765 | 4180 | 4718 | middle | easy | single | single | 0.999998 | 1.000000 |
| 六安港LUANGANG | 12873 | 942 | easy | easy | vertical | vertical | 1.000000 | 1.000000 |
| 六安港LUANGANG | 13226 | 942 | easy | easy | vertical | vertical | 1.000000 | 1.000000 |
| 六安港LUANGANG | 13311 | 942 | easy | easy | vertical | vertical | 1.000000 | 1.000000 |

### hard_to_easy pair 示例

| label | lq_index | hq_index | lq_quality | hq_quality | lq_structure | hq_structure | lq_confidence | hq_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 六安港LUANGANG | 20719 | 942 | hard | easy | vertical | vertical | 1.000000 | 1.000000 |
| 六安港LUANGANG | 26766 | 942 | hard | easy | vertical | vertical | 1.000000 | 1.000000 |
| 六安港LUANGANG | 7916 | 942 | hard | easy | vertical | vertical | 1.000000 | 1.000000 |
| 皖金舵2297WANJINDUO | 6 | 14894 | hard | easy | multi | multi | 1.000000 | 1.000000 |
| 芜湖WUHU | 25077 | 531 | hard | easy | vertical | vertical | 1.000000 | 1.000000 |

### cross_structure pair 示例

| label | lq_index | hq_index | lq_quality | hq_quality | lq_structure | hq_structure | lq_confidence | hq_confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 六安港LUANGANG | 25587 | 942 | middle | easy | single | vertical | 1.000000 | 1.000000 |
| 志远08 | 19470 | 20223 | middle | easy | multi | single | 1.000000 | 1.000000 |
| 志远08 | 4 | 20223 | middle | easy | multi | single | 1.000000 | 1.000000 |
| 志远08 | 3627 | 20223 | middle | easy | multi | single | 1.000000 | 1.000000 |
| 志远08 | 11574 | 20223 | middle | easy | multi | single | 1.000000 | 1.000000 |

## 13. 警告与限制

- 当前 top1-HQ 选择使用 `quality priority + OCR confidence`。
- OCR confidence 来自 baseline OCR，会引入 baseline 偏置。
- 同 label pair 不一定是像素对齐 pair。
- cross_structure pair 可能包含布局差异，后续 Diffusion Loss 训练需谨慎分析。


## 14. 下一步建议

下一步可基于 `pair_manifest_top1_hq.csv` 做 pair 质量可视化抽样与统计确认，然后再进入 conditional latent DiT 的 Stage 1 diffusion-loss 预训练准备。
