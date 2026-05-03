# V2-M02w0 Expanded Real-error Cache

## 1. Summary

V2-M02w-0 的目标是扩大 real-error cache，并建立比 V2-M02v 更稳定的 train/eval split，不训练 corrector。

本阶段结论：

- 成功扩大了 real-error cache。
- `train` split 仍然不是高性价比来源。此前 `train` 前 2000 样本全对，本阶段再次尝试大范围 `train replace_only` 扫描，但在低错误密度下吞吐代价过高，因此没有继续把它作为主 cache。
- 最终选择 `test/unified_lmdb` 的 `incorrect` cache 作为主 analysis cache。
- 主 cache 的 `80/20` split 成功，且 `sample_id overlap = 0`。
- 如果以 corrector 真正关心的 `replace positions` 计数，主 split 达到：
  - train `replace_positions = 1789`
  - eval `replace_positions = 481`
  - eval `samples = 346`
- 已达到最低可用规模，可以进入 `V2-M02w-1` 做 confusion-aware synthetic noise，但必须明确标注：
  - `analysis split only`
  - `not official benchmark conclusion`

额外观察：

- 更干净的 `replace_dominant` unified cache 也已导出成功，但其 held-out eval `replace_positions = 186`，略低于最低门槛 `200`，因此更适合作为辅线或后续 cleaner subset，而不是本阶段主 cache。

## 2. Files Added / Modified

V2-M02w-0 本阶段未新增或修改核心代码，直接复用已有脚本：

- `ocr_training/tools/export_parseq_corrector_cache.py`
- `ocr_training/tools/split_mdiff_corrector_cache.py`
- `ocr_training/tools/mdiff_corrector_utils.py`
- `ocr_training/tools/eval_mdiff_corrector_offline.py`

本阶段新增：

- `reports/V2-M02w0_expanded_real_error_cache.md`

兼容性确认：

- `ocr_training/configs/main.yaml` 未修改。
- baseline `forward()` 默认行为未改变。
- 未训练 corrector。
- 未进入 V2-M03。

## 3. Export Attempts

说明：

- `train replace_only 30000` 进行了启动尝试，但在低错误密度下扫描代价过高，未等待其完成，因此不作为正式统计结果。
- 下表仅列出有完整导出 summary 的 cache。

| cache | split | filter_mode | scan_limit | max_export | total_scanned | total_exported | baseline_incorrect | replace_only | replace_dominant | wrong_positions | correct_context |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V2-M02w0_cache_test_unified_replace_only_bs64` | `test/unified_lmdb` | `replace_only` | 5000 | 1000 | 5000 | 433 | 502 | 233 | 300 | 1987 | 200 |
| `V2-M02w0_cache_test_unified_replace_dominant_bs64` | `test/unified_lmdb` | `replace_dominant` | 5000 | 2000 | 5000 | 633 | 502 | 233 | 300 | 1987 | 400 |
| `V2-M02w0_cache_test_unified_replace_dominant_bs64_scan12000` | `test/unified_lmdb` | `replace_dominant` | 12000 | 3000 | 6884 | 1059 | 1131 | 459 | 632 | 4632 | 600 |
| `V2-M02w0_cache_test_unified_incorrect_bs64` | `test/unified_lmdb` | `incorrect` | 12000 | 3000 | 6884 | 1731 | 1131 | 459 | 632 | 4632 | 600 |

补充：

- `test/unified_lmdb` 实际数据量为 `6884`，因此 `scan_limit=12000` 最终等价于“扫描完整 unified”。
- `wrong_positions` 来自 exporter summary，用于表示整轮扫描中对齐得到的真实错误位总量。

## 4. Selected Main Cache

主 cache 选择：

- path: `ocr_training/outputs/V2-M02w0_cache_test_unified_incorrect_bs64`

选择原因：

1. 是本阶段唯一明确达到最低规模要求的 cache。
2. split 后 `eval samples` 和 `replace positions` 都足够稳定。
3. 保留了 `600` 个 correct-context samples，适合后续 preservation / harmful change 评估。
4. 虽然它不是最干净的子集，但对于 V2-M02w-1 来说可以：
   - 继续只在 `replace positions` 上训练
   - 或者再从该大 cache 内部派生 cleaner subset

主 cache 关键统计：

- total_exported: `1731`
- baseline_incorrect_samples: `1131`
- correct_context_samples: `600`
- scanned wrong positions: `4632`
- scanned correct positions: `90487`
- top replace pairs:
  - `6->8: 50`
  - `8->6: 42`
  - `8->9: 39`
  - `9->8: 30`
  - `8->0: 20`
  - `6->0: 20`
  - `9->0: 20`
  - `1->0: 20`
  - `6->5: 19`
  - `0->9: 18`
- confidence stats:
  - `correct_token_conf_mean = 0.9985`
  - `wrong_token_conf_mean = 0.9531`
  - `low_conf_token_count = 469`
- length stats:
  - `gt_len_mean = 13.77`
  - `pred_len_mean = 13.82`
  - `long_21plus_count = 1243`

插删比例：

- `replace = 2270`
- `insert = 2058`
- `delete = 2362`

判断：

- 该 cache 不能被直接当作“纯 replace training set”使用；
- 但它非常适合作为更大、更稳定的 analysis / source cache；
- V2-M02w-1 训练时应继续使用 `replace positions` 监督，或从其中再切出 cleaner subset。

## 5. Train/Eval Split Summary

主 split 路径：

- `ocr_training/outputs/V2-M02w0_cache_main_split_incorrect`

split 参数：

- train_ratio: `0.8`
- seed: `2026`
- stratify: `sample_type`
- sample_id overlap: `0`

这里的 `wrong_positions` 使用对 split cache 重新做 alignment 后的 `replace_count` 统计，因为这更贴近 MDiff corrector 当前仅处理 replace correction 的设定。

| split | samples | wrong_positions | correct_context_samples | originally_correct_positions | originally_wrong_positions |
|---|---:|---:|---:|---:|---:|
| train | 1385 | 1789 | 480 | 15966 | 1789 |
| eval | 346 | 481 | 120 | 3989 | 481 |

附加 sanity：

- train features: `pred_token_ids [1385, 51]`, `decoder_hidden [1385, 51, 768]`
- eval features: `pred_token_ids [346, 51]`, `decoder_hidden [346, 51, 768]`
- split 后 train/eval cache 已确认可被后续工具正常读取

对照的 cleaner split：

- `ocr_training/outputs/V2-M02w0_cache_main_split_scan12000`
- `replace_dominant` split 统计：
  - train samples: `847`
  - eval samples: `212`
  - train replace_positions: `735`
  - eval replace_positions: `186`

结论：

- `replace_dominant` 很接近门槛，但 held-out eval `186 < 200`
- 因此保留为 cleaner auxiliary cache，不作为本阶段主 cache

## 6. Data Sufficiency Assessment

最低标准要求：

- eval samples `>= 100`
- eval wrong positions `>= 200`
- train wrong positions `>= 500`
- correct-context `>= 10%`

主 cache split 结果：

- eval samples: `346`
- eval replace_positions: `481`
- train replace_positions: `1789`
- correct-context ratio: `600 / 1731 = 34.7%`

判断：

- 已达到最低可用规模
- 也达到“较可靠规模”的核心 wrong-position 指标：
  - eval wrong positions `481 < 500`，略低于理想中的较可靠 eval wrong `500`
  - train wrong positions `1789 >= 1500`

因此：

- 可以继续做 V2-M02w-1 正式 confusion-aware synthetic noise 对照
- 但仍应明确标注这是 `analysis split only`

## 7. Confusion Statistics

以下 confusion table 只从主 split 的 `train` 部分构建，`eval` 未参与构建：

- path:
  - `ocr_training/outputs/V2-M02w0_confusion_table_main_train/confusion_table.json`
  - `ocr_training/outputs/V2-M02w0_confusion_table_main_train/confusion_table.csv`

Top 20 replace pairs：

| pred_token | gt_token | count | avg_base_conf | avg_position |
|---|---:|---:|---:|---:|
| `6` | `8` | 44 | 0.8343 | 5.59 |
| `8` | `6` | 34 | 0.8986 | 6.06 |
| `8` | `9` | 30 | 0.8395 | 5.93 |
| `9` | `8` | 27 | 0.8179 | 5.67 |
| `8` | `0` | 17 | 0.9017 | 6.12 |
| `1` | `0` | 15 | 0.7209 | 5.47 |
| `8` | `1` | 15 | 0.8599 | 5.13 |
| `6` | `0` | 14 | 0.7753 | 5.93 |
| `0` | `2` | 14 | 0.8571 | 5.29 |
| `0` | `1` | 14 | 0.8756 | 5.07 |
| `8` | `3` | 14 | 0.9086 | 6.00 |
| `6` | `5` | 13 | 0.9007 | 6.23 |
| `O` | `A` | 13 | 0.9937 | 13.46 |
| `0` | `9` | 13 | 0.8625 | 5.15 |
| `0` | `6` | 13 | 0.8288 | 5.46 |
| `8` | `5` | 13 | 0.8929 | 6.46 |
| `5` | `6` | 12 | 0.8278 | 5.33 |
| `1` | `2` | 12 | 0.7884 | 4.67 |
| `7` | `1` | 11 | 0.8238 | 5.91 |
| `5` | `9` | 11 | 0.7910 | 5.73 |

这些 pair 与 V2-M02u / V2-M02v 的观察高度一致：

- 数字视觉混淆仍然占主导：`6/8/9/1/0/5/3/2/7`
- 字母混淆 `O -> A` 也依然出现
- 高频错误主要集中在中间数字位，`avg_position` 多数落在 `5~6`

## 8. Recommendation

建议进入 `V2-M02w-1`，但要明确采用以下数据策略：

1. 主 source cache 使用：
   - `outputs/V2-M02w0_cache_main_split_incorrect`
2. 训练监督继续只作用于 `replace positions`
3. `replace_dominant` split 保留为 cleaner 对照或辅助子集：
   - `outputs/V2-M02w0_cache_main_split_scan12000`
4. confusion-aware synthetic noise 的 confusion table 只从主 split 的 `train` 构建
5. 结果表述必须继续保持：
   - `analysis split only`
   - `not official benchmark conclusion`

不建议：

- 直接进入 V2-M03
- 直接把整个 incorrect cache 当成“无过滤训练目标”
- 在本阶段继续追求 full benchmark 结论
