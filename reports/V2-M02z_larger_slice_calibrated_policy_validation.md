# V2-M02z Larger Slice Calibrated Policy Validation

`analysis split only`  
`not official benchmark conclusion`

## 1. Summary

V2-M02y 的 best deployable policy 已在 `outputs/V2-M02w0_cache_main_split_incorrect/eval` 上成功复现，口径与上一阶段一致。随后从 `test/unified_lmdb` 导出一个更大的 `full_incorrect` analysis cache，并离线切出 `replace_dominant / low_conf / replace_only / long_21plus / OOV / hard_slice / confusion_pair` 子集。

主结论是：`low_conf_or_confusion` calibrated policy 在更大的 held-out analysis slices 上不是偶然有效。它在 `full_incorrect` 和 `replace_dominant` 两个主 slice 上都显著优于 default `low_conf` policy，同时在 `replace_only / OOV / confusion_pair` 等 slice 上保持正增益。最好结果出现在 `replace_only`，`correction_rate=0.3865`，`corrected_accuracy gain=+0.3333`。

风险上，`full_incorrect` 的 `harmful_change_rate=0.0117`，仍在 `<=0.02` 范围内；但更极端的 `hard_slice` 为 `0.0357`，`low_conf` slice 为 `0.5000`。其中 `low_conf` 的异常高 harmful 来自该 slice 只有 2 个 correct-context 样本，1 个被破坏就会把该比率抬到 0.5，不代表大规模退化，但它仍然说明该策略在极端 slice 上还不够稳。

因此当前不建议直接把这一版 calibrated policy 视为正式 benchmark 默认策略；但它已经足够证明 corrector 路线在 larger held-out analysis 上仍有价值。下一步更合理的是继续做 `candidate top-1 ranking / confidence calibration`，而不是回到 selected-mask coverage。

## 2. Files Added / Modified

- Modified: `ocr_training/tools/eval_mdiff_corrector_offline.py`
- Modified: `ocr_training/tools/export_parseq_corrector_cache.py`
- Added: `ocr_training/tools/filter_mdiff_corrector_cache.py`
- Generated: `ocr_training/outputs/V2-M02z_*`
- Added: `reports/V2-M02z_larger_slice_calibrated_policy_validation.md`

确认：

- `configs/main.yaml` 未修改。
- `maevit_infonce_plm` baseline forward 默认行为未修改。
- 未进入 V2-M03。
- 未实现 insert/delete correction。
- 未新增 encoder_memory branch。

## 3. Data Slices

| slice | path | filter_mode | total_exported | replace_positions | correct_context | metadata_available |
| ----- | ---- | ----------- | -------------: | ----------------: | --------------: | ------------------ |
| full_incorrect | `outputs/V2-M02z_cache_full_incorrect` | `incorrect` | 1731 | 4632 | 600 | true |
| replace_dominant | `outputs/V2-M02z_cache_replace_dominant` | `replace_dominant` | 632 | 1850 | 0 | true |
| low_conf | `outputs/V2-M02z_cache_low_conf` | `low_conf` | 338 | 1024 | 2 | true |
| replace_only | `outputs/V2-M02z_cache_replace_only` | `replace_only` | 459 | 921 | 0 | true |
| long_21plus | `outputs/V2-M02z_cache_long_21plus` | `incorrect` + `min_length=21` | 343 | 683 | 0 | true |
| OOV | `outputs/V2-M02z_cache_oov` | `incorrect` + `vocabulary_type=OOV` | 618 | 1228 | 0 | true |
| hard_slice | `outputs/V2-M02z_cache_hard_slice` | `hard_slice` | 1200 | 2259 | 84 | true |
| confusion_pair | `outputs/V2-M02z_cache_confusion_pair` | `confusion_pair` | 804 | 2270 | 0 | true |

说明：

- 所有 larger slices 都来自 `test/unified_lmdb` 的 analysis cache 构造，结论仅用于 analysis validation。
- `metadata_available=true`，可做 `quality / vocabulary_type / structure_type / resolution_type / length_bucket` 子集分析。
- confusion table 来源为 `outputs/V2-M02w1_confusion_table/confusion_table.json`。

## 4. Default Reproduction

先在 V2-M02w0 主 eval cache 上复现 V2-M02y calibrated policy，确认 evaluator 口径未漂移。

| cache | policy | correction_rate | corrected_acc | harmful_change_rate |
| ----- | ------ | --------------: | ------------: | ------------------: |
| `outputs/V2-M02w0_cache_main_split_incorrect/eval` | `low_conf_or_confusion` (`tau_low=0.90`, `tau_corr=0.30`, `tau_keep=0.90`, `delta_gain=-0.10`) | 0.0561 | 0.3815 | 0.0000 |

复现结果与 V2-M02y 一致，可继续做 larger slice validation。

## 5. Slice Evaluation Results

| slice | policy | samples | replace_positions | baseline_acc | corrected_acc | gain | correction_rate | preservation_rate | harmful_change_rate | oracle@1 | oracle@5 | changed_tokens |
| ----- | ------ | ------: | ----------------: | -----------: | ------------: | ---: | --------------: | ----------------: | ------------------: | -------: | -------: | -------------: |
| full_incorrect | default `low_conf` | 1731 | 2270 | 0.3466 | 0.3668 | +0.0202 | 0.0348 | 0.9995 | 0.0017 | 0.3022 | 0.5890 | 107 |
| full_incorrect | calibrated `low_conf_or_confusion` | 1731 | 2270 | 0.3466 | 0.4310 | +0.0843 | 0.1727 | 0.9941 | 0.0117 | 0.3022 | 0.5877 | 695 |
| replace_dominant | default `low_conf` | 632 | 1850 | 0.0000 | 0.0570 | +0.0570 | 0.0416 | 0.9994 | 0.0000 | 0.3535 | 0.6486 | 97 |
| replace_dominant | calibrated `low_conf_or_confusion` | 632 | 1850 | 0.0000 | 0.2421 | +0.2421 | 0.2038 | 0.9904 | 0.0000 | 0.3535 | 0.6476 | 565 |
| low_conf | calibrated `low_conf_or_confusion` | 338 | 1024 | 0.0059 | 0.2249 | +0.2189 | 0.2119 | 0.9803 | 0.5000 | 0.3076 | 0.5918 | 404 |
| replace_only | calibrated `low_conf_or_confusion` | 459 | 921 | 0.0000 | 0.3333 | +0.3333 | 0.3865 | 0.9934 | 0.0000 | 0.6363 | 0.9164 | 452 |
| long_21plus | calibrated `low_conf_or_confusion` | 343 | 683 | 0.0000 | 0.0816 | +0.0816 | 0.1127 | 0.9929 | 0.0000 | 0.1859 | 0.5212 | 174 |
| OOV | calibrated `low_conf_or_confusion` | 618 | 1228 | 0.0000 | 0.1246 | +0.1246 | 0.1694 | 0.9905 | 0.0000 | 0.3021 | 0.5733 | 372 |
| hard_slice | calibrated `low_conf_or_confusion` | 1200 | 2259 | 0.0700 | 0.1900 | +0.1200 | 0.1709 | 0.9917 | 0.0357 | 0.2988 | 0.5857 | 684 |
| confusion_pair | calibrated `low_conf_or_confusion` | 804 | 2270 | 0.0000 | 0.1903 | +0.1903 | 0.1727 | 0.9890 | 0.0000 | 0.3022 | 0.5877 | 656 |

观察：

- calibrated policy 在两个主对照 slice 上都显著优于 default：
  - `full_incorrect`: `correction_rate 0.0348 -> 0.1727`
  - `replace_dominant`: `correction_rate 0.0416 -> 0.2038`
- `replace_only` 是最强 slice，说明当前 corrector 的核心能力仍然集中在 replace 错误。
- `long_21plus` 仍有效，但明显弱于短串和 cleaner slices。
- `hard_slice` 超过 harmful 阈值，说明 hardest slice 上还不够稳。

## 6. Pair-level Results

| slice | pair | support | corrected | correction_rate | oracle@1 | oracle@5 |
| ----- | ---- | ------: | --------: | --------------: | -------: | -------: |
| full_incorrect | `6->8` | 50 | 37 | 0.7400 | 0.9000 | 0.9800 |
| full_incorrect | `8->6` | 42 | 10 | 0.2381 | 0.3571 | 0.9286 |
| full_incorrect | `8->9` | 39 | 3 | 0.0769 | 0.1026 | 0.7436 |
| full_incorrect | `9->8` | 30 | 19 | 0.6333 | 0.7667 | 1.0000 |
| full_incorrect | `1->0` | 20 | 11 | 0.5500 | 0.7500 | 1.0000 |
| full_incorrect | `8->0` | 20 | 8 | 0.4000 | 0.6000 | 1.0000 |
| full_incorrect | `6->0` | 20 | 10 | 0.5000 | 0.6000 | 0.9500 |
| full_incorrect | `O->A` | 13 | 0 | 0.0000 | 0.0769 | 0.7692 |
| replace_dominant | `6->8` | 45 | 33 | 0.7333 | 0.8889 | 0.9778 |
| replace_dominant | `8->6` | 36 | 10 | 0.2778 | 0.4167 | 0.9167 |
| replace_dominant | `8->9` | 34 | 3 | 0.0882 | 0.1176 | 0.7647 |
| replace_dominant | `9->8` | 26 | 17 | 0.6538 | 0.8077 | 1.0000 |
| replace_dominant | `1->0` | 20 | 11 | 0.5500 | 0.7500 | 1.0000 |
| replace_dominant | `8->0` | 18 | 7 | 0.3889 | 0.6111 | 1.0000 |
| replace_dominant | `6->0` | 18 | 10 | 0.5556 | 0.6667 | 1.0000 |
| replace_dominant | `O->A` | 12 | 0 | 0.0000 | 0.0833 | 0.8333 |
| replace_only | `6->8` | 37 | 31 | 0.8378 | 0.9189 | 0.9730 |
| replace_only | `8->6` | 32 | 10 | 0.3125 | 0.4688 | 1.0000 |
| replace_only | `8->9` | 25 | 3 | 0.1200 | 0.1600 | 0.8800 |
| replace_only | `9->8` | 20 | 17 | 0.8500 | 0.9500 | 1.0000 |
| replace_only | `1->0` | 17 | 9 | 0.5294 | 0.7059 | 1.0000 |
| replace_only | `8->0` | 14 | 7 | 0.5000 | 0.7143 | 1.0000 |
| replace_only | `6->0` | 13 | 9 | 0.6923 | 0.8462 | 1.0000 |
| replace_only | `O->A` | 1 | 0 | 0.0000 | 1.0000 | 1.0000 |

相对 default 的主 slice 改善最明显的是：

- `full_incorrect 6->8`: `0.2200 -> 0.7400`
- `full_incorrect 9->8`: `0.1000 -> 0.6333`
- `full_incorrect 6->0`: `0.1500 -> 0.5000`
- `replace_dominant 6->8`: `0.2222 -> 0.7333`
- `replace_dominant 9->8`: `0.1154 -> 0.6538`

结论：

- 高频数字混淆对 `6/8/9/1/0` 持续受益，尤其 `6->8 / 9->8 / 1->0 / 6->0`。
- `8->9` 仍弱，`O->A` 基本没有转化为实际修正，尽管 `oracle@5` 并不低，说明剩余瓶颈已偏向 ranking/calibration，而不是 mask coverage。

## 7. Subset / Metadata Analysis

metadata 可用，按 `full_incorrect` 内部子集统计如下：

| subset | sample_count | gain_accuracy | correction_rate | harmful_change_rate |
| ------ | -----------: | ------------: | --------------: | ------------------: |
| `vocabulary_type=IV` | 1113 | +0.0620 | 0.1766 | 0.0117 |
| `vocabulary_type=OOV` | 618 | +0.1246 | 0.1694 | 0.0000 |
| `quality=easy` | 690 | +0.0203 | 0.2624 | 0.0117 |
| `quality=hard` | 353 | +0.1048 | 0.1443 | 0.0000 |
| `quality=middle` | 688 | +0.1381 | 0.1835 | 0.0000 |
| `length_bucket=long_21plus` | 425 | +0.0612 | 0.1127 | 0.0244 |
| `structure_type=single_line` | 519 | +0.1387 | 0.2407 | 0.0189 |
| `structure_type=multi_lines` | 1174 | +0.0622 | 0.1452 | 0.0091 |
| `structure_type=vertical` | 38 | +0.0263 | 0.1111 | 0.0000 |

结论：

- `OOV` 并未失效，gain 仍为正，说明 calibrated policy 不只是在 IV 上工作。
- `long_21plus` 仍有效，但 correction 明显下降，且 harmful 超过 0.02，长串是明确弱点。
- `single_line` 最友好，`multi_lines` 和 `vertical` 更难，符合多错误耦合预期。
- `low_conf_sample=true` 的 gain 很高，但该子集 correct-context 仅 2 个，harmful ratio 会被单个样本放大，不适合单独当作稳定性结论。

## 8. Case Study

### Case 1: default 不改，calibrated 成功修对

- `sample_id`: `unified_lmdb:000000015`
- GT: `浙余杭货02039ZHEYUHANGHUO`
- baseline pred: `浙余杭货02839ZHEYUHANGHUO`
- default corrected: `浙余杭货02839ZHEYUHANGHUO`
- calibrated corrected: `浙余杭货02039ZHEYUHANGHUO`
- changed positions: `[6]`
- base_conf: `[0.9956]`
- corr_conf: `[0.9797]`
- success/failure reason: 高频 `8->0` confusion，被 confusion prior 放行；default `low_conf` 因高置信错误漏选。

### Case 2: 高频数字混淆继续受益

- `sample_id`: `unified_lmdb:000000456`
- GT: `兴航8888`
- baseline pred: `兴航8838`
- default corrected: `兴航8838`
- calibrated corrected: `兴航8888`
- changed positions: `[4]`
- base_conf: `[0.8838]`
- corr_conf: `[0.9949]`
- success/failure reason: `3->8` 在 calibrated policy 下被放行并成功修正，说明增益不只来自低置信位。

### Case 3: calibrated harmful change

- `sample_id`: `unified_lmdb:000000626`
- GT: `苏盐货41128`
- baseline pred: `苏盐货11128`
- default corrected: `苏盐货11128`
- calibrated corrected: `苏盐货11228`
- changed positions: `[5]`
- base_conf: `[0.9927]`
- corr_conf: `[0.8974]`
- success/failure reason: 原本正确位置 `1` 被改成 `2`，属于 calibrated 过松导致的 correct-context 破坏。

### Case 4: replace 修对了，但 sample 仍然错，因为剩余 insert/delete 没法处理

- `sample_id`: `unified_lmdb:000001558`
- GT: `浙越城货0078绍兴港`
- baseline pred: `浙越城货0076绍兴港ZHEYUECHENGHUOSHAOXINGGANG`
- default corrected: `浙越城货0076绍兴港ZHEYUECHENGHUOSHAOXINGGANG`
- calibrated corrected: `浙越城货0078绍兴港ZHEYUECHENGHUOSHAOXINGGANG`
- changed positions: `[7]`
- base_conf: `[0.9424]`
- corr_conf: `[0.9909]`
- success/failure reason: `6->8` replacement 修对，但样本仍残留 26 个 delete 对齐问题；这是 replace-only 范围限制，不是 selected-mask 问题。

### Case 5: oracle@K contains GT，但仍然不改

- `sample_id`: `unified_lmdb:000000037`
- GT: `芜湖海联5166WUHUHAILIAN`
- baseline pred: `芜湖51666芜湖WUHULIAN`
- default corrected: `芜湖51666芜湖WUHULIAN`
- calibrated corrected: `芜湖51666芜湖WUHULIAN`
- changed positions: `[]`
- base_conf: `[]`
- corr_conf: `[]`
- success/failure reason: 多个位置的 GT 出现在 `corr_top5`，例如 `W->H` 位置 `top5=W,H,Z,Y,S`，但 `top1` 仍是 baseline token，导致 oracle 可达但实际无改动。

### Case 6: calibrated 既修对一个 replace，又额外引入 harmful

- `sample_id`: `unified_lmdb:000001296`
- GT: `浙余杭货02078ZHEYUHANGHUO`
- baseline pred: `浙余杭货02878ZHEYUHANGHUO`
- default corrected: `浙余杭货02878ZHEYUHANGHUO`
- calibrated corrected: `浙余杭货02008ZHEYUHANGHUO`
- changed positions: `[6, 7]`
- base_conf: `[0.9941, 0.7090]`
- corr_conf: `[0.9023, 0.5462]`
- success/failure reason: position 6 的 `8->0` 修对，但 position 7 的正确 `7` 被错改为 `0`；这是 hardest slice 上 pair-extraneous activation 的典型失败样本。

## 9. Analysis

### 9.1 calibrated policy 是否稳定泛化

是，但稳定性是有边界的。

- 在两条主验证线 `full_incorrect` 和 `replace_dominant` 上，calibrated policy 都显著优于 default。
- 在 `replace_only / OOV / confusion_pair / long_21plus` 上也都保持正 gain。
- 但 `hard_slice` 的 `harmful_change_rate=0.0357`，超过目标阈值；`low_conf` slice 的 harmful 也不稳。

因此更准确的表述是：它在 larger held-out analysis slices 上**不是偶然有效**，但还**不是无条件稳定**。

### 9.2 哪些 slice 效果最好，哪些最差

最好：

- `replace_only`: `correction_rate=0.3865`, `gain=+0.3333`
- `replace_dominant`: `correction_rate=0.2038`, `gain=+0.2421`

较差：

- `long_21plus`: `correction_rate=0.1127`
- `hard_slice`: `harmful_change_rate=0.0357`

这说明当前系统最适合“replace 为主、长度适中、候选空间较干净”的错误；一旦进入长串、多错误耦合、insert/delete 干扰更重的样本，收益会下降。

### 9.3 harmful 是否仍可控

部分可控，不是完全可控。

- `full_incorrect=0.0117`，可接受。
- `replace_dominant / replace_only / OOV / confusion_pair = 0.0`，很好。
- `hard_slice=0.0357`，不可接受。
- `low_conf=0.5000` 虽然绝对分母只有 2 个 correct-context 样本，但从指标定义上看仍然失败。

因此 calibrated policy 还不能在 hardest slices 上宣称“稳健无害”。

### 9.4 高频 confusion pair 是否持续受益

是。

- `6->8 / 9->8 / 1->0 / 6->0` 在多个 slice 上持续提升。
- `8->6` 也有提升，但仍明显低于 `6->8`。
- `8->9` 和 `O->A` 仍是弱项，尤其 `O->A` 说明 alphabet-side ranking 还不够好。

### 9.5 当前主要瓶颈

已经不是 selected-mask coverage。

证据：

- `selected_wrong_coverage` 在多数 slice 上接近饱和，通常在 `0.99` 左右。
- 但 `oracle@1` 与 `correction_rate` 之间仍存在明显 gap，例如：
  - `full_incorrect`: `0.3022 - 0.1727 = 0.1295`
  - `replace_only`: `0.6363 - 0.3865 = 0.2497`
- `O->A`、`8->9` 等 pair 的 `oracle@5` 高而 `correction_rate` 低。

因此当前主瓶颈是：

1. `candidate top-1 ranking`
2. `confidence calibration`
3. `replace-only` 范围对 insert/delete-heavy 样本的天然上限

### 9.6 是否可以把 calibrated policy 作为后续默认 inference 策略

对 replace-oriented analysis，可以。

- 它明显优于 default `low_conf`，且在主 slice 上 harmful 仍可控。

对更正式 benchmark-style validation，还不能直接固定为最终默认。

- 原因是 `hard_slice` 和极端 `low_conf` slice 还没有稳定满足 `harmful_change_rate <= 0.02`。

## 10. Recommendation

建议下一步做 `V2-M02w2: pair-specific synthetic noise / loss calibration`，而不是立刻进入更正式 benchmark 结论。

原因：

- `selected-mask coverage` 已基本到位，不再是主矛盾。
- 当前主要损失来自 `top-1 ranking / confidence calibration`。
- `hard_slice` 和长串场景还存在明显弱点。
- 等 ranking/calibration 再推进一轮后，再做 `V2-M02zz` 的 untouched holdout benchmark-style validation 更稳妥。

结论性建议：

- `low_conf_or_confusion` 可以保留为后续 corrector analysis 的默认 deployable policy。
- 但在正式 benchmark-style validation 前，应先继续优化 `pair-specific candidate ranking`，并关注 `8->9 / O->A / long_21plus / insert-delete-heavy` 场景。
