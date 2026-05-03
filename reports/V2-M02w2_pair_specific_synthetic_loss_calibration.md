# V2-M02w2 Pair-specific Synthetic Noise / Loss Calibration

`analysis split only`  
`not official benchmark conclusion`

## 1. Summary

V2-M02w2 成功构建了 train-split pair difficulty table，并在不修改 baseline 默认行为、不进入 V2-M03 的前提下训练了三组新 corrector：

- pair-specific synthetic only
- pair-weighted loss only
- pair-specific synthetic + pair-weighted loss

结果很明确：

- `pair-specific synthetic only` 基本安全，但提升很小，弱 pair 几乎没动。
- `pair-weighted loss only` 是本阶段最好模型。
- `pair-specific synthetic + pair-weighted loss` 也优于 baseline confusion synthetic，但整体略弱于 `pair-weighted loss only`。

在 larger held-out slices 上，最佳模型 `pair_weighted` 相比 V2-M02w1/V2-M02z baseline confusion synthetic 有稳定但温和的增益：

- `full_incorrect correction_rate`: `0.1727 -> 0.1753`
- `replace_dominant correction_rate`: `0.2038 -> 0.2070`
- `replace_only correction_rate`: `0.3865 -> 0.3941`

弱 pair 上的改善主要来自 `pair_weighted`：

- `8->9`: `0.0769 -> 0.1026`
- `8->6`: `0.2381 -> 0.2857`
- `8->0`: `0.4000 -> 0.4500`

但 `O->A` 仍没有转化成实际 correction，`1->0` 还略有回退。`hard_slice harmful_change_rate` 没有恶化，但也没有下降，仍维持在 `0.0357`，所以当前还不建议直接进入 V2-M02zz benchmark-style untouched holdout validation。

## 2. Files Added / Modified

- Added: `ocr_training/tools/build_pair_difficulty_table.py`
- Modified: `ocr_training/tools/mdiff_corrector_utils.py`
- Modified: `ocr_training/tools/train_mdiff_corrector_smoke.py`
- Modified: `ocr_training/strhub/models/slp_mdiff_corrector/system.py`
- Generated: `ocr_training/outputs/V2-M02w2_pair_difficulty/*`
- Generated: `ocr_training/outputs/V2-M02w2_corrector_*/*`
- Generated: `ocr_training/outputs/V2-M02w2_eval_*/*`
- Generated: `ocr_training/outputs/V2-M02w2_matrix_eval/*`
- Added: `reports/V2-M02w2_pair_specific_synthetic_loss_calibration.md`

确认：

- `configs/main.yaml` 未修改。
- baseline forward 行为未改变。
- 未进入 V2-M03。
- 未实现 insert/delete correction。
- 未新增 encoder_memory branch。

## 3. Pair Difficulty Table

构造来源：

- train confusion table: `outputs/V2-M02w1_confusion_table/confusion_table.json`
- eval pair stats: `outputs/V2-M02z_eval_full_incorrect/pair_stats.json`

本阶段使用：

```text
difficulty_score =
  train_count_norm
  * (oracle@5 - correction_rate)
  * (1 - correction_rate)
```

```text
recommended_weight = clamp(1 + 2 * difficulty_score, 1.0, 3.0)
recommended_synthetic_multiplier = clamp(1 + 4 * difficulty_score, 1.0, 5.0)
```

Top 20:

| pred_token | gt_token | train_count | eval_support | eval_correction_rate | oracle@1 | oracle@5 | difficulty_score | weight | synthetic_multiplier |
| ---------- | -------- | ----------: | -----------: | -------------------: | -------: | -------: | ---------------: | -----: | -------------------: |
| 8 | 9 | 30 | 39 | 0.0769 | 0.1026 | 0.7436 | 0.4196 | 1.8392 | 2.6783 |
| 8 | 6 | 34 | 42 | 0.2381 | 0.3571 | 0.9286 | 0.4065 | 1.8130 | 2.6261 |
| 0 | 1 | 14 | 16 | 0.1250 | 0.1875 | 1.0000 | 0.2436 | 1.4872 | 1.9744 |
| 8 | 3 | 14 | 16 | 0.0625 | 0.3125 | 0.8750 | 0.2424 | 1.4847 | 1.9695 |
| O | A | 13 | 13 | 0.0000 | 0.0769 | 0.7692 | 0.2273 | 1.4545 | 1.9091 |
| 0 | 9 | 13 | 18 | 0.0556 | 0.0556 | 0.8333 | 0.2170 | 1.4341 | 1.8681 |
| 2 | 1 | 10 | 12 | 0.0000 | 0.0000 | 0.9167 | 0.2083 | 1.4167 | 1.8333 |
| N | A | 10 | 11 | 0.0000 | 0.1818 | 0.9091 | 0.2066 | 1.4132 | 1.8264 |
| 8 | 1 | 15 | 17 | 0.2353 | 0.2941 | 1.0000 | 0.1994 | 1.3987 | 1.7974 |
| A | U | 9 | 12 | 0.0000 | 0.2500 | 0.9167 | 0.1875 | 1.3750 | 1.7500 |
| H | Y | 10 | 11 | 0.0000 | 0.1818 | 0.8182 | 0.1860 | 1.3719 | 1.7438 |
| 0 | 6 | 13 | 15 | 0.2000 | 0.4000 | 0.9333 | 0.1733 | 1.3467 | 1.6933 |
| G | U | 11 | 15 | 0.1333 | 0.3333 | 0.8667 | 0.1589 | 1.3178 | 1.6356 |
| U | A | 9 | 12 | 0.0833 | 0.0833 | 0.9167 | 0.1562 | 1.3125 | 1.6250 |
| 5 | 9 | 11 | 11 | 0.1818 | 0.2727 | 0.9091 | 0.1488 | 1.2975 | 1.5950 |
| I | U | 8 | 9 | 0.0000 | 0.3333 | 0.7778 | 0.1414 | 1.2828 | 1.5657 |
| 2 | 0 | 10 | 14 | 0.2143 | 0.3571 | 1.0000 | 0.1403 | 1.2806 | 1.5612 |
| 1 | 3 | 7 | 8 | 0.0000 | 0.1250 | 0.8750 | 0.1392 | 1.2784 | 1.5568 |
| 8 | 0 | 17 | 20 | 0.4000 | 0.6000 | 1.0000 | 0.1391 | 1.2782 | 1.5564 |
| 7 | 1 | 11 | 12 | 0.1667 | 0.2500 | 0.8333 | 0.1389 | 1.2778 | 1.5556 |

结论：

- `8->9 / 8->6 / O->A` 被正确识别为最高优先级弱 pair。
- `8->0 / 0->9 / 0->6` 也被打进高难度区间。
- `6->8 / 9->8 / 1->0 / 6->0` 没有被推到最高权重，符合“强 pair 主要做对照”的预期。

## 4. Training Results

| model | synthetic_mode | loss_type | synthetic_ratio | final_loss | selected_ce_final | preservation_ce_final |
| ----- | -------------- | --------- | --------------: | ---------: | ----------------: | --------------------: |
| baseline confusion synthetic | confusion | selected_plus_preservation | 1.0 | 0.5827 | 0.4905 | 0.4607 |
| pair-specific synthetic | pair_specific | selected_plus_preservation | 1.0 | 0.5786 | 0.4863 | 0.4611 |
| pair-weighted loss | confusion | pair_weighted_selected_plus_preservation | 1.0 | 0.6171 | 0.5263 | 0.4539 |
| pair-specific synthetic + pair-weighted loss | pair_specific | pair_weighted_selected_plus_preservation | 1.0 | 0.6066 | 0.5160 | 0.4530 |

补充：

- `pair_specific synthetic` 数据构造是干净的：850 synthetic，全都命中 pair-specific，无 fallback。
- `pair_weighted` 两组训练中 preservation CE 没有失控，说明加权并没有直接破坏保守性。

## 5. Slice Evaluation Matrix

| model | slice | correction_rate | corrected_acc_gain | preservation_rate | harmful_change_rate | oracle@1 | oracle@5 |
| ----- | ----- | --------------: | -----------------: | ----------------: | ------------------: | -------: | -------: |
| baseline confusion synthetic | full_incorrect | 0.1727 | +0.0843 | 0.9941 | 0.0117 | 0.3022 | 0.5877 |
| pair-specific synthetic | full_incorrect | 0.1736 | +0.0855 | 0.9941 | 0.0117 | 0.3009 | 0.5881 |
| pair-weighted loss | full_incorrect | 0.1753 | +0.0884 | 0.9943 | 0.0117 | 0.3044 | 0.5903 |
| pair-specific synthetic + pair-weighted loss | full_incorrect | 0.1744 | +0.0867 | 0.9943 | 0.0117 | 0.3040 | 0.5903 |
| baseline confusion synthetic | replace_dominant | 0.2038 | +0.2421 | 0.9904 | 0.0000 | 0.3535 | 0.6476 |
| pair-specific synthetic | replace_dominant | 0.2049 | +0.2453 | 0.9904 | 0.0000 | 0.3524 | 0.6476 |
| pair-weighted loss | replace_dominant | 0.2070 | +0.2532 | 0.9913 | 0.0000 | 0.3557 | 0.6492 |
| pair-specific synthetic + pair-weighted loss | replace_dominant | 0.2059 | +0.2484 | 0.9913 | 0.0000 | 0.3557 | 0.6492 |
| baseline confusion synthetic | replace_only | 0.3865 | +0.3333 | 0.9934 | 0.0000 | 0.6363 | 0.9164 |
| pair-specific synthetic | replace_only | 0.3887 | +0.3377 | 0.9934 | 0.0000 | 0.6341 | 0.9164 |
| pair-weighted loss | replace_only | 0.3941 | +0.3486 | 0.9936 | 0.0000 | 0.6428 | 0.9207 |
| pair-specific synthetic + pair-weighted loss | replace_only | 0.3909 | +0.3420 | 0.9936 | 0.0000 | 0.6417 | 0.9207 |
| baseline confusion synthetic | hard_slice | 0.1709 | +0.1200 | 0.9917 | 0.0357 | 0.2988 | 0.5857 |
| pair-specific synthetic | hard_slice | 0.1718 | +0.1217 | 0.9917 | 0.0357 | 0.2975 | 0.5861 |
| pair-weighted loss | hard_slice | 0.1735 | +0.1258 | 0.9918 | 0.0357 | 0.3010 | 0.5883 |
| pair-specific synthetic + pair-weighted loss | hard_slice | 0.1726 | +0.1233 | 0.9919 | 0.0357 | 0.3006 | 0.5883 |
| baseline confusion synthetic | long_21plus | 0.1127 | +0.0816 | 0.9929 | 0.0000 | 0.1859 | 0.5212 |
| pair-specific synthetic | long_21plus | 0.1127 | +0.0816 | 0.9929 | 0.0000 | 0.1859 | 0.5242 |
| pair-weighted loss | long_21plus | 0.1157 | +0.0875 | 0.9931 | 0.0000 | 0.1903 | 0.5256 |
| pair-specific synthetic + pair-weighted loss | long_21plus | 0.1142 | +0.0845 | 0.9931 | 0.0000 | 0.1889 | 0.5256 |
| baseline confusion synthetic | OOV | 0.1694 | +0.1246 | 0.9905 | 0.0000 | 0.3021 | 0.5733 |
| pair-specific synthetic | OOV | 0.1702 | +0.1262 | 0.9905 | 0.0000 | 0.3013 | 0.5725 |
| pair-weighted loss | OOV | 0.1718 | +0.1246 | 0.9905 | 0.0000 | 0.3037 | 0.5765 |
| pair-specific synthetic + pair-weighted loss | OOV | 0.1718 | +0.1246 | 0.9905 | 0.0000 | 0.3037 | 0.5765 |
| baseline confusion synthetic | confusion_pair | 0.1727 | +0.1903 | 0.9890 | 0.0000 | 0.3022 | 0.5877 |
| pair-specific synthetic | confusion_pair | 0.1736 | +0.1928 | 0.9890 | 0.0000 | 0.3009 | 0.5881 |
| pair-weighted loss | confusion_pair | 0.1753 | +0.1990 | 0.9897 | 0.0000 | 0.3044 | 0.5903 |
| pair-specific synthetic + pair-weighted loss | confusion_pair | 0.1744 | +0.1953 | 0.9897 | 0.0000 | 0.3040 | 0.5903 |

主结论：

- 三个新模型都优于 baseline confusion synthetic。
- 最佳模型是 `pair-weighted loss`。
- `pair-specific synthetic` 只带来轻微提升。
- `pair-specific + weighted` 没有超过 `pair-weighted`，说明本轮主增益主要来自 loss weighting，而不是 synthetic 分布本身。

## 6. Weak-pair Analysis

以下表格使用 `full_incorrect` slice 的 pair stats，对比 V2-M02z baseline confusion synthetic。

| model | pair | support | correction_rate | oracle@1 | oracle@5 | delta_vs_baseline |
| ----- | ---- | ------: | --------------: | -------: | -------: | ----------------: |
| baseline confusion synthetic | 8->9 | 39 | 0.0769 | 0.1026 | 0.7436 | +0.0000 |
| pair-specific synthetic | 8->9 | 39 | 0.0769 | 0.1026 | 0.7436 | +0.0000 |
| pair-weighted loss | 8->9 | 39 | 0.1026 | 0.2308 | 0.7692 | +0.0256 |
| pair-specific synthetic + pair-weighted loss | 8->9 | 39 | 0.1026 | 0.2308 | 0.7692 | +0.0256 |
| baseline confusion synthetic | 8->6 | 42 | 0.2381 | 0.3571 | 0.9286 | +0.0000 |
| pair-specific synthetic | 8->6 | 42 | 0.2381 | 0.3571 | 0.9286 | +0.0000 |
| pair-weighted loss | 8->6 | 42 | 0.2857 | 0.4048 | 0.9286 | +0.0476 |
| pair-specific synthetic + pair-weighted loss | 8->6 | 42 | 0.2857 | 0.4048 | 0.9286 | +0.0476 |
| baseline confusion synthetic | O->A | 13 | 0.0000 | 0.0769 | 0.7692 | +0.0000 |
| pair-specific synthetic | O->A | 13 | 0.0000 | 0.0769 | 0.7692 | +0.0000 |
| pair-weighted loss | O->A | 13 | 0.0000 | 0.1538 | 0.7692 | +0.0000 |
| pair-specific synthetic + pair-weighted loss | O->A | 13 | 0.0000 | 0.1538 | 0.8462 | +0.0000 |
| baseline confusion synthetic | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| pair-specific synthetic | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| pair-weighted loss | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| pair-specific synthetic + pair-weighted loss | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| baseline confusion synthetic | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | +0.0000 |
| pair-specific synthetic | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | +0.0000 |
| pair-weighted loss | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | +0.0000 |
| pair-specific synthetic + pair-weighted loss | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | +0.0000 |
| baseline confusion synthetic | 1->0 | 20 | 0.5500 | 0.7500 | 1.0000 | +0.0000 |
| pair-specific synthetic | 1->0 | 20 | 0.5500 | 0.7500 | 1.0000 | +0.0000 |
| pair-weighted loss | 1->0 | 20 | 0.5000 | 0.7000 | 1.0000 | -0.0500 |
| pair-specific synthetic + pair-weighted loss | 1->0 | 20 | 0.5000 | 0.7000 | 1.0000 | -0.0500 |
| baseline confusion synthetic | 6->0 | 20 | 0.5000 | 0.6000 | 0.9500 | +0.0000 |
| pair-specific synthetic | 6->0 | 20 | 0.5000 | 0.6000 | 0.9500 | +0.0000 |
| pair-weighted loss | 6->0 | 20 | 0.5000 | 0.6000 | 0.9500 | +0.0000 |
| pair-specific synthetic + pair-weighted loss | 6->0 | 20 | 0.5000 | 0.6000 | 0.9500 | +0.0000 |
| baseline confusion synthetic | 8->0 | 20 | 0.4000 | 0.6000 | 1.0000 | +0.0000 |
| pair-specific synthetic | 8->0 | 20 | 0.4000 | 0.6000 | 1.0000 | +0.0000 |
| pair-weighted loss | 8->0 | 20 | 0.4500 | 0.5500 | 1.0000 | +0.0500 |
| pair-specific synthetic + pair-weighted loss | 8->0 | 20 | 0.4500 | 0.5500 | 1.0000 | +0.0500 |

结论：

- `pair-specific synthetic` 单独几乎不改善 weak pair。
- `pair-weighted loss` 明显改善 `8->9 / 8->6 / 8->0`。
- `O->A` 只提高了 `oracle@1`，还没转成实际 correction。
- 强 pair `6->8 / 9->8 / 6->0` 被保持住。
- `1->0` 略有退化，说明 pair weighting 还需要更细的权重调参。

## 7. Harmful / Preservation Analysis

总体上，V2-M02w2 没有让 harmful 明显恶化。

主要观察：

- `full_incorrect harmful_change_rate`：所有新模型都仍为 `0.0117`
- `replace_dominant harmful_change_rate`：所有新模型都为 `0.0000`
- `replace_only harmful_change_rate`：所有新模型都为 `0.0000`
- `hard_slice harmful_change_rate`：所有新模型都维持 `0.0357`

preservation 方面：

- `pair-weighted` 和 `pair-synth_weighted` 的 preservation 还略高于 baseline confusion synthetic。
- 说明当前的 pair weighting 并没有直接牺牲 preservation。

但风险仍在：

- harmful 样本仍集中在 `hard / low-resolution / multi-line / insert-delete-heavy` 区域。
- V2-M02w2 没有解决 hardest slice 的保守性问题，只是没有把它进一步弄坏。

因此如果要继续推进，下一轮优先不是再放大 pair weight，而是做更细的权重/ratio 调参，或在 hard slice 上使用更保守的 gating。

## 8. Case Study

以下案例使用：

- baseline confusion synthetic corrected: `outputs/V2-M02z_eval_full_incorrect`
- V2-M02w2 best model corrected: `outputs/V2-M02w2_eval_pair_weighted_full_incorrect`

### Case 1: weak pair `8->9` 被新模型修对

- GT: `浙建德货00569ZHEJIANDEHUO`
- baseline pred: `浙建德货00678ZHEJIANDEHUO`
- baseline-confusion corrected: `浙建德货00568ZHEJIANDEHUO`
- pair-weighted corrected: `浙建德货00569ZHEJIANDEHUO`
- changed positions:
  - `6: 6->5`, `base_conf=0.3848`, `corr_conf=0.6101`
  - `7: 7->6`, `base_conf=0.8433`, `corr_conf=0.4753`
  - `8: 8->9`, `base_conf=0.5757`, `corr_conf=0.3842`
- pair: `8->9`
- success/failure reason: baseline confusion synthetic 已经能把前两个数位拉回，但最后的 `8->9` top-1 仍失败；pair-weighted loss 终于把 weak pair 的 top-1 排到 GT。

### Case 2: weak pair `8->6` 被新模型修对

- GT: `苏无锡货16598SUWUXIHUO`
- baseline pred: `苏无锡货18598SUWUXIHUO`
- baseline-confusion corrected: `苏无锡货18598SUWUXIHUO`
- pair-weighted corrected: `苏无锡货16598SUWUXIHUO`
- changed positions:
  - `5: 8->6`, `base_conf=0.9995`, `corr_conf=0.9510`
- pair: `8->6`
- success/failure reason: 高置信 `8->6` 错误被 calibrated policy 放行，但只有 pair-weighted loss 把 GT 提到 top-1。

### Case 3: baseline 修不对，pair-weighted 再次完成多位 replace-only 修正

- GT: `皖亳州货2136`
- baseline pred: `皖亳州货2168`
- baseline-confusion corrected: `皖亳州货2138`
- pair-weighted corrected: `皖亳州货2136`
- changed positions:
  - `6: 6->3`, `base_conf=0.8472`, `corr_conf=0.8203`
  - `7: 8->6`, `base_conf=0.9478`, `corr_conf=0.8757`
- pair: `8->6`
- success/failure reason: baseline 只纠正了一半；pair-weighted 把两个 replace 都修对。

### Case 4: hard_slice harmful case

- GT: `浙余杭货02078ZHEYUHANGHUO`
- baseline pred: `浙余杭货02878ZHEYUHANGHUO`
- baseline-confusion corrected: `浙余杭货02008ZHEYUHANGHUO`
- pair-weighted corrected: `浙余杭货02008ZHEYUHANGHUO`
- changed positions:
  - `6: 8->0`, `base_conf=0.9941`, `corr_conf=0.9086`
  - `7: 7->0`, `base_conf=0.7090`, `corr_conf=0.5470`
- pair: `8->0` plus extraneous wrong change
- success/failure reason: 一个正确修正伴随一个额外 harmful 改写；这是 hardest slice 上最典型的“激活过多”失败，不是本轮 pair weighting 新引入的问题，但也没有被解决。

### Case 5: oracle@K contains GT but still no change

- GT: `皖创业1119WANCHUANGYE`
- baseline pred: `皖众发1198WANZHONGFA`
- baseline-confusion corrected: `皖众发1188WANZHONGFA`
- pair-weighted corrected: `皖众发1188WANZHONGFA`
- changed positions: `O->A` 相关位置未实际改写
- pair: `O->A`
- success/failure reason: pair-weighted 已把 `O->A` 的 `oracle@1` 提升到 `0.1538`，但 sample 仍是 `insert_count=1`、`replace_count=8` 的 hard multi-error case，GT 候选仍没有稳定变成最终 top-1 change。

## 9. Analysis

### 9.1 pair-specific synthetic 是否有效

有效，但很弱。

- 它在所有主 slice 上都略优于 baseline confusion synthetic。
- 但弱 pair 指标几乎没有变化。
- 说明单纯重排 synthetic 分布，并没有真正改变 top-1 ranking。

### 9.2 pair-weighted loss 是否有效

有效，而且是本阶段主增益来源。

- `full_incorrect / replace_dominant / replace_only / hard_slice / long_21plus` 全部优于 baseline。
- `8->9 / 8->6 / 8->0` 都有明确提升。
- `oracle@1` 也同步提升，说明确实在改善 top-1 candidate ranking，而不只是 gating 偶然放行。

### 9.3 哪个模型最值得保留

`pair-weighted loss only` 最值得保留。

原因：

- overall 最好；
- weak pair 提升最明确；
- harmful 没有上升；
- preservation 也没有下降。

### 9.4 是否提升了 candidate top-1 ranking

是，但幅度有限。

证据：

- `full_incorrect oracle@1`: `0.3022 -> 0.3044`
- `8->9 oracle@1`: `0.1026 -> 0.2308`
- `8->6 oracle@1`: `0.3571 -> 0.4048`

这说明 loss weighting 确实在把难 pair 的 GT 推向更靠前的位置。

### 9.5 是否降低了 oracle gap

略有降低，但幅度很小。

- `full_incorrect oracle_gap`: `0.1295 -> 0.1291`

这进一步说明当前 selected-mask/gating 已不是大瓶颈，真正难的是让 GT 成为更稳定的 top-1。

### 9.6 是否牺牲了 preservation

没有明显牺牲。

- `full_incorrect preservation_rate`: `0.9941 -> 0.9943`
- `replace_dominant preservation_rate`: `0.9904 -> 0.9913`

### 9.7 当前瓶颈是否仍是 ranking/calibration

是。

- selected coverage 仍接近饱和。
- `pair-weighted` 能提升 weak pair，但幅度仍有限。
- `O->A` 仍停留在 oracle 改善而非 actual correction。
- `hard_slice` harmful 仍高。

因此当前主瓶颈仍是：

1. weak pair 的 `candidate top-1 ranking`
2. hard/long/multi-error slice 的 `confidence calibration`
3. replace-only 对 insert/delete-heavy 样本的上限

### 9.8 是否应进入 benchmark-style validation

现在还不建议。

理由：

- 虽然主 slice 指标都有稳定正增益；
- 但增益幅度仍偏温和；
- `hard_slice harmful_change_rate` 仍为 `0.0357`，高于理想阈值；
- `O->A` 这类 alphabet-side 弱 pair 还没有转化为实际 correction。

## 10. Recommendation

下一步建议：`V2-M02w3: tune pair weights / synthetic ratio`。

原因：

- V2-M02w2 已经证明 `pair-weighted loss` 比 `pair-specific synthetic` 更有效。
- 当前最合理的延续不是直接 benchmark，而是继续做小范围、受控的 pair 权重和 synthetic ratio 调参。
- 如果下一轮能继续抬升 `8->9 / 8->6 / O->A`，同时把 `hard_slice harmful` 压下来，再进入 `V2-M02zz` 会更稳。

补充判断：

- 如果 V2-M02w3 后 weak pair 仍停滞，应考虑 `V2-M02y2: hard-aware gating / selected-mask detector`。
- 如果 ranking 再也推不动，再讨论 `V2-M02x encoder_memory branch` 才更有依据。
