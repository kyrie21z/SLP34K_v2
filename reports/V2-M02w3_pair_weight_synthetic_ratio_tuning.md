# V2-M02w3 Pair Weight / Synthetic Ratio Tuning

analysis split only  
not official benchmark conclusion

## 1. Summary

V2-M02w3 在不改结构、不进入 V2-M03 的前提下，对 `pair_weight_alpha / pair_weight_max / lambda_preservation / synthetic_ratio` 做了小范围、可解释的受控调参。共训练了 5 组模型：B/C/D/E 必跑，F 为可选 negative control。整体最优模型是 `aw1p0_wmax3p0_lp0p5`，即在 V2-M02w2 best 的 `pair_weighted loss only` 基础上，将 `lambda_preservation` 从 `0.2` 提高到 `0.5`，其余保持 `pair_weight_alpha=1.0, pair_weight_max=3.0, synthetic_mode=confusion, synthetic_ratio=1.0`。

该模型稳定超过 V2-M02w2 best：`full_incorrect correction_rate 0.1753 -> 0.1894`，`replace_dominant 0.2070 -> 0.2238`，`replace_only 0.3941 -> 0.4256`，`full_incorrect harmful_change_rate 0.0117 -> 0.0100`。但 `hard_slice harmful_change_rate` 仍停留在 `0.0357`，没有降到目标 `<=0.02`。weak pair 上，`8->9` 有继续改善，`8->6` 只有更激进权重配置 `aw2p0_wmax3p0_lp0p3` 才明显提升；`O->A` 仍无法稳定转化为 actual correction。整体看，w3 已经超过 w2 的 ranking 上限一点，但主风险仍在 hard / long / multi-error slice 的 harmful 控制，因此暂不建议进入 `V2-M02zz`。

## 2. Files Added / Modified

- Modified: [train_mdiff_corrector_smoke.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/train_mdiff_corrector_smoke.py)
- Added: [V2-M02w3_pair_weight_synthetic_ratio_tuning.md](/mnt/data/zyx/SLP34K_v2/reports/V2-M02w3_pair_weight_synthetic_ratio_tuning.md)

确认：

- `main.yaml` 未修改。
- baseline forward 行为未改变。
- 未进入 V2-M03。
- 未实现 insert/delete correction。
- 未新增 encoder_memory branch。

## 3. Hyperparameter Grid

| model | pair_weight_alpha | pair_weight_max | lambda_preservation | synthetic_mode | synthetic_ratio |
| ----- | ----------------: | --------------: | ------------------: | -------------- | --------------: |
| baseline_w2_pair_weighted | 1.0 | 3.0 | 0.2 | confusion | 1.0 |
| aw1p5_wmax2p0_lp0p2 | 1.5 | 2.0 | 0.2 | confusion | 1.0 |
| aw2p0_wmax3p0_lp0p2 | 2.0 | 3.0 | 0.2 | confusion | 1.0 |
| aw2p0_wmax3p0_lp0p3 | 2.0 | 3.0 | 0.3 | confusion | 1.0 |
| aw1p0_wmax3p0_lp0p5 | 1.0 | 3.0 | 0.5 | confusion | 1.0 |
| pair_synth_ratio2_aw1p0 | 1.0 | 3.0 | 0.2 | pair_specific | 2.0 |

## 4. Training Results

`runtime` 为本轮终端记录的 wall time；`peak_memory` 仅 baseline 有可靠记录，其余训练 summary 未持久化该字段，因此记为 `N/A`。

| model | final_loss | selected_ce_final | preservation_ce_final | runtime | peak_memory |
| ----- | ---------: | ----------------: | --------------------: | ------: | ----------: |
| baseline_w2_pair_weighted | 0.6171 | 0.5263 | 0.4539 | 38.23s | 608.17 MB |
| aw1p5_wmax2p0_lp0p2 | 0.6312 | 0.5411 | 0.4502 | 37.95s | N/A |
| aw2p0_wmax3p0_lp0p2 | 0.6431 | 0.5537 | 0.4468 | 37.99s | N/A |
| aw2p0_wmax3p0_lp0p3 | 0.6685 | 0.5504 | 0.3939 | 28.20s | N/A |
| aw1p0_wmax3p0_lp0p5 | 0.6632 | 0.4983 | 0.3298 | 37.65s | N/A |
| pair_synth_ratio2_aw1p0 | 0.4165 | 0.3477 | 0.3442 | 27.93s | N/A |

## 5. Slice Evaluation Matrix

analysis split only  
not official benchmark conclusion

| model | slice | correction_rate | corrected_acc_gain | preservation_rate | harmful_change_rate | oracle@1 | oracle@5 |
| ----- | ----- | --------------: | -----------------: | ----------------: | ------------------: | -------: | -------: |
| baseline_w2_pair_weighted | full_incorrect | 0.1753 | 0.0884 | 0.9943 | 0.0117 | 0.3044 | 0.5903 |
| aw1p5_wmax2p0_lp0p2 | full_incorrect | 0.1758 | 0.0890 | 0.9944 | 0.0117 | 0.3079 | 0.5907 |
| aw2p0_wmax3p0_lp0p2 | full_incorrect | 0.1771 | 0.0907 | 0.9943 | 0.0117 | 0.3101 | 0.5925 |
| aw2p0_wmax3p0_lp0p3 | full_incorrect | 0.1824 | 0.0924 | 0.9945 | 0.0117 | 0.3145 | 0.5947 |
| aw1p0_wmax3p0_lp0p5 | full_incorrect | 0.1894 | 0.0988 | 0.9945 | 0.0100 | 0.3141 | 0.6000 |
| baseline_w2_pair_weighted | replace_dominant | 0.2070 | 0.2532 | 0.9913 | 0.0000 | 0.3557 | 0.6492 |
| aw1p5_wmax2p0_lp0p2 | replace_dominant | 0.2076 | 0.2547 | 0.9913 | 0.0000 | 0.3605 | 0.6492 |
| aw2p0_wmax3p0_lp0p2 | replace_dominant | 0.2092 | 0.2595 | 0.9911 | 0.0000 | 0.3627 | 0.6503 |
| aw2p0_wmax3p0_lp0p3 | replace_dominant | 0.2157 | 0.2642 | 0.9917 | 0.0000 | 0.3676 | 0.6530 |
| aw1p0_wmax3p0_lp0p5 | replace_dominant | 0.2238 | 0.2801 | 0.9919 | 0.0000 | 0.3686 | 0.6578 |
| baseline_w2_pair_weighted | replace_only | 0.3941 | 0.3486 | 0.9936 | 0.0000 | 0.6428 | 0.9207 |
| aw1p5_wmax2p0_lp0p2 | replace_only | 0.3963 | 0.3508 | 0.9936 | 0.0000 | 0.6515 | 0.9207 |
| aw2p0_wmax3p0_lp0p2 | replace_only | 0.3996 | 0.3573 | 0.9936 | 0.0000 | 0.6569 | 0.9207 |
| aw2p0_wmax3p0_lp0p3 | replace_only | 0.4104 | 0.3638 | 0.9942 | 0.0000 | 0.6678 | 0.9229 |
| aw1p0_wmax3p0_lp0p5 | replace_only | 0.4256 | 0.3856 | 0.9942 | 0.0000 | 0.6764 | 0.9273 |
| baseline_w2_pair_weighted | hard_slice | 0.1735 | 0.1258 | 0.9918 | 0.0357 | 0.3010 | 0.5883 |
| aw1p5_wmax2p0_lp0p2 | hard_slice | 0.1740 | 0.1267 | 0.9920 | 0.0357 | 0.3046 | 0.5888 |
| aw2p0_wmax3p0_lp0p2 | hard_slice | 0.1749 | 0.1283 | 0.9919 | 0.0357 | 0.3068 | 0.5905 |
| aw2p0_wmax3p0_lp0p3 | hard_slice | 0.1802 | 0.1308 | 0.9921 | 0.0357 | 0.3112 | 0.5927 |
| aw1p0_wmax3p0_lp0p5 | hard_slice | 0.1873 | 0.1392 | 0.9920 | 0.0357 | 0.3108 | 0.5981 |
| baseline_w2_pair_weighted | long_21plus | 0.1157 | 0.0875 | 0.9931 | 0.0000 | 0.1903 | 0.5256 |
| aw1p5_wmax2p0_lp0p2 | long_21plus | 0.1171 | 0.0904 | 0.9931 | 0.0000 | 0.1918 | 0.5271 |
| aw2p0_wmax3p0_lp0p2 | long_21plus | 0.1186 | 0.0933 | 0.9933 | 0.0000 | 0.1977 | 0.5329 |
| aw2p0_wmax3p0_lp0p3 | long_21plus | 0.1215 | 0.0933 | 0.9931 | 0.0000 | 0.2020 | 0.5329 |
| aw1p0_wmax3p0_lp0p5 | long_21plus | 0.1245 | 0.0962 | 0.9931 | 0.0000 | 0.1991 | 0.5344 |

可选 Model F 作为 negative control：

- `full_incorrect correction_rate = 0.1388`, `harmful_change_rate = 0.0133`
- `replace_dominant correction_rate = 0.1627`
- `replace_only correction_rate = 0.3029`
- `hard_slice harmful_change_rate = 0.0476`

这说明把 `pair_specific synthetic_ratio` 提高到 `2.0` 会明显伤害整体泛化与 hard slice 稳定性。

## 6. Weak-pair Analysis

| model | pair | support | correction_rate | oracle@1 | oracle@5 | delta_vs_w2_best |
| ----- | ---- | ------: | --------------: | -------: | -------: | ---------------: |
| baseline_w2_pair_weighted | 8->9 | 39 | 0.1026 | 0.2308 | 0.7692 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 8->9 | 39 | 0.1538 | 0.3077 | 0.7692 | +0.0513 |
| aw1p0_wmax3p0_lp0p5 | 8->9 | 39 | 0.1282 | 0.2564 | 0.7692 | +0.0256 |
| baseline_w2_pair_weighted | 8->6 | 42 | 0.2857 | 0.4048 | 0.9286 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 8->6 | 42 | 0.3095 | 0.5238 | 0.9286 | +0.0238 |
| aw1p0_wmax3p0_lp0p5 | 8->6 | 42 | 0.2857 | 0.4048 | 0.9286 | +0.0000 |
| baseline_w2_pair_weighted | O->A | 13 | 0.0000 | 0.1538 | 0.7692 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | O->A | 13 | 0.0000 | 0.1538 | 0.7692 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | O->A | 13 | 0.0000 | 0.0769 | 0.7692 | +0.0000 |
| baseline_w2_pair_weighted | 8->0 | 20 | 0.4500 | 0.5500 | 1.0000 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 8->0 | 20 | 0.4500 | 0.6000 | 1.0000 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | 8->0 | 20 | 0.4500 | 0.6500 | 1.0000 | +0.0000 |
| baseline_w2_pair_weighted | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | 6->8 | 50 | 0.7400 | 0.9000 | 0.9800 | +0.0000 |
| baseline_w2_pair_weighted | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 9->8 | 30 | 0.6333 | 0.7667 | 1.0000 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | 9->8 | 30 | 0.6000 | 0.8000 | 1.0000 | -0.0333 |
| baseline_w2_pair_weighted | 1->0 | 20 | 0.5000 | 0.7000 | 1.0000 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 1->0 | 20 | 0.5000 | 0.7000 | 1.0000 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | 1->0 | 20 | 0.5000 | 0.8000 | 1.0000 | +0.0000 |
| baseline_w2_pair_weighted | 6->0 | 20 | 0.5000 | 0.6000 | 0.9500 | 0.0000 |
| aw2p0_wmax3p0_lp0p3 | 6->0 | 20 | 0.5000 | 0.6500 | 0.9500 | +0.0000 |
| aw1p0_wmax3p0_lp0p5 | 6->0 | 20 | 0.5000 | 0.6500 | 0.9500 | +0.0000 |

结论：

- 调大 pair weight 确实能推 weak pair，最典型是 `8->9` 与 `8->6`，最佳配置是 `aw2p0_wmax3p0_lp0p3`。
- 但 `O->A` 依然完全卡在 ranking / calibration，w3 的可部署配置没有把 oracle 候选转化成 actual correction。
- `aw1p0_wmax3p0_lp0p5` 不是每个 weak pair 都最强，但它在 major slice 上总体收益最大，因此是更稳妥的全局 best。
- strong/control pairs 整体保持稳定，仅 `9->8` 在 `aw1p0_wmax3p0_lp0p5` 上小幅回落。

## 7. Harmful / Preservation Analysis

`full_incorrect harmful_change_rate` 在 `lambda_preservation=0.5` 时从 `0.0117` 降到 `0.0100`，同时 `correction_rate` 反而继续升到 `0.1894`，说明更高 preservation 并没有简单压死 correction，反而改善了训练后的稳态校正行为。

但 `hard_slice harmful_change_rate` 在 B/C/D/E 四组上都维持 `0.0357`，没有低于 V2-M02z / V2-M02w2 的风险线。这意味着 current harmful 不是单纯由 global preservation weight 不足导致，更像是 hard / multi-error / insert-delete-heavy 样本上的 ranking 错误与过度替换叠加。换句话说，`lambda_preservation` 能改善常规 slice，但压不住最难 slice。

`pair_synth_ratio2_aw1p0` 是清晰的反例：虽然个别 weak pair 看起来更激进，甚至 `O->A` 出现了少量 actual correction，但 major slice 整体明显下降，`hard_slice harmful_change_rate` 还升到 `0.0476`。因此 synthetic ratio 增大并不是本阶段正确方向。

## 8. Case Study

以下案例均来自 `full_incorrect`，对比 `V2-M02w2 pair_weighted` 与 `V2-M02w3 aw1p0_wmax3p0_lp0p5`。

### Case 1: weak pair `8->9` 被 w3 修对

`sample_id = unified_lmdb:000003120`

- GT: `皖亳州货1399亳州港`
- baseline pred: `皖亳州货1398亳州港`
- V2-M02w2 corrected: `皖亳州货1398亳州港`
- V2-M02w3 corrected: `皖亳州货1399亳州港`
- changed positions: `pos7, 8->9`
- base_conf: `0.6411`
- corr_conf: `0.4426`
- pair: `8->9`
- success/failure reason: w3 在弱数字 pair 上更愿意接受低 base-conf 的正确 top-1 replacement。

### Case 2: w3 把 w2 的 partial fix 推成 exact fix

`sample_id = unified_lmdb:000005830`

- GT: `苏连云港货6028`
- baseline pred: `苏连云港货1128`
- V2-M02w2 corrected: `苏连云港货6008`
- V2-M02w3 corrected: `苏连云港货6028`
- changed positions: `1->6`, `1->0`
- base_conf: `0.5879`, `0.9639`
- corr_conf: `0.9808`, `0.9184`
- pair: mixed numeric weak pair
- success/failure reason: w3 更好地完成了双位置 replace 的联合 ranking。

### Case 3: 高置信错误字符也能被 w3 纠正

`sample_id = unified_lmdb:000001646`

- GT: `皖阜南货1288WANFUNANHUO`
- baseline pred: `皖埠南货1288WANBUNANHUO`
- V2-M02w2 corrected: `皖埠南货1288WANFUNANHUO`
- V2-M02w3 corrected: `皖阜南货1288WANFUNANHUO`
- changed positions: `埠->阜`, `B->F`
- base_conf: `1.0`, `1.0`
- corr_conf: `0.9279`, `0.9235`
- pair: non-digit hard replace
- success/failure reason: selected coverage 已够，w3 改善的是高置信错误的 top-1 ranking。

### Case 4: multi-position 港口名修正

`sample_id = unified_lmdb:000001896`

- GT: `菏泽港`
- baseline pred: `武汉港`
- V2-M02w2 corrected: `菏汉港`
- V2-M02w3 corrected: `菏泽港`
- changed positions: `武->菏`, `汉->泽`
- base_conf: `1.0`, `0.6899`
- corr_conf: `0.9533`, `0.3264`
- pair: Chinese replace pair
- success/failure reason: w3 对 multi-error replace 的联合修复更完整。

### Case 5: `9->6` 数字 pair 改善

`sample_id = unified_lmdb:000001814`

- GT: `浙平湖货01076`
- baseline pred: `浙平湖货01079`
- V2-M02w2 corrected: `浙平湖货01078`
- V2-M02w3 corrected: `浙平湖货01076`
- changed positions: `9->6`
- base_conf: `0.7778`
- corr_conf: `0.4720`
- pair: `9->6`
- success/failure reason: w3 更好地压制了错误的近邻候选。

### Case 6: hard slice harmful 仍存在

`sample_id = unified_lmdb:000001564`

- GT: `浙越城货0671绍兴港ZHEYUECHENGHUOSHAOXINGGANG`
- baseline pred: `浙越城货0671绍兴港`
- V2-M02w2 corrected: `浙越城货0671绍兴港`
- V2-M02w3 corrected: `浙越城货0671盐兴港`
- changed positions: `绍->盐`
- base_conf: high on kept prefix
- corr_conf: low-to-mid on harmful replacement
- pair: insert/delete-heavy sample
- success/failure reason: replace-only corrector 在 truncation / insert-delete-heavy 样本上仍可能把局部 candidate 误当成可替换位置。

### Case 7: `O->A` 仍停留在 oracle 候选层

`sample_id = unified_lmdb:000001606`

- GT: `皖创业1119WANCHUANGYE`
- baseline pred: `皖众发1198WANZHONGFA`
- V2-M02w2 corrected: `皖众发1188WANZHONGFA`
- V2-M02w3 corrected: `皖众发1188WANZHONGFA`
- changed positions: unrelated numeric edits only
- base_conf: mixed
- corr_conf: mixed
- pair: `O->A`
- success/failure reason: `O->A` 的 GT 仍常在 top-k 里，但没有稳定进入 top-1 或通过最终校正规则。

## 9. Analysis

1. 调大 pair weight 是否有效？  
有效，但收益主要集中在弱数字 pair。`aw2p0_wmax3p0_lp0p3` 对 `8->9`、`8->6` 的改善最明显，说明 pair-weight 方向本身是对的。

2. 调高 preservation 是否有效？  
对常规 slice 有效。`lambda_preservation=0.5` 同时提高了 `full_incorrect / replace_dominant / replace_only` 的 correction，并把 `full_incorrect harmful` 压到 `0.0100`。但它没有解决 `hard_slice harmful=0.0357`。

3. 调 synthetic ratio 是否有效？  
无效。`pair_specific synthetic_ratio=2.0` 虽然能让少量弱 pair 更激进，但总体明显退化，尤其伤害 `hard_slice`，说明 synthetic 分布扩大不是当前主方向。

4. 哪个模型最值得保留？  
`aw1p0_wmax3p0_lp0p5` 最值得保留，原因是 major slice 全面最好，且 harmful 没有恶化。若只想继续打 weak pair，可把 `aw2p0_wmax3p0_lp0p3` 作为辅助对照。

5. 是否突破了 V2-M02w2 的 ranking 上限？  
有小幅突破。`full_incorrect correction_rate 0.1753 -> 0.1894`，`replace_only 0.3941 -> 0.4256`，说明 ranking 还没完全到顶，但提升幅度已经开始变小。

6. hard_slice 是否仍是风险点？  
是。w3 没有把 hard harmful 压到 `<=0.02`，说明问题不只是 global preservation，而是 hard / long / multi-error / insert-delete-heavy 样本上的局部替换决策仍不稳。

7. 是否应进入 benchmark-style validation？  
暂不建议。虽然 w3 在主要 slice 上稳定超过 w2 best，但 `hard_slice harmful_change_rate` 仍高于目标，`O->A` 也没有真正打通。现在更合理的下一步是 hard-aware gating，而不是直接进入 untouched holdout benchmark-style validation。

## 10. Recommendation

推荐下一步进入：

`V2-M02y2: hard-aware gating / selected-mask detector`

理由：

- w3 已证明小范围 loss / preservation 调参还能继续推高 overall correction；
- 但 `hard_slice harmful` 对 global weight 调整不敏感；
- 说明下一阶段更值得做的是 sample-aware / hard-aware 的推理保守性控制，而不是继续堆 synthetic ratio；
- 在 hard-aware gating 没压住前，不建议直接进入 `V2-M02zz`。
