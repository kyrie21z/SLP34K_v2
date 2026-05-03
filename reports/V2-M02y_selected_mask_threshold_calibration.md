# V2-M02y Selected-mask / Threshold Calibration

`analysis split only`

`not official benchmark conclusion`

## 1. Summary

V2-M02y 的目标是在不重新训练模型的前提下，把 V2-M02w-1 `confusion synthetic` checkpoint 已有的候选能力转化为更多实际修正。

本阶段结论：

- 成功复现 V2-M02w-1 default：
  - `policy = low_conf`
  - `tau_low=0.70 / tau_corr=0.80 / tau_keep=0.90 / delta_gain=0.05`
  - `correction_rate = 0.0166`
- 全局阈值 sweep 显示，真正敏感的是：
  - `tau_low`
  - `tau_corr`
- `tau_keep / delta_gain` 在最佳区域几乎不影响结果。
- best global threshold on `low_conf`：
  - `tau_low=0.90`
  - `tau_corr=0.30`
  - `tau_keep=0.90`
  - `delta_gain=-0.10`
  - `correction_rate = 0.0437`
  - `harmful_change_rate = 0.0000`
- best deployable policy：
  - `low_conf_or_confusion`
  - same thresholds as best global
  - `correction_rate = 0.0561`
  - `corrected_accuracy = 0.3815`
  - `gain_accuracy = +0.0347`
  - `harmful_change_rate = 0.0000`
- `oracle_replace` / `topk_oracle` 在相同 gating 下只比 best deployable 高一个 replace position：
  - `0.0561 -> 0.0582`
- 这说明：
  - default 的主要瓶颈先是 `selected-mask coverage`
  - 一旦加入 confusion prior，剩余瓶颈主要变成 `confidence calibration / conservative rule`

判断：

- corrector 路线值得继续。
- 下一步优先建议 `V2-M02z`，先把 calibrated deployable policy 拿到更大、更难的 held-out split 上验证，而不是直接进入 V2-M03。

## 2. Files Added / Modified

修改：

- `ocr_training/tools/eval_mdiff_corrector_offline.py`
- `ocr_training/tools/mdiff_corrector_utils.py`

新增：

- `ocr_training/tools/build_pair_thresholds.py`
- `reports/V2-M02y_selected_mask_threshold_calibration.md`

确认：

- `ocr_training/configs/main.yaml` 未修改。
- baseline `maevit_infonce_plm` forward 默认行为未改变。
- 未进入 V2-M03。
- 未实现 insert/delete correction。
- 未新增 encoder_memory branch。

## 3. Default Reproduction

| policy | tau_low | tau_corr | tau_keep | delta_gain | correction_rate | preservation_rate | harmful_change_rate | corrected_acc | changed_token_count |
| ------ | ------: | -------: | -------: | ---------: | --------------: | ----------------: | ------------------: | ------------: | ------------------: |
| low_conf | 0.70 | 0.80 | 0.90 | 0.05 | 0.0166 | 0.9995 | 0.0000 | 0.3584 | 14 |

复现结论：

- 与 V2-M02w-1 一致。
- 说明新 evaluator 的 alignment-based replace 统计口径没有漂移。

## 4. Global Threshold Sweep

筛选条件：

`harmful_change_rate <= 0.02`

`preservation_rate >= 0.98`

Top 10：

| rank | tau_low | tau_corr | tau_keep | delta_gain | correction_rate | preservation_rate | harmful_change_rate | corrected_acc | changed_token_count |
| ---: | ------: | -------: | -------: | ---------: | --------------: | ----------------: | ------------------: | ------------: | ------------------: |
| 1 | 0.90 | 0.30 | 0.90 | -0.10 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 2 | 0.90 | 0.30 | 0.90 | 0.00 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 3 | 0.90 | 0.30 | 0.90 | 0.05 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 4 | 0.90 | 0.30 | 0.90 | 0.10 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 5 | 0.90 | 0.30 | 0.95 | -0.10 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 6 | 0.90 | 0.30 | 0.95 | 0.00 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 7 | 0.90 | 0.30 | 0.95 | 0.05 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 8 | 0.90 | 0.30 | 0.95 | 0.10 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 9 | 0.90 | 0.30 | 0.99 | -0.10 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |
| 10 | 0.90 | 0.30 | 0.99 | 0.00 | 0.0437 | 0.9962 | 0.0000 | 0.3728 | 68 |

观察：

- 最优解全部集中在：
  - `tau_low = 0.90`
  - `tau_corr = 0.30`
- `tau_keep` 与 `delta_gain` 在最佳区域几乎无影响。
- 对比 default：
  - `correction_rate: 0.0166 -> 0.0437`
  - `corrected_accuracy: 0.3584 -> 0.3728`
  - `oracle_gap: 0.1310 -> 0.1040`
  - `gating_gap: 8 -> 1`

结论：

- 在 `low_conf` policy 下，default gating 明显过保守。
- 放松阈值本身就能显著提高 correction，而不引入 sample-level harmful change。

## 5. Selected-mask Policy Comparison

比较点：

- fixed thresholds = best global threshold
- `tau_low=0.90 / tau_corr=0.30 / tau_keep=0.90 / delta_gain=-0.10`

| policy | selected_positions | selected_wrong_coverage | selected_correct_coverage | correction_rate | harmful_change_rate | oracle@1 | oracle@5 |
| ------ | -----------------: | ----------------------: | ------------------------: | --------------: | ------------------: | -------: | -------: |
| low_conf | 152 | 0.2058 | 0.0103 | 0.0437 | 0.0000 | 0.1476 | 0.4615 |
| confusion_pair | 4766 | 0.9605 | 0.9742 | 0.0561 | 0.0000 | 0.1476 | 0.4553 |
| low_conf_or_confusion | 4773 | 0.9688 | 0.9749 | 0.0561 | 0.0000 | 0.1476 | 0.4553 |
| oracle_replace | 481 | 1.0000 | 0.0000 | 0.0582 | 0.0000 | 0.1476 | 0.4595 |
| topk_oracle | 219 | 0.4553 | 0.0000 | 0.0582 | 0.0000 | 0.1476 | 0.4636 |

deployable policies：

- `low_conf`
- `confusion_pair`
- `low_conf_or_confusion`

oracle analysis policies：

- `oracle_replace`
- `topk_oracle`

关键结论：

- `low_conf` 的 selected-mask coverage 明显不足：
  - `selected_wrong_coverage = 0.2058`
- `confusion_pair` / `low_conf_or_confusion` 把 replace 位置 coverage 提到接近全覆盖：
  - `0.9605`
  - `0.9688`
- `low_conf_or_confusion` 相比 `low_conf`：
  - `correction_rate: 0.0437 -> 0.0561`
  - `corrected_accuracy: 0.3728 -> 0.3815`
  - `harmful_change_rate` 仍为 `0`
- `best deployable` 与 `oracle_replace` 只差：
  - `0.0561 vs 0.0582`

判断：

- default 低 correction 的第一原因是 `low_conf` mask 漏掉大量高置信 replace 错误。
- 一旦 deployable confusion prior 加入，剩余 gap 很小，selected-mask coverage 不再是第一瓶颈。

## 6. Pair-specific Calibration

这里比较：

- `low_conf_or_confusion` global thresholds
  - `tau_low=0.60 / tau_corr=0.60 / tau_keep=0.80 / delta_gain=0.00`
- `low_conf_or_confusion + pair_thresholds.json`
  - pair thresholds 仅来源于 `train` confusion table 规则：
    - digit pair with `count>=10` -> `tau_corr=0.50`
    - alphabet pair with `count>=8` -> `tau_corr=0.55`

整体结果：

- global: `correction_rate = 0.0270`
- pair-specific: `correction_rate = 0.0291`
- harmful 仍为 `0`

重点 pair：

| pair | support | default_corrected | calibrated_corrected | correction_rate | oracle@1 | oracle@5 |
| ---- | ------: | ----------------: | -------------------: | --------------: | -------: | -------: |
| 6->8 | 6 | 4 | 4 | 0.6667 | 0.8333 | 1.0000 |
| 8->6 | 8 | 0 | 0 | 0.0000 | 0.0000 | 0.8750 |
| 8->9 | 9 | 0 | 0 | 0.0000 | 0.0000 | 0.3333 |
| 9->8 | 3 | 1 | 1 | 0.3333 | 1.0000 | 1.0000 |
| 1->0 | 5 | 0 | 0 | 0.0000 | 0.4000 | 1.0000 |
| 8->0 | 3 | 0 | 0 | 0.0000 | 0.3333 | 1.0000 |
| 6->0 | 6 | 1 | 2 | 0.3333 | 0.3333 | 1.0000 |
| O->A | 0 | N/A | N/A | N/A | N/A | N/A |

结论：

- pair-specific calibration 只有轻微帮助。
- 本轮 held-out eval 上唯一明确受益的是 `6->0`。
- 对 `6->8 / 8->6 / 8->9 / 1->0 / O->A` 没有新增收益。

## 7. Position Diagnostics

诊断文件：

- `outputs/V2-M02y_calibration/position_diagnostics.csv`

使用对象：

- best deployable policy：`low_conf_or_confusion`
- thresholds：`0.90 / 0.30 / 0.90 / -0.10`

总结：

- selected 但未改的错误位，主要原因是 `corr_top1` 仍等于 baseline token：
  - `277` 个 replace positions
- selected 但未改，且 `corr_top1_conf < tau_corr`：
  - `61` 个 replace positions
- selected 但未改，主要被 `tau_keep / delta_gain` 规则挡住：
  - `70` 个 replace positions
- `oracle@5 contains GT but no change`：
  - selected 后仍 no-change：`176`
  - 未 selected 就 no-change：`2`
- changed but wrong：
  - `31` 个 replace positions
- changed correct positions：
  - `17` 个 position
  - 但没有造成 sample-level harmful change，所以 `harmful_change_rate` 仍为 `0.0000`

解释：

- 在 best deployable policy 下，selected-mask coverage 已经接近饱和。
- 剩余问题主要是：
  - top-1 ranking 仍然偏向 baseline token
  - 少量位置仍被 confidence gate 挡住
- 因此当前瓶颈已经从“选不到”转到“candidate top-1 / confidence calibration”。

## 8. Case Study

### Case 1: default 未修，calibrated 修对

```text
sample_id:         unified_lmdb:000000015
GT:                浙余杭货02039ZHEYUHANGHUO
baseline pred:     浙余杭货02839ZHEYUHANGHUO
default corrected: 浙余杭货02839ZHEYUHANGHUO
calibrated corr:   浙余杭货02039ZHEYUHANGHUO
changed positions: [6]
base_conf:         [0.9956]
corr_conf:         [0.9797]
selected policy:   low_conf_or_confusion
reason:            high-confidence 8->0 error was missed by default low_conf gate but released by calibrated deployable policy
```

### Case 2: default 未修，calibrated 修对，多位数字修复

```text
sample_id:         unified_lmdb:000001563
GT:                浙越城货0628绍兴港
baseline pred:     浙越城货0696绍兴港
default corrected: 浙越城货0696绍兴港
calibrated corr:   浙越城货0628绍兴港
changed positions: [6, 7]
base_conf:         [0.4460, 0.6362]
corr_conf:         [0.3779, 0.6873]
selected policy:   low_conf_or_confusion
reason:            calibrated gating allows two replace edits that default leaves untouched
```

### Case 3: 9/8 相关，default 未修，calibrated 修对

```text
sample_id:         unified_lmdb:000003716
GT:                顺祥1168SHUNXIANG
baseline pred:     顺祥1169SHUNXIANG
default corrected: 顺祥1169SHUNXIANG
calibrated corr:   顺祥1168SHUNXIANG
changed positions: [5]
base_conf:         [0.9810]
corr_conf:         [0.9835]
selected policy:   low_conf_or_confusion
reason:            high-confidence 9->8 style digit confusion is outside default low_conf coverage
```

### Case 4: 1/8 相关，default 未修，calibrated 修对

```text
sample_id:         unified_lmdb:000006257
GT:                皖仁和8899WANRENHE
baseline pred:     皖仁和1899WANRENHE
default corrected: 皖仁和1899WANRENHE
calibrated corr:   皖仁和8899WANRENHE
changed positions: [3]
base_conf:         [0.7349]
corr_conf:         [0.5544]
selected policy:   low_conf_or_confusion
reason:            confusion prior exposes a replace candidate that default low_conf never considers
```

### Case 5: calibrated 改了，但样本仍未完全修对

```text
sample_id:         unified_lmdb:000000568
GT:                浙衢州货00009
baseline pred:     浙杭州货00609
default corrected: 浙杭州货00609
calibrated corr:   浙杭州货00009
changed positions: [6]
base_conf:         [0.6421]
corr_conf:         [0.5212]
selected policy:   low_conf_or_confusion
reason:            one replace error is fixed, but other non-replace discrepancies remain outside current corrector scope
```

### Case 6: oracle@K contains GT but still no change

```text
sample_id:         unified_lmdb:000000056
GT:                浙建德货00659ZHEJIANDEHUO
baseline pred:     浙建德货00600ZHEJIANDEHUO
default corrected: 浙建德货00600ZHEJIANDEHUO
calibrated corr:   浙建德货00600ZHEJIANDEHUO
changed positions: []
selected policy:   low_conf_or_confusion
reason:            pair 0->5 / 0->9 is selected, but corr_top1 remains baseline 0; candidate ranking, not mask, is the blocker
```

### Case 7: default 已正确，calibrated 保持不变

```text
sample_id:         unified_lmdb:000000011
GT:                浙上虞货0677
baseline pred:     浙上虞货0677
default corrected: 浙上虞货0677
calibrated corr:   浙上虞货0677
changed positions: []
selected policy:   low_conf_or_confusion
reason:            no regression on already-correct sample
```

sample-level harmful case：

- none observed

## 9. Analysis

`analysis split only`

`not official benchmark conclusion`

1. correction_rate 低主要是 gating 还是 candidate ranking？

- default 下先是 `selected-mask coverage + conservative gating`。
- best low_conf sweep 后，`gating_gap` 从 `8 -> 1`，说明 default 阈值确实过严。
- 加入 confusion prior 后，deployable policy 几乎追平 oracle selected-mask，剩余问题更像 `candidate ranking / confidence calibration`。

2. selected-mask coverage 是否不足？

- 对 default `low_conf`，是明显不足：
  - `selected_wrong_coverage = 0.1268`
- 对 best `low_conf_or_confusion`，已经不再是主瓶颈：
  - `selected_wrong_coverage = 0.9688`

3. low-confidence mask 是否漏掉高置信错误？

- 是。
- best global threshold 需要把 `tau_low` 放到 `0.90`。
- 多个新增成功样本的 baseline 错位都带有很高置信度，例如：
  - `000000015` base conf `0.9956`
  - `000003716` base conf `0.9810`

4. confusion-pair mask 是否有帮助？

- 有明显帮助。
- 在 best global threshold 下：
  - `low_conf correction_rate = 0.0437`
  - `confusion_pair correction_rate = 0.0561`
  - `low_conf_or_confusion correction_rate = 0.0561`
- 说明 deployable confusion prior 能补足 default low-confidence detector 的漏检。

5. pair-specific threshold 是否有帮助？

- 有，但增益很小。
- `0.0270 -> 0.0291`
- 主要只改善了 `6->0`。
- 当前主收益不在 pair-specific 阈值，而在全局阈值放宽和 confusion-aware mask。

6. harmful_change 是否仍可控？

- 可控。
- 所有 deployable policy 的 sample-level `harmful_change_rate = 0.0000`。
- `preservation_rate` 最低也仍有 `0.9957`。

7. 是否应继续 corrector 路线？

- 应继续。
- 当前 held-out analysis split 上，best deployable policy 已经把：
  - `correction_rate` 从 `0.0166` 提升到 `0.0561`
  - `corrected_accuracy gain` 从 `+0.0116` 提升到 `+0.0347`
- 这是实质性增益，且 harmful 可控。

## 10. Recommendation

首选下一步：

1. `V2-M02z: larger hard/OOV/test split validation with calibrated policy`

理由：

- 本阶段已经证明 calibrated deployable policy 在 held-out analysis split 上明显优于 V2-M02w-1 default。
- harmful 仍为 `0`，已达到“值得带去更难 split 做外部验证”的条件。
- 当前不宜直接进入 V2-M03；先验证 calibrated policy 的泛化稳定性更合理。

次选：

2. `V2-M02y2: train selected-mask detector`

条件：

- 如果 V2-M02z 发现 `low_conf_or_confusion` 在更大 split 上因为 selected too broad 而退化，再单独训练更精细的 selected-mask detector。

当前不优先：

- `V2-M02x encoder_memory branch`
- `V2-M02w2 synthetic ratio / pair-specific noise`
- 暂停 corrector 路线

