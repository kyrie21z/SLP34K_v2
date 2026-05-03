# V2-M02w1 Confusion-aware Synthetic Noise

`analysis split only`

`not official benchmark conclusion`

## 1. Summary

V2-M02w-1 的目标是在 expanded real-error split 上验证 confusion-aware synthetic noise 是否能提升 MDiff corrector 的 held-out replace correction 泛化能力。

本阶段结果：

- 成功从 `outputs/V2-M02w0_cache_main_split_incorrect/train` 构建 `V2-M02w1` confusion table，且 `eval` 未参与构建。
- 成功训练并评估三组 `token_decoder_hidden` corrector：
  - `real-only`
  - `real + random synthetic`
  - `real + confusion synthetic`
- 在默认阈值 `tau_low=0.70 / tau_corr=0.80 / tau_keep=0.90 / delta_gain=0.05` 下：
  - `real-only` eval `correction_rate = 0.0062`
  - `random synthetic` eval `correction_rate = 0.0062`
  - `confusion synthetic` eval `correction_rate = 0.0166`
- `confusion synthetic` 在 held-out eval 上优于 `real-only` 和 `random synthetic`，同时保持：
  - `preservation_rate = 0.9995`
  - `harmful_change_rate = 0.0000`
- threshold sweep 显示当前最优点在更宽松的 `tau_corr=0.60`，`correction_rate` 可进一步升到 `0.0229`，且 harmful 仍为 `0`。

结论判断：

- confusion-aware synthetic 确实带来正向增益，但绝对增益仍小。
- oracle 指标和 threshold sweep 都说明当前主瓶颈更偏 `gating / threshold calibration`，不是 preservation 崩坏。
- 因此 corrector 路线仍值得继续，但下一步应优先进入 `V2-M02y: selected-mask / threshold calibration`，而不是直接进入 V2-M03。

## 2. Files Added / Modified

修改：

- `ocr_training/tools/build_confusion_table.py`
- `ocr_training/tools/train_mdiff_corrector_smoke.py`
- `ocr_training/tools/eval_mdiff_corrector_offline.py`

新增：

- `reports/V2-M02w1_confusion_aware_synthetic_noise.md`

边界确认：

- `ocr_training/configs/main.yaml` 未修改。
- `maevit_infonce_plm` baseline 默认行为未改变。
- 未进入 V2-M03。
- 未实现 insert/delete correction。
- 未实现 encoder_memory branch。

## 3. Confusion Table

构建命令：

```bash
python tools/build_confusion_table.py \
  --cache_dir outputs/V2-M02w0_cache_main_split_incorrect/train \
  --output_dir outputs/V2-M02w1_confusion_table \
  --min_count 1 \
  --top_k 50
```

说明：

- confusion table 只从 `train` split 构建。
- `eval` split 未参与构建。
- `pred_token` 表示 baseline 错误预测字符。
- `gt_token` 表示真实字符。
- synthetic corruption 方向为 `gt -> pred`。
- JSON 中保留了全部 `895` 个 pair；CSV 导出 top `50` 预览。

Top 20：

| pred_token | gt_token | count | segment_type | avg_base_conf | avg_position |
| ---------- | -------- | ----: | ------------ | ------------: | -----------: |
| 6 | 8 | 44 | digit | 0.8343 | 5.59 |
| 8 | 6 | 34 | digit | 0.8986 | 6.06 |
| 8 | 9 | 30 | digit | 0.8395 | 5.93 |
| 9 | 8 | 27 | digit | 0.8179 | 5.67 |
| 8 | 0 | 17 | digit | 0.9017 | 6.12 |
| 1 | 0 | 15 | digit | 0.7209 | 5.47 |
| 8 | 1 | 15 | digit | 0.8599 | 5.13 |
| 6 | 0 | 14 | digit | 0.7753 | 5.93 |
| 0 | 2 | 14 | digit | 0.8571 | 5.29 |
| 0 | 1 | 14 | digit | 0.8756 | 5.07 |
| 8 | 3 | 14 | digit | 0.9086 | 6.00 |
| 6 | 5 | 13 | digit | 0.9007 | 6.23 |
| O | A | 13 | alphabet | 0.9937 | 13.46 |
| 0 | 9 | 13 | digit | 0.8625 | 5.15 |
| 0 | 6 | 13 | digit | 0.8288 | 5.46 |
| 8 | 5 | 13 | digit | 0.8929 | 6.46 |
| 5 | 6 | 12 | digit | 0.8278 | 5.33 |
| 1 | 2 | 12 | digit | 0.7884 | 4.67 |
| 7 | 1 | 11 | digit | 0.8238 | 5.91 |
| 5 | 9 | 11 | digit | 0.7910 | 5.73 |

## 4. Synthetic Noise Design

### random synthetic

- 从 `train` split GT 中采样。
- 只改 `replacement`，不改长度。
- 不改 `EOS / PAD / BOS`。
- 每个 synthetic 样本默认改 `1~2` 个位置。
- `segment-preserving=true` 时只做同类型替换：
  - `digit -> digit`
  - `alphabet -> alphabet`
  - `chinese -> chinese`

### confusion synthetic

- 从 confusion table 中按 `gt_token` 查找候选 `pred_token`。
- synthetic corruption 方向固定为 `gt -> pred`。
- 同样只做 replacement，不改长度，不改 `EOS / PAD / BOS`。
- `segment-preserving=true` 时仅保留 `pred` 与 `gt` segment 一致的 confusion pair。
- 每个位置从该 `gt_token` 的 top-`20` confusion 候选中按 `count` 加权采样。

### Sample count

训练源：

- real train samples used by dataset: `850`
  - `baseline replace-only wrong samples = 370`
  - `correct-context samples = 480`

synthetic 扩增：

| model | synthetic_mode | synthetic_ratio | synthetic_sample_count | fallback_count | skip_count |
| ----- | -------------- | --------------: | ---------------------: | -------------: | ---------: |
| real-only | none | 0.0 | 0 | 0 | 0 |
| random | random | 1.0 | 850 | 0 | 0 |
| confusion | confusion | 1.0 | 850 | 0 | 0 |

观察：

- confusion synthetic 的 `850/850` 样本都直接命中 confusion pair。
- 没有 fallback 到 random synthetic。
- 没有因为无可用位置而 skip。

## 5. Training Results

| model | synthetic_mode | synthetic_ratio | max_steps | initial_loss | final_loss | selected_ce_final | preservation_ce_final |
| ----- | -------------- | --------------: | --------: | -----------: | ---------: | ----------------: | --------------------: |
| real-only | none | 0.0 | 500 | 7.9359 | 1.0784 | 0.9624 | 0.5802 |
| random synthetic | random | 1.0 | 500 | 7.7743 | 0.3907 | 0.3100 | 0.4032 |
| confusion synthetic | confusion | 1.0 | 500 | 7.4324 | 0.5827 | 0.4905 | 0.4607 |

说明：

- synthetic 增广显著降低了训练 loss。
- `random` 的训练 loss 最低，但 held-out 泛化并不优于 `confusion`。
- 说明训练集拟合更容易不等于 held-out replace correction 更强。

## 6. Eval Results

同一 held-out eval split，alignment-based replace metrics：

| model | baseline_acc | corrected_acc | gain | correction_rate | preservation_rate | harmful_change_rate | replace_error_reduction | oracle@1 | oracle@3 | oracle@5 | changed_token_count |
| ----- | -----------: | ------------: | ---: | --------------: | ----------------: | ------------------: | ----------------------: | -------: | -------: | -------: | ------------------: |
| real-only | 0.3468 | 0.3497 | 0.0029 | 0.0062 | 0.9995 | 0.0000 | 0.0062 | 0.1289 | 0.3035 | 0.4428 | 11 |
| random synthetic | 0.3468 | 0.3555 | 0.0087 | 0.0062 | 0.9992 | 0.0000 | 0.0062 | 0.1393 | 0.3389 | 0.4657 | 7 |
| confusion synthetic | 0.3468 | 0.3584 | 0.0116 | 0.0166 | 0.9995 | 0.0000 | 0.0166 | 0.1476 | 0.3451 | 0.4615 | 14 |

附加统计：

- eval `originally_wrong_positions = 481`
- eval `originally_correct_positions = 3989`

关键观察：

- `confusion synthetic` 相对 `real-only`：
  - `correction_rate` 从 `0.0062 -> 0.0166`
  - `corrected_accuracy` 从 `0.3497 -> 0.3584`
  - `oracle@1` 从 `0.1289 -> 0.1476`
- `confusion synthetic` 相对 `random synthetic`：
  - `correction_rate` 从 `0.0062 -> 0.0166`
  - `corrected_accuracy` 从 `0.3555 -> 0.3584`
  - `oracle@1` 从 `0.1393 -> 0.1476`
  - `oracle@3` 从 `0.3389 -> 0.3451`
  - `oracle@5` 略低于 random：`0.4615 < 0.4657`

## 7. Threshold Sweep

confusion synthetic sweep 结果显示，主要敏感项是 `tau_corr`。

| model | tau_corr | tau_keep | delta_gain | correction_rate | preservation_rate | harmful_change_rate | changed_token_count |
| ----- | -------: | -------: | ---------: | --------------: | ----------------: | ------------------: | ------------------: |
| confusion synthetic default | 0.80 | 0.90 | 0.05 | 0.0166 | 0.9995 | 0.0000 | 14 |
| confusion synthetic best | 0.60 | 0.80 | 0.00 | 0.0229 | 0.9992 | 0.0000 | 24 |

说明：

- 当 `tau_corr=0.60` 时，`tau_keep` 和 `delta_gain` 对结果几乎没有额外影响。
- 更宽松 gating 可以把 `correction_rate` 从 `8 / 481` 提升到 `11 / 481`。
- preservation 仍高于 `0.999`，harmful 仍为 `0`。

判断：

- candidate 已经比 `real-only` 更强，但默认阈值仍压制了部分可修正位置。
- 当前确实存在 `gating / calibration` 瓶颈。

## 8. Pair-level Analysis

下表使用 `confusion synthetic` 默认阈值结果：

| pair | support | corrected | correction_rate | oracle@1 | oracle@5 |
| ---- | ------: | --------: | --------------: | -------: | -------: |
| 9->0 | 11 | 2 | 0.1818 | 0.7273 | 1.0000 |
| 8->9 | 9 | 0 | 0.0000 | 0.0000 | 0.3333 |
| 8->6 | 8 | 0 | 0.0000 | 0.0000 | 0.8750 |
| 6->0 | 6 | 1 | 0.1667 | 0.3333 | 1.0000 |
| 6->5 | 6 | 0 | 0.0000 | 0.1667 | 0.8333 |
| 6->8 | 6 | 3 | 0.5000 | 0.8333 | 1.0000 |
| 1->0 | 5 | 0 | 0.0000 | 0.4000 | 1.0000 |
| 0->9 | 5 | 0 | 0.0000 | 0.0000 | 0.8000 |

相对对照组的重点变化：

- `6->8`：
  - real-only `1 / 6`
  - random `1 / 6`
  - confusion `3 / 6`
- `6->0`：
  - real-only `0 / 6`
  - random `0 / 6`
  - confusion `1 / 6`
- `9->0`：
  - real-only `0 / 11`
  - random `2 / 11`
  - confusion `2 / 11`

结论：

- 改善主要集中在数字视觉混淆，尤其是 `6/8`，以及少量 `6/0`、`9/0`。
- 字母类如 `A/U`、`U/A`、`I/G` 在本阶段没有转化为实际 correction gain。

## 9. Case Study

### Case 1: confusion 成功而 real/random 都失败

```text
sample_id: unified_lmdb:000004213
GT:                浙富阳货00917
baseline pred:     浙富阳货00911
real-only:         浙富阳货00911
random-synthetic:  浙富阳货00911
confusion-synth:   浙富阳货00917
changed positions: [8]
base_conf:         [0.6113]
corr_conf(conf):   [0.8951]
reason:            1->7 single-position digit correction only confusion model executes
```

### Case 2: oracle@K contains GT but no change

```text
sample_id: unified_lmdb:000000015
GT:                浙余杭货02039ZHEYUHANGHUO
baseline pred:     浙余杭货02839ZHEYUHANGHUO
real-only:         浙余杭货02839ZHEYUHANGHUO
random-synthetic:  浙余杭货02839ZHEYUHANGHUO
confusion-synth:   浙余杭货02839ZHEYUHANGHUO
changed positions: []
replace positions: [6]
reason:            8->0 candidate exists, but gating still blocks actual edit
```

### Case 3: confusion 只修对其中一个 replace

```text
sample_id: unified_lmdb:000005997
GT:                新航机9686XINHANGJI
baseline pred:     新航机9668XINHANGJI
real-only:         新航机9668XINHANGJI
random-synthetic:  新航机9668XINHANGJI
confusion-synth:   新航机9688XINHANGJI
changed positions: [5]
base_conf:         [0.5386]
corr_conf(conf):   [0.8134]
reason:            one digit repaired, but sample still not exact
```

### Case 4: random/confusion 都会动，但都只做部分修复

```text
sample_id: unified_lmdb:000004490
GT:                皖庐江货9018
baseline pred:     皖庐江货9958
real-only:         皖庐江货9958
random-synthetic:  皖庐江货0958
confusion-synth:   皖庐江货0958
changed positions: [4]
base_conf:         [0.6138]
corr_conf(random): [0.8866]
corr_conf(conf):   [0.9440]
reason:            candidate edit happens, but second wrong digit remains
```

### Case 5: real-only harmful tendency avoided, confusion fixes replace but sample still blocked by insert/delete

```text
sample_id: unified_lmdb:000006423
GT:                苏泗洪货1278宿迁
baseline pred:     苏泗洪货1275宿迁港
real-only:         苏泗洪货1273宿迁港
random-synthetic:  苏泗洪货1275宿迁港
confusion-synth:   苏泗洪货1278宿迁港
changed positions: [7]
base_conf:         [0.4170]
corr_conf(real):   [0.9785]
corr_conf(conf):   [0.8475]
reason:            confusion model repairs the replace position, but insert/delete scope remains outside current corrector
```

### Case 6: confusion improves one token inside a high-noise sample but full-sample accuracy still impossible

```text
sample_id: unified_lmdb:000001346
GT:                恒祥通008HENGXIANGTONG
baseline pred:     皖翔通996WANXIANGTONG
real-only:         皖翔通996WANXIANGTONG
random-synthetic:  皖翔通996WANXIANGTONG
confusion-synth:   皖翔通998WANXIANGTONG
changed positions: [5]
base_conf:         [0.5688]
corr_conf(conf):   [0.8147]
reason:            local candidate quality improved, but sample-level errors are too many
```

harmful cases：

- none observed in all three models at current thresholds。

## 10. Analysis

`analysis split only`

`not official benchmark conclusion`

1. confusion-aware synthetic 是否提升泛化？

- 是，但增益小。
- 默认阈值下 `correction_rate` 从 `0.0062 -> 0.0166`，`corrected_accuracy` 从 `0.3497 -> 0.3584`。

2. 是否优于 random synthetic？

- 是。
- `correction_rate`、`corrected_accuracy`、`oracle@1`、`oracle@3` 都优于 random。
- 但 `oracle@5` 略低于 random，说明候选质量提升不是单调全方位的。

3. 是否提升 candidate quality？

- 部分提升。
- `oracle@1`：`0.1289 -> 0.1393 -> 0.1476`
- `oracle@3`：`0.3035 -> 0.3389 -> 0.3451`
- `oracle@5`：`0.4428 -> 0.4657 -> 0.4615`
- 结论是 top-1 / top-3 更强，但 top-5 不稳定。

4. 是否把 oracle@K 转化为 actual correction？

- 只转化了一小部分。
- confusion 默认阈值 `oracle@1 = 0.1476`，但 `correction_rate = 0.0166`。
- best sweep `correction_rate = 0.0229`，仍远低于 oracle。
- 这说明当前瓶颈明显不是“完全没有候选”，而是 `gating / calibration` 没把候选变成实际修改。

5. harmful_change 是否可控？

- 是。
- 三组 `harmful_change_rate = 0.0000`。
- `preservation_rate` 全部大于 `0.9992`，明显高于 `0.98` 门槛。

6. 当前瓶颈更像 data / gating / candidate / model capacity 哪一个？

- 第一瓶颈：`gating / threshold calibration`
- 第二瓶颈：`model capacity / local-only correction scope`
- 不是 preservation 崩坏。
- 也不是完全没有数据作用，因为 confusion synthetic 已经比 random 更强。

7. 是否建议继续 corrector 路线？

- 建议继续，但只建议继续到 `V2-M02y`。
- 当前 held-out split 上有正向信号，而且 harmful 极低。
- 但增益幅度还不足以支撑“benchmark-ready”或“full benchmark conclusion”。

## 11. Recommendation

首选下一步：

1. `V2-M02y: selected-mask / threshold calibration`

原因：

- confusion synthetic 已经证明 `candidate quality` 有提升。
- best sweep 在 harmful 不变的前提下，把 `correction_rate` 从 `0.0166` 提高到 `0.0229`。
- 这说明最有性价比的下一步不是继续堆更多 synthetic ratio，而是先把已有候选释放出来。

备选次序：

2. `V2-M02w2: tune synthetic ratio / pair-specific noise`
3. `V2-M02z: larger hard/OOV/test split validation`

当前不建议：

- 直接进入 V2-M03
- 暂停 corrector 路线
- 立即上 `encoder_memory branch`

