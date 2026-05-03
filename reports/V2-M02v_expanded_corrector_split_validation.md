# V2-M02v Expanded Corrector Split Validation

## 1. Summary

V2-M02v 基于 V2-M02u 的真实错误 `replace_only` cache，完成了 expanded cache 的 train/eval split、`token_decoder_hidden` 与 `token_only` 的 GPU 训练，以及 held-out eval split 的 offline correction 评估。

主实验 cache 沿用 `val/SLP34K_lmdb_test` 上已导出的 `replace_only` real-error cache：

- `total_exported = 128`
- `replace_only_samples = 102`
- `correct_context_samples = 26`
- `decoder_hidden shape = [128, 51, 768]`

然后新增独立 split 脚本，按 `sample_type` 做 `80/20` stratified split：

- train: `103` samples, `156` wrong positions, `21` correct-context samples
- eval: `25` samples, `46` wrong positions, `5` correct-context samples
- `sample_id overlap = 0`

held-out eval 的结论是：

- `token_decoder_hidden` 的默认阈值下 `correction_rate = 0.0217`
- `token_only` 的默认阈值下 `correction_rate = 0.0217`
- 两者 `harmful_change_rate = 0.0`
- 两者 `corrected_accuracy = baseline_accuracy = 0.2`

因此，V2-M02v 只能说明 corrector 在未见真实错误上仍存在弱泛化的 token-level correction 能力，但还没有形成稳定的 sample-level accuracy gain。`decoder_hidden` 的主要优势仍体现在 candidate quality：

- eval `oracle@1/3/5`:
  - `token_decoder_hidden = 0.1957 / 0.3043 / 0.5000`
  - `token_only = 0.0217 / 0.1739 / 0.3043`

放松 gating 后，`token_decoder_hidden` 的 eval `correction_rate` 可从 `0.0217` 提高到 `0.0652`，而 `harmful_change_rate` 仍为 `0.0`。这说明当前瓶颈更像“小样本 + gating/selection 偏保守 + 泛化不足”，而不是 corrector 完全无候选能力。

结论：

1. corrector 路线不该暂停；
2. 但当前 split-based 泛化证据仍偏弱；
3. 下一步更值得进入 `V2-M02w` 做 confusion-aware synthetic noise，而不是直接上更复杂结构。

## 2. Files Added / Modified

本阶段实际使用并确认的相关文件：

- Modified: `ocr_training/tools/export_parseq_corrector_cache.py`
- Added: `ocr_training/tools/split_mdiff_corrector_cache.py`
- Modified: `ocr_training/tools/train_mdiff_corrector_smoke.py`
- Modified: `ocr_training/tools/eval_mdiff_corrector_offline.py`
- Modified: `ocr_training/tools/mdiff_corrector_utils.py`
- Existing and reused: `ocr_training/strhub/models/slp_mdiff_corrector/modules.py`
- Existing and reused: `ocr_training/strhub/models/slp_mdiff_corrector/system.py`

兼容性确认：

- `ocr_training/strhub/models/maevit_infonce_plm/system.py` 仅保留此前新增的 `forward_with_aux()` hook，本阶段未改变 baseline 默认 `forward()` 行为。
- `ocr_training/configs/main.yaml` 未修改。
- 未进入 V2-M03。
- 未实现 insert/delete correction。

## 3. Cache Export Summary

主实验 cache 来源：

- Command:

```bash
python tools/export_parseq_corrector_cache.py \
  --checkpoint checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --split val \
  --output_dir outputs/V2-M02u_corrector_cache_replace_only_val \
  --batch_size 8 \
  --export_decoder_hidden true \
  --export_encoder_memory false \
  --topk 8 \
  --filter-mode replace_only \
  --scan-limit 3000 \
  --max-export 128 \
  --include-correct-ratio 0.2 \
  --device cuda
```

导出摘要：

- split: `val`
- resolved subset: `SLP34K_lmdb_test`
- scan-limit: `3000`
- max-export: `128`
- total-scanned: `1479`
- total-exported: `128`
- baseline-correct-samples: `1230`
- baseline-incorrect-samples: `249`
- replace-only-samples: `102`
- replace-dominant-samples: `139`
- insert-delete-samples: `147`
- correct-context-samples: `26`
- originally-wrong-positions: `988`
- originally-correct-positions: `19756`

`op_counts`:

- `correct = 19756`
- `replace = 514`
- `delete = 474`
- `insert = 522`
- `eos_late = 65`
- `eos_early = 76`

`top_replace_pairs`:

- `8->6`: `9`
- `8->9`: `9`
- `6->8`: `9`
- `9->8`: `8`
- `1->0`: `6`
- `8->3`: `6`
- `6->9`: `5`
- `1->2`: `5`
- `8->1`: `5`
- `O->A`: `5`

`confidence_stats`:

- `correct_token_conf_mean = 0.9987`
- `wrong_token_conf_mean = 0.9553`
- `low_conf_token_count = 94`

`length_stats`:

- `gt_len_mean = 14.06`
- `pred_len_mean = 14.03`
- `long_21plus_count = 274`

feature shapes:

- `pred_token_ids`: `[128, 51]`
- `gt_token_ids`: `[128, 51]`
- `pred_token_conf`: `[128, 51]`
- `topk_indices`: `[128, 51, 8]`
- `topk_values`: `[128, 51, 8]`
- `valid_length`: `[128]`
- `eos_position`: `[128]`
- `decoder_hidden`: `[128, 51, 768]`

## 4. Train/Eval Split Summary

Command:

```bash
python tools/split_mdiff_corrector_cache.py \
  --cache_dir outputs/V2-M02u_corrector_cache_replace_only_val \
  --output_dir outputs/V2-M02v_corrector_cache_replace_only_split \
  --train_ratio 0.8 \
  --seed 2026 \
  --stratify sample_type
```

split 结果：

- train sample count: `103`
- eval sample count: `25`
- train wrong positions: `156`
- eval wrong positions: `46`
- train correct-context samples: `21`
- eval correct-context samples: `5`
- sample_id overlap: `0`
- split seed: `2026`

评价：

- expanded cache 数量满足最小要求 `>=128`
- held-out eval 仍有真实错误位，适合做泛化验证
- 但 eval 只有 `25` 个样本、`46` 个 wrong positions，统计方差仍偏大

## 5. Training Results

### `token_decoder_hidden`

- contract type: `token_decoder_hidden`
- batch size: `8`
- max steps: `500`
- device: `cuda:0`
- peak memory: `608.17 MB`
- runtime: `7.24 s`
- dataset size: `103`
- baseline replace-only samples: `82`
- correct-context samples: `21`
- selected positions: `156`
- preserve positions: `1319`
- initial loss: `7.7907`
- final loss: `0.0082`
- initial selected CE: `6.4883`
- final selected CE: `0.0027`
- initial preservation CE: `6.5117`
- final preservation CE: `0.0276`

### `token_only`

- contract type: `token_only`
- batch size: `8`
- max steps: `500`
- device: `cuda:0`
- peak memory: `598.81 MB`
- runtime: `6.66 s`
- dataset size: `103`
- baseline replace-only samples: `82`
- correct-context samples: `21`
- selected positions: `156`
- preserve positions: `1319`
- initial loss: `8.1898`
- final loss: `0.0140`
- initial selected CE: `6.8680`
- final selected CE: `0.0024`
- initial preservation CE: `6.6090`
- final preservation CE: `0.0577`

训练观察：

- 两个模型都能在 train split 上收敛；
- `token_decoder_hidden` 的 preservation CE 明显更低，说明其更容易在保留正确 token 的同时做局部更新；
- `token_only` 也能拟合 train split，但 preservation 更脆弱。

## 6. Offline Evaluation Results

| model | split | baseline_acc | corrected_acc | gain | correction_rate | preservation_rate | harmful_change_rate | replace_error_reduction | oracle@1 | oracle@3 | oracle@5 | changed_token_count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token_decoder_hidden | train | 0.2039 | 0.3204 | 0.1165 | 0.1859 | 1.0000 | 0.0000 | 0.1859 | 1.0000 | 1.0000 | 1.0000 | 29 |
| token_decoder_hidden | eval | 0.2000 | 0.2000 | 0.0000 | 0.0217 | 0.9967 | 0.0000 | 0.0217 | 0.1957 | 0.3043 | 0.5000 | 2 |
| token_only | train | 0.2039 | 0.3204 | 0.1165 | 0.1859 | 0.9985 | 0.0000 | 0.1859 | 0.6026 | 0.9744 | 0.9936 | 31 |
| token_only | eval | 0.2000 | 0.2000 | 0.0000 | 0.0217 | 0.9967 | 0.0000 | 0.0217 | 0.0217 | 0.1739 | 0.3043 | 4 |

解释：

1. train split 上两者几乎一样，说明小数据下都能拟合 cache。
2. eval split 上两者默认阈值的 `correction_rate` 持平，且都没有 sample-level accuracy gain。
3. 但 `token_decoder_hidden` 的 `oracle@1/3/5` 明显高于 `token_only`，说明 hidden 分支在 held-out 样本上能给出更好的候选分布。
4. 两者 `harmful_change_rate = 0.0`，说明 conservative inference 仍然有效。

## 7. Threshold Sweep

只对 `token_decoder_hidden` 的 held-out eval split 做阈值 sweep。

| tau_corr | tau_keep | delta_gain | correction_rate | preservation_rate | harmful_change_rate | changed_token_count |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.90 | 0.05 | 0.0217 | 0.9967 | 0.0000 | 2 |
| 0.60 | 0.80 | 0.00 | 0.0652 | 0.9967 | 0.0000 | 5 |
| 0.90 | 0.95 | 0.10 | 0.0217 | 1.0000 | 0.0000 | 1 |

结论：

- 放松 gating 可以提升 held-out `correction_rate`；
- 但 `corrected_accuracy` 仍没有提升；
- `harmful_change_rate` 没有恶化，说明阈值仍可继续校准；
- 当前主瓶颈不是“阈值一改就好”，而是 held-out 候选虽更好，但不足以稳定修正整条样本。

## 8. Case Study

以下都来自 held-out eval split。

### Case 1: no-change but oracle contains GT

- sample_id: `SLP34K_lmdb_test:000000001`
- GT: `浙富阳货00268`
- baseline: `浙桐庐货00268`
- hidden corrected: `浙桐庐货00268`
- token_only corrected: `浙桐庐货00268`
- changed positions: `[]`
- reason: `oracle_contains_gt_but_no_change`

说明：候选里已有 GT，但 conservative update 没有触发，属于“候选质量存在但 gating/selected-mask 未放行”的典型例子。

### Case 2: no-change but oracle contains GT

- sample_id: `SLP34K_lmdb_test:000000187`
- GT: `浙平湖货02308`
- baseline: `浙平湖货02366`
- hidden corrected: `浙平湖货02366`
- token_only corrected: `浙平湖货02366`
- changed positions: `[]`
- reason: `oracle_contains_gt_but_no_change`

说明：数字替换错误是主要模式，但默认阈值下 corrector 过于保守。

### Case 3: hidden changed, token_only kept, but still failed

- sample_id: `SLP34K_lmdb_test:000000752`
- GT: `浙萧山货23736杭州港`
- baseline: `浙萧山货23721杭州港`
- hidden corrected: `浙萧山货23731杭州港`
- token_only corrected: `浙萧山货23721杭州港`
- hidden changed positions: `[7]`
- success/failure: hidden 做了局部替换，但仍未命中 GT

说明：这是 hidden 分支在 held-out 上“有动作但修不准”的例子，反映 candidate quality 虽高于 token_only，但还不够稳定。

### Case 4: token_only changed, hidden kept, both failed

- sample_id: `SLP34K_lmdb_test:000001092`
- GT: `浙嘉善货03298嘉兴ZHEJIASHANHUOJIAXING`
- baseline: `浙嘉善货03266嘉兴ZHEJIASHANHUOJIAXING`
- hidden corrected: `浙嘉善货03266嘉兴ZHEJIASHANHUOJIAXING`
- token_only corrected: `浙嘉善货03238嘉兴ZHEJIASHANHUOJIAXING`
- token_only changed positions: `[7, 8]`
- success/failure: token_only 改动更激进，但没有修正成功

说明：token-only 更容易受语言先验驱动，可能会更主动改 token，但不一定更接近视觉证据。

### Case 5: both changed, both failed

- sample_id: `SLP34K_lmdb_test:000001122`
- GT: `苏盐城货091800`
- baseline: `苏盐城货091808`
- hidden corrected: `苏盐城货091898`
- token_only corrected: `苏盐城货091832`
- hidden changed positions: `[8]`
- token_only changed positions: `[8, 9]`
- success/failure: 两者都尝试修正，但都没有落到 GT

说明：这是当前 held-out 泛化的核心问题，模型能感知“这里该改”，但还不能稳定选对 replacement token。

### Case 6: semantic region failure with oracle hint

- sample_id: `SLP34K_lmdb_test:000000430`
- GT: `苏徐州货8888`
- baseline: `苏振航货6888`
- hidden corrected: `苏振航货6888`
- token_only corrected: `苏振航货6888`
- changed positions: `[]`
- reason: `oracle_contains_gt_but_no_change`

说明：这种区域名 + 数字混合错误更难，仅靠当前 local replace-only corrector 还不够。

## 9. Analysis

### 1. expanded cache 是否足够

足够完成第一轮 split-based validation，但还不够支撑稳定结论。

- 优点：`128` 条主实验样本已经显著强于 V2-M02u 的 overfit 设定。
- 问题：held-out eval 只有 `25` 条样本、`46` 个 wrong positions，规模仍偏小。

### 2. corrector 是否在 eval split 泛化

有，但很弱。

- `token_decoder_hidden` eval `correction_rate = 0.0217 > 0`
- 放松 gating 后可到 `0.0652`

这证明 corrector 不是只会记住 train cache，但泛化强度还不够转化为 sample-level accuracy gain。

### 3. decoder_hidden 是否优于 token_only

如果只看默认阈值下的 eval `correction_rate`，两者持平；如果看 candidate quality，`decoder_hidden` 仍优于 `token_only`。

关键证据：

- eval `oracle@1`: `0.1957 > 0.0217`
- eval `oracle@3`: `0.3043 > 0.1739`
- eval `oracle@5`: `0.5000 > 0.3043`
- loose gating 下 hidden 的 `correction_rate` 可提升到 `0.0652`

因此，更准确的结论是：

- `decoder_hidden` 在 held-out 上提供了更强的候选质量；
- 但当前 pipeline 还没把这部分优势稳定转成默认阈值下的 sample-level gain。

### 4. harmful change 是否可控

可控。

- 默认与 sweep 设置下，`harmful_change_rate` 都是 `0.0`
- eval `preservation_rate` 始终接近 `1.0`

这说明 conservative inference 设计是有效的。

### 5. 阈值是否是主要瓶颈

不是唯一瓶颈，但确实影响 correction activation。

- loose gating 让 `correction_rate` 从 `0.0217` 提到 `0.0652`
- 但 `corrected_accuracy` 仍然没有提升

所以问题不仅在 gating，也在 candidate 质量和训练样本规模。

### 6. oracle@K 是否显示 candidate quality

是，尤其对 `token_decoder_hidden` 很明显。

eval 上：

- hidden `oracle@5 = 0.5000`
- token_only `oracle@5 = 0.3043`

说明 hidden 分支确实学到比 token-only 更有价值的视觉/序列对齐信息。

### 7. 是否需要 confusion-aware synthetic noise

需要，而且应作为下一阶段优先项。

原因：

- 真实错误以 `6/8/9/1/0` 等视觉混淆为主；
- held-out 上模型知道“该改”，但未必选对具体字符；
- confusion-aware noise 更适合补足这种局部替换泛化。

### 8. 是否需要 encoder_memory branch

当前不需要进入主线。

现阶段更合理的顺序是：

1. 扩大 real-error split；
2. 加 confusion-aware synthetic noise；
3. 校准 selected-mask / thresholds；
4. 之后再判断 `encoder_memory` 是否值得接入。

### 9. 是否建议继续 corrector 路线

建议继续，但继续的方式应是“加强数据与泛化训练”，不是“立刻加复杂结构”。

当前证据链是成立的：

- pipeline 成立；
- real-error smoke 有正增益；
- split-based eval 上仍有非零 correction；
- hidden 候选质量优于 token-only；
- harmful change 可控。

不足之处是：

- sample-level eval gain 尚未建立；
- held-out 规模仍小；
- 泛化仍偏弱。

## 10. Recommendation

推荐下一步进入 `V2-M02w: confusion-aware synthetic noise`。

建议顺序：

1. `V2-M02w`
   - 在现有 `token_decoder_hidden` corrector 上加入 confusion-aware synthetic noise；
   - 保持 replace-only，不引入 insert/delete；
   - 继续使用 train/eval split 验证是否提升 held-out `correction_rate` 和 `corrected_accuracy`。

2. `V2-M02y`
   - 做 selected-mask / threshold calibration；
   - 尝试把 hidden 的更高 `oracle@K` 转成实际 correction gain。

3. `V2-M02z`
   - 扩大 split validation 到更多 hard/OOV/test slice。

当前不建议：

- 直接进入 V2-M03；
- 直接上 encoder_memory branch；
- 直接追求 full benchmark 结论。
