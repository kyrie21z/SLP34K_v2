# V2-M02u Real-error Corrector Validation

说明：本阶段目标是验证 real-error corrector，而不是完整 benchmark。未修改 `configs/main.yaml` 默认模型，未改变原 baseline `forward()` 默认行为，未实现 insert/delete correction，未进入 V2-M03。由于 `train` 前 2000 样本扫描结果为 baseline 全对，主验证切换到 `val` real-error slice；该 slice 用于分析和 smoke/overfit validation，不作为正式泛化结论。

## 1. Summary

- **是否成功导出真实错误样本：成功。**
  - `train` 扫描前 `2000` 样本：`baseline_incorrect_samples = 0`
  - `val` 扫描：成功导出包含真实错误的 `incorrect` 和 `replace_only` cache
- **是否成功训练 corrector：成功。**
  - `token_decoder_hidden` real-error smoke：`loss 7.6240 -> 0.0674`
  - `token_only` real-error smoke：`loss 8.0276 -> 0.5918`
- **是否成功评估真实 correction：成功。**
  - `token_decoder_hidden` 在 real-error cache 上：
    - `correction_rate = 0.1881`
    - `corrected_accuracy = 0.3125`，相对 baseline `0.2031` 提升 `+0.1094`
    - `preservation_rate = 1.0`
    - `harmful_change_rate = 0.0`
- **corrector 是否能修正真实 replace 错误：能。**
  - real-error cache 上 `originally_wrong_positions = 202`
  - 成功修正比例非零
- **harmful change 是否可控：可控。**
  - 本轮 `token_decoder_hidden` 和 `token_only` 都没有把原本正确样本改错
- **是否建议扩大训练：建议。**
  - 当前结果已经证明 corrector 路线在 real-error slice 上成立，但仍是 smoke/overfit 级验证；下一步应扩大 cache，并建立 train/eval split。

## 2. Files Added / Modified

### 重点修改

- [ocr_training/tools/export_parseq_corrector_cache.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/export_parseq_corrector_cache.py)
  - 新增 `filter-mode / scan-limit / max-export / include-correct-ratio` 等筛选能力
  - 新增错误统计、replace pair 统计、置信度统计、长度统计
- [ocr_training/tools/train_mdiff_corrector_smoke.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/train_mdiff_corrector_smoke.py)
  - 新增 `token_only` 合同支持
  - real-error cache 上默认优先用 baseline 错误样本，不强制混 synthetic
  - correct-context 样本进入 preservation training
- [ocr_training/tools/eval_mdiff_corrector_offline.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/eval_mdiff_corrector_offline.py)
  - 新增 `token_only` 兼容
  - 新增 case study 输出
- [ocr_training/tools/mdiff_corrector_utils.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/mdiff_corrector_utils.py)
  - alignment 增强，返回细粒度 `steps`
  - 支持 replace pair 统计

### 相关已有文件

- [ocr_training/strhub/models/slp_mdiff_corrector/system.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff_corrector/system.py)
- [ocr_training/strhub/models/slp_mdiff_corrector/modules.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff_corrector/modules.py)

### 兼容性确认

- `maevit_infonce_plm/system.py` 本阶段未再改变默认行为
- `main.yaml` 未修改
- 原 baseline `forward()` 仍保持 non-invasive 兼容

## 3. Export Results

### 3.1 Train Scan Result

命令思路：

- `split=train`
- `filter-mode=replace_only`
- `scan-limit=2000`

结果：

- `baseline_correct_samples = 2000`
- `baseline_incorrect_samples = 0`
- `replace_only_samples = 0`
- `originally_wrong_positions = 0`

结论：

- exporter 正常
- 但 baseline 在这段 train slice 上太强，不适合做 real-error validation

### 3.2 Val Incorrect Cache

路径：

- [manifest.jsonl](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_incorrect_val/manifest.jsonl)
- [features_0000.npz](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_incorrect_val/features_0000.npz)
- [export_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_incorrect_val/export_summary.json)

设置：

- `split = val`
- `filter-mode = incorrect`
- `scan-limit = 3000`
- `max-export = 256`
- `include-correct-ratio = 0.2`

结果：

- `total_scanned = 1235`
- `total_exported = 256`
- `baseline_correct_samples = 1030`
- `baseline_incorrect_samples = 205`
- `replace_only_samples = 85`
- `replace_dominant_samples = 117`
- `insert_delete_samples = 120`
- `low_conf_samples = 60`
- `originally_wrong_positions = 854`
- `originally_correct_positions = 16485`

shape summary：

- `pred_token_ids [256, 51]`
- `gt_token_ids [256, 51]`
- `pred_token_conf [256, 51]`
- `topk_indices [256, 51, 8]`
- `topk_values [256, 51, 8]`
- `decoder_hidden [256, 51, 768]`

### 3.3 Val Replace-only Cache

路径：

- [manifest.jsonl](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_replace_only_val/manifest.jsonl)
- [features_0000.npz](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_replace_only_val/features_0000.npz)
- [export_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_cache_replace_only_val/export_summary.json)

设置：

- `split = val`
- `filter-mode = replace_only`
- `scan-limit = 3000`
- `max-export = 128`
- `include-correct-ratio = 0.2`

结果：

- `total_scanned = 1479`
- `total_exported = 128`
- `baseline_correct_samples = 1230`
- `baseline_incorrect_samples = 249`
- `replace_only_samples = 102`
- `replace_dominant_samples = 139`
- `insert_delete_samples = 147`
- `low_conf_samples = 76`
- `correct_context_samples = 26`
- `originally_wrong_positions = 988`
- `originally_correct_positions = 19756`

shape summary：

- `pred_token_ids [128, 51]`
- `gt_token_ids [128, 51]`
- `pred_token_conf [128, 51]`
- `topk_indices [128, 51, 8]`
- `topk_values [128, 51, 8]`
- `decoder_hidden [128, 51, 768]`

主训练/评估使用的 real-error cache 是这份 `replace_only_val` cache。

## 4. Real-error Dataset Analysis

### 错误样本是否足够

足够用于 smoke。

- replace-only exported wrong samples: `102`
- correct-context exported samples: `26`
- total exported samples: `128`
- `selected_positions_count = 202`

### replace 是否占主导

在 `incorrect_val` 扫描统计中：

- `replace = 438`
- `delete = 416`
- `insert = 417`

说明：

- 真实错误并不只是 replace，insert/delete 也很多
- 这进一步支持当前阶段先限制到 `replace_only` 子集做验证

### 最常见 replace pairs

来自 `replace_only_val` cache：

- `8 -> 6`
- `8 -> 9`
- `6 -> 8`
- `9 -> 8`
- `1 -> 0`
- `8 -> 3`
- `6 -> 9`
- `1 -> 2`
- `8 -> 1`
- `O -> A`

这与 V2-M02s 预期的视觉混淆模式一致。

### low-confidence 与 wrong token 的关系

`replace_only_val` 扫描统计：

- `correct_token_conf_mean = 0.9987`
- `wrong_token_conf_mean = 0.9553`
- `low_conf_token_count = 94`

结论：

- 错误 token 平均置信度低于正确 token
- 但差距没有大到足以完全线性分离
- 这说明 low-confidence mask 有用，但并不等于 oracle

### 是否适合 corrector 训练

适合。

原因：

1. 存在真实 replace-only 错误样本
2. `decoder_hidden` 已导出
3. correct-context 样本可用于 preservation
4. visual confusion pairs 与 corrector 假设匹配

## 5. Corrector Training Results

### Token + Decoder Hidden

输出：

- [train_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_realerr_smoke/train_summary.json)
- [last.ckpt](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_realerr_smoke/checkpoints/last.ckpt)

配置：

- `contract_type = token_decoder_hidden`
- `batch_size = 8`
- `max_steps = 200`
- `device = cuda:0`

结果：

- `dataset_size = 128`
- `baseline_sample_count = 102`
- `correct_context_sample_count = 26`
- `synthetic_sample_count = 0`
- `selected_positions_count = 202`
- `preserve_positions_count = 1625`
- `initial_loss = 7.6240`
- `final_loss = 0.0674`
- `initial_selected_ce = 6.3160`
- `final_selected_ce = 0.0156`
- `initial_preservation_ce = 6.5398`
- `final_preservation_ce = 0.2588`
- `runtime_sec = 14.94`
- `peak_memory_mb = 608.17`

### Token Only

输出：

- [train_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_tokenonly_smoke/train_summary.json)
- [last.ckpt](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02u_corrector_tokenonly_smoke/checkpoints/last.ckpt)

配置：

- `contract_type = token_only`
- 其他与上面一致

结果：

- `initial_loss = 8.0276`
- `final_loss = 0.5918`
- `initial_selected_ce = 6.7065`
- `final_selected_ce = 0.1780`
- `initial_preservation_ce = 6.6051`
- `final_preservation_ce = 2.0687`
- `runtime_sec = 14.88`
- `peak_memory_mb = 597.02`

直接比较：

- `token_decoder_hidden` 收敛更快、更低
- `token_only` 的 preservation loss 收敛明显更差

## 6. Offline Evaluation Results

说明：以下是在 `replace_only_val` cache 上的 smoke/overfit evaluation，不是泛化结论。

| model | baseline_acc | corrected_acc | gain | correction_rate | preservation_rate | harmful_change_rate | replace_error_reduction | oracle@1 | oracle@3 | oracle@5 | changed_token_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token_decoder_hidden | 0.2031 | 0.3125 | 0.1094 | 0.1881 | 1.0000 | 0.0000 | 0.1881 | 1.0000 | 1.0000 | 1.0000 | 38 |
| token_only | 0.2031 | 0.3125 | 0.1094 | 0.1436 | 0.9975 | 0.0000 | 0.1436 | 0.6337 | 0.8663 | 0.9109 | 34 |

结论：

- 两个模型都能在 real-error cache 上产生非零 correction
- `token_decoder_hidden` 的 `correction_rate` 更高
- `token_decoder_hidden` 的 oracle@K 明显更强
- `token_only` preservation 略差，但仍未出现 harmful sample-level changes

## 7. Threshold Sweep

对 `token_decoder_hidden` 做了小阈值 sweep，只改 evaluator 阈值，不改训练：

| tau_corr | tau_keep | delta_gain | correction_rate | preservation_rate | harmful_change_rate | changed_token_count |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.90 | 0.05 | 0.1881 | 1.0000 | 0.0000 | 38 |
| 0.70 | 0.80 | 0.00 | 0.1881 | 1.0000 | 0.0000 | 38 |
| 0.90 | 0.95 | 0.10 | 0.1881 | 1.0000 | 0.0000 | 38 |

结论：

- 对当前这批样本，阈值 sweep 没有改变结果
- 说明已发生的改动都非常高置信；未改样本并不是被这几个阈值简单卡住
- 更可能的限制在模型本身的 candidate quality 或 selected-mask 覆盖，而不是 evaluator gating

## 8. Case Study

来自 `token_decoder_hidden` evaluator 的代表样本：

### Corrected Success

1. `GT = 浙平湖货02085ZHEPINGHUHUO`
   - baseline: `浙平湖货02055ZHEPINGHUHUO`
   - corrected: `浙平湖货02085ZHEPINGHUHUO`
   - changed position: `7`
   - `base_conf = 0.6494`
   - `corr_conf = 0.9966`

2. `GT = 皖仁和8899阜阳港WANRENHEFUYANGGANG`
   - baseline: `皖仁和8899阜阳港WANREXHEFUYANGGANG`
   - corrected: `皖仁和8899阜阳港WANRENHEFUYANGGANG`
   - changed position: `15`
   - `base_conf = 0.5664`
   - `corr_conf = 0.9949`

### Oracle Contains GT But No Change

1. `GT = 浙富阳货00268`
   - baseline: `浙桐庐货00268`
   - corrected: unchanged
   - 说明模型 top-k 内含 GT 候选，但最终没有触发正确替换

2. `GT = 汇凯88HUIKAI`
   - baseline: `汇凯99HUIKAI`
   - corrected: unchanged

3. `GT = 徐州XUZHOU`
   - baseline: `州州XUZHOU`
   - corrected: unchanged

### Harmful Changes

- 本轮未观察到 harmful sample-level changes

## 9. Analysis

1. **corrector 是否能修正真实 replace 错误？**
   - 能。
   - `token_decoder_hidden` 在 real-error cache 上达到 `correction_rate = 0.1881`。
2. **decoder_hidden 是否比 token_only 有帮助？**
   - 有帮助。
   - `correction_rate: 0.1881 > 0.1436`
   - `oracle@1/3/5` 全面高于 `token_only`
   - 训练收敛也明显更好
3. **conservative inference 是否过严？**
   - 当前证据不支持“只是阈值过严”。
   - 因为小阈值 sweep 几乎不改变结果。
4. **harmful change 是否可控？**
   - 可控。
   - 当前 `harmful_change_rate = 0.0`
   - `preservation_rate` 接近或等于 `1.0`
5. **是否应该扩大 cache？**
   - 应该。
   - 当前已经证明方向有效，但样本量仍然偏小，而且是 smoke/overfit 设置。
6. **是否需要 encoder_memory 分支？**
   - 现在还不是优先级最高。
   - 当前 `decoder_hidden` 分支已经显示增益，应先把它在更大 cache 上做稳。
7. **是否需要 confusion-aware mask？**
   - 需要，且现在已有数据支撑。
   - `8/6`, `8/9`, `6/8`, `9/8`, `1/0` 等高频混淆已经出现。

## 10. Recommendation

不建议进入 V2-M03。建议按以下顺序继续：

1. **V2-M02v：扩大 cache + train/eval split**
   - 优先扩大 `val/test hard slice` 或构造更多 `replace_only` / `replace_dominant` cache
   - 建立明确的 train/eval split，避免只做 overfit-style evaluation
2. **V2-M02w：加入 confusion-aware synthetic noise**
   - 基于本轮 top replace pairs 构造更贴近真实错误的 synthetic corruption
3. **V2-M02x：加入 encoder_memory branch**
   - 作为附加分支做 ablation，而不是替换 `decoder_hidden` 主线
4. **V2-M02y：selected-mask / calibration**
   - 当前阈值 sweep 不敏感，更值得研究 mask 覆盖率与 candidate generation，而不是单纯调 gate

当前阶段结论：

- corrector 路线**不应暂停**
- `decoder_hidden` 分支优于 `token_only`
- 下一步最值得做的是 **扩大 real-error cache，并做更严格的 split-based validation**

