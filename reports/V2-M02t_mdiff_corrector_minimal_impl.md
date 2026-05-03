# V2-M02t MDiff Corrector Minimal Implementation

说明：本阶段只实现 offline cache export + small-sample smoke validation。未修改 `configs/main.yaml` 默认模型，未覆盖已有 `slp_mdiff`，未实现 insert/delete correction，未进入 V2-M03。Python 运行环境使用 `conda activate slpr_ocr`；由于沙箱内 `torch.cuda.is_available()==False`，实际 exporter/train/eval 通过无沙箱命令使用 GPU 完成。

## 1. Summary

- 已完成 `MDiff as PARSeq corrector` 的最小可运行版本：
  - baseline `forward_with_aux()` hook
  - offline cache exporter
  - `slp_mdiff_corrector` 模型包
  - smoke training script
  - offline evaluator
- cache export 成功，`decoder_hidden` 成功导出，cache 结构符合 `[N, 51, ...]` 固定长度设计。
- corrector smoke 通过：
  - `loss` 从 `7.8563` 降到 `0.0282`
  - `selected_ce` 从 `6.5577` 降到 `0.0103`
  - `preservation_ce` 从 `6.4934` 降到 `0.0894`
- offline evaluator 通过，能输出 correction-specific metrics。
- 本次 `limit=64` 的 train cache 恰好全部是 baseline 正确样本：
  - `baseline_accuracy = 1.0`
  - `originally_wrong_positions = 0`
  - 因此 `correction_rate / oracle@K / replace_error_reduction` 全为 `0`
  - 这说明 **pipeline 成立，但当前样本切片不适合验证真实 correction gain**

## 2. Files Added / Modified

### Added

- [ocr_training/strhub/models/slp_mdiff_corrector/__init__.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff_corrector/__init__.py)
- [ocr_training/strhub/models/slp_mdiff_corrector/modules.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff_corrector/modules.py)
- [ocr_training/strhub/models/slp_mdiff_corrector/system.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff_corrector/system.py)
- [ocr_training/configs/model/slp_mdiff_corrector.yaml](/mnt/data/zyx/SLP34K_v2/ocr_training/configs/model/slp_mdiff_corrector.yaml)
- [ocr_training/tools/mdiff_corrector_utils.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/mdiff_corrector_utils.py)
- [ocr_training/tools/export_parseq_corrector_cache.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/export_parseq_corrector_cache.py)
- [ocr_training/tools/train_mdiff_corrector_smoke.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/train_mdiff_corrector_smoke.py)
- [ocr_training/tools/eval_mdiff_corrector_offline.py](/mnt/data/zyx/SLP34K_v2/ocr_training/tools/eval_mdiff_corrector_offline.py)

### Modified

- [ocr_training/strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/maevit_infonce_plm/system.py)
  - 新增 `forward_with_aux(...)`
  - 保持原 `forward()` 默认行为不变

## 3. Baseline Export

### Checkpoint and Data

- checkpoint:
  - `checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- requested split/subset:
  - `--split train --subset unified_lmdb`
- actual resolved dataset:
  - `data/train/SLP34K_lmdb_train`
- 说明：
  - train split 不存在 `unified_lmdb` 这种 test-style subset 目录
  - exporter 按实际 train LMDB root 解析并在 summary 中记录 `resolved_subset = SLP34K_lmdb_train`

### Output Paths

- manifest:
  - [manifest.jsonl](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02t_corrector_cache_smoke_gpu2/manifest.jsonl)
- feature shard:
  - [features_0000.npz](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02t_corrector_cache_smoke_gpu2/features_0000.npz)
- export summary:
  - [export_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02t_corrector_cache_smoke_gpu2/export_summary.json)

### Exported Fields

- `gt_text`
- `pred_text`
- `gt_token_ids`
- `pred_token_ids`
- `pred_token_conf`
- `topk_indices`
- `topk_values`
- `decoder_hidden`
- `valid_length`
- `eos_position`
- `sample_id`
- `lmdb_index`
- `alignment_ops`
- `metadata` if available

本次 `train` 样本的 `metadata` 为 `null`。

### Shape Summary

- `pred_token_ids`: `[64, 51]`
- `gt_token_ids`: `[64, 51]`
- `pred_token_conf`: `[64, 51]`
- `topk_indices`: `[64, 51, 8]`
- `topk_values`: `[64, 51, 8]`
- `decoder_hidden`: `[64, 51, 768]`
- `valid_length`: `[64]`
- `eos_position`: `[64]`

### Sanity Checks

- manifest rows: `64`
- `pred_token_ids` range: `[0, 570]`
- `gt_token_ids` range: `[0, 570]`
- `valid_length` range: `[2, 36]`
- `eos_position` range: `[2, 36]`

### Alignment Op Counts

- `correct = 820`
- `replace = 0`
- `insert = 0`
- `delete = 0`
- `replace_only_samples = 64`
- `insert_delete_samples = 0`

直接结论：

- alignment helper 可运行；
- 但这 64 个 train 样本全是 baseline 正确样本，没有真实 replace 错误进入 cache。

## 4. Corrector Architecture

### Contract Type

- `token_decoder_hidden`

### Inputs

- `pred_token_ids [B, T]`
- `pred_token_conf [B, T]`
- `correction_mask [B, T]`
- `decoder_hidden [B, T, D]`

第一版默认：

- `use_encoder_memory = false`

### Fusion Method

- token embedding
- confidence scalar embedding
- correction mask embedding
- decoder hidden linear projection
- learned positional embedding
- additive fusion
- `TransformerEncoder` blocks

### Output Head

- final norm
- linear head
- logits `[B, T, 569]`

### Output Classes

- `EOS + 568 chars = 569`
- 不输出 `BOS / PAD / MASK`

## 5. Loss and Inference

### Loss

实现：

- `selected-position CE + preservation CE`

公式：

`L = CE(logits[selected_mask], gt[selected_mask]) + lambda_preservation * CE(logits[preserve_mask], gt[preserve_mask])`

本次配置：

- `lambda_preservation = 0.2`

实现约束：

- PAD 不进入 loss
- selected mask 不选 EOS
- selected 为空时，correction CE 置零，但 preservation loss 仍可保留
- target id 强制限制在 `[0, 568]`

### Conservative Inference Rule

对每个 token position：

```text
if not selected_mask[t]:
    keep
elif corr_id == base_id:
    keep
elif corr_conf < tau_corr:
    keep
elif base_conf >= tau_keep and (corr_conf - base_conf) < delta_gain:
    keep
else:
    change
```

本次阈值：

- `tau_low = 0.70`
- `tau_corr = 0.80`
- `tau_keep = 0.90`
- `delta_gain = 0.05`

实现约束：

- 不修改 EOS
- 不修改 PAD
- 不做插入
- 不做删除
- 不改长度

## 6. Smoke Training Results

### Run

- command:
  - `python tools/train_mdiff_corrector_smoke.py --cache_dir outputs/V2-M02t_corrector_cache_smoke_gpu2 --output_dir outputs/V2-M02t_corrector_smoke_gpu2 --contract_type token_decoder_hidden --replace_only true --loss_type selected_plus_preservation --lambda_preservation 0.2 --batch_size 8 --max_steps 100 --device cuda`
- device:
  - `cuda:0`
- runtime:
  - `8.37 sec`
- peak memory:
  - `608.17 MB`

### Loss

- `initial_loss = 7.8563`
- `final_loss = 0.0282`
- `initial_selected_ce = 6.5577`
- `final_selected_ce = 0.0103`
- `initial_preservation_ce = 6.4934`
- `final_preservation_ce = 0.0894`

### Training Data Composition

- `dataset_size = 64`
- `baseline_sample_count = 0`
- `synthetic_sample_count = 64`

说明：

- 因为本次 cache 的 64 个 baseline 样本全部预测正确，没有真实 replace 错误；
- smoke train 实际上完全由 `GT synthetic local noise` 样本驱动；
- 这仍然满足 V2-M02t 的最小目标：验证 corrector contract、loss、训练与保存链路成立。

### Checkpoint

- [last.ckpt](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02t_corrector_smoke_gpu2/checkpoints/last.ckpt)

## 7. Offline Evaluation

### Output

- [eval_summary.json](/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02t_corrector_smoke_gpu2/eval_summary.json)

### Metrics

- `baseline_accuracy = 1.0`
- `corrected_accuracy = 1.0`
- `gain_accuracy = 0.0`
- `correction_rate = 0.0`
- `preservation_rate = 1.0`
- `harmful_change_rate = 0.0`
- `replace_error_reduction = 0.0`
- `oracle@1 = 0.0`
- `oracle@3 = 0.0`
- `oracle@5 = 0.0`
- `changed_token_count = 0`
- `changed_sample_rate = 0.0`
- `originally_wrong_positions = 0`
- `originally_correct_positions = 884`

解释：

- evaluator 本身是通的；
- conservative inference 也按设计没有改动 EOS/长度，也没有对正确样本做 harmful change；
- 但因为这 64 个 baseline 样本原本就全对，无法观察 correction gain。

## 8. Compatibility

已确认：

- 原 baseline `forward()` 默认行为不变
- `main.yaml` 未修改
- 未处理 insert/delete
- 未处理 EOS length edit
- 未进入 V2-M03

额外一致性检查：

- 单样本 GPU 检查 `forward(images, max_length=...)` 与 `forward_with_aux(images, max_length=...)[\"logits\"]`
  - `forward_equal = True`
  - `max_abs_diff = 0.0`

因此当前 hook 是 non-invasive 的。

## 9. Risks and Next Step

### Risks

1. 本次导出的 `train limit=64` 样本没有真实 baseline 错误，导致 offline correction 指标基本退化为 preservation check。
2. 当前 smoke train 全靠 synthetic local noise，尚未验证“真实 baseline 错误 -> corrector 修正”的效果。
3. 第一版默认不使用 `encoder_memory`，因此还没有回答 `decoder_hidden` 之外的视觉分支是否能进一步增益。

### 是否值得扩大 cache

- 值得。
- 下一轮应优先扩大 cache 或主动挖掘 baseline 错误样本，而不是继续在这 64 个全对样本上重复训练/评估。

### 是否需要 `encoder_memory` 分支

- 当前不急。
- 先把 `decoder_hidden` 分支在真实错误样本上跑通，再做 `use_encoder_memory=true` 的对照更合理。

### 是否需要 confusion-aware mask

- 需要，但不应在 V2-M02t 先做。
- 更合理的顺序是：
  - 先扩大 cache
  - 统计真实 confusion
  - 再决定 confusion-aware mask

### 是否建议进入 V2-M02u

- 建议进入 `V2-M02u`，但前提不是“加更复杂模型”，而是：
  - 扩大 cache
  - 优先采样真实 baseline 错误样本
  - 在真实 replace 错误上验证 `decoder_hidden corrector`

推荐下一步：

1. 导出更大 train cache，或导出 hard/OOV/low-confidence slice。
2. 增加“只收集 baseline 错误样本”的 exporter 过滤选项。
3. 在真实 replace 错误样本上复跑 smoke/short-train。
4. 之后再决定是否加入 `encoder_memory` 分支和 confusion-aware mask。

