# V2-M02h Official-core MDiff Receipt

## 1. Summary

- 已实现 SLP 适配版 official-core MDiff decoder。
- 已保留旧 `PlainMDiffDecoder`，可通过 `decoder_core: plain|official` 切换。
- 当前 `slp_mdiff.yaml` 默认改为：
  - `decoder_core: official`
  - `inference_mode: parallel`
  - `loss_mode: official_masked_normalized`
- 已实现 official-style parallel decoding：full `[MASK]` 输入，一步 decoder/head 输出 logits，不做多步 full-argmax 回灌。
- 已完成 Stage 0/1/2 smoke check。
- 已完成单卡 1000-step short training。
- 结论：official-core + parallel decoding 没有解决 EOS/repetition collapse。step 100 起 `eos_rate=1.0, avg_pred_len=0`，后期只有很短的重复片段，如 `AA` / `AAA0NNN`。
- 不建议进入 V2-M03。下一步应做 V2-M02h-fix：修 official-core decoder conditioning。

## 2. Files Changed

修改/新增文件：

- `ocr_training/strhub/models/slp_mdiff/modules.py`
  - 保留 `PlainMDiffDecoder`
  - 新增 `SinusoidalPositionalEncoding`
  - 新增 `OfficialStyleMlp`
  - 新增 `OfficialStyleTransformerBlock`
  - 新增 `OfficialCoreMDiffDecoder`
- `ocr_training/strhub/models/slp_mdiff/system.py`
  - 新增 `decoder_core`
  - 新增 `inference_mode`
  - 新增 `official_masked_normalized` loss
  - 新增 SLP special-id / shape / CE target assertions
  - 保留 Plain decoder 可切换
- `ocr_training/configs/model/slp_mdiff.yaml`
  - 默认切到 official core、parallel inference、official normalized loss
  - 默认启用 baseline encoder migration
- `ocr_training/tools/v2_m02h_official_core_train_check.py`
  - 新增 Stage 0/1/2/short-train 检查脚本
- `ocr_training/tools/v2_m02h_conditioning_check.py`
  - 新增 collapse 后最小 conditioning diagnostic 脚本
- `reports/V2-M02h_official_core_mdiff_receipt.md`
  - 本报告

生成输出：

- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/official_core_parallel_1000steps_precheck_summary.json`
- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/official_core_parallel_1000steps_dummy_forward_summary.json`
- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/official_core_parallel_1000steps_real_batch_summary.json`
- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/official_core_parallel_1000steps_train_summary.json`
- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/conditioning_summary.json`
- `ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/checkpoints/slp_mdiff_official_core_parallel_1000steps_last.ckpt`

## 3. Official-core Implementation Details

### Embedding

decoder input vocab 保持 SLP 体系：

- EOS + chars + BOS + PAD + MASK
- `input_vocab_size = 572`

`OfficialCoreMDiffDecoder` 使用现有 `TokenEmbedding`，即 `nn.Embedding` 后乘以 `sqrt(embed_dim)`。

实现位置：

- `ocr_training/strhub/models/slp_mdiff/modules.py:181-224`

### Fixed sinusoidal PE

新增 `SinusoidalPositionalEncoding`：

- 支持 `max_length=51`
- buffer shape: `[1, 51, 768]`
- forward: `hidden + pe[:, :length]` 后 dropout

实现位置：

- `ocr_training/strhub/models/slp_mdiff/modules.py:111-124`

### Learned PE

`OfficialCoreMDiffDecoder` 内部新增：

- `self.position_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))`
- 初始化 `trunc_normal_(std=0.02)`

forward 中：

```python
hidden = self.text_embed(noised_token_ids)
hidden = self.positional_encoding(hidden) + self.position_embed[:, :length]
```

实现位置：

- `ocr_training/strhub/models/slp_mdiff/modules.py:196-205`
- `ocr_training/strhub/models/slp_mdiff/modules.py:220-224`

### TransformerBlock

新增 `OfficialStyleTransformerBlock`：

1. self-attention，无 causal mask；
2. residual + dropout + `norm1`；
3. cross-attention，text hidden 作为 query，visual memory 作为 key/value；
4. residual + dropout + `norm2`；
5. MLP；
6. residual + dropout + `norm3`。

实现位置：

- `ocr_training/strhub/models/slp_mdiff/modules.py:145-178`

本阶段使用 PyTorch `nn.MultiheadAttention(batch_first=True)`，没有复制 OpenOCR 文件。

### Self-attention

- 不传 causal mask。
- 不实现 sample_k branch。
- Plain decoder 仍保留旧 `token_padding_mask` 行为；official core 不传 token padding mask。

### Cross-attention

- query: decoder text hidden `[B, 51, 768]`
- key/value: SLP MAE encoder memory `[B, 197, 768]`

新增 shape assertions：

- memory 必须是 rank-3；
- batch 必须匹配；
- memory dim 必须等于 `embed_dim`。

实现位置：

- `ocr_training/strhub/models/slp_mdiff/modules.py:207-224`

### Head / Output Projection

official core 下：

- `self.head = nn.Linear(embed_dim, output_num_classes, bias=False)`
- `output_num_classes = 569`

输出类别只包含：

- EOS
- 568 个真实字符

不输出：

- BOS
- PAD
- MASK

实现位置：

- `ocr_training/strhub/models/slp_mdiff/system.py:103-112`

### Special IDs

保持 SLP tokenizer 体系：

- `len(tokenizer) = 571`
- `eos_id = 0`
- `chars = 1..568`
- `bos_id = 569`
- `pad_id = 570`
- `mask_id = 571`
- `input_vocab_size = 572`
- `output_num_classes = 569`

断言位置：

- `ocr_training/strhub/models/slp_mdiff/system.py:114-152`

### Loss Implementation

新增 `official_masked_normalized`：

```text
loss_positions = masked_positions & non_pad
token_loss = CE(logits[loss_positions], labels[loss_positions], reduction='none')
token_loss = token_loss / p_mask_at_positions
token_loss = token_loss / (length + 1)_at_positions
loss = sum(token_loss) / batch_size
```

实现位置：

- `ocr_training/strhub/models/slp_mdiff/system.py:263-304`

### Inference Implementation

新增 `inference_mode`：

- `parallel`
- `iterative_full_feedback`

默认 `parallel`：

```python
input_ids = full MASK [B, 51]
hidden = decode(input_ids, memory)
logits = head(hidden)
return logits
```

不再执行：

```python
for step:
    input_ids = logits.argmax(dim=-1)
```

实现位置：

- `ocr_training/strhub/models/slp_mdiff/system.py:330-351`

## 4. SLP Compatibility

- tokenizer length: `571`
- EOS id: `0`
- BOS id: `569`
- PAD id: `570`
- MASK id: `571`
- input vocab size: `572`
- output classes: `569`
- max label length: `50`
- decoder sequence length: `51`
- encoder memory shape in conditioning check: `[5, 197, 768]`
- dummy logits shape: `[1, 51, 569]`
- probe/train logits shape: `[5, 51, 569]`
- baseline encoder migration:
  - baseline ckpt: `checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
  - encoder missing: `0`
  - encoder unexpected: `0`
- checkpoint loading compatibility:
  - 新 official checkpoint 可保存；
  - conditioning check 使用该 checkpoint 加载，`missing_key_count=0`，`unexpected_key_count=0`。
- Plain decoder 仍可通过 `decoder_core=plain` 切换。
- 为兼容旧 Plain checkpoint，`Model.__init__` 的 Python 默认仍是 `decoder_core="plain"` / `inference_mode="iterative_full_feedback"`；配置文件默认使用 official。

## 5. Loss and Inference

### official_masked_normalized loss

本阶段沿用现有 random/full/random_or_full noising，不引入 all-mask strategies，不引入 TRN。

变量定义：

- `labels = clean_targets = tokenizer.encode(labels)[:, 1:]`
  - 已去掉 BOS；
  - 包含真实字符、EOS、PAD；
- `masked_indices`
  - 当前 batch 中被替换成 MASK 的 non-PAD 位置；
- `p_mask`
  - 每条样本实际 `masked_valid_count / valid_count`；
  - full mask 样本为 `1.0`；
  - random mask 样本按实际 mask 比例计算；
  - clamp 到 `1e-6` 防止除零；
- `length`
  - 每条样本 non-PAD count；
  - 包括 EOS；
- loss 归一化中使用 `(length + 1)`，对齐官方 `length + 1` 风格。

PAD/EOS/MASK 处理：

- PAD 不进入 CE target positions；
- MASK 不会被 classifier 预测；
- BOS 不会被 classifier 预测；
- CE target 必须在 `[0, 568]`；
- EOS 是合法 target，id 为 `0`。

### parallel decoding 为什么用于本阶段

V2-M02d 的明显风险之一是 full-argmax iterative feedback：一旦 EOS 或某个字符早期占优，后续 step 会把错误结果作为输入继续强化。

V2-M02h 使用 official-style parallel decoding，目的是隔离这个反馈路径：

- 输入全 MASK；
- 一步 decode；
- 不把 argmax 回灌。

这不是论文最终最强 inference 策略，只是本阶段排查 decoder core 与 feedback collapse 的工程验证路径。

### 与 iterative_full_feedback 的区别

- `parallel`: 一步输出，不回灌；
- `iterative_full_feedback`: 保留旧行为，每步完整 argmax 回灌所有位置。

## 6. Smoke Test Results

### py_compile

命令：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
python -m py_compile strhub/models/slp_mdiff/modules.py strhub/models/slp_mdiff/system.py tools/v2_m02h_official_core_train_check.py
python -m py_compile tools/v2_m02h_conditioning_check.py
```

结果：通过。

### Hydra instantiate / precheck

结果：

```text
decoder_core: official
inference_mode: parallel
loss_mode: official_masked_normalized
tokenizer_len: 571
eos_id: 0
bos_id: 569
pad_id: 570
mask_id: 571
input_vocab_size: 572
output_num_classes: 569
head_bias: false
baseline encoder missing/unexpected: 0/0
```

### Dummy forward

输入：

- `images: [1, 3, 224, 224]`

输出：

- `logits: [1, 51, 569]`
- finite: `true`

### Real batch forward/backward

真实 batch：

- device: `cuda:0`
- image shape: `[2, 3, 224, 224]`
- examples:
  - `杭州港HANGZHOUGANG`
  - `兆平918ZHAOPING`

结果：

```text
loss: 6.8439049721
loss_finite: true
loss_numel: 17
masked_count: 17
pad_in_loss: 0
out_of_range_targets: 0
p_mask_mean: 0.59375
length_mean: 15.0
```

## 7. Short Training Diagnostics

训练设置：

- device: single GPU `cuda:0`
- batch size: `2`
- max_steps: `1000`
- precision: `16`
- num_workers: `0`
- freeze_encoder: `true`
- init_encoder_from_baseline_ckpt: `true`
- decoder_core: `official`
- inference_mode: `parallel`
- loss_mode: `official_masked_normalized`

| step | loss | eos_rate | avg_pred_len | repeat_ratio | unique_char_count | eos_prob_mean | eos_rank_mean | all_positions_same_top1_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | N/A | 0.000 | 51.00 | 0.672 | 8.40 | 0.0018 | 191.95 | 0.000 |
| 1 | 5.9461 | 0.000 | 51.00 | 0.672 | 8.40 | 0.0018 | 191.95 | 0.000 |
| 100 | 6.0851 | 1.000 | 0.00 | 0.776 | 0.00 | 0.0163 | 2.23 | 0.000 |
| 200 | 4.7980 | 1.000 | 0.00 | 0.896 | 0.00 | 0.0729 | 1.19 | 0.000 |
| 300 | 4.7545 | 1.000 | 0.00 | 0.884 | 0.00 | 0.0991 | 1.25 | 0.000 |
| 400 | 4.6878 | 1.000 | 0.00 | 0.804 | 0.00 | 0.0848 | 1.39 | 0.000 |
| 500 | 4.0554 | 1.000 | 5.00 | 0.732 | 2.40 | 0.0729 | 2.07 | 0.000 |
| 600 | 4.1785 | 1.000 | 5.00 | 0.728 | 2.40 | 0.0758 | 1.88 | 0.000 |
| 700 | 3.9877 | 1.000 | 1.80 | 0.740 | 1.00 | 0.0817 | 1.68 | 0.000 |
| 800 | 3.9385 | 1.000 | 2.80 | 0.720 | 1.40 | 0.0823 | 1.71 | 0.000 |
| 900 | 4.3694 | 1.000 | 2.80 | 0.724 | 1.40 | 0.0813 | 1.75 | 0.000 |
| 1000 | 4.2829 | 1.000 | 2.80 | 0.732 | 1.40 | 0.0815 | 1.75 | 0.000 |

`losses_finite: true`

## 8. Probe Predictions

Final step 1000 fixed probe samples:

| GT | Pred | pred_len | contains_eos | repeat_ratio | unique_char_count |
|---|---|---:|---|---:|---:|
| 浙萧山货23765 | AA | 2 | true | 0.720 | 1 |
| 六安港LUANGANG | AA | 2 | true | 0.700 | 1 |
| 苏常州货068SUCHANGZHOUHUO | AAA0NNN | 7 | true | 0.760 | 3 |
| 志远08 | AA | 2 | true | 0.720 | 1 |
| 翔运989XIANGYUN | A | 1 | true | 0.760 | 1 |

结论：

- step 0/1 不是 EOS collapse；
- step 100/200/300 已经全 EOS 截断为空；
- step 500 后出现短非空片段，但仍 `eos_rate=1.0`，且 repetition 很强；
- `all_positions_same_top1_ratio=0.0`，说明不再是“所有位置完全同一 top1”，但仍是 EOS + repetition collapse。

## 9. GPU and Runtime

- GPU: `NVIDIA RTX PRO 6000 Blackwell`
- batch size: `2`
- precision: `16`
- num_workers: `0`
- max_steps: `1000`
- peak allocated: `1.411 GB`
- peak reserved: `1.457 GB`
- 是否 <40GB: 是
- checkpoint size: `987 MB`

Checkpoint:

```text
/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/checkpoints/slp_mdiff_official_core_parallel_1000steps_last.ckpt
```

## 10. Collapse Comparison Against V2-M02d

V2-M02d old Plain decoder：

- step 200/300 后 `eos_rate=1.0`
- `avg_pred_len=0`
- 严重 EOS collapse
- 早期曾出现所有位置输出同一字符

V2-M02h official-core parallel：

- step 100 已 `eos_rate=1.0`
- step 100/200/300/400 `avg_pred_len=0`
- step 500 后有短非空输出，但仍全样本 contains EOS；
- final `avg_pred_len=2.8`
- final `unique_char_count=1.4`
- final probe 主要是 `AA`、`A`、`AAA0NNN`
- `all_positions_same_top1_ratio=0.0`

改善点：

- 移除 full-argmax feedback 后，不再是每个样本所有位置完全同一 top1。
- 能在后期输出极短非空片段。

未改善点：

- 仍在 step 100 快速进入全 EOS。
- 仍然存在强 repetition。
- 输出与 GT 基本无关。

最可能原因：

- conditioning 仍弱，decoder logits 对不同图像不够敏感。
- official-core 的结构移植降低了 full-feedback 风险，但没有解决图像条件利用问题。
- 当前训练 noising 仍很简化，且没有 TRN/LC/BLC，但在进入这些机制前，需要先修 conditioning。

### Conditioning Diagnostic

输出文件：

```text
/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02h_official_core_parallel_1000steps/conditioning_summary.json
```

关键结果：

```text
same_image_different_positions_cosine: 0.9589
different_images_same_position_cosine: 0.9957
real_vs_zero_memory.mean_abs_diff: 0.1046
real_vs_zero_memory.top1_changed_rate: 0.2784
real_vs_shuffled_memory.mean_abs_diff: 0.0835
real_vs_shuffled_memory.top1_changed_rate: 0.1490
position_embedding_zero_out.mean_abs_diff: 0.0097
position_embedding_zero_out.top1_changed_rate: 0.0235
```

解释：

- `different_images_same_position_cosine≈0.996` 很高，说明不同图片的同一位置 logits 极其相似。
- real memory vs zero/shuffled memory 能改变一部分 top1，说明 memory 不是完全无效；
- 但变化幅度不足，仍不足以形成可用的 image-conditioned decoding。
- learned position embedding zero-out 影响很小，说明当前模型主要没有靠 learned PE 区分输出。

## 11. Compatibility

确认：

- 未修改 `ocr_training/configs/main.yaml`。
- 未修改原 `maevit_infonce_plm/system.py` 行为。
- 未修改数据。
- 未删除任何文件。
- 未进入 V2-M03。
- 未实现 SLP-aware TRN。
- 未实现 generic TRN / reflect loss。
- 未实现完整 all-mask strategies。
- 未实现 LC/BLC remask。
- 未实现 segment-aware denoising。
- 未实现 pinyin consistency。
- Plain decoder 仍可通过 `decoder_core=plain` 切换。
- 已额外执行 Plain switch precheck：
  - `decoder_core=plain`
  - `inference_mode=iterative_full_feedback`
  - `loss_mode=masked_or_eos`
  - instantiate 通过，`head_bias=true`

注意：

- repo 中 `maevit_infonce_plm/system.py`、`maevit_plm/system.py`、`models/utils.py` 已有本任务前存在的本地修改；V2-M02h 没有修改这些文件。

## 12. Recommendation

推荐顺序：

1. **V2-M02h-fix：修 official-core decoder conditioning**
   - 当前最小下一步。
   - 重点查：
     - decoder cross-attn 是否有效利用 MAE memory；
     - CLS token 是否应去除；
     - visual adapter 是否需要 `linear_ln` / norm；
     - official-style block 的 norm/attention 实现是否还需更贴近 OpenOCR custom attention；
     - training target 是否过度鼓励 EOS；
     - full-mask/random mask mix 是否造成初期 EOS 捷径。

2. **V2-M02f：EOS bias / length penalty**
   - 仅作为轻量诊断或临时抑制 EOS 的工具。
   - 不应掩盖 conditioning 问题。

3. **V2-M02i：实现 official LC/BLC inference**
   - 暂缓。
   - 在 parallel decoding 已经能产生非空、非强重复、图像相关输出后再做。

4. **暂停并重审架构**
   - 如果 conditioning fix 后仍无法让 real/shuffled/zero memory 产生足够差异，则需要重审 MAE memory 到 MDiff decoder 的接口。

5. **V2-M03：all-mask strategies + generic TRN**
   - 当前不建议进入。
   - 条件未满足：official-core parallel decoding 没有明显缓解 collapse，也没有输出稳定的非空、非单字符重复文本。
