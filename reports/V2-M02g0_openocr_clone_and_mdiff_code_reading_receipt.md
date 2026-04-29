# V2-M02g0 OpenOCR Clone and MDiff Code Reading Receipt

## 1. Summary

- 已成功将官方 OpenOCR 参考仓库 clone 到 `/mnt/data/zyx/OpenOCR_ref`。
- `OpenOCR_ref` 当前位于 `main` 分支，commit 为 `ae934e5699e820b4e27b42f667629fb1505c0b1b`。
- 已找到官方 `MDiffDecoder`：`/mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py`。
- 已找到官方 MDiff4STR 配置：
  - `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml`
  - `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/svtrv2_mdiffdecoder_small.yml`
- 当前 SLP34K_v2 的 `PlainMDiffDecoder` 不是 OpenOCR 官方 `MDiffDecoder` 的代码级直接参考实现，而是一个思想级简化实现。
- 不建议继续把当前 Plain decoder 作为主线优化对象；它更适合作为诊断 baseline。
- 建议 V2-M02h 优先移植 OpenOCR `MDiffDecoder` core 到 SLP34K_v2，但应做适配实现，不要把 OpenOCR 文件直接复制进项目。
- 不建议现在进入 V2-M03。`all-mask strategies`、generic TRN、LC/BLC remask、segment-aware denoising、pinyin consistency 都应等官方 decoder core 对齐后再做。

## 2. OpenOCR Repository Status

- path: `/mnt/data/zyx/OpenOCR_ref`
- branch: `main`
- latest commit: `ae934e5 Merge pull request #200 from YesianRohn/patch-1`
- full SHA: `ae934e5699e820b4e27b42f667629fb1505c0b1b`
- `git status`: `nothing to commit, working tree clean`
- 是否 clean: 是

最近 5 个 commit：

```text
ae934e5 Merge pull request #200 from YesianRohn/patch-1
196a4b5 Disable cuDNN benchmarking and fix global step parsing
45bfc0c add finetune unirec
a5be92a fix: avoid occasional 799 resize output for fixed 800x800 input
b1b8346 style: simplify resize fix comment wording
```

## 3. MDiff4STR File Map

- official decoder file: `/mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py`
- official mdiff4str readme: `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/readme.md`
- official base config: `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml`
- official small config: `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/svtrv2_mdiffdecoder_small.yml`
- official label/noising encode: `/mnt/data/zyx/OpenOCR_ref/openrec/preprocess/mdiff_label_encode.py`
- official loss wrapper: `/mnt/data/zyx/OpenOCR_ref/openrec/losses/mdiff_loss.py`
- shared decoder utilities: `/mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/nrtr_decoder.py`
- 论文机制对照使用页面：
  - https://arxiv.org/abs/2512.01422v1
  - https://arxiv.org/html/2512.01422v1

`find . -iname "*mdiff*"` 的结果：

```text
./configs/rec/mdiff4str
./configs/rec/mdiff4str/svtrv2_mdiffdecoder_small.yml
./configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml
./openrec/preprocess/mdiff_label_encode.py
./openrec/losses/mdiff_loss.py
./openrec/modeling/decoders/mdiff_decoder.py
```

## 4. Official MDiffDecoder Code Reading

### 4.1 类定义位置

官方 `MDiffDecoder` 定义在：

- `/mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py:9`

### 4.2 `__init__` 参数

`__init__` 参数包括：

- `in_channels`
- `out_channels`
- `nhead`
- `num_decoder_layers`
- `max_len`
- `attention_dropout_rate`
- `residual_dropout_rate`
- `scale_embedding`
- `parallel_decoding`
- `autoregressive_decoding`
- `sampler_step`
- `low_confidence_decoding`
- `random_mask_decoding`
- `semi_autoregressive_decoding`
- `cloze_mask_decoding`
- `rec_loss_weight`
- `reflect_loss_weight`
- `sample_k`
- `temperature`

证据：`mdiff_decoder.py:28-47`。

### 4.3 `mask_token_id`

官方定义：

- `self.mask_token_id = out_channels - 2`

原因是 `MDiffLabelEncode.add_special_char()` 将特殊 token 布局设为：

```text
[EOS] + characters + [MASK, PAD]
```

因此：

- EOS: `0`
- MASK: `out_channels - 2`
- PAD: `out_channels - 1`

证据：`mdiff_decoder.py:49-52`，`mdiff_label_encode.py:310-312`。

### 4.4 `ignore_index`

官方定义：

- `self.ignore_index = out_channels - 1`

也就是 PAD id。训练 CE 中使用 `ignore_index=self.ignore_index` 忽略 PAD。

证据：`mdiff_decoder.py:49-52`，`mdiff_decoder.py:125-131`，`mdiff_decoder.py:142-145`。

### 4.5 token embedding

官方 token embedding 使用 OpenOCR 的 `Embeddings`：

- `vocab=self.out_channels`
- `padding_idx=0`
- `scale_embedding=scale_embedding`

`Embeddings` 本身是 `nn.Embedding`，权重初始化为 `std=d_model**-0.5`，forward 时可乘以 `sqrt(d_model)`。

一个需要注意的细节：OpenOCR 这里传入 `padding_idx=0`，而官方 MDiff 特殊 token 中 id `0` 是 EOS，不是 PAD。这个实现细节不应在 SLP34K_v2 中盲目照搬，V2-M02h 应做 shape/id 断言。

证据：`mdiff_decoder.py:68-73`，`nrtr_decoder.py:426-439`。

### 4.6 position embedding / query embedding

官方没有单独的 query embedding。文本 token embedding 后叠加两种位置信号：

- 固定 sinusoidal `PositionalEncoding`
- 可学习 `pos_embed`，shape 为 `[1, max_len + 1, d_model]`

这比当前 SLP Plain decoder 的单一 learned position table 更强。

证据：`mdiff_decoder.py:74-80`，`nrtr_decoder.py:326-353`。

### 4.7 decoder layer

官方 decoder 是 `nn.ModuleList`，内部堆叠 `TransformerBlock`。每个 block 启用：

- self-attention
- cross-attention
- MLP
- residual dropout
- LayerNorm

官方 block 使用 OpenOCR 自定义 `MultiheadAttention`，不是 PyTorch `nn.MultiheadAttention`。

证据：`mdiff_decoder.py:81-91`，`mdiff_decoder.py:524-587`。

### 4.8 self-attention 是否有 mask

官方 `forward_train`、`forward_decoding`、各 inference path 都传入 `self_mask=None`。因此 MDiff decoder 的 self-attention 是双向、无 causal mask 的。

证据：`mdiff_decoder.py:122-124`，`mdiff_decoder.py:240-246`，`mdiff_decoder.py:570-575`。

### 4.9 cross-attention 如何接收视觉 feature

官方 cross-attention 使用：

- text hidden states 作为 query
- image encoder memory 作为 key/value

`MDiffDecoder.forward()` 注释里说明 `src` 形状为 `(B, sN, C)`，target 为 `(B, tN, C)`。

当 `sample_k > 0` 时，官方在 self-attn 与 cross-attn 之间 reshape，把 `sample_k` 和 token 维度组织成适合 cross-attn 的形状。

证据：`mdiff_decoder.py:205-212`，`mdiff_decoder.py:577-585`。

### 4.10 output projection / classifier

官方 classifier 为：

- `self.tgt_word_prj = nn.Linear(d_model, self.out_channels - 2, bias=False)`

也就是说输出类别只包含：

- EOS
- 真实字符

不输出：

- MASK
- PAD

证据：`mdiff_decoder.py:95-104`。

### 4.11 `forward_train`

`forward_train` 输入 data 解包为：

```python
labels, reflect_ids, noisy_batch, masked_indices, p_mask, length = data
```

执行流程：

1. 将 `noisy_batch` 送入 token embedding。
2. 叠加 fixed PE 与 learned PE。
3. 经过所有 decoder layers。
4. classifier 得到 logits。
5. 只在 `masked_indices` 上计算 denoising CE。
6. CE 使用 `ignore_index=self.ignore_index`。
7. loss 会除以 `p_mask` 和 `length + 1` 做归一化。
8. 如果 `reflect_ids is not None`，额外跑一条 reflect/token replacement 分支，计算 full-sequence correction loss。

证据：`mdiff_decoder.py:112-148`。

### 4.12 `forward_train_all`

`forward_train_all` 是 `sample_k > 0` 时使用的训练分支。

它期望每个样本有 K 组：

- `reflect_ids_all`
- `noisy_batch_all`
- `masked_indices_all`
- `p_mask_all`

流程：

1. 将 K 组 noised tokens flatten 后 embedding。
2. reshape 成 `[bs, sample_k, L, dim]`。
3. 经过 decoder。
4. 对每个 k 分别计算 denoising loss 与 reflect loss。
5. 最后对 K 个 loss 求平均。

证据：`mdiff_decoder.py:150-203`。

注意：当前 OpenOCR `main` 的 config 设置 `Decoder.sample_k: 3`，但 `MDiffLabelEncode.train_all_layer` 默认是 `False`。如果没有额外传 `train_all_layer=True`，label encoder 不会输出 K 组样本。这看起来是当前官方 main 分支里的 config/code 形状风险；V2-M02h 不应盲目复制 `sample_k=3`，应先做 shape smoke test。

### 4.13 token replacement noise 对应代码

token replacement noise 在官方代码中主要对应：

- `MDiffLabelEncode.reflect_random_idices()`
- `MDiffDecoder.forward_train()` / `forward_train_all()` 中的 `reflect_ids` 分支

`reflect_random_idices()` 会以随机概率把 token 替换成随机 vocabulary id，构造非 mask 类型噪声。

证据：`mdiff_label_encode.py:254-262`，`mdiff_decoder.py:133-146`。

### 4.14 all mask strategies 对应代码

官方多种 mask/noising strategy 主要在：

- `/mnt/data/zyx/OpenOCR_ref/openrec/preprocess/mdiff_label_encode.py`

包括：

- full mask
- random mask
- left-to-right mask
- right-to-left mask
- block mask
- cloze mask
- fallback random mask

默认 `mask_tpye=[0, 1, 2, 3, 4, 5]`。

证据：`mdiff_label_encode.py:14-40`，`mdiff_label_encode.py:41-75`，`mdiff_label_encode.py:197-252`。

### 4.15 low-confidence / block low-confidence decoding

官方 inference 里：

- low-confidence decoding: `forward_low_confidence_decoding`
- block-style low-confidence decoding: 最接近 `forward_semi_autoregressive_decoding`

LC 会只把高于平均置信度的 token 写回，低置信度 token 继续保持 mask。

BLC-like 逻辑会先构造 block 有效区域，再在当前 block 内按置信度选择写回。

证据：`mdiff_decoder.py:296-343`，`mdiff_decoder.py:394-464`。

### 4.16 `sampler_step`、`sample_k`、`max_len`

- `sampler_step`: iterative denoising/remask 的迭代步数，用于 LC/random/semi-AR 等 decoding path。
- `sample_k`: 训练时每个样本的 noised variants 数量；`sample_k > 0` 会让训练进入 `forward_train_all`。
- `max_len`: 最大文本长度，不含额外 EOS slot；decoder 实际长度为 `max_len + 1`。

证据：`mdiff_decoder.py:53-66`，`mdiff_decoder.py:217-236`，`mdiff_decoder.py:266-271`。

### 4.17 inference 是否完整回灌每一步 argmax

官方主 inference path 不等价于当前 SLP 的“每一步完整 argmax 回灌”。

- parallel decoding: 从 full mask 一步输出，不迭代回灌。
- AR decoding: 每次只写回当前位置。
- LC/random/BLC: 只选择部分位置写回，低置信度或 EOS 后位置继续保留 mask。

当前 SLP Plain decoder 的推理是每一步把所有位置 argmax 全部写回，这与官方策略差异很大。

证据：`mdiff_decoder.py:264-272`，`mdiff_decoder.py:296-343`，`mdiff_decoder.py:345-392`，`mdiff_decoder.py:394-485`。

### 4.18 EOS / PAD / ignore index 处理

官方特殊 id：

- EOS: `0`
- MASK: `out_channels - 2`
- PAD / ignore: `out_channels - 1`

官方 classifier 不输出 MASK/PAD。

官方 iterative inference 用 `get_masked_indice_after_eos()` 找到第一处 EOS 后的位置，并把 EOS 后的位置视为需要 mask 的位置。

证据：`mdiff_decoder.py:49-52`，`mdiff_decoder.py:95-104`，`mdiff_decoder.py:274-294`。

### 4.19 是否存在防止所有位置同质化的机制

官方实现并没有一个单独叫“anti-homogeneous”的模块，但有多处机制降低所有位置同质化风险：

- fixed sinusoidal PE + learned PE 的双位置编码。
- 多种 noising strategy，不只 random/full。
- denoising loss 按 `p_mask` 和 `length + 1` 归一化。
- token replacement / reflect branch 训练模型修正非 mask 错误 token。
- LC/BLC inference 不会把所有位置的 argmax 无条件完整回灌。
- EOS 后位置有专门 mask 处理逻辑。

这些机制都比当前 SLP Plain decoder 更接近论文里的 MDiff4STR。

### 4.20 与论文 Fig.3 / Fig.4 / Table 2 的对应关系

- Fig.3(a) denoising training：对应 `MDiffLabelEncode.forward_process()` + `MDiffDecoder.forward_train()` 的 denoising loss。
- Fig.3(c) token-replacement correction training：对应 `reflect_random_idices()` + `reflect_loss` 分支。
- Fig.3(d) correction-capable inference：代码中有 `forward_reflect()`，且 decoder 在训练中同时见过 mask noise 与 replacement noise。
- Fig.4(a) random mask：对应 `random_mask()`。
- Fig.4(b) full mask：对应 `full_mask()` 和 `forward_parallel_decoding()`。
- Fig.4(c)/(d) right/left remask：对应 `left_to_right_mask()` / `right_to_left_mask()`；inference 中有 AR path，但没有完整对称的 reverse AR eval path。
- Fig.4(e) cloze remask：对应 `forward_cloze_mask_decoding()`。
- Fig.4(f) LC：对应 `forward_low_confidence_decoding()`。
- Fig.4(g) BLC：最接近 `forward_semi_autoregressive_decoding()`。
- Table 2 中 `R`、`R+All`、TRN 的代码映射：
  - `R`: random mask
  - `R+All`: `mask_tpye=[0,1,2,3,4,5]`
  - TRN: `reflect_random_idices()` + `reflect_loss`

## 5. Official MDiff4STR Config Reading

### 5.1 encoder

base/small 都使用：

- `Encoder.name: SVTRv2LNConvTwo33`

base:

- `dims: [128, 256, 384]`
- `depths: [6, 6, 6]`
- `num_heads: [4, 8, 12]`
- `feat2d: False`
- `last_stage: false`

small:

- `dims: [128, 256, 384]`
- `depths: [3, 6, 3]`
- `num_heads: [4, 8, 12]`
- `feat2d: False`
- `last_stage: false`

### 5.2 decoder hidden dim

decoder 的 `d_model` 来自 encoder 输出通道。官方这两个配置中 encoder 最后一层 dim 是 `384`，因此 decoder hidden dim 实际为 `384`。

### 5.3 decoder layers

- base: `num_decoder_layers: 6`
- small: `num_decoder_layers: 3`

### 5.4 attention heads

- base/small 都是 `nhead: 6`

### 5.5 max_text_length

- base/small 都是 `max_text_length: 25`
- decoder 实际 sequence length 是 `max_len + 1`，即额外包含 EOS slot。

### 5.6 vocabulary / character_dict_path

- `character_dict_path: ./tools/utils/EN_symbol_dict.txt`
- `use_space_char: False`

`EN_symbol_dict.txt` 有 93 行字符；加上 EOS/MASK/PAD 后，MDiff label vocab 是 96 类，其中 classifier 输出 `out_channels - 2 = 94` 类，即 EOS + 93 个真实字符。

### 5.7 sampler_step

- base/small 都是 `sampler_step: 3`

### 5.8 sample_k

- base/small 都是 `sample_k: 3`

但如前所述，当前 official config 与 `MDiffLabelEncode.train_all_layer` 默认值之间存在潜在 shape 风险，不能直接搬到 SLP34K_v2。

### 5.9 loss 配置

官方配置：

- `Loss.name: MDiffLoss`

`MDiffLoss.forward()` 只是返回 `{'loss': predicts}`。真正 loss 已经由 `MDiffDecoder.forward_train()` 或 `forward_train_all()` 计算。

### 5.10 label encode / postprocess

训练：

- `MDiffLabelEncode`

eval：

- `ARLabelEncode`

postprocess：

- `ARLabelDecode`

### 5.11 train dataset / eval dataset

train dataset 在 config 的 `Train.dataset`：

- `RatioDataSetTVResize`
- Union14M-L-LMDB-Filtered 路径
- `DecodeImagePIL`
- `PARSeqAugPIL`
- `MDiffLabelEncode`
- `KeepKeys`

eval dataset 在 config 的 `Eval.dataset`：

- `RatioDataSetTVResize`
- benchmark LMDB 路径
- `DecodeImagePIL`
- `ARLabelEncode`
- `KeepKeys`

### 5.12 不能直接搬到 SLP34K_v2 的配置

- `SVTRv2LNConvTwo33` encoder 与对应 checkpoint 假设。
- Union14M / benchmark 数据路径。
- OpenOCR preprocess/postprocess 类名。
- English symbol dictionary。
- `max_text_length: 25`，因为当前 SLP 配置是 `max_label_length: 50`。
- 官方 optimizer/batch size/scheduler 配置。
- `sample_k: 3`，在未确认 `train_all_layer` 与 batch shape 前不能直接用。

### 5.13 可以迁移为 SLP34K_v2 默认值或参考值的配置

- `sampler_step: 3` 可作为 `denoise_steps` / official sampler 参考。
- decoder depth/head 比例可参考，但要按 SLP `embed_dim` 适配。
- classifier 输出不包含 MASK/PAD。
- `max_len + 1` 包含 EOS slot。
- EOS 在输出类中，MASK/PAD 只作为输入/ignore special token。
- self-attention 不用 causal mask。

## 6. Current PlainMDiffDecoder Reading

当前实现文件：

- `/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff/modules.py`
- `/mnt/data/zyx/SLP34K_v2/ocr_training/strhub/models/slp_mdiff/system.py`
- `/mnt/data/zyx/SLP34K_v2/ocr_training/configs/model/slp_mdiff.yaml`

当前实现特点：

- 使用 SLP34K_v2 的 MAE ViT encoder，而不是 OpenOCR SVTRv2。
- `PlainMDiffDecoder` 使用 PyTorch `nn.MultiheadAttention`。
- 只有 learned positional embedding，没有 fixed sinusoidal PE。
- decoder layer 是 pre-norm 风格，而官方 block 更接近 post-residual norm。
- 训练 noising 在 `system.py` 内部生成，只支持 `random`、`full`、`random_or_full`。
- 没有官方 `MDiffLabelEncode` 的 all-mask strategy。
- 没有 `reflect_ids` / token replacement noise 分支。
- 没有 `forward_train_all` / `sample_k` 分支。
- inference 从 full mask 开始，但每一步把所有位置 `argmax` 结果完整回灌。
- output head 输出 `len(tokenizer)-2` 类，排除 BOS/PAD/MASK，只输出 EOS + 真实字符；这一点与官方 classifier 思路基本一致。

## 7. Side-by-side Difference Table

| Aspect | OpenOCR MDiffDecoder | SLP34K_v2 PlainMDiffDecoder | Difference | Collapse Risk |
|---|---|---|---|---|
| encoder feature shape | SVTRv2LNConvTwo33，输出序列特征，官方配置最终 dim=384 | MAE ViT，输出 `[B, patches+CLS, embed_dim]`，默认 768 | encoder 类型、token 数、CLS token、预训练域都不同 | Medium，conditioning mismatch 可能放大 collapse |
| text token embedding | OpenOCR `Embeddings(out_channels, d_model, padding_idx=0)` | 自定义 `TokenEmbedding(input_vocab_size, embed_dim, padding_idx=pad_id)` | scaling 类似，但 special-id 和 padding index 不同 | Low-Medium |
| position embedding | fixed sinusoidal PE + learned `pos_embed` | 只有 learned `position_embed` | 官方位置信号更强 | Medium，弱位置区分可能导致同质输出 |
| decoder layer structure | 自定义 `TransformerBlock`，self-attn + cross-attn + MLP | PyTorch MHA pre-norm block + final norm | 结构差异明显 | Medium-High |
| self-attention mask | `self_mask=None`，双向无 causal mask | 传入 `token_padding_mask` mask PAD | 当前会 mask PAD 输入位置，官方没有这种路径 | Medium |
| cross-attention implementation | 自定义 attention，text query，memory key/value | PyTorch MHA，text query，memory key/value | 语义类似，但实现和 norm 顺序不同 | Medium |
| output projection | `Linear(d_model, out_channels-2)`，不输出 MASK/PAD | `Linear(embed_dim, len(tokenizer)-2)`，不输出 BOS/PAD/MASK | 基本对齐 | Low |
| mask token id | `out_channels - 2` | `len(tokenizer)` | special-id layout 不同 | Low if consistent |
| ignore index | `out_channels - 1`，即 PAD | 没有显式 ignore_index，先筛选 loss positions | loss plumbing 不同 | Medium |
| noising input | `MDiffLabelEncode` 产生多种 noising 和 reflect ids | `system.py` 内部只产生 random/full/random_or_full | 重大简化 | High |
| forward_train | decoder 内部计算 denoise CE、p_mask/length 归一化、reflect loss | system 外部对 selected positions 做 CE | 不对齐 | High |
| forward_train_all | `sample_k > 0` 时 K-sample training branch | 缺失 | 缺官方分支 | Medium |
| inference update rule | PD 一步、AR 单位置、LC/BLC 选择性写回 | 每步完整 argmax 回灌所有位置 | 重大差异 | High |
| sampler_step | 控制 LC/random/BLC 等 iterative decoding 步数 | `denoise_steps` 控制 full-argmax 迭代次数 | 数值类似，语义不同 | High |
| sample_k | 控制训练时 K 个 noised variants | 缺失 | 缺官方训练机制 | Medium |
| low-confidence remask | 有 `forward_low_confidence_decoding` | 缺失 | V2-M03 项，不应现在实现 | Medium-High |
| block remask | 有 semi-AR/block confidence 逻辑 | 缺失 | V2-M03 项，不应现在实现 | Medium-High |
| token replacement noise | `reflect_ids` + `reflect_loss` | 缺失 | 论文关键机制缺失 | High，但建议 V2-M03 再系统做 |
| EOS handling | EOS=0；iterative inference 对 EOS 后位置 remask | EOS=0；decode 截断；inference 会把 EOS 完整回灌 | 当前 EOS 可自强化为空串 | High |
| length handling | `max_len+1`；loss 除以 `length+1` 和 `p_mask` | `max_label_length+1`；无 p_mask/length 归一化 | loss scale 不同 | Medium |
| head output classes | EOS + real chars，不含 MASK/PAD | EOS + real chars，不含 BOS/PAD/MASK | 基本对齐 | Low |
| initialization | classifier 自定义 normal init 后 `_init_weights` | 项目 `init_weights`，排除 encoder | init 不同 | Low-Medium |
| config field mapping | `max_text_length=25`、hidden=384、heads=6、layers=3/6、sampler=3 | `max_label_length=50`、hidden=768、heads=12、layers=6、denoise=3 | 有适配合理性，但 sampler 语义不同 | Medium |

### 7.1 V2-M02 允许的简化

- 使用当前 SLP MAE encoder，而不是替换成 SVTRv2。
- 不做 SLP-aware TRN。
- 不做 generic token replacement noise。
- 不做 LC/BLC remask。
- 不做 segment-aware denoising。
- 不做 pinyin consistency。
- 保留 SLP `max_label_length=50`。
- 保留 output head 不输出 MASK/PAD 的设计。

### 7.2 可能导致当前 collapse 的差异

- inference 每一步完整 argmax 回灌。
- 缺少 fixed sinusoidal PE + learned PE 的官方组合。
- decoder block 与官方实现差异较大。
- 训练路径没有官方 `p_mask` / `length` normalization。
- 缺少 reflect/token replacement correction branch。
- noising strategy 过少。
- EOS 被完整回灌后容易形成自强化，最后 decode 为空字符串。

### 7.3 V2-M02h 必须修复的差异

- 移植官方 decoder core：embedding、fixed PE + learned PE、TransformerBlock、classifier convention、`max_len+1`。
- 对齐 special-id mapping：EOS / MASK / PAD / output classes。
- 改掉 full-argmax iterative feedback，至少提供官方 parallel decoding 语义作为默认安全路径。
- 对齐 `forward_train` 的输入/输出契约，保留 SLP data interface。
- 加入 shape/id assertions，先防止静默错位。

### 7.4 应留到 V2-M03 的差异

- generic token replacement noise。
- all-mask strategies 完整训练套件。
- LC/BLC remask。
- segment-aware denoising。
- pinyin consistency。

### 7.5 是否建议直接移植官方 decoder core

建议移植官方 decoder core，但不是直接复制 OpenOCR 文件。应在 SLP34K_v2 中实现一个适配版 core，最小化改变数据、训练入口和现有工程结构。

## 8. Collapse Risk Assessment

### 8.1 当前 decoder 过度简化

风险：High。

当前 Plain decoder 缺失官方 decoder core、官方 train loss path、reflect branch、all-mask noising、官方 decoding semantics。collapse 更像架构/推理语义偏差，而不是单纯 loss mode 问题。

### 8.2 position embedding 不足

风险：Medium。

官方使用 fixed PE + learned PE。当前只有 learned PE。初期所有位置输出同一字符，与 position separation 不足相容。

### 8.3 cross-attention 实现差异

风险：Medium。

当前和官方都做 text-to-visual cross-attn，但 attention 实现、norm 顺序、memory 来源不同。SLP memory 还包含 MAE CLS token，这与官方 SVTR sequence feature 不同。

### 8.4 inference update rule 差异

风险：High。

当前每步完整 argmax 回灌，一旦某个错误字符或 EOS 在早期占优，会被下一步作为输入继续强化。后期全 EOS、decode 为空字符串，与这个机制高度一致。

### 8.5 EOS / max_len 处理差异

风险：High。

官方 iterative decoding 有 EOS 后位置 remask 逻辑。当前没有类似保护，并且 EOS 仍是可输出类。全量回灌会放大 EOS bias。

### 8.6 mask token / ignore index 差异

风险：Medium。

当前 special id 内部基本自洽，但没有官方 decoder/input/output class contract 的明确约束。V2-M02h 应补齐断言。

### 8.7 noising strategy 缺失

风险：High。

论文和官方实现都强调 noising gap 与 TRN。当前 random/full 简化足以做 smoke test，但不足以说明 MDiff4STR 机制已被正确复现。

总体判断：当前 collapse 很可能来自 decoder core 与 inference update rule 没有对齐官方实现，而不是 loss mode 小修小补能解决的问题。

## 9. Recommendation

推荐顺序：

1. **V2-M02h：移植 OpenOCR MDiffDecoder core 到 SLP34K_v2**
   - 最高优先级。
   - 做适配实现，不直接复制 OpenOCR 文件。
   - 先对齐 core 和默认 decoding 语义。

2. **V2-M02g：先做 conditioning diagnostic**
   - 可作为 V2-M02h 后的验证项。
   - 不应替代 core 对齐。

3. **V2-M02f：EOS bias / length penalty**
   - 低优先级。
   - 只有当 official-core 对齐后仍 EOS collapse，再考虑。

4. **V2-M03：all-mask strategies + generic TRN**
   - 暂不进入。
   - 应等 V2-M02h 完成后再做，否则会混淆诊断。

5. **暂停并重审架构**
   - 只有当 V2-M02h 对齐后仍无法利用视觉条件时再考虑。

## 10. Appendix: Evidence Snippets

### A. 官方 decoder init 与 special ids

```text
    28	    def __init__(self,
    29	                 in_channels,
    30	                 out_channels,
    31	                 nhead=None,
    32	                 num_decoder_layers=6,
    33	                 max_len=25,
    34	                 attention_dropout_rate=0.0,
    35	                 residual_dropout_rate=0.1,
    36	                 scale_embedding=True,
    37	                 parallel_decoding=False,
    38	                 autoregressive_decoding=False,
    39	                 sampler_step=5,
    40	                 low_confidence_decoding=False,
    41	                 random_mask_decoding=False,
    42	                 semi_autoregressive_decoding=False,
    43	                 cloze_mask_decoding=False,
    44	                 rec_loss_weight=1.0,
    45	                 reflect_loss_weight=1.0,
    46	                 sample_k=0,
    47	                 temperature=1.0):
    48	        super(MDiffDecoder, self).__init__()
    49	        self.out_channels = out_channels
    50	        self.ignore_index = out_channels - 1
    51	        self.mask_token_id = out_channels - 2
    52	        self.eos = 0
```

### B. 官方 embedding、position、decoder stack

```text
    68	        self.embedding = Embeddings(
    69	            d_model=d_model,
    70	            vocab=self.out_channels,
    71	            padding_idx=0,
    72	            scale_embedding=scale_embedding,
    73	        )
    74	        self.pos_embed = nn.Parameter(torch.zeros(
    75	            [1, self.max_len + 1, d_model], dtype=torch.float32),
    76	                                      requires_grad=True)
    77	        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    78	        self.positional_encoding = PositionalEncoding(
    79	            dropout=residual_dropout_rate, dim=d_model)
    80	
    81	        self.decoder = nn.ModuleList([
    82	            TransformerBlock(
    83	                d_model,
    84	                nhead,
    85	                dim_feedforward,
    86	                attention_dropout_rate,
    87	                residual_dropout_rate,
    88	                with_self_attn=True,
    89	                with_cross_attn=True,
    90	            ) for i in range(num_decoder_layers)
    91	        ])
```

### C. 官方 classifier

```text
    95	        self.d_model = d_model
    96	        self.nhead = nhead
    97	        self.tgt_word_prj = nn.Linear(d_model,
    98	                                      self.out_channels - 2,
    99	                                      bias=False)
   100	        w0 = np.random.normal(0.0, d_model**-0.5,
   101	                              (d_model, self.out_channels - 2)).astype(
   102	                                  np.float32)
   103	        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())
   104	        self.apply(self._init_weights)
```

### D. 官方 `forward_train`

```text
   112	    def forward_train(self, memory, data=None):
   113	        labels, reflect_ids, noisy_batch, masked_indices, p_mask, length = data
   114	        p_mask = p_mask[:, None].repeat(1, labels.shape[1])
   115	        noisy_data_length = length + 1
   116	        noisy_data_length = noisy_data_length[:,
   117	                                              None].repeat(1, labels.shape[1])
   118	
   119	        tgts = self.embedding(noisy_batch)
   120	        tgts = self.positional_encoding(tgts) + self.pos_embed
   121	
   122	        for decoder_layer in self.decoder:
   123	            tgts = decoder_layer(tgts, memory, self_mask=None)
   124	        logits = self.tgt_word_prj(tgts)
   125	        token_loss = F.cross_entropy(
   126	            logits[masked_indices],
   127	            labels[masked_indices],
   128	            reduction='none',
   129	            ignore_index=self.ignore_index) / p_mask[masked_indices]
   130	        loss = torch.sum(
   131	            token_loss / noisy_data_length[masked_indices]) / labels.shape[0]
```

### E. 官方 reflect/token replacement loss

```text
   133	        if reflect_ids is not None:
   134	            reflect_tgts = self.embedding(reflect_ids)
   135	            reflect_tgts = self.positional_encoding(
   136	                reflect_tgts) + self.pos_embed
   137	            for decoder_layer in self.decoder:
   138	                reflect_tgts = decoder_layer(reflect_tgts,
   139	                                             memory,
   140	                                             self_mask=None)
   141	            reflect_logits = self.tgt_word_prj(reflect_tgts)
   142	            reflect_loss = F.cross_entropy(reflect_logits.flatten(0, 1),
   143	                                           labels.flatten(0, 1),
   144	                                           reduction='mean',
   145	                                           ignore_index=self.ignore_index)
   146	            loss = self.rec_loss_weight * loss + self.reflect_loss_weight * reflect_loss
```

### F. 官方 `forward_train_all`

```text
   150	    def forward_train_all(self, memory, data=None):
   151	
   152	        labels, reflect_ids_all, noisy_batch_all, masked_indices_all, p_mask_all, length = data
   153	        bs, L = labels.shape
   154	        tgts = self.embedding(noisy_batch_all.flatten(0, 1))
   155	        tgts = self.positional_encoding(tgts) + self.pos_embed
   156	        tgts = tgts.reshape(bs, self.sample_k, L, -1)
   157	
   158	        for decoder_layer in self.decoder:
   159	            tgts = decoder_layer(tgts,
   160	                                 memory,
   161	                                 self_mask=None,
   162	                                 sample_k=self.sample_k)
   163	        logits_all = self.tgt_word_prj(tgts)  # bs, sample_k, L, c_num
```

### G. 官方 inference dispatch

```text
   217	        if self.training:
   218	            if self.sample_k > 0:
   219	                res = self.forward_train_all(src, data)
   220	            else:
   221	                res = self.forward_train(src, data)
   222	        else:
   223	            if self.pd:
   224	                res = self.forward_parallel_decoding(src)
   225	            elif self.ar:
   226	                res = self.forward_autoregressive_decoding(src)
   227	            elif self.lc:
   228	                res = self.forward_low_confidence_decoding(src)
   229	            elif self.rm:
   230	                res = self.forward_random_mask_decoding(src)
   231	            elif self.semiar:
   232	                res = self.forward_semi_autoregressive_decoding(src)
   233	            elif self.cm:
   234	                res = self.forward_cloze_mask_decoding(src)
   235	            else:
   236	                res = self.forward_parallel_decoding(src)
```

### H. 官方 parallel decoding

```text
   264	    def forward_parallel_decoding(self, src):
   265	        bs = src.shape[0]
   266	        noisy_batch = torch.full((bs, self.max_len + 1),
   267	                                 self.mask_token_id,
   268	                                 dtype=torch.int64,
   269	                                 device=src.get_device())
   270	        tgts = self.forward_decoding(src, noisy_batch)
   271	        logits = F.softmax(self.tgt_word_prj(tgts), -1)
   272	        return logits
```

### I. 官方 EOS 后 mask helper

```text
   274	    def get_masked_indice_after_eos(self, noisy_batch):
   275	        """Get the indices of the masked tokens after the first EOS token."""
   276	        # noisy_batch: [batch_size, max_len + 1]
   277	        eos_mask = noisy_batch == self.eos  # [batch_size, seq_len]
   278	
   279	        # 找到每行第一个eos的位置
   280	        eos_indices = eos_mask.float().argmax(dim=1)  # [batch_size]
   281	
   282	        # 如果没有eos，argmax会返回0，但我们不想在这些地方mask，需要过滤
   283	        eos_exists = eos_mask.any(dim=1)  # [batch_size]
   284	
   285	        batch_size, seq_len = noisy_batch.shape
   286	        arange = torch.arange(seq_len,
   287	                              device=noisy_batch.device).unsqueeze(0).expand(
   288	                                  batch_size, -1)  # [batch_size, seq_len]
   289	
   290	        # 创建掩码：只对eos之后的token设为True
   291	        masked_indices = arange > eos_indices.unsqueeze(1)
   292	        masked_indices = masked_indices | ~eos_exists.unsqueeze(1)
   293	
   294	        return masked_indices
```

### J. 官方 low-confidence decoding

```text
   307	        for step_i in range(self.sampler_step):
   308	
   309	            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)
   310	            pred_step = self.tgt_word_prj(tgts)
   311	            pred_step = F.softmax(pred_step, -1)
   312	            if step_i == 0:
   313	                logits = pred_step.clone()
   314	            logits[masked_indices_pre] = pred_step[masked_indices_pre]
   315	            pred_step_prob, pred_step_index = torch.max(
   316	                pred_step, dim=-1)  # [bs, max_len + 1], [bs, max_len + 1]
   317	            masked_indices_eos = self.get_masked_indice_after_eos(
   318	                pred_step_index
   319	            )  # [bs, max_len + 1] bool tensor False False(eos) True True ..
   320	
   321	            # 仅计算mask token位置以及eos之前token的平均概率
   322	            valid_indices = masked_indices_pre & ~masked_indices_eos
   323	            pred_step_prob = pred_step_prob * valid_indices.float()
   324	            pred_step_prob_avg = pred_step_prob.sum(
   325	                dim=1, keepdim=True) / valid_indices.sum(
   326	                    dim=1, keepdim=True)  # [bs, 1]
   327	
   328	            # 高于平均置信度的token
   329	            top_confidence_mask = pred_step_prob > pred_step_prob_avg
   330	            top_confidence_mask = top_confidence_mask & valid_indices
   331	            noisy_batch[top_confidence_mask] = pred_step_index[
   332	                top_confidence_mask]
```

### K. 官方 block-style decoding

```text
   424	            block_vaild_indices = torch.full((bs, self.max_len + 1),
   425	                                             False,
   426	                                             dtype=torch.bool,
   427	                                             device=src.get_device())
   428	
   429	            if step_i <= 2:
   430	                if self.sampler_step > 2:
   431	                    block_vaild_indices[:, :block_size * (step_i + 1)] = True
   432	                else:
   433	                    block_vaild_indices = ~block_vaild_indices
   434	            elif step_i >= self.sampler_step - 2:
   435	                block_vaild_indices[:, block_size * (step_i - 1):] = True
   436	            else:
   437	                block_vaild_indices[:, block_size * (step_i - 1):block_size *
   438	                                    (step_i + 1)] = True
   439	
   440	            # 仅计算mask token位置, eos之前token以及当前block中token的平均概率
   441	            valid_indices = masked_indices_pre & ~masked_indices_eos & block_vaild_indices
   442	            pred_step_prob = pred_step_prob * valid_indices.float()
   443	            pred_step_prob_avg = pred_step_prob.sum(
   444	                dim=1, keepdim=True) / valid_indices.sum(
   445	                    dim=1, keepdim=True)  # [bs, 1]
   446	
   447	            # 高于平均置信度的token
   448	            top_confidence_mask = pred_step_prob > pred_step_prob_avg
   449	            top_confidence_mask = top_confidence_mask & valid_indices
   450	
   451	            noisy_batch[top_confidence_mask] = pred_step_index[
   452	                top_confidence_mask]
```

### L. 官方 TransformerBlock

```text
   563	    def forward(self,
   564	                tgt,
   565	                memory=None,
   566	                self_mask=None,
   567	                cross_mask=None,
   568	                sample_k=0):
   569	
   570	        if self.with_self_attn:
   571	            if sample_k > 0:
   572	                bs, _, L, Dim = tgt.shape
   573	                tgt = tgt.flatten(0, 1)
   574	            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
   575	            tgt = self.norm1(tgt + self.dropout1(tgt1))
   576	
   577	        if self.with_cross_attn:
   578	            if sample_k > 0:
   579	                tgt = tgt.reshape(bs, sample_k, L, Dim).flatten(1, 2)
   580	            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
   581	            tgt = self.norm2(tgt + self.dropout2(tgt2))
   582	        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
```

### M. 官方 label noising 与 special chars

```text
     7	class MDiffLabelEncode(BaseRecLabelEncode):
     8	    """Convert between text-label and text-index."""
     9	
    10	    MASK = '<mask>'
    11	    EOS = '</s>'
    12	    PAD = '<pad>'
    13	
    14	    def __init__(self,
    15	                 max_text_length,
    16	                 character_dict_path=None,
    17	                 use_space_char=False,
    18	                 semi_ar=False,
    19	                 mask_tpye=[0, 1, 2, 3, 4, 5],
    20	                 train_all_layer=False,
    21	                 sample_num=1,
    22	                 **kwargs):
```

```text
    41	    def random_mask(self, text):
    42	        l = len(text)
    43	        p_mask = random.random()
    44	
    45	        noisy_batch = text[:]
    46	        masked_indices = [False] * l
    47	        none_pad_indices = []
    48	        for i in range(l):
    49	            if random.random() < p_mask and text[i] != self.dict[self.PAD]:
    50	                noisy_batch[i] = self.dict[self.MASK]
    51	                masked_indices[i] = True
    52	            if text[i] != self.dict[self.PAD]:
    53	                none_pad_indices.append(i)
    54	            if noisy_batch[i] == self.dict[self.PAD]:
    55	                noisy_batch[i] = self.dict[self.MASK]
    56	
    57	        if not any(masked_indices) and len(none_pad_indices) > 0:
    58	            idx = random.choice(none_pad_indices)
    59	            noisy_batch[idx] = self.dict[self.MASK]
    60	            masked_indices[idx] = True
    61	        return noisy_batch, masked_indices
```

```text
   197	    def forward_process(self, text):
   198	
   199	        rand_choice = random.choice(self.mask_tpye)
   200	        if rand_choice == 0:  # 并行mask full mask
   201	            return self.full_mask(text)
   202	        elif rand_choice == 1 and len(text) > 2:  # 正向自回归 right mask
   203	            return self.left_to_right_mask(text)
   204	        elif rand_choice == 2 and len(text) > 2:  # 反向自回归 left mask
   205	            return self.right_to_left_mask(text)
   206	        elif rand_choice == 3 and len(text) > 2:  # block mask
   207	            rand_step = min(random.randint(2, 6), len(text))
   208	            if rand_step <= 1:  # len(text) <= 1
   209	                return self.full_mask(text)
   210	            block_size = len(text) // rand_step
```

```text
   254	    def reflect_random_idices(self, text, eps=1e-3):
   255	        l = len(text)
   256	        t = random.random()
   257	        p_mask = (1 - eps) * t + eps
   258	        reflect_ids = text[:]
   259	        for i in range(l):
   260	            if random.random() < p_mask:
   261	                reflect_ids[i] = random.randint(0, len(self.dict) - 1)
   262	        return reflect_ids
```

```text
   310	    def add_special_char(self, dict_character):
   311	        dict_character = [self.EOS] + dict_character + [self.MASK, self.PAD]
   312	        return dict_character
```

### N. 官方 base config

```text
    41	  Encoder:
    42	    name: SVTRv2LNConvTwo33
    43	    use_pos_embed: False
    44	    dims: [128, 256, 384]
    45	    depths: [6, 6, 6]
    46	    num_heads: [4, 8, 12]
    47	    mixer: [['Conv','Conv','Conv','Conv','Conv','Conv'],['Conv','Conv','FGlobal','Global','Global','Global'],['Global','Global','Global','Global','Global','Global']]
    48	    local_k: [[5, 5], [5, 5], [-1, -1]]
    49	    sub_k: [[1, 1], [2, 1], [-1, -1]]
    50	    last_stage: false
    51	    feat2d: False
```

```text
    52	  Decoder:
    53	    name: MDiffDecoder
    54	    num_decoder_layers: 6
    55	    nhead: 6
    56	    max_len: *max_text_length
    57	    parallel_decoding: False
    58	    autoregressive_decoding: False
    59	    low_confidence_decoding: False
    60	    random_mask_decoding: False
    61	    semi_autoregressive_decoding: False
    62	    cloze_mask_decoding: False
    63	    sampler_step: 3
    64	    sample_k: &sample_k 3
    65	    temperature: 1.0
```

### O. 官方 train/eval config

```text
    91	    transforms:
    92	      - DecodeImagePIL: # load image
    93	          img_mode: RGB
    94	      - PARSeqAugPIL:
    95	      - MDiffLabelEncode: # Class handling label
    96	          character_dict_path: *character_dict_path
    97	          use_space_char: *use_space_char
    98	          max_text_length: *max_text_length
    99	          sample_num: *sample_k
   100	      - KeepKeys:
   101	          keep_keys: ['image', 'label', 'reflect_ids', 'noisy_batch', 'masked_indices', 'p_mask', 'length'] # dataloader will return list in this order
```

```text
   130	    transforms:
   131	      - DecodeImagePIL: # load image
   132	          img_mode: RGB
   133	      - ARLabelEncode: # Class handling label
   134	          character_dict_path: *character_dict_path
   135	          use_space_char: *use_space_char
   136	          max_text_length: *max_text_length
   137	      - KeepKeys:
   138	          keep_keys: ['image', 'label', 'length'] # dataloader will return list in this order
```

### P. 当前 SLP PlainMDiffDecoder

```text
    35	class PlainMDiffDecoderLayer(nn.Module):
    36	    def __init__(
    37	        self,
    38	        embed_dim: int,
    39	        num_heads: int,
    40	        mlp_ratio: float = 4.0,
    41	        dropout: float = 0.1,
    42	        layer_norm_eps: float = 1e-5,
    43	    ) -> None:
    44	        super().__init__()
    45	        mlp_dim = int(embed_dim * mlp_ratio)
    46	        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    47	        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    48	        self.linear1 = nn.Linear(embed_dim, mlp_dim)
    49	        self.linear2 = nn.Linear(mlp_dim, embed_dim)
```

```text
    58	    def forward(self, hidden: Tensor, memory: Tensor, token_padding_mask: Optional[Tensor] = None) -> Tensor:
    59	        q = self.norm1(hidden)
    60	        hidden = hidden + self.dropout1(
    61	            self.self_attn(q, q, q, key_padding_mask=token_padding_mask, need_weights=False)[0]
    62	        )
    63	
    64	        hidden = hidden + self.dropout2(
    65	            self.cross_attn(self.norm2(hidden), memory, memory, need_weights=False)[0]
    66	        )
```

```text
    86	        self.max_length = max_length
    87	        self.text_embed = TokenEmbedding(input_vocab_size, embed_dim, padding_idx=padding_idx)
    88	        self.position_embed = nn.Parameter(torch.empty(1, max_length, embed_dim))
    89	        self.dropout = nn.Dropout(dropout)
    90	        self.layers = nn.ModuleList(
    91	            [
    92	                PlainMDiffDecoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
    93	                for _ in range(depth)
    94	            ]
    95	        )
    96	        self.norm = nn.LayerNorm(embed_dim)
    97	        nn.init.trunc_normal_(self.position_embed, std=0.02)
```

### Q. 当前 SLP special ids、noising、inference

```text
    58	        self.max_label_length = max_label_length
    59	        self.embed_dim = embed_dim
    60	        self.denoise_steps = denoise_steps
    61	        self.mask_ratio = mask_ratio
    62	        self.mask_strategy = mask_strategy
    63	        self.loss_mode = loss_mode
    64	        self.freeze_encoder = freeze_encoder
    65	        self.mask_id = len(self.tokenizer)
    66	        self.input_vocab_size = len(self.tokenizer) + 1
    67	        self.output_num_classes = len(self.tokenizer) - 2
```

```text
   156	    def _make_noised_inputs(self, clean_targets: Tensor, return_full_mask_rows: bool = False):
   157	        if (clean_targets == self.mask_id).any():
   158	            raise ValueError("Clean targets unexpectedly contain mask_id")
   159	
   160	        device = clean_targets.device
   161	        valid = clean_targets != self.pad_id
   162	        full_mask_rows = self._select_mask_strategy(clean_targets.shape[0], device)
   163	        random_mask = (torch.rand(clean_targets.shape, device=device) < self.mask_ratio) & valid
   164	        masked_positions = torch.where(full_mask_rows[:, None], valid, random_mask)
   165	
   166	        for row_idx in range(masked_positions.shape[0]):
   167	            if valid[row_idx].any() and not masked_positions[row_idx].any():
   168	                valid_indices = valid[row_idx].nonzero(as_tuple=False).flatten()
   169	                choice = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]
   170	                masked_positions[row_idx, choice] = True
   171	
   172	        noised = clean_targets.clone()
   173	        noised[masked_positions] = self.mask_id
```

```text
   215	    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
   216	        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
   217	        num_steps = max_length + 1
   218	        batch_size = images.shape[0]
   219	        memory = self.encode(images)
   220	        input_ids = torch.full(
   221	            (batch_size, num_steps),
   222	            self.mask_id,
   223	            dtype=torch.long,
   224	            device=images.device,
   225	        )
   226	
   227	        logits = None
   228	        for _ in range(max(1, self.denoise_steps)):
   229	            hidden = self.decode(input_ids, memory)
   230	            logits = self.head(hidden)
   231	            input_ids = logits.argmax(dim=-1)
   232	        return logits
```
