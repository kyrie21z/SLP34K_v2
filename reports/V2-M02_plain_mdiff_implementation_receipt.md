# V2-M02 Plain MDiff 实现回执

## 1. Summary

已为 SLP34K_v2 实现 Plain MDiff Decoder Replacement 的最小可运行链路：

```text
image -> original MAE visual encoder -> identity visual adapter -> Plain MDiff decoder -> Linear head -> logits
```

实现保留原 SLP MAE encoder 初始化路径，用最小 Transformer-style denoising decoder 替换 PARSeq/AR decoder，并保持 baseline 输出类别约定：`568 chars + EOS = 569` logits。本阶段未启动正式训练，也未运行长时间 evaluation。

## 2. Implemented Files

- `ocr_training/strhub/models/slp_mdiff/__init__.py`
  - 暴露 `Model`。
- `ocr_training/strhub/models/slp_mdiff/modules.py`
  - 新增 `TokenEmbedding`、`VisualAdapter`、`PlainMDiffDecoderLayer` 和 `PlainMDiffDecoder`。
- `ocr_training/strhub/models/slp_mdiff/system.py`
  - 新增 Lightning 模型封装、encoder 加载、noising、loss、full-mask iterative inference，以及可用于 synthetic 验证的 `training_step`。
- `ocr_training/configs/model/slp_mdiff.yaml`
  - 新增 `model=slp_mdiff` 的 Hydra 配置。
- `ocr_training/strhub/models/utils.py`
  - 仅为 `_get_model_class()` 增加 `slp_mdiff` 路径识别分支，原 `maevit_infonce_plm` 和 `maevit_plm` 分支保持不变。

## 3. Architecture

- Encoder：对 `img_size=[224, 224]`、`embed_dim=768` 复用 `strhub.models.models_mae.mae_vit_base_patch16_224x224()`，并用 `strict=False` 加载 `mae_pretrained_path`。
- Baseline checkpoint migration：已实现可选配置 `init_encoder_from_baseline_ckpt=true`；该路径会从 Lightning checkpoint 中提取 `encoder.*` 权重并加载到 MAE encoder。默认值保持 `false`。
- Visual adapter：默认使用 `identity`。代码中已保留 `linear_ln` hook，后续可扩展为 `Linear + LayerNorm` adapter。
- MDiff decoder：由 token embedding、learned position embedding 和 `N` 个 decoder block 组成。每个 block 包含 token states 上的双向 self-attention、到 visual memory 的 cross-attention、FFN、residual 和 LayerNorm。
- Text embedding：输入 vocab size 为 `len(tokenizer) + 1`，额外 id 仅作为 input-only `[MASK]`。
- Head：`nn.Linear(768, len(tokenizer) - 2)`，随机初始化。
- Output classes：`569`，对应 `charset + EOS`。
- `forward_train`：`training_step()` 编码 label，移除 BOS，执行 random/full mask noising，decode 后在选定 denoising 位置计算 CE loss。
- `forward_infer`：`forward()` 从 `[MASK] * (max_label_length + 1)` 开始，迭代 `denoise_steps` 后返回 logits。最终字符串截断依赖现有 tokenizer 的 EOS decode 规则。

## 4. Token / Label Design

- `len(tokenizer)=571`：`568 chars + EOS + BOS + PAD`。
- `EOS=0`，字符 id 为 `1..568`，`BOS=569`，`PAD=570`。
- `mask_id=571`。
- Decoder input vocab size：`572`。
- Head output dim：`569`，只覆盖 `EOS + chars`。
- Clean targets 使用 `tokenizer.encode(labels)[:, 1:]`，因此训练 loss 前会丢弃 BOS。
- Clean targets 不应包含 `mask_id`；代码中已加入 guard。
- PAD 不参与 noising loss。
- BOS 不会作为 head 输出类别。
- MASK 只用于 decoder 输入，不会由 head 输出。
- Loss positions：`(masked_positions OR eos_position) AND non-PAD`。这样即使 EOS 没有被 random masking 采样到，也会保持 EOS 监督。

## 5. Verification Results

环境：

- 使用 conda 环境：`slpr_ocr`。
- 通过 `CUDA_VISIBLE_DEVICES=''` 禁用 CUDA。
- 未使用 GPU。

检查结果：

- `python -m py_compile strhub/models/slp_mdiff/system.py strhub/models/slp_mdiff/modules.py`
  - 通过。
- `python -m py_compile strhub/models/utils.py strhub/models/slp_mdiff/__init__.py`
  - 通过。
- Hydra instantiate：
  - 使用 `model=slp_mdiff`、`trainer.gpus=0`、`model.denoise_steps=1`。
  - 通过。
  - `tokenizer_len=571 mask_id=571 input_vocab=572 head_out=569`。
  - MAE load info：`missing=0`，`unexpected=104`。这是 MAE pretrain checkpoint 中存在额外 MAE-side keys 导致的预期现象。
- Dummy CPU forward：
  - 输入：`[1, 3, 224, 224]`。
  - 输出 logits：`(1, 51, 569)`。
- Synthetic `training_step`：
  - Synthetic label：tokenizer 前三个字符，`睢荷焦`。
  - Loss：`6.254663`。
  - Finite：`True`。
- Validation-style `forward_logits_loss`：
  - Logits：`(1, 4, 569)`。
  - Loss：`6.853469`。
  - Loss numel：`4`。
  - Finite：`True`。
- GPU memory：
  - 未使用 GPU；peak CUDA memory 不适用。
  - 未接近 40GB 显存上限。
- `test.py` loading：
  - 未运行完整 `test.py`，原因是当前没有真实或小型 `slp_mdiff` checkpoint，保存完整 MAE+decoder checkpoint 会为本最小阶段产生较大的额外 artifact。
  - 已检查 class dispatch：`_get_model_class('outputs/slp_mdiff/checkpoints/last.ckpt')` 可解析到 `strhub.models.slp_mdiff.system.Model`。

## 6. Compatibility

- V2-M02 未有意改变原 `maevit_infonce_plm` 行为。
- V2-M02 未有意改变原 `maevit_plm` 行为。
- `utils.py` 只增加了 `slp_mdiff` checkpoint path 识别。
- 原 `maevit_infonce_plm` 和 `maevit_plm` checkpoint loading 分支仍保留。
- 未修改 `configs/main.yaml` 默认模型。
- 未修改数据文件。

## 7. Known Risks

- Mask token index risk：当前 `mask_id=len(tokenizer)` 对现有 tokenizer 是安全的，但未来若 tokenizer 重排，必须继续保持 head/output mapping 兼容。
- EOS / length handling risk：iterative decoding 返回固定长度 logits，并依赖 tokenizer EOS 截断；当前未实现基于置信度的 early stopping。
- Encoder checkpoint migration risk：baseline ckpt 提取逻辑支持 `encoder.*`，但本阶段尚未用真实 baseline Lightning checkpoint 验证。
- Noising strategy too simple risk：当前只实现 random mask/full mask；尚未实现 LC/BLC、segment-aware denoising、generic TRN 或 SLP-aware TRN。
- Evaluation artifact insufficient risk：当前只完成 dummy CPU forward 和 synthetic loss 检查，还没有真实数据 accuracy 信号。
- 由于任务约束和 GPU 显存不可用，本阶段尚未进行真实训练。

## 8. Next Step Recommendation

- V2-M02b：GPU 可用后进行 small-batch real-data overfit test。
- V2-M03：移植 all-mask strategies 和 generic TRN。
- V2-M04：实现 SLP-aware TRN。
