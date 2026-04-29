# V2-M01b Paper-Code Mapping Receipt

## 1. Executive Summary

- 当前 `ocr_training` 代码完整实现了论文 OCR baseline 的 Stage 2 recognition path：`SLP image -> MAE/ViT visual encoder -> PARSeq/XLNet-style two-stream decoder -> linear head -> tokenizer decode`。主类是 `ocr_training/strhub/models/maevit_infonce_plm/system.py:184` 的 `Model`。
- MAE Stage 1 预训练代码存在，但和 OCR 训练入口分离：入口在 `mae/main_pretrain.py:29` / `mae/main_pretrain.py:154`，重建 decoder 与 MSE loss 在 `mae/models_mae.py:58`、`mae/models_mae.py:235`。OCR 训练代码只加载预训练好的 encoder checkpoint：`ocr_training/strhub/models/maevit_infonce_plm/system.py:213`。
- Semantic enhancement 已实现：训练时使用 CLIP text encoder 与 MAE CLS visual feature 做 InfoNCE，`total_loss = ocr_loss + 0.1 * cross_modal_loss` 在 `ocr_training/strhub/models/maevit_infonce_plm/system.py:471`。推理 `forward()` 调用 `encode(images)`，不传 labels，因此不会计算 CLIP text feature。
- 与论文描述的主要差异/简化：CLIP 在推理模型构造时仍会被实例化并保留 text tower，虽然不参与 forward 计算；评估脚本不读取 LMDB `meta-*` 字段，不能直接复现论文按 quality/structure 等 metadata 细分的 grouped accuracy。
- 对后续 MDiff Decoder 替换最重要的结论：应保留 MAE encoder、charset/tokenizer、`head` 输出类别约定、image/data config；替换点在 `encode(images)` 之后、`head` 之前，即替换当前 `decode()`/AR loop/PLM permutation decoder。必须保持输出 logits shape `[B, max_label_length + 1, 569]`，其中 569 = SLP34K 568 字符 + EOS。

## 2. Paper Module to Code Mapping Table

| Paper Module | Paper Description | Code File | Class / Function / Config | Evidence | Training or Inference | Notes |
|---|---|---|---|---|---|---|
| SLP image input | SLP image as OCR input | `ocr_training/configs/main.yaml`; `ocr_training/strhub/data/module.py` | `model.img_size: [224,224]`; `SceneTextDataModule.get_transform()` | `main.yaml:10`; `module.py:57-70` | Both | 图像 resize/to tensor/normalize 后进入 model。 |
| Overall encoder-decoder | image -> encoder -> decoder -> text | `ocr_training/strhub/models/maevit_infonce_plm/system.py` | `Model`, `forward()`, `encode()`, `decode()` | `system.py:184`; `system.py:265`; `system.py:280`; `system.py:294` | Both | 完整实现 OCR baseline Stage 2 path。 |
| ViT / MAE visual encoder | Strong visual encoder | `ocr_training/strhub/models/models_mae.py`; `system.py` | `MaskedAutoencoderViT_Encoder`; `mae_vit_base_patch16_224x224` | `models_mae.py:22-65`; `models_mae.py:91-105`; `system.py:204-217` | Both | 224x224 使用 patch16 ViT-base encoder。 |
| MAE pretraining | mask patches -> reconstruction | `mae/main_pretrain.py`; `mae/models_mae.py` | `main()`; `MaskedAutoencoderViT` | `main_pretrain.py:29-63`; `models_mae.py:14-30`; `README.md:143-158` | Stage 1 training only | 当前 repo 有独立 Stage 1 入口，但不在 `ocr_training/train.py`。 |
| reconstruction decoder | MAE decoder reconstructs pixels | `mae/models_mae.py` | `decoder_embed`, `mask_token`, `decoder_blocks`, `decoder_pred`, `forward_decoder()` | `models_mae.py:58-80`; `models_mae.py:204-233` | Stage 1 training only | OCR `ocr_training/strhub/models/models_mae.py` 是 encoder-only，不含 reconstruction decoder。 |
| semantic enhancement | visual/text semantic consistency | `system.py` | `encode(img, labels)` | `system.py:269-275`; `system.py:439-443` | Training | 只在 `training_step` 传 labels 时启用。 |
| CLIP text encoder | Text feature T from CLIP | `system.py`; `clip/model.py` | `clip.load("ViT-B/16")`; `encode_text()` | `system.py:219`; `clip/model.py:405-423` | Training branch | 删除 visual tower：`system.py:221`。 |
| CLIP visual tower | Not used for enhancement | `system.py` | `del self.clip_model.visual` | `system.py:221` | Neither | 符合只用 CLIP text encoder；但 CLIP text tower仍在模型内存中。 |
| visual-text projector | visual CLS -> CLIP 512D | `system.py` | `self.proj = nn.Parameter(... embed_dim, 512)` | `system.py:237-238` | Training | 使用 `mae_memory[:,0,:] @ self.proj`。 |
| InfoNCE loss | contrastive visual-text loss | `system.py` | `InfoNCELoss`; `info_nce()` | `system.py:64-119`; `system.py:169-174`; `system.py:273` | Training | 默认 temperature=0.1；batch 内 off-diagonal negatives。 |
| CE recognition loss | recognition CE | `system.py`; `base.py` | `F.cross_entropy(...)` | `system.py:459`; `system.py:468`; `base.py:178-185` | Training/validation | 训练 loss 是 PLM permutations 上的 CE；validation 使用 `forward_logits_loss()` CE。 |
| PARSeq / AR decoder | auto-regressive/two-stream decoder | `modules.py`; `system.py` | `DecoderLayer`, `Decoder`, `decode()` | `modules.py:95-168`; `system.py:280-291` | Both | `DecoderLayer` docstring 明确 two-stream XLNet。 |
| token embedding | token ids to hidden vectors | `system.py`; `modules.py` | `self.text_embed`; `TokenEmbedding` | `system.py:254`; `modules.py:234-242` | Both | embedding vocab 包含 charset + EOS/BOS/PAD。 |
| position query | decoder query positions | `system.py` | `self.pos_queries` | `system.py:257`; `system.py:303-304` | Both | 长度 `max_label_length + 1`，含 EOS 位置。 |
| permutation mask | permutation LM masks | `system.py` | `gen_tgt_perms()`; `generate_attn_masks()` | `system.py:365-437`; `system.py:446-457` | Training | 训练使用 PARSeq-style permutation language modeling。 |
| refinement iteration | iterative cloze refinement | `system.py`; config | `self.refine_iters`; forward refinement block | `system.py:201-202`; `system.py:345-360`; `maevit_infonce_plm.yaml:20-22` | Inference | 默认 `refine_iters: 1`。 |
| classifier head | hidden -> char logits | `system.py` | `self.head = nn.Linear(embed_dim, len(tokenizer)-2)` | `system.py:252-254` | Both | SLP34K 下输出 569 类：EOS + 568 charset，不含 BOS/PAD。 |
| tokenizer / charset | label encode/decode | `ocr_training/strhub/data/utils.py`; charset yaml | `Tokenizer`; `SLP34K_568.yaml` | `utils.py:102-127`; `SLP34K_568.yaml:3-4` | Both | `slpr_ocr` 验证 charset length = 568, unique = 568。 |
| max_label_length | output length cap | `configs/main.yaml`; `data/dataset.py` | `model.max_label_length`; label filter | `main.yaml:10-11`; `dataset.py:99-101` | Both | SLP 默认 50；推理 `num_steps=51`。 |
| training entry | Stage 2 OCR training | `ocr_training/train.py`; configs | Hydra `main()`; instantiate model/data | `train.py:58-108`; `main.yaml:1-5` | Training | 支持 Hydra CLI override；README 给出 baseline 命令。 |
| test / evaluation entry | checkpoint evaluation | `ocr_training/test.py`; `base.py` | `main()`; `test_step()` | `test.py:64-145`; `base.py:100-131` | Inference/eval | 输出 Accuracy、1-NED、Confidence、Label Length。 |
| dataset config | LMDB train/val/test dirs | `configs/dataset/SLP34K.yaml`; `data/module.py` | `train_dir`, `val_dir`, `test_dir` | `SLP34K.yaml:1-5`; `module.py:73-109` | Both | 路径为 `data/train`, `data/val`, `data/test` 下对应 LMDB。 |
| checkpoint loading | MAE and full OCR checkpoint | `system.py`; `models/utils.py`; `train.py` | `torch.load(mae_pretrained_path)`; `load_from_checkpoint()`; `ckpt_path` | `system.py:213-217`; `utils.py:77-84`; `train.py:97-108`; `main.yaml:43-44` | Both | MAE checkpoint strict=False 加载 encoder；Lightning checkpoint 保存/恢复全模型。 |

## 3. Forward Path Walkthrough

1. `batch -> image tensor`
   - DataModule: `ocr_training/strhub/data/module.py:57-70` 构造 transform：resize to `img_size`, `ToTensor`, `Normalize`。
   - Config: `ocr_training/configs/main.yaml:10` 默认 `img_size: [224,224]`。
   - Tensor shape: `[B, 3, 224, 224]`。

2. `image tensor -> encode -> memory`
   - Call path: `Model.forward(images)` at `system.py:294` calls `memory = self.encode(images)` at `system.py:301`。
   - `encode()` at `system.py:265-277` calls `self.encoder(img)`。
   - Encoder class: `MaskedAutoencoderViT_Encoder.forward()` at `ocr_training/strhub/models/models_mae.py:63-65` returns all tokens.
   - 224x224 shape: patch size 16 (`models_mae.py:91-99`), so 14 x 14 = 196 patches; CLS token appended at `models_mae.py:51-55`; output memory shape `[B, 197, 768]`。
   - This is the code counterpart of the paper Figure 3 ViT Encoder: encoder-only MAE/ViT tokens are passed directly as decoder memory.

3. `memory -> decoder hidden`
   - Inference AR loop: `system.py:309-338`。
   - Token ids `tgt_in` start as `[B, num_steps]`, with BOS at position 0 (`system.py:310-312`)。
   - `decode(tgt_in[:, :j], memory, ...)` at `system.py:320-326` uses token embedding and position query (`system.py:283-291`) and calls `self.decoder`。
   - `decode()` returns hidden states, not logits. Shape per AR step is `[B, 1, 768]`; after refinement/non-AR full decode it is `[B, T, 768]`。

4. `decoder hidden -> head -> logits`
   - Head definition: `self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)` at `system.py:253`。
   - AR loop logits: `p_i = self.head(tgt_out)` at `system.py:328`; concatenated logits shape `[B, <=51, 569]` for SLP34K.
   - Refinement logits: `logits = self.head(tgt_out)` at `system.py:360`。

5. `logits -> decode string`
   - Evaluation path: `BaseSystem._eval_step()` softmaxes logits at `base.py:120` and calls `self.tokenizer.decode(probs)` at `base.py:121`。
   - Tokenizer greedy decodes max ids at `utils.py:92-99`, truncates before EOS at `utils.py:118-127`, then `CharsetAdapter` adapts prediction before exact-match/NED at `base.py:123-128`。
   - This path is shared by validation/test; training manually computes CE in `Model.training_step()` and does not decode strings.

## 4. Training Path Walkthrough

Training call path starts at `Model.training_step()` in `ocr_training/strhub/models/maevit_infonce_plm/system.py:439`:

```text
batch(images, labels)
  -> tokenizer.encode(labels)
  -> encode(images, labels)
  -> tgt_in / tgt_out
  -> gen_tgt_perms + generate_attn_masks
  -> decode(tgt_in, memory, masks)
  -> head(out)
  -> CE loss
  -> InfoNCE loss
  -> ocr_loss + 0.1 * cross_modal_loss
```

- Label encoding: `tgt = self.tokenizer.encode(labels, self._device)` at `system.py:441`。`Tokenizer.encode()` prepends BOS and appends EOS at `utils.py:113-116`。
- Text/visual semantic branch: `memory, cross_modal_loss = self.encode(images, labels)` at `system.py:443`。Inside `encode()`, labels are CLIP-tokenized (`system.py:270`), CLIP text features are computed (`system.py:271`), and InfoNCE is `self.InfoNCELoss(mae_memory[:,0,:] @ self.proj, text_features)` (`system.py:273`)。
- Target split: `tgt_in = tgt[:, :-1]`, `tgt_out = tgt[:, 1:]` at `system.py:447-448`。This makes the model predict every next token including EOS, excluding BOS from targets。
- Permutation LM: `gen_tgt_perms()` and `generate_attn_masks()` are called at `system.py:446` and `system.py:456`。This corresponds to PARSeq permutation language modeling, not diffusion/noising。
- CE loss: `F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)` at `system.py:459` and again at `system.py:468`。This is the paper recognition CE term `Lce`。
- InfoNCE loss: `InfoNCELoss` computes cross entropy over cosine-similarity logits at `system.py:169-174`。This is the paper semantic enhancement term `LinfoNCE`。
- Total loss: `loss = ocr_loss + 0.1 * cross_modal_loss` at `system.py:471`。The alpha weight is hard-coded as `0.1` in code, not in YAML。
- Validation: `BaseSystem.validation_step()` calls `_eval_step(..., True)` at `base.py:154-155`, which calls `forward_logits_loss()` at `base.py:108-110`。For `CrossEntropySystem`, that is CE-only (`base.py:178-185`); validation does not compute InfoNCE。
- Gradients:
  - MAE encoder is trainable unless individual params are frozen elsewhere; optimizer includes encoder params at `system.py:490-504`。
  - Decoder/head/text embedding/projector are included in the non-encoder/non-CLIP optimizer groups at `system.py:495-507`。
  - CLIP text modules are explicitly set `requires_grad=False` for token embedding, transformer, and LN at `system.py:223-235`; all `clip_model` params are also excluded from optimizer by name at `system.py:496`。
  - `self.proj` is not part of `clip_model`, so it is trainable and receives InfoNCE gradients。

## 5. Inference Path Walkthrough

Inference uses `Model.forward(images, max_length=None)`:

```text
image
  -> encode(images)
  -> AR loop from BOS
  -> head at each step
  -> greedy argmax token
  -> stop when all samples have EOS or max steps reached
  -> optional refinement
  -> tokenizer.decode()
```

- CLIP usage: inference calls `self.encode(images)` at `system.py:301` without labels, so the CLIP branch at `system.py:269-275` is skipped。
- Semantic enhancement usage: no InfoNCE, no CLIP text features, no label input during inference。
- AR decode: `self.decode_ar` default is true from `maevit_infonce_plm.yaml:20-22`。The loop initializes BOS at `system.py:310-312`, predicts one position at a time at `system.py:314-329`, and writes argmax token back into `tgt_in` at `system.py:330-332`。
- EOS stop: when every batch item has at least one EOS, the test-time loop breaks at `system.py:333-336`。
- Length control: default `max_label_length=50` at `main.yaml:11`; forward sets `num_steps=max_length+1` for EOS at `system.py:296-299`。
- Refinement: default `refine_iters: 1` at `maevit_infonce_plm.yaml:21-22`。If enabled, code builds a cloze-like mask and reruns decoder at `system.py:345-360`。
- Prediction string: `BaseSystem._eval_step()` applies softmax and tokenizer decode at `base.py:120-121`。Tokenizer removes EOS from returned text but keeps EOS probability in confidence at `utils.py:118-127`。
- Paper claim check: code matches "semantic enhancement only used during training" computationally. Nuance: `self.clip_model` is still instantiated and stored in the model at `system.py:219-235`, so there is no inference forward compute overhead, but there is model construction/checkpoint memory overhead unless deployment strips that branch.

## 6. Stage 1 MAE Pretraining Status

- 当前 repo 能直接执行论文 Stage 1 MAE pretraining，入口是 `mae/main_pretrain.py`。README 给出 SLP34K 命令：`cd ./mae` 后运行 `main_pretrain.py --data_path pretrain_data/SLP34K_lmdb_train --mask_ratio 0.75 --model mae_vit_base_patch16_224x224 ...`，见 `README.md:143-158`。
- Stage 1 代码不在 `ocr_training/train.py` 中。OCR 训练入口 `ocr_training/train.py:58-108` 只做 recognition fine-tuning。
- Reconstruction decoder 存在于 `mae/models_mae.py`：decoder embedding/mask token/blocks/prediction head 在 `models_mae.py:58-80`，`forward_decoder()` 在 `models_mae.py:204-233`。
- MSE reconstruction loss 存在：`loss = (pred - target)**2` at `mae/models_mae.py:247`，并只在 masked patches 上平均 at `mae/models_mae.py:250`。
- Stage 1 checkpoint save format包含 `'model': model_without_ddp.state_dict()`，见 `mae/util/misc.py:312-326`。OCR Stage 2 从 `mae_pretrained_path` 读取该 `'model'` key 并 strict=False 加载 encoder，见 `system.py:213-217`。
- Stage 1 与 Stage 2 分离明确：Stage 1 使用 `mae/` 下的 `MaskedAutoencoderViT` full MAE；Stage 2 使用 `ocr_training/strhub/models/models_mae.py` 下的 `MaskedAutoencoderViT_Encoder` encoder-only。

## 7. Semantic Enhancement Status

- 已实现。入口在 `Model.encode(img, labels=None)`，只有 labels 非空才计算 semantic enhancement：`system.py:269-275`。
- 位于 encoder-level：visual feature 使用 `mae_memory[:, 0, :]`，即 MAE/ViT CLS token；经 `self.proj` 投影到 512D CLIP text space。
- CLIP text encoder 加载于 `system.py:219`，visual tower 在 `system.py:221` 删除。`clip.tokenize(labels)` 在 `system.py:270`，`encode_text()` 实现在 `clip/model.py:405-423`。
- CLIP text tower冻结：token embedding、transformer、LN 被 `requires_grad=False` 并置 eval，见 `system.py:223-235`；optimizer 还排除所有 `"clip_model"` 参数，见 `system.py:496`。
- InfoNCE 与论文 Figure 6 / 公式一致：`L = Lce + alpha * LinfoNCE` 在代码中是 `loss = ocr_loss + 0.1 * cross_modal_loss` (`system.py:471`)。
- 推理无 semantic branch compute：`forward()` 调 `encode(images)` 不传 labels (`system.py:301`)。
- 实现差异/注意点：alpha 固定写死在代码，不是 config key；CLIP text tower虽然不参与推理 forward，但模型构造仍加载它。

## 8. Decoder Status

- 当前 decoder 是 PARSeq/XLNet-style two-stream Transformer decoder。证据：`DecoderLayer` docstring 写明 "two-stream attention (XLNet)" at `modules.py:95-97`；训练使用 permutation masks (`system.py:365-437`)；推理使用 AR loop (`system.py:309-338`)。
- 它就是后续计划替换的 PARSeq/AR Decoder 部分。最小替换范围是 `Model.decode()`、`Model.forward()` 中 AR/refinement 控制流，以及 `training_step()` 中 PLM permutation loss 逻辑。
- `decode()` 返回 hidden states，不返回 logits：`system.py:280-291` 直接返回 `self.decoder(...)`。logits 独立由 `self.head(...)` 生成：inference at `system.py:328/360`，training at `system.py:458/467`。
- `head` 独立于 decoder，在 `system.py:253` 定义。因此 MDiff Decoder 如果输出 hidden dim 仍为 768，可以复用 `head` 结构和可能的权重。
- `head` 会随 Lightning checkpoint 一起保存，并在 `load_from_checkpoint(..., strict=False)` 中恢复；相关入口是 `train.py:97-108` 和 `models/utils.py:77-84`。
- 应复用：
  - MAE encoder 与 `mae_pretrained_path` 加载逻辑；
  - `Tokenizer` / charset / EOS+BOS+PAD ID 约定；
  - `self.head` 输出类别约定；
  - DataModule 和 `test.py` 的基本 exact-match/NED evaluation。
- 应废弃或绕开：
  - `gen_tgt_perms()` / `generate_attn_masks()`；
  - AR `decode_ar` loop；
  - PARSeq refinement block；
  - PLM CE over permutations。
- 推荐 MDiff 插入点：`memory = self.encode(images)` 之后、`logits = self.head(hidden)` 之前。实现上建议新建模型 package，避免改动 baseline `maevit_infonce_plm/system.py`。

## 9. Evaluation Gap Against Paper

- 当前测试入口：`ocr_training/test.py:64-145`。
- Checkpoint 加载：`load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)` at `test.py:99`；模型类选择/Lightning loading 在 `ocr_training/strhub/models/utils.py:77-84`。
- 测试集构造：`SceneTextDataModule(... args.test_dir ...)` at `test.py:101-102`；`test_dataloaders()` 从 `root_dir/test/test_dir/subset` 读取 LMDB at `module.py:104-109`。
- 指标：
  - Accuracy: exact string match in `base.py:127-128`。
  - NED: `edit_distance(pred, gt) / max(len(pred), len(gt))` in `base.py:125-126`，`test.py:131` 输出 `100 * (1 - mean)`。
  - Confidence: decoded token probabilities product in `base.py:123`，`test.py:132` 输出均值百分比。
  - Label Length: `base.py:130` and `test.py:133`。
- Subset row: `test.py:117-134` 对每个 subset 产生 row，`print_results_table()` 在 `test.py:41-61` 输出 `Combined`。
- 6884 unified_lmdb: 在 `slpr_ocr` 环境中读取 `ocr_training/data/test/SLP34K_lmdb_benchmark/unified_lmdb` 得到 `num-samples=6884`，且存在 `meta-000006884`。当前 `test.py --test_data SLP34K --test_dir SLP34K_lmdb_benchmark` 可把 `unified_lmdb` 当一个 subset row 评估。
- single/multi/vertical: 当前数据目录包含 `single-line_lmdb`, `multi-lines_lmdb`, `vertical_lmdb`，DataModule 的 glob 支持这些目录级 row（`module.py:28-29`）。注意该 glob 是相对当前工作目录 `./data/...`，README 要求先 `cd ./ocr_training`。
- easy/hard: Not found in current repo as evaluator grouping. 搜索 `quality/structure/vocabulary_type/resolution_type/structure_type` 只在 LMDB meta 中发现，源码未读取这些字段。原因：evaluation script simplified；metadata exists but is ignored by `LmdbDataset` and `test.py`。
- Paper table reproduction gap:
  - 当前脚本可复现 exact accuracy/NED/confidence/length 的 subset table。
  - 不能直接按 LMDB `meta-*` 字段做 easy/hard、quality、structure、OOV/IV 等统一 benchmark 内部分组。
  - 需要新增 evaluator：读取 `meta-%09d`，对 predictions/labels 按 `quality`, `structure`, `vocabulary_type`, `resolution_type`, `structure_type` 聚合，并输出 paper table rows 与可追溯 CSV/JSONL。

## 10. Implications for SLP34K_v2

1. 必须保留的模块：
   - `ocr_training/strhub/models/models_mae.py` encoder-only MAE/ViT；
   - `mae_pretrained_path` checkpoint loading contract；
   - `Tokenizer` special token order：EOS first, BOS/PAD last；
   - SLP34K_568 charset；
   - `head` 输出 EOS+charset 的 569 类设计；
   - DataModule LMDB pipeline and basic evaluator。

2. 需要替换的模块：
   - `Model.decode()` 当前 two-stream decoder；
   - `forward()` 中 `decode_ar` loop and refinement；
   - `training_step()` 中 permutation masks and PLM CE accumulation。

3. 必须继承的配置项：
   - `model.img_size: [224,224]`;
   - `model.max_label_length: 50`;
   - `charset=SLP34K_568`;
   - `dataset=SLP34K`;
   - `model.embed_dim: 768` if reusing head/encoder directly;
   - `model.mae_pretrained_path`。

4. tokenizer/head 兼容性要求：
   - 输出 logits class ids 必须覆盖 `[EOS] + charset`，不覆盖 BOS/PAD。
   - `PAD` target 可继续作为 `ignore_index`，即使 pad id 超出 logits class range。
   - 如果 MDiff 需要 `[MASK]` token，建议作为 input-only token 处理，避免破坏 head 输出类数和 checkpoint head 兼容性。

5. 可先关闭或隔离的训练逻辑：
   - CLIP/InfoNCE semantic branch 可作为可选 auxiliary；为了先验证 decoder replacement，可先只做 denoising CE。
   - PARSeq permutation loss 应关闭，因为它和 diffusion-style noising 目标不一致。

6. 最推荐的 MDiff 插入点：
   - `memory = self.encode(images)` 之后；
   - 新 decoder 输出 `[B, max_label_length + 1, 768]`；
   - 原 `self.head` 投射到 `[B, max_label_length + 1, 569]`。

7. 最大工程风险：
   - token id 对齐风险，尤其 `[MASK]` 与 EOS/BOS/PAD/head 类别之间的关系；
   - checkpoint loading 风险，`load_from_checkpoint()` 当前只识别 `maevit_infonce_plm` 与 `maevit_plm` (`utils.py:46-54`)；
   - evaluator 风险，当前没有 metadata grouped evaluator，无法直接判断 MDiff 在论文分组上的收益；
   - inference length/EOS 处理风险，diffusion full-mask decoding 不再天然逐步遇 EOS early-stop。

## 11. Appendix: Evidence Snippets

### A1. OCR model init and config contract

`ocr_training/strhub/models/maevit_infonce_plm/system.py:184-202`

```python
   184	class Model(CrossEntropySystem):
   185	
   186	    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
   187	                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
   188	                 img_size: Sequence[int], embed_dim: int, mae_pretrained_path: str,
   189	                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
   190	                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
   191	                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
   192	        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
   193	        self.save_hyperparameters()
   200	        self.max_label_length = max_label_length
   201	        self.decode_ar = decode_ar
   202	        self.refine_iters = refine_iters
```

### A2. MAE encoder selection and checkpoint loading

`ocr_training/strhub/models/maevit_infonce_plm/system.py:204-217`

```python
   204	        import strhub.models.models_mae as models_mae
   205	        if img_size[0] == 32 and img_size[1] == 128:
   206	            if embed_dim == 384:
   207	                mae_model = getattr(models_mae, 'mae_vit_base_patch4_384_32x128')()
   208	            elif embed_dim == 768:
   209	                mae_model = getattr(models_mae, 'mae_vit_base_patch4_768_32x128')()
   210	        elif img_size[0] == img_size[1] == 224:
   211	            mae_model = getattr(models_mae, 'mae_vit_base_patch16_224x224')()
   213	        chkpt_dir = mae_pretrained_path
   214	        checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
   216	        mae_model.load_state_dict(checkpoint['model'], strict=False)
   217	        self.encoder = mae_model
```

### A3. Encoder-only MAE/ViT output includes CLS token

`ocr_training/strhub/models/models_mae.py:32-65`

```python
    32	        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
    33	        num_patches = self.patch_embed.num_patches #196    224/16 ,224/16 = 14*14
    35	        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    36	        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
    44	    def forward_encoder(self, x):
    46	        x = self.patch_embed(x)  
    52	        cls_token = self.cls_token + self.pos_embed[:, :1, :] 
    53	        cls_tokens = cls_token.expand(x.shape[0], -1, -1) 
    54	        x = torch.cat((cls_tokens, x), dim=1) 
    59	        x = self.norm(x)
    61	        return x
    63	    def forward(self, imgs): 
    64	        latent = self.forward_encoder(imgs) 
    65	        return latent
```

### A4. Semantic enhancement CLIP branch

`ocr_training/strhub/models/maevit_infonce_plm/system.py:219-240`

```python
   219	        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
   221	        del self.clip_model.visual   
   223	            for param in self.clip_model.token_embedding.parameters():
   224	                param.requires_grad = False
   227	            self.clip_model.positional_embedding.requires_grad = False 
   229	            for param in self.clip_model.transformer.parameters():
   230	                param.requires_grad = False 
   233	            for param in self.clip_model.ln_final.parameters():
   234	                param.requires_grad = False 
   237	        scale = embed_dim ** -0.5
   238	        self.proj =  nn.Parameter(scale * torch.randn(embed_dim, 512))
   240	        self.InfoNCELoss = InfoNCELoss()
```

### A5. encode() computes InfoNCE only when labels are present

`ocr_training/strhub/models/maevit_infonce_plm/system.py:265-277`

```python
   265	    def encode(self, img,labels = None):
   267	        mae_memory = self.encoder(img)
   269	        if labels is not None:
   270	            labels = clip.tokenize(labels).to(mae_memory.device)
   271	            text_features = self.clip_model.encode_text(labels) 
   273	            loss = self.InfoNCELoss(mae_memory[:,0,:]@self.proj, text_features)
   275	            return mae_memory,loss
   276	        else:
   277	            return mae_memory
```

### A6. decode() returns hidden states

`ocr_training/strhub/models/maevit_infonce_plm/system.py:280-291`

```python
   280	    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
   283	        N, L = tgt.shape #B,tgt_len
   285	        null_ctx = self.text_embed(tgt[:, :1])
   286	        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
   287	        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1)) #B*T*768
   289	            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1) #B*T*768
   291	        return self.decoder(query=tgt_query, content=tgt_emb, memory=memory, query_mask=tgt_query_mask, content_mask=tgt_mask, content_key_padding_mask=tgt_padding_mask)
```

### A7. AR inference loop and EOS early stop

`ocr_training/strhub/models/maevit_infonce_plm/system.py:309-338`

```python
   309	        if self.decode_ar:
   310	            tgt_in = torch.full((bs, num_steps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
   311	            tgt_in[:, 0] = self.tokenizer.bos_id
   313	            logits = []
   314	            for i in range(num_steps):
   320	                tgt_out = self.decode(
   328	                p_i = self.head(tgt_out)
   329	                logits.append(p_i)
   330	                if j < num_steps:
   332	                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
   334	                    if testing and (tgt_in == self.tokenizer.eos_id).any(dim=-1).all():
   336	                        break
   338	            logits = torch.cat(logits, dim=1)
```

### A8. Training CE + InfoNCE total loss

`ocr_training/strhub/models/maevit_infonce_plm/system.py:439-471`

```python
   439	    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
   440	        images, labels = batch
   441	        tgt = self.tokenizer.encode(labels, self._device) # B * (T+1)
   443	        memory,cross_modal_loss = self.encode(images,labels) # B*197*768
   446	        tgt_perms = self.gen_tgt_perms(tgt) # k*(T+1)
   447	        tgt_in = tgt[:, :-1] #B*32
   448	        tgt_out = tgt[:, 1:] #B*32
   456	            tgt_mask, query_mask = self.generate_attn_masks(perm) 
   457	            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
   458	            logits = self.head(out).flatten(end_dim=1)
   459	            ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
   468	        ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
   470	        ocr_loss /= loss_numel
   471	        loss =ocr_loss + 0.1*cross_modal_loss
```

### A9. Two-stream decoder and token embedding

`ocr_training/strhub/models/maevit_infonce_plm/modules.py:95-168`

```python
    95	class DecoderLayer(nn.Module):
    96	    """A Transformer decoder layer supporting two-stream attention (XLNet)
   102	        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
   103	        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
   141	    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
   145	        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
   152	class Decoder(nn.Module):
   161	    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
   163	        for i, mod in enumerate(self.layers):
   167	        query = self.norm(query)
   168	        return query
```

`ocr_training/strhub/models/maevit_infonce_plm/modules.py:234-242`

```python
   234	class TokenEmbedding(nn.Module):
   236	    def __init__(self, charset_size: int, embed_dim: int):
   238	        self.embedding = nn.Embedding(charset_size, embed_dim)
   241	    def forward(self, tokens: torch.Tensor):
   242	        return math.sqrt(self.embed_dim) * self.embedding(tokens)
```

### A10. Tokenizer special tokens and EOS decode

`ocr_training/strhub/data/utils.py:102-127`

```python
   102	class Tokenizer(BaseTokenizer):
   103	    BOS = '[B]'
   104	    EOS = '[E]'
   105	    PAD = '[P]'
   107	    def __init__(self, charset: str) -> None:
   108	        specials_first = (self.EOS,)
   109	        specials_last = (self.BOS, self.PAD)
   111	        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]
   113	    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
   114	        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
   116	        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
   118	    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
   121	            eos_idx = ids.index(self.eos_id)
   125	        ids = ids[:eos_idx]
   126	        probs = probs[:eos_idx + 1]
```

### A11. MAE reconstruction decoder and MSE loss

`mae/models_mae.py:58-80`

```python
    58	        # MAE decoder specifics
    59	        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    61	        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    63	        self.decoder_pos_embed = nn.Parameter(
    67	        self.decoder_blocks = nn.ModuleList([
    77	        self.decoder_norm = norm_layer(decoder_embed_dim)
    78	        self.decoder_pred = nn.Linear(
    79	            decoder_embed_dim, patch_size**2 * in_chans,
    80	            bias=True)  # decoder to patch
```

`mae/models_mae.py:235-257`

```python
   235	    def forward_loss(self, imgs, pred, mask):
   241	        target = self.patchify(imgs)
   247	        loss = (pred - target)**2
   248	        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
   250	        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
   253	    def forward(self, imgs, mask_ratio=0.75):
   254	        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
   255	        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
   256	        loss = self.forward_loss(imgs, pred, mask)
   257	        return loss, pred, mask, imgs
```

### A12. DataModule and Dataset read labels/images only

`ocr_training/strhub/data/module.py:104-112`

```python
   104	    def test_dataloaders(self, subset):
   105	        transform = self.get_transform(self.img_size, rotation=self.rotation)
   106	        root = PurePath(self.root_dir,'test',self.test_dir)
   107	        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
   108	                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
   109	                                   transform=transform) for s in subset}
   110	        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
   111	                              pin_memory=True, collate_fn=self.collate_fn)
   112	                for k, v in datasets.items()}
```

`ocr_training/strhub/data/dataset.py:85-115`

```python
    85	    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
    88	        with self._create_env() as env, env.begin() as txn:
    89	            num_samples = int(txn.get('num-samples'.encode()))
    92	            for index in range(num_samples):
    94	                label_key = f'label-{index:09d}'.encode()
    95	                label = txn.get(label_key).decode()
    99	                # Filter by length before removing unsupported characters. The original label might be too long.
   100	                if len(label) > max_label_len:
   101	                    continue
   102	                label = charset_adapter(label)
   113	                self.labels.append(label)
   114	                self.filtered_index_list.append(index)
   115	        return len(self.labels)
```

### A13. Evaluation metrics

`ocr_training/strhub/models/base.py:120-131`

```python
   120	        probs = logits.softmax(-1)
   121	        preds, probs = self.tokenizer.decode(probs)
   122	        for pred, prob, gt in zip(preds, probs, labels):
   123	            confidence += prob.prod().item()
   124	            pred = self.charset_adapter(pred)
   125	            # Follow ICDAR 2019 definition of N.E.D.
   126	            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
   127	            if pred == gt:
   128	                correct += 1
   129	            total += 1
   130	            label_length += len(pred)
   131	        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))
```

### A14. Hydra defaults and SLP baseline config

`ocr_training/configs/main.yaml:1-16`

```yaml
     1	defaults:
     2	  - _self_
     3	  - model: maevit_infonce_plm
     4	  - charset: SLP34K_568
     5	  - dataset: SLP34K
    10	  img_size: [224, 224] # [ height, width ]
    11	  max_label_length: 50
    16	  batch_size: 32
```

`ocr_training/configs/model/maevit_infonce_plm.yaml:1-22`

```yaml
     1	name: maevit_infonce_plm
     2	_target_: strhub.models.maevit_infonce_plm.system.Model
     5	embed_dim: 768
     6	mae_pretrained_path: pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth
     7	dec_num_heads: 12
     9	dec_depth: 3
    13	perm_num: 6
    14	perm_forward: true
    15	perm_mirrored: true
    20	# Decoding mode (test)
    21	decode_ar: true
    22	refine_iters: 1
```

### A15. `slpr_ocr` verification commands

Executed in project root with conda env `slpr_ocr`:

```text
$ python -c "... count charset ..."
charset_len 568
unique_len 568

$ python -c "... read unified_lmdb ..."
num_samples 6884
has_meta_6884 {"id": 6884, "quality": "middle", "structure": "vertical", "vocabulary_type": "OOV", "resolution_type": "low", "structure_type": "vertical"}
```
