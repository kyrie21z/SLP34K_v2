# V2-M02r Encoder-Decoder Incompatibility Analysis

说明：本报告正文使用中文；英文仅保留在用户指定的标题/章节名、论文题名、代码标识、配置项和必要术语中。未启动训练，未修改模型代码、数据或默认配置。

## 1. Executive Summary

- **不能直接模块替换。** SLP34K encoder 与 MDiff4STR decoder 都处在 encoder-decoder 框架中，但不是同构接口。SLP34K 的“强”来自 MAE/ViT encoder 与 PARSeq/AR decoder 在 Stage 2 中共同 fine-tune 后形成的可读 memory；MDiff4STR 的“强”来自 SVTRv2 encoder 与 masked denoising decoder 联合训练后的 denoising 能力。
- **当前 collapse 的最可能根因**不是单一 loss、EOS、adapter 或 CLS 问题，而是 **SLP MAE memory 不具备 MDiff full-mask decoder 所需的字符位置对齐表示**。`[B,197,768]` 的 CLS + 14x14 patch memory 不是天然的 OCR text-line sequence memory。
- **责任排序**：representation/interface mismatch 最高；frozen encoder + random decoder 的 non-coadaptation 次之；SLP 规范输出序列和复杂布局第三；noising/TRN/LC-BLC 缺失第四；EOS bias/loss 设计是症状放大器；tokenizer/head/checkpoint/CLS 单点问题已明显降权。
- **已有实验反证很强。** V2-M02c/d/h/h-fix 已经排除了 PAD 进 loss、target id 越界、baseline encoder migration 失败、full-argmax feedback 作为唯一原因、loss position 过稀疏作为唯一原因、CLS token 作为唯一原因。
- **`linear_ln` 只解决数值投影，不解决结构对齐。** 它把不同图像同位置 logits cosine 从约 0.996 降到约 0.964，并提高 real-vs-shuffled sensitivity，但仍然 EOS/repetition collapse。
- **不建议继续 blind training。** 当前路线下多跑 1000/3000 step 只会继续优化错误捷径；应先改变适配策略。
- **推荐路线**：首选把 MDiff 作为 PARSeq 预测的 corrector；次选增加 OCR neck 并做 joint/auxiliary warmup；暂缓 direct full-mask decoder replacement；不进入 V2-M03。

## 2. Evidence Inventory

| Evidence | Source | Key Finding | Implication |
| -------- | ------ | ----------- | ----------- |
| SLP34K main paper | 官方 AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/32569；用户给定的 `/mnt/data/...strong baseline.pdf` 本地未找到 | 摘要说明 baseline 的核心是 self-supervised pretraining + semantic enhancement 得到 strong visual encoder。 | 论文主张是增强 encoder，不是声明 encoder memory 可被任意 decoder 直接读取。 |
| SLP34K README | `README.md` | README 明确两阶段：MAE self-supervised pretraining；semantic enhancement 用视觉-文本 contrastive learning 强化 encoder。训练命令使用 `224x224`、`max_label_length=50`、`maevit_infonce_plm`。 | baseline 与 224 square MAE ViT memory 和 PARSeq-style recognition path 绑定。 |
| SLP34K supplementary | `/mnt/data/zyx/SLP34K_v2/aaai2025_ocr_supp.pdf` | 补充材料说明复杂/乱序 SLP 统一按“中文字符、数字、英文字母”标注；Figure 6 显示 semantic enhancement 在 encoder side 且只训练时使用；实现细节为 224x224、16x16 patch、max length 50。 | SLP 的输出顺序和 semantic objective 是任务特定的，不等同于自然空间阅读顺序。 |
| SLP34K code | `ocr_training/strhub/models/maevit_infonce_plm/system.py`、`models_mae.py`、`modules.py` | encoder 返回包含 CLS 的全部 ViT tokens；InfoNCE 使用 `mae_memory[:,0,:]`；decoder 是 PARSeq/XLNet-style two-stream cross-attention，训练时与 encoder 共同适配。 | 原 decoder 会学习如何读这种 memory；这不保证 MDiff 能直接读。 |
| SLP tokenizer/head code | `ocr_training/strhub/data/utils.py`、`base.py`、`SLP34K_568.yaml` | tokenizer 为 EOS + 568 字符 + BOS + PAD；head 输出 569 类，即 EOS + 字符，不输出 BOS/PAD。 | 当前 SLP-MDiff 的 special id/head contract 与官方 MDiff 原则一致，纯 tokenizer mismatch 不像主因。 |
| OpenOCR MDiffDecoder code | `/mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py` | 官方 MDiff 使用 EOS=0、MASK=out_channels-2、PAD=out_channels-1；文本 embedding 叠加 fixed PE + learned PE；bidirectional self-attn；cross-attn 读 visual memory；classifier 排除 MASK/PAD。 | MDiff 是 denoising decoder，不是任意 encoder 的通用替换头。 |
| OpenOCR MDiffLabelEncode code | `/mnt/data/zyx/OpenOCR_ref/openrec/preprocess/mdiff_label_encode.py` | noising 包含 full、left-to-right、right-to-left、block、cloze、random；reflect/token replacement 构造非 mask 噪声。 | 官方 MDiff 的训练假设包含 denoising curriculum 和 TRN，不是只有 random/full mask。 |
| OpenOCR configs | `/mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/*.yml` | encoder 是 `SVTRv2LNConvTwo33`；输入尺度 32x128；final dim 384；decoder max_len=25、sampler_step=3、sample_k=3；训练数据为 Union14M 或中文 benchmark。 | 官方系统是 SVTRv2 + MDiff 联合训练，不是 frozen MAE encoder + random decoder。 |
| MDiff4STR paper | arXiv：https://arxiv.org/abs/2512.01422 | 论文指出 vanilla MDM 在 STR 中落后于 ARM，关键问题是 noising gap 和 overconfidence，因此提出六种 noising strategies 与 token-replacement noise。 | 即便在原生 STR 设置下，vanilla MDM 也不够；当前简化 direct replacement 更低于原始假设。 |
| V2-M02c report | `reports/V2-M02c_short_training_diagnostics_receipt.md` | baseline encoder migration、real batch forward/backward、checkpoint save/load 都通过；1000 step 后从重复字符退化为 EOS/空串。 | 训练管线和加载不是主阻塞。 |
| V2-M02d report | `reports/V2-M02d_loss_mode_ablation_receipt.md` | `masked_or_eos`、`all_non_pad`、`full_mask_all_non_pad` 都 collapse；PAD 不进 loss；target id 不越界。 | loss position 稀疏和 PAD/id bug 不能解释根因。 |
| V2-M02g0 report | `reports/V2-M02g0_openocr_clone_and_mdiff_code_reading_receipt.md` | Plain decoder 不是官方代码级移植；官方有更丰富 PE、TransformerBlock、noising、TRN 和 inference variants。 | Plain failure 说明初版实现不完整，但官方-core failure 后问题上升为接口/表示层面。 |
| V2-M02h report | `reports/V2-M02h_official_core_mdiff_receipt.md` | official-core + parallel decoding 仍 collapse；`different_images_same_position_cosine≈0.9957`；shuffled memory top1 changed rate 约 0.149。 | full-argmax feedback 不是唯一原因；image-conditioned logits 太同质。 |
| V2-M02h-fix report | `reports/V2-M02h_fix_conditioning_ablation_receipt.md` | drop CLS 几乎无效；`linear_ln` 改善 conditioning 指标但不改善解码；partial-mask probe 只有极弱改善。 | 简单 adapter 和 CLS removal 不足；缺的是 OCR alignment。 |

## 3. Paper-level Comparison

| Dimension | SLP34K Baseline | MDiff4STR | Incompatibility |
| --------- | --------------- | --------- | --------------- |
| task target | 船舶牌照识别，混合中文、数字、英文，且有复杂布局。 | 通用 scene text recognition，覆盖 regular/irregular/occluded/Chinese text。 | SLP 有规范序和船牌结构，generic MDiff 不知道这些先验。 |
| encoder role | 论文重点是 strong visual encoder：MAE 预训练 + encoder-side semantic enhancement。 | encoder 是 SVTRv2，与 denoising decoder 一起训练。 | SLP encoder 的强不是“MDiff-readable memory”的强。 |
| decoder role | PARSeq/XLNet-style two-stream decoder，通过 AR/permutation training 读 memory。 | bidirectional mask diffusion decoder，从 noised tokens 恢复 clean tokens。 | AR/permutation reading 与 full-mask denoising 对 memory 要求不同。 |
| visual feature requirement | ViT CLS + patch tokens，并用 CLS-text InfoNCE 做全局语义增强。 | OCR-specific SVTRv2 sequence feature。 | 一个是 square patch representation，一个是 text-oriented visual sequence。 |
| training objective | PLM/AR recognition CE + CLS-level InfoNCE。 | masked denoising CE + mask/length normalization + reflect/TRN correction。 | 一个强调全局语义和 AR 可读，一个强调 token-level denoising。 |
| inference behavior | 默认 AR decode，并带 refinement。 | 从 full mask 开始，或使用 LC/BLC/AR/random/cloze 等 denoising。 | full-mask first step 没有历史文本上下文，更依赖 visual grounding。 |
| sequence order assumption | 补充材料说明乱序样本也统一按中文、数字、英文标注。 | generic STR label sequence 通常就是识别序列。 | SLP output index 不一定单调对应图像空间位置。 |
| degradation handling | 主要靠 MAE 和 semantic enhancement 强化视觉 encoder。 | 主要靠 masked denoising、TRN、remask 处理预测不确定性。 | SLP 的鲁棒性在 encoder side；MDiff 的优势在 decoder-side uncertainty handling。 |
| language/structure prior | 没有 decoder-side LM fusion；CLIP text branch 只训练时用。 | decoder self-attn 提供通用双向上下文，但无 SLP segment/pinyin prior。 | MDiff 的 generic context 不等于 SLP 结构先验。 |
| expected coupling | encoder 与 PARSeq decoder 在 Stage 2 共同 fine-tune。 | SVTRv2 与 MDiffDecoder 在 OpenOCR 中共同训练。 | frozen SLP encoder + random MDiff decoder 同时违背两边的 co-adaptation 假设。 |

**明确回答：** 论文层面，SLP34K encoder + MDiff4STR decoder 不是“模块可替换”的同构组合。SLP34K 论文证明的是 encoder 在原 PARSeq/AR 路径中强；MDiff4STR 论文证明的是 SVTRv2 + MDiff denoising 体系强。两者可作为研究材料组合，但需要显式设计 representation bridge、训练范式和 SLP structure prior。

## 4. Code-level Interface Comparison

| Interface | SLP34K Code | MDiff4STR Code | Risk |
| --------- | ----------- | -------------- | ---- |
| feature shape | 224x224 patch16 输出 `[B,197,768]`，包含 CLS + 14x14 patches。 | SVTRv2 config 最终 dim 384，输入 32x128，输出更接近 text-line visual sequence。 | 长度、维度和空间语义都不同。 |
| feature source | MAE ViT encoder，baseline 中与 PARSeq decoder 共同 fine-tune。 | `BaseRecognizer` 中 `SVTRv2LNConvTwo33 -> MDiffDecoder` 直接连成一个 recognizer。 | 官方 MDiff memory 来自联合训练的 OCR encoder，不是任意 visual feature。 |
| CLS token | SLP memory 含 CLS，InfoNCE 使用 CLS。 | SVTRv2 memory 没有同类 CLS token。 | CLS 可能引入 global bias，但 H1 drop-CLS 证明它不是唯一问题。 |
| patch/grid token | SLP 是 14x14 square patch flatten。 | SVTRv2 使用 conv/local/global mixer 和 height/width downsampling。 | SLP patch order 不是 OCR 字符顺序。 |
| feature normalization | MAE encoder 末端 LayerNorm。 | SVTRv2 混合 BatchNorm/LayerNorm。 | 数值分布有风险，但 `linear_ln` 无法独立修复。 |
| positional/spatial encoding | MAE 内部 fixed sin-cos patch PE。 | SVTRv2 通过 OCR-specific conv/local/global 结构编码空间；MDiff text side 有 fixed+learned PE。 | text PE 不能自动生成 visual-to-text alignment。 |
| decoder query init | PARSeq 使用 BOS/null context、text embedding、learned `pos_queries` 和 AR/permutation masks。 | MDiff 使用 noised token ids + fixed PE + learned PE + bidirectional self-attn。 | full MASK 时所有位置 token 内容几乎相同，只剩 position 与 visual memory 区分。 |
| cross-attention | token query attend to all MAE tokens。 | token hidden attend to SVTRv2 memory。 | 算子相似，学习到的 alignment 完全不同。 |
| output length | SLP `max_label_length=50`，输出 51 含 EOS。 | English config `max_text_length=25`，中文 config 另设。 | SLP 序列更长，charset 更大，full-mask 难度更高。 |
| tokenizer/special ids | EOS=0，chars=1..568，BOS=569，PAD=570；SLP-MDiff 增加 input-only MASK=571。 | EOS=0，chars，MASK=out_channels-2，PAD=out_channels-1。 | 当前已经对齐原则，id mismatch 不是主因。 |
| head classes | 输出 569 类：EOS + charset。 | `tgt_word_prj` 输出 `out_channels-2`：EOS + chars。 | 当前 head 设计正确。 |
| labels/noising tensors | 当前 SLP-MDiff 使用 `Tokenizer.encode(labels)[:,1:]`、random/full noising、official normalized loss。 | 官方需要 `labels, reflect_ids, noisy_batch, masked_indices, p_mask, length`，且有 all-layer/sample_k branch。 | 当前训练缺少完整 noising/TRN，但 official-core collapse 说明它不是唯一问题。 |

**明确回答：** conditioning diagnostic 中 `different_images_same_position_cosine` 过高，说明同一输出位置的 logits 主要由共享的位置/语言/decoder bias 决定，不同图像的 visual memory 没有提供足够可判别的字符证据。real-vs-zero/shuffle 能改变部分 top1，说明 memory 不是完全无效；但它的作用不足以支撑 image-conditioned recovery。

## 5. Representation Mismatch Analysis

1. **MAE reconstruction feature 不等于 OCR-localized feature。** MAE 预训练学习的是遮挡 patch 重建能力和鲁棒 visual representation；它不会强制某个 output token 位置能直接 attend 到对应字符。
2. **CLS-level semantic enhancement 不等于 token-level character evidence。** SLP34K 使用 `mae_memory[:,0,:] @ proj` 与 CLIP text feature 做 InfoNCE，强化的是全局图文语义一致性，不是每个 patch 到每个字符位置的对齐。
3. **AR/PARSeq-readable memory 不等于 full-mask denoising-readable memory。** AR decoder 有 BOS、历史字符和 permutation/causal context；full-mask MDiff 的第一步没有历史文本，只能靠位置和 visual memory。
4. **全局语义一致性不等于位置级恢复。** SLP 中中文和拼音可能语义冗余，全局表示可以知道“这是什么船牌”，但仍不知道第 i 个规范输出 token 应该从哪个局部区域读取。
5. **强 encoder 的“强”必须带下游条件。** 对 PARSeq 强，意味着原 decoder 经过共同训练后能读；对 MDiff 强，意味着 full-mask/partial-mask denoising decoder 能按位置恢复字符。当前证据显示二者不是同一种强。

**明确结论：**

```text
Strong visual encoder for PARSeq != directly usable visual memory for MDiff.
```

我不同意“SLP34K encoder 很强，所以应该适配 MDiff”的隐含前提。代码和实验都支持相反判断：当前 SLP memory 是强视觉表示，但不是直接 denoising-readable 的 OCR sequence memory。

## 6. Training Paradigm Mismatch Analysis

当前 V2-M02 训练范式是 frozen SLP encoder + random MDiff decoder。这只是一个模块可替换性的诊断，不符合 MDiff4STR 原始训练假设。

- SLP34K baseline 的 Stage 2 会训练 encoder、decoder、head 和 visual-text projector；CLIP text tower 冻结，但 encoder 不等于固定特征抽取器。
- OpenOCR 的 `BaseRecognizer` 按 `encoder -> decoder` 构建整模型，MDiff config 训练的是 SVTRv2 + MDiffDecoder 整体。
- MDiff4STR 论文明确指出 vanilla MDM 存在 noising gap 和 overconfidence，需要六种 noising strategy 与 token-replacement noise。
- 当前 SLP 数据只有 27,501 train images，却有 568 字符表、max length 50、复杂布局；随机初始化 full-mask decoder 在这种条件下从零学习视觉对齐非常困难。
- frozen encoder 阻止 memory 向 denoising-readable 方向调整，decoder 只能学习边际频率、EOS 和重复片段等捷径。

**明确回答：** 当前训练范式不符合 MDiff4STR 原始假设。后续如果继续 replacement，需要 warmup/curriculum，例如 teacher-forcing/partial-mask、AR/CTC auxiliary、OCR neck，或者部分 unfreeze encoder/neck，而不是直接 full-mask denoising。

## 7. Sequence and Layout Mismatch Analysis

SLP34K 的 label 并不总是简单的空间阅读序列。补充材料说明，很多 SLP 存在 layout disorder，并统一按中文字符、数字、英文字母标注。数据还包含 single-line、multi-line、vertical、reverse、discontinuous、low-lighting 等样本。

这会破坏 generic full-mask decoder 的隐含前提：

- 输出位置 0/1/2 不一定单调对应图像从左到右或从上到下的位置。
- 多行和竖排样本需要 layout prior。
- reverse/discontinuous 样本使空间证据更非单调。
- 英文拼音与中文段有冗余，但 generic MDiff 不知道 pinyin consistency。
- 大字符表 + 长序列 + EOS 位置可变，使 EOS/repetition shortcut 更容易成为早期收敛点。

**明确回答：** SLP34K 的规范输出序列会显著削弱 generic STR decoder 的空间-序列对齐假设。它不说明 MDiff 不能用于 SLP，但说明 generic full-mask decoder 不应直接替换 PARSeq decoder。

## 8. Root Cause Ranking

| Rank | Root Cause | Evidence | Confidence | How to Verify / Fix |
| ---: | ---------- | -------- | ---------- | ------------------- |
| 1 | visual memory not character-aligned | SLP memory 是 CLS + 14x14 MAE patches；same-position logits 跨图像 cosine 极高；`linear_ln` 改善 sensitivity 但不改善 decode。 | High | 加 OCR neck，把 `[B,196,768]` reshape 为 `[B,14,14,768]` 后做 row-wise/text-line projection；先用 teacher-forcing/partial-mask CE 验证 real-vs-shuffled gap。 |
| 2 | frozen encoder-decoder non-coadaptation | SLP baseline 与 OpenOCR MDiff 都是 encoder-decoder 联合训练；当前只训练随机 decoder。 | High | unfreeze neck/encoder 局部层，配 AR/CTC auxiliary warmup；验证 memory 是否变得可读。 |
| 3 | full-mask generation too hard for current data scale | SLP 训练集 27,501，568 字符，max length 50，布局复杂；full-mask first step 无文本上下文。 | High | 改为 partial-mask/PARSeq-prediction correction；检查 full-mask 是否唯一失败模式。 |
| 4 | no OCR neck / spatial adapter | OpenOCR 是 OCR-specific SVTRv2 sequence；SLP 是 square ViT patch grid。 | High | 最小 2D-to-1D neck；短程 probe 而非长训练。 |
| 5 | no structured output prior | SLP 有中文/数字/英文段、pinyin redundancy、layout prior。 | Medium-High | segment embeddings、layout-aware position prior、pinyin consistency；先做 no-train/short probe。 |
| 6 | insufficient noising strategies / TRN | 官方 MDiff 依赖六种 noising 与 token replacement；当前未实现。 | Medium | 仅在 memory 可读后再加 all-mask/TRN/LC-BLC。 |
| 7 | EOS bias / inference collapse | 多个 run 中 EOS 低绝对概率但 rank 1；parallel 后仍 collapse。 | Medium | EOS penalty/length prior 只能作为症状诊断，不能替代 visual alignment 修复。 |
| 8 | tokenizer/head mismatch | 已有断言和 debug 证明 ids/head/loss target 对齐。 | Low | 保留断言即可。 |
| 9 | CLS interference | H1 drop-CLS 几乎无效。 | Low | 保留开关，不作为主线。 |
| 10 | simple adapter insufficiency | `linear_ln` 有效提升指标但不能解码。 | High as insufficiency | adapter 必须与 OCR neck/训练范式一起改。 |

**当前所有 evidence 最支持的 root cause：** SLP MAE/InfoNCE visual memory 没有被组织成 MDiff full-mask decoder 可直接按输出位置读取的字符级 memory，并且 frozen setup 阻止了 encoder-memory interface 的共同适配。

## 9. What Has Been Ruled Out

1. **pure tokenizer id mismatch**：SLP-MDiff 明确断言 tokenizer length、EOS/PAD/MASK id、head classes；V2-M02d debug 证明 target id 在合法范围。
2. **PAD entering loss**：V2-M02d 显示 `pad_in_loss=0`；代码断言禁止 PAD/MASK/BOS 进入 CE target。
3. **checkpoint loading failure**：V2-M02c/d/h 多次显示 baseline encoder migration 和 checkpoint load missing/unexpected 为 0/0。
4. **baseline encoder migration failure**：真实 batch forward/backward 已通过，固定 probe 可跑。
5. **full-argmax feedback as sole cause**：V2-M02h 改为 parallel decoding，不再回灌 argmax，但仍 collapse。
6. **loss position sparsity as sole cause**：`all_non_pad` 和 `full_mask_all_non_pad` 仍 collapse。
7. **CLS token as sole cause**：H1 drop-CLS 后 conditioning 和输出基本不变。
8. **identity adapter as sole cause**：H2/H3 `linear_ln` 明显增强 memory sensitivity，但仍 collapse。
9. **Plain decoder incompleteness as sole cause**：official-core 后仍 collapse，说明问题不是只在 Plain 简化实现。
10. **继续训练更久是合理下一步**：loss 下降但 decode 早期退化，说明优化在学捷径；没有理论/代码/实验依据支持 blind longer training。

## 10. Revised Research Direction Options

| Option | Core Idea | Expected Benefit | Risk | Minimal Validation |
| ------ | --------- | ---------------- | ---- | ------------------ |
| Option C: MDiff as corrector | 用 PARSeq baseline prediction、low-confidence tokens 或 GT-noised sequence 作为输入，让 MDiff 做修正，而不是从 full mask 起步。 | 保留 baseline 已学到的 visual-to-sequence alignment，降低 denoising 难度。 | 贡献从“替换 decoder”变成“校正模块”；依赖 baseline 预测质量。 | freeze baseline，输入 prediction/noised tokens，只监督低置信或 masked positions；比较 real/shuffled visual 与 text context ablation。 |
| Option A: Add OCR neck between SLP Encoder and MDiff Decoder | `MAE patch memory -> [B,14,14,768] -> OCR neck/row-wise projector -> text-oriented memory -> MDiff`。 | 把 square ViT memory 转为更像 STR encoder output 的序列特征。 | neck 设计可能变复杂，且需要 joint training。 | 先做 shape-only neck + teacher-forcing/partial-mask probe；要求 real memory 明显优于 shuffled memory。 |
| Option B: Jointly Train Encoder + MDiff with AR Auxiliary Warmup | 加 AR/CTC auxiliary，让 memory 先保持字符可读，再逐步提高 mask ratio。 | 恢复 encoder-decoder coadaptation，降低 random decoder 难度。 | 变量增多，可能掩盖 MDiff 贡献。 | short warmup 中 AR/CTC 不 collapse，MDiff partial-mask CE 有 real-vs-shuffle gap。 |
| Option D: Structure-aware MDiff Decoder for SLP | 引入 Chinese/number/English segment embedding、layout-aware position prior、pinyin consistency、SLP-aware TRN。 | 直接处理 SLP 规范序和结构冗余。 | 容易过拟合规则，设计范围扩大。 | 仅从 label pattern 派生 segment ids，先验证 segment-conditioned partial-mask recovery。 |
| Option E: Stop Full Replacement, Return to MAS / Candidate Generation | 把 MDiff 定位为候选生成器、reranker 或 uncertainty module。 | 风险最低，能保住 strong baseline。 | 架构新颖性较弱。 | 围绕 PARSeq output 生成候选并重排，在 hard/OOV/multi-line subset 上比较。 |

推荐排序：

```text
1. 首选：Option C, MDiff as corrector
2. 次选：Option A + Option B, OCR neck + joint/auxiliary warmup
3. 后续：Option D, structure-aware MDiff
4. 保底：Option E, candidate/reranker
5. 暂缓：direct full-mask decoder replacement
```

## 11. Recommendation

不建议继续把“MDiff Decoder 直接替换 PARSeq Decoder”作为当前主线。这个方向还可以作为长期研究问题，但当前版本必须暂停。

建议：

1. **首选 Option C：MDiff as corrector。** 让 MDiff 修正已有识别序列，而不是从 `[MASK] * 51` 起步。
2. **次选 Option A + B：OCR neck + joint/auxiliary warmup。** 如果必须坚持 replacement，就先让 memory 变成 text-oriented sequence，再训练 MDiff。
3. **暂缓 direct full-mask replacement。** 只有 teacher-forcing/partial-mask probe 证明 memory 可读后才恢复。
4. **暂停实验**：1000/3000-step blind full-mask ablation、generic TRN、LC/BLC、SLP-aware TRN、segment-aware denoising、pinyin consistency。
5. **保留代码**：`slp_mdiff` plain/official core、adapter/drop-CLS 开关、conditioning diagnostic、partial-mask probe、special-id/head assertions。

**是否进入 V2-M03：不建议。**

## 12. Next Codex Task Proposal

```text
你现在进入 SLP34K_v2 的 V2-M02s 阶段。

目标：基于 V2-M02r 报告，设计 "MDiff as PARSeq corrector" 的最小验证方案。

严格边界：
1. 不修改 maevit_infonce_plm 原 baseline 行为。
2. 不修改 configs/main.yaml 默认模型。
3. 不启动长训练。
4. 不实现 all-mask/TRN/LC/BLC。
5. 不进入 V2-M03。

任务：
1. 阅读 V2-M02r report。
2. 设计 corrector contract：
   - input: image + baseline PARSeq prediction/logits 或 GT-noised token sequence；
   - visual memory: SLP encoder memory 或 baseline decoder hidden，需论证；
   - target: clean SLP token sequence。
3. 优先写 design report，不盲目改代码。
4. 如需代码，只实现 no-train 或极短 partial-mask probe：
   - 对比 full-mask vs PARSeq/GT partial context；
   - 对比 real memory vs shuffled memory；
   - 输出 CE gap 和 top1_changed_rate。
5. 输出 reports/V2-M02s_mdiff_corrector_design.md。
```

## 13. Appendix: Evidence Snippets

### A1. 论文/补充材料证据

- 主论文用户给定本地路径未找到；使用官方 AAAI 页面核对标题、摘要和引用：https://ojs.aaai.org/index.php/AAAI/article/view/32569。
- 本地补充材料：`/mnt/data/zyx/SLP34K_v2/aaai2025_ocr_supp.pdf`。
- 补充材料第 1 页：复杂/乱序 SLP 统一按中文字符、数字、英文字母顺序标注。
- 补充材料 Figure 6：semantic enhancement 在 encoder side，且 training phase 使用；不是 decoder-side LM fusion。
- 补充材料 implementation details：224x224 input、16x16 MAE patches、max length 50、MAE mask ratio 0.75、pretraining 1500 epochs、recognition training 100 epochs。
- `image/README/baseline.pdf` 显示 Figure 3 pipeline：Stage 1 MAE reconstruction；Stage 2 ViT encoder + AR decoder + CE/InfoNCE。

### A2. SLP MAE encoder 返回 CLS + patch tokens

```bash
nl -ba ocr_training/strhub/models/models_mae.py | sed -n '32,61p'
```

```text
    32	        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
    33	        num_patches = self.patch_embed.num_patches #196    224/16 ,224/16 = 14*14
    35	        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    36	        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
    44	    def forward_encoder(self, x):
    46	        x = self.patch_embed(x)
    49	        x = x + self.pos_embed[:, 1:, :]
    52	        cls_token = self.cls_token + self.pos_embed[:, :1, :]
    53	        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    54	        x = torch.cat((cls_tokens, x), dim=1)
    59	        x = self.norm(x)
    61	        return x
```

### A3. SLP semantic enhancement 使用 CLS feature

```bash
nl -ba ocr_training/strhub/models/maevit_infonce_plm/system.py | sed -n '265,277p'
```

```text
   265	    def encode(self, img,labels = None):
   267	        mae_memory = self.encoder(img)
   269	        if labels is not None:
   270	            labels = clip.tokenize(labels).to(mae_memory.device)
   271	            text_features = self.clip_model.encode_text(labels)
   273	            loss = self.InfoNCELoss(mae_memory[:,0,:]@self.proj, text_features)
   275	            return mae_memory,loss
   277	            return mae_memory
```

### A4. SLP PARSeq decoder cross-attends to memory

```bash
nl -ba ocr_training/strhub/models/maevit_infonce_plm/modules.py | sed -n '124,149p'
```

```text
   124	    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
   131	        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
   134	        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
   135	        tgt = tgt + self.dropout2(tgt2)
   141	    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
   145	        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
   149	        return query, content
```

### A5. SLP training 同时优化 OCR loss 与 InfoNCE

```bash
nl -ba ocr_training/strhub/models/maevit_infonce_plm/system.py | sed -n '439,472p'
```

```text
   439	    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
   441	        tgt = self.tokenizer.encode(labels, self._device)
   443	        memory,cross_modal_loss = self.encode(images,labels) # B*197*768
   446	        tgt_perms = self.gen_tgt_perms(tgt) # k*(T+1)
   457	            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
   459	            ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
   470	        ocr_loss /= loss_numel
   471	        loss =ocr_loss + 0.1*cross_modal_loss
```

### A6. SLP tokenizer/head convention

```bash
nl -ba ocr_training/strhub/data/utils.py | sed -n '102,127p'
```

```text
   102	class Tokenizer(BaseTokenizer):
   103	    BOS = '[B]'
   104	    EOS = '[E]'
   105	    PAD = '[P]'
   108	        specials_first = (self.EOS,)
   109	        specials_last = (self.BOS, self.PAD)
   111	        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]
   114	        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
   116	        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
   121	            eos_idx = ids.index(self.eos_id)
   125	        ids = ids[:eos_idx]
```

### A7. 当前 SLP-MDiff 冻结 encoder

```bash
nl -ba ocr_training/strhub/models/slp_mdiff/system.py | sed -n '89,101p'
```

```text
    89	        self.encoder = self._build_encoder(img_size, embed_dim)
    92	        if init_encoder_from_baseline_ckpt:
    93	            self.encoder_load_info["baseline_ckpt"] = self._load_baseline_encoder(baseline_ckpt_path)
    94	        if freeze_encoder:
    95	            for param in self.encoder.parameters():
    96	                param.requires_grad = False
    97	            self.encoder.eval()
    99	        self.visual_adapter = (
   100	            VisualAdapter(embed_dim, embed_dim, visual_adapter_type) if use_visual_adapter else nn.Identity()
```

### A8. 当前 visual memory 仍是 token-level adapter

```bash
nl -ba ocr_training/strhub/models/slp_mdiff/system.py | sed -n '207,232p'
```

```text
   207	    def encode(self, images: Tensor) -> Tensor:
   208	        if self.freeze_encoder:
   209	            with torch.no_grad():
   210	                memory = self.encoder(images)
   213	        return self.prepare_visual_memory(memory)
   215	    def prepare_visual_memory(self, memory: Tensor) -> Tensor:
   221	        if self.drop_cls_token:
   224	            memory = memory[:, 1:, :]
   225	        memory = self.visual_adapter(memory)
   232	        return memory
```

### A9. 官方 MDiff special ids 与 embedding

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py | sed -n '49,79p'
```

```text
    49	        self.out_channels = out_channels
    50	        self.ignore_index = out_channels - 1
    51	        self.mask_token_id = out_channels - 2
    52	        self.eos = 0
    54	        d_model = in_channels
    68	        self.embedding = Embeddings(
    69	            d_model=d_model,
    70	            vocab=self.out_channels,
    71	            padding_idx=0,
    74	        self.pos_embed = nn.Parameter(torch.zeros(
    75	            [1, self.max_len + 1, d_model], dtype=torch.float32),
    78	        self.positional_encoding = PositionalEncoding(
```

### A10. 官方 MDiff train loss 与 reflect branch

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py | sed -n '112,148p'
```

```text
   112	    def forward_train(self, memory, data=None):
   113	        labels, reflect_ids, noisy_batch, masked_indices, p_mask, length = data
   119	        tgts = self.embedding(noisy_batch)
   120	        tgts = self.positional_encoding(tgts) + self.pos_embed
   123	            tgts = decoder_layer(tgts, memory, self_mask=None)
   124	        logits = self.tgt_word_prj(tgts)
   125	        token_loss = F.cross_entropy(
   126	            logits[masked_indices],
   127	            labels[masked_indices],
   129	            ignore_index=self.ignore_index) / p_mask[masked_indices]
   133	        if reflect_ids is not None:
   141	            reflect_logits = self.tgt_word_prj(reflect_tgts)
   146	            loss = self.rec_loss_weight * loss + self.reflect_loss_weight * reflect_loss
```

### A11. 官方 MDiff noising strategies

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/preprocess/mdiff_label_encode.py | sed -n '197,252p'
```

```text
   197	    def forward_process(self, text):
   199	        rand_choice = random.choice(self.mask_tpye)
   200	        if rand_choice == 0:  # 并行mask full mask
   201	            return self.full_mask(text)
   202	        elif rand_choice == 1 and len(text) > 2:  # 正向自回归 right mask
   203	            return self.left_to_right_mask(text)
   204	        elif rand_choice == 2 and len(text) > 2:  # 反向自回归 left mask
   205	            return self.right_to_left_mask(text)
   206	        elif rand_choice == 3 and len(text) > 2:  # block mask
   234	        elif rand_choice == 4 and len(text) > 2:  # cloze mask
   245	        else:  # random mask
   247	            noisy_batch, masked_indices = self.random_mask(text)
```

### A12. 官方 MDiff parallel decoding 从 full mask 开始

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/modeling/decoders/mdiff_decoder.py | sed -n '264,272p'
```

```text
   264	    def forward_parallel_decoding(self, src):
   265	        bs = src.shape[0]
   266	        noisy_batch = torch.full((bs, self.max_len + 1),
   267	                                 self.mask_token_id,
   270	        tgts = self.forward_decoding(src, noisy_batch)
   271	        logits = F.softmax(self.tgt_word_prj(tgts), -1)
   272	        return logits
```

### A13. OpenOCR encoder-decoder 是整体 recognizer

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/modeling/base_recognizer.py | sed -n '32,68p'
```

```text
    32	        # build backbone
    37	            config['Encoder']['in_channels'] = in_channels
    38	            self.encoder = build_encoder(config['Encoder'])
    39	            in_channels = self.encoder.out_channels
    41	        # build decoder
    46	            config['Decoder']['in_channels'] = in_channels
    47	            self.decoder = build_decoder(config['Decoder'])
    62	    def forward(self, x, data=None):
    65	        if self.use_encoder:
    66	            x = self.encoder(x)
    67	        if self.use_decoder:
    68	            x = self.decoder(x, data=data)
```

### A14. OpenOCR MDiff config 使用 SVTRv2 而不是 frozen MAE

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml | sed -n '36,65p'
```

```text
    36	Architecture:
    38	  algorithm: MDiff4STR
    41	  Encoder:
    42	    name: SVTRv2LNConvTwo33
    44	    dims: [128, 256, 384]
    45	    depths: [6, 6, 6]
    47	    mixer: [['Conv','Conv','Conv','Conv','Conv','Conv'],['Conv','Conv','FGlobal','Global','Global','Global'],['Global','Global','Global','Global','Global','Global']]
    52	  Decoder:
    53	    name: MDiffDecoder
    54	    num_decoder_layers: 6
    56	    max_len: *max_text_length
    63	    sampler_step: 3
    64	    sample_k: &sample_k 3
```

### A15. SVTRv2 保留 OCR 空间序列结构

```bash
nl -ba /mnt/data/zyx/OpenOCR_ref/openrec/modeling/encoders/svtrv2_lnconv_two33.py | sed -n '345,384p'
```

```text
   345	class POPatchEmbed(nn.Module):
   356	        self.patch_embed = nn.Sequential(
   357	            ConvBNLayer(
   361	                stride=2,
   366	            ConvBNLayer(
   370	                stride=2,
   378	        if flatten:
   379	            self.patch_embed.append(FlattenTranspose())
   381	    def forward(self, x):
   382	        sz = x.shape[2:]
   383	        x = self.patch_embed(x)
   384	        return x, [sz[0] // 4, sz[1] // 4]
```

### A16. V2-M02h conditioning evidence

来源：`reports/V2-M02h_official_core_mdiff_receipt.md`，"Conditioning Diagnostic"。

```text
same_image_different_positions_cosine: 0.9589
different_images_same_position_cosine: 0.9957
real_vs_zero_memory.top1_changed_rate: 0.2784
real_vs_shuffled_memory.top1_changed_rate: 0.1490
position_embedding_zero_out.top1_changed_rate: 0.0235
```

解释：memory perturbation 有影响，但同一输出位置跨图像 logits 仍几乎相同。

### A17. V2-M02h-fix adapter evidence

来源：`reports/V2-M02h_fix_conditioning_ablation_receipt.md`，"Conditioning Diagnostics" 和 "Analysis"。

```text
H1 drop_cls identity diff_img_same_pos_cos: 0.9958
H2 keep_cls linear_ln diff_img_same_pos_cos: 0.9642
H3 drop_cls linear_ln diff_img_same_pos_cos: 0.9645
H2/H3 real_shuffle_top1_change: 0.2118
H2/H3 still collapse from step 200 and final outputs are empty or short repeated fragments.
```

解释：线性投影/归一化增强了 visual sensitivity，但没有产生缺失的 OCR alignment。
