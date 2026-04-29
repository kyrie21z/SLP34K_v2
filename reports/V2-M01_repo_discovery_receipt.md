# V2-M01 仓库梳理回执

## 1. 仓库状态

- 仓库根目录：`/mnt/data/zyx/SLP34K_v2`
- 当前分支：`main`
- 最新提交：`bc2b68e Ignore Codex config`
- `git status`：不干净。开始梳理时已有以下本地修改：
  - `ocr_training/strhub/models/maevit_infonce_plm/system.py`
  - `ocr_training/strhub/models/maevit_plm/system.py`
- 现有本地 diff 仅涉及两个 `system.py` 中的 `torch.load(..., weights_only=False)` 和文件末尾换行规范化。本次任务没有回退或编辑这些模型文件。
- `repo_file_index_depth3.txt`：已生成，共 55 行。
- 基础产物目录：已确认或创建 `reports/`、`outputs/`、`logs/`、`docs/`。

## 2. 核心文件地图

| 角色 | 路径 | 状态 | 作用 |
|---|---|---:|---|
| 训练入口 | `ocr_training/train.py` | 存在 | Hydra 训练入口。加载配置，解析 data root，实例化 `config.model` 和 `config.data`，构建 Lightning `Trainer`，然后调用 `trainer.fit(..., ckpt_path=config.ckpt_path)`。 |
| 测试入口 | `ocr_training/test.py` | 存在 | 当前评估入口。通过 `strhub.models.utils.load_from_checkpoint` 加载 checkpoint，构建 `SceneTextDataModule`，评估指定 benchmark subset，并写出 `${checkpoint}.log.txt`。 |
| unified evaluation 入口 | `ocr_training/evaluation/evaluate_unified.py` | 缺失 | 当前仓库不存在 `ocr_training/evaluation/` 目录。最接近的等价入口是 `ocr_training/test.py`；其中会把 `unified_lmdb` 作为 SLP34K subset 之一评估。 |
| 主配置 | `ocr_training/configs/main.yaml` | 存在 | Hydra 默认配置选择 `model: maevit_infonce_plm`、`charset: SLP34K_568`、`dataset: SLP34K`；定义 image size、max label length、batch size、trainer 默认值、输出目录和 `ckpt_path`。 |
| 模型配置 | `ocr_training/configs/model/maevit_infonce_plm.yaml` | 存在 | 定义 `_target_: strhub.models.maevit_infonce_plm.system.Model`、MAE 预训练路径、decoder 深度/头数、PARSeq permutation 配置、AR decode、refinement 和学习率倍率。 |
| 数据集配置 | `ocr_training/configs/dataset/SLP34K.yaml` | 存在 | 设置 `train_dir: SLP34K_lmdb_train`、`val_dir: SLP34K_lmdb_test`、`test_dir: SLP34K_lmdb_benchmark`。 |
| 模型系统文件 | `ocr_training/strhub/models/maevit_infonce_plm/system.py` | 存在 | 当前模型定义。用户给出的 `strhub/models/maevit_infonce_plm/system.py` 在 repo 根目录下不存在；本仓库的 `strhub` 位于 `ocr_training/` 内。 |

## 3. 当前 Baseline 架构

- 模型类：`strhub.models.maevit_infonce_plm.system.Model`，继承自 `CrossEntropySystem`。
- Encoder：在 `Model.__init__` 中从 `strhub.models.models_mae` 初始化。默认 SLP 配置 `img_size=[224,224]` 时选择 `mae_vit_base_patch16_224x224`，从 `mae_pretrained_path` 加载权重，然后赋值为 `self.encoder = mae_model`。
- Encoder feature shape：默认 224x224、patch size 16，对应 14 x 14 个图像 patch，加上 CLS token，因此 `memory` 为 `[B, 197, 768]`。如果走 32x128 分支，则为 8 x 32 个 patch 加 CLS，即 `[B, 257, D]`。
- 辅助文本分支：`clip.load("ViT-B/16")` 加载 CLIP，删除 visual tower，冻结 CLIP text 模块。训练时在 `mae_memory[:,0,:] @ self.proj` 与 CLIP text feature 之间计算 InfoNCE。
- Decoder：PARSeq/XLNet 风格 two-stream Transformer decoder，由 `ocr_training/strhub/models/maevit_infonce_plm/modules.py` 中的 `DecoderLayer` 和 `Decoder` 构成。
- `decode()` 输入：`tgt` token id `[B, T]`，视觉 `memory` `[B, S, 768]`，以及可选 attention/padding/query mask。
- `decode()` 输出：decoder hidden states `[B, T, 768]`。它返回的是 decoder feature，不是 logits；logits 由 `self.head` 生成。
- Head/classifier：`self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)`。对 SLP34K_568，tokenizer 包含 568 个字符以及 EOS/BOS/PAD，head 输出 `568 + 1 = 569` 类，即 charset 加 EOS，不包含 BOS/PAD。
- Token embedding：`self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)`，输入空间包含 charset、EOS、BOS、PAD。
- Forward path：`images -> encode(images) -> memory -> AR decode loop 或 NAR branch -> decode(...) -> head(...) -> logits`。默认配置为 `decode_ar: true` 且 `refine_iters: 1`。
- 推理 token/logit shape：默认 `max_label_length=50`，因此 `num_steps=51`，包含 EOS。`forward()` 返回 logits `[B, <=51, 569]`；测试时 AR early stop 可能缩短序列长度。
- 训练 path：`training_step` 将 labels encode 为 `[B, label_len + 2]`，构造 `tgt_in=tgt[:,:-1]` 和 `tgt_out=tgt[:,1:]`，生成 permutation mask，调用 `decode(tgt_in, memory, ...)`，再经过 `head` 和 CE loss，最终加上 `0.1 * cross_modal_loss`。
- Charset 来源：Hydra 默认 `charset: SLP34K_568` 会加载 `ocr_training/configs/charset/SLP34K_568.yaml`；`train.py` 还支持将 `config.model.charset_train` 作为文件路径读取。
- Max label length 来源：`ocr_training/configs/main.yaml` 设置 `model.max_label_length: 50`；data module 通过 `${model.max_label_length}` 继承该值，并过滤超过该长度的标签。
- Checkpoint loading：
  - MAE encoder 预训练权重在 `Model.__init__` 中通过 `mae_pretrained_path` 加载：`torch.load(..., map_location='cpu', weights_only=False)`，然后 `mae_model.load_state_dict(checkpoint['model'], strict=False)`。
  - 训练恢复通过 `config.ckpt_path` 传给 `trainer.fit`。
  - 测试通过 `strhub.models.utils.load_from_checkpoint`；普通 checkpoint path 会按字符串匹配选择模型类，然后调用 `ModelClass.load_from_checkpoint(..., strict=False, **kwargs)`。

## 4. 评估协议

- 评估脚本：当前 repo 使用 `ocr_training/test.py`；`ocr_training/evaluation/evaluate_unified.py` 不存在。
- 推荐运行目录：README 示例先执行 `cd ./ocr_training` 再运行 `./test.py`，因为默认 `--data_root data` 和 SLP subset discovery 都相对于 `ocr_training`。
- Checkpoint 输入：位置参数 `checkpoint`，例如 `checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`。
- 测试数据路径：从 `ocr_training` 目录运行时，默认路径为 `data/test/SLP34K_lmdb_benchmark`。
- 当前 checkout 中发现的 SLP34K subsets：`IV_lmdb`、`OOV_lmdb`、`low_lmdb`、`multi-lines_lmdb`、`normal_lmdb`、`single-line_lmdb`、`unified_lmdb`、`vertical_lmdb`。
- Unified benchmark size：`unified_lmdb` 的 LMDB 元数据包含 `num-samples=6884`，因此它对应 6884 unified benchmark subset。`--test_data SLP34K` 会评估所有 SLP subset 目录，不只是 `unified_lmdb`。
- 模型加载：`test.py` 在 `--test_data SLP34K` 时硬编码 SLP charset，并注入 SLP MAE pretrained path，然后调用 `load_from_checkpoint(...).eval().to(device)`。
- 输出指标：每个 dataset row 输出 `# samples`、`Accuracy`、`1 - NED`、`Confidence`、`Label Length`，并给出加权 `Combined` row。
- 指标定义：
  - accuracy：charset adaptation 后的 exact string match。
  - NED：`edit_distance(pred, gt) / max(len(pred), len(gt))`，报告为 `100 * (1 - mean_ned)`。
  - confidence：decoded token probabilities 的乘积，取平均后按百分比报告。
  - label length：预测文本长度的平均值。
- 输出产物：仅写出 `${checkpoint}.log.txt`。`test.py` 不生成 CSV/JSON。
- `pred_vs_gt_alignment.jsonl`：当前 repo 不生成。
- replace/insert/delete breakdown：当前 repo 不生成。
- hard/OOV/long_21+/single subset breakdown：
  - OOV 和 single-line 作为 LMDB subset 名称存在。
  - 未找到 `hard` 和 `long_21+` breakdown。
  - 除 LMDB 目录级 row 外，没有更细粒度 subset report。
- 与 baseline 对齐比较方法：从 `ocr_training` 目录使用同一 `test.py` 命令、相同 `--test_data SLP34K --test_dir SLP34K_lmdb_benchmark`、相同 charset、相同 image size/max length 评估 SLP34K_v2，并以 `unified_lmdb` row 作为 6884 benchmark 的主对齐指标。保留各 subset row 用于诊断。除非历史 baseline 明确使用 `Combined`，否则不应把 `Combined` 作为 unified 主指标。

## 5. MDiff Decoder 候选插入点

- 推荐插入位置：位于 `encode(images)` 之后、`head` 之前，用 MDiff denoising decoder 替换当前 PARSeq `decode()` 行为，并让其在 visual memory 条件下对 masked token states 去噪。
- 推荐文件策略：新建 model package/file，而不是在现有 baseline `system.py` 中加分支。这样可以冻结强 baseline，并让 Hydra 模型选择显式化。
- 建议的新模型 target：`ocr_training/strhub/models/slp_mdiff/system.py`，配置中使用 `_target_: strhub.models.slp_mdiff.system.Model`。
- Encoder 复用：可以复用。MAE encoder 和 `mae_pretrained_path` loading 逻辑可以直接沿用。
- Visual adapter：如果 MDiff hidden dim 仍为 768，则技术上不是必须，因为 encoder memory 已经是 `[B, S, 768]`。但建议保留一个显式 adapter hook（`Identity` 或 `Linear + LayerNorm`），因为目标结构中包含 `visual adapter`，也方便后续维度变化。
- Head 复用：如果 MDiff hidden dim 为 768，且输出仍为 charset 加 EOS（SLP34K_568 下为 569 类），则 classifier 形状可以复用。是否直接复用现有 checkpoint 中的 `head` 权重，需要另行确定；结构上兼容。
- Label encode：现有 `Tokenizer` 可以提供 charset/EOS/BOS/PAD id，但 MDiff 需要 mask-token 语义。当前 tokenizer 没有 `[MASK]`，V2-M02 应增加 MDiff 专用 noising/encoding 逻辑，或小范围扩展 tokenizer，加入 input-only mask id。
- Loss：需要新增 denoising CE loss。最小版本应对 masked positions 或 all non-pad positions 计算 clean target CE，并明确 masking 口径。
- Inference：需要新增 full-mask iterative decoding。当前 AR loop 和 refinement mask 是 PARSeq 专用，不适合作为 MDiff 推理算法复用。
- 必须新增的配置项：
  - 新 `model` config 名称和 `_target_`
  - `mdiff_depth`、`mdiff_num_heads`、`mdiff_mlp_ratio`、`mdiff_dropout`
  - `mask_ratio` 或 mask schedule，以及 full-mask inference 选项
  - `denoise_steps` / inference steps
  - `visual_adapter` 相关设置
  - `max_label_length: 50`，charset 继续继承 SLP34K_568
  - 如果保留 CLIP auxiliary，则增加可选 `use_infonce_aux` / `infonce_weight`
- 最大风险点：
  - tokenizer/index 对齐，尤其 input-only `[MASK]` 与输出类别排除 BOS/PAD 的关系。
  - checkpoint loading：`strhub.models.utils._get_model_class` 当前只识别 `maevit_infonce_plm` 和 `maevit_plm`。
  - 当前评估脚本没有细粒度 artifact，因此 baseline 对比能力有限，除非新增独立 evaluator。
  - full-mask iterative decoding 中的 EOS/长度处理。
  - V2-M02 是否保留当前 CLIP InfoNCE auxiliary，还是只隔离 decoder replacement 与 denoising loss。

## 6. V2-M02 最小实现计划

1. 新增 model package：
   - `ocr_training/strhub/models/slp_mdiff/__init__.py`
   - `ocr_training/strhub/models/slp_mdiff/system.py`
   - `ocr_training/strhub/models/slp_mdiff/modules.py`

2. 新增配置：
   - `ocr_training/configs/model/slp_mdiff.yaml`
   - 可选 `ocr_training/configs/experiment/slp_mdiff.yaml`

3. 只修改训练/评估必要的模型选择入口：
   - 修改 `ocr_training/strhub/models/utils.py`，使 `load_from_checkpoint` 能解析 `slp_mdiff` checkpoint。
   - 在明确切换实验前，不修改 `ocr_training/configs/main.yaml` 默认模型。

4. 训练目标：
   - 复用 MAE visual encoder 和 SLP charset/max length。
   - 构造带 EOS/PAD 的 clean token targets。
   - 实现 random mask / full mask noising。
   - 在 encoder memory 条件下，从 masked inputs 预测 clean tokens。
   - 对 non-pad 或 masked positions 优化基础 denoising CE。

5. 推理目标：
   - 从长度 `max_label_length + 1` 的 full-mask sequence 开始。
   - 执行配置化的 iterative denoising steps。
   - 对 EOS 加 charset 的 logits 使用现有 tokenizer decode 规则得到预测文本。

6. 最小验收标准：
   - Hydra 可以实例化 `model=slp_mdiff`。
   - dummy forward 输入 `[B,3,224,224]` 时返回 `[B,51,569]`。
   - 小 real batch 或 synthetic batch 上一次 `training_step` 能跑通，无 shape/index 错误。
   - `ocr_training/test.py` 可以加载 SLP-MDiff checkpoint，并输出 `unified_lmdb` row。
   - baseline 文件 `maevit_infonce_plm/system.py` 和 `maevit_plm/system.py` 保持行为不变，除非用户明确要求修改。

## 7. 不确定项与待确认问题

- V2-M02 是否保留当前 CLIP InfoNCE auxiliary（`0.1 * cross_modal_loss`）以维持 baseline 连续性，还是去掉它以隔离 MDiff decoder replacement？
- 主 baseline 对比是否严格使用 6884 样本的 `unified_lmdb` row，还是也需要保留历史报告中的 `Combined` row？
- MDiff decoder hidden dim 是否固定为 768 以直接兼容 encoder/head，还是希望通过 adapter 接入不同 decoder width？
- Plain V2-M02 初始使用多少个 denoising inference steps？
- 是否需要新增 unified evaluator 来写出 `pred_vs_gt_alignment.jsonl` 和 edit-operation breakdown，还是 V2-M02 先只复用当前 `test.py`？
- 当前 `strhub.models.utils._get_config()` 引用了缺失的 `configs/charset/94_full.yaml`。这主要影响 `pretrained=<model_id>` 创建路径，不影响普通 checkpoint loading；如果后续需要 pretrained alias，应修复该路径。
