# V2-M02s MDiff Corrector Design

说明：本报告正文使用中文；英文仅保留在用户指定的标题/章节名、代码标识、配置项和必要术语中。当前阶段未训练 MDiff corrector，未修改 `maevit_infonce_plm/system.py` 行为，未修改默认 `main.yaml`，未删除数据，未进入 V2-M03。`ocr_training/evaluation/` 与 `ocr_training/evaluation/analyze_m05_alignment.py` 在当前仓库中不存在，以下统一标记为 `missing`。

## 1. Executive Summary

- **为什么从 direct replacement 改为 corrector：** V2-M02r 已经说明，SLP34K baseline 的 MAE/InfoNCE encoder memory 虽然对原 PARSeq/AR decoder 可读，但不具备直接支撑 MDiff full-mask decoder 的 character-aligned visual memory。继续做 full-mask replacement 的实验价值低。
- **corrector 的核心假设：** baseline 已经学会了 `image -> visual memory -> aligned decoder hidden -> token prediction`。因此比起让 MDiff 从 `[MASK] * T` 直接生成整串，更合理的是让 MDiff 接收 baseline prediction 及其 decoder hidden，在局部位置上做保守修正。
- **推荐 contract：** 优先 `Contract 3: Token + PARSeq decoder hidden corrector`；次选 `Contract 2: Token + encoder memory corrector`；`Contract 1: Token-only corrector` 只建议作为最小对照组。
- **推荐训练数据：** 两阶段。Stage 1 用 GT 构造 local synthetic noise 做 correction pretraining；Stage 2 用 frozen PARSeq 预测缓存做真实错误分布修正训练。第一版只处理 replace-aligned errors，不处理 insert/delete。
- **推荐 loss：** `selected-position CE + preservation CE`。即对选中的错误位置计算修正 CE，同时对原本正确位置施加较弱 preservation 约束，防止把正确 token 改错。
- **推荐 inference rule：** 只允许修改 selected positions；corrector 置信度不足不改；原 PARSeq 置信度高且 gain 不明显不改；EOS/长度先不轻易改；第一版只处理 replace。
- **推荐下一阶段 V2-M02t：** 不改原 baseline 行为；新增离线 export 工具和 corrector package；先导出 prediction cache；先做 token/decoder-hidden corrector；先只支持 replace；只做小样本 overfit/smoke 验证。
- **是否需要训练：** 当前 V2-M02s 不需要训练。V2-M02t 需要极小规模 corrector 训练或 overfit check，但不是长训练。
- **是否建议进入 V2-M03：** 不建议。先完成 V2-M02t 的 corrector 最小验证。

## 2. Baseline Export Feasibility

当前 baseline inference 主路径：

`ocr_training/test.py:99-124 -> model.test_step() -> strhub/models/base.py:100-131 _eval_step() -> model.forward() -> maevit_infonce_plm/system.py:294-362 -> tokenizer.decode()`

`slpr_ocr` 环境下的 1-sample smoke check 已验证：

- `logits_shape = (1, 51, 569)`
- `memory_shape = (1, 197, 768)`
- 手工重放 AR loop 可拼出 `decoder_hidden_shape = (1, 18, 768)`
- LMDB `meta-*` 实际键名为：`id`, `quality`, `resolution_type`, `structure`, `structure_type`, `vocabulary_type`

### 2.1 Export Table

| Field | Available Now | Source File / Function | Need Code Change | Notes |
|---|---|---|---|---|
| `sample_id` | No | `strhub/data/dataset.py:120-136` | Yes | 当前 `__getitem__()` 只返回 `(img, label)`，不返回 filtered LMDB index。 |
| `image_path` | No | `strhub/data/dataset.py:61-67, 120-136` | Yes | LMDB 样本没有单图路径；可导出 `lmdb_dir + lmdb_index` 作为稳定定位。 |
| `lmdb_index` | Indirect only | `strhub/data/dataset.py:64-67, 124-125` | Yes | 内部有 `filtered_index_list`，但未随 batch 暴露。 |
| `subset_name` | Yes | `test.py:117-134`, `data/module.py:104-112` | No | dataloader 外层循环已知道 subset。 |
| `gt_text` | Yes | `base.py:100-131` | No | batch label 已提供。 |
| `pred_text` | Yes | `base.py:120-124`, `data/utils.py:79-99` | No | `tokenizer.decode(probs)` 直接得到。 |
| `is_correct` | Indirect only | `base.py:122-129` | No for exporter; Yes for current `test.py` output | 逻辑现成，但当前只聚合到 batch 级。新 exporter 直接逐样本判断即可。 |
| `layout/difficulty/...` metadata | No in current loader | `data/dataset.py:120-136`, `test.py:117-134` | Yes | 当前代码不读取 `meta-*`；根据既有仓库 discovery，LMDB meta 至少含 `quality`, `structure`, `vocabulary_type`, `resolution_type`, `structure_type`。 |
| `pred_token_ids` | Indirect only | `base.py:120-121`, `data/utils.py:92-98` | No for exporter | `probs.argmax(-1)` 可得 raw ids；EOS 截断需复用 tokenizer 过滤逻辑。 |
| `gt_token_ids` | Yes | `data/utils.py:113-116` | No | `tokenizer.encode(labels)` 直接得到。 |
| `pred_token_confidence` | Indirect only | `data/utils.py:92-98, 118-127` | No for exporter | `probs.max(-1)` 后按 EOS 截断即可；当前只聚合成 sequence confidence product。 |
| `pred_token_logits [T,C]` | Yes | `base.py:117-121`, `maevit_infonce_plm/system.py:294-362` | No | `forward()` 已返回 `logits`。 |
| `topk logits` | No explicit API | same as above | No for exporter | 可从 `logits.topk(k)` 派生。 |
| `eos_position` | Indirect only | `Tokenizer._filter()` at `data/utils.py:118-127` | No for exporter | 从 raw top1 ids 首个 `eos_id` 推出。 |
| `valid_length` | Indirect only | same as above | No for exporter | `len(pred_text)` 或 `eos_position` 可得。 |
| `edit alignment` | No | N/A | Yes, but exporter-side only | 当前 repo 没有 alignment 导出函数；建议在新 exporter 中新增 Levenshtein op trace。 |
| `encoder_memory [S,D]` | Yes | `maevit_infonce_plm/system.py:265-277, 301` | No for exporter | `model.encode(images)` 直接可得；当前 `forward()` 不返回它。 |
| `decoder_hidden [T,D]` | Not directly returned | `maevit_infonce_plm/system.py:320-328, 357-360, 457-458` | Yes, non-invasive hook | 训练时 `out = self.decode(...)` 已存在；推理时每步 `tgt_out` 只在局部变量。 |
| `cross-attn weights` | Not returned | `maevit_infonce_plm/modules.py:124-145` | Yes | `forward_stream()` 已拿到 `ca_weights`，但当前上层丢弃。V2-M02t 非必需。 |

### 2.2 直接结论

1. **当前 baseline 能直接拿到：**
   - `gt_text`
   - `pred_text`
   - `logits [T,C]`
   - `gt_token_ids`
   - `encoder_memory [S,D]`
   - subset 名称
2. **当前 baseline 不能直接拿到但可由 exporter 外围派生：**
   - `pred_token_ids`
   - `pred_token_confidence`
   - `eos_position`
   - `valid_length`
   - `is_correct`
   - `topk logits`
3. **当前 baseline 没有现成接口，需要新增 non-invasive hook 或专用 exporter：**
   - `sample_id / lmdb_index / meta-*`
   - `decoder_hidden [T,D]`
   - `edit alignment`
   - 若未来需要，`cross-attn weights`

### 2.3 Recommended Hook Points

不建议改 `test.py` 的现有评估行为。建议新增独立 exporter，并只在 exporter 路径下增加以下 non-invasive hook：

1. `ocr_training/strhub/models/maevit_infonce_plm/system.py`
   - 新增 `forward_with_aux(images, max_length=None, return_memory=False, return_hidden=False, return_token_ids=False)`。
   - 保持现有 `forward()` 返回 `logits` 不变，内部可复用 `forward_with_aux(...)[\"logits\"]`。
   - 在 AR loop 的 `tgt_out = self.decode(...)` 处收集每步 hidden；在 refine 分支中收集最终 `tgt_out`。
2. `ocr_training/strhub/data/dataset.py`
   - 不改原 `__getitem__()` 默认返回值。
   - 新增 exporter 专用 dataset wrapper，或给 `LmdbDataset` 增加 `return_meta=False` 开关，仅在 exporter 中返回 `lmdb_index`、`root`、`meta_json`。
3. 新增 exporter-side helper
   - `decode_with_ids(logits, tokenizer)`：同时返回 `pred_text`、`pred_token_ids`、`pred_token_confidence`、`eos_position`、`valid_length`。
   - `align_pred_gt(pred_ids, gt_ids)`：返回 replace/insert/delete/eos_early/eos_late 对齐标签。

### 2.4 Cache Size and Format Recommendation

按 SLP34K baseline 常见形状估算，若使用 `fp16`：

- `encoder_memory [197,768]`: `197 * 768 * 2 bytes ≈ 296 KB / sample`
- `decoder_hidden [51,768]`: `51 * 768 * 2 bytes ≈ 76 KB / sample`
- `logits [51,569]`: `51 * 569 * 2 bytes ≈ 57 KB / sample`
- 三者全存约 `429 KB / sample`
- 若 train set 约 `27.5k` 样本，全量缓存约 `11.5 GB`，再加索引和 JSON 开销会更高

因此建议：

- **样本级 manifest：** `jsonl`
  - 适合 `sample_id / subset / lmdb_dir / lmdb_index / gt_text / pred_text / alignment / metadata / feature pointer`
- **稠密张量：** `npz` 分 shard 存 `fp16/int16`
  - 推荐用于 `pred_token_ids / pred_token_conf / topk logits / decoder_hidden / encoder_memory`
- **不推荐：**
  - 纯 `jsonl` 存 dense arrays：太大
  - 纯 `lmdb` 重复塞 arrays：调试不便
  - 无压缩全量 `pt` 单文件：单文件过大，增量处理差

### 2.5 Minimal Proposed Cache Schema

#### `manifest.jsonl`

```json
{
  "sample_id": "unified_lmdb:000012345",
  "subset": "unified_lmdb",
  "lmdb_dir": "data/test/SLP34K_lmdb_benchmark/unified_lmdb",
  "lmdb_index": 12345,
  "gt_text": "苏宿货123",
  "pred_text": "苏客货123",
  "is_correct": false,
  "valid_length": 6,
  "eos_position": 6,
  "metadata": {
    "quality": "middle",
    "structure": "vertical",
    "vocabulary_type": "OOV",
    "resolution_type": "low",
    "structure_type": "vertical"
  },
  "alignment_ops": ["correct", "replace", "correct", "correct", "correct", "correct"],
  "feature_shard": "features_0003.npz",
  "feature_index": 217
}
```

#### `features_0003.npz`

推荐字段：

- `pred_token_ids`: `[N,T]`, `int16`
- `gt_token_ids`: `[N,T]`, `int16`
- `pred_token_conf`: `[N,T]`, `float16`
- `topk_indices`: `[N,T,K]`, `int16`
- `topk_values`: `[N,T,K]`, `float16`
- `decoder_hidden`: `[N,T,D]`, `float16`, optional
- `encoder_memory`: `[N,S,D]`, `float16`, optional
- `valid_length`: `[N]`, `int16`
- `eos_position`: `[N]`, `int16`

V2-M02t 建议默认：

- 必存：`pred_token_ids`, `gt_token_ids`, `pred_token_conf`, `valid_length`, `eos_position`
- 推荐存：`topk_indices`, `topk_values`, `decoder_hidden`
- 可选存：`encoder_memory`
- **不建议默认缓存全量 logits**；更推荐 top-k logits，或按需开关导出。

## 3. Corrector Contract Options

### 3.1 Comparison

| Contract | Inputs | Pros | Cons | Recommendation |
|---|---|---|---|---|
| Contract 1: Token-only corrector | `pred_token_ids`, confidence embedding, correction mask | 最简单；不依赖 visual feature export；适合做“是不是纯 LM”下限对照 | 极易退化为语言模型；无法证明视觉证据有效 | 仅作 baseline/control，不建议作为主线 |
| Contract 2: Token + encoder memory corrector | `pred_token_ids`, confidence embedding, correction mask, `encoder_memory [B,197,768]` | 仍然保留图像证据；结构比 full replacement 更容易 | 直接读 encoder memory 的难点仍在；corrector 可能仍读不懂 `[197,768]` | 次选 |
| Contract 3: Token + PARSeq decoder hidden corrector | `pred_token_ids`, confidence embedding, correction mask, `decoder_hidden [B,T,D]`, optional `encoder_memory` | 复用 baseline 已学好的 visual-to-sequence alignment；输入最贴近修正任务；最符合“局部 refinement”定位 | 依赖 hidden export；研究上更像 refinement head | **首选** |

### 3.2 Recommended Contract

推荐 `Contract 3: Token + PARSeq decoder hidden corrector` 作为 V2-M02t 最小实现。

原因：

1. `decoder_hidden` 比 `encoder_memory` 更接近“当前 token 位置正在看什么”的表示。
2. V2-M02r 的失败根因就在于直接从 `encoder_memory` 做 full-mask generation 太难；而 corrector 的目的是**降低接口难度**。
3. 如果 corrector 只做 replace 修正，`pred_token_ids + confidence + decoder_hidden` 已足够构成一个清晰、可解释、保守的 contract。

### 3.3 Minimal Input Set for V2-M02t

建议输入：

- `pred_token_ids [B,T]`
- `pred_token_conf [B,T]`
- `correction_mask [B,T]`
- `decoder_hidden [B,T,D]`
- optional: `encoder_memory [B,S,D]`
- optional: `topk token candidates [B,T,K]`

第一版不建议强依赖：

- full `encoder_memory`
- EOS/length editing
- insert/delete action head

## 4. Correction Mask Design

### 4.1 Oracle Mask

定义：

- 训练时基于 `pred` 与 `GT` 的 alignment，只 mask 错误位置

用途：

- 上限分析
- 对齐简洁的 replace-only 训练

适用性：

- 训练：适用
- 推理：不适用

结论：

- V2-M02t 可以把它作为离线 supervision mask 和 upper-bound evaluator，但不能作为真实 inference mask。

### 4.2 Low-confidence Mask

定义：

- `mask = pred_token_conf < tau_low`

用途：

- 真实 inference
- 最简单的 deployable rule

适用性：

- 训练：适用，可与 oracle mask 混合
- 推理：适用

结论：

- V2-M02t 的默认 inference mask 应该从这里开始。

### 4.3 Confusion-aware Mask

定义：

- 优先 mask 视觉混淆字符或 SLP 常见错位字符
- 例如 `6/8`, `9/8`, `1/0`, alphabet-digit confusion，中文近形字

用途：

- 把 corrector 训练集中在“值得修”的位置
- 增强 SLP-specific error focus

适用性：

- 训练：适用
- 推理：适用，但建议作为 low-confidence 的补充，而不是单独规则

结论：

- 更适合在 Stage 2 cache 统计真实 confusion matrix 后再细化；V2-M02t 先保留接口，不必先实现大表。

### 4.4 Synthetic Noise Mask

定义：

- 从 GT 构造局部 noised sequence，并记录 noised positions
- 噪声类型：
  - random replacement
  - visual-confusion replacement
  - segment-preserving replacement

用途：

- corrector 预训练
- 不依赖 baseline prediction cache

适用性：

- 训练：强适用
- 推理：不适用

结论：

- Stage 1 应以 synthetic noise mask 为主，但**不要从 full mask 开始**，只做 local noise。

### 4.5 Recommended Usage Matrix

| Mask Type | Stage 1 Synthetic Pretraining | Stage 2 Baseline Correction Training | Real Inference |
|---|---|---|---|
| Oracle | Optional upper bound | Yes, as supervision mask | No |
| Low-confidence | Simulated only | Yes | Yes |
| Confusion-aware | Yes | Yes | Yes, as supplement |
| Synthetic noise | Yes | No | No |

## 5. Training Data Design

### 5.1 Stage 1: Synthetic Correction Pretraining

构造：

- `clean = GT`
- `noised = apply_local_noise(GT)`
- `target = GT`
- `mask = noised_positions`

推荐噪声：

1. random replacement
2. visual-confusion replacement
3. segment-preserving replacement
   - 中文位替换成中文
   - 数字位替换成数字
   - 字母位替换成字母

不建议：

- full mask
- 大量 deletion/insertion
- 复杂 TRN/LC/BLC

第一阶段目标不是拟合真实 baseline error，而是让 corrector 学会：

- 看上下文修 token
- 在 mask 外保持稳定
- 不把整串全部重写

### 5.2 Stage 2: PARSeq Prediction Correction

构造：

- 对 frozen baseline 跑 train set inference，导出：
  - `pred_text`
  - `pred_token_ids`
  - `pred_token_conf`
  - `decoder_hidden`
  - optional `encoder_memory`
  - alignment ops
- 训练时：
  - `input = pred sequence + confidence + hidden + mask`
  - `target = GT aligned tokens`

### 5.3 Length Mismatch Handling

V2-M02t 第一版建议：

1. **只保留 replace-aligned samples 或 replace-dominant positions**
   - 若 `pred_len == gt_len`，直接 position-wise 对齐
   - 若存在 insert/delete，先记为 unsupported，不进入第一版训练主集
2. **对轻微长度不一致样本，可先做 alignment 再截取 replace positions**
   - 但第一版 inference 仍不允许 insert/delete action

理由：

- 这能把 corrector 问题收缩成“局部分类修正”，避免过早把任务升级成 sequence edit transducer。

### 5.4 Insert/Delete Handling

当前建议：

- Stage 1：默认不做
- Stage 2：统计但不训练主线
- Inference：暂缓

可记录 alignment label：

- `correct`
- `replace`
- `insert`
- `delete`
- `eos_early`
- `eos_late`

但 V2-M02t 只在 `replace` 上更新。

### 5.5 Replace-only First Policy

明确建议：

1. 第一阶段只做 replace-aligned samples 是合理的
2. 第一版优先处理 replace errors
3. 插删和长度修正留到后续版本

### 5.6 Correct-token Preservation

建议保留 preservation supervision。

因为 corrector 最大风险不是“修不动”，而是“把正确 token 改错”。因此即使 V2-M02t 只做 replace 训练，也应在 loss 中显式鼓励：

- mask 外 token 保持原本正确状态
- 高置信位置不要乱改

## 6. Loss Design

### 6.1 Loss 1: selected-position CE

公式：

`L_corr = CE(logits[mask], gt[mask])`

优点：

- 最简单

风险：

- 对非 mask 位置没有保护
- 容易出现“修错一处，破坏多处”

### 6.2 Loss 2: selected CE + preservation CE

推荐公式：

`L = CE(logits[selected_mask], gt[selected_mask]) + lambda_pres * CE(logits[preserve_mask], gt[preserve_mask])`

其中：

- `selected_mask`: 需要修正的位置
- `preserve_mask`: 当前正确或高置信保留的位置
- `lambda_pres`: 小于 1，例如 `0.1 ~ 0.3`

优点：

- 直接约束“不要把对的改错”
- 与 conservative inference 目标一致

风险：

- `lambda_pres` 太大时会抑制修正能力

### 6.3 Loss 3: conservative update loss

思路：

- 只在 `corrector_conf - baseline_conf > margin` 时鼓励修改

优点：

- 更贴合部署需求

风险：

- 当前阶段过早复杂化
- 需要更精细的 baseline vs corrector score 对齐

### 6.4 Recommendation

V2-M02t 推荐 **Loss 2: selected CE + preservation CE**。

原因：

1. 比 `selected-position CE` 更稳
2. 比 margin-based conservative loss 更简单
3. 直接服务于核心目标：**降低 harmful change**

## 7. Inference Design

### 7.1 Conservative Rule

第一版 corrector inference 规则建议：

1. 只允许修改 `selected positions`
2. 若 `corrector_conf < tau_corr`，不修改
3. 若 `baseline_conf >= tau_keep` 且 `corrector_gain < delta_gain`，不修改
4. EOS/长度默认不修改
5. 若 `corrector_top1 == baseline_token`，视为 keep
6. 第一版只处理 replace，不处理 insert/delete

### 7.2 Suggested Decision Logic

对每个可改位置 `t`：

- `base_id = pred_token_ids[t]`
- `base_conf = pred_token_conf[t]`
- `corr_id = argmax(corrector_logits[t])`
- `corr_conf = max(softmax(corrector_logits[t]))`
- `gain = corr_conf - base_conf`

决策：

```text
if t not in selected_mask:
    keep
elif corr_id == base_id:
    keep
elif corr_conf < tau_corr:
    keep
elif base_conf >= tau_keep and gain < delta_gain:
    keep
else:
    change to corr_id
```

### 7.3 Output Contract

inference 输出建议包含：

- `corrected_pred`
- `changed_positions`
- `baseline_pred`
- `correction_confidence`
- `keep_or_change` per position
- `selected_mask`

这样评估时可以精确统计：

- 改了多少
- 改对了多少
- 误改了多少

## 8. Evaluation Design

不要只看 overall accuracy。corrector 必须做 correction-specific evaluation。

### 8.1 Sample-level Metrics

设：

- `S_wrong`: baseline 原本预测错误的样本集合
- `S_right`: baseline 原本预测正确的样本集合
- `S_changed`: 被 corrector 改动过的样本集合

定义：

- `baseline_accuracy = # baseline_correct_samples / # all_samples`
- `corrected_accuracy = # corrected_correct_samples / # all_samples`
- `harmful_change_rate = # {s in S_right | corrected_pred != GT} / # S_right`

补充：

- `changed_sample_rate = # S_changed / # all_samples`
- `gain_accuracy = corrected_accuracy - baseline_accuracy`

### 8.2 Token-level Metrics

设：

- `P_wrong`: baseline 原本错误的位置集合
- `P_right`: baseline 原本正确的位置集合
- `P_changed`: 被 corrector 修改的位置集合

定义：

- `correction_rate = # {p in P_wrong | corrected_token == GT_token} / # P_wrong`
- `preservation_rate = # {p in P_right | corrected_token == GT_token} / # P_right`
- `changed_token_count = # P_changed`
- `replace_error_reduction = (# baseline_replace_errors - # corrected_replace_errors) / # baseline_replace_errors`

### 8.3 Confidence-aware Metrics

对 low-confidence 掩码区域：

- `low_conf_precision = # changed_low_conf_positions_corrected / # changed_low_conf_positions`
- `low_conf_recall = # changed_low_conf_positions_corrected / # baseline_wrong_low_conf_positions`

对 high-confidence 错误：

- `high_conf_wrong_correction_rate = # {baseline_wrong & high_conf & corrected_to_GT} / # {baseline_wrong & high_conf}`

### 8.4 Oracle@K

如果 corrector 输出 top-k 候选：

- `oracle@K = # {wrong positions where GT in topK(corrector_logits)} / # wrong positions`

这个指标可回答：

- 模型是不是“知道 GT 候选，只是 gating 太保守”

### 8.5 Recommended Full Metric List

- `baseline_accuracy`
- `corrected_accuracy`
- `gain_accuracy`
- `correction_rate`
- `preservation_rate`
- `harmful_change_rate`
- `replace_error_reduction`
- `high_conf_wrong_correction_rate`
- `low_conf_precision`
- `low_conf_recall`
- `oracle@1`
- `oracle@3`
- `oracle@5`
- `changed_token_count`
- `changed_sample_rate`

### 8.6 Subset Metrics

至少按以下切分统计：

- `unified`
- `OOV`
- `IV`
- `single-line`
- `multi-lines`
- `vertical`
- `low`
- `long_21+` if token length cache available
- `hard` if future exporter 读取 LMDB `quality/structure` 分组

### 8.7 Evaluation Notes

1. 当前 repo 的 `test.py` 只能输出 subset 聚合，不足以支撑 correction-specific metrics。
2. V2-M02t 需要新增 offline evaluator：
   - 读 manifest/cache
   - 读 corrector 输出
   - 统一按 sample/token/subset 统计

## 9. V2-M02t Minimal Implementation Plan

### 9.1 Scope

V2-M02t 的最小实现必须满足：

1. 不修改原 baseline 行为
2. 新建 corrector package 或在 `slp_mdiff` 下新增 `corrector` 分支
3. 先做 offline cache export
4. 先做 `GT synthetic noise + token/decoder-hidden corrector`
5. 先只处理 replace errors
6. 不处理 insert/delete
7. 不跑完整训练
8. 只做小样本 overfit 验证

### 9.2 Recommended File Plan

#### New files

- `ocr_training/strhub/models/slp_mdiff_corrector/__init__.py`
- `ocr_training/strhub/models/slp_mdiff_corrector/modules.py`
- `ocr_training/strhub/models/slp_mdiff_corrector/system.py`
- `ocr_training/configs/model/slp_mdiff_corrector.yaml`
- `ocr_training/tools/export_parseq_corrector_cache.py`
- `ocr_training/tools/train_mdiff_corrector_smoke.py`
- `ocr_training/tools/eval_mdiff_corrector_offline.py`

#### Modified files

- `ocr_training/strhub/models/maevit_infonce_plm/system.py`
  - 只新增 `forward_with_aux()` 或 exporter hook
  - 不改变 `forward()` 默认返回值和 baseline 逻辑
- 可选：`ocr_training/strhub/data/dataset.py`
  - 只加 exporter 专用 metadata/index 读取开关
  - 默认行为不变

#### Config files

- `ocr_training/configs/model/slp_mdiff_corrector.yaml`
  - `contract_type=token_decoder_hidden`
  - `use_encoder_memory=false` 默认
  - `max_label_length=50`
  - `loss_type=selected_plus_preservation`
  - `replace_only=true`
  - `cache_topk=8`

### 9.3 Minimal Commands

#### 1. Export baseline correction cache

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
python tools/export_parseq_corrector_cache.py \
  --checkpoint checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --split train \
  --subset unified_lmdb \
  --output_dir outputs/V2-M02t_corrector_cache_smoke \
  --batch_size 8 \
  --export_decoder_hidden true \
  --export_encoder_memory false \
  --topk 8 \
  --limit 64
```

#### 2. Synthetic-noise smoke train

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
python tools/train_mdiff_corrector_smoke.py \
  --config-name main \
  model=slp_mdiff_corrector \
  model.contract_type=token_decoder_hidden \
  model.replace_only=true \
  data.batch_size=8 \
  +trainer.max_steps=50 \
  +trainer.limit_val_batches=2 \
  run_tag=V2-M02t_corrector_smoke
```

#### 3. Offline correction evaluation

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
python tools/eval_mdiff_corrector_offline.py \
  --cache_dir outputs/V2-M02t_corrector_cache_smoke \
  --corrector_ckpt outputs/V2-M02t_corrector_smoke/checkpoints/last.ckpt \
  --subset unified_lmdb \
  --tau_low 0.70 \
  --tau_corr 0.80 \
  --tau_keep 0.90 \
  --delta_gain 0.05
```

### 9.4 Acceptance Criteria

V2-M02t 最低验收标准：

1. exporter 能稳定导出：
   - `pred_text`
   - `pred_token_ids`
   - `pred_token_conf`
   - `gt_token_ids`
   - `decoder_hidden`
   - `alignment_ops`
2. corrector smoke train 能在极小样本上明显 overfit
3. inference 只改 selected replace positions，不改 EOS/长度
4. offline evaluator 能输出 correction-specific metrics
5. 相对 baseline，极小样本上：
   - `correction_rate > 0`
   - `preservation_rate` 高于 `0.98` 的目标区间更合理
   - `harmful_change_rate` 明显低

## 10. Risks and Open Questions

1. **corrector 是否会过度依赖 PARSeq？**
   - 会。尤其在 Contract 3 下，corrector 可能只学会“在 baseline hidden 上做浅层再分类”。
   - 这不是当前阶段的阻塞，而是后续需要用 visual ablation 证明其不是纯 LM。
2. **decoder hidden 是否可导出？**
   - 代码上可导出。训练分支已有 `out = self.decode(...)`，推理 AR loop 里也有 `tgt_out` 局部张量，只是当前未返回。
   - 风险不在可导出性，而在接口设计要保持 non-invasive。
3. **corrector 是否会改错正确 token？**
   - 这是最大风险。
   - 因此必须在 loss 和 inference 上同时保守化，并显式统计 `preservation_rate` 和 `harmful_change_rate`。
4. **insert/delete 如何处理？**
   - 第一版不处理。
   - 否则 corrector 会从 token refinement 变成 sequence edit model，复杂度显著上升。
5. **visual memory 是否仍有作用？**
   - 有可能有，但第一优先级应低于 `decoder_hidden`。
   - 更合理的研究路径是把 `decoder_hidden` 作为主输入，把 `encoder_memory` 作为可选附加分支做 ablation。
6. **如何证明不是简单语言模型？**
   - 需要做 ablation：
     - token-only vs token+hidden
     - hidden real vs shuffled
     - hidden only vs hidden+memory
   - 如果 token-only 接近最好结果，则 corrector 创新性较弱。
7. **如何形成 SLP-specific innovation？**
   - 不应一上来堆复杂模块。
   - 更稳妥的路径是：
     - 先证明 `decoder_hidden corrector` 有效
     - 再加入 SLP-specific confusion-aware mask、segment-preserving noise、subset-aware evaluation
   - 这样创新链条更清楚：先建立 generic corrector，再加 SLP-specific prior。

## 11. Next Codex Prompt Draft

```text
你现在进入 SLP34K_v2 项目的 V2-M02t 阶段。

项目根目录：
/mnt/data/zyx/SLP34K_v2/

当前目标：
实现 “MDiff as PARSeq corrector” 的最小可运行版本，仅做 offline cache export + small-sample smoke validation。

严格边界：
1. 不修改原 maevit_infonce_plm baseline 默认行为。
2. 不修改 configs/main.yaml 默认模型。
3. 不删除数据。
4. 不进入 V2-M03。
5. 不实现 insert/delete correction。
6. 不实现 TRN / LC / BLC / SLP-aware TRN。
7. 不跑长时间训练或完整 evaluation。

本阶段要做：
1. 新增 baseline exporter：
   - 导出 pred_text / pred_token_ids / pred_token_conf / gt_token_ids / alignment_ops
   - 导出 decoder_hidden
   - 可选导出 encoder_memory
   - 输出 manifest.jsonl + sharded npz
2. 新增 corrector package：
   - 推荐 contract: token + decoder_hidden + confidence + correction_mask
   - 第一版只做 replace-only correction
3. 新增 synthetic-noise dataset path：
   - 从 GT 构造 local replacement noise
   - target = GT
4. 新增最小 loss：
   - selected-position CE + preservation CE
5. 新增保守 inference rule：
   - only selected positions
   - low-confidence gating
   - high-confidence keep
   - no EOS/length edits
6. 新增 offline evaluator：
   - baseline_accuracy
   - corrected_accuracy
   - correction_rate
   - preservation_rate
   - harmful_change_rate
   - replace_error_reduction

要求：
1. 先读 reports/V2-M02s_mdiff_corrector_design.md
2. 先实现 exporter，再实现 corrector
3. 只在小样本上 smoke / overfit
4. 所有代码改动必须可追溯到本阶段目标
5. 最终生成报告：
   reports/V2-M02t_mdiff_corrector_minimal_impl.md

最终回复请包含：
1. 新增文件
2. 修改文件
3. 运行命令
4. smoke 结果
5. 风险与下一步建议
```

## 12. Direct Answers to Core Questions

1. **当前 SLP34K baseline 的 inference path 在哪里？**
   - 在 `ocr_training/test.py:99-124` 发起；
   - `strhub/models/base.py:100-131` 做 test-time decode 与聚合；
   - `ocr_training/strhub/models/maevit_infonce_plm/system.py:294-362` 是核心 `forward()`；
   - `encode()` 在 `system.py:265-277`；
   - `decode()` 在 `system.py:280-291`；
   - token decode 在 `strhub/data/utils.py:79-99`。
2. **当前 PARSeq decoder 是否能导出 predicted string / token ids / per-token logits / per-token confidence / encoder memory / decoder hidden states？**
   - `predicted string`: 能
   - `token ids`: 能派生，但当前不直接返回
   - `per-token logits`: 能
   - `per-token confidence`: 能派生，但当前不直接返回
   - `encoder memory`: 能，通过 `model.encode(images)`
   - `decoder hidden states`: 当前不能直接返回，但代码里已经存在对应张量，适合加 non-invasive hook
3. **如果不能直接导出，需要在哪些函数中增加 non-invasive hook？**
   - 首选 `maevit_infonce_plm/system.py` 的 `forward()`，新增 `forward_with_aux()`
   - 辅助 `data/dataset.py` exporter-only metadata/index 返回
   - exporter helper 中新增 decode/alignment 工具
4. **MDiff corrector 应该使用哪些输入？**
   - 推荐 `pred_token_ids + pred_token_conf + correction_mask + decoder_hidden`
   - `encoder_memory` 作为 optional
5. **corrector 的 target 是 GT token sequence 还是只修正 selected positions？**
   - target 仍是 GT token sequence；但 V2-M02t 的 loss 和 inference 只在 selected replace positions 上允许修正，同时对保留位置加 preservation 约束
6. **correction mask 如何定义？**
   - 训练：synthetic noised positions、oracle wrong positions、low-confidence positions
   - 推理：默认 low-confidence mask，后续可加 confusion-aware 补充
7. **训练数据应从 GT synthetic noise 构造，还是从 frozen PARSeq prediction 构造？**
   - 两阶段都要
   - Stage 1: GT synthetic local noise
   - Stage 2: frozen PARSeq prediction cache
8. **inference 时如何避免把正确 token 改错？**
   - selected-position only
   - correction confidence threshold
   - baseline high-confidence keep rule
   - gain margin
   - EOS/length freeze
   - preservation-aware training
9. **如何评估 correction 是否有效？**
   - 不能只看 overall accuracy
   - 需要 `correction_rate`, `preservation_rate`, `harmful_change_rate`, `replace_error_reduction`, `oracle@K`, confidence-aware precision/recall, subset metrics
