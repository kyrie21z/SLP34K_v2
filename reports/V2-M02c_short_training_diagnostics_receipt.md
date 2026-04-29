# V2-M02c Short Training Diagnostics Receipt

## 1. Summary

- Stage 0：通过。Baseline checkpoint 存在，`encoder.*` key 数量 150，加载到 `slp_mdiff` encoder 的 missing/unexpected 均为 0；真实 probe batch 可读取。
- Stage A：完成。使用 baseline final encoder 初始化，`freeze_encoder=true`，单卡训练 1000 steps。
- Stage B：未执行。Stage A 虽然 loss 明显下降，但预测退化没有缓解，step 200 后固定 probe 全部变成首位 EOS/空字符串。
- Checkpoint：已保存并成功用 `load_from_checkpoint()` 加载，missing/unexpected key 均为 0。
- 显存：低于 40GB。Peak allocated/reserved 为 `1.4110GB / 1.4570GB`。
- 重复退化：形式发生变化但未真正缓解。初始和 step 100 是全长重复字符；step 200 后变为全 EOS 空输出。
- 是否建议进入 V2-M03：不建议。当前最小下一步应先做 all non-PAD CE 对照实验，并检查/修复 EOS 与 loss position 退化。

## 2. Commands

脚本编译：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
python -m py_compile tools/v2_m02c_train_diagnose.py
```

Stage 0 precheck：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02c \
CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02c_train_diagnose.py \
  --mode precheck \
  --device cpu \
  --run-name stage0_precheck \
  --batch-size 2 \
  --num-workers 0 \
  --denoise-steps 1 \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 5 \
  --output-dir outputs/V2-M02c_stage0_precheck
```

Debug loss positions：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02c \
CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02c_train_diagnose.py \
  --mode debug_loss_positions \
  --device cpu \
  --run-name debug_loss_positions \
  --batch-size 2 \
  --num-workers 0 \
  --denoise-steps 1 \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 2 \
  --output-dir outputs/V2-M02c_debug_loss_positions
```

Stage A 1000-step training：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02c \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02c_train_diagnose.py \
  --mode train \
  --device cuda \
  --run-name stageA_1000steps_baseline_encoder \
  --batch-size 2 \
  --num-workers 0 \
  --max-steps 1000 \
  --denoise-steps 1 \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --diagnose-every 100 \
  --num-probe-samples 5 \
  --precision 16 \
  --output-dir outputs/V2-M02c_stageA_1000steps
```

Stage C checkpoint load + quick inference：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02c \
CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02c_train_diagnose.py \
  --mode load \
  --device cpu \
  --run-name stageC_load_quick_inference \
  --batch-size 2 \
  --num-workers 0 \
  --denoise-steps 1 \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 5 \
  --output-dir outputs/V2-M02c_stageC_load \
  --checkpoint-path outputs/V2-M02c_stageA_1000steps/checkpoints/slp_mdiff_stageA_1000steps_baseline_encoder_last.ckpt
```

## 3. Environment and GPU

- Conda env：`slpr_ocr`
- `CUDA_VISIBLE_DEVICES`：Stage A 使用 `0`
- GPU 型号：`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Stage A 启动前 GPU snapshot：
  - `memory.used=79461MiB`
  - `memory.free=17779MiB`
  - `memory.total=97887MiB`
  - `utilization.gpu=99%`
- Peak allocated：`1.4110GB`
- Peak reserved：`1.4570GB`
- 是否低于 40GB：是
- Batch size：`2`
- Precision：`16`
- `num_workers`：`0`
- Git branch：`main`
- Git commit：`bc2b68e633bb13d95f0fd0902a4c7ea02e9aa481`

## 4. Baseline Encoder Migration

- Baseline checkpoint path：`checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- 是否存在：是
- `encoder.*` key 数量：`150`
- Missing keys：`0`
- Unexpected keys：`0`
- 是否成功加载：是
- 是否 freeze encoder：是，`freeze_encoder=true`
- Probe batch shape：`[5, 3, 224, 224]`
- Probe labels：
  - `浙萧山货23765`
  - `六安港LUANGANG`
  - `苏常州货068SUCHANGZHOUHUO`
  - `志远08`
  - `翔运989XIANGYUN`

## 5. Training Curve

| step | loss | eos_rate | avg_pred_len | repeat_ratio | unique_char_count | eos_prob_mean | eos_rank_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | NA | 0.000 | 51.00 | 0.872 | 1.60 | 0.0007 | 520.75 |
| 1 | 6.401634 | 0.000 | 51.00 | 0.872 | 1.60 | 0.0007 | 520.58 |
| 100 | 6.228631 | 0.000 | 51.00 | 1.000 | 1.00 | 0.0037 | 28.43 |
| 200 | 5.732715 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0260 | 1.00 |
| 300 | 5.047743 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0573 | 1.00 |
| 400 | 4.862370 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0765 | 1.00 |
| 500 | 4.897429 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0812 | 1.00 |
| 600 | 4.769792 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0796 | 1.00 |
| 700 | 4.838542 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0821 | 1.00 |
| 800 | 4.769792 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0819 | 1.00 |
| 900 | 4.863363 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0810 | 1.00 |
| 1000 | 4.417332 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0811 | 1.00 |

结论：loss 明显下降，但 decode 指标退化。step 100 为全长单字符重复；step 200 后 EOS 成为每个位置 top-1，decode 为空字符串。

## 6. Probe Predictions

固定 probe sample：

- Sample 1 GT：`浙萧山货23765`
- Sample 2 GT：`六安港LUANGANG`
- Sample 3 GT：`苏常州货068SUCHANGZHOUHUO`

| step | sample | pred | pred_len | contains_eos | repeat_ratio |
|---:|---:|---|---:|---|---:|
| 0 | 1 | `邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 1.000 |
| 0 | 2 | `超超邦超邦超邦邦邦邦邦超邦超邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 0.580 |
| 0 | 3 | `超超超超超超超超超超超超超超超超邦超超超超邦超超...` | 51 | False | 0.820 |
| 1 | 1 | `邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 1.000 |
| 1 | 2 | `超超邦超邦超邦邦邦邦邦超邦超邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 0.580 |
| 1 | 3 | `超超超超超超超超超超超超超超超超邦超超超超邦超超...` | 51 | False | 0.820 |
| 100 | 1 | `邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 1.000 |
| 100 | 2 | `邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 1.000 |
| 100 | 3 | `邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦邦...` | 51 | False | 1.000 |
| 200 | 1 | `` | 0 | True | 1.000 |
| 200 | 2 | `` | 0 | True | 1.000 |
| 200 | 3 | `` | 0 | True | 1.000 |
| 300 | 1 | `` | 0 | True | 1.000 |
| 300 | 2 | `` | 0 | True | 1.000 |
| 300 | 3 | `` | 0 | True | 1.000 |
| 400 | 1 | `` | 0 | True | 1.000 |
| 400 | 2 | `` | 0 | True | 1.000 |
| 400 | 3 | `` | 0 | True | 1.000 |
| 500 | 1 | `` | 0 | True | 1.000 |
| 500 | 2 | `` | 0 | True | 1.000 |
| 500 | 3 | `` | 0 | True | 1.000 |
| 600 | 1 | `` | 0 | True | 1.000 |
| 600 | 2 | `` | 0 | True | 1.000 |
| 600 | 3 | `` | 0 | True | 1.000 |
| 700 | 1 | `` | 0 | True | 1.000 |
| 700 | 2 | `` | 0 | True | 1.000 |
| 700 | 3 | `` | 0 | True | 1.000 |
| 800 | 1 | `` | 0 | True | 1.000 |
| 800 | 2 | `` | 0 | True | 1.000 |
| 800 | 3 | `` | 0 | True | 1.000 |
| 900 | 1 | `` | 0 | True | 1.000 |
| 900 | 2 | `` | 0 | True | 1.000 |
| 900 | 3 | `` | 0 | True | 1.000 |
| 1000 | 1 | `` | 0 | True | 1.000 |
| 1000 | 2 | `` | 0 | True | 1.000 |
| 1000 | 3 | `` | 0 | True | 1.000 |

完整 5 个固定样本的每个诊断点结果保存在：

`ocr_training/outputs/V2-M02c_stageA_1000steps/stageA_1000steps_baseline_encoder_summary.json`

## 7. Checkpoint Save and Load

- Checkpoint path：`/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02c_stageA_1000steps/checkpoints/slp_mdiff_stageA_1000steps_baseline_encoder_last.ckpt`
- Checkpoint size：`1034932931` bytes，约 `987MB`
- `load_from_checkpoint`：成功
- Model class：`strhub.models.slp_mdiff.system.Model`
- `state_dict` key 数量：`264`
- Missing key count：`0`
- Unexpected key count：`0`
- Quick inference：成功，logits shape 为 `[5, 51, 569]`
- Quick inference 结果：`eos_rate=1.0`，`avg_pred_len=0.0`，`repeat_ratio=1.0`，`unique_char_count=0.0`

## 8. Degeneration Analysis

- 是否仍然全长 51：训练初期和 step 100 是全长 51；step 200 后不再是全长字符输出，而是首位 EOS，decode 后长度为 0。
- 是否仍然重复单字符：step 100 是 51 个重复 `邦`；step 200 后变成 EOS top-1，空输出。重复字符形式缓解了，但退化变成更严重的全 EOS/空字符串。
- EOS 是否学到：EOS 被“过度学到”。step 200 后 `eos_rank_mean=1.0`，说明 EOS 在各位置成为 top-1。
- 重复字符问题是否缓解：没有形成有效文本输出，因此不能视为真正缓解。
- 当前最可能的问题来源：
  - 当前 denoising CE 只监督 `masked_positions OR eos_position`，未 mask 的非 PAD 字符位置没有直接监督。
  - EOS 在每条样本中总是进入 loss，并且在 full-mask inference 中一旦 argmax 为 EOS，就会作为下一步输入继续强化空输出。
  - 训练序列长度来自 batch 内 label 长度，推理固定为 51，长位置上的行为约束不足。
  - Plain MDiff 当前没有长度建模、EOS reweight、confidence-based remask 或 anti-collapse 机制。

## 9. Debug Findings

执行 `--mode debug_loss_positions` 的结果：

- Labels：`浙萧山货23765`，`六安港LUANGANG`
- `eos_id=0`
- `pad_id=570`
- `mask_id=571`
- `head_out=569`
- EOS target 数量：`2`
- PAD targets in loss：`0`
- 超出 head 范围的 target 数量：`0`
- EOS target ratio in loss：`0.10`
- 判断：
  - EOS 确实进入了 loss positions。
  - PAD 没有进入 loss。
  - Target id 与 head output 范围对齐。
  - Tokenizer decode 正确遇 EOS 截断；quick inference 中首位 EOS 被 decode 为空字符串。
  - Full-mask inference 中 `argmax` 输出范围为 `0..568`，可作为 decoder 下一步输入，未发现越界。
  - 当前更可疑的是训练目标过稀疏和 EOS 监督/推理闭环导致的 collapse，而不是 tokenizer id 映射错误。

## 10. Recommendation

排序建议：

1. 先增加 all non-PAD CE 对照实验。最小改动是在 V2-M02d 或 V2-M02c-fix 中新增可配置 loss mode：`masked_or_eos` vs `all_non_pad`，保持其他结构不变，验证是否能避免全 EOS/重复字符 collapse。
2. 同时检查 EOS / loss_positions / tokenizer 对齐的实现细节，尤其是不同 label 长度 batch 下 EOS 监督和 51 长度 inference 的差异。
3. 若 all non-PAD CE 能改善 decode，再考虑更长 freeze_encoder 训练。
4. 暂不建议进入 V2-M03；当前 Plain MDiff 最小训练目标尚未稳定，直接加入 all-mask strategies + generic TRN 会掩盖基础退化问题。
