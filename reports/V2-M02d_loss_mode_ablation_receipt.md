# V2-M02d Loss Mode Ablation Receipt

## 1. Summary

- 已在 `slp_mdiff` 中实现可配置 `loss_mode`，支持 `masked_or_eos`、`all_non_pad`、`full_mask_all_non_pad`。
- 已完成 `debug_loss_positions`：PAD 没有进入 loss，target id 没有越界，baseline encoder 加载正常。
- 已完成 `all_non_pad` 1000-step 真实数据对照训练，loss finite 且下降，但 300 step 起仍进入全 EOS 空串。
- 已按要求追加 `full_mask_all_non_pad` 1000-step 对照，loss finite 且下降，但 200 step 起进入全 EOS 空串。
- 两个新 checkpoint 都可以通过 `strhub.models.utils.load_from_checkpoint()` 加载，`missing/unexpected=0`。
- 单卡训练峰值显存为 `1.4110 GB allocated / 1.4570 GB reserved`，低于 40GB。
- 结论：`all_non_pad` 没有缓解 EOS collapse。问题不能简单归因于旧版 loss supervision 过稀疏；更可能仍在 loss/noising/inference 设计，尤其是 full-mask argmax inference 下 EOS 作为全局高频终止类被过早选为所有位置 top-1。
- 当前不建议进入 V2-M03。

## 2. Files Changed

- `ocr_training/strhub/models/slp_mdiff/system.py`
  - 新增 `loss_mode` 参数校验与保存。
  - 新增 `_get_loss_positions()`，集中实现三种 loss position 规则。
  - 扩展 `_make_noised_inputs(..., return_full_mask_rows=True)`，为 `full_mask_all_non_pad` 提供当前 batch 的 full-mask row 标记。
- `ocr_training/configs/model/slp_mdiff.yaml`
  - 新增默认配置：`loss_mode: masked_or_eos`，保留旧行为可复现。
- `ocr_training/tools/v2_m02c_train_diagnose.py`
  - 新增 `--loss-mode` 参数。
  - `debug_loss_positions` 输出 loss position 统计、EOS 占比、PAD/越界检查。
  - 训练 summary 记录 `loss_mode`。
- `ocr_training/tools/v2_m02d_loss_ablation.py`
  - 新增 V2-M02d wrapper，复用 V2-M02c 诊断脚本入口。
- `reports/V2-M02d_loss_mode_ablation_receipt.md`
  - 本回执。

未修改 `ocr_training/configs/main.yaml`，未修改数据，未修改原 `maevit_infonce_plm/system.py` 行为。

## 3. Loss Mode Implementation

`clean_targets = tokenizer.encode(labels)[:, 1:]`，即移除 BOS，保留字符、EOS、PAD。head 输出维度为 `569`，对应 `charset + EOS`，不输出 BOS/PAD/MASK。

- `masked_or_eos`
  - `loss_positions = (masked_positions OR eos_positions) AND non_pad`
  - 这是 V2-M02/V2-M02c 旧逻辑。
- `all_non_pad`
  - `loss_positions = non_pad`
  - 所有字符与 EOS 都参与 CE，PAD 被排除。
- `full_mask_all_non_pad`
  - 若当前样本行使用 full mask：`loss_positions = non_pad`
  - 若当前样本行使用 random mask：`loss_positions = (masked_positions OR eos_positions) AND non_pad`
  - 由于 full-mask 行本身已经 mask 所有 non-PAD token，因此在本实现中它与旧逻辑差异主要出现在 random-mask 行。

PAD 始终不参与 loss。EOS 在三种模式中均作为合法输出类参与 loss。debug 显示 loss 中 target id 均在 `[0, 568]` 范围内。

## 4. Debug Loss Position Results

Stage 0 真实 batch probe：`batch_size=2`，labels 为 `浙萧山货23765`、`六安港LUANGANG`，`full_mask_rows=[False, True]`。

| loss_mode | non_pad_count | masked_count | eos_target_count | loss_position_count | eos_ratio_in_loss | pad_in_loss | out_of_range |
|---|---:|---:|---:|---:|---:|---:|---:|
| masked_or_eos | 22 | 19 | 2 | 20 | 0.1000 | 0 | 0 |
| all_non_pad | 22 | 19 | 2 | 22 | 0.0909 | 0 | 0 |
| full_mask_all_non_pad | 22 | 19 | 2 | 20 | 0.1000 | 0 | 0 |

Precheck 结果：

- baseline checkpoint：`checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- baseline exists：true
- encoder key count：150
- encoder load missing/unexpected：0/0
- probe image shape：`[5, 3, 224, 224]`
- encoded label shape：`[5, 23]`
- tokenizer length：571
- `eos_id=0`，`pad_id=570`，`mask_id=571`，`head_out=569`

## 5. Training Diagnostics

旧 `masked_or_eos` 行引用 V2-M02c 同配置 1000-step 结果，未重复长跑。V2-M02d 新跑了 `all_non_pad` 与 `full_mask_all_non_pad`。

| loss_mode | step | loss | eos_rate | avg_pred_len | repeat_ratio | unique_char_count | eos_prob_mean | eos_rank_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| masked_or_eos (V2-M02c) | 0 | - | 0.000 | 51.00 | 0.872 | 1.60 | 0.0007 | 520.75 |
| masked_or_eos (V2-M02c) | 100 | 6.2286 | 0.000 | 51.00 | 1.000 | 1.00 | 0.0037 | 28.43 |
| masked_or_eos (V2-M02c) | 200 | 5.7327 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0260 | 1.00 |
| masked_or_eos (V2-M02c) | 1000 | 4.4173 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0811 | 1.00 |
| all_non_pad | 0 | - | 0.000 | 51.00 | 0.872 | 1.60 | 0.0007 | 520.75 |
| all_non_pad | 100 | 6.1440 | 0.000 | 51.00 | 1.000 | 1.00 | 0.0029 | 62.16 |
| all_non_pad | 200 | 5.7399 | 0.000 | 51.00 | 1.000 | 1.00 | 0.0168 | 2.00 |
| all_non_pad | 300 | 5.0614 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0389 | 1.00 |
| all_non_pad | 400 | 4.8331 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0563 | 1.00 |
| all_non_pad | 500 | 4.8842 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0636 | 1.00 |
| all_non_pad | 600 | 4.8426 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0635 | 1.00 |
| all_non_pad | 700 | 4.9650 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0643 | 1.00 |
| all_non_pad | 800 | 4.4611 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0657 | 1.00 |
| all_non_pad | 900 | 4.7607 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0653 | 1.00 |
| all_non_pad | 1000 | 4.4565 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0655 | 1.00 |
| full_mask_all_non_pad | 0 | - | 0.000 | 51.00 | 0.872 | 1.60 | 0.0007 | 520.75 |
| full_mask_all_non_pad | 100 | 6.2286 | 0.000 | 51.00 | 1.000 | 1.00 | 0.0037 | 28.43 |
| full_mask_all_non_pad | 200 | 5.7327 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0260 | 1.00 |
| full_mask_all_non_pad | 300 | 5.0477 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0573 | 1.00 |
| full_mask_all_non_pad | 400 | 4.8624 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0765 | 1.00 |
| full_mask_all_non_pad | 500 | 4.8974 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0812 | 1.00 |
| full_mask_all_non_pad | 600 | 4.7698 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0796 | 1.00 |
| full_mask_all_non_pad | 700 | 4.8385 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0821 | 1.00 |
| full_mask_all_non_pad | 800 | 4.7698 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0819 | 1.00 |
| full_mask_all_non_pad | 900 | 4.8634 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0810 | 1.00 |
| full_mask_all_non_pad | 1000 | 4.4173 | 1.000 | 0.00 | 1.000 | 0.00 | 0.0811 | 1.00 |

## 6. Probe Predictions

固定 probe samples：

- Sample 1 GT：`浙萧山货23765`
- Sample 2 GT：`六安港LUANGANG`
- Sample 3 GT：`苏常州货068SUCHANGZHOUHUO`
- Sample 4 GT：`志远08`
- Sample 5 GT：`翔运989XIANGYUN`

`masked_or_eos` 旧行为，引用 V2-M02c：

- Step 100：预测为 51 个同一字符重复，`eos_rate=0`，`avg_pred_len=51`。
- Step 200 以后：所有 probe 预测为空字符串，`contains_eos=true`，`eos_rank_mean=1`。

`all_non_pad` 新行为：

- Step 100：所有 probe 预测为 51 个 `邦`，`contains_eos=false`。
- Step 200：所有 probe 预测为 51 个 `H`，EOS rank 已升到 2。
- Step 300 到 1000：所有 probe 预测为空字符串，`contains_eos=true`，`eos_rank_mean=1`。

`full_mask_all_non_pad` 新行为：

- Step 100：所有 probe 预测为 51 个 `邦`，`contains_eos=false`。
- Step 200 到 1000：所有 probe 预测为空字符串，`contains_eos=true`，`eos_rank_mean=1`。

Checkpoint quick inference：

- `all_non_pad` checkpoint：`eos_rate=1.0`，`avg_pred_len=0.0`，`repeat_ratio=1.0`，`eos_prob_mean=0.0655`，`eos_rank_mean=1.0`。
- `full_mask_all_non_pad` checkpoint：`eos_rate=1.0`，`avg_pred_len=0.0`，`repeat_ratio=1.0`，`eos_prob_mean=0.0811`，`eos_rank_mean=1.0`。

## 7. GPU and Runtime

- conda env：`slpr_ocr`
- `CUDA_VISIBLE_DEVICES=0`
- GPU：`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- 启动前 `nvidia-smi`：used `72521 MiB`，free `24719 MiB`，total `97887 MiB`；服务器已有其他进程占用显存。
- 本阶段训练均为单卡、`batch_size=2`、`precision=16`、`num_workers=0`、`denoise_steps=1`、`freeze_encoder=true`。
- `all_non_pad` peak：allocated `1.4110 GB`，reserved `1.4570 GB`。
- `full_mask_all_non_pad` peak：allocated `1.4110 GB`，reserved `1.4570 GB`。
- 两个 run 均低于 40GB。
- 未出现 NaN/Inf，`losses_finite=true`。

Checkpoint：

- `all_non_pad`：
  - `/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02d_stageB_all_non_pad_1000steps/checkpoints/slp_mdiff_stageB_all_non_pad_1000steps_last.ckpt`
  - size：`987 MB`
  - `load_from_checkpoint=true`
  - `missing/unexpected=0/0`
- `full_mask_all_non_pad`：
  - `/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02d_stageC_full_mask_all_non_pad_1000steps/checkpoints/slp_mdiff_stageC_full_mask_all_non_pad_1000steps_last.ckpt`
  - size：`987 MB`
  - `load_from_checkpoint=true`
  - `missing/unexpected=0/0`

## 8. Commands

静态检查：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
python -m py_compile strhub/models/slp_mdiff/system.py tools/v2_m02c_train_diagnose.py tools/v2_m02d_loss_ablation.py
```

Precheck：

```bash
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02d_loss_ablation.py \
  --mode precheck --device cpu \
  --run-name precheck_loss_ablation \
  --batch-size 2 --num-workers 0 --denoise-steps 1 \
  --loss-mode all_non_pad \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 5 \
  --output-dir outputs/V2-M02d_precheck
```

Debug loss positions：

```bash
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02d_loss_ablation.py --mode debug_loss_positions --device cpu \
  --batch-size 2 --num-workers 0 --denoise-steps 1 \
  --loss-mode masked_or_eos \
  --freeze-encoder --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 2 --output-dir outputs/V2-M02d_debug_masked_or_eos

HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02d_loss_ablation.py --mode debug_loss_positions --device cpu \
  --batch-size 2 --num-workers 0 --denoise-steps 1 \
  --loss-mode all_non_pad \
  --freeze-encoder --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 2 --output-dir outputs/V2-M02d_debug_all_non_pad

HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02d_loss_ablation.py --mode debug_loss_positions --device cpu \
  --batch-size 2 --num-workers 0 --denoise-steps 1 \
  --loss-mode full_mask_all_non_pad \
  --freeze-encoder --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --num-probe-samples 2 --output-dir outputs/V2-M02d_debug_full_mask_all_non_pad
```

Training：

```bash
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02d_loss_ablation.py \
  --mode train --device cuda \
  --run-name stageB_all_non_pad_1000steps \
  --batch-size 2 --num-workers 0 --max-steps 1000 \
  --denoise-steps 1 --loss-mode all_non_pad \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --diagnose-every 100 --num-probe-samples 5 --precision 16 \
  --output-dir outputs/V2-M02d_stageB_all_non_pad_1000steps

HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02d_loss_ablation.py \
  --mode train --device cuda \
  --run-name stageC_full_mask_all_non_pad_1000steps \
  --batch-size 2 --num-workers 0 --max-steps 1000 \
  --denoise-steps 1 --loss-mode full_mask_all_non_pad \
  --freeze-encoder \
  --init-encoder-from-baseline-ckpt \
  --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --diagnose-every 100 --num-probe-samples 5 --precision 16 \
  --output-dir outputs/V2-M02d_stageC_full_mask_all_non_pad_1000steps
```

Checkpoint load：

```bash
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02d_loss_ablation.py \
  --mode load --device cuda \
  --run-name load_stageB_all_non_pad \
  --loss-mode all_non_pad \
  --checkpoint-path outputs/V2-M02d_stageB_all_non_pad_1000steps/checkpoints/slp_mdiff_stageB_all_non_pad_1000steps_last.ckpt \
  --num-probe-samples 5 \
  --output-dir outputs/V2-M02d_load_stageB_all_non_pad

HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/zyx/mpl_v2m02d \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02d_loss_ablation.py \
  --mode load --device cuda \
  --run-name load_stageC_full_mask_all_non_pad \
  --loss-mode full_mask_all_non_pad \
  --checkpoint-path outputs/V2-M02d_stageC_full_mask_all_non_pad_1000steps/checkpoints/slp_mdiff_stageC_full_mask_all_non_pad_1000steps_last.ckpt \
  --num-probe-samples 5 \
  --output-dir outputs/V2-M02d_load_stageC_full_mask_all_non_pad
```

## 9. Analysis

1. 问题是否主要来自 adapter？
   - 当前证据不支持 adapter 是主要原因。V2-M02d 沿用 identity visual adapter，且 baseline encoder migration 正常；但需要后续验证 decoder 是否实际利用 visual memory。

2. 问题是否主要来自 decoder 主体代码？
   - 不能排除。`all_non_pad` 后仍然出现所有位置同一预测，再转为所有位置 EOS，说明仅靠更密集 CE 不能让 decoder 学到位置差异和图像条件差异。下一步需要检查 position embedding、cross-attention 输出和 logits 在不同图像/位置上的变化。

3. 问题是否主要来自 loss/noising/inference 设计？
   - 仍然是最可疑方向。`all_non_pad` 消除了“loss 太稀疏”的主要变量，但 full-mask inference 仍是每个位置独立 argmax，EOS 只需要成为边际概率 top-1 就会让所有位置截断为空。最终 EOS 概率只有约 `0.065`，但已是 569 类中的 rank 1。

4. `all_non_pad` 是否能支撑判断？
   - 能支撑“旧 loss 过稀疏不是唯一根因”。它把 loss position 从 20 增加到 22，EOS ratio 从 0.1000 降到 0.0909，但 collapse 仍发生，只是从约 200 step 推迟到约 300 step。

5. 还需要验证什么？
   - one-step logits 是否对所有位置高度相同。
   - decoder hidden 是否随 position embedding 变化。
   - cross-attention 是否随 visual memory 变化。
   - head bias / EOS logit 是否过快成为全局 top-1。
   - full-mask inference 是否需要 EOS bias、length penalty，或训练/推理阶段更一致的 denoising 目标。

## 10. Recommendation

不建议进入 V2-M03。

建议下一步排序：

1. **V2-M02g：检查 decoder 是否利用 visual memory。**
   - 做最小 logits/hidden 诊断：同图不同 position、不同图同 position、memory zero-out、memory shuffle、cross-attention 输出范数。
   - 如果视觉 memory 和 position 对 logits 几乎无影响，应先修 decoder/conditioning。
2. **V2-M02f：加入 EOS bias / length penalty 诊断。**
   - 若 V2-M02g 证明 decoder 有使用 visual memory，再做低风险 EOS logit bias、禁止 early EOS 或 length penalty 对照。
3. 暂不进入 **V2-M03：all-mask strategies + generic TRN**。
   - 当前 Plain MDiff 基础链路仍无法避免全 EOS collapse，贸然移植更复杂策略会掩盖根因。
