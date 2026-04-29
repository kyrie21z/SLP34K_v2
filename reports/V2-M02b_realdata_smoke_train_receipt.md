# V2-M02b Real-data Smoke Training Receipt

## 1. Summary

- Stage A 真实数据 smoke train：完成。真实 SLP34K LMDB batch 可读取，`slp_mdiff` 可在单卡上完成 forward/backward，10 steps loss finite。
- Stage B small overfit：完成。执行 100 steps，loss 从 `6.428013` 到 `6.287946`，有轻微下降趋势；未出现 NaN/Inf 或 shape/index/device 错误。
- Checkpoint 保存：完成。Stage A 和 Stage B 均保存了 Lightning checkpoint。
- Checkpoint loading：完成。`strhub.models.utils.load_from_checkpoint()` 可加载 Stage B checkpoint，missing/unexpected key 数量均为 0。
- Baseline encoder migration：完成。原 baseline checkpoint 存在，`encoder.*` key 数量 150，加载到 `slp_mdiff` MAE encoder 的 missing/unexpected 均为 0。
- 显存限制：满足。Stage B peak allocated/reserved 为 `1.4105GB / 1.4492GB`，低于 40GB。
- 注意：启动前整卡 free memory 只有约 `17.78GB`，低于 40GB 空闲阈值，但本次训练进程自身峰值显存远低于 40GB；未启动完整训练，未进入 V2-M03。

## 2. Commands

辅助脚本编译：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
python -m py_compile tools/v2_m02b_smoke_check.py
```

真实 LMDB batch inspect：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02b_smoke_check.py \
  --mode inspect \
  --device cpu \
  --run-name inspect \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/V2-M02b_slp_mdiff_smoke
```

Stage A smoke train：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02b_smoke_check.py \
  --mode train \
  --device cuda \
  --run-name stageA_smoke \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 10 \
  --denoise-steps 1 \
  --freeze-encoder \
  --output-dir outputs/V2-M02b_slp_mdiff_smoke_stageA
```

Stage B small overfit：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
HYDRA_FULL_ERROR=1 \
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/v2_m02b_smoke_check.py \
  --mode train \
  --device cuda \
  --run-name stageB_overfit100 \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 100 \
  --denoise-steps 1 \
  --freeze-encoder \
  --output-dir outputs/V2-M02b_slp_mdiff_smoke_stageB
```

Stage C checkpoint loading：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02b_smoke_check.py \
  --mode load \
  --device cpu \
  --run-name checkpoint_load \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/V2-M02b_slp_mdiff_smoke_stageC \
  --checkpoint-path outputs/V2-M02b_slp_mdiff_smoke_stageB/checkpoints/slp_mdiff_stageB_overfit100_last.ckpt
```

Stage D baseline encoder migration：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b CUDA_VISIBLE_DEVICES='' \
python tools/v2_m02b_smoke_check.py \
  --mode baseline \
  --device cpu \
  --run-name baseline_migration \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/V2-M02b_slp_mdiff_smoke
```

快速 decode 异常输出检查：

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr
MPLCONFIGDIR=/tmp/zyx/mpl_v2m02b CUDA_VISIBLE_DEVICES='' python - <<'PY'
import torch
from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule
ckpt='outputs/V2-M02b_slp_mdiff_smoke_stageB/checkpoints/slp_mdiff_stageB_overfit100_last.ckpt'
model=load_from_checkpoint(ckpt).eval()
hp=model.hparams
dm=SceneTextDataModule('data', 'SLP34K_lmdb_train', 'SLP34K_lmdb_test', 'SLP34K_lmdb_benchmark', hp.img_size, hp.max_label_length, hp.charset_train, hp.charset_test, 1, 0, False)
imgs, labels=next(iter(dm.train_dataloader()))
with torch.inference_mode():
    logits=model(imgs)
    preds, probs=model.tokenizer.decode(logits.softmax(-1))
print(f'label={labels[0]}')
print(f'pred={preds[0]}')
print(f'logits_shape={tuple(logits.shape)}')
print(f'pred_len={len(preds[0])}')
PY
```

## 3. Environment and GPU

- Git branch：`main`
- Git commit：`bc2b68e633bb13d95f0fd0902a4c7ea02e9aa481`
- Conda env：`slpr_ocr`
- `CUDA_VISIBLE_DEVICES`：Stage A/B 使用 `0`；CPU load/migration 使用空值。
- `PYTORCH_CUDA_ALLOC_CONF`：Stage A/B 使用 `expandable_segments:True`。
- GPU 型号：`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Stage A/B 启动前 GPU snapshot：
  - `index=0`
  - `memory.used=79461MiB`
  - `memory.free=17779MiB`
  - `memory.total=97887MiB`
  - `utilization.gpu=99%`
- Stage A peak allocated/reserved：`1.4105GB / 1.4434GB`
- Stage B peak allocated/reserved：`1.4105GB / 1.4492GB`
- 是否低于 40GB：是。

当前 git status 摘要：

```text
 M ocr_training/strhub/models/maevit_infonce_plm/system.py
 M ocr_training/strhub/models/maevit_plm/system.py
 M ocr_training/strhub/models/utils.py
?? ocr_training/configs/model/slp_mdiff.yaml
?? ocr_training/strhub/models/slp_mdiff/
?? ocr_training/tools/
?? repo_file_index_depth3.txt
?? reports/
?? scripts/
```

其中 `maevit_infonce_plm/system.py` 和 `maevit_plm/system.py` 是本阶段开始前已存在的工作区修改，本阶段未修改它们。

## 4. Data Loading

- 数据路径：`/mnt/data/zyx/SLP34K_v2/ocr_training/data`
- Train LMDB：`SLP34K_lmdb_train`
- Batch size：`1`
- `num_workers`：`0`。原因是当前 sandbox 环境下 DataLoader 多进程会触发 multiprocessing socket 权限错误。
- Image tensor shape：`[1, 3, 224, 224]`
- Label 示例：`杭州港HANGZHOUGANG`
- Tokenizer encode shape：`[1, 17]`
- Tokenizer length：`571`
- `mask_id`：`571`
- Head output dim：`569`
- 真实 LMDB batch 读取：成功。

## 5. Training Smoke Results

Stage A 配置：

- `max_steps=10`
- `batch_size=1`
- `denoise_steps=1`
- `freeze_encoder=true`
- `precision=16`
- `limit_val_batches=0`
- `num_workers=0`

Stage A 结果：

- Global step：`10`
- First loss：`6.428013`
- Mid loss：`6.504185`
- Final loss：`6.585938`
- Loss count：`10`
- Loss finite：`True`
- Forward/backward：成功。
- NaN/Inf：未出现。
- Shape/index/device error：未出现。
- Peak allocated/reserved：`1.4105GB / 1.4434GB`
- Checkpoint：`/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02b_slp_mdiff_smoke_stageA/checkpoints/slp_mdiff_stageA_smoke_last.ckpt`

## 6. Small Overfit Results

Stage B 配置：

- `max_steps=100`
- `batch_size=1`
- `denoise_steps=1`
- `freeze_encoder=true`
- `precision=16`
- `limit_val_batches=0`
- `num_workers=0`

Stage B 结果：

- Global step：`100`
- First loss：`6.428013`
- Mid loss：`6.526042`
- Final loss：`6.287946`
- Loss count：`100`
- Loss finite：`True`
- Peak allocated/reserved：`1.4105GB / 1.4492GB`
- Loss 趋势：有轻微下降，但 100 steps 仍不足以证明稳定 overfit。
- 异常输出检查：用 Stage B checkpoint 对一个真实训练样本做 CPU inference，预测为 51 个重复 `征`：
  - Label：`皖寿县货8966`
  - Pred：`征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征征`
  - Logits shape：`(1, 51, 569)`
  - 结论：输出明显退化为重复字符；这符合随机初始化 decoder/head 和极短训练的预期，不作为当前 smoke 阻塞，但进入短程训练前需要重点跟踪 EOS/重复字符问题。

## 7. Checkpoint Save and Load

- Stage A checkpoint：`/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02b_slp_mdiff_smoke_stageA/checkpoints/slp_mdiff_stageA_smoke_last.ckpt`
- Stage B checkpoint：`/mnt/data/zyx/SLP34K_v2/ocr_training/outputs/V2-M02b_slp_mdiff_smoke_stageB/checkpoints/slp_mdiff_stageB_overfit100_last.ckpt`
- Checkpoint size：约 `987MB`。
- `load_from_checkpoint` 验证对象：Stage B checkpoint。
- 是否成功加载：是。
- Model class：`strhub.models.slp_mdiff.system.Model`
- Checkpoint `state_dict` key 数量：`264`
- Missing key count：`0`
- Unexpected key count：`0`
- 是否需要 strict=false：当前 `strhub.models.utils.load_from_checkpoint()` 内部使用 `strict=False`，本次加载成功。
- 是否影响原 baseline loading：本阶段仅使用已存在的 `utils.py` 中 `slp_mdiff` 分支；未修改 `maevit_infonce_plm`/`maevit_plm` 分支。
- 未运行完整 `test.py` evaluation；本阶段只验证 checkpoint load，不做 full evaluation。

## 8. Baseline Encoder Migration

- Baseline checkpoint path：`checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- 是否存在：是。
- Checkpoint size：约 `1.6GB`
- `encoder.*` key 数量：`150`
- 是否成功提取并加载到 `slp_mdiff` MAE encoder：是。
- Missing keys：`0`
- Unexpected keys：`0`
- 建议：后续短程训练建议优先尝试使用 baseline final encoder 初始化，而不是只用 MAE pretrain encoder；这能更接近原 SLP34K strong visual encoder。

## 9. Compatibility

- 未修改 `configs/main.yaml` 默认模型。
- 未改动数据文件。
- 未删除任何文件。
- 未进入 V2-M03 功能。
- 未实现 SLP-aware TRN。
- 未实现 generic token replacement noise。
- 未实现 LC/BLC remask。
- 未实现 segment-aware denoising。
- 未实现 pinyin consistency。
- 未新增复杂 evaluator。
- 未修改原 `maevit_infonce_plm/system.py` 行为。

## 10. Problems and Fixes

- 问题：sandbox 内 PyTorch 无法初始化 CUDA/NVML，`torch.cuda.is_available()` 为 False。
  - 修复：按权限要求在 sandbox 外执行 Stage A/B GPU 命令。
- 问题：DataLoader `num_workers=2` 在当前 sandbox 下触发 multiprocessing socket `PermissionError: Operation not permitted`。
  - 修复：V2-M02b smoke 使用 `num_workers=0`。
- 问题：helper 脚本最初将 `precision` 以字符串 `"16"` 传给 PL 1.8，触发 `RuntimeError: No precision set`。
  - 修复：数字 precision 自动转换为 int。
- 问题：inspect 阶段打开 LMDB 后，训练阶段在同进程重新打开同一路径会触发 `lmdb.Error: environment is already open in this process`。
  - 修复：inspect 后递归关闭 dataset 中的 LMDB env。
- 问题：训练结束后读取 CUDA peak memory 时使用了 `model.device`，此时 model 可能已回到 CPU，触发 `ValueError: Expected a cuda device`。
  - 修复：显式使用 `torch.device("cuda:0")` 读取 memory stats。
- 问题：PL 1.8 输出 `lr_scheduler.step()` 顺序 warning。
  - 处理：本阶段未修改 optimizer/scheduler 行为；warning 未阻塞训练。后续若进入较长训练，应检查 `configure_optimizers()` 与 PL 版本兼容性。

## 11. Recommendation

- 下一步建议先做 V2-M02c：使用 `freeze_encoder=true`、baseline final encoder migration、`batch_size=1/2`、`denoise_steps=1` 做更短程的 real-data training，并重点观察 EOS 与重复字符退化。
- 如果 V2-M02c loss 和 decode 仍异常，暂停并优先修复 tokenizer/EOS/loss position 或 noising 策略问题。
- V2-M02d 可在稳定后尝试 unfreeze encoder small LR。
- V2-M03 再移植 all-mask strategies + generic TRN；当前不建议直接进入 V2-M03。
