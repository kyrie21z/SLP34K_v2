# Towards Ship License Plate Recognition in the Wild: A Large Benchmark and Strong Baseline

Paper: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32569)  
Supplementary material: [aaai2025_ocr_supp.pdf](./aaai2025_ocr_supp.pdf)  
Dataset license agreement: [English](./SLP34K_Dataset_License_Agreement.pdf) / [中文](./SLP34K数据集许可协议.pdf)

This repository now contains both:

1. The original AAAI 2025 SLP34K baseline for pre-training and recognition fine-tuning.
2. The current V2 experimental pipeline for PARSeq/MDiff-based post-correction.

![avatar](./image/README/SLP34K.png)

SLP34K is a large ship license plate (SLP) benchmark collected from a real-world waterway intelligent traffic surveillance system. Images were captured by eight surveillance cameras across eight locations over 42 months, resulting in 34,385 diverse SLP images.

![avatar](./image/README/baseline.png)

The original paper baseline uses a strong visual encoder with self-supervised pre-training and semantic enhancement. The V2 branch in this repo keeps that baseline intact, then adds an offline correction pipeline that operates on PARSeq outputs.

## Table of Contents

- [Current Status](#current-status)
- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [Dataset and Checkpoints](#dataset-and-checkpoints)
- [Original Baseline Training](#original-baseline-training)
- [V2 MDiff Corrector Workflow](#v2-mdiff-corrector-workflow)
- [Current Best V2 Setting](#current-best-v2-setting)
- [Reports](#reports)
- [Local Gradio Demo](#local-gradio-demo)

## Current Status

As of `2026-05-04`, the repo status is:

- The original baseline training and evaluation path is still available and unchanged as the main paper implementation.
- A full offline post-correction toolchain has been added under `ocr_training/tools/` and `ocr_training/strhub/models/slp_mdiff_corrector/`.
- The current V2 work is still `analysis split only`; it is not an untouched official benchmark conclusion.
- The current best corrector checkpoint is `ocr_training/outputs/V2-M02w3_corrector_aw1p0_wmax3p0_lp0p5/`.
- The current best deployable evaluation policy is `low_conf_or_confusion` with:
  - `tau_low=0.90`
  - `tau_corr=0.30`
  - `tau_keep=0.90`
  - `delta_gain=-0.10`

Current best V2 analysis result from [reports/V2-M02w3_pair_weight_synthetic_ratio_tuning.md](./reports/V2-M02w3_pair_weight_synthetic_ratio_tuning.md):

| Slice | correction_rate | gain_accuracy | harmful_change_rate |
| --- | ---: | ---: | ---: |
| `full_incorrect` | `0.1894` | `+0.0988` | `0.0100` |
| `replace_dominant` | `0.2238` | `+0.2801` | `0.0000` |
| `replace_only` | `0.4256` | `+0.3856` | `0.0000` |
| `hard_slice` | `0.1873` | `+0.1392` | `0.0357` |
| `long_21plus` | `0.1245` | `+0.0962` | `0.0000` |

Important limitations of the current V2 branch:

- It is a `replace-only` corrector.
- It does not perform insert/delete correction.
- It does not add an `encoder_memory` branch.
- `hard_slice` harmful change is still above the desired safety line, so the current best setting is not yet treated as final deployment-ready policy.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `mae/` | Self-supervised MAE pre-training code from the original baseline |
| `ocr_training/` | Recognition training, evaluation, Gradio demo, and V2 corrector work |
| `ocr_training/configs/model/slp_mdiff.yaml` | V2 MDiff model config |
| `ocr_training/configs/model/slp_mdiff_corrector.yaml` | Offline corrector model config |
| `ocr_training/strhub/models/slp_mdiff/` | MDiff baseline implementation |
| `ocr_training/strhub/models/slp_mdiff_corrector/` | Post-correction model implementation |
| `ocr_training/tools/export_parseq_corrector_cache.py` | Export baseline token-level cache with decoder hidden states |
| `ocr_training/tools/build_confusion_table.py` | Build train-split confusion table |
| `ocr_training/tools/build_pair_difficulty_table.py` | Build pair difficulty / weighting table |
| `ocr_training/tools/train_mdiff_corrector_smoke.py` | Train offline corrector on exported cache |
| `ocr_training/tools/eval_mdiff_corrector_offline.py` | Offline correction evaluation |
| `ocr_training/tools/filter_mdiff_corrector_cache.py` | Derive held-out analysis slices |
| `ocr_training/outputs/` | Generated caches, checkpoints, and evaluation outputs |
| `reports/` | Stage-by-stage experiment receipts and conclusions |

## Environment

The project still uses two environments, exactly as in the original baseline.

### 1. Self-supervised pre-training environment

```bash
cd SLP34K/mae
conda create -n slk34k_mae python=3.8
conda activate slk34k_mae
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Note: the pre-training code expects `timm==0.3.2`. If you upgrade PyTorch substantially, you may need compatibility fixes.

### 2. OCR / fine-tuning / corrector environment

```bash
cd SLP34K/ocr_training
conda create --name slk34k_rec python=3.9
conda activate slk34k_rec
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

All V2 MDiff/corrector experiments in this repo were run from the `ocr_training/` side with GPU enabled.

## Dataset and Checkpoints

Before getting the dataset, applicants must sign the dataset license agreement. Please print the agreement, hand-sign it, scan it, and email it from an official institutional email address to `liubaolongx@gmail.com` or `dongjf24@gmail.com`.

Dataset packages:

| Dataset | File Size | Format |
| --- | ---: | --- |
| `SLP34K.7z` | 153.8 MB | JPG |
| `data.7z` | 6.36 GB | LMDB |

After download, extract `data.7z` into `ocr_training/data/`.

Expected structure:

```text
ocr_training/data/
  train/
    SLP34K_lmdb_train/
    Union14m-L/
  val/
    SLP34K_lmdb_test/
    Union14m_benchmark/
  test/
    SLP34K_lmdb_train/
    SixCommon_benchmark/
    Union14m_benchmark/
```

Pretrained weights:

| Checkpoint | File Size | Usage | Link |
| --- | ---: | --- | --- |
| `pretrain_model.7z` | 2.2 GB | Pre-training weights | [Google Drive](https://drive.google.com/file/d/1K6jmsRNvnKL5om352MJ7CU3K0xA7EjtN/view?usp=drive_link) |
| `checkpoint.7z` | 2.7 GB | Recognition checkpoints | [Google Drive](https://drive.google.com/file/d/1s1VHmofcvZic0WkVxfx1PHv2F5MOS-VH/view?usp=drive_link) |

Extract the weights into `ocr_training/`.

## Original Baseline Training

The original paper training path is still available.

### Pre-training on SLP34K

```bash
cd mae
mkdir -p pretrain_data
cp -r ../ocr_training/data/train/SLP34K_lmdb_train ./pretrain_data

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
  main_pretrain.py \
  --data_path pretrain_data/SLP34K_lmdb_train \
  --mask_ratio 0.75 \
  --model mae_vit_base_patch16_224x224 \
  --output_dir pretrain_SLP34K_patch16_224x224 \
  --log_dir pretrain_SLP34K_patch16_224x224 \
  --batch_size 128 \
  --norm_pix_loss \
  --epochs 1500 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05
```

### Fine-tuning on SLP34K

```bash
cd ocr_training

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  model.img_size=[224,224] \
  charset=SLP34K_568 \
  dataset=SLP34K \
  model.batch_size=60 \
  trainer.gpus=2 \
  trainer.val_check_interval=200 \
  model=maevit_infonce_plm \
  trainer.max_epochs=100 \
  hydra.run.dir=outputs/ship/maevit_infonce_plm \
  model.max_label_length=50 \
  +trainer.accumulate_grad_batches=5 \
  model.mae_pretrained_path=./pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth
```

### Evaluation on SLP34K

```bash
cd ocr_training
./test.py outputs/ship/maevit_infonce_plm/checkpoints/last.ckpt \
  --batch_size 700 \
  --test_data SLP34K \
  --test_dir SLP34K_lmdb_benchmark
```

## V2 MDiff Corrector Workflow

The V2 branch adds an offline post-correction pipeline on top of the baseline recognizer. The most practical way to reproduce the current state is to reuse the cached analysis split and tables already stored under `ocr_training/outputs/`.

All commands below assume:

```bash
cd /mnt/data/zyx/SLP34K_v2/ocr_training
conda activate slk34k_rec
```

### 1. Reuse the current main analysis split

The main analysis split already exists at:

- `outputs/V2-M02w0_cache_main_split_incorrect/train`
- `outputs/V2-M02w0_cache_main_split_incorrect/eval`

This split is `analysis split only` and was built from `test/unified_lmdb`.

### 2. Build the confusion table

```bash
python tools/build_confusion_table.py \
  --cache_dir outputs/V2-M02w0_cache_main_split_incorrect/train \
  --output_dir outputs/V2-M02w1_confusion_table \
  --min_count 1 \
  --top_k 50
```

### 3. Build the pair difficulty table

This step uses the train confusion table plus pair statistics from the held-out analysis split.

```bash
python tools/build_pair_difficulty_table.py \
  --train_cache_dir outputs/V2-M02w0_cache_main_split_incorrect/train \
  --confusion_table outputs/V2-M02w1_confusion_table/confusion_table.json \
  --pair_stats_json outputs/V2-M02z_eval_full_incorrect/pair_stats.json \
  --output_dir outputs/V2-M02w2_pair_difficulty \
  --min_support 5
```

### 4. Train the current best corrector

This reproduces the current best V2-M02w3 training setting:

- `contract_type=token_decoder_hidden`
- `loss_type=pair_weighted_selected_plus_preservation`
- `synthetic_mode=confusion`
- `synthetic_ratio=1.0`
- `pair_weight_alpha=1.0`
- `pair_weight_max=3.0`
- `lambda_preservation=0.5`

```bash
python tools/train_mdiff_corrector_smoke.py \
  --cache_dir outputs/V2-M02w0_cache_main_split_incorrect/train \
  --output_dir outputs/V2-M02w3_corrector_aw1p0_wmax3p0_lp0p5 \
  --contract_type token_decoder_hidden \
  --replace_only true \
  --loss_type pair_weighted_selected_plus_preservation \
  --lambda_preservation 0.5 \
  --batch_size 8 \
  --max_steps 500 \
  --lr 1e-4 \
  --synthetic-mode confusion \
  --synthetic-ratio 1.0 \
  --confusion-table outputs/V2-M02w1_confusion_table/confusion_table.json \
  --confusion-topk 20 \
  --pair-weight-table outputs/V2-M02w2_pair_difficulty/pair_difficulty.json \
  --pair-weight-alpha 1.0 \
  --pair-weight-min 1.0 \
  --pair-weight-max 3.0 \
  --device cuda
```

### 5. Evaluate with the current deployable policy

```bash
python tools/eval_mdiff_corrector_offline.py \
  --cache_dir outputs/V2-M02w0_cache_main_split_incorrect/eval \
  --corrector_ckpt outputs/V2-M02w3_corrector_aw1p0_wmax3p0_lp0p5/checkpoints/last.ckpt \
  --output_json outputs/V2-M02w3_eval_aw1p0_wmax3p0_lp0p5_full_incorrect/eval_summary.json \
  --selected-mask-policy low_conf_or_confusion \
  --confusion-table outputs/V2-M02w1_confusion_table/confusion_table.json \
  --tau_low 0.90 \
  --tau_corr 0.30 \
  --tau_keep 0.90 \
  --delta_gain -0.10 \
  --pair-stats true \
  --subset-stats true \
  --save-case-study true \
  --save-position-diagnostics true \
  --device cuda
```

The resulting summary should match the checked-in analysis output at:

- `outputs/V2-M02w3_eval_aw1p0_wmax3p0_lp0p5_full_incorrect/eval_summary.json`

### Optional: rebuild the original cache yourself

If you want to rebuild the larger analysis cache from the baseline checkpoint instead of reusing the stored split, the core exporter is:

```bash
python tools/export_parseq_corrector_cache.py \
  --checkpoint checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
  --split test \
  --subset unified_lmdb \
  --output_dir outputs/V2-M02w0_cache_test_unified_incorrect_bs64 \
  --batch_size 64 \
  --export_decoder_hidden true \
  --export_encoder_memory false \
  --topk 8 \
  --filter-mode incorrect \
  --scan-limit 12000 \
  --max-export 3000 \
  --device cuda
```

## Current Best V2 Setting

Best current training setting:

| Item | Value |
| --- | --- |
| Model | `slp_mdiff_corrector` |
| Contract | `token_decoder_hidden` |
| Loss | `pair_weighted_selected_plus_preservation` |
| `lambda_preservation` | `0.5` |
| Synthetic mode | `confusion` |
| Synthetic ratio | `1.0` |
| `pair_weight_alpha` | `1.0` |
| `pair_weight_max` | `3.0` |
| Checkpoint | `ocr_training/outputs/V2-M02w3_corrector_aw1p0_wmax3p0_lp0p5/checkpoints/last.ckpt` |

Best current deployable policy:

| Item | Value |
| --- | --- |
| Policy | `low_conf_or_confusion` |
| `tau_low` | `0.90` |
| `tau_corr` | `0.30` |
| `tau_keep` | `0.90` |
| `delta_gain` | `-0.10` |

Why this is the current best:

- It improves major slices consistently over V2-M02w2.
- It reduces `full_incorrect harmful_change_rate` from `0.0117` to `0.0100`.
- It remains the most stable global choice even though some weak pairs can improve slightly more under more aggressive weighting.

Why it is not final:

- `hard_slice harmful_change_rate` is still `0.0357`.
- `O->A` remains largely an oracle/ranking problem rather than a solved correction pair.
- Insert/delete-heavy samples still expose the limit of the current replace-only formulation.

## Reports

The most useful reading order for the current repo state is:

1. [reports/V2-M02t_mdiff_corrector_minimal_impl.md](./reports/V2-M02t_mdiff_corrector_minimal_impl.md)
2. [reports/V2-M02u_real_error_corrector_validation.md](./reports/V2-M02u_real_error_corrector_validation.md)
3. [reports/V2-M02v_expanded_corrector_split_validation.md](./reports/V2-M02v_expanded_corrector_split_validation.md)
4. [reports/V2-M02w0_expanded_real_error_cache.md](./reports/V2-M02w0_expanded_real_error_cache.md)
5. [reports/V2-M02w1_confusion_aware_synthetic_noise.md](./reports/V2-M02w1_confusion_aware_synthetic_noise.md)
6. [reports/V2-M02y_selected_mask_threshold_calibration.md](./reports/V2-M02y_selected_mask_threshold_calibration.md)
7. [reports/V2-M02z_larger_slice_calibrated_policy_validation.md](./reports/V2-M02z_larger_slice_calibrated_policy_validation.md)
8. [reports/V2-M02w2_pair_specific_synthetic_loss_calibration.md](./reports/V2-M02w2_pair_specific_synthetic_loss_calibration.md)
9. [reports/V2-M02w3_pair_weight_synthetic_ratio_tuning.md](./reports/V2-M02w3_pair_weight_synthetic_ratio_tuning.md)

Short interpretation of the V2 trajectory:

- `V2-M02t` established the minimal cache/train/eval pipeline.
- `V2-M02u` and `V2-M02v` showed that real-error correction is feasible but small-sample evidence was weak.
- `V2-M02w0` expanded the analysis cache to a usable size.
- `V2-M02w1` showed that confusion-aware synthetic noise beats random synthetic.
- `V2-M02y` and `V2-M02z` showed that threshold calibration and confusion-aware selection matter more than default low-confidence gating.
- `V2-M02w2` showed that pair-weighted loss is better than pair-specific synthetic alone.
- `V2-M02w3` is the current best tuning point inside the existing V2-M02 scope.

## Local Gradio Demo

The original local demo is still available:

```bash
cd ocr_training
python gradio_SLP34K.py
```
