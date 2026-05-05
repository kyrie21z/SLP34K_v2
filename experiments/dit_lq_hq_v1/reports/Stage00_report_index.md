# Stage00 Report Index: Data Preparation and Pair Construction

## 1. Naming Rule

所有 Stage00 报告统一采用 `Stagexx_stepxx_<short_description>.md` 命名；`short_description` 使用小写英文和下划线。

## 2. Stage00 Reports

| Step | Report | Path | Status | Notes |
| --- | --- | --- | --- | --- |
| step01 | raw_dataset_schema_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step01_raw_dataset_schema_report.md` | present |  |
| step02 | train_meta_lmdb_build_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step02_train_meta_lmdb_build_report.md` | present |  |
| step03 | train_ocr_confidence_export_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step03_train_ocr_confidence_export_report.md` | present |  |
| step04 | pair_stats_top1_hq_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step04_pair_stats_top1_hq_report.md` | present |  |
| step05 | pair_visual_inspection_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step05_pair_visual_inspection_report.md` | present |  |
| step06 | pair_stats_top1_hq_visual_v2_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step06_pair_stats_top1_hq_visual_v2_report.md` | present |  |
| step07 | manual_hq_review_package_report | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/manual_hq_review/Stage00_step07_manual_hq_review_package_report.md` | present |  |

## 3. Renamed Files

| Old Name | New Name | Location | Status |
| --- | --- | --- | --- |
| raw_dataset_schema_report.md | Stage00_step01_raw_dataset_schema_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| train_meta_lmdb_build_report.md | Stage00_step02_train_meta_lmdb_build_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| train_meta_lmdb_build_report_dryrun.md | Stage00_step02_dryrun_train_meta_lmdb_build_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | missing |
| train_ocr_confidence_export_report.md | Stage00_step03_train_ocr_confidence_export_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| train_ocr_confidence_export_report_dryrun.md | Stage00_step03_dryrun_train_ocr_confidence_export_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | missing |
| pair_stats_top1_hq_report.md | Stage00_step04_pair_stats_top1_hq_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| pair_stats_top1_hq_report_dryrun.md | Stage00_step04_dryrun_pair_stats_top1_hq_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | missing |
| pair_visual_inspection_report.md | Stage00_step05_pair_visual_inspection_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| pair_stats_top1_hq_visual_v2_report.md | Stage00_step06_pair_stats_top1_hq_visual_v2_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | renamed |
| manual_hq_review_report.md | Stage00_step07_manual_hq_review_package_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/manual_hq_review` | renamed |
| repo_discovery_report.md | repo_discovery_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | legacy_unmapped |
| train_metadata_recheck_report.md | train_metadata_recheck_report.md | `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports` | legacy_unmapped |

## 4. Missing or Conflicted Files

- `train_meta_lmdb_build_report_dryrun.md` -> `Stage00_step02_dryrun_train_meta_lmdb_build_report.md` @ `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports`: missing
- `train_ocr_confidence_export_report_dryrun.md` -> `Stage00_step03_dryrun_train_ocr_confidence_export_report.md` @ `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports`: missing
- `pair_stats_top1_hq_report_dryrun.md` -> `Stage00_step04_dryrun_pair_stats_top1_hq_report.md` @ `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports`: missing
- `repo_discovery_report.md` -> `repo_discovery_report.md` @ `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports`: legacy_unmapped
- `train_metadata_recheck_report.md` -> `train_metadata_recheck_report.md` @ `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports`: legacy_unmapped

## 5. Future Report Naming Requirement

之后所有 dit_lq_hq_v1 方向报告必须使用 `Stagexx_stepxx_` 前缀。
Codex 在新建报告时必须先判断所属阶段与 step 编号。
