# Train OCR Confidence Export Report

## 1. Scope and Isolation Statement

本轮仅在 `experiments/dit_lq_hq_v1/` 下创建或覆盖脚本、manifest 和报告。
未修改原始 SLP34K 数据、现有 OCR LMDB、configs、checkpoints、outputs 或源码。
当前执行状态：成功

## 2. Environment

- Python: `/mnt/data/zyx/miniconda3/envs/slpr_ocr/bin/python`
- torch: `2.10.0+cu130`
- pytorch_lightning: `1.8.6`
- lmdb: `OK`
- PIL: `OK`
- yaml: `OK`
- hydra: `OK`
- CUDA 可用: `True`
- 实际设备: `cuda:0`
- 设备说明: `无`

## 3. Inputs and Outputs

- 输入 LMDB: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/SLP34K_lmdb_train_meta`
- 输入 checkpoint: `/mnt/data/zyx/SLP34K_v2/ocr_training/checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`
- MAE 预训练权重: `/mnt/data/zyx/SLP34K_v2/ocr_training/pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth`
- 输出 CSV: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.csv`
- 输出 JSONL: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.jsonl`
- 输出报告: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/reports/Stage00_step03_train_ocr_confidence_export_report.md`

## 4. Model Loading

- 加载函数: `strhub.models.utils.load_from_checkpoint`
- checkpoint 类型: `maevit_infonce_plm`
- 图像尺寸: `[224, 224]`
- `max_label_length`: `50`
- `charset_train` 长度: `568`
- `charset_test` 长度: `568`

## 5. Dataset Reading

- 数据集类: `ExportLmdbDataset(LmdbDataset)`
- 读取键: `image-%09d`, `label-%09d`, `meta-%09d`
- `lmdb_index`: 1-based
- 样本总数: `27501`
- 实际处理样本数: `27501`
- `batch_size`: `64`
- `num_workers`: `8`

## 6. Decoding and Confidence Definition

- 图像预处理: `SceneTextDataModule.get_transform(model.hparams.img_size, augment=False)`
- 推理路径: `model(images) -> logits.softmax(-1) -> model.tokenizer.decode(probs)`
- `pred`: greedy decode 后的字符串
- `avg_conf`: `tokenizer.decode()` 返回的预测路径 token 概率序列均值
- `min_conf`: 同一概率序列最小值
- `confidence`: 当前版本直接等于 `avg_conf`
- `nll`: 当前定义为预测路径的 `-sum(log p)`，其中 `p` 为同一 greedy 路径上的 token 概率

## 7. Export Summary

- `num_samples_total`: `27501`
- `num_samples_processed`: `27501`
- `num_correct`: `27494`
- `accuracy`: `0.999745`
- `output_csv`: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.csv`
- `output_jsonl`: `/mnt/data/zyx/SLP34K_v2/experiments/dit_lq_hq_v1/data/manifests/train_ocr_predictions.jsonl`
- `runtime_seconds`: `348.424`
- `device`: `cuda:0`
- `batch_size`: `64`

## 8. Accuracy and Confidence Statistics

- `overall_accuracy`: `0.999745`
- `mean_confidence`: `0.999901`
- `median_confidence`: `0.999999`
- `mean_avg_conf`: `0.999901`
- `mean_min_conf`: `0.998980`
- `correct_mean_confidence`: `0.999907`
- `wrong_mean_confidence`: `0.977183`

## 9. Quality / Structure Breakdown

### quality

| quality | count | accuracy | mean_confidence | mean_avg_conf | mean_min_conf |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 6233 | 0.999840 | 0.999944 | 0.999944 | 0.999402 |
| hard | 4062 | 0.999754 | 0.999823 | 0.999823 | 0.998139 |
| middle | 17206 | 0.999709 | 0.999904 | 0.999904 | 0.999025 |

### structure

| structure | count | accuracy | mean_confidence | mean_avg_conf | mean_min_conf |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi | 15384 | 0.999610 | 0.999899 | 0.999899 | 0.998766 |
| single | 8112 | 0.999877 | 0.999907 | 0.999907 | 0.999056 |
| vertical | 4005 | 1.000000 | 0.999899 | 0.999899 | 0.999643 |

### structure_type

| structure_type | count | accuracy | mean_confidence | mean_avg_conf | mean_min_conf |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_lines | 15384 | 0.999610 | 0.999899 | 0.999899 | 0.998766 |
| single_line | 8112 | 0.999877 | 0.999907 | 0.999907 | 0.999056 |
| vertical | 4005 | 1.000000 | 0.999899 | 0.999899 | 0.999643 |

## 10. Sample Records

### 前 5 条样本

| lmdb_index | label | pred | correct | confidence | quality | structure | source_path |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | 浙萧山货23765 | 浙萧山货23765 | True | 0.999999 | middle | single | train/middle&single&ng&nd&浙萧山货23765&2&0&T_20220125_07_20_05_935375.jpg |
| 2 | 六安港LUANGANG | 六安港LUANGANG | True | 1.000000 | middle | vertical | train/middle&vertical&ng&nd&六安港LUANGANG&2&2&T_20220626_15_49_06_292625.jpg |
| 3 | 苏常州货068SUCHANGZHOUHUO | 苏常州货068SUCHANGZHOUHUO | True | 1.000000 | hard | multi | train/hard&multi&ng&nd&苏常州货068SUCHANGZHOUHUO&4&2&T_20220312_10_19_05_545125.jpg |
| 4 | 志远08 | 志远08 | True | 1.000000 | middle | multi | train/middle&multi&ng&nd&志远08&5&1&O_20210608_19_41_59_330375.jpg |
| 5 | 翔运989XIANGYUN | 翔运989XIANGYUN | True | 1.000000 | easy | single | train/easy&single&ng&nd&翔运989XIANGYUN&1&2&T_20220208_21_28_24_028152.jpg |

### 错误样本示例

| lmdb_index | label | pred | correct | confidence | quality | structure | source_path |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 2249 | 绍航集168绍兴港SHAOHANGJISHAOXINGGANG | 绍航集168绍兴港 | False | 0.956074 | middle | multi | train/middle&multi&ng&nd&绍航集168绍兴港SHAOHANGJISHAOXINGGANG&-5&4&T_20220410_10_08_33_354250.jpg |
| 7541 | 皖太和货339阜阳港WANTAIHEHUOFUYANGGANG | 皖太和货339阜阳港WANTAIHEHUOFUYANG | False | 0.992906 | middle | multi | train/middle&multi&ng&nd&皖太和货339阜阳港WANTAIHEHUOFUYANGGANG&5&2&T_20220210_13_48_15_464368.jpg |
| 13407 | 浙富阳货00938 | 浙富阳货00938ZHEFUYANGHUO | False | 0.999999 | easy | single | train/easy&single&g&nd&浙富阳货00938&6&3&O_20190311_17_10_38_031250.jpg |
| 15397 | 苏泗洪货1858SUSIHONGHUO | 苏泗洪货1858 | False | 0.953325 | hard | multi | train/hard&multi&ng&d&苏泗洪货1858SUSIHONGHUO&7&2&T_20220831_10_28_24_919125.jpg |
| 19828 | 苏盐货11965盐城港SUYANHUO | 苏盐货11965盐城港 | False | 0.987734 | middle | multi | train/middle&multi&ng&nd&苏盐货11965盐城港SUYANHUO&-1&1&T_20181207_12_32_14_209600.jpg |

## 11. Warnings / Limitations

- confidence 基于 greedy decode 路径上的 token 概率，不等于完整序列后验概率。
- nll 当前是预测路径上的 -sum(log p)，不是 GT teacher-forcing NLL。
- tokenizer.decode() 返回的概率序列会包含 EOS 概率，因此 pred_length 与概率序列长度不完全相同。


## 12. Recommended Next Step

下一步建议基于 `train_ocr_predictions.csv` 和 `SLP34K_lmdb_train_meta` 构造 `train_samples_meta.csv/jsonl` 与 `same-label top1-HQ pair_manifest_top1_hq.csv`。
