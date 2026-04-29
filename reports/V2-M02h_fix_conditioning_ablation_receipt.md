# V2-M02h-fix Conditioning Ablation Receipt

## 1. Summary

- 已实现 `drop_cls_token`：在 `Model.prepare_visual_memory()` 中对 SLP MAE raw memory `[B, 197, 768]` 可选丢弃 CLS，输出 `[B, 196, 768]`。
- 已实现 `visual_adapter_type=identity|layernorm|linear_ln`：`linear_ln` 为 `Linear(768,768)+LayerNorm(768)`，参数量 592,128；`identity` 参数量 0。
- 已补充可选 `cross_attn_gate` / `cross_attn_gate_init` 和每层 norm diagnostics 记录，但本轮实验保持 `cross_attn_gate=false`，避免引入额外变量。
- H1/H2/H3 均完成 1000-step single-GPU short training 和 final checkpoint conditioning diagnostic。
- 最好的 conditioning 指标来自 H2/H3 的 `linear_ln`：`different_images_same_position_cosine` 从 V2-M02h/H1 的约 0.9957 降到约 0.964，`real_vs_shuffled.top1_changed_rate` 从 0.149 提升到 0.212。
- 但 H1/H2/H3 均未缓解 EOS/repetition collapse：H1 从 step 100 起 `eos_rate=1.0`，H2/H3 从 step 200 起 `eos_rate=1.0`；final 仍为空串或极短重复片段。
- 不建议进入 V2-M03。下一步应进入 **V2-M02h-fix2：进一步修 visual memory interface / generation diagnostic**，优先做 teacher-forcing/partial-mask conditioning 与 cross-attn 使用强度诊断，而不是加 TRN 或 LC/BLC。

## 2. Files Changed

- `ocr_training/strhub/models/slp_mdiff/modules.py`
- `ocr_training/strhub/models/slp_mdiff/system.py`
- `ocr_training/configs/model/slp_mdiff.yaml`
- `ocr_training/tools/v2_m02h_official_core_train_check.py`
- `ocr_training/tools/v2_m02h_conditioning_check.py`
- `ocr_training/tools/v2_m02h_fix_conditioning_ablation.py`
- `ocr_training/tools/v2_m02h_partial_mask_probe.py`
- `reports/V2-M02h_fix_conditioning_ablation_receipt.md`

未修改 `configs/main.yaml`，未修改原 `maevit_infonce_plm/system.py` 行为，未删除数据，未进入 V2-M03。

## 3. Implementation Details

`drop_cls_token` 实现在 `ocr_training/strhub/models/slp_mdiff/system.py:215` 的 `prepare_visual_memory()`。流程为：检查 raw memory 必须是 `[B,S,768]`；若 `drop_cls_token=true`，要求 `S>=2` 并执行 `memory[:, 1:, :]`；随后进入 `self.visual_adapter`；最后再次检查 batch size 和 feature dim 未改变。

`visual_adapter_type` 实现在 `ocr_training/strhub/models/slp_mdiff/modules.py:19`：

- `identity`: `nn.Identity()`，参数量 0。
- `layernorm`: `nn.LayerNorm(768)`，参数量 1,536。本轮未训练该消融，因为实验矩阵要求优先 H1/H2/H3。
- `linear_ln`: `nn.Sequential(nn.Linear(768,768), nn.LayerNorm(768))`，参数量 592,128。

cross-attn gate 以可选参数接入 `OfficialStyleTransformerBlock`，但 H1/H2/H3 均使用 `cross_attn_gate=false`。本轮 gate 只作为后续控制 residual 强度的预留诊断开关，没有改变默认路径。

Plain decoder 仍保留，`decoder_core: plain|official` 仍可切换。

## 4. Experiment Settings

| run | drop_cls_token | visual_adapter_type | max_steps | batch_size | freeze_encoder | peak_memory_reserved |
|---|---:|---|---:|---:|---:|---:|
| H1 | true | identity | 1000 | 2 | true | 1.4570 GB |
| H2 | false | linear_ln | 1000 | 2 | true | 1.4766 GB |
| H3 | true | linear_ln | 1000 | 2 | true | 1.4766 GB |

共同设置：single GPU，precision=16，num_workers=0，`decoder_core=official`，`inference_mode=parallel`，`loss_mode=official_masked_normalized`，`init_encoder_from_baseline_ckpt=true`，baseline ckpt 为 `checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt`。

## 5. Training Diagnostics

| run | step | loss | eos_rate | avg_pred_len | repeat_ratio | unique_char_count | eos_prob_mean | eos_rank_mean | all_positions_same_top1_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0 | - | 0.000 | 51.00 | 0.672 | 8.40 | 0.0018 | 192.06 | 0.000 |
| H1 | 1 | 5.9456 | 0.000 | 51.00 | 0.672 | 8.40 | 0.0018 | 192.06 | 0.000 |
| H1 | 100 | 6.0849 | 1.000 | 0.00 | 0.776 | 0.00 | 0.0163 | 2.22 | 0.000 |
| H1 | 200 | 4.7981 | 1.000 | 0.00 | 0.896 | 0.00 | 0.0729 | 1.19 | 0.000 |
| H1 | 500 | 4.0547 | 1.000 | 5.00 | 0.732 | 2.40 | 0.0729 | 2.07 | 0.000 |
| H1 | 1000 | 4.2830 | 1.000 | 2.80 | 0.732 | 1.40 | 0.0815 | 1.75 | 0.000 |
| H2 | 0 | - | 0.000 | 51.00 | 0.692 | 7.80 | 0.0007 | 384.95 | 0.000 |
| H2 | 1 | 6.6824 | 0.000 | 51.00 | 0.692 | 7.80 | 0.0007 | 384.53 | 0.000 |
| H2 | 100 | 5.4903 | 0.000 | 51.00 | 0.796 | 5.80 | 0.0071 | 21.95 | 0.000 |
| H2 | 200 | 5.2786 | 1.000 | 0.00 | 0.968 | 0.00 | 0.0873 | 1.04 | 0.200 |
| H2 | 500 | 4.3065 | 1.000 | 0.00 | 0.896 | 0.00 | 0.0995 | 1.24 | 0.200 |
| H2 | 1000 | 4.4154 | 1.000 | 1.40 | 0.896 | 0.40 | 0.1011 | 1.32 | 0.200 |
| H3 | 0 | - | 0.000 | 51.00 | 0.684 | 7.80 | 0.0007 | 386.00 | 0.000 |
| H3 | 1 | 6.6836 | 0.000 | 51.00 | 0.684 | 7.80 | 0.0007 | 385.60 | 0.000 |
| H3 | 100 | 5.4930 | 0.000 | 51.00 | 0.784 | 6.00 | 0.0070 | 22.14 | 0.000 |
| H3 | 200 | 5.2782 | 1.000 | 0.00 | 0.968 | 0.00 | 0.0872 | 1.04 | 0.200 |
| H3 | 500 | 4.3052 | 1.000 | 0.00 | 0.896 | 0.00 | 0.0995 | 1.24 | 0.200 |
| H3 | 1000 | 4.4142 | 1.000 | 1.40 | 0.896 | 0.40 | 0.1011 | 1.32 | 0.200 |

## 6. Conditioning Diagnostics

| run | memory_shape | same_pos_cos | diff_img_same_pos_cos | real_zero_diff | real_zero_top1_change | real_shuffle_diff | real_shuffle_top1_change | pos_zero_diff | pos_zero_top1_change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-M02h baseline | [5,197,768] | 0.9589 | 0.9957 | 0.1046 | 0.2784 | 0.0835 | 0.1490 | 0.0097 | 0.0235 |
| H1 | [5,196,768] | 0.9589 | 0.9958 | 0.1044 | 0.2784 | 0.0829 | 0.1490 | 0.0097 | 0.0235 |
| H2 | [5,197,768] | 0.9638 | 0.9642 | 0.4001 | 0.2392 | 0.2544 | 0.2118 | 0.0092 | 0.0078 |
| H3 | [5,196,768] | 0.9638 | 0.9645 | 0.4006 | 0.2392 | 0.2535 | 0.2118 | 0.0092 | 0.0078 |

结论：丢弃 CLS 本身几乎没有影响；`linear_ln` 确实增强了 logits 对 memory 内容和 shuffle 的敏感性，但仍未形成有效 OCR 条件生成。

## 7. Probe Predictions

### H1 final

| GT | Pred | pred_len | contains_eos | unique |
|---|---|---:|---:|---:|
| 浙萧山货23765 | AA | 2 | true | 1 |
| 六安港LUANGANG | AA | 2 | true | 1 |
| 苏常州货068SUCHANGZHOUHUO | AAA0NNN | 7 | true | 3 |
| 志远08 | AA | 2 | true | 1 |
| 翔运989XIANGYUN | A | 1 | true | 1 |

### H2 final

| GT | Pred | pred_len | contains_eos | unique |
|---|---|---:|---:|---:|
| 浙萧山货23765 |  | 0 | true | 0 |
| 六安港LUANGANG |  | 0 | true | 0 |
| 苏常州货068SUCHANGZHOUHUO |  | 0 | true | 0 |
| 志远08 |  | 0 | true | 0 |
| 翔运989XIANGYUN | GGGHHHH | 7 | true | 2 |

### H3 final

| GT | Pred | pred_len | contains_eos | unique |
|---|---|---:|---:|---:|
| 浙萧山货23765 |  | 0 | true | 0 |
| 六安港LUANGANG |  | 0 | true | 0 |
| 苏常州货068SUCHANGZHOUHUO |  | 0 | true | 0 |
| 志远08 |  | 0 | true | 0 |
| 翔运989XIANGYUN | GGGHHHH | 7 | true | 2 |

## 8. Partial-mask Probe

因为 H1/H2/H3 均 collapse，补做了 H3 final checkpoint 的 no-train partial-mask probe：保留每条样本前 50% non-PAD GT token，其余置 MASK。

| probe | avg_pred_len | avg_unique |
|---|---:|---:|
| full-mask | 0.00 | 0.00 |
| partial-mask keep_ratio=0.5 | 1.40 | 1.00 |

示例：

| GT | full-mask Pred | partial-mask Pred |
|---|---|---|
| 浙萧山货23765 |  | 货 |
| 六安港LUANGANG |  | NN0N |
| 苏常州货068SUCHANGZHOUHUO |  | 1 |
| 志远08 |  |  |
| 翔运989XIANGYUN |  | H |

partial-mask 只带来极弱改善，说明 decoder 不是完全无法使用 GT text context，但当前 visual-conditioned full-mask/partial-mask 恢复都很弱，仍不支持进入 V2-M03。

## 9. Analysis

1. CLS token 不是主要问题。H1 与 V2-M02h baseline 的 conditioning 数值几乎一致，输出也同样 collapse。
2. identity adapter 不足，但不是唯一问题。H2/H3 的 `linear_ln` 明显降低 different-image cosine，并提高 real-vs-shuffle sensitivity，说明 adapter 让 visual memory 更能影响 logits。
3. `linear_ln` 增强了 visual conditioning，但没有转化为可用解码。H2/H3 在 step 200 后仍全 EOS，final 仍为空或短重复。
4. different-image cosine 下降明显：H2/H3 约 0.964 vs baseline/H1 约 0.996。
5. real-vs-shuffle memory sensitivity 增强：top1_changed_rate 约 0.212 vs baseline/H1 0.149。
6. collapse 未缓解。H2/H3 的 `all_positions_same_top1_ratio=0.2` 还比 H1 更差，说明 adapter 增强影响后仍可能落入全 EOS/同质化 basin。
7. 不可以进入 V2-M03。当前证据更支持继续修 V2-M02h-fix2：检查 cross-attn 是否读取到可判别 OCR spatial pattern、是否需要 encoder memory reshape/pooling、是否需要 teacher-forcing diagnostic 或更强 image-text alignment probe。

## 10. Recommendation

推荐顺序：

1. **V2-M02h-fix2：进一步修 visual memory interface**。最小动作：做 teacher-forcing / partial-mask-by-position diagnostic，并记录 cross-attn attention entropy、cross-attn residual norm、real/shuffled memory 下 CE 差异；必要时测试 patch-token reshape 后的 2D spatial adapter 或 OCR neck。
2. **V2-M02i：实现 official LC/BLC inference**。仅在确认 decoder 已能在 teacher-forcing/partial-mask 下读图后再做。
3. **V2-M02f：EOS bias / length penalty**。可作为症状控制，但当前不是首要根因。
4. **V2-M03：all-mask strategies + generic TRN**。当前不建议进入，因为 H1/H2/H3 未输出稳定非空、非强重复、图像相关文本。
5. 暂停并重审架构。若 V2-M02h-fix2 仍证明 MAE memory 不适合直接做 OCR decoder memory，则应考虑该项。

## 11. Appendix

### Verification

- `python -m py_compile ocr_training/strhub/models/slp_mdiff/modules.py ocr_training/strhub/models/slp_mdiff/system.py ocr_training/tools/v2_m02h_official_core_train_check.py ocr_training/tools/v2_m02h_conditioning_check.py ocr_training/tools/v2_m02h_fix_conditioning_ablation.py`
- `python -m py_compile ocr_training/tools/v2_m02h_partial_mask_probe.py`
- Hydra precheck summary: `ocr_training/outputs/V2-M02h_fix_precheck/V2M02h_fix_precheck_precheck_summary.json`

### Checkpoint and JSON Paths

- H1 checkpoint: `ocr_training/outputs/V2-M02h_fix_H1_drop_cls_identity_1000steps/checkpoints/slp_mdiff_H1_drop_cls_identity_1000steps_last.ckpt`
- H1 training summary: `ocr_training/outputs/V2-M02h_fix_H1_drop_cls_identity_1000steps/H1_drop_cls_identity_1000steps_train_summary.json`
- H1 conditioning: `ocr_training/outputs/V2-M02h_fix_H1_drop_cls_identity_1000steps/conditioning_summary.json`
- H2 checkpoint: `ocr_training/outputs/V2-M02h_fix_H2_keep_cls_linear_ln_1000steps/checkpoints/slp_mdiff_H2_keep_cls_linear_ln_1000steps_last.ckpt`
- H2 training summary: `ocr_training/outputs/V2-M02h_fix_H2_keep_cls_linear_ln_1000steps/H2_keep_cls_linear_ln_1000steps_train_summary.json`
- H2 conditioning: `ocr_training/outputs/V2-M02h_fix_H2_keep_cls_linear_ln_1000steps/conditioning_summary.json`
- H3 checkpoint: `ocr_training/outputs/V2-M02h_fix_H3_drop_cls_linear_ln_1000steps/checkpoints/slp_mdiff_H3_drop_cls_linear_ln_1000steps_last.ckpt`
- H3 training summary: `ocr_training/outputs/V2-M02h_fix_H3_drop_cls_linear_ln_1000steps/H3_drop_cls_linear_ln_1000steps_train_summary.json`
- H3 conditioning: `ocr_training/outputs/V2-M02h_fix_H3_drop_cls_linear_ln_1000steps/conditioning_summary.json`
- H3 partial-mask probe: `ocr_training/outputs/V2-M02h_fix_H3_drop_cls_linear_ln_1000steps/partial_mask_probe.json`

### Actual Commands

H1:

```bash
python tools/v2_m02h_fix_conditioning_ablation.py --mode train --device cuda --run-name H1_drop_cls_identity_1000steps --batch-size 2 --num-workers 0 --max-steps 1000 --decoder-core official --inference-mode parallel --loss-mode official_masked_normalized --drop-cls-token --visual-adapter-type identity --freeze-encoder --init-encoder-from-baseline-ckpt --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt --diagnose-every 100 --num-probe-samples 5 --precision 16 --output-dir outputs/V2-M02h_fix_H1_drop_cls_identity_1000steps
```

H2:

```bash
python tools/v2_m02h_fix_conditioning_ablation.py --mode train --device cuda --run-name H2_keep_cls_linear_ln_1000steps --batch-size 2 --num-workers 0 --max-steps 1000 --decoder-core official --inference-mode parallel --loss-mode official_masked_normalized --visual-adapter-type linear_ln --freeze-encoder --init-encoder-from-baseline-ckpt --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt --diagnose-every 100 --num-probe-samples 5 --precision 16 --output-dir outputs/V2-M02h_fix_H2_keep_cls_linear_ln_1000steps
```

H3:

```bash
python tools/v2_m02h_fix_conditioning_ablation.py --mode train --device cuda --run-name H3_drop_cls_linear_ln_1000steps --batch-size 2 --num-workers 0 --max-steps 1000 --decoder-core official --inference-mode parallel --loss-mode official_masked_normalized --drop-cls-token --visual-adapter-type linear_ln --freeze-encoder --init-encoder-from-baseline-ckpt --baseline-ckpt-path checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt --diagnose-every 100 --num-probe-samples 5 --precision 16 --output-dir outputs/V2-M02h_fix_H3_drop_cls_linear_ln_1000steps
```

### Key Code Snippets

`system.py` 新增配置入口：

```text
38  drop_cls_token: bool = False,
39  cross_attn_gate: bool = False,
40  cross_attn_gate_init: float = 1.0,
66  if visual_adapter_type not in {"identity", "layernorm", "linear_ln"}:
78  self.drop_cls_token = drop_cls_token
79  self.visual_adapter_type = visual_adapter_type
```

`system.py` memory preprocessing：

```text
215 def prepare_visual_memory(self, memory: Tensor) -> Tensor:
216     if memory.ndim != 3:
218     if memory.shape[-1] != self.embed_dim:
221     if self.drop_cls_token:
224         memory = memory[:, 1:, :]
225     memory = self.visual_adapter(memory)
230     if memory.shape[-1] != self.embed_dim:
232     return memory
```

`modules.py` adapter：

```text
23 if adapter_type == "identity":
26     self.adapter = nn.Identity()
27 elif adapter_type == "layernorm":
30     self.adapter = nn.LayerNorm(out_dim)
31 elif adapter_type == "linear_ln":
32     self.adapter = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))
```

`modules.py` cross-attn diagnostics:

```text
186 hidden2 = self.cross_attn(hidden, memory, memory, need_weights=False)[0]
187 cross_residual = hidden2 * self.cross_attn_gate if self.use_cross_attn_gate else hidden2
188 hidden = self.norm2(hidden + self.dropout2(cross_residual))
192 self.last_diagnostics = {
193     "self_attn_output_norm": ...
194     "cross_attn_output_norm": ...
195     "ffn_output_norm": ...
196     "hidden_norm": ...
```

### Error and Fix Log

- 未遇到代码语法错误；`py_compile` 通过。
- Hydra precheck 通过，`drop_cls_token=true + linear_ln` 可实例化，adapter 参数量 592,128。
- 三个 1000-step run 均完成，无 shape/index/device error，peak GPU reserved 均小于 40GB。
- conditioning diagnostic checkpoint load 均为 `missing_key_count=0`、`unexpected_key_count=0`。
