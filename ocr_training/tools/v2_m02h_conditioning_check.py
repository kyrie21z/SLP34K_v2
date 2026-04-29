#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.utils import instantiate, to_absolute_path
from omegaconf import open_dict


def readfile(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return "".join(f.read().splitlines())


def make_config(args: argparse.Namespace):
    overrides = [
        "model=slp_mdiff",
        f"model.batch_size={args.batch_size}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        f"model.decoder_core={args.decoder_core}",
        f"model.inference_mode={args.inference_mode}",
        f"model.loss_mode={args.loss_mode}",
        f"model.drop_cls_token={str(args.drop_cls_token).lower()}",
        f"model.visual_adapter_type={args.visual_adapter_type}",
        f"model.cross_attn_gate={str(args.cross_attn_gate).lower()}",
        f"model.cross_attn_gate_init={args.cross_attn_gate_init}",
        f"model.freeze_encoder={str(args.freeze_encoder).lower()}",
        f"model.init_encoder_from_baseline_ckpt={str(args.init_encoder_from_baseline_ckpt).lower()}",
        f"model.baseline_ckpt_path={args.baseline_ckpt_path}",
        f"trainer.gpus={1 if args.device == 'cuda' else 0}",
        "+trainer.accumulate_grad_batches=1",
    ]
    with initialize(config_path="../configs", version_base="1.2"):
        cfg = compose(config_name="main", overrides=overrides)
    if os.path.isfile(cfg.model.charset_train):
        cfg.model.charset_train = readfile(cfg.model.charset_train)
        cfg.model.charset_test = cfg.model.charset_train
    with open_dict(cfg):
        cfg.data.root_dir = to_absolute_path(cfg.data.root_dir)
        cfg.data.augment = False
    return cfg


def close_dataset_envs(dataset: Any) -> None:
    datasets = getattr(dataset, "datasets", None)
    if datasets is not None:
        for child in datasets:
            close_dataset_envs(child)
        return
    env = getattr(dataset, "_env", None)
    if env is not None:
        env.close()
        dataset._env = None


def collect_batch(cfg, batch_size: int) -> Dict[str, Any]:
    datamodule = instantiate(cfg.data)
    dataset = datamodule.train_dataset
    images = []
    labels = []
    for idx in range(batch_size):
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    close_dataset_envs(dataset)
    return {"images": torch.stack(images, dim=0), "labels": labels}


def mean_offdiag_cosine(x: torch.Tensor) -> float:
    if x.shape[0] <= 1:
        return 1.0
    x = F.normalize(x.float(), dim=-1)
    sim = x @ x.transpose(0, 1)
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return float(sim[mask].mean().detach().cpu().item())


def same_image_different_positions_cosine(logits: torch.Tensor) -> float:
    values = [mean_offdiag_cosine(sample) for sample in logits]
    return float(sum(values) / len(values))


def different_images_same_position_cosine(logits: torch.Tensor) -> float:
    values = [mean_offdiag_cosine(logits[:, pos, :]) for pos in range(logits.shape[1])]
    return float(sum(values) / len(values))


def compare_logits(reference: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    ref_top1 = reference.argmax(dim=-1)
    other_top1 = other.argmax(dim=-1)
    return {
        "mean_abs_diff": float((reference - other).abs().mean().detach().cpu().item()),
        "max_abs_diff": float((reference - other).abs().max().detach().cpu().item()),
        "top1_changed_rate": float((ref_top1 != other_top1).float().mean().detach().cpu().item()),
    }


def decode_samples(model: pl.LightningModule, logits: torch.Tensor, labels: List[str]) -> List[Dict[str, Any]]:
    probs = logits.softmax(-1).detach().cpu()
    preds, _ = model.tokenizer.decode(probs)
    top1 = probs.argmax(dim=-1)
    rows = []
    for idx, pred in enumerate(preds):
        rows.append(
            {
                "gt": labels[idx],
                "pred": pred,
                "pred_len": len(pred),
                "unique_char_count": len(set(pred)),
                "contains_eos": bool((top1[idx] == model.eos_id).any().item()),
            }
        )
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = make_config(args)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    batch = collect_batch(cfg, args.batch_size)
    model = instantiate(cfg.model)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = [], []
    model.to(device).eval()

    images = batch["images"].to(device)
    labels = batch["labels"]
    input_ids = torch.full(
        (images.shape[0], model.max_label_length + 1),
        model.mask_id,
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        raw_memory = model.encoder(images)
        memory = model.prepare_visual_memory(raw_memory)
        real_logits = model.head(model.decode(input_ids, memory))
        zero_logits = model.head(model.decode(input_ids, torch.zeros_like(memory)))
        shuffled_memory = memory.roll(shifts=1, dims=0)
        shuffled_logits = model.head(model.decode(input_ids, shuffled_memory))

        pos_embed = getattr(model.mdiff_decoder, "position_embed", None)
        if pos_embed is not None:
            saved_pos = pos_embed.detach().clone()
            pos_embed.zero_()
            zero_pos_logits = model.head(model.decode(input_ids, memory))
            pos_embed.copy_(saved_pos)
        else:
            zero_pos_logits = real_logits.clone()

    result = {
        "checkpoint_path": args.checkpoint_path,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "device": str(device),
        "image_shape": list(images.shape),
        "raw_memory_shape": list(raw_memory.shape),
        "memory_shape": list(memory.shape),
        "logits_shape": list(real_logits.shape),
        "drop_cls_token": model.drop_cls_token,
        "visual_adapter_type": model.visual_adapter_type,
        "visual_adapter_parameter_count": model.visual_adapter_parameter_count(),
        "cross_attn_gate": model.cross_attn_gate,
        "cross_attn_gate_init": model.cross_attn_gate_init,
        "same_image_different_positions_cosine": same_image_different_positions_cosine(real_logits),
        "different_images_same_position_cosine": different_images_same_position_cosine(real_logits),
        "real_vs_zero_memory": compare_logits(real_logits, zero_logits),
        "real_vs_shuffled_memory": compare_logits(real_logits, shuffled_logits),
        "position_embedding_zero_out": compare_logits(real_logits, zero_pos_logits),
        "samples": decode_samples(model, real_logits, labels),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decoder-core", choices=["plain", "official"], default="official")
    parser.add_argument("--inference-mode", choices=["parallel", "iterative_full_feedback"], default="parallel")
    parser.add_argument(
        "--loss-mode",
        choices=["masked_or_eos", "all_non_pad", "full_mask_all_non_pad", "official_masked_normalized"],
        default="official_masked_normalized",
    )
    parser.add_argument("--drop-cls-token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-adapter-type", choices=["identity", "layernorm", "linear_ln"], default="identity")
    parser.add_argument("--cross-attn-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cross-attn-gate-init", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-encoder-from-baseline-ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", default="outputs/V2-M02h_official_core_parallel_1000steps/conditioning_summary.json")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    result = run(args)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"conditioning": result, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
