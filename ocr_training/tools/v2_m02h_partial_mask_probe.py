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
from hydra import compose, initialize
from hydra.utils import instantiate, to_absolute_path
from omegaconf import open_dict

from tools.v2_m02h_conditioning_check import collect_batch, readfile


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
        f"model.freeze_encoder={str(args.freeze_encoder).lower()}",
        f"model.init_encoder_from_baseline_ckpt={str(args.init_encoder_from_baseline_ckpt).lower()}",
        f"model.baseline_ckpt_path={args.baseline_ckpt_path}",
        f"trainer.gpus={1 if args.device == 'cuda' else 0}",
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


def repeat_ratio(ids: torch.Tensor) -> float:
    if ids.numel() <= 1:
        return 0.0
    return float((ids[1:] == ids[:-1]).float().mean().item())


def decode_rows(model, probs: torch.Tensor, labels: List[str], top1: torch.Tensor) -> List[Dict[str, Any]]:
    preds, _ = model.tokenizer.decode(probs.detach().cpu())
    rows = []
    for idx, pred in enumerate(preds):
        ids = top1[idx].detach().cpu()
        rows.append(
            {
                "gt": labels[idx],
                "pred": pred,
                "pred_len": len(pred),
                "contains_eos": bool((ids == model.eos_id).any().item()),
                "repeat_ratio": repeat_ratio(ids),
                "unique_char_count": len(set(pred)),
            }
        )
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = make_config(args)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    batch = collect_batch(cfg, args.batch_size)
    model = instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    images = batch["images"].to(device)
    labels = batch["labels"]
    with torch.inference_mode():
        memory = model.encode(images)
        clean_targets = model.tokenizer.encode(labels, device)[:, 1:]
        valid = clean_targets != model.pad_id
        position_ids = torch.arange(clean_targets.shape[1], device=device)[None, :]
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        keep_limit = (valid_count.float() * args.keep_ratio).floor().long().clamp_min(1)
        keep_mask = valid & (position_ids < keep_limit)

        partial_ids = clean_targets.clone()
        partial_ids[valid & ~keep_mask] = model.mask_id
        full_ids = torch.full_like(clean_targets, model.mask_id)

        full_logits = model.head(model.decode(full_ids, memory))
        partial_logits = model.head(model.decode(partial_ids, memory))
        full_probs = full_logits.softmax(-1)
        partial_probs = partial_logits.softmax(-1)
        full_top1 = full_probs.argmax(dim=-1)
        partial_top1 = partial_probs.argmax(dim=-1)

    full_rows = decode_rows(model, full_probs, labels, full_top1)
    partial_rows = decode_rows(model, partial_probs, labels, partial_top1)
    result = {
        "checkpoint_path": args.checkpoint_path,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "device": str(device),
        "keep_ratio": args.keep_ratio,
        "drop_cls_token": model.drop_cls_token,
        "visual_adapter_type": model.visual_adapter_type,
        "memory_shape": list(memory.shape),
        "full_mask": full_rows,
        "partial_mask": partial_rows,
        "full_avg_pred_len": sum(row["pred_len"] for row in full_rows) / len(full_rows),
        "partial_avg_pred_len": sum(row["pred_len"] for row in partial_rows) / len(partial_rows),
        "full_avg_unique": sum(row["unique_char_count"] for row in full_rows) / len(full_rows),
        "partial_avg_unique": sum(row["unique_char_count"] for row in partial_rows) / len(partial_rows),
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
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-encoder-from-baseline-ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    result = run(args)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"partial_mask_probe": result, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
