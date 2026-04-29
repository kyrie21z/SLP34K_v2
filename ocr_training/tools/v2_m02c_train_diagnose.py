#!/usr/bin/env python
import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate, to_absolute_path
from omegaconf import open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from strhub.models.utils import load_from_checkpoint


def readfile(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return "".join(f.read().splitlines())


def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def gpu_snapshot() -> str:
    return run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )


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


def make_config(args: argparse.Namespace):
    overrides = [
        "model=slp_mdiff",
        f"model.batch_size={args.batch_size}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        f"model.freeze_encoder={str(args.freeze_encoder).lower()}",
        f"model.denoise_steps={args.denoise_steps}",
        f"model.loss_mode={args.loss_mode}",
        f"model.init_encoder_from_baseline_ckpt={str(args.init_encoder_from_baseline_ckpt).lower()}",
        f"model.baseline_ckpt_path={args.baseline_ckpt_path}",
        f"trainer.gpus={1 if args.device == 'cuda' else 0}",
        "+trainer.accumulate_grad_batches=1",
    ]
    overrides.extend(args.override)
    with initialize(config_path="../configs", version_base="1.2"):
        cfg = compose(config_name="main", overrides=overrides)

    if os.path.isfile(cfg.model.charset_train):
        cfg.model.charset_train = readfile(cfg.model.charset_train)
        cfg.model.charset_test = cfg.model.charset_train

    with open_dict(cfg):
        cfg.data.root_dir = to_absolute_path(cfg.data.root_dir)
        cfg.data.augment = args.augment
    return cfg


def collect_probe_batch(cfg, num_probe_samples: int) -> Dict[str, Any]:
    datamodule = instantiate(cfg.data)
    dataset = datamodule.train_dataset
    images = []
    labels = []
    for idx in range(num_probe_samples):
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    batch = {
        "images": torch.stack(images, dim=0),
        "labels": labels,
        "data_root": cfg.data.root_dir,
        "train_dir": cfg.data.train_dir,
        "image_shape": list(torch.stack(images, dim=0).shape),
    }
    close_dataset_envs(dataset)
    del datamodule
    gc.collect()
    return batch


def repeat_ratio(ids: torch.Tensor) -> float:
    if ids.numel() <= 1:
        return 0.0
    return float((ids[1:] == ids[:-1]).float().mean().item())


def decode_probe(model: pl.LightningModule, probe: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    images = probe["images"].to(device)
    labels = probe["labels"]
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        probs = logits.softmax(-1)
        pred_probs, pred_ids = probs.max(-1)
        decoded, _ = model.tokenizer.decode(probs.detach().cpu())

    pred_ids_cpu = pred_ids.detach().cpu()
    eos_id = model.eos_id
    samples = []
    contains_eos = []
    pred_lens = []
    unique_counts = []
    repeat_ratios = []
    repeat_ratios_truncated = []
    eos_probs = probs[..., eos_id].detach().cpu()
    eos_ranks = (logits.detach().cpu() > logits.detach().cpu()[..., eos_id : eos_id + 1]).sum(-1) + 1

    for idx, ids in enumerate(pred_ids_cpu):
        eos_positions = (ids == eos_id).nonzero(as_tuple=False).flatten()
        has_eos = bool(len(eos_positions) > 0)
        trunc_end = int(eos_positions[0]) if has_eos else len(ids)
        truncated_ids = ids[: max(trunc_end, 1)]
        pred = decoded[idx]
        sample_repeat = repeat_ratio(ids)
        sample_repeat_trunc = repeat_ratio(truncated_ids)
        samples.append(
            {
                "gt": labels[idx],
                "pred": pred,
                "pred_len": len(pred),
                "contains_eos": has_eos,
                "repeat_ratio": sample_repeat,
                "repeat_ratio_truncated": sample_repeat_trunc,
                "unique_char_count": len(set(pred)),
                "eos_prob_mean": float(eos_probs[idx].mean().item()),
                "eos_rank_mean": float(eos_ranks[idx].float().mean().item()),
            }
        )
        contains_eos.append(has_eos)
        pred_lens.append(len(pred))
        unique_counts.append(len(set(pred)))
        repeat_ratios.append(sample_repeat)
        repeat_ratios_truncated.append(sample_repeat_trunc)

    return {
        "eos_rate": sum(contains_eos) / len(contains_eos),
        "avg_pred_len": sum(pred_lens) / len(pred_lens),
        "repeat_ratio": sum(repeat_ratios) / len(repeat_ratios),
        "repeat_ratio_truncated": sum(repeat_ratios_truncated) / len(repeat_ratios_truncated),
        "unique_char_count": sum(unique_counts) / len(unique_counts),
        "eos_prob_mean": float(eos_probs.mean().item()),
        "eos_rank_mean": float(eos_ranks.float().mean().item()),
        "logits_shape": list(logits.shape),
        "samples": samples,
    }


class DiagnoseCallback(Callback):
    def __init__(self, probe: Dict[str, Any], diagnose_every: int) -> None:
        self.probe = probe
        self.diagnose_every = diagnose_every
        self.losses: List[Dict[str, float]] = []
        self.diagnostics: List[Dict[str, Any]] = []

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        self._diagnose(trainer, pl_module, step=0, loss=None)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        loss_value: Optional[float] = None
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().float().cpu())
            self.losses.append({"step": trainer.global_step, "loss": loss_value})
            if trainer.global_step <= 3 or trainer.global_step % self.diagnose_every == 0:
                print(f"train_step={trainer.global_step} loss={loss_value:.6f}", flush=True)
        if trainer.global_step == 1 or trainer.global_step % self.diagnose_every == 0:
            self._diagnose(trainer, pl_module, step=trainer.global_step, loss=loss_value)

    def on_train_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if not self.diagnostics or self.diagnostics[-1]["step"] != trainer.global_step:
            last_loss = self.losses[-1]["loss"] if self.losses else None
            self._diagnose(trainer, pl_module, step=trainer.global_step, loss=last_loss)

    def _diagnose(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        step: int,
        loss: Optional[float],
    ) -> None:
        device = pl_module.device
        diag = decode_probe(pl_module, self.probe, device)
        row = {"step": step, "loss": loss, **diag}
        self.diagnostics.append(row)
        print(
            "diagnose "
            f"step={step} loss={loss} eos_rate={row['eos_rate']:.3f} "
            f"avg_pred_len={row['avg_pred_len']:.2f} repeat_ratio={row['repeat_ratio']:.3f} "
            f"unique_char_count={row['unique_char_count']:.2f} eos_prob_mean={row['eos_prob_mean']:.4f}",
            flush=True,
        )
        for sample_idx, sample in enumerate(row["samples"][:3], start=1):
            print(
                f"probe_sample={sample_idx} gt={sample['gt']} pred={sample['pred']} "
                f"pred_len={sample['pred_len']} contains_eos={sample['contains_eos']} "
                f"repeat_ratio={sample['repeat_ratio']:.3f}",
                flush=True,
            )
        pl_module.train()


def checkpoint_key_counts(model: pl.LightningModule, checkpoint_path: str) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(state.get("state_dict", {}).keys())
    model_keys = set(model.state_dict().keys())
    return {
        "state_dict_keys": len(ckpt_keys),
        "missing_key_count": len(model_keys - ckpt_keys),
        "unexpected_key_count": len(ckpt_keys - model_keys),
    }


def run_precheck(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    baseline_path = Path(args.baseline_ckpt_path)
    probe = collect_probe_batch(cfg, args.num_probe_samples)
    model = instantiate(cfg.model)
    encoded = model.tokenizer.encode(probe["labels"], torch.device("cpu"))
    state = torch.load(str(baseline_path), map_location="cpu", weights_only=False) if baseline_path.exists() else {}
    state_dict = state.get("state_dict", state) if state else {}
    encoder_keys = [key for key in state_dict if key.startswith("encoder.")]
    result = {
        "baseline_ckpt_path": str(baseline_path),
        "baseline_exists": baseline_path.exists(),
        "encoder_key_count": len(encoder_keys),
        "encoder_load_info": model.encoder_load_info.get("baseline_ckpt", {}),
        "freeze_encoder": args.freeze_encoder,
        "probe_image_shape": probe["image_shape"],
        "probe_labels": probe["labels"],
        "encoded_shape": list(encoded.shape),
        "tokenizer_len": len(model.tokenizer),
        "eos_id": model.eos_id,
        "pad_id": model.pad_id,
        "mask_id": model.mask_id,
        "head_out": model.output_num_classes,
    }
    print(json.dumps({"precheck": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_train(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).absolute()
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    probe = collect_probe_batch(cfg, args.num_probe_samples)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    callback = DiagnoseCallback(probe, args.diagnose_every)
    precision = int(args.precision) if str(args.precision).isdigit() else args.precision

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    trainer_kwargs = {
        "default_root_dir": str(output_dir),
        "max_steps": args.max_steps,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accumulate_grad_batches": 1,
        "log_every_n_steps": args.diagnose_every,
        "callbacks": [callback],
        "gradient_clip_val": args.gradient_clip_val,
        "gpus": 1 if args.device == "cuda" else 0,
        "precision": precision if args.device == "cuda" else 32,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    ckpt_path = ckpt_dir / f"slp_mdiff_{args.run_name}_last.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    peak_allocated_gb = None
    peak_reserved_gb = None
    if args.device == "cuda":
        cuda_device = torch.device("cuda:0")
        peak_allocated_gb = torch.cuda.max_memory_allocated(cuda_device) / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved(cuda_device) / 1024**3

    result = {
        "run_name": args.run_name,
        "max_steps": args.max_steps,
        "global_step": trainer.global_step,
        "batch_size": args.batch_size,
        "denoise_steps": args.denoise_steps,
        "loss_mode": args.loss_mode,
        "freeze_encoder": args.freeze_encoder,
        "init_encoder_from_baseline_ckpt": args.init_encoder_from_baseline_ckpt,
        "baseline_ckpt_path": args.baseline_ckpt_path,
        "precision": precision,
        "num_workers": args.num_workers,
        "losses": callback.losses,
        "diagnostics": callback.diagnostics,
        "losses_finite": all(math.isfinite(row["loss"]) for row in callback.losses),
        "first_loss": callback.losses[0]["loss"] if callback.losses else None,
        "final_loss": callback.losses[-1]["loss"] if callback.losses else None,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
        "checkpoint_size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else None,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }
    print(json.dumps({"training": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_load(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required for --mode load")
    model = load_from_checkpoint(args.checkpoint_path).eval()
    key_counts = checkpoint_key_counts(model, args.checkpoint_path)
    cfg = make_config(args)
    probe = collect_probe_batch(cfg, args.num_probe_samples)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    diag = decode_probe(model, probe, device)
    result = {
        "checkpoint_path": args.checkpoint_path,
        "load_from_checkpoint": True,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        **key_counts,
        "quick_inference": diag,
    }
    print(json.dumps({"checkpoint_load": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_debug_loss_positions(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    probe = collect_probe_batch(cfg, min(args.num_probe_samples, args.batch_size))
    model = instantiate(cfg.model)
    clean_targets = model.tokenizer.encode(probe["labels"], torch.device("cpu"))[:, 1:]
    noised, masked_positions, full_mask_rows = model._make_noised_inputs(
        clean_targets, return_full_mask_rows=True
    )
    non_pad_positions = clean_targets != model.pad_id
    eos_positions = clean_targets == model.eos_id
    loss_positions = model._get_loss_positions(clean_targets, masked_positions, full_mask_rows)
    target_ids = clean_targets[loss_positions]
    result = {
        "loss_mode": model.loss_mode,
        "labels": probe["labels"],
        "clean_ids": clean_targets.tolist(),
        "noised_ids": noised.tolist(),
        "full_mask_rows": full_mask_rows.tolist(),
        "eos_id": model.eos_id,
        "pad_id": model.pad_id,
        "mask_id": model.mask_id,
        "head_out": model.output_num_classes,
        "non_pad_count": int(non_pad_positions.sum().item()),
        "masked_count": int(masked_positions.sum().item()),
        "eos_target_count": int((clean_targets[non_pad_positions] == model.eos_id).sum().item()),
        "loss_position_count": int(loss_positions.sum().item()),
        "loss_positions": loss_positions.tolist(),
        "target_ids_at_loss_positions": target_ids.tolist(),
        "number_of_eos_targets": int((target_ids == model.eos_id).sum().item()),
        "number_of_pad_targets_in_loss": int((target_ids == model.pad_id).sum().item()),
        "number_of_targets_out_of_head_range": int((target_ids >= model.output_num_classes).sum().item()),
        "eos_target_ratio_in_loss": float((target_ids == model.eos_id).float().mean().item()) if target_ids.numel() else 0.0,
        "all_targets_in_head_range": bool(((target_ids >= 0) & (target_ids < model.output_num_classes)).all().item()) if target_ids.numel() else True,
    }
    print(json.dumps({"debug_loss_positions": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["precheck", "train", "load", "debug_loss_positions"], default="train")
    parser.add_argument("--run-name", default="stageA")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument(
        "--loss-mode",
        choices=["masked_or_eos", "all_non_pad", "full_mask_all_non_pad"],
        default="masked_or_eos",
    )
    parser.add_argument("--diagnose-every", type=int, default=100)
    parser.add_argument("--num-probe-samples", type=int, default=5)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-encoder-from-baseline-ckpt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--gradient-clip-val", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/V2-M02c_stageA_1000steps")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    cfg = make_config(args)
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "gpu_snapshot_before": gpu_snapshot(),
        "mode": args.mode,
        "args": vars(args),
    }

    if args.mode == "precheck":
        summary["precheck"] = run_precheck(cfg, args)
    elif args.mode == "train":
        summary["training"] = run_train(cfg, args)
    elif args.mode == "load":
        summary["checkpoint_load"] = run_load(args)
    elif args.mode == "debug_loss_positions":
        summary["debug_loss_positions"] = run_debug_loss_positions(cfg, args)

    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
