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
        f"model.decoder_core={args.decoder_core}",
        f"model.inference_mode={args.inference_mode}",
        f"model.loss_mode={args.loss_mode}",
        f"model.drop_cls_token={str(args.drop_cls_token).lower()}",
        f"model.visual_adapter_type={args.visual_adapter_type}",
        f"model.cross_attn_gate={str(args.cross_attn_gate).lower()}",
        f"model.cross_attn_gate_init={args.cross_attn_gate_init}",
        f"model.freeze_encoder={str(args.freeze_encoder).lower()}",
        f"model.denoise_steps={args.denoise_steps}",
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
    stacked = torch.stack(images, dim=0)
    batch = {
        "images": stacked,
        "labels": labels,
        "data_root": cfg.data.root_dir,
        "train_dir": cfg.data.train_dir,
        "image_shape": list(stacked.shape),
    }
    close_dataset_envs(dataset)
    del datamodule
    gc.collect()
    return batch


def first_train_batch(cfg) -> Dict[str, Any]:
    datamodule = instantiate(cfg.data)
    loader = datamodule.train_dataloader()
    images, labels = next(iter(loader))
    batch = {
        "images": images,
        "labels": labels,
        "image_shape": list(images.shape),
        "label_examples": list(labels[: min(3, len(labels))]),
    }
    close_dataset_envs(datamodule.train_dataset)
    del loader, datamodule
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
        _, pred_ids = probs.max(-1)
        decoded, _ = model.tokenizer.decode(probs.detach().cpu())

    pred_ids_cpu = pred_ids.detach().cpu()
    eos_id = model.eos_id
    samples = []
    contains_eos = []
    pred_lens = []
    unique_counts = []
    repeat_ratios = []
    all_same = []
    eos_probs = probs[..., eos_id].detach().cpu()
    logits_cpu = logits.detach().cpu()
    eos_ranks = (logits_cpu > logits_cpu[..., eos_id : eos_id + 1]).sum(-1) + 1

    for idx, ids in enumerate(pred_ids_cpu):
        eos_positions = (ids == eos_id).nonzero(as_tuple=False).flatten()
        has_eos = bool(len(eos_positions) > 0)
        pred = decoded[idx]
        sample_repeat = repeat_ratio(ids)
        sample_all_same = bool((ids == ids[0]).all().item()) if ids.numel() else False
        samples.append(
            {
                "gt": labels[idx],
                "pred": pred,
                "pred_len": len(pred),
                "contains_eos": has_eos,
                "repeat_ratio": sample_repeat,
                "unique_char_count": len(set(pred)),
                "eos_prob_mean": float(eos_probs[idx].mean().item()),
                "eos_rank_mean": float(eos_ranks[idx].float().mean().item()),
                "all_positions_same_top1": sample_all_same,
            }
        )
        contains_eos.append(has_eos)
        pred_lens.append(len(pred))
        unique_counts.append(len(set(pred)))
        repeat_ratios.append(sample_repeat)
        all_same.append(sample_all_same)

    return {
        "eos_rate": sum(contains_eos) / len(contains_eos),
        "avg_pred_len": sum(pred_lens) / len(pred_lens),
        "repeat_ratio": sum(repeat_ratios) / len(repeat_ratios),
        "unique_char_count": sum(unique_counts) / len(unique_counts),
        "eos_prob_mean": float(eos_probs.mean().item()),
        "eos_rank_mean": float(eos_ranks.float().mean().item()),
        "all_positions_same_top1_ratio": sum(all_same) / len(all_same),
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
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
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
        row = {"step": step, "loss": loss, **decode_probe(pl_module, self.probe, pl_module.device)}
        self.diagnostics.append(row)
        print(
            "diagnose "
            f"step={step} loss={loss} eos_rate={row['eos_rate']:.3f} "
            f"avg_pred_len={row['avg_pred_len']:.2f} repeat_ratio={row['repeat_ratio']:.3f} "
            f"unique_char_count={row['unique_char_count']:.2f} eos_prob_mean={row['eos_prob_mean']:.4f} "
            f"all_same={row['all_positions_same_top1_ratio']:.3f}",
            flush=True,
        )
        for sample_idx, sample in enumerate(row["samples"][:3], start=1):
            print(
                f"probe_sample={sample_idx} gt={sample['gt']} pred={sample['pred']} "
                f"pred_len={sample['pred_len']} contains_eos={sample['contains_eos']} "
                f"repeat_ratio={sample['repeat_ratio']:.3f} unique={sample['unique_char_count']}",
                flush=True,
            )
        pl_module.train()


def run_precheck(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    model = instantiate(cfg.model)
    result = {
        "decoder_core": model.decoder_core,
        "inference_mode": model.inference_mode,
        "loss_mode": model.loss_mode,
        "tokenizer_len": len(model.tokenizer),
        "eos_id": model.eos_id,
        "bos_id": model.bos_id,
        "pad_id": model.pad_id,
        "mask_id": model.mask_id,
        "input_vocab_size": model.input_vocab_size,
        "output_num_classes": model.output_num_classes,
        "head_bias": model.head.bias is not None,
        "drop_cls_token": model.drop_cls_token,
        "visual_adapter_type": model.visual_adapter_type,
        "visual_adapter_parameter_count": model.visual_adapter_parameter_count(),
        "cross_attn_gate": model.cross_attn_gate,
        "cross_attn_gate_init": model.cross_attn_gate_init,
        "encoder_load_info": model.encoder_load_info,
    }
    print(json.dumps({"precheck": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_dummy_forward(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model).to(device).eval()
    images = torch.zeros(1, 3, cfg.model.img_size[0], cfg.model.img_size[1], device=device)
    with torch.inference_mode():
        logits = model(images)
    result = {
        "device": str(device),
        "logits_shape": list(logits.shape),
        "expected_shape": [1, cfg.model.max_label_length + 1, model.output_num_classes],
        "finite": bool(torch.isfinite(logits).all().item()),
    }
    print(json.dumps({"dummy_forward": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_real_batch(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    batch = first_train_batch(cfg)
    model = instantiate(cfg.model).to(device)
    model.train()
    images = batch["images"].to(device)
    labels = batch["labels"]
    _, loss, loss_numel = model._denoising_logits_loss(images, labels)
    loss.backward()
    debug = dict(model.last_loss_debug)
    result = {
        "device": str(device),
        "image_shape": batch["image_shape"],
        "label_examples": batch["label_examples"],
        "loss": float(loss.detach().float().cpu()),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "loss_numel": int(loss_numel.detach().cpu().item()) if isinstance(loss_numel, torch.Tensor) else int(loss_numel),
        "loss_debug": debug,
    }
    print(json.dumps({"real_batch": result}, ensure_ascii=False, indent=2), flush=True)
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
        "decoder_core": args.decoder_core,
        "inference_mode": args.inference_mode,
        "loss_mode": args.loss_mode,
        "drop_cls_token": args.drop_cls_token,
        "visual_adapter_type": args.visual_adapter_type,
        "visual_adapter_parameter_count": model.visual_adapter_parameter_count(),
        "cross_attn_gate": args.cross_attn_gate,
        "cross_attn_gate_init": args.cross_attn_gate_init,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["precheck", "dummy_forward", "real_batch", "train"], default="train")
    parser.add_argument("--run-name", default="official_core_parallel_1000steps")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--denoise-steps", type=int, default=3)
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
    parser.add_argument("--diagnose-every", type=int, default=100)
    parser.add_argument("--num-probe-samples", type=int, default=5)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-encoder-from-baseline-ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--gradient-clip-val", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/V2-M02h_official_core_parallel_1000steps")
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
    elif args.mode == "dummy_forward":
        summary["dummy_forward"] = run_dummy_forward(cfg, args)
    elif args.mode == "real_batch":
        summary["real_batch"] = run_real_batch(cfg, args)
    elif args.mode == "train":
        summary["training"] = run_train(cfg, args)

    summary["gpu_snapshot_after"] = gpu_snapshot()
    summary_path = output_dir / f"{args.run_name}_{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
