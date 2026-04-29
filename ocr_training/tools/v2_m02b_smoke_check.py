#!/usr/bin/env python
import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

OCR_ROOT = Path(__file__).resolve().parents[1]
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, open_dict
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


def make_config(args: argparse.Namespace):
    overrides = [
        "model=slp_mdiff",
        f"model.batch_size={args.batch_size}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        f"model.freeze_encoder={str(args.freeze_encoder).lower()}",
        f"model.denoise_steps={args.denoise_steps}",
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


def tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).item())


class LossMemoryCallback(Callback):
    def __init__(self, log_every: int) -> None:
        self.log_every = log_every
        self.losses: List[float] = []

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().float().cpu())
            self.losses.append(loss_value)
            step = trainer.global_step
            if step <= 3 or step % self.log_every == 0:
                print(f"train_step={step} loss={loss_value:.6f}", flush=True)


def inspect_batch(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    datamodule = instantiate(cfg.data)
    loader = datamodule.train_dataloader()
    images, labels = next(iter(loader))
    model = instantiate(cfg.model)
    encoded = model.tokenizer.encode(labels, torch.device("cpu"))
    result = {
        "data_root": cfg.data.root_dir,
        "train_dir": cfg.data.train_dir,
        "batch_size": args.batch_size,
        "image_shape": list(images.shape),
        "label_examples": list(labels[: min(3, len(labels))]),
        "encoded_shape": list(encoded.shape),
        "tokenizer_len": len(model.tokenizer),
        "mask_id": model.mask_id,
        "head_out": model.output_num_classes,
    }
    print(json.dumps({"data_loading": result}, ensure_ascii=False, indent=2), flush=True)
    close_dataset_envs(datamodule.train_dataset)
    del loader, datamodule, model
    gc.collect()
    return result


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


def train_smoke(cfg, args: argparse.Namespace, data_info: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).absolute()
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    callback = LossMemoryCallback(args.log_every)

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
        "log_every_n_steps": args.log_every,
        "callbacks": [callback],
        "gradient_clip_val": args.gradient_clip_val,
    }
    if args.device == "cuda":
        precision = int(args.precision) if str(args.precision).isdigit() else args.precision
        trainer_kwargs.update({"gpus": 1, "precision": precision})
    else:
        trainer_kwargs.update({"gpus": 0, "precision": 32})

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

    losses = callback.losses
    finite_losses = all(math.isfinite(loss) for loss in losses)
    result = {
        "run_name": args.run_name,
        "max_steps": args.max_steps,
        "global_step": trainer.global_step,
        "freeze_encoder": args.freeze_encoder,
        "denoise_steps": args.denoise_steps,
        "batch_size": args.batch_size,
        "first_loss": losses[0] if losses else None,
        "mid_loss": losses[len(losses) // 2] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_count": len(losses),
        "losses_finite": finite_losses,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "data_info": data_info,
    }
    print(json.dumps({"training": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def check_checkpoint_load(checkpoint_path: str) -> Dict[str, Any]:
    model = load_from_checkpoint(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(state.get("state_dict", {}).keys())
    model_keys = set(model.state_dict().keys())
    load_result = {
        "checkpoint_path": checkpoint_path,
        "load_from_checkpoint": True,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "state_dict_keys": len(state.get("state_dict", {})),
        "missing_key_count": len(model_keys - ckpt_keys),
        "unexpected_key_count": len(ckpt_keys - model_keys),
    }
    print(json.dumps({"checkpoint_load": load_result}, ensure_ascii=False, indent=2), flush=True)
    return load_result


def check_baseline_migration(cfg, baseline_ckpt_path: str) -> Dict[str, Any]:
    path = Path(baseline_ckpt_path)
    if not path.exists():
        result = {"baseline_ckpt_path": baseline_ckpt_path, "exists": False, "verified": False}
        print(json.dumps({"baseline_migration": result}, ensure_ascii=False, indent=2), flush=True)
        return result

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    encoder_keys = [key for key in state_dict if key.startswith("encoder.")]

    with open_dict(cfg):
        cfg.model.init_encoder_from_baseline_ckpt = True
        cfg.model.baseline_ckpt_path = str(path)
        cfg.model.freeze_encoder = True
    model = instantiate(cfg.model)
    info = model.encoder_load_info.get("baseline_ckpt", {})
    result = {
        "baseline_ckpt_path": str(path),
        "exists": True,
        "encoder_key_count": len(encoder_keys),
        "verified": True,
        "load_info": info,
    }
    print(json.dumps({"baseline_migration": result}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--mode", choices=["inspect", "train", "load", "baseline"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--gradient-clip-val", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/V2-M02b_slp_mdiff_smoke")
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)
    cfg = make_config(args)

    summary: Dict[str, Any] = {
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "gpu_snapshot_before": gpu_snapshot(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "mode": args.mode,
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
    }

    if args.mode == "load":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required for --mode load")
        summary["checkpoint_load"] = check_checkpoint_load(args.checkpoint_path)
    elif args.mode == "baseline":
        summary["baseline_migration"] = check_baseline_migration(cfg, args.baseline_ckpt_path)
    else:
        data_info = inspect_batch(cfg, args)
        summary["data_loading"] = data_info
        if args.mode == "train":
            summary["training"] = train_smoke(cfg, args, data_info)

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
