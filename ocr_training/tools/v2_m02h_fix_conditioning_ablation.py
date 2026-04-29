#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["precheck", "train", "condition"], default="train")
    parser.add_argument("--run-name", default="H1_drop_cls_identity_1000steps")
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
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-encoder-from-baseline-ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-ckpt-path",
        default="checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt",
    )
    parser.add_argument("--diagnose-every", type=int, default=100)
    parser.add_argument("--num-probe-samples", type=int, default=5)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--gradient-clip-val", type=float, default=20.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--output-dir", default="outputs/V2-M02h_fix_H1_drop_cls_identity_1000steps")
    parser.add_argument("--output", default="")
    parser.add_argument("override", nargs="*")


def bool_arg(flag: str, enabled: bool) -> list[str]:
    return [flag if enabled else f"--no-{flag[2:]}"]


def base_model_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--decoder-core",
        args.decoder_core,
        "--inference-mode",
        args.inference_mode,
        "--loss-mode",
        args.loss_mode,
        "--visual-adapter-type",
        args.visual_adapter_type,
        "--cross-attn-gate-init",
        str(args.cross_attn_gate_init),
        "--baseline-ckpt-path",
        args.baseline_ckpt_path,
    ]
    cmd += bool_arg("--drop-cls-token", args.drop_cls_token)
    cmd += bool_arg("--cross-attn-gate", args.cross_attn_gate)
    cmd += bool_arg("--freeze-encoder", args.freeze_encoder)
    cmd += bool_arg("--init-encoder-from-baseline-ckpt", args.init_encoder_from_baseline_ckpt)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    if args.mode in {"precheck", "train"}:
        script = TOOLS_DIR / "v2_m02h_official_core_train_check.py"
        child_cmd = [
            sys.executable,
            str(script),
            "--mode",
            args.mode,
            "--run-name",
            args.run_name,
            "--max-steps",
            str(args.max_steps),
            "--denoise-steps",
            str(args.denoise_steps),
            "--diagnose-every",
            str(args.diagnose_every),
            "--num-probe-samples",
            str(args.num_probe_samples),
            "--precision",
            str(args.precision),
            "--gradient-clip-val",
            str(args.gradient_clip_val),
            "--output-dir",
            args.output_dir,
        ]
        child_cmd += base_model_args(args)
        child_cmd += bool_arg("--augment", args.augment)
        child_cmd += args.override
    else:
        if not args.checkpoint_path:
            raise SystemExit("--checkpoint-path is required for --mode condition")
        output = args.output or str(Path(args.output_dir) / "conditioning_summary.json")
        script = TOOLS_DIR / "v2_m02h_conditioning_check.py"
        child_cmd = [
            sys.executable,
            str(script),
            "--checkpoint-path",
            args.checkpoint_path,
            "--output",
            output,
        ]
        child_cmd += base_model_args(args)

    print("running_child=" + " ".join(child_cmd), flush=True)
    subprocess.check_call(child_cmd)


if __name__ == "__main__":
    main()
