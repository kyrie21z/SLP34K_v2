from functools import partial
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply
from torch import Tensor

from strhub.models.base import CrossEntropySystem
from strhub.models.slp_mdiff.modules import OfficialCoreMDiffDecoder, PlainMDiffDecoder, VisualAdapter
from strhub.models.utils import init_weights


class Model(CrossEntropySystem):
    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        embed_dim: int,
        mae_pretrained_path: Optional[str],
        mdiff_depth: int,
        mdiff_num_heads: int,
        mdiff_mlp_ratio: float,
        mdiff_dropout: float,
        denoise_steps: int,
        mask_ratio: float,
        mask_strategy: str,
        use_visual_adapter: bool,
        visual_adapter_type: str,
        drop_cls_token: bool = False,
        cross_attn_gate: bool = False,
        cross_attn_gate_init: float = 1.0,
        decoder_core: str = "plain",
        inference_mode: str = "iterative_full_feedback",
        loss_mode: str = "masked_or_eos",
        use_infonce_aux: bool = False,
        infonce_weight: float = 0.0,
        freeze_encoder: bool = True,
        init_encoder_from_baseline_ckpt: bool = False,
        baseline_ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        if use_infonce_aux and infonce_weight:
            raise NotImplementedError("V2-M02 plain MDiff keeps the InfoNCE auxiliary path disabled.")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")
        if mask_strategy not in {"random", "full", "random_or_full"}:
            raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")
        if decoder_core not in {"plain", "official"}:
            raise ValueError(f"Unsupported decoder_core: {decoder_core}")
        if inference_mode not in {"parallel", "iterative_full_feedback"}:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")
        if loss_mode not in {"masked_or_eos", "all_non_pad", "full_mask_all_non_pad", "official_masked_normalized"}:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        if visual_adapter_type not in {"identity", "layernorm", "linear_ln"}:
            raise ValueError(f"Unsupported visual_adapter_type: {visual_adapter_type}")

        self.max_label_length = max_label_length
        self.embed_dim = embed_dim
        self.denoise_steps = denoise_steps
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.decoder_core = decoder_core
        self.inference_mode = inference_mode
        self.loss_mode = loss_mode
        self.freeze_encoder = freeze_encoder
        self.drop_cls_token = drop_cls_token
        self.visual_adapter_type = visual_adapter_type
        self.cross_attn_gate = cross_attn_gate
        self.cross_attn_gate_init = cross_attn_gate_init
        self.mask_id = len(self.tokenizer)
        self.input_vocab_size = len(self.tokenizer) + 1
        self.output_num_classes = len(self.tokenizer) - 2
        self.encoder_load_info = {}
        self.last_loss_debug = {}
        self._assert_special_ids()

        self.encoder = self._build_encoder(img_size, embed_dim)
        if mae_pretrained_path:
            self.encoder_load_info["mae_pretrained"] = self._load_mae_pretrained(mae_pretrained_path)
        if init_encoder_from_baseline_ckpt:
            self.encoder_load_info["baseline_ckpt"] = self._load_baseline_encoder(baseline_ckpt_path)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.visual_adapter = (
            VisualAdapter(embed_dim, embed_dim, visual_adapter_type) if use_visual_adapter else nn.Identity()
        )
        decoder_kwargs = dict(
            input_vocab_size=self.input_vocab_size,
            embed_dim=embed_dim,
            depth=mdiff_depth,
            num_heads=mdiff_num_heads,
            mlp_ratio=mdiff_mlp_ratio,
            dropout=mdiff_dropout,
            max_length=max_label_length + 1,
            padding_idx=self.pad_id,
        )
        if decoder_core == "official":
            self.mdiff_decoder = OfficialCoreMDiffDecoder(
                **decoder_kwargs,
                cross_attn_gate=cross_attn_gate,
                cross_attn_gate_init=cross_attn_gate_init,
            )
            self.head = nn.Linear(embed_dim, self.output_num_classes, bias=False)
        else:
            self.mdiff_decoder = PlainMDiffDecoder(**decoder_kwargs)
            self.head = nn.Linear(embed_dim, self.output_num_classes)

        named_apply(partial(init_weights, exclude=["encoder"]), self)
        if decoder_core == "official":
            nn.init.normal_(self.head.weight, mean=0.0, std=embed_dim**-0.5)

    def _assert_special_ids(self) -> None:
        expected_tokenizer_len = 571
        expected_output_classes = 569
        if len(self.tokenizer) != expected_tokenizer_len:
            raise ValueError(f"SLP MDiff expects tokenizer length {expected_tokenizer_len}, got {len(self.tokenizer)}")
        if self.eos_id != 0:
            raise ValueError(f"SLP MDiff expects eos_id=0, got {self.eos_id}")
        if self.pad_id != 570:
            raise ValueError(f"SLP MDiff expects pad_id=570, got {self.pad_id}")
        if self.mask_id != 571:
            raise ValueError(f"SLP MDiff expects mask_id=571, got {self.mask_id}")
        if self.output_num_classes != expected_output_classes:
            raise ValueError(
                f"SLP MDiff expects output_num_classes={expected_output_classes}, got {self.output_num_classes}"
            )

    def _assert_clean_targets(self, clean_targets: Tensor) -> None:
        allowed = ((clean_targets >= 0) & (clean_targets < self.output_num_classes)) | (clean_targets == self.pad_id)
        if not bool(allowed.all().item()):
            bad_values = clean_targets[~allowed].detach().cpu().unique().tolist()
            raise ValueError(f"Clean targets must be in [0, {self.output_num_classes - 1}] or PAD; got {bad_values}")
        if bool((clean_targets == self.mask_id).any().item()):
            raise ValueError("Clean targets unexpectedly contain mask_id")
        if bool((clean_targets == self.bos_id).any().item()):
            raise ValueError("Clean targets unexpectedly contain BOS id")

    def _assert_loss_targets(self, targets: Tensor) -> None:
        if targets.numel() == 0:
            raise ValueError("No target positions selected for CE")
        if bool((targets == self.pad_id).any().item()):
            raise ValueError("PAD must not enter CE target positions")
        if bool((targets == self.mask_id).any().item()):
            raise ValueError("MASK must not enter CE target positions")
        if bool((targets == self.bos_id).any().item()):
            raise ValueError("BOS must not enter CE target positions")
        in_range = (targets >= 0) & (targets < self.output_num_classes)
        if not bool(in_range.all().item()):
            bad_values = targets[~in_range].detach().cpu().unique().tolist()
            raise ValueError(f"Loss targets must be in [0, {self.output_num_classes - 1}]; got {bad_values}")

    @staticmethod
    def _build_encoder(img_size: Sequence[int], embed_dim: int) -> nn.Module:
        import strhub.models.models_mae as models_mae

        if img_size[0] == 32 and img_size[1] == 128:
            if embed_dim == 384:
                return getattr(models_mae, "mae_vit_base_patch4_384_32x128")()
            if embed_dim == 768:
                return getattr(models_mae, "mae_vit_base_patch4_768_32x128")()
        elif img_size[0] == img_size[1] == 224 and embed_dim == 768:
            return getattr(models_mae, "mae_vit_base_patch16_224x224")()
        raise ValueError(f"Unsupported MAE encoder shape: img_size={img_size}, embed_dim={embed_dim}")

    def _load_mae_pretrained(self, checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        return {"path": checkpoint_path, "missing": len(missing), "unexpected": len(unexpected)}

    def _load_baseline_encoder(self, checkpoint_path: Optional[str]) -> dict:
        if not checkpoint_path:
            raise ValueError("baseline_ckpt_path is required when init_encoder_from_baseline_ckpt=True")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(f"No encoder.* weights found in baseline checkpoint: {checkpoint_path}")
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        return {"path": checkpoint_path, "missing": len(missing), "unexpected": len(unexpected)}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, images: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                memory = self.encoder(images)
        else:
            memory = self.encoder(images)
        return self.prepare_visual_memory(memory)

    def prepare_visual_memory(self, memory: Tensor) -> Tensor:
        if memory.ndim != 3:
            raise ValueError(f"Expected encoder memory [B, S, C], got {tuple(memory.shape)}")
        if memory.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected encoder memory dim {self.embed_dim}, got {memory.shape[-1]}")
        batch_size = memory.shape[0]
        if self.drop_cls_token:
            if memory.shape[1] < 2:
                raise ValueError(f"Cannot drop CLS from memory with length {memory.shape[1]}")
            memory = memory[:, 1:, :]
        memory = self.visual_adapter(memory)
        if memory.ndim != 3:
            raise ValueError(f"Visual adapter must return [B, S, C], got {tuple(memory.shape)}")
        if memory.shape[0] != batch_size:
            raise ValueError("Visual adapter changed batch size")
        if memory.shape[-1] != self.embed_dim:
            raise ValueError(f"Visual adapter must preserve dim {self.embed_dim}, got {memory.shape[-1]}")
        return memory

    def visual_adapter_parameter_count(self) -> int:
        return sum(param.numel() for param in self.visual_adapter.parameters())

    def decode(self, noised_token_ids: Tensor, memory: Tensor) -> Tensor:
        if self.decoder_core == "plain":
            token_padding_mask = noised_token_ids == self.pad_id
            return self.mdiff_decoder(noised_token_ids, memory, token_padding_mask)
        return self.mdiff_decoder(noised_token_ids, memory)

    def _select_mask_strategy(self, batch_size: int, device: torch.device) -> Tensor:
        if self.mask_strategy == "full":
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if self.mask_strategy == "random":
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) < 0.5

    def _make_noised_inputs(
        self,
        clean_targets: Tensor,
        return_full_mask_rows: bool = False,
        return_stats: bool = False,
    ):
        self._assert_clean_targets(clean_targets)

        device = clean_targets.device
        valid = clean_targets != self.pad_id
        full_mask_rows = self._select_mask_strategy(clean_targets.shape[0], device)
        random_mask = (torch.rand(clean_targets.shape, device=device) < self.mask_ratio) & valid
        masked_positions = torch.where(full_mask_rows[:, None], valid, random_mask)

        for row_idx in range(masked_positions.shape[0]):
            if valid[row_idx].any() and not masked_positions[row_idx].any():
                valid_indices = valid[row_idx].nonzero(as_tuple=False).flatten()
                choice = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]
                masked_positions[row_idx, choice] = True

        noised = clean_targets.clone()
        noised[masked_positions] = self.mask_id
        if return_stats:
            valid_count = valid.sum(dim=1)
            masked_valid_count = (masked_positions & valid).sum(dim=1)
            p_mask = (masked_valid_count.float() / valid_count.clamp_min(1).float()).clamp_min(1e-6)
            length = valid_count.long()
            return noised, masked_positions, full_mask_rows, p_mask, length
        if return_full_mask_rows:
            return noised, masked_positions, full_mask_rows
        return noised, masked_positions

    def _get_loss_positions(self, clean_targets: Tensor, masked_positions: Tensor, full_mask_rows: Tensor) -> Tensor:
        non_pad = clean_targets != self.pad_id
        eos_positions = clean_targets == self.eos_id

        if self.loss_mode == "masked_or_eos":
            return (masked_positions | eos_positions) & non_pad
        if self.loss_mode == "all_non_pad":
            return non_pad
        if self.loss_mode == "full_mask_all_non_pad":
            masked_or_eos = (masked_positions | eos_positions) & non_pad
            all_non_pad = non_pad
            return torch.where(full_mask_rows[:, None], all_non_pad, masked_or_eos)
        if self.loss_mode == "official_masked_normalized":
            return masked_positions & non_pad
        raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

    def _official_masked_normalized_loss(
        self,
        logits: Tensor,
        clean_targets: Tensor,
        masked_positions: Tensor,
        p_mask: Tensor,
        length: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        loss_positions = masked_positions & (clean_targets != self.pad_id)
        targets = clean_targets[loss_positions]
        self._assert_loss_targets(targets)

        token_loss = F.cross_entropy(logits[loss_positions], targets, reduction="none")
        p_mask_at_positions = p_mask[:, None].expand_as(clean_targets)[loss_positions].clamp_min(1e-6)
        length_plus_one = (length + 1).float()[:, None].expand_as(clean_targets)[loss_positions].clamp_min(1.0)
        loss = torch.sum(token_loss / p_mask_at_positions / length_plus_one) / clean_targets.shape[0]
        self.last_loss_debug = {
            "loss_position_count": int(loss_positions.sum().detach().cpu().item()),
            "masked_count": int(masked_positions.sum().detach().cpu().item()),
            "pad_in_loss": int((targets == self.pad_id).sum().detach().cpu().item()),
            "out_of_range_targets": int((targets >= self.output_num_classes).sum().detach().cpu().item()),
            "p_mask_mean": float(p_mask.detach().float().mean().cpu().item()),
            "length_mean": float(length.detach().float().mean().cpu().item()),
        }
        return loss, loss_positions.sum()

    def _denoising_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        clean_targets = self.tokenizer.encode(labels, images.device)[:, 1:]
        memory = self.encode(images)
        noised_ids, masked_positions, full_mask_rows, p_mask, length = self._make_noised_inputs(
            clean_targets, return_full_mask_rows=True, return_stats=True
        )
        hidden = self.decode(noised_ids, memory)
        logits = self.head(hidden)

        if logits.shape[-1] != self.output_num_classes:
            raise ValueError(f"Expected head output {self.output_num_classes}, got {logits.shape[-1]}")
        if self.loss_mode == "official_masked_normalized":
            loss, loss_numel = self._official_masked_normalized_loss(
                logits, clean_targets, masked_positions, p_mask, length
            )
            return logits, loss, loss_numel

        loss_positions = self._get_loss_positions(clean_targets, masked_positions, full_mask_rows)
        targets = clean_targets[loss_positions]
        self._assert_loss_targets(targets)
        loss = F.cross_entropy(logits[loss_positions], targets)
        loss_numel = loss_positions.sum()
        self.last_loss_debug = {
            "loss_position_count": int(loss_numel.detach().cpu().item()),
            "masked_count": int(masked_positions.sum().detach().cpu().item()),
            "pad_in_loss": int((targets == self.pad_id).sum().detach().cpu().item()),
            "out_of_range_targets": int((targets >= self.output_num_classes).sum().detach().cpu().item()),
            "p_mask_mean": float(p_mask.detach().float().mean().cpu().item()),
            "length_mean": float(length.detach().float().mean().cpu().item()),
        }
        return logits, loss, loss_numel

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        targets = self.tokenizer.encode(labels, images.device)[:, 1:]
        max_len = targets.shape[1] - 1
        logits = self.forward(images, max_len)
        valid = targets != self.pad_id
        loss = F.cross_entropy(logits[valid], targets[valid])
        loss_numel = valid.sum()
        return logits, loss, loss_numel

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        num_steps = max_length + 1
        batch_size = images.shape[0]
        memory = self.encode(images)
        input_ids = torch.full(
            (batch_size, num_steps),
            self.mask_id,
            dtype=torch.long,
            device=images.device,
        )

        if self.inference_mode == "parallel":
            hidden = self.decode(input_ids, memory)
            return self.head(hidden)

        logits = None
        for _ in range(max(1, self.denoise_steps)):
            hidden = self.decode(input_ids, memory)
            logits = self.head(hidden)
            input_ids = logits.argmax(dim=-1)
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        _, loss, loss_numel = self._denoising_logits_loss(images, labels)
        if getattr(self, "_trainer", None) is not None:
            self.log("loss", loss)
            self.log("ocr_loss", loss)
            self.log("loss_numel", loss_numel.float())
        return loss
