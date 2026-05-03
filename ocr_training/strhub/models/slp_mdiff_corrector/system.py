from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from strhub.data.utils import Tokenizer
from strhub.models.slp_mdiff_corrector.modules import CorrectorBackbone


class Model(nn.Module):
    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        embed_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1,
        contract_type: str = "token_decoder_hidden",
        use_encoder_memory: bool = False,
        replace_only: bool = True,
        loss_type: str = "selected_plus_preservation",
        lambda_preservation: float = 0.2,
        tau_low: float = 0.70,
        tau_corr: float = 0.80,
        tau_keep: float = 0.90,
        delta_gain: float = 0.05,
        lr: float = 1e-4,
        warmup_pct: float = 0.0,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if contract_type not in {"token_decoder_hidden", "token_only"}:
            raise ValueError(f"Unsupported contract_type: {contract_type}")
        if loss_type not in {
            "selected_plus_preservation",
            "pair_weighted_selected_plus_preservation",
        }:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.tokenizer = Tokenizer(charset_train)
        self.charset_test = charset_test
        self.max_label_length = max_label_length
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.contract_type = contract_type
        self.use_encoder_memory = use_encoder_memory
        self.replace_only = replace_only
        self.loss_type = loss_type
        self.lambda_preservation = lambda_preservation
        self.tau_low = tau_low
        self.tau_corr = tau_corr
        self.tau_keep = tau_keep
        self.delta_gain = delta_gain
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id
        self.pad_id = self.tokenizer.pad_id
        self.output_num_classes = len(self.tokenizer) - 2
        self.use_decoder_hidden = contract_type == "token_decoder_hidden"
        self.backbone = CorrectorBackbone(
            vocab_size=len(self.tokenizer),
            pad_id=self.pad_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_label_length,
            use_encoder_memory=use_encoder_memory,
            use_decoder_hidden=self.use_decoder_hidden,
        )
        self.head = nn.Linear(hidden_dim, self.output_num_classes)
        self._assert_special_ids()

    def _assert_special_ids(self) -> None:
        if self.eos_id != 0:
            raise ValueError(f"Corrector expects eos_id=0, got {self.eos_id}")
        if self.pad_id != 570:
            raise ValueError(f"Corrector expects pad_id=570, got {self.pad_id}")
        if self.output_num_classes != 569:
            raise ValueError(f"Corrector expects output_num_classes=569, got {self.output_num_classes}")

    def forward(
        self,
        pred_token_ids: Tensor,
        pred_token_conf: Tensor,
        correction_mask: Tensor,
        decoder_hidden: Tensor,
        encoder_memory: Optional[Tensor] = None,
    ) -> Tensor:
        if pred_token_ids.ndim != 2:
            raise ValueError(f"pred_token_ids must be [B, T], got {tuple(pred_token_ids.shape)}")
        if pred_token_conf.shape != pred_token_ids.shape:
            raise ValueError("pred_token_conf shape mismatch")
        if correction_mask.shape != pred_token_ids.shape:
            raise ValueError("correction_mask shape mismatch")
        if decoder_hidden.ndim != 3:
            raise ValueError(f"decoder_hidden must be [B, T, D], got {tuple(decoder_hidden.shape)}")
        if decoder_hidden.shape[:2] != pred_token_ids.shape:
            raise ValueError("decoder_hidden leading dims mismatch")
        hidden = self.backbone(
            pred_token_ids=pred_token_ids,
            pred_token_conf=pred_token_conf,
            correction_mask=correction_mask,
            decoder_hidden=decoder_hidden,
            encoder_memory=encoder_memory,
        )
        return self.head(hidden)

    def build_low_confidence_mask(self, pred_token_ids: Tensor, pred_token_conf: Tensor, tau_low: Optional[float] = None) -> Tensor:
        tau_low = self.tau_low if tau_low is None else tau_low
        valid = pred_token_ids.ne(self.pad_id) & pred_token_ids.ne(self.eos_id)
        return valid & pred_token_conf.lt(tau_low)

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        selected_mask: Tensor,
        preserve_mask: Tensor,
        pair_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if logits.shape[:2] != targets.shape:
            raise ValueError("logits/targets shape mismatch")
        if pair_weights is None:
            pair_weights = torch.ones_like(targets, dtype=logits.dtype)
        if pair_weights.shape != targets.shape:
            raise ValueError("pair_weights shape mismatch")
        valid_targets = targets.ne(self.pad_id)
        non_eos_targets = targets.ne(self.eos_id)
        selected_loss_mask = selected_mask & valid_targets & non_eos_targets
        preserve_loss_mask = preserve_mask & valid_targets
        selected_targets = targets[selected_loss_mask]
        preserve_targets = targets[preserve_loss_mask]
        if selected_targets.numel():
            self._assert_loss_targets(selected_targets)
            selected_logits = logits[selected_loss_mask]
            if self.loss_type == "pair_weighted_selected_plus_preservation":
                selected_weights = pair_weights[selected_loss_mask].to(selected_logits.dtype)
                selected_ce_unreduced = F.cross_entropy(selected_logits, selected_targets, reduction="none")
                selected_ce = (selected_ce_unreduced * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
            else:
                selected_ce = F.cross_entropy(selected_logits, selected_targets)
        else:
            selected_ce = logits.sum() * 0.0
        if preserve_targets.numel():
            self._assert_loss_targets(preserve_targets)
            preservation_ce = F.cross_entropy(logits[preserve_loss_mask], preserve_targets)
        else:
            preservation_ce = logits.sum() * 0.0
        loss = selected_ce + self.lambda_preservation * preservation_ce
        return {
            "loss": loss,
            "selected_ce": selected_ce.detach(),
            "preservation_ce": preservation_ce.detach(),
            "selected_count": torch.as_tensor(int(selected_loss_mask.sum().item()), device=logits.device),
            "preserve_count": torch.as_tensor(int(preserve_loss_mask.sum().item()), device=logits.device),
        }

    def _assert_loss_targets(self, targets: Tensor) -> None:
        if not bool(((targets >= 0) & (targets < self.output_num_classes)).all().item()):
            bad = targets[(targets < 0) | (targets >= self.output_num_classes)].detach().cpu().unique().tolist()
            raise ValueError(f"Loss targets must be in [0, {self.output_num_classes - 1}], got {bad}")

    def apply_corrections(
        self,
        pred_token_ids: Tensor,
        pred_token_conf: Tensor,
        decoder_hidden: Tensor,
        encoder_memory: Optional[Tensor] = None,
        selected_mask: Optional[Tensor] = None,
        tau_low: Optional[float] = None,
        tau_corr: Optional[float] = None,
        tau_keep: Optional[float] = None,
        delta_gain: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        tau_corr = self.tau_corr if tau_corr is None else tau_corr
        tau_keep = self.tau_keep if tau_keep is None else tau_keep
        delta_gain = self.delta_gain if delta_gain is None else delta_gain
        if selected_mask is None:
            selected_mask = self.build_low_confidence_mask(pred_token_ids, pred_token_conf, tau_low=tau_low)
        logits = self(
            pred_token_ids=pred_token_ids,
            pred_token_conf=pred_token_conf,
            correction_mask=selected_mask,
            decoder_hidden=decoder_hidden,
            encoder_memory=encoder_memory,
        )
        probs = logits.softmax(-1)
        corr_conf, corr_id = probs.max(-1)
        valid_change_positions = selected_mask & pred_token_ids.ne(self.pad_id) & pred_token_ids.ne(self.eos_id)
        keep_mask = ~valid_change_positions
        keep_mask = keep_mask | corr_id.eq(pred_token_ids)
        keep_mask = keep_mask | corr_conf.lt(tau_corr)
        keep_mask = keep_mask | (pred_token_conf.ge(tau_keep) & (corr_conf - pred_token_conf).lt(delta_gain))
        corrected_ids = pred_token_ids.clone()
        corrected_conf = pred_token_conf.clone()
        changed_mask = ~keep_mask
        corrected_ids[changed_mask] = corr_id[changed_mask]
        corrected_conf[changed_mask] = corr_conf[changed_mask]
        return {
            "logits": logits,
            "probs": probs,
            "selected_mask": selected_mask,
            "keep_mask": keep_mask,
            "changed_mask": changed_mask,
            "corrected_ids": corrected_ids,
            "corrected_conf": corrected_conf,
            "corr_id": corr_id,
            "corr_conf": corr_conf,
        }
