import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from strhub.models.utils import init_weights


class ConfidenceEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, confidence: Tensor) -> Tensor:
        return self.proj(confidence.unsqueeze(-1))


class CorrectorBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_length: int,
        use_encoder_memory: bool,
        use_decoder_hidden: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.hidden_proj = nn.Linear(embed_dim, hidden_dim) if use_decoder_hidden else None
        self.confidence_embedding = ConfidenceEmbedding(hidden_dim)
        self.mask_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length + 1, hidden_dim))
        self.encoder_memory_proj = nn.Linear(embed_dim, hidden_dim) if use_encoder_memory else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.apply(init_weights)

    def forward(
        self,
        pred_token_ids: Tensor,
        pred_token_conf: Tensor,
        correction_mask: Tensor,
        decoder_hidden: Tensor,
        encoder_memory: Optional[Tensor] = None,
    ) -> Tensor:
        token_states = self.token_embedding(pred_token_ids)
        fused = token_states
        if self.hidden_proj is not None:
            fused = fused + self.hidden_proj(decoder_hidden)
        fused = fused + self.confidence_embedding(pred_token_conf)
        fused = fused + self.mask_embedding(correction_mask.long())
        fused = fused + self.position_embedding[:, : pred_token_ids.shape[1]]
        if self.encoder_memory_proj is not None and encoder_memory is not None:
            pooled_memory = self.encoder_memory_proj(encoder_memory.mean(dim=1)).unsqueeze(1)
            fused = fused + pooled_memory
        key_padding_mask = pred_token_ids.eq(self.token_embedding.padding_idx)
        fused = self.encoder(fused, src_key_padding_mask=key_padding_mask)
        return self.norm(fused)
