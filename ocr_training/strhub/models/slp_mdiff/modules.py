import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, tokens: Tensor) -> Tensor:
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class VisualAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str = "identity") -> None:
        super().__init__()
        self.adapter_type = adapter_type
        if adapter_type == "identity":
            if in_dim != out_dim:
                raise ValueError("identity visual adapter requires in_dim == out_dim")
            self.adapter = nn.Identity()
        elif adapter_type == "layernorm":
            if in_dim != out_dim:
                raise ValueError("layernorm visual adapter requires in_dim == out_dim")
            self.adapter = nn.LayerNorm(out_dim)
        elif adapter_type == "linear_ln":
            self.adapter = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))
        else:
            raise ValueError(f"Unsupported visual_adapter_type: {adapter_type}")

    def forward(self, memory: Tensor) -> Tensor:
        return self.adapter(memory)


class PlainMDiffDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, memory: Tensor, token_padding_mask: Optional[Tensor] = None) -> Tensor:
        q = self.norm1(hidden)
        hidden = hidden + self.dropout1(
            self.self_attn(q, q, q, key_padding_mask=token_padding_mask, need_weights=False)[0]
        )

        hidden = hidden + self.dropout2(
            self.cross_attn(self.norm2(hidden), memory, memory, need_weights=False)[0]
        )

        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(hidden)))))
        hidden = hidden + self.dropout3(ffn)
        return hidden


class PlainMDiffDecoder(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        max_length: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.text_embed = TokenEmbedding(input_vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embed = nn.Parameter(torch.empty(1, max_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                PlainMDiffDecoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def forward(self, noised_token_ids: Tensor, memory: Tensor, token_padding_mask: Optional[Tensor] = None) -> Tensor:
        length = noised_token_ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"Decoder input length {length} exceeds max_length {self.max_length}")

        hidden = self.text_embed(noised_token_ids) + self.position_embed[:, :length]
        hidden = self.dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden, memory, token_padding_mask)
        return self.norm(hidden)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.dropout(hidden + self.pe[:, : hidden.shape[1]])


class OfficialStyleMlp(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return hidden


class OfficialStyleTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        cross_attn_gate: bool = False,
        cross_attn_gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.mlp = OfficialStyleMlp(embed_dim, mlp_ratio, residual_dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)
        self.use_cross_attn_gate = cross_attn_gate
        self.cross_attn_gate = (
            nn.Parameter(torch.tensor(float(cross_attn_gate_init))) if cross_attn_gate else None
        )
        self.last_diagnostics = {}

    def forward(self, hidden: Tensor, memory: Tensor) -> Tensor:
        hidden1 = self.self_attn(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm1(hidden + self.dropout1(hidden1))

        hidden2 = self.cross_attn(hidden, memory, memory, need_weights=False)[0]
        cross_residual = hidden2 * self.cross_attn_gate if self.use_cross_attn_gate else hidden2
        hidden = self.norm2(hidden + self.dropout2(cross_residual))

        ffn = self.mlp(hidden)
        hidden = self.norm3(hidden + self.dropout3(ffn))
        self.last_diagnostics = {
            "self_attn_output_norm": float(hidden1.detach().float().norm(dim=-1).mean().cpu()),
            "cross_attn_output_norm": float(hidden2.detach().float().norm(dim=-1).mean().cpu()),
            "ffn_output_norm": float(ffn.detach().float().norm(dim=-1).mean().cpu()),
            "hidden_norm": float(hidden.detach().float().norm(dim=-1).mean().cpu()),
        }
        return hidden


class OfficialCoreMDiffDecoder(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        max_length: int,
        padding_idx: Optional[int] = None,
        cross_attn_gate: bool = False,
        cross_attn_gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.text_embed = TokenEmbedding(input_vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_length, dropout)
        self.position_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.layers = nn.ModuleList(
            [
                OfficialStyleTransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    0.0,
                    dropout,
                    cross_attn_gate=cross_attn_gate,
                    cross_attn_gate_init=cross_attn_gate_init,
                )
                for _ in range(depth)
            ]
        )
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def forward(self, noised_token_ids: Tensor, memory: Tensor) -> Tensor:
        length = noised_token_ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"Decoder input length {length} exceeds max_length {self.max_length}")
        if memory.ndim != 3:
            raise ValueError(f"Expected visual memory with shape [B, S, C], got {tuple(memory.shape)}")
        if memory.shape[0] != noised_token_ids.shape[0]:
            raise ValueError(
                f"Batch mismatch between tokens ({noised_token_ids.shape[0]}) and memory ({memory.shape[0]})"
            )
        if memory.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected memory dim {self.embed_dim}, got {memory.shape[-1]}")

        hidden = self.text_embed(noised_token_ids)
        hidden = self.positional_encoding(hidden) + self.position_embed[:, :length]
        for layer in self.layers:
            hidden = layer(hidden, memory)
        return hidden

    def get_last_diagnostics(self):
        return [dict(layer.last_diagnostics) for layer in self.layers]
