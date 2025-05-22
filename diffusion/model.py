"""
Bidirectional Transformer with timestep embedding.
• `rb_loss` flag enables Rao-Blackwellised loss (compute CE only on masked tokens).
• All other architecture (FP-8 CastedLinear, Muon-friendly weight layout) matches GPT-Lab.
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
import torch._dynamo
from dataclasses import dataclass
from train_gpt import CastedLinear, Rotary, norm
from torch.nn.attention.flex_attention import flex_attention


# ------------------------------------------------------------------#
# Config -----------------------------------------------------------#
# ------------------------------------------------------------------#

@dataclass
class DiffusionConfig:
    vocab_size: int = 50257 + 1          # +1 for <mask>
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 8
    mlp_ratio: float = 4.0
    block_size: int = 65536
    dropout: float = 0.1
    mask_token_id: int = 50255
    num_steps: int = 50


# ------------------------------------------------------------------#
# Building blocks --------------------------------------------------#
# ------------------------------------------------------------------#

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.lin2 = nn.Linear(4 * dim, dim)
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor):
        half = self.lin1.in_features // 2
        inv = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / half))
        emb = torch.cat([torch.sin(t[:, None] * inv), torch.cos(t[:, None] * inv)], dim=1)
        return self.lin2(self.act(self.lin1(emb)))


class FullSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, block: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = CastedLinear(dim, dim)
        self.proj.weight.detach().zero_()
        self.rotary = Rotary(self.head_dim, block)
        self.drop = nn.Dropout(dropout)

    #@torch._dynamo.disable
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, self.heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = map(norm, (q, k))
        q, k = self.rotary(q), self.rotary(k)

        y = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=None, scale=self.scale
        ).transpose(1, 2)
        y = y.reshape(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = CastedLinear(dim, hidden)
        self.fc2 = CastedLinear(hidden, dim)
        self.fc2.weight.detach().zero_()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.attn = FullSelfAttention(cfg.model_dim, cfg.num_heads, cfg.block_size, cfg.dropout)
        self.mlp = MLP(cfg.model_dim, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(nn.LayerNorm(x.size(-1), device=x.device)(x))
        x = x + self.mlp(nn.LayerNorm(x.size(-1), device=x.device)(x))
        return x


# ------------------------------------------------------------------#
# Main model -------------------------------------------------------#
# ------------------------------------------------------------------#

class DiffusionGPT(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.model_dim)
        #self.time_emb = TimestepEmbedding(cfg.model_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.model_dim)
        #self.head = CastedLinear(cfg.model_dim, cfg.vocab_size - 1)  # exclude <mask>
        # dummy linear just to keep FP-8 / CastedLinear behaviour if you need it
        # bias=False because we’ll bypass the weight in forward()
        #self.head = CastedLinear(cfg.model_dim, cfg.vocab_size - 1)

    def forward(
        self,
        idx: torch.Tensor,
        #t: torch.Tensor,
        targets: torch.Tensor | None = None,
        rb_loss: bool = False,
        mask: torch.Tensor | None = None,
    ):
        if idx.dim() == 1:          # single sequence
            idx = idx.unsqueeze(0)  # (1, T)
        #if t.dim() == 1:
        #    t = t.unsqueeze(0)
        if targets is not None and targets.dim() == 1:                      # (T,)
            targets = targets.unsqueeze(0)          # (1, T)
        #if targets.shape[0] == 1 and idx.shape[0] > 1:
        #    targets = targets.expand(idx.size(0), -1)   # (B, T)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        B, T = idx.shape            # safe: now always 2-D
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device)).unsqueeze(0)
        #time = self.time_emb(t).unsqueeze(1)
        #time = self.time_emb(t)
        #x = self.drop(tok + pos + time)
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        emb_weights = self.token_emb.weight[:-1]
        logits = F.linear(x, emb_weights)

        if targets is None:
            return logits
        targets = targets.to(torch.int64)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        if rb_loss and mask is not None:
            loss = (loss * mask.view(-1).float()).sum() / mask.sum().clamp_min(1)
        else:
            loss = loss.mean()
        return loss
