"""
Discrete-token diffusion scheduler supporting:
  • forward schedules: uniform, linear, cosine, deterministic (Van-der-Corput)
  • remask strategies: random, low-confidence, top-p, hybrid, none
  • ReMDM-style inference scaling (target acceptance rate)
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------#
# Config ----------------------------------------------------------------
# ---------------------------------------------------------------------#

@dataclass
class MaskScheduleCfg:
    num_steps: int = 32
    mask_token_id: int = 50256

    # forward schedule
    schedule: str = "uniform"        # uniform | linear | cosine | deterministic
    deterministic_seed: int = 0      # for deterministic schedule

    # reverse-process remask
    remask: str = "random"           # random | lowconf | top_p | hybrid | none
    remask_ratio: float = 0.50       # used by random / lowconf / hybrid
    top_p: float = 0.9               # for top_p

    # ReMDM-style inference scaling
    use_inference_scaling: bool = False
    target_accept: float = 0.9

# ---------------------------------------------------------------------#
# Helpers --------------------------------------------------------------#
# ---------------------------------------------------------------------#

def _van_der_corput(n: int, base: int = 2) -> float:
    vdc, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        vdc += rem / denom
    return vdc


def _mask_fraction(t: torch.Tensor, cfg: MaskScheduleCfg) -> torch.Tensor:
    """Return per-example masking probability r∈[0,1]."""
    if cfg.schedule == "linear":
        return t.float() / cfg.num_steps
    if cfg.schedule == "cosine":
        return 1.0 - torch.cos(0.5 * torch.pi * t.float() / cfg.num_steps)
    if cfg.schedule == "uniform":
        return torch.rand_like(t.float())  # already ∈[0,1]
    if cfg.schedule == "deterministic":
        # Van-der-Corput (low-discrepancy) sequence, one value per sample
        idx = torch.arange(
            t.numel(), device=t.device, dtype=torch.long
        ) + cfg.deterministic_seed
        seq = torch.tensor(
            [_van_der_corput(int(i.item())) for i in idx], device=t.device
        )
        return seq
    raise ValueError(cfg.schedule)


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Mask logits outside the nucleus (top-p)."""
    sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
    cdf = sorted_logits.softmax(-1).cumsum(-1)
    mask = cdf > p
    mask[..., 1:] = mask[..., :-1].clone()  # shift
    mask[..., 0] = False
    logits.scatter_(
        -1, sorted_idx, torch.where(mask, torch.full_like(logits, -float("inf")), logits)
    )
    return logits


# ---------------------------------------------------------------------#
# Scheduler -----------------------------------------------------------#
# ---------------------------------------------------------------------#

class MaskScheduler:
    def __init__(self, cfg: MaskScheduleCfg):
        self.cfg = cfg

    # ----- forward ---------------------------------------------------#
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Corrupt x0 -> xt with schedule-dependent masking.
        Handles both (B, T) and (T,) inputs.
        """
        # Make `frac` broadcastable to x0’s shape
        frac = _mask_fraction(t, self.cfg).view(-1, *([1] * (x0.dim() - 1)))
        noise = torch.rand_like(x0.float())
        mask = noise < frac          # same rank as x0
        xt = x0.clone()
        xt[mask] = self.cfg.mask_token_id
        return xt, mask


    # ----- reverse step ---------------------------------------------#
    @torch.no_grad()
    def p_step(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask_prev: Optional[torch.Tensor] = None,
    ):
        logits = model(xt, t)

        # optional ReMDM scaling
        if self.cfg.use_inference_scaling:
            conf = logits.softmax(-1).max(-1).values
            accept = (conf > 0.5).float().mean()
            scale = torch.clamp(self.cfg.target_accept / accept, 0.5, 2.0)
            logits = logits * scale

        # ----- choose which tokens to REMASK ------------------------#
        pred = logits.argmax(-1)
        if self.cfg.remask == "none":
            new_mask = torch.zeros_like(pred, dtype=torch.bool)

        elif self.cfg.remask == "random":
            new_mask = torch.rand_like(pred.float()) < self.cfg.remask_ratio

        elif self.cfg.remask == "lowconf":
            conf = logits.softmax(-1).max(-1).values
            thresh = conf.quantile(self.cfg.remask_ratio)
            new_mask = conf < thresh

        elif self.cfg.remask == "top_p":
            logits_mod = _apply_top_p(logits.clone(), self.cfg.top_p)
            conf = logits_mod.softmax(-1).max(-1).values
            new_mask = conf < conf.quantile(self.cfg.remask_ratio)

        elif self.cfg.remask == "hybrid":
            conf = logits.softmax(-1).max(-1).values
            low = conf < conf.quantile(self.cfg.remask_ratio)
            rnd = torch.rand_like(pred.float()) < (self.cfg.remask_ratio / 2)
            new_mask = low | rnd
        else:
            raise ValueError(self.cfg.remask)

        # ----- update xt --------------------------------------------#
        xt_next = xt.clone()
        replace = mask_prev if mask_prev is not None else torch.ones_like(new_mask)
        xt_next[replace] = pred[replace]
        xt_next[new_mask] = self.cfg.mask_token_id
        return xt_next, new_mask
