"""Intent bottleneck modules for Intent-Head Co-Train.

See docs/intent_head_cotrain_new.md for the FiLM-modulation design and
docs/intent_head_cotrain.md for the legacy token-concat design. Four modules:

  IntentPool  : attention-pool 64 trunk action tokens into a single 256-D summary.
  IntentHead  : 256 → hidden_dim → intent_dim (→ 256) bottleneck. The narrow
                layer is supervised against rotation-invariant DCT-low magnitudes
                of the future trajectory (gt_intent). The up_project to 256 is
                optional (produce_token=False under FiLM, True for token-concat).
  SRHead      : Binary nominal/recovery classifier sharing the IntentPool output.
  IntentFiLM  : Feature-level intent modulation, gated by self-predicted p_rec.

All four modules are shared across embodiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IntentPool(nn.Module):
    """Single-query attention pool over the trunk's action tokens.

    Alternative is mean-pool, but that destroys the temporal structure of the
    64 action tokens. A learnable query with multi-head attention lets the pool
    adaptively attend to the frames most informative of the recovery intent.

    Shape contract:
        input  z_action: (B, num_action_tokens, latent_dim)
        output          : (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 256, num_heads: int = 8):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, z_action: torch.Tensor) -> torch.Tensor:
        B = z_action.shape[0]
        q = self.query.expand(B, -1, -1)              # (B, 1, D)
        out, _ = self.attn(q, z_action, z_action)     # (B, 1, D)
        return out.squeeze(1)                         # (B, D)


class IntentHead(nn.Module):
    """Narrow bottleneck that produces both the supervision signal and a token.

    Architecture::

        z_intent_in (D=256)
            └─ Linear(D → hidden) → GELU → Linear(hidden → intent_dim)
                                            │
                                 intent_latent (rank ≤ intent_dim)  ← supervised by gt_intent
                                            │
                                            Linear(intent_dim → D)
                                            │
                                 intent_token (B, 1, D)  ← prepended to action-head memory

    The rank of intent_token (as a function of batch) is bounded by intent_dim
    because of the up-projection from a low-dim representation. That is the
    actual information bottleneck — the 256-D shape is just for memory-tensor
    compatibility with trunk tokens.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 64,
        intent_dim: int = 8,
        produce_token: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.intent_dim = intent_dim
        self.produce_token = produce_token

        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intent_dim),
        )
        # up_project is only needed for the token-concat path. FiLM injects
        # intent_latent through γ/β linears inside IntentFiLM, so it skips
        # up_project entirely (saving one nn.Linear and avoiding DDP
        # "unused parameter" warnings when up_project would otherwise be
        # allocated but never reach the loss).
        if produce_token:
            self.up_project = nn.Linear(intent_dim, latent_dim)
        else:
            self.up_project = None

    def forward(
        self, z_intent_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        intent_latent = self.bottleneck(z_intent_in)                     # (B, intent_dim)
        if self.up_project is not None:
            intent_token = self.up_project(intent_latent).unsqueeze(1)   # (B, 1, latent_dim)
        else:
            intent_token = None
        return intent_latent, intent_token


class SRHead(nn.Module):
    """Binary success/recovery classifier (doc §D, Mode 2-RW+SR variant).

    Predicts ``sr_label`` (1 = recovery / off-nominal, 0 = nominal) from the
    same pooled representation that feeds ``IntentHead``. Sharing the input
    means ``L_SR`` provides additional gradient signal to ``IntentPool`` to
    learn "what's relevant for off-nominal detection" — which is the same
    structure the recovery intent feature describes, just viewed as a binary
    decision instead of a 4D regression target.

    Architecture::

        z_intent_in (B, latent_dim)
            └─ Linear(latent_dim → hidden) → GELU → Linear(hidden → 1)
                                                       │
                                            sr_logit (B,)  ← BCE vs sr_label
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_intent_in: torch.Tensor) -> torch.Tensor:
        return self.net(z_intent_in).squeeze(-1)  # (B,) logits


class IntentFiLM(nn.Module):
    """Feature-level intent modulation, gated by self-predicted p_rec.

    Replaces the Mode 2 token-concat assembly. Action head's last decoder
    layer applies::

        gate = β_min + (1 - β_min) · p_rec.detach()
        γ = to_scale(intent_latent)              # (B, latent_dim)
        β = to_shift(intent_latent)              # (B, latent_dim)
        h' = h * (1 + gate · γ) + gate · β

    Properties:

      - p_rec → 0:  h' = h * (1 + β_min · γ) + β_min · β
                    Soft baseline retains a fraction of intent so that recovery
                    signal is not lost when SR head is uncertain at deployment.
      - p_rec → 1:  h' = h * (1 + γ) + β   (full intent injection)
      - p_rec.detach():  SR head is not trained by the BC pathway. Without this
                         the BC loss could push σ(sr_logit) toward whatever
                         minimizes BC, conflicting with L_SR's binary target.

    Initialization:

      - small-init (scale=0.01) on γ/β linears, NOT zero-init. Pure zero-init
        leaves FiLM stuck at identity until BC gradient has a path through —
        which it doesn't, since BC is fine on nominal samples without intent.
        Small-init gives BC a tiny initial grip so γ/β can grow; if the data
        cares about intent, FiLM amplifies, otherwise it stays small.

    Forward shape contract::

        h:              (B, T, latent_dim)         (e.g. action queries)
        intent_latent:  (B, intent_dim)
        p_rec:          (B,)                       (already detached or not;
                                                    we re-detach defensively)
        →
        out:            (B, T, latent_dim)
    """

    def __init__(
        self,
        intent_dim: int = 4,
        latent_dim: int = 256,
        beta_min: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.intent_dim = intent_dim
        self.latent_dim = latent_dim
        self.beta_min = float(beta_min)

        self.to_scale = nn.Linear(intent_dim, latent_dim)
        self.to_shift = nn.Linear(intent_dim, latent_dim)

        # Small-init γ/β: weights ~ N(0, init_scale^2), biases = 0. See class
        # docstring for why not zero-init.
        with torch.no_grad():
            self.to_scale.weight.normal_(0.0, init_scale)
            self.to_scale.bias.zero_()
            self.to_shift.weight.normal_(0.0, init_scale)
            self.to_shift.bias.zero_()

    def forward(
        self,
        h: torch.Tensor,
        intent_latent: torch.Tensor,
        p_rec: torch.Tensor,
    ) -> torch.Tensor:
        # Defensive detach: even if caller passed a non-detached p_rec, FiLM
        # never lets BC pathway reach SRHead via the gate.
        gate = self.beta_min + (1.0 - self.beta_min) * p_rec.detach()
        gate = gate.view(-1, 1, 1)                     # (B, 1, 1)
        gamma = self.to_scale(intent_latent).unsqueeze(1)  # (B, 1, latent_dim)
        beta = self.to_shift(intent_latent).unsqueeze(1)   # (B, 1, latent_dim)
        return h * (1.0 + gate * gamma) + gate * beta
