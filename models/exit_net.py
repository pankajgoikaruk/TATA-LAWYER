# models/exit_net.py

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitNet(nn.Module):
    """
    Generic K-exit / C-class wrapper with optional local exit-to-exit hint passing.

    Expected backbone contract:
        backbone(x) -> (final_feat, taps)

    where:
        final_feat: Tensor of shape (B, final_dim)
        taps: list/tuple of Tensors, each of shape (B, tap_dim_i)

    Output:
        [logits_exit1, logits_exit2, ..., logits_exit_{K-1}, logits_final]

    Notes:
    - K = len(taps) + 1
    - Supports both:
        * explicit tap_dims/final_dim
        * reading backbone.tap_dims / backbone.final_dim
    - Optional hint path:
        * after exit i, build a small hint from that exit output
        * exit i+1 consumes only the previous hint
        * final head consumes the last hint
    """

    def __init__(
        self,
        backbone: nn.Module,
        tap_dims: Optional[Sequence[int]] = None,
        final_dim: Optional[int] = None,
        num_classes: int = 2,
        hint_dim: int = 0,
        hint_source: str = "probs",
        hint_detach: bool = True,
        hint_use_stats: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_classes = int(num_classes)

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        if tap_dims is None:
            if hasattr(backbone, "tap_dims"):
                tap_dims = getattr(backbone, "tap_dims")
            else:
                raise ValueError(
                    "tap_dims not provided and backbone has no attribute 'tap_dims'."
                )

        if final_dim is None:
            if hasattr(backbone, "final_dim"):
                final_dim = getattr(backbone, "final_dim")
            else:
                raise ValueError(
                    "final_dim not provided and backbone has no attribute 'final_dim'."
                )

        self.tap_dims = [int(d) for d in tap_dims]
        self.final_dim = int(final_dim)

        if len(self.tap_dims) == 0:
            raise ValueError("tap_dims must contain at least one tap dimension.")
        if any(d <= 0 for d in self.tap_dims):
            raise ValueError(f"All tap_dims must be positive, got {self.tap_dims}")
        if self.final_dim <= 0:
            raise ValueError(f"final_dim must be positive, got {self.final_dim}")

        # Hint config
        self.hint_dim = int(hint_dim)
        self.hint_source = str(hint_source).lower().strip()
        self.hint_detach = bool(hint_detach)
        self.hint_use_stats = bool(hint_use_stats)
        self.use_exit_hints = self.hint_dim > 0

        if self.hint_source not in {"probs", "logits"}:
            raise ValueError(
                f"hint_source must be 'probs' or 'logits', got {self.hint_source}"
            )

        # base summary: num_classes values
        # optional stats: confidence, margin, entropy = +3
        self.hint_summary_dim = self.num_classes + (3 if self.hint_use_stats else 0)

        # Early-exit heads
        # exit1 uses tap1 only
        # exit2 uses tap2 + hint1
        # ...
        self.exit_heads = nn.ModuleList()
        for i, dim in enumerate(self.tap_dims):
            in_dim = int(dim) + (self.hint_dim if self.use_exit_hints and i > 0 else 0)
            self.exit_heads.append(nn.Linear(in_dim, self.num_classes))

        # Final head consumes final_feat + last hint
        final_in_dim = self.final_dim + (self.hint_dim if self.use_exit_hints else 0)
        self.final_head = nn.Linear(final_in_dim, self.num_classes)

        # Hint projections: one projector after each early exit
        if self.use_exit_hints:
            self.hint_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hint_summary_dim, self.hint_dim),
                        nn.ReLU(),
                    )
                    for _ in range(len(self.tap_dims))
                ]
            )
        else:
            self.hint_projections = nn.ModuleList()

        # Optional backward-compatible aliases
        for i, head in enumerate(self.exit_heads, start=1):
            setattr(self, f"exit{i}", head)
        self.final = self.final_head

    @property
    def num_exits(self) -> int:
        return len(self.exit_heads) + 1

    def _make_hint(self, logits: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        src = logits.detach() if self.hint_detach else logits

        if self.hint_source == "probs":
            base = F.softmax(src, dim=1)
        else:
            base = src

        if self.hint_use_stats:
            probs = F.softmax(src, dim=1)

            conf = probs.max(dim=1, keepdim=True).values

            top2 = probs.topk(k=min(2, probs.size(1)), dim=1).values
            if top2.size(1) == 1:
                margin = top2[:, :1]
            else:
                margin = top2[:, :1] - top2[:, 1:2]

            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(
                dim=1, keepdim=True
            )

            summary = torch.cat([base, conf, margin, entropy], dim=1)
        else:
            summary = base

        return proj(summary)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_feat, taps = self.backbone(x)

        if not isinstance(taps, (list, tuple)):
            raise RuntimeError(
                f"Backbone must return taps as list/tuple, got {type(taps)}"
            )

        if len(taps) != len(self.exit_heads):
            raise RuntimeError(
                f"Backbone returned {len(taps)} taps, but ExitNet was built for "
                f"{len(self.exit_heads)} taps (tap_dims={self.tap_dims})."
            )

        logits: List[torch.Tensor] = []
        prev_hint: Optional[torch.Tensor] = None

        for i, (head, tap) in enumerate(zip(self.exit_heads, taps)):
            head_in = tap
            if self.use_exit_hints and prev_hint is not None:
                head_in = torch.cat([head_in, prev_hint], dim=1)

            lg = head(head_in)
            logits.append(lg)

            if self.use_exit_hints:
                prev_hint = self._make_hint(lg, self.hint_projections[i])

        final_in = final_feat
        if self.use_exit_hints and prev_hint is not None:
            final_in = torch.cat([final_in, prev_hint], dim=1)

        logits.append(self.final_head(final_in))
        return logits