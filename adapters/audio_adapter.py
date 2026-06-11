# adapters/audio_adapter.py

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class TinyAudioCNN(nn.Module):
    """
    5-block TinyAudioCNN backbone with configurable tap points.

    Backward-compatible default:
        tap_blocks = (1, 3)
        -> taps after block1 (C=16) and block3 (C=32)
        -> final feature after block5 (C=64)
        -> total exits = 2 taps + 1 final = 3 exits

    Example for 5 exits total:
        tap_blocks = (1, 2, 3, 4)
        -> tap_dims = [16, 24, 32, 48]
        -> final_dim = 64
        -> total exits = 4 taps + 1 final = 5 exits
    """

    _BLOCK_CHANNELS = [16, 24, 32, 48, 64]

    def __init__(self, n_mels: int = 64, tap_blocks: Sequence[int] = (1, 3)):
        super().__init__()
        self.n_mels = int(n_mels)

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.blocks = nn.ModuleList(
            [self.block1, self.block2, self.block3, self.block4, self.block5]
        )

        tb = [int(b) for b in tap_blocks]
        if len(tb) == 0:
            raise ValueError(
                "tap_blocks must contain at least one tap block, "
                "e.g. (1, 3) or (1, 2, 3, 4)."
            )
        if any(b < 1 or b > 4 for b in tb):
            raise ValueError(f"tap_blocks must be in [1..4]. Got: {tap_blocks}")

        self.tap_blocks = sorted(set(tb))
        self.tap_dims = [self._BLOCK_CHANNELS[b - 1] for b in self.tap_blocks]
        self.final_dim = self._BLOCK_CHANNELS[-1]

    @staticmethod
    def _tap_pool(feat_map: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, C, H, W) -> (B, C)
        by max over time (W) then mean over frequency (H).
        """
        return torch.amax(feat_map, dim=-1).mean(-1)

    def forward(self, x: torch.Tensor):
        taps: List[torch.Tensor] = []
        f = x

        for i, block in enumerate(self.blocks, start=1):
            f = block(f)
            if i in self.tap_blocks:
                taps.append(self._tap_pool(f))

        final_feat = f.view(f.size(0), -1)  # (B, 64)
        return final_feat, taps