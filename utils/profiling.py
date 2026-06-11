# utils/profiling.py

from __future__ import annotations

import time
from typing import Sequence

import torch


@torch.no_grad()
def measure_latency_ms(model, batch, n_warm=5, n_iter=20, device="cpu"):
    model.eval()
    batch = batch.to(device)

    for _ in range(int(n_warm)):
        _ = model(batch)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(int(n_iter)):
        _ = model(batch)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    dt = (time.perf_counter() - t0) / max(int(n_iter), 1) * 1000.0
    return float(dt)


def conv2d_flops(h, w, in_ch, out_ch, k=3, stride=1, padding=1):
    """
    Approximate Conv2d FLOPs.

    MACs = H_out * W_out * out_ch * (in_ch * k * k)
    FLOPs ≈ 2 * MACs  (multiply + add)
    """
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    macs = h_out * w_out * out_ch * (in_ch * k * k)
    return 2 * macs, h_out, w_out


def linear_flops(in_dim, out_dim):
    """
    Approximate Linear layer FLOPs.
    """
    return 2 * int(in_dim) * int(out_dim)


def estimate_flops_tiny_audiocnn(
    n_mels=64,
    frames=100,
    num_classes=2,
    tap_blocks: Sequence[int] = (1, 3),
):
    """
    Estimate cumulative FLOPs up to each exit for the dynamic TinyAudioCNN + ExitNet.

    Supported backbone structure (current refactor):
      block1: Conv(1->16,k3,p1) + MaxPool(2,2)
      block2: Conv(16->24,k3,p1) + MaxPool(2,2)
      block3: Conv(24->32,k3,p1)
      block4: Conv(32->48,k3,p1)
      block5: Conv(48->64,k3,p1) + AdaptiveAvgPool(1,1)

    Early exits are attached to tap_blocks (subset of {1,2,3,4}).
    Final exit is always after block5.

    Examples:
      tap_blocks=(1,3)       -> exits at dims [16,32] + final 64  => 3 exits
      tap_blocks=(1,2,3,4)   -> exits at dims [16,24,32,48] + final 64 => 5 exits

    Notes:
    - Pooling / tap reduction / adaptive average pool FLOPs are ignored as negligible
      relative to the convs.
    - Output keys are:
        exit1, exit2, ..., exitK
      where K = len(tap_blocks) + 1
    """

    block_channels = [16, 24, 32, 48, 64]
    valid_taps = {1, 2, 3, 4}

    tb = [int(b) for b in tap_blocks]
    if len(tb) == 0:
        raise ValueError("tap_blocks must contain at least one block, e.g. (1,3) or (1,2,3,4).")
    if any(b not in valid_taps for b in tb):
        raise ValueError(f"tap_blocks must be chosen from {sorted(valid_taps)}. Got: {tap_blocks}")

    tap_blocks = sorted(set(tb))

    flops = {}
    total = 0

    # Input spatial size
    h, w = int(n_mels), int(frames)

    # ----- Block 1 -----
    f1, h1, w1 = conv2d_flops(h, w, 1, 16, k=3, stride=1, padding=1)
    total += f1
    h1p, w1p = h1 // 2, w1 // 2  # maxpool(2,2)

    # ----- Block 2 -----
    f2, h2, w2 = conv2d_flops(h1p, w1p, 16, 24, k=3, stride=1, padding=1)
    total_b2 = total + f2
    h2p, w2p = h2 // 2, w2 // 2  # maxpool(2,2)

    # ----- Block 3 -----
    f3, h3, w3 = conv2d_flops(h2p, w2p, 24, 32, k=3, stride=1, padding=1)
    total_b3 = total_b2 + f3

    # ----- Block 4 -----
    f4, h4, w4 = conv2d_flops(h3, w3, 32, 48, k=3, stride=1, padding=1)
    total_b4 = total_b3 + f4

    # ----- Block 5 -----
    f5, h5, w5 = conv2d_flops(h4, w4, 48, 64, k=3, stride=1, padding=1)
    total_b5 = total_b4 + f5

    cumulative_by_block = {
        1: total + 0,      # after block1 conv
        2: total_b2,       # after block2 conv
        3: total_b3,       # after block3 conv
        4: total_b4,       # after block4 conv
        5: total_b5,       # after block5 conv
    }

    head_dim_by_block = {
        1: 16,
        2: 24,
        3: 32,
        4: 48,
        5: 64,
    }

    # Early exits from tap blocks
    exit_idx = 1
    for b in tap_blocks:
        flops[f"exit{exit_idx}"] = cumulative_by_block[b] + linear_flops(
            head_dim_by_block[b], num_classes
        )
        exit_idx += 1

    # Final exit after block5
    flops[f"exit{exit_idx}"] = cumulative_by_block[5] + linear_flops(
        head_dim_by_block[5], num_classes
    )

    return flops