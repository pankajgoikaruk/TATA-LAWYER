# scripts/profile_latency.py

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
import csv

import numpy as np
import torch

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg
from utils.profiling import measure_latency_ms


def load_first_test_batch(segments_csv, features_root, batch_size=16, num_workers=2):
    """
    Build loaders and return the first batch from the TEST loader.
    This batch is used to measure latency.
    """
    _, _, dl_te, label2id = make_loaders(
        segments_csv,
        features_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    x, y = next(iter(dl_te))
    return x, y, label2id


def infer_feature_shape(segments_csv, features_root):
    """
    Load one feature file from the TEST split to infer (n_mels, frames).
    """
    import pandas as pd

    seg = pd.read_csv(segments_csv)
    test_rows = seg[seg["split"] == "test"]
    if test_rows.empty:
        raise SystemExit("No TEST rows found in segments.csv")

    feat_rel = str(test_rows.iloc[0]["feat_relpath"]).replace("\\", "/")
    feat_path = os.path.join(features_root, feat_rel)
    S = np.load(feat_path)  # (n_mels, T)
    n_mels, frames = S.shape
    return int(n_mels), int(frames)


def _conv2d_flops_hw(
    H: int,
    W: int,
    in_ch: int,
    out_ch: int,
    k: int = 3,
    stride: int = 1,
    padding: int = 1,
):
    """
    Return:
      flops, H_out, W_out

    FLOPs counted as multiply+add = 2 ops.
    """
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    flops = 2 * H_out * W_out * out_ch * in_ch * k * k
    return flops, H_out, W_out


def estimate_flops_tiny_audiocnn_tapblocks(n_mels: int, frames: int, num_classes: int, tap_blocks):
    """
    K-exit FLOPs estimator consistent with adapters/audio_adapter.py:

    block1: Conv(1->16,k3,p1) + MaxPool(2x2)
    block2: Conv(16->24,k3,p1) + MaxPool(2x2)
    block3: Conv(24->32,k3,p1)
    block4: Conv(32->48,k3,p1)
    block5: Conv(48->64,k3,p1) + AdaptiveAvgPool(1,1)

    Exits:
      for each tap block in sorted(tap_blocks) (allowed 1..4):
        head_dim = channels at that block
      final exit: after block5 head_dim=64

    Returns dict:
      {"exit1":..., ..., "exitK":...}
    cumulative FLOPs up to that exit (including its linear head).
    """
    tap_blocks = sorted(set(int(b) for b in tap_blocks))
    if any(b < 1 or b > 4 for b in tap_blocks):
        raise ValueError(f"tap_blocks must be in [1..4]. Got: {tap_blocks}")

    ch = [16, 24, 32, 48, 64]
    H, W = int(n_mels), int(frames)

    flops = {}
    total = 0

    # block1 conv
    f1, h1, w1 = _conv2d_flops_hw(H, W, in_ch=1, out_ch=ch[0], k=3, stride=1, padding=1)
    total += f1
    H1, W1 = h1 // 2, w1 // 2  # maxpool

    exit_idx = 1
    if 1 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[0] * num_classes)
        exit_idx += 1

    # block2 conv
    f2, h2, w2 = _conv2d_flops_hw(H1, W1, in_ch=ch[0], out_ch=ch[1], k=3, stride=1, padding=1)
    total += f2
    H2, W2 = h2 // 2, w2 // 2  # maxpool

    if 2 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[1] * num_classes)
        exit_idx += 1

    # block3 conv
    f3, h3, w3 = _conv2d_flops_hw(H2, W2, in_ch=ch[1], out_ch=ch[2], k=3, stride=1, padding=1)
    total += f3
    if 3 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[2] * num_classes)
        exit_idx += 1

    # block4 conv
    f4, h4, w4 = _conv2d_flops_hw(h3, w3, in_ch=ch[2], out_ch=ch[3], k=3, stride=1, padding=1)
    total += f4
    if 4 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[3] * num_classes)
        exit_idx += 1

    # block5 conv (final)
    f5, h5, w5 = _conv2d_flops_hw(h4, w4, in_ch=ch[3], out_ch=ch[4], k=3, stride=1, padding=1)
    total += f5

    # final head 64 -> C
    flops[f"exit{exit_idx}"] = total + 2 * (ch[4] * num_classes)
    return flops


def append_csv_union(path: Path, row: dict):
    """
    Append a row to CSV and automatically extend header if new columns appear.
    This prevents breaking when K changes (e.g., exit4/exit5 added later).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old_fields = list(r.fieldnames or [])
        old_rows = list(r)

    new_fields = list(dict.fromkeys(old_fields + list(row.keys())))

    if new_fields == old_fields:
        filtered = {k: row.get(k, "") for k in old_fields}
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writerow(filtered)
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        w.writeheader()
        for rr in old_rows:
            w.writerow({k: rr.get(k, "") for k in new_fields})
        w.writerow({k: row.get(k, "") for k in new_fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a single run directory, e.g. runs/variant/variant_001")
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--variant", default="V0", help="Variant name label to store (e.g., V0, V1, V2).")
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto (default: auto picks cuda if available)")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size used for latency measurement.")
    ap.add_argument("--n_warm", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=20)

    # K-exit args
    ap.add_argument("--tap_blocks", default="1,3", help="Comma list like 1,2,3,4. Default 1,3 (=3 exits).")
    ap.add_argument("--n_mels", type=int, default=0, help="If 0, infer from features.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tap_blocks = tuple(int(x) for x in str(args.tap_blocks).split(",") if str(x).strip())

    # Load first test batch
    x, y, label2id = load_first_test_batch(
        args.segments_csv,
        args.features_root,
        batch_size=args.batch_size,
        num_workers=2,
    )
    batch_size = x.size(0)
    num_classes = len(label2id)

    # infer (n_mels, frames) for FLOPs
    n_mels_inf, frames = infer_feature_shape(args.segments_csv, args.features_root)
    n_mels = int(args.n_mels) if int(args.n_mels) > 0 else int(n_mels_inf)

    # Build and load model
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model_cfg = load_run_model_cfg(str(run_dir))
    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    K = model.num_exits

    # Measure latency for full forward (all exits computed)
    latency_full_ms = measure_latency_ms(
        model,
        x,
        n_warm=args.n_warm,
        n_iter=args.n_iter,
        device=device,
    )

    # Estimate FLOPs per exit (K-generic) to approximate per-exit latency
    flops = estimate_flops_tiny_audiocnn_tapblocks(
        n_mels=n_mels,
        frames=frames,
        num_classes=num_classes,
        tap_blocks=tap_blocks,
    )

    fl_full = float(flops[f"exit{K}"])
    lat_ms = {}
    for i in range(1, K + 1):
        fi = float(flops[f"exit{i}"])
        lat_ms[f"exit{i}"] = float(latency_full_ms) * (fi / max(fl_full, 1e-12))

    # Load summary.json (for MFLOPs + compute saving) if available
    summary_path = run_dir / "summary.json"
    expected_mflops = None
    full_mflops = None
    compute_saving_pct = None
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        policy = summary.get("policy_summary", {})
        expected_mflops = policy.get("expected_mflops", None)
        full_mflops = policy.get("full_mflops", None)
        compute_saving_pct = policy.get("compute_saving_pct", None)

    profiling = {
        "variant": args.variant,
        "run_id": run_dir.name,
        "device": device,
        "batch_size": int(batch_size),
        "K": int(K),
        "tap_blocks": [int(x) for x in tap_blocks],
        "n_mels": int(n_mels),
        "frames": int(frames),
        "latency_ms": lat_ms,
        "flops": flops,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": compute_saving_pct,
        "exit_hint": (model_cfg or {}).get("exit_hint", {}),
    }

    out_json = run_dir / "profiling.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profiling, f, indent=2)
    print(f"[profile_latency] Saved {out_json}")

    # Append to analysis/on_device_summary.csv
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    csv_path = analysis_dir / "on_device_summary.csv"

    row = {
        "variant": profiling["variant"],
        "run_id": profiling["run_id"],
        "device": profiling["device"],
        "batch_size": profiling["batch_size"],
        "K": profiling["K"],
        "tap_blocks": ",".join(str(x) for x in tap_blocks),
        "n_mels": profiling["n_mels"],
        "frames": profiling["frames"],
        "expected_mflops": expected_mflops if expected_mflops is not None else "",
        "full_mflops": full_mflops if full_mflops is not None else "",
        "compute_saving_pct": compute_saving_pct if compute_saving_pct is not None else "",
    }
    for i in range(1, K + 1):
        row[f"lat_exit{i}_ms"] = profiling["latency_ms"][f"exit{i}"]
        row[f"exit{i}_flops"] = float(profiling["flops"][f"exit{i}"])

    append_csv_union(csv_path, row)
    print(f"[profile_latency] Appended/updated row in {csv_path}")


if __name__ == "__main__":
    main()