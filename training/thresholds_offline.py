# training/thresholds_offline.py

from __future__ import annotations

import os
import json
import argparse
from typing import Optional

import torch
from torch.nn.functional import softmax
from sklearn.metrics import f1_score

from data.datasets import make_loaders
from utils.config import load_config
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


def _parse_tap_blocks(value) -> Optional[tuple]:
    """
    Accept:
      - None
      - "1,2,3,4"
      - [1,2,3,4]
      - (1,2,3,4)
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if value == "":
        return None

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _pad_or_trim(seq, target_len, pad_value=None):
    vals = [float(x) for x in seq]
    if len(vals) >= target_len:
        return vals[:target_len]

    if pad_value is None:
        pad_value = vals[-1] if len(vals) > 0 else 1.0

    vals = vals + [float(pad_value)] * (target_len - len(vals))
    return vals


def _load_run_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        return load_config(cfg_path) or {}
    except Exception:
        return {}


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None, temps=None, eps_temp=1e-3):
    """
    Collect validation logits once; optionally apply temperature scaling per exit.
    Returns:
        logits_list: list of tensors, one per exit
        y_true: tensor of labels
    """
    model.eval()

    all_logits = None
    all_y = []
    seen = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg = model(x)

        if all_logits is None:
            all_logits = [[] for _ in range(len(lg))]

        if temps is not None:
            temps_used = _pad_or_trim(temps, len(lg), pad_value=1.0)
            lg = [l / max(float(temps_used[i]), eps_temp) for i, l in enumerate(lg)]

        for k in range(len(lg)):
            all_logits[k].append(lg[k].detach().cpu())

        all_y.append(y.detach().cpu())
        seen += x.size(0)

        if max_samples is not None and seen >= max_samples:
            break

    if all_logits is None or len(all_y) == 0:
        raise RuntimeError("Validation loader is empty. No logits collected.")

    all_logits = [torch.cat(L, 0) for L in all_logits]
    all_y = torch.cat(all_y, 0)
    return all_logits, all_y


def eval_policy_for_tau(logits_list, y_true, tau):
    """Greedy early-exit: stop at first exit where max prob >= tau."""
    probs = [softmax(l, dim=1) for l in logits_list]
    y_hat = []

    for i in range(y_true.numel()):
        pred = None
        for k in range(len(probs) - 1):
            if float(probs[k][i].max()) >= tau:
                pred = int(torch.argmax(probs[k][i]))
                break

        if pred is None:
            pred = int(torch.argmax(probs[-1][i]))

        y_hat.append(pred)

    y_np = y_true.numpy()
    f1 = f1_score(y_np, y_hat, average="macro")
    acc = float((y_np == y_hat).mean())
    return float(f1), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument(
        "--tau",
        nargs="+",
        type=float,
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95],
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="limit val samples to speed up (0=all)",
    )
    ap.add_argument(
        "--eps_temp",
        type=float,
        default=1e-3,
        help="minimum temperature to avoid divide-by-zero",
    )

    # Generic K-exit args
    ap.add_argument(
        "--tap_blocks",
        type=str,
        default=None,
        help='Example: "1,3" for 3 exits or "1,2,3,4" for 5 exits.',
    )
    ap.add_argument("--n_mels", type=int, default=None)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load validation loader
    _, dl_val, _, label2id = make_loaders(
        args.segments_csv,
        args.features_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2) Load run config
    run_cfg = _load_run_cfg(args.run_dir)
    run_model_cfg = load_run_model_cfg(args.run_dir)
    run_features_cfg = run_cfg.get("features") or {}

    # Prefer CLI, else saved run config, else defaults
    n_mels = int(
        args.n_mels
        if args.n_mels is not None
        else run_features_cfg.get("n_mels", 64)
    )

    tap_blocks = (
        _parse_tap_blocks(args.tap_blocks)
        or _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
        or (1, 3)
    )

    # Keep C-class generic
    num_classes_data = len(label2id)
    num_classes_cli = int(args.num_classes) if args.num_classes is not None else num_classes_data
    if num_classes_cli != num_classes_data:
        print(
            f"[WARN] thresholds_offline num_classes={num_classes_cli} but dataset has "
            f"{num_classes_data}. Using dataset value."
        )
    num_classes = num_classes_data

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=run_model_cfg,
    ).to(device)

    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # 3) Load temperatures if available
    temps = None
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            temps = json.load(f).get("temperatures", None)

        if temps is None:
            print("temperature.json found but invalid; using raw logits.")
            temps = None
        else:
            temps = [float(t) for t in temps]
            temps = [max(t, args.eps_temp) for t in temps]
            print("Using temperatures:", temps)
    else:
        print("No temperature.json found; using raw logits.")

    # 4) Precompute logits once
    max_samples = None if (args.max_samples or 0) <= 0 else int(args.max_samples)

    print("Collecting validation logits...")
    logits_val, y_val = collect_val_logits(
        model,
        dl_val,
        device,
        max_samples=max_samples,
        temps=temps,
        eps_temp=args.eps_temp,
    )
    print(f"Collected {y_val.numel()} validation samples across {len(logits_val)} exits.")

    temps_used = None if temps is None else _pad_or_trim(temps, len(logits_val), pad_value=1.0)

    # 5) Sweep tau grid
    best = None
    print("Sweeping tau grid...")
    for tau in args.tau:
        f1, acc = eval_policy_for_tau(logits_val, y_val, tau)
        print(f"  tau={tau:.3f} -> macroF1={f1:.4f}, acc={acc:.4f}")

        if best is None or (f1 > best["f1"]) or (f1 == best["f1"] and acc > best["acc"]):
            best = {
                "tau": float(tau),
                "f1": float(f1),
                "acc": float(acc),
            }

    # 6) Save thresholds.json
    outpath = os.path.join(args.run_dir, "thresholds.json")
    payload = dict(best)
    payload["temperatures_used"] = temps_used
    payload["num_exits"] = int(len(logits_val))
    payload["num_classes"] = int(num_classes)
    payload["tap_blocks"] = list(tap_blocks) if tap_blocks is not None else None
    payload["exit_hint"] = (run_model_cfg or {}).get("exit_hint", {})

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved thresholds.json:", payload)


if __name__ == "__main__":
    main()