# scripts/policy_test.py

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Optional

import torch
from torch.nn.functional import softmax

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


def main(
    run_dir,
    segments_csv,
    features_root,
    policy="greedy",
    num_workers=2,
    n_mels=None,
    tap_blocks=None,
    num_classes=None,
    batch_size=64,
    device=None,
):
    if policy != "greedy":
        raise ValueError(
            f"Current scripts.policy_test.py supports only greedy policy, got: {policy}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load greedy threshold
    thresholds_path = os.path.join(run_dir, "thresholds.json")
    with open(thresholds_path, "r", encoding="utf-8") as f:
        tau_obj = json.load(f)
    tau = float(tau_obj["tau"])

    # Load temperatures
    temperature_path = os.path.join(run_dir, "temperature.json")
    with open(temperature_path, "r", encoding="utf-8") as f:
        temp_obj = json.load(f)

    temps_raw = temp_obj.get("temperatures", [])
    temps_raw = [max(float(t), 1e-3) for t in temps_raw]

    # Load test set
    _, _, dl_te, label2id = make_loaders(
        segments_csv,
        features_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Load run config
    run_cfg = _load_run_cfg(run_dir)
    run_model_cfg = load_run_model_cfg(run_dir)
    run_features_cfg = run_cfg.get("features") or {}

    # Prefer CLI, else saved run config, else defaults
    tap_blocks = (
        _parse_tap_blocks(tap_blocks)
        or _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
        or (1, 3)
    )

    if n_mels is None:
        n_mels = int(run_features_cfg.get("n_mels", 64))
    else:
        n_mels = int(n_mels)

    # Keep C-class generic
    num_classes_data = len(label2id)
    if num_classes is not None and int(num_classes) != num_classes_data:
        print(
            f"[WARN] policy_test num_classes={int(num_classes)} but dataset has "
            f"{num_classes_data}. Using dataset value."
        )
    num_classes = num_classes_data

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=run_model_cfg,
    ).to(device)

    ckpt_path = os.path.join(run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    def scale(logits, t):
        return logits / max(float(t), 1e-3)

    n = 0
    correct = 0
    exits = []

    flip_any_count = 0
    flip_count_sum = 0
    consistency_count = 0

    exit_counts = None
    temps = None
    num_exits = None

    with torch.no_grad():
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            if num_exits is None:
                num_exits = len(logits)
                temps = _pad_or_trim(temps_raw, num_exits)
                exit_counts = {f"e{k+1}": 0 for k in range(num_exits)}

            scaled_logits = [scale(lg, temps[i]) for i, lg in enumerate(logits)]
            probs = [softmax(lg, dim=1) for lg in scaled_logits]

            for i in range(x.size(0)):
                preds_all = [int(torch.argmax(p[i]).item()) for p in probs]

                # Greedy decision
                taken = len(probs) - 1
                for k in range(len(probs)):
                    if float(probs[k][i].max()) >= tau:
                        taken = k
                        break

                pred_taken = preds_all[taken]
                pred_final = preds_all[-1]

                correct += int(pred_taken == int(y[i].item()))
                exits.append(taken + 1)
                exit_counts[f"e{taken + 1}"] += 1

                # Flip metrics
                flip_any_count += int(len(set(preds_all)) > 1)
                flip_count_sum += sum(
                    1 for a, b in zip(preds_all[:-1], preds_all[1:]) if a != b
                )
                consistency_count += int(pred_taken == pred_final)

                n += 1

    if n == 0:
        raise RuntimeError("Test loader is empty. No policy results were generated.")

    acc = correct / n
    avg_exit_depth = mean(exits)
    exit_mix = {k: v / n for k, v in exit_counts.items()}
    flip_any_rate = flip_any_count / n
    avg_flip_count = flip_count_sum / n
    exit_consistency = consistency_count / n

    result = {
        "policy": "greedy",
        "accuracy": acc,
        "avg_exit_depth": avg_exit_depth,
        "n_samples": n,
        "n_segments": n,
        "num_exits": num_exits,
        "num_classes": int(num_classes),
        "tap_blocks": list(tap_blocks) if tap_blocks is not None else None,
        "n_mels": int(n_mels),
        "exit_mix": exit_mix,
        "tau": float(tau),
        "temperatures_used": temps,
        "flip_any_rate": flip_any_rate,
        "avg_flip_count": avg_flip_count,
        "exit_consistency": exit_consistency,
        "exit_hint": (run_model_cfg or {}).get("exit_hint", {}),
    }

    out_path = os.path.join(run_dir, "policy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    exit_mix_str = ", ".join(f"{k}={v:.4f}" for k, v in exit_mix.items())

    print(f"Policy test accuracy: {acc:.4f} (n_segments={n})")
    print(f"Avg exit depth: {avg_exit_depth:.3f}")
    print(f"Exit mix: {exit_mix_str}")
    print(f"Flip-any rate: {flip_any_rate:.4f}")
    print(f"Avg flip count: {avg_flip_count:.4f}")
    print(f"Exit consistency: {exit_consistency:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_caches/segments.csv")
    ap.add_argument("--features_root", default="data_caches/features")
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_mels", type=int, default=None)
    ap.add_argument(
        "--tap_blocks",
        type=str,
        default=None,
        help='Example: "1,3" for 3 exits or "1,2,3,4" for 5 exits.',
    )
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    main(
        run_dir=args.run_dir,
        segments_csv=args.segments_csv,
        features_root=args.features_root,
        policy=args.policy,
        num_workers=args.num_workers,
        n_mels=args.n_mels,
        tap_blocks=args.tap_blocks,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device,
    )