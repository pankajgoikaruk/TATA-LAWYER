# scripts/summarize_run.py

from __future__ import annotations

import os
import json
import csv
import time
import argparse
import inspect
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg
from utils.profiling import estimate_flops_tiny_audiocnn

import matplotlib.pyplot as plt


def load_json_safepath(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


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


def _infer_tap_blocks_from_run(run_dir: str) -> Optional[tuple]:
    """
    Recover tap blocks from saved artifacts if available.
    """
    candidates = [
        os.path.join(run_dir, "metrics.json"),
        os.path.join(run_dir, "report.json"),
        os.path.join(run_dir, "temperature.json"),
        os.path.join(run_dir, "thresholds.json"),
        os.path.join(run_dir, "meta.json"),
        os.path.join(run_dir, "config_used.yaml"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        if path.endswith(".json"):
            obj = load_json_safepath(path, {})
            if not isinstance(obj, dict):
                continue

            tb = None
            if isinstance(obj.get("meta"), dict):
                tb = obj["meta"].get("tap_blocks")
            if tb is None:
                tb = obj.get("tap_blocks")

            tb = _parse_tap_blocks(tb)
            if tb is not None:
                return tb

        elif path.endswith(".yaml"):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                tb = _parse_tap_blocks((cfg.get("model") or {}).get("tap_blocks"))
                if tb is not None:
                    return tb
            except Exception:
                pass

    return None


def ece_score(conf, corr, n_bins=15):
    """conf: (N,) confidences; corr: (N,) correctness (0/1)."""
    conf = np.asarray(conf)
    corr = np.asarray(corr).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_summ = []

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        sel = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)

        if not np.any(sel):
            bin_summ.append(
                {"bin": [float(lo), float(hi)], "count": 0, "acc": None, "conf": None}
            )
            continue

        p = sel.mean()
        acc = corr[sel].mean()
        cbar = conf[sel].mean()
        ece += p * abs(acc - cbar)
        bin_summ.append(
            {"bin": [float(lo), float(hi)], "count": int(sel.sum()), "acc": float(acc), "conf": float(cbar)}
        )

    return float(ece), bin_summ


def plot_hist_and_reliability(run_dir, key, conf, corr, n_bins=15):
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    plt.figure()
    plt.hist(conf, bins=30, range=(0, 1))
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Histogram - {key}")
    p_hist = os.path.join(run_dir, "plots", f"{key}_conf_hist.png")
    plt.savefig(p_hist, bbox_inches="tight")
    plt.close()

    ece, bins = ece_score(conf, corr, n_bins=n_bins)
    accs, confs = [], []
    for b in bins:
        if b["acc"] is not None:
            accs.append(b["acc"])
            confs.append(b["conf"])

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence (bin mean)")
    plt.ylabel("Accuracy (bin)")
    plt.title(f"Reliability - {key} (ECE={ece:.3f})")
    p_rel = os.path.join(run_dir, "plots", f"{key}_reliability.png")
    plt.savefig(p_rel, bbox_inches="tight")
    plt.close()

    return {
        "ece": ece,
        "bins": bins,
        "hist_path": p_hist,
        "reliability_path": p_rel,
    }


def plot_conf_vs_correct(run_dir, key, conf, corr, jitter=0.02):
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    y = np.asarray(corr).astype(float)
    x = np.asarray(conf).astype(float)
    yj = y + (np.random.rand(*y.shape) - 0.5) * jitter

    plt.figure()
    plt.scatter(x, yj, s=8, alpha=0.5)
    plt.yticks([0, 1], ["wrong", "correct"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Confidence")
    plt.ylabel("Correctness")
    plt.title(f"Confidence vs Correctness - {key}")
    p_sc = os.path.join(run_dir, "plots", f"{key}_conf_vs_correct.png")
    plt.savefig(p_sc, bbox_inches="tight")
    plt.close()

    return p_sc


@torch.no_grad()
def collect_exit_logits_on_split(model, dl, device):
    """
    Returns dict per-exit:
        probs (N,C), y (N,), conf (N,), corr (N,)
    for using that exit directly.
    """
    all_probs = None
    all_y = []

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg = model(x)
        pr = [softmax(l, dim=1).cpu() for l in lg]

        if all_probs is None:
            all_probs = [[] for _ in range(len(pr))]

        for k in range(len(pr)):
            all_probs[k].append(pr[k])

        all_y.append(y.cpu())

    if all_probs is None or len(all_y) == 0:
        return {}

    probs = [torch.cat(all_probs[k], 0).numpy() for k in range(len(all_probs))]
    ytrue = torch.cat(all_y, 0).numpy()

    out = {}
    for k in range(len(probs)):
        conf = probs[k].max(axis=1)
        pred = probs[k].argmax(axis=1)
        corr = (pred == ytrue).astype(np.float32)
        out[f"exit{k+1}"] = {
            "probs": probs[k],
            "y": ytrue,
            "conf": conf,
            "corr": corr,
        }

    return out


def _safe_estimate_flops(n_mels, frames, num_classes, tap_blocks, num_exits):
    """
    Best-effort FLOPs estimation.
    If utils.profiling is still 3-exit-specific, do not crash.
    """
    try:
        sig = inspect.signature(estimate_flops_tiny_audiocnn)
        kwargs = {
            "n_mels": int(n_mels),
            "frames": int(frames),
            "num_classes": int(num_classes),
        }
        if "tap_blocks" in sig.parameters and tap_blocks is not None:
            kwargs["tap_blocks"] = tap_blocks

        fl = estimate_flops_tiny_audiocnn(**kwargs)

        if not isinstance(fl, dict):
            return {
                "flops_raw": fl,
                "full_mflops": None,
                "expected_mflops": None,
                "compute_saving_pct": None,
                "warning": "estimate_flops_tiny_audiocnn did not return a dict.",
            }

        exit_keys = [f"exit{i+1}" for i in range(num_exits)]
        if not all(k in fl for k in exit_keys):
            return {
                "flops_raw": fl,
                "full_mflops": None,
                "expected_mflops": None,
                "compute_saving_pct": None,
                "warning": "FLOPs helper does not expose all exits for current K.",
            }

        return {
            "flops_raw": fl,
            "full_mflops": float(fl[exit_keys[-1]]) / 1e6,
            "expected_mflops": None,
            "compute_saving_pct": None,
            "warning": None,
        }

    except Exception as e:
        return {
            "flops_raw": None,
            "full_mflops": None,
            "expected_mflops": None,
            "compute_saving_pct": None,
            "warning": f"FLOPs estimation failed: {type(e).__name__}: {e}",
        }


@torch.no_grad()
def policy_eval(run_dir, segments_csv, features_root, save_plots=True, n_mels_override=None, tap_blocks_override=None):
    th = load_json_safepath(os.path.join(run_dir, "thresholds.json"), {"tau": 0.95}) or {"tau": 0.95}
    tau = float(th.get("tau", 0.95))

    temp_obj = load_json_safepath(
        os.path.join(run_dir, "temperature.json"),
        {"temperatures": [1.0, 1.0, 1.0]},
    ) or {"temperatures": [1.0, 1.0, 1.0]}
    temps = temp_obj.get("temperatures", [1.0, 1.0, 1.0])
    temps = [max(float(t), 1e-3) for t in temps]

    run_model_cfg = load_run_model_cfg(run_dir)

    tap_blocks = (
        tap_blocks_override
        or _parse_tap_blocks(th.get("tap_blocks"))
        or _parse_tap_blocks(temp_obj.get("tap_blocks"))
        or _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
        or _infer_tap_blocks_from_run(run_dir)
    )

    n_mels_cfg = None
    try:
        import yaml
        cfg_path = os.path.join(run_dir, "config_used.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            n_mels_cfg = int((cfg.get("features") or {}).get("n_mels", 64))
            if tap_blocks is None:
                tap_blocks = _parse_tap_blocks((cfg.get("model") or {}).get("tap_blocks"))
    except Exception:
        pass

    n_mels = int(n_mels_override if n_mels_override is not None else (n_mels_cfg if n_mels_cfg is not None else 64))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, dl_te, label2id = make_loaders(segments_csv, features_root, batch_size=64, num_workers=2)
    num_classes = len(label2id)

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=run_model_cfg,
    ).to(device).eval()

    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt", "best.pt"), map_location=device))

    num_exits = model.num_exits
    temps = _pad_or_trim(temps, num_exits, pad_value=1.0)

    n = 0
    correct = 0
    exits = []
    confs = []
    corrs = []

    def scale(lg, t):
        return lg / max(float(t), 1e-3)

    for x, y in dl_te:
        x, y = x.to(device), y.to(device)
        logits = [scale(l, temps[i]) for i, l in enumerate(model(x))]
        probs = [softmax(l, dim=1) for l in logits]

        for i in range(x.size(0)):
            taken = len(probs) - 1
            for k in range(len(probs) - 1):
                if float(probs[k][i].max()) >= tau:
                    taken = k
                    break

            p = probs[taken][i]
            pred = int(torch.argmax(p))
            conf = float(torch.max(p))
            corr = float(pred == int(y[i]))

            correct += int(corr)
            exits.append(taken + 1)
            n += 1
            confs.append(conf)
            corrs.append(corr)

    exit_mix = {f"e{k+1}": exits.count(k + 1) / max(n, 1) for k in range(num_exits)}
    policy_acc = correct / max(n, 1)
    avg_exit_depth = float(np.mean(exits)) if exits else None

    seg = pd.read_csv(segments_csv)
    test_row = seg[seg["split"] == "test"].iloc[0]
    feat_path = os.path.join(features_root, str(test_row["feat_relpath"]).replace("\\", "/"))
    feat_n_mels, frames = np.load(feat_path).shape

    flops_info = _safe_estimate_flops(
        n_mels=int(feat_n_mels),
        frames=int(frames),
        num_classes=num_classes,
        tap_blocks=tap_blocks,
        num_exits=num_exits,
    )

    expected_mflops = None
    full_mflops = flops_info["full_mflops"]
    compute_saving_pct = None

    if isinstance(flops_info.get("flops_raw"), dict) and full_mflops is not None:
        try:
            expected = 0.0
            for k in range(num_exits):
                expected += exit_mix[f"e{k+1}"] * flops_info["flops_raw"][f"exit{k+1}"]
            expected_mflops = expected / 1e6
            compute_saving_pct = 100.0 * (1.0 - expected_mflops / full_mflops)
        except Exception:
            expected_mflops = None
            compute_saving_pct = None

    policy_calib = {"ece": None, "bins": None, "hist_path": None, "reliability_path": None, "scatter_path": None}
    if save_plots:
        policy_calib = plot_hist_and_reliability(run_dir, "policy_test", np.array(confs), np.array(corrs), n_bins=15)
        policy_scatter = plot_conf_vs_correct(run_dir, "policy_test", np.array(confs), np.array(corrs))
        policy_calib["scatter_path"] = policy_scatter

    _, _, dl_te2, _ = make_loaders(segments_csv, features_root, batch_size=128, num_workers=2)
    per_exit = collect_exit_logits_on_split(model, dl_te2, device)
    per_exit_calib = {}

    for k in range(1, num_exits + 1):
        key = f"exit{k}"
        conf = per_exit[key]["conf"]
        corr = per_exit[key]["corr"]
        per_exit_calib[key] = plot_hist_and_reliability(run_dir, f"{key}_test", conf, corr, n_bins=15)
        sc_path = plot_conf_vs_correct(run_dir, f"{key}_test", conf, corr)
        per_exit_calib[key]["scatter_path"] = sc_path

    return {
        "tau": tau,
        "temperatures": temps,
        "num_exits": num_exits,
        "tap_blocks": list(tap_blocks) if tap_blocks is not None else None,
        "exit_mix": exit_mix,
        "avg_exit_depth": avg_exit_depth,
        "policy_test_acc": policy_acc,
        "n_mels": int(feat_n_mels),
        "frames": int(frames),
        "num_classes": num_classes,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": compute_saving_pct,
        "flops_warning": flops_info.get("warning"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "policy_calibration": policy_calib,
        "per_exit_calibration": per_exit_calib,
        "exit_hint": (run_model_cfg or {}).get("exit_hint", {}),
    }


def _append_experiments_row_safely(experiments_csv: str, row: dict):
    os.makedirs(os.path.dirname(experiments_csv), exist_ok=True)

    desired_header = list(row.keys())

    if not os.path.exists(experiments_csv):
        with open(experiments_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_header)
            w.writeheader()
            w.writerow(row)
        print("Logged row to", experiments_csv)
        return

    try:
        with open(experiments_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
    except Exception:
        existing_header = None

    if existing_header == desired_header:
        with open(experiments_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_header)
            w.writerow(row)
        print("Logged row to", experiments_csv)
        return

    root, ext = os.path.splitext(experiments_csv)
    alt_csv = root + "_kexit" + ext

    write_header = not os.path.exists(alt_csv)
    with open(alt_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=desired_header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"[summarize_run] Existing CSV schema differs; logged row to {alt_csv} instead of {experiments_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--report_name", default="summary.json")
    ap.add_argument("--experiments_csv", default="runs/experiments.csv")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--tap_blocks", type=str, default=None)
    ap.add_argument("--n_mels", type=int, default=None)
    ap.add_argument(
        "--no_log",
        action="store_true",
        help="Do not append a row to runs/experiments.csv (useful when called from run_reports.ps1).",
    )

    args = ap.parse_args()

    metrics = load_json_safepath(os.path.join(args.run_dir, "metrics.json"), {})
    report = load_json_safepath(os.path.join(args.run_dir, "report.json"), {})
    calib = load_json_safepath(os.path.join(args.run_dir, "temperature.json"), {})
    thres = load_json_safepath(os.path.join(args.run_dir, "thresholds.json"), {})

    policy = policy_eval(
        args.run_dir,
        args.segments_csv,
        args.features_root,
        save_plots=not args.no_plots,
        n_mels_override=args.n_mels,
        tap_blocks_override=_parse_tap_blocks(args.tap_blocks),
    )

    policy_results = load_json_safepath(os.path.join(args.run_dir, "policy_results.json"), {})
    clip_full = load_json_safepath(os.path.join(args.run_dir, "clip_policy_results_full.json"), {})
    clip_time = load_json_safepath(os.path.join(args.run_dir, "clip_policy_results_time.json"), {})

    if policy_results:
        policy["policy_test_acc"] = policy_results.get("accuracy", policy.get("policy_test_acc"))
        policy["avg_exit_depth"] = policy_results.get("avg_exit_depth", policy.get("avg_exit_depth"))
        policy["exit_mix"] = policy_results.get("exit_mix", policy.get("exit_mix"))
        policy["flip_any_rate"] = policy_results.get("flip_any_rate")
        policy["avg_flip_count"] = policy_results.get("avg_flip_count")
        policy["exit_consistency"] = policy_results.get("exit_consistency")
        policy["n_segments"] = policy_results.get("n_segments", policy_results.get("n_samples"))

    run_id = os.path.basename(args.run_dir.rstrip("\\/"))
    summary = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_tail": metrics.get("val", [-1])[-1] if metrics else None,
        "val_curve": metrics.get("val", []),
        "test_report": report,
        "temperature": calib,
        "thresholds": thres,
        "policy_summary": policy,
        "policy_results": policy_results,
        "clip_policy_results_full": clip_full,
        "clip_policy_results_time": clip_time,
    }

    out_json = os.path.join(args.run_dir, args.report_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved", out_json)

    if args.no_log:
        print("[summarize_run] --no_log set: skipping append to", args.experiments_csv)
        return

    per_exit_ece = {
        k: v.get("ece")
        for k, v in (policy.get("per_exit_calibration") or {}).items()
    }

    row = {
        "run_id": run_id,
        "num_exits": policy.get("num_exits"),
        "tap_blocks_json": json.dumps(policy.get("tap_blocks")),
        "tau": policy.get("tau"),
        "temperatures_json": json.dumps(policy.get("temperatures")),
        "test_acc_policy": policy.get("policy_test_acc"),
        "avg_exit_depth": policy.get("avg_exit_depth"),
        "exit_mix_json": json.dumps(policy.get("exit_mix")),
        "expected_mflops": policy.get("expected_mflops"),
        "full_mflops": policy.get("full_mflops"),
        "compute_saving_pct": policy.get("compute_saving_pct"),
        "n_mels": policy.get("n_mels"),
        "frames": policy.get("frames"),
        "num_classes": policy.get("num_classes"),
        "torch_version": policy.get("torch_version"),
        "cuda": policy.get("cuda_available"),
        "ece_policy": (policy.get("policy_calibration") or {}).get("ece"),
        "per_exit_ece_json": json.dumps(per_exit_ece),
        "flops_warning": policy.get("flops_warning"),
    }

    _append_experiments_row_safely(args.experiments_csv, row)


if __name__ == "__main__":
    main()