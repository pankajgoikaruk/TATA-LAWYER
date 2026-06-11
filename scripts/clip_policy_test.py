# scripts/clip_policy_test.py

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np
import pandas as pd
import torch

from utils.config import load_config
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def _infer_tap_blocks_from_run(run_dir: str) -> Optional[tuple]:
    """
    Try to recover tap_blocks from training artifacts.
    Priority:
      1) config_used.yaml -> model.tap_blocks
      2) metrics.json -> meta.tap_blocks
      3) temperature.json -> tap_blocks
    """
    run_model_cfg = load_run_model_cfg(run_dir)
    tb = _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
    if tb is not None:
        return tb

    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            obj = _load_json(metrics_path)
            meta = obj.get("meta", {})
            tb = meta.get("tap_blocks")
            tb = _parse_tap_blocks(tb)
            if tb is not None:
                return tb
        except Exception:
            pass

    temp_path = os.path.join(run_dir, "temperature.json")
    if os.path.exists(temp_path):
        try:
            obj = _load_json(temp_path)
            tb = obj.get("tap_blocks")
            tb = _parse_tap_blocks(tb)
            if tb is not None:
                return tb
        except Exception:
            pass

    return None


def _build_model(run_dir: str, num_classes: int, device: str, n_mels: int, tap_blocks=None):
    run_model_cfg = load_run_model_cfg(run_dir)

    if tap_blocks is None:
        tap_blocks = _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
    if tap_blocks is None:
        tap_blocks = _infer_tap_blocks_from_run(run_dir)
    if tap_blocks is None:
        tap_blocks = (1, 3)

    model = build_audio_exit_net(
        num_classes=int(num_classes),
        n_mels=int(n_mels),
        tap_blocks=tap_blocks,
        model_cfg=run_model_cfg,
    ).to(device)

    ckpt = os.path.join(run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    return model, tap_blocks, run_model_cfg


def _load_feature(features_root: Path, feat_relpath: str, device: str) -> torch.Tensor:
    path = features_root / Path(str(feat_relpath).replace("\\", "/"))
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    S = np.load(path)  # (M, T)
    x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0)  # (1,1,M,T)
    return x.to(device)


def _top1_margin(prob: torch.Tensor) -> float:
    vals, _ = torch.topk(prob, k=min(2, prob.numel()))
    if vals.numel() == 1:
        return float(vals[0].item())
    return float((vals[0] - vals[1]).item())


def _group_name_from_used(used: int) -> str:
    if used <= 2:
        return "stop_2"
    if used <= 4:
        return "stop_3_4"
    return "stop_5_plus"


def _load_temps_and_tau(run_dir: str):
    tpath = os.path.join(run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath).get("temperatures", [1.0, 1.0, 1.0])
    else:
        temps = [1.0, 1.0, 1.0]
    temps = [max(float(t), 1e-3) for t in temps]

    th_path = os.path.join(run_dir, "thresholds.json")
    if not os.path.exists(th_path):
        raise FileNotFoundError(f"thresholds.json not found in run_dir: {th_path}")
    tau = float(_load_json(th_path)["tau"])

    return temps, tau


def _greedy_decide_one(logits_list, temps, tau):
    scaled = [lg / max(float(temps[i]), 1e-3) for i, lg in enumerate(logits_list)]
    probs = [torch.softmax(lg, dim=0) for lg in scaled]
    preds = [int(torch.argmax(p).item()) for p in probs]

    taken = len(probs) - 1
    for k in range(len(probs) - 1):
        if float(torch.max(probs[k]).item()) >= tau:
            taken = k
            break

    pred_taken = preds[taken]
    pred_final = preds[-1]
    flip_any = int(len(set(preds)) > 1)
    taken_logp = torch.log_softmax(scaled[taken], dim=0)

    return {
        "taken": int(taken),
        "pred_taken": int(pred_taken),
        "pred_final": int(pred_final),
        "flip_any": int(flip_any),
        "taken_logp": taken_logp.detach().cpu(),
    }


def _per_class_from_confusion(cm, labels):
    out = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = sum(cm[i])
        out[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    return out


def _window_distribution(vals):
    vals = list(vals)
    if len(vals) == 0:
        return {
            "min": 0,
            "median": 0,
            "max": 0,
            "mean": 0.0,
            "hist": {},
            "n_clips": 0,
        }
    hist = {}
    for v in vals:
        hist[str(int(v))] = hist.get(str(int(v)), 0) + 1
    return {
        "min": int(min(vals)),
        "median": float(np.median(vals)),
        "max": int(max(vals)),
        "mean": float(np.mean(vals)),
        "hist": hist,
        "n_clips": int(len(vals)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--time_conf", type=float, default=0.95)
    ap.add_argument("--time_stable_k", type=int, default=2)
    ap.add_argument("--time_min_windows", type=int, default=2)
    ap.add_argument("--fixed_k_windows", type=int, default=3)
    ap.add_argument("--time_margin", type=float, default=0.0)

    # Generic K-exit args
    ap.add_argument("--tap_blocks", type=str, default=None,
                    help='Example: "1,3" for 3 exits or "1,2,3,4" for 5 exits.')
    ap.add_argument("--n_mels", type=int, default=None)

    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.segments_csv)
    required_cols = ["wav_relpath", "label", "start", "split", "feat_relpath"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"segments.csv missing required columns: {missing}")

    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found in segments.csv")

    test_df["wav_relpath"] = test_df["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    test_df["feat_relpath"] = test_df["feat_relpath"].astype(str).str.replace("\\", "/", regex=False)
    test_df = test_df.sort_values(["wav_relpath", "start"]).reset_index(drop=True)

    labels = sorted(df["label"].astype(str).unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}

    run_cfg = _load_run_cfg(args.run_dir)
    run_features_cfg = run_cfg.get("features") or {}

    tap_blocks = _parse_tap_blocks(args.tap_blocks)
    if tap_blocks is None:
        tap_blocks = _infer_tap_blocks_from_run(args.run_dir)

    if args.n_mels is None:
        n_mels = int(run_features_cfg.get("n_mels", 64))
    else:
        n_mels = int(args.n_mels)

    model, effective_tap_blocks, run_model_cfg = _build_model(
        run_dir=args.run_dir,
        num_classes=len(labels),
        device=device,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
    )

    features_root = Path(args.features_root)
    temps, tau = _load_temps_and_tau(args.run_dir)
    temps = _pad_or_trim(temps, model.num_exits)

    n_clips = 0
    num_exits = model.num_exits

    # Full baseline accumulators
    full_correct_clip = 0
    full_correct_seg = 0
    full_n_seg = 0
    full_compute_per_clip = []
    full_used_windows_per_clip = []
    full_exit_counts = {f"e{i+1}": 0 for i in range(num_exits)}
    full_flip_total = 0
    full_consistent_total = 0
    full_cm = [[0 for _ in labels] for _ in labels]

    # Time-exit accumulators
    time_correct_clip = 0
    time_correct_seg = 0
    time_n_seg = 0
    time_compute_per_clip = []
    time_used_windows_per_clip = []
    time_exit_counts = {f"e{i+1}": 0 for i in range(num_exits)}
    time_flip_total = 0
    time_consistent_total = 0
    time_cm = [[0 for _ in labels] for _ in labels]
    stop_reasons = {}
    stop_groups_firstK = {"stop_2": 0, "stop_3_4": 0, "stop_5_plus": 0}

    # Fixed-position diagnostics
    diag_first_correct = 0
    diag_first_n = 0
    diag_mid_correct = 0
    diag_mid_n = 0
    diag_last_correct = 0
    diag_last_n = 0

    clip_groups = list(test_df.groupby("wav_relpath", sort=False))

    with torch.no_grad():
        for wav_relpath, g in clip_groups:
            n_clips += 1
            g = g.sort_values("start").reset_index(drop=True)

            true_label = str(g.iloc[0]["label"])
            y_true = label2id[true_label]

            window_decisions = []

            # Per-window inference
            for _, row in g.iterrows():
                x = _load_feature(features_root, row["feat_relpath"], device)
                logits_list = model(x)
                logits_list = [lg.squeeze(0) for lg in logits_list]

                dec = _greedy_decide_one(logits_list, temps, tau)
                dec["true"] = y_true
                dec["wav_relpath"] = wav_relpath
                window_decisions.append(dec)

            # Full baseline: use all windows
            full_logp_sum = torch.zeros(len(labels))
            full_compute = 0

            for dec in window_decisions:
                full_correct_seg += int(dec["pred_taken"] == y_true)
                full_n_seg += 1
                full_compute += int(dec["taken"] + 1)
                full_exit_counts[f"e{dec['taken'] + 1}"] += 1
                full_flip_total += int(dec["flip_any"])
                full_consistent_total += int(dec["pred_taken"] == dec["pred_final"])
                full_logp_sum += dec["taken_logp"]

            full_clip_pred = int(torch.argmax(full_logp_sum).item())
            full_correct_clip += int(full_clip_pred == y_true)
            full_cm[y_true][full_clip_pred] += 1
            full_compute_per_clip.append(full_compute)
            full_used_windows_per_clip.append(len(window_decisions))

            # Fixed-position diagnostics
            K = max(int(args.fixed_k_windows), 1)
            n_win = len(window_decisions)

            first_slice = window_decisions[: min(K, n_win)]
            for dec in first_slice:
                diag_first_correct += int(dec["pred_taken"] == y_true)
                diag_first_n += 1

            mid_start = max((n_win - K) // 2, 0)
            mid_slice = window_decisions[mid_start: mid_start + min(K, n_win)]
            for dec in mid_slice:
                diag_mid_correct += int(dec["pred_taken"] == y_true)
                diag_mid_n += 1

            last_slice = window_decisions[max(0, n_win - K):]
            for dec in last_slice:
                diag_last_correct += int(dec["pred_taken"] == y_true)
                diag_last_n += 1

            # Time-exit over clips
            time_logp_sum = torch.zeros(len(labels))
            time_compute = 0
            time_used = 0
            clip_pred_history = []
            stop_reason = "max_windows"

            for dec in window_decisions:
                time_used += 1
                time_compute += int(dec["taken"] + 1)
                time_correct_seg += int(dec["pred_taken"] == y_true)
                time_n_seg += 1
                time_exit_counts[f"e{dec['taken'] + 1}"] += 1
                time_flip_total += int(dec["flip_any"])
                time_consistent_total += int(dec["pred_taken"] == dec["pred_final"])

                time_logp_sum += dec["taken_logp"]
                post = torch.softmax(time_logp_sum, dim=0)
                clip_pred = int(torch.argmax(post).item())
                clip_pred_history.append(clip_pred)

                conf = float(torch.max(post).item())
                margin = _top1_margin(post)

                stable_ok = False
                if len(clip_pred_history) >= args.time_stable_k:
                    tail = clip_pred_history[-args.time_stable_k:]
                    stable_ok = len(set(tail)) == 1

                if (
                    time_used >= args.time_min_windows
                    and conf >= args.time_conf
                    and margin >= args.time_margin
                    and stable_ok
                ):
                    stop_reason = "time_conf_stable"
                    break

            time_clip_pred = int(torch.argmax(time_logp_sum).item())
            time_correct_clip += int(time_clip_pred == y_true)
            time_cm[y_true][time_clip_pred] += 1
            time_compute_per_clip.append(time_compute)
            time_used_windows_per_clip.append(time_used)
            stop_reasons[stop_reason] = stop_reasons.get(stop_reason, 0) + 1
            stop_groups_firstK[_group_name_from_used(time_used)] += 1

    # Aggregate
    full_clip_acc = full_correct_clip / max(n_clips, 1)
    full_seg_acc = full_correct_seg / max(full_n_seg, 1)
    full_avg_compute = float(mean(full_compute_per_clip)) if full_compute_per_clip else 0.0
    full_avg_windows_used = float(mean(full_used_windows_per_clip)) if full_used_windows_per_clip else 0.0
    full_avg_windows_total = full_avg_windows_used
    full_avg_depth_per_used_window = (
        full_avg_compute / max(full_avg_windows_used, 1e-12)
        if full_avg_windows_used > 0 else 0.0
    )
    full_flip_rate = full_flip_total / max(full_n_seg, 1)
    full_exit_consistency = full_consistent_total / max(full_n_seg, 1)

    time_clip_acc = time_correct_clip / max(n_clips, 1)
    time_seg_acc = time_correct_seg / max(time_n_seg, 1)
    time_avg_compute = float(mean(time_compute_per_clip)) if time_compute_per_clip else 0.0
    time_avg_windows_used = float(mean(time_used_windows_per_clip)) if time_used_windows_per_clip else 0.0
    time_avg_windows_total = float(mean(full_used_windows_per_clip)) if full_used_windows_per_clip else 0.0
    time_avg_fraction_windows_used = (
        time_avg_windows_used / max(time_avg_windows_total, 1e-12)
        if time_avg_windows_total > 0 else 0.0
    )
    time_windows_saved_pct = 100.0 * (1.0 - time_avg_fraction_windows_used)
    time_compute_saved_pct = 100.0 * (1.0 - (time_avg_compute / max(full_avg_compute, 1e-12)))
    time_avg_depth_per_used_window = (
        time_avg_compute / max(time_avg_windows_used, 1e-12)
        if time_avg_windows_used > 0 else 0.0
    )
    time_flip_rate = time_flip_total / max(time_n_seg, 1)
    time_exit_consistency = time_consistent_total / max(time_n_seg, 1)

    for k in full_exit_counts:
        full_exit_counts[k] = full_exit_counts[k] / max(full_n_seg, 1)
    for k in time_exit_counts:
        time_exit_counts[k] = time_exit_counts[k] / max(time_n_seg, 1)

    first_acc = diag_first_correct / max(diag_first_n, 1)
    mid_acc = diag_mid_correct / max(diag_mid_n, 1)
    last_acc = diag_last_correct / max(diag_last_n, 1)

    hist_payload = _window_distribution(time_used_windows_per_clip)
    hist_path = os.path.join(args.run_dir, "windows_used_hist.json")
    _save_json(hist_path, hist_payload)

    common_meta = {
        "policy": "greedy",
        "K": int(num_exits),
        "tap_blocks": list(effective_tap_blocks) if effective_tap_blocks is not None else None,
        "n_mels": int(n_mels),
        "num_classes": len(labels),
        "window_distribution": hist_payload,
        "window_distribution_json_mode": "windows_used_hist.json",
        "window_distribution_json_legacy": "windows_used_hist.json",
        "diagnostic_fixed_k": int(args.fixed_k_windows),
        "diagnostic_acc_firstK": float(first_acc),
        "diagnostic_firstK_n_segments": int(diag_first_n),
        "diagnostic_acc_midK": float(mid_acc),
        "diagnostic_midK_n_segments": int(diag_mid_n),
        "diagnostic_acc_lastK": float(last_acc),
        "diagnostic_lastK_n_segments": int(diag_last_n),
        "diagnostic_stop_groups_firstK": stop_groups_firstK,
        "n_clips": int(n_clips),
        "clip_agg": "sum_logp_over_segments",
        "temperatures_used": temps,
        "tau": tau,
        "labels": labels,
        "exit_hint": (run_model_cfg or {}).get("exit_hint", {}),
    }

    full_result = {
        **common_meta,
        "segment_accuracy_over_processed_windows": float(full_seg_acc),
        "segment_accuracy_over_used_windows": float(full_seg_acc),
        "n_segments_processed_windows": int(full_n_seg),
        "n_segments_used_windows": int(full_n_seg),
        "clip_accuracy": float(full_clip_acc),
        "time_exit_enabled": False,
        "time_params": {
            "time_min_windows": int(args.time_min_windows),
            "time_stable_k": int(args.time_stable_k),
            "time_conf": float(args.time_conf),
            "time_margin": float(args.time_margin),
            "time_max_windows": None,
        },
        "avg_windows_used": float(full_avg_windows_used),
        "avg_windows_total": float(full_avg_windows_total),
        "avg_fraction_windows_used": 1.0,
        "avg_compute_units_sum_depth_over_used_windows": float(full_avg_compute),
        "avg_depth_per_used_window": float(full_avg_depth_per_used_window),
        "windows_saved_pct": 0.0,
        "compute_full_ref_avg_units": float(full_avg_compute),
        "compute_saved_pct": 0.0,
        "exit_mix_over_used_windows": full_exit_counts,
        "flip_rate_over_used_windows": float(full_flip_rate),
        "exit_consistency_taken_vs_final_over_used_windows": float(full_exit_consistency),
        "clip_confusion_matrix": full_cm,
        "per_class": _per_class_from_confusion(full_cm, labels),
        "stop_reasons": {"full_windows": int(n_clips)},
    }

    time_result = {
        **common_meta,
        "segment_accuracy_over_used_windows": float(time_seg_acc),
        "n_segments_used_windows": int(time_n_seg),
        "clip_accuracy": float(time_clip_acc),
        "time_exit_enabled": True,
        "time_params": {
            "time_min_windows": int(args.time_min_windows),
            "time_stable_k": int(args.time_stable_k),
            "time_conf": float(args.time_conf),
            "time_margin": float(args.time_margin),
            "time_max_windows": None,
        },
        "avg_windows_used": float(time_avg_windows_used),
        "avg_windows_total": float(time_avg_windows_total),
        "avg_fraction_windows_used": float(time_avg_fraction_windows_used),
        "avg_compute_units_sum_depth_over_used_windows": float(time_avg_compute),
        "avg_depth_per_used_window": float(time_avg_depth_per_used_window),
        "windows_saved_pct": float(time_windows_saved_pct),
        "compute_full_ref_avg_units": float(full_avg_compute),
        "compute_saved_pct": float(time_compute_saved_pct),
        "exit_mix_over_used_windows": time_exit_counts,
        "flip_rate_over_used_windows": float(time_flip_rate),
        "exit_consistency_taken_vs_final_over_used_windows": float(time_exit_consistency),
        "clip_confusion_matrix": time_cm,
        "per_class": _per_class_from_confusion(time_cm, labels),
        "stop_reasons": stop_reasons,
    }

    out_full = os.path.join(args.run_dir, "clip_policy_results_full.json")
    out_time = os.path.join(args.run_dir, "clip_policy_results_time.json")
    out_legacy = os.path.join(args.run_dir, "clip_policy_results.json")

    _save_json(out_full, full_result)
    _save_json(out_time, time_result)
    _save_json(out_legacy, time_result)

    print(f"[clip_policy_test] Wrote: {out_full}")
    print(f"[clip_policy_test] Wrote: {out_time}")
    print(f"[clip_policy_test] Wrote: {out_legacy}")

    full_exit_mix_str = ", ".join(f"{k}={v:.4f}" for k, v in full_exit_counts.items())
    time_exit_mix_str = ", ".join(f"{k}={v:.4f}" for k, v in time_exit_counts.items())

    print("")
    print("=== Full-clip baseline ===")
    print(f"Clip accuracy: {full_clip_acc:.4f}")
    print(f"Segment acc over processed windows: {full_seg_acc:.4f} (n_segments={full_n_seg})")
    print(
        f"Fixed-position diagnostic (K={args.fixed_k_windows}): "
        f"first={first_acc:.4f} (n_segments={diag_first_n}), "
        f"mid={mid_acc:.4f} (n_segments={diag_mid_n}), "
        f"last={last_acc:.4f} (n_segments={diag_last_n})"
    )
    print(f"Avg windows used: {full_avg_windows_used:.3f} / {full_avg_windows_total:.3f} (100.00%)")
    print("Windows saved: 0.00%")
    print(f"Avg compute units: {full_avg_compute:.3f}")
    print("Compute saved: 0.00%")
    print(f"Avg depth per used window: {full_avg_depth_per_used_window:.3f}")
    print(f"Flip-rate (used windows): {full_flip_rate:.4f}")
    print(f"Exit-consistency (taken==final over used windows): {full_exit_consistency:.4f}")
    print(f"Exit mix over used windows: {full_exit_mix_str}")
    print(f"Confusion matrix: {full_cm}")
    print(f"Per-class: {_per_class_from_confusion(full_cm, labels)}")

    print("")
    print("=== Depth×Time ===")
    print(f"Clip accuracy: {time_clip_acc:.4f}")
    print(f"Segment acc over used windows: {time_seg_acc:.4f} (n_segments={time_n_seg})")
    print(
        f"Fixed-position diagnostic (K={args.fixed_k_windows}): "
        f"first={first_acc:.4f} (n_segments={diag_first_n}), "
        f"mid={mid_acc:.4f} (n_segments={diag_mid_n}), "
        f"last={last_acc:.4f} (n_segments={diag_last_n})"
    )
    print(f"Avg windows used: {time_avg_windows_used:.3f} / {time_avg_windows_total:.3f}")
    print(f"Windows saved: {time_windows_saved_pct:.2f}%")
    print(f"Avg compute units: {time_avg_compute:.3f}")
    print(f"Avg depth per used window: {time_avg_depth_per_used_window:.3f}")
    print(f"Compute saved: {time_compute_saved_pct:.2f}%")
    print(f"Flip-rate (used windows): {time_flip_rate:.4f}")
    print(f"Exit-consistency (taken==final over used windows): {time_exit_consistency:.4f}")
    print(f"Exit mix over used windows: {time_exit_mix_str}")
    print(f"Confusion matrix: {time_cm}")
    print(f"Per-class: {_per_class_from_confusion(time_cm, labels)}")


if __name__ == "__main__":
    main()