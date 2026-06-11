# 1. load best.pt
# 2. run validation prediction
# 3. find best threshold per label
# 4. evaluate test set using per-label thresholds
# 5. compare fixed 0.5 vs tuned thresholds


# scripts/tune_multilabel_thresholds.py
#
# Tune per-label sigmoid thresholds for a trained multi-label ASHADIP model.
#
# Input:
#   runs_multilabel/<run_name>/ckpt/best.pt
#   runs_multilabel/<run_name>/config_used.json
#   multilabel_cache/metadata/multilabel_features_manifest.csv
#   multilabel_cache/features
#   multilabel_data/metadata/labels.json
#
# Output:
#   <run_dir>/threshold_tuning/multilabel_thresholds.json
#   <run_dir>/threshold_tuning/threshold_comparison.json
#
# Example:
#   python scripts\tune_multilabel_thresholds.py `
#     --run_dir "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
#     --device cpu
#
#   python scripts\tune_multilabel_thresholds.py `
#     --run_dir "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
#     --device cpu

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

# Make project root importable when running:
# python scripts\tune_multilabel_thresholds.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import make_multilabel_loaders
from utils.model_factory import build_audio_exit_net


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if not value:
        raise ValueError("tap_blocks cannot be empty.")

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def load_model_state(model, ckpt_path: Path, device: str):
    """
    Load model weights.

    Uses weights_only=True if supported by the installed torch version.
    Falls back safely for older torch versions.
    """
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)
    return model


@torch.no_grad()
def collect_probs_and_targets(model, dl, device: str):
    """
    Return:
      y_true: [N, C]
      probs_by_exit: list of [N, C], one array per exit
    """
    model.eval()

    y_parts = []
    probs_by_exit = None

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        logits_list = model(x)
        probs_list = [torch.sigmoid(logits) for logits in logits_list]

        if probs_by_exit is None:
            probs_by_exit = [[] for _ in probs_list]

        y_parts.append(y.detach().cpu().numpy())

        for k, probs in enumerate(probs_list):
            probs_by_exit[k].append(probs.detach().cpu().numpy())

    if not y_parts:
        raise RuntimeError("No data found while collecting probabilities.")

    y_true = np.concatenate(y_parts, axis=0).astype(int)
    probs_by_exit = [np.concatenate(parts, axis=0) for parts in probs_by_exit]

    return y_true, probs_by_exit


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray | float,
) -> dict[str, Any]:
    """
    Evaluate multi-label predictions.

    thresholds:
      - scalar float, e.g. 0.5
      - or vector shape [C]
    """
    if np.isscalar(thresholds):
        th = np.full(y_true.shape[1], float(thresholds), dtype=np.float32)
    else:
        th = np.asarray(thresholds, dtype=np.float32)

    y_pred = (y_prob >= th.reshape(1, -1)).astype(int)

    result = {
        "thresholds": {label: float(th[i]) for i, label in enumerate(labels)},
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "per_label": {},
    }

    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        result["per_label"][label] = {
            "threshold": float(th[i]),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }

    return result


def tune_thresholds_per_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    grid: np.ndarray,
    objective: str = "f1",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Tune one threshold per label on validation data.

    Default objective:
      maximize per-label F1.

    Returns:
      thresholds: [C]
      report: details per label
    """
    thresholds = np.zeros(len(labels), dtype=np.float32)
    report = {}

    for i, label in enumerate(labels):
        yt = y_true[:, i].astype(int)
        scores = y_prob[:, i].astype(np.float32)

        best_t = 0.5
        best_score = -1.0
        best_precision = 0.0
        best_recall = 0.0
        best_pred_pos = 0

        for t in grid:
            yp = (scores >= float(t)).astype(int)

            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)

            if objective == "f1":
                score = f1
            elif objective == "balanced_pr":
                score = 0.5 * precision + 0.5 * recall
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Tie-break rule:
            # prefer higher F1, then lower threshold for recall-sensitive labels.
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
                best_precision = float(precision)
                best_recall = float(recall)
                best_pred_pos = int(yp.sum())

        thresholds[i] = best_t

        report[label] = {
            "best_threshold": float(best_t),
            "val_best_score": float(best_score),
            "val_precision": float(best_precision),
            "val_recall": float(best_recall),
            "val_support": int(yt.sum()),
            "val_predicted_positive": int(best_pred_pos),
        }

    return thresholds, report


def make_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--step must be > 0")

    values = []
    t = float(min_threshold)

    while t <= float(max_threshold) + 1e-9:
        values.append(round(t, 6))
        t += float(step)

    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Tune per-label thresholds for trained multi-label ASHADIP model."
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Trained run directory containing config_used.json and ckpt/best.pt.",
    )

    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Default: <run_dir>/ckpt/best.pt",
    )

    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.05,
        help="Minimum threshold in search grid.",
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=0.95,
        help="Maximum threshold in search grid.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold grid step.",
    )

    parser.add_argument(
        "--objective",
        choices=["f1", "balanced_pr"],
        default="f1",
        help="Threshold tuning objective.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config_used.json.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config_used.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg = load_json(config_path)

    if args.device is None or str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "ckpt" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    manifest = Path(cfg["manifest"]).resolve()
    features_root = Path(cfg["features_root"]).resolve()
    labels_json = Path(cfg["labels_json"]).resolve()

    labels = [str(x) for x in cfg["labels"]]
    num_labels = int(cfg["num_labels"])
    n_mels = int(cfg.get("n_mels", 64))
    tap_blocks = parse_tap_blocks(cfg["tap_blocks"])
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))

    model_cfg = cfg.get("exit_hint", None)
    model_cfg = {
        "exit_hint": model_cfg if isinstance(model_cfg, dict) else {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        }
    }

    print("\nTuning multi-label thresholds")
    print("-" * 90)
    print(f"Run dir:       {run_dir}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Manifest:      {manifest}")
    print(f"Features root: {features_root}")
    print(f"Labels JSON:   {labels_json}")
    print(f"Device:        {device}")
    print(f"Labels:        {labels}")
    print(f"Tap blocks:    {tap_blocks}")
    print(f"n_mels:        {n_mels}")
    print(f"Batch size:    {batch_size}")
    print(f"Grid:          {args.min_threshold} to {args.max_threshold}, step={args.step}")
    print(f"Objective:     {args.objective}")
    print("-" * 90)

    dl_tr, dl_va, dl_te, loaded_labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        seed=int(cfg.get("seed", 42)),
        label_balance_power=0.0,
        synthetic_balance_power=0.0,
    )

    if list(loaded_labels) != list(labels):
        raise RuntimeError(
            "Label order mismatch between config and loaded dataset.\n"
            f"config labels: {labels}\n"
            f"loaded labels: {loaded_labels}"
        )

    model = build_audio_exit_net(
        num_classes=num_labels,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    model = load_model_state(model, checkpoint, device)

    y_val, val_probs_by_exit = collect_probs_and_targets(model, dl_va, device)
    y_test, test_probs_by_exit = collect_probs_and_targets(model, dl_te, device)

    num_exits = len(val_probs_by_exit)
    grid = make_threshold_grid(args.min_threshold, args.max_threshold, args.step)

    output_dir = run_dir / "threshold_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "labels": labels,
        "num_labels": num_labels,
        "num_exits": num_exits,
        "grid": {
            "min_threshold": float(args.min_threshold),
            "max_threshold": float(args.max_threshold),
            "step": float(args.step),
            "num_values": int(len(grid)),
        },
        "objective": args.objective,
        "exits": [],
    }

    print("\nThreshold tuning results")
    print("-" * 90)

    for exit_idx in range(num_exits):
        exit_no = exit_idx + 1

        y_val_prob = val_probs_by_exit[exit_idx]
        y_test_prob = test_probs_by_exit[exit_idx]

        tuned_thresholds, tune_report = tune_thresholds_per_label(
            y_true=y_val,
            y_prob=y_val_prob,
            labels=labels,
            grid=grid,
            objective=args.objective,
        )

        val_fixed = evaluate_predictions(
            y_true=y_val,
            y_prob=y_val_prob,
            labels=labels,
            thresholds=0.5,
        )
        val_tuned = evaluate_predictions(
            y_true=y_val,
            y_prob=y_val_prob,
            labels=labels,
            thresholds=tuned_thresholds,
        )
        test_fixed = evaluate_predictions(
            y_true=y_test,
            y_prob=y_test_prob,
            labels=labels,
            thresholds=0.5,
        )
        test_tuned = evaluate_predictions(
            y_true=y_test,
            y_prob=y_test_prob,
            labels=labels,
            thresholds=tuned_thresholds,
        )

        exit_payload = {
            "exit": exit_no,
            "tuned_thresholds": {
                label: float(tuned_thresholds[i])
                for i, label in enumerate(labels)
            },
            "tune_report_val": tune_report,
            "val_fixed_0p5": val_fixed,
            "val_tuned": val_tuned,
            "test_fixed_0p5": test_fixed,
            "test_tuned": test_tuned,
        }

        all_results["exits"].append(exit_payload)

        print(f"\nExit {exit_no}")
        print(
            f"  VAL  fixed: macroF1={val_fixed['macro_f1']:.4f}, "
            f"microF1={val_fixed['micro_f1']:.4f}, "
            f"exact={val_fixed['exact_match']:.4f}, "
            f"hamming={val_fixed['hamming_loss']:.4f}"
        )
        print(
            f"  VAL  tuned: macroF1={val_tuned['macro_f1']:.4f}, "
            f"microF1={val_tuned['micro_f1']:.4f}, "
            f"exact={val_tuned['exact_match']:.4f}, "
            f"hamming={val_tuned['hamming_loss']:.4f}"
        )
        print(
            f"  TEST fixed: macroF1={test_fixed['macro_f1']:.4f}, "
            f"microF1={test_fixed['micro_f1']:.4f}, "
            f"exact={test_fixed['exact_match']:.4f}, "
            f"hamming={test_fixed['hamming_loss']:.4f}"
        )
        print(
            f"  TEST tuned: macroF1={test_tuned['macro_f1']:.4f}, "
            f"microF1={test_tuned['micro_f1']:.4f}, "
            f"exact={test_tuned['exact_match']:.4f}, "
            f"hamming={test_tuned['hamming_loss']:.4f}"
        )

        if exit_no == num_exits:
            print("\n  Final-exit tuned thresholds:")
            for label in labels:
                t = exit_payload["tuned_thresholds"][label]
                before = test_fixed["per_label"][label]
                after = test_tuned["per_label"][label]
                print(
                    f"    {label:15s} t={t:.2f} | "
                    f"F1 fixed={before['f1']:.4f} -> tuned={after['f1']:.4f} | "
                    f"R fixed={before['recall']:.4f} -> tuned={after['recall']:.4f} | "
                    f"P fixed={before['precision']:.4f} -> tuned={after['precision']:.4f}"
                )

    comparison_path = output_dir / "threshold_comparison.json"
    save_json(all_results, comparison_path)

    # Also save final-exit thresholds in a small convenient file.
    final_exit = all_results["exits"][-1]
    final_thresholds_payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "selected_exit": int(num_exits),
        "labels": labels,
        "thresholds": final_exit["tuned_thresholds"],
        "test_fixed_0p5_summary": {
            k: final_exit["test_fixed_0p5"][k]
            for k in [
                "micro_f1",
                "macro_f1",
                "samples_f1",
                "exact_match",
                "hamming_loss",
                "avg_pred_labels",
                "avg_true_labels",
            ]
        },
        "test_tuned_summary": {
            k: final_exit["test_tuned"][k]
            for k in [
                "micro_f1",
                "macro_f1",
                "samples_f1",
                "exact_match",
                "hamming_loss",
                "avg_pred_labels",
                "avg_true_labels",
            ]
        },
    }

    thresholds_path = output_dir / "multilabel_thresholds.json"
    save_json(final_thresholds_payload, thresholds_path)

    print("\nSaved threshold tuning outputs:")
    print(f"  {comparison_path}")
    print(f"  {thresholds_path}")
    print("-" * 90)


if __name__ == "__main__":
    main()











