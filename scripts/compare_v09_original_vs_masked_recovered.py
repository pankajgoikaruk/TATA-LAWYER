#!/usr/bin/env python
"""
Fair recovered-domain comparison:
  original v0.9 checkpoint vs human-reviewed masked v0.9 checkpoint

Both models:
- use the same recovered human-reviewed validation rows
- tune one threshold per label on that validation subset
- use the same threshold grid and tie-break rule
- fall back to 0.50 when recovered validation has no positive support
- evaluate once on the same recovered human-reviewed test rows

No model is retrained or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import (
    MaskedMultiLabelLogMelDataset,
    load_labels,
)
from utils.model_factory import build_audio_exit_net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original_run_dir", type=Path, required=True)
    p.add_argument("--masked_run_dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features_root", type=Path, required=True)
    p.add_argument("--labels_json", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grid_start", type=float, default=0.05)
    p.add_argument("--grid_end", type=float, default=0.95)
    p.add_argument("--grid_step", type=float, default=0.01)
    p.add_argument("--fallback_threshold", type=float, default=0.50)
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tap_blocks(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def build_model(run_dir: Path, labels: list[str], device: str):
    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    config = load_json(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    tap_blocks = parse_tap_blocks(config.get("tap_blocks", [1, 3]))
    n_mels = int(config.get("n_mels", 64))
    exit_hint = config.get(
        "exit_hint",
        {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        },
    )

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg={"exit_hint": exit_hint},
    ).to(device)

    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model, config, checkpoint


def make_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


@torch.no_grad()
def collect_final_exit(model, loader, device: str):
    targets, masks, probs = [], [], []

    for x, y, mask in loader:
        x = x.to(device)
        logits = model(x)[-1]
        targets.append(y.numpy())
        masks.append(mask.numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())

    return (
        np.concatenate(targets).astype(int),
        np.concatenate(masks).astype(int),
        np.concatenate(probs).astype(np.float32),
    )


def binary_metrics(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def threshold_grid(start, end, step):
    if not (0 < start <= end < 1):
        raise ValueError("Require 0 < grid_start <= grid_end < 1")
    if step <= 0:
        raise ValueError("grid_step must be > 0")
    count = int(round((end - start) / step)) + 1
    values = start + np.arange(count) * step
    return np.round(values[values <= end + 1e-12], 6)


def tune_masked_thresholds(
    y_true,
    mask,
    probabilities,
    labels,
    grid,
    fallback_threshold,
):
    thresholds = np.full(len(labels), fallback_threshold, dtype=np.float64)
    rows = []

    for idx, label in enumerate(labels):
        known = mask[:, idx].astype(bool)
        target = y_true[known, idx]
        scores = probabilities[known, idx]
        support = int(target.sum())

        fallback_pred = (scores >= fallback_threshold).astype(int)
        fallback_metrics = binary_metrics(target, fallback_pred)

        if support == 0:
            selected = fallback_threshold
            selected_metrics = fallback_metrics
            status = "fallback_no_positive_validation_support"
        else:
            candidates = []
            for threshold in grid:
                pred = (scores >= threshold).astype(int)
                metrics = binary_metrics(target, pred)
                candidates.append(
                    {
                        "threshold": float(threshold),
                        **metrics,
                        "distance_from_0_5": abs(float(threshold) - 0.5),
                    }
                )

            best = sorted(
                candidates,
                key=lambda item: (
                    -item["f1"],
                    item["distance_from_0_5"],
                    -item["threshold"],
                ),
            )[0]
            selected = best["threshold"]
            selected_metrics = best
            status = "tuned_on_recovered_validation"

        thresholds[idx] = selected
        rows.append(
            {
                "label": label,
                "known_validation_count": int(known.sum()),
                "validation_positive_support": support,
                "fallback_threshold": fallback_threshold,
                "fallback_val_f1": fallback_metrics["f1"],
                "selected_threshold": selected,
                "selected_val_precision": selected_metrics["precision"],
                "selected_val_recall": selected_metrics["recall"],
                "selected_val_f1": selected_metrics["f1"],
                "selection_status": status,
            }
        )

    return thresholds, rows


def evaluate_masked(y_true, mask, probabilities, thresholds, labels):
    pred = (probabilities >= thresholds.reshape(1, -1)).astype(int)
    known = mask.astype(bool)
    fully_known = known.all(axis=1)

    flat_true = y_true[known]
    flat_pred = pred[known]

    per_label = {}
    label_f1 = []
    sample_f1 = []
    known_exact = []

    for idx, label in enumerate(labels):
        k = known[:, idx]
        metrics = binary_metrics(y_true[k, idx], pred[k, idx])
        metrics["known_count"] = int(k.sum())
        metrics["unknown_count"] = int((~k).sum())
        per_label[label] = metrics
        label_f1.append(metrics["f1"])

    for yt, yp, km in zip(y_true, pred, known):
        if not km.any():
            continue
        sample_f1.append(float(f1_score(yt[km], yp[km], zero_division=0)))
        known_exact.append(bool(np.all(yt[km] == yp[km])))

    return {
        "rows": int(len(y_true)),
        "known_label_decisions": int(known.sum()),
        "fully_known_rows": int(fully_known.sum()),
        "macro_f1_known_labels": float(np.mean(label_f1)),
        "micro_f1_known_decisions": float(
            f1_score(flat_true, flat_pred, zero_division=0)
        ),
        "samples_f1_known_labels": float(np.mean(sample_f1)),
        "known_labels_exact_match": float(np.mean(known_exact)),
        "exact_match_fully_known_rows": float(
            np.mean(np.all(y_true[fully_known] == pred[fully_known], axis=1))
        )
        if fully_known.any()
        else 0.0,
        "masked_hamming_loss": float(np.mean(flat_true != flat_pred)),
        "per_label": per_label,
    }


def run_model(
    name,
    run_dir,
    labels,
    val_loader,
    test_loader,
    device,
    grid,
    fallback_threshold,
):
    model, config, checkpoint = build_model(run_dir, labels, device)

    y_val, m_val, p_val = collect_final_exit(model, val_loader, device)
    y_test, m_test, p_test = collect_final_exit(model, test_loader, device)

    thresholds, threshold_rows = tune_masked_thresholds(
        y_val,
        m_val,
        p_val,
        labels,
        grid,
        fallback_threshold,
    )

    fixed = evaluate_masked(
        y_test,
        m_test,
        p_test,
        np.full(len(labels), fallback_threshold),
        labels,
    )
    tuned = evaluate_masked(
        y_test,
        m_test,
        p_test,
        thresholds,
        labels,
    )

    return {
        "name": name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "config": config,
        "thresholds": {
            label: float(thresholds[idx])
            for idx, label in enumerate(labels)
        },
        "threshold_tuning": threshold_rows,
        "test_fixed_0_5": fixed,
        "test_recovered_tuned": tuned,
    }


def main():
    args = parse_args()

    original_run = args.original_run_dir.expanduser().resolve()
    masked_run = args.masked_run_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for run_dir, name in [
        (original_run, "original run"),
        (masked_run, "masked run"),
    ]:
        if not (run_dir / "config_used.json").is_file():
            raise FileNotFoundError(f"{name} config missing: {run_dir}")
        if not (run_dir / "ckpt" / "best.pt").is_file():
            raise FileNotFoundError(f"{name} checkpoint missing: {run_dir}")

    labels = load_labels(labels_json)

    if str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    val_ds = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="val",
        filter_column="v09_masked_review_applied",
        filter_value=1,
    )
    test_ds = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_masked_review_applied",
        filter_value=1,
    )

    val_loader = make_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)
    grid = threshold_grid(args.grid_start, args.grid_end, args.grid_step)

    print("\nRecovered-domain model comparison")
    print("-" * 90)
    print(f"Recovered validation rows: {len(val_ds):,}")
    print(f"Recovered test rows:       {len(test_ds):,}")
    print(f"Device:                    {device}")
    print("-" * 90)

    original = run_model(
        "original_v09",
        original_run,
        labels,
        val_loader,
        test_loader,
        device,
        grid,
        args.fallback_threshold,
    )
    masked = run_model(
        "masked_human_reviewed_v09",
        masked_run,
        labels,
        val_loader,
        test_loader,
        device,
        grid,
        args.fallback_threshold,
    )

    comparison_rows = []
    metric_map = [
        ("Macro-F1", "macro_f1_known_labels"),
        ("Micro-F1", "micro_f1_known_decisions"),
        ("Samples-F1", "samples_f1_known_labels"),
        ("Known-label Exact", "known_labels_exact_match"),
        ("Exact Fully Known", "exact_match_fully_known_rows"),
        ("Masked Hamming", "masked_hamming_loss"),
    ]

    for display, key in metric_map:
        original_value = original["test_recovered_tuned"][key]
        masked_value = masked["test_recovered_tuned"][key]
        comparison_rows.append(
            {
                "metric": display,
                "original_v09": original_value,
                "masked_v09": masked_value,
                "masked_minus_original": masked_value - original_value,
            }
        )

    per_label_rows = []
    for label in labels:
        o = original["test_recovered_tuned"]["per_label"][label]
        m = masked["test_recovered_tuned"]["per_label"][label]
        per_label_rows.append(
            {
                "label": label,
                "original_threshold": original["thresholds"][label],
                "masked_threshold": masked["thresholds"][label],
                "original_test_f1": o["f1"],
                "masked_test_f1": m["f1"],
                "masked_minus_original_f1": m["f1"] - o["f1"],
                "test_support": m["support"],
                "known_test_count": m["known_count"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "selection_split": "same_recovered_human_reviewed_validation",
        "selection_rows": len(val_ds),
        "evaluation_split": "same_recovered_human_reviewed_test",
        "evaluation_rows": len(test_ds),
        "fallback_threshold": args.fallback_threshold,
        "original": original,
        "masked": masked,
    }
    (output_dir / "original_vs_masked_recovered_comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "original_vs_masked_recovered_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(per_label_rows).to_csv(
        output_dir / "original_vs_masked_recovered_per_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRecovered tuned-profile comparison")
    print("-" * 90)
    print(
        f"{'Metric':<24}"
        f"{'Original v0.9':>16}"
        f"{'Masked v0.9':>16}"
        f"{'Change':>12}"
    )
    for row in comparison_rows:
        print(
            f"{row['metric']:<24}"
            f"{row['original_v09']:>16.4f}"
            f"{row['masked_v09']:>16.4f}"
            f"{row['masked_minus_original']:>+12.4f}"
        )

    print("\nSaved outputs")
    print("-" * 90)
    print(output_dir / "original_vs_masked_recovered_comparison.json")
    print(output_dir / "original_vs_masked_recovered_metrics.csv")
    print(output_dir / "original_vs_masked_recovered_per_label.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
