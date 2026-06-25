#!/usr/bin/env python
"""
Tune a second per-label threshold profile for recovered low-energy audio.

Selection:
  recovered human-reviewed validation rows only

Evaluation:
  recovered human-reviewed test rows only

For labels with no positive support in recovered validation, fall back to the
already selected strict/original-domain threshold instead of tuning on no data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import MaskedMultiLabelLogMelDataset, load_labels


def load_base_module():
    path = Path(__file__).resolve().parent / "tune_v09_masked_per_label_thresholds.py"
    if not path.is_file():
        raise FileNotFoundError(
            "Required base script not found:\n"
            f"{path}\n"
            "Keep both threshold-tuning scripts inside the scripts directory."
        )
    spec = importlib.util.spec_from_file_location("v09_strict_threshold_tuner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base threshold script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--strict_thresholds_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grid_start", type=float, default=0.05)
    parser.add_argument("--grid_end", type=float, default=0.95)
    parser.add_argument("--grid_step", type=float, default=0.01)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tune_masked_per_label(
    y_true: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    grid: np.ndarray,
    fallback_thresholds: np.ndarray,
    base,
):
    thresholds = fallback_thresholds.astype(np.float64).copy()
    rows = []

    for index, label in enumerate(labels):
        known = mask[:, index].astype(bool)
        target = y_true[known, index]
        scores = probabilities[known, index]
        fallback = float(fallback_thresholds[index])

        fallback_pred = (scores >= fallback).astype(np.int64)
        fallback_metrics = (
            base.binary_metrics(target, fallback_pred)
            if len(target)
            else {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "predicted_positive": 0,
            }
        )

        positive_support = int(target.sum()) if len(target) else 0

        if len(target) == 0:
            selected = fallback
            selected_metrics = fallback_metrics
            status = "fallback_no_known_validation_rows"
        elif positive_support == 0:
            selected = fallback
            selected_metrics = fallback_metrics
            status = "fallback_no_positive_validation_support"
        else:
            candidates = []
            for threshold in grid:
                prediction = (scores >= float(threshold)).astype(np.int64)
                metrics = base.binary_metrics(target, prediction)
                candidates.append(
                    {
                        "threshold": float(threshold),
                        **metrics,
                        "distance_from_fallback": abs(float(threshold) - fallback),
                    }
                )

            best = sorted(
                candidates,
                key=lambda item: (
                    -item["f1"],
                    item["distance_from_fallback"],
                    -item["threshold"],
                ),
            )[0]
            selected = float(best["threshold"])
            selected_metrics = best
            status = "tuned_on_recovered_validation"

        thresholds[index] = selected
        rows.append(
            {
                "label": label,
                "known_validation_count": int(len(target)),
                "validation_positive_support": positive_support,
                "strict_fallback_threshold": fallback,
                "strict_fallback_precision": fallback_metrics["precision"],
                "strict_fallback_recall": fallback_metrics["recall"],
                "strict_fallback_f1": fallback_metrics["f1"],
                "recovered_threshold": selected,
                "recovered_val_precision": selected_metrics["precision"],
                "recovered_val_recall": selected_metrics["recall"],
                "recovered_val_f1": selected_metrics["f1"],
                "selection_status": status,
            }
        )

    return thresholds, rows


def main() -> int:
    args = parse_args()
    base = load_base_module()

    run_dir = args.run_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    strict_json = args.strict_thresholds_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    for path, name in [
        (manifest, "manifest"),
        (labels_json, "labels JSON"),
        (strict_json, "strict threshold JSON"),
        (config_path, "run config"),
        (checkpoint, "best checkpoint"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not features_root.is_dir():
        raise FileNotFoundError(f"features root not found: {features_root}")

    labels = load_labels(labels_json)
    config = read_json(config_path)
    strict_payload = read_json(strict_json)
    strict_map = strict_payload.get("thresholds", {})
    missing = [label for label in labels if label not in strict_map]
    if missing:
        raise ValueError(f"Strict threshold JSON is missing labels: {missing}")

    strict_thresholds = np.asarray(
        [float(strict_map[label]) for label in labels],
        dtype=np.float64,
    )

    if str(args.device).lower() == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    val_dataset = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="val",
        filter_column="v09_masked_review_applied",
        filter_value=1,
        allow_empty=False,
    )
    test_dataset = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_masked_review_applied",
        filter_value=1,
        allow_empty=False,
    )

    print("\nRecovered low-energy threshold calibration")
    print("-" * 88)
    print(f"Run directory:            {run_dir}")
    print(f"Recovered validation rows:{len(val_dataset):>7,}")
    print(f"Recovered test rows:      {len(test_dataset):>7,}")
    print(f"Device:                   {device}")
    print("-" * 88)

    model = base.build_model(labels, config, checkpoint, device)

    y_val, mask_val, prob_val = base.collect_final_exit(
        model,
        base.make_loader(val_dataset, args.batch_size, args.num_workers),
        device,
    )
    y_test, mask_test, prob_test = base.collect_final_exit(
        model,
        base.make_loader(test_dataset, args.batch_size, args.num_workers),
        device,
    )

    grid = base.make_grid(args.grid_start, args.grid_end, args.grid_step)
    recovered_thresholds, rows = tune_masked_per_label(
        y_val,
        mask_val,
        prob_val,
        labels,
        grid,
        strict_thresholds,
        base,
    )

    strict_test = base.evaluate_masked(
        y_test,
        mask_test,
        prob_test,
        strict_thresholds,
        labels,
    )
    recovered_test = base.evaluate_masked(
        y_test,
        mask_test,
        prob_test,
        recovered_thresholds,
        labels,
    )

    table = pd.DataFrame(rows)
    for index, label in enumerate(labels):
        fixed = strict_test["per_label"][label]
        tuned = recovered_test["per_label"][label]
        table.loc[index, "test_strict_profile_f1"] = fixed["f1"]
        table.loc[index, "test_recovered_profile_f1"] = tuned["f1"]
        table.loc[index, "test_f1_change"] = tuned["f1"] - fixed["f1"]

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "selection_split": "recovered_human_reviewed_validation",
        "selection_rows": int(len(y_val)),
        "evaluation_split": "recovered_human_reviewed_test",
        "evaluation_rows": int(len(y_test)),
        "fallback_profile": str(strict_json),
        "grid_start": float(args.grid_start),
        "grid_end": float(args.grid_end),
        "grid_step": float(args.grid_step),
        "thresholds": {
            label: float(recovered_thresholds[index])
            for index, label in enumerate(labels)
        },
    }

    metrics = {
        "recovered_test_with_strict_profile": strict_test,
        "recovered_test_with_recovered_profile": recovered_test,
    }

    (output_dir / "recovered_per_label_thresholds.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recovered_threshold_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    table.to_csv(
        output_dir / "recovered_threshold_tuning_by_label.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        output_dir / "recovered_validation_test_probabilities.npz",
        labels=np.asarray(labels, dtype=object),
        validation_targets=y_val,
        validation_masks=mask_val,
        validation_probabilities=prob_val,
        test_targets=y_test,
        test_masks=mask_test,
        test_probabilities=prob_test,
        strict_thresholds=strict_thresholds,
        recovered_thresholds=recovered_thresholds,
    )

    print("\nSelected recovered-domain thresholds")
    print("-" * 88)
    for label, threshold in zip(labels, recovered_thresholds):
        print(f"  {label:<30} {threshold:.2f}")

    print("\nRecovered test comparison")
    print("-" * 88)
    print(
        f"{'Metric':<28}"
        f"{'Strict profile':>16}"
        f"{'Recovered profile':>20}"
        f"{'Change':>12}"
    )
    metrics_to_print = [
        ("Macro-F1", "macro_f1_known_labels"),
        ("Micro-F1", "micro_f1_known_decisions"),
        ("Samples-F1", "samples_f1_known_labels"),
        ("Known-label exact", "known_labels_exact_match"),
        ("Exact fully known", "exact_match_fully_known_rows"),
        ("Masked Hamming", "masked_hamming_loss"),
    ]
    for display, key in metrics_to_print:
        old = strict_test[key]
        new = recovered_test[key]
        print(f"{display:<28}{old:>16.4f}{new:>20.4f}{new-old:>+12.4f}")

    print("\nSaved outputs")
    print("-" * 88)
    print(output_dir / "recovered_per_label_thresholds.json")
    print(output_dir / "recovered_threshold_tuning_by_label.csv")
    print(output_dir / "recovered_threshold_metrics.json")
    print(output_dir / "recovered_validation_test_probabilities.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
