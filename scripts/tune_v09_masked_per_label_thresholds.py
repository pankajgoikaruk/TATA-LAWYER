#!/usr/bin/env python
"""
Tune one sigmoid threshold per label for the v0.9 human-reviewed masked model.

Rules:
- Select thresholds only on the strict original validation subset.
- Evaluate the strict original test subset only after threshold selection.
- Do not modify or retrain the saved checkpoint.
- Report recovered human-reviewed test rows separately with masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import (  # noqa: E402
    MaskedMultiLabelLogMelDataset,
    load_labels,
)
from utils.model_factory import build_audio_exit_net  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grid_start", type=float, default=0.05)
    parser.add_argument("--grid_end", type=float, default=0.95)
    parser.add_argument("--grid_step", type=float, default=0.01)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        drop_last=False,
    )


@torch.no_grad()
def collect_final_exit(model, loader: DataLoader, device: str):
    model.eval()
    targets, masks, probabilities = [], [], []

    for x, y, mask in loader:
        x = x.to(device)
        logits_list = model(x)
        final_prob = torch.sigmoid(logits_list[-1])
        targets.append(y.numpy())
        masks.append(mask.numpy())
        probabilities.append(final_prob.detach().cpu().numpy())

    if not targets:
        return (
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.float32),
        )

    return (
        np.concatenate(targets, axis=0).astype(np.int64),
        np.concatenate(masks, axis=0).astype(np.int64),
        np.concatenate(probabilities, axis=0).astype(np.float32),
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def apply_thresholds(probabilities, thresholds):
    return (
        probabilities >= thresholds.reshape(1, -1)
    ).astype(np.int64)


def evaluate_fully_known(
    y_true,
    probabilities,
    thresholds,
    labels,
):
    y_pred = apply_thresholds(probabilities, thresholds)

    per_label = {}
    f1s, precisions, recalls = [], [], []

    for idx, label in enumerate(labels):
        metrics = binary_metrics(y_true[:, idx], y_pred[:, idx])
        per_label[label] = metrics
        f1s.append(metrics["f1"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    return {
        "rows": int(len(y_true)),
        "macro_f1": float(np.mean(f1s)),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "micro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "exact_match": float(
            np.mean(np.all(y_true == y_pred, axis=1))
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_predicted_labels": float(
            y_pred.sum(axis=1).mean()
        ),
        "per_label": per_label,
    }


def evaluate_masked(
    y_true,
    mask,
    probabilities,
    thresholds,
    labels,
):
    if len(y_true) == 0:
        return {
            "rows": 0,
            "known_label_decisions": 0,
            "fully_known_rows": 0,
            "macro_f1_known_labels": 0.0,
            "micro_f1_known_decisions": 0.0,
            "samples_f1_known_labels": 0.0,
            "known_labels_exact_match": 0.0,
            "exact_match_fully_known_rows": 0.0,
            "masked_hamming_loss": 0.0,
            "per_label": {},
        }

    y_pred = apply_thresholds(probabilities, thresholds)
    known = mask.astype(bool)
    fully_known = known.all(axis=1)

    flat_true = y_true[known]
    flat_pred = y_pred[known]

    per_label = {}
    per_label_f1 = []
    sample_scores = []
    known_exact = []

    for idx, label in enumerate(labels):
        label_known = known[:, idx]
        metrics = (
            binary_metrics(
                y_true[label_known, idx],
                y_pred[label_known, idx],
            )
            if label_known.any()
            else {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "predicted_positive": 0,
            }
        )
        metrics["known_count"] = int(label_known.sum())
        metrics["unknown_count"] = int((~label_known).sum())
        per_label[label] = metrics
        per_label_f1.append(metrics["f1"])

    for true_row, pred_row, known_row in zip(
        y_true, y_pred, known
    ):
        if not known_row.any():
            continue
        sample_scores.append(
            float(
                f1_score(
                    true_row[known_row],
                    pred_row[known_row],
                    zero_division=0,
                )
            )
        )
        known_exact.append(
            bool(
                np.all(
                    true_row[known_row] == pred_row[known_row]
                )
            )
        )

    exact_fully_known = (
        float(
            np.mean(
                np.all(
                    y_true[fully_known] == y_pred[fully_known],
                    axis=1,
                )
            )
        )
        if fully_known.any()
        else 0.0
    )

    return {
        "rows": int(len(y_true)),
        "known_label_decisions": int(known.sum()),
        "fully_known_rows": int(fully_known.sum()),
        "macro_f1_known_labels": float(
            np.mean(per_label_f1)
        ),
        "micro_f1_known_decisions": (
            float(f1_score(flat_true, flat_pred, zero_division=0))
            if flat_true.size
            else 0.0
        ),
        "samples_f1_known_labels": (
            float(np.mean(sample_scores))
            if sample_scores
            else 0.0
        ),
        "known_labels_exact_match": (
            float(np.mean(known_exact))
            if known_exact
            else 0.0
        ),
        "exact_match_fully_known_rows": exact_fully_known,
        "masked_hamming_loss": (
            float(np.mean(flat_true != flat_pred))
            if flat_true.size
            else 0.0
        ),
        "per_label": per_label,
    }


def make_grid(start: float, end: float, step: float):
    if step <= 0:
        raise ValueError("grid_step must be > 0.")
    if not (0 < start <= end < 1):
        raise ValueError("Require 0 < start <= end < 1.")
    count = int(round((end - start) / step)) + 1
    values = start + np.arange(count) * step
    return np.round(
        values[values <= end + 1e-12],
        6,
    )


def tune_thresholds(
    y_true,
    probabilities,
    labels,
    grid,
    fixed_threshold,
):
    thresholds = np.full(
        len(labels),
        float(fixed_threshold),
        dtype=np.float64,
    )
    rows = []

    for idx, label in enumerate(labels):
        target = y_true[:, idx]
        probability = probabilities[:, idx]
        fixed_pred = (
            probability >= fixed_threshold
        ).astype(np.int64)
        fixed = binary_metrics(target, fixed_pred)

        if int(target.sum()) == 0:
            chosen_threshold = float(fixed_threshold)
            chosen = fixed
            status = "fallback_no_positive_validation_support"
        else:
            candidates = []
            for threshold in grid:
                prediction = (
                    probability >= threshold
                ).astype(np.int64)
                metrics = binary_metrics(target, prediction)
                candidates.append(
                    {
                        "threshold": float(threshold),
                        **metrics,
                        "distance_from_0_5": abs(
                            float(threshold) - 0.5
                        ),
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
            chosen_threshold = best["threshold"]
            chosen = best
            status = "tuned_on_strict_validation"

        thresholds[idx] = chosen_threshold
        rows.append(
            {
                "label": label,
                "validation_support": int(target.sum()),
                "fixed_threshold": float(fixed_threshold),
                "fixed_precision": fixed["precision"],
                "fixed_recall": fixed["recall"],
                "fixed_f1": fixed["f1"],
                "tuned_threshold": chosen_threshold,
                "tuned_precision": chosen["precision"],
                "tuned_recall": chosen["recall"],
                "tuned_f1": chosen["f1"],
                "tuning_status": status,
            }
        )

    return thresholds, rows


def build_model(labels, config, checkpoint, device):
    tap_blocks = parse_tap_blocks(
        config.get("tap_blocks", [1, 3])
    )
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
    return model


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "threshold_tuning_strict_validation"
    )

    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    for path, description in [
        (manifest, "manifest"),
        (labels_json, "labels JSON"),
        (config_path, "run config"),
        (checkpoint, "best checkpoint"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )
    if not features_root.is_dir():
        raise FileNotFoundError(
            f"features root not found: {features_root}"
        )

    config = load_json(config_path)
    labels = load_labels(labels_json)

    if str(args.device).lower() == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = str(args.device)

    datasets = {
        "val_strict": MaskedMultiLabelLogMelDataset(
            manifest,
            features_root,
            labels_json,
            split="val",
            filter_column="v09_checkpoint_eligible",
            filter_value=1,
        ),
        "test_strict": MaskedMultiLabelLogMelDataset(
            manifest,
            features_root,
            labels_json,
            split="test",
            filter_column="v09_standard_test_eligible",
            filter_value=1,
        ),
        "test_recovered_masked":
            MaskedMultiLabelLogMelDataset(
                manifest,
                features_root,
                labels_json,
                split="test",
                filter_column="v09_masked_review_applied",
                filter_value=1,
                allow_empty=True,
            ),
    }

    print("\nPer-label threshold calibration")
    print("-" * 88)
    print(f"Run directory:          {run_dir}")
    print(f"Strict validation rows: {len(datasets['val_strict']):,}")
    print(f"Strict test rows:       {len(datasets['test_strict']):,}")
    print(
        "Recovered test rows:    "
        f"{len(datasets['test_recovered_masked']):,}"
    )
    print(f"Device:                 {device}")
    print("-" * 88)

    model = build_model(
        labels,
        config,
        checkpoint,
        device,
    )

    collected = {}
    for name, dataset in datasets.items():
        collected[name] = collect_final_exit(
            model,
            make_loader(
                dataset,
                args.batch_size,
                args.num_workers,
            ),
            device,
        )

    y_val, mask_val, prob_val = collected["val_strict"]
    y_test, mask_test, prob_test = collected["test_strict"]
    y_rec, mask_rec, prob_rec = collected[
        "test_recovered_masked"
    ]

    if not np.all(mask_val == 1):
        raise RuntimeError(
            "Strict validation contains unknown labels."
        )
    if not np.all(mask_test == 1):
        raise RuntimeError(
            "Strict test contains unknown labels."
        )

    grid = make_grid(
        args.grid_start,
        args.grid_end,
        args.grid_step,
    )
    tuned_thresholds, tuning_rows = tune_thresholds(
        y_val,
        prob_val,
        labels,
        grid,
        args.fixed_threshold,
    )
    fixed_thresholds = np.full(
        len(labels),
        args.fixed_threshold,
        dtype=np.float64,
    )

    reports = {
        "validation_fixed_0_5": evaluate_fully_known(
            y_val,
            prob_val,
            fixed_thresholds,
            labels,
        ),
        "validation_tuned": evaluate_fully_known(
            y_val,
            prob_val,
            tuned_thresholds,
            labels,
        ),
        "test_strict_fixed_0_5": evaluate_fully_known(
            y_test,
            prob_test,
            fixed_thresholds,
            labels,
        ),
        "test_strict_tuned": evaluate_fully_known(
            y_test,
            prob_test,
            tuned_thresholds,
            labels,
        ),
        "test_recovered_masked_fixed_0_5":
            evaluate_masked(
                y_rec,
                mask_rec,
                prob_rec,
                fixed_thresholds,
                labels,
            ),
        "test_recovered_masked_tuned":
            evaluate_masked(
                y_rec,
                mask_rec,
                prob_rec,
                tuned_thresholds,
                labels,
            ),
    }

    tuning_df = pd.DataFrame(tuning_rows)

    for idx, label in enumerate(labels):
        fixed_test = reports[
            "test_strict_fixed_0_5"
        ]["per_label"][label]
        tuned_test = reports[
            "test_strict_tuned"
        ]["per_label"][label]

        tuning_df.loc[
            idx, "test_fixed_precision"
        ] = fixed_test["precision"]
        tuning_df.loc[
            idx, "test_fixed_recall"
        ] = fixed_test["recall"]
        tuning_df.loc[
            idx, "test_fixed_f1"
        ] = fixed_test["f1"]
        tuning_df.loc[
            idx, "test_tuned_precision"
        ] = tuned_test["precision"]
        tuning_df.loc[
            idx, "test_tuned_recall"
        ] = tuned_test["recall"]
        tuning_df.loc[
            idx, "test_tuned_f1"
        ] = tuned_test["f1"]
        tuning_df.loc[
            idx, "test_f1_change"
        ] = tuned_test["f1"] - fixed_test["f1"]

    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_payload = {
        "selection_split": "strict_original_validation",
        "selection_rows": int(len(y_val)),
        "evaluation_split": "strict_original_test",
        "evaluation_rows": int(len(y_test)),
        "grid_start": float(args.grid_start),
        "grid_end": float(args.grid_end),
        "grid_step": float(args.grid_step),
        "fixed_threshold": float(args.fixed_threshold),
        "thresholds": {
            label: float(tuned_thresholds[idx])
            for idx, label in enumerate(labels)
        },
    }

    (output_dir / "per_label_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "threshold_tuning_metrics.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    tuning_df.to_csv(
        output_dir / "threshold_tuning_by_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        output_dir
        / "final_exit_probabilities_and_targets.npz",
        labels=np.asarray(labels, dtype=object),
        validation_targets=y_val,
        validation_masks=mask_val,
        validation_probabilities=prob_val,
        strict_test_targets=y_test,
        strict_test_masks=mask_test,
        strict_test_probabilities=prob_test,
        recovered_test_targets=y_rec,
        recovered_test_masks=mask_rec,
        recovered_test_probabilities=prob_rec,
        tuned_thresholds=tuned_thresholds,
    )

    fixed = reports["test_strict_fixed_0_5"]
    tuned = reports["test_strict_tuned"]

    print("\nSelected thresholds")
    print("-" * 88)
    for label, threshold in zip(
        labels,
        tuned_thresholds,
    ):
        print(f"  {label:<30} {threshold:.2f}")

    print("\nStrict test comparison")
    print("-" * 88)
    print(
        f"{'Metric':<20} "
        f"{'Fixed 0.5':>12} "
        f"{'Tuned':>12} "
        f"{'Change':>12}"
    )
    for display, key in [
        ("Macro-F1", "macro_f1"),
        ("Micro-F1", "micro_f1"),
        ("Samples-F1", "samples_f1"),
        ("Exact Match", "exact_match"),
        ("Hamming Loss", "hamming_loss"),
    ]:
        print(
            f"{display:<20} "
            f"{fixed[key]:>12.4f} "
            f"{tuned[key]:>12.4f} "
            f"{tuned[key] - fixed[key]:>+12.4f}"
        )

    print("\nSaved outputs")
    print("-" * 88)
    print(output_dir / "per_label_thresholds.json")
    print(output_dir / "threshold_tuning_by_label.csv")
    print(output_dir / "threshold_tuning_metrics.json")
    print(
        output_dir
        / "final_exit_probabilities_and_targets.npz"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
