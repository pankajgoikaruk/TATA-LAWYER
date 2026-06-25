#!/usr/bin/env python
"""
Final v0.9 combined evaluation for 2,119 test segments.

Domains:
- 1,961 strict/original test segments
- 158 recovered low-energy test segments

Policies:
1. original_fixed_all
2. masked_fixed_all
3. original_domain_aware
4. masked_domain_aware
5. hybrid_recommended

Recommended hybrid:
- normal/original domain -> original v0.9 model at threshold 0.50
- recovered low-energy domain -> masked model with recovered-domain thresholds
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_run_dir", type=Path, required=True)
    parser.add_argument("--masked_run_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--masked_strict_thresholds_json", type=Path, required=True)
    parser.add_argument("--recovered_comparison_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def build_model(run_dir: Path, labels: list[str], device: str):
    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    config = load_json(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

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
        n_mels=int(config.get("n_mels", 64)),
        tap_blocks=parse_tap_blocks(config.get("tap_blocks", [1, 3])),
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
    return model


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
    targets, masks, probabilities = [], [], []

    for x, y, mask in loader:
        logits = model(x.to(device))[-1]
        targets.append(y.numpy())
        masks.append(mask.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    if not targets:
        raise RuntimeError("No rows were loaded.")

    return (
        np.concatenate(targets).astype(np.int64),
        np.concatenate(masks).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
    )


def threshold_vector(payload: dict, labels: list[str], path_name: str):
    threshold_map = payload.get("thresholds")
    if not isinstance(threshold_map, dict):
        raise ValueError(f"No thresholds mapping found in {path_name}")

    missing = [label for label in labels if label not in threshold_map]
    if missing:
        raise ValueError(f"Missing thresholds in {path_name}: {missing}")

    return np.asarray(
        [float(threshold_map[label]) for label in labels],
        dtype=np.float64,
    )


def binary_metrics(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def evaluate_predictions(y_true, mask, predictions, labels):
    known = mask.astype(bool)
    fully_known = known.all(axis=1)

    flat_true = y_true[known]
    flat_pred = predictions[known]

    per_label = {}
    label_f1 = []
    sample_f1 = []
    known_exact = []

    for index, label in enumerate(labels):
        label_known = known[:, index]
        metrics = binary_metrics(
            y_true[label_known, index],
            predictions[label_known, index],
        )
        metrics["known_count"] = int(label_known.sum())
        metrics["unknown_count"] = int((~label_known).sum())
        per_label[label] = metrics
        label_f1.append(metrics["f1"])

    for true_row, pred_row, known_row in zip(y_true, predictions, known):
        if not known_row.any():
            continue
        sample_f1.append(
            float(
                f1_score(
                    true_row[known_row],
                    pred_row[known_row],
                    zero_division=0,
                )
            )
        )
        known_exact.append(
            bool(np.all(true_row[known_row] == pred_row[known_row]))
        )

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
        "exact_match_fully_known_rows": (
            float(
                np.mean(
                    np.all(
                        y_true[fully_known] == predictions[fully_known],
                        axis=1,
                    )
                )
            )
            if fully_known.any()
            else 0.0
        ),
        "masked_hamming_loss": float(np.mean(flat_true != flat_pred)),
        "per_label": per_label,
    }


def apply_thresholds(probabilities, thresholds):
    return (
        probabilities >= thresholds.reshape(1, -1)
    ).astype(np.int64)


def main() -> int:
    args = parse_args()

    original_run = args.original_run_dir.expanduser().resolve()
    masked_run = args.masked_run_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    masked_strict_json = (
        args.masked_strict_thresholds_json.expanduser().resolve()
    )
    recovered_comparison_json = (
        args.recovered_comparison_json.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    labels = load_labels(labels_json)
    fixed = np.full(
        len(labels),
        float(args.fixed_threshold),
        dtype=np.float64,
    )

    masked_strict = threshold_vector(
        load_json(masked_strict_json),
        labels,
        str(masked_strict_json),
    )

    recovered_payload = load_json(recovered_comparison_json)
    original_recovered = threshold_vector(
        recovered_payload["original"],
        labels,
        str(recovered_comparison_json) + " [original]",
    )
    masked_recovered = threshold_vector(
        recovered_payload["masked"],
        labels,
        str(recovered_comparison_json) + " [masked]",
    )

    if str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    strict_dataset = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_standard_test_eligible",
        filter_value=1,
    )
    recovered_dataset = MaskedMultiLabelLogMelDataset(
        manifest,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_masked_review_applied",
        filter_value=1,
    )

    print("\nFinal v0.9 domain-routed evaluation")
    print("-" * 94)
    print(f"Strict/original test rows: {len(strict_dataset):,}")
    print(f"Recovered test rows:       {len(recovered_dataset):,}")
    print(f"Combined test rows:        {len(strict_dataset) + len(recovered_dataset):,}")
    print(f"Device:                    {device}")
    print("-" * 94)

    strict_loader = make_loader(
        strict_dataset,
        args.batch_size,
        args.num_workers,
    )
    recovered_loader = make_loader(
        recovered_dataset,
        args.batch_size,
        args.num_workers,
    )

    original_model = build_model(original_run, labels, device)
    original_strict = collect_final_exit(
        original_model, strict_loader, device
    )
    original_recovered_data = collect_final_exit(
        original_model, recovered_loader, device
    )
    del original_model

    masked_model = build_model(masked_run, labels, device)
    masked_strict_data = collect_final_exit(
        masked_model, strict_loader, device
    )
    masked_recovered_data = collect_final_exit(
        masked_model, recovered_loader, device
    )
    del masked_model

    y_strict, m_strict, p_original_strict = original_strict
    y_recovered, m_recovered, p_original_recovered = (
        original_recovered_data
    )

    y_strict_masked, m_strict_masked, p_masked_strict = (
        masked_strict_data
    )
    y_recovered_masked, m_recovered_masked, p_masked_recovered = (
        masked_recovered_data
    )

    if not np.array_equal(y_strict, y_strict_masked):
        raise RuntimeError("Strict targets differ between model passes.")
    if not np.array_equal(m_strict, m_strict_masked):
        raise RuntimeError("Strict masks differ between model passes.")
    if not np.array_equal(y_recovered, y_recovered_masked):
        raise RuntimeError("Recovered targets differ between model passes.")
    if not np.array_equal(m_recovered, m_recovered_masked):
        raise RuntimeError("Recovered masks differ between model passes.")

    combined_targets = np.concatenate(
        [y_strict, y_recovered],
        axis=0,
    )
    combined_masks = np.concatenate(
        [m_strict, m_recovered],
        axis=0,
    )

    policies = {
        "original_fixed_all": np.concatenate(
            [
                apply_thresholds(p_original_strict, fixed),
                apply_thresholds(p_original_recovered, fixed),
            ],
            axis=0,
        ),
        "masked_fixed_all": np.concatenate(
            [
                apply_thresholds(p_masked_strict, fixed),
                apply_thresholds(p_masked_recovered, fixed),
            ],
            axis=0,
        ),
        "original_domain_aware": np.concatenate(
            [
                apply_thresholds(p_original_strict, fixed),
                apply_thresholds(
                    p_original_recovered,
                    original_recovered,
                ),
            ],
            axis=0,
        ),
        "masked_domain_aware": np.concatenate(
            [
                apply_thresholds(
                    p_masked_strict,
                    masked_strict,
                ),
                apply_thresholds(
                    p_masked_recovered,
                    masked_recovered,
                ),
            ],
            axis=0,
        ),
        "hybrid_recommended": np.concatenate(
            [
                apply_thresholds(
                    p_original_strict,
                    fixed,
                ),
                apply_thresholds(
                    p_masked_recovered,
                    masked_recovered,
                ),
            ],
            axis=0,
        ),
    }

    reports = {
        name: evaluate_predictions(
            combined_targets,
            combined_masks,
            predictions,
            labels,
        )
        for name, predictions in policies.items()
    }

    metric_keys = [
        ("Macro-F1", "macro_f1_known_labels"),
        ("Micro-F1", "micro_f1_known_decisions"),
        ("Samples-F1", "samples_f1_known_labels"),
        ("Known-label Exact", "known_labels_exact_match"),
        ("Exact Fully Known", "exact_match_fully_known_rows"),
        ("Masked Hamming", "masked_hamming_loss"),
    ]

    summary_rows = []
    for policy_name, report in reports.items():
        row = {"policy": policy_name}
        for display, key in metric_keys:
            row[display] = report[key]
        summary_rows.append(row)

    hybrid_per_label_rows = []
    hybrid_report = reports["hybrid_recommended"]
    original_report = reports["original_fixed_all"]

    for label in labels:
        baseline = original_report["per_label"][label]
        hybrid = hybrid_report["per_label"][label]
        hybrid_per_label_rows.append(
            {
                "label": label,
                "original_fixed_f1": baseline["f1"],
                "hybrid_f1": hybrid["f1"],
                "hybrid_minus_original_f1": (
                    hybrid["f1"] - baseline["f1"]
                ),
                "combined_support": hybrid["support"],
                "known_count": hybrid["known_count"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "strict_test_rows": int(len(strict_dataset)),
        "recovered_test_rows": int(len(recovered_dataset)),
        "combined_test_rows": int(
            len(strict_dataset) + len(recovered_dataset)
        ),
        "normal_domain_policy": {
            "model": "original_v09",
            "thresholds": "fixed_0.50",
        },
        "low_energy_domain_policy": {
            "model": "masked_human_reviewed_v09",
            "thresholds": {
                label: float(masked_recovered[index])
                for index, label in enumerate(labels)
            },
        },
        "reports": reports,
    }

    (output_dir / "final_domain_routed_hybrid_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "final_domain_routed_policy_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(hybrid_per_label_rows).to_csv(
        output_dir / "final_hybrid_per_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCombined 2,119-row comparison")
    print("-" * 120)
    print(
        f"{'Policy':<26}"
        f"{'Macro-F1':>12}"
        f"{'Micro-F1':>12}"
        f"{'Samples-F1':>14}"
        f"{'Known Exact':>14}"
        f"{'Full Exact':>12}"
        f"{'Hamming':>12}"
    )
    for row in summary_rows:
        print(
            f"{row['policy']:<26}"
            f"{row['Macro-F1']:>12.4f}"
            f"{row['Micro-F1']:>12.4f}"
            f"{row['Samples-F1']:>14.4f}"
            f"{row['Known-label Exact']:>14.4f}"
            f"{row['Exact Fully Known']:>12.4f}"
            f"{row['Masked Hamming']:>12.4f}"
        )

    print("\nSaved outputs")
    print("-" * 94)
    print(output_dir / "final_domain_routed_hybrid_metrics.json")
    print(output_dir / "final_domain_routed_policy_comparison.csv")
    print(output_dir / "final_hybrid_per_label.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
