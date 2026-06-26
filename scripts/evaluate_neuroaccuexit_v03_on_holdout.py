#!/usr/bin/env python
"""
Evaluate NeuroAccuExit v0.3 selected model on an untouched human ground-truth holdout.

This script is intentionally local-only. It does not modify the holdout CSV,
training manifest, checkpoints, or thresholds.

Expected input:
  1. A holdout feature manifest produced from:
       scripts/build_tata_holdout_segments.py
       scripts/extract_multilabel_features.py
  2. A trained run_dir containing:
       config_used.json
       ckpt/best.pt
  3. A tuned threshold JSON containing:
       {"thresholds": {label: threshold, ...}}

Outputs:
  - holdout_v03_eval_summary.json
  - holdout_v03_per_label_segment.csv
  - holdout_v03_per_label_parent_mean.csv
  - holdout_v03_per_label_parent_max.csv
  - holdout_v03_segment_predictions.csv
  - holdout_v03_parent_predictions_mean.csv
  - holdout_v03_parent_predictions_max.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import load_labels  # noqa: E402
from utils.model_factory import build_audio_exit_net  # noqa: E402


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def threshold_vector(payload: dict, labels: list[str], fixed_threshold: float) -> tuple[np.ndarray, dict]:
    threshold_map = payload.get("thresholds") if isinstance(payload, dict) else None
    if threshold_map is None:
        threshold_map = {}

    out = []
    clean = {}
    for label in labels:
        value = float(threshold_map.get(label, fixed_threshold))
        out.append(value)
        clean[label] = value
    return np.asarray(out, dtype=np.float64), clean


def build_model(run_dir: Path, labels: list[str], device: str):
    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")

    config = load_json(config_path)
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
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model, config


class FeatureManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features_root: Path):
        self.df = df.reset_index(drop=True).copy()
        self.features_root = Path(features_root)

        if "feat_relpath" not in self.df.columns:
            raise ValueError("Feature manifest must contain feat_relpath.")
        if not self.features_root.is_dir():
            raise FileNotFoundError(f"Features root not found: {self.features_root}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[int(index)]
        rel = str(row["feat_relpath"]).replace("\\", "/")
        path = self.features_root / Path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"Feature file not found: {path}")
        feature = np.load(path).astype(np.float32)
        if feature.ndim != 2:
            raise RuntimeError(f"Expected feature shape [n_mels, T], got {feature.shape}: {path}")
        x = torch.from_numpy(feature).float().unsqueeze(0)
        return x, int(index)


@torch.no_grad()
def collect_probs(model, dataset: Dataset, device: str, batch_size: int, num_workers: int):
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        drop_last=False,
    )

    probs_by_exit = None
    positions = []

    for x, idx in loader:
        logits_list = model(x.to(device))
        probs_list = [torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32) for logits in logits_list]

        if probs_by_exit is None:
            probs_by_exit = [[] for _ in probs_list]

        positions.append(idx.numpy().astype(np.int64))
        for exit_idx, probs in enumerate(probs_list):
            probs_by_exit[exit_idx].append(probs)

    if probs_by_exit is None:
        return np.zeros((0,), dtype=np.int64), []

    return (
        np.concatenate(positions, axis=0),
        [np.concatenate(parts, axis=0) for parts in probs_by_exit],
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray, labels: list[str]) -> dict:
    y_pred = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)

    per_label = {}
    f1s = []
    precisions = []
    recalls = []

    for idx, label in enumerate(labels):
        m = binary_metrics(y_true[:, idx], y_pred[:, idx])
        per_label[label] = m
        f1s.append(m["f1"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    return {
        "rows": int(len(y_true)),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)) if len(y_true) else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))) if len(y_true) else 0.0,
        "hamming_loss": float(hamming_loss(y_true, y_pred)) if len(y_true) else 0.0,
        "avg_true_labels": float(y_true.sum(axis=1).mean()) if len(y_true) else 0.0,
        "avg_predicted_labels": float(y_pred.sum(axis=1).mean()) if len(y_true) else 0.0,
        "per_label": per_label,
    }


def make_per_label_df(metrics: dict) -> pd.DataFrame:
    rows = []
    for label, values in metrics["per_label"].items():
        row = {"label": label}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_parent(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    parent_col: str,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if parent_col not in df.columns:
        raise ValueError(f"Parent column not found in manifest: {parent_col}")

    work = df.reset_index(drop=True).copy()
    for idx, label in enumerate(labels):
        work[f"prob_{label}"] = probabilities[:, idx]

    parent_rows = []
    parent_probs = []
    parent_targets = []

    for parent_id, group in work.groupby(parent_col, sort=False):
        out = {parent_col: parent_id, "segments": int(len(group))}

        # Segment labels should be duplicated from the parent. Use max for robustness.
        y = group[labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).max(axis=0).values.astype(np.int64)

        p_values = []
        for label in labels:
            vals = group[f"prob_{label}"].astype(float).values
            if method == "mean":
                p = float(np.mean(vals))
            elif method == "max":
                p = float(np.max(vals))
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            out[f"prob_{label}"] = p
            out[label] = int(group[label].astype(int).max())
            p_values.append(p)

        parent_rows.append(out)
        parent_probs.append(p_values)
        parent_targets.append(y)

    return (
        pd.DataFrame(parent_rows),
        np.asarray(parent_targets, dtype=np.int64),
        np.asarray(parent_probs, dtype=np.float32),
    )


def write_predictions(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    out = df.reset_index(drop=True).copy()
    preds = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)
    for idx, label in enumerate(labels):
        out[f"prob_{label}"] = probabilities[:, idx]
        out[f"pred_{label}"] = preds[:, idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--thresholds_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    parser.add_argument("--parent_col", default="parent_clip_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    feature_manifest = args.feature_manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    thresholds_json = args.thresholds_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    labels = load_labels(labels_json)
    thresholds, threshold_map = threshold_vector(
        load_json(thresholds_json),
        labels,
        fixed_threshold=float(args.fixed_threshold),
    )

    df = pd.read_csv(feature_manifest, low_memory=False)
    missing = [label for label in labels if label not in df.columns]
    if missing:
        raise ValueError(f"Feature manifest missing label columns: {missing}")

    for label in labels:
        df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)

    model, config = build_model(run_dir, labels, device)
    dataset = FeatureManifestDataset(df, features_root)

    print("\nNeuroAccuExit v0.3 holdout evaluation")
    print("-" * 90)
    print(f"Run directory:     {run_dir}")
    print(f"Feature manifest:  {feature_manifest}")
    print(f"Features root:     {features_root}")
    print(f"Labels JSON:       {labels_json}")
    print(f"Thresholds JSON:   {thresholds_json}")
    print(f"Output directory:  {output_dir}")
    print(f"Rows:              {len(df):,}")
    print(f"Device:            {device}")
    print("-" * 90)

    positions, probs_by_exit = collect_probs(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(positions) != len(df):
        raise RuntimeError(f"Collected {len(positions)} predictions for {len(df)} rows.")

    y_segment = df[labels].values.astype(np.int64)
    final_probs = probs_by_exit[-1]

    exit_metrics = []
    for exit_idx, probs in enumerate(probs_by_exit, start=1):
        item = evaluate(y_segment, probs, thresholds, labels)
        item["exit"] = int(exit_idx)
        exit_metrics.append(item)

    segment_metrics = exit_metrics[-1]

    output_dir.mkdir(parents=True, exist_ok=True)

    parent_reports = {}
    if args.parent_col in df.columns:
        for method in ("mean", "max"):
            parent_df, y_parent, p_parent = aggregate_parent(
                df=df,
                probabilities=final_probs,
                labels=labels,
                parent_col=args.parent_col,
                method=method,
            )
            metrics = evaluate(y_parent, p_parent, thresholds, labels)
            parent_reports[method] = metrics

            make_per_label_df(metrics).to_csv(
                output_dir / f"holdout_v03_per_label_parent_{method}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            write_predictions(
                parent_df,
                p_parent,
                thresholds,
                labels,
                output_dir / f"holdout_v03_parent_predictions_{method}.csv",
            )

    make_per_label_df(segment_metrics).to_csv(
        output_dir / "holdout_v03_per_label_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_predictions(
        df,
        final_probs,
        thresholds,
        labels,
        output_dir / "holdout_v03_segment_predictions.csv",
    )

    summary = {
        "task": "neuroaccuexit_v03_selected_model_on_untouched_human_ground_truth_holdout",
        "run_dir": str(run_dir),
        "feature_manifest": str(feature_manifest),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "thresholds_json": str(thresholds_json),
        "labels": labels,
        "thresholds": threshold_map,
        "rows_segment": int(len(df)),
        "device": device,
        "config": config,
        "segment_final_exit": segment_metrics,
        "exit_metrics_segment": exit_metrics,
        "parent_reports": parent_reports,
    }

    save_json(summary, output_dir / "holdout_v03_eval_summary.json")

    print("\nFinal-exit holdout results")
    print("-" * 90)
    print(
        f"Segment-level: "
        f"macroF1={segment_metrics['macro_f1']:.4f}, "
        f"microF1={segment_metrics['micro_f1']:.4f}, "
        f"samplesF1={segment_metrics['samples_f1']:.4f}, "
        f"exact={segment_metrics['exact_match']:.4f}, "
        f"hamming={segment_metrics['hamming_loss']:.4f}"
    )

    for method, metrics in parent_reports.items():
        print(
            f"Parent-{method:<4}: "
            f"macroF1={metrics['macro_f1']:.4f}, "
            f"microF1={metrics['micro_f1']:.4f}, "
            f"samplesF1={metrics['samples_f1']:.4f}, "
            f"exact={metrics['exact_match']:.4f}, "
            f"hamming={metrics['hamming_loss']:.4f}, "
            f"parents={metrics['rows']}"
        )

    print("\nSaved outputs")
    print("-" * 90)
    print(output_dir / "holdout_v03_eval_summary.json")
    print(output_dir / "holdout_v03_per_label_segment.csv")
    if parent_reports:
        print(output_dir / "holdout_v03_per_label_parent_mean.csv")
        print(output_dir / "holdout_v03_per_label_parent_max.csv")
    print(output_dir / "holdout_v03_segment_predictions.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
