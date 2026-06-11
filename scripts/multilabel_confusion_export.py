# scripts/multilabel_confusion_export.py
#
# Confusion-matrix exporter for multi-label / single-label speaker experiments.
#
# Why this exists:
#   The ASHADIP human-talk stages are stored as multi-label, but each clean
#   speaker example normally has exactly one active speaker label. This script
#   therefore saves both:
#     1) standard K x K confusion matrices using argmax-backed single-label view
#     2) per-label multi-label confusion counts: TP, FP, TN, FN
#
# Modes:
#   --eval_mode segment
#       Saves segment-level selected-policy and final-exit confusion matrices.
#
#   --eval_mode clip
#       Groups windows by parent_clip_id and saves clip-level full-clip baseline
#       and dynamic clip/window policy confusion matrices.
#
# Outputs:
#   Segment mode -> <run_dir>/multilabel_greedy_policy/confusion/
#   Clip mode    -> <run_dir>/multilabel_clip_window_policy/confusion/

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import load_labels, make_multilabel_loaders
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
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"


def fmt_float(x: Any, digits: int = 6) -> Any:
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return np.nan
        return round(x, digits)
    except Exception:
        return x


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_float_dtype(df2[col]):
            df2[col] = df2[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    headers = [str(c) for c in df2.columns]
    rows = df2.astype(str).values.tolist()
    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(str(row[i])) for row in rows], default=0)
        widths.append(max(len(h), max_cell))

    def make_row(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    out = [make_row(headers)]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        out.append(make_row(row))
    return "\n".join(out) + "\n"


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.write_text(df_to_markdown(df), encoding="utf-8")


def threshold_vector_from_mapping(mapping: dict[str, Any], labels: list[str]) -> np.ndarray:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(f"Threshold mapping missing labels: {missing}")
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def load_thresholds_by_exit(
    *,
    run_dir: Path,
    labels: list[str],
    num_exits: int,
    threshold_mode: str,
    fixed_threshold: float,
) -> list[np.ndarray]:
    if threshold_mode == "fixed_0p5":
        th = np.full(len(labels), float(fixed_threshold), dtype=np.float32)
        return [th.copy() for _ in range(num_exits)]

    comparison_path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    if comparison_path.exists():
        payload = load_json(comparison_path)
        if list(payload.get("labels", labels)) != list(labels):
            raise RuntimeError("Label order mismatch in threshold_comparison.json.")
        exits = payload.get("exits", [])
        if len(exits) < num_exits:
            raise RuntimeError(f"Threshold file has {len(exits)} exits, model has {num_exits}.")
        if threshold_mode == "tuned_per_exit":
            return [
                threshold_vector_from_mapping(exits[i].get("tuned_thresholds", {}), labels)
                for i in range(num_exits)
            ]
        if threshold_mode == "final_exit_tuned":
            th = threshold_vector_from_mapping(exits[num_exits - 1].get("tuned_thresholds", {}), labels)
            return [th.copy() for _ in range(num_exits)]

    final_path = run_dir / "threshold_tuning" / "multilabel_thresholds.json"
    if threshold_mode == "final_exit_tuned" and final_path.exists():
        payload = load_json(final_path)
        th = threshold_vector_from_mapping(payload.get("thresholds", {}), labels)
        return [th.copy() for _ in range(num_exits)]

    raise FileNotFoundError(
        f"Could not load thresholds for mode={threshold_mode}. Expected {comparison_path} or {final_path}."
    )


def probs_to_label_matrix(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    th = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (np.asarray(y_prob) >= th).astype(int)


def load_model_state(model, ckpt_path: Path, device: str):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def collect_segment_probs_from_loader(model, dl, device: str):
    model.eval()
    y_parts = []
    probs_by_exit = None
    for x, y in dl:
        x = x.to(device)
        logits_list = model(x)
        probs_list = [torch.sigmoid(logits).detach().cpu().numpy() for logits in logits_list]
        if probs_by_exit is None:
            probs_by_exit = [[] for _ in probs_list]
        y_parts.append(y.numpy())
        for k, probs in enumerate(probs_list):
            probs_by_exit[k].append(probs)
    y_true = np.concatenate(y_parts, axis=0).astype(int)
    probs_by_exit = [np.concatenate(parts, axis=0) for parts in probs_by_exit]
    return y_true, probs_by_exit


class FeatureWithMetaDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, features_root: str | Path, labels_json: str | Path, split: str):
        self.manifest_csv = Path(manifest_csv)
        self.features_root = Path(features_root)
        df = pd.read_csv(self.manifest_csv)
        self.labels = load_labels(labels_json, manifest_df=df)
        required = ["split", "feat_relpath", "parent_clip_id", "segment_index"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Manifest missing columns required for clip confusion: {missing}")
        self.df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No rows found for split={split} in {manifest_csv}")
        missing_labels = [lab for lab in self.labels if lab not in self.df.columns]
        if missing_labels:
            raise RuntimeError(f"Manifest missing label columns: {missing_labels}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        feat_path = self.features_root / Path(feat_rel)
        S = np.load(feat_path).astype(np.float32)
        x = torch.from_numpy(S).float().unsqueeze(0)
        y = torch.from_numpy(row[self.labels].astype(np.float32).values).float()
        return x, y, int(idx)


@torch.no_grad()
def collect_clip_probs(model, dataset: FeatureWithMetaDataset, batch_size: int, num_workers: int, device: str):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_by_exit = None
    y_parts = []
    idx_parts = []
    model.eval()
    for x, y, idx in dl:
        x = x.to(device)
        logits_list = model(x)
        probs_list = [torch.sigmoid(logits).detach().cpu().numpy() for logits in logits_list]
        if probs_by_exit is None:
            probs_by_exit = [[] for _ in probs_list]
        for k, probs in enumerate(probs_list):
            probs_by_exit[k].append(probs)
        y_parts.append(y.numpy())
        idx_parts.append(idx.numpy())
    y_true = np.concatenate(y_parts, axis=0).astype(int)
    idx_all = np.concatenate(idx_parts, axis=0).astype(int)
    probs_by_exit = [np.concatenate(parts, axis=0) for parts in probs_by_exit]
    return idx_all, y_true, probs_by_exit


def label_set_stability_policy(
    *,
    preds_by_exit: list[np.ndarray],
    min_exit: int,
    stable_k: int,
    allow_empty_stop: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    num_exits = len(preds_by_exit)
    n = int(preds_by_exit[0].shape[0])
    selected = np.zeros_like(preds_by_exit[-1], dtype=int)
    selected_exit_idx = np.full(n, num_exits - 1, dtype=int)
    exit_counts = {f"e{k + 1}": 0 for k in range(num_exits)}
    start_idx = int(min_exit) - 1

    for sample_idx in range(n):
        prev_vec = None
        stable_count = 0
        chosen_idx = num_exits - 1
        for exit_idx in range(start_idx, num_exits):
            current = preds_by_exit[exit_idx][sample_idx].astype(int)
            if prev_vec is not None and np.array_equal(current, prev_vec):
                stable_count += 1
            else:
                stable_count = 1
            prev_vec = current
            is_empty = int(current.sum()) == 0
            is_final = exit_idx == num_exits - 1
            empty_ok = bool(allow_empty_stop) or is_final or not is_empty
            if stable_count >= stable_k and empty_ok:
                chosen_idx = exit_idx
                break
        selected[sample_idx] = preds_by_exit[chosen_idx][sample_idx]
        selected_exit_idx[sample_idx] = chosen_idx
        exit_counts[f"e{chosen_idx + 1}"] += 1
    return selected, selected_exit_idx, exit_counts


def select_exit_for_window(
    *,
    pred_vectors: list[np.ndarray],
    min_exit: int,
    stable_k: int,
    allow_empty_stop: bool,
) -> tuple[np.ndarray, int]:
    num_exits = len(pred_vectors)
    start_idx = int(min_exit) - 1
    prev_vec = None
    stable_count = 0
    chosen_idx = num_exits - 1
    for exit_idx in range(start_idx, num_exits):
        current = pred_vectors[exit_idx].astype(int)
        if prev_vec is not None and np.array_equal(current, prev_vec):
            stable_count += 1
        else:
            stable_count = 1
        prev_vec = current
        is_empty = int(current.sum()) == 0
        is_final = exit_idx == num_exits - 1
        empty_ok = bool(allow_empty_stop) or is_final or not is_empty
        if stable_count >= stable_k and empty_ok:
            chosen_idx = exit_idx
            break
    return pred_vectors[chosen_idx].astype(int), int(chosen_idx)


def gather_selected_probs(probs_by_exit: list[np.ndarray], selected_exit_idx: np.ndarray) -> np.ndarray:
    n = int(probs_by_exit[0].shape[0])
    c = int(probs_by_exit[0].shape[1])
    out = np.zeros((n, c), dtype=np.float32)
    for i, exit_idx in enumerate(selected_exit_idx.astype(int)):
        out[i] = probs_by_exit[int(exit_idx)][i]
    return out


def labels_to_single_class(
    *,
    y_multi: np.ndarray,
    y_prob: np.ndarray | None,
    labels: list[str],
    role: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Convert multi-hot labels into a single class id for K x K confusion.

    If a row has exactly one active label, use it.
    If it has zero or multiple labels, fall back to argmax probability when
    available; otherwise use argmax over the multi-hot vector. A diagnostic
    table records how often fallback was needed.
    """
    y_multi = np.asarray(y_multi).astype(int)
    n, c = y_multi.shape
    out = np.zeros(n, dtype=int)
    statuses = []

    for i in range(n):
        active = np.flatnonzero(y_multi[i] == 1)
        if len(active) == 1:
            out[i] = int(active[0])
            statuses.append("single_positive")
        elif len(active) == 0:
            if y_prob is not None:
                out[i] = int(np.argmax(y_prob[i]))
                statuses.append("empty_argmax_probability")
            else:
                out[i] = 0
                statuses.append("empty_default_zero")
        else:
            if y_prob is not None:
                out[i] = int(active[np.argmax(y_prob[i, active])])
                statuses.append("multi_positive_argmax_among_active")
            else:
                out[i] = int(active[0])
                statuses.append("multi_positive_first_active")

    diag = (
        pd.Series(statuses, name="status")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "status", "count": "count"})
    )
    diag.insert(0, "role", role)
    diag["fraction"] = diag["count"] / max(n, 1)
    return out, diag


def per_label_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        support = int((yt == 1).sum())
        predicted_positive = int((yp == 1).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        rows.append({
            "label": label,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "support": support,
            "predicted_positive": predicted_positive,
            "precision": fmt_float(precision),
            "recall": fmt_float(recall),
            "specificity": fmt_float(specificity),
        })
    return pd.DataFrame(rows)


def save_confusion_outputs(
    *,
    out_dir: Path,
    prefix: str,
    labels: list[str],
    y_true_multi: np.ndarray,
    y_pred_multi: np.ndarray,
    y_true_prob: np.ndarray | None,
    y_pred_prob: np.ndarray | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(labels)))

    y_true_single, true_diag = labels_to_single_class(
        y_multi=y_true_multi,
        y_prob=y_true_prob,
        labels=labels,
        role="true",
    )
    y_pred_single, pred_diag = labels_to_single_class(
        y_multi=y_pred_multi,
        y_prob=y_pred_prob,
        labels=labels,
        role="pred",
    )

    cm = confusion_matrix(y_true_single, y_pred_single, labels=label_ids)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_label"
    cm_df.to_csv(out_dir / f"{prefix}_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), dtype=np.float64)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.index.name = "true_label"
    cm_norm_df.to_csv(out_dir / f"{prefix}_confusion_matrix_normalized.csv")

    per_label_df = per_label_confusion(y_true_multi, y_pred_multi, labels)
    write_table(
        per_label_df,
        out_dir / f"{prefix}_per_label_confusion.csv",
        out_dir / f"{prefix}_per_label_confusion.md",
    )

    diag_df = pd.concat([true_diag, pred_diag], ignore_index=True)
    write_table(
        diag_df,
        out_dir / f"{prefix}_single_label_conversion_diagnostics.csv",
        out_dir / f"{prefix}_single_label_conversion_diagnostics.md",
    )

    plot_confusion_matrix(
        matrix=cm,
        labels=labels,
        title=f"{prefix.replace('_', ' ').title()} Confusion Matrix",
        out_path=out_dir / f"{prefix}_confusion_matrix.png",
        normalized=False,
    )
    plot_confusion_matrix(
        matrix=cm_norm,
        labels=labels,
        title=f"{prefix.replace('_', ' ').title()} Normalized Confusion Matrix",
        out_path=out_dir / f"{prefix}_confusion_matrix_normalized.png",
        normalized=True,
    )

    return {
        "confusion_matrix_csv": str(out_dir / f"{prefix}_confusion_matrix.csv"),
        "confusion_matrix_normalized_csv": str(out_dir / f"{prefix}_confusion_matrix_normalized.csv"),
        "confusion_matrix_png": str(out_dir / f"{prefix}_confusion_matrix.png"),
        "confusion_matrix_normalized_png": str(out_dir / f"{prefix}_confusion_matrix_normalized.png"),
        "per_label_confusion_csv": str(out_dir / f"{prefix}_per_label_confusion.csv"),
        "single_label_conversion_diagnostics_csv": str(out_dir / f"{prefix}_single_label_conversion_diagnostics.csv"),
    }


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, out_path: Path, normalized: bool):
    matrix = np.asarray(matrix)
    fig_width = max(7, 0.75 * len(labels) + 3)
    fig_height = max(6, 0.65 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_model_from_config(cfg: dict[str, Any], device: str):
    labels = [str(x) for x in cfg["labels"]]
    num_labels = int(cfg["num_labels"])
    n_mels = int(cfg.get("n_mels", 64))
    tap_blocks = parse_tap_blocks(cfg["tap_blocks"])
    exit_hint_cfg = cfg.get("exit_hint", None)
    model_cfg = {
        "exit_hint": exit_hint_cfg if isinstance(exit_hint_cfg, dict) else {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        }
    }
    model = build_audio_exit_net(
        num_classes=num_labels,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)
    return model, labels, tap_blocks


def run_segment_mode(args, cfg, model, labels, run_dir: Path, device: str):
    manifest = Path(cfg["manifest"]).resolve()
    features_root = Path(cfg["features_root"]).resolve()
    labels_json = Path(cfg["labels_json"]).resolve()
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    seed = int(cfg.get("seed", 42))

    _, dl_va, dl_te, loaded_labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        seed=seed,
        label_balance_power=0.0,
        synthetic_balance_power=0.0,
    )
    if list(loaded_labels) != list(labels):
        raise RuntimeError("Label order mismatch between config and dataset.")

    eval_loader = dl_te if args.split == "test" else dl_va
    y_true, probs_by_exit = collect_segment_probs_from_loader(model, eval_loader, device)
    num_exits = len(probs_by_exit)

    thresholds_by_exit = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=num_exits,
        threshold_mode=args.threshold_mode,
        fixed_threshold=float(args.fixed_threshold),
    )
    preds_by_exit = [
        probs_to_label_matrix(probs, thresholds_by_exit[k])
        for k, probs in enumerate(probs_by_exit)
    ]

    selected_pred, selected_exit_idx, _ = label_set_stability_policy(
        preds_by_exit=preds_by_exit,
        min_exit=int(args.min_exit),
        stable_k=int(args.stable_k),
        allow_empty_stop=bool(args.allow_empty_stop),
    )
    selected_prob = gather_selected_probs(probs_by_exit, selected_exit_idx)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "multilabel_greedy_policy" / "confusion"
    outputs = {}
    outputs["segment_selected_policy"] = save_confusion_outputs(
        out_dir=out_dir,
        prefix="segment_selected_policy",
        labels=labels,
        y_true_multi=y_true,
        y_pred_multi=selected_pred,
        y_true_prob=None,
        y_pred_prob=selected_prob,
    )
    outputs["segment_final_exit"] = save_confusion_outputs(
        out_dir=out_dir,
        prefix="segment_final_exit",
        labels=labels,
        y_true_multi=y_true,
        y_pred_multi=preds_by_exit[-1],
        y_true_prob=None,
        y_pred_prob=probs_by_exit[-1],
    )
    return out_dir, outputs


def run_clip_mode(args, cfg, model, labels, run_dir: Path, device: str):
    manifest = Path(cfg["manifest"]).resolve()
    features_root = Path(cfg["features_root"]).resolve()
    labels_json = Path(cfg["labels_json"]).resolve()
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))

    dataset = FeatureWithMetaDataset(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        split=args.split,
    )
    if list(dataset.labels) != list(labels):
        raise RuntimeError("Label order mismatch between config and clip dataset.")

    idx_all, y_segment_true, probs_by_exit = collect_clip_probs(
        model, dataset, batch_size=batch_size, num_workers=int(args.num_workers), device=device
    )
    num_exits = len(probs_by_exit)
    thresholds_by_exit = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=num_exits,
        threshold_mode=args.threshold_mode,
        fixed_threshold=float(args.fixed_threshold),
    )
    preds_by_exit = [
        probs_to_label_matrix(probs, thresholds_by_exit[k])
        for k, probs in enumerate(probs_by_exit)
    ]

    df = dataset.df.iloc[idx_all].reset_index(drop=True).copy()

    clip_true = []
    full_pred = []
    full_prob = []
    dyn_pred = []
    dyn_prob = []

    for clip_id, g in df.groupby("parent_clip_id", sort=True):
        g = g.sort_values("segment_index").reset_index()
        row_indices = g["index"].astype(int).to_numpy()
        y_clip = y_segment_true[row_indices].max(axis=0).astype(int)

        final_scores = probs_by_exit[-1][row_indices]
        full_score = final_scores.mean(axis=0)
        full_labelset = (full_score >= thresholds_by_exit[-1]).astype(int)

        window_preds = []
        window_scores = []
        for global_idx in row_indices:
            pred_vectors = [preds_by_exit[k][global_idx] for k in range(num_exits)]
            selected_vec, selected_exit_idx = select_exit_for_window(
                pred_vectors=pred_vectors,
                min_exit=int(args.min_exit),
                stable_k=int(args.stable_k),
                allow_empty_stop=bool(args.allow_empty_stop),
            )
            window_preds.append(selected_vec.astype(int))
            window_scores.append(probs_by_exit[selected_exit_idx][global_idx])

        temporal_stable_count = 0
        previous_vec = None
        chosen_w = len(window_preds) - 1
        for w, current in enumerate(window_preds):
            if previous_vec is not None and np.array_equal(current, previous_vec):
                temporal_stable_count += 1
            else:
                temporal_stable_count = 1
            previous_vec = current
            used = w + 1
            is_empty = int(current.sum()) == 0
            is_final_window = w == len(window_preds) - 1
            empty_ok = bool(args.temporal_allow_empty_stop) or is_final_window or not is_empty
            enough_windows = used >= int(args.time_min_windows)
            if enough_windows and temporal_stable_count >= int(args.time_stable_k) and empty_ok:
                chosen_w = w
                break

        used_scores = np.asarray(window_scores[: chosen_w + 1], dtype=np.float32)
        chosen_pred = window_preds[chosen_w].astype(int)
        chosen_score = used_scores.mean(axis=0)

        clip_true.append(y_clip)
        full_pred.append(full_labelset)
        full_prob.append(full_score)
        dyn_pred.append(chosen_pred)
        dyn_prob.append(chosen_score)

    y_clip_true = np.asarray(clip_true).astype(int)
    y_full_pred = np.asarray(full_pred).astype(int)
    y_full_prob = np.asarray(full_prob, dtype=np.float32)
    y_dyn_pred = np.asarray(dyn_pred).astype(int)
    y_dyn_prob = np.asarray(dyn_prob, dtype=np.float32)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "multilabel_clip_window_policy" / "confusion"
    outputs = {}
    outputs["clip_full_final"] = save_confusion_outputs(
        out_dir=out_dir,
        prefix="clip_full_final",
        labels=labels,
        y_true_multi=y_clip_true,
        y_pred_multi=y_full_pred,
        y_true_prob=None,
        y_pred_prob=y_full_prob,
    )
    outputs["clip_dynamic_policy"] = save_confusion_outputs(
        out_dir=out_dir,
        prefix="clip_dynamic_policy",
        labels=labels,
        y_true_multi=y_clip_true,
        y_pred_multi=y_dyn_pred,
        y_true_prob=None,
        y_pred_prob=y_dyn_prob,
    )
    return out_dir, outputs


def main():
    parser = argparse.ArgumentParser(description="Export confusion matrices for multi-label speaker/evaluation runs.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--eval_mode", choices=["segment", "clip"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--threshold_mode",
        choices=["fixed_0p5", "tuned_per_exit", "final_exit_tuned"],
        default="fixed_0p5",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--min_exit", type=int, default=2)
    parser.add_argument("--stable_k", type=int, default=2)
    parser.add_argument("--allow_empty_stop", action="store_true")
    parser.add_argument("--time_min_windows", type=int, default=2)
    parser.add_argument("--time_stable_k", type=int, default=2)
    parser.add_argument("--temporal_allow_empty_stop", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = load_json(run_dir / "config_used.json")
    device = "cuda" if (args.device is None or str(args.device).lower() == "auto") and torch.cuda.is_available() else (args.device or "cpu")
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "ckpt" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, labels, tap_blocks = build_model_from_config(cfg, device)
    model = load_model_state(model, checkpoint, device)

    print("\nExporting confusion matrices")
    print("-" * 90)
    print(f"Run dir:        {run_dir}")
    print(f"Checkpoint:     {checkpoint}")
    print(f"Eval mode:      {args.eval_mode}")
    print(f"Split:          {args.split}")
    print(f"Labels:         {labels}")
    print(f"Tap blocks:     {tap_blocks}")
    print(f"Threshold mode: {args.threshold_mode}")
    print("-" * 90)

    if args.eval_mode == "segment":
        out_dir, outputs = run_segment_mode(args, cfg, model, labels, run_dir, device)
    else:
        out_dir, outputs = run_clip_mode(args, cfg, model, labels, run_dir, device)

    payload = {
        "run_dir": str(run_dir),
        "eval_mode": args.eval_mode,
        "split": args.split,
        "labels": labels,
        "threshold_mode": args.threshold_mode,
        "min_exit": int(args.min_exit),
        "stable_k": int(args.stable_k),
        "time_min_windows": int(args.time_min_windows),
        "time_stable_k": int(args.time_stable_k),
        "outputs": outputs,
        "notes": [
            "Standard K x K confusion matrices use a single-label argmax-backed view.",
            "Per-label confusion tables preserve the multi-label TP/FP/TN/FN interpretation.",
            "For clean speaker stages, the standard confusion matrix is valid because each example normally has one speaker label.",
        ],
    }
    save_json(payload, out_dir / f"{args.eval_mode}_confusion_results.json")

    print("Saved confusion outputs:")
    print(f"  {out_dir}")
    print("-" * 90)


if __name__ == "__main__":
    main()
