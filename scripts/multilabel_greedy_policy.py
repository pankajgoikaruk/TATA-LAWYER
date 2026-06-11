# scripts/multilabel_greedy_policy.py
#
# Sigmoid-aware multi-label greedy / label-set-stability policy evaluation.
#
# Purpose:
#   Convert the static multi-label per-exit baseline into a dynamic neural
#   network evaluation by reporting:
#     - static per-exit quality
#     - dynamic policy quality
#     - exit distribution
#     - average exit depth
#     - depth-unit compute saving estimate
#     - full policy sweep
#     - multi-label thresholded metrics
#     - probability-based AUPRC / mAP metrics
#     - label-set stability diagnostics
#
# Notes:
#   - This script uses sigmoid probabilities and per-label thresholds.
#   - Default threshold source is threshold_tuning/threshold_comparison.json,
#     because it stores tuned thresholds for every exit.
#   - Exit 1 is ignored by default through --min_exit 2, but it can be used
#     later by setting --min_exit 1.
#   - Compute saving is estimated using exit-depth units, not measured FLOPs.
#   - AUPRC/mAP is computed from probabilities, while Macro-F1, Exact Match,
#     Hamming Loss, and Jaccard are computed from thresholded label sets.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

# Make project root importable when running:
# python scripts\multilabel_greedy_policy.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import make_multilabel_loaders
from utils.model_factory import build_audio_exit_net


METRIC_KEYS = [
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "exact_match",
    "hamming_loss",
    "hamming_accuracy",
    "jaccard_score",
    "micro_precision",
    "micro_recall",
    "macro_precision",
    "macro_recall",
    "avg_true_labels",
    "avg_pred_labels",
    "label_cardinality_error",
    "label_cardinality_bias",
    "macro_auprc",
    "micro_auprc",
    "mAP",
]

POLICY_DIAGNOSTIC_KEYS = [
    "exit_consistency",
    "label_set_flip_any_rate",
    "avg_label_set_flip_count",
    "avg_label_bit_flip_count",
]


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
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    value = str(value).strip()
    if not value:
        raise ValueError("tap_blocks cannot be empty.")
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_int_list(value: str | Iterable[int], *, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        raw = str(value).strip()
        if not raw:
            return list(default)
        vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def safe_model_name(path: str | Path) -> str:
    name = Path(path).name
    name = re.sub(r"_20\d{6}_\d{6}.*$", "", name)
    return name


def fmt_float(x: Any, digits: int = 4) -> Any:
    try:
        return round(float(x), digits)
    except Exception:
        return x


def df_to_markdown(df: pd.DataFrame) -> str:
    """Simple markdown writer that does not require tabulate."""
    if df.empty:
        return "_No rows._\n"

    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_float_dtype(df2[col]):
            df2[col] = df2[col].map(lambda v: f"{v:.4f}")

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


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def write_table(df: pd.DataFrame, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(df_to_markdown(df))


def load_model_state(model, ckpt_path: Path, device: str):
    """
    Load model weights.

    Uses weights_only=True if supported by the installed torch version.
    Falls back for older torch versions.
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


def threshold_vector_from_mapping(mapping: dict[str, Any], labels: list[str]) -> np.ndarray:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(
            "Threshold mapping is missing labels:\n"
            f"{missing}\n"
            f"Available keys: {list(mapping.keys())}"
        )
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def load_thresholds_by_exit(
    *,
    run_dir: Path,
    labels: list[str],
    num_exits: int,
    threshold_mode: str,
    fixed_threshold: float,
) -> list[np.ndarray]:
    """
    Load threshold vectors for every exit.

    threshold_mode:
      fixed_0p5:
        use the scalar threshold for all exits and labels.

      tuned_per_exit:
        use threshold_tuning/threshold_comparison.json;
        each exit gets its own tuned per-label threshold vector.

      final_exit_tuned:
        use final-exit tuned thresholds for every exit.
    """
    threshold_mode = str(threshold_mode)

    if threshold_mode == "fixed_0p5":
        th = np.full(len(labels), float(fixed_threshold), dtype=np.float32)
        return [th.copy() for _ in range(num_exits)]

    comparison_path = run_dir / "threshold_tuning" / "threshold_comparison.json"

    if comparison_path.exists():
        payload = load_json(comparison_path)

        if list(payload.get("labels", labels)) != list(labels):
            raise RuntimeError(
                "Label order mismatch between config_used.json and threshold_comparison.json.\n"
                f"config labels:    {labels}\n"
                f"threshold labels: {payload.get('labels')}"
            )

        exits = payload.get("exits", [])
        if len(exits) < num_exits:
            raise RuntimeError(
                f"threshold_comparison.json contains {len(exits)} exits, "
                f"but the model produced {num_exits} exits."
            )

        if threshold_mode == "tuned_per_exit":
            thresholds = []
            for exit_idx in range(num_exits):
                exit_payload = exits[exit_idx]
                mapping = exit_payload.get("tuned_thresholds", {})
                thresholds.append(threshold_vector_from_mapping(mapping, labels))
            return thresholds

        if threshold_mode == "final_exit_tuned":
            mapping = exits[num_exits - 1].get("tuned_thresholds", {})
            th = threshold_vector_from_mapping(mapping, labels)
            return [th.copy() for _ in range(num_exits)]

    final_path = run_dir / "threshold_tuning" / "multilabel_thresholds.json"
    if threshold_mode == "final_exit_tuned" and final_path.exists():
        payload = load_json(final_path)
        mapping = payload.get("thresholds", {})
        th = threshold_vector_from_mapping(mapping, labels)
        return [th.copy() for _ in range(num_exits)]

    if threshold_mode == "tuned_per_exit":
        raise FileNotFoundError(
            "Per-exit tuned thresholds are required for --threshold_mode tuned_per_exit.\n"
            f"Expected file: {comparison_path}\n"
            "Run scripts/tune_multilabel_thresholds.py first."
        )

    raise FileNotFoundError(
        "Could not load tuned thresholds.\n"
        f"Expected: {comparison_path}\n"
        f"or:       {final_path}"
    )


def probs_to_label_matrix(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert sigmoid probabilities to a multi-hot label-set matrix."""
    th = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (np.asarray(y_prob) >= th).astype(int)


def safe_average_precision_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision for one label, with NaN for undefined all-negative labels."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob).astype(float)
    try:
        if int(yt.sum()) == 0:
            return float("nan")
        return float(average_precision_score(yt, yp))
    except Exception:
        return float("nan")


def safe_micro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        yt = np.asarray(y_true).astype(int).ravel()
        yp = np.asarray(y_prob).astype(float).ravel()
        if int(yt.sum()) == 0:
            return float("nan")
        return float(average_precision_score(yt, yp))
    except Exception:
        return float("nan")


def evaluate_multilabel_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Evaluate multi-label predictions.

    Thresholded metrics use y_pred:
      Macro-F1, Micro-F1, Samples-F1, Exact Match, Hamming Loss/Accuracy,
      Jaccard, Label Cardinality Error.

    Probability metrics use y_prob:
      macro_AUPRC / mAP, micro_AUPRC, per-label AUPRC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    ham_loss = float(hamming_loss(y_true, y_pred))
    true_card = y_true.sum(axis=1).astype(float)
    pred_card = y_pred.sum(axis=1).astype(float)

    result = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": ham_loss,
        "hamming_accuracy": float(1.0 - ham_loss),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(true_card.mean()),
        "avg_pred_labels": float(pred_card.mean()),
        "label_cardinality_error": float(np.abs(true_card - pred_card).mean()),
        "label_cardinality_bias": float((pred_card - true_card).mean()),
        "macro_auprc": float("nan"),
        "micro_auprc": float("nan"),
        "mAP": float("nan"),
        "per_label": {},
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        per_label_auprcs = []
        for i in range(len(labels)):
            ap = safe_average_precision_binary(y_true[:, i], y_prob[:, i])
            per_label_auprcs.append(ap)
        if per_label_auprcs and not np.all(np.isnan(per_label_auprcs)):
            macro_auprc = float(np.nanmean(per_label_auprcs))
        else:
            macro_auprc = float("nan")
        result["macro_auprc"] = macro_auprc
        result["mAP"] = macro_auprc
        result["micro_auprc"] = safe_micro_auprc(y_true, y_prob)
    else:
        per_label_auprcs = [float("nan")] * len(labels)

    for i, label in enumerate(labels):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        result["per_label"][label] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auprc": float(per_label_auprcs[i]) if i < len(per_label_auprcs) else float("nan"),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }

    return result


def compute_exit_sequence_stability(preds_by_exit: list[np.ndarray]) -> dict[str, float]:
    """
    Measure how much predicted label sets change across exits.

    label_set_flip_any_rate:
      fraction of samples whose predicted label set changes at least once.

    avg_label_set_flip_count:
      average number of exit-to-exit label-set changes per sample.

    avg_label_bit_flip_count:
      average number of individual label-bit changes per sample.
    """
    if len(preds_by_exit) <= 1:
        return {
            "label_set_flip_any_rate": 0.0,
            "avg_label_set_flip_count": 0.0,
            "avg_label_bit_flip_count": 0.0,
        }

    n = int(preds_by_exit[0].shape[0])
    set_flip_counts = np.zeros(n, dtype=np.float32)
    bit_flip_counts = np.zeros(n, dtype=np.float32)

    for exit_idx in range(1, len(preds_by_exit)):
        prev_pred = preds_by_exit[exit_idx - 1].astype(int)
        curr_pred = preds_by_exit[exit_idx].astype(int)
        changed_bits = np.not_equal(prev_pred, curr_pred)
        set_changed = changed_bits.any(axis=1)
        set_flip_counts += set_changed.astype(np.float32)
        bit_flip_counts += changed_bits.sum(axis=1).astype(np.float32)

    return {
        "label_set_flip_any_rate": float(np.mean(set_flip_counts > 0)),
        "avg_label_set_flip_count": float(set_flip_counts.mean()),
        "avg_label_bit_flip_count": float(bit_flip_counts.mean()),
    }


def gather_selected_probs(probs_by_exit: list[np.ndarray], selected_exit_idx: np.ndarray) -> np.ndarray:
    """Gather each sample's probability vector from the exit chosen by policy."""
    if not probs_by_exit:
        raise ValueError("probs_by_exit cannot be empty.")
    selected_exit_idx = np.asarray(selected_exit_idx).astype(int)
    n = int(probs_by_exit[0].shape[0])
    c = int(probs_by_exit[0].shape[1])
    out = np.zeros((n, c), dtype=np.float32)
    for sample_idx, exit_idx in enumerate(selected_exit_idx):
        out[sample_idx] = probs_by_exit[int(exit_idx)][sample_idx]
    return out


def static_exit_rows(
    *,
    model_name: str,
    run_dir: Path,
    split: str,
    threshold_mode: str,
    thresholds_by_exit: list[np.ndarray],
    y_true: np.ndarray,
    probs_by_exit: list[np.ndarray],
    preds_by_exit: list[np.ndarray],
    labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    details = []

    num_exits = len(preds_by_exit)
    final_pred = preds_by_exit[-1]

    for exit_idx, y_pred in enumerate(preds_by_exit):
        exit_no = exit_idx + 1
        metrics = evaluate_multilabel_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=probs_by_exit[exit_idx],
            labels=labels,
        )

        exit_consistency_to_final = float(np.mean(np.all(y_pred == final_pred, axis=1)))

        row = {
            "model": model_name,
            "run_dir": str(run_dir),
            "split": split,
            "threshold_mode": threshold_mode,
            "num_exits": int(num_exits),
            "exit": int(exit_no),
            "exit_consistency_to_final": fmt_float(exit_consistency_to_final),
        }

        for key in METRIC_KEYS:
            row[key] = fmt_float(metrics.get(key, 0.0))

        rows.append(row)
        details.append(
            {
                "exit": exit_no,
                "thresholds": thresholds_by_exit[exit_idx].tolist(),
                "metrics": metrics,
                "exit_consistency_to_final": exit_consistency_to_final,
            }
        )

    return rows, details


def label_set_stability_policy(
    *,
    preds_by_exit: list[np.ndarray],
    min_exit: int,
    stable_k: int,
    allow_empty_stop: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Apply label-set stability stopping.

    Returns:
      selected_pred: [N, C]
      selected_exit_idx: [N] zero-based exit indices
      exit_counts: {"e1": count, ..., "eK": count}
    """
    if not preds_by_exit:
        raise ValueError("preds_by_exit cannot be empty.")

    num_exits = len(preds_by_exit)
    n = int(preds_by_exit[0].shape[0])

    if min_exit < 1:
        raise ValueError(f"min_exit must be >= 1, got {min_exit}")
    if min_exit > num_exits:
        raise ValueError(f"min_exit={min_exit} exceeds num_exits={num_exits}")
    if stable_k < 1:
        raise ValueError(f"stable_k must be >= 1, got {stable_k}")

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


def policy_metrics_row(
    *,
    model_name: str,
    run_dir: Path,
    split: str,
    threshold_mode: str,
    policy_name: str,
    min_exit: int,
    stable_k: int,
    allow_empty_stop: bool,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    selected_exit_idx: np.ndarray,
    exit_counts: dict[str, int],
    labels: list[str],
    preds_by_exit: list[np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = evaluate_multilabel_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=labels,
    )

    n = int(y_true.shape[0])
    num_exits = int(len(exit_counts))
    selected_depths = selected_exit_idx.astype(float) + 1.0

    avg_exit_depth = float(selected_depths.mean())
    policy_depth_units = float(selected_depths.sum())
    full_depth_units = float(n * num_exits)
    depth_compute_saved_pct = float(100.0 * (1.0 - policy_depth_units / max(full_depth_units, 1e-12)))

    final_pred = preds_by_exit[-1]
    exit_consistency = float(np.mean(np.all(y_pred == final_pred, axis=1)))
    stability_metrics = compute_exit_sequence_stability(preds_by_exit)

    row = {
        "model": model_name,
        "run_dir": str(run_dir),
        "split": split,
        "policy": policy_name,
        "threshold_mode": threshold_mode,
        "min_exit": int(min_exit),
        "stable_k": int(stable_k),
        "allow_empty_stop": bool(allow_empty_stop),
        "n_samples": int(n),
        "num_exits": int(num_exits),
        "avg_exit_depth": fmt_float(avg_exit_depth),
        "final_exit_depth": int(num_exits),
        "policy_depth_units": fmt_float(policy_depth_units),
        "full_depth_units": fmt_float(full_depth_units),
        "depth_compute_saved_pct": fmt_float(depth_compute_saved_pct),
        "exit_consistency": fmt_float(exit_consistency),
        "label_set_flip_any_rate": fmt_float(stability_metrics["label_set_flip_any_rate"]),
        "avg_label_set_flip_count": fmt_float(stability_metrics["avg_label_set_flip_count"]),
        "avg_label_bit_flip_count": fmt_float(stability_metrics["avg_label_bit_flip_count"]),
    }

    for key in METRIC_KEYS:
        row[key] = fmt_float(metrics.get(key, 0.0))

    for k in range(num_exits):
        count = int(exit_counts.get(f"e{k + 1}", 0))
        row[f"exit{k + 1}_samples"] = count
        row[f"exit{k + 1}_fraction"] = fmt_float(count / max(n, 1))

    detail = {
        "row": row,
        "metrics": metrics,
        "exit_counts": exit_counts,
        "exit_mix": {key: float(value / max(n, 1)) for key, value in exit_counts.items()},
        "avg_exit_depth": avg_exit_depth,
        "policy_depth_units": policy_depth_units,
        "full_depth_units": full_depth_units,
        "depth_compute_saved_pct": depth_compute_saved_pct,
        "exit_consistency": exit_consistency,
        "stability_metrics": stability_metrics,
    }

    return row, detail


def make_exit_distribution_df(policy_row: dict[str, Any]) -> pd.DataFrame:
    rows = []
    n = int(policy_row["n_samples"])
    num_exits = int(policy_row["num_exits"])

    for exit_no in range(1, num_exits + 1):
        samples = int(policy_row.get(f"exit{exit_no}_samples", 0))
        rows.append(
            {
                "model": policy_row["model"],
                "split": policy_row["split"],
                "policy": policy_row["policy"],
                "threshold_mode": policy_row["threshold_mode"],
                "min_exit": policy_row["min_exit"],
                "stable_k": policy_row["stable_k"],
                "allow_empty_stop": policy_row["allow_empty_stop"],
                "exit": int(exit_no),
                "samples": samples,
                "fraction": float(samples / max(n, 1)),
            }
        )
    return pd.DataFrame(rows)


def make_compute_depth_df(policy_row: dict[str, Any]) -> pd.DataFrame:
    cols = [
        "model",
        "split",
        "policy",
        "threshold_mode",
        "min_exit",
        "stable_k",
        "allow_empty_stop",
        "n_samples",
        "num_exits",
        "avg_exit_depth",
        "final_exit_depth",
        "policy_depth_units",
        "full_depth_units",
        "depth_compute_saved_pct",
        "exit_consistency",
        "label_set_flip_any_rate",
        "avg_label_set_flip_count",
        "avg_label_bit_flip_count",
    ]
    return pd.DataFrame([{c: policy_row[c] for c in cols if c in policy_row}])


def make_per_label_policy_df(
    *,
    model_name: str,
    selected_detail: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    row_info = selected_detail["row"]
    per_label = selected_detail["metrics"]["per_label"]

    for label, vals in per_label.items():
        rows.append(
            {
                "model": model_name,
                "policy": row_info["policy"],
                "threshold_mode": row_info["threshold_mode"],
                "min_exit": row_info["min_exit"],
                "stable_k": row_info["stable_k"],
                "label": label,
                "precision": fmt_float(vals.get("precision", 0.0)),
                "recall": fmt_float(vals.get("recall", 0.0)),
                "f1": fmt_float(vals.get("f1", 0.0)),
                "auprc": fmt_float(vals.get("auprc", float("nan"))),
                "support": int(vals.get("support", 0)),
                "predicted_positive": int(vals.get("predicted_positive", 0)),
            }
        )
    return pd.DataFrame(rows)


def make_policy_summary_md(
    *,
    out_path: Path,
    model_name: str,
    run_dir: Path,
    split: str,
    threshold_mode: str,
    selected_policy_df: pd.DataFrame,
    exit_distribution_df: pd.DataFrame,
    compute_depth_df: pd.DataFrame,
    static_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    per_label_df: pd.DataFrame,
):
    lines = []

    lines.append(f"# Multi-label greedy policy summary — `{model_name}`\n")
    lines.append("## Context\n")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Evaluation split: `{split}`")
    lines.append(f"- Threshold mode: `{threshold_mode}`")
    lines.append("- Policy: sigmoid-aware label-set stability")
    lines.append("- Compute saving is estimated using depth units, not measured FLOPs/latency.")
    lines.append("- F1/Hamming/Jaccard metrics are thresholded label-set metrics.")
    lines.append("- AUPRC/mAP metrics are probability-based ranking metrics.\n")

    lines.append("## Selected dynamic early-exit policy\n")
    selected_cols = existing_cols(selected_policy_df, [
        "model", "policy", "min_exit", "stable_k",
        "macro_f1", "micro_f1", "samples_f1", "exact_match",
        "hamming_loss", "hamming_accuracy", "jaccard_score",
        "macro_auprc", "micro_auprc", "mAP",
        "avg_exit_depth", "depth_compute_saved_pct", "exit_consistency",
        "label_set_flip_any_rate", "avg_label_set_flip_count", "avg_label_bit_flip_count",
    ])
    lines.append(df_to_markdown(selected_policy_df[selected_cols]))

    lines.append("\n## Exit distribution\n")
    lines.append(df_to_markdown(exit_distribution_df))

    lines.append("\n## Compute-depth and stability estimate\n")
    lines.append(df_to_markdown(compute_depth_df))

    lines.append("\n## Static per-exit quality\n")
    compact_static_cols = existing_cols(static_df, [
        "model", "split", "threshold_mode", "exit",
        "macro_f1", "micro_f1", "samples_f1", "exact_match",
        "hamming_loss", "hamming_accuracy", "jaccard_score",
        "macro_auprc", "micro_auprc", "mAP",
        "avg_pred_labels", "label_cardinality_error",
        "exit_consistency_to_final",
    ])
    lines.append(df_to_markdown(static_df[compact_static_cols]))

    lines.append("\n## Full policy sweep\n")
    compact_sweep_cols = existing_cols(sweep_df, [
        "model", "policy", "min_exit", "stable_k", "allow_empty_stop",
        "macro_f1", "micro_f1", "samples_f1", "exact_match",
        "hamming_loss", "hamming_accuracy", "jaccard_score",
        "macro_auprc", "micro_auprc", "mAP",
        "avg_exit_depth", "depth_compute_saved_pct", "exit_consistency",
        "label_set_flip_any_rate", "avg_label_set_flip_count", "avg_label_bit_flip_count",
    ])
    lines.append(df_to_markdown(sweep_df[compact_sweep_cols]))

    lines.append("\n## Selected policy per-label quality\n")
    lines.append(df_to_markdown(per_label_df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sigmoid-aware multi-label early-exit policy using "
            "label-set stability."
        )
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Trained run directory containing config_used.json and ckpt/best.pt.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional display name for tables. Default: run_dir name without timestamp.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Default: <run_dir>/ckpt/best.pt",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split. Default: test.",
    )
    parser.add_argument(
        "--threshold_mode",
        choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"],
        default="tuned_per_exit",
        help=(
            "Threshold source. tuned_per_exit is preferred because each exit "
            "uses its own tuned per-label thresholds."
        ),
    )
    parser.add_argument(
        "--fixed_threshold",
        type=float,
        default=0.5,
        help="Scalar threshold used when --threshold_mode fixed_0p5.",
    )

    parser.add_argument(
        "--min_exit",
        type=int,
        default=2,
        help="First exit allowed for dynamic stopping. Default 2 ignores Exit 1.",
    )
    parser.add_argument(
        "--stable_k",
        type=int,
        default=2,
        help="Consecutive stable label sets required before stopping.",
    )
    parser.add_argument(
        "--allow_empty_stop",
        action="store_true",
        help=(
            "Allow early stopping on an empty predicted label set. "
            "Default false avoids stopping too early on empty predictions."
        ),
    )

    parser.add_argument(
        "--sweep_min_exits",
        default="2",
        help='Comma-separated min_exit values for appendix sweep, e.g. "1,2".',
    )
    parser.add_argument(
        "--sweep_stable_k",
        default="1,2,3",
        help='Comma-separated stable_k values for appendix sweep, e.g. "1,2,3".',
    )
    parser.add_argument(
        "--no_sweep",
        action="store_true",
        help="Disable full policy sweep and evaluate only the selected policy.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config_used.json.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: <run_dir>/multilabel_greedy_policy",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config_used.json"
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
    seed = int(cfg.get("seed", 42))

    model_name = str(args.name or safe_model_name(run_dir))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "multilabel_greedy_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve current/future hint configuration from the trained run.
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

    print("\nMulti-label greedy policy evaluation")
    print("-" * 90)
    print(f"Run dir:        {run_dir}")
    print(f"Checkpoint:     {checkpoint}")
    print(f"Manifest:       {manifest}")
    print(f"Features root:  {features_root}")
    print(f"Labels JSON:    {labels_json}")
    print(f"Output dir:     {out_dir}")
    print(f"Model name:     {model_name}")
    print(f"Device:         {device}")
    print(f"Split:          {args.split}")
    print(f"Labels:         {labels}")
    print(f"Tap blocks:     {tap_blocks}")
    print(f"Threshold mode: {args.threshold_mode}")
    print(
        f"Selected policy: min_exit={args.min_exit}, stable_k={args.stable_k}, "
        f"allow_empty_stop={bool(args.allow_empty_stop)}"
    )
    print("-" * 90)

    dl_tr, dl_va, dl_te, loaded_labels = make_multilabel_loaders(
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
        raise RuntimeError(
            "Label order mismatch between config and loaded dataset.\n"
            f"config labels: {labels}\n"
            f"loaded labels: {loaded_labels}"
        )

    eval_loader = dl_te if args.split == "test" else dl_va

    model = build_audio_exit_net(
        num_classes=num_labels,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    model = load_model_state(model, checkpoint, device)

    y_true, probs_by_exit = collect_probs_and_targets(model, eval_loader, device)
    num_exits = len(probs_by_exit)

    thresholds_by_exit = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=num_exits,
        threshold_mode=args.threshold_mode,
        fixed_threshold=float(args.fixed_threshold),
    )

    preds_by_exit = [
        probs_to_label_matrix(y_prob=probs, thresholds=thresholds_by_exit[exit_idx])
        for exit_idx, probs in enumerate(probs_by_exit)
    ]

    # Static per-exit quality.
    static_rows, static_details = static_exit_rows(
        model_name=model_name,
        run_dir=run_dir,
        split=args.split,
        threshold_mode=args.threshold_mode,
        thresholds_by_exit=thresholds_by_exit,
        y_true=y_true,
        probs_by_exit=probs_by_exit,
        preds_by_exit=preds_by_exit,
        labels=labels,
    )
    static_df = pd.DataFrame(static_rows)

    # Selected policy.
    selected_pred, selected_exit_idx, selected_exit_counts = label_set_stability_policy(
        preds_by_exit=preds_by_exit,
        min_exit=int(args.min_exit),
        stable_k=int(args.stable_k),
        allow_empty_stop=bool(args.allow_empty_stop),
    )
    selected_prob = gather_selected_probs(probs_by_exit, selected_exit_idx)

    selected_row, selected_detail = policy_metrics_row(
        model_name=model_name,
        run_dir=run_dir,
        split=args.split,
        threshold_mode=args.threshold_mode,
        policy_name="label_set_stability",
        min_exit=int(args.min_exit),
        stable_k=int(args.stable_k),
        allow_empty_stop=bool(args.allow_empty_stop),
        y_true=y_true,
        y_pred=selected_pred,
        y_prob=selected_prob,
        selected_exit_idx=selected_exit_idx,
        exit_counts=selected_exit_counts,
        labels=labels,
        preds_by_exit=preds_by_exit,
    )

    selected_policy_df = pd.DataFrame([selected_row])
    exit_distribution_df = make_exit_distribution_df(selected_row)
    compute_depth_df = make_compute_depth_df(selected_row)
    per_label_df = make_per_label_policy_df(
        model_name=model_name,
        selected_detail=selected_detail,
    )

    # Full policy sweep.
    sweep_rows = []
    sweep_details = []

    if args.no_sweep:
        sweep_rows = [selected_row]
        sweep_details = [selected_detail]
    else:
        sweep_min_exits = parse_int_list(args.sweep_min_exits, default=[int(args.min_exit)])
        sweep_stable_ks = parse_int_list(args.sweep_stable_k, default=[int(args.stable_k)])

        # Always include the selected policy even if not in the sweep strings.
        if int(args.min_exit) not in sweep_min_exits:
            sweep_min_exits.append(int(args.min_exit))
        if int(args.stable_k) not in sweep_stable_ks:
            sweep_stable_ks.append(int(args.stable_k))

        for min_exit in sweep_min_exits:
            if min_exit < 1 or min_exit > num_exits:
                print(f"[WARN] Skipping sweep min_exit={min_exit}; num_exits={num_exits}.")
                continue

            for stable_k in sweep_stable_ks:
                if stable_k < 1:
                    print(f"[WARN] Skipping sweep stable_k={stable_k}; must be >= 1.")
                    continue

                policy_pred, policy_exit_idx, policy_exit_counts = label_set_stability_policy(
                    preds_by_exit=preds_by_exit,
                    min_exit=int(min_exit),
                    stable_k=int(stable_k),
                    allow_empty_stop=bool(args.allow_empty_stop),
                )
                policy_prob = gather_selected_probs(probs_by_exit, policy_exit_idx)

                row, detail = policy_metrics_row(
                    model_name=model_name,
                    run_dir=run_dir,
                    split=args.split,
                    threshold_mode=args.threshold_mode,
                    policy_name="label_set_stability",
                    min_exit=int(min_exit),
                    stable_k=int(stable_k),
                    allow_empty_stop=bool(args.allow_empty_stop),
                    y_true=y_true,
                    y_pred=policy_pred,
                    y_prob=policy_prob,
                    selected_exit_idx=policy_exit_idx,
                    exit_counts=policy_exit_counts,
                    labels=labels,
                    preds_by_exit=preds_by_exit,
                )
                sweep_rows.append(row)
                sweep_details.append(detail)

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["min_exit", "stable_k"]
    ).reset_index(drop=True)

    # Write required tables.
    write_table(
        static_df,
        out_dir / "static_per_exit_quality.csv",
        out_dir / "static_per_exit_quality.md",
    )
    write_table(
        selected_policy_df,
        out_dir / "dynamic_early_exit_efficiency.csv",
        out_dir / "dynamic_early_exit_efficiency.md",
    )
    write_table(
        exit_distribution_df,
        out_dir / "exit_distribution.csv",
        out_dir / "exit_distribution.md",
    )
    write_table(
        compute_depth_df,
        out_dir / "compute_depth_units.csv",
        out_dir / "compute_depth_units.md",
    )
    write_table(
        sweep_df,
        out_dir / "full_policy_sweep.csv",
        out_dir / "full_policy_sweep.md",
    )
    write_table(
        per_label_df,
        out_dir / "selected_policy_per_label.csv",
        out_dir / "selected_policy_per_label.md",
    )

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "split": args.split,
        "labels": labels,
        "num_labels": num_labels,
        "num_exits": num_exits,
        "tap_blocks": list(tap_blocks),
        "n_mels": n_mels,
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": [
            {label: float(th[i]) for i, label in enumerate(labels)}
            for th in thresholds_by_exit
        ],
        "selected_policy": selected_detail,
        "static_per_exit": static_details,
        "full_policy_sweep": sweep_details,
        "metric_notes": {
            "macro_f1_micro_f1_samples_f1": "Thresholded multi-label F1 metrics.",
            "exact_match": "Strict subset accuracy; all labels must match.",
            "hamming_loss": "Fraction of wrong label decisions.",
            "hamming_accuracy": "1 - hamming_loss.",
            "jaccard_score": "Sample-wise label-set overlap.",
            "label_cardinality_error": "Mean absolute error in number of predicted labels.",
            "macro_auprc_mAP": "Mean per-label average precision from probabilities.",
            "micro_auprc": "Global average precision over all label decisions.",
            "exit_consistency": "Selected policy label set equals final-exit label set.",
            "label_set_flip_any_rate": "Fraction of samples whose label set changes across exits.",
            "avg_label_set_flip_count": "Mean number of exit-to-exit label-set changes.",
            "avg_label_bit_flip_count": "Mean number of individual label-bit changes across exits.",
        },
        "outputs": {
            "static_per_exit_quality_csv": str(out_dir / "static_per_exit_quality.csv"),
            "dynamic_early_exit_efficiency_csv": str(out_dir / "dynamic_early_exit_efficiency.csv"),
            "exit_distribution_csv": str(out_dir / "exit_distribution.csv"),
            "compute_depth_units_csv": str(out_dir / "compute_depth_units.csv"),
            "full_policy_sweep_csv": str(out_dir / "full_policy_sweep.csv"),
            "selected_policy_per_label_csv": str(out_dir / "selected_policy_per_label.csv"),
            "policy_summary_md": str(out_dir / "policy_summary.md"),
        },
        "notes": [
            "Depth compute saving is an estimate using exit depth units.",
            "Exit 1 is ignored by default when min_exit=2, but can be enabled with --min_exit 1.",
            "Empty label-set early stopping is disabled by default unless --allow_empty_stop is used.",
            "AUPRC/mAP metrics are probability-based and should be interpreted separately from thresholded F1 metrics.",
        ],
    }

    save_json(results, out_dir / "policy_results.json")

    make_policy_summary_md(
        out_path=out_dir / "policy_summary.md",
        model_name=model_name,
        run_dir=run_dir,
        split=args.split,
        threshold_mode=args.threshold_mode,
        selected_policy_df=selected_policy_df,
        exit_distribution_df=exit_distribution_df,
        compute_depth_df=compute_depth_df,
        static_df=static_df,
        sweep_df=sweep_df,
        per_label_df=per_label_df,
    )

    print("\nSelected dynamic policy")
    print("-" * 90)
    print(selected_policy_df.to_string(index=False))

    print("\nExit distribution")
    print("-" * 90)
    print(exit_distribution_df.to_string(index=False))

    print("\nSaved outputs")
    print("-" * 90)
    for key, value in results["outputs"].items():
        print(f"{key}: {value}")
    print(f"policy_results_json: {out_dir / 'policy_results.json'}")
    print("-" * 90)


if __name__ == "__main__":
    main()
