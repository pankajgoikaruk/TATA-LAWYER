# scripts/multilabel_clip_window_policy.py
#
# Multi-label clip/window-level early-exit evaluator.
#
# Purpose:
#   Adds the clip/window-efficiency metrics missing from the segment-level
#   multilabel_greedy_policy.py output.
#
# Outputs:
#   <run_dir>/multilabel_clip_window_policy/
#     - clip_full_final_quality.csv/.md
#     - clip_dynamic_window_efficiency.csv/.md
#     - clip_policy_per_label.csv/.md
#     - clip_window_distribution.csv/.md
#     - clip_policy_results.json
#     - clip_policy_summary.md
#
# Notes:
#   - This script groups segment rows by parent_clip_id.
#   - Full-clip baseline uses all windows and final-exit mean probabilities.
#   - Dynamic clip policy uses:
#       1) per-window label-set stability across exits
#       2) temporal label-set stability across windows
#   - Detection latency / missed-event / false-alarm are marked N/A for
#     persistent speaker labels. For transient event classes, add event onset
#     metadata and evaluate those separately.

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import load_labels
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


def fmt_float(x: Any, digits: int = 4) -> Any:
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return np.nan
        return round(x, digits)
    except Exception:
        return x


def safe_model_name(path: str | Path) -> str:
    name = Path(path).name
    return re.sub(r"_20\d{6}_\d{6}.*$", "", name)


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


def write_table(df: pd.DataFrame, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_md.write_text(df_to_markdown(df), encoding="utf-8")


def threshold_vector_from_mapping(mapping: dict[str, Any], labels: list[str]) -> np.ndarray:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(f"Threshold mapping is missing labels: {missing}")
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def load_thresholds_by_exit(
    *,
    run_dir: Path,
    labels: list[str],
    num_exits: int,
    threshold_mode: str,
    fixed_threshold: float,
) -> list[np.ndarray]:
    threshold_mode = str(threshold_mode)

    if threshold_mode == "fixed_0p5":
        th = np.full(len(labels), float(fixed_threshold), dtype=np.float32)
        return [th.copy() for _ in range(num_exits)]

    comparison_path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    if comparison_path.exists():
        payload = load_json(comparison_path)
        if list(payload.get("labels", labels)) != list(labels):
            raise RuntimeError("Label order mismatch between config and threshold_comparison.json.")

        exits = payload.get("exits", [])
        if len(exits) < num_exits:
            raise RuntimeError(
                f"threshold_comparison.json contains {len(exits)} exits, but model has {num_exits}."
            )

        if threshold_mode == "tuned_per_exit":
            return [
                threshold_vector_from_mapping(exits[exit_idx].get("tuned_thresholds", {}), labels)
                for exit_idx in range(num_exits)
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
        f"Could not load thresholds for mode={threshold_mode}. "
        f"Expected {comparison_path} or {final_path}."
    )


def probs_to_label_matrix(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    th = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (np.asarray(y_prob) >= th).astype(int)


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray, average: str | None):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def evaluate_multilabel_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    h_loss = float(hamming_loss(y_true, y_pred))
    true_card = y_true.sum(axis=1)
    pred_card = y_pred.sum(axis=1)

    result = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": h_loss,
        "hamming_accuracy": float(1.0 - h_loss),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(true_card.mean()),
        "avg_pred_labels": float(pred_card.mean()),
        "label_cardinality_error": float(np.mean(np.abs(pred_card - true_card))),
        "label_cardinality_bias": float(np.mean(pred_card - true_card)),
        "macro_auprc": float("nan"),
        "micro_auprc": float("nan"),
        "mAP": float("nan"),
        "per_label": {},
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=np.float32)
        result["macro_auprc"] = safe_average_precision(y_true, y_score, average="macro")
        result["micro_auprc"] = safe_average_precision(y_true, y_score, average="micro")
        result["mAP"] = result["macro_auprc"]

    per_label_ap = None
    if y_score is not None:
        per_label_ap = safe_average_precision(y_true, y_score, average=None)

    for i, label in enumerate(labels):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        result["per_label"][label] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "auprc": float(per_label_ap[i]) if per_label_ap is not None else float("nan"),
        }

    return result


class FeatureWithMetaDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_csv: str | Path,
        features_root: str | Path,
        labels_json: str | Path,
        split: str,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.features_root = Path(features_root)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        if not self.features_root.exists():
            raise FileNotFoundError(f"Features root not found: {self.features_root}")

        df = pd.read_csv(self.manifest_csv)
        self.labels = load_labels(labels_json, manifest_df=df)

        required_cols = ["split", "feat_relpath", "parent_clip_id", "segment_index"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Manifest is missing columns required for clip/window evaluation: {missing}"
            )

        self.df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No rows found for split={split} in {self.manifest_csv}")

        missing_labels = [label for label in self.labels if label not in self.df.columns]
        if missing_labels:
            raise RuntimeError(f"Manifest missing label columns: {missing_labels}")

        self.targets = self.df[self.labels].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        feat_path = self.features_root / Path(feat_rel)
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        S = np.load(feat_path).astype(np.float32)
        if S.ndim != 2:
            raise RuntimeError(f"Expected [n_mels, T], got {S.shape} for {feat_path}")

        x = torch.from_numpy(S).float().unsqueeze(0)
        y = torch.from_numpy(row[self.labels].astype(np.float32).values).float()
        return x, y, int(idx)


def load_model_state(model, ckpt_path: Path, device: str):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def collect_probs(model, dataset: FeatureWithMetaDataset, batch_size: int, num_workers: int, device: str):
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


def label_set_flip_stats(seq: list[np.ndarray]) -> dict[str, float]:
    if len(seq) <= 1:
        return {
            "label_set_flip_any": 0.0,
            "label_set_flip_count": 0.0,
            "label_bit_flip_count": 0.0,
        }

    set_flips = 0
    bit_flips = 0
    for i in range(1, len(seq)):
        prev = seq[i - 1].astype(int)
        cur = seq[i].astype(int)
        changed = cur != prev
        if bool(np.any(changed)):
            set_flips += 1
            bit_flips += int(changed.sum())

    return {
        "label_set_flip_any": float(set_flips > 0),
        "label_set_flip_count": float(set_flips),
        "label_bit_flip_count": float(bit_flips),
    }


def mode_or_nan(values: list[int]) -> float:
    if not values:
        return float("nan")
    return float(Counter(values).most_common(1)[0][0])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-label clip/window-level early-exit policy."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--name", default=None)
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
    parser.add_argument(
        "--temporal_allow_empty_stop",
        action="store_true",
        help="Allow temporal early stop on empty label set.",
    )

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

    manifest = Path(cfg["manifest"]).resolve()
    features_root = Path(cfg["features_root"]).resolve()
    labels_json = Path(cfg["labels_json"]).resolve()

    labels = [str(x) for x in cfg["labels"]]
    num_labels = int(cfg["num_labels"])
    n_mels = int(cfg.get("n_mels", 64))
    tap_blocks = parse_tap_blocks(cfg["tap_blocks"])
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = str(args.name or safe_model_name(run_dir))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "multilabel_clip_window_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print("\nMulti-label clip/window policy evaluation")
    print("-" * 90)
    print(f"Run dir:             {run_dir}")
    print(f"Checkpoint:          {checkpoint}")
    print(f"Manifest:            {manifest}")
    print(f"Features root:       {features_root}")
    print(f"Labels JSON:         {labels_json}")
    print(f"Output dir:          {out_dir}")
    print(f"Model name:          {model_name}")
    print(f"Device:              {device}")
    print(f"Split:               {args.split}")
    print(f"Labels:              {labels}")
    print(f"Depth policy:        min_exit={args.min_exit}, stable_k={args.stable_k}")
    print(f"Temporal policy:     time_min_windows={args.time_min_windows}, time_stable_k={args.time_stable_k}")
    print("-" * 90)

    dataset = FeatureWithMetaDataset(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        split=args.split,
    )
    if list(dataset.labels) != list(labels):
        raise RuntimeError("Label order mismatch between config and dataset labels.")

    model = build_audio_exit_net(
        num_classes=num_labels,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)
    model = load_model_state(model, checkpoint, device)

    idx_all, y_segment_true, probs_by_exit = collect_probs(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        device=device,
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

    # Full-clip final-exit baseline: use all windows and final-exit mean probability.
    clip_true = []
    clip_full_score = []
    clip_full_pred = []
    clip_ids = []
    clip_total_windows = []

    # Dynamic window policy state.
    dyn_pred = []
    dyn_score = []
    dyn_used_windows = []
    dyn_total_windows = []
    dyn_compute_units = []
    dyn_full_compute_units = []
    dyn_avg_exit_depth = []
    dyn_decision_time_sec = []
    dyn_flip_set_any = []
    dyn_flip_set_count = []
    dyn_flip_bit_count = []
    dyn_rows = []

    for clip_id, g in df.groupby("parent_clip_id", sort=True):
        g = g.sort_values("segment_index").reset_index()
        row_indices = g["index"].astype(int).to_numpy()

        y_clip = y_segment_true[row_indices].max(axis=0).astype(int)
        total_windows = int(len(row_indices))

        final_scores = probs_by_exit[-1][row_indices]
        full_score = final_scores.mean(axis=0)
        full_pred = (full_score >= thresholds_by_exit[-1]).astype(int)

        clip_ids.append(str(clip_id))
        clip_true.append(y_clip)
        clip_full_score.append(full_score)
        clip_full_pred.append(full_pred)
        clip_total_windows.append(total_windows)

        window_preds = []
        window_scores = []
        window_exit_depths = []

        for local_pos, global_idx in enumerate(row_indices):
            pred_vectors = [preds_by_exit[k][global_idx] for k in range(num_exits)]
            selected_vec, selected_exit_idx = select_exit_for_window(
                pred_vectors=pred_vectors,
                min_exit=int(args.min_exit),
                stable_k=int(args.stable_k),
                allow_empty_stop=bool(args.allow_empty_stop),
            )
            window_preds.append(selected_vec.astype(int))
            window_scores.append(probs_by_exit[selected_exit_idx][global_idx])
            window_exit_depths.append(int(selected_exit_idx + 1))

        temporal_stable_count = 0
        previous_vec = None
        chosen_w = total_windows - 1

        for w, current in enumerate(window_preds):
            if previous_vec is not None and np.array_equal(current, previous_vec):
                temporal_stable_count += 1
            else:
                temporal_stable_count = 1

            previous_vec = current

            used = w + 1
            is_empty = int(current.sum()) == 0
            is_final_window = w == total_windows - 1
            empty_ok = bool(args.temporal_allow_empty_stop) or is_final_window or not is_empty
            enough_windows = used >= int(args.time_min_windows)

            if enough_windows and temporal_stable_count >= int(args.time_stable_k) and empty_ok:
                chosen_w = w
                break

        used_windows = int(chosen_w + 1)
        used_preds_seq = window_preds[:used_windows]
        used_scores_seq = window_scores[:used_windows]
        used_depths = window_exit_depths[:used_windows]

        # Use current stable label set as early clip prediction.
        chosen_pred = used_preds_seq[-1].astype(int)
        chosen_score = np.mean(np.asarray(used_scores_seq, dtype=np.float32), axis=0)

        flip = label_set_flip_stats(used_preds_seq)
        end_col = "end_sec" if "end_sec" in g.columns else None
        if end_col:
            decision_time = float(g.iloc[chosen_w][end_col])
        else:
            hop_sec = float(g.iloc[0].get("hop_sec", 0.5))
            segment_sec = float(g.iloc[0].get("segment_sec", 1.0))
            decision_time = float(chosen_w * hop_sec + segment_sec)

        compute_units = float(sum(used_depths))
        full_compute = float(total_windows * num_exits)

        dyn_pred.append(chosen_pred)
        dyn_score.append(chosen_score)
        dyn_used_windows.append(used_windows)
        dyn_total_windows.append(total_windows)
        dyn_compute_units.append(compute_units)
        dyn_full_compute_units.append(full_compute)
        dyn_avg_exit_depth.append(float(np.mean(used_depths)))
        dyn_decision_time_sec.append(decision_time)
        dyn_flip_set_any.append(flip["label_set_flip_any"])
        dyn_flip_set_count.append(flip["label_set_flip_count"])
        dyn_flip_bit_count.append(flip["label_bit_flip_count"])

        dyn_rows.append({
            "model": model_name,
            "clip_id": str(clip_id),
            "split": args.split,
            "num_exits": int(num_exits),
            "total_windows": int(total_windows),
            "used_windows": int(used_windows),
            "windows_saved": int(total_windows - used_windows),
            "windows_saved_pct": float(100.0 * (1.0 - used_windows / max(total_windows, 1))),
            "avg_exit_depth": float(np.mean(used_depths)),
            "clip_compute_units": compute_units,
            "full_clip_compute_units": full_compute,
            "clip_compute_saved_pct": float(100.0 * (1.0 - compute_units / max(full_compute, 1e-12))),
            "clip_decision_time_sec": decision_time,
            "label_set_flip_any": flip["label_set_flip_any"],
            "label_set_flip_count": flip["label_set_flip_count"],
            "label_bit_flip_count": flip["label_bit_flip_count"],
            "true_labels": ",".join([labels[i] for i, v in enumerate(y_clip) if int(v) == 1]),
            "pred_labels": ",".join([labels[i] for i, v in enumerate(chosen_pred) if int(v) == 1]),
        })

    y_clip_true = np.asarray(clip_true).astype(int)
    y_full_pred = np.asarray(clip_full_pred).astype(int)
    y_full_score = np.asarray(clip_full_score, dtype=np.float32)
    y_dyn_pred = np.asarray(dyn_pred).astype(int)
    y_dyn_score = np.asarray(dyn_score, dtype=np.float32)

    full_metrics = evaluate_multilabel_predictions(
        y_true=y_clip_true,
        y_pred=y_full_pred,
        y_score=y_full_score,
        labels=labels,
    )
    dyn_metrics = evaluate_multilabel_predictions(
        y_true=y_clip_true,
        y_pred=y_dyn_pred,
        y_score=y_dyn_score,
        labels=labels,
    )

    n_clips = int(y_clip_true.shape[0])
    total_used_windows = float(np.sum(dyn_used_windows))
    total_windows_all = float(np.sum(dyn_total_windows))
    total_compute = float(np.sum(dyn_compute_units))
    total_full_compute = float(np.sum(dyn_full_compute_units))

    full_row = {
        "model": model_name,
        "run_dir": str(run_dir),
        "split": args.split,
        "clip_policy": "full_clip_final_exit_mean_prob",
        "n_clips": n_clips,
        "num_exits": int(num_exits),
        "avg_windows_used": fmt_float(np.mean(clip_total_windows)),
        "avg_windows_total": fmt_float(np.mean(clip_total_windows)),
        "windows_saved_pct": 0.0,
        "avg_exit_depth": float(num_exits),
        "avg_compute_units": fmt_float(np.mean(np.asarray(clip_total_windows) * num_exits)),
        "compute_saved_pct": 0.0,
    }
    for key in METRIC_KEYS:
        full_row[key] = fmt_float(full_metrics.get(key, np.nan))

    dyn_row = {
        "model": model_name,
        "run_dir": str(run_dir),
        "split": args.split,
        "clip_policy": "depth_then_time_label_set_stability",
        "threshold_mode": args.threshold_mode,
        "min_exit": int(args.min_exit),
        "stable_k": int(args.stable_k),
        "time_min_windows": int(args.time_min_windows),
        "time_stable_k": int(args.time_stable_k),
        "allow_empty_stop": bool(args.allow_empty_stop),
        "temporal_allow_empty_stop": bool(args.temporal_allow_empty_stop),
        "n_clips": n_clips,
        "num_exits": int(num_exits),
        "avg_windows_used": fmt_float(np.mean(dyn_used_windows)),
        "avg_windows_total": fmt_float(np.mean(dyn_total_windows)),
        "window_distribution_mode": fmt_float(mode_or_nan([int(x) for x in dyn_used_windows])),
        "windows_saved_pct": fmt_float(100.0 * (1.0 - total_used_windows / max(total_windows_all, 1e-12))),
        "avg_exit_depth": fmt_float(np.mean(dyn_avg_exit_depth)),
        "avg_compute_units": fmt_float(np.mean(dyn_compute_units)),
        "full_avg_compute_units": fmt_float(np.mean(dyn_full_compute_units)),
        "compute_saved_pct": fmt_float(100.0 * (1.0 - total_compute / max(total_full_compute, 1e-12))),
        "avg_clip_decision_time_sec": fmt_float(np.mean(dyn_decision_time_sec)),
        "label_set_flip_any_rate": fmt_float(np.mean(dyn_flip_set_any)),
        "avg_label_set_flip_count": fmt_float(np.mean(dyn_flip_set_count)),
        "avg_label_bit_flip_count": fmt_float(np.mean(dyn_flip_bit_count)),
        "event_detection_applicable": False,
        "detection_latency_sec": np.nan,
        "missed_event_rate": np.nan,
        "false_alarm_rate": np.nan,
    }
    for key in METRIC_KEYS:
        dyn_row[key] = fmt_float(dyn_metrics.get(key, np.nan))

    full_df = pd.DataFrame([full_row])
    dyn_df = pd.DataFrame([dyn_row])
    clip_details_df = pd.DataFrame(dyn_rows)

    dist_df = (
        clip_details_df.groupby("used_windows")
        .size()
        .reset_index(name="clips")
        .sort_values("used_windows")
    )
    dist_df["fraction"] = dist_df["clips"] / max(n_clips, 1)
    dist_df.insert(0, "model", model_name)
    dist_df.insert(1, "split", args.split)

    per_label_rows = []
    for label, vals in dyn_metrics["per_label"].items():
        per_label_rows.append({
            "model": model_name,
            "split": args.split,
            "clip_policy": "depth_then_time_label_set_stability",
            "label": label,
            "precision": fmt_float(vals.get("precision", 0.0)),
            "recall": fmt_float(vals.get("recall", 0.0)),
            "f1": fmt_float(vals.get("f1", 0.0)),
            "support": int(vals.get("support", 0)),
            "predicted_positive": int(vals.get("predicted_positive", 0)),
            "auprc": fmt_float(vals.get("auprc", np.nan)),
        })
    per_label_df = pd.DataFrame(per_label_rows)

    write_table(full_df, out_dir / "clip_full_final_quality.csv", out_dir / "clip_full_final_quality.md")
    write_table(dyn_df, out_dir / "clip_dynamic_window_efficiency.csv", out_dir / "clip_dynamic_window_efficiency.md")
    write_table(per_label_df, out_dir / "clip_policy_per_label.csv", out_dir / "clip_policy_per_label.md")
    write_table(dist_df, out_dir / "clip_window_distribution.csv", out_dir / "clip_window_distribution.md")
    write_table(clip_details_df, out_dir / "clip_level_details.csv", out_dir / "clip_level_details.md")

    summary_lines = [
        f"# Multi-label clip/window policy summary — `{model_name}`\n",
        "## Context\n",
        f"- Run directory: `{run_dir}`",
        f"- Split: `{args.split}`",
        f"- Threshold mode: `{args.threshold_mode}`",
        f"- Labels: `{labels}`",
        "- Full-clip baseline: all windows + final exit + mean probability.",
        "- Dynamic policy: per-window depth stability followed by temporal label-set stability.",
        "- Detection latency / missed-event / false-alarm are N/A for persistent speaker labels.\n",
        "## Full-clip baseline\n",
        df_to_markdown(full_df),
        "\n## Dynamic clip/window policy\n",
        df_to_markdown(dyn_df),
        "\n## Window distribution\n",
        df_to_markdown(dist_df),
        "\n## Per-label dynamic clip quality\n",
        df_to_markdown(per_label_df),
    ]
    (out_dir / "clip_policy_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "split": args.split,
        "labels": labels,
        "num_labels": num_labels,
        "num_exits": num_exits,
        "tap_blocks": list(tap_blocks),
        "threshold_mode": args.threshold_mode,
        "depth_policy": {
            "min_exit": int(args.min_exit),
            "stable_k": int(args.stable_k),
            "allow_empty_stop": bool(args.allow_empty_stop),
        },
        "temporal_policy": {
            "time_min_windows": int(args.time_min_windows),
            "time_stable_k": int(args.time_stable_k),
            "temporal_allow_empty_stop": bool(args.temporal_allow_empty_stop),
        },
        "full_clip_baseline": {
            "row": full_row,
            "metrics": full_metrics,
        },
        "dynamic_clip_policy": {
            "row": dyn_row,
            "metrics": dyn_metrics,
        },
        "outputs": {
            "clip_full_final_quality_csv": str(out_dir / "clip_full_final_quality.csv"),
            "clip_dynamic_window_efficiency_csv": str(out_dir / "clip_dynamic_window_efficiency.csv"),
            "clip_policy_per_label_csv": str(out_dir / "clip_policy_per_label.csv"),
            "clip_window_distribution_csv": str(out_dir / "clip_window_distribution.csv"),
            "clip_level_details_csv": str(out_dir / "clip_level_details.csv"),
            "clip_policy_summary_md": str(out_dir / "clip_policy_summary.md"),
        },
        "notes": [
            "Clip-level window efficiency is computed by grouping segment predictions by parent_clip_id.",
            "Full-clip baseline uses all windows and final exit.",
            "Dynamic clip policy first applies depth-level label-set stability per window, then temporal label-set stability over windows.",
            "Event detection latency requires event onset metadata and is not applicable to persistent speaker labels.",
        ],
    }
    save_json(results, out_dir / "clip_policy_results.json")

    print("\nFull-clip baseline")
    print("-" * 90)
    print(full_df.to_string(index=False))

    print("\nDynamic clip/window policy")
    print("-" * 90)
    print(dyn_df.to_string(index=False))

    print("\nSaved outputs")
    print("-" * 90)
    for key, value in results["outputs"].items():
        print(f"{key}: {value}")
    print(f"clip_policy_results_json: {out_dir / 'clip_policy_results.json'}")
    print("-" * 90)


if __name__ == "__main__":
    main()
