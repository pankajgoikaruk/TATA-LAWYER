# scripts/evaluate_label_aware_parent_aggregation.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score


DEFAULT_MAX_LABELS = [
    "audience_reaction_present",
    "silence_present",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o: Any):
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


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def load_labels(labels_json: Path | None, df: pd.DataFrame, exit_idx: int) -> list[str]:
    if labels_json is not None:
        payload = load_json(labels_json)
        if isinstance(payload, list):
            labels = [str(x) for x in payload]
        elif isinstance(payload, dict) and "labels" in payload:
            labels = [str(x) for x in payload["labels"]]
        else:
            raise RuntimeError(
                f"Could not read labels from {labels_json}. Expected a JSON list or a dict with key 'labels'."
            )
        return labels

    prefix = f"exit{exit_idx}_prob_"
    labels = [col[len(prefix):] for col in df.columns if col.startswith(prefix)]
    if not labels:
        raise RuntimeError(
            f"Could not infer labels. No columns found with prefix '{prefix}'. "
            "Pass --labels_json or check --exit_idx."
        )
    return labels


def threshold_slug(value: float) -> str:
    if abs(value - 0.5) < 1e-12:
        return "fixed_0p5"
    return "fixed_" + str(value).replace(".", "p").replace("-", "m")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "per_label": {},
    }

    for i, lab in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        result["per_label"][lab] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }
    return result


def validate_inputs(df: pd.DataFrame, labels: list[str], exit_idx: int) -> None:
    if "parent_clip_id" not in df.columns:
        raise RuntimeError("Input segment-probability CSV missing required column: parent_clip_id")

    missing_true = [lab for lab in labels if lab not in df.columns]
    if missing_true:
        raise RuntimeError(f"Input CSV missing true-label columns: {missing_true}")

    missing_probs = [f"exit{exit_idx}_prob_{lab}" for lab in labels if f"exit{exit_idx}_prob_{lab}" not in df.columns]
    if missing_probs:
        raise RuntimeError(f"Input CSV missing probability columns: {missing_probs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc parent-level label-aware aggregation from segment probability CSV. "
            "Use mean for stable labels and max for transient/bursty labels without retraining."
        )
    )
    parser.add_argument(
        "--segment_probs_csv",
        required=True,
        help="CSV produced by evaluate_tata_final_holdout_parent_level.py, e.g. parent_eval_segment_probs_fixed_0p5_mean.csv or *_max.csv.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for label-aware evaluation files.")
    parser.add_argument("--labels_json", default=None, help="Optional labels JSON. If omitted, labels are inferred from exit probability columns.")
    parser.add_argument("--exit_idx", type=int, default=3, help="Exit index to evaluate, default 3.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold, default 0.5.")
    parser.add_argument(
        "--max_labels",
        default=",".join(DEFAULT_MAX_LABELS),
        help="Comma-separated labels to aggregate with max. Default: audience_reaction_present,silence_present.",
    )
    parser.add_argument(
        "--mean_labels",
        default=None,
        help="Optional comma-separated labels to aggregate with mean. If omitted, all non-max labels use mean.",
    )
    parser.add_argument("--model_name", default=None, help="Optional model/run name for summary CSV.")
    parser.add_argument("--split", default="final_raw_holdout_parent_level", help="Split name for summary CSV.")
    args = parser.parse_args()

    segment_probs_csv = Path(args.segment_probs_csv)
    out_dir = Path(args.out_dir)
    labels_json = Path(args.labels_json) if args.labels_json else None
    exit_idx = int(args.exit_idx)
    threshold = float(args.threshold)
    max_labels = parse_csv_list(args.max_labels)

    if not segment_probs_csv.exists():
        raise FileNotFoundError(f"Segment probability CSV not found: {segment_probs_csv}")

    df = pd.read_csv(segment_probs_csv, low_memory=False)
    labels = load_labels(labels_json, df, exit_idx)
    validate_inputs(df, labels, exit_idx)

    unknown_max = sorted(set(max_labels) - set(labels))
    if unknown_max:
        raise RuntimeError(f"--max_labels contains labels not present in the label schema: {unknown_max}")

    if args.mean_labels:
        mean_labels = parse_csv_list(args.mean_labels)
        unknown_mean = sorted(set(mean_labels) - set(labels))
        if unknown_mean:
            raise RuntimeError(f"--mean_labels contains labels not present in the label schema: {unknown_mean}")
    else:
        mean_labels = [lab for lab in labels if lab not in set(max_labels)]

    overlap = sorted(set(mean_labels).intersection(max_labels))
    if overlap:
        raise RuntimeError(f"Labels cannot be both mean and max aggregated: {overlap}")

    missing_strategy = sorted(set(labels) - set(mean_labels) - set(max_labels))
    if missing_strategy:
        raise RuntimeError(f"No aggregation strategy assigned for labels: {missing_strategy}")

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_cols = [c for c in ["parent_clip_id", "source_file", "source_path"] if c in df.columns]
    parent_meta = df.groupby("parent_clip_id", as_index=False).first()[metadata_cols]

    parent_rows = []
    for parent_clip_id, g in df.groupby("parent_clip_id", sort=False):
        row: dict[str, Any] = {"parent_clip_id": parent_clip_id}
        if "source_file" in g.columns:
            row["source_file"] = g["source_file"].iloc[0]
        if "source_path" in g.columns:
            row["source_path"] = g["source_path"].iloc[0]

        for lab in labels:
            true_values = pd.to_numeric(g[lab], errors="coerce").fillna(0).astype(int)
            row[lab] = int(true_values.max())

            prob_col = f"exit{exit_idx}_prob_{lab}"
            probs = pd.to_numeric(g[prob_col], errors="coerce").astype(float)
            if lab in max_labels:
                row[f"prob_{lab}"] = float(probs.max())
                row[f"aggregation_{lab}"] = "max"
            else:
                row[f"prob_{lab}"] = float(probs.mean())
                row[f"aggregation_{lab}"] = "mean"

        parent_rows.append(row)

    parent_df = pd.DataFrame(parent_rows)
    # Preserve parent order from the first occurrence in the segment CSV.
    if metadata_cols:
        parent_order = parent_meta[["parent_clip_id"]].copy()
        parent_df = parent_order.merge(parent_df, on="parent_clip_id", how="left", suffixes=("", "_drop"))
        drop_cols = [c for c in parent_df.columns if c.endswith("_drop")]
        if drop_cols:
            parent_df = parent_df.drop(columns=drop_cols)

    y_true = parent_df[labels].astype(int).values
    prob_cols = [f"prob_{lab}" for lab in labels]
    y_prob = parent_df[prob_cols].astype(float).values
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, labels)
    for lab in labels:
        metrics["per_label"][lab]["aggregation"] = "max" if lab in max_labels else "mean"

    threshold_name = threshold_slug(threshold)
    model_name = args.model_name or segment_probs_csv.parent.parent.name or segment_probs_csv.parent.name

    summary_row = {
        "model": model_name,
        "threshold_mode": threshold_name,
        "aggregation": "label_aware_mean_for_8_max_for_2",
        "split": args.split,
        "exit": exit_idx,
        "parent_clips": int(len(parent_df)),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "samples_f1": metrics["samples_f1"],
        "exact_match": metrics["exact_match"],
        "hamming_loss": metrics["hamming_loss"],
        "jaccard_score": metrics["jaccard_score"],
        "avg_true_labels": metrics["avg_true_labels"],
        "avg_pred_labels": metrics["avg_pred_labels"],
    }

    per_label_rows = []
    for lab in labels:
        per_label_rows.append({"label": lab, **metrics["per_label"][lab]})

    parent_probs_path = out_dir / f"parent_label_aware_exit{exit_idx}_probabilities.csv"
    static_csv = out_dir / f"parent_holdout_static_{threshold_name}_label_aware.csv"
    per_label_csv = out_dir / f"parent_holdout_per_label_{threshold_name}_label_aware_exit{exit_idx}.csv"
    out_json = out_dir / f"parent_holdout_eval_{threshold_name}_label_aware.json"

    parent_df.to_csv(parent_probs_path, index=False)
    pd.DataFrame([summary_row]).to_csv(static_csv, index=False)
    pd.DataFrame(per_label_rows).to_csv(per_label_csv, index=False)

    result = {
        "segment_probs_csv": str(segment_probs_csv),
        "threshold": threshold,
        "threshold_mode": threshold_name,
        "exit": exit_idx,
        "labels": labels,
        "mean_labels": mean_labels,
        "max_labels": max_labels,
        "parent_clips": int(len(parent_df)),
        "segment_rows": int(len(df)),
        "metrics": metrics,
        "summary_row": summary_row,
        "outputs": {
            "parent_probabilities_csv": str(parent_probs_path),
            "static_csv": str(static_csv),
            "per_label_csv": str(per_label_csv),
        },
    }
    save_json(result, out_json)

    print("\nLabel-aware parent aggregation complete")
    print("-" * 90)
    print(f"Segment probabilities: {segment_probs_csv}")
    print(f"Parent clips:          {len(parent_df)}")
    print(f"Segments:              {len(df)}")
    print(f"Exit:                  {exit_idx}")
    print(f"Threshold:             {threshold}")
    print(f"Mean labels:           {', '.join(mean_labels)}")
    print(f"Max labels:            {', '.join(max_labels)}")
    print("")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print("\nPer-label metrics:")
    print(pd.DataFrame(per_label_rows).to_string(index=False))
    print(f"\nOutput JSON: {out_json}")


if __name__ == "__main__":
    main()
