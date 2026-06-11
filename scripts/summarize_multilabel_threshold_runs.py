# scripts/summarize_multilabel_threshold_runs.py
#
# Summarise multi-label threshold tuning runs.
#
# Input:
#   runs_multilabel/<run>/threshold_tuning/threshold_comparison.json
#
# Output:
#   runs_multilabel/summary_thresholds/
#     all_exit_metrics.csv
#     all_exit_metrics.md
#     final_exit_comparison.csv
#     final_exit_comparison.md
#     best_exit_comparison.csv
#     best_exit_comparison.md
#     final_exit_per_label.csv
#     final_exit_per_label.md
#     final_exit_thresholds.csv
#     final_exit_thresholds.md
#
# Example:
#   python scripts\summarize_multilabel_threshold_runs.py `
#     --run_dirs `
#       "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
#       "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
#     --names "3exit_nohint" "5exit_nohint" `
#     --out_dir "runs_multilabel\summary_thresholds"

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_KEYS = [
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "exact_match",
    "hamming_loss",
    "micro_precision",
    "micro_recall",
    "macro_precision",
    "macro_recall",
    "avg_true_labels",
    "avg_pred_labels",
]


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_model_name(path: str | Path) -> str:
    name = Path(path).name
    name = re.sub(r"_20\d{6}_\d{6}.*$", "", name)
    return name


def fmt_float(x, digits: int = 4):
    try:
        return round(float(x), digits)
    except Exception:
        return x


def df_to_markdown(df: pd.DataFrame) -> str:
    """
    Simple markdown table writer without requiring tabulate.
    """
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

    out = []
    out.append(make_row(headers))
    out.append("| " + " | ".join("-" * w for w in widths) + " |")

    for row in rows:
        out.append(make_row(row))

    return "\n".join(out) + "\n"


def write_table(df: pd.DataFrame, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(df_to_markdown(df))


def metric_row(
    *,
    model_name: str,
    run_dir: Path,
    num_exits: int,
    exit_no: int,
    split: str,
    threshold_mode: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model": model_name,
        "run_dir": str(run_dir),
        "num_exits": int(num_exits),
        "exit": int(exit_no),
        "split": split,
        "threshold_mode": threshold_mode,
    }

    for key in METRIC_KEYS:
        row[key] = fmt_float(metrics.get(key, 0.0))

    return row


def collect_one_run(run_dir: Path, model_name: str):
    tuning_path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    payload = load_json(tuning_path)

    labels = payload["labels"]
    num_exits = int(payload["num_exits"])

    all_metric_rows = []
    final_per_label_rows = []
    final_threshold_rows = []

    for exit_payload in payload["exits"]:
        exit_no = int(exit_payload["exit"])

        all_metric_rows.append(
            metric_row(
                model_name=model_name,
                run_dir=run_dir,
                num_exits=num_exits,
                exit_no=exit_no,
                split="val",
                threshold_mode="fixed_0p5",
                metrics=exit_payload["val_fixed_0p5"],
            )
        )
        all_metric_rows.append(
            metric_row(
                model_name=model_name,
                run_dir=run_dir,
                num_exits=num_exits,
                exit_no=exit_no,
                split="val",
                threshold_mode="tuned",
                metrics=exit_payload["val_tuned"],
            )
        )
        all_metric_rows.append(
            metric_row(
                model_name=model_name,
                run_dir=run_dir,
                num_exits=num_exits,
                exit_no=exit_no,
                split="test",
                threshold_mode="fixed_0p5",
                metrics=exit_payload["test_fixed_0p5"],
            )
        )
        all_metric_rows.append(
            metric_row(
                model_name=model_name,
                run_dir=run_dir,
                num_exits=num_exits,
                exit_no=exit_no,
                split="test",
                threshold_mode="tuned",
                metrics=exit_payload["test_tuned"],
            )
        )

    # Final exit details.
    final_exit_payload = payload["exits"][-1]
    final_exit_no = int(final_exit_payload["exit"])

    fixed_per_label = final_exit_payload["test_fixed_0p5"]["per_label"]
    tuned_per_label = final_exit_payload["test_tuned"]["per_label"]
    thresholds = final_exit_payload["tuned_thresholds"]

    for label in labels:
        fixed = fixed_per_label[label]
        tuned = tuned_per_label[label]

        final_per_label_rows.append(
            {
                "model": model_name,
                "run_dir": str(run_dir),
                "num_exits": num_exits,
                "final_exit": final_exit_no,
                "label": label,
                "threshold": fmt_float(thresholds[label]),
                "support": int(tuned.get("support", fixed.get("support", 0))),
                "fixed_precision": fmt_float(fixed.get("precision", 0.0)),
                "fixed_recall": fmt_float(fixed.get("recall", 0.0)),
                "fixed_f1": fmt_float(fixed.get("f1", 0.0)),
                "fixed_pred_pos": int(fixed.get("predicted_positive", 0)),
                "tuned_precision": fmt_float(tuned.get("precision", 0.0)),
                "tuned_recall": fmt_float(tuned.get("recall", 0.0)),
                "tuned_f1": fmt_float(tuned.get("f1", 0.0)),
                "tuned_pred_pos": int(tuned.get("predicted_positive", 0)),
                "delta_f1": fmt_float(tuned.get("f1", 0.0) - fixed.get("f1", 0.0)),
                "delta_recall": fmt_float(tuned.get("recall", 0.0) - fixed.get("recall", 0.0)),
                "delta_precision": fmt_float(tuned.get("precision", 0.0) - fixed.get("precision", 0.0)),
            }
        )

        final_threshold_rows.append(
            {
                "model": model_name,
                "num_exits": num_exits,
                "final_exit": final_exit_no,
                "label": label,
                "threshold": fmt_float(thresholds[label]),
            }
        )

    return {
        "labels": labels,
        "num_exits": num_exits,
        "all_metric_rows": all_metric_rows,
        "final_per_label_rows": final_per_label_rows,
        "final_threshold_rows": final_threshold_rows,
    }


def make_final_exit_comparison(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    test_df = all_metrics_df[all_metrics_df["split"] == "test"].copy()

    for model_name, group in test_df.groupby("model"):
        max_exit = int(group["exit"].max())

        final_rows = group[group["exit"] == max_exit].copy()
        rows.append(final_rows)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    cols = [
        "model",
        "num_exits",
        "exit",
        "threshold_mode",
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "exact_match",
        "hamming_loss",
        "avg_true_labels",
        "avg_pred_labels",
    ]

    return out[cols].sort_values(["model", "threshold_mode"]).reset_index(drop=True)


def make_best_exit_comparison(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Best test exit by macro-F1 under tuned thresholds.
    """
    rows = []

    tuned_test = all_metrics_df[
        (all_metrics_df["split"] == "test")
        & (all_metrics_df["threshold_mode"] == "tuned")
    ].copy()

    for model_name, group in tuned_test.groupby("model"):
        best_idx = group["macro_f1"].astype(float).idxmax()
        rows.append(group.loc[best_idx])

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    cols = [
        "model",
        "num_exits",
        "exit",
        "threshold_mode",
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "exact_match",
        "hamming_loss",
        "avg_true_labels",
        "avg_pred_labels",
    ]

    return out[cols].sort_values("macro_f1", ascending=False).reset_index(drop=True)


def make_readme_summary(
    final_df: pd.DataFrame,
    best_df: pd.DataFrame,
    per_label_df: pd.DataFrame,
    out_path: Path,
):
    lines = []

    lines.append("# Multi-label threshold tuning summary\n")
    lines.append("## Final-exit comparison\n")
    lines.append(df_to_markdown(final_df))
    lines.append("\n## Best tuned exit per model\n")
    lines.append(df_to_markdown(best_df))

    if not per_label_df.empty:
        lines.append("\n## Final-exit per-label F1 comparison\n")

        compact = per_label_df[
            [
                "model",
                "label",
                "threshold",
                "support",
                "fixed_f1",
                "tuned_f1",
                "delta_f1",
                "fixed_recall",
                "tuned_recall",
                "fixed_precision",
                "tuned_precision",
            ]
        ].copy()

        compact = compact.sort_values(["model", "delta_f1"], ascending=[True, False])
        lines.append(df_to_markdown(compact))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Summarise multi-label threshold tuning runs into CSV/Markdown tables."
    )

    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories to summarise.",
    )

    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names matching run_dirs.",
    )

    parser.add_argument(
        "--out_dir",
        default="runs_multilabel/summary_thresholds",
        help="Output directory for summary tables.",
    )

    args = parser.parse_args()

    run_dirs = [Path(p).resolve() for p in args.run_dirs]

    if args.names:
        if len(args.names) != len(run_dirs):
            raise ValueError("--names must have the same length as --run_dirs")
        names = [str(x) for x in args.names]
    else:
        names = [safe_model_name(p) for p in run_dirs]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nSummarising multi-label threshold runs")
    print("-" * 90)
    print(f"Output dir: {out_dir}")
    for name, run_dir in zip(names, run_dirs):
        print(f"  {name}: {run_dir}")
    print("-" * 90)

    all_metric_rows = []
    final_per_label_rows = []
    final_threshold_rows = []

    for name, run_dir in zip(names, run_dirs):
        result = collect_one_run(run_dir=run_dir, model_name=name)
        all_metric_rows.extend(result["all_metric_rows"])
        final_per_label_rows.extend(result["final_per_label_rows"])
        final_threshold_rows.extend(result["final_threshold_rows"])

    all_metrics_df = pd.DataFrame(all_metric_rows)
    final_per_label_df = pd.DataFrame(final_per_label_rows)
    thresholds_df = pd.DataFrame(final_threshold_rows)

    final_exit_df = make_final_exit_comparison(all_metrics_df)
    best_exit_df = make_best_exit_comparison(all_metrics_df)

    # Write all tables.
    write_table(
        all_metrics_df,
        out_dir / "all_exit_metrics.csv",
        out_dir / "all_exit_metrics.md",
    )

    write_table(
        final_exit_df,
        out_dir / "final_exit_comparison.csv",
        out_dir / "final_exit_comparison.md",
    )

    write_table(
        best_exit_df,
        out_dir / "best_exit_comparison.csv",
        out_dir / "best_exit_comparison.md",
    )

    write_table(
        final_per_label_df,
        out_dir / "final_exit_per_label.csv",
        out_dir / "final_exit_per_label.md",
    )

    write_table(
        thresholds_df,
        out_dir / "final_exit_thresholds.csv",
        out_dir / "final_exit_thresholds.md",
    )

    make_readme_summary(
        final_df=final_exit_df,
        best_df=best_exit_df,
        per_label_df=final_per_label_df,
        out_path=out_dir / "README_TABLES.md",
    )

    print("\nSaved summary tables:")
    print(f"  {out_dir / 'all_exit_metrics.csv'}")
    print(f"  {out_dir / 'all_exit_metrics.md'}")
    print(f"  {out_dir / 'final_exit_comparison.csv'}")
    print(f"  {out_dir / 'final_exit_comparison.md'}")
    print(f"  {out_dir / 'best_exit_comparison.csv'}")
    print(f"  {out_dir / 'best_exit_comparison.md'}")
    print(f"  {out_dir / 'final_exit_per_label.csv'}")
    print(f"  {out_dir / 'final_exit_per_label.md'}")
    print(f"  {out_dir / 'final_exit_thresholds.csv'}")
    print(f"  {out_dir / 'final_exit_thresholds.md'}")
    print(f"  {out_dir / 'README_TABLES.md'}")

    print("\nFinal-exit comparison:")
    print(final_exit_df.to_string(index=False))

    print("\nBest tuned exit per model:")
    print(best_exit_df.to_string(index=False))

    print("-" * 90)


if __name__ == "__main__":
    main()