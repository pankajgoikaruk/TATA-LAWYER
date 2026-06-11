import argparse
from pathlib import Path

import pandas as pd


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table showing the effect of window size on performance.

    Expected columns in df after aggregation:
      - segment_sec
      - n_runs
      - policy_acc_mean
      - avg_exit_depth_mean
      - compute_saving_pct_mean
      - lat_exit1_ms_mean
      - lat_exit2_ms_mean
      - lat_exit3_ms_mean
      - pipeline_minutes_mean
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Effect of window size on ASHADIP performance (V0 baseline).}")
    lines.append(r"  \label{tab:window_size_performance}")
    lines.append(r"  \begin{tabular}{rrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Window (s) & Runs & Acc$_{\text{policy}}$ (\%) & Avg exit depth & "
        r"Save (\%) & Exit1 (ms) & Exit2 (ms) & Exit3 (ms) & Pipeline (min) \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        seg = float(row["segment_sec"])
        # Nice formatting for window size (1, 10, 20, 30)
        if abs(seg - round(seg)) < 1e-6:
            window_str = f"{int(round(seg))}"
        else:
            window_str = f"{seg:.2f}"

        n_runs = int(row.get("n_runs", 1))

        def f_pct01(x):
            # convert fraction [0,1] -> percent
            try:
                return f"{float(x) * 100.0:.1f}"
            except Exception:
                return "--"

        def f_pct(x):
            # already a percentage
            try:
                return f"{float(x):.1f}"
            except Exception:
                return "--"

        def f_float2(x):
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "--"

        def f_float3(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return "--"

        acc    = f_pct01(row.get("policy_acc_mean"))
        depth  = f_float3(row.get("avg_exit_depth_mean"))
        save   = f_pct(row.get("compute_saving_pct_mean"))
        e1_ms  = f_float2(row.get("lat_exit1_ms_mean"))
        e2_ms  = f_float2(row.get("lat_exit2_ms_mean"))
        e3_ms  = f_float2(row.get("lat_exit3_ms_mean"))
        pipe_m = f_float2(row.get("pipeline_minutes_mean"))

        lines.append(
            rf"    {window_str} & {n_runs} & {acc} & {depth} & {save} & "
            rf"{e1_ms} & {e2_ms} & {e3_ms} & {pipe_m} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all_runs_csv",
        default="analysis/all_runs_summary.csv",
        help="CSV produced by compare_variants.py (per run, per variant).",
    )
    ap.add_argument(
        "--ondevice_csv",
        default="analysis/on_device_summary.csv",
        help="CSV produced by profile_latency.py (per run).",
    )
    ap.add_argument(
        "--pipeline_csv",
        default="analysis/pipeline_runtime.csv",
        help="CSV with end-to-end pipeline runtime per run (from run_full.ps1).",
    )
    ap.add_argument(
        "--out_csv",
        default="analysis/window_size_summary.csv",
        help="Output CSV summarising performance per window size.",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/window_size_vs_performance_table.tex",
        help="Output LaTeX table for window size vs performance.",
    )
    args = ap.parse_args()

    all_runs_path = Path(args.all_runs_csv)
    ondev_path    = Path(args.ondevice_csv)
    pipe_path     = Path(args.pipeline_csv)

    if not all_runs_path.exists():
        raise SystemExit(f"all_runs_summary.csv not found: {all_runs_path}")
    if not ondev_path.exists():
        raise SystemExit(f"on_device_summary.csv not found: {ondev_path}")
    if not pipe_path.exists():
        raise SystemExit(f"pipeline_runtime.csv not found: {pipe_path}")

    out_csv_path = Path(args.out_csv)
    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(all_runs_path)
    df_on  = pd.read_csv(ondev_path)
    df_pipe = pd.read_csv(pipe_path)

    if df_all.empty or df_on.empty or df_pipe.empty:
        raise SystemExit("One of the input CSVs is empty; nothing to summarise.")

    # Harmonise policy accuracy column name
    if "policy_test_acc" not in df_all.columns and "test_acc_policy" in df_all.columns:
        df_all["policy_test_acc"] = df_all["test_acc_policy"]

    # We need at least these columns:
    needed_all = {"run_id", "policy_test_acc", "exit_e1", "exit_e2", "exit_e3", "compute_saving_pct"}
    missing_all = needed_all - set(df_all.columns)
    if missing_all:
        raise SystemExit(f"Missing columns in {all_runs_path}: {missing_all}")

    needed_on = {"run_id", "lat_exit1_ms", "lat_exit2_ms", "lat_exit3_ms"}
    missing_on = needed_on - set(df_on.columns)
    if missing_on:
        raise SystemExit(f"Missing columns in {ondev_path}: {missing_on}")

    needed_pipe = {"run_id", "segment_sec", "total_minutes"}
    missing_pipe = needed_pipe - set(df_pipe.columns)
    if missing_pipe:
        raise SystemExit(f"Missing columns in {pipe_path}: {missing_pipe}")

    # Join on run_id (each run_id is unique across logs)
    df_merged = (
        df_all.merge(df_on[["run_id", "lat_exit1_ms", "lat_exit2_ms", "lat_exit3_ms"]],
                     on="run_id", how="inner")
              .merge(df_pipe[["run_id", "segment_sec", "total_minutes"]],
                     on="run_id", how="inner")
    )

    if df_merged.empty:
        raise SystemExit("No overlapping runs across all_runs_summary, on_device_summary, and pipeline_runtime.")

    # Compute avg exit depth = 1*p1 + 2*p2 + 3*p3
    df_merged["avg_exit_depth"] = (
        1.0 * df_merged["exit_e1"] +
        2.0 * df_merged["exit_e2"] +
        3.0 * df_merged["exit_e3"]
    )

    # Group by window size (segment_sec)
    grouped = df_merged.groupby("segment_sec", as_index=False).agg(
        n_runs=("run_id", "count"),
        policy_acc_mean=("policy_test_acc", "mean"),
        avg_exit_depth_mean=("avg_exit_depth", "mean"),
        compute_saving_pct_mean=("compute_saving_pct", "mean"),
        lat_exit1_ms_mean=("lat_exit1_ms", "mean"),
        lat_exit2_ms_mean=("lat_exit2_ms", "mean"),
        lat_exit3_ms_mean=("lat_exit3_ms", "mean"),
        pipeline_minutes_mean=("total_minutes", "mean"),
    )

    # Sort by window size
    grouped = grouped.sort_values("segment_sec")

    # Save CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_csv_path, index=False)
    print(f"[window_size_summary] Wrote window-size summary CSV to {out_csv_path}")

    # Save LaTeX table
    table_str = make_latex_table(grouped)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"[window_size_summary] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()
