import argparse
from pathlib import Path

import pandas as pd


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table summarising averaged performance per variant (+device).

    Expected index: MultiIndex (variant, device) or single index variant.
    Expected columns (after aggregation):
      - n_runs
      - policy_acc_mean
      - compute_saving_pct_mean
      - exit_e1_mean
      - exit_e2_mean
      - exit_e3_mean
      - expected_mflops_mean
      - full_mflops_mean
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Averaged policy accuracy and compute saving across ASHADIP variants.}")
    lines.append(r"  \label{tab:ashadip_variants_avg_summary}")
    lines.append(r"  \begin{tabular}{llrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Variant & Device & Runs & Acc$_{\text{policy}}$ (\%) & Save (\%) & "
        r"Exit1 (\%) & Exit2 (\%) & Exit3 (\%) & Exp.~MFLOPs \\"
    )
    lines.append(r"    \midrule")

    # If df has a MultiIndex, reset it to explicit columns
    if isinstance(df.index, pd.MultiIndex):
        df_iter = df.reset_index()
    else:
        df_iter = df.reset_index()

    for _, row in df_iter.iterrows():
        variant = row.get("variant", "")
        device  = row.get("device", "")

        n_runs = int(row.get("n_runs", 1))

        def f_pct01(x):
            try:
                return f"{float(x) * 100.0:.1f}"
            except Exception:
                return "--"

        def f_pct(x):
            try:
                return f"{float(x):.1f}"
            except Exception:
                return "--"

        def f_float1(x):
            try:
                return f"{float(x):.1f}"
            except Exception:
                return "--"

        acc   = f_pct01(row.get("policy_acc_mean"))
        save  = f_pct(row.get("compute_saving_pct_mean"))
        e1    = f_pct01(row.get("exit_e1_mean"))
        e2    = f_pct01(row.get("exit_e2_mean"))
        e3    = f_pct01(row.get("exit_e3_mean"))
        exp_f = f_float1(row.get("expected_mflops_mean"))

        lines.append(
            rf"    {variant} & {device} & {n_runs} & {acc} & {save} & {e1} & {e2} & {e3} & {exp_f} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default="analysis/all_runs_summary.csv",
        help="Path to all_runs_summary.csv produced by compare_variants.py",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/variants_avg_summary_table.tex",
        help="Output LaTeX file for the averaged variants summary table.",
    )
    ap.add_argument(
        "--out_csv",
        default="analysis/variants_avg_summary_table.csv",
        help="Output CSV with averaged metrics per variant/device.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    # Harmonise policy accuracy column name
    if "policy_test_acc" not in df.columns and "test_acc_policy" in df.columns:
        df["policy_test_acc"] = df["test_acc_policy"]

    required = {
        "variant",
        "device",
        "policy_test_acc",
        "compute_saving_pct",
        "exit_e1",
        "exit_e2",
        "exit_e3",
        "expected_mflops",
        "full_mflops",
        "run_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {summary_path}: {missing}")

    grouped = df.groupby(["variant", "device"])

    agg_df = grouped.agg(
        n_runs=("run_id", "count"),
        policy_acc_mean=("policy_test_acc", "mean"),
        compute_saving_pct_mean=("compute_saving_pct", "mean"),
        exit_e1_mean=("exit_e1", "mean"),
        exit_e2_mean=("exit_e2", "mean"),
        exit_e3_mean=("exit_e3", "mean"),
        expected_mflops_mean=("expected_mflops", "mean"),
        full_mflops_mean=("full_mflops", "mean"),
    )

    # Save CSV
    agg_df_reset = agg_df.reset_index()
    agg_df_reset.to_csv(out_csv_path, index=False)
    print(f"[variants_avg_to_latex] Wrote averaged variants CSV to {out_csv_path}")

    # Save LaTeX
    table_str = make_latex_table(agg_df)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"[variants_avg_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()





# import argparse
# from pathlib import Path
#
# import pandas as pd
#
#
# def make_latex_table(df: pd.DataFrame) -> str:
#     """
#     Build a LaTeX table summarising the AVERAGE performance per variant.
#
#     Input df is expected to be the output of groupby('variant').agg(...),
#     with columns such as:
#       - n_runs
#       - policy_test_acc_mean, policy_test_acc_std
#       - compute_saving_pct_mean
#       - exit_e1_mean, exit_e2_mean, exit_e3_mean
#       - expected_mflops_mean, full_mflops_mean
#     """
#
#     lines = []
#     lines.append(r"\begin{table}[ht]")
#     lines.append(r"  \centering")
#     lines.append(r"  \caption{Average early-exit performance per ASHADIP variant.}")
#     lines.append(r"  \label{tab:ashadip_variants_avg}")
#     lines.append(r"  \begin{tabular}{lrrrrrr}")
#     lines.append(r"    \toprule")
#     lines.append(
#         r"    Variant & Runs & Acc$_{\text{policy}}$ (\%) & Save (\%) & Exit1 (\%) & Exit2 (\%) & Exit3 (\%) \\"
#     )
#     lines.append(r"    \midrule")
#
#     for _, row in df.iterrows():
#         variant = row.name
#         n_runs = int(row.get("n_runs", 1))
#
#         def fmt_pct_frac(x):
#             if pd.isna(x):
#                 return "--"
#             try:
#                 return f"{float(x) * 100:.1f}"
#             except Exception:
#                 return "--"
#
#         def fmt_pct(x):
#             if pd.isna(x):
#                 return "--"
#             try:
#                 return f"{float(x):.1f}"
#             except Exception:
#                 return "--"
#
#         acc_mean = fmt_pct_frac(row.get("policy_test_acc_mean"))
#         save_mean = fmt_pct(row.get("compute_saving_pct_mean"))
#         e1_mean = fmt_pct_frac(row.get("exit_e1_mean"))
#         e2_mean = fmt_pct_frac(row.get("exit_e2_mean"))
#         e3_mean = fmt_pct_frac(row.get("exit_e3_mean"))
#
#         lines.append(
#             rf"    {variant} & {n_runs} & {acc_mean} & {save_mean} & {e1_mean} & {e2_mean} & {e3_mean} \\"
#         )
#
#     lines.append(r"    \bottomrule")
#     lines.append(r"  \end{tabular}")
#     lines.append(r"\end{table}")
#
#     return "\n".join(lines)
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--summary_csv",
#         default="analysis/all_runs_summary.csv",
#         help="Per-run summary CSV produced by compare_variants.py",
#     )
#     ap.add_argument(
#         "--out_tex",
#         default="analysis/tables/variants_avg_summary_table.tex",
#         help="Output LaTeX table (average per variant).",
#     )
#     args = ap.parse_args()
#
#     summary_path = Path(args.summary_csv)
#     if not summary_path.exists():
#         raise SystemExit(f"Summary CSV not found: {summary_path}")
#
#     out_tex_path = Path(args.out_tex)
#     out_tex_path.parent.mkdir(parents=True, exist_ok=True)
#
#     df_runs = pd.read_csv(summary_path)
#
#     if df_runs.empty:
#         raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")
#
#     if "variant" not in df_runs.columns:
#         raise SystemExit(
#             "Column 'variant' not found in all_runs_summary.csv. "
#             "Make sure compare_variants.py writes a 'variant' column."
#         )
#
#     # Group by variant and compute averages
#     grouped = df_runs.groupby("variant")
#
#     agg_df = grouped.agg(
#         n_runs=("run_id", "count"),
#         policy_test_acc_mean=("policy_test_acc", "mean"),
#         compute_saving_pct_mean=("compute_saving_pct", "mean"),
#         exit_e1_mean=("exit_e1", "mean"),
#         exit_e2_mean=("exit_e2", "mean"),
#         exit_e3_mean=("exit_e3", "mean"),
#         expected_mflops_mean=("expected_mflops", "mean"),
#         full_mflops_mean=("full_mflops", "mean"),
#     )
#
#     # (Optional) you could also compute std devs and add to the table later.
#
#     table_str = make_latex_table(agg_df)
#
#     with open(out_tex_path, "w", encoding="utf-8") as f:
#         f.write(table_str + "\n")
#
#     print(f"[variants_avg_to_latex] Wrote LaTeX table to {out_tex_path}")
#
#
# if __name__ == "__main__":
#     main()
