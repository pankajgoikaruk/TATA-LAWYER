# scripts/variants_to_latex.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_json_loads(x, default=None):
    if default is None:
        default = {}
    if pd.isna(x):
        return default
    if isinstance(x, dict):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return default


def _extract_exit_mix(row: pd.Series) -> dict:
    """
    Prefer exit_mix_json if present, else fall back to flat exit_eK columns.
    """
    exit_mix = {}

    if "exit_mix_json" in row and pd.notna(row["exit_mix_json"]):
        exit_mix = _safe_json_loads(row["exit_mix_json"], default={})

    if not isinstance(exit_mix, dict) or len(exit_mix) == 0:
        for col in row.index:
            if str(col).startswith("exit_e"):
                try:
                    k = str(col).replace("exit_", "")   # exit_e1 -> e1
                    exit_mix[k] = float(row[col])
                except Exception:
                    pass

    out = {}
    for k, v in exit_mix.items():
        ks = str(k)
        if ks.startswith("e"):
            try:
                idx = int(ks[1:])
                out[f"e{idx}"] = float(v)
            except Exception:
                pass

    return out


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table summarising multiple runs/variants.
    Dynamic K-exit safe.
    """
    sort_cols = []
    if "variant" in df.columns:
        sort_cols.append("variant")
    if "policy_test_acc" in df.columns:
        sort_cols.append("policy_test_acc")

    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df = df.sort_values(sort_cols, ascending=ascending)

    # Determine max K across all rows
    exit_mixes = [_extract_exit_mix(row) for _, row in df.iterrows()]
    max_k = 0
    for em in exit_mixes:
        if em:
            max_k = max(max_k, max(int(k[1:]) for k in em.keys()))

    if max_k == 0:
        max_k = 3  # safe fallback

    exit_headers = " & ".join([rf"Exit{k}~(\%)" for k in range(1, max_k + 1)])
    tab_cols = "llrrrr" + ("r" * max_k) + "rr"

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Policy accuracy, compute saving, and exit behaviour across ASHADIP variants.}"
    )
    lines.append(r"  \label{tab:ashadip_variants_summary}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(rf"  \begin{{tabular}}{{{tab_cols}}}")
    lines.append(r"    \toprule")
    lines.append(
        rf"    Variant & Run ID & K & $\tau$ & Acc$_{{\text{{policy}}}}$ & Save~(\%) & Avg depth & "
        rf"{exit_headers} & Exp.~MFLOPs & ECE$_{{\text{{policy}}}}$ \\"
    )
    lines.append(r"    \midrule")

    def get(row, key, default="--"):
        return row[key] if key in row and pd.notna(row[key]) else default

    def fmt_pct_frac(x):
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x) * 100.0:.1f}"
        except Exception:
            return "--"

    def fmt_pct(x):
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "--"

    def fmt_float(x, ndigits=1):
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "--"

    def fmt_tau(x):
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "--"

    def fmt_ece(x):
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "--"

    for _, row in df.iterrows():
        variant = get(row, "variant")
        run_id = get(row, "run_id")

        num_exits = get(row, "num_exits", None)
        tau = get(row, "tau", None)
        acc = get(row, "policy_test_acc", None)
        save = get(row, "compute_saving_pct", None)
        avg_depth = get(row, "avg_exit_depth", None)
        exp_flops = get(row, "expected_mflops", None)
        ece_pol = get(row, "ece_policy", None)

        exit_mix = _extract_exit_mix(row)

        # Compute avg depth if missing
        if avg_depth in (None, "--"):
            try:
                avg_depth = sum(int(k[1:]) * float(v) for k, v in exit_mix.items())
            except Exception:
                avg_depth = None

        tau_str = fmt_tau(tau)
        acc_str = fmt_pct_frac(acc)
        save_str = fmt_pct(save)
        avg_depth_str = fmt_float(avg_depth, ndigits=2)
        exp_str = fmt_float(exp_flops, ndigits=1)
        ece_str = fmt_ece(ece_pol)

        try:
            k_str = str(int(num_exits))
        except Exception:
            k_str = "--"

        exit_vals = []
        for k in range(1, max_k + 1):
            exit_vals.append(fmt_pct_frac(exit_mix.get(f"e{k}", None)))
        exit_vals_str = " & ".join(exit_vals)

        lines.append(
            rf"    {variant} & {run_id} & {k_str} & {tau_str} & {acc_str} & {save_str} & {avg_depth_str} & "
            rf"{exit_vals_str} & {exp_str} & {ece_str} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table*}")

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
        help="Output LaTeX file for the variants summary table.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    # Save sorted CSV backup next to the .tex
    table_df = df.copy()
    sort_cols = []
    if "variant" in table_df.columns:
        sort_cols.append("variant")
    if "policy_test_acc" in table_df.columns:
        sort_cols.append("policy_test_acc")
    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1)
        table_df = table_df.sort_values(sort_cols, ascending=ascending)

    out_csv_path = out_tex_path.with_suffix(".csv")
    table_df.to_csv(out_csv_path, index=False)
    print(f"[variants_to_latex] Wrote CSV backup to {out_csv_path}")

    table_str = make_latex_table(table_df)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print(f"[variants_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()