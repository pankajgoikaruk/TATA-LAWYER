# scripts/ondevice_to_latex.py

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


def _normalize_ondevice_rows(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Support both:
    1) old legacy CSV:
         lat_exit1_ms, lat_exit2_ms, lat_exit3_ms
    2) new K-exit CSV:
         num_exits, tap_blocks_json, latency_ms_json, full_forward_latency_ms
    """
    rows = []

    is_new = "latency_ms_json" in df_runs.columns
    is_old = {"lat_exit1_ms", "lat_exit2_ms", "lat_exit3_ms"}.issubset(df_runs.columns)

    if not is_new and not is_old:
        raise SystemExit(
            "CSV is missing both the new K-exit fields and the old 3-exit latency fields."
        )

    for _, r in df_runs.iterrows():
        variant = r.get("variant", "--")
        run_id = r.get("run_id", "--")
        device = r.get("device", "--")
        compute_saving_pct = r.get("compute_saving_pct", None)

        if is_new:
            latency_ms = _safe_json_loads(r.get("latency_ms_json"), default={})
            tap_blocks = _safe_json_loads(r.get("tap_blocks_json"), default=None)
            num_exits = r.get("num_exits", None)
            full_forward_latency_ms = r.get("full_forward_latency_ms", None)

            try:
                num_exits = int(num_exits) if pd.notna(num_exits) else None
            except Exception:
                num_exits = None

            if num_exits is None and isinstance(latency_ms, dict):
                keys = [
                    int(str(k).replace("exit", ""))
                    for k in latency_ms.keys()
                    if str(k).startswith("exit")
                ]
                num_exits = max(keys) if keys else None

            if full_forward_latency_ms is None and num_exits is not None:
                full_forward_latency_ms = latency_ms.get(f"exit{num_exits}", None)

            rows.append({
                "variant": variant,
                "run_id": run_id,
                "device": device,
                "num_exits": num_exits,
                "tap_blocks_json": json.dumps(tap_blocks) if tap_blocks is not None else "",
                "latency_ms": latency_ms,
                "full_forward_latency_ms": full_forward_latency_ms,
                "compute_saving_pct": compute_saving_pct,
            })

        else:
            latency_ms = {
                "exit1": r.get("lat_exit1_ms", None),
                "exit2": r.get("lat_exit2_ms", None),
                "exit3": r.get("lat_exit3_ms", None),
            }
            rows.append({
                "variant": variant,
                "run_id": run_id,
                "device": device,
                "num_exits": 3,
                "tap_blocks_json": json.dumps([1, 3]),
                "latency_ms": latency_ms,
                "full_forward_latency_ms": r.get("lat_exit3_ms", None),
                "compute_saving_pct": compute_saving_pct,
            })

    return pd.DataFrame(rows)


def _aggregate_latency(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Group by variant + device + K + tap blocks.
    Compute mean latency for each exit key present.
    """
    if df_norm.empty:
        raise SystemExit("No rows after normalization; nothing to summarize.")

    max_k = int(df_norm["num_exits"].dropna().max())

    def build_group_row(group: pd.DataFrame):
        out = {
            "n_runs": int(len(group)),
            "full_forward_latency_ms_mean": pd.to_numeric(
                group["full_forward_latency_ms"], errors="coerce"
            ).mean(),
            "compute_saving_pct_mean": pd.to_numeric(
                group["compute_saving_pct"], errors="coerce"
            ).mean(),
        }

        for k in range(1, max_k + 1):
            vals = []
            for lat_dict in group["latency_ms"]:
                if isinstance(lat_dict, dict) and f"exit{k}" in lat_dict:
                    try:
                        vals.append(float(lat_dict[f"exit{k}"]))
                    except Exception:
                        pass
            out[f"lat_exit{k}_ms_mean"] = sum(vals) / len(vals) if len(vals) > 0 else None

        return pd.Series(out)

    grouped = df_norm.groupby(
        ["variant", "device", "num_exits", "tap_blocks_json"],
        dropna=False,
    ).apply(build_group_row)

    return grouped


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Dynamic K-exit LaTeX table.
    Index: variant, device, num_exits, tap_blocks_json
    """
    max_k = 0
    for col in df.columns:
        if col.startswith("lat_exit") and col.endswith("_ms_mean"):
            try:
                k = int(col.replace("lat_exit", "").replace("_ms_mean", ""))
                max_k = max(max_k, k)
            except Exception:
                pass

    exit_cols_spec = "r" * max_k
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Average on-device latency per ASHADIP variant under dynamic K-exit settings.}")
    lines.append(r"  \label{tab:on_device_performance}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(rf"  \begin{{tabular}}{{llrl{exit_cols_spec}rr}}")
    lines.append(r"    \toprule")

    exit_headers = " & ".join([rf"Exit{k} (ms)" for k in range(1, max_k + 1)])
    lines.append(
        rf"    Variant & Device & K & Runs & {exit_headers} & Full (ms) & Save (\%) \\"
    )
    lines.append(r"    \midrule")

    def fmt_ms(x):
        if pd.isna(x):
            return "--"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "--"

    def fmt_pct(x):
        if pd.isna(x):
            return "--"
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "--"

    for (variant, device, num_exits, tap_blocks_json), row in df.iterrows():
        n_runs = int(row.get("n_runs", 1))

        exit_vals = []
        for k in range(1, max_k + 1):
            exit_vals.append(fmt_ms(row.get(f"lat_exit{k}_ms_mean")))

        full_ms = fmt_ms(row.get("full_forward_latency_ms_mean"))
        save_pct = fmt_pct(row.get("compute_saving_pct_mean"))

        exit_vals_str = " & ".join(exit_vals)
        lines.append(
            rf"    {variant} & {device} & {int(num_exits)} & {n_runs} & "
            rf"{exit_vals_str} & {full_ms} & {save_pct} \\"
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
        default="analysis/on_device_summary.csv",
        help="CSV produced by profile_latency.py (one row per run).",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/on_device_performance_table.tex",
        help="Output LaTeX file for averaged on-device performance.",
    )
    ap.add_argument(
        "--device_filter",
        choices=["all", "cpu", "cuda"],
        default="all",
        help="Filter rows by device before averaging (default: all).",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"On-device summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(summary_path)
    if df_runs.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    if args.device_filter != "all":
        df_runs = df_runs[df_runs["device"] == args.device_filter]
        if df_runs.empty:
            raise SystemExit(
                f"No rows left after filtering for device='{args.device_filter}'."
            )

    df_norm = _normalize_ondevice_rows(df_runs)
    agg_df = _aggregate_latency(df_norm)

    # CSV backup next to .tex
    out_csv_path = out_tex_path.with_suffix(".csv")
    agg_df.to_csv(out_csv_path, index=True)
    print(f"[ondevice_to_latex] Wrote CSV backup to {out_csv_path}")

    table_str = make_latex_table(agg_df)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print(f"[ondevice_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()