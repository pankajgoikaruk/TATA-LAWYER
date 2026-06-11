# scripts/compare_variants.py

from __future__ import annotations

import os
import json
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+_(\d+)$")


def find_new_run_dirs(root: Path):
    """
    NEW-ONLY layout:
      runs/<VariantDir>/<RunDir>/
        must contain: meta.json and summary.json
    """
    runs_root = root / "runs"
    if not runs_root.exists():
        return []

    run_dirs = []
    for dirpath, _, filenames in os.walk(runs_root):
        if "summary.json" not in filenames:
            continue
        if "meta.json" not in filenames:
            continue

        run_dir = Path(dirpath)
        variant_dir = run_dir.parent
        runs_dir = variant_dir.parent

        if runs_dir.name != "runs":
            continue

        run_dirs.append(run_dir)

    return run_dirs


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_variant_runid_from_meta(run_dir: Path):
    meta = load_json(run_dir / "meta.json")

    variant = meta.get("variant", None)
    run_id = meta.get("run_id", None)

    if not run_id:
        run_id = run_dir.name
    if not variant:
        variant = meta.get("variant_safe", run_dir.parent.name)

    return str(variant), str(run_id), meta


def _normalize_exit_mix(exit_mix):
    if not isinstance(exit_mix, dict):
        return {}

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


def load_summary_row(run_dir: Path):
    """
    Load meta.json + summary.json and extract key metrics for comparison.
    Dynamic K-exit safe.
    """
    variant, run_id, meta = parse_variant_runid_from_meta(run_dir)
    summ = load_json(run_dir / "summary.json")

    policy = summ.get("policy_summary", {}) or {}
    exit_mix = _normalize_exit_mix(policy.get("exit_mix", {}))
    temps = policy.get("temperatures", [])
    tap_blocks = policy.get("tap_blocks", meta.get("tap_blocks", None))
    num_exits = policy.get("num_exits", None)

    if num_exits is None and exit_mix:
        keys = [int(k[1:]) for k in exit_mix.keys()]
        num_exits = max(keys)
    if num_exits is None and isinstance(temps, list) and len(temps) > 0:
        num_exits = len(temps)

    row = {
        "run_id": run_id,
        "variant": variant,
        "num_exits": num_exits,
        "tap_blocks_json": json.dumps(tap_blocks) if tap_blocks is not None else "",
        "tau": policy.get("tau", None),
        "temperatures_json": json.dumps(temps),
        "policy_test_acc": policy.get("policy_test_acc", None),
        "avg_exit_depth": policy.get("avg_exit_depth", None),
        "exit_mix_json": json.dumps(exit_mix),
        "expected_mflops": policy.get("expected_mflops", None),
        "full_mflops": policy.get("full_mflops", None),
        "compute_saving_pct": policy.get("compute_saving_pct", None),
        "ece_policy": (policy.get("policy_calibration") or {}).get("ece", None),
        "n_mels": policy.get("n_mels", None),
        "frames": policy.get("frames", None),
        "num_classes": policy.get("num_classes", None),
        "device": meta.get("device", None),
        "segment_sec": meta.get("segment_sec", None),
        "hop_sec": meta.get("hop_sec", None),
    }

    # Backward-compatible flat columns for easier CSV/table use
    for ek, ev in exit_mix.items():
        row[f"exit_{ek}"] = ev

    # Also keep legacy names for first 3 if present
    row["exit_e1"] = exit_mix.get("e1", None)
    row["exit_e2"] = exit_mix.get("e2", None)
    row["exit_e3"] = exit_mix.get("e3", None)

    return row


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy vs compute saving
    plt.figure()
    for variant, df_v in df.groupby("variant"):
        plt.scatter(
            df_v["compute_saving_pct"],
            df_v["policy_test_acc"],
            label=variant,
            s=40,
        )
        for _, row in df_v.iterrows():
            try:
                x = float(row["compute_saving_pct"])
                y = float(row["policy_test_acc"])
            except Exception:
                continue
            plt.text(
                x,
                y,
                row["run_id"],
                fontsize=7,
                ha="left",
                va="bottom",
            )

    plt.xlabel("Compute saving (%)")
    plt.ylabel("Policy test accuracy")
    plt.title("Accuracy vs compute saving (new runs only)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_compute_saving.png", dpi=150)
    plt.close()

    # 2) Dynamic exit mix stacked bar per run
    exit_cols = sorted(
        [c for c in df.columns if c.startswith("exit_e")],
        key=lambda s: int(s.replace("exit_e", "")),
    )

    if len(exit_cols) == 0:
        print("[compare_variants] No exit mix columns found; skipping exit_mix_per_run.png")
        return

    plt.figure(figsize=(max(10, len(df) * 0.5), 4))
    df_plot = df.reset_index(drop=True)
    x = range(len(df_plot))
    bottom = None

    for idx, col in enumerate(exit_cols, start=1):
        vals = pd.to_numeric(df_plot[col], errors="coerce").fillna(0.0)
        if bottom is None:
            plt.bar(x, vals, label=f"exit{idx}")
            bottom = vals.copy()
        else:
            plt.bar(x, vals, bottom=bottom, label=f"exit{idx}")
            bottom = bottom + vals

    plt.xticks(x, df_plot["run_id"], rotation=45, ha="right")
    plt.ylabel("Fraction of samples")
    plt.title("Exit mix per run (new runs only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exit_mix_per_run.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Project root (default: current dir). Uses only ./runs/<variant>/<run>/ with meta.json+summary.json.",
    )
    ap.add_argument(
        "--out_csv",
        default="analysis/all_runs_summary.csv",
        help="Path to save aggregated CSV.",
    )
    ap.add_argument(
        "--out_dir",
        default="analysis/plots",
        help="Directory to save comparison plots.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    run_dirs = find_new_run_dirs(root)
    if not run_dirs:
        raise SystemExit(
            f"No NEW-style runs found under {root / 'runs'}.\n"
            f"Expected folders like runs/<Variant>/<RunId>/ containing meta.json and summary.json."
        )

    rows = []
    for rd in run_dirs:
        try:
            rows.append(load_summary_row(rd))
        except Exception as e:
            print(f"[warn] Skipping run at {rd} due to error: {e}")

    if not rows:
        raise SystemExit("Found run directories but could not load any rows (all failed).")

    df = pd.DataFrame(rows)

    df = df[df["run_id"].astype(str).apply(lambda s: bool(RUN_ID_RE.match(s)))]
    df = df.sort_values(["variant", "run_id"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV to {out_csv}")

    make_plots(df, Path(args.out_dir))
    print(f"Saved comparison plots under {args.out_dir}")


if __name__ == "__main__":
    main()