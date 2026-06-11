# scripts/create_tata_final_holdout.py

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_score(text: str, seed: int = 42) -> float:
    key = f"{seed}::{text}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def find_candidate_routing_csvs(run_dir: Path) -> list[Path]:
    csvs = []
    for path in run_dir.rglob("*.csv"):
        name = path.name.lower()
        full = str(path).lower()

        if any(x in name or x in full for x in ["routing", "accepted", "warning", "needs_review", "rejected", "hybrid", "fixed"]):
            csvs.append(path)

    return sorted(csvs)


def load_all_routing_rows(run_dir: Path, mode: str) -> pd.DataFrame:
    rows = []

    for csv_path in find_candidate_routing_csvs(run_dir):
        full_lower = str(csv_path).lower()

        if mode != "all" and mode.lower() not in full_lower:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "parent_clip_id" not in df.columns:
            continue

        df["routing_source_file"] = csv_path.name
        df["routing_source_path"] = str(csv_path)

        if "routing_decision" not in df.columns:
            name = csv_path.name.lower()
            if "accepted_with_warning" in name or "warning" in name:
                df["routing_decision"] = "accepted_with_warning"
            elif "accepted" in name:
                df["routing_decision"] = "accepted"
            elif "needs_review" in name:
                df["routing_decision"] = "needs_review"
            elif "rejected" in name:
                df["routing_decision"] = "rejected"
            else:
                df["routing_decision"] = "unknown"

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No routing CSVs with parent_clip_id found under: {run_dir}")

    all_df = pd.concat(rows, ignore_index=True)

    # One row per parent clip for holdout selection.
    all_df["parent_clip_id"] = all_df["parent_clip_id"].astype(str)
    all_df = all_df.sort_values(["parent_clip_id", "routing_decision", "routing_source_file"])
    all_df = all_df.drop_duplicates(subset=["parent_clip_id"], keep="first").reset_index(drop=True)

    return all_df


def select_holdout(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["_holdout_score"] = df["parent_clip_id"].astype(str).apply(lambda x: stable_score(x, seed=seed))

    selected = []

    # Stratify by routing decision so holdout has accepted/warning/review/rejected examples.
    for _, group in df.groupby("routing_decision", dropna=False):
        group = group.sort_values("_holdout_score").copy()
        n = len(group)
        k = int(round(n * frac))

        if n > 0 and frac > 0 and k == 0:
            k = 1

        selected.append(group.head(k))

    if selected:
        out = pd.concat(selected, ignore_index=True)
    else:
        out = pd.DataFrame(columns=df.columns)

    return out.drop(columns=["_holdout_score"], errors="ignore")


def make_review_template(holdout_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "parent_clip_id",
        "source_file",
        "source_path",
        "source_rel_path",
        "abs_path",
        "file_path",
        "rel_path",
        "routing_decision",
        "routing_source_file",
        "labels",
        "pred_labels",
        "parent_pred_labels",
    ]

    pred_cols = [c for c in holdout_df.columns if c.startswith("parent_pred_")]
    prob_cols = [c for c in holdout_df.columns if c.startswith("parent_prob_") or c.startswith("prob_")]

    cols = [c for c in keep_cols if c in holdout_df.columns] + pred_cols + prob_cols
    cols = list(dict.fromkeys(cols))

    review = holdout_df[cols].copy()

    # Manual ground truth columns. These must be filled by human.
    for lab in LABELS:
        review[lab] = ""

    review["manual_labels"] = ""
    review["review_status"] = "pending_final_holdout"
    review["notes"] = ""

    return review


def filter_csv_by_holdout(input_csv: Path, holdout_ids: set[str], out_csv: Path) -> dict:
    if not input_csv.exists():
        return {
            "input_csv": str(input_csv),
            "exists": False,
        }

    df = pd.read_csv(input_csv)

    if "parent_clip_id" not in df.columns:
        df.to_csv(out_csv, index=False)
        return {
            "input_csv": str(input_csv),
            "exists": True,
            "had_parent_clip_id": False,
            "input_rows": int(len(df)),
            "output_rows": int(len(df)),
            "removed_rows": 0,
            "output_csv": str(out_csv),
        }

    before = len(df)
    df["parent_clip_id"] = df["parent_clip_id"].astype(str)
    filtered = df[~df["parent_clip_id"].isin(holdout_ids)].copy()
    after = len(filtered)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_csv, index=False)

    return {
        "input_csv": str(input_csv),
        "exists": True,
        "had_parent_clip_id": True,
        "input_rows": int(before),
        "output_rows": int(after),
        "removed_rows": int(before - after),
        "output_csv": str(out_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final holdout test set for TATA/main model evaluation.")

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--scratch_root", default="human_talk_workspace/tata_v0.6_scratch")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "fixed", "all"])
    parser.add_argument("--holdout_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scratch_root = Path(args.scratch_root)
    out_root = scratch_root / "final_holdout_test"
    out_root.mkdir(parents=True, exist_ok=True)

    print("")
    print("Creating final holdout test set")
    print("-" * 90)
    print(f"Run dir:       {run_dir}")
    print(f"Scratch root:  {scratch_root}")
    print(f"Mode:          {args.mode}")
    print(f"Holdout frac:  {args.holdout_frac}")
    print(f"Output root:   {out_root}")
    print("-" * 90)

    routing_df = load_all_routing_rows(run_dir, mode=args.mode)
    holdout_df = select_holdout(routing_df, frac=args.holdout_frac, seed=args.seed)

    holdout_ids = set(holdout_df["parent_clip_id"].astype(str).tolist())

    all_routing_path = out_root / f"all_{args.mode}_routing_parent_rows.csv"
    holdout_ids_path = out_root / "final_holdout_parent_ids.csv"
    review_template_path = out_root / "final_holdout_MANUAL_LABEL_TEMPLATE.csv"

    routing_df.to_csv(all_routing_path, index=False)
    pd.DataFrame({"parent_clip_id": sorted(holdout_ids)}).to_csv(holdout_ids_path, index=False)

    review_template = make_review_template(holdout_df)
    review_template.to_csv(review_template_path, index=False)

    # Filter pseudo training candidate CSVs.
    pseudo_root = scratch_root / "pseudo_training_manifests"
    filtered_root = scratch_root / "pseudo_training_manifests_holdout_removed"
    filtered_root.mkdir(parents=True, exist_ok=True)

    filter_reports = []

    for name in [
        "tata_v06_fixed_0p5_pseudo_train_manifest.csv",
        "tata_v06_hybrid_pseudo_train_manifest.csv",
    ]:
        input_csv = pseudo_root / name
        out_csv = filtered_root / name.replace(".csv", "_HOLDOUT_REMOVED.csv")
        filter_reports.append(filter_csv_by_holdout(input_csv, holdout_ids, out_csv))

    summary = {
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "mode": args.mode,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "total_parent_rows": int(len(routing_df)),
        "holdout_parent_rows": int(len(holdout_df)),
        "routing_counts_all": routing_df["routing_decision"].value_counts().to_dict(),
        "routing_counts_holdout": holdout_df["routing_decision"].value_counts().to_dict(),
        "outputs": {
            "all_routing_parent_rows": str(all_routing_path),
            "holdout_parent_ids": str(holdout_ids_path),
            "manual_label_template": str(review_template_path),
            "filtered_pseudo_training_root": str(filtered_root),
        },
        "filter_reports": filter_reports,
        "important_rule": "Holdout rows must not be used for training, threshold tuning, pseudo-label training, or manual needs_review expansion. They are only for final evaluation after human labeling.",
    }

    summary_path = out_root / "final_holdout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = out_root / "final_holdout_summary.md"
    lines = []
    lines.append("# Final Holdout Test Set Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append(f"- Mode: `{args.mode}`")
    lines.append(f"- Total parent rows considered: `{summary['total_parent_rows']}`")
    lines.append(f"- Holdout parent rows: `{summary['holdout_parent_rows']}`")
    lines.append("")
    lines.append("## Holdout Routing Counts")
    lines.append("")
    lines.append("| Routing decision | Count |")
    lines.append("|---|---:|")
    for k, v in summary["routing_counts_holdout"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append("Do not use holdout rows for training. Manually label them and reserve only for final testing.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Final holdout created.")
    print("-" * 90)
    print(f"Total parent rows:   {len(routing_df)}")
    print(f"Holdout parent rows: {len(holdout_df)}")
    print("")
    print("Holdout routing counts:")
    print(holdout_df["routing_decision"].value_counts().to_string())
    print("")
    print(f"Manual label template: {review_template_path}")
    print(f"Summary:               {md_path}")
    print(f"Filtered pseudo CSVs:   {filtered_root}")


if __name__ == "__main__":
    main()