# scripts/create_tata_manual_review_queue.py

from __future__ import annotations

import argparse
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


def as_binary_series(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=int)

    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(default)
        .astype(int)
        .clip(0, 1)
    )


def active_label_text(row: pd.Series) -> str:
    active = [lab for lab in LABELS if int(row.get(lab, 0)) == 1]
    return "|".join(active)


def ensure_easy_label_columns(df: pd.DataFrame, prefill_from_parent_pred: bool = True) -> pd.DataFrame:
    df = df.copy()

    for lab in LABELS:
        if lab in df.columns:
            df[lab] = as_binary_series(df, lab, default=0)
            continue

        pred_col = f"parent_pred_{lab}"
        if prefill_from_parent_pred and pred_col in df.columns:
            df[lab] = as_binary_series(df, pred_col, default=0)
        else:
            df[lab] = 0

    df["manual_labels"] = df.apply(active_label_text, axis=1)

    if "review_status" not in df.columns:
        df["review_status"] = "pending"
    else:
        df["review_status"] = df["review_status"].fillna("").replace("", "pending")

    if "notes" not in df.columns:
        df["notes"] = ""

    return df


def reorder_for_easy_edit(df: pd.DataFrame) -> pd.DataFrame:
    front_cols = [
        "parent_clip_id",
        "source_file",
        "source_path",
        "source_rel_path",
        "routing_decision",
        "labels",
        *LABELS,
        "manual_labels",
        "review_status",
        "notes",
    ]

    front_cols = [c for c in front_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in front_cols]

    return df[front_cols + remaining_cols]


def find_hybrid_needs_review(run_dir: Path) -> Path:
    candidates = []

    for path in run_dir.rglob("*.csv"):
        full = str(path).lower()
        name = path.name.lower()

        if "hybrid" in full and "needs_review" in name:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No hybrid needs_review CSV found under: {run_dir}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_holdout_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    df = pd.read_csv(path)

    if "parent_clip_id" not in df.columns:
        return set()

    return set(df["parent_clip_id"].astype(str).tolist())


def write_summary_md(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# TATA v0.6 Manual Review Queue Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Final holdout rows: `{summary['final_holdout_rows']}`")
    lines.append(f"- Hybrid needs-review rows before holdout removal: `{summary['hybrid_needs_review_rows_before']}`")
    lines.append(f"- Hybrid needs-review rows after holdout removal: `{summary['hybrid_needs_review_rows_after']}`")
    lines.append(f"- Holdout rows removed from needs-review: `{summary['holdout_removed_from_needs_review']}`")
    lines.append("")
    lines.append("## Manual Editing Columns")
    lines.append("")
    lines.append("Edit only these 10 label columns plus `review_status` and `notes`:")
    lines.append("")
    for lab in LABELS:
        lines.append(f"- `{lab}`")
    lines.append("")
    lines.append("Do not edit `parent_pred_*` or `parent_prob_*` columns.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create easy-edit manual review CSVs for TATA v0.6."
    )

    parser.add_argument(
        "--scratch_root",
        default="human_talk_workspace/tata_v0.6_scratch",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
    )

    args = parser.parse_args()

    scratch_root = Path(args.scratch_root)
    run_dir = Path(args.run_dir)

    review_root = scratch_root / "manual_review_queue"
    review_root.mkdir(parents=True, exist_ok=True)

    holdout_template = scratch_root / "final_holdout_test" / "final_holdout_MANUAL_LABEL_TEMPLATE.csv"
    holdout_ids_csv = scratch_root / "final_holdout_test" / "final_holdout_parent_ids.csv"

    if not holdout_template.exists():
        raise FileNotFoundError(f"Final holdout template not found: {holdout_template}")

    holdout_ids = load_holdout_ids(holdout_ids_csv)

    # 1. Final holdout easy-edit file.
    holdout_df = pd.read_csv(holdout_template)
    holdout_easy = ensure_easy_label_columns(
        holdout_df,
        prefill_from_parent_pred=False,
    )
    holdout_easy["review_status"] = "pending_final_holdout"
    holdout_easy = reorder_for_easy_edit(holdout_easy)

    out_holdout = review_root / "01_final_holdout_MANUAL_LABEL_TEMPLATE_EASY_EDIT.csv"
    holdout_easy.to_csv(out_holdout, index=False)

    # 2. Hybrid needs-review easy-edit file with holdout removed.
    hybrid_needs_review_csv = find_hybrid_needs_review(run_dir)
    needs_df = pd.read_csv(hybrid_needs_review_csv)

    before = len(needs_df)

    if "parent_clip_id" not in needs_df.columns:
        raise RuntimeError("Hybrid needs_review CSV does not contain parent_clip_id")

    needs_df["parent_clip_id"] = needs_df["parent_clip_id"].astype(str)
    needs_df = needs_df[~needs_df["parent_clip_id"].isin(holdout_ids)].copy()

    after = len(needs_df)

    needs_easy = ensure_easy_label_columns(
        needs_df,
        prefill_from_parent_pred=True,
    )
    needs_easy["review_status"] = "pending_needs_review"
    needs_easy = reorder_for_easy_edit(needs_easy)

    out_needs = review_root / "02_hybrid_needs_review_HOLDOUT_REMOVED_MANUAL_CORRECTION_EASY_EDIT.csv"
    needs_easy.to_csv(out_needs, index=False)

    summary = {
        "generated_at": now_iso(),
        "scratch_root": str(scratch_root),
        "run_dir": str(run_dir),
        "hybrid_needs_review_source": str(hybrid_needs_review_csv),
        "final_holdout_rows": int(len(holdout_easy)),
        "hybrid_needs_review_rows_before": int(before),
        "hybrid_needs_review_rows_after": int(after),
        "holdout_removed_from_needs_review": int(before - after),
        "labels": LABELS,
        "outputs": {
            "final_holdout_easy_edit": str(out_holdout),
            "hybrid_needs_review_easy_edit": str(out_needs),
            "summary_json": str(review_root / "manual_review_queue_summary.json"),
            "summary_md": str(review_root / "manual_review_queue_summary.md"),
        },
    }

    summary_json = review_root / "manual_review_queue_summary.json"
    summary_md = review_root / "manual_review_queue_summary.md"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary_md, summary)

    print("")
    print("Manual review queue created")
    print("-" * 90)
    print(f"Final holdout easy-edit:     {out_holdout}")
    print(f"Needs-review easy-edit:      {out_needs}")
    print(f"Summary:                     {summary_md}")
    print("")
    print(f"Final holdout rows:          {len(holdout_easy)}")
    print(f"Needs-review before removal: {before}")
    print(f"Needs-review after removal:  {after}")
    print(f"Holdout removed:             {before - after}")


if __name__ == "__main__":
    main()
