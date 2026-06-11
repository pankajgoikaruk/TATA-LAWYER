# scripts/apply_v08_reviewed_delta_outputs.py
#
# Apply manually reviewed v0.8 delta files to produce corrected inputs.
#
# Reviewed inputs expected:
#   manual_review_queue/03_raw_nontarget_context_REVIEW.csv
#   manual_review_queue/04_holdout_nontarget_context_REVIEW.csv
#   manual_review_queue/06_lawyer_new_samples_REVIEW.csv
#
# Outputs:
#   corrected_training_inputs/02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_v08_context_checked.csv
#   corrected_training_inputs/06_lawyer_new_samples_REVIEWED_v08.csv
#   corrected_training_inputs/hybrid_accepted_with_warning_PLUS_REVIEWED_LAWYER_NEW_v08.csv
#   corrected_holdout/01_raw_final_holdout_GROUND_TRUTH_FINAL_v08_context_checked.csv
#
# This does NOT modify old v0.6 or v0.8 files.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_LABELS_DEFAULT = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

EVENT_LABELS_DEFAULT = [
    "music_present",
    "audience_reaction_present",
    "silence_present",
]

NON_TARGET_CLASSES_DEFAULT = [
    "Les_Brown",
    "Mel_Robbins",
    "Oprah_Winfrey",
    "Rabin_Sharma",
    "Simon_Sinek",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: Path) -> list[str]:
    payload = load_json(path)
    labels = payload["labels"] if isinstance(payload, dict) else payload
    labels = [str(x) for x in labels]
    if not labels:
        raise RuntimeError("No labels found.")
    return labels


def ensure_label_cols(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        if lab not in df.columns:
            df[lab] = 0
        df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def refresh_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = ensure_label_cols(df, labels)
    df["manual_labels"] = df.apply(
        lambda r: "|".join([lab for lab in labels if int(r.get(lab, 0)) == 1]),
        axis=1,
    )
    df["labels"] = df["manual_labels"]
    df["num_active_labels"] = df[labels].sum(axis=1).astype(int)
    return df


def source_text(df: pd.DataFrame) -> pd.Series:
    cols = ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].astype(str).agg(" ".join, axis=1)


def non_target_mask(df: pd.DataFrame, non_target_classes: list[str]) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=bool)
    text = source_text(df)
    mask = pd.Series([False] * len(df), index=df.index)
    for cls in non_target_classes:
        mask = mask | text.str.contains(cls, regex=False, na=False)
    return mask


def force_non_target_identity(
    df: pd.DataFrame,
    labels: list[str],
    target_labels: list[str],
    open_set_label: str,
    non_target_classes: list[str],
) -> pd.DataFrame:
    df = ensure_label_cols(df, labels)
    mask = non_target_mask(df, non_target_classes)
    if len(df) and mask.any():
        for lab in target_labels:
            df.loc[mask, lab] = 0
        df.loc[mask, open_set_label] = 1
    return refresh_labels(df, labels)


def reviewed_stats(df: pd.DataFrame) -> dict[str, int]:
    if "human_review_status" not in df.columns:
        return {"rows": int(len(df)), "reviewed": 0, "pending_or_blank": int(len(df))}
    status = df["human_review_status"].fillna("").astype(str).str.lower()
    return {
        "rows": int(len(df)),
        "reviewed": int((status == "reviewed").sum()),
        "pending_or_blank": int((status != "reviewed").sum()),
    }


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path, low_memory=False)


def overlay_labels_by_parent(
    base_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    labels: list[str],
    extra_cols_to_copy: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    base = base_df.copy()
    patch = patch_df.copy()

    if "parent_clip_id" not in base.columns:
        raise RuntimeError("Base CSV missing parent_clip_id")
    if "parent_clip_id" not in patch.columns:
        raise RuntimeError("Patch CSV missing parent_clip_id")

    base["parent_clip_id"] = base["parent_clip_id"].astype(str)
    patch["parent_clip_id"] = patch["parent_clip_id"].astype(str)
    patch = patch.drop_duplicates("parent_clip_id", keep="last")
    patch_index = patch.set_index("parent_clip_id")

    base_ids = set(base["parent_clip_id"].astype(str))
    updated = 0

    for i, row in base.iterrows():
        pid = str(row["parent_clip_id"])
        if pid not in patch_index.index:
            continue

        p = patch_index.loc[pid]
        for lab in labels:
            base.at[i, lab] = int(p.get(lab, 0))

        if extra_cols_to_copy:
            for col in extra_cols_to_copy:
                if col in patch.columns:
                    base.at[i, col] = p.get(col, "")

        updated += 1

    missing = sum(1 for pid in patch["parent_clip_id"].astype(str) if pid not in base_ids)
    base = refresh_labels(base, labels)

    return base, {
        "patch_rows": int(len(patch)),
        "updated_rows": int(updated),
        "patch_rows_missing_from_base": int(missing),
    }


def label_counts(df: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    return {
        lab: int(pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).sum())
        for lab in labels
        if lab in df.columns
    }


def save_csv(df: pd.DataFrame, path: Path, labels: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = refresh_labels(df, labels)
    df.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "label_counts": label_counts(df, labels),
    }


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v0.8 Human-Corrected-Balanced Apply Reviewed Delta Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Reviewed input validation",
        "",
        "| File | Rows | Reviewed | Pending/blank |",
        "|---|---:|---:|---:|",
    ]

    for name, stats in summary["reviewed_input_stats"].items():
        lines.append(f"| `{name}` | {stats['rows']} | {stats['reviewed']} | {stats['pending_or_blank']} |")

    lines += ["", "## Overlay stats", "", "| Item | Count |", "|---|---:|"]
    for key, value in summary["overlay_stats"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += ["", "## Outputs", ""]
    for name, info in summary["outputs"].items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- Path: `{info['path']}`")
        lines.append(f"- Rows: `{info['rows']}`")
        lines.append("")
        lines.append("| Label | Count |")
        lines.append("|---|---:|")
        for lab, count in info.get("label_counts", {}).items():
            lines.append(f"| `{lab}` | {count} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed v0.8 delta CSVs to produce corrected training/evaluation inputs.")
    parser.add_argument("--v06_root", default="human_talk_workspace/tata_v0.6_raw_pipeline")
    parser.add_argument("--out_root", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline")
    parser.add_argument("--labels_json", default="configs/human_talk_10label_schema.json")
    parser.add_argument("--open_set_label", default="other_speaker_present")
    parser.add_argument("--require_reviewed", action="store_true")
    args = parser.parse_args()

    v06_root = Path(args.v06_root)
    out_root = Path(args.out_root)
    labels_json = Path(args.labels_json)
    manual_dir = out_root / "manual_review_queue"

    labels = load_labels(labels_json)
    target_labels = [lab for lab in TARGET_LABELS_DEFAULT if lab in labels]
    open_set_label = str(args.open_set_label)
    non_target_classes = NON_TARGET_CLASSES_DEFAULT

    # Reviewed input files.
    raw_nt_review_path = manual_dir / "03_raw_nontarget_context_REVIEW.csv"
    holdout_nt_review_path = manual_dir / "04_holdout_nontarget_context_REVIEW.csv"
    new_samples_review_path = manual_dir / "06_lawyer_new_samples_REVIEW.csv"

    raw_nt_review = read_csv(raw_nt_review_path, "03_raw_nontarget_context_REVIEW.csv")
    holdout_nt_review = read_csv(holdout_nt_review_path, "04_holdout_nontarget_context_REVIEW.csv")
    new_samples_review = read_csv(new_samples_review_path, "06_lawyer_new_samples_REVIEW.csv")

    raw_nt_review = force_non_target_identity(raw_nt_review, labels, target_labels, open_set_label, non_target_classes)
    holdout_nt_review = force_non_target_identity(holdout_nt_review, labels, target_labels, open_set_label, non_target_classes)
    new_samples_review = refresh_labels(new_samples_review, labels)

    reviewed_input_stats = {
        "03_raw_nontarget_context_REVIEW.csv": reviewed_stats(raw_nt_review),
        "04_holdout_nontarget_context_REVIEW.csv": reviewed_stats(holdout_nt_review),
        "06_lawyer_new_samples_REVIEW.csv": reviewed_stats(new_samples_review),
    }

    if args.require_reviewed:
        not_done = {name: stats for name, stats in reviewed_input_stats.items() if stats["pending_or_blank"] > 0}
        if not_done:
            raise RuntimeError(f"Some rows are not marked reviewed: {not_done}")

    # Original v0.6 files.
    raw_base_path = v06_root / "manual_review_queue" / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv"
    holdout_base_path = v06_root / "manual_review_queue" / "01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv"
    hybrid_warning_path = v06_root / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted_with_warning.csv"

    raw_base = refresh_labels(read_csv(raw_base_path, "v0.6 corrected raw hybrid needs-review"), labels)
    holdout_base = refresh_labels(read_csv(holdout_base_path, "v0.6 holdout ground truth"), labels)
    hybrid_warning = refresh_labels(read_csv(hybrid_warning_path, "v0.6 hybrid accepted_with_warning"), labels)

    corrected_raw, raw_overlay_stats = overlay_labels_by_parent(
        raw_base,
        raw_nt_review,
        labels,
        extra_cols_to_copy=["human_review_status", "human_review_notes"],
    )
    corrected_holdout, holdout_overlay_stats = overlay_labels_by_parent(
        holdout_base,
        holdout_nt_review,
        labels,
        extra_cols_to_copy=["human_review_status", "human_review_notes"],
    )

    reviewed_new = new_samples_review.copy()
    if "dataset_role" in reviewed_new.columns:
        reviewed_new = reviewed_new[reviewed_new["dataset_role"].astype(str) == "training_candidate"].copy()
    reviewed_new = refresh_labels(reviewed_new, labels)

    # Add reviewed LAWYER new samples into warning pseudo-label group.
    hybrid_warning["v08_source_group"] = "v06_hybrid_accepted_with_warning"
    reviewed_new["v08_source_group"] = "v08_reviewed_lawyer_new_sample"

    combined_warning_plus_new = pd.concat([hybrid_warning, reviewed_new], ignore_index=True, sort=False)

    if "parent_clip_id" in combined_warning_plus_new.columns:
        combined_warning_plus_new["parent_clip_id"] = combined_warning_plus_new["parent_clip_id"].astype(str)
        priority = {
            "v08_reviewed_lawyer_new_sample": 0,
            "v06_hybrid_accepted_with_warning": 1,
        }
        combined_warning_plus_new["_priority"] = combined_warning_plus_new["v08_source_group"].map(priority).fillna(99).astype(int)
        combined_warning_plus_new = combined_warning_plus_new.sort_values(["parent_clip_id", "_priority"])
        combined_warning_plus_new = combined_warning_plus_new.drop_duplicates("parent_clip_id", keep="first")
        combined_warning_plus_new = combined_warning_plus_new.drop(columns=["_priority"], errors="ignore")

    combined_warning_plus_new = refresh_labels(combined_warning_plus_new, labels)

    out_eval = out_root / "corrected_holdout"
    out_training = out_root / "corrected_training_inputs"
    out_meta = out_root / "metadata"

    outputs = {
        "corrected_raw_hybrid_needs_review": save_csv(
            corrected_raw,
            out_training / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_v08_context_checked.csv",
            labels,
        ),
        "corrected_holdout_ground_truth": save_csv(
            corrected_holdout,
            out_eval / "01_raw_final_holdout_GROUND_TRUTH_FINAL_v08_context_checked.csv",
            labels,
        ),
        "reviewed_lawyer_new_samples": save_csv(
            reviewed_new,
            out_training / "06_lawyer_new_samples_REVIEWED_v08.csv",
            labels,
        ),
        "hybrid_warning_plus_reviewed_lawyer_new": save_csv(
            combined_warning_plus_new,
            out_training / "hybrid_accepted_with_warning_PLUS_REVIEWED_LAWYER_NEW_v08.csv",
            labels,
        ),
    }

    summary = {
        "generated_at": now_iso(),
        "experiment": "v0.8-human-corrected-balanced",
        "v06_root": str(v06_root),
        "out_root": str(out_root),
        "manual_review_queue": str(manual_dir),
        "labels": labels,
        "target_labels": target_labels,
        "open_set_label": open_set_label,
        "known_non_target_source_classes": non_target_classes,
        "reviewed_input_stats": reviewed_input_stats,
        "overlay_stats": {
            "raw_context_patch_rows": raw_overlay_stats["patch_rows"],
            "raw_context_rows_updated": raw_overlay_stats["updated_rows"],
            "raw_context_patch_rows_missing_from_base": raw_overlay_stats["patch_rows_missing_from_base"],
            "holdout_context_patch_rows": holdout_overlay_stats["patch_rows"],
            "holdout_context_rows_updated": holdout_overlay_stats["updated_rows"],
            "holdout_context_patch_rows_missing_from_base": holdout_overlay_stats["patch_rows_missing_from_base"],
        },
        "outputs": outputs,
        "next_builder_inputs": {
            "hybrid_accepted_csv": str(v06_root / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted.csv"),
            "hybrid_warning_csv": outputs["hybrid_warning_plus_reviewed_lawyer_new"]["path"],
            "corrected_needs_review_csv": outputs["corrected_raw_hybrid_needs_review"]["path"],
            "corrected_holdout_csv": outputs["corrected_holdout_ground_truth"]["path"],
        },
    }

    out_meta.mkdir(parents=True, exist_ok=True)
    summary_json = out_meta / "v08_apply_reviewed_delta_summary.json"
    summary_md = out_meta / "v08_apply_reviewed_delta_summary.md"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary_md, summary)

    print("")
    print("Applied reviewed v0.8 delta outputs")
    print("-" * 90)
    print("Reviewed input stats:")
    for name, stats in reviewed_input_stats.items():
        print(f"  {name}: rows={stats['rows']}, reviewed={stats['reviewed']}, pending_or_blank={stats['pending_or_blank']}")

    print("")
    print("Overlay stats:")
    for k, v in summary["overlay_stats"].items():
        print(f"  {k}: {v}")

    print("")
    print("Outputs:")
    for name, info in outputs.items():
        print(f"  {name}: rows={info['rows']}")
        print(f"    {info['path']}")

    print("")
    print("Next builder inputs:")
    for k, v in summary["next_builder_inputs"].items():
        print(f"  {k}: {v}")

    print("")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD:   {summary_md}")


if __name__ == "__main__":
    main()
