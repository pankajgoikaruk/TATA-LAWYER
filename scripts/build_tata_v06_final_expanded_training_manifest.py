# scripts/build_tata_v06_final_expanded_training_manifest.py

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


def to_bin(v) -> int:
    try:
        if pd.isna(v):
            return 0
        return 1 if int(float(v)) == 1 else 0
    except Exception:
        return 0


def refresh_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lab in LABELS:
        if lab not in df.columns:
            df[lab] = 0
        df[lab] = df[lab].apply(to_bin)

    df["labels"] = df.apply(
        lambda r: "|".join([lab for lab in LABELS if int(r[lab]) == 1]),
        axis=1,
    )
    df["num_active_labels"] = df[LABELS].sum(axis=1).astype(int)

    return df


def make_project_relative_feat_path(features_root: Path, feat_relpath: str) -> str:
    feat_relpath = str(feat_relpath).replace("\\", "/")
    full = features_root / Path(feat_relpath)

    # Keep path relative to repo root, because training will use FeaturesRoot "."
    return str(full).replace("\\", "/")


def prepare_seed_rows(seed_manifest: Path, seed_features_root: Path) -> pd.DataFrame:
    df = pd.read_csv(seed_manifest)

    if "feat_relpath" not in df.columns:
        raise RuntimeError("Seed manifest missing feat_relpath")

    df = refresh_labels(df)

    df["feat_relpath"] = df["feat_relpath"].apply(
        lambda x: make_project_relative_feat_path(seed_features_root, x)
    )

    df["label_source"] = "human_reviewed_seed"
    df["is_pseudo_labeled"] = 0
    df["is_corrected_needs_review"] = 0
    df["training_group"] = "seed_reviewed"

    # Keep existing seed split: train/val/test.
    if "split" not in df.columns:
        df["split"] = "train"

    return df


def load_parent_labels_from_routing(path: Path, label_source: str, training_group: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "parent_clip_id" not in df.columns:
        raise RuntimeError(f"Missing parent_clip_id in {path}")

    for lab in LABELS:
        if lab in df.columns:
            df[lab] = df[lab].apply(to_bin)
        else:
            pred_col = f"parent_pred_{lab}"
            if pred_col in df.columns:
                df[lab] = df[pred_col].apply(to_bin)
            else:
                df[lab] = 0

    df = refresh_labels(df)

    df["label_source"] = label_source
    df["training_group"] = training_group
    df["is_pseudo_labeled"] = 1
    df["is_corrected_needs_review"] = 0

    return df


def load_corrected_needs_review(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    if "parent_clip_id" not in df.columns:
        raise RuntimeError(f"Missing parent_clip_id in {path}")

    df = refresh_labels(df)

    zero = df[df["num_active_labels"] == 0].copy()
    df = df[df["num_active_labels"] > 0].copy()

    df["label_source"] = "human_corrected_needs_review"
    df["training_group"] = "raw_hybrid_needs_review_corrected"
    df["is_pseudo_labeled"] = 0
    df["is_corrected_needs_review"] = 1

    return df, zero


def map_parent_labels_to_raw_segments(
    raw_feature_manifest: Path,
    raw_features_root: Path,
    parent_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(raw_feature_manifest)

    if "parent_clip_id" not in raw_df.columns:
        raise RuntimeError("Raw feature manifest missing parent_clip_id")

    if "feat_relpath" not in raw_df.columns:
        raise RuntimeError("Raw feature manifest missing feat_relpath")

    parent_labels = parent_labels.copy()
    parent_labels["parent_clip_id"] = parent_labels["parent_clip_id"].astype(str)

    raw_df["parent_clip_id"] = raw_df["parent_clip_id"].astype(str)

    label_cols = [
        "parent_clip_id",
        *LABELS,
        "labels",
        "num_active_labels",
        "label_source",
        "training_group",
        "is_pseudo_labeled",
        "is_corrected_needs_review",
    ]

    available_label_cols = [c for c in label_cols if c in parent_labels.columns]
    merged = raw_df.merge(
        parent_labels[available_label_cols],
        on="parent_clip_id",
        how="inner",
        suffixes=("", "_parent"),
    )

    missing_parent_ids = sorted(
        set(parent_labels["parent_clip_id"].astype(str)) - set(raw_df["parent_clip_id"].astype(str))
    )
    missing_df = pd.DataFrame({"parent_clip_id": missing_parent_ids})

    # Replace dummy label columns with final parent labels.
    for lab in LABELS:
        parent_col = f"{lab}_parent"
        if parent_col in merged.columns:
            merged[lab] = merged[parent_col].apply(to_bin)
            merged = merged.drop(columns=[parent_col])

    for col in [
        "labels",
        "num_active_labels",
        "label_source",
        "training_group",
        "is_pseudo_labeled",
        "is_corrected_needs_review",
    ]:
        parent_col = f"{col}_parent"
        if parent_col in merged.columns:
            merged[col] = merged[parent_col]
            merged = merged.drop(columns=[parent_col])

    merged = refresh_labels(merged)

    merged["feat_relpath"] = merged["feat_relpath"].apply(
        lambda x: make_project_relative_feat_path(raw_features_root, x)
    )

    # Raw pseudo/corrected rows should go into training only.
    merged["split"] = "train"

    return merged, missing_df


def write_summary_md(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# TATA v0.6 Final Expanded Training Manifest Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Row Counts")
    lines.append("")
    lines.append(f"- Seed segment rows: `{summary['seed_segment_rows']}`")
    lines.append(f"- Raw expanded segment rows: `{summary['raw_expanded_segment_rows']}`")
    lines.append(f"- Final combined segment rows: `{summary['final_combined_segment_rows']}`")
    lines.append(f"- Zero-active corrected needs-review rows excluded: `{summary['zero_active_corrected_needs_review_excluded']}`")
    lines.append(f"- Missing parent segment groups: `{summary['missing_parent_segment_groups']}`")
    lines.append("")
    lines.append("## Training Groups")
    lines.append("")
    lines.append("| Group | Rows |")
    lines.append("|---|---:|")
    for k, v in summary["training_group_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Label Counts")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|---|---:|")
    for lab, count in summary["final_label_counts"].items():
        lines.append(f"| `{lab}` | {count} |")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Training Note")
    lines.append("")
    lines.append("Use `-FeaturesRoot .` when training with this combined manifest because feature paths are stored relative to the project root.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final expanded training manifest: seed + raw accepted + warning + corrected needs_review."
    )

    parser.add_argument("--seed_feature_manifest", required=True)
    parser.add_argument("--seed_features_root", required=True)

    parser.add_argument("--raw_feature_manifest", required=True)
    parser.add_argument("--raw_features_root", required=True)

    parser.add_argument("--hybrid_accepted_csv", required=True)
    parser.add_argument("--hybrid_warning_csv", required=True)
    parser.add_argument("--corrected_needs_review_csv", required=True)

    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--out_root", required=True)

    args = parser.parse_args()

    out_root = Path(args.out_root)
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    seed_df = prepare_seed_rows(
        Path(args.seed_feature_manifest),
        Path(args.seed_features_root),
    )

    accepted_parent = load_parent_labels_from_routing(
        Path(args.hybrid_accepted_csv),
        label_source="tata_hybrid_high_confidence",
        training_group="raw_hybrid_accepted",
    )

    warning_parent = load_parent_labels_from_routing(
        Path(args.hybrid_warning_csv),
        label_source="tata_hybrid_accepted_with_warning",
        training_group="raw_hybrid_accepted_with_warning",
    )

    corrected_parent, zero_corrected = load_corrected_needs_review(
        Path(args.corrected_needs_review_csv)
    )

    parent_labels = pd.concat(
        [accepted_parent, warning_parent, corrected_parent],
        ignore_index=True,
    )

    # Safety: one row per raw parent clip. If duplicated, keep corrected human row first.
    parent_labels["_priority"] = parent_labels["training_group"].map(
        {
            "raw_hybrid_needs_review_corrected": 0,
            "raw_hybrid_accepted_with_warning": 1,
            "raw_hybrid_accepted": 2,
        }
    ).fillna(9)

    parent_labels = (
        parent_labels.sort_values(["parent_clip_id", "_priority"])
        .drop_duplicates(subset=["parent_clip_id"], keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    raw_segments, missing_df = map_parent_labels_to_raw_segments(
        Path(args.raw_feature_manifest),
        Path(args.raw_features_root),
        parent_labels,
    )

    final_df = pd.concat([seed_df, raw_segments], ignore_index=True)
    final_df = refresh_labels(final_df)

    out_manifest = meta_dir / "multilabel_features_manifest.csv"
    out_parent_labels = meta_dir / "raw_parent_labels_used.csv"
    out_zero = meta_dir / "zero_active_corrected_needs_review_excluded.csv"
    out_missing = meta_dir / "missing_parent_segment_groups.csv"
    out_labels = meta_dir / "tata_v06_labels.json"
    out_summary_json = meta_dir / "final_expanded_training_summary.json"
    out_summary_md = meta_dir / "final_expanded_training_summary.md"

    final_df.to_csv(out_manifest, index=False)
    parent_labels.to_csv(out_parent_labels, index=False)
    zero_corrected.to_csv(out_zero, index=False)
    missing_df.to_csv(out_missing, index=False)

    labels_payload = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    labels_payload["labels"] = LABELS
    out_labels.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "seed_segment_rows": int(len(seed_df)),
        "raw_expanded_segment_rows": int(len(raw_segments)),
        "final_combined_segment_rows": int(len(final_df)),
        "raw_parent_labels_used": int(len(parent_labels)),
        "zero_active_corrected_needs_review_excluded": int(len(zero_corrected)),
        "missing_parent_segment_groups": int(len(missing_df)),
        "training_group_counts": final_df["training_group"].value_counts().to_dict(),
        "split_counts": final_df["split"].value_counts().to_dict(),
        "final_label_counts": {lab: int(final_df[lab].sum()) for lab in LABELS},
        "outputs": {
            "final_manifest": str(out_manifest),
            "raw_parent_labels_used": str(out_parent_labels),
            "zero_active_excluded": str(out_zero),
            "missing_parent_segments": str(out_missing),
            "labels_json": str(out_labels),
            "summary_json": str(out_summary_json),
            "summary_md": str(out_summary_md),
        },
        "training_features_root": ".",
        "important_rule": "Final raw holdout is not included. It remains only for final evaluation after manual labelling.",
    }

    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(out_summary_md, summary)

    print("")
    print("Final expanded training manifest created")
    print("-" * 90)
    print(f"Seed segment rows:       {len(seed_df)}")
    print(f"Raw expanded rows:       {len(raw_segments)}")
    print(f"Final combined rows:     {len(final_df)}")
    print(f"Zero-active excluded:    {len(zero_corrected)}")
    print(f"Missing parent segments: {len(missing_df)}")
    print("")
    print("Training group counts:")
    print(final_df["training_group"].value_counts().to_string())
    print("")
    print("Label counts:")
    for lab in LABELS:
        print(f"  {lab:28s}: {int(final_df[lab].sum())}")
    print("")
    print(f"Final manifest: {out_manifest}")
    print(f"Summary:        {out_summary_md}")
    print("")
    print("Use this FeaturesRoot when training:")
    print("  .")


if __name__ == "__main__":
    main()