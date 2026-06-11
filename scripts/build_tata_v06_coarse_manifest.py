# scripts/build_tata_v06_coarse_manifest.py

"""
Build TATA v0.6 coarse audience-reaction manifests.

Purpose:
- Convert the v0.5/v0.5_tata_2 12-label manifests into a 10-core-label design.
- Merge:
    applause_present
    laughter_present
    crowd_cheer_present
  into:
    audience_reaction_present

This does NOT change audio files or features.
It only creates new v0.6 metadata/manifest files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_SPEAKER_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

V06_LABELS = [
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

AUDIENCE_FINE_LABELS = [
    "applause_present",
    "laughter_present",
    "crowd_cheer_present",
]

OLD_12_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "applause_present",
    "laughter_present",
    "crowd_cheer_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_binary(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * len(df), index=df.index, dtype=int)

    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )


def fine_components(row: pd.Series) -> str:
    active = []
    for lab in AUDIENCE_FINE_LABELS:
        if int(row.get(lab, 0)) == 1:
            active.append(lab)
    return "|".join(active)


def refresh_label_text(df: pd.DataFrame) -> pd.DataFrame:
    active_text = []
    active_count = []

    for _, row in df.iterrows():
        active = [lab for lab in V06_LABELS if int(row.get(lab, 0)) == 1]
        active_text.append("|".join(active))
        active_count.append(len(active))

    df["labels"] = active_text
    df["num_active_labels"] = active_count
    return df


def convert_to_v06(df: pd.DataFrame, keep_fine_metadata: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Normalise old/fine label columns.
    for lab in OLD_12_LABELS:
        df[lab] = ensure_binary(df, lab)

    # Create coarse audience label.
    audience = pd.Series([0] * len(df), index=df.index, dtype=int)
    for lab in AUDIENCE_FINE_LABELS:
        audience = audience | ensure_binary(df, lab)

    df["audience_reaction_present"] = audience.astype(int)

    if keep_fine_metadata:
        df["fine_audience_components"] = df.apply(fine_components, axis=1)
        for lab in AUDIENCE_FINE_LABELS:
            df[f"fine_{lab}"] = ensure_binary(df, lab)

    # Drop fine label columns as active training labels.
    df = df.drop(columns=[c for c in AUDIENCE_FINE_LABELS if c in df.columns])

    # Ensure all v0.6 core columns exist and are binary.
    for lab in V06_LABELS:
        df[lab] = ensure_binary(df, lab)

    df = refresh_label_text(df)

    # Nice column order.
    preferred = [
        "sample_id",
        "clip_id",
        "parent_clip_id",
        "file_path",
        "abs_path",
        "rel_path",
        "segment_wav_relpath",
        "feat_relpath",
        "feature_path",
        "split",
        "primary_label",
        "labels",
        "num_active_labels",
    ]

    existing_preferred = [c for c in preferred if c in df.columns]
    remaining = [
        c for c in df.columns
        if c not in existing_preferred and c not in V06_LABELS
    ]

    df = df[existing_preferred + V06_LABELS + remaining]

    return df


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# TATA v0.6 Coarse Audience-Reaction Manifest Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Branch")
    lines.append("")
    lines.append("`agentic_data_preprocessing_v0.6`")
    lines.append("")
    lines.append("## Label Design")
    lines.append("")
    lines.append("This v0.6 experiment merges the fine audience-event labels:")
    lines.append("")
    lines.append("- `applause_present`")
    lines.append("- `laughter_present`")
    lines.append("- `crowd_cheer_present`")
    lines.append("")
    lines.append("into one coarse label:")
    lines.append("")
    lines.append("- `audience_reaction_present`")
    lines.append("")
    lines.append("## Core Labels")
    lines.append("")
    lines.append("| Label | Clip positives | Segment positives | Feature positives |")
    lines.append("|---|---:|---:|---:|")

    for lab in V06_LABELS:
        lines.append(
            f"| `{lab}` | "
            f"{summary['clip_label_counts'].get(lab, 'N/A')} | "
            f"{summary['segment_label_counts'].get(lab, 'N/A')} | "
            f"{summary['feature_label_counts'].get(lab, 'N/A')} |"
        )

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- `{k}`: `{v}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {lab: int(df[lab].sum()) for lab in V06_LABELS if lab in df.columns}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TATA v0.6 coarse audience-reaction manifests."
    )

    parser.add_argument(
        "--clip_manifest",
        default="human_talk_workspace/tata_2/metadata/tata_clip_level_manifest_training_ready.csv",
    )
    parser.add_argument(
        "--segment_manifest",
        default="human_talk_workspace/tata_2/segment_cache/metadata/tata_segment_manifest.csv",
    )
    parser.add_argument(
        "--feature_manifest",
        default="human_talk_workspace/tata_2/feature_cache/metadata/multilabel_features_manifest.csv",
    )
    parser.add_argument(
        "--out_root",
        default="human_talk_workspace/tata_v0.6",
    )

    args = parser.parse_args()

    clip_manifest = Path(args.clip_manifest)
    segment_manifest = Path(args.segment_manifest)
    feature_manifest = Path(args.feature_manifest)
    out_root = Path(args.out_root)

    if not clip_manifest.exists():
        raise FileNotFoundError(f"Clip manifest not found: {clip_manifest}")
    if not segment_manifest.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest}")
    if not feature_manifest.exists():
        raise FileNotFoundError(f"Feature manifest not found: {feature_manifest}")

    metadata_dir = out_root / "metadata"
    segment_meta_dir = out_root / "segment_cache" / "metadata"
    feature_meta_dir = out_root / "feature_cache" / "metadata"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    segment_meta_dir.mkdir(parents=True, exist_ok=True)
    feature_meta_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("Building TATA v0.6 coarse audience-reaction manifests")
    print("-" * 90)
    print(f"Clip manifest:    {clip_manifest}")
    print(f"Segment manifest: {segment_manifest}")
    print(f"Feature manifest: {feature_manifest}")
    print(f"Output root:      {out_root}")
    print("-" * 90)

    clip_df = convert_to_v06(pd.read_csv(clip_manifest), keep_fine_metadata=True)
    seg_df = convert_to_v06(pd.read_csv(segment_manifest), keep_fine_metadata=True)
    feat_df = convert_to_v06(pd.read_csv(feature_manifest), keep_fine_metadata=True)

    out_clip = metadata_dir / "tata_v06_clip_level_manifest_training_ready.csv"
    out_segment = segment_meta_dir / "tata_v06_segment_manifest.csv"
    out_feature = feature_meta_dir / "multilabel_features_manifest.csv"
    out_labels = metadata_dir / "tata_v06_labels.json"
    out_segment_labels = segment_meta_dir / "tata_labels.json"
    out_summary_json = metadata_dir / "tata_v06_coarse_summary.json"
    out_summary_md = metadata_dir / "tata_v06_coarse_summary.md"

    clip_df.to_csv(out_clip, index=False)
    seg_df.to_csv(out_segment, index=False)
    feat_df.to_csv(out_feature, index=False)

    labels_payload = {
        "branch": "agentic_data_preprocessing_v0.6",
        "task": "tiny_audio_triage",
        "labeling_level": "coarse_audience_reaction",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": V06_LABELS,
        "target_speaker_labels": TARGET_SPEAKER_LABELS,
        "non_target_speech_labels": ["other_speaker_present"],
        "event_background_labels": [
            "music_present",
            "audience_reaction_present",
            "silence_present",
        ],
        "merged_from": {
            "audience_reaction_present": AUDIENCE_FINE_LABELS,
        },
    }

    write_json(out_labels, labels_payload)
    write_json(out_segment_labels, labels_payload)

    summary = {
        "generated_at": now_iso(),
        "branch": "agentic_data_preprocessing_v0.6",
        "inputs": {
            "clip_manifest": str(clip_manifest),
            "segment_manifest": str(segment_manifest),
            "feature_manifest": str(feature_manifest),
        },
        "outputs": {
            "clip_manifest": str(out_clip),
            "segment_manifest": str(out_segment),
            "feature_manifest": str(out_feature),
            "labels_json": str(out_labels),
            "segment_labels_json": str(out_segment_labels),
            "summary_json": str(out_summary_json),
            "summary_md": str(out_summary_md),
        },
        "rows": {
            "clip_rows": int(len(clip_df)),
            "segment_rows": int(len(seg_df)),
            "feature_rows": int(len(feat_df)),
        },
        "clip_label_counts": label_counts(clip_df),
        "segment_label_counts": label_counts(seg_df),
        "feature_label_counts": label_counts(feat_df),
    }

    write_json(out_summary_json, summary)
    write_md(out_summary_md, summary)

    print("")
    print("TATA v0.6 manifest build complete")
    print("-" * 90)
    print(f"Clip rows:    {len(clip_df)}")
    print(f"Segment rows: {len(seg_df)}")
    print(f"Feature rows: {len(feat_df)}")
    print("")
    print("Feature label counts:")
    for lab in V06_LABELS:
        print(f"  {lab:28s}: {int(feat_df[lab].sum())}")
    print("")
    print(f"Labels JSON: {out_labels}")
    print(f"Summary MD:  {out_summary_md}")


if __name__ == "__main__":
    main()