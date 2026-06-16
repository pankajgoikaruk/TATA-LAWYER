#!/usr/bin/env python
"""
Build the v0.9 nine-label manual-review CSV from the completed
low-energy silence review queue.

Tri-state annotation:
    1  = confidently present
    0  = confidently absent
   -1  = reviewed but uncertain / unknown
 blank = not reviewed yet

The existing review_silence_present value is preserved as trusted human
ground truth and is not one of the editable nine-label columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


NINE_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "audience_reaction_present",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the v0.9 nine-label manual-review CSV."
    )
    default_repo = Path(__file__).resolve().parents[1]
    default_review_root = (
        default_repo
        / "human_talk_workspace"
        / "tata_v0.9_pipeline"
        / "tata_triage_model"
        / "manual_review"
        / "low_energy_recovery_v09"
    )

    parser.add_argument(
        "--source_csv",
        type=Path,
        default=default_review_root / "low_energy_silence_review_queue_v09.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=default_review_root / "low_energy_9label_manual_review_v09.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV.",
    )
    parser.add_argument(
        "--no_expected_count_check",
        action="store_true",
        help="Skip the expected 1,018 / 271 / 747 count checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    source_csv = args.source_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not source_csv.is_file():
        raise FileNotFoundError(f"Source review CSV not found:\n{source_csv}")

    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists:\n{output_csv}\n"
            "Use --overwrite only when you intentionally want to rebuild it."
        )

    df = pd.read_csv(source_csv, low_memory=False)

    required = {
        "candidate_id",
        "clip_id",
        "split",
        "start_sec",
        "end_sec",
        "review_audio_file",
        "review_silence_present",
        "review_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")

    if df["candidate_id"].astype(str).duplicated().any():
        examples = (
            df.loc[
                df["candidate_id"].astype(str).duplicated(),
                "candidate_id",
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(f"Duplicate candidate_id values: {examples}")

    status = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    not_reviewed = status.ne("reviewed")
    if not_reviewed.any():
        raise ValueError(
            f"{int(not_reviewed.sum())} rows are not marked reviewed "
            "in the silence review queue."
        )

    silence = pd.to_numeric(
        df["review_silence_present"],
        errors="coerce",
    )
    invalid_silence = silence.isna() | ~silence.isin([0, 1])
    if invalid_silence.any():
        examples = (
            df.loc[
                invalid_silence,
                ["candidate_id", "review_silence_present"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "review_silence_present must contain only 0 or 1. "
            f"Examples: {examples}"
        )
    df["review_silence_present"] = silence.astype(int)

    positive_count = int((df["review_silence_present"] == 1).sum())
    negative_count = int((df["review_silence_present"] == 0).sum())

    if not args.no_expected_count_check:
        expected = {
            "rows": 1018,
            "silence_positive": 271,
            "silence_negative": 747,
        }
        actual = {
            "rows": int(len(df)),
            "silence_positive": positive_count,
            "silence_negative": negative_count,
        }
        if actual != expected:
            raise ValueError(
                "Unexpected reviewed queue counts.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                "Use --no_expected_count_check only after confirming the source."
            )

    # Preserve the earlier silence-review fields under explicit names.
    df = df.rename(
        columns={
            "review_status": "silence_review_status",
            "review_notes": "silence_review_notes",
            "review_keep_segment": "silence_review_keep_segment_original",
        }
    )

    # Add a directly usable path to each exported one-second review WAV.
    review_audio_root = source_csv.parent / "audio"
    df["review_audio_path"] = [
        str((review_audio_root / str(name)).resolve())
        for name in df["review_audio_file"].astype(str)
    ]

    # New editable tri-state columns. Blank means not reviewed yet.
    for label in NINE_LABELS:
        df[f"review_{label}"] = pd.NA

    df["review_9label_status"] = "pending"
    df["review_9label_notes"] = ""
    df["review_unknown_reason"] = ""

    # Prioritise trusted evaluation rows first.
    split_priority = {"test": 1, "val": 2, "train": 3}
    df["review_order_priority"] = (
        df["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(split_priority)
        .fillna(9)
        .astype(int)
    )

    sort_columns = [
        "review_order_priority",
        "clip_id",
        "start_sec",
        "candidate_id",
    ]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    # Put review-facing columns first; retain all source/provenance columns after them.
    front = [
        "review_order_priority",
        "candidate_id",
        "clip_id",
        "split",
        "start_sec",
        "end_sec",
        "review_audio_file",
        "review_audio_path",
        "audio_path",
        "review_silence_present",
        "silence_review_status",
    ]

    editable = [f"review_{label}" for label in NINE_LABELS]
    review_admin = [
        "review_9label_status",
        "review_9label_notes",
        "review_unknown_reason",
    ]

    front = [column for column in front if column in df.columns]
    ordered = front + editable + review_admin
    remaining = [column for column in df.columns if column not in ordered]
    df = df[ordered + remaining]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n=== v0.9 nine-label manual-review CSV ===")
    print(f"Rows:                         {len(df):,}")
    print(f"Trusted silence positives:    {positive_count:,}")
    print(f"Trusted silence negatives:    {negative_count:,}")
    print(f"Editable label columns:       {len(NINE_LABELS)}")
    print(f"Test rows first:              {(df['split'].astype(str).str.lower() == 'test').sum():,}")
    print(f"Validation rows second:       {(df['split'].astype(str).str.lower() == 'val').sum():,}")
    print(f"Training rows last:           {(df['split'].astype(str).str.lower() == 'train').sum():,}")
    print(f"Saved:                        {output_csv}")
    print("\nAnnotation values: 1=present, 0=absent, -1=unknown, blank=not reviewed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
