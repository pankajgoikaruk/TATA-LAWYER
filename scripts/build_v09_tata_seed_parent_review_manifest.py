#!/usr/bin/env python
"""
Build the editable v0.9 TATA parent-level review manifest.

This script:
1. Reads the immutable 2,074-row tata_2 parent baseline.
2. Reads the immutable 12,469-row feature manifest to recover the original
   parent-level train/val/test split without leakage.
3. Converts the old 12-label schema into the canonical 10-label schema:
      applause OR laughter OR crowd_cheer -> audience_reaction_present
4. Preserves every original baseline column unchanged.
5. Adds editable v09_* label, split, review, and provenance columns.
6. Creates a rare-event priority queue and an auditable build summary.

It never modifies the baseline files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_LABELS = [
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

FINE_AUDIENCE_LABELS = [
    "applause_present",
    "laughter_present",
    "crowd_cheer_present",
]


def parse_args() -> argparse.Namespace:
    script_repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build the v0.9 editable TATA seed parent review manifest."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=script_repo_root,
        help="TATA-LAWYER repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--expected_parent_rows",
        type=int,
        default=2074,
        help="Expected number of parent rows.",
    )
    parser.add_argument(
        "--expected_feature_rows",
        type=int,
        default=12469,
        help="Expected number of segment-feature rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated Step 2 outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def require_columns(
    frame: pd.DataFrame, required: list[str], description: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {missing}"
        )


def coerce_binary(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            bad_count = int(converted.isna().sum())
            raise ValueError(
                f"{description}.{column} contains {bad_count} non-numeric values."
            )

        invalid = ~converted.isin([0, 1])
        if invalid.any():
            examples = converted.loc[invalid].head(5).tolist()
            raise ValueError(
                f"{description}.{column} is not binary. Examples: {examples}"
            )

        frame[column] = converted.astype("int8")


def ensure_output_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Generated output already exists:\n{path}\n"
            "Re-run with --overwrite only when you intentionally want to rebuild it."
        )


def build_parent_split_map(features: pd.DataFrame) -> pd.Series:
    split_counts = features.groupby("source_file", dropna=False)["split"].nunique()
    leaked = split_counts[split_counts != 1]
    if not leaked.empty:
        raise ValueError(
            "Parent split leakage or missing split values were detected for "
            f"{len(leaked)} source files."
        )

    split_map = (
        features[["source_file", "split"]]
        .drop_duplicates()
        .set_index("source_file")["split"]
    )

    valid_splits = {"train", "val", "test"}
    observed = set(split_map.dropna().astype(str).unique())
    unexpected = sorted(observed - valid_splits)
    if unexpected:
        raise ValueError(f"Unexpected split values were found: {unexpected}")

    return split_map


def active_label_string(row: pd.Series, prefixed_labels: list[str]) -> str:
    active = [
        label.removeprefix("v09_")
        for label in prefixed_labels
        if int(row[label]) == 1
    ]
    return "|".join(active)


def review_reason(row: pd.Series) -> str:
    audience = int(row["v09_audience_reaction_present"]) == 1
    silence = int(row["v09_silence_present"]) == 1
    other = int(row["v09_other_speaker_present"]) == 1

    if audience and silence:
        return "verify_audience_reaction_and_silence"
    if silence:
        return "verify_silence_label"
    if audience:
        return "verify_transient_audience_reaction"
    if other:
        return "verify_other_speaker_context"
    return "standard_seed_review"


def review_priority(row: pd.Series) -> int:
    if int(row["v09_silence_present"]) == 1:
        return 1
    if int(row["v09_audience_reaction_present"]) == 1:
        return 2
    if int(row["v09_other_speaker_present"]) == 1:
        return 3
    return 4


def serialisable_counts(series: pd.Series) -> dict[str, int]:
    return OrderedDict(
        (str(index), int(value))
        for index, value in series.sort_index().items()
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    v09_root = repo_root / "human_talk_workspace" / "tata_v0.9_pipeline"

    parent_baseline = (
        v09_root
        / "tata_triage_model"
        / "metadata"
        / "tata2_parent_manifest_12label_2074_BASELINE.csv"
    )
    feature_baseline = (
        v09_root
        / "tata_triage_model"
        / "feature_cache"
        / "metadata"
        / "tata_seed_features_manifest_10label_12469_BASELINE.csv"
    )
    schema_path = (
        v09_root / "shared" / "human_talk_10label_schema.json"
    )

    output_manifest = (
        v09_root
        / "tata_triage_model"
        / "metadata"
        / "tata_seed_parent_manifest_v09_REVIEW.csv"
    )
    rare_event_queue = (
        v09_root
        / "tata_triage_model"
        / "metadata"
        / "tata_seed_parent_rare_event_review_v09.csv"
    )
    summary_path = (
        v09_root
        / "shared"
        / "correction_ledgers"
        / "v09_seed_parent_review_build_summary.json"
    )
    label_counts_path = (
        v09_root
        / "shared"
        / "correction_ledgers"
        / "v09_seed_parent_initial_label_counts.csv"
    )

    require_file(parent_baseline, "Frozen parent baseline")
    require_file(feature_baseline, "Frozen feature baseline")
    require_file(schema_path, "Canonical 10-label schema")

    for output in [
        output_manifest,
        rare_event_queue,
        summary_path,
        label_counts_path,
    ]:
        ensure_output_available(output, args.overwrite)
        output.parent.mkdir(parents=True, exist_ok=True)

    print("[STEP 2] Loading immutable v0.9 baselines...")
    parents = pd.read_csv(parent_baseline, low_memory=False)
    features = pd.read_csv(feature_baseline, low_memory=False)

    with schema_path.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)

    if len(parents) != args.expected_parent_rows:
        raise ValueError(
            f"Expected {args.expected_parent_rows} parent rows, found {len(parents)}."
        )
    if len(features) != args.expected_feature_rows:
        raise ValueError(
            f"Expected {args.expected_feature_rows} feature rows, found {len(features)}."
        )

    require_columns(
        parents,
        [
            "clip_id",
            "file_name",
            "file_path",
            "source_group",
            "source_subfolder",
            "primary_label",
            *EXPECTED_LABELS[:-2],
            *FINE_AUDIENCE_LABELS,
            "silence_present",
        ],
        "Parent baseline",
    )
    require_columns(
        features,
        ["parent_clip_id", "source_file", "split"],
        "Feature baseline",
    )

    schema_labels = schema.get("labels")
    if schema_labels != EXPECTED_LABELS:
        raise ValueError(
            "The shared schema labels do not match the expected canonical "
            f"10-label order.\nExpected: {EXPECTED_LABELS}\nFound: {schema_labels}"
        )

    original_columns = parents.columns.tolist()
    if parents["file_name"].duplicated().any():
        duplicates = parents.loc[
            parents["file_name"].duplicated(keep=False), "file_name"
        ].head(10).tolist()
        raise ValueError(f"Duplicate parent file names found: {duplicates}")

    binary_source_columns = [
        "Brene_Brown",
        "Eckhart_Tolle",
        "Eric_Thomas",
        "Gary_Vee",
        "Jay_Shetty",
        "Nick_Vujicic",
        "other_speaker_present",
        "music_present",
        *FINE_AUDIENCE_LABELS,
        "silence_present",
    ]
    coerce_binary(parents, binary_source_columns, "Parent baseline")

    split_map = build_parent_split_map(features)

    parent_files = set(parents["file_name"].astype(str))
    feature_files = set(split_map.index.astype(str))
    missing_from_features = sorted(parent_files - feature_files)
    unexpected_feature_parents = sorted(feature_files - parent_files)

    if missing_from_features or unexpected_feature_parents:
        raise ValueError(
            "The frozen parent and feature manifests do not contain the same "
            "2,074 source files.\n"
            f"Missing from feature manifest: {len(missing_from_features)}\n"
            f"Unexpected feature parents: {len(unexpected_feature_parents)}"
        )

    print("[STEP 2] Building canonical 10-label editable columns...")
    output = parents.copy()

    output["v09_split"] = output["file_name"].map(split_map)
    if output["v09_split"].isna().any():
        raise ValueError("At least one parent could not be assigned a v0.9 split.")

    for label in EXPECTED_LABELS:
        if label == "audience_reaction_present":
            output[f"v09_{label}"] = (
                output[FINE_AUDIENCE_LABELS].max(axis=1).astype("int8")
            )
        else:
            output[f"v09_{label}"] = output[label].astype("int8")

    v09_label_columns = [f"v09_{label}" for label in EXPECTED_LABELS]
    output["v09_num_active_labels"] = (
        output[v09_label_columns].sum(axis=1).astype("int16")
    )

    if (output["v09_num_active_labels"] < 1).any():
        bad = output.loc[
            output["v09_num_active_labels"] < 1,
            ["clip_id", "file_name", "source_group", "source_subfolder"],
        ].head(10)
        raise ValueError(
            "Canonical conversion produced zero-active-label rows:\n"
            f"{bad.to_string(index=False)}"
        )

    output["v09_labels"] = output.apply(
        active_label_string, axis=1, prefixed_labels=v09_label_columns
    )

    output["v09_review_priority_rank"] = output.apply(
        review_priority, axis=1
    ).astype("int8")
    output["v09_review_reason"] = output.apply(review_reason, axis=1)
    output["v09_review_status"] = "pending"
    output["v09_review_action"] = ""
    output["v09_review_notes"] = ""
    output["v09_reviewer"] = ""
    output["v09_reviewed_utc"] = ""
    output["v09_label_changed"] = 0
    output["v09_keep_for_tata_training"] = 1
    output["v09_source_version"] = "tata2_12label_to_v06_10label"
    output["v09_manifest_role"] = "tata_triage_seed_parent_review"

    # Prove that all original columns survived the transformation.
    missing_original_columns = [
        column for column in original_columns if column not in output.columns
    ]
    if missing_original_columns:
        raise AssertionError(
            f"Original columns were lost: {missing_original_columns}"
        )

    rare_mask = (
        (output["v09_audience_reaction_present"] == 1)
        | (output["v09_silence_present"] == 1)
    )
    rare = output.loc[rare_mask].copy()
    rare = rare.sort_values(
        by=[
            "v09_review_priority_rank",
            "source_group",
            "source_subfolder",
            "file_name",
        ],
        kind="stable",
    )

    print("[STEP 2] Writing generated review files...")
    output.to_csv(output_manifest, index=False, encoding="utf-8")
    rare.to_csv(rare_event_queue, index=False, encoding="utf-8")

    label_counts = pd.DataFrame(
        {
            "label": EXPECTED_LABELS,
            "positive_parent_count": [
                int(output[f"v09_{label}"].sum()) for label in EXPECTED_LABELS
            ],
        }
    )
    label_counts.to_csv(label_counts_path, index=False, encoding="utf-8")

    split_parent_counts = serialisable_counts(
        output["v09_split"].value_counts()
    )
    split_segment_counts = serialisable_counts(
        features["split"].value_counts()
    )
    active_label_distribution = serialisable_counts(
        output["v09_num_active_labels"].value_counts()
    )

    summary = OrderedDict(
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        repo_root=str(repo_root),
        parent_baseline=str(parent_baseline),
        parent_baseline_sha256=sha256_file(parent_baseline),
        feature_baseline=str(feature_baseline),
        feature_baseline_sha256=sha256_file(feature_baseline),
        schema_path=str(schema_path),
        schema_sha256=sha256_file(schema_path),
        output_manifest=str(output_manifest),
        output_manifest_sha256=sha256_file(output_manifest),
        rare_event_queue=str(rare_event_queue),
        rare_event_queue_sha256=sha256_file(rare_event_queue),
        parent_rows=int(len(output)),
        unique_parent_files=int(output["file_name"].nunique()),
        feature_segment_rows=int(len(features)),
        feature_unique_parents=int(features["source_file"].nunique()),
        unmatched_parent_files=0,
        parent_split_leakage_groups=0,
        parent_split_counts=split_parent_counts,
        segment_split_counts=split_segment_counts,
        canonical_label_count=len(EXPECTED_LABELS),
        canonical_labels=EXPECTED_LABELS,
        audience_merge_rule="applause_present OR laughter_present OR crowd_cheer_present",
        rare_event_review_rows=int(len(rare)),
        audience_positive_parents=int(
            output["v09_audience_reaction_present"].sum()
        ),
        silence_positive_parents=int(output["v09_silence_present"].sum()),
        active_label_count_distribution=active_label_distribution,
        review_status_initial="pending",
        baseline_files_modified=False,
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n[COMPLETE] Step 2 generated the editable v0.9 parent review manifest.")
    print(f"Full review manifest: {output_manifest}")
    print(f"Rare-event queue:     {rare_event_queue}")
    print(f"Build summary:        {summary_path}")
    print(f"Label counts:         {label_counts_path}")
    print("\nVerified counts:")
    print(f"  Parent rows:         {len(output):,}")
    print(f"  Feature rows:        {len(features):,}")
    print(f"  Rare-event rows:     {len(rare):,}")
    print(f"  Audience positives:  {int(output['v09_audience_reaction_present'].sum()):,}")
    print(f"  Silence positives:   {int(output['v09_silence_present'].sum()):,}")
    print(f"  Parent splits:       {split_parent_counts}")
    print("\nNo baseline file was modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
