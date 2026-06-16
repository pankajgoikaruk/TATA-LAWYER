#!/usr/bin/env python
"""
Build a non-destructive v0.9 human-reviewed masked feature manifest.

For the 1,018 manually reviewed low-energy segments:
  - review value 1 -> target 1, mask 1
  - review value 0 -> target 0, mask 1
  - review value -1 -> target placeholder 0, mask 0

For all pre-existing trusted rows:
  - retain current binary targets
  - set every label mask to 1

The script never modifies the source recovered manifest or feature files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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

NINE_REVIEW_LABELS = LABELS[:-1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build v0.9 human-reviewed masked training manifest."
    )
    p.add_argument("--source_manifest", type=Path, required=True)
    p.add_argument("--review_csv", type=Path, required=True)
    p.add_argument("--output_manifest", type=Path, required=True)
    p.add_argument("--reports_dir", type=Path, required=True)
    p.add_argument("--expected_review_rows", type=int, default=1018)
    p.add_argument("--overwrite", action="store_true")
    return p


def normalise_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def numeric_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.isin([0, 1])
    if invalid.any():
        examples = series.loc[invalid].head(10).tolist()
        raise ValueError(f"{name} must contain only 0/1. Examples: {examples}")
    return values.astype(np.int8)


def numeric_tristate(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.isin([-1, 0, 1])
    if invalid.any():
        examples = series.loc[invalid].head(10).tolist()
        raise ValueError(f"{name} must contain only -1/0/1. Examples: {examples}")
    return values.astype(np.int8)


def rounded_time(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("start_sec/end_sec contain non-numeric values.")
    return values.round(3)


def main() -> int:
    args = parser().parse_args()

    source_manifest = args.source_manifest.expanduser().resolve()
    review_csv = args.review_csv.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()

    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found:\n{source_manifest}")
    if not review_csv.is_file():
        raise FileNotFoundError(f"Review CSV not found:\n{review_csv}")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists:\n{output_manifest}\n"
            "Use --overwrite only when intentionally rebuilding it."
        )

    manifest = pd.read_csv(source_manifest, low_memory=False)
    review = pd.read_csv(review_csv, low_memory=False)

    if len(review) != int(args.expected_review_rows):
        raise ValueError(
            f"Expected {args.expected_review_rows} reviewed rows, found {len(review)}."
        )

    required_manifest = {"clip_id", "split", "start_sec", "end_sec", *LABELS}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"Source manifest missing columns: {sorted(missing_manifest)}"
        )

    required_review = {
        "candidate_id",
        "clip_id",
        "split",
        "start_sec",
        "end_sec",
        "review_9label_status",
        "review_silence_present",
        *[f"review_{label}" for label in NINE_REVIEW_LABELS],
    }
    missing_review = required_review - set(review.columns)
    if missing_review:
        raise ValueError(f"Review CSV missing columns: {sorted(missing_review)}")

    # Validate source binary targets.
    for label in LABELS:
        manifest[label] = numeric_binary(manifest[label], label)

    # Validate review completion.
    status = normalise_id(review["review_9label_status"]).str.lower()
    incomplete = status.ne("reviewed")
    if incomplete.any():
        examples = review.loc[
            incomplete,
            ["candidate_id", "clip_id", "review_9label_status"],
        ].head(10)
        raise ValueError(
            f"{int(incomplete.sum())} rows are not marked reviewed.\n"
            f"{examples.to_string(index=False)}"
        )

    # Validate review labels.
    reviewed_values: dict[str, pd.Series] = {}
    for label in NINE_REVIEW_LABELS:
        col = f"review_{label}"
        reviewed_values[label] = numeric_tristate(review[col], col)
    reviewed_values["silence_present"] = numeric_binary(
        review["review_silence_present"],
        "review_silence_present",
    )

    review = review.copy()
    review["candidate_id"] = normalise_id(review["candidate_id"])
    review["clip_id"] = normalise_id(review["clip_id"])
    review["split"] = normalise_id(review["split"]).str.lower()
    review["_start_key"] = rounded_time(review["start_sec"])
    review["_end_key"] = rounded_time(review["end_sec"])

    if review["candidate_id"].eq("").any():
        raise ValueError("Review CSV contains blank candidate_id.")
    if review["candidate_id"].duplicated().any():
        dups = review.loc[
            review["candidate_id"].duplicated(False),
            ["candidate_id", "clip_id", "start_sec", "end_sec"],
        ].head(20)
        raise ValueError(f"Duplicate review candidate_id values:\n{dups}")

    # Initialise masks and provenance.
    out = manifest.copy()
    for label in LABELS:
        out[f"mask_{label}"] = np.int8(1)

    out["v09_masked_review_applied"] = np.int8(0)
    out["v09_review_candidate_id"] = ""
    out["v09_review_has_unknown"] = np.int8(0)
    out["v09_review_known_label_count"] = np.int8(len(LABELS))
    out["v09_review_unknown_label_count"] = np.int8(0)
    out["v09_evaluation_group"] = "original_trusted"
    out["v09_checkpoint_eligible"] = np.int8(
        out["split"].astype(str).str.lower().eq("val")
    )
    out["v09_standard_test_eligible"] = np.int8(
        out["split"].astype(str).str.lower().eq("test")
    )

    out["_clip_key"] = normalise_id(out["clip_id"])
    out["_split_key"] = normalise_id(out["split"]).str.lower()
    out["_start_key"] = rounded_time(out["start_sec"])
    out["_end_key"] = rounded_time(out["end_sec"])

    # Prefer exact candidate-id matching. Fall back to segment identity.
    candidate_col = None
    for possible in (
        "recovery_candidate_id",
        "candidate_id",
        "v09_recovery_candidate_id",
    ):
        if possible in out.columns:
            candidate_col = possible
            break

    candidate_to_rows: dict[str, list[int]] = {}
    if candidate_col is not None:
        for idx, cid in normalise_id(out[candidate_col]).items():
            if cid:
                candidate_to_rows.setdefault(cid, []).append(int(idx))

    key_to_rows: dict[tuple[str, str, float, float], list[int]] = {}
    for idx, row in out[
        ["_clip_key", "_split_key", "_start_key", "_end_key"]
    ].iterrows():
        key = (
            str(row["_clip_key"]),
            str(row["_split_key"]),
            float(row["_start_key"]),
            float(row["_end_key"]),
        )
        key_to_rows.setdefault(key, []).append(int(idx))

    matched_manifest_indices: list[int] = []
    match_methods: list[str] = []

    for r_idx, r in review.iterrows():
        candidate_id = str(r["candidate_id"])
        matches = candidate_to_rows.get(candidate_id, [])
        method = "candidate_id"

        if len(matches) != 1:
            key = (
                str(r["clip_id"]),
                str(r["split"]),
                float(r["_start_key"]),
                float(r["_end_key"]),
            )
            matches = key_to_rows.get(key, [])
            method = "clip_split_start_end"

        if len(matches) != 1:
            raise ValueError(
                "Could not uniquely match reviewed row to source manifest:\n"
                f"candidate_id={candidate_id}, clip_id={r['clip_id']}, "
                f"split={r['split']}, start={r['start_sec']}, end={r['end_sec']}, "
                f"matches={matches}"
            )

        m_idx = int(matches[0])
        matched_manifest_indices.append(m_idx)
        match_methods.append(method)

        unknown_count = 0
        for label in LABELS:
            value = int(reviewed_values[label].iloc[r_idx])
            mask_col = f"mask_{label}"

            if value == -1:
                # Placeholder target is ignored because mask=0.
                out.at[m_idx, label] = np.int8(0)
                out.at[m_idx, mask_col] = np.int8(0)
                unknown_count += 1
            else:
                out.at[m_idx, label] = np.int8(value)
                out.at[m_idx, mask_col] = np.int8(1)

        out.at[m_idx, "v09_masked_review_applied"] = np.int8(1)
        out.at[m_idx, "v09_review_candidate_id"] = candidate_id
        out.at[m_idx, "v09_review_has_unknown"] = np.int8(unknown_count > 0)
        out.at[m_idx, "v09_review_known_label_count"] = np.int8(
            len(LABELS) - unknown_count
        )
        out.at[m_idx, "v09_review_unknown_label_count"] = np.int8(unknown_count)
        out.at[m_idx, "v09_evaluation_group"] = "recovered_human_reviewed"

        # Recovered rows are excluded from the strict original-v0.9 checkpoint
        # and standard test comparison. They remain available for masked metrics.
        out.at[m_idx, "v09_checkpoint_eligible"] = np.int8(0)
        out.at[m_idx, "v09_standard_test_eligible"] = np.int8(0)

    if len(set(matched_manifest_indices)) != len(review):
        raise ValueError(
            "Multiple reviewed rows matched the same manifest row. "
            "The build has been stopped."
        )

    # Final safety checks.
    reviewed_mask = out["v09_masked_review_applied"].eq(1)
    if int(reviewed_mask.sum()) != len(review):
        raise RuntimeError(
            f"Expected {len(review)} reviewed manifest rows, "
            f"found {int(reviewed_mask.sum())}."
        )

    for label in LABELS:
        if not set(out[label].dropna().astype(int).unique()).issubset({0, 1}):
            raise RuntimeError(f"Final target column {label} is not binary.")
        if not set(out[f"mask_{label}"].dropna().astype(int).unique()).issubset(
            {0, 1}
        ):
            raise RuntimeError(f"Final mask column mask_{label} is not binary.")

    checkpoint_val_rows = int(
        (
            out["split"].astype(str).str.lower().eq("val")
            & out["v09_checkpoint_eligible"].eq(1)
        ).sum()
    )
    standard_test_rows = int(
        (
            out["split"].astype(str).str.lower().eq("test")
            & out["v09_standard_test_eligible"].eq(1)
        ).sum()
    )

    # Remove temporary matching keys.
    out = out.drop(
        columns=["_clip_key", "_split_key", "_start_key", "_end_key"],
        errors="ignore",
    )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_manifest, index=False, encoding="utf-8-sig")

    unknown_by_label = {
        label: int(
            (
                out["v09_masked_review_applied"].eq(1)
                & out[f"mask_{label}"].eq(0)
            ).sum()
        )
        for label in LABELS
    }

    reviewed_label_summary = []
    for label in LABELS:
        reviewed_rows = out.loc[reviewed_mask]
        known = reviewed_rows[f"mask_{label}"].eq(1)
        reviewed_label_summary.append(
            {
                "label": label,
                "reviewed_known": int(known.sum()),
                "reviewed_unknown": int((~known).sum()),
                "reviewed_positive": int(
                    (known & reviewed_rows[label].eq(1)).sum()
                ),
                "reviewed_negative": int(
                    (known & reviewed_rows[label].eq(0)).sum()
                ),
            }
        )

    pd.DataFrame(reviewed_label_summary).to_csv(
        reports_dir / "v09_masked_reviewed_label_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_summary = pd.DataFrame(
        {
            "candidate_id": review["candidate_id"].tolist(),
            "clip_id": review["clip_id"].tolist(),
            "split": review["split"].tolist(),
            "start_sec": review["start_sec"].tolist(),
            "end_sec": review["end_sec"].tolist(),
            "match_method": match_methods,
            "manifest_row_index_zero_based": matched_manifest_indices,
        }
    )
    match_summary.to_csv(
        reports_dir / "v09_masked_review_match_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "source_manifest": str(source_manifest),
        "review_csv": str(review_csv),
        "output_manifest": str(output_manifest),
        "source_rows": int(len(manifest)),
        "output_rows": int(len(out)),
        "review_rows": int(len(review)),
        "review_rows_matched": int(reviewed_mask.sum()),
        "review_rows_fully_known": int(
            (
                reviewed_mask
                & out["v09_review_unknown_label_count"].eq(0)
            ).sum()
        ),
        "review_rows_partially_known": int(
            (
                reviewed_mask
                & out["v09_review_unknown_label_count"].gt(0)
            ).sum()
        ),
        "unknown_by_label": unknown_by_label,
        "split_counts": {
            str(k): int(v)
            for k, v in out["split"].astype(str).value_counts().to_dict().items()
        },
        "strict_checkpoint_val_rows": checkpoint_val_rows,
        "strict_standard_test_rows": standard_test_rows,
        "candidate_id_match_count": int(
            sum(method == "candidate_id" for method in match_methods)
        ),
        "fallback_segment_match_count": int(
            sum(method == "clip_split_start_end" for method in match_methods)
        ),
        "source_files_modified": False,
    }

    with (reports_dir / "v09_masked_manifest_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== v0.9 human-reviewed masked manifest ===")
    print(f"Source rows:                   {len(manifest):,}")
    print(f"Output rows:                   {len(out):,}")
    print(f"Reviewed rows matched:         {int(reviewed_mask.sum()):,}")
    print(
        "Fully known reviewed rows:     "
        f"{summary['review_rows_fully_known']:,}"
    )
    print(
        "Partially known reviewed rows: "
        f"{summary['review_rows_partially_known']:,}"
    )
    print(f"Strict checkpoint val rows:    {checkpoint_val_rows:,}")
    print(f"Strict standard test rows:     {standard_test_rows:,}")
    print(f"Candidate-ID matches:          {summary['candidate_id_match_count']:,}")
    print(f"Fallback segment matches:      {summary['fallback_segment_match_count']:,}")
    print(f"Output manifest:               {output_manifest}")
    print(f"Reports:                       {reports_dir}")
    print("Source files modified:          No")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
