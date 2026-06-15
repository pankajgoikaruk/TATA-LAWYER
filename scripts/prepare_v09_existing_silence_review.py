#!/usr/bin/env python
"""
Prepare Step 3 of the TATA-LAWYER v0.9 workflow:
manual verification of all existing parent clips labelled silence_present=1.

The script does NOT change the v0.9 parent manifest. It creates:
  - a compact editable silence review CSV;
  - copied review audio files;
  - an audit summary JSON;
  - review instructions.

After manual review, a later script will apply the reviewed corrections safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Prepare the v0.9 existing-silence manual-review batch."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=default_repo,
        help="TATA-LAWYER repository root.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help=(
            "Root containing human_talk_tata_seed_dataset. "
            "Defaults to <repo_root>/dataset/human_talk_tata_seed_dataset."
        ),
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=100,
        help="Expected number of existing silence-positive parent clips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a previously generated Step 3 review package.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: object, fallback: str = "unknown") -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


def normalise_path_text(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().strip('"')
    if not text:
        return None
    return Path(text)


def build_filename_index(dataset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            index[path.name.lower()].append(path.resolve())
    return index


def score_candidate(row: pd.Series, candidate: Path) -> int:
    candidate_text = str(candidate).lower().replace("\\", "/")
    score = 0

    for column, weight in [
        ("source_subfolder", 8),
        ("primary_label", 4),
        ("source_group", 2),
    ]:
        if column in row and not pd.isna(row[column]):
            token = str(row[column]).strip().lower().replace(" ", "_")
            if token and token in candidate_text:
                score += weight

    return score


def resolve_audio(
    row: pd.Series,
    filename_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    for column in ["file_path", "abs_path", "source_path"]:
        if column in row:
            candidate = normalise_path_text(row[column])
            if candidate is not None and candidate.is_file():
                return candidate.resolve(), f"existing_{column}"

    file_name = str(row["file_name"]).strip()
    candidates = filename_index.get(file_name.lower(), [])

    if not candidates:
        return None, "not_found"
    if len(candidates) == 1:
        return candidates[0], "dataset_filename_unique"

    ranked = sorted(
        ((score_candidate(row, path), str(path).lower(), path) for path in candidates),
        reverse=True,
    )
    best_score = ranked[0][0]
    best = [item[2] for item in ranked if item[0] == best_score]

    if len(best) == 1:
        return best[0], "dataset_filename_disambiguated"

    return None, f"ambiguous_{len(candidates)}_matches"


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input review manifest is missing columns: {missing}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else (
            repo_root / "dataset" / "human_talk_tata_seed_dataset"
        ).resolve()
    )

    v09_root = repo_root / "human_talk_workspace" / "tata_v0.9_pipeline"
    input_manifest = (
        v09_root
        / "tata_triage_model"
        / "metadata"
        / "tata_seed_parent_manifest_v09_REVIEW.csv"
    )

    review_root = (
        v09_root
        / "tata_triage_model"
        / "manual_review"
        / "silence_existing_v09"
    )
    audio_root = review_root / "audio"
    review_csv = review_root / "silence_existing_review_sheet_v09.csv"
    missing_csv = review_root / "silence_existing_missing_or_ambiguous_v09.csv"
    summary_json = review_root / "silence_existing_review_package_summary.json"
    instructions_md = review_root / "README_SILENCE_REVIEW.md"

    if not input_manifest.is_file():
        raise FileNotFoundError(f"Step 2 manifest was not found:\n{input_manifest}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Seed dataset root was not found. Expected:\n"
            f"{dataset_root}\n"
            "Pass --dataset_root with the correct location."
        )

    if review_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Step 3 output already exists:\n{review_root}\n"
                "Use --overwrite only to rebuild it intentionally."
            )
        shutil.rmtree(review_root)

    audio_root.mkdir(parents=True, exist_ok=True)

    print("[STEP 3] Reading the v0.9 parent review manifest...")
    frame = pd.read_csv(input_manifest, low_memory=False)
    require_columns(
        frame,
        [
            "clip_id",
            "file_name",
            "source_group",
            "source_subfolder",
            "primary_label",
            "v09_split",
            "v09_audience_reaction_present",
            "v09_silence_present",
            "v09_other_speaker_present",
            "v09_keep_for_tata_training",
        ],
    )

    silence = frame.loc[
        pd.to_numeric(frame["v09_silence_present"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()

    if len(silence) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} silence-positive parents, "
            f"but found {len(silence)}."
        )

    silence = silence.sort_values(
        ["source_group", "source_subfolder", "file_name"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"[STEP 3] Indexing WAV files under:\n{dataset_root}")
    filename_index = build_filename_index(dataset_root)
    if not filename_index:
        raise ValueError(f"No WAV files were found under:\n{dataset_root}")

    rows: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []

    print(f"[STEP 3] Preparing {len(silence)} silence-review audio copies...")
    for zero_index, (_, row) in enumerate(silence.iterrows()):
        review_order = zero_index + 1
        source_audio, resolution_method = resolve_audio(row, filename_index)

        base_record = {
            "review_order": review_order,
            "clip_id": row["clip_id"],
            "file_name": row["file_name"],
            "source_group": row["source_group"],
            "source_subfolder": row["source_subfolder"],
            "primary_label": row["primary_label"],
            "split": row["v09_split"],
            "current_audience_reaction_present": int(
                row["v09_audience_reaction_present"]
            ),
            "current_silence_present": 1,
            "current_other_speaker_present": int(
                row["v09_other_speaker_present"]
            ),
            "corrected_silence_present": 1,
            "review_status": "pending",
            "review_action": "",
            "keep_for_tata_training": int(
                row["v09_keep_for_tata_training"]
            ),
            "review_notes": "",
            "reviewer": "",
            "reviewed_utc": "",
            "resolution_method": resolution_method,
        }

        if source_audio is None:
            unresolved.append(
                {
                    **base_record,
                    "source_audio_path": "",
                    "review_audio_path": "",
                }
            )
            continue

        clip_token = safe_name(row["clip_id"], f"row_{review_order:04d}")
        source_name = safe_name(source_audio.stem, f"audio_{review_order:04d}")
        destination_name = (
            f"SILENCE_{review_order:04d}__{clip_token}__"
            f"{source_name}{source_audio.suffix.lower()}"
        )
        destination = audio_root / destination_name
        shutil.copy2(source_audio, destination)

        rows.append(
            {
                **base_record,
                "source_audio_path": str(source_audio),
                "review_audio_path": str(destination.resolve()),
                "source_audio_sha256": sha256_file(source_audio),
            }
        )

    review_frame = pd.DataFrame(rows)
    unresolved_frame = pd.DataFrame(unresolved)

    review_frame.to_csv(review_csv, index=False, encoding="utf-8")
    unresolved_frame.to_csv(missing_csv, index=False, encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256_file(input_manifest),
        "dataset_root": str(dataset_root),
        "expected_silence_rows": int(args.expected_rows),
        "silence_rows_found": int(len(silence)),
        "audio_files_resolved_and_copied": int(len(review_frame)),
        "missing_or_ambiguous_audio": int(len(unresolved_frame)),
        "review_csv": str(review_csv),
        "audio_root": str(audio_root),
        "baseline_manifest_modified": False,
        "parent_review_manifest_modified": False,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    instructions = """# v0.9 Existing Silence Review

Review each audio file in numerical order and edit only the following columns in
`silence_existing_review_sheet_v09.csv`:

- `corrected_silence_present`
  - `1`: the clip genuinely contains an audible silent region.
  - `0`: the old silence label is incorrect.
- `review_status`
  - change `pending` to `reviewed`.
- `review_action`
  - use `keep`, `correct_label`, or `exclude_bad_audio`.
- `keep_for_tata_training`
  - normally `1`; use `0` only for corrupt/unusable audio.
- `review_notes`
- `reviewer`
- `reviewed_utc`

Do not rename audio files and do not edit identifier/path columns.

This Step 3 package does not modify the main v0.9 parent review manifest.
A later apply script will validate and merge the reviewed corrections.
"""
    instructions_md.write_text(instructions, encoding="utf-8")

    print("\n[COMPLETE] Existing-silence review package created.")
    print(f"Review CSV:       {review_csv}")
    print(f"Review audio:     {audio_root}")
    print(f"Instructions:     {instructions_md}")
    print(f"Resolved/copied:  {len(review_frame):,}")
    print(f"Missing/ambiguous:{len(unresolved_frame):,}")

    if len(unresolved_frame):
        print(
            "\nSome audio files could not be resolved. Check:\n"
            f"{missing_csv}"
        )
        return 2

    print("\nNo baseline or parent review manifest was modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
