#!/usr/bin/env python
"""
Build the v0.9 TATA triage feature cache from:

1. The frozen 12,469-row v0.6 segment-feature manifest and feature arrays.
2. The final reviewed 2,101-parent v0.9 manifest.
3. The 27 newly verified silence clips.

What this script does
---------------------
- Resolves every old segment to one of the original 2,074 parents.
- Replaces old segment labels with the final reviewed v0.9 parent labels.
- Preserves the original train/val/test parent split.
- Automatically locates the old v0.6 feature cache in either the current
  repository or the sibling NeuroAccuExit-ASHADIP repository.
- Reuses old .npy arrays through Windows hard links (copy fallback).
- Segments the 27 verified silence clips into 1-second windows with 0.5-second hop.
- Does NOT remove low-energy windows; these are deliberately verified silence data.
- Extracts new log-mel features using the same ASHADIP transform implementation.
- Auto-detects whether the old cache used CMVN.
- Assigns all 27 new parents to train only.
- Populates complete metadata for every new silence segment.
- Recomputes labels/num_active_labels for every segment after review changes.
- Clears stale fine-audience evidence whenever the reviewed audience label is 0.
- Populates consistent timing aliases and canonical feature/audio paths.
- Writes one combined final feature manifest and audit reports.

Default mode is validation/dry-run. Add --apply to create files.
Old manifests, old features, and source audio are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import soundfile as sf

# Ensure repository root is importable when executed as:
# python scripts\build_v09_tata_feature_cache.py
REPO_FROM_SCRIPT = Path(__file__).resolve().parents[1]
if str(REPO_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(REPO_FROM_SCRIPT))

import librosa  # noqa: E402
from data.transforms_audio import bandpass, cmvn_feat, to_logmel  # noqa: E402


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

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg",
    ".aac", ".wma", ".aiff", ".aif", ".aifc", ".au",
}

FEATURE_PATH_CANDIDATES = [
    "feat_relpath",
    "feature_relpath",
    "feature_path",
    "npy_path",
    "feature_file",
    "features_path",
]

DIRECT_CLIP_ID_CANDIDATES = [
    "clip_id",
    "parent_clip_id",
    "source_clip_id",
    "parent_id",
]

DIRECT_FILE_NAME_CANDIDATES = [
    "file_name",
    "parent_file_name",
    "source_file_name",
    "orig_file_name",
]

PATH_TO_FILE_NAME_CANDIDATES = [
    "orig_relpath",
    "source_file",
    "raw_file",
    "parent_file",
    "source_path",
    "orig_filepath",
    "wav_path",
    "audio_path",
    "wav_relpath",
    "clean_relpath",
]

SPLIT_CANDIDATES = ["split", "v09_split", "dataset_split"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/")


def normalise_id(value: object) -> str:
    return normalise_text(value).lower()


def basename_key(value: object) -> str:
    text = normalise_text(value)
    return Path(text).name.lower() if text else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_token(value: object, max_len: int = 100) -> str:
    import re

    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-") or "item"
    return text[:max_len]


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def binary_series(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad = int(values.isna().sum())
        raise ValueError(f"{name} contains {bad} blank/non-numeric values.")
    invalid = ~values.isin([0, 1])
    if invalid.any():
        examples = sorted(values[invalid].unique().tolist())[:10]
        raise ValueError(f"{name} must contain only 0/1. Found: {examples}")
    return values.astype("int8")


def resolve_feature_column(frame: pd.DataFrame) -> str:
    for column in FEATURE_PATH_CANDIDATES:
        if column in frame.columns:
            nonempty = frame[column].astype(str).str.strip().ne("")
            if nonempty.any():
                return column
    raise ValueError(
        "Could not find the feature-path column in the old manifest. "
        f"Expected one of: {FEATURE_PATH_CANDIDATES}"
    )


def resolve_split_column(frame: pd.DataFrame) -> str:
    for column in SPLIT_CANDIDATES:
        if column in frame.columns:
            values = frame[column].astype(str).str.strip().str.lower()
            if values.isin(["train", "val", "test"]).all():
                return column
    raise ValueError(
        "Could not find a valid train/val/test split column in the old manifest."
    )


def parent_lookup_maps(parents: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    require_columns(parents, ["clip_id", "file_name"], "Final parent manifest")

    clip_pairs = parents[["clip_id"]].copy()
    clip_pairs["key"] = clip_pairs["clip_id"].map(normalise_id)
    if clip_pairs["key"].eq("").any() or clip_pairs["key"].duplicated().any():
        raise ValueError("Parent clip_id values must be non-empty and unique.")
    clip_map = dict(zip(clip_pairs["key"], clip_pairs["clip_id"].astype(str)))

    file_pairs = parents[["clip_id", "file_name"]].copy()
    file_pairs["key"] = file_pairs["file_name"].map(basename_key)
    if file_pairs["key"].eq("").any():
        raise ValueError("Parent file_name values must be non-empty.")
    duplicated = file_pairs["key"].duplicated(keep=False)
    if duplicated.any():
        examples = file_pairs.loc[duplicated, "file_name"].head(10).tolist()
        raise ValueError(
            "Parent file_name values are not unique, so basename matching is unsafe. "
            f"Examples: {examples}"
        )
    file_map = dict(zip(file_pairs["key"], file_pairs["clip_id"].astype(str)))
    return clip_map, file_map


def evaluate_mapping(
    old: pd.DataFrame,
    column: str,
    transform: Callable[[object], str],
    lookup: dict[str, str],
) -> tuple[int, pd.Series]:
    keys = old[column].map(transform)
    mapped = keys.map(lookup)
    return int(mapped.notna().sum()), mapped


def resolve_old_parent_mapping(
    old: pd.DataFrame,
    parents: pd.DataFrame,
    explicit_old_key: str | None = None,
    explicit_parent_key: str | None = None,
) -> tuple[str, str, pd.Series, list[dict]]:
    """
    Resolve each old segment row to final parent clip_id.

    Returns:
      old column, mapping mode, mapped clip_id series, diagnostics
    """
    clip_map, file_map = parent_lookup_maps(parents)
    candidates: list[tuple[str, str, Callable[[object], str], dict[str, str]]] = []

    if explicit_old_key:
        if explicit_old_key not in old.columns:
            raise ValueError(
                f"--old_parent_key_col not found in old manifest: {explicit_old_key}"
            )
        mode = explicit_parent_key or "auto"
        if mode in {"clip_id", "id"}:
            candidates.append((explicit_old_key, "clip_id", normalise_id, clip_map))
        elif mode in {"file_name", "filename", "basename"}:
            candidates.append((explicit_old_key, "file_name", basename_key, file_map))
        else:
            candidates.extend([
                (explicit_old_key, "clip_id", normalise_id, clip_map),
                (explicit_old_key, "file_name", basename_key, file_map),
            ])
    else:
        for column in DIRECT_CLIP_ID_CANDIDATES:
            if column in old.columns:
                candidates.append((column, "clip_id", normalise_id, clip_map))
        for column in DIRECT_FILE_NAME_CANDIDATES:
            if column in old.columns:
                candidates.append((column, "file_name", basename_key, file_map))
        for column in PATH_TO_FILE_NAME_CANDIDATES:
            if column in old.columns:
                candidates.append((column, "file_name", basename_key, file_map))

    if not candidates:
        raise ValueError(
            "No possible parent-identity column was found in the old feature manifest."
        )

    diagnostics: list[dict] = []
    best = None
    for column, mode, transform, lookup in candidates:
        matched, mapped = evaluate_mapping(old, column, transform, lookup)
        record = {
            "old_column": column,
            "parent_mode": mode,
            "matched_rows": matched,
            "total_rows": int(len(old)),
            "coverage": float(matched / max(len(old), 1)),
        }
        diagnostics.append(record)
        score = (matched, 1 if mode == "clip_id" else 0)
        if best is None or score > best[0]:
            best = (score, column, mode, mapped)

    assert best is not None
    _, column, mode, mapped = best
    if mapped.isna().any():
        unmatched = int(mapped.isna().sum())
        examples = old.loc[mapped.isna(), column].head(10).tolist()
        raise ValueError(
            f"Best parent mapping ({column} -> {mode}) matched "
            f"{len(old) - unmatched:,}/{len(old):,} rows, but {unmatched:,} remain "
            f"unmatched. Examples: {examples}\n"
            "Use --old_parent_key_col and --parent_key_mode if the correct key is known."
        )

    return column, mode, mapped.astype(str), diagnostics


def resolve_old_feature_path(value: object, old_features_root: Path) -> Path:
    raw = normalise_text(value)
    if not raw:
        raise ValueError("Encountered an empty feature path in the old manifest.")

    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            old_features_root / path,
            old_features_root.parent / path,
        ])
        # Some manifests already prefix paths with "features/".
        parts = path.parts
        if parts and parts[0].lower() == "features":
            candidates.append(old_features_root.joinpath(*parts[1:]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Old feature array was not found for manifest value: {value}\n"
        f"Tried: {candidates}"
    )


def relative_legacy_destination(source: Path, old_features_root: Path) -> Path:
    try:
        rel = source.relative_to(old_features_root.resolve())
        return Path("v09_final") / "legacy" / rel
    except ValueError:
        digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:12]
        return (
            Path("v09_final")
            / "legacy_external"
            / f"{digest}_{safe_token(source.name)}"
        )


def detect_feature_normalisation(
    feature_paths: list[Path],
    max_samples: int = 30,
) -> tuple[bool, tuple[int, ...], dict]:
    if not feature_paths:
        raise ValueError("No old feature paths were provided for inspection.")

    sample_indices = np.linspace(
        0, len(feature_paths) - 1, num=min(max_samples, len(feature_paths)), dtype=int
    )
    mean_abs_values = []
    std_error_values = []
    shapes = []

    for index in sample_indices:
        array = np.load(feature_paths[int(index)], mmap_mode="r")
        if array.ndim != 2:
            raise ValueError(
                f"Old feature must be 2-D log-mel, got shape {array.shape}: "
                f"{feature_paths[int(index)]}"
            )
        shapes.append(tuple(int(x) for x in array.shape))
        row_means = np.mean(array, axis=1)
        row_stds = np.std(array, axis=1)
        mean_abs_values.append(float(np.median(np.abs(row_means))))
        std_error_values.append(float(np.median(np.abs(row_stds - 1.0))))

    shape_counts = Counter(shapes)
    expected_shape, expected_shape_count = shape_counts.most_common(1)[0]
    if len(shape_counts) != 1:
        raise ValueError(
            "Old feature cache contains inconsistent sampled shapes: "
            f"{dict(shape_counts)}"
        )

    median_mean_abs = float(np.median(mean_abs_values))
    median_std_error = float(np.median(std_error_values))
    cmvn_detected = median_mean_abs < 0.05 and median_std_error < 0.15

    diagnostics = {
        "sampled_feature_count": int(len(sample_indices)),
        "sample_shape": list(expected_shape),
        "shape_consensus_count": int(expected_shape_count),
        "median_abs_per_mel_mean": median_mean_abs,
        "median_abs_per_mel_std_minus_one": median_std_error,
        "cmvn_detected": bool(cmvn_detected),
    }
    return cmvn_detected, expected_shape, diagnostics


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, dtype="float32")
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False)
        y = np.asarray(y, dtype=np.float32)

    if y.ndim > 1:
        if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)
    if int(sr) != int(target_sr):
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    if y.size:
        y = y - float(np.mean(y))
    return np.asarray(y, dtype=np.float32), int(sr)


def evenly_spaced_starts(starts: list[int], max_keep: int) -> list[int]:
    if max_keep <= 0 or len(starts) <= max_keep:
        return starts
    indexes = np.linspace(0, len(starts) - 1, num=max_keep, dtype=int)
    indexes = np.unique(indexes)
    return [starts[int(index)] for index in indexes]


def pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= target_length:
        return y[:target_length]
    return np.pad(y, (0, target_length - len(y)), mode="constant").astype(
        np.float32
    )


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Existing destination differs in size:\n{destination}"
            )
        return "existing"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_new_row_template(columns: list[str]) -> dict:
    return {column: "" for column in columns}


def assign_if_present(row: dict, column: str, value: object) -> None:
    if column in row:
        row[column] = value


def normalise_slashes(value: object) -> str:
    """Return a stable forward-slash path string without changing its meaning."""
    return normalise_text(value)


def path_relative_to_repo(path: Path, repo: Path) -> str:
    """Return a repository-relative path when possible, otherwise the basename."""
    try:
        return str(path.resolve().relative_to(repo.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def resolve_existing_source_path(
    value: object,
    repo: Path,
    legacy_repo: Path,
) -> str:
    """
    Resolve legacy source-audio paths without inventing a nonexistent location.

    Old rows often store either:
      human_talk_tata_seed_dataset/.../file.wav
    or:
      target_speaker/.../file.wav
    """
    raw = normalise_text(value)
    if not raw:
        return ""

    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            repo / "dataset" / path,
            repo / "dataset" / "human_talk_tata_seed_dataset" / path,
            legacy_repo / "dataset" / path,
            legacy_repo / "dataset" / "human_talk_tata_seed_dataset" / path,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    # Keep the original trace instead of silently pointing to a false file.
    return raw


def active_label_text(frame: pd.DataFrame) -> pd.Series:
    """Build canonical pipe-separated 10-label text in schema order."""
    return frame[LABELS].apply(
        lambda row: "|".join(
            label
            for label, value in zip(LABELS, row.tolist())
            if int(value) == 1
        ),
        axis=1,
    )


def canonical_primary_label(row: pd.Series) -> str:
    """
    Return a canonical primary label compatible with the reviewed 10-label schema.

    Fine audience source classes are mapped to audience_reaction_present only when
    that reviewed label remains active. Otherwise the first active 10-label class
    is used. A reviewed all-zero row is explicitly marked rather than retaining a
    stale source-class label.
    """
    current = str(row.get("primary_label", "") or "").strip()
    if current in LABELS and int(row[current]) == 1:
        return current
    if (
        current in {
            "applause_present",
            "laughter_present",
            "crowd_cheer_present",
        }
        and int(row["audience_reaction_present"]) == 1
    ):
        return "audience_reaction_present"
    for label in LABELS:
        if int(row[label]) == 1:
            return label
    return "none_active_reviewed"


def recompute_reviewed_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute all label-derived metadata after parent-level corrections."""
    frame = frame.copy()
    for label in LABELS:
        frame[label] = binary_series(frame[label], label)

    frame["num_active_labels"] = frame[LABELS].sum(axis=1).astype("int16")
    frame["labels"] = active_label_text(frame)
    frame["primary_label"] = frame.apply(canonical_primary_label, axis=1)

    fine_columns = [
        "fine_applause_present",
        "fine_laughter_present",
        "fine_crowd_cheer_present",
    ]
    for column in fine_columns:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0)
            .astype("int8")
        )

    audience_zero = frame["audience_reaction_present"].eq(0)
    frame.loc[audience_zero, fine_columns] = 0
    if "fine_audience_components" not in frame.columns:
        frame["fine_audience_components"] = ""
    frame.loc[audience_zero, "fine_audience_components"] = ""

    # Rebuild the fine-component text from remaining active fine labels, avoiding
    # stale text after human correction.
    component_pairs = [
        ("fine_applause_present", "applause_present"),
        ("fine_laughter_present", "laughter_present"),
        ("fine_crowd_cheer_present", "crowd_cheer_present"),
    ]
    audience_one = frame["audience_reaction_present"].eq(1)
    frame.loc[audience_one, "fine_audience_components"] = frame.loc[
        audience_one, fine_columns
    ].apply(
        lambda row: "|".join(
            output_name
            for (column, output_name), value in zip(
                component_pairs, row.tolist()
            )
            if int(value) == 1
        ),
        axis=1,
    )
    return frame


def validate_manifest_completeness(
    frame: pd.DataFrame,
    feature_column: str,
) -> dict:
    """Fail before writing if core training or new-segment metadata is incomplete."""
    required_all = [
        "sample_id",
        "parent_clip_id",
        "abs_path",
        "segment_wav_relpath",
        feature_column,
        "feature_path",
        "split",
        "primary_label",
        "labels",
        "num_active_labels",
        "start_sec",
        "end_sec",
        "segment_sec",
        "hop_sec",
        "source_file",
        "source_path",
        "source_rel_path",
        "is_clean_seed",
        "is_synthetic",
        "labeling_level",
        "feature_shape",
        "feature_sample_rate",
        "feature_clip_sec",
        "feature_n_mels",
        "feature_n_fft",
        "feature_win_ms",
        "feature_hop_ms",
        "feature_cmvn",
        *LABELS,
    ]
    missing_columns = [column for column in required_all if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Final manifest is missing required columns: {missing_columns}")

    new_mask = frame["v09_data_origin"].astype(str).eq("new_verified_silence")
    new_frame = frame.loc[new_mask]
    blank_counts: dict[str, int] = {}
    for column in required_all:
        values = new_frame[column]
        blank = values.isna() | values.astype(str).str.strip().isin(["", "nan", "None"])
        if int(blank.sum()) > 0:
            blank_counts[column] = int(blank.sum())
    if blank_counts:
        raise ValueError(
            "New silence rows still contain blank required metadata: "
            f"{blank_counts}"
        )

    expected_num = frame[LABELS].sum(axis=1).astype(int)
    actual_num = pd.to_numeric(frame["num_active_labels"], errors="coerce")
    num_mismatch = int((actual_num != expected_num).sum())
    expected_labels = active_label_text(frame)
    label_mismatch = int(
        (frame["labels"].fillna("").astype(str) != expected_labels).sum()
    )

    fine_columns = [
        "fine_applause_present",
        "fine_laughter_present",
        "fine_crowd_cheer_present",
    ]
    fine_positive = frame[fine_columns].sum(axis=1).gt(0)
    fine_inconsistency = int(
        (frame["audience_reaction_present"].eq(0) & fine_positive).sum()
    )

    if num_mismatch or label_mismatch or fine_inconsistency:
        raise ValueError(
            "Final metadata validation failed: "
            f"num_active_labels mismatches={num_mismatch}, "
            f"labels mismatches={label_mismatch}, "
            f"fine-audience inconsistencies={fine_inconsistency}"
        )

    return {
        "new_required_metadata_blank_cells": 0,
        "num_active_labels_mismatches": 0,
        "labels_mismatches": 0,
        "fine_audience_inconsistencies": 0,
        "reviewed_all_zero_segments": int(
            frame["num_active_labels"].astype(int).eq(0).sum()
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_old_features_root(
    repo: Path,
    explicit_root: Path | None,
    expected_npy_count: int,
) -> tuple[Path, str, int]:
    """
    Resolve the legacy v0.6 feature directory.

    Search order:
    1. Explicit --old_features_root.
    2. Current repository:
       <repo>/human_talk_workspace/tata_v0.6_scratch/feature_cache/features
    3. Sibling legacy repository:
       <repo.parent>/NeuroAccuExit-ASHADIP/human_talk_workspace/
       tata_v0.6_scratch/feature_cache/features

    An automatically discovered directory is accepted only when it contains
    the expected number of .npy files. This prevents silently using an
    incomplete or unrelated feature cache.
    """
    if explicit_root is not None:
        path = explicit_root.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Explicit old v0.6 features root was not found:\n{path}"
            )
        npy_count = sum(1 for _ in path.rglob("*.npy"))
        if npy_count != expected_npy_count:
            raise ValueError(
                f"Explicit old feature root contains {npy_count:,} .npy files; "
                f"expected {expected_npy_count:,}:\n{path}"
            )
        return path, "explicit_argument", npy_count

    candidates = [
        (
            repo
            / "human_talk_workspace"
            / "tata_v0.6_scratch"
            / "feature_cache"
            / "features",
            "current_repository",
        ),
        (
            repo.parent
            / "NeuroAccuExit-ASHADIP"
            / "human_talk_workspace"
            / "tata_v0.6_scratch"
            / "feature_cache"
            / "features",
            "sibling_legacy_repository",
        ),
    ]

    # Also detect the sibling repository case-insensitively.
    try:
        for sibling in repo.parent.iterdir():
            if (
                sibling.is_dir()
                and sibling.name.lower() == "neuroaccuexit-ashadip"
            ):
                candidates.append(
                    (
                        sibling
                        / "human_talk_workspace"
                        / "tata_v0.6_scratch"
                        / "feature_cache"
                        / "features",
                        "sibling_legacy_repository_case_detected",
                    )
                )
    except OSError:
        pass

    checked: list[tuple[Path, str, int | None]] = []
    seen: set[str] = set()

    for candidate, source in candidates:
        candidate = candidate.resolve()
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)

        if not candidate.is_dir():
            checked.append((candidate, source, None))
            continue

        npy_count = sum(1 for _ in candidate.rglob("*.npy"))
        checked.append((candidate, source, npy_count))
        if npy_count == expected_npy_count:
            return candidate, source, npy_count

    details = "\n".join(
        f"  - {path} [{source}]: "
        + ("not found" if count is None else f"{count:,} .npy files")
        for path, source, count in checked
    )
    raise FileNotFoundError(
        "Could not automatically locate a complete legacy v0.6 feature cache.\n"
        f"Expected {expected_npy_count:,} .npy files.\n"
        f"Checked:\n{details}\n\n"
        "Provide the correct location explicitly with:\n"
        '  --old_features_root "C:\\path\\to\\feature_cache\\features"'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final v0.9 TATA segment-feature cache."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=REPO_FROM_SCRIPT,
    )
    parser.add_argument(
        "--final_parent_manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--old_feature_manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--old_features_root",
        type=Path,
        default=None,
        help=(
            "Optional path to the legacy v0.6 feature arrays. When omitted, "
            "the script checks the current repository first and then the sibling "
            "NeuroAccuExit-ASHADIP repository automatically."
        ),
    )
    parser.add_argument(
        "--new_silence_root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out_feature_cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--old_parent_key_col",
        default=None,
        help="Optional old-manifest parent key override.",
    )
    parser.add_argument(
        "--parent_key_mode",
        choices=["auto", "clip_id", "file_name"],
        default="auto",
    )
    parser.add_argument("--expected_old_segments", type=int, default=12469)
    parser.add_argument("--expected_old_parents", type=int, default=2074)
    parser.add_argument("--expected_new_parents", type=int, default=27)
    parser.add_argument("--expected_final_parents", type=int, default=2101)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--segment_hop", type=float, default=0.5)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_ms", type=int, default=25)
    parser.add_argument("--hop_ms", type=int, default=10)
    parser.add_argument("--bandpass_low", type=float, default=100.0)
    parser.add_argument("--bandpass_high", type=float, default=3000.0)
    parser.add_argument(
        "--max_segments_per_new_parent",
        type=int,
        default=0,
        help=(
            "0 = automatically use the median old segment count among corrected "
            "silence-positive parents."
        ),
    )
    parser.add_argument(
        "--force_cmvn",
        choices=["auto", "yes", "no"],
        default="auto",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create hardlinks/features/manifests. Without this flag, validate only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace final output manifest/reports if they already exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo_root.expanduser().resolve()

    v09_root = repo / "human_talk_workspace" / "tata_v0.9_pipeline"
    triage_root = v09_root / "tata_triage_model"

    final_parent_manifest = (
        args.final_parent_manifest.expanduser().resolve()
        if args.final_parent_manifest
        else (
            triage_root
            / "metadata"
            / "tata_seed_parent_manifest_v09_FINAL_REVIEWED.csv"
        ).resolve()
    )
    old_feature_manifest = (
        args.old_feature_manifest.expanduser().resolve()
        if args.old_feature_manifest
        else (
            triage_root
            / "feature_cache"
            / "metadata"
            / "tata_seed_features_manifest_10label_12469_BASELINE.csv"
        ).resolve()
    )
    (
        old_features_root,
        old_features_root_source,
        old_features_npy_count,
    ) = discover_old_features_root(
        repo=repo,
        explicit_root=args.old_features_root,
        expected_npy_count=int(args.expected_old_segments),
    )
    new_silence_root = (
        args.new_silence_root.expanduser().resolve()
        if args.new_silence_root
        else (repo / "dataset" / "new_verified_rare_event_audio" / "silence").resolve()
    )
    out_cache = (
        args.out_feature_cache.expanduser().resolve()
        if args.out_feature_cache
        else (triage_root / "feature_cache").resolve()
    )

    out_features_root = out_cache / "features"
    output_manifest = (
        out_cache / "metadata" / "multilabel_features_manifest_v09_FINAL.csv"
    )
    new_segments_csv = (
        out_cache / "metadata" / "new_verified_silence_segments_v09.csv"
    )
    mapping_diagnostics_csv = (
        v09_root
        / "shared"
        / "correction_ledgers"
        / "v09_old_segment_parent_mapping_diagnostics.csv"
    )
    label_counts_csv = (
        v09_root
        / "shared"
        / "correction_ledgers"
        / "v09_final_segment_label_counts.csv"
    )
    summary_json = (
        v09_root
        / "shared"
        / "correction_ledgers"
        / "v09_feature_cache_build_summary.json"
    )
    segment_wav_root = (
        out_cache / "segment_wavs" / "new_verified_silence"
    )

    require_file(final_parent_manifest, "Final reviewed parent manifest")
    require_file(old_feature_manifest, "Frozen old feature manifest")
    require_dir(old_features_root, "Old v0.6 features root")
    require_dir(new_silence_root, "New verified silence audio directory")

    parents = pd.read_csv(final_parent_manifest, low_memory=False)
    old = pd.read_csv(old_feature_manifest, low_memory=False)

    require_columns(
        parents,
        [
            "clip_id",
            "file_name",
            "v09_split",
            "v09_keep_for_tata_training",
            *[f"v09_{label}" for label in LABELS],
        ],
        "Final parent manifest",
    )

    if len(parents) != args.expected_final_parents:
        raise ValueError(
            f"Expected {args.expected_final_parents} final parents, found {len(parents)}."
        )
    if len(old) != args.expected_old_segments:
        raise ValueError(
            f"Expected {args.expected_old_segments} old segment rows, found {len(old)}."
        )

    for label in LABELS:
        parents[f"v09_{label}"] = binary_series(
            parents[f"v09_{label}"], f"v09_{label}"
        )
    parents["v09_keep_for_tata_training"] = binary_series(
        parents["v09_keep_for_tata_training"],
        "v09_keep_for_tata_training",
    )

    if int((parents["v09_keep_for_tata_training"] == 0).sum()) > 0:
        raise ValueError(
            "The final parent manifest still contains parents excluded from TATA "
            "training. Resolve them before building the feature cache."
        )

    old_key_column, mapping_mode, mapped_clip_ids, diagnostics = (
        resolve_old_parent_mapping(
            old,
            parents,
            explicit_old_key=args.old_parent_key_col,
            explicit_parent_key=(
                None if args.parent_key_mode == "auto" else args.parent_key_mode
            ),
        )
    )
    old = old.copy()
    old["_v09_clip_id"] = mapped_clip_ids

    old_parent_count = int(old["_v09_clip_id"].nunique())
    if old_parent_count != args.expected_old_parents:
        raise ValueError(
            f"Expected {args.expected_old_parents} old parents in the segment cache, "
            f"resolved {old_parent_count}."
        )

    parent_index = parents.set_index(parents["clip_id"].astype(str), drop=False)
    old_parent_ids = set(old["_v09_clip_id"].astype(str))
    final_parent_ids = set(parents["clip_id"].astype(str))
    if not old_parent_ids.issubset(final_parent_ids):
        raise ValueError("Some old segment parents are absent from the final manifest.")

    feature_column = resolve_feature_column(old)
    split_column = resolve_split_column(old)

    # Resolve and validate all 12,469 legacy feature arrays.
    old_sources: list[Path] = []
    legacy_relpaths: list[str] = []
    for value in old[feature_column].tolist():
        source = resolve_old_feature_path(value, old_features_root)
        old_sources.append(source)
        legacy_relpaths.append(
            str(relative_legacy_destination(source, old_features_root)).replace(
                "\\", "/"
            )
        )

    if len(set(legacy_relpaths)) != len(legacy_relpaths):
        duplicates = [
            item for item, count in Counter(legacy_relpaths).items() if count > 1
        ][:10]
        raise ValueError(
            "Legacy destination feature paths are not unique. "
            f"Examples: {duplicates}"
        )

    detected_cmvn, expected_feature_shape, feature_diagnostics = (
        detect_feature_normalisation(old_sources)
    )
    if args.force_cmvn == "yes":
        use_cmvn = True
    elif args.force_cmvn == "no":
        use_cmvn = False
    else:
        use_cmvn = detected_cmvn

    n_mels = int(expected_feature_shape[0])

    # Final parent rows corresponding to the 27 source files in the new folder.
    audio_files = sorted(
        [
            path.resolve()
            for path in new_silence_root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    if len(audio_files) != args.expected_new_parents:
        raise ValueError(
            f"Expected {args.expected_new_parents} new silence audio files, "
            f"found {len(audio_files)}."
        )

    audio_by_name = {}
    for path in audio_files:
        key = path.name.lower()
        if key in audio_by_name:
            raise ValueError(f"Duplicate new-audio filename: {path.name}")
        audio_by_name[key] = path

    new_parent_mask = parents["file_name"].astype(str).str.lower().isin(audio_by_name)
    new_parents = parents.loc[new_parent_mask].copy()
    if len(new_parents) != args.expected_new_parents:
        missing_names = sorted(
            set(audio_by_name)
            - set(parents["file_name"].astype(str).str.lower())
        )
        raise ValueError(
            f"Only {len(new_parents)} of {args.expected_new_parents} new audio files "
            f"were found in the final parent manifest. Missing: {missing_names[:10]}"
        )

    if not (new_parents["v09_split"].astype(str).str.lower() == "train").all():
        raise ValueError("All 27 new silence parents must be assigned to train.")
    if not (new_parents["v09_silence_present"] == 1).all():
        raise ValueError("Every new verified silence parent must have silence_present=1.")
    non_silence_labels = [
        f"v09_{label}" for label in LABELS if label != "silence_present"
    ]
    if int(new_parents[non_silence_labels].sum().sum()) != 0:
        raise ValueError(
            "New verified pure-silence parents contain unexpected additional labels."
        )

    # Derive a balanced cap from existing corrected silence-positive parents.
    segment_counts = old.groupby("_v09_clip_id").size()
    old_silence_ids = set(
        parents.loc[
            parents["clip_id"].astype(str).isin(old_parent_ids)
            & (parents["v09_silence_present"] == 1),
            "clip_id",
        ].astype(str)
    )
    old_silence_counts = segment_counts[
        segment_counts.index.astype(str).isin(old_silence_ids)
    ]
    reference_counts = (
        old_silence_counts if len(old_silence_counts) else segment_counts
    )
    auto_cap = max(1, int(round(float(reference_counts.median()))))
    segment_cap = (
        int(args.max_segments_per_new_parent)
        if args.max_segments_per_new_parent > 0
        else auto_cap
    )

    # Relabel old segment rows using the final parent manifest.
    old_final = old.drop(columns=["_v09_clip_id"]).copy()
    old_final["clip_id"] = mapped_clip_ids.values
    old_final["parent_clip_id"] = mapped_clip_ids.values
    old_final[feature_column] = legacy_relpaths

    parent_split_map = parents.set_index(
        parents["clip_id"].astype(str)
    )["v09_split"].astype(str).str.lower()
    old_final[split_column] = mapped_clip_ids.map(parent_split_map).values
    old_final["split"] = mapped_clip_ids.map(parent_split_map).values

    for label in LABELS:
        label_map = parents.set_index(
            parents["clip_id"].astype(str)
        )[f"v09_{label}"]
        old_final[label] = mapped_clip_ids.map(label_map).astype("int8").values

    old_final["v09_data_origin"] = "reused_v06_feature"
    old_final["v09_parent_labels_reviewed"] = 1

    # Preserve all old columns, while adding universal columns needed for new rows.
    universal_columns = [
        "sample_id",
        "clip_id",
        "parent_clip_id",
        "file_name",
        "abs_path",
        "segment_wav_relpath",
        "feat_relpath",
        "feature_path",
        "split",
        "primary_label",
        "labels",
        "num_active_labels",
        "start_sec",
        "end_sec",
        "segment_sec",
        "hop_sec",
        "wav_relpath",
        "orig_relpath",
        "source_file",
        "source_path",
        "source_rel_path",
        "parent_file",
        "is_clean_seed",
        "is_synthetic",
        "labeling_level",
        "feature_shape",
        "feature_sample_rate",
        "feature_clip_sec",
        "feature_n_mels",
        "feature_n_fft",
        "feature_win_ms",
        "feature_hop_ms",
        "feature_cmvn",
        "fine_audience_components",
        "fine_applause_present",
        "fine_laughter_present",
        "fine_crowd_cheer_present",
        "start",
        "parent_start",
        "duration",
        "segment_index_parent",
        *LABELS,
        "v09_data_origin",
        "v09_parent_labels_reviewed",
    ]
    final_columns = list(old_final.columns)
    for column in universal_columns:
        if column not in final_columns:
            final_columns.append(column)
    old_final = old_final.reindex(columns=final_columns, fill_value="")

    # Complete/normalise legacy aliases without recomputing old feature arrays.
    legacy_repo = repo.parent / "NeuroAccuExit-ASHADIP"
    old_final["feat_relpath"] = old_final[feature_column].map(normalise_slashes)
    old_final["segment_wav_relpath"] = old_final["segment_wav_relpath"].map(
        normalise_slashes
    )
    old_final["feature_path"] = old_final["feat_relpath"].map(
        lambda value: str((out_features_root / Path(str(value))).resolve())
    )
    old_final["abs_path"] = old_final["abs_path"].map(
        lambda value: str(Path(str(value)).resolve())
        if normalise_text(value)
        else ""
    )
    old_final["source_file"] = old_final["source_file"].map(
        lambda value: Path(normalise_text(value)).name
    )
    old_final["file_name"] = old_final["source_file"]
    old_final["source_rel_path"] = old_final["source_rel_path"].map(
        normalise_slashes
    )
    old_final["source_path"] = old_final.apply(
        lambda row: resolve_existing_source_path(
            normalise_text(row["source_path"])
            or normalise_text(row["source_rel_path"]),
            repo,
            legacy_repo,
        ),
        axis=1,
    )
    old_final["parent_file"] = old_final["source_path"]
    old_final["orig_relpath"] = old_final["source_rel_path"]
    old_final["wav_relpath"] = old_final["source_rel_path"]

    old_final["start"] = pd.to_numeric(
        old_final["start_sec"], errors="coerce"
    ).fillna(0.0)
    old_final["parent_start"] = old_final["start"]
    old_final["duration"] = pd.to_numeric(
        old_final["segment_sec"], errors="coerce"
    ).fillna(float(args.segment_sec))
    old_final["segment_index_parent"] = old_final.groupby(
        "parent_clip_id", sort=False
    ).cumcount().astype("int32")

    # Prepare new segments and features in memory/dry-run metadata.
    new_rows = []
    new_feature_payloads: list[tuple[Path, np.ndarray]] = []
    new_segment_payloads: list[tuple[Path, np.ndarray, int]] = []

    target_samples = int(round(args.segment_sec * args.sample_rate))
    hop_samples = int(round(args.segment_hop * args.sample_rate))
    if target_samples <= 0 or hop_samples <= 0:
        raise ValueError("segment_sec and segment_hop must be positive.")

    for _, parent in new_parents.sort_values("file_name").iterrows():
        audio_path = audio_by_name[str(parent["file_name"]).lower()]
        y, sr = load_audio(audio_path, args.sample_rate)

        # Match established ASHADIP cleaning, but deliberately do not reject
        # low-energy windows because these clips are verified silence positives.
        if y.size:
            y = bandpass(
                y,
                sr,
                float(args.bandpass_low),
                float(args.bandpass_high),
            ).astype(np.float32)
            peak = float(np.max(np.abs(y)) + 1e-9)
            if peak > 0:
                y = (0.8913 * y / peak).astype(np.float32)

        if len(y) <= target_samples:
            starts = [0]
        else:
            starts = list(
                range(0, max(len(y) - target_samples + 1, 1), hop_samples)
            )
            if not starts:
                starts = [0]
        starts = evenly_spaced_starts(starts, segment_cap)

        for local_index, start_sample in enumerate(starts):
            clip = pad_or_trim(
                y[start_sample : start_sample + target_samples],
                target_samples,
            )
            segment_key = (
                f"{parent['clip_id']}|{start_sample}|{local_index}|"
                f"{audio_path.name}"
            )
            digest = hashlib.md5(segment_key.encode("utf-8")).hexdigest()[:16]
            segment_name = f"silence_present_seg_{digest}.wav"
            feature_name = f"silence_present_seg_{digest}.npy"

            segment_rel = (
                Path("segment_wavs")
                / "new_verified_silence"
                / "train"
                / segment_name
            )
            feature_rel = (
                Path("v09_final")
                / "new_verified_silence"
                / "train"
                / feature_name
            )

            feature = to_logmel(
                clip,
                sr,
                n_mels=n_mels,
                n_fft=args.n_fft,
                win_ms=args.win_ms,
                hop_ms=args.hop_ms,
            )
            if use_cmvn:
                feature = cmvn_feat(feature)
            if tuple(feature.shape) != tuple(expected_feature_shape):
                raise ValueError(
                    "New feature shape does not match old cache. "
                    f"Old={expected_feature_shape}, new={feature.shape}. "
                    "Check sample rate/n_fft/win_ms/hop_ms."
                )

            row = build_new_row_template(final_columns)
            start_sec = float(start_sample / sr)
            end_sec = float(start_sec + args.segment_sec)
            parent_token = safe_token(audio_path.stem)
            while "__" in parent_token:
                parent_token = parent_token.replace("__", "_")
            sample_id = f"{parent_token}_seg{local_index:04d}"
            source_rel = path_relative_to_repo(audio_path, repo)
            segment_abs = (out_cache / segment_rel).resolve()
            feature_abs = (out_features_root / feature_rel).resolve()

            assign_if_present(row, feature_column, str(feature_rel).replace("\\", "/"))
            assign_if_present(row, "feat_relpath", str(feature_rel).replace("\\", "/"))
            assign_if_present(row, "feature_path", str(feature_abs))
            assign_if_present(row, "sample_id", sample_id)
            assign_if_present(row, "clip_id", str(parent["clip_id"]))
            assign_if_present(row, "parent_clip_id", str(parent["clip_id"]))
            assign_if_present(row, "file_name", str(parent["file_name"]))
            assign_if_present(row, "abs_path", str(segment_abs))
            assign_if_present(row, split_column, "train")
            assign_if_present(row, "split", "train")
            assign_if_present(row, "primary_label", "silence_present")
            assign_if_present(row, "labels", "silence_present")
            assign_if_present(row, "num_active_labels", 1)
            assign_if_present(row, "wav_relpath", source_rel)
            assign_if_present(row, "orig_relpath", source_rel)
            assign_if_present(row, "source_file", audio_path.name)
            assign_if_present(row, "source_path", str(audio_path.resolve()))
            assign_if_present(row, "source_rel_path", source_rel)
            assign_if_present(row, "parent_file", str(audio_path.resolve()))
            assign_if_present(
                row,
                "segment_wav_relpath",
                str(segment_rel).replace("\\", "/"),
            )
            assign_if_present(row, "start_sec", start_sec)
            assign_if_present(row, "end_sec", end_sec)
            assign_if_present(row, "segment_sec", float(args.segment_sec))
            assign_if_present(row, "hop_sec", float(args.segment_hop))
            assign_if_present(row, "start", start_sec)
            assign_if_present(row, "parent_start", start_sec)
            assign_if_present(row, "duration", float(args.segment_sec))
            assign_if_present(row, "segment_index_parent", int(local_index))
            assign_if_present(row, "is_clean_seed", 1)
            assign_if_present(row, "is_synthetic", 0)
            assign_if_present(
                row,
                "labeling_level",
                "human_verified_parent_propagated_to_segment",
            )
            assign_if_present(
                row,
                "feature_shape",
                f"{expected_feature_shape[0]}x{expected_feature_shape[1]}",
            )
            assign_if_present(row, "feature_sample_rate", int(args.sample_rate))
            assign_if_present(row, "feature_clip_sec", float(args.segment_sec))
            assign_if_present(row, "feature_n_mels", int(n_mels))
            assign_if_present(row, "feature_n_fft", int(args.n_fft))
            assign_if_present(row, "feature_win_ms", int(args.win_ms))
            assign_if_present(row, "feature_hop_ms", int(args.hop_ms))
            assign_if_present(row, "feature_cmvn", int(use_cmvn))
            assign_if_present(row, "fine_audience_components", "")
            assign_if_present(row, "fine_applause_present", 0)
            assign_if_present(row, "fine_laughter_present", 0)
            assign_if_present(row, "fine_crowd_cheer_present", 0)
            assign_if_present(row, "v09_data_origin", "new_verified_silence")
            assign_if_present(row, "v09_parent_labels_reviewed", 1)

            for label in LABELS:
                row[label] = 1 if label == "silence_present" else 0

            # Populate the selected old key column consistently where possible.
            if old_key_column in row:
                if mapping_mode == "clip_id":
                    row[old_key_column] = str(parent["clip_id"])
                else:
                    row[old_key_column] = str(parent["file_name"])

            new_rows.append(row)
            new_feature_payloads.append((out_features_root / feature_rel, feature))
            new_segment_payloads.append(
                (out_cache / segment_rel, clip, sr)
            )

    new_frame = pd.DataFrame(new_rows, columns=final_columns)
    combined = pd.concat([old_final, new_frame], ignore_index=True)

    # Recompute all label-derived fields after parent-level corrections and
    # normalise canonical paths for both reused and newly extracted features.
    combined = recompute_reviewed_metadata(combined)
    combined["feat_relpath"] = combined[feature_column].map(normalise_slashes)
    combined[feature_column] = combined["feat_relpath"]
    combined["segment_wav_relpath"] = combined["segment_wav_relpath"].map(
        normalise_slashes
    )
    combined["feature_path"] = combined["feat_relpath"].map(
        lambda value: str((out_features_root / Path(str(value))).resolve())
    )
    combined["source_rel_path"] = combined["source_rel_path"].map(
        normalise_slashes
    )
    combined["wav_relpath"] = combined["wav_relpath"].map(normalise_slashes)
    combined["orig_relpath"] = combined["orig_relpath"].map(normalise_slashes)

    completeness_audit = validate_manifest_completeness(
        combined,
        feature_column=feature_column,
    )

    # Final invariants.
    if len(combined) != len(old) + len(new_frame):
        raise RuntimeError("Combined row-count invariant failed.")
    if combined[feature_column].astype(str).duplicated().any():
        examples = combined.loc[
            combined[feature_column].astype(str).duplicated(keep=False),
            feature_column,
        ].head(10).tolist()
        raise ValueError(f"Duplicate final feature paths: {examples}")

    combined_parent_count = int(combined["clip_id"].astype(str).nunique())
    if combined_parent_count != args.expected_final_parents:
        raise ValueError(
            f"Expected {args.expected_final_parents} combined parents, "
            f"found {combined_parent_count}."
        )

    leakage = (
        combined.groupby(combined["clip_id"].astype(str))["split"]
        .nunique()
        .gt(1)
    )
    if leakage.any():
        bad = leakage[leakage].index[:10].tolist()
        raise ValueError(f"Parent split leakage detected: {bad}")

    label_counts = pd.DataFrame(
        {
            "label": LABELS,
            "positive_segments": [
                int(pd.to_numeric(combined[label], errors="raise").sum())
                for label in LABELS
            ],
        }
    )

    summary = {
        "generated_utc": utc_now(),
        "mode": "apply" if args.apply else "dry_run",
        "final_parent_manifest": str(final_parent_manifest),
        "old_feature_manifest": str(old_feature_manifest),
        "old_features_root": str(old_features_root),
        "old_features_root_source": old_features_root_source,
        "old_features_npy_count": int(old_features_npy_count),
        "new_silence_root": str(new_silence_root),
        "output_manifest": str(output_manifest),
        "feature_path_column": feature_column,
        "old_split_column": split_column,
        "old_parent_mapping_column": old_key_column,
        "old_parent_mapping_mode": mapping_mode,
        "old_segment_rows": int(len(old)),
        "old_parent_count": old_parent_count,
        "new_parent_count": int(len(new_parents)),
        "new_segment_rows": int(len(new_frame)),
        "new_segments_per_parent_cap": int(segment_cap),
        "auto_cap_from_old_silence_median": int(auto_cap),
        "old_silence_segment_count_min": (
            int(reference_counts.min()) if len(reference_counts) else None
        ),
        "old_silence_segment_count_median": (
            float(reference_counts.median()) if len(reference_counts) else None
        ),
        "old_silence_segment_count_max": (
            int(reference_counts.max()) if len(reference_counts) else None
        ),
        "combined_segment_rows": int(len(combined)),
        "combined_parent_count": combined_parent_count,
        "split_segment_counts": {
            str(key): int(value)
            for key, value in combined["split"].value_counts().items()
        },
        "feature_shape": list(expected_feature_shape),
        "n_mels": n_mels,
        "cmvn_used_for_new_features": bool(use_cmvn),
        "feature_detection": feature_diagnostics,
        "manifest_completeness_audit": completeness_audit,
        "timing_policy": {
            "segment_sec": float(args.segment_sec),
            "new_segment_hop_sec": float(args.segment_hop),
            "legacy_timing_preserved": True,
        },
        "legacy_reuse_method": (
            "hardlink with copy fallback" if args.apply else "planned"
        ),
        "source_manifests_modified": False,
        "source_audio_modified": False,
    }

    print("\n=== v0.9 TATA feature-cache validation ===")
    print(f"Final parents:                 {len(parents):,}")
    print(f"Old segment rows:              {len(old):,}")
    print(f"Old features root:             {old_features_root}")
    print(f"Old features source:           {old_features_root_source}")
    print(f"Old .npy arrays found:         {old_features_npy_count:,}")
    print(f"Old parents resolved:          {old_parent_count:,}")
    print(
        f"Old mapping:                   {old_key_column} -> {mapping_mode}"
    )
    print(f"Legacy feature column:         {feature_column}")
    print(f"Legacy feature shape:          {expected_feature_shape}")
    print(f"CMVN detected/used:            {detected_cmvn}/{use_cmvn}")
    print(f"New silence parents:           {len(new_parents):,}")
    print(f"New segment cap per parent:    {segment_cap}")
    print(f"New silence segment rows:      {len(new_frame):,}")
    print(f"Combined segment rows:         {len(combined):,}")
    print(f"Combined unique parents:       {combined_parent_count:,}")
    print(
        "Combined split rows:          "
        f"{combined['split'].value_counts().to_dict()}"
    )
    print(
        "New required metadata blanks: "
        f"{completeness_audit['new_required_metadata_blank_cells']}"
    )
    print(
        "Derived-label mismatches:     "
        f"{completeness_audit['labels_mismatches'] + completeness_audit['num_active_labels_mismatches']}"
    )
    print(
        "Fine-audience inconsistencies:"
        f" {completeness_audit['fine_audience_inconsistencies']}"
    )
    print(f"Output manifest:               {output_manifest}")

    if not args.apply:
        print("\n[DRY RUN COMPLETE] No files were created or modified.")
        print("Run the same command with --apply after checking these counts.")
        return 0

    for path in [
        output_manifest,
        new_segments_csv,
        mapping_diagnostics_csv,
        label_counts_csv,
        summary_json,
    ]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists:\n{path}\nUse --overwrite to rebuild."
            )

    # Reuse old arrays through hard links; copy only if hard links are unavailable.
    reuse_counts = Counter()
    for source, rel in zip(old_sources, legacy_relpaths):
        destination = out_features_root / Path(rel)
        method = link_or_copy(source, destination)
        reuse_counts[method] += 1

    # Write newly extracted feature arrays and inspectable segment WAVs.
    for destination, feature in new_feature_payloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, feature)

    for destination, clip, sr in new_segment_payloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, clip, sr)

    for report_path in [
        output_manifest,
        new_segments_csv,
        mapping_diagnostics_csv,
        label_counts_csv,
        summary_json,
    ]:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_manifest, index=False, encoding="utf-8")
    new_frame.to_csv(new_segments_csv, index=False, encoding="utf-8")
    pd.DataFrame(diagnostics).sort_values(
        ["matched_rows", "parent_mode"], ascending=[False, True]
    ).to_csv(mapping_diagnostics_csv, index=False, encoding="utf-8")
    label_counts.to_csv(label_counts_csv, index=False, encoding="utf-8")

    summary["legacy_reuse_counts"] = {
        str(key): int(value) for key, value in reuse_counts.items()
    }
    summary["output_manifest_sha256"] = sha256_file(output_manifest)
    write_json(summary_json, summary)

    print("\n[COMPLETE] v0.9 TATA feature cache created.")
    print(f"Legacy arrays reused:          {dict(reuse_counts)}")
    print(f"New arrays extracted:          {len(new_feature_payloads):,}")
    print(f"New segment WAVs written:      {len(new_segment_payloads):,}")
    print(f"Final feature manifest:        {output_manifest}")
    print(f"Build summary:                 {summary_json}")
    print("No v0.6 features, source manifests, or source audio were modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
