#!/usr/bin/env python
"""
Apply human-reviewed low-energy segment recovery to TATA v0.9.

Purpose
-------
1. Keep ALL manually reviewed low-energy candidates:
   - human silence positives become positive silence segment examples;
   - human silence negatives become hard-negative silence examples.
2. Merge them into their original parent timelines.
3. Detect parent-level silence when at least two consecutive reviewed
   1-second windows are silent.
4. Build a NEW, self-contained feature cache without modifying the existing
   v0.9 parent manifest, feature manifest, features, or source audio.

Default rule
------------
Two silent windows are consecutive when their start times differ by one
review hop (normally 0.5 seconds), within a tolerance. With 1-second windows
and 0.5-second hop, this represents approximately 1.5 seconds of continuous
silence evidence.

Important label policy
----------------------
- silence_present on recovered rows: authoritative human segment review.
- the other nine labels: inherited from the reviewed row's parent labels,
  falling back to the current parent manifest.
- ALL rows with review_status=reviewed are incorporated, regardless of the
  old review_keep_segment field. Existing timeline rows are updated in place;
  only genuinely missing windows are appended. The old field is retained only
  as provenance.
- existing feature-manifest rows are not relabelled. The recovered parent
  manifest is updated independently for parent-level evaluation/routing.

Feature compatibility
---------------------
Recovered features mirror the legacy pipeline:
- mono, 16 kHz
- whole-parent DC removal
- optional FFT bandpass (default 50-7600 Hz)
- whole-parent peak normalisation to 0.8913
- 64-bin log-mel, n_fft=1024, win=25 ms, hop=10 ms
- per-mel CMVN
The resulting shape is matched to an existing feature (normally 64x101).

Run modes
---------
Dry run (default): validates everything and prints the intended changes.
Apply: writes a new cache under silence_recovered_v09.

No existing files are overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SCRIPT_VERSION = "3.0-windows-safe-finalisation"

DEFAULT_LABELS = [
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

FEATURE_COLUMNS = [
    "feat_relpath",
    "feature_path",
    "feature_relpath",
    "feat_path",
    "npy_path",
]

START_COLUMNS = [
    "start_sec",
    "start",
    "parent_start",
    "segment_start_sec",
    "original_start",
]

END_COLUMNS = [
    "end_sec",
    "end",
    "segment_end_sec",
]

DURATION_COLUMNS = [
    "duration",
    "duration_sec",
    "segment_sec",
    "feature_clip_sec",
]

ABS_AUDIO_COLUMNS = [
    "filepath",
    "abs_path",
    "audio_path",
    "segment_path",
    "segment_wav_path",
    "final_path",
]

SOURCE_AUDIO_COLUMNS = [
    "source_path",
    "source_file",
    "raw_file",
    "parent_file",
]

ID_COLUMNS = [
    "sample_id",
    "segment_id",
    "feature_id",
]

PROVENANCE_COLUMNS = [
    "v09_data_origin",
    "silence_label_source",
    "other_labels_source",
    "recovery_candidate_id",
    "recovery_parent_audio_path",
    "recovery_raw_rms_dbfs",
    "recovery_raw_peak_dbfs",
    "recovery_zero_crossing_rate",
    "recovery_spectral_flatness",
    "recovery_energy_band",
    "recovery_review_priority",
    "recovery_review_keep_segment_original",
    "recovery_review_notes",
    "silence_recovery_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm_path_text(value: Any) -> str:
    return norm_text(value).strip('"').replace("\\", "/")


def safe_name(value: Any, max_len: int = 90) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", norm_text(value))
    return (text.strip("._-") or "item")[:max_len]


def as_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | ~values.isin([0, 1])
    if bad.any():
        examples = series[bad].head(10).tolist()
        raise ValueError(
            f"{name} must contain only 0/1 for reviewed rows. "
            f"Invalid examples: {examples}"
        )
    return values.astype(int)


def load_labels(labels_json: Path | None) -> list[str]:
    if labels_json is not None and labels_json.is_file():
        payload = json.loads(labels_json.read_text(encoding="utf-8"))
        labels = payload.get("labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
    return list(DEFAULT_LABELS)


def detect_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
    required: bool = True,
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(
            f"Could not find {description}. Tried columns: {list(candidates)}"
        )
    return None


def labels_string(active: list[str], separator: str) -> str:
    return separator.join(active)


def detect_label_separator(df: pd.DataFrame) -> str:
    if "labels" not in df.columns:
        return "|"
    for value in df["labels"].dropna().astype(str):
        if "|" in value:
            return "|"
        if ";" in value:
            return ";"
        if "," in value:
            return ","
    return "|"


def parent_label_column(df: pd.DataFrame, label: str) -> str:
    preferred = f"v09_{label}"
    if preferred in df.columns:
        return preferred
    if label in df.columns:
        return label
    raise ValueError(
        f"Parent manifest is missing label column '{preferred}' or '{label}'."
    )


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return None


def strip_features_prefix(path_text: str) -> Path:
    p = Path(path_text.replace("\\", "/"))
    parts = list(p.parts)
    lowered = [part.lower() for part in parts]
    if "features" in lowered:
        index = lowered.index("features")
        parts = parts[index + 1 :]
    if not parts:
        raise ValueError(f"Invalid feature path: {path_text}")
    return Path(*parts)


def resolve_old_feature(
    value: Any,
    old_features_root: Path,
    repo_root: Path,
    feature_manifest: Path,
) -> tuple[Path, Path]:
    text = norm_path_text(value)
    if not text:
        raise FileNotFoundError("Empty feature path.")

    raw = Path(text)
    stripped = strip_features_prefix(text)

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                old_features_root / raw,
                old_features_root / stripped,
                repo_root / raw,
                feature_manifest.parent / raw,
                feature_manifest.parent.parent / raw,
            ]
        )

    resolved = first_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Could not resolve feature file.\n"
            f"Manifest value: {text}\n"
            f"Tried under: {old_features_root}"
        )

    try:
        relative = resolved.relative_to(old_features_root.resolve())
    except ValueError:
        relative = stripped

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative feature path: {relative}")
    return resolved, relative


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def safe_load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, dtype="float32")
        return to_mono(y), int(sr)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=None, mono=False)
        return to_mono(np.asarray(y, dtype=np.float32)), int(sr)


def fft_bandpass(
    y: np.ndarray,
    sr: int,
    low_hz: float | None,
    high_hz: float | None,
) -> np.ndarray:
    if low_hz is None or high_hz is None:
        return np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.asarray(y, dtype=np.float32)

    high_hz = min(float(high_hz), (sr / 2.0) - 1e-6)
    low_hz = max(float(low_hz), 0.0)
    if low_hz >= high_hz:
        raise ValueError(
            f"Invalid bandpass after Nyquist adjustment: {low_hz}, {high_hz}"
        )

    spectrum = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(len(y), 1.0 / sr)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    filtered = np.fft.irfft(
        spectrum * mask.astype(np.float64),
        n=len(y),
    )
    return np.asarray(filtered, dtype=np.float32)


def legacy_preprocess_parent(
    path: Path,
    target_sr: int,
    bandpass_low: float | None,
    bandpass_high: float | None,
) -> tuple[np.ndarray, int]:
    y, sr = safe_load_audio(path)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = int(target_sr)

    y = np.asarray(y, dtype=np.float32)
    if y.size:
        y = y - float(np.mean(y))

    y = fft_bandpass(y, sr, bandpass_low, bandpass_high)

    peak = float(np.max(np.abs(y)) + 1e-9) if y.size else 1e-9
    if peak > 0:
        y = (0.8913 * y / peak).astype(np.float32)

    return y, int(sr)


def pad_or_trim(y: np.ndarray, length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.shape[0] >= length:
        return y[:length]
    return np.pad(y, (0, length - y.shape[0]), mode="constant").astype(
        np.float32
    )


def rms_dbfs(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def extract_logmel(
    clip: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    win_ms: int,
    hop_ms: int,
    cmvn: bool,
    target_shape: tuple[int, int],
) -> np.ndarray:
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)

    spectrum = librosa.feature.melspectrogram(
        y=np.asarray(clip, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=n_mels,
        power=2.0,
    )
    feature = librosa.power_to_db(spectrum, ref=np.max).astype(np.float32)

    if cmvn:
        mean = feature.mean(axis=1, keepdims=True)
        std = feature.std(axis=1, keepdims=True) + 1e-8
        feature = ((feature - mean) / std).astype(np.float32)

    target_mels, target_frames = target_shape
    if feature.shape[0] != target_mels:
        raise ValueError(
            f"Recovered feature has {feature.shape[0]} mel bins; "
            f"expected {target_mels}."
        )

    if feature.shape[1] > target_frames:
        feature = feature[:, :target_frames]
    elif feature.shape[1] < target_frames:
        pad = target_frames - feature.shape[1]
        feature = np.pad(
            feature,
            ((0, 0), (0, pad)),
            mode="constant",
        ).astype(np.float32)

    return feature.astype(np.float32)


def find_reference_feature_shape(
    feature_df: pd.DataFrame,
    feature_col: str,
    old_features_root: Path,
    repo_root: Path,
    feature_manifest: Path,
) -> tuple[int, int]:
    for value in feature_df[feature_col].dropna().head(100):
        try:
            resolved, _ = resolve_old_feature(
                value,
                old_features_root,
                repo_root,
                feature_manifest,
            )
            feature = np.load(resolved, mmap_mode="r")
            if feature.ndim == 2:
                return int(feature.shape[0]), int(feature.shape[1])
        except Exception:
            continue
    raise RuntimeError(
        "Could not load an existing feature to determine the reference shape."
    )


def count_consecutive_pairs(
    positive_starts: list[float],
    hop_sec: float,
    tolerance_sec: float,
) -> tuple[int, list[tuple[float, float]]]:
    starts = sorted(set(round(float(x), 6) for x in positive_starts))
    pairs = []
    for left, right in zip(starts, starts[1:]):
        if abs((right - left) - hop_sec) <= tolerance_sec:
            pairs.append((left, right))
    return len(pairs), pairs


def feature_relative_path(candidate_id: str, split: str, clip_id: str) -> Path:
    return (
        Path("recovered_low_energy")
        / safe_name(split)
        / safe_name(clip_id)
        / f"{safe_name(candidate_id)}.npy"
    )


def wav_relative_path(candidate_id: str, split: str, clip_id: str) -> Path:
    return (
        Path("recovered_low_energy")
        / safe_name(split)
        / safe_name(clip_id)
        / f"{safe_name(candidate_id)}.wav"
    )


def assign_if_present(row: pd.Series, columns: Iterable[str], value: Any) -> None:
    for column in columns:
        if column in row.index:
            row[column] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply reviewed low-energy silence recovery to TATA v0.9."
    )

    default_repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo_root", type=Path, default=default_repo)
    parser.add_argument("--review_csv", type=Path, default=None)
    parser.add_argument("--parent_manifest", type=Path, default=None)
    parser.add_argument("--feature_manifest", type=Path, default=None)
    parser.add_argument("--old_features_root", type=Path, default=None)
    parser.add_argument("--labels_json", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=None)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_ms", type=int, default=25)
    parser.add_argument("--feature_hop_ms", type=int, default=10)
    parser.add_argument("--cmvn", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--bandpass_low", type=float, default=50.0)
    parser.add_argument("--bandpass_high", type=float, default=7600.0)
    parser.add_argument(
        "--disable_bandpass",
        action="store_true",
        help="Disable the legacy wideband FFT filter.",
    )

    parser.add_argument(
        "--consecutive_tolerance_sec",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the new parent manifest, feature cache, and reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing silence_recovered_v09 output directory.",
    )
    parser.add_argument(
        "--skip_feature_link_validation",
        action="store_true",
        help="Not recommended. Skip resolving every existing feature in dry run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(f"[script] version: {SCRIPT_VERSION}")

    repo_root = args.repo_root.expanduser().resolve()
    v09_root = repo_root / "human_talk_workspace" / "tata_v0.9_pipeline"
    triage_root = v09_root / "tata_triage_model"

    review_csv = (
        args.review_csv.expanduser().resolve()
        if args.review_csv
        else (
            triage_root
            / "manual_review"
            / "low_energy_recovery_v09"
            / "low_energy_silence_review_queue_v09.csv"
        ).resolve()
    )
    parent_manifest = (
        args.parent_manifest.expanduser().resolve()
        if args.parent_manifest
        else (
            triage_root
            / "metadata"
            / "tata_seed_parent_manifest_v09_FINAL_REVIEWED.csv"
        ).resolve()
    )
    feature_manifest = (
        args.feature_manifest.expanduser().resolve()
        if args.feature_manifest
        else (
            triage_root
            / "feature_cache"
            / "metadata"
            / "multilabel_features_manifest_v09_FINAL.csv"
        ).resolve()
    )
    old_features_root = (
        args.old_features_root.expanduser().resolve()
        if args.old_features_root
        else (triage_root / "feature_cache" / "features").resolve()
    )
    labels_json = (
        args.labels_json.expanduser().resolve()
        if args.labels_json
        else (v09_root / "shared" / "human_talk_10label_schema.json").resolve()
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root
        else (triage_root / "silence_recovered_v09").resolve()
    )

    for path, name in [
        (review_csv, "review CSV"),
        (parent_manifest, "parent manifest"),
        (feature_manifest, "feature manifest"),
        (old_features_root, "old features root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found:\n{path}")

    labels = load_labels(labels_json)
    if "silence_present" not in labels:
        raise ValueError("labels_json does not contain silence_present.")

    review = pd.read_csv(review_csv, low_memory=False)
    parents = pd.read_csv(parent_manifest, low_memory=False)
    features = pd.read_csv(feature_manifest, low_memory=False)

    required_review = {
        "candidate_id",
        "clip_id",
        "split",
        "audio_path",
        "start_sec",
        "end_sec",
        "segment_sec",
        "hop_sec",
        "review_silence_present",
        "review_status",
    }
    missing = required_review - set(review.columns)
    if missing:
        raise ValueError(f"Review CSV missing columns: {sorted(missing)}")

    if review["candidate_id"].astype(str).duplicated().any():
        examples = (
            review.loc[
                review["candidate_id"].astype(str).duplicated(),
                "candidate_id",
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(f"Duplicate candidate_id values: {examples}")

    statuses = review["review_status"].fillna("").astype(str).str.strip().str.lower()
    not_reviewed = statuses.ne("reviewed")
    if not_reviewed.any():
        examples = review.loc[
            not_reviewed,
            ["candidate_id", "review_status"],
        ].head(10)
        raise ValueError(
            "The recovery CSV is not fully reviewed. "
            f"Pending/non-reviewed rows: {int(not_reviewed.sum())}.\n"
            "Examples:\n"
            f"{examples.to_string(index=False)}\n"
            "Use the locally saved CSV containing your completed manual review."
        )

    review["review_silence_present"] = as_binary(
        review["review_silence_present"],
        "review_silence_present",
    )

    review["start_sec"] = pd.to_numeric(
        review["start_sec"],
        errors="raise",
    ).astype(float)
    review["end_sec"] = pd.to_numeric(
        review["end_sec"],
        errors="raise",
    ).astype(float)
    review["segment_sec"] = pd.to_numeric(
        review["segment_sec"],
        errors="raise",
    ).astype(float)
    review["hop_sec"] = pd.to_numeric(
        review["hop_sec"],
        errors="raise",
    ).astype(float)

    invalid_duration = (
        (review["end_sec"] <= review["start_sec"])
        | (review["segment_sec"] <= 0)
        | (review["hop_sec"] <= 0)
    )
    if invalid_duration.any():
        raise ValueError(
            "Invalid timing values in review CSV:\n"
            + review.loc[
                invalid_duration,
                [
                    "candidate_id",
                    "start_sec",
                    "end_sec",
                    "segment_sec",
                    "hop_sec",
                ],
            ]
            .head(10)
            .to_string(index=False)
        )

    if "clip_id" not in parents.columns:
        raise ValueError("Parent manifest is missing clip_id.")
    if "clip_id" not in features.columns:
        raise ValueError("Feature manifest is missing clip_id.")
    if "split" not in features.columns:
        raise ValueError("Feature manifest is missing split.")
    if "feat_relpath" not in features.columns:
        raise ValueError("Feature manifest is missing feat_relpath.")

    parent_clip_ids = set(parents["clip_id"].astype(str))
    feature_clip_ids = set(features["clip_id"].astype(str))
    review_clip_ids = set(review["clip_id"].astype(str))

    absent_parent = sorted(review_clip_ids - parent_clip_ids)
    absent_feature = sorted(review_clip_ids - feature_clip_ids)
    if absent_parent:
        raise ValueError(
            f"{len(absent_parent)} reviewed clip IDs are absent from parent manifest. "
            f"Examples: {absent_parent[:10]}"
        )
    if absent_feature:
        raise ValueError(
            f"{len(absent_feature)} reviewed clip IDs are absent from feature manifest. "
            f"Examples: {absent_feature[:10]}"
        )

    parent_split_col = detect_column(
        parents,
        ["v09_split", "split"],
        "parent split column",
    )
    existing_start_col = detect_column(
        features,
        START_COLUMNS,
        "feature-manifest start-time column",
    )

    parent_split_map = (
        parents.assign(_clip=parents["clip_id"].astype(str))
        .set_index("_clip")[parent_split_col]
        .astype(str)
        .str.lower()
        .to_dict()
    )

    split_mismatch = []
    for row in review.itertuples(index=False):
        expected = parent_split_map[str(row.clip_id)]
        actual = str(row.split).strip().lower()
        if actual != expected:
            split_mismatch.append(
                {
                    "candidate_id": row.candidate_id,
                    "clip_id": row.clip_id,
                    "review_split": actual,
                    "parent_split": expected,
                }
            )
    if split_mismatch:
        raise ValueError(
            "Review/parent split mismatches detected:\n"
            + pd.DataFrame(split_mismatch).head(10).to_string(index=False)
        )

    duplicate_timeline_keys = review.duplicated(
        subset=["clip_id", "start_sec"],
        keep=False,
    )
    if duplicate_timeline_keys.any():
        raise ValueError(
            "Duplicate reviewed timeline positions:\n"
            + review.loc[
                duplicate_timeline_keys,
                ["candidate_id", "clip_id", "start_sec"],
            ]
            .head(20)
            .to_string(index=False)
        )

    existing_times = defaultdict(list)
    numeric_existing_starts = pd.to_numeric(
        features[existing_start_col],
        errors="coerce",
    )
    for feature_index, (clip_id, start) in enumerate(
        zip(features["clip_id"].astype(str), numeric_existing_starts)
    ):
        if not pd.isna(start):
            existing_times[clip_id].append((int(feature_index), float(start)))

    # A reviewed candidate can legitimately overlap a row already present in
    # the current feature manifest. In that case, do not append a duplicate
    # feature. Update only that existing row's silence label and provenance.
    collision_match_by_candidate: dict[str, dict[str, Any]] = {}
    collision_records = []
    used_existing_indices: dict[int, str] = {}

    for row in review.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        clip_id = str(row.clip_id)
        start_sec = float(row.start_sec)
        matches = [
            (feature_index, existing_start, abs(start_sec - existing_start))
            for feature_index, existing_start in existing_times[clip_id]
            if abs(start_sec - existing_start) <= args.consecutive_tolerance_sec
        ]
        if not matches:
            continue

        matches.sort(key=lambda item: (item[2], item[0]))
        feature_index, existing_start, difference = matches[0]

        if len(matches) > 1 and abs(matches[1][2] - difference) <= 1e-9:
            raise ValueError(
                "Ambiguous existing-window match for reviewed candidate "
                f"{candidate_id}: {matches[:5]}"
            )
        if feature_index in used_existing_indices:
            raise ValueError(
                "Two reviewed candidates map to the same existing feature row: "
                f"{used_existing_indices[feature_index]} and {candidate_id}."
            )

        used_existing_indices[feature_index] = candidate_id
        match = {
            "candidate_id": candidate_id,
            "clip_id": clip_id,
            "review_start_sec": start_sec,
            "existing_start_sec": float(existing_start),
            "start_difference_sec": float(difference),
            "existing_feature_index": int(feature_index),
        }
        collision_match_by_candidate[candidate_id] = match
        collision_records.append(match)

    collision_df = pd.DataFrame(collision_records)

    feature_col = "feat_relpath"
    reference_shape = find_reference_feature_shape(
        features,
        feature_col,
        old_features_root,
        repo_root,
        feature_manifest,
    )

    if reference_shape[0] != args.n_mels:
        raise ValueError(
            f"Existing features have shape {reference_shape}; "
            f"--n_mels is {args.n_mels}."
        )

    resolved_old_features: list[tuple[Path, Path]] = []
    if not args.skip_feature_link_validation or args.apply:
        missing_features = []
        duplicate_relpaths = set()
        seen_relpaths = set()

        for index, value in features[feature_col].items():
            try:
                resolved, relative = resolve_old_feature(
                    value,
                    old_features_root,
                    repo_root,
                    feature_manifest,
                )
                key = relative.as_posix().lower()
                if key in seen_relpaths:
                    duplicate_relpaths.add(relative.as_posix())
                seen_relpaths.add(key)
                resolved_old_features.append((resolved, relative))
            except Exception as exc:
                missing_features.append(
                    {
                        "row": int(index),
                        "feat_relpath": norm_text(value),
                        "error": str(exc),
                    }
                )

        if missing_features:
            raise FileNotFoundError(
                f"Could not resolve {len(missing_features)} existing features.\n"
                + pd.DataFrame(missing_features).head(10).to_string(index=False)
            )
        if duplicate_relpaths:
            raise ValueError(
                "Duplicate destination feature paths detected: "
                f"{sorted(duplicate_relpaths)[:10]}"
            )

    parent_label_cols = {
        label: parent_label_column(parents, label)
        for label in labels
    }

    parent_lookup = (
        parents.assign(_clip=parents["clip_id"].astype(str))
        .set_index("_clip", drop=False)
    )

    review_label_values_by_candidate = {}
    for _, item in review.iterrows():
        clip_id = str(item["clip_id"])
        parent_row = parent_lookup.loc[clip_id]
        label_values = {}

        for label in labels:
            if label == "silence_present":
                label_values[label] = int(item["review_silence_present"])
                continue

            review_column = f"parent_{label}"
            if review_column in review.columns and not pd.isna(item.get(review_column)):
                value = int(float(item[review_column]))
            else:
                value = int(float(parent_row[parent_label_cols[label]]))

            if value not in (0, 1):
                raise ValueError(
                    f"Non-binary inherited label for {clip_id}, {label}: {value}"
                )
            label_values[label] = value

        review_label_values_by_candidate[str(item["candidate_id"])] = label_values

    parent_summaries = []
    updated_parents = parents.copy()

    silence_parent_col = parent_label_cols["silence_present"]
    before_col = "v09_silence_present_before_recovery"
    updated_parents[before_col] = pd.to_numeric(
        updated_parents[silence_parent_col],
        errors="raise",
    ).astype(int)

    for column, default in [
        ("recovered_reviewed_window_count", 0),
        ("recovered_silent_window_count", 0),
        ("recovered_non_silent_window_count", 0),
        ("recovered_consecutive_silence_pair_count", 0),
        ("recovered_silence_event", 0),
        ("recovered_positive_start_times", ""),
        ("recovered_consecutive_pairs", ""),
        ("silence_parent_label_source", "existing_human_parent_review"),
    ]:
        updated_parents[column] = default

    parent_index_by_clip = {
        str(value): index
        for index, value in updated_parents["clip_id"].items()
    }

    for clip_id, group in review.groupby(review["clip_id"].astype(str), sort=True):
        positive = group[group["review_silence_present"] == 1]
        negative = group[group["review_silence_present"] == 0]

        hop_values = sorted(group["hop_sec"].round(6).unique().tolist())
        if len(hop_values) != 1:
            raise ValueError(
                f"Multiple hop sizes within parent {clip_id}: {hop_values}"
            )
        hop_sec = float(hop_values[0])

        positive_starts = positive["start_sec"].astype(float).tolist()
        pair_count, pairs = count_consecutive_pairs(
            positive_starts,
            hop_sec,
            args.consecutive_tolerance_sec,
        )
        event = int(pair_count >= 1)

        parent_index = parent_index_by_clip[clip_id]
        old_parent_silence = int(
            updated_parents.at[parent_index, before_col]
        )
        new_parent_silence = max(old_parent_silence, event)

        updated_parents.at[parent_index, silence_parent_col] = new_parent_silence
        updated_parents.at[
            parent_index,
            "recovered_reviewed_window_count",
        ] = int(len(group))
        updated_parents.at[
            parent_index,
            "recovered_silent_window_count",
        ] = int(len(positive))
        updated_parents.at[
            parent_index,
            "recovered_non_silent_window_count",
        ] = int(len(negative))
        updated_parents.at[
            parent_index,
            "recovered_consecutive_silence_pair_count",
        ] = int(pair_count)
        updated_parents.at[
            parent_index,
            "recovered_silence_event",
        ] = event
        updated_parents.at[
            parent_index,
            "recovered_positive_start_times",
        ] = "|".join(f"{value:.3f}" for value in sorted(positive_starts))
        updated_parents.at[
            parent_index,
            "recovered_consecutive_pairs",
        ] = "|".join(
            f"{left:.3f}->{right:.3f}"
            for left, right in pairs
        )

        if old_parent_silence == 1:
            source = "existing_human_parent_review"
        elif event == 1:
            source = "two_consecutive_human_reviewed_silent_windows"
        else:
            source = "existing_parent_negative_no_recovered_event"

        updated_parents.at[
            parent_index,
            "silence_parent_label_source",
        ] = source

        parent_summaries.append(
            {
                "clip_id": clip_id,
                "split": parent_split_map[clip_id],
                "reviewed_window_count": int(len(group)),
                "silent_window_count": int(len(positive)),
                "non_silent_window_count": int(len(negative)),
                "hop_sec": hop_sec,
                "consecutive_pair_count": int(pair_count),
                "consecutive_pairs": "|".join(
                    f"{left:.3f}->{right:.3f}"
                    for left, right in pairs
                ),
                "old_parent_silence_present": old_parent_silence,
                "recovered_silence_event": event,
                "new_parent_silence_present": new_parent_silence,
                "parent_label_changed_0_to_1": int(
                    old_parent_silence == 0 and new_parent_silence == 1
                ),
            }
        )

    parent_summary_df = pd.DataFrame(parent_summaries)

    parent_sep = detect_label_separator(updated_parents)
    if "labels" in updated_parents.columns:
        for index, row in updated_parents.iterrows():
            active = [
                label
                for label in labels
                if int(row[parent_label_cols[label]]) == 1
            ]
            updated_parents.at[index, "labels"] = labels_string(
                active,
                parent_sep,
            )
    for count_col in ["num_active_labels", "num_labels"]:
        if count_col in updated_parents.columns:
            updated_parents[count_col] = [
                sum(
                    int(updated_parents.at[index, parent_label_cols[label]])
                    for label in labels
                )
                for index in updated_parents.index
            ]

    review_positive_count = int((review["review_silence_present"] == 1).sum())
    review_negative_count = int((review["review_silence_present"] == 0).sum())
    changed_parent_count = int(
        parent_summary_df["parent_label_changed_0_to_1"].sum()
        if len(parent_summary_df)
        else 0
    )

    existing_window_update_count = int(len(collision_match_by_candidate))
    newly_appended_review_count = int(len(review) - existing_window_update_count)
    intended_total_rows = int(len(features) + newly_appended_review_count)

    append_review = review[
        ~review["candidate_id"].astype(str).isin(
            set(collision_match_by_candidate.keys())
        )
    ].copy()
    intended_split_counts = (
        pd.concat(
            [
                features["split"].astype(str).str.lower(),
                append_review["split"].astype(str).str.lower(),
            ],
            ignore_index=True,
        )
        .value_counts()
        .to_dict()
    )

    summary = {
        "generated_utc": utc_now(),
        "mode": "apply" if args.apply else "dry_run",
        "review_csv": str(review_csv),
        "input_parent_manifest": str(parent_manifest),
        "input_feature_manifest": str(feature_manifest),
        "input_features_root": str(old_features_root),
        "output_root": str(out_root),
        "labels": labels,
        "existing_parent_rows": int(len(parents)),
        "existing_feature_rows": int(len(features)),
        "reviewed_recovered_rows": int(len(review)),
        "reviewed_existing_windows_updated_in_place": existing_window_update_count,
        "reviewed_missing_windows_appended": newly_appended_review_count,
        "reviewed_silence_positive_rows": review_positive_count,
        "reviewed_silence_negative_hard_rows": review_negative_count,
        "affected_parent_count": int(review["clip_id"].astype(str).nunique()),
        "parents_changed_from_silence_0_to_1": changed_parent_count,
        "new_feature_manifest_rows": intended_total_rows,
        "new_split_counts": {
            str(key): int(value)
            for key, value in intended_split_counts.items()
        },
        "reference_feature_shape": list(reference_shape),
        "feature_preprocessing": {
            "sample_rate": int(args.sample_rate),
            "bandpass_low": (
                None if args.disable_bandpass else float(args.bandpass_low)
            ),
            "bandpass_high": (
                None if args.disable_bandpass else float(args.bandpass_high)
            ),
            "parent_peak_normalisation": 0.8913,
            "n_mels": int(args.n_mels),
            "n_fft": int(args.n_fft),
            "win_ms": int(args.win_ms),
            "feature_hop_ms": int(args.feature_hop_ms),
            "cmvn": bool(args.cmvn),
            "logmel_db_reference": "per-segment maximum",
        },
        "policy": {
            "all_reviewed_rows_included": True,
            "existing_reviewed_windows_updated_in_place": True,
            "only_truly_missing_reviewed_windows_appended": True,
            "review_keep_segment_controls_inclusion": False,
            "recovered_silence_label": "human_segment_review",
            "other_recovered_labels": "parent_inherited",
            "parent_silence_event": (
                "at least two reviewed silent windows separated by one hop"
            ),
            "existing_parent_positive_is_preserved": True,
            "existing_feature_rows_relabelled": False,
        },
        "existing_window_matches": collision_records,
        "existing_files_modified": False,
    }

    print("\n=== TATA v0.9 reviewed low-energy recovery ===")
    print(f"Mode:                              {summary['mode']}")
    print(f"Existing parent rows:              {len(parents):,}")
    print(f"Existing feature rows:             {len(features):,}")
    print(f"Reviewed rows to incorporate:      {len(review):,}")
    print(f"  Existing rows updated in place:  {existing_window_update_count:,}")
    print(f"  Missing rows to append:           {newly_appended_review_count:,}")
    print(f"  Silence positives:               {review_positive_count:,}")
    print(f"  Silence hard negatives:          {review_negative_count:,}")
    print(
        "Affected parents:                  "
        f"{review['clip_id'].astype(str).nunique():,}"
    )
    print(f"Parents changed silence 0 -> 1:    {changed_parent_count:,}")
    print(f"New feature-manifest rows:         {intended_total_rows:,}")
    print(f"New split counts:                  {summary['new_split_counts']}")
    print(f"Reference feature shape:           {reference_shape}")
    print(f"Output root:                       {out_root}")

    if not args.apply:
        print("\n[DRY RUN COMPLETE] No files were created or modified.")
        print("Run again with --apply after reviewing this summary.")
        return 0

    staging = out_root.with_name(out_root.name + ".__staging__")
    if staging.exists():
        shutil.rmtree(staging)

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists:\n{out_root}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(out_root)

    metadata_root = staging / "metadata"
    reports_root = staging / "reports"
    new_cache_root = staging / "feature_cache"
    new_features_root = new_cache_root / "features"
    new_manifest_root = new_cache_root / "metadata"
    new_wav_root = staging / "segment_wavs"

    for directory in [
        metadata_root,
        reports_root,
        new_features_root,
        new_manifest_root,
        new_wav_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        new_features = features.copy()
        normalised_old_relpaths = []
        link_counts = {"hardlink": 0, "copy": 0}

        if not resolved_old_features:
            for value in features[feature_col]:
                resolved_old_features.append(
                    resolve_old_feature(
                        value,
                        old_features_root,
                        repo_root,
                        feature_manifest,
                    )
                )

        for resolved, relative in resolved_old_features:
            destination = new_features_root / relative
            method = hardlink_or_copy(resolved, destination)
            link_counts[method] += 1
            normalised_old_relpaths.append(relative.as_posix())

        new_features["feat_relpath"] = normalised_old_relpaths

        for column in PROVENANCE_COLUMNS:
            if column not in new_features.columns:
                new_features[column] = pd.Series(
                    [""] * len(new_features),
                    index=new_features.index,
                    dtype="object",
                )
            else:
                new_features[column] = new_features[column].astype("object")

        if "labels" in new_features.columns:
            new_features["labels"] = new_features["labels"].astype("object")

        feature_sep = detect_label_separator(new_features)
        updated_existing_rows = []

        # Apply human silence review to candidates that already have a
        # feature-manifest row. Preserve the feature itself and all other
        # labels; update only silence metadata and derived label fields.
        review_by_candidate = review.assign(
            _candidate=review["candidate_id"].astype(str)
        ).set_index("_candidate", drop=False)

        for candidate_id, match in collision_match_by_candidate.items():
            feature_index = int(match["existing_feature_index"])
            item = review_by_candidate.loc[candidate_id]
            human_silence = int(item["review_silence_present"])

            new_features.at[feature_index, "silence_present"] = human_silence
            new_features.at[feature_index, "silence_label_source"] = (
                "human_segment_review_existing_window"
            )
            new_features.at[feature_index, "recovery_candidate_id"] = candidate_id
            new_features.at[feature_index, "recovery_parent_audio_path"] = str(
                item.get("audio_path", "")
            )
            new_features.at[feature_index, "recovery_raw_rms_dbfs"] = item.get(
                "raw_rms_dbfs", ""
            )
            new_features.at[feature_index, "recovery_raw_peak_dbfs"] = item.get(
                "raw_peak_dbfs", ""
            )
            new_features.at[feature_index, "recovery_zero_crossing_rate"] = item.get(
                "zero_crossing_rate", ""
            )
            new_features.at[feature_index, "recovery_spectral_flatness"] = item.get(
                "spectral_flatness", ""
            )
            new_features.at[feature_index, "recovery_energy_band"] = item.get(
                "energy_band", ""
            )
            new_features.at[feature_index, "recovery_review_priority"] = item.get(
                "review_priority", ""
            )
            new_features.at[
                feature_index, "recovery_review_keep_segment_original"
            ] = item.get("review_keep_segment", "")
            new_features.at[feature_index, "recovery_review_notes"] = item.get(
                "review_notes", ""
            )
            new_features.at[feature_index, "silence_recovery_action"] = (
                "updated_existing_window"
            )

            active_labels = [
                label
                for label in labels
                if int(float(new_features.at[feature_index, label])) == 1
            ]
            if "labels" in new_features.columns:
                new_features.at[feature_index, "labels"] = labels_string(
                    active_labels,
                    feature_sep,
                )
            for count_col in ["num_active_labels", "num_labels"]:
                if count_col in new_features.columns:
                    new_features.at[feature_index, count_col] = int(
                        len(active_labels)
                    )

            record = dict(match)
            record.update(
                {
                    "review_silence_present": human_silence,
                    "feat_relpath": str(
                        new_features.at[feature_index, "feat_relpath"]
                    ),
                    "old_row_updated": 1,
                }
            )
            updated_existing_rows.append(record)

        templates = {
            str(clip_id): group.iloc[0].copy()
            for clip_id, group in new_features.groupby(
                new_features["clip_id"].astype(str),
                sort=False,
            )
        }

        recovered_rows = []
        timeline_rows = []

        bandpass_low = None if args.disable_bandpass else args.bandpass_low
        bandpass_high = None if args.disable_bandpass else args.bandpass_high

        parent_audio_cache: dict[str, tuple[np.ndarray, int]] = {}

        ordered_review = review.sort_values(
            ["split", "clip_id", "start_sec", "candidate_id"]
        ).reset_index(drop=True)

        for item_index, item in ordered_review.iterrows():
            clip_id = str(item["clip_id"])
            candidate_id = str(item["candidate_id"])
            split = str(item["split"]).strip().lower()
            audio_path = Path(str(item["audio_path"])).expanduser()

            existing_match = collision_match_by_candidate.get(candidate_id)
            if existing_match is not None:
                feature_index = int(existing_match["existing_feature_index"])
                timeline_rows.append(
                    {
                        "clip_id": clip_id,
                        "split": split,
                        "candidate_id": candidate_id,
                        "start_sec": float(item["start_sec"]),
                        "end_sec": float(item["end_sec"]),
                        "timeline_source": "human_review_updated_existing_window",
                        "silence_present": int(item["review_silence_present"]),
                        "silence_label_source": (
                            "human_segment_review_existing_window"
                        ),
                        "feat_relpath": str(
                            new_features.at[feature_index, "feat_relpath"]
                        ),
                        "segment_wav_path": "",
                        "parent_audio_path": str(audio_path),
                    }
                )
                continue

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Parent audio not found for {candidate_id}:\n{audio_path}"
                )

            cache_key = str(audio_path.resolve()).lower()
            if cache_key not in parent_audio_cache:
                parent_audio_cache[cache_key] = legacy_preprocess_parent(
                    audio_path,
                    args.sample_rate,
                    bandpass_low,
                    bandpass_high,
                )

            parent_audio, sr = parent_audio_cache[cache_key]
            start_sec = float(item["start_sec"])
            segment_sec = float(item["segment_sec"])
            start_sample = int(round(start_sec * sr))
            segment_samples = int(round(segment_sec * sr))
            segment = pad_or_trim(
                parent_audio[start_sample : start_sample + segment_samples],
                segment_samples,
            )

            relative_feature = feature_relative_path(
                candidate_id,
                split,
                clip_id,
            )
            relative_wav = wav_relative_path(
                candidate_id,
                split,
                clip_id,
            )
            feature_destination = new_features_root / relative_feature
            wav_destination = new_wav_root / relative_wav

            feature_destination.parent.mkdir(parents=True, exist_ok=True)
            wav_destination.parent.mkdir(parents=True, exist_ok=True)

            feature = extract_logmel(
                segment,
                sr,
                args.n_mels,
                args.n_fft,
                args.win_ms,
                args.feature_hop_ms,
                args.cmvn,
                reference_shape,
            )
            np.save(feature_destination, feature)
            sf.write(wav_destination, segment, sr)

            template = templates[clip_id].copy()
            label_values = review_label_values_by_candidate[candidate_id]

            recovered_id = f"recovered_{candidate_id}"
            assign_if_present(template, ID_COLUMNS, recovered_id)

            template["clip_id"] = clip_id
            template["split"] = split
            template["feat_relpath"] = relative_feature.as_posix()

            assign_if_present(template, START_COLUMNS, start_sec)
            assign_if_present(template, END_COLUMNS, float(item["end_sec"]))
            assign_if_present(template, DURATION_COLUMNS, segment_sec)
            assign_if_present(
                template,
                ABS_AUDIO_COLUMNS,
                str(wav_destination.resolve()),
            )

            if "segment_wav_relpath" in template.index:
                template["segment_wav_relpath"] = relative_wav.as_posix()

            if "file_name" in template.index:
                template["file_name"] = str(
                    item.get(
                        "review_audio_file",
                        relative_wav.name,
                    )
                )

            if "segment_index" in template.index:
                template["segment_index"] = int(item_index)

            for label in labels:
                template[label] = int(label_values[label])

            active_labels = [
                label
                for label in labels
                if int(label_values[label]) == 1
            ]
            if "labels" in template.index:
                template["labels"] = labels_string(
                    active_labels,
                    feature_sep,
                )
            for count_col in ["num_active_labels", "num_labels"]:
                if count_col in template.index:
                    template[count_col] = int(len(active_labels))

            if "feature_shape" in template.index:
                template["feature_shape"] = (
                    f"{reference_shape[0]}x{reference_shape[1]}"
                )
            if "feature_sample_rate" in template.index:
                template["feature_sample_rate"] = int(args.sample_rate)
            if "sample_rate" in template.index:
                template["sample_rate"] = int(args.sample_rate)
            if "feature_clip_sec" in template.index:
                template["feature_clip_sec"] = segment_sec
            if "feature_n_mels" in template.index:
                template["feature_n_mels"] = int(args.n_mels)
            if "feature_n_fft" in template.index:
                template["feature_n_fft"] = int(args.n_fft)
            if "feature_win_ms" in template.index:
                template["feature_win_ms"] = int(args.win_ms)
            if "feature_hop_ms" in template.index:
                template["feature_hop_ms"] = int(args.feature_hop_ms)
            if "feature_cmvn" in template.index:
                template["feature_cmvn"] = int(bool(args.cmvn))

            template["v09_data_origin"] = (
                "recovered_low_energy_human_reviewed"
            )
            template["silence_label_source"] = "human_segment_review"
            template["other_labels_source"] = "parent_inherited"
            template["recovery_candidate_id"] = candidate_id
            template["recovery_parent_audio_path"] = str(
                audio_path.resolve()
            )
            template["recovery_raw_rms_dbfs"] = item.get(
                "raw_rms_dbfs",
                "",
            )
            template["recovery_raw_peak_dbfs"] = item.get(
                "raw_peak_dbfs",
                "",
            )
            template["recovery_zero_crossing_rate"] = item.get(
                "zero_crossing_rate",
                "",
            )
            template["recovery_spectral_flatness"] = item.get(
                "spectral_flatness",
                "",
            )
            template["recovery_energy_band"] = item.get(
                "energy_band",
                "",
            )
            template["recovery_review_priority"] = item.get(
                "review_priority",
                "",
            )
            template["recovery_review_keep_segment_original"] = item.get(
                "review_keep_segment",
                "",
            )
            template["recovery_review_notes"] = item.get(
                "review_notes",
                "",
            )
            template["silence_recovery_action"] = "appended_missing_window"
            template["recovery_legacy_processed_rms_dbfs"] = rms_dbfs(
                segment
            )

            recovered_rows.append(template)

            timeline_rows.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "candidate_id": candidate_id,
                    "start_sec": start_sec,
                    "end_sec": float(item["end_sec"]),
                    "timeline_source": (
                        "recovered_low_energy_human_reviewed"
                    ),
                    "silence_present": int(
                        item["review_silence_present"]
                    ),
                    "silence_label_source": "human_segment_review",
                    "feat_relpath": relative_feature.as_posix(),
                    "segment_wav_path": str(wav_destination.resolve()),
                    "parent_audio_path": str(audio_path.resolve()),
                }
            )

        recovered_df = pd.DataFrame(recovered_rows)
        merged_features = pd.concat(
            [new_features, recovered_df],
            ignore_index=True,
            sort=False,
        )

        if "global_index" in merged_features.columns:
            merged_features["global_index"] = np.arange(
                len(merged_features),
                dtype=np.int64,
            )

        if merged_features["feat_relpath"].astype(str).duplicated().any():
            duplicates = (
                merged_features.loc[
                    merged_features["feat_relpath"]
                    .astype(str)
                    .duplicated(),
                    "feat_relpath",
                ]
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"Duplicate feat_relpath after merge: {duplicates}"
            )

        missing_new_features = [
            value
            for value in merged_features["feat_relpath"].astype(str)
            if not (new_features_root / Path(value)).is_file()
        ]
        if missing_new_features:
            raise FileNotFoundError(
                f"{len(missing_new_features)} merged feature files are missing. "
                f"Examples: {missing_new_features[:10]}"
            )

        split_leakage = (
            merged_features.assign(
                _clip=merged_features["clip_id"].astype(str),
                _split=merged_features["split"].astype(str).str.lower(),
            )
            .groupby("_clip")["_split"]
            .nunique()
        )
        leaking = split_leakage[split_leakage > 1]
        if len(leaking):
            raise ValueError(
                "Split leakage after merge. Examples: "
                f"{leaking.head(10).to_dict()}"
            )

        actual_split_counts = (
            merged_features["split"]
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )

        if len(merged_features) != intended_total_rows:
            raise ValueError(
                f"Unexpected merged row count: {len(merged_features)}; "
                f"expected {intended_total_rows}."
            )

        summary["actual_feature_manifest_rows"] = int(
            len(merged_features)
        )
        summary["actual_split_counts"] = {
            str(key): int(value)
            for key, value in actual_split_counts.items()
        }
        summary["old_feature_materialisation"] = link_counts
        summary["existing_feature_rows_updated_by_review"] = int(
            len(updated_existing_rows)
        )
        summary["new_recovered_feature_count"] = int(
            len(recovered_df)
        )
        summary["new_recovered_wav_count"] = int(
            len(recovered_df)
        )

        output_parent_manifest = (
            metadata_root
            / "tata_seed_parent_manifest_v09_SILENCE_RECOVERED.csv"
        )
        output_feature_manifest = (
            new_manifest_root
            / "multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"
        )
        output_recovered_rows = (
            metadata_root
            / "recovered_low_energy_segments_v09.csv"
        )
        output_updated_existing_rows = (
            metadata_root
            / "human_review_updates_to_existing_windows_v09.csv"
        )
        output_parent_summary = (
            reports_root
            / "recovered_silence_parent_summary_v09.csv"
        )
        output_timeline = (
            reports_root
            / "merged_parent_timeline_v09.csv"
        )
        output_summary = (
            reports_root
            / "silence_recovery_summary_v09.json"
        )

        updated_parents.to_csv(
            output_parent_manifest,
            index=False,
            encoding="utf-8",
        )
        merged_features.to_csv(
            output_feature_manifest,
            index=False,
            encoding="utf-8",
        )
        recovered_df.to_csv(
            output_recovered_rows,
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(updated_existing_rows).to_csv(
            output_updated_existing_rows,
            index=False,
            encoding="utf-8",
        )
        parent_summary_df.to_csv(
            output_parent_summary,
            index=False,
            encoding="utf-8",
        )
        # Full chronological timeline: all existing rows plus all recovered rows.
        timeline_start_col = detect_column(
            merged_features,
            START_COLUMNS,
            "merged feature start-time column",
        )
        timeline_end_col = detect_column(
            merged_features,
            END_COLUMNS,
            "merged feature end-time column",
            required=False,
        )
        timeline_df = pd.DataFrame({
            "clip_id": merged_features["clip_id"].astype(str),
            "split": merged_features["split"].astype(str).str.lower(),
            "start_sec": pd.to_numeric(
                merged_features[timeline_start_col], errors="coerce"
            ),
            "feat_relpath": merged_features["feat_relpath"].astype(str),
            "silence_present": pd.to_numeric(
                merged_features["silence_present"], errors="coerce"
            ).fillna(0).astype(int),
        })
        if timeline_end_col is not None:
            timeline_df["end_sec"] = pd.to_numeric(
                merged_features[timeline_end_col], errors="coerce"
            )
        else:
            duration_col = detect_column(
                merged_features,
                DURATION_COLUMNS,
                "merged feature duration column",
                required=False,
            )
            if duration_col is not None:
                timeline_df["end_sec"] = timeline_df["start_sec"] + pd.to_numeric(
                    merged_features[duration_col], errors="coerce"
                )
            else:
                timeline_df["end_sec"] = np.nan

        if "v09_data_origin" in merged_features.columns:
            timeline_df["timeline_source"] = merged_features[
                "v09_data_origin"
            ].fillna("").astype(str)
        else:
            timeline_df["timeline_source"] = "existing_feature_manifest"
        timeline_df.loc[
            timeline_df["timeline_source"].eq(""), "timeline_source"
        ] = "existing_feature_manifest"

        if "silence_label_source" in merged_features.columns:
            timeline_df["silence_label_source"] = merged_features[
                "silence_label_source"
            ].fillna("").astype(str)
        else:
            timeline_df["silence_label_source"] = ""
        timeline_df.loc[
            timeline_df["silence_label_source"].eq(""),
            "silence_label_source",
        ] = "existing_manifest_parent_inherited_or_prior"

        if "recovery_candidate_id" in merged_features.columns:
            timeline_df["recovery_candidate_id"] = merged_features[
                "recovery_candidate_id"
            ].fillna("").astype(str)
        else:
            timeline_df["recovery_candidate_id"] = ""

        timeline_df = timeline_df.sort_values(
            ["split", "clip_id", "start_sec", "feat_relpath"]
        ).reset_index(drop=True)
        timeline_df["timeline_index_within_parent"] = (
            timeline_df.groupby("clip_id").cumcount()
        )
        timeline_df["previous_start_sec"] = (
            timeline_df.groupby("clip_id")["start_sec"].shift(1)
        )
        timeline_df["next_start_sec"] = (
            timeline_df.groupby("clip_id")["start_sec"].shift(-1)
        )
        timeline_df.to_csv(
            output_timeline,
            index=False,
            encoding="utf-8",
        )
        output_summary.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        promotion_error = None
        promoted = False
        for attempt in range(1, 11):
            try:
                staging.rename(out_root)
                promoted = True
                break
            except PermissionError as exc:
                promotion_error = exc
                print(
                    f"[finalise] Windows rename attempt {attempt}/10 failed: {exc}"
                )
                time.sleep(1.5)

        effective_root = out_root if promoted else staging
        summary["promotion_succeeded"] = bool(promoted)
        summary["actual_output_root"] = str(effective_root)
        summary["promotion_error"] = (
            "" if promotion_error is None else str(promotion_error)
        )
        effective_summary = (
            effective_root / "reports" / "silence_recovery_summary_v09.json"
        )
        effective_summary.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        if not promoted:
            marker = effective_root / "WINDOWS_PROMOTION_PENDING.txt"
            marker.write_text(
                "The dataset build completed successfully, but Windows denied "
                "the final directory rename. Close File Explorer/PyCharm views "
                "inside this folder, then rename this directory from "
                "silence_recovered_v09.__staging__ to silence_recovered_v09.\n"
                f"Last error: {promotion_error}\n",
                encoding="utf-8",
            )
            print(
                "\n[WARNING] Build completed, but Windows still blocked "
                "the final folder rename."
            )
            print(
                "Completed output has been PRESERVED at: "
                f"{effective_root}"
            )

    except Exception:
        if staging.exists():
            print(
                "\n[RECOVERY] Staging output was preserved at: "
                f"{staging}"
            )
        raise

    print("\n[COMPLETE] Silence-recovered v0.9 cache created.")
    print(
        "Parent manifest:                   "
        f"{effective_root / 'metadata' / 'tata_seed_parent_manifest_v09_SILENCE_RECOVERED.csv'}"
    )
    print(
        "Feature manifest:                  "
        f"{effective_root / 'feature_cache' / 'metadata' / 'multilabel_features_manifest_v09_SILENCE_RECOVERED.csv'}"
    )
    print(
        "Features root:                     "
        f"{effective_root / 'feature_cache' / 'features'}"
    )
    print(
        "Parent recovery report:            "
        f"{effective_root / 'reports' / 'recovered_silence_parent_summary_v09.csv'}"
    )
    print(
        "Summary:                           "
        f"{effective_root / 'reports' / 'silence_recovery_summary_v09.json'}"
    )
    print(
        "Final folder promotion:             "
        + ("Succeeded" if promoted else "Pending manual rename")
    )
    print("Existing v0.9 files modified:       No")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
