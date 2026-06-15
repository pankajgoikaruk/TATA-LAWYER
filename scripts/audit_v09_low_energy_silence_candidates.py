#!/usr/bin/env python
"""
Audit low-energy 1-second windows that may be absent from the current v0.9
TATA feature manifest.

This is a non-destructive audit:
- It does not change labels, manifests, features, or source audio.
- It measures RMS on raw audio BEFORE peak normalisation.
- It uses 1.0-second windows and 0.5-second hop.
- It compares candidate start times with starts already represented in the
  final v0.9 feature manifest.
- It writes a review CSV and optional WAV clips only when --apply is used.

The audit focuses on the original 2,074 parents represented by reused v0.6
features. The 27 newly added silence parents are excluded because they were
already processed with the v0.9 silence-aware path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg",
    ".aac", ".wma", ".aiff", ".aif", ".aifc", ".au",
}

PATH_COLUMNS = [
    "source_path",
    "source_file",
    "raw_file",
    "parent_file",
    "abs_path",
    "filepath",
    "audio_path",
    "final_path",
]

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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_path_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().strip('"').replace("\\", "/")


def safe_token(value: object, max_len: int = 80) -> str:
    import re
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return (text.strip("._-") or "item")[:max_len]


def rms_dbfs(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def peak_dbfs(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    peak = float(np.max(np.abs(y)))
    return float(20.0 * np.log10(max(peak, 1e-12)))


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, dtype="float32")
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=False)
        y = np.asarray(y, dtype=np.float32)

    y = to_mono(y)
    sr = int(sr)
    if sr != int(target_sr):
        y = librosa.resample(y, orig_sr=sr, target_sr=int(target_sr))
        sr = int(target_sr)
    return np.asarray(y, dtype=np.float32), sr


def pad_or_trim(y: np.ndarray, length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)), mode="constant").astype(np.float32)


def zero_crossing_rate(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(y)).astype(np.float32))))


def spectral_flatness(y: np.ndarray, n_fft: int = 1024) -> float:
    y = np.asarray(y, dtype=np.float32)
    if not np.any(np.abs(y) > 0):
        return 0.0
    value = librosa.feature.spectral_flatness(y=y, n_fft=n_fft)
    return float(np.mean(value))


def discover_search_roots(repo: Path, explicit: list[Path]) -> list[Path]:
    roots: list[Path] = []

    for path in explicit:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            roots.append(resolved)

    defaults = [
        repo / "dataset",
        repo / "human_talk_workspace",
        repo.parent / "NeuroAccuExit-ASHADIP" / "dataset",
        repo.parent / "NeuroAccuExit-ASHADIP" / "human_talk_workspace",
    ]
    for path in defaults:
        if path.is_dir():
            roots.append(path.resolve())

    unique = []
    seen = set()
    for path in roots:
        key = str(path).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def build_basename_index(roots: list[Path], wanted_names: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            key = path.name.lower()
            if key in wanted_names:
                index[key].append(path.resolve())
    return index


def existing_path_from_values(values: Iterable[object], repo: Path) -> Path | None:
    for value in values:
        text = normalise_path_text(value)
        if not text:
            continue
        path = Path(text)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(repo / path)
            candidates.append(repo.parent / "NeuroAccuExit-ASHADIP" / path)
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except OSError:
                pass
            if candidate.is_file():
                return candidate
    return None


def resolve_parent_audio(
    clip_id: str,
    file_name: str,
    parent_row: pd.Series,
    segment_rows: pd.DataFrame,
    repo: Path,
    basename_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    values = []
    for column in PATH_COLUMNS:
        if column in parent_row.index:
            values.append(parent_row[column])
    for column in PATH_COLUMNS:
        if column in segment_rows.columns:
            values.extend(segment_rows[column].dropna().head(20).tolist())

    path = existing_path_from_values(values, repo)
    if path is not None:
        return path, "manifest_path"

    matches = basename_index.get(str(file_name).lower(), [])
    if len(matches) == 1:
        return matches[0], "basename_search"
    if len(matches) > 1:
        return None, f"ambiguous_basename:{len(matches)}"
    return None, "not_found"


def routing_label(raw_rms_dbfs: float, parent_silence: int) -> tuple[str, str]:
    if raw_rms_dbfs <= -60.0:
        energy_band = "near_digital_silence"
    elif raw_rms_dbfs <= -50.0:
        energy_band = "very_low_energy"
    elif raw_rms_dbfs <= -40.0:
        energy_band = "low_energy"
    else:
        energy_band = "borderline_low_energy"

    if int(parent_silence) == 1:
        priority = "high"
    elif raw_rms_dbfs <= -55.0:
        priority = "medium"
    else:
        priority = "low"
    return energy_band, priority


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit potentially discarded low-energy v0.9 TATA windows."
    )
    default_repo = Path(__file__).resolve().parents[1]
    p.add_argument("--repo_root", type=Path, default=default_repo)
    p.add_argument("--parent_manifest", type=Path, default=None)
    p.add_argument("--feature_manifest", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument(
        "--audio_search_root",
        type=Path,
        action="append",
        default=[],
        help="Optional additional root to search for parent audio. Repeatable.",
    )
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--segment_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument(
        "--candidate_max_dbfs",
        type=float,
        default=-35.0,
    )
    p.add_argument(
        "--existing_start_tolerance_sec",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--max_candidates_per_parent",
        type=int,
        default=5,
    )
    p.add_argument("--export_audio", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    repo = args.repo_root.expanduser().resolve()
    v09 = repo / "human_talk_workspace" / "tata_v0.9_pipeline"

    parent_manifest = (
        args.parent_manifest.expanduser().resolve()
        if args.parent_manifest
        else (
            v09
            / "tata_triage_model"
            / "metadata"
            / "tata_seed_parent_manifest_v09_FINAL_REVIEWED.csv"
        ).resolve()
    )
    feature_manifest = (
        args.feature_manifest.expanduser().resolve()
        if args.feature_manifest
        else (
            v09
            / "tata_triage_model"
            / "feature_cache"
            / "metadata"
            / "multilabel_features_manifest_v09_FINAL.csv"
        ).resolve()
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root
        else (
            v09
            / "tata_triage_model"
            / "manual_review"
            / "low_energy_recovery_v09"
        ).resolve()
    )

    if not parent_manifest.is_file():
        raise FileNotFoundError(f"Parent manifest not found:\n{parent_manifest}")
    if not feature_manifest.is_file():
        raise FileNotFoundError(f"Feature manifest not found:\n{feature_manifest}")

    parents = pd.read_csv(parent_manifest, low_memory=False)
    features = pd.read_csv(feature_manifest, low_memory=False)

    required_parent = {"clip_id", "file_name", "v09_split", "v09_silence_present"}
    missing = required_parent - set(parents.columns)
    if missing:
        raise ValueError(f"Parent manifest missing columns: {sorted(missing)}")

    required_feature = {"clip_id", "split", "start_sec", "v09_data_origin"}
    missing = required_feature - set(features.columns)
    if missing:
        raise ValueError(f"Feature manifest missing columns: {sorted(missing)}")

    original_ids = set(
        features.loc[
            features["v09_data_origin"].astype(str) == "reused_v06_feature",
            "clip_id",
        ].astype(str)
    )
    original_parents = parents[
        parents["clip_id"].astype(str).isin(original_ids)
    ].copy()

    if len(original_parents) != 2074:
        raise ValueError(
            f"Expected 2,074 original parents, found {len(original_parents)}."
        )

    feature_groups = {
        str(clip_id): group.copy()
        for clip_id, group in features.groupby(features["clip_id"].astype(str))
    }

    wanted_names = set(
        original_parents["file_name"].astype(str).str.lower().tolist()
    )
    search_roots = discover_search_roots(repo, args.audio_search_root)
    basename_index = build_basename_index(search_roots, wanted_names)

    win = int(round(args.segment_sec * args.sample_rate))
    hop = int(round(args.hop_sec * args.sample_rate))

    candidate_rows = []
    candidate_audio = []
    missing_audio_rows = []
    scanned_windows = 0
    represented_windows = 0
    low_energy_missing_windows = 0

    for _, parent in original_parents.sort_values("clip_id").iterrows():
        clip_id = str(parent["clip_id"])
        file_name = str(parent["file_name"])
        seg_rows = feature_groups.get(clip_id, pd.DataFrame())

        audio_path, resolution = resolve_parent_audio(
            clip_id, file_name, parent, seg_rows, repo, basename_index
        )

        if audio_path is None:
            missing_audio_rows.append(
                {
                    "clip_id": clip_id,
                    "file_name": file_name,
                    "split": str(parent["v09_split"]),
                    "resolution_status": resolution,
                }
            )
            continue

        y, sr = load_audio(audio_path, args.sample_rate)
        if len(y) <= win:
            starts = [0]
        else:
            starts = list(range(0, max(len(y) - win + 1, 1), hop)) or [0]

        existing_starts = (
            pd.to_numeric(seg_rows["start_sec"], errors="coerce")
            .dropna()
            .astype(float)
            .tolist()
            if not seg_rows.empty
            else []
        )

        parent_candidates = []
        for start_sample in starts:
            scanned_windows += 1
            start_sec = float(start_sample / sr)

            if any(
                abs(start_sec - existing) <= args.existing_start_tolerance_sec
                for existing in existing_starts
            ):
                represented_windows += 1
                continue

            clip = pad_or_trim(y[start_sample:start_sample + win], win)
            raw_rms = rms_dbfs(clip)
            if raw_rms > float(args.candidate_max_dbfs):
                continue

            low_energy_missing_windows += 1
            raw_peak = peak_dbfs(clip)
            zcr = zero_crossing_rate(clip)
            flatness = spectral_flatness(clip)
            energy_band, priority = routing_label(
                raw_rms, int(parent["v09_silence_present"])
            )

            candidate_key = f"{clip_id}|{start_sec:.3f}|{args.segment_sec:.3f}"
            digest = hashlib.md5(candidate_key.encode("utf-8")).hexdigest()[:12]
            review_name = (
                f"{safe_token(Path(file_name).stem)}"
                f"__start_{start_sec:08.3f}"
                f"__{digest}.wav"
            )

            row = {
                "candidate_id": f"low_energy_{digest}",
                "clip_id": clip_id,
                "file_name": file_name,
                "split": str(parent["v09_split"]).lower(),
                "audio_path": str(audio_path),
                "audio_resolution": resolution,
                "start_sec": round(start_sec, 6),
                "end_sec": round(start_sec + args.segment_sec, 6),
                "segment_sec": float(args.segment_sec),
                "hop_sec": float(args.hop_sec),
                "raw_rms_dbfs": round(raw_rms, 6),
                "raw_peak_dbfs": round(raw_peak, 6),
                "zero_crossing_rate": round(zcr, 8),
                "spectral_flatness": round(flatness, 8),
                "energy_band": energy_band,
                "review_priority": priority,
                "parent_silence_present": int(parent["v09_silence_present"]),
                "review_audio_file": review_name,
                "suggested_action": (
                    "manual_check_silence"
                    if int(parent["v09_silence_present"]) == 1
                    else "manual_check_pause_or_low_level_audio"
                ),
                "review_silence_present": "",
                "review_keep_segment": "",
                "review_status": "pending",
                "review_notes": "",
            }
            for label in LABELS:
                parent_col = f"v09_{label}"
                row[f"parent_{label}"] = (
                    int(parent[parent_col])
                    if parent_col in parent.index and not pd.isna(parent[parent_col])
                    else ""
                )

            parent_candidates.append((row, clip, sr))

        priority_order = {"high": 0, "medium": 1, "low": 2}
        parent_candidates.sort(
            key=lambda item: (
                priority_order.get(item[0]["review_priority"], 9),
                item[0]["raw_rms_dbfs"],
                item[0]["start_sec"],
            )
        )
        if args.max_candidates_per_parent > 0:
            parent_candidates = parent_candidates[:args.max_candidates_per_parent]

        for item in parent_candidates:
            candidate_rows.append(item[0])
            candidate_audio.append(item)

    candidate_df = pd.DataFrame(candidate_rows)
    missing_audio_df = pd.DataFrame(missing_audio_rows)

    summary = {
        "generated_utc": utc_now(),
        "mode": "apply" if args.apply else "dry_run",
        "original_parent_count": int(len(original_parents)),
        "resolved_parent_audio_count": int(
            len(original_parents) - len(missing_audio_df)
        ),
        "missing_or_ambiguous_parent_audio_count": int(len(missing_audio_df)),
        "scanned_candidate_grid_windows": int(scanned_windows),
        "already_represented_windows": int(represented_windows),
        "missing_windows_at_or_below_candidate_max_dbfs_before_cap": int(
            low_energy_missing_windows
        ),
        "review_candidates_after_per_parent_cap": int(len(candidate_df)),
        "candidate_max_dbfs": float(args.candidate_max_dbfs),
        "segment_sec": float(args.segment_sec),
        "hop_sec": float(args.hop_sec),
        "sample_rate": int(args.sample_rate),
        "max_candidates_per_parent": int(args.max_candidates_per_parent),
        "source_data_modified": False,
        "current_v09_manifest_modified": False,
    }

    print("\n=== v0.9 low-energy silence-recovery audit ===")
    print(f"Original parents scanned:          {len(original_parents):,}")
    print(f"Parent audio resolved:            {len(original_parents)-len(missing_audio_df):,}")
    print(f"Parent audio missing/ambiguous:    {len(missing_audio_df):,}")
    print(f"1-second grid windows scanned:     {scanned_windows:,}")
    print(f"Already represented windows:       {represented_windows:,}")
    print(f"Low-energy missing windows:       {low_energy_missing_windows:,} before parent cap")
    print(f"Review candidates retained:        {len(candidate_df):,}")
    if len(candidate_df):
        print(f"Candidates by priority:          {candidate_df['review_priority'].value_counts().to_dict()}")
        print(f"Candidates by split:             {candidate_df['split'].value_counts().to_dict()}")
        print(f"From silence-positive parents:   {int((candidate_df['parent_silence_present']==1).sum()):,}")
    print(f"Output root:                       {out_root}")

    if not args.apply:
        print("\n[DRY RUN COMPLETE] No files were created or modified.")
        print("Run with --apply to write the review queue.")
        return 0

    review_csv = out_root / "low_energy_silence_review_queue_v09.csv"
    missing_csv = out_root / "missing_or_ambiguous_parent_audio_v09.csv"
    summary_json = out_root / "low_energy_silence_audit_summary_v09.json"
    audio_root = out_root / "audio"

    outputs = [review_csv, missing_csv, summary_json]
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Audit outputs already exist. Use --overwrite to rebuild:\n"
                + "\n".join(existing)
            )

    out_root.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(review_csv, index=False, encoding="utf-8")
    missing_audio_df.to_csv(missing_csv, index=False, encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.export_audio:
        audio_root.mkdir(parents=True, exist_ok=True)
        for row, clip, sr in candidate_audio:
            sf.write(audio_root / row["review_audio_file"], clip, sr)

    print("\n[COMPLETE] Audit outputs written.")
    print(f"Review queue:                      {review_csv}")
    print(f"Missing-audio report:              {missing_csv}")
    print(f"Summary:                           {summary_json}")
    if args.export_audio:
        print(f"Review WAVs:                       {audio_root}")
    print("No existing dataset or manifest was modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
