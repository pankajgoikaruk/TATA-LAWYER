# scripts/add_lawyer_silence_signal_features_v08.py
#
# Reusable acoustic silence-feature enrichment for LAWYER.
#
# Purpose
# -------
# LAWYER already uses TATA segment probabilities. This script adds signal-based
# evidence for silence_present by computing acoustic features from the source
# audio for each 1-sec segment:
#
#   - silence_rms_dbfs
#   - silence_peak_dbfs
#   - silence_zcr
#   - silence_speech_activity_ratio
#   - silence_is_acoustic_silent
#
# It does NOT overwrite the input CSV. It writes an enriched output CSV.
#
# Example:
#   python scripts/add_lawyer_silence_signal_features_v08.py ^
#     --input_csv human_talk_workspace\tata_v0.6_raw_pipeline\raw_tata_pseudo_routing\raw_segment_predictions.csv ^
#     --output_csv human_talk_workspace\tata_v0.8_raw_pipeline\raw_tata_pseudo_routing\raw_segment_predictions_with_silence_signal.csv

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def read_audio_any(path: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_mono(y), int(sr)
    except Exception:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if sr == target_sr:
        return y

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sr), int(target_sr))
        return resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def dbfs(value: float, eps: float = 1e-12) -> float:
    value = max(float(value), eps)
    return float(20.0 * np.log10(value))


def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> list[np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return []

    if len(x) <= frame_len:
        return [x]

    frames = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frames.append(x[start:start + frame_len])

    if not frames:
        frames.append(x)

    return frames


def zero_crossing_rate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return 0.0

    signs = np.signbit(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def compute_segment_features(
    segment: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    speech_frame_db_threshold: float,
    rms_silence_threshold_db: float,
    speech_activity_threshold: float,
) -> dict[str, Any]:
    segment = np.asarray(segment, dtype=np.float32)

    if len(segment) == 0:
        return {
            "silence_rms_dbfs": -120.0,
            "silence_peak_dbfs": -120.0,
            "silence_zcr": 0.0,
            "silence_speech_activity_ratio": 0.0,
            "silence_is_acoustic_silent": 1,
        }

    rms = float(np.sqrt(np.mean(np.square(segment)) + 1e-12))
    peak = float(np.max(np.abs(segment)) + 1e-12)

    rms_db = dbfs(rms)
    peak_db = dbfs(peak)
    zcr = zero_crossing_rate(segment)

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frames = frame_signal(segment, frame_len, hop_len)

    frame_rms_db = []
    for frame in frames:
        fr = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        frame_rms_db.append(dbfs(fr))

    if frame_rms_db:
        speech_activity_ratio = float(np.mean(np.asarray(frame_rms_db) > float(speech_frame_db_threshold)))
    else:
        speech_activity_ratio = 0.0

    is_silent = int(
        rms_db <= float(rms_silence_threshold_db)
        and speech_activity_ratio <= float(speech_activity_threshold)
    )

    return {
        "silence_rms_dbfs": float(rms_db),
        "silence_peak_dbfs": float(peak_db),
        "silence_zcr": float(zcr),
        "silence_speech_activity_ratio": float(speech_activity_ratio),
        "silence_is_acoustic_silent": int(is_silent),
    }


def infer_segment_timing(
    row: pd.Series,
    *,
    seg_index_in_parent: int,
    sample_rate: int,
    segment_sec: float,
    hop_sec: float,
    start_col: str,
    end_col: str,
) -> tuple[float, float]:
    if start_col in row.index and not pd.isna(row[start_col]):
        try:
            start_sec = float(row[start_col])
            if end_col in row.index and not pd.isna(row[end_col]):
                end_sec = float(row[end_col])
            else:
                end_sec = start_sec + float(segment_sec)
            return start_sec, end_sec
        except Exception:
            pass

    start_sec = float(seg_index_in_parent) * float(hop_sec)
    end_sec = start_sec + float(segment_sec)
    return start_sec, end_sec


def crop_or_pad(y: np.ndarray, start_sec: float, end_sec: float, sr: int) -> np.ndarray:
    start = max(0, int(round(start_sec * sr)))
    end = max(start, int(round(end_sec * sr)))

    segment = y[start:end].astype(np.float32)
    target_len = max(1, end - start)

    if len(segment) >= target_len:
        return segment[:target_len]

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(segment)] = segment
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add acoustic silence signal features to TATA segment prediction CSV."
    )

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", default=None)

    parser.add_argument("--source_path_col", default="source_path")
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--start_sec_col", default="start_sec")
    parser.add_argument("--end_sec_col", default="end_sec")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)

    parser.add_argument("--frame_ms", type=float, default=25.0)
    parser.add_argument("--frame_hop_ms", type=float, default=10.0)

    parser.add_argument("--speech_frame_db_threshold", type=float, default=-45.0)
    parser.add_argument("--rms_silence_threshold_db", type=float, default=-45.0)
    parser.add_argument("--speech_activity_threshold", type=float, default=0.15)

    parser.add_argument("--progress_every", type=int, default=250)

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if args.source_path_col not in df.columns:
        raise RuntimeError(f"Missing source path column: {args.source_path_col}")

    if args.parent_id_col not in df.columns:
        raise RuntimeError(f"Missing parent id column: {args.parent_id_col}")

    print("")
    print("Adding LAWYER silence signal features")
    print("-" * 90)
    print(f"Input CSV:  {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows:       {len(df)}")
    print("-" * 90)

    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    features = []
    errors = []

    parent_counts: dict[str, int] = {}

    for idx, row in df.iterrows():
        parent_id = str(row[args.parent_id_col])
        seg_index = parent_counts.get(parent_id, 0)
        parent_counts[parent_id] = seg_index + 1

        source_path = Path(str(row[args.source_path_col]))

        try:
            if str(source_path) not in audio_cache:
                y, sr = read_audio_any(source_path)
                y = resample_audio(y, sr, int(args.sample_rate))
                audio_cache[str(source_path)] = (y.astype(np.float32), int(args.sample_rate))

            y, sr = audio_cache[str(source_path)]

            start_sec, end_sec = infer_segment_timing(
                row,
                seg_index_in_parent=seg_index,
                sample_rate=int(args.sample_rate),
                segment_sec=float(args.segment_sec),
                hop_sec=float(args.hop_sec),
                start_col=str(args.start_sec_col),
                end_col=str(args.end_sec_col),
            )

            segment = crop_or_pad(y, start_sec, end_sec, sr)

            feat = compute_segment_features(
                segment,
                sr,
                frame_ms=float(args.frame_ms),
                hop_ms=float(args.frame_hop_ms),
                speech_frame_db_threshold=float(args.speech_frame_db_threshold),
                rms_silence_threshold_db=float(args.rms_silence_threshold_db),
                speech_activity_threshold=float(args.speech_activity_threshold),
            )
            feat["silence_signal_error"] = ""

        except Exception as e:
            feat = {
                "silence_rms_dbfs": np.nan,
                "silence_peak_dbfs": np.nan,
                "silence_zcr": np.nan,
                "silence_speech_activity_ratio": np.nan,
                "silence_is_acoustic_silent": 0,
                "silence_signal_error": str(e),
            }
            errors.append({
                "row_index": int(idx),
                "parent_clip_id": parent_id,
                "source_path": str(source_path),
                "error": str(e),
            })

        features.append(feat)

        if int(args.progress_every) > 0 and ((idx + 1) % int(args.progress_every) == 0 or (idx + 1) == len(df)):
            print(f"[silence-signal] processed {idx + 1}/{len(df)}")

    feat_df = pd.DataFrame(features)
    out_df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary = {
        "generated_at": now_iso(),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(out_df)),
        "unique_audio_files": int(len(audio_cache)),
        "errors": int(len(errors)),
        "acoustic_silent_segments": int(pd.to_numeric(out_df["silence_is_acoustic_silent"], errors="coerce").fillna(0).astype(int).sum()),
        "parameters": {
            "sample_rate": int(args.sample_rate),
            "segment_sec": float(args.segment_sec),
            "hop_sec": float(args.hop_sec),
            "frame_ms": float(args.frame_ms),
            "frame_hop_ms": float(args.frame_hop_ms),
            "speech_frame_db_threshold": float(args.speech_frame_db_threshold),
            "rms_silence_threshold_db": float(args.rms_silence_threshold_db),
            "speech_activity_threshold": float(args.speech_activity_threshold),
        },
        "columns_added": [
            "silence_rms_dbfs",
            "silence_peak_dbfs",
            "silence_zcr",
            "silence_speech_activity_ratio",
            "silence_is_acoustic_silent",
            "silence_signal_error",
        ],
        "error_examples": errors[:20],
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Silence signal enrichment complete")
    print("-" * 90)
    print(f"Rows:                      {len(out_df)}")
    print(f"Unique audio files cached: {len(audio_cache)}")
    print(f"Errors:                    {len(errors)}")
    print(f"Acoustic silent segments:  {summary['acoustic_silent_segments']}")
    print(f"Output CSV:                {output_csv}")
    print(f"Summary:                   {summary_json}")


if __name__ == "__main__":
    main()
