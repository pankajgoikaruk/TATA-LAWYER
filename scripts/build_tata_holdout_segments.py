# scripts/build_tata_holdout_segments.py
#
# Reusable final-holdout segment builder.
#
# This version does NOT hard-code label names. It reads labels from:
#   --labels_json configs/human_talk_10label_schema.json
#
# Expected labels_json formats supported:
#   {"labels": ["label_a", "label_b", ...]}
# or:
#   ["label_a", "label_b", ...]

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


def load_labels(labels_json: Path) -> list[str]:
    if not labels_json.exists():
        raise FileNotFoundError(f"labels_json not found: {labels_json}")

    payload = json.loads(labels_json.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        labels = payload
    elif isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        labels = payload["labels"]
    else:
        raise RuntimeError(
            "labels_json must be either a list of labels or a dict containing key 'labels'."
        )

    labels = [str(x) for x in labels]
    if not labels:
        raise RuntimeError("labels_json contains no labels.")

    if len(labels) != len(set(labels)):
        duplicates = sorted({x for x in labels if labels.count(x) > 1})
        raise RuntimeError(f"labels_json contains duplicate labels: {duplicates}")

    return labels


def to_bin(v: Any) -> int:
    try:
        if pd.isna(v):
            return 0
        return 1 if int(float(v)) == 1 else 0
    except Exception:
        return 0


def active_text(row: pd.Series, labels: list[str]) -> str:
    return "|".join([lab for lab in labels if int(row[lab]) == 1])


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
    if sr == target_sr:
        return y.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, target_sr)
        return resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(
            y.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr,
        ).astype(np.float32)


def normalise_audio(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    if len(y) == 0:
        return y

    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))

    if peak > 1e-8:
        y = y / max(peak, 1.0)

    return y.astype(np.float32)


def crop_or_pad(segment: np.ndarray, target_len: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)
    if len(segment) >= target_len:
        return segment[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(segment)] = segment
    return out


def ensure_required_columns(df: pd.DataFrame, required_cols: set[str]) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labelled fixed-length segment manifest from final holdout parent CSV."
    )

    parser.add_argument("--holdout_csv", required=True)
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--include_tail", action="store_true")

    parser.add_argument("--split_name", default="test")
    parser.add_argument("--task_name", default="final_raw_holdout_ground_truth")
    parser.add_argument("--labeling_level", default="final_raw_holdout_ground_truth")

    parser.add_argument("--audio_path_col", default="source_path")
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--source_file_col", default="source_file")
    parser.add_argument("--source_rel_path_col", default="source_rel_path")
    parser.add_argument("--source_class_col", default="source_class_dir")
    parser.add_argument("--primary_label_col", default="primary_label")

    args = parser.parse_args()

    holdout_csv = Path(args.holdout_csv)
    labels_json = Path(args.labels_json)
    out_dir = Path(args.out_dir)

    labels = load_labels(labels_json)

    wav_root = out_dir / "segment_wavs"
    meta_dir = out_dir / "metadata"

    wav_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(holdout_csv, low_memory=False)

    required = {
        args.parent_id_col,
        args.audio_path_col,
        args.source_file_col,
        *labels,
    }
    ensure_required_columns(df, required)

    for lab in labels:
        df[lab] = df[lab].apply(to_bin)

    df["labels"] = df.apply(lambda row: active_text(row, labels), axis=1)
    df["num_active_labels"] = df[labels].sum(axis=1).astype(int)

    zero = df[df["num_active_labels"] == 0].copy()
    if len(zero):
        zero_path = meta_dir / "zero_active_holdout_rows.csv"
        zero.to_csv(zero_path, index=False)
        raise RuntimeError(
            f"Found {len(zero)} zero-active holdout rows. Fix before evaluation. "
            f"Saved: {zero_path}"
        )

    segment_len = int(round(args.sample_rate * args.segment_sec))
    hop_len = int(round(args.sample_rate * args.hop_sec))

    if segment_len <= 0:
        raise RuntimeError("segment_len must be > 0. Check sample_rate and segment_sec.")
    if hop_len <= 0:
        raise RuntimeError("hop_len must be > 0. Check sample_rate and hop_sec.")

    rows = []
    errors = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parent_id = str(row[args.parent_id_col])
        audio_path = Path(str(row[args.audio_path_col]))

        if not audio_path.exists():
            errors.append({
                "parent_clip_id": parent_id,
                "source_path": str(audio_path),
                "error": "missing_file",
            })
            continue

        try:
            y, sr = read_audio_any(audio_path)
            y = resample_audio(y, sr, args.sample_rate)
            y = normalise_audio(y)
        except Exception as e:
            errors.append({
                "parent_clip_id": parent_id,
                "source_path": str(audio_path),
                "error": str(e),
            })
            continue

        if len(y) <= segment_len:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(y) - segment_len + 1), hop_len))
            final_start = max(0, len(y) - segment_len)
            if args.include_tail and final_start not in starts:
                starts.append(final_start)

        for seg_idx, start in enumerate(starts):
            end = start + segment_len
            segment = crop_or_pad(y[start:end], segment_len)

            sample_id = f"{parent_id}_seg{seg_idx:04d}"
            rel = Path("holdout") / f"{sample_id}.wav"
            out_wav = wav_root / rel
            out_wav.parent.mkdir(parents=True, exist_ok=True)

            sf.write(out_wav, segment, args.sample_rate)

            out_row = {
                "sample_id": sample_id,
                "parent_clip_id": parent_id,
                "abs_path": str(out_wav.resolve()),
                "segment_wav_relpath": str(rel),
                "split": str(args.split_name),
                "start_sec": round(start / args.sample_rate, 4),
                "end_sec": round(min(end, len(y)) / args.sample_rate, 4),
                "segment_sec": float(args.segment_sec),
                "hop_sec": float(args.hop_sec),
                "source_file": row.get(args.source_file_col, ""),
                "source_path": str(audio_path),
                "source_rel_path": row.get(args.source_rel_path_col, ""),
                "source_class_dir": row.get(args.source_class_col, ""),
                "primary_label": row.get(args.primary_label_col, ""),
                "labels": row["labels"],
                "num_active_labels": int(row["num_active_labels"]),
                "labeling_level": str(args.labeling_level),
                "is_clean_seed": 0,
                "is_synthetic": 0,
            }

            for lab in labels:
                out_row[lab] = int(row[lab])

            rows.append(out_row)

    rows_df = pd.DataFrame(rows)

    out_manifest = meta_dir / "final_holdout_segment_manifest.csv"
    out_errors = meta_dir / "final_holdout_segment_errors.csv"
    out_labels = meta_dir / "labels.json"
    out_summary = meta_dir / "final_holdout_segment_summary.json"

    rows_df.to_csv(out_manifest, index=False)
    pd.DataFrame(errors).to_csv(out_errors, index=False)

    label_payload = {
        "task": str(args.task_name),
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": labels,
        "source_labels_json": str(labels_json),
    }
    out_labels.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "task": str(args.task_name),
        "holdout_csv": str(holdout_csv),
        "labels_json": str(labels_json),
        "labels": labels,
        "holdout_parent_rows": int(len(df)),
        "segment_rows": int(len(rows_df)),
        "errors": int(len(errors)),
        "sample_rate": int(args.sample_rate),
        "segment_sec": float(args.segment_sec),
        "hop_sec": float(args.hop_sec),
        "include_tail": bool(args.include_tail),
        "label_counts_parent": {lab: int(df[lab].sum()) for lab in labels},
        "label_counts_segment": {
            lab: int(rows_df[lab].sum()) for lab in labels
        } if len(rows_df) else {},
        "outputs": {
            "segment_manifest": str(out_manifest),
            "errors_csv": str(out_errors),
            "labels_json": str(out_labels),
            "summary_json": str(out_summary),
        },
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Final holdout segment manifest created")
    print("-" * 90)
    print(f"Labels:       {len(labels)}")
    print(f"Parent rows:  {len(df)}")
    print(f"Segment rows: {len(rows_df)}")
    print(f"Errors:       {len(errors)}")
    print(f"Manifest:     {out_manifest}")
    print(f"Errors CSV:   {out_errors}")
    print(f"Labels JSON:  {out_labels}")


if __name__ == "__main__":
    main()
