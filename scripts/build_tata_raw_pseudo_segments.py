# scripts/build_tata_raw_pseudo_segments.py

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


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
    except Exception as sf_error:
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None, mono=True)
            return y.astype(np.float32), int(sr)
        except Exception as librosa_error:
            raise RuntimeError(
                f"Could not read audio: {path}\n"
                f"soundfile error: {sf_error}\n"
                f"librosa error: {librosa_error}"
            )


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)

    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        return resample_poly(y, up, down).astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build raw pseudo-pool 1-sec segments for TATA v0.6 inference."
    )

    parser.add_argument("--raw_parent_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--include_tail", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)

    args = parser.parse_args()

    raw_parent_manifest = Path(args.raw_parent_manifest)
    out_dir = Path(args.out_dir)
    segment_root = out_dir / "segment_wavs"
    metadata_dir = out_dir / "metadata"

    segment_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_parent_manifest)

    required = {"parent_clip_id", "source_path", "source_file", "source_rel_path"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Raw parent manifest missing columns: {sorted(missing)}")

    segment_len = int(round(args.sample_rate * args.segment_sec))
    hop_len = int(round(args.sample_rate * args.hop_sec))

    out_rows = []
    errors = []

    print("")
    print("Building raw pseudo-pool TATA segments")
    print("-" * 90)
    print(f"Input parent manifest: {raw_parent_manifest}")
    print(f"Rows:                  {len(df)}")
    print(f"Output dir:            {out_dir}")
    print(f"Sample rate:           {args.sample_rate}")
    print(f"Segment seconds:       {args.segment_sec}")
    print(f"Hop seconds:           {args.hop_sec}")
    print("-" * 90)

    for idx, row in df.reset_index(drop=True).iterrows():
        parent_clip_id = str(row["parent_clip_id"])
        audio_path = Path(str(row["source_path"]))

        if not audio_path.exists():
            errors.append({
                "parent_clip_id": parent_clip_id,
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
                "parent_clip_id": parent_clip_id,
                "source_path": str(audio_path),
                "error": str(e),
            })
            continue

        if len(y) == 0:
            errors.append({
                "parent_clip_id": parent_clip_id,
                "source_path": str(audio_path),
                "error": "empty_audio",
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

            sample_id = f"{parent_clip_id}_seg{seg_idx:04d}"
            seg_rel = Path("raw") / f"{sample_id}.wav"
            seg_path = segment_root / seg_rel
            seg_path.parent.mkdir(parents=True, exist_ok=True)

            sf.write(seg_path, segment, args.sample_rate)

            out_row = {
                "sample_id": sample_id,
                "parent_clip_id": parent_clip_id,
                "abs_path": str(seg_path.resolve()),
                "segment_wav_relpath": str(seg_rel),
                "split": "raw",
                "start_sec": round(start / args.sample_rate, 4),
                "end_sec": round(min(end, len(y)) / args.sample_rate, 4),
                "segment_sec": args.segment_sec,
                "hop_sec": args.hop_sec,
                "labels": "",
                "num_active_labels": 0,
                "primary_label": "",
                "source_file": row.get("source_file", audio_path.name),
                "source_path": str(audio_path),
                "source_rel_path": row.get("source_rel_path", ""),
                "source_class_dir": row.get("source_class_dir", ""),
                "is_clean_seed": 0,
                "is_synthetic": 0,
                "labeling_level": "raw_unlabelled_for_tata_inference",
            }

            # Dummy labels only for feature-extraction compatibility.
            for lab in LABELS:
                out_row[lab] = 0

            out_rows.append(out_row)

        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"[raw_segments] processed parents: {idx + 1}/{len(df)} | segments: {len(out_rows)}")

    out_manifest = metadata_dir / "raw_pseudo_pool_segment_manifest.csv"
    out_labels = metadata_dir / "tata_v06_labels.json"
    out_errors = metadata_dir / "raw_pseudo_pool_segment_errors.csv"
    out_summary = metadata_dir / "raw_pseudo_pool_segment_summary.json"

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_manifest, index=False)

    pd.DataFrame(errors).to_csv(out_errors, index=False)

    labels_payload = {
        "task": "tata_v06_raw_pseudo_pool_inference",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": LABELS,
    }
    out_labels.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "input_parent_manifest": str(raw_parent_manifest),
        "output_manifest": str(out_manifest),
        "labels_json": str(out_labels),
        "errors_csv": str(out_errors),
        "parent_rows": int(len(df)),
        "segments_created": int(len(out_df)),
        "errors": int(len(errors)),
        "segments_per_parent_mean": float(out_df.groupby("parent_clip_id").size().mean()) if len(out_df) else 0.0,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Raw pseudo-pool segment build complete")
    print("-" * 90)
    print(f"Segments: {len(out_df)}")
    print(f"Errors:   {len(errors)}")
    print(f"Manifest: {out_manifest}")
    print(f"Errors:   {out_errors}")


if __name__ == "__main__":
    main()