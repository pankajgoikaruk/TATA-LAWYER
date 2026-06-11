# scripts/extract_features.py

"""
Feature extraction for ASHADIP audio segments.

This version is backward-compatible with the old metadata-only pipeline and also
supports the new physical segment WAV pipeline.

Priority when reading audio:
1. If segments.csv contains segment_wav_relpath, read cache/<segment_wav_relpath>.
   This means features are extracted from the exported fixed-length 1s WAV clips.
2. Otherwise, fall back to the old behavior: read cache/clean/<wav_relpath> and
   slice using start/duration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from data.transforms_audio import cmvn_feat, to_logmel


def _pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros if shorter than required length; trim if longer."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] >= length:
        return x[:length]
    return np.pad(x, (0, length - x.shape[0]), mode="constant").astype(np.float32)


def _to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def _safe_stem_from_relpath(rel: str, start_i: int, dur_i: int) -> str:
    p = Path(str(rel).replace("\\", "/"))
    return str(p.with_suffix("")) + f"_si{start_i:09d}_di{dur_i:09d}.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--win_ms", type=int, default=25)
    ap.add_argument("--hop_ms", type=int, default=10)
    ap.add_argument("--cmvn", action="store_true")
    ap.add_argument(
        "--pad_short",
        action="store_true",
        help="If a clip is shorter than expected, pad zeros. Recommended for exported segment WAVs.",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=500,
        help="Print progress every N segments. 0 disables progress printing.",
    )
    args = ap.parse_args()

    cache = Path(args.cache)
    seg_path = cache / "segments.csv"
    if not seg_path.exists():
        raise SystemExit(f"segments.csv not found: {seg_path}")

    seg = pd.read_csv(seg_path)
    if "wav_relpath" not in seg.columns:
        raise SystemExit("segments.csv missing required column: wav_relpath")

    use_exported_segments = (
        "segment_wav_relpath" in seg.columns
        and seg["segment_wav_relpath"].notna().any()
        and seg["segment_wav_relpath"].astype(str).str.len().gt(0).any()
    )

    feat_root = cache / "features"
    feat_root.mkdir(parents=True, exist_ok=True)

    seg_out = seg.copy()
    seg_out["wav_relpath"] = seg_out["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    if "segment_wav_relpath" in seg_out.columns:
        seg_out["segment_wav_relpath"] = (
            seg_out["segment_wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
        )

    feats = []
    current_audio_key = None
    current_y = None
    current_sr = None

    total = len(seg_out)
    for i, row in enumerate(seg_out.itertuples(index=False), start=1):
        if args.progress_every and (i % args.progress_every == 0):
            print(f"[extract_features] processed {i}/{total} segments...")

        if use_exported_segments:
            audio_rel = str(getattr(row, "segment_wav_relpath"))
            audio_path = cache / audio_rel
            start_s = 0.0
            dur_s = float(getattr(row, "duration"))
        else:
            audio_rel = str(getattr(row, "wav_relpath"))
            audio_path = cache / "clean" / audio_rel
            start_s = float(getattr(row, "start"))
            dur_s = float(getattr(row, "duration"))

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for feature extraction: {audio_path}")

        audio_key = str(audio_path)
        if audio_key != current_audio_key:
            y, sr = sf.read(audio_path, dtype="float32")
            current_y = _to_mono(y)
            current_sr = int(sr)
            current_audio_key = audio_key

        start_i = int(round(start_s * current_sr))
        dur_i = int(round(dur_s * current_sr))
        clip = current_y[start_i:start_i + dur_i]

        if clip.shape[0] != dur_i:
            if args.pad_short:
                clip = _pad_to_length(clip, dur_i)
            else:
                feats.append("")
                continue

        S = to_logmel(clip, current_sr, args.n_mels, args.n_fft, args.win_ms, args.hop_ms)
        if args.cmvn:
            S = cmvn_feat(S)

        # For exported segment WAVs, mirror their path under features/.
        # For old metadata-only rows, use the old unique start/duration naming scheme.
        if use_exported_segments:
            out_rel = str(Path(audio_rel).with_suffix(".npy")).replace("\\", "/")
        else:
            out_rel = _safe_stem_from_relpath(audio_rel, start_i, dur_i).replace("\\", "/")

        out_path = feat_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, S)
        feats.append(out_rel)

    seg_out["feat_relpath"] = feats

    nonempty = seg_out["feat_relpath"].astype(str)
    nonempty = nonempty[nonempty.str.len() > 0]
    if nonempty.duplicated().any():
        dups = nonempty[nonempty.duplicated()].head(5).tolist()
        raise SystemExit(f"feat_relpath has duplicates (examples): {dups}")

    seg_out.to_csv(seg_path, index=False)
    print("Saved features to", feat_root)
    if use_exported_segments:
        print("Feature source: exported physical segment WAVs via segment_wav_relpath.")
    else:
        print("Feature source: parent clean WAVs via wav_relpath + start/duration.")
    print("Updated segments.csv with feat_relpath.")


if __name__ == "__main__":
    main()
