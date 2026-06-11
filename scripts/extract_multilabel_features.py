# scripts/extract_multilabel_features.py
#
# Extract log-mel features for the multi-label audio manifest.
#
# Input:
#   multilabel_data/metadata/multilabel_train_manifest.csv
#
# Output:
#   multilabel_cache/features/{split}/{sample_id}.npy
#   multilabel_cache/metadata/multilabel_features_manifest.csv
#
# Usage:
#   python scripts\extract_multilabel_features.py `
#     --manifest "multilabel_data\metadata\multilabel_train_manifest.csv" `
#     --labels_json "multilabel_data\metadata\labels.json" `
#     --out_cache "multilabel_cache" `
#     --sample_rate 16000 `
#     --clip_sec 1.0 `
#     --n_mels 64 `
#     --n_fft 1024 `
#     --win_ms 25 `
#     --hop_ms 10 `
#     --cmvn

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Make project root importable when running:
# python scripts\extract_multilabel_features.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

from data.transforms_audio import cmvn_feat, to_logmel


def safe_name(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return y

    if y.ndim == 2:
        # soundfile normally returns [samples, channels]
        return y.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unsupported audio shape: {y.shape}")


def read_audio_any(path: Path) -> tuple[np.ndarray, int]:
    """
    Read audio with soundfile first.
    If soundfile fails, fall back to librosa.
    """
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        y = to_mono(y)
        return y.astype(np.float32), int(sr)

    except Exception as sf_error:
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=None, mono=True)
            return y.astype(np.float32), int(sr)

        except Exception as librosa_error:
            raise RuntimeError(
                "Could not read audio file with soundfile or librosa.\n"
                f"File: {path}\n\n"
                f"soundfile error:\n{sf_error}\n\n"
                f"librosa error:\n{librosa_error}\n\n"
                "Possible fixes:\n"
                "1. Convert this file to WAV/FLAC, or\n"
                "2. Exclude this format from the manifest, or\n"
                "3. Install FFmpeg for compressed audio decoding."
            )


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        y2 = resample_poly(y, up, down)
        return y2.astype(np.float32)

    except Exception:
        try:
            import librosa

            y2 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            return y2.astype(np.float32)

        except Exception as e:
            raise RuntimeError(
                f"Could not resample from {sr} Hz to {target_sr} Hz. "
                f"Install scipy or librosa. Original error: {e}"
            )


def crop_or_pad_start(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Deterministic feature extraction:
      - if longer, take the first target_len samples
      - if shorter, zero-pad
    """
    y = np.asarray(y, dtype=np.float32)

    if len(y) >= target_len:
        return y[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[:len(y)] = y
    return out


def load_fixed_audio(path: Path, target_sr: int, target_len: int) -> np.ndarray:
    y, sr = read_audio_any(path)

    if len(y) == 0:
        raise RuntimeError(f"Empty audio file: {path}")

    y = resample_audio(y, sr, target_sr)
    y = crop_or_pad_start(y, target_len)

    # Remove DC offset.
    y = y - float(np.mean(y))

    return y.astype(np.float32)


def load_labels(labels_json: Path) -> list[str]:
    with labels_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(f"Invalid labels.json. Missing labels list: {labels_json}")

    return [str(x) for x in labels]


def validate_manifest(df: pd.DataFrame, labels: list[str]) -> None:
    required = {
        "sample_id",
        "abs_path",
        "split",
        "labels",
        "is_clean_seed",
        "is_synthetic",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")

    missing_label_cols = [lab for lab in labels if lab not in df.columns]
    if missing_label_cols:
        raise RuntimeError(
            "Manifest missing multi-hot label columns:\n"
            f"{missing_label_cols}"
        )


def make_feature_relpath(row: pd.Series, used: set[str]) -> str:
    split = safe_name(row["split"])
    sample_id = safe_name(row["sample_id"])

    if not sample_id:
        sample_id = safe_name(Path(str(row["abs_path"])).stem)

    rel = f"{split}/{sample_id}.npy"

    # Avoid rare duplicate sample_id collisions.
    if rel not in used:
        used.add(rel)
        return rel

    i = 2
    while True:
        rel2 = f"{split}/{sample_id}_{i}.npy"
        if rel2 not in used:
            used.add(rel2)
            return rel2
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Extract log-mel features for multi-label audio manifest."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to multilabel_train_manifest.csv",
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        help="Path to labels.json",
    )
    parser.add_argument(
        "--out_cache",
        default="multilabel_cache",
        help="Output cache directory.",
    )

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_sec", type=float, default=1.0)

    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_ms", type=int, default=25)
    parser.add_argument("--hop_ms", type=int, default=10)
    parser.add_argument("--cmvn", action="store_true")

    parser.add_argument(
        "--progress_every",
        type=int,
        default=200,
        help="Print progress every N files. Use 0 to disable.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    labels_json = Path(args.labels_json).resolve()
    out_cache = Path(args.out_cache).resolve()

    feature_root = out_cache / "features"
    metadata_root = out_cache / "metadata"
    out_manifest = metadata_root / "multilabel_features_manifest.csv"

    feature_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_json)

    df = pd.read_csv(manifest_path)
    validate_manifest(df, labels)

    target_len = int(round(args.sample_rate * args.clip_sec))

    print("\nExtracting multi-label log-mel features")
    print("-" * 90)
    print(f"Input manifest:  {manifest_path}")
    print(f"Labels JSON:     {labels_json}")
    print(f"Output cache:    {out_cache}")
    print(f"Feature root:    {feature_root}")
    print(f"Output manifest: {out_manifest}")
    print(f"Rows:            {len(df)}")
    print(f"Labels ({len(labels)}): {labels}")
    print(f"Sample rate:     {args.sample_rate}")
    print(f"Clip seconds:    {args.clip_sec}")
    print(f"n_mels:          {args.n_mels}")
    print(f"n_fft:           {args.n_fft}")
    print(f"win_ms:          {args.win_ms}")
    print(f"hop_ms:          {args.hop_ms}")
    print(f"CMVN:            {args.cmvn}")
    print("-" * 90)

    out_rows = []
    used_feat_relpaths: set[str] = set()

    total = len(df)

    for idx, row in df.iterrows():
        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"[extract_multilabel_features] processed {idx + 1}/{total}")

        audio_path = Path(str(row["abs_path"]))

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y = load_fixed_audio(
            path=audio_path,
            target_sr=args.sample_rate,
            target_len=target_len,
        )

        feat = to_logmel(
            y,
            args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            win_ms=args.win_ms,
            hop_ms=args.hop_ms,
        )

        if args.cmvn:
            feat = cmvn_feat(feat)

        feat_relpath = make_feature_relpath(row, used_feat_relpaths)
        feat_path = feature_root / feat_relpath
        feat_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(feat_path, feat.astype(np.float32))

        new_row = row.to_dict()
        new_row["feat_relpath"] = feat_relpath
        new_row["feature_path"] = str(feat_path)
        new_row["feature_shape"] = "x".join(str(x) for x in feat.shape)
        new_row["feature_sample_rate"] = args.sample_rate
        new_row["feature_clip_sec"] = args.clip_sec
        new_row["feature_n_mels"] = args.n_mels
        new_row["feature_n_fft"] = args.n_fft
        new_row["feature_win_ms"] = args.win_ms
        new_row["feature_hop_ms"] = args.hop_ms
        new_row["feature_cmvn"] = int(bool(args.cmvn))

        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)

    # Stable order.
    split_order = {"train": 0, "val": 1, "test": 2, "unsplit": 3}
    out_df["_split_order"] = out_df["split"].map(split_order).fillna(99)
    out_df = out_df.sort_values(
        by=["_split_order", "is_synthetic", "primary_label", "sample_id"],
        ascending=[True, True, True, True],
    ).drop(columns=["_split_order"])

    out_df.to_csv(out_manifest, index=False)

    print("\nFeature extraction completed.")
    print(f"Features saved to: {feature_root}")
    print(f"Updated manifest:  {out_manifest}")
    print(f"Rows:              {len(out_df)}")

    print("\nSplit counts:")
    print(out_df["split"].value_counts().to_string())

    print("\nClean/synthetic counts:")
    print(out_df["is_synthetic"].value_counts().rename(index={0: "clean", 1: "synthetic"}).to_string())

    print("\nLabel-positive counts:")
    for lab in labels:
        print(f"  {lab}: {int(out_df[lab].sum())}")


if __name__ == "__main__":
    main()