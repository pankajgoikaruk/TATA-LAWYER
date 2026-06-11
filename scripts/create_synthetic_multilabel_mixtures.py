# scripts/create_synthetic_multilabel_mixtures.py
#
# Create synthetic multi-label audio mixtures from a clean seed manifest.
#
# Input:
#   multilabel_data/metadata/clean_seed_manifest.csv
#   multilabel_data/metadata/labels.json
#
# Output:
#   multilabel_data/synthetic_mixed/audio/{train,val,test}/*.wav
#   multilabel_data/metadata/synthetic_mixed_manifest.csv
#   optionally: multilabel_data/metadata/multilabel_train_manifest.csv
#
# Example:
#   rain_0001.wav + thunderstorm_0003.wav
#   -> synthetic_train_000001.wav
#   -> labels: rain=1, thunderstorm=1
#
# Usage:
#   python scripts\create_synthetic_multilabel_mixtures.py `
#     --seed_manifest "multilabel_data\metadata\clean_seed_manifest.csv" `
#     --labels_json "multilabel_data\metadata\labels.json" `
#     --out_audio_root "multilabel_data\synthetic_mixed\audio" `
#     --out_manifest "multilabel_data\metadata\synthetic_mixed_manifest.csv" `
#     --combined_out "multilabel_data\metadata\multilabel_train_manifest.csv" `
#     --num_train 1000 `
#     --num_val 200 `
#     --num_test 200 `
#     --mix_size_min 2 `
#     --mix_size_max 2 `
#     --sample_rate 16000 `
#     --clip_sec 1.0 `
#     --seed 42

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return y

    # soundfile usually returns shape [samples, channels]
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unsupported audio shape: {y.shape}")


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target_sr.

    First tries scipy. If scipy is not available, tries librosa.
    """
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


def crop_or_pad(
    y: np.ndarray,
    target_len: int,
    rng: random.Random,
    mode: str = "random_crop",
    short_mode: str = "pad",
) -> np.ndarray:
    """
    Convert audio to exactly target_len samples.

    Long audio:
      random_crop: choose random crop
      start_crop: take beginning

    Short audio:
      pad: zero pad
      loop: repeat until length is enough
    """
    y = np.asarray(y, dtype=np.float32)

    if len(y) == target_len:
        return y

    if len(y) > target_len:
        if mode == "random_crop":
            start = rng.randint(0, len(y) - target_len)
            return y[start:start + target_len].astype(np.float32)

        if mode == "start_crop":
            return y[:target_len].astype(np.float32)

        raise ValueError(f"Unknown crop mode: {mode}")

    # shorter than target_len
    if short_mode == "pad":
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(y)] = y
        return out

    if short_mode == "loop":
        if len(y) == 0:
            return np.zeros(target_len, dtype=np.float32)
        reps = int(np.ceil(target_len / len(y)))
        return np.tile(y, reps)[:target_len].astype(np.float32)

    raise ValueError(f"Unknown short_mode: {short_mode}")


def read_audio_any(path: Path) -> tuple[np.ndarray, int]:
    """
    Read audio with soundfile first.
    If soundfile fails, fall back to librosa.

    Why:
      - soundfile is good for WAV/FLAC.
      - librosa/audioread can often handle MP3/M4A depending on system support.
    """
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        y = to_mono(y)
        return y.astype(np.float32), int(sr)

    except Exception as sf_error:
        try:
            import librosa

            # sr=None keeps original sample rate.
            # mono=True converts stereo/multi-channel to mono.
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
                "2. Exclude .m4a from the clean seed manifest, or\n"
                "3. Install FFmpeg so librosa/audioread can decode M4A files."
            )


def load_audio_fixed(
    path: Path,
    target_sr: int,
    target_len: int,
    rng: random.Random,
    crop_mode: str,
    short_mode: str,
) -> np.ndarray:
    y, sr = read_audio_any(path)

    if len(y) == 0:
        raise RuntimeError(f"Empty audio file: {path}")

    y = resample_audio(y, int(sr), target_sr)
    y = crop_or_pad(
        y=y,
        target_len=target_len,
        rng=rng,
        mode=crop_mode,
        short_mode=short_mode,
    )

    # Remove DC offset.
    y = y - float(np.mean(y))

    return y.astype(np.float32)


def normalize_component(y: np.ndarray, target_rms: float = 0.10) -> np.ndarray:
    """
    RMS-normalise each component before applying random gain.

    This prevents one source from dominating only because it was recorded louder.
    """
    r = rms(y)
    if r < 1e-6:
        return y.astype(np.float32)
    return (y / r * target_rms).astype(np.float32)


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0

    if max_abs < 1e-9:
        return y

    if max_abs > peak:
        y = y / max_abs * peak

    return y.astype(np.float32)


def parse_labels_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(f"labels.json missing valid 'labels' list: {path}")

    labels = [str(x) for x in labels]
    return labels


def get_audio_path(row: pd.Series) -> Path:
    """
    Prefer abs_path from the seed manifest.
    """
    if "abs_path" in row and pd.notna(row["abs_path"]) and str(row["abs_path"]).strip():
        return Path(str(row["abs_path"]))

    raise RuntimeError(
        "Seed manifest row does not contain abs_path. "
        "Please rebuild clean_seed_manifest.csv with build_multilabel_seed_manifest.py."
    )


def build_candidates_by_split_label(
    seed_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, dict[str, list[Path]]]:
    """
    candidates[split][label] = list of audio paths
    """
    candidates: dict[str, dict[str, list[Path]]] = {}

    required_cols = {"split", "primary_label", "abs_path"}
    missing = required_cols - set(seed_df.columns)
    if missing:
        raise RuntimeError(f"Seed manifest missing required columns: {sorted(missing)}")

    for split in sorted(seed_df["split"].astype(str).unique()):
        candidates[split] = {label: [] for label in labels}

    for _, row in seed_df.iterrows():
        split = str(row["split"])
        label = str(row["primary_label"])

        if label not in labels:
            continue

        path = get_audio_path(row)
        if not path.exists():
            raise FileNotFoundError(f"Audio path not found from manifest: {path}")

        candidates.setdefault(split, {lab: [] for lab in labels})
        candidates[split].setdefault(label, [])
        candidates[split][label].append(path)

    return candidates


def choose_labels_for_mix(
    labels: list[str],
    candidates_for_split: dict[str, list[Path]],
    mix_size_min: int,
    mix_size_max: int,
    rng: random.Random,
) -> list[str]:
    available_labels = [
        label for label in labels
        if label in candidates_for_split and len(candidates_for_split[label]) > 0
    ]

    if len(available_labels) < mix_size_min:
        raise RuntimeError(
            f"Not enough labels with available clean seed files. "
            f"Available={available_labels}, required at least={mix_size_min}"
        )

    max_size = min(mix_size_max, len(available_labels))
    min_size = min(mix_size_min, max_size)

    k = rng.randint(min_size, max_size)
    return sorted(rng.sample(available_labels, k=k))


def make_one_mixture(
    chosen_labels: list[str],
    candidates_for_split: dict[str, list[Path]],
    target_sr: int,
    target_len: int,
    rng: random.Random,
    crop_mode: str,
    short_mode: str,
    gain_min_db: float,
    gain_max_db: float,
    component_target_rms: float,
    peak: float,
) -> tuple[np.ndarray, list[Path], list[float]]:
    mix = np.zeros(target_len, dtype=np.float32)
    source_paths: list[Path] = []
    gains_db: list[float] = []

    for label in chosen_labels:
        source_path = rng.choice(candidates_for_split[label])
        source_paths.append(source_path)

        y = load_audio_fixed(
            path=source_path,
            target_sr=target_sr,
            target_len=target_len,
            rng=rng,
            crop_mode=crop_mode,
            short_mode=short_mode,
        )

        y = normalize_component(y, target_rms=component_target_rms)

        gain_db = rng.uniform(gain_min_db, gain_max_db)
        gain = db_to_amp(gain_db)
        y = y * gain

        mix += y.astype(np.float32)
        gains_db.append(float(gain_db))

    mix = peak_normalize(mix, peak=peak)
    return mix.astype(np.float32), source_paths, gains_db


def parse_num_by_split(args) -> dict[str, int]:
    return {
        "train": int(args.num_train),
        "val": int(args.num_val),
        "test": int(args.num_test),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic multi-label mixtures from clean seed audio."
    )

    parser.add_argument(
        "--seed_manifest",
        required=True,
        help="Path to clean_seed_manifest.csv",
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        help="Path to labels.json",
    )
    parser.add_argument(
        "--out_audio_root",
        default="multilabel_data/synthetic_mixed/audio",
        help="Output directory for synthetic audio WAV files.",
    )
    parser.add_argument(
        "--out_manifest",
        default="multilabel_data/metadata/synthetic_mixed_manifest.csv",
        help="Output CSV manifest for synthetic mixtures.",
    )
    parser.add_argument(
        "--combined_out",
        default=None,
        help=(
            "Optional combined manifest path. "
            "If provided, seed_manifest + synthetic_mixed_manifest are merged here."
        ),
    )

    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_val", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=200)

    parser.add_argument(
        "--mix_size_min",
        type=int,
        default=2,
        help="Minimum number of labels/components per synthetic mixture.",
    )
    parser.add_argument(
        "--mix_size_max",
        type=int,
        default=2,
        help="Maximum number of labels/components per synthetic mixture.",
    )

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_sec", type=float, default=1.0)

    parser.add_argument(
        "--gain_min_db",
        type=float,
        default=-6.0,
        help="Minimum random component gain in dB.",
    )
    parser.add_argument(
        "--gain_max_db",
        type=float,
        default=0.0,
        help="Maximum random component gain in dB.",
    )
    parser.add_argument(
        "--component_target_rms",
        type=float,
        default=0.10,
        help="Target RMS before random gain.",
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=0.95,
        help="Peak normalisation limit.",
    )

    parser.add_argument(
        "--crop_mode",
        choices=["random_crop", "start_crop"],
        default="random_crop",
        help="How to crop long source clips.",
    )
    parser.add_argument(
        "--short_mode",
        choices=["pad", "loop"],
        default="pad",
        help="How to handle source clips shorter than clip_sec.",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mix_size_min < 1:
        raise ValueError("--mix_size_min must be >= 1")

    if args.mix_size_max < args.mix_size_min:
        raise ValueError("--mix_size_max must be >= --mix_size_min")

    seed_manifest = Path(args.seed_manifest).resolve()
    labels_json = Path(args.labels_json).resolve()
    out_audio_root = Path(args.out_audio_root).resolve()
    out_manifest = Path(args.out_manifest).resolve()

    out_audio_root.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    labels = parse_labels_json(labels_json)
    seed_df = pd.read_csv(seed_manifest)

    candidates = build_candidates_by_split_label(seed_df, labels)
    num_by_split = parse_num_by_split(args)

    target_len = int(round(args.sample_rate * args.clip_sec))

    print("\nCreating synthetic multi-label mixtures")
    print("-" * 90)
    print(f"Seed manifest:  {seed_manifest}")
    print(f"Labels JSON:    {labels_json}")
    print(f"Output audio:   {out_audio_root}")
    print(f"Output manifest:{out_manifest}")
    print(f"Labels ({len(labels)}): {labels}")
    print(f"Sample rate:    {args.sample_rate}")
    print(f"Clip seconds:   {args.clip_sec}")
    print(f"Mix size:       {args.mix_size_min} to {args.mix_size_max}")
    print(f"Gain range:     {args.gain_min_db} dB to {args.gain_max_db} dB")
    print(f"Seed:           {args.seed}")
    print("-" * 90)

    print("\nClean seed availability:")
    for split in ["train", "val", "test", "unsplit"]:
        if split not in candidates:
            continue
        counts = {lab: len(candidates[split].get(lab, [])) for lab in labels}
        total = sum(counts.values())
        print(f"  {split}: total={total}, per_label={counts}")

    rows = []
    global_index = 1

    for split, n_mix in num_by_split.items():
        if n_mix <= 0:
            continue

        if split not in candidates:
            raise RuntimeError(
                f"Split '{split}' was requested but not found in seed manifest."
            )

        split_out_dir = out_audio_root / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        candidates_for_split = candidates[split]

        print(f"\nGenerating {n_mix} mixtures for split: {split}")

        for i in range(1, n_mix + 1):
            chosen_labels = choose_labels_for_mix(
                labels=labels,
                candidates_for_split=candidates_for_split,
                mix_size_min=args.mix_size_min,
                mix_size_max=args.mix_size_max,
                rng=rng,
            )

            mix, source_paths, gains_db = make_one_mixture(
                chosen_labels=chosen_labels,
                candidates_for_split=candidates_for_split,
                target_sr=args.sample_rate,
                target_len=target_len,
                rng=rng,
                crop_mode=args.crop_mode,
                short_mode=args.short_mode,
                gain_min_db=args.gain_min_db,
                gain_max_db=args.gain_max_db,
                component_target_rms=args.component_target_rms,
                peak=args.peak,
            )

            sample_id = f"synthetic_{split}_{i:06d}"
            out_name = f"{sample_id}.wav"
            out_path = split_out_dir / out_name

            sf.write(out_path, mix, args.sample_rate, subtype="PCM_16")

            relpath = out_path.relative_to(out_audio_root).as_posix()

            row = {
                "sample_id": sample_id,
                "filepath": relpath,
                "abs_path": str(out_path),
                "source_file": out_name,
                "source_stem": sample_id,
                "source_ext": ".wav",
                "class_dir": "synthetic_mixed",
                "primary_label": "mixed",
                "labels": ";".join(chosen_labels),
                "num_labels": len(chosen_labels),
                "is_clean_seed": 0,
                "is_synthetic": 1,
                "split": split,
                "source_paths": ";".join(str(p) for p in source_paths),
                "source_names": ";".join(p.name for p in source_paths),
                "source_labels": ";".join(chosen_labels),
                "component_gains_db": ";".join(f"{x:.3f}" for x in gains_db),
                "sample_rate": args.sample_rate,
                "duration_sec": args.clip_sec,
                "global_index": global_index,
            }

            for lab in labels:
                row[lab] = 1 if lab in chosen_labels else 0

            rows.append(row)
            global_index += 1

            if i % 100 == 0 or i == n_mix:
                print(f"  {split}: {i}/{n_mix} done")

    if not rows:
        raise RuntimeError("No synthetic mixtures were generated.")

    fieldnames = [
        "sample_id",
        "filepath",
        "abs_path",
        "source_file",
        "source_stem",
        "source_ext",
        "class_dir",
        "primary_label",
        "labels",
        "num_labels",
        "is_clean_seed",
        "is_synthetic",
        "split",
        "source_paths",
        "source_names",
        "source_labels",
        "component_gains_db",
        "sample_rate",
        "duration_sec",
        "global_index",
    ] + labels

    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    synthetic_df = pd.DataFrame(rows)

    print("\nSynthetic manifest created:")
    print(f"  {out_manifest}")
    print(f"  Rows: {len(synthetic_df)}")

    print("\nSynthetic split counts:")
    print(synthetic_df["split"].value_counts().to_string())

    print("\nSynthetic label-positive counts:")
    for lab in labels:
        print(f"  {lab}: {int(synthetic_df[lab].sum())}")

    if args.combined_out:
        combined_out = Path(args.combined_out).resolve()
        combined_out.parent.mkdir(parents=True, exist_ok=True)

        # Align columns from seed and synthetic manifests.
        seed_df_copy = seed_df.copy()
        synthetic_df_copy = synthetic_df.copy()

        all_columns = list(dict.fromkeys(list(seed_df_copy.columns) + list(synthetic_df_copy.columns)))

        for col in all_columns:
            if col not in seed_df_copy.columns:
                seed_df_copy[col] = ""
            if col not in synthetic_df_copy.columns:
                synthetic_df_copy[col] = ""

        combined = pd.concat(
            [
                seed_df_copy[all_columns],
                synthetic_df_copy[all_columns],
            ],
            ignore_index=True,
        )

        # Stable order.
        split_order = {"train": 0, "val": 1, "test": 2, "unsplit": 3}
        combined["_split_order"] = combined["split"].map(split_order).fillna(99)
        combined = combined.sort_values(
            by=["_split_order", "is_synthetic", "primary_label", "sample_id"],
            ascending=[True, True, True, True],
        ).drop(columns=["_split_order"])

        combined.to_csv(combined_out, index=False)

        print("\nCombined manifest created:")
        print(f"  {combined_out}")
        print(f"  Rows: {len(combined)}")

        print("\nCombined split counts:")
        print(combined["split"].value_counts().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()