# scripts/prepare_human_talk_segments.py
#
# Prepare parent-child segment manifests for human-talk speaker experiments.
#
# Expected input:
#   human_talk_dataset/
#     Les_Brown/Les_Brown__0001.wav
#     Simon_Sinek/Simon_Sinek__0001.wav
#
# Output is stage-contained:
#   human_talk_workspace/stages/<stage>/data/
#     metadata/labels.json
#     metadata/human_talk_parent_manifest.csv
#     metadata/multilabel_train_manifest.csv
#     segments/{train,val,test}/{class}/{segment_id}.wav
#
# The output manifest is compatible with scripts/extract_multilabel_features.py.
#
# Important fix:
#   Filename stems are normalized without collapsing repeated underscores.
#   Therefore Les_Brown__0001.wav remains Les_Brown__0001 and is correctly
#   detected when --filename_separator "__" is used.

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

AUDIO_EXTS_DEFAULT = ".wav,.flac,.mp3,.ogg,.m4a,.aac,.wma"


def safe_name(text: str, preserve_case: bool = True) -> str:
    """Normalize class/folder names. Repeated underscores may be collapsed here."""
    text = str(text).strip()
    if not preserve_case:
        text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_stem_preserve_separator(text: str, preserve_case: bool = True) -> str:
    """
    Normalize filename stems without collapsing repeated underscores.

    This preserves the rename separator:
      Les_Brown__0001 -> Les_Brown__0001
    """
    text = str(text).strip()
    if not preserve_case:
        text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def natural_key(path: Path):
    text = path.name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = to_mono(y)
    if len(y) == 0:
        raise RuntimeError(f"Empty audio file: {path}")
    return y.astype(np.float32), int(sr)


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, target_sr)
        y2 = resample_poly(y, target_sr // gcd, sr // gcd)
        return y2.astype(np.float32)
    except Exception:
        try:
            import librosa
            return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Could not resample {sr} -> {target_sr}. Install scipy or librosa. Error: {e}")


def collect_files(raw_root: Path, classes: list[str], exts: set[str]) -> dict[str, list[Path]]:
    missing = [cls for cls in classes if not (raw_root / cls).is_dir()]
    if missing:
        available = sorted([p.name for p in raw_root.iterdir() if p.is_dir()]) if raw_root.exists() else []
        raise FileNotFoundError(
            "Requested human-talk class folder(s) not found.\n"
            f"Missing: {missing}\n"
            f"RawRoot: {raw_root}\n"
            f"Available folders: {available}\n\n"
            "You are probably pointing RawRoot to the old environmental dataset. Use -RawRoot human_talk_dataset."
        )
    out: dict[str, list[Path]] = {}
    for cls in classes:
        cls_dir = raw_root / cls
        files = sorted([p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts], key=natural_key)
        if not files:
            raise RuntimeError(f"No audio files found for class: {cls_dir.name}")
        out[cls_dir.name] = files
    return out


def parse_renamed_sample(path: Path, class_name: str, separator: str) -> tuple[bool, int | None, str]:
    """Parse filenames such as Les_Brown__0001.wav."""
    cls_prefix = safe_name(class_name, preserve_case=True)
    stem = safe_stem_preserve_separator(path.stem, preserve_case=True)
    pattern = rf"^{re.escape(cls_prefix)}{re.escape(separator)}(\d+)$"
    match = re.match(pattern, stem, flags=re.IGNORECASE)
    if match:
        return True, int(match.group(1)), stem
    if stem.lower().startswith(cls_prefix.lower()):
        return False, None, stem
    return False, None, f"{cls_prefix}{separator}{stem}"


def split_parent_files(files: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0
    return {"train": files[:n_train], "val": files[n_train:n_train + n_val], "test": files[n_train + n_val:]}


def window_starts(n_samples: int, sr: int, segment_sec: float, hop_sec: float, min_keep_sec: float) -> list[int]:
    seg_len = int(round(segment_sec * sr))
    hop_len = int(round(hop_sec * sr))
    if n_samples < int(round(min_keep_sec * sr)):
        return []
    if n_samples <= seg_len:
        return [0]
    starts = list(range(0, n_samples - seg_len + 1, hop_len))
    final_start = n_samples - seg_len
    if starts and starts[-1] != final_start:
        starts.append(final_start)
    return starts


def crop_or_pad(y: np.ndarray, start: int, seg_len: int) -> np.ndarray:
    end = start + seg_len
    chunk = y[start:min(end, len(y))]
    if len(chunk) == seg_len:
        return chunk.astype(np.float32)
    out = np.zeros(seg_len, dtype=np.float32)
    out[:len(chunk)] = chunk
    return out


def unique_id(base: str, used: set[str], separator: str) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        candidate = f"{base}{separator}dup{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human-talk parent-child segment manifest.")
    parser.add_argument("--raw_root", default="human_talk_dataset")
    parser.add_argument("--out_root", required=True, help="Stage data root, e.g. human_talk_workspace/stages/clean2_balanced/data")
    parser.add_argument("--classes", required=True, help="Comma-separated speaker classes for this stage.")
    parser.add_argument("--exts", default=AUDIO_EXTS_DEFAULT)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--min_keep_sec", type=float, default=0.25)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_to_min", action="store_true")
    parser.add_argument("--clips_per_class", type=int, default=0)
    parser.add_argument("--filename_separator", default="__")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    metadata_root = out_root / "metadata"
    segments_root = out_root / "segments"
    if not raw_root.exists():
        raise FileNotFoundError(f"RawRoot does not exist: {raw_root}")
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    metadata_root.mkdir(parents=True, exist_ok=True)
    segments_root.mkdir(parents=True, exist_ok=True)
    classes = parse_csv_list(args.classes)
    if not classes:
        raise RuntimeError("--classes is required for controlled staged human-talk experiments.")
    exts = {x.strip().lower() for x in args.exts.split(",") if x.strip()}
    rng = random.Random(args.seed)
    files_by_class = collect_files(raw_root, classes, exts)
    labels = list(files_by_class.keys())
    counts = {cls: len(files) for cls, files in files_by_class.items()}
    target = min(counts.values()) if args.balance_to_min else (args.clips_per_class if args.clips_per_class > 0 else None)
    selected_by_class = {}
    for cls, files in files_by_class.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        if target is not None:
            shuffled = shuffled[:min(target, len(shuffled))]
        selected_by_class[cls] = sorted(shuffled, key=natural_key)
    selected_counts = {cls: len(files) for cls, files in selected_by_class.items()}
    print("\nPreparing human-talk segments")
    print("-" * 90)
    print(f"Raw root:            {raw_root.resolve()}")
    print(f"Output root:         {out_root.resolve()}")
    print(f"Classes:             {labels}")
    print(f"Original counts:     {counts}")
    print(f"Selected counts:     {selected_counts}")
    print(f"Balance to min:      {args.balance_to_min}")
    print(f"Filename separator:  {args.filename_separator}")
    print(f"Segment sec:         {args.segment_sec}")
    print(f"Hop sec:             {args.hop_sec}")
    print(f"Sample rate:         {args.sample_rate}")
    print("-" * 90)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    (metadata_root / "labels.json").write_text(json.dumps({"labels": labels, "label_to_idx": label_to_idx}, indent=2), encoding="utf-8")
    parent_rows = []
    segment_rows = []
    used_parent_ids = set()
    used_segment_ids = set()
    seg_len = int(round(args.segment_sec * args.sample_rate))
    for class_name, files in selected_by_class.items():
        split_files = split_parent_files(files, args.train_ratio, args.val_ratio, args.seed)
        for split, split_list in split_files.items():
            for path in split_list:
                renamed_ok, sample_index, parsed_parent_id = parse_renamed_sample(path, class_name, args.filename_separator)
                parent_id = unique_id(parsed_parent_id, used_parent_ids, args.filename_separator)
                try:
                    y, sr = read_audio(path)
                    original_sr = sr
                    original_duration_sec = float(len(y) / sr)
                    y = resample_audio(y, sr, args.sample_rate)
                    y = y - float(np.mean(y))
                    parent_duration_sec = float(len(y) / args.sample_rate)
                    starts = window_starts(len(y), args.sample_rate, args.segment_sec, args.hop_sec, args.min_keep_sec)
                    read_status = "ok"
                    error = ""
                except Exception as e:
                    original_sr = ""
                    original_duration_sec = 0.0
                    parent_duration_sec = 0.0
                    y = np.zeros(0, dtype=np.float32)
                    starts = []
                    read_status = "error"
                    error = repr(e)
                parent_rows.append({
                    "parent_clip_id": parent_id,
                    "class_name": class_name,
                    "source_file": path.name,
                    "source_path": str(path),
                    "abs_source_path": str(path.resolve()),
                    "split": split,
                    "renamed_format_ok": int(bool(renamed_ok)),
                    "sample_index": "" if sample_index is None else int(sample_index),
                    "filename_separator": args.filename_separator,
                    "original_sample_rate": original_sr,
                    "target_sample_rate": args.sample_rate,
                    "original_duration_sec": original_duration_sec,
                    "parent_duration_sec": parent_duration_sec,
                    "num_segments": len(starts),
                    "read_status": read_status,
                    "error": error,
                })
                for seg_idx, start in enumerate(starts):
                    segment_id_base = f"{parent_id}_seg{seg_idx:04d}"
                    segment_id = unique_id(segment_id_base, used_segment_ids, args.filename_separator)
                    start_sec = float(start / args.sample_rate)
                    end_sec = float(start_sec + args.segment_sec)
                    chunk = crop_or_pad(y, start, seg_len)
                    out_rel = Path(split) / safe_name(class_name, preserve_case=True) / f"{segment_id}.wav"
                    out_path = segments_root / out_rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(out_path), chunk, args.sample_rate, subtype="PCM_16")
                    row = {
                        "sample_id": segment_id,
                        "filepath": str(out_path),
                        "abs_path": str(out_path.resolve()),
                        "source_file": path.name,
                        "source_path": str(path),
                        "source_stem": path.stem,
                        "primary_label": class_name,
                        "labels": class_name,
                        "split": split,
                        "is_clean_seed": 1,
                        "is_synthetic": 0,
                        "parent_clip_id": parent_id,
                        "segment_id": segment_id,
                        "segment_index": seg_idx,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "segment_sec": args.segment_sec,
                        "hop_sec": args.hop_sec,
                        "parent_duration_sec": parent_duration_sec,
                        "renamed_format_ok": int(bool(renamed_ok)),
                        "sample_index": "" if sample_index is None else int(sample_index),
                    }
                    for lab in labels:
                        row[lab] = 1 if lab == class_name else 0
                    segment_rows.append(row)
    parent_df = pd.DataFrame(parent_rows)
    segment_df = pd.DataFrame(segment_rows)
    parent_manifest = metadata_root / "human_talk_parent_manifest.csv"
    segment_manifest = metadata_root / "multilabel_train_manifest.csv"
    parent_df.to_csv(parent_manifest, index=False)
    segment_df.to_csv(segment_manifest, index=False)
    print("\nPreparation completed.")
    print(f"Parent manifest:  {parent_manifest}")
    print(f"Segment manifest: {segment_manifest}")
    print(f"Labels JSON:      {metadata_root / 'labels.json'}")
    print(f"Segments root:    {segments_root}")
    print(f"Parent clips:     {len(parent_df)}")
    print(f"Segments:         {len(segment_df)}")
    print("\nParent split counts:")
    print(parent_df.groupby(["split", "class_name"]).size().to_string())
    print("\nSegment split counts:")
    print(segment_df.groupby(["split", "primary_label"]).size().to_string())
    print("\nRenamed format check:")
    check = parent_df.groupby("class_name")["renamed_format_ok"].agg(["count", "sum"])
    check["incorrect"] = check["count"] - check["sum"]
    print(check.to_string())


if __name__ == "__main__":
    main()
