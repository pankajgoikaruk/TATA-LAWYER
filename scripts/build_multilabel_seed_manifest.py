# scripts/build_multilabel_seed_manifest.py
#
# Build a multi-label seed manifest from a clean single-label folder structure.
#
# Expected input:
#   multilabel_data/clean_seed/
#   ├─ car_crash/
#   │  ├─ car_crash_0001.wav
#   │  └─ car_crash_0002.wav
#   ├─ conversation/
#   ├─ engine_idling/
#   ├─ fireworks/
#   ├─ gun_shot/
#   ├─ rain/
#   ├─ scream/
#   ├─ thunderstorm/
#   └─ wind/
#
# Output:
#   multilabel_data/metadata/clean_seed_manifest.csv
#   multilabel_data/metadata/labels.json
#
# Each clean file receives exactly one positive label.
# Example:
#   rain/rain_0001.wav -> rain=1, thunderstorm=0, gun_shot=0, ...
#
# Usage:
#   python scripts\build_multilabel_seed_manifest.py `
#     --root "multilabel_data\clean_seed" `
#     --out "multilabel_data\metadata\clean_seed_manifest.csv" `
#     --labels_json "multilabel_data\metadata\labels.json" `
#     --seed 42

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


def natural_key(path: Path):
    """
    Natural filename sorting:
      file_2.wav before file_10.wav
    """
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def safe_label_name(name: str) -> str:
    """
    Convert folder names into safe label names.

    Examples:
      "car crash" -> "car_crash"
      "rain_ thunderstorm" -> "rain_thunderstorm"
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def parse_audio_exts(exts_arg: str | None) -> set[str]:
    if not exts_arg:
        return set(DEFAULT_AUDIO_EXTS)

    exts = set()
    for item in exts_arg.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        exts.add(item)

    if not exts:
        raise ValueError("No valid audio extensions were provided.")

    return exts


def parse_labels(labels_arg: str | None) -> list[str] | None:
    if not labels_arg:
        return None

    labels = [safe_label_name(x) for x in labels_arg.split(",") if x.strip()]
    labels = [x for x in labels if x]

    if not labels:
        raise ValueError("--labels was provided but no valid labels were found.")

    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels found in --labels: {labels}")

    return labels


def collect_class_dirs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )


def collect_audio_files(class_dir: Path, audio_exts: set[str], recursive: bool) -> list[Path]:
    if recursive:
        candidates = class_dir.rglob("*")
    else:
        candidates = class_dir.iterdir()

    files = [
        p for p in candidates
        if p.is_file() and p.suffix.lower() in audio_exts
    ]

    return sorted(files, key=natural_key)


def make_split_for_label(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[Path, str]:
    """
    Create a simple per-label split.

    For very small classes:
      n=1 -> train
      n=2 -> train/test
      n>=3 -> train/val/test if ratios allow
    """
    files = list(files)
    rng.shuffle(files)

    n = len(files)
    if n == 0:
        return {}

    if n == 1:
        return {files[0]: "train"}

    if n == 2:
        return {
            files[0]: "train",
            files[1]: "test",
        }

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Ensure valid non-empty train split.
    if n_train < 1:
        n_train = 1

    # For n >= 3, keep at least one validation and one test item when ratios are non-zero.
    if val_ratio > 0 and n_val < 1:
        n_val = 1

    if test_ratio > 0 and n_test < 1:
        n_test = 1

    # Fix overflow.
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    # Fix underflow.
    while n_train + n_val + n_test < n:
        n_train += 1

    split_map = {}

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:n_train + n_val + n_test]

    for p in train_files:
        split_map[p] = "train"
    for p in val_files:
        split_map[p] = "val"
    for p in test_files:
        split_map[p] = "test"

    return split_map


def main():
    parser = argparse.ArgumentParser(
        description="Build a clean multi-label seed manifest from class folders."
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing clean single-label class folders.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output manifest CSV path. "
            "Default: <root_parent>/metadata/clean_seed_manifest.csv"
        ),
    )
    parser.add_argument(
        "--labels_json",
        default=None,
        help=(
            "Output labels JSON path. "
            "Default: same folder as manifest, labels.json"
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional comma-separated label order. "
            "Example: car_crash,conversation,engine_idling,fireworks,gun_shot,rain,scream,thunderstorm,wind"
        ),
    )
    parser.add_argument(
        "--audio_exts",
        default=None,
        help=(
            "Comma-separated audio extensions. "
            "Default: .wav,.flac,.mp3,.ogg,.m4a,.aac,.wma"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for audio files recursively inside each class folder.",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Do not create train/val/test split; set split=unsplit for all rows.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train split ratio. Default: 0.70",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment. Default: 42",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    audio_exts = parse_audio_exts(args.audio_exts)
    requested_labels = parse_labels(args.labels)

    if args.out:
        out_csv = Path(args.out).resolve()
    else:
        # Example:
        #   root = multilabel_data/clean_seed
        #   out = multilabel_data/metadata/clean_seed_manifest.csv
        out_csv = root.parent / "metadata" / "clean_seed_manifest.csv"

    if args.labels_json:
        labels_json = Path(args.labels_json).resolve()
    else:
        labels_json = out_csv.parent / "labels.json"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_json.parent.mkdir(parents=True, exist_ok=True)

    class_dirs = collect_class_dirs(root)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {root}")

    # Map sanitized label -> class directory.
    label_to_dir: dict[str, Path] = {}
    for class_dir in class_dirs:
        label = safe_label_name(class_dir.name)
        if not label:
            raise RuntimeError(f"Invalid empty label generated from folder: {class_dir}")

        if label in label_to_dir:
            raise RuntimeError(
                "Two folders become the same safe label name:\n"
                f"  {label_to_dir[label]}\n"
                f"  {class_dir}\n"
                f"Safe label: {label}"
            )

        label_to_dir[label] = class_dir

    discovered_labels = sorted(label_to_dir.keys())

    if requested_labels is not None:
        labels = requested_labels

        missing = [x for x in labels if x not in label_to_dir]
        if missing:
            raise RuntimeError(
                "Some requested labels do not exist as folders under root:\n"
                f"  Missing: {missing}\n"
                f"  Available: {discovered_labels}"
            )

        extra = [x for x in discovered_labels if x not in labels]
        if extra:
            print(
                "\nWARNING: Some folders exist but are not included in --labels, so they will be ignored:"
            )
            for x in extra:
                print(f"  - {x}")

    else:
        labels = discovered_labels

    label_to_id = {label: i for i, label in enumerate(labels)}

    print("\nBuilding clean multi-label seed manifest")
    print("-" * 80)
    print(f"Root:        {root}")
    print(f"Output CSV:  {out_csv}")
    print(f"Labels JSON: {labels_json}")
    print(f"Audio exts:  {', '.join(sorted(audio_exts))}")
    print(f"Recursive:   {args.recursive}")
    print(f"Split:       {'disabled' if args.no_split else 'train/val/test'}")
    print(f"Labels ({len(labels)}): {labels}")
    print("-" * 80)

    rng = random.Random(args.seed)

    rows = []
    label_counts = Counter()
    split_counts = Counter()
    label_split_counts = defaultdict(Counter)

    for label in labels:
        class_dir = label_to_dir[label]
        audio_files = collect_audio_files(class_dir, audio_exts, recursive=args.recursive)

        print(f"\nClass: {label}")
        print(f"Folder: {class_dir}")
        print(f"Audio files: {len(audio_files)}")

        if len(audio_files) == 0:
            print(f"WARNING: No supported audio files found for label: {label}")
            continue

        if args.no_split:
            split_map = {p: "unsplit" for p in audio_files}
        else:
            split_map = make_split_for_label(
                files=audio_files,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                rng=rng,
            )

        for audio_path in audio_files:
            relpath = audio_path.relative_to(root).as_posix()
            split = split_map[audio_path]

            row = {
                "sample_id": audio_path.stem,
                "filepath": relpath,
                "abs_path": str(audio_path),
                "source_file": audio_path.name,
                "source_stem": audio_path.stem,
                "source_ext": audio_path.suffix.lower(),
                "class_dir": class_dir.name,
                "primary_label": label,
                "labels": label,
                "num_labels": 1,
                "is_clean_seed": 1,
                "is_synthetic": 0,
                "split": split,
            }

            # Multi-hot label columns.
            for lab in labels:
                row[lab] = 1 if lab == label else 0

            rows.append(row)

            label_counts[label] += 1
            split_counts[split] += 1
            label_split_counts[label][split] += 1

    if not rows:
        raise RuntimeError("No audio rows were created. Check your root path and audio extensions.")

    # Stable row order:
    # split -> primary_label -> filepath
    split_order = {"train": 0, "val": 1, "test": 2, "unsplit": 3}
    rows = sorted(
        rows,
        key=lambda r: (
            split_order.get(r["split"], 99),
            r["primary_label"],
            r["filepath"],
        ),
    )

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
    ] + labels

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels_payload = {
        "task": "multi_label_audio",
        "labels": labels,
        "label_to_id": label_to_id,
        "num_labels": len(labels),
        "manifest": str(out_csv),
        "root": str(root),
        "audio_extensions": sorted(audio_exts),
        "split_enabled": not args.no_split,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
    }

    with labels_json.open("w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)

    print("\n" + "-" * 80)
    print("Manifest created successfully.")
    print(f"Rows:        {len(rows)}")
    print(f"CSV:         {out_csv}")
    print(f"Labels JSON: {labels_json}")

    print("\nClass counts:")
    for label in labels:
        print(f"  {label}: {label_counts[label]}")

    print("\nSplit counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print("\nPer-class split counts:")
    for label in labels:
        counts = label_split_counts[label]
        print(
            f"  {label}: "
            f"train={counts.get('train', 0)}, "
            f"val={counts.get('val', 0)}, "
            f"test={counts.get('test', 0)}, "
            f"unsplit={counts.get('unsplit', 0)}"
        )


if __name__ == "__main__":
    main()