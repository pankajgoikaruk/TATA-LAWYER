# scripts/rename_wavs_by_class.py
#
# Generic class-folder audio renaming utility.
#
# Default naming:
#   class_name_0001.wav
#   class_name_0002.flac
#
# Human-talk/speaker naming example:
#   Les_Brown__0001.wav
#   Mel_Robbins__0001.wav
#
# Supports:
#   - dry-run first
#   - safe two-stage rename
#   - collision checks
#   - rename_manifest.csv for traceability
#   - configurable separator
#   - optional case preservation
#   - manifest path outside the raw dataset root
#
# Manifest columns:
#   class_dir,old_path,new_path,old_name,new_name,old_ext,new_ext,action
#
# Example dry run for environmental data:
#   python scripts\rename_wavs_by_class.py `
#     --root multilabel_data\clean_seed `
#     --manifest rename_manifest.csv
#
# Example apply for environmental data:
#   python scripts\rename_wavs_by_class.py `
#     --root multilabel_data\clean_seed `
#     --manifest rename_manifest.csv `
#     --apply
#
# Example dry run for human-talk data:
#   python scripts\rename_wavs_by_class.py `
#     --root human_talk_dataset `
#     --manifest human_talk_workspace\metadata\rename_manifest.csv `
#     --separator "__" `
#     --preserve_case
#
# Example apply for human-talk data:
#   python scripts\rename_wavs_by_class.py `
#     --root human_talk_dataset `
#     --manifest human_talk_workspace\metadata\rename_manifest.csv `
#     --separator "__" `
#     --preserve_case `
#     --apply
#
# Important path behaviour:
#   --manifest rename_manifest.csv
#       writes to: <root>/rename_manifest.csv
#
#   --manifest human_talk_workspace\metadata\rename_manifest.csv
#       writes to that project-level path, relative to the current working directory.
#
# Therefore, for human-talk experiments, keep the rename manifest here:
#   human_talk_workspace/metadata/rename_manifest.csv

from __future__ import annotations

import argparse
import csv
import re
import uuid
from pathlib import Path


AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}


def natural_key(path: Path):
    """
    Sort filenames naturally:
      file_2.wav before file_10.wav
    """
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def safe_label_name(name: str, preserve_case: bool = False) -> str:
    """
    Convert folder name into a safe filename prefix.

    Examples:
      "car crash" -> "car_crash"
      "rain_ thunderstorm" -> "rain_thunderstorm"
      "Les Brown" -> "Les_Brown" when preserve_case=True
      "Les Brown" -> "les_brown" when preserve_case=False
    """
    name = name.strip()

    if not preserve_case:
        name = name.lower()

    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def collect_class_dirs(root: Path) -> list[Path]:
    """
    Collect immediate class folders under root.

    Example:
      multilabel_data/clean_seed/car_crash
      multilabel_data/clean_seed/conversation
      human_talk_dataset/Les_Brown
      human_talk_dataset/Simon_Sinek
    """
    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )


def collect_audio_files(class_dir: Path) -> list[Path]:
    """
    Collect supported audio files directly inside a class folder.

    This does not search recursively by design. This avoids accidentally
    renaming generated segments or nested output files.
    """
    return sorted(
        [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ],
        key=natural_key,
    )


def parse_existing_index(
    label: str,
    separator: str,
    path: Path,
    preserve_case: bool = False,
) -> int | None:
    """
    Detect whether a file is already named like:
      label_0001.wav
      label__0001.wav
      Label__0001.wav

    Returns the number if matched, otherwise None.
    """
    stem = path.stem if preserve_case else path.stem.lower()
    label_cmp = label if preserve_case else label.lower()
    sep_cmp = separator if preserve_case else separator.lower()

    pattern = rf"^{re.escape(label_cmp)}{re.escape(sep_cmp)}(\d+)$"
    match = re.match(pattern, stem)
    if not match:
        return None
    return int(match.group(1))


def make_index(i: int, digits: int) -> str:
    if digits > 0:
        return str(i).zfill(digits)
    return str(i)


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    """
    Backward-compatible manifest handling.

    Case 1:
      --manifest rename_manifest.csv
      -> writes inside root/rename_manifest.csv, same as the older script.

    Case 2:
      --manifest human_talk_workspace/metadata/rename_manifest.csv
      -> writes exactly there relative to the current working directory.

    Case 3:
      --manifest C:/.../rename_manifest.csv
      -> writes to that absolute path.
    """
    manifest = Path(manifest_arg)

    if manifest.is_absolute():
        return manifest

    # Preserve old behaviour for simple filename only.
    if manifest.parent == Path("."):
        return root / manifest

    # If the user provided a folder path, respect it relative to cwd.
    return manifest.resolve()


def planned_action(old_path: Path, new_path: Path, apply: bool) -> str:
    prefix = "" if apply else "dryrun_"
    if old_path.resolve() == new_path.resolve():
        return f"{prefix}skip_same_name"
    return f"{prefix}rename"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename audio files inside each class directory using "
            "class-name + separator + index while preserving original file extensions."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing class folders, e.g. "
            "multilabel_data/clean_seed or human_talk_dataset"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index for each class. Default: 1",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Zero padding. Example: --digits 4 gives class_0001.wav. Default: 4",
    )
    parser.add_argument(
        "--separator",
        default="_",
        help=(
            "Separator between class prefix and index. "
            "Default '_' gives class_0001.wav. "
            "Use '__' for Class__0001.wav."
        ),
    )
    parser.add_argument(
        "--preserve_case",
        action="store_true",
        help=(
            "Preserve class folder case in output filenames. "
            "Without this, class prefixes are lowercased for backward compatibility."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="rename_manifest.csv",
        help=(
            "CSV file to store old/new filename mapping. "
            "If a simple filename is given, it is written inside --root for backward compatibility. "
            "If a path is given, it is written to that path relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--skip_already_named",
        action="store_true",
        help=(
            "Skip files already named like class_0001.ext or Class__0001.ext. "
            "Useful if you already renamed some files and only want to rename remaining files."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only dry-run is shown.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    class_dirs = collect_class_dirs(root)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {root}")

    manifest_path = resolve_manifest_path(root, args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"\nRoot: {root}")
    print(f"Mode: {'APPLY / RENAME FILES' if args.apply else 'DRY RUN ONLY'}")
    print(f"Supported audio extensions: {', '.join(sorted(AUDIO_EXTS))}")
    print(f"Separator: {repr(args.separator)}")
    print(f"Preserve case: {args.preserve_case}")
    print(f"Skip already named files: {args.skip_already_named}")
    print(f"Manifest: {manifest_path}")
    print("-" * 90)

    total_audio_files = 0
    total_planned_rename = 0
    total_same_name = 0
    total_skipped = 0

    for class_dir in class_dirs:
        label = safe_label_name(class_dir.name, preserve_case=args.preserve_case)
        audio_files = collect_audio_files(class_dir)

        print(f"\nClass: {class_dir.name} -> prefix: {label}")
        print(f"Found audio files: {len(audio_files)}")

        total_audio_files += len(audio_files)

        if not audio_files:
            continue

        planned: list[tuple[Path, Path, str]] = []
        skipped: list[Path] = []

        # If skipping already named files, continue numbering after existing max index.
        existing_indices = []
        if args.skip_already_named:
            for p in audio_files:
                idx = parse_existing_index(
                    label=label,
                    separator=args.separator,
                    path=p,
                    preserve_case=args.preserve_case,
                )
                if idx is not None:
                    existing_indices.append(idx)

            next_index = max(existing_indices, default=args.start - 1) + 1
        else:
            next_index = args.start

        for old_path in audio_files:
            existing_idx = parse_existing_index(
                label=label,
                separator=args.separator,
                path=old_path,
                preserve_case=args.preserve_case,
            )

            if args.skip_already_named and existing_idx is not None:
                skipped.append(old_path)
                action = "skip_already_named" if args.apply else "dryrun_skip_already_named"
                rows.append(
                    {
                        "class_dir": class_dir.name,
                        "old_path": str(old_path),
                        "new_path": str(old_path),
                        "old_name": old_path.name,
                        "new_name": old_path.name,
                        "old_ext": old_path.suffix.lower(),
                        "new_ext": old_path.suffix.lower(),
                        "action": action,
                    }
                )
                continue

            index = make_index(next_index, args.digits)

            # Preserve original extension correctly:
            # .wav stays .wav, .flac stays .flac, .mp3 stays .mp3
            new_name = f"{label}{args.separator}{index}{old_path.suffix.lower()}"
            new_path = class_dir / new_name
            action = planned_action(old_path, new_path, apply=args.apply)

            planned.append((old_path, new_path, action))
            next_index += 1

        rename_items = [
            (old_path, new_path)
            for old_path, new_path, action in planned
            if action.endswith("rename")
        ]
        same_name_items = [
            (old_path, new_path)
            for old_path, new_path, action in planned
            if action.endswith("skip_same_name")
        ]

        print(f"Planned renames: {len(rename_items)}")
        print(f"Already correct names: {len(same_name_items)}")
        print(f"Skipped already named: {len(skipped)}")

        total_planned_rename += len(rename_items)
        total_same_name += len(same_name_items)
        total_skipped += len(skipped)

        # Safety check 1: target names must be unique in the plan.
        target_names_lower = [new_path.name.lower() for _, new_path, _ in planned]
        if len(target_names_lower) != len(set(target_names_lower)):
            raise RuntimeError(f"Duplicate target filenames detected in planned rename: {class_dir}")

        # Safety check 2: do not overwrite files that are not part of this rename plan.
        planned_old_paths = {old_path.resolve() for old_path, _, _ in planned}
        for old_path, new_path, action in planned:
            if action.endswith("skip_same_name"):
                continue

            if new_path.exists() and new_path.resolve() not in planned_old_paths:
                raise RuntimeError(
                    "Target file already exists and is not part of this rename plan:\n"
                    f"  old: {old_path}\n"
                    f"  new: {new_path}\n"
                    "Use --skip_already_named or inspect the directory manually."
                )

        # Show first few rename examples only.
        shown = 0
        for old_path, new_path, action in planned:
            if not action.endswith("rename"):
                continue
            print(f"  {old_path.name}  ->  {new_path.name}")
            shown += 1
            if shown >= 10:
                break

        if len(rename_items) > 10:
            print(f"  ... {len(rename_items) - 10} more")

        # Store manifest rows.
        for old_path, new_path, action in planned:
            rows.append(
                {
                    "class_dir": class_dir.name,
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "old_name": old_path.name,
                    "new_name": new_path.name,
                    "old_ext": old_path.suffix.lower(),
                    "new_ext": new_path.suffix.lower(),
                    "action": action,
                }
            )

        if args.apply and rename_items:
            # Two-stage rename avoids collisions, for example:
            # file_a.wav -> car_crash_0001.wav
            # file_b.wav -> car_crash_0002.wav
            #
            # Even if target names overlap old names, temporary names keep it safe.
            temp_pairs: list[tuple[Path, Path]] = []

            for old_path, new_path in rename_items:
                temp_name = f".tmp_rename_{uuid.uuid4().hex}{old_path.suffix.lower()}"
                temp_path = old_path.with_name(temp_name)
                old_path.rename(temp_path)
                temp_pairs.append((temp_path, new_path))

            for temp_path, new_path in temp_pairs:
                temp_path.rename(new_path)

    # Write manifest.
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_dir",
                "old_path",
                "new_path",
                "old_name",
                "new_name",
                "old_ext",
                "new_ext",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "-" * 90)
    print(f"Total supported audio files found: {total_audio_files}")
    print(f"Total planned renames: {total_planned_rename}")
    print(f"Total already correct names: {total_same_name}")
    print(f"Total skipped already named: {total_skipped}")
    print(f"Manifest written to: {manifest_path}")

    if not args.apply:
        print("\nDry run completed. No files were renamed.")
        print("Run again with --apply to actually rename files.")
    else:
        print("\nRenaming completed successfully.")


if __name__ == "__main__":
    main()
