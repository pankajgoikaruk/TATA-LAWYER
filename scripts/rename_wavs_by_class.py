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
#   - continuation after the highest matching index found in a separate
#     reference dataset and/or CSV manifest
#   - cross-dataset filename collision protection
#
# Example for new verified silence clips
# --------------------------------------
# Folder layout:
#   dataset/new_verified_rare_event_audio/silence/<27 audio files>
#
# Dry run:
#   python scripts\rename_wavs_by_class.py `
#     --root "dataset\new_verified_rare_event_audio" `
#     --separator "__" `
#     --reference_root "dataset\human_talk_tata_seed_dataset" `
#     --reference_manifest "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\metadata\tata_seed_parent_manifest_v09_RARE_EVENTS_CORRECTED.csv" `
#     --manifest "human_talk_workspace\tata_v0.9_pipeline\shared\correction_ledgers\v09_new_silence_rename_manifest.csv"
#
# Apply after checking the dry-run mapping:
#   rerun the same command with --apply
#
# Important:
#   --root must contain immediate class folders. Therefore, when the audio is in
#   dataset/new_verified_rare_event_audio/silence, use
#   dataset/new_verified_rare_event_audio as --root.
#
# Manifest columns:
#   class_dir,old_path,new_path,old_name,new_name,old_ext,new_ext,
#   assigned_index,reference_max_index,action

from __future__ import annotations

import argparse
import csv
import re
import uuid
from pathlib import Path
from typing import Iterable


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

# Known filename/path columns used by the project's manifests.
REFERENCE_FILENAME_COLUMNS = (
    "file_name",
    "new_name",
    "source_file",
    "audio_file",
    "audio_path",
    "file_path",
    "abs_path",
    "source_audio_path",
    "review_audio_path",
)


def natural_key(path: Path):
    """Sort filenames naturally: file_2.wav before file_10.wav."""
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def safe_label_name(name: str, preserve_case: bool = False) -> str:
    """
    Convert a folder name into a safe filename prefix.

    Examples:
      "car crash" -> "car_crash"
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
    """Collect immediate class folders under root."""
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def collect_audio_files(class_dir: Path) -> list[Path]:
    """
    Collect supported audio files directly inside a class folder.

    This intentionally does not search recursively, avoiding generated
    segments or nested output files.
    """
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS
        ],
        key=natural_key,
    )


def parse_existing_index_from_name(
    label: str,
    separator: str,
    filename: str,
    preserve_case: bool = False,
) -> int | None:
    """
    Return the numeric index from a standard filename.

    Examples:
      silence__0001.wav -> 1
      class_0012.flac   -> 12
    """
    path = Path(str(filename).strip().strip('"'))
    stem = path.stem if preserve_case else path.stem.lower()
    label_cmp = label if preserve_case else label.lower()
    separator_cmp = separator if preserve_case else separator.lower()

    pattern = rf"^{re.escape(label_cmp)}{re.escape(separator_cmp)}(\d+)$"
    match = re.match(pattern, stem)
    return int(match.group(1)) if match else None


def parse_existing_index(
    label: str,
    separator: str,
    path: Path,
    preserve_case: bool = False,
) -> int | None:
    """Path-based wrapper retained for compatibility with the old script."""
    return parse_existing_index_from_name(
        label=label,
        separator=separator,
        filename=path.name,
        preserve_case=preserve_case,
    )


def make_index(index: int, digits: int) -> str:
    if digits > 0:
        return str(index).zfill(digits)
    return str(index)


def resolve_manifest_path(root: Path, manifest_arg: str) -> Path:
    """
    Backward-compatible manifest path handling.

    A simple filename is written inside --root. A path containing folders is
    resolved relative to the current working directory unless absolute.
    """
    manifest = Path(manifest_arg)

    if manifest.is_absolute():
        return manifest

    if manifest.parent == Path("."):
        return root / manifest

    return manifest.resolve()


def planned_action(old_path: Path, new_path: Path, apply: bool) -> str:
    prefix = "" if apply else "dryrun_"
    if old_path.resolve() == new_path.resolve():
        return f"{prefix}skip_same_name"
    return f"{prefix}rename"


def iter_reference_audio_names(reference_root: Path) -> Iterable[str]:
    """Yield supported audio filenames recursively from a reference dataset."""
    for path in reference_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            yield path.name


def iter_reference_manifest_names(reference_manifest: Path) -> Iterable[str]:
    """
    Yield filenames from recognised filename/path columns in a CSV manifest.
    """
    with reference_manifest.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        matched_columns = [
            column
            for column in REFERENCE_FILENAME_COLUMNS
            if column in fieldnames
        ]

        if not matched_columns:
            raise RuntimeError(
                "Reference manifest has no recognised filename/path column.\n"
                f"Manifest: {reference_manifest}\n"
                f"Expected one of: {', '.join(REFERENCE_FILENAME_COLUMNS)}"
            )

        for row in reader:
            for column in matched_columns:
                value = str(row.get(column, "") or "").strip()
                if not value:
                    continue

                filename = Path(value.strip('"')).name
                if Path(filename).suffix.lower() in AUDIO_EXTS:
                    yield filename


def collect_reference_names(
    reference_root: Path | None,
    reference_manifest: Path | None,
) -> set[str]:
    """
    Collect case-insensitive filenames from a reference dataset and manifest.
    """
    names: set[str] = set()

    if reference_root is not None:
        names.update(
            filename.lower()
            for filename in iter_reference_audio_names(reference_root)
        )

    if reference_manifest is not None:
        names.update(
            filename.lower()
            for filename in iter_reference_manifest_names(
                reference_manifest
            )
        )

    return names


def reference_indices_for_label(
    reference_names: set[str],
    label: str,
    separator: str,
    preserve_case: bool,
) -> list[int]:
    """Extract all matching indices for one class from reference filenames."""
    indices: list[int] = []

    for filename in reference_names:
        index = parse_existing_index_from_name(
            label=label,
            separator=separator,
            filename=filename,
            preserve_case=preserve_case,
        )
        if index is not None:
            indices.append(index)

    return indices


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename audio files inside each immediate class directory using "
            "class-name + separator + index while preserving extensions."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing immediate class folders, e.g. "
            "dataset/new_verified_rare_event_audio"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "Minimum starting index for each class. When reference sources "
            "contain larger indices, numbering continues after their maximum."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help=(
            "Zero padding. Example: --digits 4 gives silence__0001.wav. "
            "Default: 4"
        ),
    )
    parser.add_argument(
        "--separator",
        default="_",
        help=(
            "Separator between class prefix and index. Use '__' for "
            "silence__0001.wav."
        ),
    )
    parser.add_argument(
        "--preserve_case",
        action="store_true",
        help="Preserve class-folder case in output filename prefixes.",
    )
    parser.add_argument(
        "--manifest",
        default="rename_manifest.csv",
        help="CSV file storing the old/new filename mapping.",
    )
    parser.add_argument(
        "--skip_already_named",
        action="store_true",
        help=(
            "Skip local files already following the selected class naming "
            "pattern. Useful when resuming a partially renamed directory."
        ),
    )
    parser.add_argument(
        "--reference_root",
        default=None,
        help=(
            "Optional existing dataset root scanned recursively. Numbering "
            "continues after its highest matching class index, and its "
            "filenames are protected from reuse."
        ),
    )
    parser.add_argument(
        "--reference_manifest",
        default=None,
        help=(
            "Optional existing CSV manifest. Recognised filename/path columns "
            "are scanned for highest indices and collision protection."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, run a dry run only.",
    )

    args = parser.parse_args()

    if args.start < 1:
        raise ValueError("--start must be at least 1.")
    if args.digits < 0:
        raise ValueError("--digits cannot be negative.")

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    reference_root = (
        Path(args.reference_root).resolve()
        if args.reference_root
        else None
    )
    if reference_root is not None and not reference_root.is_dir():
        raise FileNotFoundError(
            f"Reference dataset root not found: {reference_root}"
        )

    reference_manifest = (
        Path(args.reference_manifest).resolve()
        if args.reference_manifest
        else None
    )
    if (
        reference_manifest is not None
        and not reference_manifest.is_file()
    ):
        raise FileNotFoundError(
            f"Reference manifest not found: {reference_manifest}"
        )

    class_dirs = collect_class_dirs(root)
    if not class_dirs:
        raise RuntimeError(
            f"No immediate class directories found under: {root}\n"
            "For .../new_verified_rare_event_audio/silence, pass "
            ".../new_verified_rare_event_audio as --root."
        )

    manifest_path = resolve_manifest_path(root, args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    reference_names = collect_reference_names(
        reference_root=reference_root,
        reference_manifest=reference_manifest,
    )

    rows: list[dict[str, object]] = []

    print(f"\nRoot: {root}")
    print(
        "Mode: "
        + ("APPLY / RENAME FILES" if args.apply else "DRY RUN ONLY")
    )
    print(
        "Supported audio extensions: "
        + ", ".join(sorted(AUDIO_EXTS))
    )
    print(f"Separator: {args.separator!r}")
    print(f"Preserve case: {args.preserve_case}")
    print(f"Skip already named files: {args.skip_already_named}")
    print(f"Reference root: {reference_root or 'not supplied'}")
    print(
        "Reference manifest: "
        f"{reference_manifest or 'not supplied'}"
    )
    print(f"Protected reference filenames: {len(reference_names):,}")
    print(f"Rename manifest: {manifest_path}")
    print("-" * 90)

    total_audio_files = 0
    total_planned_rename = 0
    total_same_name = 0
    total_skipped = 0

    for class_dir in class_dirs:
        label = safe_label_name(
            class_dir.name,
            preserve_case=args.preserve_case,
        )
        audio_files = collect_audio_files(class_dir)

        print(f"\nClass: {class_dir.name} -> prefix: {label}")
        print(f"Found audio files: {len(audio_files)}")
        total_audio_files += len(audio_files)

        if not audio_files:
            continue

        local_existing_indices: list[int] = []
        if args.skip_already_named:
            for path in audio_files:
                index = parse_existing_index(
                    label=label,
                    separator=args.separator,
                    path=path,
                    preserve_case=args.preserve_case,
                )
                if index is not None:
                    local_existing_indices.append(index)

        reference_indices = reference_indices_for_label(
            reference_names=reference_names,
            label=label,
            separator=args.separator,
            preserve_case=args.preserve_case,
        )

        reference_max_index = max(reference_indices, default=0)
        local_max_index = max(local_existing_indices, default=0)
        next_index = max(
            args.start - 1,
            reference_max_index,
            local_max_index,
        ) + 1

        print(f"Reference maximum index: {reference_max_index}")
        print(f"Local maximum existing index: {local_max_index}")
        print(f"First new index: {next_index}")

        planned: list[tuple[Path, Path, str, int]] = []
        skipped: list[Path] = []

        for old_path in audio_files:
            existing_index = parse_existing_index(
                label=label,
                separator=args.separator,
                path=old_path,
                preserve_case=args.preserve_case,
            )

            if args.skip_already_named and existing_index is not None:
                skipped.append(old_path)
                action = (
                    "skip_already_named"
                    if args.apply
                    else "dryrun_skip_already_named"
                )
                rows.append(
                    {
                        "class_dir": class_dir.name,
                        "old_path": str(old_path),
                        "new_path": str(old_path),
                        "old_name": old_path.name,
                        "new_name": old_path.name,
                        "old_ext": old_path.suffix.lower(),
                        "new_ext": old_path.suffix.lower(),
                        "assigned_index": existing_index,
                        "reference_max_index": reference_max_index,
                        "action": action,
                    }
                )
                continue

            assigned_index = next_index
            index_text = make_index(assigned_index, args.digits)
            new_name = (
                f"{label}{args.separator}{index_text}"
                f"{old_path.suffix.lower()}"
            )
            new_path = class_dir / new_name
            action = planned_action(
                old_path,
                new_path,
                apply=args.apply,
            )

            planned.append(
                (old_path, new_path, action, assigned_index)
            )
            next_index += 1

        rename_items = [
            (old_path, new_path)
            for old_path, new_path, action, _ in planned
            if action.endswith("rename")
        ]
        same_name_items = [
            (old_path, new_path)
            for old_path, new_path, action, _ in planned
            if action.endswith("skip_same_name")
        ]

        print(f"Planned renames: {len(rename_items)}")
        print(f"Already correct names: {len(same_name_items)}")
        print(f"Skipped already named: {len(skipped)}")

        total_planned_rename += len(rename_items)
        total_same_name += len(same_name_items)
        total_skipped += len(skipped)

        # Safety check 1: all proposed target names must be unique.
        target_names_lower = [
            new_path.name.lower()
            for _, new_path, _, _ in planned
        ]
        if len(target_names_lower) != len(set(target_names_lower)):
            raise RuntimeError(
                "Duplicate target filenames detected in the planned rename: "
                f"{class_dir}"
            )

        # Safety check 2: never reuse a protected reference filename.
        for old_path, new_path, action, _ in planned:
            if action.endswith("skip_same_name"):
                continue

            if new_path.name.lower() in reference_names:
                raise RuntimeError(
                    "Target filename already exists in the protected "
                    "reference dataset or manifest:\n"
                    f"  old: {old_path}\n"
                    f"  proposed: {new_path.name}\n"
                    "No files were renamed."
                )

        # Safety check 3: never overwrite unrelated files in the new folder.
        planned_old_paths = {
            old_path.resolve()
            for old_path, _, _, _ in planned
        }
        for old_path, new_path, action, _ in planned:
            if action.endswith("skip_same_name"):
                continue

            if (
                new_path.exists()
                and new_path.resolve() not in planned_old_paths
            ):
                raise RuntimeError(
                    "Target file already exists and is not part of this "
                    "rename plan:\n"
                    f"  old: {old_path}\n"
                    f"  new: {new_path}\n"
                    "No files were renamed."
                )

        shown = 0
        for old_path, new_path, action, _ in planned:
            if not action.endswith("rename"):
                continue

            print(f"  {old_path.name}  ->  {new_path.name}")
            shown += 1
            if shown >= 10:
                break

        if len(rename_items) > 10:
            print(f"  ... {len(rename_items) - 10} more")

        for old_path, new_path, action, assigned_index in planned:
            rows.append(
                {
                    "class_dir": class_dir.name,
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "old_name": old_path.name,
                    "new_name": new_path.name,
                    "old_ext": old_path.suffix.lower(),
                    "new_ext": new_path.suffix.lower(),
                    "assigned_index": assigned_index,
                    "reference_max_index": reference_max_index,
                    "action": action,
                }
            )

        if args.apply and rename_items:
            # Two-stage rename prevents collisions among files being renamed.
            temp_pairs: list[tuple[Path, Path, Path]] = []

            try:
                for old_path, new_path in rename_items:
                    temp_name = (
                        f".tmp_rename_{uuid.uuid4().hex}"
                        f"{old_path.suffix.lower()}"
                    )
                    temp_path = old_path.with_name(temp_name)
                    old_path.rename(temp_path)
                    temp_pairs.append((temp_path, old_path, new_path))

                for temp_path, _, new_path in temp_pairs:
                    temp_path.rename(new_path)

            except Exception:
                # Best-effort rollback for temporary files not yet finalised.
                for temp_path, old_path, new_path in reversed(temp_pairs):
                    if temp_path.exists() and not old_path.exists():
                        temp_path.rename(old_path)
                    elif new_path.exists() and not old_path.exists():
                        new_path.rename(old_path)
                raise

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_dir",
                "old_path",
                "new_path",
                "old_name",
                "new_name",
                "old_ext",
                "new_ext",
                "assigned_index",
                "reference_max_index",
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
        print(
            "Inspect the manifest, then rerun the same command with --apply."
        )
    else:
        print("\nRenaming completed successfully.")


if __name__ == "__main__":
    main()
