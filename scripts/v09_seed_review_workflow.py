#!/usr/bin/env python
"""TATA-LAWYER v0.9 seed-review workflow.

Commands:
  apply-silence    Apply the completed 100-row silence review.
  prepare-audience Create the audience-reaction review package.
  apply-audience   Apply the completed audience review.
  add-new-silence  Add 27 verified silence clips as train-only augmentation.

Every stage writes a new manifest. Earlier manifests are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LABELS = [
    "Brene_Brown", "Eckhart_Tolle", "Eric_Thomas", "Gary_Vee",
    "Jay_Shetty", "Nick_Vujicic", "other_speaker_present",
    "music_present", "audience_reaction_present", "silence_present",
]
FINE_AUDIENCE = ["applause_present", "laughter_present", "crowd_cheer_present"]
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: object, fallback: str = "unknown") -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    return re.sub(r"\s+", "_", text)[:120]


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def ensure_new(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists:\n{path}\n"
            "Use --overwrite only when intentionally rebuilding this stage."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or (~values.isin([0, 1])).any():
        raise ValueError(f"{name} must contain only 0 or 1 and no blanks.")
    return values.astype("int8")


def is_reviewed(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("reviewed")


def fill_review_identity(review: pd.DataFrame, reviewer: str) -> pd.DataFrame:
    review = review.copy()
    done = is_reviewed(review["review_status"])
    reviewer_blank = review["reviewer"].isna() | review["reviewer"].astype(str).str.strip().eq("")
    utc_blank = review["reviewed_utc"].isna() | review["reviewed_utc"].astype(str).str.strip().eq("")
    review.loc[done & reviewer_blank, "reviewer"] = reviewer
    review.loc[done & utc_blank, "reviewed_utc"] = utc_now()
    return review


def recompute(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    columns = [f"v09_{label}" for label in LABELS]
    require_columns(frame, columns, "Parent manifest")
    for column in columns:
        frame[column] = binary(frame[column], column)
    frame["v09_num_active_labels"] = frame[columns].sum(axis=1).astype("int16")
    frame["v09_labels"] = frame[columns].apply(
        lambda row: "|".join(label for label, value in zip(LABELS, row.tolist()) if int(value) == 1),
        axis=1,
    )
    frame["v09_zero_active_after_review"] = (frame["v09_num_active_labels"] == 0).astype("int8")
    return frame


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def paths(repo_root: Path) -> dict[str, Path]:
    v09 = repo_root / "human_talk_workspace" / "tata_v0.9_pipeline"
    triage = v09 / "tata_triage_model"
    dataset = repo_root / "dataset" / "human_talk_tata_seed_dataset"
    return {
        "v09": v09,
        "metadata": triage / "metadata",
        "manual": triage / "manual_review",
        "ledgers": v09 / "shared" / "correction_ledgers",
        "dataset": dataset,
        "new_silence": dataset / "new_verified_rare_event_audio" / "silence",
    }


def audio_index(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            result[path.name.lower()].append(path.resolve())
    return result


def choose_audio(row: pd.Series, index: dict[str, list[Path]]) -> Path:
    for column in ["file_path", "abs_path", "source_path"]:
        if column in row and not pd.isna(row[column]):
            candidate = Path(str(row[column]).strip().strip('"'))
            if candidate.is_file():
                return candidate.resolve()
    name = str(row["file_name"]).strip()
    candidates = index.get(name.lower(), [])
    if not candidates:
        raise FileNotFoundError(f"No audio file found for {name}")
    if len(candidates) == 1:
        return candidates[0]
    tokens = [
        str(row.get("source_subfolder", "")).strip().lower().replace(" ", "_"),
        str(row.get("source_group", "")).strip().lower().replace(" ", "_"),
        str(row.get("primary_label", "")).strip().lower().replace(" ", "_"),
    ]
    scored = []
    for candidate in candidates:
        text = str(candidate).lower().replace("\\", "/")
        scored.append((sum(1 for token in tokens if token and token in text), str(candidate), candidate))
    scored.sort(reverse=True)
    best_score = scored[0][0]
    best = [item[2] for item in scored if item[0] == best_score]
    if len(best) != 1:
        raise RuntimeError(f"Ambiguous audio matches for {name}: {candidates}")
    return best[0]


def apply_silence(args: argparse.Namespace) -> int:
    p = paths(args.repo_root)
    source = p["metadata"] / "tata_seed_parent_manifest_v09_REVIEW.csv"
    review_path = p["manual"] / "silence_existing_v09" / "silence_existing_review_sheet_v09.csv"
    output_path = p["metadata"] / "tata_seed_parent_manifest_v09_SILENCE_CORRECTED.csv"
    ledger_path = p["ledgers"] / "v09_silence_corrections_applied.csv"
    summary_path = p["ledgers"] / "v09_silence_corrections_summary.json"
    require_file(source, "Parent review manifest")
    require_file(review_path, "Silence review sheet")
    for item in [output_path, ledger_path, summary_path]:
        ensure_new(item, args.overwrite)

    parents = pd.read_csv(source, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)
    require_columns(parents, ["clip_id", "v09_silence_present", "v09_keep_for_tata_training"], "Parent manifest")
    require_columns(review, [
        "clip_id", "current_silence_present", "corrected_silence_present",
        "review_status", "review_action", "keep_for_tata_training",
        "review_notes", "reviewer", "reviewed_utc",
    ], "Silence review sheet")
    if len(review) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} reviewed silence rows, found {len(review)}.")
    if review["clip_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate clip IDs exist in the silence review sheet.")
    if not is_reviewed(review["review_status"]).all():
        raise ValueError(f"{int((~is_reviewed(review['review_status'])).sum())} silence rows remain pending.")

    review["current_silence_present"] = binary(review["current_silence_present"], "current_silence_present")
    review["corrected_silence_present"] = binary(review["corrected_silence_present"], "corrected_silence_present")
    review["keep_for_tata_training"] = binary(review["keep_for_tata_training"], "keep_for_tata_training")
    review = fill_review_identity(review, args.reviewer)

    parents["clip_id"] = parents["clip_id"].astype(str)
    review["clip_id"] = review["clip_id"].astype(str)
    indexed = review.set_index("clip_id")
    missing = set(indexed.index) - set(parents["clip_id"])
    if missing:
        raise ValueError(f"{len(missing)} reviewed IDs are missing from the parent manifest.")

    output = parents.copy()
    mask = output["clip_id"].isin(indexed.index)
    before = output.loc[mask, ["clip_id", "v09_silence_present"]].set_index("clip_id")["v09_silence_present"]
    output.loc[mask, "v09_silence_present"] = output.loc[mask, "clip_id"].map(indexed["corrected_silence_present"]).astype("int8")
    output.loc[mask, "v09_keep_for_tata_training"] = output.loc[mask, "clip_id"].map(indexed["keep_for_tata_training"]).astype("int8")
    for target, source_col in [
        ("v09_silence_review_action", "review_action"),
        ("v09_silence_review_notes", "review_notes"),
        ("v09_silence_reviewer", "reviewer"),
        ("v09_silence_reviewed_utc", "reviewed_utc"),
    ]:
        output.loc[mask, target] = output.loc[mask, "clip_id"].map(indexed[source_col])
    output.loc[mask, "v09_silence_review_status"] = "reviewed"
    changed_ids = {
        clip_id for clip_id in indexed.index
        if int(before.loc[clip_id]) != int(indexed.loc[clip_id, "corrected_silence_present"])
    }
    output["v09_silence_label_changed"] = output["clip_id"].isin(changed_ids).astype("int8")
    if "v09_label_changed" in output.columns:
        output.loc[output["clip_id"].isin(changed_ids), "v09_label_changed"] = 1
    output = recompute(output)

    output.to_csv(output_path, index=False, encoding="utf-8")
    review.to_csv(review_path, index=False, encoding="utf-8")
    review.assign(
        silence_label_changed=(review["current_silence_present"] != review["corrected_silence_present"]).astype("int8")
    ).to_csv(ledger_path, index=False, encoding="utf-8")
    write_json(summary_path, {
        "generated_utc": utc_now(),
        "review_rows": len(review),
        "changed_labels": len(changed_ids),
        "silence_positive_after": int(output["v09_silence_present"].sum()),
        "zero_active_rows_temporary": int(output["v09_zero_active_after_review"].sum()),
        "output_manifest": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_manifest_modified": False,
    })
    print("[COMPLETE] Silence corrections applied to a new manifest.")
    print(f"Output: {output_path}")
    print(f"Changed labels: {len(changed_ids)}")
    print(f"Silence positives after: {int(output['v09_silence_present'].sum())}")
    return 0


def prepare_audience(args: argparse.Namespace) -> int:
    p = paths(args.repo_root)
    source = p["metadata"] / "tata_seed_parent_manifest_v09_SILENCE_CORRECTED.csv"
    silence_review_path = p["manual"] / "silence_existing_v09" / "silence_existing_review_sheet_v09.csv"
    review_root = p["manual"] / "audience_reaction_existing_v09"
    audio_root = review_root / "audio"
    review_path = review_root / "audience_reaction_existing_review_sheet_v09.csv"
    unresolved_path = review_root / "audience_reaction_existing_missing_or_ambiguous_v09.csv"
    summary_path = review_root / "audience_reaction_review_package_summary.json"
    readme_path = review_root / "README_AUDIENCE_REACTION_REVIEW.md"
    require_file(source, "Silence-corrected manifest")
    require_file(silence_review_path, "Completed silence review sheet")
    require_dir(args.dataset_root, "Seed dataset root")
    if review_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Audience review package already exists:\n{review_root}")
        shutil.rmtree(review_root)
    audio_root.mkdir(parents=True, exist_ok=True)

    parents = pd.read_csv(source, low_memory=False)
    silence_review = pd.read_csv(silence_review_path, low_memory=False)
    require_columns(parents, [
        "clip_id", "file_name", "source_group", "source_subfolder", "primary_label", "v09_split",
        "v09_audience_reaction_present", "v09_silence_present", "v09_music_present",
        "v09_other_speaker_present", "v09_keep_for_tata_training",
    ], "Silence-corrected manifest")
    silence_ids = set(silence_review["clip_id"].astype(str))
    mask = (binary(parents["v09_audience_reaction_present"], "v09_audience_reaction_present") == 1) | parents["clip_id"].astype(str).isin(silence_ids)
    candidates = parents.loc[mask].sort_values(["source_group", "source_subfolder", "file_name"], kind="stable").reset_index(drop=True)
    index = audio_index(args.dataset_root)
    rows, unresolved = [], []
    for order, (_, row) in enumerate(candidates.iterrows(), start=1):
        try:
            source_audio = choose_audio(row, index)
        except Exception as exc:
            unresolved.append({"review_order": order, "clip_id": row["clip_id"], "file_name": row["file_name"], "reason": str(exc)})
            continue
        destination = audio_root / (
            f"AUDIENCE_{order:04d}__{safe_name(row['clip_id'], f'row_{order:04d}')}__"
            f"{safe_name(source_audio.stem, f'audio_{order:04d}')}{source_audio.suffix.lower()}"
        )
        shutil.copy2(source_audio, destination)
        current = int(row["v09_audience_reaction_present"])
        rows.append({
            "review_order": order,
            "clip_id": row["clip_id"],
            "file_name": row["file_name"],
            "source_group": row["source_group"],
            "source_subfolder": row["source_subfolder"],
            "primary_label": row["primary_label"],
            "split": row["v09_split"],
            "included_because_current_audience_positive": current,
            "included_because_silence_reviewed": int(str(row["clip_id"]) in silence_ids),
            "current_audience_reaction_present": current,
            "corrected_audience_reaction_present": current,
            "current_silence_present": int(row["v09_silence_present"]),
            "current_music_present": int(row["v09_music_present"]),
            "current_other_speaker_present": int(row["v09_other_speaker_present"]),
            "review_status": "pending",
            "review_action": "",
            "keep_for_tata_training": int(row["v09_keep_for_tata_training"]),
            "review_notes": "",
            "reviewer": "",
            "reviewed_utc": "",
            "source_audio_path": str(source_audio),
            "review_audio_path": str(destination.resolve()),
            "source_audio_sha256": sha256_file(source_audio),
        })
    pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8")
    pd.DataFrame(unresolved).to_csv(unresolved_path, index=False, encoding="utf-8")
    readme_path.write_text(
        "# Audience-Reaction Review\n\n"
        "Check only whether applause, laughter, cheering, or audience/crowd reaction is present.\n"
        "Set corrected_audience_reaction_present to 1 or 0, mark review_status=reviewed, "
        "and use review_action=keep or correct_label. Ignore silence and music in this pass.\n",
        encoding="utf-8",
    )
    write_json(summary_path, {
        "generated_utc": utc_now(),
        "current_audience_positive_parents": int(parents["v09_audience_reaction_present"].sum()),
        "silence_review_parent_ids": len(silence_ids),
        "union_review_rows": len(candidates),
        "resolved_audio_rows": len(rows),
        "unresolved_audio_rows": len(unresolved),
        "source_manifest_modified": False,
    })
    print("[COMPLETE] Audience-reaction review package created.")
    print(f"Review rows: {len(candidates)}")
    print(f"Resolved audio: {len(rows)}")
    print(f"Unresolved audio: {len(unresolved)}")
    print(f"Review CSV: {review_path}")
    return 2 if unresolved else 0


def apply_audience(args: argparse.Namespace) -> int:
    p = paths(args.repo_root)
    source = p["metadata"] / "tata_seed_parent_manifest_v09_SILENCE_CORRECTED.csv"
    review_path = p["manual"] / "audience_reaction_existing_v09" / "audience_reaction_existing_review_sheet_v09.csv"
    output_path = p["metadata"] / "tata_seed_parent_manifest_v09_RARE_EVENTS_CORRECTED.csv"
    ledger_path = p["ledgers"] / "v09_audience_reaction_corrections_applied.csv"
    summary_path = p["ledgers"] / "v09_audience_reaction_corrections_summary.json"
    require_file(source, "Silence-corrected manifest")
    require_file(review_path, "Audience review sheet")
    for item in [output_path, ledger_path, summary_path]:
        ensure_new(item, args.overwrite)
    parents = pd.read_csv(source, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)
    require_columns(review, [
        "clip_id", "current_audience_reaction_present", "corrected_audience_reaction_present",
        "review_status", "review_action", "keep_for_tata_training", "review_notes", "reviewer", "reviewed_utc",
    ], "Audience review sheet")
    if review["clip_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate clip IDs exist in the audience review sheet.")
    if not is_reviewed(review["review_status"]).all():
        raise ValueError(f"{int((~is_reviewed(review['review_status'])).sum())} audience rows remain pending.")
    review["current_audience_reaction_present"] = binary(review["current_audience_reaction_present"], "current_audience_reaction_present")
    review["corrected_audience_reaction_present"] = binary(review["corrected_audience_reaction_present"], "corrected_audience_reaction_present")
    review["keep_for_tata_training"] = binary(review["keep_for_tata_training"], "keep_for_tata_training")
    review = fill_review_identity(review, args.reviewer)
    parents["clip_id"] = parents["clip_id"].astype(str)
    review["clip_id"] = review["clip_id"].astype(str)
    indexed = review.set_index("clip_id")
    if set(indexed.index) - set(parents["clip_id"]):
        raise ValueError("Some reviewed audience clip IDs are missing from the parent manifest.")
    output = parents.copy()
    mask = output["clip_id"].isin(indexed.index)
    before = output.loc[mask, ["clip_id", "v09_audience_reaction_present"]].set_index("clip_id")["v09_audience_reaction_present"]
    output.loc[mask, "v09_audience_reaction_present"] = output.loc[mask, "clip_id"].map(indexed["corrected_audience_reaction_present"]).astype("int8")
    output.loc[mask, "v09_keep_for_tata_training"] = output.loc[mask, "clip_id"].map(indexed["keep_for_tata_training"]).astype("int8")
    for target, source_col in [
        ("v09_audience_review_action", "review_action"),
        ("v09_audience_review_notes", "review_notes"),
        ("v09_audience_reviewer", "reviewer"),
        ("v09_audience_reviewed_utc", "reviewed_utc"),
    ]:
        output.loc[mask, target] = output.loc[mask, "clip_id"].map(indexed[source_col])
    output.loc[mask, "v09_audience_review_status"] = "reviewed"
    changed_ids = {
        clip_id for clip_id in indexed.index
        if int(before.loc[clip_id]) != int(indexed.loc[clip_id, "corrected_audience_reaction_present"])
    }
    output["v09_audience_label_changed"] = output["clip_id"].isin(changed_ids).astype("int8")
    if "v09_label_changed" in output.columns:
        output.loc[output["clip_id"].isin(changed_ids), "v09_label_changed"] = 1
    output = recompute(output)
    output.to_csv(output_path, index=False, encoding="utf-8")
    review.to_csv(review_path, index=False, encoding="utf-8")
    review.assign(
        audience_label_changed=(review["current_audience_reaction_present"] != review["corrected_audience_reaction_present"]).astype("int8")
    ).to_csv(ledger_path, index=False, encoding="utf-8")
    write_json(summary_path, {
        "generated_utc": utc_now(),
        "review_rows": len(review),
        "changed_labels": len(changed_ids),
        "audience_positive_after": int(output["v09_audience_reaction_present"].sum()),
        "silence_positive_after": int(output["v09_silence_present"].sum()),
        "zero_active_rows_after_review": int(output["v09_zero_active_after_review"].sum()),
        "output_manifest": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_manifest_modified": False,
    })
    print("[COMPLETE] Audience corrections applied to a new manifest.")
    print(f"Output: {output_path}")
    print(f"Changed labels: {len(changed_ids)}")
    return 0


def add_new_silence(args: argparse.Namespace) -> int:
    p = paths(args.repo_root)
    source = p["metadata"] / "tata_seed_parent_manifest_v09_RARE_EVENTS_CORRECTED.csv"
    output_path = p["metadata"] / "tata_seed_parent_manifest_v09_FINAL_REVIEWED.csv"
    inventory_path = p["ledgers"] / "v09_new_verified_silence_inventory.csv"
    summary_path = p["ledgers"] / "v09_new_verified_silence_add_summary.json"
    require_file(source, "Rare-event-corrected manifest")
    require_dir(args.new_silence_root, "New verified silence directory")
    for item in [output_path, inventory_path, summary_path]:
        ensure_new(item, args.overwrite)
    parents = pd.read_csv(source, low_memory=False)
    require_columns(parents, ["clip_id", "file_name", "v09_split"] + [f"v09_{label}" for label in LABELS], "Parent manifest")
    audio_files = sorted(
        [path.resolve() for path in args.new_silence_root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda path: str(path).lower(),
    )
    if len(audio_files) != args.expected_new_rows:
        raise ValueError(f"Expected {args.expected_new_rows} new silence files, found {len(audio_files)}.")
    existing_names = set(parents["file_name"].astype(str).str.lower())
    duplicates = [path.name for path in audio_files if path.name.lower() in existing_names]
    if duplicates:
        raise ValueError(f"New files duplicate existing manifest names: {duplicates[:10]}")
    now = utc_now()
    rows, inventory = [], []
    for number, audio_path in enumerate(audio_files, start=1):
        digest = sha256_file(audio_path)
        clip_id = f"v09_verified_silence_{number:03d}_{digest[:10]}"
        row = {column: "" for column in parents.columns}
        assignments = {
            "clip_id": clip_id, "sample_id": clip_id, "file_name": audio_path.name,
            "source_file": audio_path.name, "file_path": str(audio_path), "abs_path": str(audio_path),
            "source_group": "events", "source_subfolder": "new_verified_rare_event_audio/silence",
            "primary_label": "silence", "label_group": "events", "review_status": "reviewed",
            "exclude_from_tata_training": 0, "needs_manual_check": 0, "split": "train", "v09_split": "train",
            "v09_review_status": "reviewed", "v09_review_action": "keep",
            "v09_review_notes": "new verified pure-silence augmentation",
            "v09_reviewer": args.reviewer, "v09_reviewed_utc": now,
            "v09_label_changed": 0, "v09_keep_for_tata_training": 1,
            "v09_source_version": "v0.9_new_verified_silence",
            "v09_manifest_role": "tata_triage_seed_parent",
            "v09_zero_active_after_review": 0,
        }
        for column, value in assignments.items():
            if column in row:
                row[column] = value
        for label in [
            "Brene_Brown", "Eckhart_Tolle", "Eric_Thomas", "Gary_Vee", "Jay_Shetty",
            "Nick_Vujicic", "other_speaker_present", "music_present", *FINE_AUDIENCE,
            "audience_reaction_present", "silence_present",
        ]:
            if label in row:
                row[label] = 1 if label == "silence_present" else 0
        for label in LABELS:
            row[f"v09_{label}"] = 1 if label == "silence_present" else 0
        row["v09_num_active_labels"] = 1
        row["v09_labels"] = "silence_present"
        rows.append(row)
        inventory.append({
            "clip_id": clip_id, "file_name": audio_path.name, "source_audio_path": str(audio_path),
            "sha256": digest, "assigned_split": "train", "labels": "silence_present",
            "reviewer": args.reviewer, "reviewed_utc": now,
        })
    combined = pd.concat([parents, pd.DataFrame(rows, columns=parents.columns)], ignore_index=True)
    combined = recompute(combined)
    if combined["clip_id"].astype(str).duplicated().any() or combined["file_name"].astype(str).str.lower().duplicated().any():
        raise ValueError("Duplicate clip IDs or file names exist after adding new silence clips.")
    combined.to_csv(output_path, index=False, encoding="utf-8")
    pd.DataFrame(inventory).to_csv(inventory_path, index=False, encoding="utf-8")
    write_json(summary_path, {
        "generated_utc": now,
        "input_parent_rows": len(parents),
        "new_verified_silence_rows": len(rows),
        "final_parent_rows": len(combined),
        "new_rows_assigned_split": "train",
        "reason_for_train_only": "Preserve existing validation/test parents and avoid evaluation contamination.",
        "final_silence_positive_parents": int(combined["v09_silence_present"].sum()),
        "final_audience_positive_parents": int(combined["v09_audience_reaction_present"].sum()),
        "zero_active_rows_final": int(combined["v09_zero_active_after_review"].sum()),
        "output_manifest": str(output_path),
        "output_sha256": sha256_file(output_path),
        "audio_files_moved": False,
        "source_manifest_modified": False,
    })
    print("[COMPLETE] New verified silence clips added as train-only augmentation.")
    print(f"New clips: {len(rows)}")
    print(f"Final parent rows: {len(combined)}")
    print(f"Final manifest: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="TATA-LAWYER v0.9 seed-review workflow")
    parser.add_argument("--repo_root", type=Path, default=default_repo)
    parser.add_argument("--reviewer", default="Pankaj Goikar")
    parser.add_argument("--overwrite", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    p1 = commands.add_parser("apply-silence")
    p1.add_argument("--expected_rows", type=int, default=100)
    p1.set_defaults(func=apply_silence)
    p2 = commands.add_parser("prepare-audience")
    p2.add_argument("--dataset_root", type=Path, default=None)
    p2.set_defaults(func=prepare_audience)
    p3 = commands.add_parser("apply-audience")
    p3.set_defaults(func=apply_audience)
    p4 = commands.add_parser("add-new-silence")
    p4.add_argument("--new_silence_root", type=Path, default=None)
    p4.add_argument("--expected_new_rows", type=int, default=27)
    p4.set_defaults(func=add_new_silence)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    p = paths(args.repo_root)
    if hasattr(args, "dataset_root"):
        args.dataset_root = p["dataset"].resolve() if args.dataset_root is None else args.dataset_root.expanduser().resolve()
    if hasattr(args, "new_silence_root"):
        args.new_silence_root = p["new_silence"].resolve() if args.new_silence_root is None else args.new_silence_root.expanduser().resolve()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
