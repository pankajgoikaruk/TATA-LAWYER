# scripts/prepare_v08_delta_from_v06_trusted_base.py
#
# Prepare a SMALL manual-review queue for:
#   v0.8-human-corrected-balanced
#
# Strategy:
#   v0.6 trusted base
#   + only LAWYER delta/new suggestions
#   + raw non-target context correction
#   + holdout non-target context correction
#
# This avoids re-reviewing all v0.6 labels.
#
# It creates:
#   human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/manual_review_queue/
#
# It does NOT modify existing v0.6/v0.8 files.
#
# Example:
#   python scripts\prepare_v08_delta_from_v06_trusted_base.py ^
#     --v06_root human_talk_workspace\tata_v0.6_raw_pipeline ^
#     --v08_root human_talk_workspace\tata_v0.8_raw_pipeline ^
#     --out_root human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline ^
#     --labels_json configs\human_talk_10label_schema.json ^
#     --lawyer_config configs\lawyer_v08_human_talk.json

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

DEFAULT_EVENT_LABELS = [
    "music_present",
    "audience_reaction_present",
    "silence_present",
]

DEFAULT_NON_TARGET_CLASSES = [
    "Les_Brown",
    "Mel_Robbins",
    "Oprah_Winfrey",
    "Rabin_Sharma",
    "Simon_Sinek",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(labels_json: Path) -> list[str]:
    payload = load_json(labels_json)
    labels = payload["labels"] if isinstance(payload, dict) else payload
    labels = [str(x) for x in labels]
    if not labels:
        raise RuntimeError(f"No labels found in {labels_json}")
    return labels


def load_lawyer_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def get_label_groups(labels: list[str], lawyer_config: dict[str, Any]):
    groups = lawyer_config.get("label_groups", {}) if isinstance(lawyer_config, dict) else {}
    source_matching = lawyer_config.get("source_matching", {}) if isinstance(lawyer_config, dict) else {}

    target_labels = [str(x) for x in groups.get("target_labels", DEFAULT_TARGET_LABELS)]
    event_labels = [str(x) for x in groups.get("event_labels", DEFAULT_EVENT_LABELS)]
    open_set_label = str(groups.get("open_set_label", "other_speaker_present"))
    non_target_classes = [str(x) for x in source_matching.get("known_non_target_source_classes", DEFAULT_NON_TARGET_CLASSES)]

    target_labels = [x for x in target_labels if x in labels]
    event_labels = [x for x in event_labels if x in labels]

    missing = [x for x in target_labels + event_labels + [open_set_label] if x not in labels]
    if missing:
        raise RuntimeError(f"Configured labels missing from label schema: {missing}")

    return target_labels, open_set_label, event_labels, non_target_classes


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file, using empty dataframe: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def ensure_label_cols(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        if lab not in df.columns:
            df[lab] = 0
        df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int)
    return df


def refresh_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = ensure_label_cols(df, labels)
    df["manual_labels"] = df.apply(
        lambda r: "|".join([lab for lab in labels if int(r.get(lab, 0)) == 1]),
        axis=1,
    )
    df["labels"] = df["manual_labels"]
    df["num_active_labels"] = df[labels].sum(axis=1).astype(int)
    return df


def source_text(df: pd.DataFrame) -> pd.Series:
    cols = ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].astype(str).agg(" ".join, axis=1)


def non_target_mask(df: pd.DataFrame, non_target_classes: list[str]) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=bool)
    text = source_text(df)
    mask = pd.Series([False] * len(df), index=df.index)
    for cls in non_target_classes:
        mask = mask | text.str.contains(cls, regex=False, na=False)
    return mask


def force_known_nontarget_identity(
    df: pd.DataFrame,
    *,
    labels: list[str],
    target_labels: list[str],
    open_set_label: str,
    non_target_classes: list[str],
) -> pd.DataFrame:
    df = ensure_label_cols(df, labels)
    mask = non_target_mask(df, non_target_classes)
    if len(df) and mask.any():
        for lab in target_labels:
            df.loc[mask, lab] = 0
        df.loc[mask, open_set_label] = 1
    return refresh_labels(df, labels)


def add_review_columns(
    df: pd.DataFrame,
    *,
    review_file_type: str,
    review_priority: str,
    review_task: str,
    review_instruction: str,
    dataset_role: str,
) -> pd.DataFrame:
    df = df.copy()

    insert_cols = [
        ("review_file_type", review_file_type),
        ("review_priority", review_priority),
        ("review_task", review_task),
        ("review_instruction", review_instruction),
        ("dataset_role", dataset_role),
        ("human_review_status", "pending"),
        ("human_review_notes", ""),
    ]

    for idx, (col, val) in enumerate(insert_cols):
        if col in df.columns:
            df[col] = val
        else:
            df.insert(idx, col, val)

    return df


def label_vector_str(row: pd.Series, labels: list[str]) -> str:
    return "|".join(str(int(row.get(lab, 0))) for lab in labels)


def active_text_from_prefixed(row: pd.Series, labels: list[str], prefix: str) -> str:
    return "|".join([lab for lab in labels if int(row.get(prefix + lab, 0)) == 1])


def make_trusted_base(v06_root: Path, labels: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    paths = {
        "v06_hybrid_accepted": v06_root / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted.csv",
        "v06_hybrid_accepted_with_warning": v06_root / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted_with_warning.csv",
        "v06_hybrid_needs_review_corrected": v06_root / "manual_review_queue" / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv",
    }

    dfs = []
    counts = {}

    for name, path in paths.items():
        df = read_csv_or_empty(path)
        counts[name] = int(len(df))

        if len(df) == 0:
            continue

        df = refresh_labels(df, labels)
        df["v06_base_source"] = name
        df["v06_base_path"] = str(path)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(), counts

    base = pd.concat(dfs, ignore_index=True, sort=False)

    if "parent_clip_id" not in base.columns:
        raise RuntimeError("Trusted base files must contain parent_clip_id")

    base["parent_clip_id"] = base["parent_clip_id"].astype(str)

    # If a clip appears more than once, prefer manually corrected needs_review > accepted_with_warning > accepted.
    priority_map = {
        "v06_hybrid_needs_review_corrected": 0,
        "v06_hybrid_accepted_with_warning": 1,
        "v06_hybrid_accepted": 2,
    }
    base["_v06_priority"] = base["v06_base_source"].map(priority_map).fillna(99).astype(int)
    base = base.sort_values(["parent_clip_id", "_v06_priority"]).drop_duplicates("parent_clip_id", keep="first")
    base = base.drop(columns=["_v06_priority"], errors="ignore")
    base = refresh_labels(base, labels)

    return base, counts


def add_prefixed_labels(df: pd.DataFrame, labels: list[str], prefix: str) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        if lab in df.columns:
            df[prefix + lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int)
        else:
            df[prefix + lab] = 0
    df[prefix + "labels_text"] = df.apply(lambda r: active_text_from_prefixed(r, labels, prefix), axis=1)
    return df


def choose_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def merge_lawyer_with_v06(lawyer: pd.DataFrame, v06_base: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    lawyer = lawyer.copy()
    v06_base = v06_base.copy()

    lawyer["parent_clip_id"] = lawyer["parent_clip_id"].astype(str)
    v06_base["parent_clip_id"] = v06_base["parent_clip_id"].astype(str)

    lawyer = refresh_labels(lawyer, labels)
    v06_base = refresh_labels(v06_base, labels)

    lawyer = add_prefixed_labels(lawyer, labels, "lawyer_")
    v06_base = add_prefixed_labels(v06_base, labels, "v06_")

    # Keep useful v0.6 columns only to avoid duplicate chaos.
    v06_keep = [
        "parent_clip_id",
        "v06_base_source",
        "v06_base_path",
        "v06_labels_text",
        *["v06_" + lab for lab in labels],
    ]

    merged = lawyer.merge(v06_base[choose_existing_cols(v06_base, v06_keep)], on="parent_clip_id", how="left")

    for lab in labels:
        if "v06_" + lab not in merged.columns:
            merged["v06_" + lab] = pd.NA

    merged["exists_in_v06_trusted_base"] = merged["v06_base_source"].notna().astype(int)
    merged["lawyer_labels_text"] = merged.apply(lambda r: active_text_from_prefixed(r, labels, "lawyer_"), axis=1)
    merged["v06_labels_text"] = merged["v06_labels_text"].fillna("")

    changed = []
    for _, row in merged.iterrows():
        if int(row["exists_in_v06_trusted_base"]) == 0:
            changed.append(False)
            continue
        diff = False
        for lab in labels:
            lv = int(row.get("lawyer_" + lab, 0))
            vv = int(row.get("v06_" + lab, 0))
            if lv != vv:
                diff = True
                break
        changed.append(diff)

    merged["lawyer_changed_v06_labels"] = pd.Series(changed, index=merged.index).astype(int)
    return merged


def prefill_from_v06_or_lawyer(df: pd.DataFrame, labels: list[str], prefer: str) -> pd.DataFrame:
    df = df.copy()
    if prefer not in {"v06", "lawyer"}:
        raise ValueError("prefer must be 'v06' or 'lawyer'")

    for lab in labels:
        src_col = f"{prefer}_{lab}"
        if src_col in df.columns:
            df[lab] = pd.to_numeric(df[src_col], errors="coerce").fillna(0).astype(int)
        elif lab not in df.columns:
            df[lab] = 0
    return refresh_labels(df, labels)


def add_delta_summary_cols(df: pd.DataFrame, labels: list[str], event_labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    changed_labs = []

    for _, row in df.iterrows():
        labs = []
        for lab in labels:
            if pd.notna(row.get("v06_" + lab, pd.NA)):
                if int(row.get("v06_" + lab, 0)) != int(row.get("lawyer_" + lab, 0)):
                    labs.append(lab)
        changed_labs.append("|".join(labs))

    df["changed_label_names"] = changed_labs
    df["changed_event_label_names"] = df["changed_label_names"].apply(
        lambda s: "|".join([lab for lab in str(s).split("|") if lab in event_labels])
    )
    return df


def save_csv(df: pd.DataFrame, path: Path, labels: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = refresh_labels(df, labels) if len(df) else df
    df.to_csv(path, index=False)

    label_counts = {}
    for lab in labels:
        if lab in df.columns:
            label_counts[lab] = int(pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).sum())

    return {
        "path": str(path),
        "rows": int(len(df)),
        "label_counts": label_counts,
    }


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    target_labels = summary["target_labels"]
    open_set_label = summary["open_set_label"]
    event_labels = summary["event_labels"]
    non_target_classes = summary["known_non_target_source_classes"]

    txt = f"""# v0.8-human-corrected-balanced Delta Review Queue

Generated: `{summary["generated_at"]}`

This folder uses **v0.6 as trusted base** and creates only smaller delta/context review files.

## Do not edit frozen copies

Files starting with `00_` are reference/copy files.

## Known non-target speakers

```text
{chr(10).join(non_target_classes)}
```

For these rows, keep:

```text
{chr(10).join([lab + " = 0" for lab in target_labels])}
{open_set_label} = 1
```

Only manually review/edit:

```text
{chr(10).join(event_labels)}
```

## Main files to edit

### `03_raw_nontarget_context_REVIEW.csv`

These come from the v0.6 trusted manually corrected hybrid raw file.

You already reviewed speaker identity. Now check only:

```text
{chr(10).join(event_labels)}
```

### `04_holdout_nontarget_context_REVIEW.csv`

These are holdout rows. Keep separate. Do not use them for training.

Only check:

```text
{chr(10).join(event_labels)}
```

### `05_lawyer_delta_changed_labels_REVIEW.csv`

These are rows already present in v0.6 trusted base, but LAWYER suggests different labels.

Default editable label columns are prefilled from **v0.6**, because v0.6 is trusted.

Use helper columns:

```text
v06_<label>
lawyer_<label>
changed_label_names
changed_event_label_names
```

Only change the editable label columns if you agree with LAWYER after listening.

### `06_lawyer_new_samples_REVIEW.csv`

These are LAWYER accepted/warning rows not found in v0.6 trusted base.

Default editable label columns are prefilled from **LAWYER**.

Review before using for training.

### `07_training_delta_master_REVIEW.csv`

This combines training-side review candidates only. It excludes holdout rows.

## After editing

For every edited row, set:

```text
human_review_status = reviewed
```

Optionally fill:

```text
human_review_notes
```

The script that rebuilds the final corrected dataset can refresh:

```text
labels
manual_labels
num_active_labels
```

if you forget, but it is better to keep them consistent.

## Summary

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v0.8 delta review queue from trusted v0.6 base.")
    parser.add_argument("--v06_root", default="human_talk_workspace/tata_v0.6_raw_pipeline")
    parser.add_argument("--v08_root", default="human_talk_workspace/tata_v0.8_raw_pipeline")
    parser.add_argument("--out_root", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline")
    parser.add_argument("--labels_json", default="configs/human_talk_10label_schema.json")
    parser.add_argument("--lawyer_config", default="configs/lawyer_v08_human_talk.json")
    parser.add_argument("--lawyer_dir", default="")
    parser.add_argument("--archive_existing_queue", action="store_true")
    args = parser.parse_args()

    v06_root = Path(args.v06_root)
    v08_root = Path(args.v08_root)
    out_root = Path(args.out_root)
    labels_json = Path(args.labels_json)
    lawyer_config_path = Path(args.lawyer_config) if args.lawyer_config else None
    lawyer_dir = Path(args.lawyer_dir) if args.lawyer_dir else v08_root / "raw_tata_pseudo_routing" / "lawyer_v08"

    labels = load_labels(labels_json)
    lawyer_config = load_lawyer_config(lawyer_config_path)
    target_labels, open_set_label, event_labels, non_target_classes = get_label_groups(labels, lawyer_config)

    manual_dir = out_root / "manual_review_queue"
    if args.archive_existing_queue and manual_dir.exists():
        archive_dir = out_root / f"manual_review_queue_ARCHIVED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        manual_dir.rename(archive_dir)
        print(f"[INFO] Archived existing queue to: {archive_dir}")

    manual_dir.mkdir(parents=True, exist_ok=True)

    # Input paths.
    raw_corrected_path = v06_root / "manual_review_queue" / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv"
    holdout_path = v06_root / "manual_review_queue" / "01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv"
    lawyer_all_path = lawyer_dir / "lawyer_v08_parent_labels_all.csv"

    # Frozen copies.
    shutil.copy2(raw_corrected_path, manual_dir / "00_v06_raw_hybrid_corrected_TRUSTED_COPY_DO_NOT_EDIT.csv")
    shutil.copy2(holdout_path, manual_dir / "00_v06_holdout_TRUSTED_COPY_DO_NOT_EDIT.csv")
    if lawyer_all_path.exists():
        shutil.copy2(lawyer_all_path, manual_dir / "00_lawyer_v08_parent_labels_all_AUTO_COPY_DO_NOT_EDIT.csv")

    v06_base, v06_counts = make_trusted_base(v06_root, labels)
    lawyer_all = read_csv_or_empty(lawyer_all_path)
    lawyer_all = refresh_labels(lawyer_all, labels) if len(lawyer_all) else lawyer_all

    raw_corrected = refresh_labels(read_csv_or_empty(raw_corrected_path), labels)
    holdout = refresh_labels(read_csv_or_empty(holdout_path), labels)

    outputs = {}

    # 01 trusted base index for transparency.
    outputs["01_v06_trusted_base_index.csv"] = save_csv(
        v06_base,
        manual_dir / "01_v06_trusted_base_index.csv",
        labels,
    )

    # 03 raw non-target context review from v0.6 trusted raw corrected file.
    raw_nt = raw_corrected[non_target_mask(raw_corrected, non_target_classes)].copy()
    raw_nt = force_known_nontarget_identity(
        raw_nt,
        labels=labels,
        target_labels=target_labels,
        open_set_label=open_set_label,
        non_target_classes=non_target_classes,
    )
    raw_nt = add_review_columns(
        raw_nt,
        review_file_type="raw_nontarget_context_review",
        review_priority="P1",
        review_task="check_background_event_labels_only",
        review_instruction=f"Keep target speakers=0 and {open_set_label}=1. Edit only: {', '.join(event_labels)}.",
        dataset_role="training_candidate",
    )
    outputs["03_raw_nontarget_context_REVIEW.csv"] = save_csv(
        raw_nt,
        manual_dir / "03_raw_nontarget_context_REVIEW.csv",
        labels,
    )

    # 04 holdout non-target context review.
    holdout_nt = holdout[non_target_mask(holdout, non_target_classes)].copy()
    holdout_nt = force_known_nontarget_identity(
        holdout_nt,
        labels=labels,
        target_labels=target_labels,
        open_set_label=open_set_label,
        non_target_classes=non_target_classes,
    )
    holdout_nt = add_review_columns(
        holdout_nt,
        review_file_type="holdout_nontarget_context_review",
        review_priority="P1_HOLDOUT",
        review_task="correct_holdout_background_event_ground_truth",
        review_instruction=f"HOLDOUT ONLY. Keep target speakers=0 and {open_set_label}=1. Edit only: {', '.join(event_labels)}.",
        dataset_role="holdout_ground_truth",
    )
    outputs["04_holdout_nontarget_context_REVIEW.csv"] = save_csv(
        holdout_nt,
        manual_dir / "04_holdout_nontarget_context_REVIEW.csv",
        labels,
    )

    if len(lawyer_all) and len(v06_base):
        merged = merge_lawyer_with_v06(lawyer_all, v06_base, labels)
        merged = add_delta_summary_cols(merged, labels, event_labels)

        # 05 rows present in v0.6 but LAWYER suggests changed labels.
        changed = merged[(merged["exists_in_v06_trusted_base"] == 1) & (merged["lawyer_changed_v06_labels"] == 1)].copy()
        changed = prefill_from_v06_or_lawyer(changed, labels, prefer="v06")
        changed = add_review_columns(
            changed,
            review_file_type="lawyer_delta_changed_labels",
            review_priority="P2",
            review_task="review_lawyer_changed_label_suggestion",
            review_instruction="Editable labels are prefilled from trusted v0.6. Compare v06_* vs lawyer_* columns and change only if LAWYER is correct.",
            dataset_role="training_candidate",
        )
        outputs["05_lawyer_delta_changed_labels_REVIEW.csv"] = save_csv(
            changed,
            manual_dir / "05_lawyer_delta_changed_labels_REVIEW.csv",
            labels,
        )

        # 06 new LAWYER accepted/warning samples not present in v0.6 trusted base.
        if "routing_decision" in merged.columns:
            accepted_mask = merged["routing_decision"].astype(str).isin(["accepted", "accepted_with_warning"])
        else:
            accepted_mask = pd.Series([True] * len(merged), index=merged.index)

        new_samples = merged[(merged["exists_in_v06_trusted_base"] == 0) & accepted_mask].copy()
        new_samples = prefill_from_v06_or_lawyer(new_samples, labels, prefer="lawyer")
        new_samples = add_review_columns(
            new_samples,
            review_file_type="lawyer_new_samples",
            review_priority="P3",
            review_task="review_new_lawyer_accepted_or_warning_sample",
            review_instruction="This sample was not in the v0.6 trusted base. Editable labels are prefilled from LAWYER; review before training.",
            dataset_role="training_candidate",
        )
        outputs["06_lawyer_new_samples_REVIEW.csv"] = save_csv(
            new_samples,
            manual_dir / "06_lawyer_new_samples_REVIEW.csv",
            labels,
        )

        training_master = pd.concat([raw_nt, changed, new_samples], ignore_index=True, sort=False)
    else:
        changed = pd.DataFrame()
        new_samples = pd.DataFrame()
        outputs["05_lawyer_delta_changed_labels_REVIEW.csv"] = save_csv(changed, manual_dir / "05_lawyer_delta_changed_labels_REVIEW.csv", labels)
        outputs["06_lawyer_new_samples_REVIEW.csv"] = save_csv(new_samples, manual_dir / "06_lawyer_new_samples_REVIEW.csv", labels)
        training_master = raw_nt.copy()

    # Deduplicate master by parent ID, preserving P1 raw non-target first, then changed, then new.
    if len(training_master):
        priority_order = {
            "raw_nontarget_context_review": 0,
            "lawyer_delta_changed_labels": 1,
            "lawyer_new_samples": 2,
        }
        training_master["_priority_sort"] = training_master["review_file_type"].map(priority_order).fillna(99).astype(int)
        training_master = training_master.sort_values(["parent_clip_id", "_priority_sort"])
        training_master = training_master.drop_duplicates("parent_clip_id", keep="first")
        training_master = training_master.drop(columns=["_priority_sort"], errors="ignore")
        training_master = refresh_labels(training_master, labels)

    outputs["07_training_delta_master_REVIEW.csv"] = save_csv(
        training_master,
        manual_dir / "07_training_delta_master_REVIEW.csv",
        labels,
    )

    summary = {
        "generated_at": now_iso(),
        "experiment": "v0.8-human-corrected-balanced",
        "strategy": "v0.6 trusted base + LAWYER delta/new suggestions + non-target context corrections",
        "v06_root": str(v06_root),
        "v08_root": str(v08_root),
        "out_root": str(out_root),
        "manual_review_queue": str(manual_dir),
        "labels_json": str(labels_json),
        "lawyer_config": str(lawyer_config_path) if lawyer_config_path else None,
        "lawyer_dir": str(lawyer_dir),
        "labels": labels,
        "target_labels": target_labels,
        "open_set_label": open_set_label,
        "event_labels": event_labels,
        "known_non_target_source_classes": non_target_classes,
        "v06_base_source_counts_before_dedup": v06_counts,
        "v06_trusted_base_unique_parent_rows": int(len(v06_base)),
        "lawyer_parent_rows": int(len(lawyer_all)),
        "outputs": outputs,
        "manual_review_rule": {
            "known_non_target_identity": {
                "target_labels": "force/keep 0",
                open_set_label: "force/keep 1",
                "edit_only": event_labels,
            },
            "changed_delta_rows": "prefilled from trusted v0.6; compare LAWYER helper columns before changing",
            "new_lawyer_rows": "prefilled from LAWYER; review before training",
            "holdout_rows": "correct separately and never mix into training",
        },
    }

    summary_path = manual_dir / "00_review_queue_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(manual_dir / "00_README_DELTA_REVIEW_INSTRUCTIONS.md", summary)

    print("")
    print("v0.8 delta review queue prepared")
    print("-" * 90)
    print(f"Manual review queue: {manual_dir}")
    print("")
    for name, info in outputs.items():
        print(f"{name}: rows={info['rows']}")
    print("")
    print(f"Summary:      {summary_path}")
    print(f"Instructions: {manual_dir / '00_README_DELTA_REVIEW_INSTRUCTIONS.md'}")


if __name__ == "__main__":
    main()
