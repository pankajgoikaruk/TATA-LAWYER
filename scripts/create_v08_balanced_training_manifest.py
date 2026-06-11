# scripts/create_v08_balanced_training_manifest.py
#
# Create a balanced training manifest for:
#   v0.8-human-corrected-balanced
#
# This script does NOT change labels.
# It only down-samples over-large "background-heavy" rows, mainly:
#   other_speaker_present-only
#   other_speaker_present + music_present only
#   music_present-only
#
# It keeps all rows that contain:
#   - any target speaker label
#   - audience_reaction_present
#   - silence_present
#   - clean seed rows, if is_clean_seed exists
#
# Default goal:
#   cap other_speaker_present total at about 3x median target-speaker count,
#   but never remove protected rows.
#
# Example:
#   python scripts\create_v08_balanced_training_manifest.py ^
#     --input_manifest human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset\metadata\multilabel_features_manifest.csv ^
#     --out_root human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced ^
#     --labels_json configs\human_talk_10label_schema.json ^
#     --lawyer_config configs\lawyer_v08_human_talk.json ^
#     --other_cap_multiplier 3.0 ^
#     --seed 42

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: Path) -> list[str]:
    payload = load_json(path)
    labels = payload["labels"] if isinstance(payload, dict) else payload
    labels = [str(x) for x in labels]
    if not labels:
        raise RuntimeError("No labels found.")
    return labels


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def get_label_groups(labels: list[str], config: dict[str, Any]) -> tuple[list[str], str, list[str]]:
    groups = config.get("label_groups", {}) if isinstance(config, dict) else {}

    target_labels = [str(x) for x in groups.get("target_labels", DEFAULT_TARGET_LABELS)]
    event_labels = [str(x) for x in groups.get("event_labels", DEFAULT_EVENT_LABELS)]
    open_set_label = str(groups.get("open_set_label", "other_speaker_present"))

    target_labels = [x for x in target_labels if x in labels]
    event_labels = [x for x in event_labels if x in labels]

    if open_set_label not in labels:
        raise RuntimeError(f"open_set_label not in labels: {open_set_label}")

    return target_labels, open_set_label, event_labels


def ensure_label_cols(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        if lab not in df.columns:
            df[lab] = 0
        df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def refresh_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    df = ensure_label_cols(df, labels)
    df["labels"] = df.apply(
        lambda r: "|".join([lab for lab in labels if int(r.get(lab, 0)) == 1]),
        axis=1,
    )
    if "manual_labels" in df.columns:
        df["manual_labels"] = df["labels"]
    df["num_active_labels"] = df[labels].sum(axis=1).astype(int)
    return df


def label_counts(df: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    return {
        lab: int(pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).sum())
        for lab in labels
        if lab in df.columns
    }


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    if "split" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}


def group_counts(df: pd.DataFrame) -> dict[str, int]:
    if "training_group" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["training_group"].value_counts().sort_index().items()}


def make_combo_key(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    return df[labels].astype(int).astype(str).agg("".join, axis=1)


def sample_eligible_rows(
    eligible: pd.DataFrame,
    n_keep: int,
    labels: list[str],
    seed: int,
) -> pd.DataFrame:
    if n_keep <= 0 or len(eligible) == 0:
        return eligible.iloc[0:0].copy()

    if n_keep >= len(eligible):
        return eligible.copy()

    eligible = eligible.copy()
    eligible["_balance_combo"] = make_combo_key(eligible, labels)

    # Stratify by split and label-combo if possible.
    strat_cols = []
    if "split" in eligible.columns:
        strat_cols.append("split")
    strat_cols.append("_balance_combo")

    groups = list(eligible.groupby(strat_cols, dropna=False))
    total = len(eligible)

    sampled_parts = []
    remaining_need = n_keep
    remainders = []

    for key, g in groups:
        exact = n_keep * len(g) / total
        base_n = int(np.floor(exact))
        base_n = min(base_n, len(g))
        if base_n > 0:
            sampled_parts.append(g.sample(n=base_n, random_state=seed))
        remaining_need -= base_n
        remainders.append((exact - base_n, key, g, base_n))

    if remaining_need > 0:
        # Allocate leftover to groups with largest fractional remainder.
        remainders = sorted(remainders, key=lambda x: x[0], reverse=True)
        for _, key, g, already in remainders:
            if remaining_need <= 0:
                break
            available = len(g) - already
            if available <= 0:
                continue

            take = min(available, remaining_need)
            already_idx = set()
            if sampled_parts:
                # Slow but okay for this dataset.
                already_sampled = pd.concat(sampled_parts, ignore_index=False)
                already_idx = set(already_sampled.index)

            candidates = g.loc[[idx for idx in g.index if idx not in already_idx]]
            sampled_parts.append(candidates.sample(n=take, random_state=seed + remaining_need))
            remaining_need -= take

    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else eligible.iloc[0:0].copy()
    sampled = sampled.drop(columns=["_balance_combo"], errors="ignore")

    # Safety if tiny rounding mismatch.
    if len(sampled) > n_keep:
        sampled = sampled.sample(n=n_keep, random_state=seed)

    return sampled


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v0.8 Human-Corrected-Balanced Manifest Balancing Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Row counts",
        "",
        f"- Input rows: `{summary['input_rows']}`",
        f"- Protected rows kept: `{summary['protected_rows_kept']}`",
        f"- Eligible heavy/background rows before: `{summary['eligible_heavy_rows_before']}`",
        f"- Eligible heavy/background rows kept: `{summary['eligible_heavy_rows_kept']}`",
        f"- Dropped rows: `{summary['dropped_rows']}`",
        f"- Balanced rows: `{summary['balanced_rows']}`",
        "",
        "## Label counts before/after",
        "",
        "| Label | Before | After |",
        "|---|---:|---:|",
    ]

    for lab in summary["labels"]:
        before = summary["label_counts_before"].get(lab, 0)
        after = summary["label_counts_after"].get(lab, 0)
        lines.append(f"| `{lab}` | {before} | {after} |")

    lines += [
        "",
        "## Split counts before/after",
        "",
        "| Split | Before | After |",
        "|---|---:|---:|",
    ]

    split_keys = sorted(set(summary["split_counts_before"]) | set(summary["split_counts_after"]))
    for key in split_keys:
        lines.append(f"| `{key}` | {summary['split_counts_before'].get(key, 0)} | {summary['split_counts_after'].get(key, 0)} |")

    lines += [
        "",
        "## Training group counts before/after",
        "",
        "| Group | Before | After |",
        "|---|---:|---:|",
    ]

    group_keys = sorted(set(summary["training_group_counts_before"]) | set(summary["training_group_counts_after"]))
    for key in group_keys:
        lines.append(f"| `{key}` | {summary['training_group_counts_before'].get(key, 0)} | {summary['training_group_counts_after'].get(key, 0)} |")

    lines += [
        "",
        "## Outputs",
        "",
        f"- Balanced manifest: `{summary['outputs']['balanced_manifest']}`",
        f"- Dropped rows audit: `{summary['outputs']['dropped_rows_audit']}`",
        f"- Summary JSON: `{summary['outputs']['summary_json']}`",
        "",
        "Training note: use `-FeaturesRoot .` if your original manifest summary says feature paths are relative to the project root.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced v0.8-human-corrected-balanced training manifest.")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--labels_json", default="configs/human_talk_10label_schema.json")
    parser.add_argument("--lawyer_config", default="configs/lawyer_v08_human_talk.json")
    parser.add_argument("--other_cap_multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    out_root = Path(args.out_root)
    labels = load_labels(Path(args.labels_json))
    config = load_config(Path(args.lawyer_config) if args.lawyer_config else None)

    target_labels, open_set_label, event_labels = get_label_groups(labels, config)

    audience_label = "audience_reaction_present" if "audience_reaction_present" in labels else None
    silence_label = "silence_present" if "silence_present" in labels else None
    music_label = "music_present" if "music_present" in labels else None

    df = pd.read_csv(input_manifest, low_memory=False)
    df = refresh_labels(df, labels)

    before_counts = label_counts(df, labels)
    target_counts = [before_counts[lab] for lab in target_labels]
    target_median = int(np.median(target_counts)) if target_counts else 0

    other_cap = int(round(args.other_cap_multiplier * target_median))

    target_any = df[target_labels].sum(axis=1) > 0
    audience_any = df[audience_label] == 1 if audience_label else pd.Series([False] * len(df), index=df.index)
    silence_any = df[silence_label] == 1 if silence_label else pd.Series([False] * len(df), index=df.index)

    if "is_clean_seed" in df.columns:
        clean_seed = df["is_clean_seed"].astype(str).isin(["1", "True", "true"])
    else:
        clean_seed = pd.Series([False] * len(df), index=df.index)

    # Protected rows are always kept.
    protected = target_any | audience_any | silence_any | clean_seed

    # Eligible rows are background-heavy only. These rows have no target/audience/silence.
    eligible = ~protected

    protected_df = df[protected].copy()
    eligible_df = df[eligible].copy()

    protected_other_count = int(protected_df[open_set_label].sum()) if open_set_label in protected_df.columns else 0

    # Number of eligible rows with other_speaker_present needed to reach cap.
    # If protected rows already exceed cap, keep no eligible other rows.
    other_needed = max(0, other_cap - protected_other_count)

    if open_set_label in eligible_df.columns:
        eligible_other = eligible_df[eligible_df[open_set_label] == 1].copy()
        eligible_non_other = eligible_df[eligible_df[open_set_label] != 1].copy()
    else:
        eligible_other = eligible_df.copy()
        eligible_non_other = eligible_df.iloc[0:0].copy()

    sampled_other = sample_eligible_rows(
        eligible_other,
        n_keep=other_needed,
        labels=labels,
        seed=args.seed,
    )

    # Keep a small amount of music-only non-other rows, if any.
    # This avoids deleting all pure music background cases.
    if music_label and len(eligible_non_other):
        music_only_keep = min(len(eligible_non_other), max(100, int(0.25 * target_median)))
        sampled_non_other = sample_eligible_rows(
            eligible_non_other,
            n_keep=music_only_keep,
            labels=labels,
            seed=args.seed + 17,
        )
    else:
        sampled_non_other = eligible_non_other.iloc[0:0].copy()

    balanced = pd.concat([protected_df, sampled_other, sampled_non_other], ignore_index=False)
    balanced = balanced.sort_index().reset_index(drop=True)
    balanced = refresh_labels(balanced, labels)

    kept_index = set(balanced.index)
    # Because reset_index loses original index, build dropped using a stable helper.
    df_with_idx = df.copy()
    df_with_idx["_original_row_index"] = np.arange(len(df_with_idx))
    balanced_source = pd.concat([protected_df, sampled_other, sampled_non_other], ignore_index=False)
    kept_original = set(balanced_source.index)
    dropped = df_with_idx[~df_with_idx["_original_row_index"].isin(kept_original)].copy()

    out_meta = out_root / "metadata"
    out_meta.mkdir(parents=True, exist_ok=True)

    balanced_manifest = out_meta / "multilabel_features_manifest_balanced.csv"
    dropped_audit = out_meta / "dropped_heavy_background_rows_audit.csv"
    summary_json = out_meta / "balanced_manifest_summary.json"
    summary_md = out_meta / "balanced_manifest_summary.md"

    balanced.to_csv(balanced_manifest, index=False)
    dropped.to_csv(dropped_audit, index=False)

    summary = {
        "generated_at": now_iso(),
        "experiment": "v0.8-human-corrected-balanced",
        "input_manifest": str(input_manifest),
        "labels": labels,
        "target_labels": target_labels,
        "open_set_label": open_set_label,
        "event_labels": event_labels,
        "other_cap_multiplier": args.other_cap_multiplier,
        "target_median_count_before": target_median,
        "other_cap_requested": other_cap,
        "protected_other_count": protected_other_count,
        "input_rows": int(len(df)),
        "protected_rows_kept": int(len(protected_df)),
        "eligible_heavy_rows_before": int(len(eligible_df)),
        "eligible_heavy_other_rows_before": int(len(eligible_other)),
        "eligible_heavy_other_rows_kept": int(len(sampled_other)),
        "eligible_non_other_rows_before": int(len(eligible_non_other)),
        "eligible_non_other_rows_kept": int(len(sampled_non_other)),
        "eligible_heavy_rows_kept": int(len(sampled_other) + len(sampled_non_other)),
        "dropped_rows": int(len(dropped)),
        "balanced_rows": int(len(balanced)),
        "label_counts_before": before_counts,
        "label_counts_after": label_counts(balanced, labels),
        "split_counts_before": split_counts(df),
        "split_counts_after": split_counts(balanced),
        "training_group_counts_before": group_counts(df),
        "training_group_counts_after": group_counts(balanced),
        "outputs": {
            "balanced_manifest": str(balanced_manifest),
            "dropped_rows_audit": str(dropped_audit),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
        "protected_rule": "kept all rows with any target speaker OR audience_reaction_present OR silence_present OR is_clean_seed",
        "downsample_rule": "downsampled only unprotected background-heavy rows to reduce other_speaker_present dominance",
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary_md, summary)

    print("")
    print("Balanced manifest created")
    print("-" * 90)
    print(f"Input rows:              {summary['input_rows']}")
    print(f"Protected rows kept:     {summary['protected_rows_kept']}")
    print(f"Eligible heavy rows:     {summary['eligible_heavy_rows_before']}")
    print(f"Eligible heavy kept:     {summary['eligible_heavy_rows_kept']}")
    print(f"Dropped rows:            {summary['dropped_rows']}")
    print(f"Balanced rows:           {summary['balanced_rows']}")
    print("")
    print("Label counts before -> after:")
    for lab in labels:
        print(f"  {lab}: {summary['label_counts_before'].get(lab, 0)} -> {summary['label_counts_after'].get(lab, 0)}")
    print("")
    print(f"Balanced manifest: {balanced_manifest}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Summary MD:        {summary_md}")


if __name__ == "__main__":
    main()
