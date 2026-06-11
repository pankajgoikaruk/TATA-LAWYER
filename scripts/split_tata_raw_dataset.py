# scripts/split_tata_raw_dataset.py

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


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


LABELS_V06 = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_score(text: str, seed: int) -> float:
    key = f"{seed}::{text}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def safe_id(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def discover_audio(raw_root: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in AUDIO_EXTS:
            continue

        rel_path = path.relative_to(raw_root)
        parts = rel_path.parts

        if len(parts) >= 2:
            class_dir = parts[0]
        else:
            class_dir = "unknown"

        parent_clip_id = safe_id(f"{class_dir}__{path.stem}")

        rows.append(
            {
                "parent_clip_id": parent_clip_id,
                "source_class_dir": class_dir,
                "source_file": path.name,
                "source_stem": path.stem,
                "source_ext": path.suffix.lower(),
                "source_path": str(path),
                "source_rel_path": str(rel_path),
                "raw_root": str(raw_root),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No audio files found under: {raw_root}")

    # Protect against duplicate IDs.
    if df["parent_clip_id"].duplicated().any():
        counts = {}
        fixed = []
        for val in df["parent_clip_id"].astype(str):
            counts[val] = counts.get(val, 0) + 1
            if counts[val] == 1:
                fixed.append(val)
            else:
                fixed.append(f"{val}__dup{counts[val]:03d}")
        df["parent_clip_id"] = fixed

    return df


def split_df(df: pd.DataFrame, holdout_frac: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["_score"] = df["source_rel_path"].astype(str).apply(lambda x: stable_score(x, seed))

    split_parts = []

    for _, group in df.groupby("source_class_dir", dropna=False):
        group = group.sort_values("_score").copy()
        n = len(group)

        k_holdout = int(round(n * holdout_frac))
        if n > 0 and holdout_frac > 0 and k_holdout == 0:
            k_holdout = 1

        group["raw_split_role"] = "raw_pseudo_pool"
        group.iloc[:k_holdout, group.columns.get_loc("raw_split_role")] = "raw_final_holdout"

        split_parts.append(group)

    out = pd.concat(split_parts, ignore_index=True)
    out = out.drop(columns=["_score"], errors="ignore")
    return out


def make_holdout_template(holdout_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "parent_clip_id",
        "source_class_dir",
        "source_file",
        "source_path",
        "source_rel_path",
        "raw_split_role",
    ]

    template = holdout_df[cols].copy()

    for lab in LABELS_V06:
        template[lab] = ""

    template["manual_labels"] = ""
    template["review_status"] = "pending_final_holdout"
    template["notes"] = ""

    return template


def write_md(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# TATA v0.6 Raw Dataset Split Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append(f"- Raw root: `{summary['raw_root']}`")
    lines.append(f"- Total audio files: `{summary['total_audio_files']}`")
    lines.append(f"- Pseudo pool files: `{summary['pseudo_pool_files']}`")
    lines.append(f"- Final holdout files: `{summary['final_holdout_files']}`")
    lines.append(f"- Holdout fraction: `{summary['holdout_frac']}`")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append("")
    lines.append("## Class-wise counts")
    lines.append("")
    lines.append("| Class | Total | Pseudo pool | Final holdout |")
    lines.append("|---|---:|---:|---:|")

    for row in summary["class_counts"]:
        lines.append(
            f"| `{row['source_class_dir']}` | {row['total']} | "
            f"{row['raw_pseudo_pool']} | {row['raw_final_holdout']} |"
        )

    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Important rule")
    lines.append("")
    lines.append("The final holdout set must not be used for TATA routing, main-model training, threshold tuning, or pseudo-label expansion. It must be manually labelled and reserved only for final testing.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split raw human_talk_dataset into pseudo pool and final holdout.")

    parser.add_argument("--raw_root", default="human_talk_dataset")
    parser.add_argument("--out_root", default="human_talk_workspace/tata_v0.6_raw_pipeline")
    parser.add_argument("--holdout_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    metadata_dir = out_root / "metadata"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = discover_audio(raw_root)
    split = split_df(df, holdout_frac=args.holdout_frac, seed=args.seed)

    pseudo_df = split[split["raw_split_role"] == "raw_pseudo_pool"].copy()
    holdout_df = split[split["raw_split_role"] == "raw_final_holdout"].copy()

    holdout_template = make_holdout_template(holdout_df)

    out_all = metadata_dir / "raw_parent_manifest_all.csv"
    out_pseudo = metadata_dir / "raw_pseudo_pool_parent_manifest.csv"
    out_holdout = metadata_dir / "raw_final_holdout_parent_manifest.csv"
    out_template = metadata_dir / "raw_final_holdout_MANUAL_LABEL_TEMPLATE.csv"
    out_summary_json = metadata_dir / "raw_split_summary.json"
    out_summary_md = metadata_dir / "raw_split_summary.md"

    split.to_csv(out_all, index=False)
    pseudo_df.to_csv(out_pseudo, index=False)
    holdout_df.to_csv(out_holdout, index=False)
    holdout_template.to_csv(out_template, index=False)

    class_counts = []
    for class_name, group in split.groupby("source_class_dir"):
        class_counts.append(
            {
                "source_class_dir": class_name,
                "total": int(len(group)),
                "raw_pseudo_pool": int((group["raw_split_role"] == "raw_pseudo_pool").sum()),
                "raw_final_holdout": int((group["raw_split_role"] == "raw_final_holdout").sum()),
            }
        )

    summary = {
        "generated_at": now_iso(),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "holdout_frac": float(args.holdout_frac),
        "seed": int(args.seed),
        "total_audio_files": int(len(split)),
        "pseudo_pool_files": int(len(pseudo_df)),
        "final_holdout_files": int(len(holdout_df)),
        "class_counts": class_counts,
        "outputs": {
            "all_parent_manifest": str(out_all),
            "pseudo_pool_parent_manifest": str(out_pseudo),
            "final_holdout_parent_manifest": str(out_holdout),
            "final_holdout_manual_label_template": str(out_template),
            "summary_json": str(out_summary_json),
            "summary_md": str(out_summary_md),
        },
    }

    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(out_summary_md, summary)

    print("")
    print("Raw dataset split complete")
    print("-" * 90)
    print(f"Raw root:          {raw_root}")
    print(f"Output root:       {out_root}")
    print(f"Total audio files: {len(split)}")
    print(f"Pseudo pool:       {len(pseudo_df)}")
    print(f"Final holdout:     {len(holdout_df)}")
    print("")
    print("Class-wise counts:")
    for row in class_counts:
        print(
            f"  {row['source_class_dir']:24s} "
            f"total={row['total']:5d} "
            f"pseudo={row['raw_pseudo_pool']:5d} "
            f"holdout={row['raw_final_holdout']:4d}"
        )
    print("")
    print(f"Pseudo pool manifest:   {out_pseudo}")
    print(f"Holdout template:       {out_template}")
    print(f"Summary:                {out_summary_md}")


if __name__ == "__main__":
    main()