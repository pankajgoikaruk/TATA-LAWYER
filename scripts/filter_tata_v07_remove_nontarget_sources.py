# scripts/filter_tata_v07_remove_nontarget_sources.py
#
# Reusable manifest filtering script.
#
# This version does NOT hard-code source classes or file mappings.
# It reads them from:
#   --config configs/filter_v07_target_only.json
#
# The script filters CSV manifests only. It does not delete or move audio.

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if "exclude_source_classes" not in config or not isinstance(config["exclude_source_classes"], list):
        raise RuntimeError("Config must contain list: exclude_source_classes")

    if "files" not in config or not isinstance(config["files"], list) or not config["files"]:
        raise RuntimeError("Config must contain non-empty list: files")

    config = dict(config)
    config.setdefault("match_columns", ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"])
    config.setdefault("copy_files", [])
    config.setdefault("summary_name", "filter_summary")
    config.setdefault("note", "Filtered manifests are copied/created only; original audio and source manifests are untouched.")

    for i, item in enumerate(config["files"]):
        if not isinstance(item, dict):
            raise RuntimeError(f"files[{i}] must be an object")
        for key in ["name", "src", "dst"]:
            if key not in item:
                raise RuntimeError(f"files[{i}] missing required key: {key}")

    return config


def resolve_template_path(template: str, src_root: Path, dst_root: Path) -> Path:
    value = str(template)
    value = value.replace("{src_root}", str(src_root))
    value = value.replace("{dst_root}", str(dst_root))
    return Path(value)


def is_excluded_row(row: pd.Series, exclude_classes: list[str], match_columns: list[str]) -> bool:
    text = " ".join(str(row.get(c, "")) for c in match_columns)
    return any(cls in text for cls in exclude_classes)


def filter_csv(
    *,
    src: Path,
    dst: Path,
    name: str,
    exclude_classes: list[str],
    match_columns: list[str],
) -> dict[str, Any]:
    df = pd.read_csv(src, low_memory=False)
    mask = df.apply(
        lambda row: is_excluded_row(row, exclude_classes, match_columns),
        axis=1,
    )

    kept = df[~mask].copy()
    removed = df[mask].copy()

    dst.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(dst, index=False)

    removed_path = dst.with_name(dst.stem + "_REMOVED_ROWS.csv")
    removed.to_csv(removed_path, index=False)

    return {
        "name": name,
        "src": str(src),
        "dst": str(dst),
        "removed_rows_csv": str(removed_path),
        "original_rows": int(len(df)),
        "kept_rows": int(len(kept)),
        "removed_rows": int(len(removed)),
        "status": "ok",
    }


def copy_optional_files(
    *,
    config: dict[str, Any],
    src_root: Path,
    dst_root: Path,
) -> list[dict[str, Any]]:
    outputs = []

    for item in config.get("copy_files", []):
        name = str(item.get("name", "copy_file"))
        src = resolve_template_path(str(item["src"]), src_root, dst_root)
        dst = resolve_template_path(str(item["dst"]), src_root, dst_root)

        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            outputs.append({
                "name": name,
                "src": str(src),
                "dst": str(dst),
                "status": "copied",
            })
        else:
            outputs.append({
                "name": name,
                "src": str(src),
                "dst": str(dst),
                "status": "missing",
            })

    return outputs


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = []
    lines.append(f"# {summary.get('title', 'Manifest Filter Summary')}")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Excluded source classes")
    lines.append("")
    for cls in summary["exclude_source_classes"]:
        lines.append(f"- `{cls}`")

    lines.append("")
    lines.append("## Match columns")
    lines.append("")
    for col in summary["match_columns"]:
        lines.append(f"- `{col}`")

    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| File | Original | Kept | Removed | Status |")
    lines.append("|---|---:|---:|---:|---|")

    for item in summary["outputs"]:
        if item.get("status") != "ok":
            lines.append(f"| `{item['name']}` | - | - | - | `{item.get('status', 'unknown')}` |")
        else:
            lines.append(
                f"| `{item['name']}` | {item['original_rows']} | {item['kept_rows']} | {item['removed_rows']} | `{item['status']}` |"
            )

    if summary.get("copied_files"):
        lines.append("")
        lines.append("## Copied files")
        lines.append("")
        lines.append("| File | Status |")
        lines.append("|---|---|")
        for item in summary["copied_files"]:
            lines.append(f"| `{item['name']}` | `{item['status']}` |")

    lines.append("")
    lines.append(summary.get("note", "Original files were not modified."))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable CSV manifest source-class filter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--src_root", default="human_talk_workspace/tata_v0.6_raw_pipeline")
    parser.add_argument("--dst_root", default="human_talk_workspace/tata_v0.7_raw_pipeline")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = validate_config(load_json(config_path))

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    exclude_classes = [str(x) for x in config["exclude_source_classes"]]
    match_columns = [str(x) for x in config["match_columns"]]

    outputs = []

    for item in config["files"]:
        name = str(item["name"])
        src = resolve_template_path(str(item["src"]), src_root, dst_root)
        dst = resolve_template_path(str(item["dst"]), src_root, dst_root)

        if src.exists():
            outputs.append(
                filter_csv(
                    src=src,
                    dst=dst,
                    name=name,
                    exclude_classes=exclude_classes,
                    match_columns=match_columns,
                )
            )
        else:
            outputs.append({
                "name": name,
                "src": str(src),
                "dst": str(dst),
                "status": "missing",
            })

    copied_files = copy_optional_files(
        config=config,
        src_root=src_root,
        dst_root=dst_root,
    )

    metadata_dir = dst_root / str(config.get("metadata_dir", "metadata"))
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_name = str(config.get("summary_name", "filter_summary"))
    summary_json = metadata_dir / f"{summary_name}.json"
    summary_md = metadata_dir / f"{summary_name}.md"

    summary = {
        "generated_at": now_iso(),
        "title": config.get("title", "Manifest Filter Summary"),
        "config": str(config_path),
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "exclude_source_classes": sorted(exclude_classes),
        "match_columns": match_columns,
        "outputs": outputs,
        "copied_files": copied_files,
        "note": config.get("note", "Original files were not modified."),
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary_md, summary)

    print("Filtered manifests created")
    print("-" * 90)
    print(f"Config:      {config_path}")
    print(f"Source root: {src_root}")
    print(f"Output root: {dst_root}")
    print(f"Summary:     {summary_json}")

    for item in outputs:
        if item.get("status") == "ok":
            print(
                f"{item['name']}: original={item['original_rows']} "
                f"kept={item['kept_rows']} removed={item['removed_rows']}"
            )
        else:
            print(f"{item['name']}: {item.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
