# scripts/audit_human_talk_dataset.py
#
# Audit a folder-structured human-talk / speaker dataset.
#
# Expected input:
#   human_talk_dataset/
#     Les_Brown/Les_Brown__0001.wav
#     Simon_Sinek/Simon_Sinek__0001.wav
#     ...
#
# This script is intentionally strict enough to prevent accidentally processing
# the old environmental dataset folders such as rain/gun_shot/car_crash.
#
# Outputs:
#   <out_dir>/human_talk_audit.csv
#   <out_dir>/human_talk_audit_summary.md
#
# Important fix:
#   Filename stems are normalized without collapsing repeated underscores.
#   Therefore Les_Brown__0001.wav remains Les_Brown__0001 and is correctly
#   detected when --filename_separator "__" is used.

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

AUDIO_EXTS_DEFAULT = ".wav,.flac,.mp3,.ogg,.m4a,.aac,.wma"


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


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


def safe_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * math.log10(max(float(x), eps)))


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


def collect_class_dirs(raw_root: Path, classes: list[str]) -> list[Path]:
    if classes:
        missing = [cls for cls in classes if not (raw_root / cls).is_dir()]
        if missing:
            available = sorted([p.name for p in raw_root.iterdir() if p.is_dir()]) if raw_root.exists() else []
            raise FileNotFoundError(
                "Requested human-talk class folder(s) not found.\n"
                f"Missing: {missing}\n"
                f"RawRoot: {raw_root}\n"
                f"Available folders: {available}\n\n"
                "This usually means RawRoot is pointing to the wrong dataset. "
                "Use -RawRoot human_talk_dataset, not event_dataset, if event_dataset contains environmental classes."
            )
        return [raw_root / cls for cls in classes]
    return sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def collect_audio_files(raw_root: Path, classes: list[str], exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for cls_dir in collect_class_dirs(raw_root, classes):
        for p in sorted(cls_dir.rglob("*"), key=natural_key):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return files


def parse_renamed_sample(path: Path, class_name: str, separator: str) -> dict:
    """
    Parse renamed samples such as:
      Les_Brown__0001.wav
      Simon_Sinek__0450.wav
    """
    cls_prefix = safe_name(class_name, preserve_case=True)
    stem = safe_stem_preserve_separator(path.stem, preserve_case=True)

    pattern = rf"^{re.escape(cls_prefix)}{re.escape(separator)}(\d+)$"
    match = re.match(pattern, stem, flags=re.IGNORECASE)

    if match:
        return {
            "parent_clip_id": stem,
            "renamed_format_ok": 1,
            "sample_index": int(match.group(1)),
            "expected_prefix": cls_prefix,
        }

    # Avoid duplicated IDs like Les_Brown__Les_Brown_0450 if the file already
    # starts with the class prefix but uses an unexpected separator.
    if stem.lower().startswith(cls_prefix.lower()):
        fallback_parent_id = stem
    else:
        fallback_parent_id = f"{cls_prefix}{separator}{stem}"

    return {
        "parent_clip_id": fallback_parent_id,
        "renamed_format_ok": 0,
        "sample_index": "",
        "expected_prefix": cls_prefix,
    }


def inspect_audio(path: Path, raw_root: Path, silence_dbfs: float, separator: str) -> dict:
    class_name = path.parent.name
    parsed = parse_renamed_sample(path, class_name, separator)
    row = {
        "class_name": class_name,
        "filename": path.name,
        "parent_clip_id": parsed["parent_clip_id"],
        "renamed_format_ok": parsed["renamed_format_ok"],
        "sample_index": parsed["sample_index"],
        "expected_prefix": parsed["expected_prefix"],
        "rel_path": str(path.relative_to(raw_root)),
        "abs_path": str(path.resolve()),
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "subtype": None,
        "frames": None,
        "rms": None,
        "rms_dbfs": None,
        "peak": None,
        "peak_dbfs": None,
        "silence_ratio": None,
        "clipping_ratio": None,
        "read_status": "ok",
        "quality_flag": "ok",
        "error": "",
    }
    try:
        info = sf.info(str(path))
        row["duration_sec"] = float(info.duration)
        row["sample_rate"] = int(info.samplerate)
        row["channels"] = int(info.channels)
        row["subtype"] = str(info.subtype)
        row["frames"] = int(info.frames)
        y, _ = sf.read(str(path), dtype="float32", always_2d=False)
        y = to_mono(y)
        if len(y) == 0:
            row["read_status"] = "error"
            row["quality_flag"] = "empty"
            row["error"] = "empty audio"
            return row
        y = y - float(np.mean(y))
        abs_y = np.abs(y)
        rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
        peak = float(np.max(abs_y))
        silence_amp = 10.0 ** (silence_dbfs / 20.0)
        silence_ratio = float(np.mean(abs_y < silence_amp))
        clipping_ratio = float(np.mean(abs_y >= 0.999))
        row["rms"] = rms
        row["rms_dbfs"] = safe_db(rms)
        row["peak"] = peak
        row["peak_dbfs"] = safe_db(peak)
        row["silence_ratio"] = silence_ratio
        row["clipping_ratio"] = clipping_ratio
        flags = []
        if float(info.duration) < 0.25:
            flags.append("too_short")
        if silence_ratio > 0.85:
            flags.append("mostly_silence")
        if clipping_ratio > 0.001:
            flags.append("possible_clipping")
        if rms < 1e-4:
            flags.append("very_low_energy")
        if not row["renamed_format_ok"]:
            flags.append("filename_not_standard")
        row["quality_flag"] = "|".join(flags) if flags else "ok"
        return row
    except Exception as e:
        row["read_status"] = "error"
        row["quality_flag"] = "read_error"
        row["error"] = repr(e)
        return row


def write_summary(df: pd.DataFrame, out_md: Path, clean_classes: set[str]) -> None:
    lines = []
    lines.append("# Human-talk dataset audit summary\n\n")
    lines.append(f"Total files: **{len(df)}**\n\n")
    lines.append(f"Classes audited: **{df['class_name'].nunique()}**\n\n")
    lines.append("## Class counts\n\n")
    lines.append("| Class | Files | Manually cleaned |\n|---|---:|---:|\n")
    for cls, n in df.groupby("class_name").size().sort_index().items():
        lines.append(f"| `{cls}` | {int(n)} | {int(cls in clean_classes)} |\n")
    lines.append("\n## Renamed filename check\n\n")
    lines.append("| Class | Files | Correct format | Incorrect |\n|---|---:|---:|---:|\n")
    check = df.groupby("class_name")["renamed_format_ok"].agg(["count", "sum"]).reset_index()
    for _, r in check.iterrows():
        total = int(r["count"])
        ok = int(r["sum"])
        lines.append(f"| `{r['class_name']}` | {total} | {ok} | {total - ok} |\n")
    lines.append("\n## Duration summary by class\n\n")
    lines.append("| Class | Count | Min | Median | Mean | Max |\n|---|---:|---:|---:|---:|---:|\n")
    dur = df.dropna(subset=["duration_sec"]).groupby("class_name")["duration_sec"].agg(["count", "min", "median", "mean", "max"]).reset_index()
    for _, r in dur.iterrows():
        lines.append(f"| `{r['class_name']}` | {int(r['count'])} | {float(r['min']):.3f} | {float(r['median']):.3f} | {float(r['mean']):.3f} | {float(r['max']):.3f} |\n")
    lines.append("\n## Quality flags\n\n")
    lines.append("| Quality flag | Files |\n|---|---:|\n")
    for flag, n in df.groupby("quality_flag").size().sort_values(ascending=False).items():
        lines.append(f"| `{flag}` | {int(n)} |\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit human-talk speaker dataset.")
    parser.add_argument("--raw_root", default="human_talk_dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--classes", default="", help="Comma-separated classes to audit. Empty = all folders.")
    parser.add_argument("--clean_classes", default="")
    parser.add_argument("--filename_separator", default="__")
    parser.add_argument("--exts", default=AUDIO_EXTS_DEFAULT)
    parser.add_argument("--silence_dbfs", type=float, default=-45.0)
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not raw_root.exists():
        raise FileNotFoundError(f"RawRoot does not exist: {raw_root}")
    classes = parse_csv_list(args.classes)
    clean_classes = set(parse_csv_list(args.clean_classes))
    exts = {x.strip().lower() for x in args.exts.split(",") if x.strip()}
    files = collect_audio_files(raw_root, classes, exts)
    if not files:
        raise RuntimeError(f"No audio files found under {raw_root}")
    print("\nAuditing human-talk dataset")
    print("-" * 90)
    print(f"Raw root:           {raw_root.resolve()}")
    print(f"Output dir:         {out_dir.resolve()}")
    print(f"Classes requested:  {classes if classes else 'ALL'}")
    print(f"Audio files:        {len(files)}")
    print(f"Filename separator: {args.filename_separator}")
    print("-" * 90)
    rows = []
    for i, p in enumerate(files, start=1):
        if i % 500 == 0:
            print(f"[audit] {i}/{len(files)}")
        row = inspect_audio(p, raw_root, args.silence_dbfs, args.filename_separator)
        row["manual_cleaned"] = int(row["class_name"] in clean_classes)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["class_name", "sample_index", "filename"]).reset_index(drop=True)
    out_csv = out_dir / "human_talk_audit.csv"
    out_md = out_dir / "human_talk_audit_summary.md"
    df.to_csv(out_csv, index=False)
    write_summary(df, out_md, clean_classes)
    print("\nAudit completed.")
    print(f"CSV:     {out_csv}")
    print(f"Summary: {out_md}")
    print("\nClass counts:")
    print(df.groupby("class_name").size().sort_index().to_string())
    print("\nRenamed format check:")
    fmt = df.groupby("class_name")["renamed_format_ok"].agg(["count", "sum"])
    fmt["incorrect"] = fmt["count"] - fmt["sum"]
    print(fmt.to_string())
    print("\nQuality flags:")
    print(df.groupby("quality_flag").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
