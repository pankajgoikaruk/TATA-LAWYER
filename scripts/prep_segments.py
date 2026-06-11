# scripts/prep_segments.py

"""
Unified ASHADIP audio preprocessing and segmentation.

Design goals
------------
1. Auto-discover any class folders under --root.
2. Support mixed-length raw audio through input_mode=segment.
3. Support already clipped datasets through input_mode=ready.
4. Always keep parent-file traceability in segments.csv.
5. Export physical fixed-length segment WAVs for reuse and inspection.
6. Keep downstream training compatible by preserving wav_relpath as the parent clip key.

Important columns in segments.csv
---------------------------------
- wav_relpath: parent cleaned WAV relative to cache/clean; used for clip grouping.
- segment_wav_relpath: exported segment WAV relative to cache; used by extract_features.py.
- parent_start: start time inside the parent cleaned WAV.
- start: kept as 0.0 because features are extracted from the physical segment WAV.
- split_key: file/group key used for leakage-safe splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from data.transforms_audio import bandpass


SUPPORTED_EXTS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".au", ".mp3", ".m4a"
}


def rms_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(y, dtype=np.float32) ** 2)) + 1e-9))


def safe_read_audio(path: Path, dtype: str = "float32") -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Read audio with soundfile first; fall back to librosa for compressed formats."""
    try:
        y, sr = sf.read(path, dtype=dtype)
        return y, int(sr)
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception as e:
            warnings.warn(f"Skipping unreadable file: {path} ({e})")
            return None, None


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    # librosa mono=False gives (channels, samples); soundfile gives (samples, channels)
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        warnings.warn(f"--config was provided but file not found: {p}. Ignoring.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalise_bandpass_value(value):
    """Return [low, high] or None. Accepts YAML lists, strings, null/none/off."""
    if value is None:
        return None

    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in {"", "none", "null", "false", "off", "0"}:
            return None
        raw = raw.replace(";", ",")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid bandpass value: {value}. Use [low, high], 'low,high', or null."
            )
        return [float(parts[0]), float(parts[1])]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) != 2:
            raise ValueError(
                f"Invalid bandpass list: {value}. Use exactly two values or null."
            )
        return [float(value[0]), float(value[1])]

    if value is False:
        return None

    raise ValueError(f"Invalid bandpass value: {value}")


def _parse_json_map(text: str) -> Dict[str, int]:
    """
    Robust parser for per-label segment caps.

    Accepts:
      - JSON: {"gun_shot":0,"rain":5}
      - PowerShell-escaped JSON: {\"gun_shot\":0,\"rain\":5}
      - CLI-safe syntax: gun_shot=0,rain=5 or gun_shot:0,rain:5
      - @path/to/file.json containing any of the above
    """
    if not text or not str(text).strip():
        return {}

    raw = str(text).strip()

    # File-based caps avoid shell quoting issues entirely.
    if raw.startswith("@"):
        cap_path = Path(raw[1:])
        if not cap_path.exists():
            raise argparse.ArgumentTypeError(
                f"Cap file not found for --max_segments_per_label_json: {cap_path}"
            )
        raw = cap_path.read_text(encoding="utf-8").strip()

    # Remove literal wrapper quotes if a shell left them in the string.
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()

    candidates = [raw]

    # Windows/PowerShell may pass JSON as {\"label\":5}; convert to normal JSON.
    if '\\"' in raw:
        candidates.append(raw.replace('\\"', '"'))

    # Try generic unicode unescape too, but keep it non-fatal.
    try:
        decoded = bytes(raw, "utf-8").decode("unicode_escape")
        if decoded not in candidates:
            candidates.append(decoded)
    except Exception:
        pass

    last_error = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            if not isinstance(data, dict):
                raise argparse.ArgumentTypeError(
                    "--max_segments_per_label_json must be a JSON object."
                )
            return {str(k): int(v) for k, v in data.items()}
        except Exception as e:
            last_error = e

    # Fallback parser for CLI-safe syntax: gun_shot=0,rain=5 or {gun_shot:0,rain:5}
    simple = raw.strip().strip("{}")
    if simple:
        out: Dict[str, int] = {}
        try:
            for item in simple.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    key, value = item.split("=", 1)
                elif ":" in item:
                    key, value = item.split(":", 1)
                else:
                    raise ValueError(f"missing ':' or '=' in item: {item}")
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                if not key:
                    raise ValueError(f"empty label name in item: {item}")
                out[key] = int(value)
            if out:
                return out
        except Exception as e:
            last_error = e

    raise argparse.ArgumentTypeError(
        "Invalid value for --max_segments_per_label_json. "
        "Use JSON like '{\"rain\":5}', CLI-safe syntax like 'rain=5,wind=5', "
        "or a file reference like '@configs/caps.json'. "
        f"Original error: {last_error}"
    )


def two_stage_parse():
    """Parse --config first, then use YAML defaults while allowing CLI overrides."""
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default="", help="Optional YAML config, e.g. configs/audio_moth.yaml")
    cfg0, rest = p0.parse_known_args()

    cfg = load_yaml(cfg0.config)
    seed_default = int(cfg.get("seed", 42))

    split_cfg = cfg.get("split", {}) or {}
    train_default = float(split_cfg.get("train", 0.7))
    val_default = float(split_cfg.get("val", 0.15))
    test_default = float(split_cfg.get("test", 0.15))
    strat_default = bool(split_cfg.get("stratify", True))

    audio_cfg = cfg.get("audio", {}) or {}
    sr_default = int(audio_cfg.get("sample_rate", 16000))
    seg_default = float(audio_cfg.get("segment_sec", 1.0))
    hop_default = float(audio_cfg.get("segment_hop", 0.5))
    silence_default = float(audio_cfg.get("silence_dbfs", -40))
    try:
        bp_default = _normalise_bandpass_value(audio_cfg.get("bandpass", [100, 3000]))
    except Exception as e:
        warnings.warn(f"Invalid audio.bandpass in config ({e}); falling back to [100, 3000].")
        bp_default = [100, 3000]

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=cfg0.config)
    p.add_argument("--root", default="data")
    p.add_argument("--cache", default="data_cache")

    p.add_argument("--sr", type=int, default=sr_default)
    p.add_argument("--segment_sec", type=float, default=seg_default)
    p.add_argument("--hop", type=float, default=hop_default)
    p.add_argument("--silence_dbfs", type=float, default=silence_default)
    p.add_argument(
        "--bandpass",
        nargs="*",
        default=bp_default,
        help="Optional bandpass as two values, e.g. --bandpass 50 7600. Omit to use YAML. Use YAML null to disable.",
    )

    p.add_argument("--seed", type=int, default=seed_default)
    p.add_argument("--train_frac", type=float, default=train_default)
    p.add_argument("--val_frac", type=float, default=val_default)
    p.add_argument("--test_frac", type=float, default=test_default)

    p.add_argument(
        "--input_mode",
        choices=["segment", "ready"],
        default="segment",
        help="segment: raw mixed-length audio. ready: already fixed-length clips.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional explicit class folders. Default: auto-discover subfolders under --root.",
    )
    p.add_argument(
        "--min_keep_sec",
        type=float,
        default=0.25,
        help="Short files >= this duration are kept as one padded segment.",
    )
    p.add_argument(
        "--max_segments_per_file_default",
        type=int,
        default=0,
        help="Default max segments per parent file. 0 = keep all.",
    )
    p.add_argument(
        "--max_segments_per_label_json",
        type=str,
        default="",
        help='Optional JSON caps per label, e.g. "{""fireworks"":5,""gunshot"":0}". 0 = keep all.',
    )
    p.add_argument(
        "--split_unit",
        choices=["file", "group"],
        default="file",
        help="file: split by source file. group: split related files by a group id.",
    )
    p.add_argument(
        "--group_regex",
        type=str,
        default="",
        help="Optional regex for split_unit=group. Use first capture group or named group 'group'.",
    )
    p.add_argument(
        "--export_segment_wavs",
        dest="export_segment_wavs",
        action="store_true",
        default=True,
        help="Export physical fixed-length segment WAVs. Default: enabled.",
    )
    p.add_argument(
        "--no_export_segment_wavs",
        dest="export_segment_wavs",
        action="store_false",
        help="Disable segment WAV export. Not recommended for this project.",
    )
    p.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Delete existing clean/features/segment_wavs/CSV artifacts inside the cache before rebuilding.",
    )
    p.add_argument(
        "--ready_metadata_dir",
        type=str,
        default="",
        help=(
            "Optional metadata directory for ready-mode datasets. "
            "If omitted, prep_segments auto-detects root/metadata or root.parent/metadata. "
            "Expected files: train.csv, val.csv, test.csv with final_path/source_file_id/start_sec."
        ),
    )

    if strat_default:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=True)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")
    else:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")

    args = p.parse_args(rest)
    args.bandpass = _normalise_bandpass_value(args.bandpass)
    args.max_segments_per_label = _parse_json_map(args.max_segments_per_label_json)
    args._cfg = cfg
    return args


SPLIT_NAMES = ("train", "val", "test")


def is_ready_split_root(root: Path) -> bool:
    """Return True for prepared datasets laid out as root/train/class/*.wav etc."""
    return any((root / split).is_dir() for split in SPLIT_NAMES)


def list_label_dirs(
    root: Path,
    explicit_labels: Optional[Sequence[str]] = None,
    input_mode: str = "segment",
) -> List[str]:
    """Discover labels for both raw class folders and ready train/val/test folders."""
    ready_split = str(input_mode).lower() == "ready" and is_ready_split_root(root)

    if explicit_labels:
        labels = [str(x).strip() for x in explicit_labels if str(x).strip()]
        if ready_split:
            missing = []
            for lab in labels:
                present = any((root / split / lab).is_dir() for split in SPLIT_NAMES)
                if not present:
                    missing.append(lab)
            if missing:
                raise SystemExit(
                    f"These label folders were not found under any train/val/test split in {root}: {missing}"
                )
        else:
            missing = [lab for lab in labels if not (root / lab).is_dir()]
            if missing:
                raise SystemExit(f"These label folders were not found under {root}: {missing}")
        return labels

    if ready_split:
        found = set()
        for split in SPLIT_NAMES:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            found.update(
                p.name for p in split_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        labels = sorted(found)
        if not labels:
            raise SystemExit(f"No class folders found under train/val/test splits in {root}")
        return labels

    labels = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not labels:
        raise SystemExit(f"No class folders found under {root}")
    return labels


def iter_audio_files(label_dir: Path) -> Iterable[Path]:
    for p in sorted(label_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _norm_path_text(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def find_ready_metadata_dir(root: Path, explicit: str = "") -> Optional[Path]:
    """Find prepared_data2/metadata for ready-mode datasets, if available."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        root / "metadata",
        root.parent / "metadata",
        root.parent.parent / "metadata",
    ])
    for cand in candidates:
        if cand and cand.exists() and cand.is_dir():
            has_split_csv = all((cand / f"{split}.csv").exists() for split in SPLIT_NAMES)
            if has_split_csv:
                return cand
    return None


def _resolve_ready_final_path(root: Path, final_path_value: object, split: str, label: str, segment_id: str) -> Path:
    """Resolve metadata final_path robustly across Windows/Linux and relative roots."""
    raw = _norm_path_text(final_path_value)
    candidates: List[Path] = []

    if raw:
        p = Path(raw)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(Path.cwd() / p)
            candidates.append(root.parent.parent / p)
            candidates.append(root.parent / p)
        parts = list(Path(raw).parts)
        norm_parts = [str(x).replace("\\", "/") for x in parts]
        for i, part in enumerate(norm_parts):
            if part in SPLIT_NAMES and i + 2 < len(norm_parts):
                candidates.append(root.joinpath(*norm_parts[i:]))
                break
        candidates.append(root / split / label / Path(raw).name)

    if segment_id:
        candidates.append(root / split / label / f"{segment_id}.wav")

    seen = set()
    for cand in candidates:
        try:
            key = str(cand.resolve())
        except Exception:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand

    if raw:
        return root / split / label / Path(raw).name
    return root / split / label / f"{segment_id}.wav"


def load_ready_metadata_rows(root: Path, args, labels: Sequence[str]) -> Tuple[List[dict], Optional[Path]]:
    """Load source-aware prepared dataset metadata for ready mode."""
    meta_dir = find_ready_metadata_dir(root, getattr(args, "ready_metadata_dir", ""))
    if meta_dir is None:
        return [], None

    allowed_labels = set(str(x) for x in labels)
    rows: List[dict] = []
    for split in SPLIT_NAMES:
        csv_path = meta_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        required = {"segment_id", "label", "source_file_id", "final_path", "split"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise SystemExit(f"Ready metadata file {csv_path} missing required columns: {missing}")

        for rec in df.to_dict(orient="records"):
            label = str(rec.get("label", "")).strip()
            if label not in allowed_labels:
                continue
            rec_split = str(rec.get("split", split)).strip().lower()
            if rec_split not in SPLIT_NAMES:
                rec_split = split
            segment_id = str(rec.get("segment_id", "")).strip()
            src_path = _resolve_ready_final_path(root, rec.get("final_path", ""), rec_split, label, segment_id)
            if not src_path.exists():
                warnings.warn(f"Ready metadata references missing final_path; skipping: {src_path}")
                continue

            source_file_id = str(rec.get("source_file_id", "") or "").strip()
            source_path = str(rec.get("source_path", "") or "").strip()
            if not source_file_id:
                source_file_id = f"{label}__{hashlib.md5(_norm_path_text(source_path or segment_id).encode('utf-8')).hexdigest()[:12]}"

            out = dict(rec)
            out.update({
                "src_path": src_path,
                "label": label,
                "predefined_split": rec_split,
                "source_file_id": source_file_id,
                "source_id": source_file_id,
                "group_id": source_file_id,
                "source_file": source_path,
                "raw_file": source_path,
                "parent_file": source_path,
                "ready_metadata_dir": str(meta_dir),
                "ready_metadata_used": True,
            })
            rows.append(out)

    return rows, meta_dir


def _safe_token(text: str, max_len: int = 32) -> str:
    """Create a short filesystem-safe token for Windows-friendly paths."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    if not token:
        token = "x"
    return token[:max_len]


def stable_wav_name(label: str, orig_relpath: str) -> str:
    """Short deterministic parent WAV name to avoid Windows path-length failures."""
    label_safe = _safe_token(label, max_len=24)
    h = hashlib.md5(orig_relpath.encode("utf-8")).hexdigest()[:16]
    return f"{label_safe}_parent_{h}.wav"

def get_group_key(orig_relpath: str, group_regex: str = "") -> str:
    rel = str(orig_relpath).replace("\\", "/")
    if group_regex:
        m = re.search(group_regex, rel)
        if m:
            if "group" in m.groupdict():
                return str(m.group("group"))
            if m.groups():
                return str(m.group(1))
            return str(m.group(0))
        warnings.warn(f"group_regex did not match {rel}; falling back to source file key.")
        return rel

    parts = Path(rel).parts
    # Expected: label/session/file.wav -> group=session. Otherwise fallback to file.
    if len(parts) >= 3:
        return str(parts[1])
    return rel


def clean_audio_file(
    src: Path,
    label: str,
    root: Path,
    clean_root: Path,
    args,
    predefined_split: Optional[str] = None,
    ready_meta: Optional[dict] = None,
) -> Optional[dict]:
    try:
        orig_relpath = os.path.relpath(src, root).replace("\\", "/")
    except ValueError:
        orig_relpath = _norm_path_text(src)
    y, sr = safe_read_audio(src, dtype="float32")
    if y is None or sr is None:
        return None

    y = to_mono(y)
    if sr != args.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
        sr = int(args.sr)

    y = np.asarray(y, dtype=np.float32)
    if y.size > 0:
        y = y - float(np.mean(y))

    if args.bandpass:
        y = bandpass(y, sr, float(args.bandpass[0]), float(args.bandpass[1]))

    peak = float(np.max(np.abs(y)) + 1e-9) if y.size > 0 else 1e-9
    if peak > 0:
        y = (0.8913 * y / peak).astype(np.float32)

    out = clean_root / "parents" / (predefined_split or "unsplit") / label / stable_wav_name(label, orig_relpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr)

    clean_relpath = os.path.relpath(out, clean_root).replace("\\", "/")
    row = {
        "filepath": str(out),
        "clean_relpath": clean_relpath,
        "label": label,
        "duration": float(len(y) / sr),
        "orig_filepath": str(src),
        "orig_relpath": orig_relpath,
        "orig_ext": src.suffix.lower(),
        "orig_sr": int(sr),
        "predefined_split": predefined_split or "",
    }

    if ready_meta:
        source_file_id = str(ready_meta.get("source_file_id", "") or "").strip()
        source_path = str(ready_meta.get("source_path", "") or ready_meta.get("source_file", "") or "").strip()
        row.update({
            "segment_id": str(ready_meta.get("segment_id", "") or ""),
            "source_file_id": source_file_id,
            "source_id": str(ready_meta.get("source_id", source_file_id) or source_file_id),
            "group_id": str(ready_meta.get("group_id", source_file_id) or source_file_id),
            "source_file": source_path,
            "raw_file": str(ready_meta.get("raw_file", source_path) or source_path),
            "parent_file": str(ready_meta.get("parent_file", source_path) or source_path),
            "source_path": source_path,
            "final_path": str(ready_meta.get("final_path", "") or ""),
            "segment_index": ready_meta.get("segment_index", ""),
            "start_sec": ready_meta.get("start_sec", ""),
            "end_sec": ready_meta.get("end_sec", ""),
            "selection_rank": ready_meta.get("selection_rank", ""),
            "selection_method": ready_meta.get("selection_method", ""),
            "ready_metadata_used": bool(ready_meta.get("ready_metadata_used", True)),
            "ready_metadata_dir": str(ready_meta.get("ready_metadata_dir", "") or ""),
        })
    return row


def select_evenly_spaced_starts(starts: List[int], max_keep: int) -> List[int]:
    if max_keep <= 0 or len(starts) <= max_keep:
        return starts
    idx = np.linspace(0, len(starts) - 1, num=max_keep, dtype=int)
    idx = np.unique(idx)
    return [starts[int(i)] for i in idx]


def label_cap(label: str, default_cap: int, cap_map: Dict[str, int]) -> int:
    return int(cap_map.get(str(label), default_cap))


def build_candidate_segments(manifest: pd.DataFrame, clean_root: Path, args) -> pd.DataFrame:
    seg_rows = []
    dropped_short = []

    for _, r in manifest.iterrows():
        parent_rel = str(r["clean_relpath"]).replace("\\", "/")
        parent_path = clean_root / parent_rel
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"]).replace("\\", "/")
        predefined_split = str(r.get("predefined_split", "") or "").strip().lower()

        if args.input_mode == "ready" and str(r.get("source_file_id", "") or "").strip():
            group_id = str(r.get("group_id", "") or r.get("source_file_id", "")).strip()
            logical_wav_relpath = f"ready_group/{label}/{group_id}.wav"
            split_key = group_id
        else:
            group_id = orig_relpath if args.split_unit == "file" else get_group_key(orig_relpath, args.group_regex)
            logical_wav_relpath = parent_rel
            split_key = orig_relpath if args.split_unit == "file" else get_group_key(orig_relpath, args.group_regex)

        source_file = str(r.get("source_file", "") or r.get("source_path", "") or "")
        raw_file = str(r.get("raw_file", "") or source_file)
        parent_file = str(r.get("parent_file", "") or source_file)

        y, sr = sf.read(parent_path, dtype="float32")
        y = to_mono(y)
        win = int(round(float(args.segment_sec) * sr))
        hop = int(round(float(args.hop) * sr))
        n = int(len(y))
        dur_sec = float(n / sr) if sr else 0.0

        if args.input_mode == "ready":
            if dur_sec < float(args.min_keep_sec):
                dropped_short.append((parent_rel, dur_sec))
                continue
            if rms_dbfs(y) < float(args.silence_dbfs):
                continue
            starts = [0]
        else:
            if n < win:
                if dur_sec < float(args.min_keep_sec):
                    dropped_short.append((parent_rel, dur_sec))
                    continue
                if rms_dbfs(y) < float(args.silence_dbfs):
                    continue
                starts = [0]
            else:
                all_starts = list(range(0, max(n - win + 1, 0), max(hop, 1)))
                valid_starts = []
                for s in all_starts:
                    seg = y[s:s + win]
                    if rms_dbfs(seg) >= float(args.silence_dbfs):
                        valid_starts.append(s)
                max_keep = label_cap(label, args.max_segments_per_file_default, args.max_segments_per_label)
                starts = select_evenly_spaced_starts(valid_starts, max_keep)

        max_keep_effective = label_cap(label, args.max_segments_per_file_default, args.max_segments_per_label)
        for local_segment_index, s in enumerate(starts):
            if args.input_mode == "ready":
                clean_parent_start = 0.0
                try:
                    original_start = float(r.get("start_sec", 0.0) or 0.0)
                except Exception:
                    original_start = 0.0
                try:
                    segment_index_value = int(float(r.get("segment_index", local_segment_index) or local_segment_index))
                except Exception:
                    segment_index_value = int(local_segment_index)
            else:
                clean_parent_start = float(s / sr)
                original_start = float(s / sr)
                segment_index_value = int(local_segment_index)

            seg_rows.append({
                "wav_relpath": logical_wav_relpath,          # logical parent key for clip grouping
                "clean_relpath": parent_rel,                 # actual cleaned WAV used for export
                "orig_relpath": orig_relpath,
                "label": label,
                "parent_start": clean_parent_start,
                "start": original_start,                     # original parent time; used by clip_policy_test sorting
                "duration": float(args.segment_sec),
                "split_key": split_key,
                "group_id": group_id,
                "source_file_id": str(r.get("source_file_id", "") or group_id),
                "source_id": str(r.get("source_id", "") or group_id),
                "source_file": source_file,
                "raw_file": raw_file,
                "parent_file": parent_file,
                "source_path": str(r.get("source_path", "") or source_file),
                "final_path": str(r.get("final_path", "") or ""),
                "segment_id": str(r.get("segment_id", "") or ""),
                "original_start_sec": original_start,
                "original_end_sec": r.get("end_sec", ""),
                "selection_rank": r.get("selection_rank", ""),
                "selection_method": r.get("selection_method", ""),
                "ready_metadata_used": bool(r.get("ready_metadata_used", False)),
                "ready_metadata_dir": str(r.get("ready_metadata_dir", "") or ""),
                "predefined_split": predefined_split,
                "parent_duration_sec": dur_sec,
                "input_mode": str(args.input_mode),
                "segment_index_parent": segment_index_value,
                "segments_before_cap": int(len(valid_starts) if args.input_mode == "segment" and n >= win else len(starts)),
                "max_segments_per_file_effective": int(max_keep_effective),
            })

    if dropped_short:
        print(
            f"\nDropped {len(dropped_short)} files shorter than min_keep_sec={args.min_keep_sec} sec "
            f"(showing up to 10):"
        )
        for rel, dur in dropped_short[:10]:
            print(f" - {rel} ({dur:.3f}s)")
        if len(dropped_short) > 10:
            print(" ... (more dropped)")

    seg_df = pd.DataFrame(seg_rows)
    if len(seg_df) == 0:
        raise SystemExit("No segments above silence threshold; try raising --silence_dbfs, e.g. -55.")
    return seg_df


def split_by_key(
    seg_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify: bool,
    key_col: str = "split_key",
):
    """Split by parent file/group key, then map split to all child segments."""
    from sklearn.model_selection import train_test_split

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {total:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("Split ratios must be > 0 for train/val/test.")

    if key_col not in seg_df.columns:
        raise ValueError(f"Missing split key column: {key_col}")

    key_df = seg_df[[key_col, "label"]].drop_duplicates().reset_index(drop=True)
    label_counts = key_df.groupby(key_col)["label"].nunique()
    if label_counts.max() > 1:
        warnings.warn(
            "Some split keys contain multiple labels. Stratification will use the first observed label per key."
        )

    key_df = key_df.groupby(key_col, as_index=False).agg(label=("label", "first"))
    keys = key_df[key_col].values
    labels = key_df["label"].values

    temp_frac = val_frac + test_frac
    try:
        if stratify:
            keys_train, keys_temp, y_train, y_temp = train_test_split(
                keys, labels, test_size=temp_frac, stratify=labels, random_state=seed
            )
        else:
            keys_train, keys_temp, y_train, y_temp = train_test_split(
                keys, labels, test_size=temp_frac, random_state=seed, shuffle=True
            )

        test_within_temp = test_frac / temp_frac
        if stratify:
            keys_val, keys_test = train_test_split(
                keys_temp, test_size=test_within_temp, stratify=y_temp, random_state=seed
            )
        else:
            keys_val, keys_test = train_test_split(
                keys_temp, test_size=test_within_temp, random_state=seed, shuffle=True
            )
    except ValueError as e:
        warnings.warn(f"Stratified split failed ({e}). Falling back to non-stratified split.")
        keys_train, keys_temp = train_test_split(keys, test_size=temp_frac, random_state=seed, shuffle=True)
        keys_val, keys_test = train_test_split(
            keys_temp, test_size=(test_frac / temp_frac), random_state=seed, shuffle=True
        )

    split_map = {k: "train" for k in keys_train}
    split_map.update({k: "val" for k in keys_val})
    split_map.update({k: "test" for k in keys_test})

    seg_df = seg_df.copy()
    seg_df["split"] = seg_df[key_col].map(split_map)
    if seg_df["split"].isna().any():
        missing = seg_df.loc[seg_df["split"].isna(), key_col].unique()[:10]
        raise SystemExit(f"Some split keys were not assigned a split: {missing}")

    seg_counts = seg_df["split"].value_counts().to_dict()
    key_counts = key_df.assign(split=key_df[key_col].map(split_map))["split"].value_counts().to_dict()
    return seg_df, seg_counts, key_counts


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant").astype(np.float32)


def export_segments(seg_df: pd.DataFrame, cache: Path, clean_root: Path, args) -> pd.DataFrame:
    """Write physical fixed-length segment WAVs and add segment_wav_relpath."""
    segment_root = cache / "segment_wavs"
    segment_root.mkdir(parents=True, exist_ok=True)

    out_rels = []
    current_parent_rel = None
    current_y = None
    current_sr = None

    for idx, row in seg_df.reset_index(drop=True).iterrows():
        parent_rel = str(row["clean_relpath"]).replace("\\", "/")
        if parent_rel != current_parent_rel:
            parent_path = clean_root / parent_rel
            y, sr = sf.read(parent_path, dtype="float32")
            current_y = to_mono(y)
            current_sr = int(sr)
            current_parent_rel = parent_rel

        start_i = int(round(float(row["parent_start"]) * current_sr))
        dur_i = int(round(float(row["duration"]) * current_sr))
        clip = current_y[start_i:start_i + dur_i]
        clip = pad_or_trim(clip, dur_i)

        label = str(row["label"])
        split = str(row["split"])
        src_key = f"{row['orig_relpath']}|{row['parent_start']:.6f}|{idx}"
        h = hashlib.md5(src_key.encode("utf-8")).hexdigest()[:16]
        label_safe = _safe_token(label, max_len=24)
        fname = f"{label_safe}_seg_{h}.wav"

        out = segment_root / split / label_safe / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.export_segment_wavs:
            sf.write(out, clip, current_sr)

        out_rels.append(os.path.relpath(out, cache).replace("\\", "/"))

    seg_df = seg_df.copy().reset_index(drop=True)
    seg_df["segment_wav_relpath"] = out_rels
    return seg_df


def print_inventory_summary(inv_df: pd.DataFrame):
    print("\n=== Audio inventory summary ===")
    print(f"Files found: {len(inv_df)}")
    print("Labels:", sorted(inv_df["label"].unique().tolist()))
    if len(inv_df) == 0:
        return
    summary = (
        inv_df.groupby("label")
        .agg(
            files=("label", "size"),
            min_sec=("duration", "min"),
            median_sec=("duration", "median"),
            mean_sec=("duration", "mean"),
            max_sec=("duration", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


def remove_cache_artifacts(cache: Path):
    for name in ["clean", "features", "segment_wavs"]:
        p = cache / name
        if p.exists():
            shutil.rmtree(p)
    for name in [
        "segments.csv", "manifest.csv", "moths_manifest.csv", "audio_inventory.csv",
        "audio_inventory_by_label.csv"
    ]:
        p = cache / name
        if p.exists():
            p.unlink()


def main():
    args = two_stage_parse()

    root = Path(args.root)
    cache = Path(args.cache)
    clean_root = cache / "clean"

    if args.force_rebuild:
        remove_cache_artifacts(cache)

    clean_root.mkdir(parents=True, exist_ok=True)
    ready_split_root = args.input_mode == "ready" and is_ready_split_root(root)
    labels = list_label_dirs(root, args.labels, input_mode=args.input_mode)

    rows = []
    skipped = []

    if ready_split_root:
        print("[prep_segments] Ready-mode split dataset detected: using existing train/val/test folders.")

        metadata_rows, metadata_dir = load_ready_metadata_rows(root, args, labels)
        if metadata_rows:
            print(f"[prep_segments] Ready metadata detected: {metadata_dir}")
            print("[prep_segments] Using source_file_id/source_path/start_sec metadata for clip grouping.")
            for rec in metadata_rows:
                src = Path(rec["src_path"])
                label = str(rec["label"])
                split = str(rec["predefined_split"])
                row = clean_audio_file(src, label, root, clean_root, args, predefined_split=split, ready_meta=rec)
                if row is None:
                    skipped.append(str(src))
                    continue
                rows.append(row)
        else:
            print("[prep_segments] No ready metadata found; falling back to one ready WAV = one clip group.")
            for split in SPLIT_NAMES:
                split_dir = root / split
                if not split_dir.is_dir():
                    continue
                for label in labels:
                    label_dir = split_dir / label
                    if not label_dir.is_dir():
                        warnings.warn(f"Label folder missing for split={split}: {label_dir}")
                        continue
                    for src in iter_audio_files(label_dir):
                        row = clean_audio_file(src, label, root, clean_root, args, predefined_split=split)
                        if row is None:
                            skipped.append(str(src))
                            continue
                        rows.append(row)
    else:
        if args.input_mode == "ready":
            print("[prep_segments] Ready mode without train/val/test folders: will create a leakage-safe split from class folders.")
        for label in labels:
            for src in iter_audio_files(root / label):
                row = clean_audio_file(src, label, root, clean_root, args)
                if row is None:
                    skipped.append(str(src))
                    continue
                rows.append(row)

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable files (showing up to 10):")
        for s in skipped[:10]:
            print(" -", s)
        if len(skipped) > 10:
            print(" ... (more skipped)")

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise SystemExit("No valid audio files found. Check --root and class folders.")

    cache.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cache / "manifest.csv", index=False)
    # Backward-compatible filename for older scripts/docs.
    manifest.to_csv(cache / "moths_manifest.csv", index=False)
    manifest.to_csv(cache / "audio_inventory.csv", index=False)
    (
        manifest.groupby("label")
        .agg(
            files=("label", "size"),
            min_sec=("duration", "min"),
            median_sec=("duration", "median"),
            mean_sec=("duration", "mean"),
            max_sec=("duration", "max"),
        )
        .reset_index()
        .to_csv(cache / "audio_inventory_by_label.csv", index=False)
    )
    print_inventory_summary(manifest)
    print(f"Bandpass: {args.bandpass if args.bandpass else 'disabled'}")

    seg_df = build_candidate_segments(manifest, clean_root, args)

    if ready_split_root:
        seg_df = seg_df.copy()
        seg_df["split"] = seg_df["predefined_split"].astype(str).str.lower()
        bad_splits = sorted(set(seg_df["split"].unique()) - set(SPLIT_NAMES))
        if bad_splits:
            raise SystemExit(f"Invalid predefined splits in ready dataset: {bad_splits}")
        seg_counts = seg_df["split"].value_counts().to_dict()
        key_counts = seg_df[["split_key", "split"]].drop_duplicates()["split"].value_counts().to_dict()
    else:
        seg_df, seg_counts, key_counts = split_by_key(
            seg_df,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
            seed=int(args.seed),
            stratify=bool(args.stratify),
            key_col="split_key",
        )

    seg_df = export_segments(seg_df, cache, clean_root, args)

    # Keep stable column order while preserving any future extra columns.
    preferred = [
        "orig_relpath", "clean_relpath", "wav_relpath", "segment_wav_relpath",
        "label", "split", "split_key", "group_id", "source_file_id", "source_id",
        "source_file", "raw_file", "parent_file", "source_path", "final_path", "segment_id",
        "predefined_split", "parent_start", "start", "original_start_sec", "original_end_sec",
        "duration", "parent_duration_sec", "input_mode", "segment_index_parent",
        "selection_rank", "selection_method", "ready_metadata_used", "ready_metadata_dir",
        "segments_before_cap", "max_segments_per_file_effective"
    ]
    other = [c for c in seg_df.columns if c not in preferred]
    seg_df = seg_df[preferred + other]
    seg_df.to_csv(cache / "segments.csv", index=False)
    (
        seg_df.groupby(["split", "label"])
        .size()
        .rename("count")
        .reset_index()
        .to_csv(cache / "segments_by_split_label.csv", index=False)
    )
    (
        seg_df.groupby(["label"])
        .size()
        .rename("count")
        .reset_index()
        .to_csv(cache / "segments_by_label.csv", index=False)
    )
    if "source_file_id" in seg_df.columns:
        (
            seg_df.groupby(["split", "label"])
            .agg(
                segments=("label", "size"),
                sources=("source_file_id", "nunique"),
                avg_segments_per_source=("source_file_id", lambda s: float(len(s) / max(s.nunique(), 1))),
            )
            .reset_index()
            .to_csv(cache / "segments_by_split_label_source.csv", index=False)
        )
        (
            seg_df.groupby(["split", "label", "source_file_id"])
            .size()
            .rename("segments")
            .reset_index()
            .sort_values(["split", "label", "segments"], ascending=[True, True, False])
            .to_csv(cache / "segments_by_source.csv", index=False)
        )

    print("\n=== Segmentation summary ===")
    print("Split keys:", key_counts)
    print("Segments:", seg_counts)
    print("\nSegments by split/label:")
    print(seg_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))
    print(f"\nSaved segments.csv: {cache / 'segments.csv'}")
    print(f"Saved physical segment WAVs under: {cache / 'segment_wavs'}")


if __name__ == "__main__":
    main()
