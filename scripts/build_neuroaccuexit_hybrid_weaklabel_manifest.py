#!/usr/bin/env python
"""
Build the downstream NeuroAccuExit hybrid weak-label manifest.

This script consumes the frozen TATA-LAWYER v0.10.1 domain-aware
hybrid policy and writes a new manifest for downstream main-model work.

Selected policy:
  - normal/original rows        -> original v0.9 TATA model, fixed 0.50
  - recovered low-energy rows   -> masked human-reviewed v0.10 model,
                                   recovered-domain thresholds

The script does not modify the input manifest, checkpoints, thresholds,
manual-review files, or feature arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import load_labels
from utils.model_factory import build_audio_exit_net


VERSION = "tata_lawyer_v0.10.1_domain_aware_hybrid"
NORMAL_MODEL_NAME = "original_v09_tata"
RECOVERED_MODEL_NAME = "masked_v10_human_reviewed"
NORMAL_THRESHOLD_PROFILE = "fixed_0_50"
RECOVERED_THRESHOLD_PROFILE = "recovered_low_energy_profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final hybrid weak-label manifest for downstream "
            "NeuroAccuExit main-model training."
        )
    )
    parser.add_argument("--original_run_dir", type=Path, required=True)
    parser.add_argument("--masked_run_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument(
        "--recovered_thresholds_json",
        type=Path,
        required=True,
        help=(
            "JSON containing recovered-domain thresholds. Accepted formats: "
            "{'thresholds': {...}} or {'masked': {'thresholds': {...}}}."
        ),
    )
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--reports_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to include. Default: train,val,test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return str(value)

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def build_model(run_dir: Path, labels: list[str], device: str):
    config_path = run_dir / "config_used.json"
    checkpoint = run_dir / "ckpt" / "best.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = load_json(config_path)
    exit_hint = config.get(
        "exit_hint",
        {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        },
    )

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=int(config.get("n_mels", 64)),
        tap_blocks=parse_tap_blocks(config.get("tap_blocks", [1, 3])),
        model_cfg={"exit_hint": exit_hint},
    ).to(device)

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model, config


def threshold_vector_from_payload(
    payload: dict,
    labels: list[str],
    path_name: str,
) -> tuple[np.ndarray, dict]:
    """Return thresholds in label order.

    Supported payloads:
      1. {"thresholds": {label: value}}
      2. {"masked": {"thresholds": {label: value}}}
    """
    threshold_map = payload.get("thresholds")
    if threshold_map is None and isinstance(payload.get("masked"), dict):
        threshold_map = payload["masked"].get("thresholds")

    if not isinstance(threshold_map, dict):
        raise ValueError(
            f"No thresholds mapping found in {path_name}. Expected either "
            "{'thresholds': {...}} or {'masked': {'thresholds': {...}}}."
        )

    missing = [label for label in labels if label not in threshold_map]
    if missing:
        raise ValueError(f"Missing recovered thresholds in {path_name}: {missing}")

    ordered = np.asarray(
        [float(threshold_map[label]) for label in labels],
        dtype=np.float64,
    )
    clean_map = {label: float(threshold_map[label]) for label in labels}
    return ordered, clean_map


def parse_splits(raw: str) -> set[str]:
    values = {
        item.strip().lower()
        for item in str(raw).split(",")
        if item.strip()
    }
    if not values:
        raise ValueError("At least one split must be provided.")
    return values


def numeric_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.isin([0, 1])
    if invalid.any():
        examples = series.loc[invalid].head(10).tolist()
        raise ValueError(f"{name} must contain only 0/1. Examples: {examples}")
    return values.astype(np.int8)


def detect_recovered_rows(df: pd.DataFrame) -> pd.Series:
    """Detect rows that belong to the recovered low-energy domain.

    The preferred indicator is v09_masked_review_applied=1. Fallbacks are
    included so the builder remains usable with nearby v0.9/v0.10 manifests.
    """
    recovered = pd.Series(False, index=df.index)

    if "v09_masked_review_applied" in df.columns:
        values = pd.to_numeric(df["v09_masked_review_applied"], errors="coerce")
        recovered = recovered | values.eq(1).fillna(False)

    if "v09_evaluation_group" in df.columns:
        recovered = recovered | df["v09_evaluation_group"].fillna("").astype(str).str.lower().eq(
            "recovered_human_reviewed"
        )

    for column in (
        "v09_data_origin",
        "data_origin",
        "source_origin",
    ):
        if column in df.columns:
            recovered = recovered | df[column].fillna("").astype(str).str.lower().str.contains(
                "recovered", regex=False
            )

    return recovered.astype(bool)


class FeatureManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_root: Path,
        row_positions: Iterable[int],
    ):
        self.df = df
        self.features_root = Path(features_root)
        self.row_positions = [int(pos) for pos in row_positions]

        if not self.features_root.exists():
            raise FileNotFoundError(f"Features root not found: {self.features_root}")
        if "feat_relpath" not in self.df.columns:
            raise ValueError("Manifest must contain feat_relpath.")

    def __len__(self) -> int:
        return len(self.row_positions)

    def __getitem__(self, index: int):
        row_pos = self.row_positions[index]
        row = self.df.iloc[row_pos]
        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        feat_path = self.features_root / Path(feat_rel)

        if not feat_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        feature = np.load(feat_path).astype(np.float32)
        if feature.ndim != 2:
            raise RuntimeError(
                f"Expected feature shape [n_mels, T], got {feature.shape}: {feat_path}"
            )

        x = torch.from_numpy(feature).float().unsqueeze(0)
        return x, int(row_pos)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        drop_last=False,
    )


@torch.no_grad()
def collect_probabilities(
    model,
    dataset: FeatureManifestDataset,
    device: str,
    batch_size: int,
    num_workers: int,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, num_labels), dtype=np.float32),
        )

    loader = make_loader(dataset, batch_size, num_workers)
    positions_parts = []
    probabilities_parts = []

    for x, row_positions in loader:
        logits = model(x.to(device))[-1]
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        positions_parts.append(row_positions.numpy().astype(np.int64))
        probabilities_parts.append(probabilities.astype(np.float32))

    return (
        np.concatenate(positions_parts, axis=0),
        np.concatenate(probabilities_parts, axis=0),
    )


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)


def make_strict_subset_role(df: pd.DataFrame) -> list[str]:
    split = df["split"].fillna("").astype(str).str.lower()
    val_eligible = pd.Series(False, index=df.index)
    test_eligible = pd.Series(False, index=df.index)

    if "v09_checkpoint_eligible" in df.columns:
        val_eligible = pd.to_numeric(
            df["v09_checkpoint_eligible"], errors="coerce"
        ).eq(1).fillna(False)
    if "v09_standard_test_eligible" in df.columns:
        test_eligible = pd.to_numeric(
            df["v09_standard_test_eligible"], errors="coerce"
        ).eq(1).fillna(False)

    role = np.full(len(df), "not_strict_eval", dtype=object)
    role[split.eq("val") & val_eligible] = "strict_val"
    role[split.eq("test") & test_eligible] = "strict_test"
    return role.tolist()


def write_distribution_reports(
    out: pd.DataFrame,
    labels: list[str],
    reports_dir: Path,
) -> None:
    label_rows = []
    for label in labels:
        values = pd.to_numeric(out[label], errors="coerce").fillna(0).astype(int)
        positives = int(values.eq(1).sum())
        negatives = int(values.eq(0).sum())
        label_rows.append(
            {
                "label": label,
                "positive": positives,
                "negative": negatives,
                "positive_rate": positives / max(len(out), 1),
            }
        )

    pd.DataFrame(label_rows).to_csv(
        reports_dir / "hybrid_weaklabel_label_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_rows = []
    grouped = out.groupby(
        ["split", "routing_domain", "label_source_model"],
        dropna=False,
    )
    for (split, routing_domain, source_model), group in grouped:
        source_rows.append(
            {
                "split": split,
                "routing_domain": routing_domain,
                "label_source_model": source_model,
                "rows": int(len(group)),
            }
        )

    pd.DataFrame(source_rows).to_csv(
        reports_dir / "hybrid_weaklabel_source_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_rows = [
        {"split": str(split), "rows": int(count)}
        for split, count in out["split"].astype(str).value_counts().sort_index().items()
    ]
    pd.DataFrame(split_rows).to_csv(
        reports_dir / "hybrid_weaklabel_split_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    args = parse_args()

    original_run_dir = args.original_run_dir.expanduser().resolve()
    masked_run_dir = args.masked_run_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    labels_json = args.labels_json.expanduser().resolve()
    thresholds_json = args.recovered_thresholds_json.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()

    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output manifest already exists: {output_manifest}\n"
            "Use --overwrite only when intentionally rebuilding it."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not labels_json.is_file():
        raise FileNotFoundError(f"Labels JSON not found: {labels_json}")

    labels = load_labels(labels_json)
    splits = parse_splits(args.splits)
    fixed_thresholds = np.full(
        len(labels),
        float(args.fixed_threshold),
        dtype=np.float64,
    )
    recovered_thresholds, recovered_threshold_map = threshold_vector_from_payload(
        load_json(thresholds_json),
        labels,
        str(thresholds_json),
    )

    if str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    df = pd.read_csv(manifest_path, low_memory=False)
    if "split" not in df.columns:
        raise ValueError("Manifest must contain split.")

    missing_labels = [label for label in labels if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Manifest missing target columns: {missing_labels}")

    for label in labels:
        df[label] = numeric_binary(df[label], label)

    split_values = df["split"].fillna("").astype(str).str.lower()
    include_mask = split_values.isin(splits)
    if not include_mask.any():
        raise RuntimeError(f"No rows found for requested splits: {sorted(splits)}")

    working = df.loc[include_mask].copy().reset_index(drop=True)
    recovered_mask = detect_recovered_rows(working).to_numpy(dtype=bool)
    normal_mask = ~recovered_mask

    normal_positions = np.where(normal_mask)[0].tolist()
    recovered_positions = np.where(recovered_mask)[0].tolist()

    print("\nNeuroAccuExit hybrid weak-label manifest builder")
    print("-" * 96)
    print(f"Input manifest:            {manifest_path}")
    print(f"Rows included:             {len(working):,}")
    print(f"Normal/original rows:      {len(normal_positions):,}")
    print(f"Recovered low-energy rows: {len(recovered_positions):,}")
    print(f"Device:                    {device}")
    print("-" * 96)

    original_model, original_config = build_model(
        original_run_dir,
        labels,
        device,
    )
    normal_dataset = FeatureManifestDataset(
        working,
        features_root,
        normal_positions,
    )
    normal_row_positions, normal_probabilities = collect_probabilities(
        original_model,
        normal_dataset,
        device,
        args.batch_size,
        args.num_workers,
        len(labels),
    )
    del original_model

    masked_model, masked_config = build_model(
        masked_run_dir,
        labels,
        device,
    )
    recovered_dataset = FeatureManifestDataset(
        working,
        features_root,
        recovered_positions,
    )
    recovered_row_positions, recovered_probabilities = collect_probabilities(
        masked_model,
        recovered_dataset,
        device,
        args.batch_size,
        args.num_workers,
        len(labels),
    )
    del masked_model

    all_probabilities = np.full(
        (len(working), len(labels)),
        np.nan,
        dtype=np.float32,
    )
    all_predictions = np.zeros((len(working), len(labels)), dtype=np.int8)

    if len(normal_row_positions):
        normal_predictions = apply_thresholds(normal_probabilities, fixed_thresholds)
        all_probabilities[normal_row_positions] = normal_probabilities
        all_predictions[normal_row_positions] = normal_predictions

    if len(recovered_row_positions):
        recovered_predictions = apply_thresholds(
            recovered_probabilities,
            recovered_thresholds,
        )
        all_probabilities[recovered_row_positions] = recovered_probabilities
        all_predictions[recovered_row_positions] = recovered_predictions

    if np.isnan(all_probabilities).any():
        raise RuntimeError("Some rows did not receive probabilities.")

    out = working.copy()

    # Preserve the previous manifest targets and masks before replacing label
    # columns with final hybrid weak labels.
    for label_index, label in enumerate(labels):
        out[f"upstream_manifest_{label}"] = out[label].astype(np.int8)
        mask_col = f"mask_{label}"
        if mask_col in out.columns:
            out[f"upstream_{mask_col}"] = pd.to_numeric(
                out[mask_col],
                errors="coerce",
            ).fillna(1).astype(np.int8)
        out[label] = all_predictions[:, label_index].astype(np.int8)
        out[f"prob_{label}"] = all_probabilities[:, label_index].astype(np.float32)
        out[mask_col] = np.int8(1)

    out["routing_domain"] = np.where(
        recovered_mask,
        "recovered_low_energy",
        "normal_original",
    )
    out["label_source_model"] = np.where(
        recovered_mask,
        RECOVERED_MODEL_NAME,
        NORMAL_MODEL_NAME,
    )
    out["threshold_profile_name"] = np.where(
        recovered_mask,
        RECOVERED_THRESHOLD_PROFILE,
        NORMAL_THRESHOLD_PROFILE,
    )
    out["threshold_profile_path"] = np.where(
        recovered_mask,
        str(thresholds_json),
        "fixed_0.50",
    )
    out["is_low_energy_recovered"] = recovered_mask.astype(np.int8)
    out["is_original_trusted_row"] = (~recovered_mask).astype(np.int8)
    out["strict_subset_role"] = make_strict_subset_role(out)
    out["weak_label_generation_version"] = VERSION
    out["label_reliability_group"] = np.where(
        recovered_mask,
        "masked_v10_recovered_model_prediction",
        "original_v09_model_prediction",
    )
    out["has_any_unknown_label"] = np.int8(0)
    out["known_label_count"] = np.int16(len(labels))
    out["unknown_label_count"] = np.int16(0)

    # Place downstream-facing provenance columns near the front while keeping
    # all original metadata columns available for traceability.
    front_columns = [
        column
        for column in [
            "clip_id",
            "segment_id",
            "split",
            "start_sec",
            "end_sec",
            "audio_path",
            "feat_relpath",
            "routing_domain",
            "label_source_model",
            "threshold_profile_name",
            "threshold_profile_path",
            "is_low_energy_recovered",
            "is_original_trusted_row",
            "strict_subset_role",
            "weak_label_generation_version",
            "label_reliability_group",
            "has_any_unknown_label",
            "known_label_count",
            "unknown_label_count",
        ]
        if column in out.columns
    ]
    label_columns = labels
    prob_columns = [f"prob_{label}" for label in labels]
    mask_columns = [f"mask_{label}" for label in labels]
    ordered = front_columns + label_columns + prob_columns + mask_columns
    ordered = list(dict.fromkeys(ordered))
    remaining = [column for column in out.columns if column not in ordered]
    out = out[ordered + remaining]

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_manifest, index=False, encoding="utf-8-sig")

    write_distribution_reports(out, labels, reports_dir)

    summary = {
        "version": VERSION,
        "input_manifest": str(manifest_path),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "output_manifest": str(output_manifest),
        "reports_dir": str(reports_dir),
        "rows": int(len(out)),
        "splits_included": sorted(splits),
        "split_counts": {
            str(k): int(v)
            for k, v in out["split"].astype(str).value_counts().sort_index().to_dict().items()
        },
        "routing_counts": {
            str(k): int(v)
            for k, v in out["routing_domain"].value_counts().to_dict().items()
        },
        "label_source_counts": {
            str(k): int(v)
            for k, v in out["label_source_model"].value_counts().to_dict().items()
        },
        "normal_domain_policy": {
            "model": NORMAL_MODEL_NAME,
            "run_dir": str(original_run_dir),
            "threshold_profile": NORMAL_THRESHOLD_PROFILE,
            "fixed_threshold": float(args.fixed_threshold),
        },
        "recovered_low_energy_policy": {
            "model": RECOVERED_MODEL_NAME,
            "run_dir": str(masked_run_dir),
            "threshold_profile": RECOVERED_THRESHOLD_PROFILE,
            "thresholds_json": str(thresholds_json),
            "thresholds": recovered_threshold_map,
        },
        "model_configs": {
            "original": {
                "tap_blocks": original_config.get("tap_blocks"),
                "n_mels": original_config.get("n_mels", 64),
                "num_exits": original_config.get("num_exits"),
                "threshold": original_config.get("threshold"),
            },
            "masked": {
                "tap_blocks": masked_config.get("tap_blocks"),
                "n_mels": masked_config.get("n_mels", 64),
                "num_exits": masked_config.get("num_exits"),
                "threshold": masked_config.get("threshold"),
                "checkpoint_selection": masked_config.get("checkpoint_selection"),
            },
        },
        "label_positive_counts": {
            label: int(pd.to_numeric(out[label], errors="coerce").fillna(0).astype(int).sum())
            for label in labels
        },
        "source_files_modified": False,
    }
    save_json(summary, reports_dir / "hybrid_weaklabel_manifest_summary.json")

    print("\nSaved outputs")
    print("-" * 96)
    print(output_manifest)
    print(reports_dir / "hybrid_weaklabel_manifest_summary.json")
    print(reports_dir / "hybrid_weaklabel_label_distribution.csv")
    print(reports_dir / "hybrid_weaklabel_source_distribution.csv")
    print(reports_dir / "hybrid_weaklabel_split_distribution.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
