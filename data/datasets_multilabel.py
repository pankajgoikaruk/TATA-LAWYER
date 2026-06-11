# data/datasets_multilabel.py
#
# Multi-label dataset loader for ASHADIP audio tagging.
#
# Input manifest:
#   multilabel_cache/metadata/multilabel_features_manifest.csv
#
# Feature root:
#   multilabel_cache/features
#
# Returns:
#   x: torch.FloatTensor, shape [1, n_mels, T]
#   y: torch.FloatTensor, shape [num_labels]
#
# Example y:
#   rain + thunderstorm:
#   [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def load_labels(labels_json: str | Path | None, manifest_df: pd.DataFrame | None = None) -> list[str]:
    """
    Load label order from labels.json.

    labels.json is preferred because it guarantees stable label order.
    """
    if labels_json is not None:
        labels_json = Path(labels_json)
        if not labels_json.exists():
            raise FileNotFoundError(f"labels_json not found: {labels_json}")

        with labels_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        labels = payload.get("labels")
        if not isinstance(labels, list) or not labels:
            raise RuntimeError(f"Invalid labels.json. Missing non-empty 'labels' list: {labels_json}")

        return [str(x) for x in labels]

    if manifest_df is None:
        raise ValueError("Either labels_json or manifest_df must be provided.")

    # Fallback inference. Avoid metadata columns.
    metadata_cols = {
        "sample_id",
        "filepath",
        "abs_path",
        "source_file",
        "source_stem",
        "source_ext",
        "class_dir",
        "primary_label",
        "labels",
        "num_labels",
        "is_clean_seed",
        "is_synthetic",
        "split",
        "source_paths",
        "source_names",
        "source_labels",
        "component_gains_db",
        "sample_rate",
        "duration_sec",
        "global_index",
        "feat_relpath",
        "feature_path",
        "feature_shape",
        "feature_sample_rate",
        "feature_clip_sec",
        "feature_n_mels",
        "feature_n_fft",
        "feature_win_ms",
        "feature_hop_ms",
        "feature_cmvn",
    }

    possible = []
    for col in manifest_df.columns:
        if col in metadata_cols:
            continue

        values = set(pd.Series(manifest_df[col]).dropna().astype(int).unique().tolist())
        if values.issubset({0, 1}):
            possible.append(str(col))

    if not possible:
        raise RuntimeError(
            "Could not infer label columns from manifest. "
            "Please provide labels_json."
        )

    return sorted(possible)


class MultiLabelLogMelDataset(Dataset):
    """
    Multi-label log-mel dataset.

    Each row must contain:
      - split
      - feat_relpath
      - one binary column per label
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        features_root: str | Path,
        labels_json: str | Path | None = None,
        split: str = "train",
        allow_empty: bool = False,
    ):
        manifest_csv = Path(manifest_csv)
        features_root = Path(features_root)

        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

        if not features_root.exists():
            raise FileNotFoundError(f"Features root not found: {features_root}")

        df = pd.read_csv(manifest_csv, low_memory=False)

        if "split" not in df.columns:
            raise ValueError(f"'split' column not found in {manifest_csv}")

        if "feat_relpath" not in df.columns:
            raise ValueError(
                f"'feat_relpath' column not found in {manifest_csv}. "
                "Did you run scripts/extract_multilabel_features.py?"
            )

        self.labels = load_labels(labels_json, manifest_df=df)
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.num_labels = len(self.labels)

        missing_label_cols = [lab for lab in self.labels if lab not in df.columns]
        if missing_label_cols:
            raise ValueError(
                "Manifest is missing these label columns:\n"
                f"{missing_label_cols}"
            )

        self.all_df = df.reset_index(drop=True)
        self.df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
        self.split = str(split)
        self.manifest_csv = manifest_csv
        self.features_root = features_root

        if len(self.df) == 0 and not allow_empty:
            raise RuntimeError(
                f"No rows found for split='{split}' in manifest:\n{manifest_csv}"
            )

        # Precompute targets for speed and sampler support.
        if len(self.df) > 0:
            self.targets = self.df[self.labels].astype(np.float32).values
        else:
            self.targets = np.zeros((0, self.num_labels), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        feat_path = self.features_root / Path(feat_rel)

        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        S = np.load(feat_path).astype(np.float32)  # [n_mels, T]

        if S.ndim != 2:
            raise RuntimeError(
                f"Expected feature shape [n_mels, T], got {S.shape} for {feat_path}"
            )

        x = torch.from_numpy(S).float().unsqueeze(0)  # [1, n_mels, T]

        y_np = row[self.labels].astype(np.float32).values
        y = torch.from_numpy(y_np).float()  # [num_labels]

        return x, y

    def label_positive_counts(self) -> Dict[str, int]:
        if len(self.df) == 0:
            return {label: 0 for label in self.labels}

        counts = self.df[self.labels].astype(int).sum(axis=0).to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def label_negative_counts(self) -> Dict[str, int]:
        positives = self.label_positive_counts()
        n = len(self.df)
        return {label: int(n - positives[label]) for label in self.labels}

    def label_prevalence(self) -> Dict[str, float]:
        positives = self.label_positive_counts()
        n = max(len(self.df), 1)
        return {label: float(positives[label] / n) for label in self.labels}

    def pos_weight_tensor(self, max_value: float | None = 20.0) -> torch.FloatTensor:
        """
        Positive class weights for BCEWithLogitsLoss.

        pos_weight[c] = negative_count[c] / positive_count[c]

        This helps when some labels are less frequent.
        """
        positives = self.label_positive_counts()
        negatives = self.label_negative_counts()

        weights = []
        for label in self.labels:
            pos = max(float(positives[label]), 1.0)
            neg = max(float(negatives[label]), 0.0)
            w = neg / pos

            if max_value is not None:
                w = min(w, float(max_value))

            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(
        self,
        label_balance_power: float = 0.0,
        synthetic_balance_power: float = 0.0,
    ) -> torch.DoubleTensor:
        """
        Build per-sample weights for WeightedRandomSampler.

        label_balance_power:
          0.0 -> no label balancing
          0.5 -> moderate inverse-frequency balancing
          1.0 -> stronger inverse-frequency balancing

        synthetic_balance_power:
          0.0 -> no clean/synthetic balancing
          1.0 -> reduce dominance of whichever group is larger
        """
        if len(self.df) == 0:
            return torch.ones(0, dtype=torch.double)

        weights = np.ones(len(self.df), dtype=np.float64)

        label_balance_power = float(label_balance_power or 0.0)
        synthetic_balance_power = float(synthetic_balance_power or 0.0)

        if label_balance_power > 0:
            y = self.targets.astype(np.float64)
            positive_counts = np.maximum(y.sum(axis=0), 1.0)

            # Inverse-frequency per positive label.
            inv = 1.0 / (positive_counts ** label_balance_power)

            sample_label_weights = []
            for row_y in y:
                active = np.where(row_y > 0.5)[0]
                if len(active) == 0:
                    sample_label_weights.append(1.0)
                else:
                    sample_label_weights.append(float(np.mean(inv[active])))

            weights *= np.asarray(sample_label_weights, dtype=np.float64)

        if synthetic_balance_power > 0 and "is_synthetic" in self.df.columns:
            group_values = self.df["is_synthetic"].astype(str)
            group_counts = group_values.value_counts().to_dict()

            group_weights = np.array(
                [
                    1.0 / max(float(group_counts[str(v)]), 1.0) ** synthetic_balance_power
                    for v in group_values.tolist()
                ],
                dtype=np.float64,
            )

            weights *= group_weights

        # Normalise for readability/debugging.
        mean_w = float(np.mean(weights)) if len(weights) else 1.0
        if mean_w > 0:
            weights = weights / mean_w

        return torch.as_tensor(weights, dtype=torch.double)


def _seed_worker(worker_id: int):
    """
    Deterministic DataLoader worker seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_train_sampler(
    ds_tr: MultiLabelLogMelDataset,
    label_balance_power: float,
    synthetic_balance_power: float,
    generator: Optional[torch.Generator] = None,
):
    if float(label_balance_power or 0.0) <= 0 and float(synthetic_balance_power or 0.0) <= 0:
        return None

    weights = ds_tr.sample_weights(
        label_balance_power=label_balance_power,
        synthetic_balance_power=synthetic_balance_power,
    )

    if weights.numel() == 0:
        return None

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_multilabel_loaders(
    manifest_csv: str | Path,
    features_root: str | Path,
    labels_json: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int | None = None,
    label_balance_power: float = 0.0,
    synthetic_balance_power: float = 0.0,
):
    """
    Create train/val/test DataLoaders.

    Returns:
      dl_train, dl_val, dl_test, labels
    """
    ds_tr = MultiLabelLogMelDataset(
        manifest_csv=manifest_csv,
        features_root=features_root,
        labels_json=labels_json,
        split="train",
    )

    ds_va = MultiLabelLogMelDataset(
        manifest_csv=manifest_csv,
        features_root=features_root,
        labels_json=labels_json,
        split="val",
    )

    ds_te = MultiLabelLogMelDataset(
        manifest_csv=manifest_csv,
        features_root=features_root,
        labels_json=labels_json,
        split="test",
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    train_sampler = _make_train_sampler(
        ds_tr=ds_tr,
        label_balance_power=label_balance_power,
        synthetic_balance_power=synthetic_balance_power,
        generator=generator,
    )

    if train_sampler is not None:
        print(
            "[INFO] Multi-label WeightedRandomSampler enabled: "
            f"label_balance_power={label_balance_power}, "
            f"synthetic_balance_power={synthetic_balance_power}"
        )

    print("[INFO] Multi-label dataset loaded")
    print(f"  labels: {ds_tr.labels}")
    print(f"  train rows: {len(ds_tr)}")
    print(f"  val rows:   {len(ds_va)}")
    print(f"  test rows:  {len(ds_te)}")

    print("[INFO] Train positive label counts:")
    for label, count in ds_tr.label_positive_counts().items():
        print(f"  {label}: {count}")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator if train_sampler is None else None,
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )

    return dl_tr, dl_va, dl_te, ds_tr.labels


def make_pos_weight_from_train(
    manifest_csv: str | Path,
    features_root: str | Path,
    labels_json: str | Path | None = None,
    max_value: float | None = 20.0,
) -> torch.FloatTensor:
    """
    Helper for training script.

    Usage:
      pos_weight = make_pos_weight_from_train(...)
      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    """
    ds_tr = MultiLabelLogMelDataset(
        manifest_csv=manifest_csv,
        features_root=features_root,
        labels_json=labels_json,
        split="train",
    )

    return ds_tr.pos_weight_tensor(max_value=max_value)