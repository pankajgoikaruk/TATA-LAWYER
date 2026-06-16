# data/datasets_multilabel_masked.py
#
# Mask-aware multi-label dataset loader for partially reviewed audio labels.
#
# Each sample returns:
#   x:    FloatTensor [1, n_mels, T]
#   y:    FloatTensor [num_labels]
#   mask: FloatTensor [num_labels]
#
# mask[c] = 1 -> label c contributes to loss/metrics
# mask[c] = 0 -> label c is unknown and ignored

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_labels(
    labels_json: str | Path,
) -> list[str]:
    labels_json = Path(labels_json)
    if not labels_json.exists():
        raise FileNotFoundError(f"labels_json not found: {labels_json}")

    with labels_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(
            f"Invalid labels.json; expected a non-empty 'labels' list: {labels_json}"
        )
    return [str(x) for x in labels]


class MaskedMultiLabelLogMelDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        features_root: str | Path,
        labels_json: str | Path,
        split: str,
        filter_column: str | None = None,
        filter_value: int | str | None = 1,
        allow_empty: bool = False,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.features_root = Path(features_root)
        self.split = str(split)

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        if not self.features_root.exists():
            raise FileNotFoundError(f"Features root not found: {self.features_root}")

        df = pd.read_csv(self.manifest_csv, low_memory=False)

        required = {"split", "feat_relpath"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest missing required columns {sorted(missing)}: "
                f"{self.manifest_csv}"
            )

        self.labels = load_labels(labels_json)
        self.mask_columns = [f"mask_{label}" for label in self.labels]
        self.num_labels = len(self.labels)

        missing_labels = [c for c in self.labels if c not in df.columns]
        missing_masks = [c for c in self.mask_columns if c not in df.columns]
        if missing_labels:
            raise ValueError(f"Missing target columns: {missing_labels}")
        if missing_masks:
            raise ValueError(f"Missing mask columns: {missing_masks}")

        split_values = df["split"].fillna("").astype(str).str.strip().str.lower()
        selected = df.loc[split_values.eq(self.split.lower())].copy()

        if filter_column is not None:
            if filter_column not in selected.columns:
                raise ValueError(
                    f"Filter column '{filter_column}' not found in manifest."
                )
            if filter_value is None:
                selected = selected.loc[selected[filter_column].notna()].copy()
            else:
                left = selected[filter_column]
                numeric_left = pd.to_numeric(left, errors="coerce")
                try:
                    numeric_value = float(filter_value)
                    selected = selected.loc[numeric_left.eq(numeric_value)].copy()
                except (TypeError, ValueError):
                    selected = selected.loc[
                        left.fillna("").astype(str).str.strip().eq(str(filter_value))
                    ].copy()

        self.df = selected.reset_index(drop=True)

        if len(self.df) == 0 and not allow_empty:
            suffix = (
                ""
                if filter_column is None
                else f", {filter_column}={filter_value}"
            )
            raise RuntimeError(
                f"No rows for split='{self.split}'{suffix} in {self.manifest_csv}"
            )

        if len(self.df):
            targets_df = self.df[self.labels].apply(
                pd.to_numeric, errors="coerce"
            )
            masks_df = self.df[self.mask_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            invalid_targets = targets_df.isna() | ~targets_df.isin([0, 1])
            invalid_masks = masks_df.isna() | ~masks_df.isin([0, 1])

            if invalid_targets.any().any():
                bad = np.argwhere(invalid_targets.to_numpy())[:10]
                examples = [
                    {
                        "row": int(r),
                        "label": self.labels[int(c)],
                        "value": self.df.iloc[int(r)][self.labels[int(c)]],
                    }
                    for r, c in bad
                ]
                raise ValueError(f"Targets must be binary 0/1. Examples: {examples}")

            if invalid_masks.any().any():
                bad = np.argwhere(invalid_masks.to_numpy())[:10]
                examples = [
                    {
                        "row": int(r),
                        "mask": self.mask_columns[int(c)],
                        "value": self.df.iloc[int(r)][self.mask_columns[int(c)]],
                    }
                    for r, c in bad
                ]
                raise ValueError(f"Masks must be binary 0/1. Examples: {examples}")

            self.targets = targets_df.to_numpy(dtype=np.float32)
            self.masks = masks_df.to_numpy(dtype=np.float32)
        else:
            self.targets = np.zeros((0, self.num_labels), dtype=np.float32)
            self.masks = np.zeros((0, self.num_labels), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        feat_path = self.features_root / Path(feat_rel)

        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        feature = np.load(feat_path).astype(np.float32)
        if feature.ndim != 2:
            raise RuntimeError(
                f"Expected [n_mels, T], got {feature.shape}: {feat_path}"
            )

        x = torch.from_numpy(feature).float().unsqueeze(0)
        y = torch.from_numpy(self.targets[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        return x, y, mask

    def known_positive_negative_counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for idx, label in enumerate(self.labels):
            known = self.masks[:, idx] > 0.5
            target = self.targets[:, idx] > 0.5
            out[label] = {
                "known": int(known.sum()),
                "unknown": int((~known).sum()),
                "positive": int((known & target).sum()),
                "negative": int((known & ~target).sum()),
            }
        return out

    def pos_weight_tensor(
        self,
        max_value: float | None = 20.0,
    ) -> torch.FloatTensor:
        counts = self.known_positive_negative_counts()
        weights = []
        for label in self.labels:
            positive = max(float(counts[label]["positive"]), 1.0)
            negative = max(float(counts[label]["negative"]), 0.0)
            value = negative / positive
            if max_value is not None:
                value = min(value, float(max_value))
            weights.append(value)
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(
        self,
        label_balance_power: float = 0.0,
        synthetic_balance_power: float = 0.0,
    ) -> torch.DoubleTensor:
        if len(self.df) == 0:
            return torch.ones(0, dtype=torch.double)

        weights = np.ones(len(self.df), dtype=np.float64)
        label_balance_power = float(label_balance_power or 0.0)
        synthetic_balance_power = float(synthetic_balance_power or 0.0)

        if label_balance_power > 0:
            known_positive = (self.targets > 0.5) & (self.masks > 0.5)
            positive_counts = np.maximum(known_positive.sum(axis=0), 1.0)
            inverse = 1.0 / (positive_counts ** label_balance_power)

            row_weights = []
            for row_positive in known_positive:
                active = np.where(row_positive)[0]
                row_weights.append(
                    1.0 if len(active) == 0 else float(np.mean(inverse[active]))
                )
            weights *= np.asarray(row_weights, dtype=np.float64)

        if synthetic_balance_power > 0 and "is_synthetic" in self.df.columns:
            groups = self.df["is_synthetic"].fillna("").astype(str)
            group_counts = groups.value_counts().to_dict()
            group_weights = np.asarray(
                [
                    1.0
                    / max(float(group_counts[str(value)]), 1.0)
                    ** synthetic_balance_power
                    for value in groups
                ],
                dtype=np.float64,
            )
            weights *= group_weights

        mean_weight = float(np.mean(weights)) if len(weights) else 1.0
        if mean_weight > 0:
            weights /= mean_weight
        return torch.as_tensor(weights, dtype=torch.double)


def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(
    dataset: MaskedMultiLabelLogMelDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int | None,
    sampler=None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=False,
        drop_last=False,
        worker_init_fn=_seed_worker if int(num_workers) > 0 else None,
        generator=generator,
    )


def make_masked_multilabel_loaders(
    manifest_csv: str | Path,
    features_root: str | Path,
    labels_json: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int | None = 42,
    label_balance_power: float = 0.0,
    synthetic_balance_power: float = 0.0,
):
    # Training uses every train row, with unknown labels masked.
    ds_train = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="train",
    )

    # Strict original validation/test subsets preserve direct comparability
    # with the old v0.9 experiment.
    ds_val_strict = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="val",
        filter_column="v09_checkpoint_eligible",
        filter_value=1,
    )
    ds_test_strict = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_standard_test_eligible",
        filter_value=1,
    )

    # All rows support masked secondary evaluation.
    ds_val_all = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="val",
    )
    ds_test_all = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="test",
    )

    # Recovered-only subsets show performance on the manually reviewed domain.
    ds_val_recovered = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="val",
        filter_column="v09_masked_review_applied",
        filter_value=1,
        allow_empty=True,
    )
    ds_test_recovered = MaskedMultiLabelLogMelDataset(
        manifest_csv,
        features_root,
        labels_json,
        split="test",
        filter_column="v09_masked_review_applied",
        filter_value=1,
        allow_empty=True,
    )

    sampler = None
    if (
        float(label_balance_power or 0.0) > 0
        or float(synthetic_balance_power or 0.0) > 0
    ):
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(int(seed or 0))
        sampler = WeightedRandomSampler(
            weights=ds_train.sample_weights(
                label_balance_power=label_balance_power,
                synthetic_balance_power=synthetic_balance_power,
            ),
            num_samples=len(ds_train),
            replacement=True,
            generator=sampler_generator,
        )

    loaders = {
        "train": _make_loader(
            ds_train,
            batch_size,
            num_workers,
            shuffle=True,
            seed=seed,
            sampler=sampler,
        ),
        "val_strict": _make_loader(
            ds_val_strict,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
        "test_strict": _make_loader(
            ds_test_strict,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
        "val_all_masked": _make_loader(
            ds_val_all,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
        "test_all_masked": _make_loader(
            ds_test_all,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
        "val_recovered_masked": _make_loader(
            ds_val_recovered,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
        "test_recovered_masked": _make_loader(
            ds_test_recovered,
            batch_size,
            num_workers,
            shuffle=False,
            seed=seed,
        ),
    }

    datasets = {
        "train": ds_train,
        "val_strict": ds_val_strict,
        "test_strict": ds_test_strict,
        "val_all_masked": ds_val_all,
        "test_all_masked": ds_test_all,
        "val_recovered_masked": ds_val_recovered,
        "test_recovered_masked": ds_test_recovered,
    }

    print("[INFO] Masked multi-label datasets loaded")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset):,} rows")
    print(f"  labels: {ds_train.labels}")

    return loaders, datasets, ds_train.labels
