# data/datasets.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class LogMelDataset(Dataset):
    def __init__(self, segments_csv, features_root, split="train"):
        df = pd.read_csv(segments_csv)

        if "split" not in df.columns:
            raise ValueError(f"'split' column not found in {segments_csv}")
        if "feat_relpath" not in df.columns:
            raise ValueError(
                f"'feat_relpath' column not found in {segments_csv} "
                "(did you run extract_features?)"
            )
        if "label" not in df.columns:
            raise ValueError(f"'label' column not found in {segments_csv}")

        self.all_df = df.reset_index(drop=True)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = str(split)
        self.features_root = Path(features_root)

        # Stable label mapping across all splits/classes.
        labels = sorted(df["label"].astype(str).unique().tolist())
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.targets = [self.label2id[str(x)] for x in self.df["label"].astype(str).tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feat_rel = str(row["feat_relpath"]).replace("\\", "/")
        path = self.features_root / Path(feat_rel)

        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        S = np.load(path)  # (n_mels, T)
        x = torch.from_numpy(S).float().unsqueeze(0)  # (1, M, T)
        y = torch.tensor(self.label2id[str(row["label"])], dtype=torch.long)
        return x, y

    def class_counts_by_id(self) -> Dict[int, int]:
        counts = pd.Series(self.targets, dtype="int64").value_counts().to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def class_counts_by_label(self) -> Dict[str, int]:
        counts = self.df["label"].astype(str).value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def sample_weights(
        self,
        class_balance_power: float = 0.0,
        source_balance_power: float = 0.0,
    ) -> torch.DoubleTensor:
        """
        Build per-sample weights for WeightedRandomSampler.

        class_balance_power:
          0.0 -> no class balancing
          0.5 -> square-root inverse-frequency balancing
          1.0 -> full inverse-frequency balancing

        source_balance_power:
          0.0 -> no source/file balancing
          0.5/1.0 -> reduce dominance by files/groups with many segments
        """
        if len(self.df) == 0:
            return torch.ones(0, dtype=torch.double)

        weights = np.ones(len(self.df), dtype=np.float64)

        class_balance_power = float(class_balance_power or 0.0)
        source_balance_power = float(source_balance_power or 0.0)

        if class_balance_power > 0:
            class_counts = self.df["label"].astype(str).value_counts().to_dict()
            weights *= np.array(
                [
                    1.0 / max(float(class_counts[str(label)]), 1.0) ** class_balance_power
                    for label in self.df["label"].astype(str).tolist()
                ],
                dtype=np.float64,
            )

        if source_balance_power > 0:
            source_col = None
            for candidate in ["split_key", "orig_relpath", "wav_relpath", "clean_relpath"]:
                if candidate in self.df.columns:
                    source_col = candidate
                    break
            if source_col is not None:
                source_values = self.df[source_col].astype(str)
                source_counts = source_values.value_counts().to_dict()
                weights *= np.array(
                    [
                        1.0 / max(float(source_counts[str(src)]), 1.0) ** source_balance_power
                        for src in source_values.tolist()
                    ],
                    dtype=np.float64,
                )

        # Normalise only for readability/debugging; WeightedRandomSampler does not require it.
        mean_w = float(np.mean(weights)) if len(weights) else 1.0
        if mean_w > 0:
            weights = weights / mean_w
        return torch.as_tensor(weights, dtype=torch.double)


def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a deterministic seed derived from the main seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_train_sampler(
    ds_tr: LogMelDataset,
    class_balance_power: float,
    source_balance_power: float,
    generator: Optional[torch.Generator] = None,
):
    if float(class_balance_power or 0.0) <= 0 and float(source_balance_power or 0.0) <= 0:
        return None

    weights = ds_tr.sample_weights(
        class_balance_power=class_balance_power,
        source_balance_power=source_balance_power,
    )
    if weights.numel() == 0:
        return None

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_loaders(
    segments_csv,
    features_root,
    batch_size=64,
    num_workers=4,
    seed=None,
    class_balance_power: float = 0.0,
    source_balance_power: float = 0.0,
):
    ds_tr = LogMelDataset(segments_csv, features_root, "train")
    ds_va = LogMelDataset(segments_csv, features_root, "val")
    ds_te = LogMelDataset(segments_csv, features_root, "test")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    train_sampler = _make_train_sampler(
        ds_tr,
        class_balance_power=class_balance_power,
        source_balance_power=source_balance_power,
        generator=generator,
    )

    if train_sampler is not None:
        print(
            "[INFO] WeightedRandomSampler enabled: "
            f"class_balance_power={class_balance_power}, "
            f"source_balance_power={source_balance_power}"
        )
        print(f"[INFO] train class counts: {ds_tr.class_counts_by_label()}")

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

    return dl_tr, dl_va, dl_te, ds_tr.label2id
