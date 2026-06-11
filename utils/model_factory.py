from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from utils.config import load_config


def _normalize_hint_cfg(model_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model_cfg = model_cfg or {}
    hint_cfg = model_cfg.get("exit_hint", {}) or {}

    enable = bool(hint_cfg.get("enable", False))
    hint_dim = int(hint_cfg.get("dim", 0)) if enable else 0
    if hint_dim < 0:
        hint_dim = 0

    return {
        "hint_dim": hint_dim,
        "hint_source": str(hint_cfg.get("source", "probs")),
        "hint_detach": bool(hint_cfg.get("detach", True)),
        "hint_use_stats": bool(hint_cfg.get("use_stats", True)),
    }


def load_run_model_cfg(run_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return {}

    try:
        cfg = load_config(cfg_path) or {}
    except Exception:
        return {}

    return cfg.get("model") or {}


def build_audio_exit_net(
    *,
    num_classes: int,
    n_mels: int,
    tap_blocks: Sequence[int],
    model_cfg: Optional[Dict[str, Any]] = None,
):
    backbone = TinyAudioCNN(n_mels=n_mels, tap_blocks=tap_blocks)

    hint_kwargs = _normalize_hint_cfg(model_cfg)

    model = ExitNet(
        backbone=backbone,
        num_classes=num_classes,
        tap_dims=backbone.tap_dims,
        final_dim=backbone.final_dim,
        **hint_kwargs,
    )
    return model