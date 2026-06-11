# scripts/lawyer_refine_weak_labels_v08.py
#
# LAWYER: Label-Aware Weak-label Yield Estimation and Refinement
#
# Config-driven version with signal-event support.
#
# Dataset-specific labels, source classes, and thresholds are loaded from JSON.
# For silence_present, this version can combine:
#   - TATA probability evidence
#   - acoustic binary column, e.g. silence_is_acoustic_silent
#   - acoustic RMS column, e.g. silence_rms_dbfs
#   - acoustic activity column, e.g. silence_speech_activity_ratio

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64, np.integer)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def safe_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).values


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def topk_mean(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    k = max(1, min(int(k), values.size))
    return float(np.mean(np.sort(values)[-k:]))


def score_to_zone(score: float, low: float, high: float) -> str:
    score = float(score)
    if score < float(low):
        return "reject_zone"
    if score > float(high):
        return "accept_zone"
    return "uncertain_zone"


def require_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Config must contain non-empty list: {key}")
    return [str(x) for x in value]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    labels = require_list(config, "labels")
    label_set = set(labels)

    groups = config.get("label_groups", {})
    if not isinstance(groups, dict):
        raise RuntimeError("Config key 'label_groups' must be a dictionary.")

    target_labels = [str(x) for x in groups.get("target_labels", [])]
    event_labels = [str(x) for x in groups.get("event_labels", [])]
    focus_labels = [str(x) for x in groups.get("focus_labels", [])]
    open_set_label = groups.get("open_set_label")
    if open_set_label is not None:
        open_set_label = str(open_set_label)

    for group_name, group_labels in [
        ("target_labels", target_labels),
        ("event_labels", event_labels),
        ("focus_labels", focus_labels),
    ]:
        missing = [lab for lab in group_labels if lab not in label_set]
        if missing:
            raise RuntimeError(f"Config group '{group_name}' contains labels not in labels: {missing}")

    if open_set_label is not None and open_set_label not in label_set:
        raise RuntimeError(f"open_set_label is not present in labels: {open_set_label}")

    config = dict(config)
    config["labels"] = labels
    config["label_groups"] = {
        **groups,
        "target_labels": target_labels,
        "event_labels": event_labels,
        "focus_labels": focus_labels,
        "open_set_label": open_set_label,
    }

    config.setdefault("source_matching", {})
    config["source_matching"].setdefault(
        "columns",
        ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"],
    )
    config["source_matching"].setdefault("known_non_target_source_classes", [])

    config.setdefault("rules", {})
    config["rules"].setdefault("speaker_identity", {})
    config["rules"]["speaker_identity"].setdefault("alpha", 0.70)
    config["rules"]["speaker_identity"].setdefault("threshold", 0.50)
    config["rules"]["speaker_identity"].setdefault("margin_threshold", 0.10)

    config["rules"].setdefault("open_set", {})
    config["rules"]["open_set"].setdefault("direct_threshold", 0.55)
    config["rules"]["open_set"].setdefault("speech_threshold", 0.55)
    config["rules"]["open_set"].setdefault("known_max_threshold", 0.35)
    config["rules"]["open_set"].setdefault("uncertain_low", 0.35)
    config["rules"]["open_set"].setdefault("uncertain_high", 0.60)

    config["rules"].setdefault("default_event", {})
    config["rules"]["default_event"].setdefault("aggregation", "max")
    config["rules"]["default_event"].setdefault("threshold", 0.50)

    config["rules"].setdefault("transient_events", {})
    config["rules"].setdefault("signal_events", {})

    config["rules"].setdefault("known_non_target_override", {})
    config["rules"]["known_non_target_override"].setdefault("enabled", True)
    config["rules"]["known_non_target_override"].setdefault("zero_target_labels", True)
    config["rules"]["known_non_target_override"].setdefault("force_open_set_label", True)

    config["rules"].setdefault("routing", {})
    config["rules"]["routing"].setdefault("other_only_decision", "accepted_with_warning")
    config["rules"]["routing"].setdefault("target_with_event_decision", "accepted_with_warning")
    config["rules"]["routing"].setdefault("clean_single_target_decision", "accepted")

    return config


def label_prob_col(label: str) -> str:
    return f"segment_prob_{label}"


def get_configured_threshold(config: dict[str, Any], label: str) -> float:
    groups = config["label_groups"]
    rules = config["rules"]

    if label in groups.get("target_labels", []):
        return float(rules["speaker_identity"].get("threshold", 0.50))
    if label == groups.get("open_set_label"):
        return float(rules["open_set"].get("direct_threshold", 0.55))
    if label in rules.get("transient_events", {}):
        return float(rules["transient_events"][label].get("threshold", 0.50))
    if label in rules.get("signal_events", {}):
        return float(rules["signal_events"][label].get("threshold", 0.50))

    return float(rules.get("default_event", {}).get("threshold", 0.50))


def active_label_text(row: dict[str, Any], labels: list[str]) -> str:
    return "|".join([lab for lab in labels if int(row.get(lab, 0)) == 1])


def is_known_non_target_source(row: dict[str, Any], config: dict[str, Any]) -> bool:
    source_cfg = config.get("source_matching", {})
    match_columns = [str(c) for c in source_cfg.get("columns", [])]
    non_target_classes = [str(c) for c in source_cfg.get("known_non_target_source_classes", [])]
    text = " ".join(str(row.get(c, "")) for c in match_columns)
    return any(cls in text for cls in non_target_classes)


def aggregate_probs(p: np.ndarray, aggregation: str, top_k: int = 2) -> float:
    if len(p) == 0:
        return 0.0
    aggregation = str(aggregation)
    if aggregation == "max":
        return float(np.max(p))
    if aggregation == "mean":
        return float(np.mean(p))
    if aggregation == "topk_mean":
        return topk_mean(p, int(top_k))
    raise RuntimeError(f"Unknown aggregation: {aggregation}")


def signal_acoustic_score(
    *,
    label: str,
    group: pd.DataFrame,
    rule: dict[str, Any],
    base_row: dict[str, Any],
) -> float:
    """
    Compute acoustic evidence for a signal-like label.

    Supported config fields:
      binary_col: acoustic binary flag column. If any segment has 1, acoustic score = 1.
      energy_col + energy_threshold: e.g. silence_rms_dbfs <= -45.
      activity_col + activity_threshold: e.g. silence_speech_activity_ratio <= 0.15.
      acoustic_logic: "and" or "or" for energy/activity tests.

    For silence, "and" is safer:
      low RMS AND low speech activity.
    """
    reasons = []
    evidence_flags = []

    binary_col = rule.get("binary_col")
    if binary_col and binary_col in group.columns:
        binary_vals = pd.to_numeric(group[binary_col], errors="coerce").fillna(0).astype(int)
        flag = bool((binary_vals == 1).any())
        evidence_flags.append(flag)
        reasons.append(f"binary_col={binary_col},any={int(flag)}")

    energy_col = rule.get("energy_col")
    activity_col = rule.get("activity_col")

    energy_flag = None
    activity_flag = None

    if energy_col and energy_col in group.columns:
        threshold = float(rule.get("energy_threshold", -45.0))
        vals = pd.to_numeric(group[energy_col], errors="coerce")
        energy_flag = bool((vals <= threshold).any())
        reasons.append(f"energy_col={energy_col},threshold={threshold},any={int(energy_flag)}")

    if activity_col and activity_col in group.columns:
        threshold = float(rule.get("activity_threshold", 0.15))
        vals = pd.to_numeric(group[activity_col], errors="coerce")
        activity_flag = bool((vals <= threshold).any())
        reasons.append(f"activity_col={activity_col},threshold={threshold},any={int(activity_flag)}")

    acoustic_logic = str(rule.get("acoustic_logic", "or")).lower().strip()
    if energy_flag is not None and activity_flag is not None:
        if acoustic_logic == "and":
            evidence_flags.append(bool(energy_flag and activity_flag))
        elif acoustic_logic == "or":
            evidence_flags.append(bool(energy_flag or activity_flag))
        else:
            raise RuntimeError(f"Unknown acoustic_logic for {label}: {acoustic_logic}")
    elif energy_flag is not None:
        evidence_flags.append(bool(energy_flag))
    elif activity_flag is not None:
        evidence_flags.append(bool(activity_flag))

    acoustic_score = 1.0 if any(evidence_flags) else 0.0

    base_row[f"lawyer_score_{label}_acoustic"] = float(acoustic_score)
    base_row[f"lawyer_{label}_acoustic_reason"] = ";".join(reasons) if reasons else "not_used"

    return float(acoustic_score)


def compute_label_score(
    *,
    label: str,
    group: pd.DataFrame,
    base_row: dict[str, Any],
    config: dict[str, Any],
) -> float:
    rules = config["rules"]
    transient_rules = rules.get("transient_events", {})
    signal_rules = rules.get("signal_events", {})

    p = safe_float_array(group[label_prob_col(label)])

    if label in transient_rules:
        rule = transient_rules[label]
        return aggregate_probs(
            p,
            str(rule.get("aggregation", "topk_mean")),
            int(rule.get("top_k", 2)),
        )

    if label in signal_rules:
        rule = signal_rules[label]
        tata_score = aggregate_probs(
            p,
            str(rule.get("aggregation", "max")),
            int(rule.get("top_k", 2)),
        )
        acoustic_score = signal_acoustic_score(
            label=label,
            group=group,
            rule=rule,
            base_row=base_row,
        )

        combine = str(rule.get("combine", "max")).lower().strip()
        if combine == "max":
            final_score = max(tata_score, acoustic_score)
        elif combine == "tata_only":
            final_score = tata_score
        elif combine == "acoustic_only":
            final_score = acoustic_score
        elif combine == "mean":
            final_score = float(np.mean([tata_score, acoustic_score]))
        else:
            raise RuntimeError(f"Unknown signal combine rule for {label}: {combine}")

        base_row[f"lawyer_score_{label}_tata"] = float(tata_score)
        return float(final_score)

    aggregation = str(rules.get("default_event", {}).get("aggregation", "max"))
    return aggregate_probs(
        p,
        aggregation,
        int(rules.get("default_event", {}).get("top_k", 2)),
    )


def aggregate_segment_evidence(seg_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labels = config["labels"]
    target_labels = config["label_groups"].get("target_labels", [])
    open_set_label = config["label_groups"].get("open_set_label")
    speaker_rule = config["rules"]["speaker_identity"]
    alpha = float(speaker_rule.get("alpha", 0.70))

    if "parent_clip_id" not in seg_df.columns:
        raise RuntimeError("segment_predictions CSV must contain parent_clip_id")

    missing_prob_cols = [label_prob_col(label) for label in labels if label_prob_col(label) not in seg_df.columns]
    if missing_prob_cols:
        raise RuntimeError(
            "segment_predictions CSV is missing required segment probability columns:\n"
            f"{missing_prob_cols}"
        )

    meta_cols = [str(c) for c in config.get("source_matching", {}).get("columns", [])]
    meta_cols = ["parent_clip_id", *[c for c in meta_cols if c != "parent_clip_id"]]

    rows = []

    for parent_id, group in seg_df.groupby("parent_clip_id", dropna=False):
        row = {"parent_clip_id": str(parent_id), "num_segments": int(len(group))}

        for col in meta_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]

        for label in labels:
            p = safe_float_array(group[label_prob_col(label)])
            row[f"lawyer_seg_max_{label}"] = float(np.max(p)) if len(p) else 0.0
            row[f"lawyer_seg_mean_{label}"] = float(np.mean(p)) if len(p) else 0.0
            row[f"lawyer_seg_top2_{label}"] = topk_mean(p, 2)

        for label in target_labels:
            mean_p = float(row[f"lawyer_seg_mean_{label}"])
            max_p = float(row[f"lawyer_seg_max_{label}"])
            row[f"lawyer_score_{label}"] = clip01(alpha * mean_p + (1.0 - alpha) * max_p)

        target_scores = [float(row[f"lawyer_score_{label}"]) for label in target_labels]
        target_sorted = sorted(target_scores, reverse=True)
        row["lawyer_target_max_score"] = float(target_sorted[0]) if target_sorted else 0.0
        row["lawyer_target_second_score"] = float(target_sorted[1]) if len(target_sorted) > 1 else 0.0
        row["lawyer_target_margin"] = float(row["lawyer_target_max_score"] - row["lawyer_target_second_score"])

        if open_set_label:
            other_direct = float(row[f"lawyer_seg_max_{open_set_label}"])
            max_target_seg = max([float(row[f"lawyer_seg_max_{label}"]) for label in target_labels], default=0.0)
            speech_like = max(other_direct, max_target_seg)
            known_max = float(row["lawyer_target_max_score"])
            open_set_score = clip01(speech_like * (1.0 - known_max))

            row[f"lawyer_score_{open_set_label}_direct"] = other_direct
            row["lawyer_score_speech_like"] = speech_like
            row[f"lawyer_score_{open_set_label}_open_set"] = open_set_score
            row["lawyer_is_known_non_target_source"] = int(is_known_non_target_source(row, config))

            source_bonus = 1.0 if row["lawyer_is_known_non_target_source"] == 1 else 0.0
            row[f"lawyer_score_{open_set_label}"] = max(other_direct, open_set_score, source_bonus)

        for label in labels:
            if label in target_labels:
                continue
            if label == open_set_label:
                continue
            row[f"lawyer_score_{label}"] = compute_label_score(
                label=label,
                group=group,
                base_row=row,
                config=config,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def attach_parent_context(evidence_df: pd.DataFrame, parent_csv: Path | None) -> pd.DataFrame:
    if parent_csv is None:
        return evidence_df
    if not parent_csv.exists():
        raise FileNotFoundError(f"parent_csv not found: {parent_csv}")

    parent_df = pd.read_csv(parent_csv, low_memory=False)
    if "parent_clip_id" not in parent_df.columns:
        raise RuntimeError("parent_csv must contain parent_clip_id")

    parent_df = parent_df.copy()
    parent_df["parent_clip_id"] = parent_df["parent_clip_id"].astype(str)

    evidence_df = evidence_df.copy()
    evidence_df["parent_clip_id"] = evidence_df["parent_clip_id"].astype(str)

    keep_cols = ["parent_clip_id"] + [
        c for c in parent_df.columns
        if c != "parent_clip_id" and c not in evidence_df.columns
    ]

    return evidence_df.merge(parent_df[keep_cols], on="parent_clip_id", how="left")


def get_configured_threshold(config: dict[str, Any], label: str) -> float:
    groups = config["label_groups"]
    rules = config["rules"]

    if label in groups.get("target_labels", []):
        return float(rules["speaker_identity"].get("threshold", 0.50))
    if label == groups.get("open_set_label"):
        return float(rules["open_set"].get("direct_threshold", 0.55))
    if label in rules.get("transient_events", {}):
        return float(rules["transient_events"][label].get("threshold", 0.50))
    if label in rules.get("signal_events", {}):
        return float(rules["signal_events"][label].get("threshold", 0.50))
    return float(rules.get("default_event", {}).get("threshold", 0.50))


def apply_lawyer_decisions(evidence_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labels = config["labels"]
    groups = config["label_groups"]
    rules = config["rules"]

    target_labels = groups.get("target_labels", [])
    event_labels = groups.get("event_labels", [])
    open_set_label = groups.get("open_set_label")

    speaker_rule = rules["speaker_identity"]
    open_rule = rules["open_set"]
    override_rule = rules["known_non_target_override"]
    routing_rule = rules["routing"]

    target_threshold = float(speaker_rule.get("threshold", 0.50))
    target_margin_threshold = float(speaker_rule.get("margin_threshold", 0.10))

    rows = []

    for _, row in evidence_df.iterrows():
        out = row.to_dict()
        uncertain_reasons = []
        routing_reasons = []

        is_known_non_target = int(out.get("lawyer_is_known_non_target_source", 0)) == 1

        target_active = []
        target_scores = {}

        for label in target_labels:
            score = float(out.get(f"lawyer_score_{label}", 0.0))
            target_scores[label] = score
            out[label] = int(score >= target_threshold)
            out[f"parent_pred_{label}"] = int(out[label])
            if out[label] == 1:
                target_active.append(label)

        if target_scores:
            best_target = max(target_scores, key=target_scores.get)
            best_target_score = float(target_scores[best_target])
            sorted_target_scores = sorted(target_scores.values(), reverse=True)
            second_target_score = float(sorted_target_scores[1]) if len(sorted_target_scores) > 1 else 0.0
        else:
            best_target = ""
            best_target_score = 0.0
            second_target_score = 0.0

        target_margin = best_target_score - second_target_score

        out["lawyer_best_target_label"] = best_target
        out["lawyer_best_target_score"] = best_target_score
        out["lawyer_second_target_score"] = second_target_score
        out["lawyer_target_margin_after_refine"] = target_margin

        if (
            is_known_non_target
            and bool(override_rule.get("enabled", True))
            and bool(override_rule.get("zero_target_labels", True))
        ):
            for label in target_labels:
                out[label] = 0
                out[f"parent_pred_{label}"] = 0

            target_active = []
            routing_reasons.append("known_non_target_targets_forced_zero")
        else:
            if len(target_active) > 1 and target_margin < target_margin_threshold:
                uncertain_reasons.append("multi_target_low_margin")

        if open_set_label:
            direct_other = float(out.get(f"lawyer_score_{open_set_label}_direct", 0.0))
            speech_like = float(out.get("lawyer_score_speech_like", 0.0))
            known_max = float(out.get("lawyer_target_max_score", best_target_score))

            other_direct_threshold = float(open_rule.get("direct_threshold", 0.55))
            other_speech_threshold = float(open_rule.get("speech_threshold", 0.55))
            other_known_max_threshold = float(open_rule.get("known_max_threshold", 0.35))

            open_set_condition = speech_like >= other_speech_threshold and known_max <= other_known_max_threshold

            open_positive = (
                (
                    is_known_non_target
                    and bool(override_rule.get("enabled", True))
                    and bool(override_rule.get("force_open_set_label", True))
                )
                or direct_other >= other_direct_threshold
                or open_set_condition
            )

            out[open_set_label] = int(open_positive)
            out[f"parent_pred_{open_set_label}"] = int(open_positive)

            if is_known_non_target and bool(override_rule.get("force_open_set_label", True)):
                routing_reasons.append("known_non_target_forced_open_set")
            elif open_set_condition:
                routing_reasons.append("open_set_speech_low_known_target_rule")
            elif direct_other >= other_direct_threshold:
                routing_reasons.append("direct_open_set_probability_rule")

            open_score = float(out.get(f"lawyer_score_{open_set_label}", 0.0))
            out[f"lawyer_zone_{open_set_label}"] = score_to_zone(
                open_score,
                float(open_rule.get("uncertain_low", 0.35)),
                float(open_rule.get("uncertain_high", 0.60)),
            )

            if out[f"lawyer_zone_{open_set_label}"] == "uncertain_zone" and not is_known_non_target and not open_set_condition:
                uncertain_reasons.append(f"{open_set_label}_uncertain")

        for label in labels:
            if label in target_labels:
                continue
            if label == open_set_label:
                continue

            score = float(out.get(f"lawyer_score_{label}", 0.0))
            threshold = get_configured_threshold(config, label)

            out[label] = int(score >= threshold)
            out[f"parent_pred_{label}"] = int(out[label])

            zone_rule = None
            if label in rules.get("transient_events", {}):
                zone_rule = rules["transient_events"][label]
            elif label in rules.get("signal_events", {}):
                zone_rule = rules["signal_events"][label]

            if zone_rule is not None and "uncertain_low" in zone_rule and "uncertain_high" in zone_rule:
                zone = score_to_zone(score, float(zone_rule["uncertain_low"]), float(zone_rule["uncertain_high"]))
                out[f"lawyer_zone_{label}"] = zone
                if zone == "uncertain_zone":
                    uncertain_reasons.append(f"{label}_uncertain")

        num_active = int(sum(int(out.get(label, 0)) for label in labels))
        target_count = int(sum(int(out.get(label, 0)) for label in target_labels))
        event_count = int(sum(int(out.get(label, 0)) for label in event_labels))
        open_active = bool(open_set_label and int(out.get(open_set_label, 0)) == 1)

        out["num_active_labels"] = num_active
        out["labels"] = active_label_text(out, labels)
        out["manual_labels"] = out["labels"]
        out["lawyer_uncertain_reasons"] = "|".join(sorted(set(uncertain_reasons)))
        out["lawyer_routing_reason_detail"] = "|".join(sorted(set(routing_reasons)))

        if uncertain_reasons:
            out["routing_decision"] = "needs_review"
            out["routing_reason"] = "lawyer_uncertain_" + "|".join(sorted(set(uncertain_reasons)))
        elif num_active == 0:
            out["routing_decision"] = "rejected"
            out["routing_reason"] = "lawyer_no_reliable_label"
        elif target_count == 1 and not open_active and event_count == 0:
            out["routing_decision"] = str(routing_rule.get("clean_single_target_decision", "accepted"))
            out["routing_reason"] = "lawyer_single_target_clean"
        else:
            if open_active and target_count == 0:
                out["routing_decision"] = str(routing_rule.get("other_only_decision", "accepted_with_warning"))
                out["routing_reason"] = "lawyer_open_set_only_high_confidence"
            elif event_count > 0:
                out["routing_decision"] = str(routing_rule.get("target_with_event_decision", "accepted_with_warning"))
                out["routing_reason"] = "lawyer_target_or_open_set_with_context_event"
            else:
                out["routing_decision"] = "accepted_with_warning"
                out["routing_reason"] = "lawyer_multi_label_warning"

        out["routing_mode"] = str(config.get("mode_name", "lawyer_v08"))
        out["review_status"] = "lawyer_pseudo_routed"

        if "notes" not in out or pd.isna(out.get("notes")):
            out["notes"] = ""

        rows.append(out)

    return pd.DataFrame(rows)


def reorder_output_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labels = config["labels"]

    front = [
        "parent_clip_id", "source_file", "source_path", "source_rel_path", "source_class_dir",
        "routing_mode", "routing_decision", "routing_reason", "labels",
        *labels,
        "manual_labels", "review_status", "notes", "num_segments", "num_active_labels",
        "lawyer_best_target_label", "lawyer_best_target_score", "lawyer_second_target_score",
        "lawyer_target_margin_after_refine", "lawyer_is_known_non_target_source",
        "lawyer_uncertain_reasons", "lawyer_routing_reason_detail",
    ]

    score_cols = []
    for label in labels:
        score_cols.extend([
            f"lawyer_score_{label}", f"lawyer_score_{label}_direct", f"lawyer_score_{label}_open_set",
            f"lawyer_score_{label}_tata", f"lawyer_score_{label}_acoustic", f"lawyer_{label}_acoustic_reason",
            f"lawyer_seg_max_{label}", f"lawyer_seg_mean_{label}", f"lawyer_seg_top2_{label}",
            f"lawyer_zone_{label}", f"parent_pred_{label}",
        ])

    extra_cols = ["lawyer_score_speech_like", "lawyer_target_max_score", "lawyer_target_second_score", "lawyer_target_margin"]

    ordered = [c for c in front + score_cols + extra_cols if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LAWYER v0.8 Weak-label Refinement Summary",
        "",
        "LAWYER = **Label-Aware Weak-label Yield Estimation and Refinement**.",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Configuration",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Mode: `{summary['mode_name']}`",
        "",
        "## Counts",
        "",
        f"- Segment rows: `{summary['segment_rows']}`",
        f"- Parent clips: `{summary['parent_rows']}`",
        f"- Known non-target source rows: `{summary.get('known_non_target_source_rows', 'n/a')}`",
        f"- Known non-target rows with any target label still active: `{summary.get('known_non_target_rows_with_any_target_active', 'n/a')}`",
        "",
        "### Routing counts",
        "",
        "| Decision | Rows |",
        "|---|---:|",
    ]

    for key, value in summary["routing_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "### Label counts after LAWYER", "", "| Label | Positive parent clips |", "|---|---:|"])
    for label, count in summary["label_counts"].items():
        lines.append(f"| `{label}` | {count} |")

    lines.extend(["", "### Focus label counts", "", "| Label | Positive parent clips |", "|---|---:|"])
    for label, count in summary["focus_label_counts"].items():
        lines.append(f"| `{label}` | {count} |")

    if "signal_label_counts" in summary:
        lines.extend(["", "### Signal label details", "", "| Item | Count |", "|---|---:|"])
        for key, value in summary["signal_label_counts"].items():
            lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    mode_name: str,
    config: dict[str, Any],
    summary_extra: dict[str, Any],
) -> dict[str, Any]:
    labels = config["labels"]
    focus_labels = config["label_groups"].get("focus_labels", [])
    target_labels = config["label_groups"].get("target_labels", [])

    out_dir.mkdir(parents=True, exist_ok=True)
    df = reorder_output_columns(df, config)

    paths = {
        "all_parent_labels": out_dir / f"{mode_name}_parent_labels_all.csv",
        "accepted": out_dir / f"{mode_name}_accepted.csv",
        "accepted_with_warning": out_dir / f"{mode_name}_accepted_with_warning.csv",
        "needs_review": out_dir / f"{mode_name}_needs_review.csv",
        "rejected": out_dir / f"{mode_name}_rejected.csv",
        "manual_review_prefill": out_dir / f"{mode_name}_manual_review_prefill.csv",
        "config_snapshot": out_dir / f"{mode_name}_config_snapshot.json",
        "summary_json": out_dir / f"{mode_name}_summary.json",
        "summary_md": out_dir / f"{mode_name}_summary.md",
    }

    df.to_csv(paths["all_parent_labels"], index=False)
    df[df["routing_decision"] == "accepted"].to_csv(paths["accepted"], index=False)
    df[df["routing_decision"] == "accepted_with_warning"].to_csv(paths["accepted_with_warning"], index=False)
    df[df["routing_decision"] == "needs_review"].to_csv(paths["needs_review"], index=False)
    df[df["routing_decision"] == "rejected"].to_csv(paths["rejected"], index=False)
    df[df["routing_decision"] == "needs_review"].to_csv(paths["manual_review_prefill"], index=False)

    save_json(config, paths["config_snapshot"])

    known_non_target_col = pd.to_numeric(df.get("lawyer_is_known_non_target_source", 0), errors="coerce").fillna(0).astype(int)

    if target_labels:
        target_sum = df[target_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum(axis=1)
    else:
        target_sum = pd.Series([0] * len(df), index=df.index)

    signal_label_counts = {}
    for label in config["rules"].get("signal_events", {}):
        col_acoustic = f"lawyer_score_{label}_acoustic"
        col_tata = f"lawyer_score_{label}_tata"
        if col_acoustic in df.columns:
            signal_label_counts[f"{label}_acoustic_positive"] = int((pd.to_numeric(df[col_acoustic], errors="coerce").fillna(0) >= 1.0).sum())
        if col_tata in df.columns:
            signal_label_counts[f"{label}_tata_positive_at_threshold"] = int((pd.to_numeric(df[col_tata], errors="coerce").fillna(0) >= get_configured_threshold(config, label)).sum())

    summary = {
        "generated_at": now_iso(),
        **summary_extra,
        "parent_rows": int(len(df)),
        "routing_counts": {str(k): int(v) for k, v in df["routing_decision"].value_counts().to_dict().items()},
        "label_counts": {label: int(pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).sum()) for label in labels},
        "focus_label_counts": {label: int(pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).sum()) for label in focus_labels if label in df.columns},
        "known_non_target_source_rows": int(known_non_target_col.sum()),
        "known_non_target_rows_with_any_target_active": int(((known_non_target_col == 1) & (target_sum > 0)).sum()),
        "signal_label_counts": signal_label_counts,
        "outputs": {key: str(value) for key, value in paths.items()},
    }

    save_json(summary, paths["summary_json"])
    write_summary_md(paths["summary_md"], summary)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LAWYER v0.8 config-driven label-aware weak-label refinement.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--segment_predictions_csv", required=True)
    parser.add_argument("--parent_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode_name", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    segment_predictions_csv = Path(args.segment_predictions_csv)
    parent_csv = Path(args.parent_csv) if args.parent_csv else None
    out_dir = Path(args.out_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not segment_predictions_csv.exists():
        raise FileNotFoundError(f"segment_predictions_csv not found: {segment_predictions_csv}")

    config = validate_config(load_json(config_path))
    mode_name = str(args.mode_name or config.get("mode_name", "lawyer_v08"))
    config["mode_name"] = mode_name

    print("")
    print("LAWYER v0.8 config-driven weak-label refinement")
    print("-" * 90)
    print(f"Config:              {config_path}")
    print(f"Segment predictions: {segment_predictions_csv}")
    print(f"Parent context CSV:   {parent_csv if parent_csv else '(not provided)'}")
    print(f"Output dir:           {out_dir}")
    print(f"Mode name:            {mode_name}")
    print("-" * 90)

    seg_df = pd.read_csv(segment_predictions_csv, low_memory=False)
    evidence = aggregate_segment_evidence(seg_df, config)
    evidence = attach_parent_context(evidence, parent_csv)
    refined = apply_lawyer_decisions(evidence, config)

    summary_extra = {
        "method": "LAWYER: Label-Aware Weak-label Yield Estimation and Refinement",
        "implementation": "config_driven_with_silence_signal",
        "config_path": str(config_path),
        "segment_predictions_csv": str(segment_predictions_csv),
        "parent_csv": str(parent_csv) if parent_csv else None,
        "out_dir": str(out_dir),
        "mode_name": mode_name,
        "segment_rows": int(len(seg_df)),
    }

    summary = write_outputs(df=refined, out_dir=out_dir, mode_name=mode_name, config=config, summary_extra=summary_extra)

    print("")
    print("LAWYER refinement complete")
    print("-" * 90)
    print(f"Parent rows: {summary['parent_rows']}")
    print("")
    print("Routing counts:")
    print(pd.Series(summary["routing_counts"]).to_string())
    print("")
    print("Focus label counts:")
    print(pd.Series(summary["focus_label_counts"]).to_string())
    print("")
    print("Known non-target check:")
    print(f"  known_non_target_source_rows = {summary['known_non_target_source_rows']}")
    print(f"  known_non_target_rows_with_any_target_active = {summary['known_non_target_rows_with_any_target_active']}")
    if summary.get("signal_label_counts"):
        print("")
        print("Signal label details:")
        print(pd.Series(summary["signal_label_counts"]).to_string())
    print("")
    print(f"Summary: {summary['outputs']['summary_md']}")


if __name__ == "__main__":
    main()
