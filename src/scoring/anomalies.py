"""Anomaly detection for scoring mismatches."""

from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path

try:
    import tomllib
except Exception:  # pragma: no cover - fallback for older Python
    tomllib = None


@dataclass
class AnomalyThresholds:
    trace_zero: float = 0.05
    triple_zero: float = 0.05
    trace_high: float = 0.6
    both_zero: float = 0.02
    trace_perfect: float = 0.99
    triple_perfect: float = 0.99
    triple_precision_low: float = 0.1
    triple_recall_high: float = 0.9
    overcite_ratio: float = 3.0
    overcite_min_extra: int = 5


@dataclass
class AnomalyRules:
    triple_pos_trace_zero: bool = True
    trace_pos_triple_zero: bool = True
    both_zero: bool = True
    triple_perfect_trace_not: bool = False
    literal_subject_mismatch: bool = False
    overcitation_low_precision_high_recall: bool = True
    overcitation_ratio: bool = True
    impossible_found_not_false: bool = True


@dataclass
class AnomalyConfig:
    thresholds: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    rules: AnomalyRules = field(default_factory=AnomalyRules)
    ignore_qtypes: list[str] = field(default_factory=lambda: ["impossible"])


DEFAULT_CONFIG = AnomalyConfig()


def _merge_dict(target: dict, updates: dict) -> dict:
    out = dict(target)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _config_to_dict(cfg: AnomalyConfig) -> dict[str, Any]:
    return {
        "thresholds": {
            "trace_zero": cfg.thresholds.trace_zero,
            "triple_zero": cfg.thresholds.triple_zero,
            "trace_high": cfg.thresholds.trace_high,
            "both_zero": cfg.thresholds.both_zero,
            "trace_perfect": cfg.thresholds.trace_perfect,
            "triple_perfect": cfg.thresholds.triple_perfect,
            "triple_precision_low": cfg.thresholds.triple_precision_low,
            "triple_recall_high": cfg.thresholds.triple_recall_high,
            "overcite_ratio": cfg.thresholds.overcite_ratio,
            "overcite_min_extra": cfg.thresholds.overcite_min_extra,
        },
        "rules": {
            "triple_pos_trace_zero": cfg.rules.triple_pos_trace_zero,
            "trace_pos_triple_zero": cfg.rules.trace_pos_triple_zero,
            "both_zero": cfg.rules.both_zero,
            "triple_perfect_trace_not": cfg.rules.triple_perfect_trace_not,
            "literal_subject_mismatch": cfg.rules.literal_subject_mismatch,
            "overcitation_low_precision_high_recall": cfg.rules.overcitation_low_precision_high_recall,
            "overcitation_ratio": cfg.rules.overcitation_ratio,
            "impossible_found_not_false": cfg.rules.impossible_found_not_false,
        },
        "ignore_qtypes": list(cfg.ignore_qtypes),
    }


def _dict_to_config(data: dict[str, Any]) -> AnomalyConfig:
    thresholds = data.get("thresholds", {}) or {}
    rules = data.get("rules", {}) or {}
    return AnomalyConfig(
        thresholds=AnomalyThresholds(
            trace_zero=float(thresholds.get("trace_zero", DEFAULT_CONFIG.thresholds.trace_zero)),
            triple_zero=float(thresholds.get("triple_zero", DEFAULT_CONFIG.thresholds.triple_zero)),
            trace_high=float(thresholds.get("trace_high", DEFAULT_CONFIG.thresholds.trace_high)),
            both_zero=float(thresholds.get("both_zero", DEFAULT_CONFIG.thresholds.both_zero)),
            trace_perfect=float(thresholds.get("trace_perfect", DEFAULT_CONFIG.thresholds.trace_perfect)),
            triple_perfect=float(thresholds.get("triple_perfect", DEFAULT_CONFIG.thresholds.triple_perfect)),
            triple_precision_low=float(thresholds.get("triple_precision_low", DEFAULT_CONFIG.thresholds.triple_precision_low)),
            triple_recall_high=float(thresholds.get("triple_recall_high", DEFAULT_CONFIG.thresholds.triple_recall_high)),
            overcite_ratio=float(thresholds.get("overcite_ratio", DEFAULT_CONFIG.thresholds.overcite_ratio)),
            overcite_min_extra=int(thresholds.get("overcite_min_extra", DEFAULT_CONFIG.thresholds.overcite_min_extra)),
        ),
        rules=AnomalyRules(
            triple_pos_trace_zero=bool(rules.get("triple_pos_trace_zero", DEFAULT_CONFIG.rules.triple_pos_trace_zero)),
            trace_pos_triple_zero=bool(rules.get("trace_pos_triple_zero", DEFAULT_CONFIG.rules.trace_pos_triple_zero)),
            both_zero=bool(rules.get("both_zero", DEFAULT_CONFIG.rules.both_zero)),
            triple_perfect_trace_not=bool(rules.get("triple_perfect_trace_not", DEFAULT_CONFIG.rules.triple_perfect_trace_not)),
            literal_subject_mismatch=bool(rules.get("literal_subject_mismatch", DEFAULT_CONFIG.rules.literal_subject_mismatch)),
            overcitation_low_precision_high_recall=bool(rules.get("overcitation_low_precision_high_recall", DEFAULT_CONFIG.rules.overcitation_low_precision_high_recall)),
            overcitation_ratio=bool(rules.get("overcitation_ratio", DEFAULT_CONFIG.rules.overcitation_ratio)),
            impossible_found_not_false=bool(rules.get("impossible_found_not_false", DEFAULT_CONFIG.rules.impossible_found_not_false)),
        ),
        ignore_qtypes=list(data.get("ignore_qtypes", DEFAULT_CONFIG.ignore_qtypes)),
    )


def load_anomaly_config(path: Path | None) -> AnomalyConfig:
    if not path:
        return DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Anomaly config not found: {path}")
    suffix = path.suffix.lower()
    raw: dict[str, Any]
    if suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("tomllib not available to read TOML config")
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if "anomalies" in raw and isinstance(raw.get("anomalies"), dict):
        raw = raw.get("anomalies") or {}
    merged = _merge_dict(_config_to_dict(DEFAULT_CONFIG), raw)
    return _dict_to_config(merged)


def detect_anomalies(
    trace_f1: float,
    triple_f1: float,
    triple_precision: float,
    triple_recall: float,
    qtype: str | None,
    config: AnomalyConfig,
    expected_triples: list[dict] | None = None,
    received_triples: list[dict] | None = None,
    explain_found: bool | None = None,
) -> list[str]:
    t = config.thresholds
    r = config.rules
    flags: list[str] = []
    expected_count = len(expected_triples or [])
    received_count = len(received_triples or [])

    # Keep impossible validation explicit even when impossible qtypes are otherwise ignored.
    if r.impossible_found_not_false and qtype == "impossible" and explain_found is not False:
        flags.append("impossible_found_not_false")

    if qtype and qtype in config.ignore_qtypes:
        return flags

    if r.triple_pos_trace_zero and (triple_f1 > t.triple_zero and trace_f1 <= t.trace_zero):
        flags.append("triple_pos_trace_zero")
    if r.trace_pos_triple_zero and (trace_f1 >= t.trace_high and triple_f1 <= t.triple_zero):
        flags.append("trace_pos_triple_zero")
    if r.both_zero and (trace_f1 <= t.both_zero and triple_f1 <= t.both_zero):
        flags.append("both_zero")
    if r.triple_perfect_trace_not and (triple_f1 >= t.triple_perfect and trace_f1 < t.trace_perfect):
        flags.append("triple_perfect_trace_not")

    if r.literal_subject_mismatch and expected_triples and received_triples:
        if _detect_literal_subject_mismatch(expected_triples, received_triples):
            flags.append("literal_subject_mismatch")

    if (
        r.overcitation_low_precision_high_recall
        and expected_count > 0
        and received_count > 0
        and triple_recall >= t.triple_recall_high
        and triple_precision <= t.triple_precision_low
        and (received_count - expected_count) >= t.overcite_min_extra
    ):
        flags.append("overcitation_low_precision_high_recall")

    if (
        r.overcitation_ratio
        and expected_count > 0
        and received_count >= expected_count
        and (received_count - expected_count) >= t.overcite_min_extra
        and (received_count / expected_count) >= t.overcite_ratio
    ):
        flags.append("overcitation_ratio")

    return flags


def _detect_literal_subject_mismatch(
    expected_triples: list[dict],
    received_triples: list[dict],
) -> bool:
    expected_map: dict[str, set[str]] = {}
    received_map: dict[str, set[str]] = {}

    for t in expected_triples:
        obj = t.get("object")
        subj = t.get("subject")
        if not isinstance(obj, str) or not isinstance(subj, str):
            continue
        expected_map.setdefault(obj, set()).add(subj)

    for t in received_triples:
        obj = t.get("object")
        subj = t.get("subject")
        if not isinstance(obj, str) or not isinstance(subj, str):
            continue
        received_map.setdefault(obj, set()).add(subj)

    for lit in expected_map.keys() & received_map.keys():
        if expected_map[lit].isdisjoint(received_map[lit]):
            return True
    return False
