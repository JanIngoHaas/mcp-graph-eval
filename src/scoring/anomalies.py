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


@dataclass
class AnomalyRules:
    triple_pos_trace_zero: bool = True
    trace_pos_triple_zero: bool = True
    both_zero: bool = True
    triple_perfect_trace_not: bool = False
    literal_subject_mismatch: bool = False


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
        },
        "rules": {
            "triple_pos_trace_zero": cfg.rules.triple_pos_trace_zero,
            "trace_pos_triple_zero": cfg.rules.trace_pos_triple_zero,
            "both_zero": cfg.rules.both_zero,
            "triple_perfect_trace_not": cfg.rules.triple_perfect_trace_not,
            "literal_subject_mismatch": cfg.rules.literal_subject_mismatch,
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
        ),
        rules=AnomalyRules(
            triple_pos_trace_zero=bool(rules.get("triple_pos_trace_zero", DEFAULT_CONFIG.rules.triple_pos_trace_zero)),
            trace_pos_triple_zero=bool(rules.get("trace_pos_triple_zero", DEFAULT_CONFIG.rules.trace_pos_triple_zero)),
            both_zero=bool(rules.get("both_zero", DEFAULT_CONFIG.rules.both_zero)),
            triple_perfect_trace_not=bool(rules.get("triple_perfect_trace_not", DEFAULT_CONFIG.rules.triple_perfect_trace_not)),
            literal_subject_mismatch=bool(rules.get("literal_subject_mismatch", DEFAULT_CONFIG.rules.literal_subject_mismatch)),
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
    qtype: str | None,
    config: AnomalyConfig,
    expected_triples: list[dict] | None = None,
    received_triples: list[dict] | None = None,
) -> list[str]:
    if qtype and qtype in config.ignore_qtypes:
        return []

    t = config.thresholds
    r = config.rules
    flags: list[str] = []

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
