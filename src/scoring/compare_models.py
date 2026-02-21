"""Cross-model comparison focused on combined report artifacts only."""

from __future__ import annotations

import re
import shutil
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_triple_f1
from .reporting import (
    render_cross_dataset_explain_latex,
    render_cross_dataset_prf_latex,
    render_markdown_table,
    write_pareto_svg,
)

RDF_TYPE_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
EXPLANATION_TOOLS = {"cite", "explain"}


@dataclass
class ModelPricing:
    input_per_1m_usd: float
    output_per_1m_usd: float


@dataclass
class RowStats:
    model: str
    source_file: str
    question_id: str
    question: str
    qtype: str
    triple_precision: float
    triple_recall: float
    triple_f1: float
    triples_matched: int
    triples_expected: int
    triples_received: int
    runtime_tokens_total: float
    runtime_tokens_cite_explain: float
    runtime_input_tokens: float
    runtime_output_tokens: float
    elapsed_s: float
    found_flag: bool | None
    has_citation: bool
    answer: str


@dataclass
class ParetoPoint:
    model: str
    f1_without_impossible: float
    cost_per_question_usd: float
    mean_latency_s: float
    cost_frontier: bool
    latency_frontier: bool


@dataclass
class DatasetView:
    dataset: str
    table_a_by_model: dict[str, dict[str, str]]
    table_b_by_model: dict[str, dict[str, str]]
    aggregate_by_model: dict[str, dict[str, float]]
    benchmark_headers: list[str]
    benchmark_rows: list[list[str]]


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _usage_counts(usage: Any) -> tuple[float, float, float]:
    if not isinstance(usage, dict):
        return 0.0, 0.0, 0.0
    inp = _to_float(usage.get("input_tokens"))
    out = _to_float(usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return inp, out, float(total)
    return inp, out, inp + out


def _normalize_tool_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    return re.sub(r"[^a-z0-9_]+", "", text)


def _has_citation_call(row: dict[str, Any]) -> bool:
    runtime_trace = row.get("runtime_trace")
    if not isinstance(runtime_trace, list):
        return False
    for event in runtime_trace:
        if not isinstance(event, dict):
            continue
        calls = event.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            if _normalize_tool_name(call.get("name")).startswith("cite"):
                return True
    return False


def _scorable_triples(side: dict[str, Any]) -> list[dict[str, Any]]:
    triples = side.get("triples", []) or []
    return [
        t
        for t in triples
        if isinstance(t, dict)
        and str(t.get("predicate", "")) != RDF_TYPE_PREDICATE
    ]


def _runtime_token_breakdown(runtime_trace: list[Any]) -> tuple[float, float, float, float]:
    """Return (total_tokens, cite_explain_tokens, input_tokens, output_tokens)."""
    total_tokens = 0.0
    explain_tokens = 0.0
    input_tokens = 0.0
    output_tokens = 0.0

    for event in runtime_trace:
        if not isinstance(event, dict):
            continue
        usage = event.get("usage") or event.get("token_usage")
        inp, out, total = _usage_counts(usage)
        input_tokens += inp
        output_tokens += out
        total_tokens += total

        calls = event.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = []

        if total <= 0 or not calls:
            continue

        token_per_call = total / len(calls)
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip().lower()
            if name in EXPLANATION_TOOLS:
                explain_tokens += token_per_call

    return total_tokens, explain_tokens, input_tokens, output_tokens


def _load_pricing(pricing_file: Path | None) -> dict[str, ModelPricing]:
    if pricing_file is None:
        return {}
    if not pricing_file.exists():
        raise FileNotFoundError(f"Pricing file not found: {pricing_file}")

    pricing: dict[str, ModelPricing] = {}
    with pricing_file.open(newline="") as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            pricing[model] = ModelPricing(
                input_per_1m_usd=float(row.get("input_per_1m", 0.0) or 0.0),
                output_per_1m_usd=float(row.get("output_per_1m", 0.0) or 0.0),
            )
    return pricing


def _load_eval_file(path: Path) -> tuple[str, list[RowStats]]:
    with path.open("rb") as f:
        data = tomllib.load(f)

    meta: dict[str, Any] = {}
    rows: list[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("meta"), dict):
            meta = data.get("meta", {})  # legacy schema
        elif isinstance(data.get("metadata"), dict):
            meta = data.get("metadata", {})  # current schema
        if isinstance(data.get("rows"), list):
            rows = data.get("rows", [])  # legacy schema
        elif isinstance(data.get("output"), list):
            rows = data.get("output", [])  # current schema

    model = str(meta.get("model", ""))
    if not model:
        model = path.stem.replace("eval_results_", "")

    parsed: list[RowStats] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qtype = str(row.get("qtype", "unknown"))
        expected = row.get("expected", {}) if isinstance(row.get("expected"), dict) else {}
        received = row.get("received", {}) if isinstance(row.get("received"), dict) else {}

        score = compute_triple_f1(_scorable_triples(expected), _scorable_triples(received))
        precision = float(score.precision)
        recall = float(score.recall)
        f1 = float(score.f1)
        matched = int(score.matched)
        expected_count = int(score.expected_count)
        received_count = int(score.received_count)

        found_flag = _extract_found_flag(row)
        has_citation = _has_citation_call(row)

        if qtype.lower() == "impossible":
            # For impossible questions, score by whether the model explicitly marked not found.
            correct_not_found = found_flag is False
            val = 1.0 if correct_not_found else 0.0
            precision = val
            recall = val
            f1 = val
            matched = int(correct_not_found)
            expected_count = 1
            received_count = 1

        # Enforce citation compliance as a hard gate on F1.
        if not has_citation:
            f1 = 0.0

        runtime_trace = row.get("runtime_trace") if isinstance(row.get("runtime_trace"), list) else []
        runtime_total, runtime_explain, runtime_inp, runtime_out = _runtime_token_breakdown(runtime_trace)

        answer = str(row.get("answer", "")).strip()
        if not answer:
            received_answer = received.get("answer")
            if isinstance(received_answer, str):
                answer = received_answer.strip()

        parsed.append(
            RowStats(
                model=model,
                source_file=path.name,
                question_id=str(row.get("question_id", row.get("id", ""))),
                question=str(row.get("question", "")),
                qtype=qtype,
                triple_precision=precision,
                triple_recall=recall,
                triple_f1=f1,
                triples_matched=matched,
                triples_expected=expected_count,
                triples_received=received_count,
                runtime_tokens_total=runtime_total,
                runtime_tokens_cite_explain=runtime_explain,
                runtime_input_tokens=runtime_inp,
                runtime_output_tokens=runtime_out,
                elapsed_s=_to_float(row.get("elapsed_s"), 0.0),
                found_flag=found_flag,
                has_citation=has_citation,
                answer=answer,
            )
        )

    return model, parsed


def _resolve_input_files(results_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in results_dir.glob("eval_results_*.toml")
        if p.is_file() and ".scored." not in p.name
    )


def _group_mean(rows: list[RowStats], attr: str) -> float:
    values = [getattr(r, attr) for r in rows]
    return float(statistics.fmean(values)) if values else 0.0


def _group_sum(rows: list[RowStats], attr: str) -> float:
    return float(sum(getattr(r, attr) for r in rows))


def _subset_for_column(rows: list[RowStats], column: str) -> list[RowStats]:
    if column == "all_without_impossible":
        return [r for r in rows if r.qtype.lower() != "impossible"]
    if column == "all_with_impossible":
        return list(rows)
    return [r for r in rows if r.qtype == column]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _extract_found_flag(row: dict[str, Any]) -> bool | None:
    top_level = _coerce_bool(row.get("found"))
    if top_level is not None:
        return top_level

    received = row.get("received")
    if isinstance(received, dict):
        recv_found = _coerce_bool(received.get("found"))
        if recv_found is not None:
            return recv_found
        trace = received.get("trace")
        if isinstance(trace, list):
            for entry in trace:
                if not isinstance(entry, dict):
                    continue
                trace_found = _coerce_bool(entry.get("found"))
                if trace_found is not None:
                    return trace_found

    runtime_trace = row.get("runtime_trace")
    if isinstance(runtime_trace, list):
        for event in runtime_trace:
            if not isinstance(event, dict):
                continue
            calls = event.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                args = call.get("args")
                if not isinstance(args, dict):
                    continue
                found = _coerce_bool(args.get("found"))
                if found is not None:
                    return found

    return None


def _ordered_qtypes(rows: list[RowStats]) -> list[str]:
    qtypes = {str(r.qtype or "unknown") for r in rows}
    preferred = ["direct", "hop", "query_builder", "impossible"]
    ordered = [q for q in preferred if q in qtypes]
    ordered.extend(sorted(q for q in qtypes if q not in set(preferred)))
    return ordered


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _fmt_token_compact(value: float) -> str:
    n = abs(value)
    sign = "-" if value < 0 else ""
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"{sign}{int(round(n / 1_000.0))}k"
    return f"{sign}{int(round(n))}"


def _fmt_tokens_share(explain_tokens: float, total_tokens: float) -> str:
    if total_tokens <= 0:
        return "0/0 (0.0%)"
    share = 100.0 * explain_tokens / total_tokens
    return f"{_fmt_token_compact(explain_tokens)}/{_fmt_token_compact(total_tokens)} ({share:.1f}%)"


def _compact_explain_share_text(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    m = re.match(r"^\s*([^/]+)\s*/[^()]+\s*\(([^)]+)\)\s*$", text)
    if not m:
        return text
    explain = m.group(1).strip()
    pct = m.group(2).strip()
    unit = re.match(r"^([0-9]+(?:\.[0-9]+)?)([kKmM])$", explain)
    if unit:
        explain = f"{unit.group(1)}{unit.group(2).upper()}"
    return f"{explain} ({pct})"


def _prf_f1_value(prf: str) -> float | None:
    clean = prf.strip()
    if not clean:
        return None
    if "/" not in clean:
        try:
            return float(clean)
        except ValueError:
            return None
    parts = [p.strip() for p in prf.split("/")]
    if len(parts) != 3:
        return None
    try:
        return float(parts[2])
    except ValueError:
        return None


def _prf_f1_text(prf: str) -> str:
    val = _prf_f1_value(prf)
    if val is None:
        return ""
    return f"{val:.3f}"


def _truncate_text(value: str, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _ratio_cell(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def _found_stats(rows: list[RowStats]) -> dict[str, int]:
    stats = {
        "total": 0,
        "false": 0,
        "true": 0,
        "missing": 0,
        "imp_total": 0,
        "imp_false": 0,
        "nonimp_total": 0,
        "nonimp_false": 0,
    }
    for row in rows:
        qtype = row.qtype.lower()
        stats["total"] += 1
        if row.found_flag is False:
            stats["false"] += 1
        elif row.found_flag is True:
            stats["true"] += 1
        else:
            stats["missing"] += 1

        if qtype == "impossible":
            stats["imp_total"] += 1
            if row.found_flag is False:
                stats["imp_false"] += 1
        else:
            stats["nonimp_total"] += 1
            if row.found_flag is False:
                stats["nonimp_false"] += 1
    return stats


def _render_found_stats_table(model_rows: dict[str, list[RowStats]]) -> str:
    headers = [
        "model",
        "found=false",
        "found=true",
        "found_missing",
        "found=false (impossible)",
        "found=false (non-impossible)",
    ]
    rows: list[list[str]] = []
    sorted_models = sorted(
        model_rows.keys(),
        key=lambda model: (_size_class_sort_key(_model_size_class(model)), model),
    )
    for model in sorted_models:
        stats = _found_stats(model_rows[model])
        total = stats["total"]
        rows.append(
            [
                model,
                _ratio_cell(stats["false"], total),
                _ratio_cell(stats["true"], total),
                _ratio_cell(stats["missing"], total),
                _ratio_cell(stats["imp_false"], stats["imp_total"]),
                _ratio_cell(stats["nonimp_false"], stats["nonimp_total"]),
            ]
        )
    return render_markdown_table(headers, rows)


def _rows_by_model(rows: list[RowStats]) -> dict[str, list[RowStats]]:
    by_model: dict[str, list[RowStats]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)
    return by_model


def _question_key(row: RowStats) -> str:
    qid = row.question_id.strip()
    if qid:
        return qid
    return row.question.strip()


def _nemotron_fail_examples(
    model_rows: dict[str, list[RowStats]],
    *,
    nemotron_model: str,
    baseline_model: str,
    limit: int,
) -> list[dict[str, str]]:
    nemotron_rows = model_rows.get(nemotron_model, [])
    baseline_rows = model_rows.get(baseline_model, [])
    if not nemotron_rows or not baseline_rows:
        return []

    nemotron_by_key: dict[str, RowStats] = {}
    for row in nemotron_rows:
        key = _question_key(row)
        if key and key not in nemotron_by_key:
            nemotron_by_key[key] = row
    baseline_by_key: dict[str, RowStats] = {}
    for row in baseline_rows:
        key = _question_key(row)
        if key and key not in baseline_by_key:
            baseline_by_key[key] = row

    examples: list[dict[str, str]] = []
    for key in sorted(set(nemotron_by_key) & set(baseline_by_key), key=lambda x: (len(x), x)):
        n_row = nemotron_by_key[key]
        b_row = baseline_by_key[key]
        if n_row.qtype.lower() == "impossible":
            continue
        if not (n_row.triple_f1 <= 1e-12 and b_row.triple_f1 >= 0.5):
            continue

        n_counts = f"exp={n_row.triples_expected},rec={n_row.triples_received},matched={n_row.triples_matched}"
        b_counts = f"exp={b_row.triples_expected},rec={b_row.triples_received},matched={b_row.triples_matched}"
        examples.append(
            {
                "id": n_row.question_id or key,
                "qtype": n_row.qtype,
                "question": _truncate_text(n_row.question, max_chars=260) or "n/a",
                "nemotron_metrics": (
                    f"F1={n_row.triple_f1:.3f}, {n_counts}, "
                    f"found={n_row.found_flag}, cite={n_row.has_citation}"
                ),
                "baseline_metrics": (
                    f"F1={b_row.triple_f1:.3f}, {b_counts}, "
                    f"found={b_row.found_flag}, cite={b_row.has_citation}"
                ),
                "nemotron_answer": _truncate_text(n_row.answer, max_chars=260) or "n/a",
                "baseline_answer": _truncate_text(b_row.answer, max_chars=260) or "n/a",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _render_found_false_audit_markdown(
    dataset_order: list[str],
    dataset_rows_by_name: dict[str, list[RowStats]],
) -> str:
    nemotron_model = "nemotron-3-nano:30b-cloud"
    baseline_model = "ministral-3:3b-cloud"
    example_limit = 3

    combined_model_rows: dict[str, list[RowStats]] = {}
    dataset_model_rows: dict[str, dict[str, list[RowStats]]] = {}
    total_rows = 0

    for dataset in dataset_order:
        rows = dataset_rows_by_name.get(dataset, [])
        total_rows += len(rows)
        by_model = _rows_by_model(rows)
        dataset_model_rows[dataset] = by_model
        for model, model_specific_rows in by_model.items():
            combined_model_rows.setdefault(model, []).extend(model_specific_rows)

    models_with_false = sum(
        1
        for rows in combined_model_rows.values()
        if any(row.found_flag is False for row in rows)
    )

    sections = [
        "# Found=False Audit",
        "",
        "- Computed automatically from all loaded `eval_results_*.toml` rows.",
        "- `found=false` is expected for correctly handled `impossible` questions, but suspicious for non-impossible rows.",
        f"- Models with at least one `found=false`: {models_with_false}/{len(combined_model_rows)}.",
        f"- Total rows audited: {total_rows}.",
        "",
        "## Combined (all datasets)",
        "",
        _render_found_stats_table(combined_model_rows),
        "",
    ]

    for dataset in dataset_order:
        sections.extend(
            [
                f"## Dataset: {dataset}",
                "",
                _render_found_stats_table(dataset_model_rows.get(dataset, {})),
                "",
            ]
        )

        examples = _nemotron_fail_examples(
            dataset_model_rows.get(dataset, {}),
            nemotron_model=nemotron_model,
            baseline_model=baseline_model,
            limit=example_limit,
        )
        sections.append(
            f"### Examples where `{nemotron_model}` fails but `{baseline_model}` does better"
        )
        if not examples:
            sections.extend(
                [
                    "",
                    "No examples found with current thresholds (`nemotron F1 == 0` and `3B F1 >= 0.5`).",
                    "",
                ]
            )
            continue

        sections.append("")
        for idx, ex in enumerate(examples, start=1):
            sections.extend(
                [
                    f"{idx}. `id={ex['id']}` `qtype={ex['qtype']}`",
                    f"   Question: {ex['question']}",
                    f"   Nemotron: {ex['nemotron_metrics']}",
                    f"   3B: {ex['baseline_metrics']}",
                    f"   Nemotron answer: {ex['nemotron_answer']}",
                    f"   3B answer: {ex['baseline_answer']}",
                    "",
                ]
            )

    return "\n".join(sections)


def _model_size_class(model: str) -> str:
    m = model.lower()
    if "ministral-3:3b" in m:
        return "3B"
    if "ministral-3:8b" in m:
        return "8B"
    if "ministral-3:14b" in m:
        return "14B"
    if "nemotron-3-nano:30b" in m:
        return "30B"
    if "qwen3-coder-next" in m:
        return "~80B"
    if "devstral-2:123b" in m or "gpt-oss:120b" in m:
        return "~120B"
    if "glm-4.7" in m:
        return "~300B"
    if "kimi-k2.5" in m:
        return "~1T"
    return "n/a"


def _size_class_sort_key(size_class: str) -> int:
    order = {
        "3B": 0,
        "8B": 1,
        "14B": 2,
        "30B": 3,
        "~80B": 4,
        "~120B": 5,
        "~300B": 6,
        "~1T": 7,
        "n/a": 8,
    }
    return order.get(size_class, 99)


def _estimate_cost_usd(rows: list[RowStats], pricing: ModelPricing) -> tuple[float, float]:
    total_input = _group_sum(rows, "runtime_input_tokens")
    total_output = _group_sum(rows, "runtime_output_tokens")
    total_cost = (
        (total_input / 1_000_000.0) * pricing.input_per_1m_usd
        + (total_output / 1_000_000.0) * pricing.output_per_1m_usd
    )
    mean_cost = (total_cost / len(rows)) if rows else 0.0
    return mean_cost, total_cost


def _is_dominated_2d(candidate: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    cand_f1, cand_axis = candidate
    for other_f1, other_axis in others:
        if (
            other_f1 >= cand_f1
            and other_axis <= cand_axis
            and (other_f1 > cand_f1 or other_axis < cand_axis)
        ):
            return True
    return False


def _build_pareto_points_from_metrics(
    metrics: list[tuple[str, float, float, float]],
) -> list[ParetoPoint]:
    out_points: list[ParetoPoint] = []
    for model, f1, cost_val, latency in metrics:
        cost_others = [(of1, ocost) for om, of1, ocost, _ in metrics if om != model]
        latency_others = [(of1, olat) for om, of1, _, olat in metrics if om != model]
        out_points.append(
            ParetoPoint(
                model=model,
                f1_without_impossible=f1,
                cost_per_question_usd=cost_val,
                mean_latency_s=latency,
                cost_frontier=not _is_dominated_2d((f1, cost_val), cost_others),
                latency_frontier=not _is_dominated_2d((f1, latency), latency_others),
            )
        )
    out_points.sort(key=lambda p: (not p.cost_frontier, -p.f1_without_impossible))
    return out_points


def _collect_dataset_view(
    dataset: str,
    rows: list[RowStats],
    pricing_by_model: dict[str, ModelPricing],
) -> DatasetView:
    by_model: dict[str, list[RowStats]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)

    table_a_by_model: dict[str, dict[str, str]] = {}
    table_b_by_model: dict[str, dict[str, str]] = {}
    aggregate_by_model: dict[str, dict[str, float]] = {}
    qtypes = _ordered_qtypes(rows)
    benchmark_headers = [
        "model",
        *[f"{q} (P/R/F1)" for q in qtypes],
        "overall_f1_without_impossible",
        "overall_f1_with_impossible",
    ]
    benchmark_rows: list[list[str]] = []
    sorted_models = sorted(
        by_model.keys(),
        key=lambda model: (_size_class_sort_key(_model_size_class(model)), model),
    )

    for model in sorted_models:
        model_rows = by_model[model]
        without_impossible = _subset_for_column(model_rows, "all_without_impossible")
        with_impossible = _subset_for_column(model_rows, "all_with_impossible")

        p_wo = _group_mean(without_impossible, "triple_precision")
        r_wo = _group_mean(without_impossible, "triple_recall")
        f1_wo = _group_mean(without_impossible, "triple_f1")

        p_w = _group_mean(with_impossible, "triple_precision")
        r_w = _group_mean(with_impossible, "triple_recall")
        f1_w = _group_mean(with_impossible, "triple_f1")

        table_a_by_model[model] = {
            "all_without_impossible_prf": f"{_fmt(p_wo, 3)}/{_fmt(r_wo, 3)}/{_fmt(f1_wo, 3)}",
            "all_with_impossible_prf": f"{_fmt(p_w, 3)}/{_fmt(r_w, 3)}/{_fmt(f1_w, 3)}",
        }

        explain_wo = _group_sum(without_impossible, "runtime_tokens_cite_explain")
        total_wo = _group_sum(without_impossible, "runtime_tokens_total")
        explain_w = _group_sum(with_impossible, "runtime_tokens_cite_explain")
        total_w = _group_sum(with_impossible, "runtime_tokens_total")

        table_b_by_model[model] = {
            "all_without_impossible": _compact_explain_share_text(_fmt_tokens_share(explain_wo, total_wo)),
            "all_with_impossible": _compact_explain_share_text(_fmt_tokens_share(explain_w, total_w)),
        }

        pricing = pricing_by_model.get(model)
        if pricing is None:
            raise ValueError(
                f"Missing pricing for model '{model}'. Add it to reports/model_pricing.csv and rerun."
            )
        mean_cost, _ = _estimate_cost_usd(without_impossible, pricing)

        aggregate_by_model[model] = {
            "questions": float(len(without_impossible)),
            "mean_f1_without_impossible": float(f1_wo),
            "mean_est_cost_per_q_usd": float(mean_cost),
            "mean_elapsed_s_per_q": float(_group_mean(without_impossible, "elapsed_s")),
        }

        benchmark_row = [model]
        for qtype in qtypes:
            qrows = _subset_for_column(model_rows, qtype)
            if not qrows:
                benchmark_row.append("n/a")
                continue
            qp = _group_mean(qrows, "triple_precision")
            qr = _group_mean(qrows, "triple_recall")
            qf = _group_mean(qrows, "triple_f1")
            benchmark_row.append(f"{_fmt(qp, 3)}/{_fmt(qr, 3)}/{_fmt(qf, 3)}")
        benchmark_row.append(_fmt(f1_wo, 3))
        benchmark_row.append(_fmt(f1_w, 3))
        benchmark_rows.append(benchmark_row)

    return DatasetView(
        dataset=dataset,
        table_a_by_model=table_a_by_model,
        table_b_by_model=table_b_by_model,
        aggregate_by_model=aggregate_by_model,
        benchmark_headers=benchmark_headers,
        benchmark_rows=benchmark_rows,
    )


def build_combined_outputs(
    dataset_views: list[DatasetView],
    out_dir: Path,
    dataset_rows_by_name: dict[str, list[RowStats]] | None = None,
) -> dict[str, Path]:
    if not dataset_views:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_order = [
        d for d in ["hypmol", "wiproflex"] if d in {view.dataset for view in dataset_views}
    ]
    dataset_order.extend(
        sorted(view.dataset for view in dataset_views if view.dataset not in set(dataset_order))
    )
    views_by_name = {view.dataset: view for view in dataset_views}
    dataset_rows_by_name = dataset_rows_by_name or {}
    benchmark_md_by_dataset: dict[str, Path] = {}
    found_false_audit_md = out_dir / "found_false_audit.md"

    all_models = {
        model
        for view in dataset_views
        for model in set(view.table_a_by_model) | set(view.table_b_by_model)
    }
    sorted_models = sorted(
        all_models,
        key=lambda model: (_size_class_sort_key(_model_size_class(model)), model),
    )

    cross_prf_rows: list[list[str]] = []
    cross_explain_rows: list[list[str]] = []
    for model in sorted_models:
        prf_row = [_model_size_class(model), model]
        explain_row = [_model_size_class(model), model]
        for dataset in dataset_order:
            view = views_by_name.get(dataset)
            a = view.table_a_by_model.get(model, {}) if view else {}
            b = view.table_b_by_model.get(model, {}) if view else {}
            prf_row.append(_prf_f1_text(str(a.get("all_without_impossible_prf", ""))))
            prf_row.append(_prf_f1_text(str(a.get("all_with_impossible_prf", ""))))
            explain_row.append(str(b.get("all_without_impossible", "")))
            explain_row.append(str(b.get("all_with_impossible", "")))
        cross_prf_rows.append(prf_row)
        cross_explain_rows.append(explain_row)

    cross_prf_tex = out_dir / "cross_dataset_prf_table.tex"
    cross_explain_tex = out_dir / "cross_dataset_explain_table.tex"
    global_cost_pdf = out_dir / "chart_pareto_global_f1_vs_cost.pdf"
    global_latency_pdf = out_dir / "chart_pareto_global_f1_vs_latency.pdf"
    global_fig_tex = out_dir / "global_pareto_figures.tex"

    cross_prf_tex.write_text(render_cross_dataset_prf_latex(dataset_order, cross_prf_rows))
    cross_explain_tex.write_text(
        render_cross_dataset_explain_latex(dataset_order, cross_explain_rows)
    )
    for dataset in dataset_order:
        view = views_by_name[dataset]
        benchmark_md = out_dir / f"benchmark_scoring_{dataset}.md"
        benchmark_md.write_text(
            "\n".join(
                [
                    f"# Benchmark Scoring ({dataset})",
                    "",
                    "- Per-qtype cells are shown as `(P/R/F1)`.",
                    "- For `impossible`, `(P/R/F1)` is scored from `found=false` correctness (1.000 if false, else 0.000).",
                    "- Citation gate: if no `cite` tool call appears in a row's runtime trace, that row's F1 is forced to `0.000`.",
                    "- `overall_f1_without_impossible` is mean triple F1 excluding impossible questions.",
                    "- `overall_f1_with_impossible` is mean triple F1 across all questions in this dataset.",
                    "",
                    render_markdown_table(view.benchmark_headers, view.benchmark_rows),
                    "",
                ]
            )
        )
        benchmark_md_by_dataset[dataset] = benchmark_md

    global_acc: dict[str, dict[str, float]] = {}
    for dataset in dataset_order:
        view = views_by_name[dataset]
        for model, agg in view.aggregate_by_model.items():
            q = agg.get("questions", 0.0)
            if q <= 0:
                continue
            acc = global_acc.setdefault(
                model,
                {"questions": 0.0, "f1q": 0.0, "costq": 0.0, "latq": 0.0},
            )
            acc["questions"] += q
            acc["f1q"] += agg.get("mean_f1_without_impossible", 0.0) * q
            acc["costq"] += agg.get("mean_est_cost_per_q_usd", 0.0) * q
            acc["latq"] += agg.get("mean_elapsed_s_per_q", 0.0) * q

    global_metrics = [
        (
            model,
            acc["f1q"] / acc["questions"],
            acc["costq"] / acc["questions"],
            acc["latq"] / acc["questions"],
        )
        for model, acc in sorted(
            global_acc.items(), key=lambda kv: (_size_class_sort_key(_model_size_class(kv[0])), kv[0])
        )
        if acc["questions"] > 0
    ]

    global_points = _build_pareto_points_from_metrics(global_metrics)
    write_pareto_svg(
        path=global_cost_pdf,
        points=global_points,
        frontier_attr="cost_frontier",
        x_attr="cost_per_question_usd",
        x_label="Global Weighted Cost Per Question (USD)",
        y_attr="f1_without_impossible",
        y_label="Global Weighted F1 (without impossible questions)",
    )
    write_pareto_svg(
        path=global_latency_pdf,
        points=global_points,
        frontier_attr="latency_frontier",
        x_attr="mean_latency_s",
        x_label="Global Weighted Mean Latency Per Question (s)",
        y_attr="f1_without_impossible",
        y_label="Global Weighted F1 (without impossible questions)",
    )

    global_fig_tex.write_text(
        "\n".join(
            [
                "% Auto-generated by src.scoring.compare_models",
                "% Required packages in your preamble:",
                r"% \usepackage{graphicx}",
                r"% \usepackage{subcaption}",
                "",
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\begin{subfigure}[t]{\linewidth}",
                r"\centering",
                r"\includegraphics[width=\linewidth]{chart_pareto_global_f1_vs_cost.pdf}",
                r"\caption{Global Pareto: cost vs F1.}",
                r"\label{fig:global_pareto_cost}",
                r"\end{subfigure}",
                r"\vspace{0.75em}",
                r"\begin{subfigure}[t]{\linewidth}",
                r"\centering",
                r"\includegraphics[width=\linewidth]{chart_pareto_global_f1_vs_latency.pdf}",
                r"\caption{Global Pareto: latency vs F1.}",
                r"\label{fig:global_pareto_latency}",
                r"\end{subfigure}",
                r"\caption{Global weighted Pareto frontiers across datasets.}",
                r"\label{fig:global_pareto}",
                r"\end{figure}",
                "",
            ]
        )
    )

    found_false_audit_md.write_text(
        _render_found_false_audit_markdown(dataset_order, dataset_rows_by_name)
    )

    outputs = {
        "cross_dataset_prf_table_tex": cross_prf_tex,
        "cross_dataset_explain_table_tex": cross_explain_tex,
        "chart_pareto_global_f1_vs_cost_pdf": global_cost_pdf,
        "chart_pareto_global_f1_vs_latency_pdf": global_latency_pdf,
        "global_pareto_figures_tex": global_fig_tex,
        "found_false_audit_md": found_false_audit_md,
    }
    for dataset, path in benchmark_md_by_dataset.items():
        outputs[f"benchmark_scoring_{dataset}_md"] = path
    return outputs


def _resolve_default_results_root() -> Path:
    primary = Path("results")
    if primary.is_dir():
        return primary
    fallback = Path("result")
    if fallback.is_dir():
        return fallback
    return primary


def _resolve_pricing_file(cli_pricing_file: Path | None) -> Path:
    if cli_pricing_file is not None:
        return cli_pricing_file
    return Path("reports") / "model_pricing.csv"


def _prune_to_combined_artifacts(reports_root: Path, pricing_file: Path | None) -> list[Path]:
    """Keep only canonical combined outputs (plus pricing file at reports root)."""
    if not reports_root.exists() or not reports_root.is_dir():
        return []

    combined_dir = reports_root / "combined"
    keep_root_files: set[str] = set()
    if pricing_file is not None and pricing_file.parent.resolve() == reports_root.resolve():
        keep_root_files.add(pricing_file.name)

    for child in reports_root.iterdir():
        if child == combined_dir:
            continue
        if child.is_file() and child.name in keep_root_files:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)

    allowed_combined = {
        "cross_dataset_prf_table.tex",
        "cross_dataset_explain_table.tex",
        "global_pareto_figures.tex",
        "chart_pareto_global_f1_vs_cost.pdf",
        "chart_pareto_global_f1_vs_latency.pdf",
        "found_false_audit.md",
    }
    combined_dir.mkdir(parents=True, exist_ok=True)
    for child in combined_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            continue
        is_dataset_benchmark_md = (
            child.suffix == ".md" and child.name.startswith("benchmark_scoring_")
        )
        if child.name not in allowed_combined and not is_dataset_benchmark_md:
            child.unlink(missing_ok=True)

    return sorted(p for p in combined_dir.iterdir() if p.is_file())


def run(
    results_dir: Path | None = None,
    *,
    run_all: bool = False,
    pricing_file: Path | None = None,
) -> tuple[dict[str, Path], list[Path]]:
    pricing_file = _resolve_pricing_file(pricing_file)
    pricing_by_model = _load_pricing(pricing_file)

    if results_dir is None and not run_all:
        run_all = True
    if run_all and results_dir is not None:
        raise ValueError("Provide either a single results_dir or --all, not both.")

    if run_all:
        results_root = _resolve_default_results_root()
        if not results_root.exists() or not results_root.is_dir():
            raise ValueError(
                f"Results root not found: {results_root}. Expected ./results (or ./result)."
            )
        dataset_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())
        if not dataset_dirs:
            raise ValueError(f"No dataset directories found under: {results_root}")
    else:
        assert results_dir is not None
        dataset_dirs = [results_dir]

    dataset_views: list[DatasetView] = []
    dataset_rows_by_name: dict[str, list[RowStats]] = {}
    for dataset_dir in dataset_dirs:
        input_files = _resolve_input_files(dataset_dir)
        if not input_files:
            if run_all:
                continue
            raise ValueError(
                f"No input files found in directory: {dataset_dir}. "
                "Expected files named eval_results_*.toml (excluding *.scored.*)."
            )
        rows: list[RowStats] = []
        for file in input_files:
            _, parsed = _load_eval_file(file)
            rows.extend(parsed)
        if not rows:
            continue
        dataset_rows_by_name[dataset_dir.name] = rows
        dataset_views.append(
            _collect_dataset_view(
                dataset=dataset_dir.name,
                rows=rows,
                pricing_by_model=pricing_by_model,
            )
        )

    if not dataset_views:
        raise ValueError("No datasets with eval_results_*.toml found.")

    reports_root = Path("reports")
    outputs = build_combined_outputs(
        dataset_views,
        reports_root / "combined",
        dataset_rows_by_name=dataset_rows_by_name,
    )
    kept = _prune_to_combined_artifacts(reports_root, pricing_file)
    return outputs, kept
