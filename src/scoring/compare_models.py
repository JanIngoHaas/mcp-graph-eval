"""Cross-model comparison and resource analysis for eval result files."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_triple_f1
from .reporting import (
    render_cross_dataset_explain_latex as _render_cross_dataset_explain_latex,
    render_cross_dataset_prf_latex as _render_cross_dataset_prf_latex,
    render_latex_export as _render_latex_export,
    render_markdown_table as _render_markdown_table,
    write_pareto_svg as _write_pareto_svg,
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


@dataclass
class ModelAggregate:
    model: str
    questions: int
    mean_f1_with_impossible: float
    mean_f1_without_impossible: float
    mean_precision_without_impossible: float
    mean_recall_without_impossible: float
    mean_runtime_tokens_per_q: float
    mean_cite_explain_tokens_per_q: float
    cite_explain_share_pct: float
    mean_elapsed_s_per_q: float
    total_elapsed_h: float
    mean_input_tokens_per_q: float
    mean_output_tokens_per_q: float
    mean_est_cost_per_q_usd: float | None
    total_est_cost_usd: float | None


@dataclass
class ParetoPoint:
    model: str
    f1_without_impossible: float
    cost_per_question_usd: float
    mean_latency_s: float
    cost_frontier: bool
    latency_frontier: bool


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

        tool_calls = event.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        names: list[str] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name", "")).strip()
            if name:
                names.append(name)
        if not names or total <= 0:
            continue

        per_call = total / len(names)
        for name in names:
            if name in EXPLANATION_TOOLS:
                explain_tokens += per_call
    return total_tokens, explain_tokens, input_tokens, output_tokens


def _load_pricing(pricing_file: Path | None) -> dict[str, ModelPricing]:
    if pricing_file is None:
        raise ValueError("--pricing-file is required.")
    if not pricing_file.exists():
        raise ValueError(f"Pricing file not found: {pricing_file}")
    out: dict[str, ModelPricing] = {}
    with pricing_file.open() as f:
        reader = csv.DictReader(f)
        required = {"model", "input_per_1m", "output_per_1m"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Pricing file missing required columns: {sorted(missing)}. "
                "Expected: model,input_per_1m,output_per_1m"
            )
        for row in reader:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            input_raw = str(row.get("input_per_1m", "")).strip()
            output_raw = str(row.get("output_per_1m", "")).strip()
            if not input_raw or not output_raw:
                raise ValueError(
                    f"Pricing row for model '{model}' is incomplete. "
                    "Both input_per_1m and output_per_1m are required."
                )
            try:
                input_price = float(input_raw)
                output_price = float(output_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Pricing row for model '{model}' has non-numeric values: "
                    f"input_per_1m='{input_raw}', output_per_1m='{output_raw}'"
                ) from exc
            if input_price < 0 or output_price < 0:
                raise ValueError(
                    f"Pricing row for model '{model}' must be non-negative. "
                    f"Got input_per_1m={input_price}, output_per_1m={output_price}"
                )
            out[model] = ModelPricing(
                input_per_1m_usd=input_price,
                output_per_1m_usd=output_price,
            )
    if not out:
        raise ValueError(f"Pricing file has no model rows: {pricing_file}")
    return out


def _load_eval_file(path: Path) -> tuple[str, list[RowStats]]:
    data = tomllib.loads(path.read_text())
    metadata = data.get("metadata", {}) or {}
    model = str(metadata.get("model") or path.stem)
    rows = data.get("output", []) or []

    parsed: list[RowStats] = []
    for row in rows:
        expected = row.get("expected", {}) or {}
        received = row.get("received", {}) or {}
        expected_triples = _scorable_triples(expected)
        received_triples = _scorable_triples(received)
        score = compute_triple_f1(expected_triples, received_triples)
        runtime_trace = row.get("runtime_trace", []) or []
        runtime_total, runtime_explain, runtime_inp, runtime_out = _runtime_token_breakdown(runtime_trace)
        parsed.append(
            RowStats(
                model=model,
                source_file=path.name,
                question_id=str(row.get("id", "")),
                question=str(row.get("question", "")),
                qtype=str(row.get("qtype") or "unknown"),
                triple_precision=score.precision,
                triple_recall=score.recall,
                triple_f1=score.f1,
                triples_matched=score.matched,
                triples_expected=score.expected_count,
                triples_received=score.received_count,
                runtime_tokens_total=runtime_total,
                runtime_tokens_cite_explain=runtime_explain,
                runtime_input_tokens=runtime_inp,
                runtime_output_tokens=runtime_out,
                elapsed_s=_to_float(row.get("elapsed_s")),
            )
        )
    return model, parsed


def _resolve_input_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    if not results_dir.is_dir():
        raise ValueError(f"Results directory is not a directory: {results_dir}")

    candidates = sorted(results_dir.glob("eval_results_*.toml"))
    return [path for path in candidates if ".scored." not in path.name]


def _group_mean(rows: list[RowStats], attr: str) -> float:
    if not rows:
        return 0.0
    return float(statistics.fmean(getattr(r, attr) for r in rows))


def _group_sum(rows: list[RowStats], attr: str) -> float:
    return float(sum(getattr(r, attr) for r in rows))


def _group_explain_share_pct(rows: list[RowStats]) -> float:
    if not rows:
        return 0.0
    total = _group_sum(rows, "runtime_tokens_total")
    explain = _group_sum(rows, "runtime_tokens_cite_explain")
    return (100.0 * explain / total) if total > 0 else 0.0


def _column_qtypes(all_rows: list[RowStats]) -> list[str]:
    qtypes = sorted({r.qtype for r in all_rows})
    preferred = ["direct", "hop", "query_builder", "impossible"]
    ordered: list[str] = [q for q in preferred if q in qtypes]
    ordered.extend(q for q in qtypes if q not in set(preferred))
    return ordered


def _subset_for_column(rows: list[RowStats], column: str) -> list[RowStats]:
    if column == "all_without_impossible":
        return [r for r in rows if r.qtype.lower() != "impossible"]
    if column == "all_with_impossible":
        return list(rows)
    return [r for r in rows if r.qtype == column]


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
        return "approx 80B"
    if "devstral-2:123b" in m or "gpt-oss:120b" in m:
        return "approx 120B"
    if "glm-4.7" in m:
        return "approx 300B"
    if "kimi-k2.5" in m:
        return "approx 1T"
    return "Unspecified"


def _size_class_sort_key(size_class: str) -> int:
    order = {
        "3B": 0,
        "8B": 1,
        "14B": 2,
        "30B": 3,
        "approx 80B": 4,
        "approx 120B": 5,
        "approx 300B": 6,
        "approx 1T": 7,
        "Unspecified": 8,
    }
    return order.get(size_class, 99)


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


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _discover_dataset_report_dirs(results_root: Path, reports_root: Path) -> list[Path]:
    if not results_root.exists() or not results_root.is_dir():
        return []
    report_dirs: list[Path] = []
    for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        report_dir = reports_root / dataset_dir.name
        table_a = report_dir / "table_a_f1_by_model_qtype.csv"
        table_b = report_dir / "table_b_explain_overhead_share_by_model_qtype.csv"
        if table_a.exists() and table_b.exists():
            report_dirs.append(report_dir)
    return report_dirs


def build_combined_reports(report_dirs: list[Path], out_dir: Path) -> dict[str, Path]:
    if not report_dirs:
        return {}

    headers = [
        "dataset",
        "model",
        "direct_f1",
        "hop_f1",
        "query_builder_f1",
        "impossible_f1",
        "all_without_impossible_prf",
        "all_with_impossible_prf",
        "explain_direct",
        "explain_hop",
        "explain_query_builder",
        "explain_impossible",
        "explain_all_without_impossible",
        "explain_all_with_impossible",
    ]

    csv_rows: list[list[Any]] = []
    md_rows: list[list[str]] = []
    table_a_by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    table_b_by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    aggregates_by_dataset: dict[str, dict[str, dict[str, str]]] = {}

    for report_dir in report_dirs:
        dataset = report_dir.name
        table_a_rows = _load_csv_rows(report_dir / "table_a_f1_by_model_qtype.csv")
        table_b_rows = _load_csv_rows(report_dir / "table_b_explain_overhead_share_by_model_qtype.csv")
        aggregates_csv = report_dir / "model_aggregates.csv"
        aggregate_rows = _load_csv_rows(aggregates_csv) if aggregates_csv.exists() else []

        a_by_model = {
            str(row.get("model", "")): row
            for row in table_a_rows
            if str(row.get("model", "")).strip()
        }
        b_by_model = {
            str(row.get("model", "")): row
            for row in table_b_rows
            if str(row.get("model", "")).strip()
        }
        table_a_by_dataset[dataset] = a_by_model
        table_b_by_dataset[dataset] = b_by_model
        aggregates_by_dataset[dataset] = {
            str(row.get("model", "")): row
            for row in aggregate_rows
            if str(row.get("model", "")).strip()
        }

        for model in sorted(set(a_by_model) | set(b_by_model)):
            a = a_by_model.get(model, {})
            b = b_by_model.get(model, {})
            row = [
                dataset,
                model,
                str(a.get("direct_f1", "")),
                str(a.get("hop_f1", "")),
                str(a.get("query_builder_f1", "")),
                str(a.get("impossible_f1", "")),
                str(a.get("all_without_impossible_prf", "")),
                str(a.get("all_with_impossible_prf", "")),
                str(b.get("direct", "")),
                str(b.get("hop", "")),
                str(b.get("query_builder", "")),
                str(b.get("impossible", "")),
                str(b.get("all_without_impossible", "")),
                str(b.get("all_with_impossible", "")),
            ]
            csv_rows.append(row)
            md_rows.append([str(v) for v in row])

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "combined_model_table.csv"
    combined_md = out_dir / "combined_model_table.md"
    cross_prf_csv = out_dir / "cross_dataset_prf_table.csv"
    cross_prf_md = out_dir / "cross_dataset_prf_table.md"
    cross_prf_tex = out_dir / "cross_dataset_prf_table.tex"
    cross_explain_csv = out_dir / "cross_dataset_explain_table.csv"
    cross_explain_md = out_dir / "cross_dataset_explain_table.md"
    cross_explain_tex = out_dir / "cross_dataset_explain_table.tex"
    global_pareto_cost_svg = out_dir / "chart_pareto_global_f1_vs_cost.svg"
    global_pareto_latency_svg = out_dir / "chart_pareto_global_f1_vs_latency.svg"
    summary_md = out_dir / "summary.md"

    _write_csv(combined_csv, headers, csv_rows)
    combined_md.write_text(
        "# Combined Model Table\n\n"
        + _render_markdown_table(headers, md_rows)
        + "\n"
    )

    all_dataset_names = set(table_a_by_dataset) | set(table_b_by_dataset)
    dataset_order = [d for d in ["hypmol", "wiproflex"] if d in all_dataset_names]
    dataset_order.extend(sorted(d for d in all_dataset_names if d not in set(dataset_order)))
    cross_prf_headers = ["size_class", "model"]
    for dataset in dataset_order:
        cross_prf_headers.append(f"{dataset}_ans_f1")
        cross_prf_headers.append(f"{dataset}_all_f1")

    cross_explain_headers = ["size_class", "model"]
    for dataset in dataset_order:
        cross_explain_headers.append(f"{dataset}_ans_cite_explain_share")
        cross_explain_headers.append(f"{dataset}_all_cite_explain_share")

    all_models = {
        model
        for per_dataset in table_a_by_dataset.values()
        for model in per_dataset.keys()
    }
    sorted_models = sorted(
        all_models,
        key=lambda model: (_size_class_sort_key(_model_size_class(model)), model),
    )

    cross_rows: list[list[str]] = []
    cross_explain_rows: list[list[str]] = []
    for model in sorted_models:
        row = [_model_size_class(model), model]
        explain_row = [_model_size_class(model), model]
        for dataset in dataset_order:
            a_row = table_a_by_dataset.get(dataset, {}).get(model, {})
            b_row = table_b_by_dataset.get(dataset, {}).get(model, {})
            row.append(_prf_f1_text(str(a_row.get("all_without_impossible_prf", ""))))
            row.append(_prf_f1_text(str(a_row.get("all_with_impossible_prf", ""))))
            explain_row.append(
                _compact_explain_share_text(str(b_row.get("all_without_impossible", "")))
            )
            explain_row.append(
                _compact_explain_share_text(str(b_row.get("all_with_impossible", "")))
            )
        cross_rows.append(row)
        cross_explain_rows.append(explain_row)

    _write_csv(cross_prf_csv, cross_prf_headers, [[*r] for r in cross_rows])
    cross_prf_md.write_text(
        "# Cross-Dataset F1 Table\n\n"
        + _render_markdown_table(cross_prf_headers, cross_rows)
        + "\n"
    )
    cross_prf_tex.write_text(_render_cross_dataset_prf_latex(dataset_order, cross_rows))

    _write_csv(cross_explain_csv, cross_explain_headers, [[*r] for r in cross_explain_rows])
    cross_explain_md.write_text(
        "# Cross-Dataset Cite + Explanation Token Share Table\n\n"
        + _render_markdown_table(cross_explain_headers, cross_explain_rows)
        + "\n"
    )
    cross_explain_tex.write_text(
        _render_cross_dataset_explain_latex(dataset_order, cross_explain_rows)
    )

    global_acc: dict[str, dict[str, float]] = {}
    for dataset in dataset_order:
        for model, row in aggregates_by_dataset.get(dataset, {}).items():
            try:
                questions = int(float(str(row.get("questions", "")).strip()))
                f1_without = float(str(row.get("mean_f1_without_impossible", "")).strip())
                mean_cost = float(str(row.get("mean_est_cost_per_q_usd", "")).strip())
                mean_latency = float(str(row.get("mean_elapsed_s_per_q", "")).strip())
            except ValueError:
                continue
            if questions <= 0:
                continue
            acc = global_acc.setdefault(
                model,
                {"questions": 0.0, "f1q": 0.0, "costq": 0.0, "latq": 0.0},
            )
            q = float(questions)
            acc["questions"] += q
            acc["f1q"] += f1_without * q
            acc["costq"] += mean_cost * q
            acc["latq"] += mean_latency * q

    global_metrics = [
        (
            model,
            acc["f1q"] / acc["questions"],
            acc["costq"] / acc["questions"],
            acc["latq"] / acc["questions"],
        )
        for model, acc in sorted(
            global_acc.items(),
            key=lambda kv: (_size_class_sort_key(_model_size_class(kv[0])), kv[0]),
        )
        if acc["questions"] > 0
    ]
    global_pareto_points = _build_pareto_points_from_metrics(global_metrics)
    _write_pareto_svg(
        path=global_pareto_cost_svg,
        points=global_pareto_points,
        frontier_attr="cost_frontier",
        x_attr="cost_per_question_usd",
        x_label="Global Weighted Cost Per Question (USD)",
        y_attr="f1_without_impossible",
        y_label="Global Weighted F1 (without impossible questions)",
    )
    _write_pareto_svg(
        path=global_pareto_latency_svg,
        points=global_pareto_points,
        frontier_attr="latency_frontier",
        x_attr="mean_latency_s",
        x_label="Global Weighted Mean Latency Per Question (s)",
        y_attr="f1_without_impossible",
        y_label="Global Weighted F1 (without impossible questions)",
    )

    summary_md.write_text(
        "\n".join(
            [
                "# Combined Reports",
                "",
                "Source dataset report directories:",
                *[f"- `{p}`" for p in report_dirs],
                "",
                "Generated files:",
                f"- `{combined_csv}`",
                f"- `{combined_md}`",
                f"- `{cross_prf_csv}`",
                f"- `{cross_prf_md}`",
                f"- `{cross_prf_tex}`",
                f"- `{cross_explain_csv}`",
                f"- `{cross_explain_md}`",
                f"- `{cross_explain_tex}`",
                f"- `{global_pareto_cost_svg}`",
                f"- `{global_pareto_latency_svg}`",
            ]
        )
        + "\n"
    )

    return {
        "combined_model_table_csv": combined_csv,
        "combined_model_table_md": combined_md,
        "cross_dataset_prf_table_csv": cross_prf_csv,
        "cross_dataset_prf_table_md": cross_prf_md,
        "cross_dataset_prf_table_tex": cross_prf_tex,
        "cross_dataset_explain_table_csv": cross_explain_csv,
        "cross_dataset_explain_table_md": cross_explain_md,
        "cross_dataset_explain_table_tex": cross_explain_tex,
        "chart_pareto_global_f1_vs_cost_svg": global_pareto_cost_svg,
        "chart_pareto_global_f1_vs_latency_svg": global_pareto_latency_svg,
        "combined_summary_md": summary_md,
    }


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
        cost_frontier = not _is_dominated_2d((f1, cost_val), cost_others)
        latency_frontier = not _is_dominated_2d((f1, latency), latency_others)
        out_points.append(
            ParetoPoint(
                model=model,
                f1_without_impossible=f1,
                cost_per_question_usd=cost_val,
                mean_latency_s=latency,
                cost_frontier=cost_frontier,
                latency_frontier=latency_frontier,
            )
        )
    out_points.sort(key=lambda p: (not p.cost_frontier, -p.f1_without_impossible))
    return out_points


def _build_pareto_points(aggregates: list[ModelAggregate]) -> list[ParetoPoint]:
    missing_cost_models = [a.model for a in aggregates if a.mean_est_cost_per_q_usd is None]
    if missing_cost_models:
        raise ValueError(
            "Missing USD cost for models (check pricing file coverage): "
            + ", ".join(sorted(missing_cost_models))
        )
    metrics = [
        (
            a.model,
            a.mean_f1_without_impossible,
            float(a.mean_est_cost_per_q_usd),
            a.mean_elapsed_s_per_q,
        )
        for a in aggregates
    ]
    return _build_pareto_points_from_metrics(metrics)


def build_reports(
    input_files: list[Path],
    out_dir: Path,
    pricing_by_model: dict[str, ModelPricing],
) -> dict[str, Path]:
    all_rows: list[RowStats] = []
    for file in input_files:
        _, parsed = _load_eval_file(file)
        all_rows.extend(parsed)

    if not all_rows:
        raise ValueError("No rows parsed from input files.")

    qtype_cols = _column_qtypes(all_rows)
    all_cols = ["all_without_impossible", "all_with_impossible"]
    cols = qtype_cols + all_cols

    by_model: dict[str, list[RowStats]] = {}
    for row in all_rows:
        by_model.setdefault(row.model, []).append(row)
    models = sorted(by_model.keys())

    # Table A: F1 by qtype + PRF for all_* aggregates.
    headers_a = ["model"] + [f"{col}_f1" for col in qtype_cols] + [f"{col}_prf" for col in all_cols]
    table_a_csv_rows: list[list[Any]] = []
    table_a_md_rows: list[list[str]] = []
    for model in models:
        model_rows = by_model[model]
        vals_num: list[Any] = []
        vals_str: list[str] = []
        for col in qtype_cols:
            subset = _subset_for_column(model_rows, col)
            val = _group_mean(subset, "triple_f1")
            vals_num.append(val)
            vals_str.append(_fmt(val, 3))
        for col in all_cols:
            subset = _subset_for_column(model_rows, col)
            p = _group_mean(subset, "triple_precision")
            r = _group_mean(subset, "triple_recall")
            f1 = _group_mean(subset, "triple_f1")
            prf = f"{_fmt(p, 3)}/{_fmt(r, 3)}/{_fmt(f1, 3)}"
            vals_num.append(prf)
            vals_str.append(prf)
        table_a_csv_rows.append([model] + vals_num)
        table_a_md_rows.append([model] + vals_str)

    # Table B: merged explain overhead table as explain/total (share%).
    headers_b = ["model"] + cols
    table_b_csv_rows: list[list[Any]] = []
    table_b_md_rows: list[list[str]] = []
    for model in models:
        model_rows = by_model[model]
        cells: list[str] = []
        csv_cells: list[str] = []
        for col in cols:
            subset = _subset_for_column(model_rows, col)
            explain_total = _group_sum(subset, "runtime_tokens_cite_explain")
            token_total = _group_sum(subset, "runtime_tokens_total")
            cell = _fmt_tokens_share(explain_total, token_total)
            cells.append(cell)
            csv_cells.append(cell)
        table_b_md_rows.append([model] + cells)
        table_b_csv_rows.append([model] + csv_cells)

    aggregates: list[ModelAggregate] = []

    for model in models:
        model_rows = by_model[model]
        with_impossible = _subset_for_column(model_rows, "all_with_impossible")
        without_impossible = _subset_for_column(model_rows, "all_without_impossible")
        mean_tokens = _group_mean(with_impossible, "runtime_tokens_total")
        mean_explain = _group_mean(with_impossible, "runtime_tokens_cite_explain")
        mean_input = _group_mean(with_impossible, "runtime_input_tokens")
        mean_output = _group_mean(with_impossible, "runtime_output_tokens")
        share = _group_explain_share_pct(with_impossible)
        mean_elapsed = _group_mean(with_impossible, "elapsed_s")
        total_elapsed_h = _group_sum(with_impossible, "elapsed_s") / 3600.0
        mean_f1_with = _group_mean(with_impossible, "triple_f1")
        mean_f1_without = _group_mean(without_impossible, "triple_f1")
        mean_p_without = _group_mean(without_impossible, "triple_precision")
        mean_r_without = _group_mean(without_impossible, "triple_recall")
        mean_p_with = _group_mean(with_impossible, "triple_precision")
        mean_r_with = _group_mean(with_impossible, "triple_recall")
        pricing = pricing_by_model.get(model)
        if pricing is None:
            raise ValueError(
                f"Missing pricing for model '{model}'. "
                "Add it to the pricing file and rerun."
            )
        mean_cost, total_cost = _estimate_cost_usd(with_impossible, pricing)

        agg = ModelAggregate(
            model=model,
            questions=len(with_impossible),
            mean_f1_with_impossible=mean_f1_with,
            mean_f1_without_impossible=mean_f1_without,
            mean_precision_without_impossible=mean_p_without,
            mean_recall_without_impossible=mean_r_without,
            mean_runtime_tokens_per_q=mean_tokens,
            mean_cite_explain_tokens_per_q=mean_explain,
            cite_explain_share_pct=share,
            mean_elapsed_s_per_q=mean_elapsed,
            total_elapsed_h=total_elapsed_h,
            mean_input_tokens_per_q=mean_input,
            mean_output_tokens_per_q=mean_output,
            mean_est_cost_per_q_usd=mean_cost,
            total_est_cost_usd=total_cost,
        )
        aggregates.append(agg)

    # Pareto charts per 2D objective:
    # - cost chart: F1 (max) vs cost/question (min)
    # - latency chart: F1 (max) vs latency/question (min)
    pareto_points = _build_pareto_points(aggregates)

    out_dir.mkdir(parents=True, exist_ok=True)

    table_a_csv = out_dir / "table_a_f1_by_model_qtype.csv"
    table_a_md = out_dir / "table_a_f1_by_model_qtype.md"
    table_b_csv = out_dir / "table_b_explain_overhead_share_by_model_qtype.csv"
    table_b_md = out_dir / "table_b_explain_overhead_share_by_model_qtype.md"
    model_aggregates_csv = out_dir / "model_aggregates.csv"
    pareto_cost_svg = out_dir / "chart_pareto_f1_vs_cost.svg"
    pareto_latency_svg = out_dir / "chart_pareto_f1_vs_latency.svg"
    summary_md = out_dir / "summary.md"
    export_latex = out_dir / "model_comparison_export.tex"

    _write_csv(table_a_csv, headers_a, table_a_csv_rows)
    _write_csv(table_b_csv, headers_b, table_b_csv_rows)
    _write_csv(
        model_aggregates_csv,
        [
            "model",
            "questions",
            "mean_f1_without_impossible",
            "mean_f1_with_impossible",
            "mean_est_cost_per_q_usd",
            "mean_elapsed_s_per_q",
        ],
        [
            [
                a.model,
                a.questions,
                a.mean_f1_without_impossible,
                a.mean_f1_with_impossible,
                a.mean_est_cost_per_q_usd,
                a.mean_elapsed_s_per_q,
            ]
            for a in aggregates
        ],
    )

    table_a_md.write_text(_render_markdown_table(headers_a, table_a_md_rows) + "\n")
    table_b_md.write_text(_render_markdown_table(headers_b, table_b_md_rows) + "\n")
    _write_pareto_svg(
        path=pareto_cost_svg,
        points=pareto_points,
        frontier_attr="cost_frontier",
        x_attr="cost_per_question_usd",
        x_label="Cost Per Question (USD)",
        y_attr="f1_without_impossible",
        y_label="F1 (without impossible questions)",
    )
    _write_pareto_svg(
        path=pareto_latency_svg,
        points=pareto_points,
        frontier_attr="latency_frontier",
        x_attr="mean_latency_s",
        x_label="Mean Latency Per Question (s)",
        y_attr="f1_without_impossible",
        y_label="F1 (without impossible questions)",
    )

    pricing_note = (
        "- Cost columns are estimated from `runtime_trace` input/output tokens "
        "and pricing file rates (`$/1M`)."
    )
    pareto_note = (
        "- Pareto frontiers are computed per chart objective pair "
        "(cost-vs-F1 and latency-vs-F1), using dominance in those 2D spaces."
    )

    summary_md.write_text(
        "\n".join(
            [
                "# Model Comparison Summary",
                "",
                "## Table A: F1 by model and qtype",
                "",
                _render_markdown_table(headers_a, table_a_md_rows),
                "",
                "## Table B: Explanation overhead (`cite+explain / total (share%)`)",
                "",
                _render_markdown_table(headers_b, table_b_md_rows),
                "",
                "## Pareto Charts",
                "",
                f"![Pareto F1 vs Cost](chart_pareto_f1_vs_cost.svg)",
                "",
                f"![Pareto F1 vs Latency](chart_pareto_f1_vs_latency.svg)",
                "",
                "## Notes",
                "",
                "- Runtime token counts are inferred from `runtime_trace[*].usage`.",
                "- If an event has multiple tool calls, event tokens are split evenly across those calls for explain-share.",
                "- Pareto chart rendering requires matplotlib.",
                pricing_note,
                pareto_note,
            ]
        )
        + "\n"
    )
    export_latex.write_text(
        _render_latex_export(
            qtype_cols=qtype_cols,
            table_a_md_rows=table_a_md_rows,
            table_b_md_rows=table_b_md_rows,
        )
    )

    return {
        "table_a_csv": table_a_csv,
        "table_a_md": table_a_md,
        "table_b_csv": table_b_csv,
        "table_b_md": table_b_md,
        "model_aggregates_csv": model_aggregates_csv,
        "chart_pareto_f1_vs_cost_svg": pareto_cost_svg,
        "chart_pareto_f1_vs_latency_svg": pareto_latency_svg,
        "model_comparison_export_tex": export_latex,
        "summary_md": summary_md,
    }


def _run_dataset_comparison(
    results_dir: Path,
    pricing_by_model: dict[str, ModelPricing],
) -> tuple[Path, dict[str, Path]]:
    input_files = _resolve_input_files(results_dir)
    if not input_files:
        raise ValueError(
            f"No input files found in directory: {results_dir}. "
            "Expected files named eval_results_*.toml (excluding *.scored.*)."
        )
    out_dir = Path("reports") / results_dir.name
    outputs = build_reports(
        input_files=input_files,
        out_dir=out_dir,
        pricing_by_model=pricing_by_model,
    )
    return out_dir, outputs


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval result files across models.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        help="Directory containing eval_results_*.toml files for one dataset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparisons for every dataset directory under ./results.",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        default=None,
        help=(
            "Pricing CSV with columns: model,input_per_1m,output_per_1m "
            "(default: reports/model_pricing.csv)."
        ),
    )
    args = parser.parse_args()
    pricing_file = _resolve_pricing_file(args.pricing_file)
    pricing_by_model = _load_pricing(pricing_file)

    if args.results_dir is None and not args.all:
        args.all = True

    if args.all and args.results_dir is not None:
        parser.error("Provide either a single results_dir or --all, not both.")

    if args.all:
        results_root = _resolve_default_results_root()
        if not results_root.exists() or not results_root.is_dir():
            raise SystemExit(
                f"Results root not found: {results_root}. "
                "Expected ./results (or ./result)."
            )
        dataset_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())
        if not dataset_dirs:
            raise SystemExit(f"No dataset directories found under: {results_root}")
        processed = 0
        for dataset_dir in dataset_dirs:
            try:
                out_dir, outputs = _run_dataset_comparison(dataset_dir, pricing_by_model)
            except ValueError:
                # Skip non-dataset directories without eval files.
                continue
            processed += 1
            print(f"Generated reports for dataset '{dataset_dir.name}' in {out_dir}:")
            for key, path in outputs.items():
                print(f"- {key}: {path}")
        if processed == 0:
            raise SystemExit(
                f"No eval_results_*.toml files found in any dataset under: {results_root}"
            )
    else:
        assert args.results_dir is not None
        try:
            out_dir, outputs = _run_dataset_comparison(args.results_dir, pricing_by_model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Generated reports for dataset '{args.results_dir.name}' in {out_dir}:")
        for key, path in outputs.items():
            print(f"- {key}: {path}")

    # Programmatically keep a cross-dataset combined table up to date.
    reports_root = Path("reports")
    if args.all:
        results_root = _resolve_default_results_root()
    else:
        assert args.results_dir is not None
        results_root = args.results_dir.parent
    dataset_report_dirs = _discover_dataset_report_dirs(results_root, reports_root)
    combined_outputs = build_combined_reports(dataset_report_dirs, reports_root / "combined")
    if combined_outputs:
        print("Generated combined reports:")
        for key, path in combined_outputs.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
