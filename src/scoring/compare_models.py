"""Cross-model comparison and resource analysis for eval result files."""

from __future__ import annotations

import argparse
import csv
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_triple_f1

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


def _resolve_input_files(inputs: list[Path] | None) -> list[Path]:
    if not inputs:
        return sorted(Path(".").glob("eval_results_*.toml"))

    resolved: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            resolved.extend(sorted(input_path.glob("eval_results_*.toml")))
            continue
        if input_path.is_file():
            resolved.append(input_path)
            continue
        raise ValueError(f"Input path not found: {input_path}")

    # Preserve deterministic order while removing duplicates.
    return list(dict.fromkeys(resolved))


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


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _qtype_display_name(qtype: str) -> str:
    mapped = {
        "direct": "Direct",
        "hop": "Multi-hop",
        "query_builder": "Query Builder",
        "impossible": "Impossible",
    }
    if qtype in mapped:
        return mapped[qtype]
    return qtype.replace("_", " ").title()


def _latex_prf_cell(prf: str) -> str:
    parts = [p.strip() for p in prf.split("/")]
    if len(parts) != 3:
        return _latex_escape(prf)
    p, r, f1 = (_latex_escape(parts[0]), _latex_escape(parts[1]), _latex_escape(parts[2]))
    return rf"\makecell[r]{{\textbf{{P}} {p}\\\textbf{{R}} {r}\\\textbf{{F1}} {f1}}}"


def _latex_share_cell(share: str) -> str:
    value_part, sep, pct_part = share.partition(" (")
    if not sep:
        return _latex_escape(share)
    value_render = _latex_escape(value_part.replace("/", " / "))
    pct_render = _latex_escape("(" + pct_part)
    return rf"\makecell[r]{{{value_render}\\{pct_render}}}"


def _render_latex_export(
    qtype_cols: list[str],
    table_a_md_rows: list[list[str]],
    table_b_md_rows: list[list[str]],
) -> str:
    qtype_labels = [_qtype_display_name(q) for q in qtype_cols]
    colspec_a = "l" + ("c" * len(qtype_cols)) + "cc"
    colspec_b = "l" + ("c" * (len(qtype_cols) + 2))

    header_a = [r"\thead{Model}"]
    for label in qtype_labels:
        header_a.append(rf"\thead{{{_latex_escape(label)}\\F1}}")
    header_a.extend(
        [
            r"\thead{All Answerable Questions\\Precision / Recall / F1}",
            r"\thead{All Questions (Including Impossible)\\Precision / Recall / F1}",
        ]
    )

    header_b = [r"\thead{Model}"]
    for label in qtype_labels:
        header_b.append(rf"\thead{{{_latex_escape(label)} Questions\\Explain / Total (Share)}}")
    header_b.extend(
        [
            r"\thead{All Answerable Questions\\Explain / Total (Share)}",
            r"\thead{All Questions (Including Impossible)\\Explain / Total (Share)}",
        ]
    )

    table_a_lines: list[str] = []
    for row in table_a_md_rows:
        model = row[0]
        qtype_vals = row[1 : 1 + len(qtype_cols)]
        prf_without = row[1 + len(qtype_cols)]
        prf_with = row[2 + len(qtype_cols)]
        rendered = [rf"\texttt{{{_latex_escape(model)}}}"]
        rendered.extend(_latex_escape(v) for v in qtype_vals)
        rendered.append(_latex_prf_cell(prf_without))
        rendered.append(_latex_prf_cell(prf_with))
        table_a_lines.append(" & ".join(rendered) + r" \\")

    table_b_lines: list[str] = []
    for row in table_b_md_rows:
        model = row[0]
        shares = row[1:]
        rendered = [rf"\texttt{{{_latex_escape(model)}}}"]
        rendered.extend(_latex_share_cell(v) for v in shares)
        table_b_lines.append(" & ".join(rendered) + r" \\")

    return "\n".join(
        [
            "% Auto-generated by src.scoring.compare_models",
            "%",
            "% Required packages in your preamble:",
            "% \\usepackage{booktabs}",
            "% \\usepackage{makecell}",
            "% \\usepackage{graphicx}",
            "% \\usepackage{adjustbox}",
            "% \\usepackage{subcaption}",
            "% \\usepackage{svg}",
            "",
            r"\renewcommand\theadfont{\bfseries}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Model quality by question type with grouped Precision/Recall/F1 columns}",
            r"\label{tab:model_quality_by_qtype}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\textwidth}",
            rf"\begin{{tabular}}{{{colspec_a}}}",
            r"\toprule",
            " & ".join(header_a) + r" \\",
            r"\midrule",
            *table_a_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Explanation overhead by question type}",
            r"\label{tab:explanation_overhead_by_qtype}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\textwidth}",
            rf"\begin{{tabular}}{{{colspec_b}}}",
            r"\toprule",
            " & ".join(header_b) + r" \\",
            r"\midrule",
            *table_b_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\begin{subfigure}[t]{0.49\textwidth}",
            r"\centering",
            r"\includesvg[width=\linewidth]{chart_pareto_f1_vs_cost}",
            r"\caption{Cost-efficiency comparison. X-axis: estimated cost per question in USD. Y-axis: F1 score on answerable questions (impossible questions excluded).}",
            r"\label{fig:pareto_cost}",
            r"\end{subfigure}",
            r"\hfill",
            r"\begin{subfigure}[t]{0.49\textwidth}",
            r"\centering",
            r"\includesvg[width=\linewidth]{chart_pareto_f1_vs_latency}",
            r"\caption{Latency-efficiency comparison. X-axis: mean latency per question in seconds. Y-axis: F1 score on answerable questions (impossible questions excluded).}",
            r"\label{fig:pareto_latency}",
            r"\end{subfigure}",
            r"\caption{Pareto charts with explicit axis definitions.}",
            r"\label{fig:pareto_charts}",
            r"\end{figure}",
            "",
        ]
    )


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


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


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


def _build_pareto_points(aggregates: list[ModelAggregate]) -> list[ParetoPoint]:
    missing_cost_models = [a.model for a in aggregates if a.mean_est_cost_per_q_usd is None]
    if missing_cost_models:
        raise ValueError(
            "Missing USD cost for models (check pricing file coverage): "
            + ", ".join(sorted(missing_cost_models))
        )

    points: list[tuple[str, float, float, float]] = []
    for a in aggregates:
        points.append(
            (
                a.model,
                a.mean_f1_without_impossible,
                float(a.mean_est_cost_per_q_usd),
                a.mean_elapsed_s_per_q,
            )
        )

    out_points: list[ParetoPoint] = []
    for model, f1, cost_val, latency in points:
        cost_others = [(of1, ocost) for om, of1, ocost, _ in points if om != model]
        latency_others = [(of1, olat) for om, of1, _, olat in points if om != model]
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


def _write_pareto_svg(
    path: Path,
    points: list[ParetoPoint],
    frontier_attr: str,
    x_attr: str,
    x_label: str,
    y_attr: str,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for Pareto rendering. "
            "Install dependencies and rerun."
        ) from exc

    if not points:
        path.write_text("")
        return

    frontier = [p for p in points if bool(getattr(p, frontier_attr))]
    dominated = [p for p in points if not bool(getattr(p, frontier_attr))]

    plt.figure(figsize=(10, 5.2), dpi=120)

    def _plot_group(group: list[ParetoPoint], color: str, label: str) -> None:
        if not group:
            return
        xs = [getattr(p, x_attr) for p in group]
        ys = [getattr(p, y_attr) for p in group]
        plt.scatter(xs, ys, c=color, label=label, s=50, alpha=0.9, edgecolors="none")
        for p in group:
            plt.annotate(
                p.model,
                (getattr(p, x_attr), getattr(p, y_attr)),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    _plot_group(frontier, "#d62728", "Pareto frontier")
    _plot_group(dominated, "#1f77b4", "Dominated")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


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
    pareto_cost_svg = out_dir / "chart_pareto_f1_vs_cost.svg"
    pareto_latency_svg = out_dir / "chart_pareto_f1_vs_latency.svg"
    summary_md = out_dir / "summary.md"
    pricing_template_csv = out_dir / "model_pricing_template.csv"
    export_latex = out_dir / "model_comparison_export.tex"

    _write_csv(table_a_csv, headers_a, table_a_csv_rows)
    _write_csv(table_b_csv, headers_b, table_b_csv_rows)

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

    if not pricing_template_csv.exists():
        with pricing_template_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "input_per_1m", "output_per_1m"])
            for model in models:
                writer.writerow([model, "", ""])

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
        "chart_pareto_f1_vs_cost_svg": pareto_cost_svg,
        "chart_pareto_f1_vs_latency_svg": pareto_latency_svg,
        "model_comparison_export_tex": export_latex,
        "pricing_template_csv": pricing_template_csv,
        "summary_md": summary_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval result files across models.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Input eval result TOML files and/or directories. "
            "Directories are expanded as eval_results_*.toml. "
            "Default: eval_results_*.toml in cwd."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/model_comparison"),
        help="Output directory for generated tables and CSVs.",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        required=True,
        help="CSV with columns: model,input_per_1m,output_per_1m",
    )
    args = parser.parse_args()

    try:
        input_files = _resolve_input_files(args.inputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not input_files:
        raise SystemExit(
            "No input files found. Provide eval_results_*.toml files or directories containing them."
        )

    pricing_by_model = _load_pricing(args.pricing_file)
    outputs = build_reports(
        input_files=input_files,
        out_dir=args.out_dir,
        pricing_by_model=pricing_by_model,
    )
    print("Generated reports:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
