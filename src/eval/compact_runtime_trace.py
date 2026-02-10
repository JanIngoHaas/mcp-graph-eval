"""
Compact legacy eval runtime traces in existing result files.

This script rewrites `output[*].runtime_trace` from verbose callback event streams
(`llm_start`, `tool_start`, `tool_end`, ...) into compact, question-level message
history entries (`human`, `ai`, `tool`).

Usage:
  python -m src.eval.compact_runtime_trace results/hypmol/eval_results_gemini-3-flash-preview_cloud.toml
  python -m src.eval.compact_runtime_trace results/hypmol/*.toml --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.scoring.io_utils import load_data, dump_data, infer_format


_LEGACY_EVENT_TYPES = {
    "llm_start",
    "llm_end",
    "tool_start",
    "tool_end",
    "agent_action",
    "agent_finish",
    "token_usage",
}
_COMPACT_MESSAGE_TYPES = {"human", "ai", "tool", "system"}


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _looks_legacy_event_stream(runtime_trace: list[Any]) -> bool:
    for item in runtime_trace:
        if not isinstance(item, dict):
            continue
        if "step" in item:
            return True
        t = item.get("type")
        if isinstance(t, str) and t in _LEGACY_EVENT_TYPES:
            return True
    return False


def _looks_compact_message_stream(runtime_trace: list[Any]) -> bool:
    if not runtime_trace:
        return True
    for item in runtime_trace:
        if not isinstance(item, dict):
            return False
        if "type" not in item:
            return False
        t = item.get("type")
        if not isinstance(t, str) or t not in _COMPACT_MESSAGE_TYPES:
            return False
    return True


def _compact_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, list):
        return None
    compact: list[dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            compact.append(
                {
                    "name": tc.get("name"),
                    "args": _safe_jsonable(tc.get("args")),
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                }
            )
    return compact or None


def _usage_by_step(runtime_trace: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in runtime_trace:
        if event.get("type") != "token_usage":
            continue
        step = event.get("step")
        usage = event.get("usage")
        if isinstance(step, int) and isinstance(usage, dict):
            out[step] = usage
    return out


def _convert_runtime_trace(question: str, runtime_trace: list[Any]) -> list[dict[str, Any]]:
    if not runtime_trace:
        return []
    if _looks_compact_message_stream(runtime_trace):
        return [item for item in runtime_trace if isinstance(item, dict)]
    if not _looks_legacy_event_stream(runtime_trace):
        # Unknown shape, keep as-is.
        return [item for item in runtime_trace if isinstance(item, dict)]

    events = [item for item in runtime_trace if isinstance(item, dict)]
    usage_map = _usage_by_step(events)
    compact: list[dict[str, Any]] = []

    if question:
        compact.append({"type": "human", "content": question})

    last_tool_name: str | None = None
    for event in events:
        etype = event.get("type")
        if etype == "tool_start":
            tool_name = event.get("tool")
            last_tool_name = str(tool_name) if tool_name else last_tool_name
            continue

        if etype == "llm_end":
            content = event.get("content")
            tool_calls = _compact_tool_calls(event.get("tool_calls"))
            usage = usage_map.get(event.get("step")) if isinstance(event.get("step"), int) else None

            if content is None:
                content = ""
            ai_msg: dict[str, Any] = {"type": "ai", "content": _safe_jsonable(content)}
            if tool_calls:
                ai_msg["tool_calls"] = tool_calls
            if usage:
                ai_msg["usage"] = _safe_jsonable(usage)
            if ai_msg.get("content") or ai_msg.get("tool_calls") or ai_msg.get("usage"):
                compact.append(ai_msg)
            continue

        if etype == "tool_end":
            output = event.get("output")
            if output is None:
                continue
            tool_msg: dict[str, Any] = {"type": "tool", "content": _safe_jsonable(output)}
            if last_tool_name:
                tool_msg["name"] = last_tool_name
            compact.append(tool_msg)
            continue

    return compact


def _estimate_size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False))


def compact_file(path: Path, write: bool = False) -> tuple[int, int, int, int]:
    data = load_data(path)
    outputs = data.get("output")
    if not isinstance(outputs, list):
        return 0, 0, 0, 0

    changed_entries = 0
    old_total = 0
    new_total = 0
    converted_events = 0

    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        runtime_trace = entry.get("runtime_trace")
        if not isinstance(runtime_trace, list):
            continue

        question = str(entry.get("question") or "")
        old_size = _estimate_size(runtime_trace)
        new_trace = _convert_runtime_trace(question, runtime_trace)
        new_size = _estimate_size(new_trace)
        old_total += old_size
        new_total += new_size

        if new_trace != runtime_trace:
            changed_entries += 1
            converted_events += len(runtime_trace)
            entry["runtime_trace"] = new_trace

    if write and changed_entries > 0:
        fmt = infer_format(path, None)
        dump_data(data, path, fmt)

    return changed_entries, converted_events, old_total, new_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact runtime_trace in existing eval result files.")
    parser.add_argument("files", nargs="+", help="Result files (.toml/.json) to compact")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes in-place. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    any_error = False
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"[ERROR] {path}: file not found")
            any_error = True
            continue

        try:
            changed, converted_events, old_size, new_size = compact_file(path, write=args.write)
            mode = "WROTE" if args.write else "DRY-RUN"
            shrink = old_size - new_size
            pct = (shrink / old_size * 100.0) if old_size else 0.0
            print(
                f"[{mode}] {path} | changed_entries={changed} "
                f"| converted_events={converted_events} "
                f"| runtime_trace_bytes: {old_size} -> {new_size} "
                f"({pct:.1f}% smaller)"
            )
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            any_error = True

    if any_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

