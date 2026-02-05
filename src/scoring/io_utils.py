"""Load/save evaluation data in JSON or TOML."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None


def infer_format(path: Path, format_hint: str | None) -> str:
    if format_hint:
        fmt = format_hint.lower()
        if fmt not in {"json", "toml"}:
            raise ValueError(f"Unsupported format: {format_hint}")
        return fmt
    suffix = path.suffix.lower()
    if suffix in {".toml", ".tml"}:
        return "toml"
    return "json"


def load_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("tomllib not available to read TOML input")
        return tomllib.loads(text)
    return json.loads(text)


def dump_data(data: dict[str, Any], path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return
    if fmt == "toml":
        path.write_text(toml_dumps(data), encoding="utf-8")
        return
    raise ValueError(f"Unsupported format: {fmt}")


def toml_dumps(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _dump_table(lines, [], data, emit_header=False)
    return "\n".join(lines).rstrip() + "\n"


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(str(value))


def _format_key(key: str) -> str:
    if key.replace("_", "").replace("-", "").isalnum():
        return key
    return json.dumps(key)


def _format_array(values: list[Any]) -> str:
    items = ", ".join(_format_scalar(v) for v in values)
    return f"[{items}]"


def _dump_table(lines: list[str], path: list[str], data: dict[str, Any], emit_header: bool = True) -> None:
    if emit_header and path:
        header = ".".join(_format_key(p) for p in path)
        lines.append(f"[{header}]")
    scalars: dict[str, Any] = {}
    subtables: dict[str, dict[str, Any]] = {}
    array_tables: dict[str, list[dict[str, Any]]] = {}

    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            subtables[key] = value
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            array_tables[key] = value  # type: ignore[assignment]
        else:
            scalars[key] = value

    for key, value in scalars.items():
        key_out = _format_key(key)
        if isinstance(value, list):
            if not value:
                lines.append(f"{key_out} = []")
            elif all(_is_scalar(v) for v in value):
                lines.append(f"{key_out} = {_format_array(value)}")
            else:
                lines.append(f"{key_out} = {json.dumps(value)}")
        else:
            lines.append(f"{key_out} = {_format_scalar(value)}")

    if scalars and (subtables or array_tables):
        lines.append("")

    for key, table in subtables.items():
        _dump_table(lines, path + [key], table, emit_header=True)
        lines.append("")

    for key, tables in array_tables.items():
        for table in tables:
            header = ".".join(_format_key(p) for p in (path + [key]))
            lines.append(f"[[{header}]]")
            _dump_table(lines, path + [key], table, emit_header=False)
            lines.append("")
