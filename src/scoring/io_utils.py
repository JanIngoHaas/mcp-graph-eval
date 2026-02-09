"""Load/save evaluation data in JSON or TOML."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

try:
    import tomlkit
except Exception:  # pragma: no cover
    tomlkit = None

_SURROGATE_PAIR_RE = re.compile(
    r"\\u([dD][89aAbB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
)
_DROP = object()


def _replace_surrogate_pair(match: re.Match[str]) -> str:
    high = int(match.group(1), 16)
    low = int(match.group(2), 16)
    codepoint = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
    return f"\\U{codepoint:08X}"


def _repair_toml_surrogate_escapes(text: str) -> str:
    """Convert JSON-style UTF-16 surrogate pairs into TOML-valid \\U escapes."""
    return _SURROGATE_PAIR_RE.sub(_replace_surrogate_pair, text)


def _sanitize_for_toml(value: Any) -> Any:
    """Recursively drop None values so TOML serialization can succeed."""
    if value is None:
        return _DROP
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            child_clean = _sanitize_for_toml(child)
            if child_clean is _DROP:
                continue
            cleaned[key] = child_clean
        return cleaned
    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for child in value:
            child_clean = _sanitize_for_toml(child)
            if child_clean is _DROP:
                continue
            cleaned_list.append(child_clean)
        return cleaned_list
    return value


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
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            repaired = _repair_toml_surrogate_escapes(text)
            if repaired == text:
                raise
            try:
                return tomllib.loads(repaired)
            except tomllib.TOMLDecodeError:
                raise exc
    return json.loads(text)


def dump_data(data: dict[str, Any], path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return
    if fmt == "toml":
        if tomlkit is None:
            raise RuntimeError("tomlkit not available to write TOML output")
        cleaned = _sanitize_for_toml(data)
        if cleaned is _DROP:
            cleaned = {}
        path.write_text(tomlkit.dumps(cleaned), encoding="utf-8")
        return
    raise ValueError(f"Unsupported format: {fmt}")
