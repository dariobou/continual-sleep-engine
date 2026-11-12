"""Configuration loader and JSONL / SHA256 I/O utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar

import yaml

from src.schemas import SafetyThresholdConfig, _CompatModel

T = TypeVar("T", bound=_CompatModel)


def load_yaml_config(path: str | Path) -> dict:
    """Load and parse a YAML configuration file safely."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_safety_thresholds(path: str | Path = "configs/safety_thresholds.yaml") -> SafetyThresholdConfig:
    """Load SafetyThresholdConfig from YAML."""
    raw = load_yaml_config(path)
    return SafetyThresholdConfig(**raw.get("thresholds", {}))


def append_jsonl(path: str | Path, item: _CompatModel) -> None:
    """Append a validated Pydantic model instance as a single line to a JSONL file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(item.to_json_line() + "\n")


def read_jsonl(path: str | Path, model_cls: type[T]) -> list[T]:
    """Read and validate all lines of a JSONL file into Pydantic model instances."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[T] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(model_cls.from_json_line(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid record at {file_path}:{line_num}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, items: list[_CompatModel]) -> None:
    """Write a list of Pydantic models to a JSONL file atomically."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_json_line() + "\n")


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 digest of a file (e.g., GGUF model binary)."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
