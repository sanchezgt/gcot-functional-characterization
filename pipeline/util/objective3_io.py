"""Entrada, selección y salida compartidas por las etapas del objetivo 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No se encontraron registros en {path}.")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        raise ValueError(
            "El archivo JSON debe contener un objeto o una lista de objetos."
        )

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Línea {line_number}: JSON inválido: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"Línea {line_number}: cada registro debe ser un objeto JSON."
            )
        records.append(row)

    if not records:
        raise ValueError(f"No se encontraron registros en {path}.")
    return records


def records_by_trace_id(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        trace_id = record.get("trace_id") or record.get("id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"{label} {index}: falta trace_id.")
        if trace_id in indexed:
            raise ValueError(f"{label}: trace_id duplicado {trace_id!r}.")
        indexed[trace_id] = record
    return indexed


def select_records(
    records: list[dict[str, Any]],
    trace_id: str | None,
    index: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if trace_id is not None and index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")

    selected = records
    if trace_id is not None:
        selected = [record for record in records if record.get("trace_id") == trace_id]
        if not selected:
            raise ValueError(f"No existe un registro con trace_id={trace_id!r}.")
    elif index is not None:
        if index < 0 or index >= len(records):
            raise ValueError(f"--index debe estar entre 0 y {len(records) - 1}.")
        selected = [records[index]]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit debe ser mayor que 0.")
        selected = selected[:limit]

    if not selected:
        raise ValueError("La selección quedó vacía.")
    return selected


def write_records(
    records: list[dict[str, Any]],
    output_path: Path | None,
    pretty: bool,
) -> None:
    single = len(records) == 1
    if output_path is None:
        if pretty or single:
            payload: Any = records[0] if single else records
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for record in records:
                sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix.lower() == ".json":
            payload = records[0] if single else records
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2 if pretty else None,
            )
            handle.write("\n")
        else:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
