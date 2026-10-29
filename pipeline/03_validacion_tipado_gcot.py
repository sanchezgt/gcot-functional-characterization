"""Validación estructural del tipado producido por el estimador.

La etapa no reinterpreta semánticamente el texto. Comprueba únicamente el
contrato: correspondencia segmento-nodo, identificadores, offsets, tipos
admitidos y campos de ambigüedad.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from util.pipeline_canonical import ensure_pipeline_record, set_validation_stage
from util.sgotgraph_schema import validate_nodes


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida estructuralmente el tipado antes de construir aristas."
    )
    parser.add_argument(
        "input_path", type=Path, help="Ruta al archivo JSON o JSONL tipado."
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


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
        raise ValueError("El JSON debe contener un objeto o una lista de objetos.")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Línea {line_number}: JSON inválido: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Línea {line_number}: cada registro debe ser un objeto."
            )
        records.append(value)
    if not records:
        raise ValueError(f"No se encontraron registros en {path}.")
    return records


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    pipeline = ensure_pipeline_record(record)
    segments = pipeline.get("segments")
    nodes = pipeline.get("nodes")
    errors: list[str] = []
    if not isinstance(segments, list):
        errors.append("segments debe ser una lista.")
        segments = []
    if not isinstance(nodes, list):
        errors.append("nodes debe ser una lista.")
        nodes = []
    if any(not isinstance(item, dict) for item in segments):
        errors.append("Todos los segmentos deben ser objetos.")
    if any(not isinstance(item, dict) for item in nodes):
        errors.append("Todos los nodos deben ser objetos.")
    if not errors:
        errors.extend(validate_nodes(segments, nodes))
    return set_validation_stage(
        pipeline,
        "typing_review",
        {
            "status": "valid" if not errors else "invalid",
            "validation_kind": "structural_only",
            "errors": errors,
        },
    )


def write_output(
    results: list[dict[str, Any]],
    output_path: Path | None,
    pretty: bool,
) -> None:
    single = len(results) == 1
    if output_path is None:
        if pretty or single:
            json.dump(
                results[0] if single else results,
                sys.stdout,
                ensure_ascii=False,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            for result in results:
                sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix.lower() == ".json":
            json.dump(
                results[0] if single else results,
                handle,
                ensure_ascii=False,
                indent=2 if pretty else None,
            )
            handle.write("\n")
        else:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    configure_stdio()
    args = parse_args()
    try:
        results = [validate_record(record) for record in load_records(args.input_path)]
        write_output(results, args.output, args.pretty)
        return int(
            any(
                result["validation"]["typing_review"]["status"] == "invalid"
                for result in results
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
