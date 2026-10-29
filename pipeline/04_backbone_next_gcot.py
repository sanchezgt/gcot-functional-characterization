from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from util.pipeline_canonical import ensure_pipeline_record, set_validation_stage
from util.sgotgraph_schema import make_provenance


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el backbone NEXT a partir de nodos tipados de SGoTGraph."
    )
    parser.add_argument("input_path", type=Path, help="Ruta al archivo JSON o JSONL tipado.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Ruta de salida. Si no se indica, imprime en stdout.",
    )
    parser.add_argument("--trace-id", help="Procesa solo el registro con este id.")
    parser.add_argument(
        "--index",
        type=int,
        help="Procesa solo el registro en esta posicion 0-based.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Procesa solo los primeros N registros seleccionados.",
    )
    parser.add_argument("--pretty", action="store_true", help="Imprime JSON con indentacion.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No se encontraron registros en {path}.")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("El archivo JSON debe contener un objeto o una lista.")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Linea {line_number}: JSON invalido: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Linea {line_number}: cada registro debe ser un objeto JSON.")
        records.append(row)
    return records


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
        raise ValueError("La seleccion quedo vacia.")
    return selected


def validate_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not nodes:
        return ["El registro no contiene nodos."]

    segment_ids = [node.get("segment_id") for node in nodes]
    if any(not segment_id for segment_id in segment_ids):
        errors.append("Hay nodos sin segment_id.")
    if len(set(segment_ids)) != len(segment_ids):
        errors.append("Hay segment_id duplicados.")

    previous_end = None
    for index, node in enumerate(nodes, start=1):
        start_char = node.get("start_char")
        end_char = node.get("end_char")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            errors.append(f"Nodo {index} sin offsets validos.")
            continue
        if start_char >= end_char:
            errors.append(f"Nodo {index} con rango invalido.")
        if previous_end is not None and start_char < previous_end:
            errors.append(f"Nodo {index} rompe el orden de offsets.")
        previous_end = end_char

    return errors


def build_next_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for i in range(1, len(nodes)):
        source = nodes[i - 1]["segment_id"]
        target = nodes[i]["segment_id"]
        edges.append(
            {
                "edge_id": f"e_next_{i}",
                "source": source,
                "target": target,
                "edge_type": "NEXT",
                "provenance": make_provenance("deterministic", "backbone"),
            }
        )
    return edges


def validate_backbone(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_edges = max(0, len(nodes) - 1)
    if len(edges) != expected_edges:
        errors.append(
            f"Numero de aristas NEXT invalido: esperado {expected_edges}, recibido {len(edges)}."
        )

    for index, node in enumerate(nodes):
        if index == 0:
            continue
        expected_source = nodes[index - 1]["segment_id"]
        expected_target = node["segment_id"]
        edge = edges[index - 1] if index - 1 < len(edges) else None
        if edge is None:
            errors.append(f"Falta arista NEXT para {expected_target}.")
            continue
        if edge.get("edge_type") != "NEXT":
            errors.append(f"Arista {edge.get('edge_id')} no es NEXT.")
        if edge.get("source") != expected_source or edge.get("target") != expected_target:
            errors.append(
                f"Arista NEXT incorrecta: esperado {expected_source} -> {expected_target}."
            )

    return errors


def build_backbone(record: dict[str, Any]) -> dict[str, Any]:
    pipeline = ensure_pipeline_record(record)
    trace_id = pipeline.get("trace_id")
    nodes = pipeline.get("nodes")
    if not trace_id:
        raise ValueError("Cada registro debe tener trace_id.")
    if not isinstance(nodes, list):
        raise ValueError(f"El registro {trace_id!r} no tiene una lista de nodos.")

    node_errors = validate_nodes(nodes)
    if node_errors:
        raise ValueError(f"Error en nodos de {trace_id!r}: {' | '.join(node_errors)}")

    enriched_nodes = []
    for position, node in enumerate(nodes, start=1):
        enriched_node = dict(node)
        enriched_node["position"] = position
        enriched_nodes.append(enriched_node)

    edges = build_next_edges(enriched_nodes)
    backbone_errors = validate_backbone(enriched_nodes, edges)
    if backbone_errors:
        raise ValueError(f"Error en backbone de {trace_id!r}: {' | '.join(backbone_errors)}")

    pipeline["nodes"] = enriched_nodes
    pipeline["edges"] = edges
    return set_validation_stage(
        pipeline,
        "graph",
        {
            "status": "valid",
            "backbone": "valid",
        },
    )


def write_output(results: list[dict[str, Any]], output_path: Path | None, pretty: bool) -> None:
    single_output = len(results) == 1

    if output_path is None:
        if pretty or single_output:
            payload = results[0] if single_output else results
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            for result in results:
                sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix.lower() == ".json":
            payload = results[0] if single_output else results
            json.dump(
                payload,
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
        records = load_records(args.input_path)
        selected = select_records(records, args.trace_id, args.index, args.limit)
        results = [build_backbone(record) for record in selected]
        write_output(results, args.output, args.pretty)
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
