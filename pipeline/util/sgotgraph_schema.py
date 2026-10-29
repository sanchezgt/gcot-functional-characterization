"""Contrato canónico y validación de SGoTGraph para el objetivo específico 2.

El módulo no depende de un proveedor LLM. Distingue invariantes estructurales
duros de diagnósticos automáticos no bloqueantes; no reinterpreta ni completa
decisiones semánticas.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Iterable

from util.consensus import validate_consensus_record


SCHEMA_VERSION = "sgotgraph-objective2-1.1"
PIPELINE_VERSION = "objective2-1.1"

NODE_TYPES = {"PROP", "METHOD", "EVID", "ALT", "CHECK", "REVISE", "OUT"}
SEMANTIC_EDGE_TYPES = {"SUPPORTS", "ALTERNATIVE_TO", "CHECKS", "REVISES", "LEADS_TO"}
EDGE_TYPES = {"NEXT", *SEMANTIC_EDGE_TYPES}
PROVENANCE_ORIGINS = {"deterministic", "llm_proposed", "rule_filtered", "human_adjudicated"}
PROVENANCE_FIELDS = {
    "origin",
    "stage",
    "provider",
    "model",
    "prompt_hash",
    "response_id",
    "system_fingerprint",
    "reasoning_effort",
}
RECORD_FIELDS = {
    "schema_version",
    "pipeline_version",
    "trace_id",
    "source_record",
    "trace_text",
    "segments",
    "nodes",
    "edges",
    "validation",
    "audit",
}
SEGMENT_FIELDS = {"segment_id", "text", "start_char", "end_char", "provenance"}
NODE_REQUIRED_FIELDS = {
    "segment_id",
    "text",
    "start_char",
    "end_char",
    "position",
    "node_type",
    "provenance",
}
# ``consensus`` (opcional desde 1.1) registra el voto multimuestra que
# produjo la decisión; nunca sustituye a ``provenance``.
NODE_FIELDS = {
    *NODE_REQUIRED_FIELDS,
    "review_status",
    "ambiguity_codes",
    "consensus",
}
EDGE_REQUIRED_FIELDS = {"edge_id", "source", "target", "edge_type", "provenance"}
EDGE_FIELDS = {*EDGE_REQUIRED_FIELDS, "consensus"}

ALLOWED_TARGET_TYPES: dict[tuple[str, str], set[str]] = {
    ("PROP", "SUPPORTS"): {"PROP", "REVISE", "OUT"},
    ("PROP", "LEADS_TO"): {"PROP", "REVISE", "OUT"},
    ("METHOD", "LEADS_TO"): {"PROP", "REVISE", "OUT"},
    ("EVID", "SUPPORTS"): {"PROP", "REVISE", "OUT"},
    ("ALT", "ALTERNATIVE_TO"): {"PROP", "METHOD", "ALT"},
    ("CHECK", "CHECKS"): {"PROP", "METHOD", "EVID", "ALT"},
    ("REVISE", "REVISES"): {"PROP", "METHOD", "EVID", "ALT", "CHECK"},
    ("REVISE", "LEADS_TO"): {"PROP", "REVISE", "OUT"},
}

BACKWARD_EDGE_TYPES = {"CHECKS", "REVISES", "ALTERNATIVE_TO"}
FORWARD_EDGE_TYPES = {"LEADS_TO"}


def text_hash(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def make_provenance(
    origin: str,
    stage: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    prompt_hash: str | None = None,
    response_id: str | None = None,
    system_fingerprint: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"origin": origin, "stage": stage}
    if provider is not None:
        value["provider"] = provider
    if model is not None:
        value["model"] = model
    if prompt_hash is not None:
        value["prompt_hash"] = prompt_hash
    if response_id is not None:
        value["response_id"] = response_id
    if system_fingerprint is not None:
        value["system_fingerprint"] = system_fingerprint
    if reasoning_effort is not None:
        value["reasoning_effort"] = reasoning_effort
    return value


def append_audit_event(record: dict[str, Any], stage: str, event: dict[str, Any]) -> None:
    audit = record.setdefault("audit", {})
    events = audit.setdefault(stage, [])
    events.append(event)


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: dict[str, Any] = {}
    repeated: dict[str, Any] = {}
    for value in values:
        try:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            key = repr(value)
        if key in seen:
            repeated[key] = value
        seen[key] = value
    return list(repeated.values())


def _validate_object_fields(
    value: dict[Any, Any],
    label: str,
    *,
    required: set[str],
    allowed: set[str],
) -> list[str]:
    errors: list[str] = []
    keys = set(value)
    missing = required - keys
    if missing:
        errors.append(
            f"{label}: faltan campos obligatorios: "
            f"{sorted(str(field) for field in missing)}."
        )
    unknown = keys - allowed
    if unknown:
        errors.append(
            f"{label}: campos no permitidos: "
            f"{sorted(str(field) for field in unknown)}."
        )
    return errors


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_provenance(value: Any, object_label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{object_label}: provenance es obligatoria y debe ser un objeto."]
    errors = _validate_object_fields(
        value,
        f"{object_label} provenance",
        required={"origin", "stage"},
        allowed=PROVENANCE_FIELDS,
    )
    origin = value.get("origin")
    if not isinstance(origin, str) or origin not in PROVENANCE_ORIGINS:
        errors.append(f"{object_label}: provenance.origin inválido.")
    if not isinstance(value.get("stage"), str) or not value.get("stage"):
        errors.append(f"{object_label}: provenance.stage es obligatorio.")
    for field in ("provider", "model", "prompt_hash"):
        if field in value and not isinstance(value[field], str):
            errors.append(
                f"{object_label}: provenance.{field} debe ser una cadena."
            )
    return errors


def validate_segments(trace_text: str, segments: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not segments:
        return ["segments debe contener al menos un segmento."]

    ids = [segment.get("segment_id") for segment in segments]
    if any(not isinstance(value, str) or not value for value in ids):
        errors.append("Todos los segmentos deben tener segment_id no vacío.")
    if duplicates := _duplicates(ids):
        errors.append(
            f"segment_id duplicados: {sorted(str(value) for value in duplicates)}."
        )

    cursor = 0
    reconstructed: list[str] = []
    for index, segment in enumerate(segments, start=1):
        label = f"Segmento {segment.get('segment_id') or index}"
        errors.extend(
            _validate_object_fields(
                segment,
                label,
                required=SEGMENT_FIELDS,
                allowed=SEGMENT_FIELDS,
            )
        )
        start = segment.get("start_char")
        end = segment.get("end_char")
        text = segment.get("text")
        errors.extend(_validate_provenance(segment.get("provenance"), label))
        if not _is_integer(start) or not _is_integer(end):
            errors.append(f"{label}: offsets no enteros.")
            continue
        if start != cursor:
            errors.append(f"{label}: start_char={start}; se esperaba {cursor}.")
        if start < 0 or end <= start or end > len(trace_text):
            errors.append(f"{label}: intervalo fuera de rango, vacío o invertido.")
            continue
        if not isinstance(text, str) or not text:
            errors.append(f"{label}: text debe ser una cadena no vacía.")
            continue
        if trace_text[start:end] != text:
            errors.append(
                f"{label}: text no coincide con "
                "trace_text[start_char:end_char]."
            )
        reconstructed.append(text)
        cursor = end

    if cursor != len(trace_text):
        errors.append(
            f"La cobertura termina en {cursor}; la traza termina en {len(trace_text)}."
        )
    if "".join(reconstructed) != trace_text:
        errors.append("La concatenación de segmentos no reconstruye exactamente trace_text.")
    return errors


def validate_nodes(
    segments: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if len(nodes) != len(segments):
        errors.append(
            f"Debe existir un nodo por segmento: {len(nodes)} != {len(segments)}."
        )
    expected_ids = [segment.get("segment_id") for segment in segments]
    node_ids = [node.get("segment_id") for node in nodes]
    if node_ids != expected_ids:
        errors.append(
            "Los nodos no preservan exactamente el orden e identidad de los segmentos."
        )

    segment_by_id = {
        segment["segment_id"]: segment
        for segment in segments
        if isinstance(segment.get("segment_id"), str)
    }
    for index, node in enumerate(nodes, start=1):
        node_id = node.get("segment_id")
        label = f"Nodo {node_id or index}"
        errors.extend(
            _validate_object_fields(
                node,
                label,
                required=NODE_REQUIRED_FIELDS,
                allowed=NODE_FIELDS,
            )
        )
        node_type = node.get("node_type")
        errors.extend(_validate_provenance(node.get("provenance"), label))
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{label}: segment_id debe ser una cadena no vacía.")
        if not isinstance(node_type, str) or node_type not in NODE_TYPES:
            errors.append(f"{label}: node_type inválido: {node_type!r}.")
        if not isinstance(node.get("text"), str):
            errors.append(f"{label}: text debe ser una cadena.")
        if not _is_integer(node.get("start_char")) or not _is_integer(
            node.get("end_char")
        ):
            errors.append(f"{label}: offsets no enteros.")
        position_value = node.get("position")
        if not _is_integer(position_value) or position_value < 1:
            errors.append(f"{label}: position debe ser un entero positivo.")
        elif position_value != index:
            errors.append(
                f"{label}: position={position_value}; se esperaba {index}."
            )
        if "review_status" in node and not isinstance(node["review_status"], str):
            errors.append(f"{label}: review_status debe ser una cadena.")
        if "ambiguity_codes" in node and (
            not isinstance(node["ambiguity_codes"], list)
            or not all(isinstance(value, str) for value in node["ambiguity_codes"])
        ):
            errors.append(f"{label}: ambiguity_codes debe contener cadenas.")
        if "consensus" in node:
            errors.extend(validate_consensus_record(node["consensus"], label))
        segment = segment_by_id.get(node_id)
        if segment is None:
            continue
        for field in ("text", "start_char", "end_char"):
            if node.get(field) != segment.get(field):
                errors.append(f"{label}: {field} difiere del segmento.")
    return errors


def validate_edges(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    review: list[dict[str, Any]] = []
    node_by_id = {
        node["segment_id"]: node
        for node in nodes
        if isinstance(node.get("segment_id"), str)
    }
    position = {
        node["segment_id"]: index
        for index, node in enumerate(nodes, start=1)
        if isinstance(node.get("segment_id"), str)
    }

    edge_ids = [edge.get("edge_id") for edge in edges]
    if any(not isinstance(edge_id, str) or not edge_id for edge_id in edge_ids):
        errors.append("Todas las aristas deben tener edge_id no vacío.")
    if duplicates := _duplicates(edge_ids):
        errors.append(
            f"edge_id duplicados: {sorted(str(value) for value in duplicates)}."
        )

    next_edges = [edge for edge in edges if edge.get("edge_type") == "NEXT"]
    expected_next = max(0, len(nodes) - 1)
    if len(next_edges) != expected_next:
        errors.append(
            f"Cantidad NEXT inválida: {len(next_edges)}; se esperaba {expected_next}."
        )
    for index in range(expected_next):
        if index >= len(next_edges):
            break
        expected = (
            nodes[index].get("segment_id"),
            nodes[index + 1].get("segment_id"),
        )
        observed = (next_edges[index].get("source"), next_edges[index].get("target"))
        if observed != expected:
            errors.append(f"NEXT {index + 1}: {observed}; se esperaba {expected}.")

    semantic_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges, start=1):
        edge_id = edge.get("edge_id")
        label = f"Arista {edge_id or index}"
        errors.extend(
            _validate_object_fields(
                edge,
                label,
                required=EDGE_REQUIRED_FIELDS,
                allowed=EDGE_FIELDS,
            )
        )
        edge_type = edge.get("edge_type")
        source = edge.get("source")
        target = edge.get("target")
        errors.extend(_validate_provenance(edge.get("provenance"), label))
        if "consensus" in edge:
            errors.extend(validate_consensus_record(edge["consensus"], label))
        if not isinstance(edge_type, str) or edge_type not in EDGE_TYPES:
            errors.append(f"{label}: edge_type inválido {edge_type!r}.")
            continue
        if not isinstance(source, str) or not isinstance(target, str):
            errors.append(f"{label}: source y target deben ser cadenas.")
            continue
        if source not in node_by_id or target not in node_by_id:
            errors.append(f"{label}: origen o destino inexistente.")
            continue
        if source == target:
            errors.append(f"{label}: no se permiten autoaristas.")
        if edge_type == "NEXT":
            continue
        pair = (source, target)
        if pair in semantic_pairs:
            errors.append(f"Más de una arista semántica para el par {source}->{target}.")
        semantic_pairs.add(pair)

        source_type = node_by_id[source].get("node_type")
        target_type = node_by_id[target].get("node_type")
        allowed_targets = ALLOWED_TARGET_TYPES.get((source_type, edge_type))
        if allowed_targets is None or target_type not in allowed_targets:
            errors.append(
                f"{label}: combinación "
                f"{source_type}-{edge_type}->{target_type} no permitida."
            )
        if edge_type in BACKWARD_EDGE_TYPES and position[target] >= position[source]:
            errors.append(f"{label}: {edge_type} debe apuntar hacia contenido previo.")
        if edge_type in FORWARD_EDGE_TYPES and position[target] <= position[source]:
            errors.append(f"{label}: LEADS_TO debe apuntar hacia contenido posterior.")

    semantic_by_source: dict[str, set[str]] = {}
    for edge in edges:
        edge_type = edge.get("edge_type")
        source = edge.get("source")
        if edge_type in SEMANTIC_EDGE_TYPES and isinstance(source, str):
            semantic_by_source.setdefault(source, set()).add(edge_type)
    expected_anchor = {
        "CHECK": "CHECKS",
        "REVISE": "REVISES",
        "ALT": "ALTERNATIVE_TO",
    }
    for node in nodes:
        expected = expected_anchor.get(node.get("node_type"))
        node_id = node.get("segment_id")
        if (
            expected
            and isinstance(node_id, str)
            and expected not in semantic_by_source.get(node_id, set())
        ):
            review.append(
                {
                    "code": "MISSING_SEMANTIC_ANCHOR",
                    "segment_id": node_id,
                    "expected_edge_type": expected,
                }
            )

    out_ids = {
        node["segment_id"]
        for node in nodes
        if node.get("node_type") == "OUT"
        and isinstance(node.get("segment_id"), str)
    }
    for out_id in out_ids:
        if not any(
            edge.get("target") == out_id and edge.get("edge_type") in {"SUPPORTS", "LEADS_TO"}
            for edge in edges
        ):
            review.append({"code": "UNANCHORED_OUTPUT", "segment_id": out_id})
    if not out_ids:
        review.append({"code": "NO_EXPLICIT_OUTPUT"})
    return errors, review


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "status": "invalid",
            "hard_errors": ["El registro debe ser un objeto JSON."],
            "warnings": [],
        }
    hard_errors: list[str] = []
    warnings: list[dict[str, Any]] = []
    hard_errors.extend(
        _validate_object_fields(
            record,
            "Registro",
            required=RECORD_FIELDS,
            allowed=RECORD_FIELDS,
        )
    )
    if record.get("schema_version") != SCHEMA_VERSION:
        hard_errors.append(f"schema_version debe ser {SCHEMA_VERSION!r}.")
    if record.get("pipeline_version") != PIPELINE_VERSION:
        hard_errors.append(f"pipeline_version debe ser {PIPELINE_VERSION!r}.")
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        hard_errors.append("trace_id es obligatorio.")
    if not isinstance(record.get("source_record"), dict):
        hard_errors.append("source_record debe ser un objeto.")
    if not isinstance(record.get("validation"), dict):
        hard_errors.append("validation debe ser un objeto.")
    if not isinstance(record.get("audit"), dict):
        hard_errors.append("audit debe ser un objeto.")
    trace_text = record.get("trace_text")
    if not isinstance(trace_text, str) or trace_text == "":
        hard_errors.append("trace_text debe ser una cadena no vacía.")
        trace_text = ""
    segments = record.get("segments")
    nodes = record.get("nodes")
    edges = record.get("edges")
    if not isinstance(segments, list):
        hard_errors.append("segments debe ser una lista.")
        segments = []
    else:
        for index, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                hard_errors.append(f"Segmento {index}: debe ser un objeto.")
        segments = [segment for segment in segments if isinstance(segment, dict)]
    if not isinstance(nodes, list):
        hard_errors.append("nodes debe ser una lista.")
        nodes = []
    else:
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                hard_errors.append(f"Nodo {index}: debe ser un objeto.")
        nodes = [node for node in nodes if isinstance(node, dict)]
    if not isinstance(edges, list):
        hard_errors.append("edges debe ser una lista.")
        edges = []
    else:
        for index, edge in enumerate(edges, start=1):
            if not isinstance(edge, dict):
                hard_errors.append(f"Arista {index}: debe ser un objeto.")
        edges = [edge for edge in edges if isinstance(edge, dict)]

    hard_errors.extend(validate_segments(trace_text, segments))
    hard_errors.extend(validate_nodes(segments, nodes))
    edge_errors, edge_warnings = validate_edges(nodes, edges)
    hard_errors.extend(edge_errors)
    warnings.extend(edge_warnings)

    status = "invalid" if hard_errors else "valid"
    return {
        "status": status,
        "hard_errors": hard_errors,
        "warnings": warnings,
    }
