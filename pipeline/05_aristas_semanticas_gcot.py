"""Etapa 05: inferencia LLM de aristas semánticas de SGoTGraph.

Estimador canónico: un nodo fuente por solicitud, con todos los demás nodos
como candidatos (carga completa) y un inventario de opciones admisibles. El
código no elige destinos, no aplica prioridades semánticas y no trunca
relaciones: comprueba el contrato formal y devuelve la salida al estimador si
lo incumple. Un nodo sin opciones admisibles no se consulta.

Variantes de transporte no canónicas (``--payload compact``, lotes de
fuentes, ventana de candidatos): quedan disponibles para ablaciones y trazas
muy largas, marcadas en el artefacto como ``estimator_variant``. La medición
de septiembre de 2026 mostró que todas cambian lo que el estimador decide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from util.llm_provider import (
    DEFAULT_OPENAI_MODEL,
    LLMClient,
    add_llm_provider_args,
    build_llm_client,
    finalize_llm_args,
    request_structured_response,
    structured_request_hash,
)
from util.llm_checkpoint import (
    ItemCheckpoint,
    canonical_json_hash,
    checkpoint_path_for_trace,
)
from util.edge_batching import (
    DEFAULT_PAYLOAD,
    PAYLOAD_COMPACT,
    PAYLOAD_FULL,
    validate_transport_combination,
    DEFAULT_MAX_OPTIONS_PER_REQUEST,
    DEFAULT_SOURCES_PER_REQUEST,
    candidate_policy_label,
    chunk_sources,
    estimator_variant,
    validate_batch_offset,
    validate_candidate_window,
    validate_max_options,
    validate_sources_per_request,
)
from util.consensus import (
    CONSENSUS_METHOD_VERSION,
    DEFAULT_CONSENSUS_SAMPLES,
    consensus_record,
    majority_set,
    validate_sample_count,
)
from util.pipeline_canonical import ensure_pipeline_record, set_validation_stage
from util.sgotgraph_schema import (
    ALLOWED_TARGET_TYPES,
    BACKWARD_EDGE_TYPES,
    FORWARD_EDGE_TYPES,
    SEMANTIC_EDGE_TYPES,
    append_audit_event,
    make_provenance,
    validate_edges,
)
from util.stage_cache import implementation_tree_hash


DEFAULT_EDGE_MODEL = DEFAULT_OPENAI_MODEL
RELATION_OPTION_TRANSPORT_VERSION = "relation-option-refs-v1"
COMPACT_TRANSPORT_VERSION = "relation-option-refs-v2-batched"


SYSTEM_PROMPT = """You infer outgoing semantic edges for one fixed node in an
SGoTGraph representation of a generated reasoning trace.

The segmentation, node types, offsets, and NEXT backbone are already fixed.
Use the complete trace context supplied through the current node and all other
nodes. The semantic edge ontology is:
- SUPPORTS: from content that provides an explicit reason or grounding to the
  content it supports.
- ALTERNATIVE_TO: from an ALT node to the identifiable content or route with
  which it competes.
- CHECKS: from a CHECK node to the identifiable content it evaluates.
- REVISES: from a REVISE node to the identifiable content it corrects,
  replaces, retracts, or adjusts.
- LEADS_TO: from earlier content whose execution or use directly produces
  later content.

Rules:
1. Return all and only justified outgoing relations from the current node.
2. Select relations exclusively through the relation_ref identifiers supplied
   in the allowed option inventory. Never reproduce a target or edge type.
3. Textual adjacency, topic similarity, compatible vocabulary, or NEXT alone
   never justify a semantic edge.
4. Do not infer hidden model states or use external task correctness.
5. Do not invent nodes or relation types.
6. At most one semantic relation may be returned for an ordered source-target
   pair. If more than one interpretation is plausible, select the single
   relation best supported by the operational definitions; there is no global
   priority among relation types.
7. An empty relation_refs array is valid when no relation is justified.
Return exactly one JSON object matching the requested schema.
"""


USER_PROMPT_TEMPLATE = """Infer outgoing semantic edges for the current node.

Return:
- source: the current segment_id
- relation_refs: an array of identifiers from the allowed option inventory

Allowed source-relation-target options:
{allowed_relation_options_json}

The relation_refs array must be a subset of those exact option identifiers.
Returning an empty array is correct when none is semantically justified.

Current node:
{current_node_json}

All candidate nodes:
{candidate_nodes_json}
"""


REPAIR_PROMPT_SUFFIX = """

The previous output violated the deterministic relation contract. Return the
complete corrected output for the same current node. Preserve relation_refs
that remain explicitly justified, and revise or remove every invalid
reference. Use only relation_ref identifiers from the exact allowed options
above. Do not add a relation merely to satisfy validation.

Validation errors:
{validation_errors_json}

Previous output:
{previous_output_json}
"""


COMPACT_SYSTEM_PROMPT = """You infer outgoing semantic edges for each of the listed
source nodes of an SGoTGraph representation of a generated reasoning trace,
independently and with the same definitions. The grouping of source nodes in
this request is a transport convenience and carries no semantic information.

The segmentation, node types, offsets, and NEXT backbone are already fixed.
Use the complete trace context supplied through the ordered list of all nodes.
Each source node has its own inventory of allowed relation options. The
semantic edge ontology is:
- SUPPORTS: from content that provides an explicit reason or grounding to the
  content it supports.
- ALTERNATIVE_TO: from an ALT node to the identifiable content or route with
  which it competes.
- CHECKS: from a CHECK node to the identifiable content it evaluates.
- REVISES: from a REVISE node to the identifiable content it corrects,
  replaces, retracts, or adjusts.
- LEADS_TO: from earlier content whose execution or use directly produces
  later content.

Rules:
1. For each source node, return all and only justified outgoing relations.
2. Select relations exclusively through the relation_ref identifiers of that
   source node's own inventory. Never reproduce a target or edge type, and
   never use an identifier from another source node's inventory.
3. Textual adjacency, topic similarity, compatible vocabulary, or NEXT alone
   never justify a semantic edge.
4. Do not infer hidden model states or use external task correctness.
5. Do not invent nodes or relation types.
6. At most one semantic relation may be returned for an ordered source-target
   pair. If more than one interpretation is plausible, select the single
   relation best supported by the operational definitions; there is no global
   priority among relation types.
7. An empty relation_refs array is valid when no relation is justified.
8. Return exactly one entry per listed source node and no entry for any other
   node.
Return exactly one JSON object matching the requested schema.
"""


COMPACT_USER_PROMPT_TEMPLATE = """Infer outgoing semantic edges for each source node.

Return:
- sources: one entry per source node, each with
  - source: that source node's segment_id
  - relation_refs: an array of identifiers taken from that node's
    allowed_relation_options

In allowed_relation_options each edge type maps to a list of
[relation_ref, target_segment_id] pairs. Each relation_refs array must be a
subset of that node's exact identifiers. Returning an empty array is correct
when none is semantically justified.

Source nodes with their allowed relation options:
{current_nodes_json}

All nodes of the trace in textual order (complete context):
{all_nodes_json}
"""


COMPACT_REPAIR_PROMPT_SUFFIX = """

The previous output violated the deterministic relation contract for the
source nodes listed above. Return the complete corrected output for exactly
those source nodes. Preserve relation_refs that remain explicitly justified,
and revise or remove every invalid reference. Use only relation_ref
identifiers from each node's exact allowed options above. Do not add a
relation merely to satisfy validation.

Validation errors by source:
{validation_errors_json}

Previous output for these sources:
{previous_output_json}
"""


class NodeRelationsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    relation_refs: list[str]


class BatchRelationsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[NodeRelationsDraft]

def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Construye aristas semánticas SGoTGraph mediante inferencia LLM "
            "sobre el conjunto completo de nodos."
        )
    )
    parser.add_argument(
        "input_path", type=Path, help="JSON o JSONL con el backbone NEXT."
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--trace-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    add_llm_provider_args(parser, DEFAULT_EDGE_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_EDGE_REASONING_EFFORT", "low"),
        choices=["none", "minimal", "low", "medium", "high"],
        help=(
            "Esfuerzo de razonamiento del analizador. Forma parte del "
            "instrumento: el canónico usa low. "
            "Default: OPENAI_EDGE_REASONING_EFFORT o low."
        ),
    )
    parser.add_argument(
        "--prev-window",
        type=int,
        help="Opción retirada: provoca un error explícito; use --candidate-window.",
    )
    parser.add_argument(
        "--future-window",
        type=int,
        help="Opción retirada: provoca un error explícito; use --candidate-window.",
    )
    parser.add_argument(
        "--payload",
        choices=[PAYLOAD_FULL, PAYLOAD_COMPACT],
        default=DEFAULT_PAYLOAD,
        help=(
            "Carga del prompt. full (predeterminado, canónico): un nodo por "
            "solicitud con todos los candidatos, offsets e inventario plano. "
            "compact: NO canónico; habilita lotes y ventana y reduce tokens de "
            "entrada, pero cambia lo que el estimador decide."
        ),
    )
    parser.add_argument(
        "--sources-per-request",
        type=int,
        default=DEFAULT_SOURCES_PER_REQUEST,
        help=(
            "Nodos fuente consecutivos que comparten una solicitud (el "
            "contexto se envía una vez por lote; la aceptación sigue siendo "
            "por fuente). Opcional y NO canónico: exige --payload compact; no "
            "cambia las relaciones expresables, pero sí lo que el estimador "
            "decide (menos densidad, menos relaciones lejanas, sesgo por "
            f"posición). Default: {DEFAULT_SOURCES_PER_REQUEST}."
        ),
    )
    parser.add_argument(
        "--batch-offset",
        type=int,
        default=0,
        help=(
            "Acorta el primer lote en este número de fuentes para desplazar "
            "las fronteras de los lotes (medición de sesgo). Solo tiene "
            "efecto con --payload compact y varios nodos por solicitud. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--max-options-per-request",
        type=int,
        default=DEFAULT_MAX_OPTIONS_PER_REQUEST,
        help=(
            "Tope de opciones admisibles por solicitud; un lote se corta antes "
            "de superarlo. Sin efecto en el estimador canónico (una fuente "
            f"por solicitud). Default: {DEFAULT_MAX_OPTIONS_PER_REQUEST}."
        ),
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=None,
        help=(
            "Opcional y NO canónico: limita los destinos a los nodos situados "
            "a esta distancia de posición o menos. Restringe las relaciones "
            "expresables; el artefacto queda marcado como variante "
            "'windowed'. Default: sin ventana."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "Intentos máximos por muestra y nodo ante una salida semántica "
            "inválida. Default: 3."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por nodo fuente; se conservan las "
            "aristas presentes en una mayoría estricta. 1 reproduce el "
            f"estimador de una sola muestra. Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Directorio opcional para persistir decisiones aceptadas por "
            "nodo fuente."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        action="store_true",
        help=(
            "Reutiliza decisiones del checkpoint solo si su identidad y "
            "validación coinciden exactamente."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="En JSONL, omite trace_id ya presentes y agrega los nuevos.",
    )
    return finalize_llm_args(parser.parse_args(), DEFAULT_EDGE_MODEL)


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
        selected = [
            record for record in records if record.get("trace_id") == trace_id
        ]
        if not selected:
            raise ValueError(f"No existe trace_id={trace_id!r}.")
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


def load_existing_trace_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.suffix.lower() == ".json":
        return set()
    result: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace_id = value.get("trace_id") if isinstance(value, dict) else None
        if isinstance(trace_id, str):
            result.add(trace_id)
    return result


def append_jsonl_output(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def select_candidate_nodes(
    nodes: list[dict[str, Any]],
    current_index: int,
    prev_window: int | None = None,
    future_window: int | None = None,
    candidate_window: int | None = None,
) -> list[dict[str, Any]]:
    """Candidatos de destino para un nodo fuente.

    Sin ventana devuelve todos los demás nodos. ``candidate_window`` limita los
    candidatos a los nodos situados a esa distancia de posición o menos; es
    una restricción operacional explícita que se registra en la auditoría,
    marca la variante como no canónica y está desactivada por defecto.
    ``prev_window`` y ``future_window`` se conservan por compatibilidad de API
    y no intervienen.
    """

    _ = (prev_window, future_window)
    window = validate_candidate_window(candidate_window)
    current_position = nodes[current_index]["position"]
    candidates: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if index == current_index:
            continue
        delta = node["position"] - current_position
        if window is not None and abs(delta) > window:
            continue
        candidate = dict(node)
        candidate["relative_position"] = abs(delta)
        candidate["temporal_relation"] = "previous" if delta < 0 else "next"
        candidates.append(candidate)
    return candidates


def _allowed_relation_options(
    current_node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
    prefixed: bool = False,
) -> list[dict[str, str]]:
    source_type = current_node["node_type"]
    options: list[dict[str, str]] = []
    for target in candidate_nodes:
        for edge_type in sorted(SEMANTIC_EDGE_TYPES):
            allowed_targets = ALLOWED_TARGET_TYPES.get(
                (source_type, edge_type)
            )
            if (
                allowed_targets is None
                or target["node_type"] not in allowed_targets
            ):
                continue
            if (
                edge_type in BACKWARD_EDGE_TYPES
                and target["position"] >= current_node["position"]
            ):
                continue
            if (
                edge_type in FORWARD_EDGE_TYPES
                and target["position"] <= current_node["position"]
            ):
                continue
            options.append(
                {
                    "relation_ref": "",
                    "source": current_node["segment_id"],
                    "edge_type": edge_type,
                    "target": target["segment_id"],
                }
            )
    # Alias planos en el transporte canónico; con prefijo de fuente en la
    # carga compacta, donde un alias ajeno debe caer fuera del inventario.
    prefix = f"{current_node['segment_id']}:" if prefixed else ""
    for option_index, option in enumerate(options, start=1):
        option["relation_ref"] = f"{prefix}R{option_index:04d}"
    return options


def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {
        key: node[key]
        for key in ("segment_id", "node_type", "position", "text")
    }


def _options_payload(options: list[dict[str, str]]) -> dict[str, list[list[str]]]:
    """Inventario denso: tipo de arista -> pares [alias, destino]."""

    payload: dict[str, list[list[str]]] = {}
    for option in options:
        payload.setdefault(option["edge_type"], []).append(
            [option["relation_ref"], option["target"]]
        )
    return payload


def build_source_item(
    node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
    payload: str = DEFAULT_PAYLOAD,
) -> dict[str, Any]:
    """Un nodo fuente con sus candidatos e inventario de opciones admisibles."""

    options = _allowed_relation_options(
        node, candidate_nodes, prefixed=payload == PAYLOAD_COMPACT
    )
    return {
        "payload": payload,
        "node": node,
        "candidates": candidate_nodes,
        "options": options,
        "option_by_ref": {option["relation_ref"]: option for option in options},
    }


def build_batch_prompt(
    batch: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> str:
    """Prompt de un lote: contexto completo una vez, inventario por fuente.

    La carga es compacta: por nodo solo identificador, tipo, posición y texto
    íntegro; el inventario agrupa las opciones por tipo de arista sin repetir
    la fuente. Los offsets no intervienen en la selección de alias y se omiten.
    """

    current_nodes_json = json.dumps(
        [
            {
                **_node_payload(item["node"]),
                "allowed_relation_options": _options_payload(item["options"]),
            }
            for item in batch
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    all_nodes_json = json.dumps(
        [_node_payload(node) for node in nodes],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return COMPACT_USER_PROMPT_TEMPLATE.format(
        current_nodes_json=current_nodes_json,
        all_nodes_json=all_nodes_json,
    )


def build_prompt(
    current_node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
) -> str:
    current_payload = {
        key: current_node[key]
        for key in (
            "segment_id",
            "node_type",
            "position",
            "start_char",
            "end_char",
            "text",
        )
    }
    candidate_payload = [
        {
            key: node[key]
            for key in (
                "segment_id",
                "node_type",
                "position",
                "start_char",
                "end_char",
                "text",
                "relative_position",
                "temporal_relation",
            )
        }
        for node in candidate_nodes
    ]
    return USER_PROMPT_TEMPLATE.format(
        allowed_relation_options_json=json.dumps(
            _allowed_relation_options(current_node, candidate_nodes),
            ensure_ascii=False,
            indent=2,
        ),
        current_node_json=json.dumps(
            current_payload, ensure_ascii=False, indent=2
        ),
        candidate_nodes_json=json.dumps(
            candidate_payload, ensure_ascii=False, indent=2
        ),
    )


def call_model(
    client: LLMClient,
    model: str,
    batch: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    reasoning_effort: str,
    verbose: bool,
    user_prompt: str | None = None,
    attempt: int = 1,
    max_attempts: int = 1,
) -> BatchRelationsDraft:
    """Consulta al LLM por las fuentes de ``batch``.

    Con la carga completa (canónica) el lote tiene una sola fuente y la
    solicitud es la original: prompt de un nodo y respuesta
    ``NodeRelationsDraft``, que se envuelve para compartir el resto del flujo.
    """

    source_ids = ",".join(item["node"]["segment_id"] for item in batch)
    log(
        f"[{source_ids}] Infiriendo aristas salientes con "
        f"{client.provider}:{model} (intento {attempt}/{max_attempts}).",
        verbose,
    )
    if batch[0]["payload"] == PAYLOAD_FULL:
        (item,) = batch
        response = request_structured_response(
            client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
            if user_prompt is not None
            else build_prompt(item["node"], item["candidates"]),
            response_model=NodeRelationsDraft,
            reasoning_effort=reasoning_effort,
        )
        return BatchRelationsDraft(sources=[response.parsed])
    response = request_structured_response(
        client,
        model=model,
        system_prompt=COMPACT_SYSTEM_PROMPT,
        user_prompt=user_prompt
        if user_prompt is not None
        else build_batch_prompt(batch, nodes),
        response_model=BatchRelationsDraft,
        reasoning_effort=reasoning_effort,
    )
    return response.parsed


def validate_node_relations(
    current_node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
    draft: NodeRelationsDraft,
    batch_source_ids: list[str] | None = None,
    option_by_ref: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    errors: list[str] = []
    source_id = current_node["segment_id"]
    if option_by_ref is None:
        option_by_ref = {
            option["relation_ref"]: option
            for option in _allowed_relation_options(
                current_node,
                candidate_nodes,
            )
        }
    if draft.source != source_id:
        errors.append(
            f"source={draft.source!r}; se esperaba {source_id!r}."
        )
    seen_refs: set[str] = set()
    seen_targets: set[str] = set()
    for relation_ref in draft.relation_refs:
        if relation_ref in seen_refs:
            errors.append(f"relation_ref duplicado: {relation_ref!r}.")
            continue
        seen_refs.add(relation_ref)
        option = option_by_ref.get(relation_ref)
        if option is None:
            owner = relation_ref.split(":", 1)[0] if ":" in relation_ref else None
            if (
                batch_source_ids
                and owner in batch_source_ids
                and owner != source_id
            ):
                errors.append(
                    f"relation_ref {relation_ref!r} pertenece a otra fuente "
                    f"del lote ({owner!r})."
                )
            else:
                errors.append(
                    f"relation_ref inexistente o no admitido: "
                    f"{relation_ref!r}."
                )
            continue
        target = option["target"]
        if target in seen_targets:
            errors.append(
                f"Más de una relación para {source_id}->{target}."
            )
        seen_targets.add(target)
    return errors


def validate_batch_by_source(
    batch: list[dict[str, Any]],
    draft: BatchRelationsDraft,
) -> tuple[dict[str, list[str]], list[str]]:
    """Errores por fuente pedida y notas del lote no atribuibles a una fuente.

    Una fuente sin entrada o con entradas repetidas es inválida; una entrada
    para una fuente no solicitada no invalida a las demás y se registra como
    nota del lote.
    """

    batch_ids = [item["node"]["segment_id"] for item in batch]
    returned = [entry.source for entry in draft.sources]
    notes: list[str] = []
    extra = sorted({source_id for source_id in returned if source_id not in batch_ids})
    if extra:
        notes.append(f"Entradas para fuentes no solicitadas (ignoradas): {extra}.")
    errors_by_source: dict[str, list[str]] = {}
    for item in batch:
        source_id = item["node"]["segment_id"]
        entries = [entry for entry in draft.sources if entry.source == source_id]
        if not entries:
            errors_by_source[source_id] = ["Falta la entrada de esta fuente."]
            continue
        if len(entries) > 1:
            errors_by_source[source_id] = [
                f"Entradas repetidas para la fuente ({len(entries)})."
            ]
            continue
        errors_by_source[source_id] = validate_node_relations(
            item["node"],
            item["candidates"],
            entries[0],
            batch_ids,
            item["option_by_ref"],
        )
    return errors_by_source, notes


def validate_batch_relations(
    batch: list[dict[str, Any]],
    draft: BatchRelationsDraft,
) -> list[str]:
    """Contrato del lote como lista plana de errores con prefijo de fuente."""

    errors_by_source, notes = validate_batch_by_source(batch, draft)
    errors = list(notes)
    for source_id, source_errors in errors_by_source.items():
        errors.extend(f"[{source_id}] {error}" for error in source_errors)
    return errors

def _semantic_edge_id(
    trace_id: str,
    source: str,
    target: str,
    edge_type: str,
) -> str:
    payload = json.dumps(
        [trace_id, source, target, edge_type],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "e_sem_" + sha256(payload.encode("utf-8")).hexdigest()


def _validate_cached_sample(
    cached: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    draft = NodeRelationsDraft.model_validate(cached["draft"])
    prompt_hash = cached["prompt_hash"]
    attempts = cached["attempts"]
    history = cached["history"]
    if not isinstance(prompt_hash, str) or not prompt_hash.startswith(
        "sha256:"
    ):
        raise ValueError("prompt_hash inválido.")
    if not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("attempts inválido.")
    if (
        not isinstance(history, list)
        or len(history) != attempts
        or not all(isinstance(event, dict) for event in history)
    ):
        raise ValueError("history inválido.")
    final_event = history[-1]
    if (
        final_event.get("attempt") != attempts
        or final_event.get("prompt_hash") != prompt_hash
        or final_event.get("validation_status") != "valid"
        or final_event.get("validation_errors") != []
    ):
        raise ValueError("El evento final no coincide con la decisión.")
    validation_errors = validate_node_relations(
        item["node"],
        item["candidates"],
        draft,
        option_by_ref=item["option_by_ref"],
    )
    if validation_errors:
        raise ValueError(" | ".join(validation_errors))
    return {
        "draft": draft,
        "prompt_hash": prompt_hash,
        "attempts": attempts,
        "history": history,
    }


def load_cached_source(
    checkpoint: ItemCheckpoint,
    item: dict[str, Any],
    samples: int,
) -> list[dict[str, Any]] | None:
    """Recupera las ``samples`` muestras aceptadas de una fuente.

    Solo se reutiliza una entrada con exactamente ``samples`` muestras que
    revalidan contra el inventario actual; cualquier discrepancia la descarta.
    """

    source_id = item["node"]["segment_id"]
    cached = checkpoint.get(source_id)
    if cached is None:
        return None
    try:
        if cached["consensus_samples"] != samples:
            raise ValueError("El número de muestras del checkpoint difiere.")
        stored = cached["samples"]
        if not isinstance(stored, list) or len(stored) != samples:
            raise ValueError("samples inválido.")
        results = [_validate_cached_sample(sample, item) for sample in stored]
    except (KeyError, TypeError, ValueError, ValidationError):
        checkpoint.discard(source_id)
        return None
    checkpoint.mark_reused(source_id)
    return results


def sample_node_relations(
    client: LLMClient,
    model: str,
    item: dict[str, Any],
    nodes: list[dict[str, Any]],
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    sample_index: int,
) -> dict[str, Any]:
    """Una muestra válida del estimador canónico para un nodo fuente.

    Prompt, errores de validación y sufijo de reparación son los originales:
    una salida inválida se devuelve completa al LLM con sus errores hasta
    ``max_attempts`` veces. El código nunca recorta ni completa la selección.
    """

    current_node, candidates = item["node"], item["candidates"]
    source_id = current_node["segment_id"]
    base_prompt = build_prompt(current_node, candidates)
    validation_errors: list[str] = []
    previous_output: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt
        if previous_output is not None:
            prompt += REPAIR_PROMPT_SUFFIX.format(
                validation_errors_json=json.dumps(
                    validation_errors,
                    ensure_ascii=False,
                    indent=2,
                ),
                previous_output_json=json.dumps(
                    previous_output,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        prompt_hash = structured_request_hash(client, SYSTEM_PROMPT, prompt)
        batch_draft = call_model(
            client,
            model,
            [item],
            nodes,
            reasoning_effort,
            verbose,
            prompt,
            attempt,
            max_attempts,
        )
        if len(batch_draft.sources) != 1:
            raise ValueError(
                f"El estimador canónico devolvió {len(batch_draft.sources)} "
                f"entradas para {source_id}; se esperaba una."
            )
        draft = batch_draft.sources[0]
        validation_errors = validate_node_relations(
            current_node,
            candidates,
            draft,
            option_by_ref=item["option_by_ref"],
        )
        history.append(
            {
                "sample": sample_index,
                "attempt": attempt,
                "prompt_hash": prompt_hash,
                "validation_status": (
                    "valid" if not validation_errors else "invalid"
                ),
                "validation_errors": validation_errors,
            }
        )
        if not validation_errors:
            return {
                "draft": draft,
                "prompt_hash": prompt_hash,
                "attempts": attempt,
                "history": history,
            }
        previous_output = draft.model_dump()
        log(
            f"[{source_id}] Muestra {sample_index}: salida inválida: "
            + " | ".join(validation_errors),
            verbose,
        )
    raise ValueError(
        f"Salida inválida para {source_id} (muestra {sample_index}) tras "
        f"{max_attempts} intentos: " + " | ".join(validation_errors)
    )


def query_sources(
    client: LLMClient,
    model: str,
    pending: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    sample_index: int,
) -> dict[str, dict[str, Any]]:
    """Una muestra válida por fuente para un subconjunto de fuentes del lote.

    Todas las fuentes pendientes se consultan juntas con el contexto completo.
    Una fuente cuya entrada cumple el contrato queda aceptada tal cual; solo
    las inválidas se vuelven a consultar, en el mismo contexto y con sus
    errores, hasta ``max_attempts``. El código nunca recorta ni completa una
    selección: la unidad de aceptación es la fuente.
    """

    if pending[0]["payload"] == PAYLOAD_FULL:
        return {
            item["node"]["segment_id"]: sample_node_relations(
                client,
                model,
                item,
                nodes,
                reasoning_effort,
                verbose,
                max_attempts,
                sample_index,
            )
            for item in pending
        }

    accepted: dict[str, dict[str, Any]] = {}
    history: dict[str, list[dict[str, Any]]] = {
        item["node"]["segment_id"]: [] for item in pending
    }
    previous_entries: dict[str, dict[str, Any]] = {}
    errors_by_source: dict[str, list[str]] = {}
    for attempt in range(1, max_attempts + 1):
        pending_ids = [item["node"]["segment_id"] for item in pending]
        prompt = build_batch_prompt(pending, nodes)
        if attempt > 1:
            prompt += COMPACT_REPAIR_PROMPT_SUFFIX.format(
                validation_errors_json=json.dumps(
                    {source_id: errors_by_source.get(source_id, []) for source_id in pending_ids},
                    ensure_ascii=False,
                    indent=2,
                ),
                previous_output_json=json.dumps(
                    {
                        "sources": [
                            previous_entries[source_id]
                            for source_id in pending_ids
                            if source_id in previous_entries
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        prompt_hash = structured_request_hash(client, COMPACT_SYSTEM_PROMPT, prompt)
        draft = call_model(
            client,
            model,
            pending,
            nodes,
            reasoning_effort,
            verbose,
            prompt,
            attempt,
            max_attempts,
        )
        errors_by_source, notes = validate_batch_by_source(pending, draft)
        entries = {entry.source: entry for entry in draft.sources}
        still_pending: list[dict[str, Any]] = []
        for item in pending:
            source_id = item["node"]["segment_id"]
            source_errors = errors_by_source[source_id]
            history[source_id].append(
                {
                    "sample": sample_index,
                    "attempt": attempt,
                    "requested_sources": pending_ids,
                    "prompt_hash": prompt_hash,
                    "validation_status": "valid" if not source_errors else "invalid",
                    "validation_errors": source_errors,
                    "batch_notes": notes,
                }
            )
            if source_errors:
                still_pending.append(item)
                if source_id in entries:
                    previous_entries[source_id] = entries[source_id].model_dump()
                continue
            accepted[source_id] = {
                "draft": entries[source_id],
                "prompt_hash": prompt_hash,
                "attempts": attempt,
                "history": history[source_id],
            }
        if not still_pending:
            return accepted
        log(
            f"[{','.join(item['node']['segment_id'] for item in still_pending)}] "
            f"Muestra {sample_index}: entradas inválidas; se vuelven a consultar "
            f"(intento {attempt}/{max_attempts}).",
            verbose,
        )
        pending = still_pending
    failed = {
        item["node"]["segment_id"]: errors_by_source.get(item["node"]["segment_id"], [])
        for item in pending
    }
    raise ValueError(
        f"Salida inválida para las fuentes {sorted(failed)} (muestra "
        f"{sample_index}) tras {max_attempts} intentos: "
        + " | ".join(
            f"[{source_id}] " + " ; ".join(errors)
            for source_id, errors in failed.items()
        )
    )


def build_semantic_edges(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    prev_window: int | None,
    future_window: int | None,
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int = 3,
    checkpoint_dir: Path | None = None,
    resume_checkpoint: bool = False,
    samples: int = 1,
    sources_per_request: int = DEFAULT_SOURCES_PER_REQUEST,
    candidate_window: int | None = None,
    batch_offset: int = 0,
    max_options_per_request: int = DEFAULT_MAX_OPTIONS_PER_REQUEST,
    payload: str = DEFAULT_PAYLOAD,
) -> dict[str, Any]:
    """Construye las aristas semánticas por fuente (o por lotes) y consenso.

    ``payload="full"`` (predeterminado) es el estimador canónico: una fuente
    por solicitud con la carga completa original. ``payload="compact"``
    habilita la carga compacta, los lotes y la ventana, todos no canónicos.

    Los nodos fuente con inventario no vacío se agrupan en bloques
    consecutivos de hasta ``sources_per_request`` fuentes (y hasta
    ``max_options_per_request`` opciones); cada bloque comparte el contexto
    completo y se muestrea ``samples`` veces. La aceptación, el checkpoint y el
    voto son por fuente: una fuente ya aceptada o recuperada del checkpoint no
    se vuelve a consultar, y solo las pendientes se piden en el contexto del
    bloque. Se conservan las aristas presentes en una mayoría estricta de las
    muestras. ``candidate_window`` (opcional) limita los destinos a esa
    distancia de posición y marca el artefacto como variante no canónica. Un
    nodo sin opciones admisibles no se consulta: el contrato ya excluye toda
    arista saliente.

    En el estimador canónico cada bloque es una sola fuente. La función usa
    ``samples=1`` por defecto (estimador de una muestra); el consenso
    canónico ``k = 3`` lo fija la línea de comandos de la etapa y de los
    orquestadores.

    Procedencia: ``prompt_hashes_by_source`` y ``provenance.prompt_hash`` son
    la huella de la solicitud BASE del bloque. Las huellas efectivas de cada
    intento, incluidas las de reparación y las de un bloque compacto reanudado
    solo con sus fuentes pendientes, están en ``attempt_history_by_source``.
    """

    pipeline = ensure_pipeline_record(record)
    trace_id = pipeline.get("trace_id")
    nodes = pipeline.get("nodes")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("Cada registro debe tener trace_id.")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{trace_id!r} no contiene nodos tipados.")
    if max_attempts <= 0:
        raise ValueError("max_attempts debe ser mayor que 0.")
    validate_sample_count(samples)
    validate_sources_per_request(sources_per_request)
    validate_batch_offset(batch_offset, sources_per_request)
    validate_candidate_window(candidate_window)
    validate_max_options(max_options_per_request)
    validate_transport_combination(payload, sources_per_request, candidate_window)
    candidate_policy = candidate_policy_label(candidate_window)
    variant = estimator_variant(candidate_window, sources_per_request, payload)
    transport = (
        RELATION_OPTION_TRANSPORT_VERSION
        if payload == PAYLOAD_FULL
        else COMPACT_TRANSPORT_VERSION
    )

    checkpoint = ItemCheckpoint(
        checkpoint_path_for_trace(
            checkpoint_dir,
            "05_aristas_semanticas",
            trace_id,
        ),
        identity={
            "stage": "05_aristas_semanticas",
            "implementation_hash": implementation_tree_hash(
                Path(__file__).resolve().parent
            ),
            "input_hash": canonical_json_hash(
                {
                    "trace_id": trace_id,
                    "trace_text": pipeline.get("trace_text"),
                    "nodes": nodes,
                    "edges": pipeline.get("edges", []),
                }
            ),
            "provider": client.provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "consensus_samples": samples,
            "sources_per_request": sources_per_request,
            "batch_offset": batch_offset,
            "max_options_per_request": max_options_per_request,
            "candidate_window": candidate_window,
            "candidate_policy": candidate_policy,
            "payload": payload,
            "transport": transport,
        },
        resume=resume_checkpoint,
    )

    proposed: list[dict[str, Any]] = []
    per_node_counts: dict[str, int] = {}
    prompt_hashes: dict[str, str] = {}
    attempts_by_source: dict[str, int] = {}
    samples_by_source: dict[str, int] = {}
    attempt_history_by_source: dict[str, list[dict[str, Any]]] = {}
    minority_relations_by_source: dict[str, list[dict[str, Any]]] = {}
    excluded_by_window_by_source: dict[str, int] = {}
    skipped_sources: list[str] = []
    repaired_sources: list[str] = []
    mixed_prompt_sources: list[str] = []
    cross_source_alias_errors = 0
    items: list[dict[str, Any]] = []
    for index, current_node in enumerate(nodes):
        candidates = select_candidate_nodes(
            nodes,
            index,
            prev_window,
            future_window,
            candidate_window,
        )
        source_id = current_node["segment_id"]
        if candidate_window is not None:
            excluded_by_window_by_source[source_id] = (
                len(nodes) - 1 - len(candidates)
            )
        item = build_source_item(current_node, candidates, payload)
        if not item["options"]:
            # Sin opciones admisibles no hay decisión semántica que tomar: el
            # contrato ya excluye toda arista saliente de este nodo.
            skipped_sources.append(source_id)
            per_node_counts[source_id] = 0
            attempts_by_source[source_id] = 0
            samples_by_source[source_id] = 0
            attempt_history_by_source[source_id] = [
                {
                    "sample": 0,
                    "attempt": 0,
                    "skipped": "empty_inventory",
                    "validation_status": "valid",
                    "validation_errors": [],
                }
            ]
            log(
                f"[{source_id}] Sin opciones admisibles; no se consulta al LLM.",
                verbose,
            )
            continue
        items.append(item)

    request_batches: list[dict[str, Any]] = []
    batches = chunk_sources(
        items,
        sources_per_request,
        offset=batch_offset,
        max_options=max_options_per_request,
        weight=lambda item: len(item["options"]),
    )
    for batch_index, batch in enumerate(batches, start=1):
        batch_ids = [item["node"]["segment_id"] for item in batch]
        batch_prompt_hash = (
            structured_request_hash(
                client,
                SYSTEM_PROMPT,
                build_prompt(batch[0]["node"], batch[0]["candidates"]),
            )
            if payload == PAYLOAD_FULL
            else structured_request_hash(
                client,
                COMPACT_SYSTEM_PROMPT,
                build_batch_prompt(batch, nodes),
            )
        )
        for source_id in batch_ids:
            prompt_hashes[source_id] = batch_prompt_hash

        samples_for_source: dict[str, list[dict[str, Any]]] = {}
        pending: list[dict[str, Any]] = []
        for item in batch:
            cached = load_cached_source(checkpoint, item, samples)
            if cached is None:
                pending.append(item)
            else:
                samples_for_source[item["node"]["segment_id"]] = cached
        pending_ids = [item["node"]["segment_id"] for item in pending]
        if pending_ids and len(pending_ids) < len(batch_ids):
            log(
                f"[{','.join(batch_ids)}] Lote parcialmente en checkpoint; se "
                f"consultan solo {pending_ids}.",
                verbose,
            )
        elif not pending_ids:
            log(
                f"[{','.join(batch_ids)}] Aristas recuperadas del checkpoint.",
                verbose,
            )
        if pending:
            per_sample = [
                query_sources(
                    client,
                    model,
                    pending,
                    nodes,
                    reasoning_effort,
                    verbose,
                    max_attempts,
                    sample_index,
                )
                for sample_index in range(1, samples + 1)
            ]
            for item in pending:
                source_id = item["node"]["segment_id"]
                samples_for_source[source_id] = [
                    result[source_id] for result in per_sample
                ]
                checkpoint.put(
                    source_id,
                    {
                        "consensus_samples": samples,
                        "batch_sources": batch_ids,
                        "batch_prompt_hash": batch_prompt_hash,
                        "samples": [
                            {
                                "draft": sample["draft"].model_dump(),
                                "prompt_hash": sample["prompt_hash"],
                                "attempts": sample["attempts"],
                                "history": sample["history"],
                            }
                            for sample in samples_for_source[source_id]
                        ],
                    },
                )
        request_batches.append(
            {
                "batch_index": batch_index,
                "sources": batch_ids,
                "queried_sources": pending_ids,
                "prompt_hash": batch_prompt_hash,
                "option_count": sum(len(item["options"]) for item in batch),
            }
        )

        for item in batch:
            source_id = item["node"]["segment_id"]
            option_by_ref = item["option_by_ref"]
            source_samples = samples_for_source[source_id]
            attempts_by_source[source_id] = sum(
                sample["attempts"] for sample in source_samples
            )
            samples_by_source[source_id] = len(source_samples)
            events = [event for sample in source_samples for event in sample["history"]]
            attempt_history_by_source[source_id] = events
            if any(sample["attempts"] > 1 for sample in source_samples):
                repaired_sources.append(source_id)
            if len({sample["prompt_hash"] for sample in source_samples}) > 1:
                mixed_prompt_sources.append(source_id)
            cross_source_alias_errors += sum(
                1
                for event in events
                for error in event.get("validation_errors", [])
                if "pertenece a otra fuente" in error
            )
            vote = majority_set(
                [
                    [
                        (
                            option_by_ref[relation_ref]["source"],
                            option_by_ref[relation_ref]["target"],
                            option_by_ref[relation_ref]["edge_type"],
                        )
                        for relation_ref in sample["draft"].relation_refs
                    ]
                    for sample in source_samples
                ]
            )
            per_node_counts[source_id] = len(vote["kept"])
            for source, target, edge_type in vote["kept"]:
                relation: dict[str, Any] = {
                    "source": source,
                    "target": target,
                    "edge_type": edge_type,
                }
                if len(source_samples) > 1:
                    relation["consensus"] = consensus_record(
                        vote["samples"],
                        vote["support"][(source, target, edge_type)],
                    )
                proposed.append(relation)
            if len(source_samples) > 1:
                minority_relations_by_source[source_id] = [
                    {
                        "source": source,
                        "target": target,
                        "edge_type": edge_type,
                        "support": support,
                        "samples": vote["samples"],
                    }
                    for (source, target, edge_type), support in vote["support"].items()
                    if support < vote["threshold"]
                ]

    semantic_edges: list[dict[str, Any]] = []
    for relation in proposed:
        edge: dict[str, Any] = {
            "edge_id": _semantic_edge_id(
                trace_id,
                relation["source"],
                relation["target"],
                relation["edge_type"],
            ),
            "source": relation["source"],
            "target": relation["target"],
            "edge_type": relation["edge_type"],
            "provenance": make_provenance(
                "llm_proposed",
                "semantic_edges",
                provider=client.provider,
                model=model,
                prompt_hash=prompt_hashes[relation["source"]],
                reasoning_effort=reasoning_effort,
            ),
        }
        if "consensus" in relation:
            edge["consensus"] = relation["consensus"]
        semantic_edges.append(edge)
    consensus_method = CONSENSUS_METHOD_VERSION if samples > 1 else None
    next_edges = [
        edge
        for edge in pipeline.get("edges", [])
        if isinstance(edge, dict) and edge.get("edge_type") == "NEXT"
    ]
    pipeline["edges"] = [*next_edges, *semantic_edges]
    edge_errors, edge_warnings = validate_edges(nodes, pipeline["edges"])
    if edge_errors:
        raise ValueError(
            f"Las aristas de {trace_id!r} incumplen el contrato: "
            + " | ".join(edge_errors)
        )
    checkpoint_summary = {
        "enabled": checkpoint.path is not None,
        "load_status": checkpoint.load_status,
        "reused_source_ids": checkpoint.reused_item_ids,
    }
    transport_summary = {
        "payload": payload,
        "candidate_policy": candidate_policy,
        "candidate_window": candidate_window,
        "estimator_variant": variant,
        "sources_per_request": sources_per_request,
        "batch_offset": batch_offset,
        "max_options_per_request": max_options_per_request,
        "request_batch_count": len(request_batches),
    }
    append_audit_event(
        pipeline,
        "semantic_edges",
        {
            "decision": "accepted",
            **transport_summary,
            "request_batches": request_batches,
            "excluded_by_window_by_source": excluded_by_window_by_source,
            "transport": transport,
            "semantic_edge_count": len(semantic_edges),
            "relations_by_source": per_node_counts,
            "prompt_hashes_by_source": prompt_hashes,
            "attempts_by_source": attempts_by_source,
            "attempt_history_by_source": attempt_history_by_source,
            "repaired_sources": repaired_sources,
            "mixed_prompt_sources": mixed_prompt_sources,
            "cross_source_alias_errors": cross_source_alias_errors,
            "samples": samples,
            "consensus_method": consensus_method,
            "samples_by_source": samples_by_source,
            "minority_relations_by_source": minority_relations_by_source,
            "skipped_empty_inventory": skipped_sources,
            "checkpoint": checkpoint_summary,
        },
    )
    return set_validation_stage(
        pipeline,
        "semantic_edges",
        {
            "status": "valid",
            "provider": client.provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "transport": transport,
            **transport_summary,
            "prompt_hashes_by_source": prompt_hashes,
            "attempts_by_source": attempts_by_source,
            "semantic_edge_count": len(semantic_edges),
            "samples": samples,
            "consensus_method": consensus_method,
            "skipped_empty_inventory_count": len(skipped_sources),
            "checkpoint": checkpoint_summary,
            "warnings": edge_warnings,
        },
    )

def main() -> int:
    configure_stdio()
    args = parse_args()
    try:
        if args.resume_checkpoint and args.checkpoint_dir is None:
            raise ValueError(
                "--resume-checkpoint requiere --checkpoint-dir."
            )
        if args.prev_window is not None or args.future_window is not None:
            raise ValueError(
                "--prev-window y --future-window fueron retirados; use "
                "--candidate-window (variante no canónica) o ninguna ventana."
            )
        records = select_records(
            load_records(args.input_path),
            args.trace_id,
            args.index,
            args.limit,
        )
        append_mode = (
            args.resume
            and args.output is not None
            and args.output.suffix.lower() != ".json"
        )
        if append_mode:
            existing = load_existing_trace_ids(args.output)
            records = [
                record
                for record in records
                if record.get("trace_id") not in existing
            ]
            if not records:
                log("No hay registros nuevos para procesar.", args.verbose)
                return 0
        client = build_llm_client(
            args.provider,
            args.api_key,
            args.base_url,
            args.request_timeout_seconds,
            args.transport_max_retries,
        )
        results = [
            build_semantic_edges(
                client,
                args.model,
                record,
                args.prev_window,
                args.future_window,
                args.reasoning_effort,
                args.verbose,
                args.max_attempts,
                args.checkpoint_dir,
                args.resume_checkpoint,
                samples=args.samples,
                sources_per_request=args.sources_per_request,
                candidate_window=args.candidate_window,
                batch_offset=args.batch_offset,
                max_options_per_request=args.max_options_per_request,
                payload=args.payload,
            )
            for record in records
        ]
        if append_mode:
            assert args.output is not None
            append_jsonl_output(results, args.output)
        else:
            write_output(results, args.output, args.pretty)
        return 0
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
