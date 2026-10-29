"""Contrato canónico del objetivo 3: episodios, relaciones, rutas y perfiles.

Las decisiones semánticas llegan a este módulo como salidas estructuradas de un
estimador automático (o, por compatibilidad, como una importación externa). El
código no descubre funciones por palabras clave ni completa decisiones
semánticas ausentes: normaliza, valida, asigna identificadores estables y deriva
únicamente objetos definidos de forma determinista.
"""

from __future__ import annotations

from copy import deepcopy

import copy
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path
from typing import Any, Iterable

from util.consensus import validate_consensus_record
from util.sgotgraph_schema import (
    SCHEMA_VERSION as OBJECTIVE2_SCHEMA_VERSION,
    validate_record as validate_objective2_record,
)


SCHEMA_VERSION = "sgotgraph-objective3-1.1"
PIPELINE_VERSION = "objective3-1.1"
PROFILE_VERSION = "functional-profile-1.0"

EPISODE_TYPES = {
    "PROGRESSION",
    "VERIFICATION",
    "CORRECTION",
    "SUPPORT",
    "ALTERNATIVE",
}

OUTCOMES_BY_TYPE: dict[str, set[str]] = {
    "PROGRESSION": {
        "ADVANCES",
        "REACHES_OUTPUT",
        "INTERRUPTED",
        "ABANDONED",
        "UNRESOLVED",
    },
    "VERIFICATION": {"CONFIRMED", "REFUTED", "INCONCLUSIVE"},
    "CORRECTION": {"RECOVERED", "PARTIAL", "FAILED", "UNRESOLVED"},
    "SUPPORT": {
        "ATTACHED_ACTIVE",
        "ATTACHED_SUPERSEDED",
        "CHALLENGED",
        "UNRESOLVED",
    },
    "ALTERNATIVE": {"ADOPTED", "REJECTED", "MERGED", "UNRESOLVED"},
}

ANCHOR_ROLES = {
    "TARGET",
    "CONTRAST",
    "SUPPORTED",
    "ENTRY",
    "EXIT",
    "STATE",
    "OTHER",
}
REQUIRED_ANCHOR_ROLE = {
    "VERIFICATION": "TARGET",
    "CORRECTION": "TARGET",
    "SUPPORT": "SUPPORTED",
    "ALTERNATIVE": "CONTRAST",
}
AMBIGUITY_CODES = {
    "AMBIGUOUS_TYPE",
    "BOUNDARY_UNCERTAIN",
    "ANCHOR_UNCERTAIN",
    "OUTCOME_UNCERTAIN",
    "MIXED_FUNCTION",
    "INSUFFICIENT_TEXT",
    "SEGMENTATION_CONSTRAINT",
}

TEMPORAL_RELATION_TYPES = {"IMMEDIATELY_PRECEDES_ONSET"}
FUNCTIONAL_RELATION_TYPES = {
    "CONTINUES",
    "INTERRUPTS",
    "RESUMES",
    "EMBEDS",
    "EVALUATES",
    "REPLACES",
    "SUPPORTS_EPISODE",
    "COMPETES_WITH",
}
EPISODE_RELATION_TYPES = TEMPORAL_RELATION_TYPES | FUNCTIONAL_RELATION_TYPES

ROUTE_CRITERIA = {
    "PROGRESSION": "progression_trajectory",
    "VERIFICATION": "verification_effect",
    "CORRECTION": "correction_recovery",
    "SUPPORT": "support_attachment",
    "ALTERNATIVE": "alternative_resolution",
}
OPEN_OUTCOMES = {"UNRESOLVED", "INCONCLUSIVE"}

PROVENANCE_ORIGINS = {
    "deterministic",
    "llm_inferred",
    "imported",
    # Se aceptan únicamente para importar artefactos históricos.
    "llm_proposed",
    "human_independent",
    "human_adjudicated",
    "rule_filtered",
}
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
    "sgot_graph",
    "episodes",
    "episode_relations",
    "progression_lines",
    "routes",
    "profile",
    "configurations",
    "validation",
    "provenance",
    "audit",
}
SPAN_FIELDS = {"start_char", "end_char", "text"}
ANCHOR_FIELDS = {*SPAN_FIELDS, "anchor_role"}
EVIDENCE_FIELDS = {"text_spans", "witness_node_ids", "witness_edge_ids"}
EPISODE_REQUIRED_FIELDS = {
    "episode_id",
    "episode_type",
    "members",
    "anchors",
    "entry_anchor",
    "exit_anchor",
    "outcome",
    "ambiguity_codes",
    "evidence",
    "provenance",
}
# ``consensus`` (opcional desde 1.1) registra el voto multimuestra que
# produjo el episodio; nunca sustituye a ``provenance``.
EPISODE_FIELDS = {*EPISODE_REQUIRED_FIELDS, "consensus"}
RELATION_FIELDS = {
    "relation_id",
    "relation_type",
    "source_episode_id",
    "target_episode_id",
    "evidence",
    "provenance",
}
PROGRESSION_LINE_FIELDS = {
    "line_id",
    "episode_ids",
    "relation_ids",
    "provenance",
}
ROUTE_EVIDENCE_FIELDS = {
    "source_episode_ids",
    "source_relation_ids",
    "entry_boundary_episode_ids",
    "exit_boundary_episode_ids",
}
ROUTE_FIELDS = {
    "route_id",
    "route_type",
    "focal_criterion",
    "focus_episode_ids",
    "episode_ids",
    "relation_ids",
    "onset_order",
    "entry_boundary_episode_ids",
    "exit_boundary_episode_ids",
    "entry_anchor",
    "exit_anchor",
    "route_status",
    "evidence",
    "provenance",
}
CONFIGURATION_FIELDS = {
    "route_id",
    "route_type",
    "canonical_form",
    "configuration_signature",
    "provenance",
}
RECORD_PROVENANCE_FIELDS = {
    "schema_version",
    "pipeline_version",
    "runtime",
    "stages",
}
RUNTIME_FIELDS = {
    "python_version",
    "python_implementation",
    "platform",
    "code_commit",
    "source_hash",
}
IMPLEMENTATION_FILES = (
    "01_segmentacion_gcot.py",
    "01_1_refinamiento_segmentacion_gcot.py",
    "02_tipado_gcot.py",
    "03_validacion_tipado_gcot.py",
    "04_backbone_next_gcot.py",
    "05_aristas_semanticas_gcot.py",
    "06_episodios_funcionales_gcot.py",
    "07_relaciones_episodios_gcot.py",
    "08_rutas_perfiles_gcot.py",
    "09_validacion_objetivo3_gcot.py",
    "pyproject.toml",
    "uv.lock",
    "schemas/sgotgraph_objective2.schema.json",
    "schemas/sgotgraph_objective3.schema.json",
    "util/consensus.py",
    "util/edge_batching.py",
    "util/llm_checkpoint.py",
    "util/llm_provider.py",
    "util/objective3_consensus.py",
    "util/objective3_hierarchical.py",
    "util/objective3_llm.py",
    "util/objective3_semantic_audit.py",
    "util/objective3_stability.py",
    "util/compare_objective3_runs.py",
    "util/objective3_io.py",
    "util/pipeline_canonical.py",
    "util/run_pipeline_completo_gcot.py",
    "util/run_pipeline_gcot.py",
    "util/run_pipeline_objetivo3.py",
    "util/sgotgraph_objective3.py",
    "util/sgotgraph_schema.py",
    "util/stage_cache.py",
    "util/trace_alias.py",
    "util/validar_sgotgraph_objetivo2.py",
    "util/validar_sgotgraph_objetivo3.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, payload: Any, length: int | None = None) -> str:
    digest = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if length is not None:
        digest = digest[:length]
    return f"{prefix}_{digest}"


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_token_character(value: str) -> bool:
    return value.isalnum() or value == "_"


def nearest_token_aligned_interval(
    trace_text: str, start: int, end: int
) -> tuple[int, int]:
    """Intervalo más próximo que no corta tokens, expandiendo hacia fuera.

    Es una pista mecánica para el mensaje de reparación: describe la regla de
    frontera, no decide qué contenido pertenece a un episodio.
    """

    aligned_start = start
    while (
        aligned_start > 0
        and _is_token_character(trace_text[aligned_start - 1])
        and _is_token_character(trace_text[aligned_start])
    ):
        aligned_start -= 1
    aligned_end = end
    while (
        aligned_end < len(trace_text)
        and _is_token_character(trace_text[aligned_end - 1])
        and _is_token_character(trace_text[aligned_end])
    ):
        aligned_end += 1
    return aligned_start, aligned_end


def _cuts_token_boundary(trace_text: str, start: int, end: int) -> bool:
    cuts_left = (
        start > 0
        and _is_token_character(trace_text[start - 1])
        and _is_token_character(trace_text[start])
    )
    cuts_right = (
        end < len(trace_text)
        and _is_token_character(trace_text[end - 1])
        and _is_token_character(trace_text[end])
    )
    return cuts_left or cuts_right


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    repeated: dict[str, Any] = {}
    for value in values:
        key = canonical_json(value) if _json_serializable(value) else repr(value)
        if key in seen:
            repeated[key] = value
        seen.add(key)
    return list(repeated.values())


def _json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_provenance(
    value: Any,
    *,
    default_origin: str,
    default_stage: str,
) -> dict[str, Any]:
    result = (
        {key: item for key, item in value.items() if key in PROVENANCE_FIELDS}
        if isinstance(value, dict)
        else {}
    )
    result.setdefault("origin", default_origin)
    result.setdefault("stage", default_stage)
    return result


@lru_cache(maxsize=1)
def _implementation_source_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = sha256()
    for relative_path in IMPLEMENTATION_FILES:
        encoded_path = relative_path.encode("utf-8")
        path = root / relative_path
        content = path.read_bytes() if path.is_file() else b"<missing>"
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


@lru_cache(maxsize=1)
def _cached_runtime_metadata() -> tuple[str, str, str, str | None, str]:
    commit = os.getenv("SGOTGRAPH_CODE_COMMIT")
    if not commit:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            candidate = completed.stdout.strip()
            if completed.returncode == 0 and candidate:
                commit = candidate
        except (OSError, subprocess.SubprocessError):
            commit = None
    return (
        platform.python_version(),
        platform.python_implementation(),
        platform.platform(),
        commit,
        _implementation_source_hash(),
    )


def runtime_metadata() -> dict[str, Any]:
    (
        python_version,
        python_implementation,
        platform_name,
        commit,
        source_hash,
    ) = _cached_runtime_metadata()
    return {
        "python_version": python_version,
        "python_implementation": python_implementation,
        "platform": platform_name,
        "code_commit": commit,
        "source_hash": source_hash,
    }


def normalize_span(value: Any, *, anchor: bool = False) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = {
        "start_char": source.get("start_char"),
        "end_char": source.get("end_char"),
        "text": source.get("text"),
    }
    if anchor:
        result["anchor_role"] = source.get("anchor_role")
    return result


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    start = span.get("start_char")
    end = span.get("end_char")
    return (
        start if _is_integer(start) else 10**18,
        end if _is_integer(end) else 10**18,
        str(span.get("text")),
    )


def normalize_evidence(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    raw_spans = source.get("text_spans")
    spans = (
        [normalize_span(item) for item in raw_spans]
        if isinstance(raw_spans, list)
        else []
    )
    node_ids = source.get("witness_node_ids")
    edge_ids = source.get("witness_edge_ids")
    return {
        "text_spans": sorted(spans, key=_span_key),
        "witness_node_ids": sorted(
            {item for item in node_ids if isinstance(item, str)}
        )
        if isinstance(node_ids, list)
        else [],
        "witness_edge_ids": sorted(
            {item for item in edge_ids if isinstance(item, str)}
        )
        if isinstance(edge_ids, list)
        else [],
    }


def _merge_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    spans = {
        canonical_json(span): span
        for span in [*left["text_spans"], *right["text_spans"]]
    }
    return {
        "text_spans": sorted(spans.values(), key=_span_key),
        "witness_node_ids": sorted(
            set(left["witness_node_ids"]) | set(right["witness_node_ids"])
        ),
        "witness_edge_ids": sorted(
            set(left["witness_edge_ids"]) | set(right["witness_edge_ids"])
        ),
    }


def normalize_episode(
    raw: Any,
    *,
    episode_type: str | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else {}
    raw_members = value.get("members")
    raw_anchors = value.get("anchors")
    raw_ambiguity = value.get("ambiguity_codes")
    normalized = {
        "episode_id": value.get("episode_id"),
        "episode_type": episode_type or value.get("episode_type"),
        "members": sorted(
            [normalize_span(item) for item in raw_members]
            if isinstance(raw_members, list)
            else [],
            key=_span_key,
        ),
        "anchors": sorted(
            [normalize_span(item, anchor=True) for item in raw_anchors]
            if isinstance(raw_anchors, list)
            else [],
            key=_span_key,
        ),
        "entry_anchor": normalize_span(value["entry_anchor"], anchor=True)
        if isinstance(value.get("entry_anchor"), dict)
        else None,
        "exit_anchor": normalize_span(value["exit_anchor"], anchor=True)
        if isinstance(value.get("exit_anchor"), dict)
        else None,
        "outcome": value.get("outcome"),
        "ambiguity_codes": sorted(
            {item for item in raw_ambiguity if isinstance(item, str)}
        )
        if isinstance(raw_ambiguity, list)
        else [],
        "evidence": normalize_evidence(value.get("evidence")),
        "provenance": normalize_provenance(
            value.get("provenance"),
            default_origin=str(provenance.get("origin", "imported")),
            default_stage=str(provenance.get("stage", "06_episodes")),
        ),
    }
    if isinstance(value.get("consensus"), dict):
        normalized["consensus"] = deepcopy(value["consensus"])
    return normalized


def episode_envelope(episode: dict[str, Any]) -> tuple[int, int]:
    intervals = _member_intervals(episode)
    if not intervals:
        return (10**18, 10**18)
    return (
        min(start for start, _ in intervals),
        max(end for _, end in intervals),
    )


def _episode_identity_payload(trace_id: str, episode: dict[str, Any]) -> list[Any]:
    return [
        trace_id,
        episode.get("episode_type"),
        episode.get("members"),
        episode.get("anchors"),
    ]


def _canonicalize_episodes(
    trace_id: str,
    raw_episodes: list[Any],
    default_provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups: dict[str, dict[str, Any]] = {}
    source_id_map: dict[str, str] = {}

    for raw in raw_episodes:
        raw_type = raw.get("episode_type") if isinstance(raw, dict) else None
        item = normalize_episode(
            raw,
            episode_type=raw_type if isinstance(raw_type, str) else None,
            provenance=default_provenance,
        )
        identity = canonical_json(
            [
                item["episode_type"],
                item["members"],
                item["anchors"],
            ]
        )
        canonical_id = stable_id("ep", _episode_identity_payload(trace_id, item))
        old_id = raw.get("episode_id") if isinstance(raw, dict) else None
        if isinstance(old_id, str) and old_id:
            source_id_map[old_id] = canonical_id
        item["episode_id"] = canonical_id

        previous = groups.get(identity)
        if previous is None:
            groups[identity] = item
            continue
        if previous["outcome"] != item["outcome"]:
            raise ValueError(
                "Dos candidatos con la misma identidad estructural proponen "
                f"desenlaces incompatibles: {canonical_id}."
            )
        for field in ("entry_anchor", "exit_anchor"):
            if (
                previous[field] is not None
                and item[field] is not None
                and previous[field] != item[field]
            ):
                raise ValueError(
                    f"Dos candidatos equivalentes proponen {field} incompatibles: "
                    f"{canonical_id}."
                )
            if previous[field] is None:
                previous[field] = item[field]
        previous["ambiguity_codes"] = sorted(
            set(previous["ambiguity_codes"]) | set(item["ambiguity_codes"])
        )
        previous["evidence"] = _merge_evidence(
            previous["evidence"], item["evidence"]
        )
        if "consensus" not in previous and "consensus" in item:
            previous["consensus"] = item["consensus"]

    episodes = sorted(
        groups.values(),
        key=lambda item: (
            *episode_envelope(item),
            str(item.get("episode_type")),
            str(item.get("episode_id")),
        ),
    )
    return episodes, source_id_map


def _objective3_record(
    sgot_graph: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    provenance_events: list[dict[str, Any]],
    audit_event: dict[str, Any],
) -> dict[str, Any]:
    trace_id = sgot_graph.get("trace_id")
    record = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "trace_id": trace_id,
        "sgot_graph": copy.deepcopy(sgot_graph),
        "episodes": episodes,
        "episode_relations": [],
        "progression_lines": [],
        "routes": [],
        "profile": None,
        "configurations": [],
        "validation": {},
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "runtime": runtime_metadata(),
            "stages": copy.deepcopy(provenance_events),
        },
        "audit": [copy.deepcopy(audit_event)],
    }
    report = validate_record(record, stage="episodes")
    record["validation"] = {"status": report["status"], "episodes": report}
    return record


def build_objective3_record_from_candidates(
    sgot_graph: dict[str, Any],
    raw_episodes: list[Any],
    *,
    default_provenance: dict[str, Any],
    provenance_events: list[dict[str, Any]] | None = None,
    action: str = "episodes_inferred",
) -> dict[str, Any]:
    trace_id = sgot_graph.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("SGoTGraph no contiene trace_id.")
    if sgot_graph.get("schema_version") != OBJECTIVE2_SCHEMA_VERSION:
        raise ValueError(
            f"SGoTGraph debe usar schema_version={OBJECTIVE2_SCHEMA_VERSION!r}."
        )
    objective2_validation = validate_objective2_record(sgot_graph)
    if objective2_validation["status"] == "invalid":
        details = "; ".join(objective2_validation["hard_errors"][:3])
        raise ValueError(f"SGoTGraph de entrada es inválido: {details}")
    if not isinstance(raw_episodes, list):
        raise ValueError("raw_episodes debe ser una lista.")

    episodes, source_id_map = _canonicalize_episodes(
        trace_id, raw_episodes, default_provenance
    )
    audit_event = {
        "stage": "06_episodes",
        "action": action,
        "episode_count": len(episodes),
        "source_episode_id_map": source_id_map,
    }
    events = provenance_events or [default_provenance]
    return _objective3_record(
        sgot_graph,
        episodes,
        provenance_events=events,
        audit_event=audit_event,
    )


def _annotation_provenance(annotation: dict[str, Any], stage: str) -> dict[str, Any]:
    raw = annotation.get("annotation_provenance")
    origin = annotation.get("annotation_origin")
    return normalize_provenance(
        raw,
        default_origin=origin if isinstance(origin, str) else "imported",
        default_stage=stage,
    )


def build_objective3_record(
    base_graph: dict[str, Any],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    """Importa episodios externos usando el contrato 1.0.

    Esta función se conserva para reproducir anotaciones históricas; el flujo
    automático normal utiliza ``build_objective3_record_from_candidates``.
    """

    trace_id = base_graph.get("trace_id")
    annotation_trace_id = annotation.get("trace_id") or annotation.get("id")
    if annotation_trace_id != trace_id:
        raise ValueError(
            f"La anotación {annotation_trace_id!r} no corresponde al grafo "
            f"{trace_id!r}."
        )
    raw_episodes = annotation.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("La anotación debe contener episodes como lista.")
    provenance = _annotation_provenance(annotation, "06_episodes_import")
    return build_objective3_record_from_candidates(
        base_graph,
        raw_episodes,
        default_provenance=provenance,
        provenance_events=[provenance],
        action="episodes_imported",
    )


def normalize_episode_relation(
    raw: Any,
    *,
    trace_id: str,
    provenance: dict[str, Any],
    source_id_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else {}
    source = value.get("source_episode_id") or value.get("source")
    target = value.get("target_episode_id") or value.get("target")
    if source_id_map:
        source = source_id_map.get(source, source)
        target = source_id_map.get(target, target)
    relation_type = value.get("relation_type")
    relation_id = (
        stable_id("rel", [trace_id, relation_type, source, target])
        if all(isinstance(item, str) and item for item in (relation_type, source, target))
        else None
    )
    return {
        "relation_id": relation_id,
        "relation_type": relation_type,
        "source_episode_id": source,
        "target_episode_id": target,
        "evidence": normalize_evidence(value.get("evidence")),
        "provenance": normalize_provenance(
            value.get("provenance"),
            default_origin=str(provenance.get("origin", "imported")),
            default_stage=str(provenance.get("stage", "07_episode_relations")),
        ),
    }


def _canonicalize_functional_relations(
    record: dict[str, Any],
    raw_relations: list[Any],
    provenance: dict[str, Any],
    source_id_map: dict[str, str] | None,
) -> list[dict[str, Any]]:
    trace_id = record["trace_id"]
    grouped: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for raw in raw_relations:
        relation = normalize_episode_relation(
            raw,
            trace_id=trace_id,
            provenance=provenance,
            source_id_map=source_id_map,
        )
        relation_type = relation["relation_type"]
        if relation_type in TEMPORAL_RELATION_TYPES:
            raise ValueError(
                "IMMEDIATELY_PRECEDES_ONSET es una derivación determinista y no "
                "puede importarse ni inferirse."
            )
        key = (
            relation_type,
            relation["source_episode_id"],
            relation["target_episode_id"],
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = relation
        else:
            previous["evidence"] = _merge_evidence(
                previous["evidence"], relation["evidence"]
            )
    return sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("relation_type")),
            str(item.get("source_episode_id")),
            str(item.get("target_episode_id")),
        ),
    )


def _deterministic_temporal_relations(
    episodes: list[dict[str, Any]],
    trace_id: str,
) -> list[dict[str, Any]]:
    by_onset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        start, _ = episode_envelope(episode)
        if start != 10**18:
            by_onset[start].append(episode)
    levels = sorted(by_onset)
    provenance = {
        "origin": "deterministic",
        "stage": "07_episode_relations",
    }
    relations: list[dict[str, Any]] = []
    for left, right in zip(levels, levels[1:]):
        for source in sorted(by_onset[left], key=lambda item: item["episode_id"]):
            for target in sorted(by_onset[right], key=lambda item: item["episode_id"]):
                source_id = source["episode_id"]
                target_id = target["episode_id"]
                relation_type = "IMMEDIATELY_PRECEDES_ONSET"
                relations.append(
                    {
                        "relation_id": stable_id(
                            "rel",
                            [trace_id, relation_type, source_id, target_id],
                        ),
                        "relation_type": relation_type,
                        "source_episode_id": source_id,
                        "target_episode_id": target_id,
                        "evidence": {
                            "text_spans": [],
                            "witness_node_ids": [],
                            "witness_edge_ids": [],
                        },
                        "provenance": provenance,
                    }
                )
    return relations


def _source_episode_id_map(record: dict[str, Any]) -> dict[str, str]:
    audit = record.get("audit")
    if not isinstance(audit, list):
        return {}
    for event in audit:
        if isinstance(event, dict) and isinstance(
            event.get("source_episode_id_map"), dict
        ):
            return {
                key: value
                for key, value in event["source_episode_id_map"].items()
                if isinstance(key, str) and isinstance(value, str)
            }
    return {}


def add_episode_relations_from_candidates(
    record: dict[str, Any],
    raw_relations: list[Any],
    *,
    default_provenance: dict[str, Any],
    provenance_events: list[dict[str, Any]] | None = None,
    source_id_map: dict[str, str] | None = None,
    action: str = "functional_relations_inferred",
) -> dict[str, Any]:
    source_validation = validate_record(record, stage="episodes")
    if source_validation["status"] == "INVALID":
        details = "; ".join(
            item["message"] for item in source_validation["errors"][:3]
        )
        raise ValueError(f"No se pueden relacionar episodios inválidos: {details}")
    if not isinstance(raw_relations, list):
        raise ValueError("raw_relations debe ser una lista.")

    result = copy.deepcopy(record)
    functional = _canonicalize_functional_relations(
        result,
        raw_relations,
        default_provenance,
        source_id_map,
    )
    temporal = _deterministic_temporal_relations(
        result["episodes"], result["trace_id"]
    )
    # La precedencia se serializa por niveles de inicio; las relaciones
    # funcionales ya llegan ordenadas por tipo y extremos.
    result["episode_relations"] = [*temporal, *functional]
    result["progression_lines"] = []
    result["routes"] = []
    result["profile"] = None
    result["configurations"] = []
    result.setdefault("audit", []).append(
        {
            "stage": "07_episode_relations",
            "action": action,
            "functional_relation_count": len(functional),
            "temporal_relation_count": len(temporal),
        }
    )
    stages = result.setdefault("provenance", {}).setdefault("stages", [])
    stages.extend(copy.deepcopy(provenance_events or [default_provenance]))
    report = validate_record(result, stage="relations")
    result.setdefault("validation", {})["relations"] = report
    result["validation"]["status"] = report["status"]
    return result


def add_episode_relations(
    record: dict[str, Any],
    annotation: dict[str, Any],
    *,
    derive_temporal: bool = True,
) -> dict[str, Any]:
    if not derive_temporal:
        raise ValueError(
            "El contrato 1.0 exige derivar IMMEDIATELY_PRECEDES_ONSET."
        )
    trace_id = record.get("trace_id")
    annotation_trace_id = annotation.get("trace_id") or annotation.get("id")
    if annotation_trace_id != trace_id:
        raise ValueError(
            f"La anotación {annotation_trace_id!r} no corresponde al registro "
            f"{trace_id!r}."
        )
    raw_relations = annotation.get("episode_relations")
    if not isinstance(raw_relations, list):
        raise ValueError("La anotación debe contener episode_relations como lista.")
    provenance = _annotation_provenance(annotation, "07_relations_import")
    return add_episode_relations_from_candidates(
        record,
        raw_relations,
        default_provenance=provenance,
        provenance_events=[provenance],
        source_id_map=_source_episode_id_map(record),
        action="functional_relations_imported",
    )


def _member_intervals(episode: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (span["start_char"], span["end_char"])
        for span in episode.get("members", [])
        if isinstance(span, dict)
        and _is_integer(span.get("start_char"))
        and _is_integer(span.get("end_char"))
        and span["end_char"] > span["start_char"]
    ]


def _functional_relations(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        relation
        for relation in record.get("episode_relations", [])
        if isinstance(relation, dict)
        and relation.get("relation_type") in FUNCTIONAL_RELATION_TYPES
    ]


def _components(
    episode_ids: Iterable[str],
    relations: Iterable[dict[str, Any]],
) -> list[set[str]]:
    ordered_ids = list(dict.fromkeys(episode_ids))
    adjacency = {episode_id: set() for episode_id in ordered_ids}
    for relation in relations:
        source = relation.get("source_episode_id")
        target = relation.get("target_episode_id")
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    components: list[set[str]] = []
    seen: set[str] = set()
    for episode_id in ordered_ids:
        if episode_id in seen:
            continue
        component: set[str] = set()
        stack = [episode_id]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            seen.add(current)
            stack.extend(adjacency[current] - component)
        components.append(component)
    return components


def _episode_order(
    episode_id: str,
    episode_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int, str]:
    return (*episode_envelope(episode_by_id[episode_id]), episode_id)


def derive_progression_lines(record: dict[str, Any]) -> list[dict[str, Any]]:
    episode_by_id = {
        item["episode_id"]: item
        for item in record.get("episodes", [])
        if isinstance(item, dict) and isinstance(item.get("episode_id"), str)
    }
    progression_ids = [
        episode_id
        for episode_id, episode in episode_by_id.items()
        if episode.get("episode_type") == "PROGRESSION"
    ]
    core_relations = [
        relation
        for relation in _functional_relations(record)
        if relation.get("relation_type") in {"CONTINUES", "RESUMES"}
        and relation.get("source_episode_id") in progression_ids
        and relation.get("target_episode_id") in progression_ids
    ]
    components = _components(progression_ids, core_relations)
    components.sort(
        key=lambda component: min(
            _episode_order(episode_id, episode_by_id)
            for episode_id in component
        )
    )
    provenance = {"origin": "deterministic", "stage": "08_routes_profiles"}
    lines: list[dict[str, Any]] = []
    for component in components:
        episode_ids = sorted(
            component,
            key=lambda episode_id: _episode_order(episode_id, episode_by_id),
        )
        relation_ids = sorted(
            relation["relation_id"]
            for relation in core_relations
            if relation["source_episode_id"] in component
            and relation["target_episode_id"] in component
        )
        lines.append(
            {
                "line_id": stable_id(
                    "pline",
                    [record["trace_id"], episode_ids, relation_ids],
                ),
                "episode_ids": episode_ids,
                "relation_ids": relation_ids,
                "provenance": provenance,
            }
        )
    return lines


def _focal_components(
    record: dict[str, Any],
    route_type: str,
) -> list[set[str]]:
    episode_by_id = {
        item["episode_id"]: item
        for item in record.get("episodes", [])
        if isinstance(item, dict) and isinstance(item.get("episode_id"), str)
    }
    focal_ids = [
        episode_id
        for episode_id, episode in episode_by_id.items()
        if episode.get("episode_type") == route_type
    ]
    relations = [
        relation
        for relation in _functional_relations(record)
        if relation.get("relation_type") in {"CONTINUES", "RESUMES"}
        and relation.get("source_episode_id") in focal_ids
        and relation.get("target_episode_id") in focal_ids
    ]
    components = _components(focal_ids, relations)
    components.sort(
        key=lambda component: min(
            _episode_order(episode_id, episode_by_id)
            for episode_id in component
        )
    )
    return components


def _onset_order(
    episode_ids: list[str],
    episode_by_id: dict[str, dict[str, Any]],
) -> list[list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for episode_id in episode_ids:
        start, _ = episode_envelope(episode_by_id[episode_id])
        groups[start].append(episode_id)
    return [
        sorted(groups[start])
        for start in sorted(groups)
    ]


def _route_status(
    focus_ids: list[str],
    episode_by_id: dict[str, dict[str, Any]],
) -> str:
    is_open = [
        episode_by_id[episode_id].get("outcome") in OPEN_OUTCOMES
        for episode_id in focus_ids
    ]
    if all(is_open):
        return "OPEN"
    if not any(is_open):
        return "RESOLVED"
    return "MIXED"


def derive_routes(record: dict[str, Any]) -> list[dict[str, Any]]:
    episode_by_id = {
        item["episode_id"]: item
        for item in record.get("episodes", [])
        if isinstance(item, dict) and isinstance(item.get("episode_id"), str)
    }
    functional = _functional_relations(record)
    provenance = {"origin": "deterministic", "stage": "08_routes_profiles"}
    routes: list[dict[str, Any]] = []

    for route_type in sorted(EPISODE_TYPES):
        for core in _focal_components(record, route_type):
            neighbors: set[str] = set()
            for relation in functional:
                source = relation.get("source_episode_id")
                target = relation.get("target_episode_id")
                if source in core and target in episode_by_id and target not in core:
                    neighbors.add(target)
                if target in core and source in episode_by_id and source not in core:
                    neighbors.add(source)
            route_episode_set = set(core) | neighbors
            episode_ids = sorted(
                route_episode_set,
                key=lambda episode_id: _episode_order(
                    episode_id, episode_by_id
                ),
            )
            focus_ids = sorted(
                core,
                key=lambda episode_id: _episode_order(
                    episode_id, episode_by_id
                ),
            )
            induced_relations = [
                relation
                for relation in functional
                if relation.get("source_episode_id") in route_episode_set
                and relation.get("target_episode_id") in route_episode_set
            ]
            relation_ids = sorted(
                relation["relation_id"] for relation in induced_relations
            )
            earliest_start = min(
                episode_envelope(episode_by_id[item])[0] for item in episode_ids
            )
            latest_end = max(
                episode_envelope(episode_by_id[item])[1] for item in episode_ids
            )
            entry_ids = [
                item
                for item in episode_ids
                if episode_envelope(episode_by_id[item])[0] == earliest_start
            ]
            exit_ids = [
                item
                for item in episode_ids
                if episode_envelope(episode_by_id[item])[1] == latest_end
            ]
            entry_anchor = (
                copy.deepcopy(episode_by_id[entry_ids[0]].get("entry_anchor"))
                if len(entry_ids) == 1
                else None
            )
            exit_anchor = (
                copy.deepcopy(episode_by_id[exit_ids[0]].get("exit_anchor"))
                if len(exit_ids) == 1
                else None
            )
            criterion = ROUTE_CRITERIA[route_type]
            route_id = stable_id(
                "route",
                [
                    record["trace_id"],
                    route_type,
                    criterion,
                    focus_ids,
                    episode_ids,
                    relation_ids,
                ],
            )
            evidence = {
                "source_episode_ids": episode_ids,
                "source_relation_ids": relation_ids,
                "entry_boundary_episode_ids": entry_ids,
                "exit_boundary_episode_ids": exit_ids,
            }
            routes.append(
                {
                    "route_id": route_id,
                    "route_type": route_type,
                    "focal_criterion": criterion,
                    "focus_episode_ids": focus_ids,
                    "episode_ids": episode_ids,
                    "relation_ids": relation_ids,
                    "onset_order": _onset_order(episode_ids, episode_by_id),
                    "entry_boundary_episode_ids": entry_ids,
                    "exit_boundary_episode_ids": exit_ids,
                    "entry_anchor": entry_anchor,
                    "exit_anchor": exit_anchor,
                    "route_status": _route_status(focus_ids, episode_by_id),
                    "evidence": evidence,
                    "provenance": provenance,
                }
            )
    routes.sort(
        key=lambda route: (
            _episode_order(route["focus_episode_ids"][0], episode_by_id),
            route["route_type"],
            route["route_id"],
        )
    )
    return routes


def _route_canonical_form(
    route: dict[str, Any],
    episode_by_id: dict[str, dict[str, Any]],
    relation_by_id: dict[str, dict[str, Any]],
) -> str:
    """Canoniza preservando los niveles de inicio y permutando solo empates.

    Toda biyección que preserve el orden estricto de inicio debe mapear cada
    nivel de inicio al nivel correspondiente. Por ello basta enumerar las
    permutaciones dentro de los niveles con igual inicio, que suelen ser
    pequeños, en lugar de enumerar |F_q|! biyecciones.
    """

    onset_levels = route["onset_order"]
    level_permutations = [
        list(permutations(level)) if len(level) > 1 else [tuple(level)]
        for level in onset_levels
    ]
    focus = set(route["focus_episode_ids"])
    best: str | None = None
    for ordered_levels in product(*level_permutations):
        ordered_ids = [item for level in ordered_levels for item in level]
        index = {episode_id: position for position, episode_id in enumerate(ordered_ids)}
        labels = [
            [
                episode_by_id[episode_id]["episode_type"],
                episode_by_id[episode_id]["outcome"],
                episode_id in focus,
            ]
            for episode_id in ordered_ids
        ]
        order_matrix = [
            [
                int(
                    episode_envelope(episode_by_id[left])[0]
                    < episode_envelope(episode_by_id[right])[0]
                )
                for right in ordered_ids
            ]
            for left in ordered_ids
        ]
        relations = sorted(
            [
                index[relation["source_episode_id"]],
                relation["relation_type"],
                index[relation["target_episode_id"]],
            ]
            for relation_id in route["relation_ids"]
            for relation in [relation_by_id[relation_id]]
        )
        candidate = canonical_json(
            {
                "route_type": route["route_type"],
                "nodes": labels,
                "order": order_matrix,
                "relations": relations,
            }
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(f"La ruta {route.get('route_id')!r} no contiene episodios.")
    return best


def derive_configurations(record: dict[str, Any]) -> list[dict[str, Any]]:
    episode_by_id = {
        item["episode_id"]: item for item in record.get("episodes", [])
    }
    relation_by_id = {
        item["relation_id"]: item for item in record.get("episode_relations", [])
    }
    provenance = {"origin": "deterministic", "stage": "08_routes_profiles"}
    configurations = []
    for route in record.get("routes", []):
        canonical_form = _route_canonical_form(
            route, episode_by_id, relation_by_id
        )
        signature = (
            "sha256:"
            + sha256(canonical_form.encode("utf-8")).hexdigest()
        )
        configurations.append(
            {
                "route_id": route["route_id"],
                "route_type": route["route_type"],
                "canonical_form": canonical_form,
                "configuration_signature": signature,
                "provenance": provenance,
            }
        )
    return configurations


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _coverage(
    intervals: list[tuple[int, int]],
    trace_length: int,
) -> dict[str, Any]:
    chars = sum(end - start for start, end in _merge_intervals(intervals))
    return {
        "chars": chars,
        "ratio": round(chars / trace_length, 6) if trace_length else 0.0,
    }


def _intervals_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def derive_profile(record: dict[str, Any]) -> dict[str, Any]:
    trace_text = record.get("sgot_graph", {}).get("trace_text", "")
    trace_length = len(trace_text) if isinstance(trace_text, str) else 0
    episodes = record.get("episodes", [])
    relations = record.get("episode_relations", [])
    routes = record.get("routes", [])
    progression_lines = record.get("progression_lines", [])
    configurations = record.get("configurations", [])

    episode_counts = {item: 0 for item in sorted(EPISODE_TYPES)}
    outcome_counts = {
        episode_type: {
            outcome: 0 for outcome in sorted(OUTCOMES_BY_TYPE[episode_type])
        }
        for episode_type in sorted(EPISODE_TYPES)
    }
    intervals_by_type = {item: [] for item in EPISODE_TYPES}
    all_intervals: list[tuple[int, int]] = []
    noncontiguous = 0
    unresolved = 0
    for episode in episodes:
        episode_type = episode.get("episode_type")
        intervals = _member_intervals(episode)
        if episode_type in EPISODE_TYPES:
            episode_counts[episode_type] += 1
            intervals_by_type[episode_type].extend(intervals)
            outcome = episode.get("outcome")
            if outcome in outcome_counts[episode_type]:
                outcome_counts[episode_type][outcome] += 1
        all_intervals.extend(intervals)
        merged = _merge_intervals(intervals)
        if len(merged) > 1:
            noncontiguous += 1
        if episode.get("outcome") in OPEN_OUTCOMES:
            unresolved += 1

    overlap_pairs = 0
    nested_pairs = 0
    for index, left in enumerate(episodes):
        left_intervals = _member_intervals(left)
        left_envelope = episode_envelope(left)
        for right in episodes[index + 1 :]:
            right_intervals = _member_intervals(right)
            if any(
                _intervals_overlap(a, b)
                for a in left_intervals
                for b in right_intervals
            ):
                overlap_pairs += 1
            right_envelope = episode_envelope(right)
            strict_left = (
                left_envelope[0] <= right_envelope[0]
                and right_envelope[1] <= left_envelope[1]
                and left_envelope != right_envelope
            )
            strict_right = (
                right_envelope[0] <= left_envelope[0]
                and left_envelope[1] <= right_envelope[1]
                and left_envelope != right_envelope
            )
            if strict_left or strict_right:
                nested_pairs += 1

    relation_counts = {
        item: 0 for item in sorted(EPISODE_RELATION_TYPES)
    }
    for relation in relations:
        relation_type = relation.get("relation_type")
        if relation_type in relation_counts:
            relation_counts[relation_type] += 1

    route_counts = {item: 0 for item in sorted(EPISODE_TYPES)}
    route_sizes: list[int] = []
    context_ratio_by_route: dict[str, float] = {}
    episode_route_membership = {
        episode["episode_id"]: 0 for episode in episodes
    }
    for route in routes:
        route_type = route.get("route_type")
        if route_type in route_counts:
            route_counts[route_type] += 1
        episode_ids = route.get("episode_ids", [])
        focus_ids = route.get("focus_episode_ids", [])
        route_sizes.append(len(episode_ids))
        context_ratio_by_route[route["route_id"]] = round(
            (len(episode_ids) - len(focus_ids)) / len(episode_ids), 6
        )
        for episode_id in episode_ids:
            if episode_id in episode_route_membership:
                episode_route_membership[episode_id] += 1

    route_overlap: list[dict[str, Any]] = []
    for index, left in enumerate(routes):
        left_ids = set(left["episode_ids"])
        for right in routes[index + 1 :]:
            right_ids = set(right["episode_ids"])
            intersection = left_ids & right_ids
            union = left_ids | right_ids
            route_overlap.append(
                {
                    "left_route_id": left["route_id"],
                    "right_route_id": right["route_id"],
                    "shared_episode_count": len(intersection),
                    "jaccard": round(len(intersection) / len(union), 6)
                    if union
                    else 0.0,
                }
            )

    functional_components = _components(
        [episode["episode_id"] for episode in episodes],
        _functional_relations(record),
    )
    line_sizes = [len(line["episode_ids"]) for line in progression_lines]
    context_values = list(context_ratio_by_route.values())
    membership_values = list(episode_route_membership.values())
    overlap_values = [item["jaccard"] for item in route_overlap]
    signatures = Counter(
        item["configuration_signature"] for item in configurations
    )
    provenance = {"origin": "deterministic", "stage": "08_routes_profiles"}
    return {
        "profile_version": PROFILE_VERSION,
        "trace_length_chars": trace_length,
        "episode_count": len(episodes),
        "episode_counts_by_type": episode_counts,
        "outcome_counts_by_type": outcome_counts,
        "relation_count": len(relations),
        "functional_relation_count": sum(
            relation.get("relation_type") in FUNCTIONAL_RELATION_TYPES
            for relation in relations
        ),
        "relation_counts_by_type": relation_counts,
        "route_count": len(routes),
        "route_counts_by_type": route_counts,
        "configuration_count": len(configurations),
        "configuration_multiplicity": dict(sorted(signatures.items())),
        "coverage": {
            "overall": _coverage(all_intervals, trace_length),
            "by_type": {
                episode_type: _coverage(
                    intervals_by_type[episode_type], trace_length
                )
                for episode_type in sorted(EPISODE_TYPES)
            },
        },
        "organization": {
            "progression_line_count": len(progression_lines),
            "progression_line_sizes": line_sizes,
            "functional_component_count": len(functional_components),
            "noncontiguous_episode_count": noncontiguous,
            "overlapping_episode_pair_count": overlap_pairs,
            "nested_episode_pair_count": nested_pairs,
            "unresolved_or_inconclusive_episode_count": unresolved,
            "functional_diversity": round(
                sum(value > 0 for value in episode_counts.values())
                / len(EPISODE_TYPES),
                6,
            ),
        },
        "route_structure": {
            "multi_episode_route_count": sum(size > 1 for size in route_sizes),
            "mean_episode_count": round(
                sum(route_sizes) / len(route_sizes), 6
            )
            if route_sizes
            else 0.0,
            "max_episode_count": max(route_sizes, default=0),
            "context_ratio_by_route": dict(sorted(context_ratio_by_route.items())),
            "mean_context_ratio": round(
                sum(context_values) / len(context_values), 6
            )
            if context_values
            else 0.0,
            "episode_route_membership": dict(
                sorted(episode_route_membership.items())
            ),
            "mean_episode_route_membership": round(
                sum(membership_values) / len(membership_values), 6
            )
            if membership_values
            else 0.0,
            "overlapping_route_pair_count": sum(
                item["shared_episode_count"] > 0 for item in route_overlap
            ),
            "mean_route_jaccard": round(
                sum(overlap_values) / len(overlap_values), 6
            )
            if overlap_values
            else 0.0,
            "max_route_jaccard": max(overlap_values, default=0.0),
            "pairwise_route_overlap": route_overlap,
        },
        "source_episode_ids": sorted(
            episode["episode_id"] for episode in episodes
        ),
        "source_relation_ids": sorted(
            relation["relation_id"] for relation in relations
        ),
        "source_route_ids": sorted(route["route_id"] for route in routes),
        "provenance": provenance,
    }


def derive_routes_and_profile(record: dict[str, Any]) -> dict[str, Any]:
    source_validation = validate_record(record, stage="relations")
    if source_validation["status"] == "INVALID":
        details = "; ".join(
            item["message"] for item in source_validation["errors"][:3]
        )
        raise ValueError(
            "No se pueden derivar rutas desde episodios o relaciones inválidos: "
            + details
        )
    result = copy.deepcopy(record)
    result["progression_lines"] = derive_progression_lines(result)
    result["routes"] = derive_routes(result)
    result["configurations"] = derive_configurations(result)
    result["profile"] = derive_profile(result)
    result.setdefault("audit", []).append(
        {
            "stage": "08_routes_profiles",
            "action": "routes_profile_configurations_derived",
            "progression_line_count": len(result["progression_lines"]),
            "route_count": len(result["routes"]),
            "configuration_count": len(result["configurations"]),
        }
    )
    result.setdefault("provenance", {}).setdefault("stages", []).append(
        {"origin": "deterministic", "stage": "08_routes_profiles"}
    )
    report = validate_record(result, stage="complete")
    result.setdefault("validation", {})["complete"] = report
    result["validation"]["status"] = report["status"]
    return result


def _error(
    errors: list[dict[str, Any]],
    stage: str,
    code: str,
    message: str,
    object_id: str | None = None,
) -> None:
    errors.append(
        {
            "stage": stage,
            "code": code,
            "object_id": object_id,
            "message": message,
        }
    )


def _validate_fields(
    value: dict[str, Any],
    required: set[str],
    allowed: set[str],
    *,
    stage: str,
    object_id: str | None,
    errors: list[dict[str, Any]],
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        _error(
            errors,
            stage,
            "MISSING_FIELDS",
            f"Faltan campos obligatorios: {missing}.",
            object_id,
        )
    if unknown:
        _error(
            errors,
            stage,
            "UNKNOWN_FIELDS",
            f"Campos no permitidos: {unknown}.",
            object_id,
        )


def _validate_provenance(
    value: Any,
    *,
    stage: str,
    object_id: str | None,
    errors: list[dict[str, Any]],
) -> None:
    if not isinstance(value, dict):
        _error(
            errors,
            stage,
            "INVALID_PROVENANCE",
            "provenance debe ser un objeto.",
            object_id,
        )
        return
    _validate_fields(
        value,
        {"origin", "stage"},
        PROVENANCE_FIELDS,
        stage=stage,
        object_id=object_id,
        errors=errors,
    )
    if value.get("origin") not in PROVENANCE_ORIGINS:
        _error(
            errors,
            stage,
            "INVALID_PROVENANCE_ORIGIN",
            f"Origen de procedencia inválido: {value.get('origin')!r}.",
            object_id,
        )
    if not isinstance(value.get("stage"), str) or not value.get("stage"):
        _error(
            errors,
            stage,
            "INVALID_PROVENANCE_STAGE",
            "provenance.stage debe ser una cadena no vacía.",
            object_id,
        )
    for field in PROVENANCE_FIELDS - {"origin", "stage"}:
        if field in value and not isinstance(value[field], str):
            _error(
                errors,
                stage,
                "INVALID_PROVENANCE_FIELD",
                f"provenance.{field} debe ser una cadena.",
                object_id,
            )


def _validate_span(
    value: Any,
    trace_text: str,
    *,
    anchor: bool,
    stage: str,
    object_id: str | None,
    errors: list[dict[str, Any]],
) -> bool:
    if not isinstance(value, dict):
        _error(
            errors,
            stage,
            "INVALID_SPAN",
            "El intervalo debe ser un objeto.",
            object_id,
        )
        return False
    fields = ANCHOR_FIELDS if anchor else SPAN_FIELDS
    _validate_fields(
        value,
        fields,
        fields,
        stage=stage,
        object_id=object_id,
        errors=errors,
    )
    start = value.get("start_char")
    end = value.get("end_char")
    text = value.get("text")
    valid = True
    if (
        not _is_integer(start)
        or not _is_integer(end)
        or start < 0
        or end <= start
        or end > len(trace_text)
    ):
        _error(
            errors,
            stage,
            "SPAN_OUT_OF_RANGE",
            f"Intervalo inválido [{start!r}, {end!r}).",
            object_id,
        )
        valid = False
    if not isinstance(text, str) or not text:
        _error(
            errors,
            stage,
            "INVALID_SPAN_TEXT",
            "El texto del intervalo debe ser una cadena no vacía.",
            object_id,
        )
        valid = False
    if valid and trace_text[start:end] != text:
        _error(
            errors,
            stage,
            "SPAN_TEXT_MISMATCH",
            "El texto no coincide exactamente con trace_text[start_char:end_char].",
            object_id,
        )
        valid = False
    if valid and _cuts_token_boundary(trace_text, start, end):
        aligned_start, aligned_end = nearest_token_aligned_interval(
            trace_text, start, end
        )
        _error(
            errors,
            stage,
            "SPAN_BOUNDARY_INSIDE_TOKEN",
            (
                f"El intervalo [{start}, {end}) = {trace_text[start:end]!r} "
                "corta un token alfanumérico en uno de sus límites. El "
                "intervalo alineado más próximo es "
                f"[{aligned_start}, {aligned_end}) = "
                f"{trace_text[aligned_start:aligned_end]!r}; verifique los "
                "offsets 0-based y semiabiertos."
            ),
            object_id,
        )
        valid = False
    if anchor and value.get("anchor_role") not in ANCHOR_ROLES:
        _error(
            errors,
            stage,
            "INVALID_ANCHOR_ROLE",
            f"anchor_role inválido: {value.get('anchor_role')!r}.",
            object_id,
        )
        valid = False
    return valid


def _node_projection(
    episode: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> set[str]:
    result: set[str] = set()
    intervals = _member_intervals(episode)
    for node in nodes:
        node_id = node.get("segment_id")
        start = node.get("start_char")
        end = node.get("end_char")
        if (
            isinstance(node_id, str)
            and _is_integer(start)
            and _is_integer(end)
            and any(max(start, left) < min(end, right) for left, right in intervals)
        ):
            result.add(node_id)
    return result


def _validate_evidence(
    value: Any,
    trace_text: str,
    *,
    allowed_node_ids: set[str],
    allowed_edge_ids: set[str],
    require_witness: bool,
    stage: str,
    object_id: str | None,
    errors: list[dict[str, Any]],
) -> None:
    if not isinstance(value, dict):
        _error(
            errors,
            stage,
            "INVALID_EVIDENCE",
            "evidence debe ser un objeto.",
            object_id,
        )
        return
    _validate_fields(
        value,
        EVIDENCE_FIELDS,
        EVIDENCE_FIELDS,
        stage=stage,
        object_id=object_id,
        errors=errors,
    )
    spans = value.get("text_spans")
    node_ids = value.get("witness_node_ids")
    edge_ids = value.get("witness_edge_ids")
    if not isinstance(spans, list):
        _error(
            errors,
            stage,
            "INVALID_EVIDENCE_SPANS",
            "evidence.text_spans debe ser una lista.",
            object_id,
        )
        spans = []
    for span in spans:
        _validate_span(
            span,
            trace_text,
            anchor=False,
            stage=stage,
            object_id=object_id,
            errors=errors,
        )
    if not isinstance(node_ids, list) or not all(
        isinstance(item, str) for item in node_ids
    ):
        _error(
            errors,
            stage,
            "INVALID_WITNESS_NODES",
            "witness_node_ids debe contener únicamente cadenas.",
            object_id,
        )
        node_ids = []
    if not isinstance(edge_ids, list) or not all(
        isinstance(item, str) for item in edge_ids
    ):
        _error(
            errors,
            stage,
            "INVALID_WITNESS_EDGES",
            "witness_edge_ids debe contener únicamente cadenas.",
            object_id,
        )
        edge_ids = []
    unknown_nodes = sorted(set(node_ids) - allowed_node_ids)
    unknown_edges = sorted(set(edge_ids) - allowed_edge_ids)
    if unknown_nodes:
        _error(
            errors,
            stage,
            "WITNESS_NODE_OUTSIDE_PROJECTION",
            f"Nodos testigo fuera de la proyección permitida: {unknown_nodes}.",
            object_id,
        )
    if unknown_edges:
        _error(
            errors,
            stage,
            "WITNESS_EDGE_OUTSIDE_PROJECTION",
            f"Aristas testigo fuera de la proyección permitida: {unknown_edges}.",
            object_id,
        )
    if require_witness and not spans and not node_ids and not edge_ids:
        _error(
            errors,
            stage,
            "MISSING_WITNESS",
            "La decisión semántica requiere al menos un testigo verificable.",
            object_id,
        )


def _validate_episodes(
    record: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stage = "episodes"
    trace_text = record.get("sgot_graph", {}).get("trace_text", "")
    nodes = record.get("sgot_graph", {}).get("nodes", [])
    edges = record.get("sgot_graph", {}).get("edges", [])
    edge_by_id = {
        edge.get("edge_id"): edge
        for edge in edges
        if isinstance(edge, dict) and isinstance(edge.get("edge_id"), str)
    }
    episodes = record.get("episodes")
    if not isinstance(episodes, list):
        _error(errors, stage, "INVALID_EPISODES", "episodes debe ser una lista.")
        return {}
    episode_by_id: dict[str, dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        object_id = (
            episode.get("episode_id")
            if isinstance(episode, dict)
            else f"episode[{index}]"
        )
        if not isinstance(episode, dict):
            _error(
                errors,
                stage,
                "INVALID_EPISODE",
                "Cada episodio debe ser un objeto.",
                str(object_id),
            )
            continue
        _validate_fields(
            episode,
            EPISODE_REQUIRED_FIELDS,
            EPISODE_FIELDS,
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        if "consensus" in episode:
            for message in validate_consensus_record(
                episode["consensus"], "episode"
            ):
                _error(errors, stage, "INVALID_CONSENSUS", message, str(object_id))
        episode_id = episode.get("episode_id")
        episode_type = episode.get("episode_type")
        if not isinstance(episode_id, str) or not episode_id:
            _error(
                errors,
                stage,
                "INVALID_EPISODE_ID",
                "episode_id debe ser una cadena no vacía.",
                str(object_id),
            )
        elif episode_id in episode_by_id:
            _error(
                errors,
                stage,
                "DUPLICATE_EPISODE_ID",
                "episode_id duplicado.",
                episode_id,
            )
        else:
            episode_by_id[episode_id] = episode
        if episode_type not in EPISODE_TYPES:
            _error(
                errors,
                stage,
                "INVALID_EPISODE_TYPE",
                f"episode_type inválido: {episode_type!r}.",
                str(object_id),
            )
        members = episode.get("members")
        if not isinstance(members, list) or not members:
            _error(
                errors,
                stage,
                "EMPTY_EPISODE_MEMBERS",
                "members debe ser una lista no vacía.",
                str(object_id),
            )
            members = []
        valid_intervals: list[tuple[int, int]] = []
        for span in members:
            if _validate_span(
                span,
                trace_text,
                anchor=False,
                stage=stage,
                object_id=str(object_id),
                errors=errors,
            ):
                valid_intervals.append(
                    (span["start_char"], span["end_char"])
                )
        if members != sorted(members, key=_span_key):
            _error(
                errors,
                stage,
                "UNSORTED_MEMBERS",
                "Los miembros deben estar ordenados por posición.",
                str(object_id),
            )
        if any(
            right[0] < left[1]
            for left, right in zip(valid_intervals, valid_intervals[1:])
        ):
            _error(
                errors,
                stage,
                "OVERLAPPING_EPISODE_MEMBERS",
                "Los miembros de un episodio no pueden solaparse entre sí.",
                str(object_id),
            )
        anchors = episode.get("anchors")
        if not isinstance(anchors, list):
            _error(
                errors,
                stage,
                "INVALID_ANCHORS",
                "anchors debe ser una lista.",
                str(object_id),
            )
            anchors = []
        for anchor in anchors:
            _validate_span(
                anchor,
                trace_text,
                anchor=True,
                stage=stage,
                object_id=str(object_id),
                errors=errors,
            )
        required_role = REQUIRED_ANCHOR_ROLE.get(episode_type)
        if required_role and not any(
            isinstance(anchor, dict)
            and anchor.get("anchor_role") == required_role
            for anchor in anchors
        ):
            _error(
                errors,
                stage,
                "MISSING_SPECIALIZED_ANCHOR",
                f"{episode_type} requiere al menos un ancla {required_role}.",
                str(object_id),
            )
        required_anchor_intervals = [
            (anchor["start_char"], anchor["end_char"])
            for anchor in anchors
            if isinstance(anchor, dict)
            and anchor.get("anchor_role") == required_role
            and _is_integer(anchor.get("start_char"))
            and _is_integer(anchor.get("end_char"))
            and anchor["end_char"] > anchor["start_char"]
        ]
        if valid_intervals and required_anchor_intervals:
            merged_anchors = _merge_intervals(required_anchor_intervals)
            member_chars = sum(end - start for start, end in valid_intervals)
            anchored_member_chars = sum(
                max(
                    0,
                    min(member_end, anchor_end)
                    - max(member_start, anchor_start),
                )
                for member_start, member_end in valid_intervals
                for anchor_start, anchor_end in merged_anchors
            )
            if anchored_member_chars >= member_chars:
                _error(
                    errors,
                    stage,
                    "OPERATION_COINCIDENT_WITH_SPECIALIZED_ANCHOR",
                    (
                        f"Los miembros de {episode_type} no pueden quedar "
                        f"totalmente cubiertos por sus anclas {required_role}; "
                        "debe existir contenido observable que realice la "
                        "operación además del contenido sobre el que actúa."
                    ),
                    str(object_id),
                )
        if (
            episode_type == "CORRECTION"
            and valid_intervals
            and required_anchor_intervals
            and not any(
                member_end > target_end
                for _, member_end in valid_intervals
                for _, target_end in required_anchor_intervals
            )
        ):
            _error(
                errors,
                stage,
                "CORRECTION_TARGET_NOT_PRIOR_TO_OPERATION",
                (
                    "Al menos un ancla TARGET de CORRECTION debe preceder "
                    "contenido miembro que materialice la revisión."
                ),
                str(object_id),
            )
        for field, role in (("entry_anchor", "ENTRY"), ("exit_anchor", "EXIT")):
            anchor = episode.get(field)
            if anchor is not None:
                _validate_span(
                    anchor,
                    trace_text,
                    anchor=True,
                    stage=stage,
                    object_id=str(object_id),
                    errors=errors,
                )
                if isinstance(anchor, dict) and anchor.get("anchor_role") != role:
                    _error(
                        errors,
                        stage,
                        "INVALID_BOUNDARY_ANCHOR_ROLE",
                        f"{field} debe usar anchor_role={role}.",
                        str(object_id),
                    )
        if (
            episode_type in OUTCOMES_BY_TYPE
            and episode.get("outcome") not in OUTCOMES_BY_TYPE[episode_type]
        ):
            _error(
                errors,
                stage,
                "INVALID_OUTCOME",
                f"Desenlace incompatible con {episode_type}: "
                f"{episode.get('outcome')!r}.",
                str(object_id),
            )
        ambiguity = episode.get("ambiguity_codes")
        if not isinstance(ambiguity, list) or not all(
            isinstance(item, str) and item in AMBIGUITY_CODES
            for item in ambiguity
        ):
            _error(
                errors,
                stage,
                "INVALID_AMBIGUITY_CODES",
                "ambiguity_codes contiene valores no admitidos.",
                str(object_id),
            )
        projection = _node_projection(episode, nodes)
        if members and not projection:
            _error(
                errors,
                stage,
                "EMPTY_NODE_PROJECTION",
                "El episodio no intersecta ningún nodo de SGoTGraph.",
                str(object_id),
            )
        projected_edges = {
            edge_id
            for edge_id, edge in edge_by_id.items()
            if edge.get("edge_type") != "NEXT"
            and (
                edge.get("source") in projection
                or edge.get("target") in projection
            )
        }
        _validate_evidence(
            episode.get("evidence"),
            trace_text,
            allowed_node_ids=projection,
            allowed_edge_ids=projected_edges,
            require_witness=True,
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        _validate_provenance(
            episode.get("provenance"),
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        if isinstance(episode_id, str):
            expected_id = stable_id(
                "ep",
                _episode_identity_payload(record.get("trace_id", ""), episode),
            )
            if episode_id != expected_id:
                _error(
                    errors,
                    stage,
                    "NONCANONICAL_EPISODE_ID",
                    "episode_id no coincide con su identidad estructural.",
                    episode_id,
                )
    ordered = sorted(
        episodes,
        key=lambda item: (
            *episode_envelope(item),
            str(item.get("episode_type")),
            str(item.get("episode_id")),
        )
        if isinstance(item, dict)
        else (10**18, 10**18, "", ""),
    )
    if episodes != ordered:
        _error(
            errors,
            stage,
            "UNSORTED_EPISODES",
            "episodes no preserva el orden canónico.",
        )
    valid_episode_values = [
        episode
        for episode in episodes
        if isinstance(episode, dict)
        and episode.get("episode_type") in EPISODE_TYPES
    ]
    for index, left in enumerate(valid_episode_values):
        for right in valid_episode_values[index + 1 :]:
            if left.get("episode_type") == right.get("episode_type"):
                continue
            if not any(
                _intervals_overlap(left_interval, right_interval)
                for left_interval in _member_intervals(left)
                for right_interval in _member_intervals(right)
            ):
                continue
            missing_marker = [
                episode.get("episode_id")
                for episode in (left, right)
                if "MIXED_FUNCTION"
                not in (
                    episode.get("ambiguity_codes")
                    if isinstance(episode.get("ambiguity_codes"), list)
                    else []
                )
            ]
            if missing_marker:
                _error(
                    errors,
                    stage,
                    "UNMARKED_CROSS_TYPE_MEMBER_OVERLAP",
                    (
                        "Todo solapamiento de miembros entre tipos distintos "
                        "debe conservar MIXED_FUNCTION en ambos episodios; "
                        f"falta en {missing_marker}."
                    ),
                    (
                        f"{left.get('episode_id')}|"
                        f"{right.get('episode_id')}"
                    ),
                )
    return episode_by_id


def _anchor_linked(
    source_episode: dict[str, Any],
    target_episode: dict[str, Any],
    role: str,
) -> bool:
    target_intervals = _member_intervals(target_episode)
    target_anchor_keys = {
        (
            anchor.get("start_char"),
            anchor.get("end_char"),
        )
        for anchor in target_episode.get("anchors", [])
        if isinstance(anchor, dict)
    }
    for anchor in source_episode.get("anchors", []):
        if not isinstance(anchor, dict) or anchor.get("anchor_role") != role:
            continue
        interval = (anchor.get("start_char"), anchor.get("end_char"))
        if not all(_is_integer(value) for value in interval):
            continue
        if interval in target_anchor_keys or any(
            _intervals_overlap(interval, member) for member in target_intervals
        ):
            return True
    return False


def _anchor_intervals(
    episode: dict[str, Any],
    roles: set[str],
) -> list[tuple[int, int]]:
    values = [
        *episode.get("anchors", []),
        episode.get("entry_anchor"),
        episode.get("exit_anchor"),
    ]
    intervals: list[tuple[int, int]] = []
    for value in values:
        if not isinstance(value, dict) or value.get("anchor_role") not in roles:
            continue
        start = value.get("start_char")
        end = value.get("end_char")
        if _is_integer(start) and _is_integer(end):
            intervals.append((start, end))
    return intervals


def _anchors_correspond(
    source: dict[str, Any],
    target: dict[str, Any],
    roles: set[str],
) -> bool:
    return any(
        _intervals_overlap(left, right)
        for left in _anchor_intervals(source, roles)
        for right in _anchor_intervals(target, roles)
    )


def _has_leads_to_path(
    record: dict[str, Any],
    earlier: dict[str, Any],
    later: dict[str, Any],
) -> bool:
    graph = record.get("sgot_graph", {})
    nodes = graph.get("nodes", [])
    earlier_nodes = _node_projection(earlier, nodes)
    later_nodes = _node_projection(later, nodes)
    if not earlier_nodes or not later_nodes:
        return False

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("edge_type") != "LEADS_TO":
            continue
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, str) and isinstance(target, str):
            adjacency[source].add(target)

    pending = list(earlier_nodes)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        if current in later_nodes:
            return True
        seen.add(current)
        pending.extend(adjacency.get(current, set()) - seen)
    return False


def _continuity_referent_linked(
    record: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    episode_type = source.get("episode_type")
    if episode_type == "PROGRESSION":
        return _anchors_correspond(
            source,
            target,
            {"STATE", "ENTRY", "EXIT"},
        ) or _has_leads_to_path(record, target, source)

    required_role = REQUIRED_ANCHOR_ROLE.get(str(episode_type))
    return bool(
        required_role
        and _anchors_correspond(source, target, {required_role})
    )


def _has_intervening_functional_change(
    record: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    source_start, _ = episode_envelope(source)
    target_start, _ = episode_envelope(target)
    source_id = source.get("episode_id")
    target_id = target.get("episode_id")
    source_type = source.get("episode_type")
    episodes = [
        item
        for item in record.get("episodes", [])
        if isinstance(item, dict)
    ]

    for episode in episodes:
        if episode.get("episode_id") in {source_id, target_id}:
            continue
        onset, _ = episode_envelope(episode)
        if (
            target_start < onset < source_start
            and episode.get("episode_type") != source_type
        ):
            return True

    episode_by_id = {
        item.get("episode_id"): item
        for item in episodes
        if isinstance(item.get("episode_id"), str)
    }
    for relation in record.get("episode_relations", []):
        if (
            isinstance(relation, dict)
            and relation.get("relation_type") == "INTERRUPTS"
            and relation.get("target_episode_id") == target_id
        ):
            interrupter = episode_by_id.get(relation.get("source_episode_id"))
            if interrupter is not None:
                onset, _ = episode_envelope(interrupter)
                if target_start < onset < source_start:
                    return True
    return False


def _relation_compatibility_error(
    record: dict[str, Any],
    relation: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> str | None:
    relation_type = relation.get("relation_type")
    source_type = source.get("episode_type")
    target_type = target.get("episode_type")
    source_start, source_end = episode_envelope(source)
    target_start, target_end = episode_envelope(target)
    if relation_type in {"CONTINUES", "RESUMES"}:
        if source_type != target_type:
            return f"{relation_type} exige episodios del mismo tipo."
        if source_start <= target_start:
            return f"{relation_type} debe ir del episodio posterior al previo."
        if not _continuity_referent_linked(record, source, target):
            return (
                f"{relation_type} exige un referente común testificado por "
                "anclas correspondientes o, para PROGRESSION, por una cadena "
                "LEADS_TO."
            )
        if relation_type == "RESUMES" and not _has_intervening_functional_change(
            record,
            source,
            target,
        ):
            return (
                "RESUMES exige una interrupción o un cambio funcional "
                "intermedio observable."
            )
    elif relation_type == "INTERRUPTS":
        target_remains_open = (
            source_start <= target_end
            or target.get("outcome") in {"INTERRUPTED", *OPEN_OUTCOMES}
        )
        if not (target_start < source_start and target_remains_open):
            return (
                "INTERRUPTS exige que el episodio interrumpido esté abierto "
                "cuando comienza el interruptor."
            )
    elif relation_type == "EMBEDS":
        if not (
            target_start <= source_start
            and source_end <= target_end
            and (source_start, source_end) != (target_start, target_end)
        ):
            return "EMBEDS exige que la fuente esté contenida en el destino."
    elif relation_type == "EVALUATES":
        if source_type != "VERIFICATION" or not _anchor_linked(
            source, target, "TARGET"
        ):
            return "EVALUATES exige una fuente VERIFICATION con TARGET vinculado."
    elif relation_type == "REPLACES":
        if source_type != "CORRECTION" or not _anchor_linked(
            source, target, "TARGET"
        ):
            return "REPLACES exige una fuente CORRECTION con TARGET vinculado."
    elif relation_type == "SUPPORTS_EPISODE":
        if source_type != "SUPPORT" or not _anchor_linked(
            source, target, "SUPPORTED"
        ):
            return (
                "SUPPORTS_EPISODE exige una fuente SUPPORT con ancla "
                "SUPPORTED vinculada."
            )
    elif relation_type == "COMPETES_WITH":
        if (
            source_type != "ALTERNATIVE"
            or target_type not in {"PROGRESSION", "ALTERNATIVE"}
            or not _anchor_linked(source, target, "CONTRAST")
        ):
            return (
                "COMPETES_WITH exige una fuente ALTERNATIVE, un destino "
                "PROGRESSION/ALTERNATIVE y un CONTRAST vinculado."
            )
    return None


def _validate_relations(
    record: dict[str, Any],
    episode_by_id: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stage = "relations"
    trace_text = record.get("sgot_graph", {}).get("trace_text", "")
    nodes = record.get("sgot_graph", {}).get("nodes", [])
    edges = record.get("sgot_graph", {}).get("edges", [])
    edge_by_id = {
        edge.get("edge_id"): edge
        for edge in edges
        if isinstance(edge, dict) and isinstance(edge.get("edge_id"), str)
    }
    relations = record.get("episode_relations")
    if not isinstance(relations, list):
        _error(
            errors,
            stage,
            "INVALID_RELATIONS",
            "episode_relations debe ser una lista.",
        )
        return {}
    relation_by_id: dict[str, dict[str, Any]] = {}
    triples: list[tuple[Any, Any, Any]] = []
    projection = {
        episode_id: _node_projection(episode, nodes)
        for episode_id, episode in episode_by_id.items()
    }
    for index, relation in enumerate(relations):
        object_id = (
            relation.get("relation_id")
            if isinstance(relation, dict)
            else f"relation[{index}]"
        )
        if not isinstance(relation, dict):
            _error(
                errors,
                stage,
                "INVALID_RELATION",
                "Cada relación debe ser un objeto.",
                str(object_id),
            )
            continue
        _validate_fields(
            relation,
            RELATION_FIELDS,
            RELATION_FIELDS,
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        relation_id = relation.get("relation_id")
        relation_type = relation.get("relation_type")
        source_id = relation.get("source_episode_id")
        target_id = relation.get("target_episode_id")
        if not isinstance(relation_id, str) or not relation_id:
            _error(
                errors,
                stage,
                "INVALID_RELATION_ID",
                "relation_id debe ser una cadena no vacía.",
                str(object_id),
            )
        elif relation_id in relation_by_id:
            _error(
                errors,
                stage,
                "DUPLICATE_RELATION_ID",
                "relation_id duplicado.",
                relation_id,
            )
        else:
            relation_by_id[relation_id] = relation
        if relation_type not in EPISODE_RELATION_TYPES:
            _error(
                errors,
                stage,
                "INVALID_RELATION_TYPE",
                f"relation_type inválido: {relation_type!r}.",
                str(object_id),
            )
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            _error(
                errors,
                stage,
                "INVALID_RELATION_ENDPOINT",
                "source_episode_id y target_episode_id deben ser cadenas.",
                str(object_id),
            )
            continue
        if source_id not in episode_by_id or target_id not in episode_by_id:
            _error(
                errors,
                stage,
                "UNKNOWN_RELATION_ENDPOINT",
                "La relación referencia episodios inexistentes.",
                str(object_id),
            )
            continue
        if source_id == target_id:
            _error(
                errors,
                stage,
                "SELF_RELATION",
                "No se permiten autorrelaciones.",
                str(object_id),
            )
        triples.append((relation_type, source_id, target_id))
        expected_id = stable_id(
            "rel",
            [record.get("trace_id"), relation_type, source_id, target_id],
        )
        if relation_id != expected_id:
            _error(
                errors,
                stage,
                "NONCANONICAL_RELATION_ID",
                "relation_id no coincide con sus extremos y tipo.",
                str(object_id),
            )
        source = episode_by_id[source_id]
        target = episode_by_id[target_id]
        source_projection = projection[source_id]
        target_projection = projection[target_id]
        allowed_nodes = source_projection | target_projection
        allowed_edges = {
            edge_id
            for edge_id, edge in edge_by_id.items()
            if edge.get("edge_type") != "NEXT"
            and (
                (
                    edge.get("source") in source_projection
                    and edge.get("target") in target_projection
                )
                or (
                    edge.get("source") in target_projection
                    and edge.get("target") in source_projection
                )
            )
        }
        is_functional = relation_type in FUNCTIONAL_RELATION_TYPES
        _validate_evidence(
            relation.get("evidence"),
            trace_text,
            allowed_node_ids=allowed_nodes,
            allowed_edge_ids=allowed_edges,
            require_witness=is_functional,
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        _validate_provenance(
            relation.get("provenance"),
            stage=stage,
            object_id=str(object_id),
            errors=errors,
        )
        if is_functional:
            compatibility_error = _relation_compatibility_error(
                record, relation, source, target
            )
            if compatibility_error:
                _error(
                    errors,
                    stage,
                    "INCOMPATIBLE_FUNCTIONAL_RELATION",
                    compatibility_error,
                    str(object_id),
                )
        elif relation_type in TEMPORAL_RELATION_TYPES:
            if relation.get("provenance", {}).get("origin") != "deterministic":
                _error(
                    errors,
                    stage,
                    "NONDETERMINISTIC_TEMPORAL_RELATION",
                    "IMMEDIATELY_PRECEDES_ONSET debe ser determinista.",
                    str(object_id),
                )
    if duplicate_triples := _duplicates(triples):
        _error(
            errors,
            stage,
            "DUPLICATE_RELATION_TRIPLE",
            f"Relaciones duplicadas: {duplicate_triples}.",
        )
    if episode_by_id:
        expected_temporal = _deterministic_temporal_relations(
            list(episode_by_id.values()), record.get("trace_id", "")
        )
        actual_temporal = [
            relation
            for relation in relations
            if isinstance(relation, dict)
            and relation.get("relation_type") in TEMPORAL_RELATION_TYPES
        ]
        if actual_temporal != expected_temporal:
            _error(
                errors,
                stage,
                "NONCANONICAL_TEMPORAL_RELATIONS",
                "Las relaciones temporales no coinciden con la derivación canónica.",
            )
    return relation_by_id


def _validate_exact_derivation(
    actual: Any,
    expected: Any,
    *,
    stage: str,
    code: str,
    label: str,
    errors: list[dict[str, Any]],
) -> None:
    if actual != expected:
        _error(
            errors,
            stage,
            code,
            f"{label} no coincide con la derivación determinista canónica.",
        )


def validate_record(
    record: Any,
    *,
    stage: str = "complete",
) -> dict[str, Any]:
    if stage not in {"episodes", "relations", "complete"}:
        raise ValueError(f"Etapa de validación inválida: {stage!r}.")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(record, dict):
        _error(
            errors,
            stage,
            "INVALID_RECORD",
            "El registro debe ser un objeto JSON.",
        )
        return {
            "stage": stage,
            "status": "INVALID",
            "errors": errors,
            "warnings": warnings,
        }

    _validate_fields(
        record,
        RECORD_FIELDS,
        RECORD_FIELDS,
        stage=stage,
        object_id=record.get("trace_id")
        if isinstance(record.get("trace_id"), str)
        else None,
        errors=errors,
    )
    if record.get("schema_version") != SCHEMA_VERSION:
        _error(
            errors,
            stage,
            "INVALID_SCHEMA_VERSION",
            f"schema_version debe ser {SCHEMA_VERSION!r}.",
        )
    if record.get("pipeline_version") != PIPELINE_VERSION:
        _error(
            errors,
            stage,
            "INVALID_PIPELINE_VERSION",
            f"pipeline_version debe ser {PIPELINE_VERSION!r}.",
        )
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        _error(
            errors,
            stage,
            "INVALID_TRACE_ID",
            "trace_id debe ser una cadena no vacía.",
        )
    graph = record.get("sgot_graph")
    if not isinstance(graph, dict):
        _error(
            errors,
            stage,
            "INVALID_SGOT_GRAPH",
            "sgot_graph debe ser un objeto.",
        )
        graph = {}
    else:
        if graph.get("schema_version") != OBJECTIVE2_SCHEMA_VERSION:
            _error(
                errors,
                stage,
                "INVALID_SGOT_GRAPH_VERSION",
                f"sgot_graph debe usar {OBJECTIVE2_SCHEMA_VERSION!r}.",
            )
        if trace_id and graph.get("trace_id") != trace_id:
            _error(
                errors,
                stage,
                "TRACE_ID_MISMATCH",
                "sgot_graph.trace_id no coincide con trace_id.",
            )
        try:
            graph_report = validate_objective2_record(graph)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            graph_report = {
                "status": "invalid",
                "hard_errors": [str(exc)],
                "warnings": [],
            }
        if graph_report["status"] == "invalid":
            _error(
                errors,
                stage,
                "INVALID_SGOT_GRAPH_CONTRACT",
                "sgot_graph incumple el contrato del objetivo 2: "
                + "; ".join(graph_report["hard_errors"][:3]),
            )
        graph_warnings = graph_report.get(
            "warnings",
            graph_report.get("review_findings", []),
        )
        for finding in graph_warnings:
            warnings.append(
                {
                    "stage": stage,
                    "code": "SGOT_GRAPH_DIAGNOSTIC",
                    "object_id": trace_id,
                    "message": canonical_json(finding),
                }
            )
    if not isinstance(record.get("validation"), dict):
        _error(
            errors,
            stage,
            "INVALID_VALIDATION_OBJECT",
            "validation debe ser un objeto.",
        )
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        _error(
            errors,
            stage,
            "INVALID_RECORD_PROVENANCE",
            "provenance debe ser un objeto.",
        )
    else:
        _validate_fields(
            provenance,
            RECORD_PROVENANCE_FIELDS,
            RECORD_PROVENANCE_FIELDS,
            stage=stage,
            object_id=trace_id if isinstance(trace_id, str) else None,
            errors=errors,
        )
        if provenance.get("schema_version") != SCHEMA_VERSION:
            _error(
                errors,
                stage,
                "INVALID_PROVENANCE_SCHEMA_VERSION",
                f"provenance.schema_version debe ser {SCHEMA_VERSION!r}.",
                trace_id if isinstance(trace_id, str) else None,
            )
        if provenance.get("pipeline_version") != PIPELINE_VERSION:
            _error(
                errors,
                stage,
                "INVALID_PROVENANCE_PIPELINE_VERSION",
                f"provenance.pipeline_version debe ser {PIPELINE_VERSION!r}.",
                trace_id if isinstance(trace_id, str) else None,
            )
        provenance_stages = provenance.get("stages")
        if not isinstance(provenance_stages, list) or not all(
            isinstance(item, dict) for item in provenance_stages
        ):
            _error(
                errors,
                stage,
                "INVALID_STAGE_PROVENANCE",
                "provenance.stages debe ser una lista de objetos.",
                trace_id if isinstance(trace_id, str) else None,
            )
        else:
            for item in provenance_stages:
                if (
                    item.get("origin") not in PROVENANCE_ORIGINS
                    or not isinstance(item.get("stage"), str)
                    or not item.get("stage")
                ):
                    _error(
                        errors,
                        stage,
                        "INVALID_STAGE_PROVENANCE_ENTRY",
                        "Cada evento de procedencia requiere origin y stage válidos.",
                        trace_id if isinstance(trace_id, str) else None,
                    )
                    break
        runtime = provenance.get("runtime")
        if not isinstance(runtime, dict):
            _error(
                errors,
                stage,
                "INVALID_RUNTIME_PROVENANCE",
                "provenance.runtime debe ser un objeto.",
                trace_id if isinstance(trace_id, str) else None,
            )
        else:
            _validate_fields(
                runtime,
                RUNTIME_FIELDS,
                RUNTIME_FIELDS,
                stage=stage,
                object_id=trace_id if isinstance(trace_id, str) else None,
                errors=errors,
            )
            for field in RUNTIME_FIELDS - {"code_commit"}:
                if not isinstance(runtime.get(field), str) or not runtime.get(field):
                    _error(
                        errors,
                        stage,
                        "INVALID_RUNTIME_FIELD",
                        f"provenance.runtime.{field} debe ser una cadena no vacía.",
                        trace_id if isinstance(trace_id, str) else None,
                    )
            code_commit = runtime.get("code_commit")
            if code_commit is not None and (
                not isinstance(code_commit, str) or not code_commit
            ):
                _error(
                    errors,
                    stage,
                    "INVALID_CODE_COMMIT",
                    "provenance.runtime.code_commit debe ser una cadena o null.",
                    trace_id if isinstance(trace_id, str) else None,
                )
            source_hash = runtime.get("source_hash")
            if isinstance(source_hash, str) and (
                not source_hash.startswith("sha256:")
                or len(source_hash) != len("sha256:") + 64
            ):
                _error(
                    errors,
                    stage,
                    "INVALID_SOURCE_HASH",
                    "provenance.runtime.source_hash debe ser un SHA-256 completo.",
                    trace_id if isinstance(trace_id, str) else None,
                )
    if not isinstance(record.get("audit"), list):
        _error(
            errors,
            stage,
            "INVALID_AUDIT",
            "audit debe ser una lista.",
        )

    episode_by_id = _validate_episodes(record, errors)
    if stage in {"relations", "complete"}:
        _validate_relations(record, episode_by_id, errors)
    else:
        for field in ("episode_relations", "progression_lines", "routes", "configurations"):
            if record.get(field) != []:
                _error(
                    errors,
                    stage,
                    "PREMATURE_DERIVATION",
                    f"{field} debe permanecer vacío en la etapa de episodios.",
                )
        if record.get("profile") is not None:
            _error(
                errors,
                stage,
                "PREMATURE_PROFILE",
                "profile debe ser null en la etapa de episodios.",
            )

    if stage == "relations":
        for field in ("progression_lines", "routes", "configurations"):
            if record.get(field) != []:
                _error(
                    errors,
                    stage,
                    "PREMATURE_DERIVATION",
                    f"{field} debe permanecer vacío antes de la etapa 08.",
                )
        if record.get("profile") is not None:
            _error(
                errors,
                stage,
                "PREMATURE_PROFILE",
                "profile debe ser null antes de la etapa 08.",
            )

    if stage == "complete" and not errors:
        _validate_exact_derivation(
            record.get("progression_lines"),
            derive_progression_lines(record),
            stage=stage,
            code="NONCANONICAL_PROGRESSION_LINES",
            label="progression_lines",
            errors=errors,
        )
        _validate_exact_derivation(
            record.get("routes"),
            derive_routes(record),
            stage=stage,
            code="NONCANONICAL_ROUTES",
            label="routes",
            errors=errors,
        )
        if not errors:
            _validate_exact_derivation(
                record.get("configurations"),
                derive_configurations(record),
                stage=stage,
                code="NONCANONICAL_CONFIGURATIONS",
                label="configurations",
                errors=errors,
            )
        if not errors:
            _validate_exact_derivation(
                record.get("profile"),
                derive_profile(record),
                stage=stage,
                code="NONCANONICAL_PROFILE",
                label="profile",
                errors=errors,
            )

    # Estas comprobaciones dan mensajes locales incluso cuando la igualdad global
    # de la derivación todavía no puede calcularse.
    if stage == "complete":
        if not isinstance(record.get("progression_lines"), list):
            _error(
                errors,
                stage,
                "INVALID_PROGRESSION_LINES",
                "progression_lines debe ser una lista.",
            )
        if not isinstance(record.get("routes"), list):
            _error(errors, stage, "INVALID_ROUTES", "routes debe ser una lista.")
        if not isinstance(record.get("configurations"), list):
            _error(
                errors,
                stage,
                "INVALID_CONFIGURATIONS",
                "configurations debe ser una lista.",
            )
        if not isinstance(record.get("profile"), dict):
            _error(
                errors,
                stage,
                "INVALID_PROFILE",
                "profile debe ser un objeto en la salida completa.",
            )

    return {
        "stage": stage,
        "status": "INVALID" if errors else "VALID",
        "errors": errors,
        "warnings": warnings,
    }


__all__ = [
    "AMBIGUITY_CODES",
    "ANCHOR_ROLES",
    "EPISODE_RELATION_TYPES",
    "EPISODE_TYPES",
    "FUNCTIONAL_RELATION_TYPES",
    "OUTCOMES_BY_TYPE",
    "PIPELINE_VERSION",
    "PROFILE_VERSION",
    "REQUIRED_ANCHOR_ROLE",
    "ROUTE_CRITERIA",
    "SCHEMA_VERSION",
    "TEMPORAL_RELATION_TYPES",
    "add_episode_relations",
    "add_episode_relations_from_candidates",
    "build_objective3_record",
    "build_objective3_record_from_candidates",
    "canonical_json",
    "derive_configurations",
    "derive_profile",
    "derive_progression_lines",
    "derive_routes",
    "derive_routes_and_profile",
    "episode_envelope",
    "normalize_episode",
    "normalize_episode_relation",
    "runtime_metadata",
    "stable_id",
    "validate_record",
]
