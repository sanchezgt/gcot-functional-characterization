"""Estimación jerárquica de episodios mediante bloques solapados.

El módulo organiza el contexto y valida contratos operacionales. Todas las
decisiones sobre existencia, tipo, límites, anclas, desenlace y evidencia
pertenecen a estimadores LLM.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from util.llm_provider import (
    LLMClient,
    request_structured_response,
    structured_request_hash,
)
from util.trace_alias import prompt_trace_alias
from util.objective3_llm import (
    EPISODE_REPAIR_SUFFIX,
    EPISODE_SPECIFICATIONS,
    EPISODE_SYSTEM_PROMPT,
    EPISODE_TYPES,
    EpisodeBatchDraft,
    EpisodeDraft,
    _graph_payload,
    _episode_candidate_contract_errors,
    _materialize_episode_candidate,
    _persist_semantic_attempt,
    _persist_semantic_request_failure,
    _reconcile_episode_candidates,
    _response_provenance,
    _validation_errors_json,
    infer_episodes,
)
from util.sgotgraph_objective3 import (
    AMBIGUITY_CODES,
    ANCHOR_ROLES,
    OUTCOMES_BY_TYPE,
    build_objective3_record_from_candidates,
    canonical_json,
    episode_envelope,
    normalize_episode,
)


DEFAULT_BLOCK_CORE_NODES = 32
DEFAULT_BLOCK_HALO_NODES = 8
DEFAULT_BLOCK_SHIFT_NODES = 0

SYNTHESIS_ACTIONS = {
    "ADDED",
    "MERGED",
    "PRESERVED",
    "REVISED",
}
MERGE_BASIS_CODES = {
    "COMPLEMENTARY_FRAGMENTS",
    "SAME_FUNCTIONAL_REFERENT",
    "SAME_OBSERVABLE_OPERATION",
    "SINGLE_DEPENDENT_OUTCOME",
}
REMOVAL_REASON_CODES = {
    "DUPLICATE_OR_REDUNDANT_PROPOSAL",
    "INCOMPLETE_LOCAL_OPERATION",
    "INSUFFICIENT_EVIDENCE",
    "NOT_SUPPORTED_IN_FULL_CONTEXT",
    "TYPE_DEFINITION_NOT_MET",
}


class SynthesizedEpisodeDraft(EpisodeDraft):
    model_config = ConfigDict(extra="forbid")

    source_proposal_refs: list[str]
    synthesis_action: Literal["ADDED", "MERGED", "REVISED"]
    merge_basis_codes: list[str]


class RemovedProposalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_ref: str
    reason_code: str


class EpisodeSynthesisBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    episode_type: str
    preserved_proposal_refs: list[str]
    episodes: list[SynthesizedEpisodeDraft]
    removed_proposals: list[RemovedProposalDraft]


LOCAL_EPISODE_SYSTEM_PROMPT = (
    """You propose functional episodes from one exact visible block of an
automatically generated reasoning trace.

The block is not the complete trace. It contains a non-overlapping ownership
core and a left/right context halo. Use global character offsets exactly as
provided. Propose only complete episodes of the requested type whose earliest
member begins inside the ownership core. Halo content supplies context but
cannot own a candidate. If the visible evidence is insufficient to represent a
complete episode, omit it; a later global synthesis examines the full trace.

Formal episode rules:"""
    + (
        EPISODE_SYSTEM_PROMPT.split(
            "Formal episode rules:",
            maxsplit=1,
        )[1]
    ).replace(
        "Return exactly one JSON object matching the requested schema.",
        """Additional block rules:
15. Every member, anchor, evidence span, witness node, and witness edge must be
    visible in this block.
16. The earliest member start must lie in the half-open global character
    interval [core_start_char, core_end_char).
17. Do not infer content hidden outside the visible interval.
18. Return all and only the complete requested-type episodes owned by this
    core. Return an empty array when none is supported.

Return exactly one JSON object matching the requested schema.""",
    )
)


LOCAL_EPISODE_USER_TEMPLATE = """Propose all locally observable episodes of
type {episode_type}.

Type specification:
{type_specification}

Allowed outcomes:
{outcomes}

Required specialized anchor:
{required_anchor}

Allowed anchor roles:
{anchor_roles}

Allowed ambiguity codes:
{ambiguity_codes}

The output must use trace_id={trace_id!r} and
episode_type={episode_type!r}.

Block input:
{input_json}
"""


SYNTHESIS_SYSTEM_PROMPT = (
    """You synthesize the final complete set of one functional episode type
for an automatically generated reasoning trace.

You receive the complete trace, complete SGoTGraph, and local proposals from
overlapping blocks. Local proposals are evidence-bearing suggestions, not
fixed labels and not a complete answer. Apply the same episode definition,
exclusions, anchors, outcomes, and formal rules used by the global estimator.

You may:
- preserve a supported proposal;
- revise or extend a proposal when full context changes its exact boundaries,
  anchors, outcome, or evidence;
- merge proposals that express one operation on the same referent;
- remove a proposal that does not satisfy the definition in full context;
- add an episode omitted locally, including a discontinuous or non-local
  episode.

When a proposal remains semantically unchanged, put only its proposal_ref in
preserved_proposal_refs. Do not reproduce its members, anchors, outcome,
ambiguity, or evidence; the pipeline materializes that episode literally from
the referenced proposal.

For every final episode returned in episodes, list every local proposal_ref
that materially contributed to the observable operation. Material contribution
requires at least one exact character of a proposal member to remain inside a
final member; shared type, topic, anchor, outcome, or final answer is not
contribution. Do not enlarge members merely to manufacture overlap. Classify
each such episode with exactly one synthesis_action:
- REVISED: exactly one proposal_ref, with at least one retained member
  character, but a full-context change to the semantic episode;
- MERGED: at least two proposal_refs that are complementary observations or
  fragments of one observable operation;
- ADDED: a genuinely new global episode with no source_proposal_refs.

For MERGED, merge_basis_codes must include SAME_OBSERVABLE_OPERATION and may
also include COMPLEMENTARY_FRAGMENTS, SAME_FUNCTIONAL_REFERENT, or
SINGLE_DEPENDENT_OUTCOME. Same type, same referent, or same outcome alone never
justifies merging independent operations. merge_basis_codes must be empty for
all other actions.

PRESERVED is represented only by an identifier in preserved_proposal_refs; it
is not an allowed synthesis_action inside episodes.

Put every unused proposal in removed_proposals with one reason_code:
DUPLICATE_OR_REDUNDANT_PROPOSAL, INCOMPLETE_LOCAL_OPERATION,
INSUFFICIENT_EVIDENCE, NOT_SUPPORTED_IN_FULL_CONTEXT, or
TYPE_DEFINITION_NOT_MET. Each proposal_ref must appear exactly once, either in
preserved_proposal_refs, in one final episode, or in removed_proposals.

Do not prefer a candidate because several windows produced it. Do not maximize
episode count or textual coverage.

Formal episode rules:"""
    + EPISODE_SYSTEM_PROMPT.split("Formal episode rules:", maxsplit=1)[1]
)


SYNTHESIS_USER_TEMPLATE = """Synthesize the final complete episode set for
type {episode_type}.

Type specification:
{type_specification}

Allowed outcomes:
{outcomes}

Required specialized anchor:
{required_anchor}

Allowed anchor roles:
{anchor_roles}

Allowed ambiguity codes:
{ambiguity_codes}

The output must use trace_id={trace_id!r} and
episode_type={episode_type!r}.

Every proposal_ref must appear exactly once in preserved_proposal_refs, in one
final episode's source_proposal_refs, or in removed_proposals.

For an unchanged proposal, return only its identifier in
preserved_proposal_refs. Never copy its semantic fields into episodes.

Allowed synthesis_action values inside episodes:
{synthesis_actions}

Allowed merge basis codes:
{merge_basis_codes}

Allowed removal reason codes:
{removal_reason_codes}

Input:
{input_json}
"""


SYNTHESIS_REPAIR_SUFFIX = """

The previous synthesis did not satisfy the deterministic contract. Return the
complete corrected synthesis for this same type. Preserve supported semantic
decisions, but account for every proposal exactly once, make every synthesis
operation explicit, and correct any invalid episode.

candidate_index is 1-based. Candidates materialized from the episodes array
come first, followed by proposals materialized from preserved_proposal_refs in
the listed order. The allowed witness sets are exhaustive for the candidate's
current members. A source_proposal_ref may remain attached only when its member
intervals literally overlap the final member intervals. Otherwise remove that
reference with an explicit reason, or represent a genuinely new global episode
as ADDED with no source refs. Do not add or enlarge an episode merely to
satisfy validation.

Deterministic validation errors:
{validation_errors}

Previous output:
{previous_output_json}
"""


def _positive_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} debe ser un entero mayor que 0.")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{name} debe ser un entero no negativo.")
    return value


def build_episode_blocks(
    sgot_graph: dict[str, Any],
    *,
    core_nodes: int = DEFAULT_BLOCK_CORE_NODES,
    halo_nodes: int = DEFAULT_BLOCK_HALO_NODES,
    shift_nodes: int = DEFAULT_BLOCK_SHIFT_NODES,
) -> list[dict[str, Any]]:
    """Construye bloques deterministas sobre la secuencia canónica de nodos."""

    core_nodes = _positive_integer(core_nodes, "core_nodes")
    halo_nodes = _nonnegative_integer(halo_nodes, "halo_nodes")
    shift_nodes = _nonnegative_integer(shift_nodes, "shift_nodes")
    if shift_nodes >= core_nodes:
        raise ValueError("shift_nodes debe ser menor que core_nodes.")

    trace_id = sgot_graph.get("trace_id")
    trace_text = sgot_graph.get("trace_text")
    nodes = sgot_graph.get("nodes")
    edges = sgot_graph.get("edges")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("SGoTGraph no contiene trace_id válido.")
    if not isinstance(trace_text, str):
        raise ValueError("SGoTGraph no contiene trace_text válido.")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("SGoTGraph debe contener una lista no vacía de nodos.")
    if not isinstance(edges, list):
        raise ValueError("SGoTGraph debe contener una lista de aristas.")
    expected_positions = list(range(1, len(nodes) + 1))
    positions = [node.get("position") for node in nodes]
    if positions != expected_positions:
        raise ValueError(
            "Los nodos deben estar en orden canónico con posiciones "
            "consecutivas desde 1."
        )

    core_ranges: list[tuple[int, int]] = []
    cursor = 0
    if shift_nodes:
        core_ranges.append((0, min(shift_nodes, len(nodes))))
        cursor = shift_nodes
    while cursor < len(nodes):
        end = min(cursor + core_nodes, len(nodes))
        core_ranges.append((cursor, end))
        cursor = end

    alignment = "start0" if shift_nodes == 0 else f"shift{shift_nodes}"
    blocks: list[dict[str, Any]] = []
    for index, (core_start, core_end) in enumerate(core_ranges, start=1):
        visible_start = max(0, core_start - halo_nodes)
        visible_end = min(len(nodes), core_end + halo_nodes)
        visible_nodes = nodes[visible_start:visible_end]
        visible_ids = {
            node.get("segment_id")
            for node in visible_nodes
            if isinstance(node.get("segment_id"), str)
        }
        visible_edges = [
            edge
            for edge in edges
            if edge.get("source") in visible_ids
            and edge.get("target") in visible_ids
        ]
        core_start_char = nodes[core_start].get("start_char")
        core_end_char = nodes[core_end - 1].get("end_char")
        visible_start_char = nodes[visible_start].get("start_char")
        visible_end_char = nodes[visible_end - 1].get("end_char")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                core_start_char,
                core_end_char,
                visible_start_char,
                visible_end_char,
            )
        ):
            raise ValueError("Los nodos contienen offsets no enteros.")
        block_id = (
            f"{alignment}_b{index:03d}_"
            f"p{core_start + 1:04d}-{core_end:04d}"
        )
        blocks.append(
            {
                "block_id": block_id,
                "alignment": alignment,
                "core_start_index": core_start,
                "core_end_index": core_end,
                "visible_start_index": visible_start,
                "visible_end_index": visible_end,
                "core_start_position": core_start + 1,
                "core_end_position": core_end,
                "visible_start_position": visible_start + 1,
                "visible_end_position": visible_end,
                "core_start_char": core_start_char,
                "core_end_char": core_end_char,
                "visible_start_char": visible_start_char,
                "visible_end_char": visible_end_char,
                "trace_id": trace_id,
                "global_trace_length_chars": len(trace_text),
                "visible_trace_text": trace_text[
                    visible_start_char:visible_end_char
                ],
                "nodes": visible_nodes,
                "edges": visible_edges,
            }
        )
    return blocks


def _block_payload(block: dict[str, Any]) -> dict[str, Any]:
    return {
        # Alias opaco solo en la carga del prompt; el registro del bloque
        # conserva el identificador real.
        "trace_id": prompt_trace_alias(block["trace_id"]),
        "global_trace_length_chars": block["global_trace_length_chars"],
        "block": {
            key: block[key]
            for key in (
                "block_id",
                "alignment",
                "core_start_position",
                "core_end_position",
                "visible_start_position",
                "visible_end_position",
                "core_start_char",
                "core_end_char",
                "visible_start_char",
                "visible_end_char",
            )
        },
        "visible_trace_slice": {
            "start_char": block["visible_start_char"],
            "end_char": block["visible_end_char"],
            "text": block["visible_trace_text"],
        },
        "nodes": [
            {
                "segment_id": node.get("segment_id"),
                "node_type": node.get("node_type"),
                "start_char": node.get("start_char"),
                "end_char": node.get("end_char"),
                "text": node.get("text"),
            }
            for node in block["nodes"]
        ],
        "edges": [
            {
                "edge_id": edge.get("edge_id"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "edge_type": edge.get("edge_type"),
            }
            for edge in block["edges"]
        ],
    }


def _error(
    code: str,
    message: str,
    **context: Any,
) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


def _valid_span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start_char")
    end = value.get("end_char")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or end <= start
    ):
        return None
    return start, end


def _candidate_block_errors(
    candidates: list[dict[str, Any]],
    block: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    visible_start = block["visible_start_char"]
    visible_end = block["visible_end_char"]
    core_start = block["core_start_char"]
    core_end = block["core_end_char"]
    visible_node_ids = {
        str(node.get("segment_id")) for node in block["nodes"]
    }
    visible_edge_ids = {
        str(edge.get("edge_id")) for edge in block["edges"]
    }

    for index, candidate in enumerate(candidates, start=1):
        members = [
            interval
            for interval in (
                _valid_span(span)
                for span in candidate.get("members", [])
            )
            if interval is not None
        ]
        if members:
            earliest_start = min(start for start, _ in members)
            if not core_start <= earliest_start < core_end:
                errors.append(
                    _error(
                        "MEMBER_START_OUTSIDE_OWNERSHIP_CORE",
                        (
                            f"Candidato {index}: el primer miembro comienza "
                            f"en {earliest_start}; el núcleo es "
                            f"[{core_start}, {core_end})."
                        ),
                        candidate_index=index,
                        member_start=earliest_start,
                        ownership_core={
                            "start_char": core_start,
                            "end_char": core_end,
                        },
                    )
                )

        span_groups: list[tuple[str, list[Any]]] = [
            ("members", list(candidate.get("members", []))),
            ("anchors", list(candidate.get("anchors", []))),
            (
                "evidence.text_spans",
                list(
                    candidate.get("evidence", {}).get(
                        "text_spans",
                        [],
                    )
                ),
            ),
        ]
        for field in ("entry_anchor", "exit_anchor"):
            value = candidate.get(field)
            if value is not None:
                span_groups.append((field, [value]))
        for label, spans in span_groups:
            for span in spans:
                interval = _valid_span(span)
                if interval is None:
                    continue
                start, end = interval
                if start < visible_start or end > visible_end:
                    errors.append(
                        _error(
                            "SPAN_OUTSIDE_VISIBLE_BLOCK",
                            (
                                f"Candidato {index}, {label}: "
                                f"[{start}, {end}) queda fuera de "
                                f"[{visible_start}, {visible_end})."
                            ),
                            candidate_index=index,
                            field=label,
                            interval={
                                "start_char": start,
                                "end_char": end,
                            },
                            visible_interval={
                                "start_char": visible_start,
                                "end_char": visible_end,
                            },
                        )
                    )
        evidence = candidate.get("evidence", {})
        for node_id in evidence.get("witness_node_ids", []):
            if str(node_id) not in visible_node_ids:
                errors.append(
                    _error(
                        "WITNESS_NODE_OUTSIDE_VISIBLE_BLOCK",
                        f"Candidato {index}: nodo no visible {node_id!r}.",
                        candidate_index=index,
                        invalid_witness_node_id=str(node_id),
                        visible_witness_node_ids=sorted(visible_node_ids),
                    )
                )
        for edge_id in evidence.get("witness_edge_ids", []):
            if str(edge_id) not in visible_edge_ids:
                errors.append(
                    _error(
                        "WITNESS_EDGE_OUTSIDE_VISIBLE_BLOCK",
                        f"Candidato {index}: arista no visible {edge_id!r}.",
                        candidate_index=index,
                        invalid_witness_edge_id=str(edge_id),
                        visible_witness_edge_ids=sorted(visible_edge_ids),
                    )
                )
    return errors


def _formal_candidate_errors(
    sgot_graph: dict[str, Any],
    candidates: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    return _episode_candidate_contract_errors(
        sgot_graph,
        candidates,
        provenance,
        action="hierarchical_candidate_contract_check",
    )


def _infer_local_type_block(
    client: LLMClient,
    *,
    model: str,
    sgot_graph: dict[str, Any],
    block: dict[str, Any],
    episode_type: str,
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    semantic_audit_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specification = EPISODE_SPECIFICATIONS[episode_type]
    trace_id = sgot_graph["trace_id"]
    trace_text = sgot_graph["trace_text"]
    base_prompt = LOCAL_EPISODE_USER_TEMPLATE.format(
        episode_type=episode_type,
        type_specification=(
            specification["definition"]
            + "\nExclusion: "
            + specification["exclusion"]
        ),
        outcomes=", ".join(sorted(OUTCOMES_BY_TYPE[episode_type])),
        required_anchor=specification["required_anchor"],
        anchor_roles=", ".join(sorted(ANCHOR_ROLES)),
        ambiguity_codes=", ".join(sorted(AMBIGUITY_CODES)),
        trace_id=prompt_trace_alias(trace_id),
        input_json=canonical_json(_block_payload(block)),
    )
    previous_output: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = []
    provenance_events: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        user_prompt = base_prompt
        if previous_output is not None:
            user_prompt += EPISODE_REPAIR_SUFFIX.format(
                validation_errors=_validation_errors_json(
                    validation_errors
                ),
                previous_output_json=canonical_json(previous_output),
            )
        prompt_hash = structured_request_hash(
            client,
            LOCAL_EPISODE_SYSTEM_PROMPT,
            user_prompt,
        )
        if verbose:
            print(
                f"[{trace_id}] Bloque {block['block_id']}: "
                f"{episode_type} (intento {attempt}/{max_attempts}).",
                file=sys.stderr,
                flush=True,
            )
        stage = (
            f"06_episode_proposals:{episode_type}:"
            f"{block['block_id']}"
        )
        try:
            response = request_structured_response(
                client,
                model=model,
                system_prompt=LOCAL_EPISODE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=EpisodeBatchDraft,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            _persist_semantic_request_failure(
                semantic_audit_path,
                trace_id=trace_id,
                stage=stage,
                client=client,
                model=model,
                prompt_hash=prompt_hash,
                reasoning_effort=reasoning_effort,
                attempt=attempt,
                exception=exc,
                context={
                    "method": "hierarchical",
                    "phase": "local_proposal",
                    "block_id": block["block_id"],
                    "alignment": block["alignment"],
                    "episode_type": episode_type,
                },
            )
            raise
        batch = response.parsed
        validation_errors = []
        if batch.trace_id != prompt_trace_alias(trace_id):
            validation_errors.append(
                _error(
                    "TRACE_ID_MISMATCH",
                    (
                        f"trace_id={batch.trace_id!r}; se esperaba "
                        f"{prompt_trace_alias(trace_id)!r}."
                    ),
                )
            )
        if batch.episode_type != episode_type:
            validation_errors.append(
                _error(
                    "EPISODE_TYPE_MISMATCH",
                    (
                        f"episode_type={batch.episode_type!r}; se esperaba "
                        f"{episode_type!r}."
                    ),
                )
            )
        provenance = _response_provenance(
            stage=stage,
            client=client,
            model=model,
            prompt_hash=prompt_hash,
            reasoning_effort=reasoning_effort,
            response=response,
        )
        candidates = [
            _materialize_episode_candidate(
                episode,
                episode_type=episode_type,
                trace_text=trace_text,
                provenance=provenance,
            )
            for episode in batch.episodes
        ]
        validation_errors.extend(
            _candidate_block_errors(candidates, block)
        )
        validation_errors.extend(
            _formal_candidate_errors(
                sgot_graph,
                candidates,
                provenance,
            )
        )
        event = _persist_semantic_attempt(
            semantic_audit_path,
            trace_id=trace_id,
            event={
                **provenance,
                "method": "hierarchical",
                "phase": "local_proposal",
                "block_id": block["block_id"],
                "alignment": block["alignment"],
                "core_start_position": block["core_start_position"],
                "core_end_position": block["core_end_position"],
                "visible_start_position": block[
                    "visible_start_position"
                ],
                "visible_end_position": block["visible_end_position"],
                "episode_type": episode_type,
                "attempt": attempt,
                "candidate_count": len(candidates),
                "validation_status": (
                    "VALID" if not validation_errors else "INVALID"
                ),
                "validation_error_codes": [
                    error["code"] for error in validation_errors
                ],
            },
            validation_errors=validation_errors,
            response_payload=batch.model_dump(),
        )
        provenance_events.append(event)
        if not validation_errors:
            return candidates, provenance_events
        previous_output = batch.model_dump()

    details = "; ".join(
        f"{error['code']}: {error['message']}"
        for error in validation_errors[:4]
    )
    raise ValueError(
        f"El bloque {block['block_id']} agotó sus intentos para "
        f"{episode_type}: {details}"
    )


def _proposal_semantic_payload(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "episode_type",
            "members",
            "anchors",
            "entry_anchor",
            "exit_anchor",
            "outcome",
            "ambiguity_codes",
            "evidence",
        )
    }


def _normalized_semantic_payload(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_episode(
        candidate,
        episode_type=(
            str(candidate["episode_type"])
            if isinstance(candidate.get("episode_type"), str)
            else None
        ),
        provenance={
            "origin": "deterministic",
            "stage": "synthesis_action_contract_check",
        },
    )
    return _proposal_semantic_payload(normalized)


def _consolidate_exact_proposals(
    candidates_by_block: list[tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    source_blocks: dict[str, set[str]] = defaultdict(set)
    source_inferences: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for candidate, block_id in candidates_by_block:
        payload = _proposal_semantic_payload(candidate)
        key = canonical_json(payload)
        grouped.setdefault(key, payload)
        source_blocks[key].add(block_id)
        inference = {
            "block_id": block_id,
            "alignment": block_id.split("_b", maxsplit=1)[0],
            "provenance": dict(candidate.get("provenance", {})),
        }
        source_inferences[key][canonical_json(inference)] = inference
    ordered = sorted(
        grouped,
        key=lambda key: (
            episode_envelope(grouped[key]),
            key,
        ),
    )
    return [
        {
            "proposal_ref": f"P{index:04d}",
            **grouped[key],
            "source_block_ids": sorted(source_blocks[key]),
            "source_inferences": [
                source_inferences[key][inference_key]
                for inference_key in sorted(source_inferences[key])
            ],
        }
        for index, key in enumerate(ordered, start=1)
    ]


def _proposal_prompt_payload(
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Elimina metadatos repetidos sin retirar información semántica."""

    return {
        key: value
        for key, value in proposal.items()
        if key != "source_inferences"
    } | {
        "source_inference_count": len(
            proposal.get("source_inferences", [])
        )
    }


def _member_intervals(value: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        interval
        for interval in (
            _valid_span(span) for span in value.get("members", [])
        )
        if interval is not None
    ]


def _member_contribution_stats(
    proposal: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    proposal_intervals = _member_intervals(proposal)
    final_intervals = _member_intervals(candidate)
    overlap_chars = sum(
        max(
            0,
            min(proposal_end, final_end)
            - max(proposal_start, final_start),
        )
        for proposal_start, proposal_end in proposal_intervals
        for final_start, final_end in final_intervals
    )
    proposal_chars = sum(
        end - start for start, end in proposal_intervals
    )
    final_chars = sum(end - start for start, end in final_intervals)
    return {
        "overlap_chars": overlap_chars,
        "proposal_member_chars": proposal_chars,
        "final_member_chars": final_chars,
        "proposal_overlap_ratio": (
            overlap_chars / proposal_chars if proposal_chars else None
        ),
        "final_overlap_ratio": (
            overlap_chars / final_chars if final_chars else None
        ),
    }


def _synthesis_reference_errors(
    batch: EpisodeSynthesisBatchDraft,
    *,
    trace_id: str,
    episode_type: str,
    proposals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if batch.trace_id != prompt_trace_alias(trace_id):
        errors.append(
            _error(
                "TRACE_ID_MISMATCH",
                (
                    f"trace_id={batch.trace_id!r}; se esperaba "
                    f"{prompt_trace_alias(trace_id)!r}."
                ),
            )
        )
    if batch.episode_type != episode_type:
        errors.append(
            _error(
                "EPISODE_TYPE_MISMATCH",
                (
                    f"episode_type={batch.episode_type!r}; se esperaba "
                    f"{episode_type!r}."
                ),
            )
        )
    proposal_by_ref = {
        str(proposal["proposal_ref"]): proposal
        for proposal in proposals
    }
    expected = set(proposal_by_ref)
    preserved = list(batch.preserved_proposal_refs)
    synthesized: list[str] = [
        proposal_ref
        for episode in batch.episodes
        for proposal_ref in episode.source_proposal_refs
    ]
    used = [*preserved, *synthesized]
    removed = [
        item.proposal_ref for item in batch.removed_proposals
    ]
    for label, values in (
        ("used_proposal_refs", used),
        ("removed_proposals", removed),
    ):
        unknown = sorted(set(values) - expected)
        if unknown:
            errors.append(
                _error(
                    "UNKNOWN_PROPOSAL_REF",
                    f"{label} contiene referencias desconocidas: {unknown}.",
                )
            )
        duplicates = sorted(
            value for value in set(values) if values.count(value) > 1
        )
        if duplicates:
            errors.append(
                _error(
                    "DUPLICATE_PROPOSAL_REF",
                    f"{label} repite referencias: {duplicates}.",
                )
            )
    overlap = sorted(set(used) & set(removed))
    if overlap:
        errors.append(
            _error(
                "PROPOSAL_USED_AND_REMOVED",
                f"Referencias usadas y retiradas simultáneamente: {overlap}.",
            )
        )
    missing = sorted(expected - set(used) - set(removed))
    if missing:
        errors.append(
            _error(
                "UNACCOUNTED_PROPOSAL_REF",
                f"Faltan decisiones para las propuestas: {missing}.",
            )
        )

    for removed_index, item in enumerate(
        batch.removed_proposals,
        start=1,
    ):
        if item.reason_code not in REMOVAL_REASON_CODES:
            errors.append(
                _error(
                    "INVALID_REMOVAL_REASON",
                    (
                        f"removed_proposals[{removed_index}] usa "
                        f"reason_code={item.reason_code!r}."
                    ),
                    removed_proposal_index=removed_index,
                    proposal_ref=item.proposal_ref,
                    allowed_reason_codes=sorted(REMOVAL_REASON_CODES),
                )
            )

    for index, (episode, candidate) in enumerate(
        zip(batch.episodes, candidates),
        start=1,
    ):
        action = episode.synthesis_action
        refs = list(episode.source_proposal_refs)
        bases = list(episode.merge_basis_codes)
        context = {
            "candidate_index": index,
            "synthesis_action": action,
            "source_proposal_refs": refs,
        }
        if action not in SYNTHESIS_ACTIONS:
            errors.append(
                _error(
                    "INVALID_SYNTHESIS_ACTION",
                    (
                        f"Candidato {index}: synthesis_action={action!r} "
                        "no está admitida."
                    ),
                    **context,
                    allowed_synthesis_actions=sorted(SYNTHESIS_ACTIONS),
                )
            )
        invalid_bases = sorted(set(bases) - MERGE_BASIS_CODES)
        duplicate_bases = sorted(
            value for value in set(bases) if bases.count(value) > 1
        )
        if invalid_bases or duplicate_bases:
            errors.append(
                _error(
                    "INVALID_MERGE_BASIS_CODES",
                    (
                        f"Candidato {index}: bases no admitidas "
                        f"{invalid_bases}; repetidas {duplicate_bases}."
                    ),
                    **context,
                    merge_basis_codes=bases,
                    allowed_merge_basis_codes=sorted(MERGE_BASIS_CODES),
                )
            )
        if action != "MERGED" and bases:
            errors.append(
                _error(
                    "MERGE_BASIS_WITHOUT_MERGE",
                    (
                        f"Candidato {index}: merge_basis_codes debe quedar "
                        f"vacío para {action!r}."
                    ),
                    **context,
                    merge_basis_codes=bases,
                )
            )
        if action == "ADDED" and refs:
            errors.append(
                _error(
                    "ADDED_EPISODE_HAS_SOURCE_REFS",
                    (
                        f"Candidato {index}: ADDED requiere cero "
                        "source_proposal_refs."
                    ),
                    **context,
                )
            )
        if action == "REVISED" and len(refs) != 1:
            errors.append(
                _error(
                    "INVALID_SINGLE_SOURCE_ACTION",
                    (
                        f"Candidato {index}: {action} requiere exactamente "
                        f"una referencia; recibió {len(refs)}."
                    ),
                    **context,
                )
            )
        if action == "MERGED":
            if len(refs) < 2:
                errors.append(
                    _error(
                        "MERGE_REQUIRES_MULTIPLE_PROPOSALS",
                        (
                            f"Candidato {index}: MERGED requiere al menos "
                            "dos referencias."
                        ),
                        **context,
                    )
                )
            if "SAME_OBSERVABLE_OPERATION" not in bases:
                errors.append(
                    _error(
                        "MERGE_LACKS_OPERATION_BASIS",
                        (
                            f"Candidato {index}: MERGED debe declarar "
                            "SAME_OBSERVABLE_OPERATION."
                        ),
                        **context,
                        merge_basis_codes=bases,
                    )
                )

        known_refs = [
            proposal_ref
            for proposal_ref in refs
            if proposal_ref in proposal_by_ref
        ]
        if action == "REVISED" and len(known_refs) == 1:
            proposal = proposal_by_ref[known_refs[0]]
            exact = canonical_json(
                _normalized_semantic_payload(proposal)
            ) == canonical_json(_normalized_semantic_payload(candidate))
            if exact:
                errors.append(
                    _error(
                        "REVISED_EPISODE_UNCHANGED",
                        (
                            f"Candidato {index}: REVISED es idéntico a "
                            f"{known_refs[0]}; muévalo a "
                            "preserved_proposal_refs."
                        ),
                        **context,
                        proposal_ref=known_refs[0],
                    )
                )

        for proposal_ref in known_refs:
            proposal = proposal_by_ref[proposal_ref]
            contribution = _member_contribution_stats(
                proposal,
                candidate,
            )
            if contribution["overlap_chars"] > 0:
                continue
            errors.append(
                _error(
                    "PROPOSAL_WITHOUT_LITERAL_CONTRIBUTION",
                    (
                        f"Candidato {index}: {proposal_ref} se declaró "
                        "contribuyente, pero ningún carácter de sus miembros "
                        "permanece en los miembros finales."
                    ),
                    **context,
                    proposal_ref=proposal_ref,
                    proposal_member_intervals=[
                        {
                            "start_char": start,
                            "end_char": end,
                        }
                        for start, end in _member_intervals(proposal)
                    ],
                    final_member_intervals=[
                        {
                            "start_char": start,
                            "end_char": end,
                        }
                        for start, end in _member_intervals(candidate)
                    ],
                    contribution=contribution,
                )
            )
    return errors


def _synthesized_episode_draft(
    value: SynthesizedEpisodeDraft,
) -> EpisodeDraft:
    return EpisodeDraft.model_validate(
        value.model_dump(
            exclude={
                "source_proposal_refs",
                "synthesis_action",
                "merge_basis_codes",
            }
        )
    )


def _preserved_proposal_candidate(
    proposal: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    candidate = deepcopy(_proposal_semantic_payload(proposal))
    candidate["provenance"] = deepcopy(provenance)
    return candidate


def _synthesize_type(
    client: LLMClient,
    *,
    model: str,
    sgot_graph: dict[str, Any],
    episode_type: str,
    proposals: list[dict[str, Any]],
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    semantic_audit_path: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    specification = EPISODE_SPECIFICATIONS[episode_type]
    trace_id = sgot_graph["trace_id"]
    trace_text = sgot_graph["trace_text"]
    synthesis_input = {
        **_graph_payload(sgot_graph),
        "local_proposals": [
            _proposal_prompt_payload(proposal)
            for proposal in proposals
        ],
    }
    base_prompt = SYNTHESIS_USER_TEMPLATE.format(
        episode_type=episode_type,
        type_specification=(
            specification["definition"]
            + "\nExclusion: "
            + specification["exclusion"]
        ),
        outcomes=", ".join(sorted(OUTCOMES_BY_TYPE[episode_type])),
        required_anchor=specification["required_anchor"],
        anchor_roles=", ".join(sorted(ANCHOR_ROLES)),
        ambiguity_codes=", ".join(sorted(AMBIGUITY_CODES)),
        synthesis_actions=", ".join(
            sorted(SYNTHESIS_ACTIONS - {"PRESERVED"})
        ),
        merge_basis_codes=", ".join(sorted(MERGE_BASIS_CODES)),
        removal_reason_codes=", ".join(
            sorted(REMOVAL_REASON_CODES)
        ),
        trace_id=prompt_trace_alias(trace_id),
        input_json=canonical_json(synthesis_input),
    )
    previous_output: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = []
    provenance_events: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    synthesis_audit: dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        user_prompt = base_prompt
        if previous_output is not None:
            user_prompt += SYNTHESIS_REPAIR_SUFFIX.format(
                validation_errors=_validation_errors_json(
                    validation_errors
                ),
                previous_output_json=canonical_json(previous_output),
            )
        prompt_hash = structured_request_hash(
            client,
            SYNTHESIS_SYSTEM_PROMPT,
            user_prompt,
        )
        if verbose:
            print(
                f"[{trace_id}] Síntesis global {episode_type} "
                f"(intento {attempt}/{max_attempts}).",
                file=sys.stderr,
                flush=True,
            )
        stage = f"06_episode_synthesis:{episode_type}"
        try:
            response = request_structured_response(
                client,
                model=model,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=EpisodeSynthesisBatchDraft,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            _persist_semantic_request_failure(
                semantic_audit_path,
                trace_id=trace_id,
                stage=stage,
                client=client,
                model=model,
                prompt_hash=prompt_hash,
                reasoning_effort=reasoning_effort,
                attempt=attempt,
                exception=exc,
                context={
                    "method": "hierarchical",
                    "phase": "global_synthesis",
                    "episode_type": episode_type,
                    "proposal_count": len(proposals),
                },
            )
            raise
        batch = response.parsed
        provenance = _response_provenance(
            stage=stage,
            client=client,
            model=model,
            prompt_hash=prompt_hash,
            reasoning_effort=reasoning_effort,
            response=response,
        )
        proposal_by_ref = {
            str(proposal["proposal_ref"]): proposal
            for proposal in proposals
        }
        synthesized_candidates = [
            _materialize_episode_candidate(
                _synthesized_episode_draft(episode),
                episode_type=episode_type,
                trace_text=trace_text,
                provenance=provenance,
            )
            for episode in batch.episodes
        ]
        preserved_candidates = [
            _preserved_proposal_candidate(
                proposal_by_ref[proposal_ref],
                provenance=provenance,
            )
            for proposal_ref in batch.preserved_proposal_refs
            if proposal_ref in proposal_by_ref
        ]
        candidates = [
            *synthesized_candidates,
            *preserved_candidates,
        ]
        validation_errors = _synthesis_reference_errors(
            batch,
            trace_id=trace_id,
            episode_type=episode_type,
            proposals=proposals,
            candidates=synthesized_candidates,
        )
        validation_errors.extend(
            _formal_candidate_errors(
                sgot_graph,
                candidates,
                provenance,
            )
        )
        event = _persist_semantic_attempt(
            semantic_audit_path,
            trace_id=trace_id,
            event={
                **provenance,
                "method": "hierarchical",
                "phase": "global_synthesis",
                "episode_type": episode_type,
                "attempt": attempt,
                "proposal_count": len(proposals),
                "candidate_count": len(candidates),
                "validation_status": (
                    "VALID" if not validation_errors else "INVALID"
                ),
                "validation_error_codes": [
                    error["code"] for error in validation_errors
                ],
            },
            validation_errors=validation_errors,
            response_payload=batch.model_dump(),
        )
        provenance_events.append(event)
        if not validation_errors:
            final_candidates = [
                {
                    "source_proposal_refs": sorted(
                        episode.source_proposal_refs
                    ),
                    "synthesis_action": episode.synthesis_action,
                    "merge_basis_codes": sorted(
                        episode.merge_basis_codes
                    ),
                    "candidate": candidate,
                }
                for episode, candidate in zip(
                    batch.episodes,
                    synthesized_candidates,
                )
            ]
            final_candidates.extend(
                {
                    "source_proposal_refs": [proposal_ref],
                    "synthesis_action": "PRESERVED",
                    "merge_basis_codes": [],
                    "candidate": candidate,
                }
                for proposal_ref, candidate in zip(
                    batch.preserved_proposal_refs,
                    preserved_candidates,
                )
            )
            synthesis_audit = {
                "episode_type": episode_type,
                "proposal_count": len(proposals),
                "proposal_registry": proposals,
                "final_candidate_count": len(candidates),
                "removed_proposals": sorted(
                    [
                        item.model_dump()
                        for item in batch.removed_proposals
                    ],
                    key=lambda item: item["proposal_ref"],
                ),
                "removed_proposal_refs": sorted(
                    item.proposal_ref
                    for item in batch.removed_proposals
                ),
                "final_candidates": [
                    {
                        "source_proposal_refs": decision[
                            "source_proposal_refs"
                        ],
                        "synthesis_action": decision[
                            "synthesis_action"
                        ],
                        "merge_basis_codes": decision[
                            "merge_basis_codes"
                        ],
                        "source_contributions": [
                            {
                                "proposal_ref": proposal_ref,
                                **_member_contribution_stats(
                                    proposal_by_ref[proposal_ref],
                                    candidate,
                                ),
                            }
                            for proposal_ref in decision[
                                "source_proposal_refs"
                            ]
                        ],
                        "episode": _proposal_semantic_payload(candidate),
                        "members": candidate["members"],
                        "anchors": candidate["anchors"],
                        "outcome": candidate["outcome"],
                    }
                    for decision in final_candidates
                    for candidate in [decision["candidate"]]
                ],
            }
            return candidates, provenance_events, synthesis_audit
        previous_output = batch.model_dump()

    details = "; ".join(
        f"{error['code']}: {error['message']}"
        for error in validation_errors[:4]
    )
    raise ValueError(
        f"La síntesis de {episode_type} agotó sus intentos: {details}"
    )


def infer_episodes_hierarchical(
    client: LLMClient,
    *,
    model: str,
    sgot_graph: dict[str, Any],
    reasoning_effort: str = "low",
    verbose: bool = False,
    max_attempts: int = 3,
    core_nodes: int = DEFAULT_BLOCK_CORE_NODES,
    halo_nodes: int = DEFAULT_BLOCK_HALO_NODES,
    shift_nodes: int = DEFAULT_BLOCK_SHIFT_NODES,
    semantic_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Estima episodios con propuestas locales y síntesis global por tipo."""

    core_nodes = _positive_integer(core_nodes, "core_nodes")
    halo_nodes = _nonnegative_integer(halo_nodes, "halo_nodes")
    shift_nodes = _nonnegative_integer(shift_nodes, "shift_nodes")
    if shift_nodes >= core_nodes:
        raise ValueError("shift_nodes debe ser menor que core_nodes.")
    if max_attempts <= 0:
        raise ValueError("max_attempts debe ser mayor que 0.")
    nodes = sgot_graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("SGoTGraph debe contener nodos.")

    if len(nodes) <= core_nodes:
        result = infer_episodes(
            client,
            model=model,
            sgot_graph=sgot_graph,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
            max_attempts=max_attempts,
            semantic_audit_path=semantic_audit_path,
        )
        result.setdefault("provenance", {}).setdefault("stages", []).append(
            {
                "origin": "deterministic",
                "stage": "06_hierarchical_global_fallback",
                "method": "hierarchical",
                "node_count": len(nodes),
                "core_nodes": core_nodes,
                "halo_nodes": halo_nodes,
                "shift_nodes": shift_nodes,
            }
        )
        result.setdefault("audit", []).append(
            {
                "stage": "06_hierarchical",
                "action": "global_fallback",
                "node_count": len(nodes),
                "core_nodes": core_nodes,
                "halo_nodes": halo_nodes,
                "shift_nodes": shift_nodes,
            }
        )
        return result

    blocks = build_episode_blocks(
        sgot_graph,
        core_nodes=core_nodes,
        halo_nodes=halo_nodes,
        shift_nodes=shift_nodes,
    )
    provenance_events: list[dict[str, Any]] = []
    synthesized_candidates: list[dict[str, Any]] = []
    synthesis_audits: list[dict[str, Any]] = []
    proposal_counts_by_type: dict[str, int] = {}

    for episode_type in sorted(EPISODE_TYPES):
        local_candidates: list[tuple[dict[str, Any], str]] = []
        for block in blocks:
            candidates, events = _infer_local_type_block(
                client,
                model=model,
                sgot_graph=sgot_graph,
                block=block,
                episode_type=episode_type,
                reasoning_effort=reasoning_effort,
                verbose=verbose,
                max_attempts=max_attempts,
                semantic_audit_path=semantic_audit_path,
            )
            provenance_events.extend(events)
            local_candidates.extend(
                (candidate, block["block_id"])
                for candidate in candidates
            )
        proposals = _consolidate_exact_proposals(local_candidates)
        proposal_counts_by_type[episode_type] = len(proposals)
        candidates, events, synthesis_audit = _synthesize_type(
            client,
            model=model,
            sgot_graph=sgot_graph,
            episode_type=episode_type,
            proposals=proposals,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
            max_attempts=max_attempts,
            semantic_audit_path=semantic_audit_path,
        )
        provenance_events.extend(events)
        synthesized_candidates.extend(candidates)
        synthesis_audits.append(synthesis_audit)

    default_provenance = {
        "origin": "llm_inferred",
        "stage": "06_episodes_hierarchical",
        "provider": client.provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    candidate_record = build_objective3_record_from_candidates(
        sgot_graph,
        synthesized_candidates,
        default_provenance=default_provenance,
        provenance_events=provenance_events,
        action="hierarchical_episode_candidates_synthesized",
    )
    (
        reconciled_episodes,
        reconciliation_provenance,
        reconciliation_audit,
    ) = _reconcile_episode_candidates(
        client,
        model=model,
        record=candidate_record,
        reasoning_effort=reasoning_effort,
        verbose=verbose,
        max_attempts=max_attempts,
        semantic_audit_path=semantic_audit_path,
    )
    result = build_objective3_record_from_candidates(
        sgot_graph,
        reconciled_episodes,
        default_provenance=default_provenance,
        provenance_events=[
            *provenance_events,
            *reconciliation_provenance,
        ],
        action="episodes_inferred_hierarchically_and_reconciled",
    )
    result["audit"].append(
        {
            "stage": "06_hierarchical",
            "action": "local_proposals_and_global_synthesis",
            "method": "hierarchical",
            "core_nodes": core_nodes,
            "halo_nodes": halo_nodes,
            "shift_nodes": shift_nodes,
            "alignment": blocks[0]["alignment"],
            "node_count": len(nodes),
            "block_count": len(blocks),
            "blocks": [
                {
                    key: block[key]
                    for key in (
                        "block_id",
                        "core_start_position",
                        "core_end_position",
                        "visible_start_position",
                        "visible_end_position",
                        "core_start_char",
                        "core_end_char",
                        "visible_start_char",
                        "visible_end_char",
                    )
                }
                for block in blocks
            ],
            "proposal_counts_by_type": proposal_counts_by_type,
            "synthesis": synthesis_audits,
        }
    )
    result["audit"].append(reconciliation_audit)
    return result


__all__ = [
    "DEFAULT_BLOCK_CORE_NODES",
    "DEFAULT_BLOCK_HALO_NODES",
    "DEFAULT_BLOCK_SHIFT_NODES",
    "EpisodeSynthesisBatchDraft",
    "RemovedProposalDraft",
    "SynthesizedEpisodeDraft",
    "build_episode_blocks",
    "infer_episodes_hierarchical",
]
