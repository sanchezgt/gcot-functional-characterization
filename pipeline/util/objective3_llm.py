"""Estimadores LLM para las etapas semánticas 06 y 07.

El módulo solicita decisiones estructuradas al modelo. Las restricciones del
contrato se verifican posteriormente en ``sgotgraph_objective3``; aquí no se
usan palabras clave, ventanas posicionales ni reglas que sustituyan una
decisión semántica del estimador.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from util.consensus import DEFAULT_MAX_EXTRA_SAMPLES, validate_sample_count
from util.trace_alias import prompt_trace_alias
from util.objective3_consensus import (
    DEFAULT_EPISODE_IOU_THRESHOLD,
    consensus_episodes,
    validate_iou_threshold,
)
from util.llm_provider import (
    DEFAULT_OPENAI_MODEL,
    LLMClient,
    request_structured_response,
    structured_request_hash,
)
from util.sgotgraph_objective3 import (
    AMBIGUITY_CODES,
    ANCHOR_ROLES,
    EPISODE_TYPES,
    FUNCTIONAL_RELATION_TYPES,
    OUTCOMES_BY_TYPE,
    add_episode_relations_from_candidates,
    build_objective3_record_from_candidates,
)
from util.objective3_semantic_audit import append_semantic_attempt


DEFAULT_OBJECTIVE3_MODEL = DEFAULT_OPENAI_MODEL


class SpanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_char: int
    end_char: int


class AnchorDraft(SpanDraft):
    model_config = ConfigDict(extra="forbid")

    anchor_role: str


class EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_spans: list[SpanDraft]
    witness_node_ids: list[str]
    witness_edge_ids: list[str]


class EpisodeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    members: list[SpanDraft]
    anchors: list[AnchorDraft]
    entry_anchor: Optional[AnchorDraft]
    exit_anchor: Optional[AnchorDraft]
    outcome: str
    ambiguity_codes: list[str]
    evidence: EvidenceDraft


class EpisodeBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    episode_type: str
    episodes: list[EpisodeDraft]


class EpisodeReconciliationDecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_ref: str
    decision: str


class RetainedEpisodeOverlapDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_episode_ref: str
    second_episode_ref: str
    distinction_basis: list[str]


class EpisodeReconciliationBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    decisions: list[EpisodeReconciliationDecisionDraft]
    retained_overlaps: list[RetainedEpisodeOverlapDraft]


class RelationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_type: str
    source_episode_ref: str
    target_episode_ref: str
    evidence: EvidenceDraft


class RelationBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    relations: list[RelationDraft]


EPISODE_SPECIFICATIONS = {
    "PROGRESSION": {
        "definition": (
            "Introduces or transforms a state, result, decision, or procedure "
            "belonging to a currently active line of advancement."
        ),
        "exclusion": (
            "Exclude content whose dominant function is only to evaluate, "
            "correct, support, or compete with that line, and exclude "
            "repetition of content without an observable advance."
        ),
        "required_anchor": "none",
    },
    "VERIFICATION": {
        "definition": (
            "Performs an observable evaluative operation on an identifiable "
            "statement, operation, method, or result. Re-deriving an already "
            "obtained result by an independent route, or checking it against "
            "a stated reference, and comparing the two is a verification even "
            "without explicit evaluative vocabulary. The check that reveals an "
            "error is a verification with outcome REFUTED in its own right, "
            "even when a CORRECTION follows and overlaps its end."
        ),
        "exclusion": (
            "Exclude a calculation or assertion merely because it is correct, "
            "as well as repetition, confidence statements, or corrections that "
            "do not contain an observable act of evaluation. Merely justifying "
            "why a rule, formula, or method applies is SUPPORT, not "
            "verification; it becomes verification only when the trace "
            "explicitly checks that setup or definition as an evaluative act "
            "(for example, double-checking a definition before computing). "
            "Evaluating or rejecting a competing option inside an ALTERNATIVE "
            "episode is that alternative's outcome, not a separate "
            "verification. Words such as 'confirmed', 'checked', or 'verified' "
            "that report events in the problem's own world (a test confirmed a "
            "diagnosis, a document confirms a decision) describe facts, not an "
            "evaluative operation of the trace. The TARGET marks "
            "what is evaluated; the members mark the evaluative operation."
        ),
        "required_anchor": "TARGET",
    },
    "CORRECTION": {
        "definition": (
            "Explicitly invalidates, replaces, restricts, or adjusts earlier "
            "content."
        ),
        "exclusion": (
            "Exclude doubt, monitoring, or verification without an observable "
            "modification of the evaluated content. Rejecting a never-adopted "
            "competing option is not by itself a correction, and neither is "
            "switching to an ALTERNATIVE path while leaving the earlier line "
            "unfinished: that is the alternative's outcome, unless the earlier "
            "content is explicitly declared wrong. The check that detects the "
            "error is a VERIFICATION; the correction begins where earlier "
            "content is invalidated or replaced and may overlap the end of "
            "that check, but it does not absorb the check. The TARGET must "
            "identify content that occurs before the observable revision."
        ),
        "required_anchor": "TARGET",
    },
    "SUPPORT": {
        "definition": (
            "Provides a reason, evidence, grounding, or constraint for other "
            "identifiable content."
        ),
        "exclusion": (
            "Exclude additional information that has no identifiable supported "
            "content or explicit justificatory function; repeating a claim; "
            "executing the next step of the solution; and asserting that "
            "something is correct without an observable ground. The "
            "supporting operation and its SUPPORTED content must remain "
            "distinguishable, and the relation must be one of justification, "
            "not of proximity, topical similarity, or continuation."
        ),
        "required_anchor": "SUPPORTED",
    },
    "ALTERNATIVE": {
        "definition": (
            "Introduces or develops a path that competes with an identifiable "
            "active line. The competing path may be introduced without an "
            "explicit marker such as 'alternatively': what matters is that a "
            "different route, method, or candidate toward the same objective "
            "is considered or developed."
        ),
        "exclusion": (
            "Exclude examples, reformulations, or digressions that do not "
            "establish a competing path, and exclude a step of the active line "
            "that is later found wrong and revised: that step belongs to the "
            "line and its revision is a CORRECTION, not a rejected alternative. "
            "Re-deriving a result that the line has already obtained, by "
            "another route, in order to compare with it is a VERIFICATION, not "
            "an alternative: an alternative is considered instead of the "
            "active line, before its result is settled. "
            "Evaluation or rejection may form part "
            "of an alternative's outcome, but does not automatically establish "
            "a separate VERIFICATION or CORRECTION episode; when the competing "
            "path is proposed and then rejected or adopted in the same passage, "
            "the episode extends through that rejection or adoption, which is "
            "its outcome. The CONTRAST anchor "
            "marks the step of the active line the alternative competes with: "
            "the step it departs from or, when the line resumes after the "
            "alternative, the step it resumes with. It lies outside the "
            "alternative's own proposal, it is a step of the solution rather "
            "than the shared problem facts, and it never marks the "
            "alternative itself."
        ),
        "required_anchor": "CONTRAST",
    },
}

RECONCILIATION_DECISIONS = {"KEEP", "REMOVE"}
RECONCILIATION_DISTINCTION_BASES = {
    "DISTINCT_REFERENT",
    "DISTINCT_ANCHOR",
    "DISTINCT_RELATION",
    "DISTINCT_OUTCOME",
}


EPISODE_SYSTEM_PROMPT = """You identify functional episodes in an automatically
generated reasoning trace.

You receive the complete trace and its SGoTGraph. Identify episodes only for
the requested functional type. The trace is an observable textual artifact;
do not infer hidden model states and do not judge truth against an external
answer.

Formal episode rules:
1. An episode is a higher-level discursive operation made of one or more exact
   text intervals that act on the same functional referent and contribute to
   one observable outcome.
2. Members may be discontinuous only when the same operation is interrupted,
   embedded, explicitly supported, or linked non-locally. Members of one
   episode must not overlap each other.
3. Separate episodes of the same type when the referent or anchor changes,
   when outcomes are independent, or when the later group is not necessary to
   interpret the earlier operation.
4. Episodes of different types may overlap only when each independently
   satisfies its definition and has a distinguishable referent, anchor,
   relation, or outcome.
5. Node types and graph edges are evidence signals, not equivalence rules. Do
   not reduce an episode to a keyword or force it to match one node.
6. Every member, anchor, and evidence span must contain exact 0-based,
   half-open character offsets into trace_text. The pipeline attaches the
   corresponding literal text deterministically after validating the bounds.
   A span boundary must never cut an alphanumeric token.
7. A specialized anchor identifies the content on which the operation acts; it
   is not a substitute for the operation itself. It may point outside or within
   the episode members, but the specialized anchors must not cover all member
   content. At least some member content must observably realize the functional
   operation. For CORRECTION, at least one TARGET must precede member content
   that realizes the revision. ENTRY and EXIT anchors represent observable
   boundary states and must be null when no such state is explicit.
8. Evidence must include at least one exact text span, projected node id, or
   projected edge id that supports the decision. A witness node must intersect
   an episode member. A witness edge must have at least one endpoint in that
   member projection and must be semantic rather than NEXT. A relevant
   contextual interval outside the members may be cited as an evidence text
   span. Never invent identifiers.
9. Return an empty episodes array when this functional type is absent.
10. Do not create missing content, repair the trace, or use task correctness.
11. For PROGRESSION, include exact STATE, ENTRY, or EXIT anchors whenever an
    observable state, objective, result, or procedure may establish continuity
    with another progression episode. Do not invent an anchor when it is not
    textually observable.
12. A correct operation, a selected answer, an option rejection, or a changed
    value does not by itself determine VERIFICATION, ALTERNATIVE, or CORRECTION.
    Apply the functional definition and exclusions for the requested type.
13. Boundaries. An episode begins at the first interval that unequivocally
    realizes its function; a preparatory or intention statement belongs to it
    only when it is functionally part of the operation. An episode ends at its
    explicit outcome, at the return to another dominant function, at the last
    interval that contributes to the same functional objective, or at the end
    of the trace when it remains unresolved.
14. Episodes do not partition the trace. Do not maximize coverage: content that
    does not realize the requested operation is left out. There is no expected
    number of episodes for a trace of any length; do not merge distinct
    operations to shorten the list, and do not split one operation to lengthen
    it. Two groups of the same type are separate episodes when they have
    different anchors, independent outcomes, an observable return to another
    function between them, or when neither is needed to interpret the other.
Return exactly one JSON object matching the requested schema.
"""


EPISODE_USER_TEMPLATE = """Identify all episodes of type {episode_type}.

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

Input:
{input_json}
"""

EPISODE_REPAIR_SUFFIX = """

The previous output did not satisfy the deterministic formal contract.
Return the complete corrected output for this same episode type. Preserve
semantic decisions that remain supported, but revise or remove any candidate
whose members, anchors, outcome, or evidence cannot satisfy the definitions.
Do not add a candidate merely to satisfy validation.

When present, candidate_index is 1-based and refers to the episodes array in
the previous output. candidate_indices identifies every previous candidate
that canonicalized to the same structural object. The listed
allowed_witness_node_ids and allowed_witness_edge_ids are the exhaustive
identifier sets permitted by the current member intervals; choose evidence
that semantically supports the episode from those sets, cite an exact text
span, or revise/remove the unsupported candidate. Do not expand members merely
to make an identifier pass validation.

Deterministic validation errors:
{validation_errors}

Previous output:
{previous_output_json}
"""


RECONCILIATION_SYSTEM_PROMPT = """You reconcile functional episode candidates
that were inferred independently by type from an automatically generated
reasoning trace.

The input lists every candidate and the pairs whose member intervals intersect
across different episode types. Those pairs were detected mechanically; overlap
does not prove either compatibility or incompatibility.

For every candidate involved in at least one listed pair, return exactly one
KEEP or REMOVE decision:
- KEEP only when the candidate independently satisfies its full type definition
  after all competing candidates and the complete trace are considered.
- REMOVE when the candidate is a duplicate interpretation, confuses an outcome
  with a separate operation, lacks its own observable operation, or otherwise
  fails its type definition.
- Do not impose a global priority among episode types and do not force a single
  label. Multiple overlapping candidates may remain only when the shared text
  genuinely performs each operation and the memberships are distinguishable by
  referent, anchor, functional relation, or outcome.

For every listed overlap pair whose two candidates are both kept, return exactly
one retained_overlaps entry. Use the pair's episode refs and one or more of these
codes to state the independent basis:
DISTINCT_REFERENT, DISTINCT_ANCHOR, DISTINCT_RELATION, DISTINCT_OUTCOME.
Do not return retained_overlaps entries for removed candidates or unlisted
pairs. Candidates not involved in a listed pair are already fixed and require
no decision. Do not alter spans, anchors, outcomes, evidence, or the trace.
Do not judge task correctness or infer hidden model states.
Return exactly one JSON object matching the requested schema.
"""


RECONCILIATION_USER_TEMPLATE = """Reconcile the cross-type episode overlaps.

The output must use trace_id={trace_id!r}.
Return one decision for each of these episode refs:
{conflict_episode_refs}

Allowed decisions:
{decisions}

Allowed distinction bases:
{distinction_bases}

Input:
{input_json}
"""


RECONCILIATION_REPAIR_SUFFIX = """

The previous reconciliation output did not satisfy the deterministic contract.
Return the complete corrected reconciliation for the same candidates. Preserve
supported semantic decisions, but fix missing, duplicate, unknown, or
incompatible decisions and overlap justifications. Do not change the episode
candidates and do not create decisions for non-conflicting candidates.

Deterministic validation errors:
{validation_errors}

Previous output:
{previous_output_json}
"""


RELATION_SYSTEM_PROMPT = """You infer functional relations between already fixed
functional episodes in an automatically generated reasoning trace.

All episodes and the complete SGoTGraph are visible. Each episode has a short,
deterministic episode_ref used only in this estimation request. Refer to
episodes exclusively through those refs. Inspect the whole set; no
positional window or lexical prefilter limits candidate pairs. Return every
functional relation that is explicitly justified and return no relation when
the evidence is insufficient.

Relation ontology and direction:
- CONTINUES: from a later episode to an earlier episode of the same type whose
  operation it continues; both preserve the same functional referent. For
  PROGRESSION, that referent must be witnessed by corresponding STATE, ENTRY,
  or EXIT anchors, or by a directed LEADS_TO chain from the earlier episode
  projection to the later projection.
- INTERRUPTS: from the interrupting episode to an episode that had begun and
  remained functionally open when the interruption started.
- RESUMES: from a later episode to an earlier episode of the same type whose
  operation it resumes after an intervening interruption or functional change.
  It uses the same referent-witness rule as CONTINUES.
- EMBEDS: from a functionally distinct episode contained within the envelope
  of its containing episode.
- EVALUATES: from VERIFICATION to the evaluated episode; the source TARGET
  anchor must link to the target.
- REPLACES: from CORRECTION to the displaced episode; the source TARGET anchor
  must link to the target.
- SUPPORTS_EPISODE: from SUPPORT to the supported episode; the source SUPPORTED
  anchor must link to the target.
- COMPETES_WITH: from ALTERNATIVE to PROGRESSION or ALTERNATIVE; the source
  CONTRAST anchor must link to the target.

An anchor links to a target when its exact interval intersects a target member
or equals one of the target's anchors. Proximity, topic similarity, shared
vocabulary, or textual order alone never establishes a functional relation.
Every relation requires an exact evidence interval, a node in the union of both
episode projections, or an edge that crosses between those projections. Span
outputs contain only exact 0-based, half-open offsets; the pipeline attaches
literal text deterministically. NEXT is sequential scaffolding and cannot be
used as a semantic witness. Span boundaries must never cut an alphanumeric
token. Do not output
IMMEDIATELY_PRECEDES_ONSET; the pipeline derives it deterministically. Do not
create self-relations or invent identifiers. Multiple relation types for a pair
are allowed only when each independently satisfies its definition.
Return exactly one JSON object matching the requested schema.
"""


RELATION_USER_TEMPLATE = """Infer all justified functional relations.

The output must use trace_id={trace_id!r}.
Allowed relation types:
{relation_types}

Input:
{input_json}
"""

RELATION_REPAIR_SUFFIX = """

The previous output did not satisfy the deterministic formal contract.
Return the complete corrected relation set. Preserve semantic decisions that
remain supported, but revise or remove any relation that violates its
definition, direction, anchor linkage, or evidence requirements. Do not add a
relation merely to satisfy validation.

Deterministic validation errors:
{validation_errors}

Previous output:
{previous_output_json}
"""


def _graph_payload(sgot_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        # Alias opaco: el prompt no recibe el identificador real, que
        # revela dataset y generador.
        "trace_id": prompt_trace_alias(sgot_graph["trace_id"]),
        "trace_text": sgot_graph["trace_text"],
        "nodes": [
            {
                "segment_id": node["segment_id"],
                "node_type": node["node_type"],
                "start_char": node["start_char"],
                "end_char": node["end_char"],
                "text": node["text"],
            }
            for node in sgot_graph.get("nodes", [])
        ],
        "edges": [
            {
                "edge_id": edge["edge_id"],
                "source": edge["source"],
                "target": edge["target"],
                "edge_type": edge["edge_type"],
            }
            for edge in sgot_graph.get("edges", [])
        ],
    }


def _materialize_span_text(
    span: dict[str, Any],
    trace_text: str,
) -> dict[str, Any]:
    value = dict(span)
    start = value.get("start_char")
    end = value.get("end_char")
    value["text"] = (
        trace_text[start:end]
        if isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(trace_text)
        else None
    )
    return value


def _materialize_evidence_text(
    evidence: dict[str, Any],
    trace_text: str,
) -> dict[str, Any]:
    value = dict(evidence)
    value["text_spans"] = [
        _materialize_span_text(span, trace_text)
        for span in value.get("text_spans", [])
    ]
    return value


def _materialize_episode_candidate(
    episode: EpisodeDraft,
    *,
    episode_type: str,
    trace_text: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    value = episode.model_dump()
    value["episode_type"] = episode_type
    value["members"] = [
        _materialize_span_text(span, trace_text)
        for span in value.get("members", [])
    ]
    value["anchors"] = [
        _materialize_span_text(anchor, trace_text)
        for anchor in value.get("anchors", [])
    ]
    for field in ("entry_anchor", "exit_anchor"):
        if isinstance(value.get(field), dict):
            value[field] = _materialize_span_text(value[field], trace_text)
    value["evidence"] = _materialize_evidence_text(
        value.get("evidence", {}),
        trace_text,
    )
    value["provenance"] = provenance
    return value


def _materialize_relation_candidate(
    relation: RelationDraft,
    *,
    trace_text: str,
    provenance: dict[str, Any],
    episode_id_by_ref: dict[str, str],
) -> dict[str, Any]:
    value = relation.model_dump()
    source_ref = value.pop("source_episode_ref")
    target_ref = value.pop("target_episode_ref")
    value["source_episode_id"] = episode_id_by_ref.get(
        source_ref,
        source_ref,
    )
    value["target_episode_id"] = episode_id_by_ref.get(
        target_ref,
        target_ref,
    )
    value["evidence"] = _materialize_evidence_text(
        value.get("evidence", {}),
        trace_text,
    )
    value["provenance"] = provenance
    return value


def _relation_episode_payload(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    payload: list[dict[str, Any]] = []
    episode_id_by_ref: dict[str, str] = {}
    episode_ref_by_id: dict[str, str] = {}
    for index, episode in enumerate(episodes, start=1):
        episode_ref = f"E{index:04d}"
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Cada episodio debe tener episode_id canónico.")
        episode_id_by_ref[episode_ref] = episode_id
        episode_ref_by_id[episode_id] = episode_ref
        payload.append(
            {
                "episode_ref": episode_ref,
                "episode_type": episode.get("episode_type"),
                "members": episode.get("members"),
                "anchors": episode.get("anchors"),
                "entry_anchor": episode.get("entry_anchor"),
                "exit_anchor": episode.get("exit_anchor"),
                "outcome": episode.get("outcome"),
                "ambiguity_codes": episode.get("ambiguity_codes"),
                "evidence": episode.get("evidence"),
            }
        )
    return payload, episode_id_by_ref, episode_ref_by_id


def _valid_interval(value: Any) -> tuple[int, int] | None:
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
    return (start, end)


def _candidate_projection_context(
    sgot_graph: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_index: int,
) -> dict[str, Any]:
    member_intervals = [
        interval
        for interval in (
            _valid_interval(span)
            for span in candidate.get("members", [])
        )
        if interval is not None
    ]
    allowed_node_ids = {
        str(node.get("segment_id"))
        for node in sgot_graph.get("nodes", [])
        if isinstance(node, dict)
        and isinstance(node.get("segment_id"), str)
        and (
            node_interval := _valid_interval(node)
        )
        is not None
        and any(
            max(node_interval[0], member_start)
            < min(node_interval[1], member_end)
            for member_start, member_end in member_intervals
        )
    }
    allowed_edge_ids = {
        str(edge.get("edge_id"))
        for edge in sgot_graph.get("edges", [])
        if isinstance(edge, dict)
        and isinstance(edge.get("edge_id"), str)
        and edge.get("edge_type") != "NEXT"
        and (
            edge.get("source") in allowed_node_ids
            or edge.get("target") in allowed_node_ids
        )
    }
    evidence = candidate.get("evidence", {})
    witness_node_ids = (
        evidence.get("witness_node_ids", [])
        if isinstance(evidence, dict)
        else []
    )
    witness_edge_ids = (
        evidence.get("witness_edge_ids", [])
        if isinstance(evidence, dict)
        else []
    )
    return {
        "candidate_index": candidate_index,
        "candidate_type": candidate.get("episode_type"),
        "member_intervals": [
            {"start_char": start, "end_char": end}
            for start, end in member_intervals
        ],
        "allowed_witness_node_ids": sorted(allowed_node_ids),
        "allowed_witness_edge_ids": sorted(allowed_edge_ids),
        "invalid_witness_node_ids": sorted(
            {
                str(node_id)
                for node_id in witness_node_ids
                if str(node_id) not in allowed_node_ids
            }
        ),
        "invalid_witness_edge_ids": sorted(
            {
                str(edge_id)
                for edge_id in witness_edge_ids
                if str(edge_id) not in allowed_edge_ids
            }
        ),
    }


def _episode_candidate_contract_errors(
    sgot_graph: dict[str, Any],
    candidates: list[dict[str, Any]],
    provenance: dict[str, Any],
    *,
    action: str,
) -> list[dict[str, Any]]:
    """Valida candidatos y traduce IDs canónicos a índices reparables."""

    tagged_candidates = []
    source_ids: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        source_id = f"candidate[{index}]"
        source_ids.append(source_id)
        tagged_candidates.append({**candidate, "episode_id": source_id})
    try:
        probe = build_objective3_record_from_candidates(
            sgot_graph,
            tagged_candidates,
            default_provenance=provenance,
            provenance_events=[provenance],
            action=action,
        )
    except ValueError as exc:
        individual_failures: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            try:
                build_objective3_record_from_candidates(
                    sgot_graph,
                    [{**candidate, "episode_id": f"candidate[{index}]"}],
                    default_provenance=provenance,
                    provenance_events=[provenance],
                    action=f"{action}_individual_probe",
                )
            except ValueError as individual_exc:
                individual_failures.append(
                    {
                        "code": "CANDIDATE_CANONICALIZATION_ERROR",
                        "message": str(individual_exc),
                        **_candidate_projection_context(
                            sgot_graph,
                            candidate,
                            candidate_index=index,
                        ),
                    }
                )
        if individual_failures:
            return individual_failures

        structural_groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates, start=1):
            identity = json.dumps(
                {
                    "episode_type": candidate.get("episode_type"),
                    "members": candidate.get("members"),
                    "anchors": candidate.get("anchors"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            structural_groups.setdefault(identity, []).append(index)
        conflicting_groups = [
            indices
            for indices in structural_groups.values()
            if len(indices) > 1
        ]
        conflicting_indices = sorted(
            {
                index
                for indices in conflicting_groups
                for index in indices
            }
        )
        value: dict[str, Any] = {
            "code": "CANDIDATE_CANONICALIZATION_ERROR",
            "message": str(exc),
            "candidate_count": len(candidates),
        }
        if conflicting_indices:
            value["candidate_indices"] = conflicting_indices
            value["conflicting_structural_groups"] = conflicting_groups
            value["candidate_contexts"] = [
                _candidate_projection_context(
                    sgot_graph,
                    candidates[index - 1],
                    candidate_index=index,
                )
                for index in conflicting_indices
            ]
        return [value]

    source_id_map = {}
    for audit_event in probe.get("audit", []):
        if isinstance(audit_event, dict) and isinstance(
            audit_event.get("source_episode_id_map"),
            dict,
        ):
            source_id_map.update(audit_event["source_episode_id_map"])
    indices_by_object_id: dict[str, list[int]] = {}
    for index, source_id in enumerate(source_ids, start=1):
        object_id = source_id_map.get(source_id)
        if isinstance(object_id, str):
            indices_by_object_id.setdefault(object_id, []).append(index)

    feedback: list[dict[str, Any]] = []
    for error in probe["validation"]["episodes"]["errors"]:
        value = dict(error)
        object_id = error.get("object_id")
        candidate_indices = indices_by_object_id.get(
            object_id,
            [],
        )
        if candidate_indices:
            contexts = [
                _candidate_projection_context(
                    sgot_graph,
                    candidates[index - 1],
                    candidate_index=index,
                )
                for index in candidate_indices
            ]
            value["candidate_indices"] = candidate_indices
            if len(contexts) == 1:
                value.update(contexts[0])
            else:
                value["candidate_contexts"] = contexts
        feedback.append(value)
    return feedback


def _persist_semantic_attempt(
    semantic_audit_path: Path | None,
    *,
    trace_id: str,
    event: dict[str, Any],
    validation_errors: list[dict[str, Any]],
    response_payload: Any | None,
) -> dict[str, Any]:
    journal_event = {
        **event,
        "validation_errors": [
            dict(error) for error in validation_errors
        ],
    }
    attempt_id = append_semantic_attempt(
        semantic_audit_path,
        trace_id=trace_id,
        event=journal_event,
        response_payload=response_payload,
    )
    return {
        **event,
        **(
            {
                "semantic_attempt_id": attempt_id,
                "semantic_attempt_journaled": True,
            }
            if attempt_id is not None
            else {}
        ),
    }


def _persist_semantic_request_failure(
    semantic_audit_path: Path | None,
    *,
    trace_id: str,
    stage: str,
    client: LLMClient,
    model: str,
    prompt_hash: str,
    reasoning_effort: str,
    attempt: int,
    exception: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    error = {
        "code": "SEMANTIC_REQUEST_FAILED",
        "message": str(exception),
        "exception_type": type(exception).__name__,
    }
    event = {
        "origin": "llm_request_failed",
        "stage": stage,
        "provider": client.provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "reasoning_effort": reasoning_effort,
        "attempt": attempt,
        "validation_status": "REQUEST_FAILED",
        "validation_error_codes": [error["code"]],
        **(context or {}),
    }
    _persist_semantic_attempt(
        semantic_audit_path,
        trace_id=trace_id,
        event=event,
        validation_errors=[error],
        response_payload=None,
    )


def _episode_members_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_intervals = [
        interval
        for interval in (
            _valid_interval(span) for span in left.get("members", [])
        )
        if interval is not None
    ]
    right_intervals = [
        interval
        for interval in (
            _valid_interval(span) for span in right.get("members", [])
        )
        if interval is not None
    ]
    return any(
        max(left_start, right_start) < min(left_end, right_end)
        for left_start, left_end in left_intervals
        for right_start, right_end in right_intervals
    )


def _cross_type_overlap_pairs(
    episode_payload: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(episode_payload):
        for right in episode_payload[index + 1 :]:
            if left.get("episode_type") == right.get("episode_type"):
                continue
            if not _episode_members_overlap(left, right):
                continue
            pairs.append(
                (
                    str(left.get("episode_ref")),
                    str(right.get("episode_ref")),
                )
            )
    return pairs


def _reconciliation_error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    object_id: str | None = None,
) -> None:
    value: dict[str, Any] = {"code": code, "message": message}
    if object_id is not None:
        value["object_id"] = object_id
    errors.append(value)


def _validate_reconciliation_batch(
    batch: EpisodeReconciliationBatchDraft,
    *,
    trace_id: str,
    conflict_refs: list[str],
    overlap_pairs: list[tuple[str, str]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[tuple[str, str], list[str]],
]:
    errors: list[dict[str, Any]] = []
    if batch.trace_id != prompt_trace_alias(trace_id):
        _reconciliation_error(
            errors,
            "TRACE_ID_MISMATCH",
            (
                f"trace_id={batch.trace_id!r}; se esperaba "
                f"{prompt_trace_alias(trace_id)!r}."
            ),
        )

    expected_refs = set(conflict_refs)
    ref_rank = {episode_ref: index for index, episode_ref in enumerate(conflict_refs)}
    decisions: dict[str, str] = {}
    for item in batch.decisions:
        episode_ref = item.episode_ref
        if episode_ref not in expected_refs:
            _reconciliation_error(
                errors,
                "UNKNOWN_RECONCILIATION_EPISODE_REF",
                f"Referencia no conflictiva o desconocida: {episode_ref!r}.",
                episode_ref,
            )
            continue
        if episode_ref in decisions:
            _reconciliation_error(
                errors,
                "DUPLICATE_RECONCILIATION_DECISION",
                f"La referencia {episode_ref!r} tiene más de una decisión.",
                episode_ref,
            )
            continue
        decisions[episode_ref] = item.decision
        if item.decision not in RECONCILIATION_DECISIONS:
            _reconciliation_error(
                errors,
                "INVALID_RECONCILIATION_DECISION",
                (
                    f"Decisión inválida para {episode_ref!r}: "
                    f"{item.decision!r}."
                ),
                episode_ref,
            )

    missing_refs = [
        episode_ref for episode_ref in conflict_refs if episode_ref not in decisions
    ]
    if missing_refs:
        _reconciliation_error(
            errors,
            "MISSING_RECONCILIATION_DECISIONS",
            f"Faltan decisiones para: {missing_refs}.",
        )

    expected_pairs = set(overlap_pairs)

    def canonical_pair(first: str, second: str) -> tuple[str, str]:
        if ref_rank.get(first, 10**9) <= ref_rank.get(second, 10**9):
            return (first, second)
        return (second, first)

    retained: dict[tuple[str, str], list[str]] = {}
    for item in batch.retained_overlaps:
        pair = canonical_pair(item.first_episode_ref, item.second_episode_ref)
        pair_id = f"{pair[0]}|{pair[1]}"
        if item.first_episode_ref == item.second_episode_ref:
            _reconciliation_error(
                errors,
                "SELF_RETAINED_OVERLAP",
                "Un solapamiento retenido requiere dos referencias distintas.",
                pair_id,
            )
            continue
        if pair not in expected_pairs:
            _reconciliation_error(
                errors,
                "UNKNOWN_RETAINED_OVERLAP",
                f"El par {pair_id} no figura entre los solapamientos de entrada.",
                pair_id,
            )
            continue
        if pair in retained:
            _reconciliation_error(
                errors,
                "DUPLICATE_RETAINED_OVERLAP",
                f"El par {pair_id} aparece más de una vez.",
                pair_id,
            )
            continue
        bases = sorted(set(item.distinction_basis))
        invalid_bases = sorted(
            set(bases) - RECONCILIATION_DISTINCTION_BASES
        )
        if not bases or invalid_bases:
            _reconciliation_error(
                errors,
                "INVALID_OVERLAP_DISTINCTION_BASIS",
                (
                    f"El par {pair_id} requiere al menos una base admitida; "
                    f"valores no admitidos: {invalid_bases}."
                ),
                pair_id,
            )
            continue
        if not all(decisions.get(ref) == "KEEP" for ref in pair):
            _reconciliation_error(
                errors,
                "JUSTIFICATION_FOR_REMOVED_OVERLAP",
                (
                    f"El par {pair_id} solo puede justificarse cuando ambas "
                    "decisiones son KEEP."
                ),
                pair_id,
            )
            continue
        retained[pair] = bases

    for pair in overlap_pairs:
        if (
            all(decisions.get(episode_ref) == "KEEP" for episode_ref in pair)
            and pair not in retained
        ):
            pair_id = f"{pair[0]}|{pair[1]}"
            _reconciliation_error(
                errors,
                "MISSING_RETAINED_OVERLAP_JUSTIFICATION",
                (
                    f"El par retenido {pair_id} requiere una base de "
                    "distinción explícita."
                ),
                pair_id,
            )

    return errors, decisions, retained


def _relation_validation_feedback(
    result: dict[str, Any],
    errors: list[dict[str, Any]],
    episode_ref_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    relation_by_id = {
        relation.get("relation_id"): relation
        for relation in result.get("episode_relations", [])
        if isinstance(relation, dict)
    }
    feedback: list[dict[str, Any]] = []
    for error in errors:
        value = dict(error)
        relation = relation_by_id.get(error.get("object_id"))
        if isinstance(relation, dict):
            value["relation"] = {
                "relation_type": relation.get("relation_type"),
                "source_episode_ref": episode_ref_by_id.get(
                    relation.get("source_episode_id"),
                    relation.get("source_episode_id"),
                ),
                "target_episode_ref": episode_ref_by_id.get(
                    relation.get("target_episode_id"),
                    relation.get("target_episode_id"),
                ),
            }
        feedback.append(value)
    return feedback


def _validation_errors_json(errors: list[dict[str, Any]]) -> str:
    return json.dumps(errors, ensure_ascii=False, indent=2)


def _response_provenance(
    *,
    stage: str,
    client: LLMClient,
    model: str,
    prompt_hash: str,
    reasoning_effort: str,
    response: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "origin": "llm_inferred",
        "stage": stage,
        "provider": client.provider,
        "model": response.model or model,
        "prompt_hash": prompt_hash,
        "reasoning_effort": reasoning_effort,
    }
    if response.response_id:
        value["response_id"] = response.response_id
    if response.system_fingerprint:
        value["system_fingerprint"] = response.system_fingerprint
    return value


def _reconcile_episode_candidates(
    client: LLMClient,
    *,
    model: str,
    record: dict[str, Any],
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    semantic_audit_path: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    trace_id = record["trace_id"]
    episode_payload, episode_id_by_ref, _ = _relation_episode_payload(
        record.get("episodes", [])
    )
    overlap_pairs = _cross_type_overlap_pairs(episode_payload)
    conflict_ref_set = {
        episode_ref for pair in overlap_pairs for episode_ref in pair
    }
    conflict_refs = [
        episode["episode_ref"]
        for episode in episode_payload
        if episode["episode_ref"] in conflict_ref_set
    ]

    if not overlap_pairs:
        provenance = {
            "origin": "deterministic",
            "stage": "06_episode_reconciliation",
            "candidate_count": len(episode_payload),
            "conflict_episode_count": 0,
            "cross_type_overlap_pair_count": 0,
            "status": "not_required",
        }
        audit = {
            "stage": "06_episode_reconciliation",
            "action": "cross_type_reconciliation_not_required",
            "candidate_count": len(episode_payload),
            "conflict_episode_count": 0,
            "cross_type_overlap_pair_count": 0,
            "removed_episode_ids": [],
            "mixed_function_episode_ids": [],
            "retained_overlap_justifications": [],
        }
        return list(record.get("episodes", [])), [provenance], audit

    reconciliation_input = {
        **_graph_payload(record["sgot_graph"]),
        "episode_type_specifications": {
            episode_type: {
                **EPISODE_SPECIFICATIONS[episode_type],
                "allowed_outcomes": sorted(
                    OUTCOMES_BY_TYPE[episode_type]
                ),
            }
            for episode_type in sorted(EPISODE_TYPES)
        },
        "episodes": episode_payload,
        "cross_type_member_overlap_pairs": [
            {
                "first_episode_ref": first,
                "second_episode_ref": second,
            }
            for first, second in overlap_pairs
        ],
    }
    base_prompt = RECONCILIATION_USER_TEMPLATE.format(
        trace_id=prompt_trace_alias(trace_id),
        conflict_episode_refs=", ".join(conflict_refs),
        decisions=", ".join(sorted(RECONCILIATION_DECISIONS)),
        distinction_bases=", ".join(
            sorted(RECONCILIATION_DISTINCTION_BASES)
        ),
        input_json=json.dumps(
            reconciliation_input,
            ensure_ascii=False,
            indent=2,
        ),
    )
    previous_output: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = []
    provenance_events: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    retained: dict[tuple[str, str], list[str]] = {}

    for attempt in range(1, max_attempts + 1):
        user_prompt = base_prompt
        if previous_output is not None:
            user_prompt += RECONCILIATION_REPAIR_SUFFIX.format(
                validation_errors=_validation_errors_json(validation_errors),
                previous_output_json=json.dumps(
                    previous_output,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        prompt_hash = structured_request_hash(
            client,
            RECONCILIATION_SYSTEM_PROMPT,
            user_prompt,
        )
        if verbose:
            print(
                f"[{trace_id}] Reconciliando {len(overlap_pairs)} "
                f"solapamientos entre tipos (intento "
                f"{attempt}/{max_attempts}) con "
                f"{client.provider}:{model}.",
                file=sys.stderr,
                flush=True,
            )
        try:
            response = request_structured_response(
                client,
                model=model,
                system_prompt=RECONCILIATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=EpisodeReconciliationBatchDraft,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            _persist_semantic_request_failure(
                semantic_audit_path,
                trace_id=trace_id,
                stage="06_episode_reconciliation",
                client=client,
                model=model,
                prompt_hash=prompt_hash,
                reasoning_effort=reasoning_effort,
                attempt=attempt,
                exception=exc,
                context={
                    "candidate_count": len(episode_payload),
                    "conflict_episode_count": len(conflict_refs),
                    "cross_type_overlap_pair_count": len(overlap_pairs),
                },
            )
            raise
        batch = response.parsed
        validation_errors, decisions, retained = (
            _validate_reconciliation_batch(
                batch,
                trace_id=trace_id,
                conflict_refs=conflict_refs,
                overlap_pairs=overlap_pairs,
            )
        )
        provenance = _response_provenance(
            stage="06_episode_reconciliation",
            client=client,
            model=model,
            prompt_hash=prompt_hash,
            reasoning_effort=reasoning_effort,
            response=response,
        )
        event = _persist_semantic_attempt(
            semantic_audit_path,
            trace_id=trace_id,
            event={
                **provenance,
                "attempt": attempt,
                "candidate_count": len(episode_payload),
                "conflict_episode_count": len(conflict_refs),
                "cross_type_overlap_pair_count": len(overlap_pairs),
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
            break
        previous_output = batch.model_dump()

    if validation_errors:
        details = "; ".join(
            f"{error['code']}: {error['message']}"
            for error in validation_errors[:4]
        )
        raise ValueError(
            "La reconciliación de episodios agotó sus intentos sin "
            f"satisfacer el contrato: {details}"
        )

    kept_refs = {
        episode["episode_ref"]
        for episode in episode_payload
        if episode["episode_ref"] not in conflict_ref_set
        or decisions.get(episode["episode_ref"]) == "KEEP"
    }
    mixed_refs = {
        episode_ref
        for pair in retained
        for episode_ref in pair
    }
    episodes_by_id = {
        episode["episode_id"]: episode
        for episode in record.get("episodes", [])
    }
    kept_episodes: list[dict[str, Any]] = []
    for episode_ref, episode_id in episode_id_by_ref.items():
        if episode_ref not in kept_refs:
            continue
        episode = episodes_by_id[episode_id]
        if episode_ref in mixed_refs:
            episode = {
                **episode,
                "ambiguity_codes": sorted(
                    {
                        *episode.get("ambiguity_codes", []),
                        "MIXED_FUNCTION",
                    }
                ),
            }
        kept_episodes.append(episode)

    decision_audit = [
        {
            "episode_ref": episode_ref,
            "episode_id": episode_id_by_ref[episode_ref],
            "decision": decisions[episode_ref],
        }
        for episode_ref in conflict_refs
    ]
    retained_audit = [
        {
            "first_episode_ref": pair[0],
            "first_episode_id": episode_id_by_ref[pair[0]],
            "second_episode_ref": pair[1],
            "second_episode_id": episode_id_by_ref[pair[1]],
            "distinction_basis": bases,
        }
        for pair, bases in sorted(retained.items())
    ]
    removed_episode_ids = [
        episode_id_by_ref[episode_ref]
        for episode_ref in conflict_refs
        if decisions[episode_ref] == "REMOVE"
    ]
    mixed_episode_ids = sorted(
        episode_id_by_ref[episode_ref] for episode_ref in mixed_refs
    )
    audit = {
        "stage": "06_episode_reconciliation",
        "action": "cross_type_overlaps_reconciled",
        "candidate_count": len(episode_payload),
        "conflict_episode_count": len(conflict_refs),
        "cross_type_overlap_pair_count": len(overlap_pairs),
        "decisions": decision_audit,
        "removed_episode_ids": removed_episode_ids,
        "mixed_function_episode_ids": mixed_episode_ids,
        "retained_overlap_justifications": retained_audit,
    }
    return kept_episodes, provenance_events, audit


def _sample_episode_type(
    client: LLMClient,
    *,
    model: str,
    sgot_graph: dict[str, Any],
    trace_id: str,
    trace_text: str,
    episode_type: str,
    base_prompt: str,
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    semantic_audit_path: Path | None,
    sample_index: int,
    provenance_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Una muestra válida del estimador de un tipo de episodio.

    Una salida que incumple el contrato se devuelve al LLM con sus errores
    hasta ``max_attempts`` veces; el código nunca repara contenido semántico.
    """

    previous_output: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        user_prompt = base_prompt
        if previous_output is not None:
            user_prompt += EPISODE_REPAIR_SUFFIX.format(
                validation_errors=_validation_errors_json(
                    validation_errors
                ),
                previous_output_json=json.dumps(
                    previous_output,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        prompt_hash = structured_request_hash(
            client,
            EPISODE_SYSTEM_PROMPT,
            user_prompt,
        )
        if verbose:
            print(
                f"[{trace_id}] Infiriendo episodios {episode_type} "
                f"(muestra {sample_index}, intento {attempt}/{max_attempts}) con "
                f"{client.provider}:{model}.",
                file=sys.stderr,
                flush=True,
            )
        stage = f"06_episodes:{episode_type}"
        try:
            response = request_structured_response(
                client,
                model=model,
                system_prompt=EPISODE_SYSTEM_PROMPT,
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
                context={"episode_type": episode_type, "sample": sample_index},
            )
            raise
        batch = response.parsed
        validation_errors = []
        if batch.trace_id != prompt_trace_alias(trace_id):
            validation_errors.append(
                {
                    "code": "TRACE_ID_MISMATCH",
                    "message": (
                        f"trace_id={batch.trace_id!r}; se esperaba "
                        f"{prompt_trace_alias(trace_id)!r}."
                    ),
                }
            )
        if batch.episode_type != episode_type:
            validation_errors.append(
                {
                    "code": "EPISODE_TYPE_MISMATCH",
                    "message": (
                        f"episode_type={batch.episode_type!r}; se "
                        f"esperaba {episode_type!r}."
                    ),
                }
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
            _episode_candidate_contract_errors(
                sgot_graph,
                candidates,
                provenance,
                action="episode_batch_contract_check",
            )
        )
        validation_status = (
            "VALID" if not validation_errors else "INVALID"
        )
        event = _persist_semantic_attempt(
            semantic_audit_path,
            trace_id=trace_id,
            event={
                **provenance,
                "episode_type": episode_type,
                "sample": sample_index,
                "attempt": attempt,
                "candidate_count": len(candidates),
                "validation_status": validation_status,
                "validation_error_codes": [
                    error["code"] for error in validation_errors
                ],
            },
            validation_errors=validation_errors,
            response_payload=batch.model_dump(),
        )
        provenance_events.append(event)
        if validation_status == "VALID":
            break
        previous_output = batch.model_dump()

    if validation_errors:
        details = "; ".join(
            f"{error['code']}: {error['message']}"
            for error in validation_errors[:4]
        )
        raise ValueError(
            f"La identificación de {episode_type} agotó sus intentos: "
            f"{details}"
        )
    return candidates


def infer_episodes(
    client: LLMClient,
    *,
    model: str,
    sgot_graph: dict[str, Any],
    reasoning_effort: str = "low",
    verbose: bool = False,
    max_attempts: int = 3,
    semantic_audit_path: Path | None = None,
    samples: int = 1,
    consensus_iou_threshold: float = DEFAULT_EPISODE_IOU_THRESHOLD,
    max_extra_samples: int = DEFAULT_MAX_EXTRA_SAMPLES,
) -> dict[str, Any]:
    """Identifica episodios por tipo; con ``samples > 1`` aplica consenso.

    Cada tipo se muestrea ``samples`` veces y se conservan los episodios
    presentes en una mayoría estricta de las muestras (emparejados por IoU de
    miembros >= ``consensus_iou_threshold``). Con ``samples=1`` se reproduce el
    estimador de una sola muestra. La reconciliación entre tipos se ejecuta
    después sobre el conjunto resultante.
    """

    trace_id = sgot_graph.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("SGoTGraph no contiene trace_id.")
    trace_text = sgot_graph.get("trace_text")
    if not isinstance(trace_text, str):
        raise ValueError("SGoTGraph no contiene trace_text válido.")
    if max_attempts <= 0:
        raise ValueError("max_attempts debe ser mayor que 0.")
    validate_sample_count(samples)
    validate_iou_threshold(consensus_iou_threshold)
    if max_extra_samples < 0:
        raise ValueError("max_extra_samples no puede ser negativo.")
    input_json = json.dumps(
        _graph_payload(sgot_graph), ensure_ascii=False, indent=2
    )
    raw_episodes: list[dict[str, Any]] = []
    provenance_events: list[dict[str, Any]] = []
    consensus_audit: list[dict[str, Any]] = []

    for episode_type in sorted(EPISODE_TYPES):
        specification = EPISODE_SPECIFICATIONS[episode_type]
        base_prompt = EPISODE_USER_TEMPLATE.format(
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
            input_json=input_json,
        )
        sample_candidates: list[list[dict[str, Any]]] = []
        failed_samples: list[dict[str, Any]] = []
        sample_index = 0
        while len(sample_candidates) < samples:
            sample_index += 1
            try:
                sample_candidates.append(
                    _sample_episode_type(
                        client,
                        model=model,
                        sgot_graph=sgot_graph,
                        trace_id=trace_id,
                        trace_text=trace_text,
                        episode_type=episode_type,
                        base_prompt=base_prompt,
                        reasoning_effort=reasoning_effort,
                        verbose=verbose,
                        max_attempts=max_attempts,
                        semantic_audit_path=semantic_audit_path,
                        sample_index=sample_index,
                        provenance_events=provenance_events,
                    )
                )
            except ValueError as exc:
                # Con consenso, una muestra que agota sus intentos se sustituye
                # por otra independiente (con tope); con una sola muestra se
                # conserva la política de cierre. Nada se repara ni completa.
                if samples == 1 or len(failed_samples) >= max_extra_samples:
                    raise
                failed_samples.append({"sample": sample_index, "error": str(exc)})
                if verbose:
                    print(
                        f"[{trace_id}] Muestra {sample_index} de {episode_type} "
                        f"descartada tras agotar intentos; se toma otra muestra.",
                        file=sys.stderr,
                        flush=True,
                    )
        if samples == 1:
            raw_episodes.extend(sample_candidates[0])
            continue
        aggregated = consensus_episodes(
            sample_candidates,
            iou_threshold=consensus_iou_threshold,
        )
        raw_episodes.extend(aggregated["episodes"])
        consensus_audit.append(
            {
                "episode_type": episode_type,
                "sample_draws": sample_index,
                "failed_samples": failed_samples,
                **aggregated["audit"],
            }
        )

    default_provenance = {
        "origin": "llm_inferred",
        "stage": "06_episodes",
        "provider": client.provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    candidate_record = build_objective3_record_from_candidates(
        sgot_graph,
        raw_episodes,
        default_provenance=default_provenance,
        provenance_events=provenance_events,
        action="episode_candidates_inferred_by_type",
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
        action="episodes_inferred_and_reconciled",
    )
    if samples > 1:
        result["audit"].append(
            {
                "stage": "06_episode_consensus",
                "action": "episode_candidates_aggregated_by_majority",
                "samples": samples,
                "iou_threshold": float(consensus_iou_threshold),
                "by_type": consensus_audit,
            }
        )
    result["audit"].append(reconciliation_audit)
    return result


def infer_episode_relations(
    client: LLMClient,
    *,
    model: str,
    record: dict[str, Any],
    reasoning_effort: str = "low",
    verbose: bool = False,
    max_attempts: int = 3,
    semantic_audit_path: Path | None = None,
) -> dict[str, Any]:
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("El registro no contiene trace_id.")
    trace_text = record.get("sgot_graph", {}).get("trace_text")
    if not isinstance(trace_text, str):
        raise ValueError("El registro no contiene trace_text válido.")
    if max_attempts <= 0:
        raise ValueError("max_attempts debe ser mayor que 0.")
    episode_payload, episode_id_by_ref, episode_ref_by_id = (
        _relation_episode_payload(record.get("episodes", []))
    )
    payload = {
        **_graph_payload(record["sgot_graph"]),
        "episodes": episode_payload,
    }
    base_prompt = RELATION_USER_TEMPLATE.format(
        trace_id=prompt_trace_alias(trace_id),
        relation_types=", ".join(sorted(FUNCTIONAL_RELATION_TYPES)),
        input_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    previous_output: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = []
    provenance_events: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        user_prompt = base_prompt
        if previous_output is not None:
            user_prompt += RELATION_REPAIR_SUFFIX.format(
                validation_errors=_validation_errors_json(
                    validation_errors
                ),
                previous_output_json=json.dumps(
                    previous_output,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        prompt_hash = structured_request_hash(
            client,
            RELATION_SYSTEM_PROMPT,
            user_prompt,
        )
        if verbose:
            print(
                f"[{trace_id}] Infiriendo relaciones funcionales "
                f"(intento {attempt}/{max_attempts}) con "
                f"{client.provider}:{model}.",
                file=sys.stderr,
                flush=True,
            )
        try:
            response = request_structured_response(
                client,
                model=model,
                system_prompt=RELATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=RelationBatchDraft,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            _persist_semantic_request_failure(
                semantic_audit_path,
                trace_id=trace_id,
                stage="07_episode_relations",
                client=client,
                model=model,
                prompt_hash=prompt_hash,
                reasoning_effort=reasoning_effort,
                attempt=attempt,
                exception=exc,
            )
            raise
        batch = response.parsed
        validation_errors = []
        if batch.trace_id != prompt_trace_alias(trace_id):
            validation_errors.append(
                {
                    "code": "TRACE_ID_MISMATCH",
                    "message": (
                        f"trace_id={batch.trace_id!r}; se esperaba "
                        f"{prompt_trace_alias(trace_id)!r}."
                    ),
                }
            )
        provenance = _response_provenance(
            stage="07_episode_relations",
            client=client,
            model=model,
            prompt_hash=prompt_hash,
            reasoning_effort=reasoning_effort,
            response=response,
        )
        raw_relations = [
            _materialize_relation_candidate(
                relation,
                trace_text=trace_text,
                provenance=provenance,
                episode_id_by_ref=episode_id_by_ref,
            )
            for relation in batch.relations
        ]
        try:
            attempt_result = add_episode_relations_from_candidates(
                record,
                raw_relations,
                default_provenance=provenance,
                provenance_events=[],
                action="functional_relations_contract_check",
            )
        except ValueError as exc:
            attempt_result = None
            validation_errors.append(
                {
                    "code": "RELATION_CANONICALIZATION_ERROR",
                    "message": str(exc),
                }
            )
        if attempt_result is not None:
            validation = attempt_result["validation"]["relations"]
            validation_errors.extend(
                _relation_validation_feedback(
                    attempt_result,
                    validation["errors"],
                    episode_ref_by_id,
                )
            )
        validation_status = (
            "VALID" if not validation_errors else "INVALID"
        )
        event = _persist_semantic_attempt(
            semantic_audit_path,
            trace_id=trace_id,
            event={
                **provenance,
                "attempt": attempt,
                "candidate_count": len(raw_relations),
                "validation_status": validation_status,
                "validation_error_codes": [
                    error["code"] for error in validation_errors
                ],
            },
            validation_errors=validation_errors,
            response_payload=batch.model_dump(),
        )
        provenance_events.append(event)
        if validation_status == "VALID":
            result = attempt_result
            break
        previous_output = batch.model_dump()

    if validation_errors or result is None:
        details = "; ".join(
            f"{error['code']}: {error['message']}"
            for error in validation_errors[:4]
        )
        raise ValueError(
            "La inferencia de relaciones agotó sus intentos: "
            f"{details}"
        )
    result["audit"][-1]["action"] = "functional_relations_inferred"
    result["provenance"]["stages"].extend(provenance_events)
    return result


__all__ = [
    "DEFAULT_OBJECTIVE3_MODEL",
    "EpisodeBatchDraft",
    "EpisodeDraft",
    "EpisodeReconciliationBatchDraft",
    "EpisodeReconciliationDecisionDraft",
    "RelationBatchDraft",
    "RelationDraft",
    "RetainedEpisodeOverlapDraft",
    "infer_episode_relations",
    "infer_episodes",
]
