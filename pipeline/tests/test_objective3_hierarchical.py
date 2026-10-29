from __future__ import annotations

from copy import deepcopy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.sgotgraph_schema import (
    PIPELINE_VERSION as OBJECTIVE2_PIPELINE_VERSION,
    SCHEMA_VERSION as OBJECTIVE2_SCHEMA_VERSION,
)
from util.trace_alias import prompt_trace_alias
from util.llm_provider import StructuredResponse
from util import objective3_hierarchical
from util.objective3_hierarchical import (
    EpisodeSynthesisBatchDraft,
    RemovedProposalDraft,
    SynthesizedEpisodeDraft,
    build_episode_blocks,
    infer_episodes_hierarchical,
)
from util.objective3_llm import (
    EPISODE_SYSTEM_PROMPT,
    EpisodeBatchDraft,
    EpisodeDraft,
    EvidenceDraft,
    SpanDraft,
)
from util.sgotgraph_schema import make_provenance, validate_record


def synthetic_graph(node_count: int = 7) -> dict:
    pieces = [f"Step {index:02d}. " for index in range(1, node_count + 1)]
    trace_text = "".join(pieces)
    provenance = make_provenance("deterministic", "synthetic_test")
    segments = []
    nodes = []
    cursor = 0
    for index, piece in enumerate(pieces, start=1):
        segment = {
            "segment_id": f"s{index}",
            "text": piece,
            "start_char": cursor,
            "end_char": cursor + len(piece),
            "provenance": provenance,
        }
        segments.append(segment)
        nodes.append(
            {
                **segment,
                "position": index,
                "node_type": "OUT" if index == node_count else "PROP",
            }
        )
        cursor += len(piece)
    edges = [
        {
            "edge_id": f"e_next_{index}",
            "source": f"s{index}",
            "target": f"s{index + 1}",
            "edge_type": "NEXT",
            "provenance": make_provenance(
                "deterministic",
                "synthetic_backbone",
            ),
        }
        for index in range(1, node_count)
    ]
    graph = {
        "schema_version": OBJECTIVE2_SCHEMA_VERSION,
        "pipeline_version": OBJECTIVE2_PIPELINE_VERSION,
        "trace_id": "hierarchical_synthetic_case",
        "source_record": {"problem_id": "synthetic"},
        "trace_text": trace_text,
        "segments": segments,
        "nodes": nodes,
        "edges": edges,
        "validation": {},
        "audit": {},
    }
    assert validate_record(graph)["status"] == "valid"
    return graph


def episode_draft_for_node(graph: dict, position: int) -> EpisodeDraft:
    node = graph["nodes"][position - 1]
    member = SpanDraft(
        start_char=node["start_char"],
        end_char=node["end_char"],
    )
    return EpisodeDraft(
        members=[member],
        anchors=[],
        entry_anchor=None,
        exit_anchor=None,
        outcome="ADVANCES",
        ambiguity_codes=[],
        evidence=EvidenceDraft(
            text_spans=[member],
            witness_node_ids=[node["segment_id"]],
            witness_edge_ids=[],
        ),
    )


def prompt_episode_type(user_prompt: str) -> str:
    match = re.search(r"episode_type='([^']+)'", user_prompt)
    if match is None:
        raise AssertionError("El prompt no contiene episode_type.")
    return match.group(1)


def structured_response(parsed) -> StructuredResponse:
    return StructuredResponse(
        parsed=parsed,
        provider="mock",
        model="mock-model",
        response_id="mock-response",
    )


class EpisodeBlockTests(unittest.TestCase):
    def test_local_and_synthesis_prompts_reuse_all_global_rules(
        self,
    ) -> None:
        rules = EPISODE_SYSTEM_PROMPT.split(
            "Formal episode rules:",
            maxsplit=1,
        )[1]
        for required_text in (
            "11. For PROGRESSION",
            "12. A correct operation",
        ):
            self.assertIn(required_text, rules)
            self.assertIn(
                required_text,
                objective3_hierarchical.LOCAL_EPISODE_SYSTEM_PROMPT,
            )
            self.assertIn(
                required_text,
                objective3_hierarchical.SYNTHESIS_SYSTEM_PROMPT,
            )

    def test_blocks_cover_each_core_node_once_and_preserve_context(
        self,
    ) -> None:
        graph = synthetic_graph(7)
        blocks = build_episode_blocks(
            graph,
            core_nodes=3,
            halo_nodes=1,
            shift_nodes=0,
        )

        self.assertEqual(
            [
                (
                    block["core_start_position"],
                    block["core_end_position"],
                    block["visible_start_position"],
                    block["visible_end_position"],
                )
                for block in blocks
            ],
            [(1, 3, 1, 4), (4, 6, 3, 7), (7, 7, 6, 7)],
        )
        owned_positions = [
            position
            for block in blocks
            for position in range(
                block["core_start_position"],
                block["core_end_position"] + 1,
            )
        ]
        self.assertEqual(owned_positions, list(range(1, 8)))
        for block in blocks:
            self.assertEqual(
                block["visible_trace_text"],
                graph["trace_text"][
                    block["visible_start_char"]:block["visible_end_char"]
                ],
            )
            visible_ids = {
                node["segment_id"] for node in block["nodes"]
            }
            self.assertTrue(
                all(
                    edge["source"] in visible_ids
                    and edge["target"] in visible_ids
                    for edge in block["edges"]
                )
            )

    def test_shifted_boundaries_preserve_exact_unique_ownership(self) -> None:
        graph = synthetic_graph(7)
        start0 = build_episode_blocks(
            graph,
            core_nodes=3,
            halo_nodes=1,
            shift_nodes=0,
        )
        shifted = build_episode_blocks(
            graph,
            core_nodes=3,
            halo_nodes=1,
            shift_nodes=2,
        )

        self.assertEqual(
            [
                (
                    block["core_start_position"],
                    block["core_end_position"],
                )
                for block in shifted
            ],
            [(1, 2), (3, 5), (6, 7)],
        )
        self.assertNotEqual(
            [
                (
                    block["core_start_position"],
                    block["core_end_position"],
                )
                for block in start0
            ],
            [
                (
                    block["core_start_position"],
                    block["core_end_position"],
                )
                for block in shifted
            ],
        )
        self.assertEqual(
            [
                position
                for block in shifted
                for position in range(
                    block["core_start_position"],
                    block["core_end_position"] + 1,
                )
            ],
            list(range(1, 8)),
        )

    def test_invalid_block_parameters_are_rejected(self) -> None:
        graph = synthetic_graph(3)
        invalid_parameters = [
            {"core_nodes": 0, "halo_nodes": 1, "shift_nodes": 0},
            {"core_nodes": 2, "halo_nodes": -1, "shift_nodes": 0},
            {"core_nodes": 2, "halo_nodes": 1, "shift_nodes": -1},
            {"core_nodes": 2, "halo_nodes": 1, "shift_nodes": 2},
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    build_episode_blocks(graph, **parameters)


class HierarchicalInferenceTests(unittest.TestCase):
    def test_short_graph_uses_exact_global_route(self) -> None:
        graph = synthetic_graph(2)
        global_result = {
            "episodes": [{"episode_id": "unchanged"}],
            "provenance": {"stages": []},
            "audit": [],
        }
        client = SimpleNamespace(provider="mock")

        with patch.object(
            objective3_hierarchical,
            "infer_episodes",
            return_value=deepcopy(global_result),
        ) as mocked_global:
            result = infer_episodes_hierarchical(
                client,
                model="mock-model",
                sgot_graph=graph,
                core_nodes=2,
                halo_nodes=1,
                shift_nodes=0,
            )

        mocked_global.assert_called_once()
        self.assertEqual(result["episodes"], global_result["episodes"])
        self.assertEqual(
            result["audit"][-1]["action"],
            "global_fallback",
        )
        self.assertEqual(
            result["provenance"]["stages"][-1]["stage"],
            "06_hierarchical_global_fallback",
        )

    def test_hierarchical_orchestration_is_valid_and_traceable(self) -> None:
        graph = synthetic_graph(5)
        request_models: list[type] = []

        def fake_request(
            _client,
            *,
            model,
            system_prompt,
            user_prompt,
            response_model,
            reasoning_effort,
        ):
            del model, system_prompt, reasoning_effort
            request_models.append(response_model)
            episode_type = prompt_episode_type(user_prompt)
            if response_model is EpisodeBatchDraft:
                episodes = []
                if (
                    episode_type == "PROGRESSION"
                    and '"block_id":"start0_b001_' in user_prompt
                ):
                    episodes = [episode_draft_for_node(graph, 1)]
                return structured_response(
                    EpisodeBatchDraft(
                        trace_id=prompt_trace_alias(graph["trace_id"]),
                        episode_type=episode_type,
                        episodes=episodes,
                    )
                )
            if response_model is EpisodeSynthesisBatchDraft:
                return structured_response(
                    EpisodeSynthesisBatchDraft(
                        trace_id=prompt_trace_alias(graph["trace_id"]),
                        episode_type=episode_type,
                        preserved_proposal_refs=(
                            ["P0001"]
                            if episode_type == "PROGRESSION"
                            else []
                        ),
                        episodes=[],
                        removed_proposals=[],
                    )
                )
            raise AssertionError(f"Modelo de respuesta inesperado: {response_model}")

        with patch.object(
            objective3_hierarchical,
            "request_structured_response",
            side_effect=fake_request,
        ):
            result = infer_episodes_hierarchical(
                SimpleNamespace(provider="mock"),
                model="mock-model",
                sgot_graph=graph,
                max_attempts=2,
                core_nodes=2,
                halo_nodes=1,
                shift_nodes=0,
            )

        self.assertEqual(
            request_models.count(EpisodeBatchDraft),
            15,
        )
        self.assertEqual(
            request_models.count(EpisodeSynthesisBatchDraft),
            5,
        )
        self.assertEqual(
            result["validation"]["episodes"]["status"],
            "VALID",
        )
        self.assertEqual(len(result["episodes"]), 1)
        hierarchical_audit = next(
            item
            for item in result["audit"]
            if item.get("stage") == "06_hierarchical"
        )
        self.assertEqual(hierarchical_audit["block_count"], 3)
        progression_audit = next(
            item
            for item in hierarchical_audit["synthesis"]
            if item["episode_type"] == "PROGRESSION"
        )
        proposal = progression_audit["proposal_registry"][0]
        self.assertEqual(proposal["proposal_ref"], "P0001")
        self.assertEqual(
            proposal["source_block_ids"],
            ["start0_b001_p0001-0002"],
        )
        self.assertEqual(
            proposal["source_inferences"][0]["provenance"]["model"],
            "mock-model",
        )
        self.assertEqual(
            progression_audit["final_candidates"][0][
                "source_proposal_refs"
            ],
            ["P0001"],
        )
        self.assertEqual(
            progression_audit["final_candidates"][0][
                "synthesis_action"
            ],
            "PRESERVED",
        )
        self.assertEqual(
            progression_audit["final_candidates"][0][
                "source_contributions"
            ][0]["proposal_overlap_ratio"],
            1.0,
        )

    def test_local_ownership_violation_is_repaired(self) -> None:
        graph = synthetic_graph(5)
        block = build_episode_blocks(
            graph,
            core_nodes=2,
            halo_nodes=1,
            shift_nodes=0,
        )[1]
        prompts: list[str] = []
        invalid = EpisodeBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            episodes=[episode_draft_for_node(graph, 2)],
        )
        repaired = EpisodeBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            episodes=[],
        )

        def fake_request(_client, **kwargs):
            prompts.append(kwargs["user_prompt"])
            parsed = invalid if len(prompts) == 1 else repaired
            return structured_response(parsed)

        with patch.object(
            objective3_hierarchical,
            "request_structured_response",
            side_effect=fake_request,
        ):
            candidates, events = (
                objective3_hierarchical._infer_local_type_block(
                    SimpleNamespace(provider="mock"),
                    model="mock-model",
                    sgot_graph=graph,
                    block=block,
                    episode_type="PROGRESSION",
                    reasoning_effort="low",
                    verbose=False,
                    max_attempts=2,
                )
            )

        self.assertEqual(candidates, [])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["validation_status"], "INVALID")
        self.assertIn(
            "MEMBER_START_OUTSIDE_OWNERSHIP_CORE",
            prompts[1],
        )

    def test_synthesis_reference_omission_is_repaired(self) -> None:
        graph = synthetic_graph(3)
        candidate = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
        }
        candidate["members"][0]["text"] = graph["nodes"][0]["text"]
        candidate["evidence"]["text_spans"][0]["text"] = graph["nodes"][0][
            "text"
        ]
        candidate["provenance"] = {
            "origin": "llm_inferred",
            "stage": "test",
            "provider": "mock",
            "model": "mock-model",
        }
        proposals = objective3_hierarchical._consolidate_exact_proposals(
            [(candidate, "start0_b001_p0001-0002")]
        )
        prompts: list[str] = []
        missing = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=[],
            episodes=[],
            removed_proposals=[],
        )
        repaired = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=[],
            episodes=[],
            removed_proposals=[
                RemovedProposalDraft(
                    proposal_ref="P0001",
                    reason_code="NOT_SUPPORTED_IN_FULL_CONTEXT",
                )
            ],
        )

        def fake_request(_client, **kwargs):
            prompts.append(kwargs["user_prompt"])
            parsed = missing if len(prompts) == 1 else repaired
            return structured_response(parsed)

        with patch.object(
            objective3_hierarchical,
            "request_structured_response",
            side_effect=fake_request,
        ):
            candidates, events, audit = (
                objective3_hierarchical._synthesize_type(
                    SimpleNamespace(provider="mock"),
                    model="mock-model",
                    sgot_graph=graph,
                    episode_type="PROGRESSION",
                    proposals=proposals,
                    reasoning_effort="low",
                    verbose=False,
                    max_attempts=2,
                )
            )

        self.assertEqual(candidates, [])
        self.assertEqual(len(events), 2)
        self.assertEqual(audit["removed_proposal_refs"], ["P0001"])
        self.assertIn("UNACCOUNTED_PROPOSAL_REF", prompts[1])

    def test_preserved_proposal_is_materialized_literally_by_reference(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        members = [
            SpanDraft(
                start_char=node["start_char"],
                end_char=node["end_char"],
            )
            for node in graph["nodes"]
        ]
        source_draft = EpisodeDraft(
            members=members,
            anchors=[],
            entry_anchor=None,
            exit_anchor=None,
            outcome="ADVANCES",
            ambiguity_codes=[],
            evidence=EvidenceDraft(
                text_spans=[
                    SpanDraft(
                        start_char=graph["nodes"][0]["start_char"],
                        end_char=graph["nodes"][-1]["end_char"],
                    )
                ],
                witness_node_ids=["s1", "s2", "s3"],
                witness_edge_ids=[],
            ),
        )
        source_candidate = (
            objective3_hierarchical._materialize_episode_candidate(
                source_draft,
                episode_type="PROGRESSION",
                trace_text=graph["trace_text"],
                provenance={
                    "origin": "llm_inferred",
                    "stage": "local",
                    "provider": "mock",
                    "model": "mock-model",
                },
            )
        )
        proposals = objective3_hierarchical._consolidate_exact_proposals(
            [(source_candidate, "start0_b001_p0001-0003")]
        )
        response_batch = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=["P0001"],
            episodes=[],
            removed_proposals=[],
        )

        with patch.object(
            objective3_hierarchical,
            "request_structured_response",
            return_value=structured_response(response_batch),
        ):
            candidates, events, audit = (
                objective3_hierarchical._synthesize_type(
                    SimpleNamespace(provider="mock"),
                    model="mock-model",
                    sgot_graph=graph,
                    episode_type="PROGRESSION",
                    proposals=proposals,
                    reasoning_effort="low",
                    verbose=False,
                    max_attempts=1,
                )
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            objective3_hierarchical._proposal_semantic_payload(
                candidates[0]
            ),
            objective3_hierarchical._proposal_semantic_payload(
                proposals[0]
            ),
        )
        self.assertEqual(
            candidates[0]["evidence"]["text_spans"],
            proposals[0]["evidence"]["text_spans"],
        )
        self.assertEqual(events[0]["validation_status"], "VALID")
        self.assertEqual(
            audit["final_candidates"][0]["synthesis_action"],
            "PRESERVED",
        )
        self.assertEqual(
            audit["final_candidates"][0]["source_proposal_refs"],
            ["P0001"],
        )

    def test_preserved_is_not_a_full_episode_action(self) -> None:
        graph = synthetic_graph(1)
        with self.assertRaises(ValueError):
            SynthesizedEpisodeDraft(
                **episode_draft_for_node(graph, 1).model_dump(),
                source_proposal_refs=["P0001"],
                synthesis_action="PRESERVED",
                merge_basis_codes=[],
            )

    def test_formal_feedback_identifies_candidate_and_allowed_witnesses(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        candidate = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
        }
        candidate["members"][0]["text"] = graph["nodes"][0]["text"]
        candidate["evidence"]["text_spans"][0]["text"] = graph["nodes"][0][
            "text"
        ]
        candidate["evidence"]["witness_node_ids"] = ["s2"]
        provenance = {
            "origin": "llm_inferred",
            "stage": "test",
            "provider": "mock",
            "model": "mock-model",
        }
        candidate["provenance"] = provenance

        errors = objective3_hierarchical._formal_candidate_errors(
            graph,
            [candidate],
            provenance,
        )

        witness_error = next(
            error
            for error in errors
            if error["code"] == "WITNESS_NODE_OUTSIDE_PROJECTION"
        )
        self.assertEqual(witness_error["candidate_index"], 1)
        self.assertEqual(witness_error["candidate_indices"], [1])
        self.assertEqual(
            witness_error["allowed_witness_node_ids"],
            ["s1"],
        )
        self.assertEqual(
            witness_error["invalid_witness_node_ids"],
            ["s2"],
        )
        self.assertEqual(
            witness_error["member_intervals"],
            [
                {
                    "start_char": graph["nodes"][0]["start_char"],
                    "end_char": graph["nodes"][0]["end_char"],
                }
            ],
        )

    def test_canonicalization_conflict_identifies_all_prior_candidates(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        base = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
        }
        base["members"][0]["text"] = graph["nodes"][0]["text"]
        base["evidence"]["text_spans"][0]["text"] = graph["nodes"][0][
            "text"
        ]
        provenance = {
            "origin": "llm_inferred",
            "stage": "test",
            "provider": "mock",
            "model": "mock-model",
        }
        first = {**deepcopy(base), "provenance": provenance}
        second = {**deepcopy(base), "provenance": provenance}
        second["outcome"] = "INTERRUPTED"

        errors = objective3_hierarchical._formal_candidate_errors(
            graph,
            [first, second],
            provenance,
        )

        self.assertEqual(len(errors), 1)
        self.assertEqual(
            errors[0]["code"],
            "CANDIDATE_CANONICALIZATION_ERROR",
        )
        self.assertEqual(errors[0]["candidate_indices"], [1, 2])
        self.assertEqual(
            errors[0]["conflicting_structural_groups"],
            [[1, 2]],
        )

    def test_used_proposal_requires_literal_member_contribution(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        proposal_candidate = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
        }
        proposal_candidate["members"][0]["text"] = graph["nodes"][0][
            "text"
        ]
        proposal_candidate["evidence"]["text_spans"][0]["text"] = (
            graph["nodes"][0]["text"]
        )
        proposals = [
            {
                "proposal_ref": "P0001",
                **proposal_candidate,
                "source_block_ids": ["start0_b001_p0001-0002"],
                "source_inferences": [],
            }
        ]
        final_draft = episode_draft_for_node(graph, 2)
        synthesized = SynthesizedEpisodeDraft(
            **final_draft.model_dump(),
            source_proposal_refs=["P0001"],
            synthesis_action="REVISED",
            merge_basis_codes=[],
        )
        batch = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=[],
            episodes=[synthesized],
            removed_proposals=[],
        )
        provenance = {
            "origin": "llm_inferred",
            "stage": "test",
            "provider": "mock",
            "model": "mock-model",
        }
        candidate = objective3_hierarchical._materialize_episode_candidate(
            objective3_hierarchical._synthesized_episode_draft(synthesized),
            episode_type="PROGRESSION",
            trace_text=graph["trace_text"],
            provenance=provenance,
        )

        errors = objective3_hierarchical._synthesis_reference_errors(
            batch,
            trace_id=graph["trace_id"],
            episode_type="PROGRESSION",
            proposals=proposals,
            candidates=[candidate],
        )

        contribution_error = next(
            error
            for error in errors
            if error["code"]
            == "PROPOSAL_WITHOUT_LITERAL_CONTRIBUTION"
        )
        self.assertEqual(contribution_error["candidate_index"], 1)
        self.assertEqual(contribution_error["proposal_ref"], "P0001")

    def test_explicit_merge_accepts_two_literal_contributions(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        proposals = []
        members = []
        for position in (1, 2):
            draft = episode_draft_for_node(graph, position)
            proposals.append(
                {
                    "proposal_ref": f"P{position:04d}",
                    "episode_type": "PROGRESSION",
                    **draft.model_dump(),
                    "source_block_ids": [
                        f"start0_b00{position}_p0001-0002"
                    ],
                    "source_inferences": [],
                }
            )
            members.extend(draft.members)
        merged = SynthesizedEpisodeDraft(
            members=members,
            anchors=[],
            entry_anchor=None,
            exit_anchor=None,
            outcome="ADVANCES",
            ambiguity_codes=[],
            evidence=EvidenceDraft(
                text_spans=members,
                witness_node_ids=["s1", "s2"],
                witness_edge_ids=[],
            ),
            source_proposal_refs=["P0001", "P0002"],
            synthesis_action="MERGED",
            merge_basis_codes=[
                "SAME_OBSERVABLE_OPERATION",
                "COMPLEMENTARY_FRAGMENTS",
            ],
        )
        batch = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=[],
            episodes=[merged],
            removed_proposals=[],
        )
        provenance = {
            "origin": "llm_inferred",
            "stage": "test",
            "provider": "mock",
            "model": "mock-model",
        }
        candidate = objective3_hierarchical._materialize_episode_candidate(
            objective3_hierarchical._synthesized_episode_draft(merged),
            episode_type="PROGRESSION",
            trace_text=graph["trace_text"],
            provenance=provenance,
        )

        errors = objective3_hierarchical._synthesis_reference_errors(
            batch,
            trace_id=graph["trace_id"],
            episode_type="PROGRESSION",
            proposals=proposals,
            candidates=[candidate],
        )

        self.assertEqual(errors, [])

    def test_failed_synthesis_keeps_every_attempt_in_durable_journal(
        self,
    ) -> None:
        graph = synthetic_graph(3)
        local_candidate = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
        }
        local_candidate["members"][0]["text"] = graph["nodes"][0]["text"]
        local_candidate["evidence"]["text_spans"][0]["text"] = (
            graph["nodes"][0]["text"]
        )
        local_candidate["provenance"] = {
            "origin": "llm_inferred",
            "stage": "local",
            "provider": "mock",
            "model": "mock-model",
            "prompt_hash": "sha256:local",
        }
        proposals = objective3_hierarchical._consolidate_exact_proposals(
            [(local_candidate, "start0_b001_p0001-0002")]
        )
        final_draft = episode_draft_for_node(graph, 2)
        invalid_batch = EpisodeSynthesisBatchDraft(
            trace_id=prompt_trace_alias(graph["trace_id"]),
            episode_type="PROGRESSION",
            preserved_proposal_refs=[],
            episodes=[
                SynthesizedEpisodeDraft(
                    **final_draft.model_dump(),
                    source_proposal_refs=["P0001"],
                    synthesis_action="REVISED",
                    merge_basis_codes=[],
                )
            ],
            removed_proposals=[],
        )
        prompts: list[str] = []

        def fake_request(_client, **kwargs):
            prompts.append(kwargs["user_prompt"])
            return structured_response(invalid_batch)

        with tempfile.TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "attempts.jsonl"
            with patch.object(
                objective3_hierarchical,
                "request_structured_response",
                side_effect=fake_request,
            ):
                with self.assertRaises(ValueError):
                    objective3_hierarchical._synthesize_type(
                        SimpleNamespace(provider="mock"),
                        model="mock-model",
                        sgot_graph=graph,
                        episode_type="PROGRESSION",
                        proposals=proposals,
                        reasoning_effort="low",
                        verbose=False,
                        max_attempts=2,
                        semantic_audit_path=journal,
                    )

            records = [
                json.loads(line)
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line
            ]

        self.assertEqual(len(prompts), 2)
        self.assertNotIn('"source_inferences"', prompts[0])
        self.assertIn(
            "PROPOSAL_WITHOUT_LITERAL_CONTRIBUTION",
            prompts[1],
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(
            all(
                record["event"]["validation_status"] == "INVALID"
                for record in records
            )
        )
        self.assertTrue(
            all(
                "PROPOSAL_WITHOUT_LITERAL_CONTRIBUTION"
                in record["event"]["validation_error_codes"]
                for record in records
            )
        )
        self.assertTrue(
            all(record["response_payload"] is not None for record in records)
        )

    def test_only_exact_duplicates_are_consolidated(self) -> None:
        graph = synthetic_graph(3)
        base_candidate = {
            "episode_type": "PROGRESSION",
            **episode_draft_for_node(graph, 1).model_dump(),
            "provenance": {
                "origin": "llm_inferred",
                "stage": "call_one",
                "provider": "mock",
                "model": "mock-model",
                "prompt_hash": "sha256:one",
            },
        }
        duplicate = deepcopy(base_candidate)
        duplicate["provenance"]["stage"] = "call_two"
        duplicate["provenance"]["prompt_hash"] = "sha256:two"
        distinct = deepcopy(base_candidate)
        distinct["outcome"] = "INTERRUPTED"

        proposals = objective3_hierarchical._consolidate_exact_proposals(
            [
                (base_candidate, "start0_b001_p0001-0002"),
                (duplicate, "start0_b002_p0003-0003"),
                (distinct, "start0_b002_p0003-0003"),
            ]
        )

        self.assertEqual(len(proposals), 2)
        consolidated = next(
            proposal
            for proposal in proposals
            if proposal["outcome"] == "ADVANCES"
        )
        self.assertEqual(
            consolidated["source_block_ids"],
            [
                "start0_b001_p0001-0002",
                "start0_b002_p0003-0003",
            ],
        )
        self.assertEqual(len(consolidated["source_inferences"]), 2)


if __name__ == "__main__":
    unittest.main()
