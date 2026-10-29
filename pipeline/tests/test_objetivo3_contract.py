from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.sgotgraph_schema import (
    PIPELINE_VERSION as OBJECTIVE2_PIPELINE_VERSION,
    SCHEMA_VERSION as OBJECTIVE2_SCHEMA_VERSION,
)
from util.trace_alias import prompt_trace_alias
from util.llm_provider import LLMClient, StructuredResponse
from util.objective3_llm import (
    AnchorDraft,
    EpisodeBatchDraft,
    EpisodeDraft,
    EpisodeReconciliationBatchDraft,
    EpisodeReconciliationDecisionDraft,
    EvidenceDraft,
    RelationBatchDraft,
    RelationDraft,
    RetainedEpisodeOverlapDraft,
    SpanDraft,
    infer_episode_relations,
    infer_episodes,
)
from util.sgotgraph_objective3 import (
    FUNCTIONAL_RELATION_TYPES,
    TEMPORAL_RELATION_TYPES,
    add_episode_relations,
    build_objective3_record,
    derive_routes_and_profile,
    validate_record,
)
from util.sgotgraph_schema import (
    make_provenance,
    validate_record as validate_objective2,
)


ROOT = PIPELINE_ROOT


def span(
    trace: str,
    text: str,
    *,
    occurrence: int = 1,
    anchor_role: str | None = None,
) -> dict:
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = trace.index(text, cursor)
        cursor = start + len(text)
    value = {"start_char": start, "end_char": start + len(text), "text": text}
    if anchor_role is not None:
        value["anchor_role"] = anchor_role
    return value


def base_graph() -> dict:
    pieces = [
        "Propose A. ",
        "Check A. ",
        "A fails. ",
        "Replace A with B. ",
        "Answer B.",
    ]
    node_types = ["PROP", "CHECK", "PROP", "REVISE", "OUT"]
    trace = "".join(pieces)
    provenance = make_provenance("human_adjudicated", "synthetic_contract_test")
    segments = []
    nodes = []
    cursor = 0
    for index, (piece, node_type) in enumerate(zip(pieces, node_types), start=1):
        segment_id = f"s{index}"
        segment = {
            "segment_id": segment_id,
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
                "node_type": node_type,
            }
        )
        cursor += len(piece)
    edges = [
        {
            "edge_id": f"e_next_{index}",
            "source": f"s{index}",
            "target": f"s{index + 1}",
            "edge_type": "NEXT",
            "provenance": make_provenance("deterministic", "backbone"),
        }
        for index in range(1, len(nodes))
    ]
    edges.extend(
        [
            {
                "edge_id": "e_check",
                "source": "s2",
                "target": "s1",
                "edge_type": "CHECKS",
                "provenance": provenance,
            },
            {
                "edge_id": "e_revise",
                "source": "s4",
                "target": "s1",
                "edge_type": "REVISES",
                "provenance": provenance,
            },
            {
                "edge_id": "e_output",
                "source": "s4",
                "target": "s5",
                "edge_type": "LEADS_TO",
                "provenance": provenance,
            },
        ]
    )
    record = {
        "schema_version": OBJECTIVE2_SCHEMA_VERSION,
        "pipeline_version": OBJECTIVE2_PIPELINE_VERSION,
        "trace_id": "synthetic_case",
        "source_record": {"problem_id": "synthetic_problem"},
        "trace_text": trace,
        "segments": segments,
        "nodes": nodes,
        "edges": edges,
        "validation": {},
        "audit": {},
    }
    assert validate_objective2(record)["status"] == "valid"
    return record


def episode(
    trace: str,
    episode_id: str,
    episode_type: str,
    member_text: str,
    outcome: str,
    *,
    anchors: list[dict] | None = None,
    witness_node_ids: list[str] | None = None,
) -> dict:
    member = span(trace, member_text)
    provenance = {"origin": "imported", "stage": "synthetic_contract_test"}
    return {
        "episode_id": episode_id,
        "episode_type": episode_type,
        "members": [member],
        "anchors": anchors or [],
        "entry_anchor": None,
        "exit_anchor": None,
        "outcome": outcome,
        "ambiguity_codes": [],
        "evidence": {
            "text_spans": [member],
            "witness_node_ids": witness_node_ids or [],
            "witness_edge_ids": [],
        },
        "provenance": provenance,
    }


def relation(
    trace: str,
    relation_id: str,
    relation_type: str,
    source: str,
    target: str,
    witness_text: str,
    *,
    witness_edges: list[str] | None = None,
) -> dict:
    return {
        "relation_id": relation_id,
        "relation_type": relation_type,
        "source_episode_id": source,
        "target_episode_id": target,
        "evidence": {
            "text_spans": [span(trace, witness_text)],
            "witness_node_ids": [],
            "witness_edge_ids": witness_edges or [],
        },
        "provenance": {
            "origin": "imported",
            "stage": "synthetic_contract_test",
        },
    }


def annotation() -> dict:
    graph = base_graph()
    trace = graph["trace_text"]
    target = span(trace, "Propose A. ", anchor_role="TARGET")
    state = span(trace, "Propose A. ", anchor_role="STATE")
    episodes = [
        episode(
            trace,
            "p1",
            "PROGRESSION",
            "Propose A. ",
            "INTERRUPTED",
            anchors=[state],
            witness_node_ids=["s1"],
        ),
        episode(
            trace,
            "v1",
            "VERIFICATION",
            "Check A. A fails. ",
            "REFUTED",
            anchors=[target],
            witness_node_ids=["s2", "s3"],
        ),
        episode(
            trace,
            "c1",
            "CORRECTION",
            "Replace A with B. ",
            "RECOVERED",
            anchors=[target],
            witness_node_ids=["s4"],
        ),
        episode(
            trace,
            "p2",
            "PROGRESSION",
            "Answer B.",
            "REACHES_OUTPUT",
            anchors=[state],
            witness_node_ids=["s5"],
        ),
    ]
    relations = [
        relation(
            trace,
            "r_eval",
            "EVALUATES",
            "v1",
            "p1",
            "Check A. ",
            witness_edges=["e_check"],
        ),
        relation(trace, "r_interrupt", "INTERRUPTS", "v1", "p1", "Check A. "),
        relation(
            trace,
            "r_replace",
            "REPLACES",
            "c1",
            "p1",
            "Replace A with B. ",
            witness_edges=["e_revise"],
        ),
        relation(trace, "r_resume", "RESUMES", "p2", "p1", "Answer B."),
    ]
    return {
        "trace_id": graph["trace_id"],
        "annotation_provenance": {
            "origin": "imported",
            "stage": "synthetic_contract_test",
        },
        "episodes": episodes,
        "episode_relations": relations,
    }


def complete_record() -> dict:
    annotations = annotation()
    stage06 = build_objective3_record(base_graph(), annotations)
    stage07 = add_episode_relations(stage06, annotations)
    return derive_routes_and_profile(stage07)


def by_type(record: dict, episode_type: str) -> list[dict]:
    return [
        item
        for item in record["episodes"]
        if item["episode_type"] == episode_type
    ]


class Objective3EpisodeTests(unittest.TestCase):
    def test_canonical_episode_contract(self) -> None:
        record = build_objective3_record(base_graph(), annotation())
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(len(record["episodes"]), 4)
        self.assertIn("sgot_graph", record)
        self.assertNotIn("base_graph", record)
        self.assertNotIn("annotation_status", record)

    def test_specialized_episode_requires_anchor(self) -> None:
        annotations = annotation()
        annotations["episodes"][2]["anchors"] = []
        record = build_objective3_record(base_graph(), annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "MISSING_SPECIALIZED_ANCHOR"
                for item in report["errors"]
            )
        )

    def test_specialized_anchor_cannot_replace_the_operation(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        target = span(trace, "Propose A. ", anchor_role="TARGET")
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [
                episode(
                    trace,
                    "v1",
                    "VERIFICATION",
                    "Propose A. ",
                    "CONFIRMED",
                    anchors=[target],
                )
            ],
        }
        record = build_objective3_record(graph, annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"]
                == "OPERATION_COINCIDENT_WITH_SPECIALIZED_ANCHOR"
                for item in report["errors"]
            )
        )

    def test_correction_target_must_precede_revision_content(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        later_target = span(trace, "Answer B.", anchor_role="TARGET")
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [
                episode(
                    trace,
                    "c1",
                    "CORRECTION",
                    "Replace A with B. ",
                    "RECOVERED",
                    anchors=[later_target],
                )
            ],
        }
        record = build_objective3_record(graph, annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "CORRECTION_TARGET_NOT_PRIOR_TO_OPERATION"
                for item in report["errors"]
            )
        )

    def test_cross_type_member_overlap_requires_mixed_function(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        target = span(trace, "Propose A. ", anchor_role="TARGET")
        progression = episode(
            trace,
            "p1",
            "PROGRESSION",
            "Propose A. ",
            "ADVANCES",
        )
        verification = episode(
            trace,
            "v1",
            "VERIFICATION",
            "Propose A. Check A. ",
            "CONFIRMED",
            anchors=[target],
        )
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [progression, verification],
        }
        record = build_objective3_record(graph, annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "UNMARKED_CROSS_TYPE_MEMBER_OVERLAP"
                for item in report["errors"]
            )
        )

        progression["ambiguity_codes"] = ["MIXED_FUNCTION"]
        verification["ambiguity_codes"] = ["MIXED_FUNCTION"]
        marked_record = build_objective3_record(graph, annotations)
        self.assertEqual(
            validate_record(marked_record, stage="episodes")["status"],
            "VALID",
        )

    def test_trace_identity_is_enforced(self) -> None:
        annotations = annotation()
        annotations["trace_id"] = "another_case"
        with self.assertRaises(ValueError):
            build_objective3_record(base_graph(), annotations)

    def test_episode_edge_witness_may_have_one_projected_endpoint(self) -> None:
        annotations = annotation()
        verification = next(
            item
            for item in annotations["episodes"]
            if item["episode_type"] == "VERIFICATION"
        )
        verification["evidence"]["witness_edge_ids"] = ["e_check"]
        record = build_objective3_record(base_graph(), annotations)
        self.assertEqual(
            validate_record(record, stage="episodes")["status"],
            "VALID",
        )

    def test_next_cannot_be_an_episode_witness(self) -> None:
        annotations = annotation()
        verification = next(
            item
            for item in annotations["episodes"]
            if item["episode_type"] == "VERIFICATION"
        )
        verification["evidence"]["witness_edge_ids"] = ["e_next_2"]
        record = build_objective3_record(base_graph(), annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "WITNESS_EDGE_OUTSIDE_PROJECTION"
                for item in report["errors"]
            )
        )

    def test_span_boundary_cannot_cut_an_alphanumeric_token(self) -> None:
        annotations = annotation()
        correction = next(
            item
            for item in annotations["episodes"]
            if item["episode_type"] == "CORRECTION"
        )
        correction["anchors"] = [
            span(
                base_graph()["trace_text"],
                "ropose A. ",
                anchor_role="TARGET",
            )
        ]
        record = build_objective3_record(base_graph(), annotations)
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "SPAN_BOUNDARY_INSIDE_TOKEN"
                for item in report["errors"]
            )
        )

    def test_episode_ids_are_stable_and_ignore_import_ids(self) -> None:
        first = build_objective3_record(base_graph(), annotation())
        changed = annotation()
        for index, item in enumerate(changed["episodes"]):
            item["episode_id"] = f"arbitrary_{index}"
        second = build_objective3_record(base_graph(), changed)
        self.assertEqual(
            [item["episode_id"] for item in first["episodes"]],
            [item["episode_id"] for item in second["episodes"]],
        )


class Objective3RelationTests(unittest.TestCase):
    def test_temporal_relations_are_derived_not_imported(self) -> None:
        annotations = annotation()
        trace = base_graph()["trace_text"]
        annotations["episode_relations"].append(
            relation(
                trace,
                "r_manual_time",
                "IMMEDIATELY_PRECEDES_ONSET",
                "p1",
                "v1",
                "Propose A. ",
            )
        )
        stage06 = build_objective3_record(base_graph(), annotations)
        with self.assertRaises(ValueError):
            add_episode_relations(stage06, annotations)

    def test_functional_relation_requires_witness(self) -> None:
        annotations = annotation()
        annotations["episode_relations"][0]["evidence"] = {
            "text_spans": [],
            "witness_node_ids": [],
            "witness_edge_ids": [],
        }
        stage06 = build_objective3_record(base_graph(), annotations)
        stage07 = add_episode_relations(stage06, annotations)
        report = validate_record(stage07, stage="relations")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(item["code"] == "MISSING_WITNESS" for item in report["errors"])
        )
        with self.assertRaises(ValueError):
            derive_routes_and_profile(stage07)

    def test_progression_continuity_requires_a_referent_witness(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [
                episode(
                    trace,
                    "p1",
                    "PROGRESSION",
                    "Propose A. ",
                    "INTERRUPTED",
                ),
                episode(
                    trace,
                    "v1",
                    "VERIFICATION",
                    "Check A. ",
                    "CONFIRMED",
                    anchors=[
                        span(
                            trace,
                            "Propose A. ",
                            anchor_role="TARGET",
                        )
                    ],
                ),
                episode(
                    trace,
                    "p2",
                    "PROGRESSION",
                    "Answer B.",
                    "REACHES_OUTPUT",
                ),
            ],
            "episode_relations": [
                relation(
                    trace,
                    "r_resume",
                    "RESUMES",
                    "p2",
                    "p1",
                    "Answer B.",
                )
            ],
        }
        stage06 = build_objective3_record(graph, annotations)
        stage07 = add_episode_relations(stage06, annotations)
        report = validate_record(stage07, stage="relations")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                item["code"] == "INCOMPATIBLE_FUNCTIONAL_RELATION"
                and "referente común" in item["message"]
                for item in report["errors"]
            )
        )

    def test_only_temporal_relations_do_not_merge_routes(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [
                episode(trace, "p1", "PROGRESSION", "Propose A. ", "ADVANCES"),
                episode(
                    trace,
                    "p2",
                    "PROGRESSION",
                    "Answer B.",
                    "REACHES_OUTPUT",
                ),
            ],
            "episode_relations": [],
        }
        stage06 = build_objective3_record(graph, annotations)
        stage07 = add_episode_relations(stage06, annotations)
        final = derive_routes_and_profile(stage07)
        temporal = [
            item
            for item in final["episode_relations"]
            if item["relation_type"] in TEMPORAL_RELATION_TYPES
        ]
        self.assertEqual(len(temporal), 1)
        self.assertEqual(len(final["routes"]), 2)
        self.assertEqual(len(final["progression_lines"]), 2)
        self.assertTrue(
            all(len(route["episode_ids"]) == 1 for route in final["routes"])
        )

    def test_equal_onsets_remain_temporally_incomparable(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        target = span(trace, "Propose A. ", anchor_role="TARGET")
        progression = episode(
            trace,
            "p1",
            "PROGRESSION",
            "Propose A. ",
            "ADVANCES",
        )
        verification = episode(
            trace,
            "v1",
            "VERIFICATION",
            "Propose A. Check A. ",
            "CONFIRMED",
            anchors=[target],
        )
        progression["ambiguity_codes"] = ["MIXED_FUNCTION"]
        verification["ambiguity_codes"] = ["MIXED_FUNCTION"]
        annotations = {
            "trace_id": graph["trace_id"],
            "annotation_provenance": {
                "origin": "imported",
                "stage": "synthetic_contract_test",
            },
            "episodes": [
                progression,
                verification,
                episode(
                    trace,
                    "p2",
                    "PROGRESSION",
                    "Answer B.",
                    "REACHES_OUTPUT",
                ),
            ],
            "episode_relations": [],
        }
        stage06 = build_objective3_record(graph, annotations)
        stage07 = add_episode_relations(stage06, annotations)
        p1 = by_type(stage07, "PROGRESSION")[0]["episode_id"]
        p2 = by_type(stage07, "PROGRESSION")[1]["episode_id"]
        v1 = by_type(stage07, "VERIFICATION")[0]["episode_id"]
        temporal_pairs = {
            (item["source_episode_id"], item["target_episode_id"])
            for item in stage07["episode_relations"]
            if item["relation_type"] in TEMPORAL_RELATION_TYPES
        }
        self.assertEqual(temporal_pairs, {(p1, p2), (v1, p2)})
        self.assertNotIn((p1, v1), temporal_pairs)
        self.assertNotIn((v1, p1), temporal_pairs)


class Objective3RouteAndProfileTests(unittest.TestCase):
    def test_complete_record_is_valid(self) -> None:
        record = complete_record()
        report = validate_record(record)
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(report["errors"], [])

    def test_routes_use_focal_core_and_one_hop_context(self) -> None:
        record = complete_record()
        progression = next(
            route
            for route in record["routes"]
            if route["route_type"] == "PROGRESSION"
        )
        self.assertEqual(len(progression["focus_episode_ids"]), 2)
        self.assertEqual(len(progression["episode_ids"]), 4)
        verification = next(
            route
            for route in record["routes"]
            if route["route_type"] == "VERIFICATION"
        )
        self.assertEqual(len(verification["focus_episode_ids"]), 1)
        self.assertEqual(len(verification["episode_ids"]), 2)

    def test_all_route_relations_are_functional_and_induced(self) -> None:
        record = complete_record()
        for route in record["routes"]:
            episode_ids = set(route["episode_ids"])
            expected = {
                relation["relation_id"]
                for relation in record["episode_relations"]
                if relation["relation_type"] in FUNCTIONAL_RELATION_TYPES
                and relation["source_episode_id"] in episode_ids
                and relation["target_episode_id"] in episode_ids
            }
            self.assertEqual(set(route["relation_ids"]), expected)

    def test_profile_and_configurations_are_recomputed(self) -> None:
        record = complete_record()
        self.assertEqual(len(record["configurations"]), len(record["routes"]))
        self.assertTrue(
            all(
                item["configuration_signature"].startswith("sha256:")
                for item in record["configurations"]
            )
        )
        record["profile"]["episode_count"] += 1
        report = validate_record(record)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(item["code"] == "NONCANONICAL_PROFILE" for item in report["errors"])
        )

    def test_cli_validator_accepts_a_canonical_artifact(self) -> None:
        record = complete_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact_path = temp / "00_sgotgraph_objetivo3.json"
            report_path = temp / "validation.json"
            artifact_path.write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "09_validacion_objetivo3_gcot.py",
                    str(artifact_path),
                    "-o",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            self.assertEqual(report["valid"], 1)
            self.assertEqual(report["invalid"], 0)

    def test_validator_rejects_malformed_values_without_crashing(self) -> None:
        malformed_cases = []
        malformed_route = complete_record()
        malformed_route["routes"][0]["episode_ids"] = [{"not": "an_id"}]
        malformed_cases.append(malformed_route)
        malformed_relation = complete_record()
        malformed_relation["episode_relations"][0][
            "source_episode_id"
        ] = {"not": "an_id"}
        malformed_cases.append(malformed_relation)
        missing_required_field = complete_record()
        del missing_required_field["episodes"][0]["outcome"]
        malformed_cases.append(missing_required_field)
        unexpected_top_level_field = complete_record()
        unexpected_top_level_field["legacy_route"] = []
        malformed_cases.append(unexpected_top_level_field)
        for record in malformed_cases:
            with self.subTest(record=record):
                self.assertEqual(validate_record(record)["status"], "INVALID")
        self.assertEqual(validate_record([])["status"], "INVALID")


class Objective3LLMOrchestrationTests(unittest.TestCase):
    def test_episode_inference_calls_one_estimator_per_type(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        member_span = span(trace, "Propose A. ")
        member = SpanDraft(
            start_char=member_span["start_char"],
            end_char=member_span["end_char"],
        )
        responses: list[str] = []

        def fake_request(*args, **kwargs):
            user_prompt = kwargs["user_prompt"]
            episode_type = next(
                item
                for item in (
                    "ALTERNATIVE",
                    "CORRECTION",
                    "PROGRESSION",
                    "SUPPORT",
                    "VERIFICATION",
                )
                if f"episode_type='{item}'" in user_prompt
            )
            responses.append(episode_type)
            episodes = []
            if episode_type == "PROGRESSION":
                episodes = [
                    EpisodeDraft(
                        members=[member],
                        anchors=[],
                        entry_anchor=None,
                        exit_anchor=None,
                        outcome="ADVANCES",
                        ambiguity_codes=[],
                        evidence=EvidenceDraft(
                            text_spans=[member],
                            witness_node_ids=["s1"],
                            witness_edge_ids=[],
                        ),
                    )
                ]
            parsed = EpisodeBatchDraft(
                trace_id=prompt_trace_alias(graph["trace_id"]),
                episode_type=episode_type,
                episodes=episodes,
            )
            return StructuredResponse(
                parsed=parsed,
                provider="openai",
                model="mock-model",
                response_id=f"response-{episode_type}",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch(
            "util.objective3_llm.request_structured_response",
            side_effect=fake_request,
        ):
            record = infer_episodes(
                client,
                model="mock-model",
                sgot_graph=graph,
                reasoning_effort="low",
            )
        self.assertEqual(set(responses), {
            "ALTERNATIVE",
            "CORRECTION",
            "PROGRESSION",
            "SUPPORT",
            "VERIFICATION",
        })
        self.assertEqual(record["validation"]["status"], "VALID")
        self.assertEqual(len(record["episodes"]), 1)
        self.assertEqual(
            record["episodes"][0]["provenance"]["origin"], "llm_inferred"
        )
        self.assertEqual(record["episodes"][0]["members"][0]["text"], "Propose A. ")

    def test_cross_type_reconciliation_repairs_and_marks_retained_overlap(
        self,
    ) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        member_value = span(trace, "Replace A with B. ")
        target_value = span(trace, "Propose A. ")
        member = SpanDraft(
            start_char=member_value["start_char"],
            end_char=member_value["end_char"],
        )
        target = AnchorDraft(
            start_char=target_value["start_char"],
            end_char=target_value["end_char"],
            anchor_role="TARGET",
        )
        contrast = AnchorDraft(
            start_char=target_value["start_char"],
            end_char=target_value["end_char"],
            anchor_role="CONTRAST",
        )
        evidence = EvidenceDraft(
            text_spans=[member],
            witness_node_ids=["s4"],
            witness_edge_ids=[],
        )
        type_calls: list[str] = []
        reconciliation_prompts: list[str] = []

        def fake_request(*args, **kwargs):
            response_model = kwargs["response_model"]
            if response_model is EpisodeBatchDraft:
                user_prompt = kwargs["user_prompt"]
                episode_type = next(
                    item
                    for item in (
                        "ALTERNATIVE",
                        "CORRECTION",
                        "PROGRESSION",
                        "SUPPORT",
                        "VERIFICATION",
                    )
                    if f"episode_type='{item}'" in user_prompt
                )
                type_calls.append(episode_type)
                episodes = []
                if episode_type == "ALTERNATIVE":
                    episodes = [
                        EpisodeDraft(
                            members=[member],
                            anchors=[contrast],
                            entry_anchor=None,
                            exit_anchor=None,
                            outcome="REJECTED",
                            ambiguity_codes=[],
                            evidence=evidence,
                        )
                    ]
                elif episode_type == "CORRECTION":
                    episodes = [
                        EpisodeDraft(
                            members=[member],
                            anchors=[target],
                            entry_anchor=None,
                            exit_anchor=None,
                            outcome="RECOVERED",
                            ambiguity_codes=[],
                            evidence=evidence,
                        )
                    ]
                return StructuredResponse(
                    parsed=EpisodeBatchDraft(
                        trace_id=prompt_trace_alias(graph["trace_id"]),
                        episode_type=episode_type,
                        episodes=episodes,
                    ),
                    provider="openai",
                    model="mock-model",
                    response_id=f"episode-{episode_type}",
                )

            self.assertIs(response_model, EpisodeReconciliationBatchDraft)
            reconciliation_prompts.append(kwargs["user_prompt"])
            retained = []
            if len(reconciliation_prompts) == 2:
                retained = [
                    RetainedEpisodeOverlapDraft(
                        first_episode_ref="E0001",
                        second_episode_ref="E0002",
                        distinction_basis=["DISTINCT_OUTCOME"],
                    )
                ]
            return StructuredResponse(
                parsed=EpisodeReconciliationBatchDraft(
                    trace_id=prompt_trace_alias(graph["trace_id"]),
                    decisions=[
                        EpisodeReconciliationDecisionDraft(
                            episode_ref="E0001",
                            decision="KEEP",
                        ),
                        EpisodeReconciliationDecisionDraft(
                            episode_ref="E0002",
                            decision="KEEP",
                        ),
                    ],
                    retained_overlaps=retained,
                ),
                provider="openai",
                model="mock-model",
                response_id=f"reconciliation-{len(reconciliation_prompts)}",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch(
            "util.objective3_llm.request_structured_response",
            side_effect=fake_request,
        ):
            record = infer_episodes(
                client,
                model="mock-model",
                sgot_graph=graph,
                max_attempts=2,
            )

        self.assertEqual(len(type_calls), 5)
        self.assertEqual(len(reconciliation_prompts), 2)
        self.assertIn(
            '"episode_type_specifications"',
            reconciliation_prompts[0],
        )
        self.assertIn(
            '"required_anchor": "TARGET"',
            reconciliation_prompts[0],
        )
        self.assertIn(
            '"allowed_outcomes"',
            reconciliation_prompts[0],
        )
        self.assertIn(
            "MISSING_RETAINED_OVERLAP_JUSTIFICATION",
            reconciliation_prompts[1],
        )
        self.assertEqual(record["validation"]["status"], "VALID")
        self.assertEqual(len(record["episodes"]), 2)
        self.assertTrue(
            all(
                "MIXED_FUNCTION" in item["ambiguity_codes"]
                for item in record["episodes"]
            )
        )
        reconciliation_audit = record["audit"][-1]
        self.assertEqual(
            reconciliation_audit["action"],
            "cross_type_overlaps_reconciled",
        )
        self.assertEqual(
            len(reconciliation_audit["retained_overlap_justifications"]),
            1,
        )

    def test_cross_type_reconciliation_can_remove_a_false_candidate(
        self,
    ) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        member_value = span(trace, "Replace A with B. ")
        target_value = span(trace, "Propose A. ")
        member = SpanDraft(
            start_char=member_value["start_char"],
            end_char=member_value["end_char"],
        )
        evidence = EvidenceDraft(
            text_spans=[member],
            witness_node_ids=["s4"],
            witness_edge_ids=[],
        )
        reconciliation_calls = 0

        def fake_request(*args, **kwargs):
            nonlocal reconciliation_calls
            if kwargs["response_model"] is EpisodeBatchDraft:
                user_prompt = kwargs["user_prompt"]
                episode_type = next(
                    item
                    for item in (
                        "ALTERNATIVE",
                        "CORRECTION",
                        "PROGRESSION",
                        "SUPPORT",
                        "VERIFICATION",
                    )
                    if f"episode_type='{item}'" in user_prompt
                )
                episodes = []
                if episode_type in {"ALTERNATIVE", "CORRECTION"}:
                    anchor_role = (
                        "CONTRAST"
                        if episode_type == "ALTERNATIVE"
                        else "TARGET"
                    )
                    episodes = [
                        EpisodeDraft(
                            members=[member],
                            anchors=[
                                AnchorDraft(
                                    start_char=target_value["start_char"],
                                    end_char=target_value["end_char"],
                                    anchor_role=anchor_role,
                                )
                            ],
                            entry_anchor=None,
                            exit_anchor=None,
                            outcome=(
                                "REJECTED"
                                if episode_type == "ALTERNATIVE"
                                else "RECOVERED"
                            ),
                            ambiguity_codes=[],
                            evidence=evidence,
                        )
                    ]
                return StructuredResponse(
                    parsed=EpisodeBatchDraft(
                        trace_id=prompt_trace_alias(graph["trace_id"]),
                        episode_type=episode_type,
                        episodes=episodes,
                    ),
                    provider="openai",
                    model="mock-model",
                    response_id=f"episode-{episode_type}",
                )

            reconciliation_calls += 1
            return StructuredResponse(
                parsed=EpisodeReconciliationBatchDraft(
                    trace_id=prompt_trace_alias(graph["trace_id"]),
                    decisions=[
                        EpisodeReconciliationDecisionDraft(
                            episode_ref="E0001",
                            decision="REMOVE",
                        ),
                        EpisodeReconciliationDecisionDraft(
                            episode_ref="E0002",
                            decision="KEEP",
                        ),
                    ],
                    retained_overlaps=[],
                ),
                provider="openai",
                model="mock-model",
                response_id="reconciliation-remove",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch(
            "util.objective3_llm.request_structured_response",
            side_effect=fake_request,
        ):
            record = infer_episodes(
                client,
                model="mock-model",
                sgot_graph=graph,
                max_attempts=2,
            )

        self.assertEqual(reconciliation_calls, 1)
        self.assertEqual(record["validation"]["status"], "VALID")
        self.assertEqual(
            [item["episode_type"] for item in record["episodes"]],
            ["CORRECTION"],
        )
        self.assertEqual(
            len(record["audit"][-1]["removed_episode_ids"]),
            1,
        )

    def test_relation_aliases_map_to_canonical_ids_and_support_repair(self) -> None:
        graph = base_graph()
        stage06 = build_objective3_record(graph, annotation())
        reference_by_id = {
            item["episode_id"]: f"E{index:04d}"
            for index, item in enumerate(stage06["episodes"], start=1)
        }
        verification = next(
            item
            for item in stage06["episodes"]
            if item["episode_type"] == "VERIFICATION"
        )
        progression = next(
            item
            for item in stage06["episodes"]
            if item["episode_type"] == "PROGRESSION"
            and item["members"][0]["text"] == "Propose A. "
        )
        evidence_span = span(graph["trace_text"], "Check A. ")
        evidence = EvidenceDraft(
            text_spans=[
                SpanDraft(
                    start_char=evidence_span["start_char"],
                    end_char=evidence_span["end_char"],
                )
            ],
            witness_node_ids=["s1", "s2"],
            witness_edge_ids=["e_check"],
        )
        calls = 0
        captured_prompts: list[str] = []

        def fake_request(*args, **kwargs):
            nonlocal calls
            calls += 1
            captured_prompts.append(kwargs["user_prompt"])
            source = progression if calls == 1 else verification
            target = verification if calls == 1 else progression
            return StructuredResponse(
                parsed=RelationBatchDraft(
                    trace_id=prompt_trace_alias(graph["trace_id"]),
                    relations=[
                        RelationDraft(
                            relation_type="EVALUATES",
                            source_episode_ref=reference_by_id[
                                source["episode_id"]
                            ],
                            target_episode_ref=reference_by_id[
                                target["episode_id"]
                            ],
                            evidence=evidence,
                        )
                    ],
                ),
                provider="openai",
                model="mock-model",
                response_id=f"response-{calls}",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch(
            "util.objective3_llm.request_structured_response",
            side_effect=fake_request,
        ):
            stage07 = infer_episode_relations(
                client,
                model="mock-model",
                record=stage06,
                max_attempts=2,
            )

        self.assertEqual(calls, 2)
        self.assertEqual(stage07["validation"]["status"], "VALID")
        inferred = next(
            item
            for item in stage07["episode_relations"]
            if item["relation_type"] == "EVALUATES"
        )
        self.assertEqual(
            inferred["source_episode_id"],
            verification["episode_id"],
        )
        self.assertEqual(
            inferred["target_episode_id"],
            progression["episode_id"],
        )
        self.assertIn('"episode_ref": "E0001"', captured_prompts[0])
        self.assertNotIn(verification["episode_id"], captured_prompts[0])
        self.assertIn("Deterministic validation errors", captured_prompts[1])

    def test_relation_inference_is_global_and_can_return_empty(self) -> None:
        graph = base_graph()
        annotations = annotation()
        stage06 = build_objective3_record(graph, annotations)
        captured_prompt = ""

        def fake_request(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs["user_prompt"]
            return StructuredResponse(
                parsed=RelationBatchDraft(
                    trace_id=prompt_trace_alias(graph["trace_id"]), relations=[]
                ),
                provider="openai",
                model="mock-model",
                response_id="response-relations",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch(
            "util.objective3_llm.request_structured_response",
            side_effect=fake_request,
        ):
            stage07 = infer_episode_relations(
                client,
                model="mock-model",
                record=stage06,
            )
        self.assertEqual(stage07["validation"]["status"], "VALID")
        self.assertIn('"episodes"', captured_prompt)
        self.assertEqual(
            [
                item
                for item in stage07["episode_relations"]
                if item["relation_type"] in FUNCTIONAL_RELATION_TYPES
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
