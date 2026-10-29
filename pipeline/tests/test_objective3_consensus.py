from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from tests.test_objetivo3_contract import base_graph, span  # noqa: E402
from util.llm_provider import LLMClient, StructuredResponse  # noqa: E402
from util.objective3_consensus import (  # noqa: E402
    ANCHOR_UNCERTAIN,
    BOUNDARY_UNCERTAIN,
    OUTCOME_UNCERTAIN,
    cluster_episode_samples,
    consensus_episodes,
)
from util.objective3_llm import (  # noqa: E402
    AnchorDraft,
    EpisodeBatchDraft,
    EpisodeDraft,
    EvidenceDraft,
    SpanDraft,
    infer_episodes,
)
from util.sgotgraph_objective3 import (  # noqa: E402
    build_objective3_record_from_candidates,
    validate_record,
)
from util.trace_alias import prompt_trace_alias  # noqa: E402


def episode(
    members: list[tuple[int, int]],
    *,
    outcome: str = "ADVANCES",
    anchors: list[tuple[str, int, int]] | None = None,
    codes: list[str] | None = None,
) -> dict:
    return {
        "episode_type": "PROGRESSION",
        "members": [{"start_char": a, "end_char": b, "text": "x" * (b - a)} for a, b in members],
        "anchors": [
            {"anchor_role": role, "start_char": a, "end_char": b, "text": "y" * (b - a)}
            for role, a, b in (anchors or [])
        ],
        "entry_anchor": None,
        "exit_anchor": None,
        "outcome": outcome,
        "ambiguity_codes": list(codes or []),
        "evidence": {"text_spans": [], "witness_node_ids": ["s1"], "witness_edge_ids": []},
        "provenance": {"origin": "llm_inferred", "stage": "06_episodes"},
    }


class EpisodeClusteringTests(unittest.TestCase):
    def test_majority_keeps_matched_episodes_and_drops_minority(self) -> None:
        # Tres muestras: A aparece en las tres (con límites ligeramente distintos),
        # B en dos, C solo en una.
        samples = [
            [episode([(0, 100)]), episode([(200, 300)], outcome="ADVANCES")],
            [episode([(0, 90)]), episode([(205, 300)], outcome="REACHES_OUTPUT"), episode([(400, 450)])],
            [episode([(5, 100)])],
        ]
        result = consensus_episodes(samples, iou_threshold=0.5)
        kept = result["episodes"]
        self.assertEqual(len(kept), 2)
        a, b = kept
        self.assertEqual(a["consensus"]["support"], 3)
        self.assertEqual(a["consensus"]["samples"], 3)
        self.assertTrue(a["consensus"]["unanimous"])
        self.assertLess(a["consensus"]["member_iou_mean"], 1.0)
        self.assertIn(BOUNDARY_UNCERTAIN, a["ambiguity_codes"])
        self.assertNotIn(OUTCOME_UNCERTAIN, a["ambiguity_codes"])
        # el medoide de A es el candidato con mayor IoU media: (0,100)
        self.assertEqual(a["members"][0]["start_char"], 0)
        self.assertEqual(a["members"][0]["end_char"], 100)
        self.assertEqual(b["consensus"]["support"], 2)
        self.assertIn(OUTCOME_UNCERTAIN, b["ambiguity_codes"])
        self.assertEqual(b["consensus"]["votes"], {"ADVANCES": 1, "REACHES_OUTPUT": 1})
        self.assertEqual(b["consensus"]["tie_break"], "first_sample")
        self.assertEqual(b["outcome"], "ADVANCES")
        audit = result["audit"]
        self.assertEqual(audit["candidate_counts_by_sample"], [2, 3, 1])
        self.assertEqual(audit["kept_count"], 2)
        self.assertEqual(audit["minority_count"], 1)
        self.assertEqual(audit["majority_threshold"], 2)
        self.assertEqual(
            [item["status"] for item in audit["clusters"]],
            ["kept", "kept", "minority"],
        )

    def test_anchor_disagreement_is_marked_without_changing_members(self) -> None:
        samples = [
            [episode([(0, 50)], anchors=[("STATE", 60, 70)])],
            [episode([(0, 50)], anchors=[("STATE", 80, 90)])],
            [episode([(0, 50)], anchors=[("STATE", 60, 70)])],
        ]
        kept = consensus_episodes(samples)["episodes"]
        self.assertEqual(len(kept), 1)
        self.assertIn(ANCHOR_UNCERTAIN, kept[0]["ambiguity_codes"])
        self.assertNotIn(BOUNDARY_UNCERTAIN, kept[0]["ambiguity_codes"])
        self.assertEqual(kept[0]["consensus"]["member_iou_mean"], 1.0)
        self.assertEqual(kept[0]["anchors"][0]["start_char"], 60)

    def test_one_sample_reproduces_the_single_estimator(self) -> None:
        clusters = cluster_episode_samples([[episode([(0, 10)]), episode([(20, 30)])]])
        self.assertEqual(len(clusters), 2)
        kept = consensus_episodes([[episode([(0, 10)])]])["episodes"]
        self.assertEqual(kept[0]["consensus"]["support"], 1)
        self.assertEqual(kept[0]["ambiguity_codes"], [])

    def test_each_sample_contributes_at_most_one_candidate_per_cluster(self) -> None:
        # Dos candidatos casi idénticos en la misma muestra no inflan el soporte.
        samples = [
            [episode([(0, 100)]), episode([(0, 95)])],
            [episode([(0, 100)])],
            [episode([(0, 100)])],
        ]
        result = consensus_episodes(samples)
        supports = sorted(item["support"] for item in result["audit"]["clusters"])
        self.assertEqual(supports, [1, 3])
        self.assertEqual(len(result["episodes"]), 1)

    def test_threshold_and_sample_validation(self) -> None:
        with self.assertRaises(ValueError):
            consensus_episodes([], iou_threshold=0.5)
        with self.assertRaises(ValueError):
            consensus_episodes([[episode([(0, 10)])]], iou_threshold=0)


class InferEpisodesConsensusTests(unittest.TestCase):
    def test_infer_episodes_samples_each_type_and_aggregates(self) -> None:
        graph = base_graph()
        trace = graph["trace_text"]
        long_member = span(trace, "Propose A. ")
        short_member = span(trace, "Propose A")
        far_member = span(trace, "Answer B.")
        calls: list[tuple[str, int]] = []

        def fake_request(*args, **kwargs):
            user_prompt = kwargs["user_prompt"]
            episode_type = next(
                item
                for item in ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")
                if f"episode_type='{item}'" in user_prompt
            )
            sample = sum(1 for kind, _ in calls if kind == episode_type) + 1
            calls.append((episode_type, sample))
            episodes: list[EpisodeDraft] = []
            if episode_type == "PROGRESSION":
                member = short_member if sample == 2 else long_member
                episodes.append(
                    EpisodeDraft(
                        members=[SpanDraft(start_char=member["start_char"], end_char=member["end_char"])],
                        anchors=[],
                        entry_anchor=None,
                        exit_anchor=None,
                        outcome="ADVANCES" if sample != 3 else "REACHES_OUTPUT",
                        ambiguity_codes=[],
                        evidence=EvidenceDraft(
                            text_spans=[SpanDraft(start_char=member["start_char"], end_char=member["end_char"])],
                            witness_node_ids=["s1"],
                            witness_edge_ids=[],
                        ),
                    )
                )
                if sample == 1:
                    # candidato minoritario: solo en la primera muestra
                    episodes.append(
                        EpisodeDraft(
                            members=[SpanDraft(start_char=far_member["start_char"], end_char=far_member["end_char"])],
                            anchors=[],
                            entry_anchor=None,
                            exit_anchor=None,
                            outcome="REACHES_OUTPUT",
                            ambiguity_codes=[],
                            evidence=EvidenceDraft(
                                text_spans=[],
                                witness_node_ids=["s5"],
                                witness_edge_ids=[],
                            ),
                        )
                    )
            parsed = EpisodeBatchDraft(
                trace_id=prompt_trace_alias(graph["trace_id"]), episode_type=episode_type, episodes=episodes
            )
            return StructuredResponse(
                parsed=parsed, provider="openai", model="mock-model", response_id=f"r-{episode_type}-{sample}"
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch("util.objective3_llm.request_structured_response", side_effect=fake_request):
            record = infer_episodes(
                client, model="mock-model", sgot_graph=graph, reasoning_effort="low", samples=3
            )
        self.assertEqual(len(calls), 15)
        self.assertEqual(record["validation"]["status"], "VALID")
        self.assertEqual(len(record["episodes"]), 1)
        kept = record["episodes"][0]
        self.assertEqual(kept["members"][0]["text"], "Propose A. ")
        self.assertEqual(kept["consensus"]["support"], 3)
        self.assertEqual(kept["consensus"]["votes"], {"ADVANCES": 2, "REACHES_OUTPUT": 1})
        self.assertEqual(kept["outcome"], "ADVANCES")
        self.assertIn(BOUNDARY_UNCERTAIN, kept["ambiguity_codes"])
        self.assertIn(OUTCOME_UNCERTAIN, kept["ambiguity_codes"])
        consensus_events = [
            event for event in record["audit"] if event.get("stage") == "06_episode_consensus"
        ]
        self.assertEqual(len(consensus_events), 1)
        by_type = {item["episode_type"]: item for item in consensus_events[0]["by_type"]}
        self.assertEqual(by_type["PROGRESSION"]["candidate_counts_by_sample"], [2, 1, 1])
        self.assertEqual(by_type["PROGRESSION"]["minority_count"], 1)
        self.assertEqual(by_type["SUPPORT"]["kept_count"], 0)
        self.assertEqual(validate_record(record, stage="episodes")["status"], "VALID")

    def test_single_sample_has_no_consensus_field(self) -> None:
        graph = base_graph()
        member = span(graph["trace_text"], "Propose A. ")

        def fake_request(*args, **kwargs):
            user_prompt = kwargs["user_prompt"]
            episode_type = next(
                item
                for item in ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")
                if f"episode_type='{item}'" in user_prompt
            )
            episodes = []
            if episode_type == "PROGRESSION":
                episodes = [
                    EpisodeDraft(
                        members=[SpanDraft(start_char=member["start_char"], end_char=member["end_char"])],
                        anchors=[],
                        entry_anchor=None,
                        exit_anchor=None,
                        outcome="ADVANCES",
                        ambiguity_codes=[],
                        evidence=EvidenceDraft(text_spans=[], witness_node_ids=["s1"], witness_edge_ids=[]),
                    )
                ]
            return StructuredResponse(
                parsed=EpisodeBatchDraft(trace_id=prompt_trace_alias(graph["trace_id"]), episode_type=episode_type, episodes=episodes),
                provider="openai",
                model="mock-model",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch("util.objective3_llm.request_structured_response", side_effect=fake_request) as calls:
            record = infer_episodes(client, model="mock-model", sgot_graph=graph, reasoning_effort="low")
        self.assertEqual(calls.call_count, 5)
        self.assertNotIn("consensus", record["episodes"][0])
        self.assertFalse(
            any(event.get("stage") == "06_episode_consensus" for event in record["audit"])
        )

    def test_consensus_field_is_validated(self) -> None:
        graph = base_graph()
        member = span(graph["trace_text"], "Propose A. ")

        def fake_request(*args, **kwargs):
            user_prompt = kwargs["user_prompt"]
            episode_type = next(
                item
                for item in ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")
                if f"episode_type='{item}'" in user_prompt
            )
            episodes = []
            if episode_type == "PROGRESSION":
                episodes = [
                    EpisodeDraft(
                        members=[SpanDraft(start_char=member["start_char"], end_char=member["end_char"])],
                        anchors=[],
                        entry_anchor=None,
                        exit_anchor=None,
                        outcome="ADVANCES",
                        ambiguity_codes=[],
                        evidence=EvidenceDraft(text_spans=[], witness_node_ids=["s1"], witness_edge_ids=[]),
                    )
                ]
            return StructuredResponse(
                parsed=EpisodeBatchDraft(trace_id=prompt_trace_alias(graph["trace_id"]), episode_type=episode_type, episodes=episodes),
                provider="openai",
                model="mock-model",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch("util.objective3_llm.request_structured_response", side_effect=fake_request):
            record = infer_episodes(client, model="mock-model", sgot_graph=graph, reasoning_effort="low", samples=3)
        self.assertEqual(validate_record(record, stage="episodes")["status"], "VALID")
        record["episodes"][0]["consensus"]["support"] = 7
        report = validate_record(record, stage="episodes")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any(error["code"] == "INVALID_CONSENSUS" for error in report["errors"]))


class FailedSampleReplacementTests(unittest.TestCase):
    def _run(self, samples: int, max_attempts: int = 1):
        graph = base_graph()
        trace = graph["trace_text"]
        good = span(trace, "Propose A. ")
        calls: list[tuple[str, int]] = []

        def fake_request(*args, **kwargs):
            user_prompt = kwargs["user_prompt"]
            episode_type = next(
                item
                for item in ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")
                if f"episode_type='{item}'" in user_prompt
            )
            draw = sum(1 for kind, _ in calls if kind == episode_type) + 1
            calls.append((episode_type, draw))
            episodes: list[EpisodeDraft] = []
            if episode_type == "PROGRESSION":
                # La primera muestra corta un token alfanumérico y agota sus
                # intentos; las siguientes son válidas.
                start = good["start_char"] + 2 if draw == 1 else good["start_char"]
                episodes.append(
                    EpisodeDraft(
                        members=[SpanDraft(start_char=start, end_char=good["end_char"])],
                        anchors=[],
                        entry_anchor=None,
                        exit_anchor=None,
                        outcome="ADVANCES",
                        ambiguity_codes=[],
                        evidence=EvidenceDraft(
                            text_spans=[], witness_node_ids=["s1"], witness_edge_ids=[]
                        ),
                    )
                )
            return StructuredResponse(
                parsed=EpisodeBatchDraft(
                    trace_id=kwargs["user_prompt"].split("trace_id=")[1].split("'")[1],
                    episode_type=episode_type,
                    episodes=episodes,
                ),
                provider="openai",
                model="mock-model",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch("util.objective3_llm.request_structured_response", side_effect=fake_request):
            record = infer_episodes(
                client,
                model="mock-model",
                sgot_graph=graph,
                reasoning_effort="low",
                samples=samples,
                max_attempts=max_attempts,
            )
        return record, calls

    def test_failed_sample_is_replaced_when_consensus_is_active(self) -> None:
        record, calls = self._run(samples=3)
        self.assertEqual(sum(1 for kind, _ in calls if kind == "PROGRESSION"), 4)
        self.assertEqual(record["validation"]["status"], "VALID")
        kept = record["episodes"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["consensus"]["support"], 3)
        consensus_event = next(
            event for event in record["audit"] if event.get("stage") == "06_episode_consensus"
        )
        progression = next(
            item for item in consensus_event["by_type"] if item["episode_type"] == "PROGRESSION"
        )
        self.assertEqual(progression["sample_draws"], 4)
        self.assertEqual(len(progression["failed_samples"]), 1)
        self.assertEqual(progression["failed_samples"][0]["sample"], 1)

    def test_single_sample_keeps_fail_closed_policy(self) -> None:
        with self.assertRaises(ValueError):
            self._run(samples=1)


class TokenBoundaryHintTests(unittest.TestCase):
    def test_error_message_reports_text_and_nearest_aligned_interval(self) -> None:
        from util.sgotgraph_objective3 import nearest_token_aligned_interval

        trace = "= 495 * 15 = $7,425. Alternatively, we can compute"
        start = trace.index("Alternatively")
        self.assertEqual(
            nearest_token_aligned_interval(trace, start, start + len("Alternativel")),
            (start, start + len("Alternatively")),
        )
        # un límite que ya está alineado no se mueve
        self.assertEqual(nearest_token_aligned_interval(trace, start, start + 13), (start, start + 13))
        graph = base_graph()
        member = span(graph["trace_text"], "Propose A. ")
        bad = {"start_char": member["start_char"] + 2, "end_char": member["end_char"]}
        record = build_objective3_record_from_candidates(
            graph,
            [
                {
                    "episode_type": "PROGRESSION",
                    "members": [{**bad, "text": graph["trace_text"][bad["start_char"]:bad["end_char"]]}],
                    "anchors": [],
                    "entry_anchor": None,
                    "exit_anchor": None,
                    "outcome": "ADVANCES",
                    "ambiguity_codes": [],
                    "evidence": {"text_spans": [], "witness_node_ids": ["s1"], "witness_edge_ids": []},
                }
            ],
            default_provenance={"origin": "llm_inferred", "stage": "06_episodes"},
        )
        report = validate_record(record, stage="episodes")
        message = next(
            error["message"] for error in report["errors"] if error["code"] == "SPAN_BOUNDARY_INSIDE_TOKEN"
        )
        self.assertIn("'opose A. '", message)
        self.assertIn("'Propose A. '", message)


if __name__ == "__main__":
    unittest.main()
