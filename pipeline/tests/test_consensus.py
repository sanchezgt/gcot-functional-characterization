from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.consensus import (  # noqa: E402
    AMBIGUITY_TIE,
    AMBIGUITY_UNSTABLE,
    CONSENSUS_METHOD_VERSION,
    consensus_record,
    has_tie,
    majority_label,
    majority_set,
    majority_threshold,
    validate_consensus_record,
)
from util.sgotgraph_schema import make_provenance, validate_record  # noqa: E402


def load_stage_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def per_source(edges_module, fake):
    """Adapta una respuesta simulada por nodo fuente a la firma por lote."""

    def batch_fake(client, model, batch, _nodes, effort, verbose, user_prompt, attempt, max_attempts):
        return edges_module.BatchRelationsDraft(
            sources=[
                fake(client, model, item["node"], item["candidates"], effort, verbose, user_prompt, attempt, max_attempts)
                for item in batch
            ]
        )

    return batch_fake


class ConsensusHelpersTests(unittest.TestCase):
    def test_majority_threshold_is_a_strict_majority(self) -> None:
        self.assertEqual(majority_threshold(1), 1)
        self.assertEqual(majority_threshold(3), 2)
        self.assertEqual(majority_threshold(4), 3)
        self.assertEqual(majority_threshold(5), 3)
        with self.assertRaises(ValueError):
            majority_threshold(0)

    def test_majority_label_reports_votes_and_agreement(self) -> None:
        vote = majority_label(["PROP", "CHECK", "PROP"])
        self.assertEqual(vote["label"], "PROP")
        self.assertEqual(vote["votes"], {"PROP": 2, "CHECK": 1})
        self.assertEqual(vote["support"], 2)
        self.assertAlmostEqual(vote["agreement"], 2 / 3)
        self.assertFalse(vote["unanimous"])
        self.assertFalse(vote["tie"])
        self.assertIsNone(vote["tie_break"])

    def test_majority_label_tie_uses_sampling_order_not_type(self) -> None:
        self.assertTrue(has_tie(["OUT", "PROP", "EVID"]))
        first = majority_label(["OUT", "PROP", "EVID"])
        self.assertEqual(first["label"], "OUT")
        self.assertTrue(first["tie"])
        self.assertEqual(first["tie_break"], "first_sample")
        reordered = majority_label(["EVID", "PROP", "OUT"])
        self.assertEqual(reordered["label"], "EVID")
        self.assertFalse(has_tie(["PROP"]))
        self.assertTrue(majority_label(["PROP"])["unanimous"])

    def test_majority_set_keeps_strict_majority_in_first_seen_order(self) -> None:
        vote = majority_set(
            [
                [("s1", "s2", "SUPPORTS"), ("s1", "s3", "LEADS_TO")],
                [("s1", "s3", "LEADS_TO")],
                [("s1", "s2", "SUPPORTS"), ("s1", "s2", "SUPPORTS")],
            ]
        )
        self.assertEqual(
            vote["kept"],
            [("s1", "s2", "SUPPORTS"), ("s1", "s3", "LEADS_TO")],
        )
        self.assertEqual(vote["support"][("s1", "s2", "SUPPORTS")], 2)
        self.assertEqual(vote["threshold"], 2)
        single = majority_set([[("a", "b", "SUPPORTS")]])
        self.assertEqual(single["kept"], [("a", "b", "SUPPORTS")])
        even = majority_set([[("x",)], [("x",)], [], []])
        self.assertEqual(even["kept"], [])

    def test_consensus_record_contract(self) -> None:
        record = consensus_record(3, 2, votes={"PROP": 2, "CHECK": 1})
        self.assertEqual(record["method"], CONSENSUS_METHOD_VERSION)
        self.assertEqual(validate_consensus_record(record, "n"), [])
        with self.assertRaises(ValueError):
            consensus_record(3, 4)
        broken = dict(record, support=5)
        self.assertTrue(validate_consensus_record(broken, "n"))
        unknown = dict(record, extra=1)
        self.assertTrue(validate_consensus_record(unknown, "n"))
        self.assertTrue(validate_consensus_record("x", "n"))


def _valid_graph_with_consensus() -> dict:
    text = "Rule. Value. Answer."
    pieces = ["Rule. ", "Value. ", "Answer."]
    types = ["METHOD", "PROP", "OUT"]
    provenance = make_provenance("llm_proposed", "typing")
    segments = []
    nodes = []
    cursor = 0
    for index, (piece, node_type) in enumerate(zip(pieces, types), start=1):
        segment = {
            "segment_id": f"s{index}",
            "text": piece,
            "start_char": cursor,
            "end_char": cursor + len(piece),
            "provenance": make_provenance("llm_proposed", "segmentation"),
        }
        cursor += len(piece)
        segments.append(segment)
        nodes.append(
            {
                **{k: segment[k] for k in ("segment_id", "text", "start_char", "end_char")},
                "position": index,
                "node_type": node_type,
                "review_status": "valid",
                "ambiguity_codes": [],
                "provenance": provenance,
                "consensus": consensus_record(3, 3, votes={node_type: 3}),
            }
        )
    edges = [
        {
            "edge_id": f"e_next_{index}",
            "source": f"s{index}",
            "target": f"s{index + 1}",
            "edge_type": "NEXT",
            "provenance": make_provenance("deterministic", "backbone"),
        }
        for index in (1, 2)
    ]
    edges.append(
        {
            "edge_id": "e_sem_1",
            "source": "s1",
            "target": "s2",
            "edge_type": "LEADS_TO",
            "provenance": make_provenance("llm_proposed", "semantic_edges"),
            "consensus": consensus_record(3, 2),
        }
    )
    from util.sgotgraph_schema import PIPELINE_VERSION, SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "trace_id": "consensus-case",
        "source_record": {},
        "trace_text": text,
        "segments": segments,
        "nodes": nodes,
        "edges": edges,
        "validation": {},
        "audit": {},
    }


class ConsensusContractTests(unittest.TestCase):
    def test_consensus_field_is_optional_and_validated(self) -> None:
        graph = _valid_graph_with_consensus()
        self.assertEqual(validate_record(graph)["status"], "valid")
        graph["nodes"][0]["consensus"]["support"] = 9
        report = validate_record(graph)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("consensus.support" in error for error in report["hard_errors"])
        )
        del graph["nodes"][0]["consensus"]
        graph["edges"][-1]["consensus"] = {"method": "x"}
        self.assertEqual(validate_record(graph)["status"], "invalid")


class TypingConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.typing = load_stage_module("typing_consensus_stage", "02_tipado_gcot.py")

    def _record(self) -> dict:
        text = "Rule. Answer."
        return {
            "trace_id": "typing-consensus",
            "trace_text": text,
            "segments": [
                {"segment_id": "s1", "text": "Rule. ", "start_char": 0, "end_char": 6},
                {"segment_id": "s2", "text": "Answer.", "start_char": 6, "end_char": 13},
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }

    def _draft(self, first: str, second: str, codes: list[str] | None = None):
        return self.typing.TypingDraft(
            trace_id="typing-consensus",
            nodes=[
                self.typing.NodeDraft(segment_id="s1", node_type=first),
                self.typing.NodeDraft(
                    segment_id="s2", node_type=second, ambiguity_codes=codes or []
                ),
            ],
        )

    def test_majority_and_tie_resolution_with_extra_sample(self) -> None:
        # s1: PROP, CHECK, PROP -> PROP no unánime. s2: OUT, PROP, EVID -> empate
        # de tres; la muestra extra (OUT) lo resuelve sin desempate por regla.
        drafts = [
            self._draft("PROP", "OUT"),
            self._draft("CHECK", "PROP", ["INSUFFICIENT_CONTEXT"]),
            self._draft("PROP", "EVID"),
            self._draft("METHOD", "OUT"),
        ]
        client = SimpleNamespace(provider="mock")
        with patch.object(self.typing, "call_model", side_effect=drafts) as calls:
            result = self.typing.type_record(
                client, "mock-model", self._record(), verbose=False, samples=3
            )
        self.assertEqual(calls.call_count, 4)
        s1, s2 = result["nodes"]
        self.assertEqual(s1["node_type"], "PROP")
        self.assertEqual(s1["consensus"]["votes"], {"PROP": 2, "CHECK": 1, "METHOD": 1})
        self.assertEqual(s1["consensus"]["samples"], 4)
        self.assertIn(AMBIGUITY_UNSTABLE, s1["ambiguity_codes"])
        self.assertNotIn(AMBIGUITY_TIE, s1["ambiguity_codes"])
        self.assertEqual(s1["review_status"], "ambiguous")
        self.assertEqual(s2["node_type"], "OUT")
        self.assertNotIn(AMBIGUITY_TIE, s2["ambiguity_codes"])
        self.assertIn("INSUFFICIENT_CONTEXT", s2["ambiguity_codes"])
        audit = result["audit"]["typing"][-1]
        self.assertEqual(audit["requested_samples"], 3)
        self.assertEqual(audit["samples"], 4)
        self.assertEqual(audit["extra_samples"], 1)
        self.assertEqual(audit["unstable_node_count"], 2)
        self.assertEqual(audit["tie_node_count"], 0)

    def test_unresolved_tie_is_marked_and_uses_sampling_order(self) -> None:
        drafts = [
            self._draft("PROP", "OUT"),
            self._draft("PROP", "PROP"),
            self._draft("PROP", "EVID"),
            self._draft("PROP", "CHECK"),
            self._draft("PROP", "ALT"),
        ]
        client = SimpleNamespace(provider="mock")
        with patch.object(self.typing, "call_model", side_effect=drafts) as calls:
            result = self.typing.type_record(
                client, "mock-model", self._record(), verbose=False, samples=3
            )
        self.assertEqual(calls.call_count, 5)
        s1, s2 = result["nodes"]
        self.assertNotIn("consensus", {})
        self.assertTrue(s1["consensus"]["unanimous"])
        self.assertEqual(s1["ambiguity_codes"], [])
        self.assertEqual(s2["node_type"], "OUT")
        self.assertEqual(s2["consensus"]["tie_break"], "first_sample")
        self.assertIn(AMBIGUITY_TIE, s2["ambiguity_codes"])
        self.assertIn(AMBIGUITY_UNSTABLE, s2["ambiguity_codes"])

    def test_single_sample_keeps_legacy_shape(self) -> None:
        client = SimpleNamespace(provider="mock")
        with patch.object(
            self.typing, "call_model", return_value=self._draft("METHOD", "OUT")
        ) as calls:
            result = self.typing.type_record(
                client, "mock-model", self._record(), verbose=False, samples=1
            )
        self.assertEqual(calls.call_count, 1)
        self.assertNotIn("consensus", result["nodes"][0])
        self.assertEqual(result["nodes"][0]["ambiguity_codes"], [])
        self.assertEqual(result["validation"]["typing"]["samples"], 1)
        self.assertIsNone(result["validation"]["typing"]["consensus_method"])

    def test_invalid_sample_is_resampled_not_repaired(self) -> None:
        bad = self.typing.TypingDraft(
            trace_id="typing-consensus",
            nodes=[self.typing.NodeDraft(segment_id="s1", node_type="PROP")],
        )
        client = SimpleNamespace(provider="mock")
        with patch.object(
            self.typing,
            "call_model",
            side_effect=[bad, self._draft("PROP", "OUT")],
        ) as calls:
            result = self.typing.type_record(
                client, "mock-model", self._record(), verbose=False, samples=1
            )
        self.assertEqual(calls.call_count, 2)
        events = result["audit"]["typing"][-1]["sample_events"]
        self.assertEqual([event["validation_status"] for event in events], ["invalid", "valid"])
        with patch.object(self.typing, "call_model", return_value=bad):
            with self.assertRaises(ValueError):
                self.typing.type_record(
                    client, "mock-model", self._record(), verbose=False, samples=1, max_attempts=2
                )


class EdgeConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = load_stage_module("edge_consensus_stage", "05_aristas_semanticas_gcot.py")

    def _graph(self) -> dict:
        graph = _valid_graph_with_consensus()
        for node in graph["nodes"]:
            del node["consensus"]
        graph["nodes"][0]["node_type"] = "PROP"
        graph["edges"] = graph["edges"][:2]
        return graph

    def test_majority_edges_and_empty_inventory_skip(self) -> None:
        calls_by_source: dict[str, int] = {}

        def edge_response(
            _client, _model, current_node, candidate_nodes, _effort, _verbose,
            _prompt, _attempt, _max_attempts,
        ):
            source = current_node["segment_id"]
            sample = calls_by_source.get(source, 0) + 1
            calls_by_source[source] = sample
            options = self.edges._allowed_relation_options(current_node, candidate_nodes)

            def ref(target: str, edge_type: str) -> str:
                return next(
                    option["relation_ref"]
                    for option in options
                    if option["target"] == target and option["edge_type"] == edge_type
                )

            refs: list[str] = []
            if source == "s1":
                if sample in (1, 2):
                    refs.append(ref("s2", "SUPPORTS"))
                if sample == 2:
                    refs.append(ref("s3", "LEADS_TO"))
            return self.edges.NodeRelationsDraft(source=source, relation_refs=refs)

        client = SimpleNamespace(provider="mock")
        with patch.object(self.edges, "call_model", side_effect=per_source(self.edges, edge_response)):
            result = self.edges.build_semantic_edges(
                client,
                "mock-model",
                self._graph(),
                prev_window=None,
                future_window=None,
                reasoning_effort="low",
                verbose=False,
                samples=3,
            )
        # s3 es OUT: sin opciones admisibles no se consulta al LLM.
        self.assertEqual(calls_by_source, {"s1": 3, "s2": 3})
        semantic = [edge for edge in result["edges"] if edge["edge_type"] != "NEXT"]
        self.assertEqual(
            [(e["source"], e["target"], e["edge_type"]) for e in semantic],
            [("s1", "s2", "SUPPORTS")],
        )
        self.assertEqual(semantic[0]["consensus"]["support"], 2)
        self.assertEqual(semantic[0]["consensus"]["samples"], 3)
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual(audit["skipped_empty_inventory"], ["s3"])
        self.assertEqual(audit["samples_by_source"], {"s1": 3, "s2": 3, "s3": 0})
        self.assertEqual(
            audit["minority_relations_by_source"]["s1"],
            [{"source": "s1", "target": "s3", "edge_type": "LEADS_TO", "support": 1, "samples": 3}],
        )
        self.assertEqual(validate_record(result)["status"], "valid")

    def test_single_sample_has_no_consensus_field(self) -> None:
        def edge_response(
            _client, _model, current_node, _candidate_nodes, _effort, _verbose,
            _prompt, _attempt, _max_attempts,
        ):
            return self.edges.NodeRelationsDraft(
                source=current_node["segment_id"], relation_refs=[]
            )

        client = SimpleNamespace(provider="mock")
        with patch.object(self.edges, "call_model", side_effect=per_source(self.edges, edge_response)) as calls:
            result = self.edges.build_semantic_edges(
                client,
                "mock-model",
                self._graph(),
                prev_window=None,
                future_window=None,
                reasoning_effort="low",
                verbose=False,
            )
        # estimador canónico: una fuente por solicitud; s3 (OUT) no se consulta.
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(result["validation"]["semantic_edges"]["request_batch_count"], 2)
        self.assertEqual(result["validation"]["semantic_edges"]["estimator_variant"], "canonical")
        self.assertTrue(all("consensus" not in edge for edge in result["edges"]))
        self.assertIsNone(result["validation"]["semantic_edges"]["consensus_method"])
        self.assertEqual(
            result["validation"]["semantic_edges"]["skipped_empty_inventory_count"], 1
        )


class SegmentationConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = load_stage_module(
            "segmentation_consensus_stage", "01_segmentacion_gcot.py"
        )

    def _response(self, ranges: list[tuple[str, str]]):
        return self.stage.ParsedSegmentationResponse(
            draft=self.stage.SegmentationDraft(
                trace_id="seg-consensus",
                segments=[
                    self.stage.SegmentRangeDraft(start_unit_id=a, end_unit_id=b)
                    for a, b in ranges
                ],
            ),
            provider="mock",
        )

    def test_cut_vote_keeps_majority_cuts_only(self) -> None:
        trace = "First. Second. Third. Fourth."
        units = self.stage.build_literal_units(trace)
        self.assertEqual([u["unit_id"] for u in units], ["u1", "u2", "u3", "u4"])
        # cortes propuestos: m1 {u1,u3}, m2 {u1}, m3 {u3, u2}; mayoría: u1 y u3.
        responses = [
            self._response([("u1", "u1"), ("u2", "u3"), ("u4", "u4")]),
            self._response([("u1", "u1"), ("u2", "u4")]),
            self._response([("u1", "u2"), ("u3", "u3"), ("u4", "u4")]),
        ]
        client = SimpleNamespace(provider="mock")
        with patch.object(self.stage, "call_model", side_effect=responses) as calls:
            result = self.stage.segment_trace(
                client,
                "mock-model",
                {"id": "seg-consensus", "reasoning_trace": trace},
                max_attempts=2,
                verbose=False,
                reasoning_effort="minimal",
                samples=3,
            )
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["First. ", "Second. Third. ", "Fourth."],
        )
        self.assertEqual("".join(s["text"] for s in result["segments"]), trace)
        audit = result["audit"]["segmentation"][-1]
        self.assertEqual(audit["samples"], 3)
        self.assertEqual(audit["cut_support"], {"u1": 2, "u2": 1, "u3": 2})
        self.assertEqual(audit["cut_threshold"], 2)
        self.assertEqual(audit["segment_count_by_sample"], [3, 2, 3])
        self.assertEqual(result["validation"]["segmentation"]["samples"], 3)

    def test_single_sample_keeps_legacy_ranges(self) -> None:
        trace = "First. Second."
        client = SimpleNamespace(provider="mock")
        with patch.object(
            self.stage,
            "call_model",
            return_value=self._response([("u1", "u2")]),
        ) as calls:
            result = self.stage.segment_trace(
                client,
                "mock-model",
                {"id": "seg-consensus", "reasoning_trace": trace},
                max_attempts=2,
                verbose=False,
                reasoning_effort="minimal",
            )
        self.assertEqual(calls.call_count, 1)
        self.assertEqual([s["text"] for s in result["segments"]], [trace])
        self.assertIsNone(result["audit"]["segmentation"][-1]["cut_support"])


class RefinementConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = load_stage_module(
            "refinement_consensus_stage", "01_1_refinamiento_segmentacion_gcot.py"
        )

    def _draft(self, ranges: list[tuple[str, str]]):
        return self.stage.RefinementDraft(
            segment_id="s1",
            decision="KEEP" if len(ranges) == 1 else "SPLIT",
            subsegment_ranges=[
                self.stage.SubsegmentRangeDraft(start_unit_id=a, end_unit_id=b)
                for a, b in ranges
            ],
        )

    def test_split_survives_only_with_majority(self) -> None:
        text = "Now I subtract five. Then I get nineteen."
        record = {
            "trace_id": "ref-consensus",
            "trace_text": text,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "provenance": make_provenance("llm_proposed", "segmentation"),
                }
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        units = self.stage.build_refinement_literal_units(text)
        last = units[-1]["unit_id"]
        # u4 = "five. " -> corte tras u4 en dos de tres muestras; u2 solo en una.
        drafts = [
            self._draft([("u1", last)]),
            self._draft([("u1", "u4"), ("u5", last)]),
            self._draft([("u1", "u2"), ("u3", "u4"), ("u5", last)]),
        ]
        client = SimpleNamespace(provider="mock")
        with patch.object(self.stage, "call_model", side_effect=drafts) as calls:
            result = self.stage.refine_record(
                client,
                "mock-model",
                record,
                reasoning_effort="low",
                min_length=0,
                max_subsegments=8,
                verbose=False,
                max_attempts=2,
                samples=3,
            )
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["Now I subtract five. ", "Then I get nineteen."],
        )
        audit = result["audit"]["segmentation_refinement"][-1]
        self.assertEqual(audit["samples"], 3)
        self.assertEqual(
            audit["decisions_by_sample_by_segment"]["s1"], ["KEEP", "SPLIT", "SPLIT"]
        )
        self.assertEqual(
            audit["cut_support_by_segment"]["s1"],
            {"support": {"u2": 1, "u4": 2}, "threshold": 2},
        )
        self.assertEqual(result["validation"]["segmentation_refinement"]["split_segments"], 1)


if __name__ == "__main__":
    unittest.main()
