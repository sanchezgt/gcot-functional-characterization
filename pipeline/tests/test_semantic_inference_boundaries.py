from __future__ import annotations

from copy import deepcopy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.sgotgraph_schema import make_provenance


ROOT = PIPELINE_ROOT


def load_script(name: str, filename: str):
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


class SemanticDecisionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.segmentation = load_script(
            "segmentation_stage", "01_segmentacion_gcot.py"
        )
        cls.refinement = load_script(
            "refinement_stage", "01_1_refinamiento_segmentacion_gcot.py"
        )
        cls.typing = load_script("typing_stage", "02_tipado_gcot.py")
        cls.typing_validation = load_script(
            "typing_validation_stage", "03_validacion_tipado_gcot.py"
        )
        cls.backbone = load_script("backbone_stage", "04_backbone_next_gcot.py")
        cls.edges = load_script("edge_stage", "05_aristas_semanticas_gcot.py")

    def test_prompts_do_not_embed_development_dataset_entities(self) -> None:
        prompts = "\n".join(
            [
                self.segmentation.SYSTEM_PROMPT,
                self.refinement.SYSTEM_PROMPT,
                self.typing.SYSTEM_PROMPT,
                self.edges.SYSTEM_PROMPT,
            ]
        ).lower()
        for term in ("maria", "pencils", "muffins", "bakery", "bottle"):
            with self.subTest(term=term):
                self.assertNotIn(term, prompts)

    def test_refinement_reviews_every_nonempty_segment(self) -> None:
        self.assertTrue(
            self.refinement.is_candidate_segment({"text": "x"}, min_length=999)
        )
        self.assertFalse(
            self.refinement.is_candidate_segment({"text": ""}, min_length=0)
        )

    def test_refinement_does_not_cap_semantically_justified_splits(self) -> None:
        segment = {"segment_id": "s1", "text": "a b c d"}
        units = self.refinement.build_refinement_literal_units(
            segment["text"]
        )
        draft = self.refinement.RefinementDraft(
            segment_id="s1",
            decision="SPLIT",
            subsegment_ranges=[
                self.refinement.SubsegmentRangeDraft(
                    start_unit_id=unit["unit_id"],
                    end_unit_id=unit["unit_id"],
                )
                for unit in units
            ],
        )
        self.assertEqual(
            self.refinement.validate_refinement(
                segment, draft, max_subsegments=2
            ),
            [],
        )

    def test_refinement_repairs_invalid_literal_ranges(self) -> None:
        trace = "State S. Apply transformation T."
        provenance = make_provenance("llm_proposed", "segmentation")
        units = self.refinement.build_refinement_literal_units(trace)
        record = {
            "trace_id": "repair-case",
            "trace_text": trace,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": trace,
                    "start_char": 0,
                    "end_char": len(trace),
                    "provenance": provenance,
                }
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        invalid = self.refinement.RefinementDraft(
            segment_id="s1",
            decision="SPLIT",
            subsegment_ranges=[
                self.refinement.SubsegmentRangeDraft(
                    start_unit_id=units[1]["unit_id"],
                    end_unit_id=units[1]["unit_id"],
                ),
                self.refinement.SubsegmentRangeDraft(
                    start_unit_id=units[2]["unit_id"],
                    end_unit_id=units[-1]["unit_id"],
                ),
            ],
        )
        repaired = self.refinement.RefinementDraft(
            segment_id="s1",
            decision="SPLIT",
            subsegment_ranges=[
                self.refinement.SubsegmentRangeDraft(
                    start_unit_id=units[0]["unit_id"],
                    end_unit_id=units[1]["unit_id"],
                ),
                self.refinement.SubsegmentRangeDraft(
                    start_unit_id=units[2]["unit_id"],
                    end_unit_id=units[-1]["unit_id"],
                ),
            ],
        )
        client = SimpleNamespace(provider="mock")

        with patch.object(
            self.refinement,
            "call_model",
            side_effect=[invalid, repaired],
        ):
            result = self.refinement.refine_record(
                client,
                "mock-model",
                record,
                reasoning_effort="low",
                min_length=0,
                max_subsegments=1,
                verbose=False,
                max_attempts=2,
            )

        self.assertEqual(
            [segment["text"] for segment in result["segments"]],
            ["State S. ", "Apply transformation T."],
        )
        validation = result["validation"]["segmentation_refinement"]
        self.assertEqual(validation["attempts_by_segment"], {"s1": 2})
        self.assertEqual(
            len(result["audit"]["segmentation_refinement"][-1]["requests"]),
            2,
        )

    def test_refinement_resumes_after_the_last_accepted_segment(
        self,
    ) -> None:
        trace = "First step. Second step."
        boundary = len("First step. ")
        provenance = make_provenance(
            "llm_proposed",
            "segmentation",
        )
        record = {
            "trace_id": "refinement-checkpoint",
            "trace_text": trace,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": trace[:boundary],
                    "start_char": 0,
                    "end_char": boundary,
                    "provenance": provenance,
                },
                {
                    "segment_id": "s2",
                    "text": trace[boundary:],
                    "start_char": boundary,
                    "end_char": len(trace),
                    "provenance": provenance,
                },
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        client = SimpleNamespace(provider="mock")

        def keep(segment):
            units = self.refinement.build_refinement_literal_units(
                segment["text"]
            )
            return self.refinement.RefinementDraft(
                segment_id=segment["segment_id"],
                decision="KEEP",
                subsegment_ranges=[
                    self.refinement.SubsegmentRangeDraft(
                        start_unit_id=units[0]["unit_id"],
                        end_unit_id=units[-1]["unit_id"],
                    )
                ],
            )

        def interrupted(
            _client,
            _model,
            _trace_id,
            segment,
            _effort,
            _verbose,
            **_kwargs,
        ):
            if segment["segment_id"] == "s2":
                raise TimeoutError("simulated timeout")
            return keep(segment)

        def resumed(
            _client,
            _model,
            _trace_id,
            segment,
            _effort,
            _verbose,
            **_kwargs,
        ):
            return keep(segment)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            with patch.object(
                self.refinement,
                "call_model",
                side_effect=interrupted,
            ):
                with self.assertRaises(TimeoutError):
                    self.refinement.refine_record(
                        client,
                        "mock-model",
                        deepcopy(record),
                        reasoning_effort="low",
                        min_length=0,
                        max_subsegments=1,
                        verbose=False,
                        max_attempts=1,
                        checkpoint_dir=checkpoint_dir,
                    )

            with patch.object(
                self.refinement,
                "call_model",
                side_effect=resumed,
            ) as resumed_call:
                result = self.refinement.refine_record(
                    client,
                    "mock-model",
                    deepcopy(record),
                    reasoning_effort="low",
                    min_length=0,
                    max_subsegments=1,
                    verbose=False,
                    max_attempts=1,
                    checkpoint_dir=checkpoint_dir,
                    resume_checkpoint=True,
                )

        self.assertEqual(resumed_call.call_count, 1)
        self.assertEqual(
            result["validation"]["segmentation_refinement"][
                "checkpoint"
            ]["reused_segment_ids"],
            ["s1"],
        )
        self.assertEqual(
            "".join(segment["text"] for segment in result["segments"]),
            trace,
        )

    def test_typing_prompt_has_no_fixed_category_priority(self) -> None:
        self.assertNotIn("Disambiguation priority", self.typing.SYSTEM_PROMPT)
        self.assertIn("no category has a general priority", self.typing.SYSTEM_PROMPT)

    def test_typing_output_passes_the_following_structural_stage(self) -> None:
        text = "First claim. Final answer."
        first_end = len("First claim. ")
        provenance = make_provenance("deterministic", "segmentation")
        record = {
            "trace_id": "typed-case",
            "trace_text": text,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": text[:first_end],
                    "start_char": 0,
                    "end_char": first_end,
                    "provenance": provenance,
                },
                {
                    "segment_id": "s2",
                    "text": text[first_end:],
                    "start_char": first_end,
                    "end_char": len(text),
                    "provenance": provenance,
                },
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        draft = self.typing.TypingDraft(
            trace_id="typed-case",
            nodes=[
                self.typing.NodeDraft(
                    segment_id="s1", node_type="PROP", ambiguity_codes=[]
                ),
                self.typing.NodeDraft(
                    segment_id="s2", node_type="OUT", ambiguity_codes=[]
                ),
            ],
        )
        client = SimpleNamespace(provider="mock")
        with patch.object(self.typing, "call_model", return_value=draft):
            typed = self.typing.type_record(client, "mock-model", record, False)

        self.assertEqual(
            [node["position"] for node in typed["nodes"]],
            [1, 2],
        )
        reviewed = self.typing_validation.validate_record(typed)
        self.assertEqual(
            reviewed["validation"]["typing_review"]["status"],
            "valid",
        )

    def test_typing_validation_is_structural_not_lexical(self) -> None:
        text = "check verify because alternative"
        provenance = make_provenance("llm_proposed", "typing")
        record = {
            "trace_id": "case",
            "trace_text": text,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "provenance": provenance,
                }
            ],
            "nodes": [
                {
                    "segment_id": "s1",
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "position": 1,
                    "node_type": "PROP",
                    "review_status": "valid",
                    "ambiguity_codes": [],
                    "provenance": provenance,
                }
            ],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        result = self.typing_validation.validate_record(record)
        review = result["validation"]["typing_review"]
        self.assertEqual(review["status"], "valid")
        self.assertEqual(review["validation_kind"], "structural_only")
        self.assertEqual(result["nodes"][0]["ambiguity_codes"], [])

    def test_edge_options_materialize_only_contract_compatible_pairs(self) -> None:
        current = {
            "segment_id": "s1",
            "node_type": "PROP",
            "position": 1,
        }
        candidates = [
            {
                "segment_id": "s2",
                "node_type": "METHOD",
                "position": 2,
            },
            {
                "segment_id": "s3",
                "node_type": "OUT",
                "position": 3,
            },
        ]

        options = self.edges._allowed_relation_options(current, candidates)

        option_triples = {
            (option["source"], option["edge_type"], option["target"])
            for option in options
        }
        self.assertNotIn(
            ("s1", "SUPPORTS", "s2"),
            option_triples,
        )
        self.assertIn(
            ("s1", "SUPPORTS", "s3"),
            option_triples,
        )
        self.assertIn(
            ("s1", "LEADS_TO", "s3"),
            option_triples,
        )
        # alias planos en el transporte canónico; con prefijo de fuente solo en
        # la carga compacta, donde un alias ajeno debe caer fuera del inventario
        self.assertEqual(
            [option["relation_ref"] for option in options],
            [
                f"R{index:04d}"
                for index in range(1, len(options) + 1)
            ],
        )

    def test_edge_stage_repairs_unknown_relation_reference(self) -> None:
        text = "Claim. Method. Answer."
        spans = [(0, 7), (7, 15), (15, len(text))]
        node_types = ["PROP", "METHOD", "OUT"]
        provenance = make_provenance("llm_proposed", "typing")
        nodes = []
        segments = []
        for index, ((start, end), node_type) in enumerate(
            zip(spans, node_types),
            start=1,
        ):
            segment_id = f"s{index}"
            segment = {
                "segment_id": segment_id,
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
                "provenance": provenance,
            }
            segments.append(segment)
            nodes.append(
                {
                    **segment,
                    "position": index,
                    "node_type": node_type,
                    "review_status": "valid",
                    "ambiguity_codes": [],
                }
            )
        record = {
            "trace_id": "edge-repair",
            "trace_text": text,
            "segments": segments,
            "nodes": nodes,
            "edges": [
                {
                    "edge_id": "e_next_1",
                    "source": "s1",
                    "target": "s2",
                    "edge_type": "NEXT",
                    "provenance": make_provenance(
                        "deterministic", "backbone"
                    ),
                },
                {
                    "edge_id": "e_next_2",
                    "source": "s2",
                    "target": "s3",
                    "edge_type": "NEXT",
                    "provenance": make_provenance(
                        "deterministic", "backbone"
                    ),
                },
            ],
            "validation": {},
            "audit": {},
        }
        client = SimpleNamespace(provider="mock")

        def edge_response(
            _client,
            _model,
            current_node,
            _candidate_nodes,
            _effort,
            _verbose,
            _user_prompt,
            attempt,
            _max_attempts,
        ):
            if current_node["segment_id"] == "s1" and attempt == 1:
                return self.edges.NodeRelationsDraft(
                    source="s1",
                    relation_refs=["R9999"],
                )
            if current_node["segment_id"] == "s1":
                options = self.edges._allowed_relation_options(
                    current_node,
                    _candidate_nodes,
                )
                selected = next(
                    option["relation_ref"]
                    for option in options
                    if option["target"] == "s3"
                    and option["edge_type"] == "SUPPORTS"
                )
                return self.edges.NodeRelationsDraft(
                    source="s1",
                    relation_refs=[selected],
                )
            return self.edges.NodeRelationsDraft(
                source=current_node["segment_id"],
                relation_refs=[],
            )

        with patch.object(self.edges, "call_model", side_effect=per_source(self.edges, edge_response)):
            result = self.edges.build_semantic_edges(
                client,
                "mock-model",
                record,
                prev_window=None,
                future_window=None,
                reasoning_effort="low",
                verbose=False,
                max_attempts=2,
            )

        semantic = [
            edge for edge in result["edges"] if edge["edge_type"] != "NEXT"
        ]
        self.assertEqual(
            [
                (edge["source"], edge["edge_type"], edge["target"])
                for edge in semantic
            ],
            [("s1", "SUPPORTS", "s3")],
        )
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual(
            audit["transport"],
            self.edges.RELATION_OPTION_TRANSPORT_VERSION,
        )
        self.assertEqual(
            result["validation"]["semantic_edges"]["transport"],
            self.edges.RELATION_OPTION_TRANSPORT_VERSION,
        )
        self.assertEqual(audit["attempts_by_source"]["s1"], 2)
        self.assertEqual(
            audit["attempt_history_by_source"]["s1"][0][
                "validation_status"
            ],
            "invalid",
        )

    def test_edge_stage_resumes_after_the_last_accepted_source(
        self,
    ) -> None:
        # Dos PROP: ambos tienen opciones admisibles (SUPPORTS en ambas
        # direcciones), así que el segundo sí llega a consultar al LLM.
        text = "Method. Answer."
        boundary = len("Method. ")
        provenance = make_provenance("llm_proposed", "typing")
        nodes = [
            {
                "segment_id": "s1",
                "text": text[:boundary],
                "start_char": 0,
                "end_char": boundary,
                "position": 1,
                "node_type": "PROP",
                "review_status": "valid",
                "ambiguity_codes": [],
                "provenance": provenance,
            },
            {
                "segment_id": "s2",
                "text": text[boundary:],
                "start_char": boundary,
                "end_char": len(text),
                "position": 2,
                "node_type": "PROP",
                "review_status": "valid",
                "ambiguity_codes": [],
                "provenance": provenance,
            },
        ]
        record = {
            "trace_id": "edge-checkpoint",
            "trace_text": text,
            "segments": [
                {
                    key: node[key]
                    for key in (
                        "segment_id",
                        "text",
                        "start_char",
                        "end_char",
                        "provenance",
                    )
                }
                for node in nodes
            ],
            "nodes": nodes,
            "edges": [
                {
                    "edge_id": "e_next",
                    "source": "s1",
                    "target": "s2",
                    "edge_type": "NEXT",
                    "provenance": make_provenance(
                        "deterministic",
                        "backbone",
                    ),
                }
            ],
            "validation": {},
            "audit": {},
        }
        client = SimpleNamespace(provider="mock")

        def interrupted(
            _client,
            _model,
            current_node,
            _candidate_nodes,
            _effort,
            _verbose,
            _user_prompt,
            _attempt,
            _max_attempts,
        ):
            if current_node["segment_id"] == "s2":
                raise TimeoutError("simulated timeout")
            return self.edges.NodeRelationsDraft(
                source=current_node["segment_id"],
                relation_refs=[],
            )

        def resumed(
            _client,
            _model,
            current_node,
            _candidate_nodes,
            _effort,
            _verbose,
            _user_prompt,
            _attempt,
            _max_attempts,
        ):
            return self.edges.NodeRelationsDraft(
                source=current_node["segment_id"],
                relation_refs=[],
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            with patch.object(
                self.edges,
                "call_model",
                side_effect=per_source(self.edges, interrupted),
            ):
                with self.assertRaises(TimeoutError):
                    self.edges.build_semantic_edges(
                        client,
                        "mock-model",
                        deepcopy(record),
                        prev_window=None,
                        future_window=None,
                        reasoning_effort="low",
                        verbose=False,
                        max_attempts=1,
                        checkpoint_dir=checkpoint_dir,
                        sources_per_request=1,
                    )

            with patch.object(
                self.edges,
                "call_model",
                side_effect=per_source(self.edges, resumed),
            ) as resumed_call:
                result = self.edges.build_semantic_edges(
                    client,
                    "mock-model",
                    deepcopy(record),
                    prev_window=None,
                    future_window=None,
                    reasoning_effort="low",
                    verbose=False,
                    max_attempts=1,
                    checkpoint_dir=checkpoint_dir,
                    resume_checkpoint=True,
                    sources_per_request=1,
                )

        self.assertEqual(resumed_call.call_count, 1)
        self.assertEqual(
            result["validation"]["semantic_edges"]["checkpoint"][
                "reused_source_ids"
            ],
            ["s1"],
        )
        self.assertEqual(
            [edge["edge_type"] for edge in result["edges"]],
            ["NEXT"],
        )

    def test_mocked_objective2_stages_form_one_valid_chain(self) -> None:
        trace = "Apply rule. Answer A."
        raw = {"id": "chain-case", "reasoning_trace": trace}
        client = SimpleNamespace(provider="mock")
        segmentation_response = self.segmentation.ParsedSegmentationResponse(
            draft=self.segmentation.SegmentationDraft(
                trace_id="chain-case",
                segments=[
                    self.segmentation.SegmentRangeDraft(
                        start_unit_id="u1",
                        end_unit_id="u1",
                    ),
                    self.segmentation.SegmentRangeDraft(
                        start_unit_id="u2",
                        end_unit_id="u2",
                    ),
                ],
            ),
            provider="mock",
            model="mock-model",
            reasoning_effort="low",
        )
        with patch.object(
            self.segmentation,
            "call_model",
            return_value=segmentation_response,
        ):
            segmented = self.segmentation.segment_trace(
                client,
                "mock-model",
                raw,
                max_attempts=1,
                verbose=False,
                reasoning_effort="low",
            )

        def keep_segment(_client, _model, _trace_id, segment, _effort, _verbose):
            units = self.refinement.build_refinement_literal_units(
                segment["text"]
            )
            return self.refinement.RefinementDraft(
                segment_id=segment["segment_id"],
                decision="KEEP",
                subsegment_ranges=[
                    self.refinement.SubsegmentRangeDraft(
                        start_unit_id=units[0]["unit_id"],
                        end_unit_id=units[-1]["unit_id"],
                    )
                ],
            )

        with patch.object(
            self.refinement,
            "call_model",
            side_effect=keep_segment,
        ):
            refined = self.refinement.refine_record(
                client,
                "mock-model",
                segmented,
                reasoning_effort="low",
                min_length=0,
                max_subsegments=1,
                verbose=False,
            )

        typing_draft = self.typing.TypingDraft(
            trace_id="chain-case",
            nodes=[
                self.typing.NodeDraft(
                    segment_id="s1", node_type="METHOD", ambiguity_codes=[]
                ),
                self.typing.NodeDraft(
                    segment_id="s2", node_type="OUT", ambiguity_codes=[]
                ),
            ],
        )
        with patch.object(self.typing, "call_model", return_value=typing_draft):
            typed = self.typing.type_record(
                client, "mock-model", refined, False
            )
        reviewed = self.typing_validation.validate_record(typed)
        self.assertEqual(
            reviewed["validation"]["typing_review"]["status"], "valid"
        )
        graph = self.backbone.build_backbone(reviewed)

        def edge_response(
            _client,
            _model,
            current_node,
            _candidate_nodes,
            _effort,
            _verbose,
            _user_prompt,
            _attempt,
            _max_attempts,
        ):
            relation_refs = []
            if current_node["segment_id"] == "s1":
                options = self.edges._allowed_relation_options(
                    current_node,
                    _candidate_nodes,
                )
                relation_refs = [
                    next(
                        option["relation_ref"]
                        for option in options
                        if option["target"] == "s2"
                        and option["edge_type"] == "LEADS_TO"
                    )
                ]
            return self.edges.NodeRelationsDraft(
                source=current_node["segment_id"],
                relation_refs=relation_refs,
            )

        with patch.object(self.edges, "call_model", side_effect=per_source(self.edges, edge_response)):
            final = self.edges.build_semantic_edges(
                client,
                "mock-model",
                graph,
                prev_window=0,
                future_window=0,
                reasoning_effort="low",
                verbose=False,
            )
        from util.sgotgraph_schema import validate_record

        report = validate_record(final)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["hard_errors"], [])
        self.assertEqual(final["validation"]["semantic_edges"]["status"], "valid")

    def test_edge_candidates_are_complete_even_with_zero_windows(self) -> None:
        nodes = [
            {
                "segment_id": f"s{index}",
                "node_type": "PROP",
                "position": index,
                "start_char": index - 1,
                "end_char": index,
                "text": str(index),
            }
            for index in range(1, 5)
        ]
        candidates = self.edges.select_candidate_nodes(
            nodes, current_index=2, prev_window=0, future_window=0
        )
        self.assertEqual(
            {item["segment_id"] for item in candidates},
            {"s1", "s2", "s4"},
        )

    def test_edge_stage_has_no_priority_or_cardinality_selector(self) -> None:
        self.assertFalse(hasattr(self.edges, "SEMANTIC_PRIORITY"))
        self.assertFalse(hasattr(self.edges, "MAX_RELATIONS_BY_TYPE"))
        self.assertFalse(hasattr(self.edges, "filter_node_relations"))


if __name__ == "__main__":
    unittest.main()
