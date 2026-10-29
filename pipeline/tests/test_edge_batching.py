"""Etapa 05: transporte canónico (una fuente por solicitud, carga completa, prompts
de referencia) y variantes (carga compacta, lotes,
ventana): contrato del lote, aceptación y reparación por fuente, consenso y
reanudación parcial. Sin API."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.edge_batching import (  # noqa: E402
    candidate_policy_label,
    chunk_sources,
    estimator_variant,
    validate_batch_offset,
    validate_candidate_window,
    validate_sources_per_request,
    validate_transport_combination,
)
from util.llm_checkpoint import checkpoint_path_for_trace  # noqa: E402
from util.llm_provider import structured_request_hash  # noqa: E402
from util.sgotgraph_schema import make_provenance, validate_record  # noqa: E402

# Huellas de referencia calculadas sobre
# graph_with_types(["METHOD", "PROP", "CHECK", "PROP", "OUT"]): el estimador
# canónico debe seguir emitiendo exactamente esos prompts.
GOLDEN_PROMPT_HASH_S2 = "sha256:1f0a2b15cda8cc879064ea3db1b042046cf0378019cd9d1edd5e0109bf149cad"
GOLDEN_PROMPT_HASH_S3 = "sha256:6970cad8c5d46d2fc658ba25688175f65a10b027ba46ff611790083dc9ebd07c"
GOLDEN_REPAIR_HASH_S2 = "sha256:42b70e82d4633298e03994131d20cf4c08996a41ce18862a407157cacf1f79b0"


def load_stage(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def graph_with_types(types: list[str], trace_id: str = "batch-case") -> dict:
    pieces = [f"Step {index}. " for index in range(1, len(types) + 1)]
    text = "".join(pieces)
    provenance = make_provenance("llm_proposed", "typing")
    segments, nodes = [], []
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
        for index in range(1, len(types))
    ]
    from util.sgotgraph_schema import PIPELINE_VERSION, SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "trace_id": trace_id,
        "source_record": {},
        "trace_text": text,
        "segments": segments,
        "nodes": nodes,
        "edges": edges,
        "validation": {},
        "audit": {},
    }


class BatchingHelpersTests(unittest.TestCase):
    def test_chunking_offset_and_option_cap(self) -> None:
        self.assertEqual(chunk_sources([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertEqual(chunk_sources([], 3), [])
        self.assertEqual(chunk_sources([1, 2, 3, 4, 5], 3, offset=1), [[1, 2], [3, 4, 5]])
        weights = {1: 40, 2: 40, 3: 40, 4: 5}
        self.assertEqual(
            chunk_sources([1, 2, 3, 4], 6, max_options=90, weight=lambda x: weights[x]),
            [[1, 2], [3, 4]],
        )
        # un elemento que solo excede el tope forma su propio bloque
        self.assertEqual(chunk_sources([1], 6, max_options=10, weight=lambda x: 500), [[1]])
        with self.assertRaises(ValueError):
            validate_sources_per_request(0)
        with self.assertRaises(ValueError):
            validate_batch_offset(3, 3)
        self.assertIsNone(validate_candidate_window(None))
        with self.assertRaises(ValueError):
            validate_candidate_window(0)
        self.assertEqual(candidate_policy_label(None), "all_other_nodes")
        self.assertEqual(candidate_policy_label(4), "window:4")

    def test_only_full_payload_per_source_without_window_is_canonical(self) -> None:
        self.assertEqual(estimator_variant(None), "canonical")
        self.assertEqual(estimator_variant(None, 1, "full"), "canonical")
        self.assertEqual(estimator_variant(None, 1, "compact"), "compact")
        self.assertEqual(estimator_variant(None, 6, "compact"), "compact_batched")
        self.assertEqual(estimator_variant(4, 1, "compact"), "compact_windowed")
        self.assertEqual(estimator_variant(4, 6, "compact"), "compact_batched_windowed")
        validate_transport_combination("full", 1, None)
        validate_transport_combination("compact", 6, 4)
        with self.assertRaises(ValueError):
            validate_transport_combination("full", 6, None)
        with self.assertRaises(ValueError):
            validate_transport_combination("full", 1, 4)
        with self.assertRaises(ValueError):
            validate_transport_combination("dense", 1, None)


class CanonicalTransportTests(unittest.TestCase):
    """El transporte por defecto reproduce byte a byte las huellas de referencia."""

    def setUp(self) -> None:
        self.edges = load_stage("edge_stage_canonical", "05_aristas_semanticas_gcot.py")
        self.graph = graph_with_types(["METHOD", "PROP", "CHECK", "PROP", "OUT"])
        self.client = SimpleNamespace(provider="openai")

    def test_canonical_prompts_match_the_july_instrument(self) -> None:
        nodes = self.graph["nodes"]
        for index, golden in ((1, GOLDEN_PROMPT_HASH_S2), (2, GOLDEN_PROMPT_HASH_S3)):
            prompt = self.edges.build_prompt(
                nodes[index], self.edges.select_candidate_nodes(nodes, index)
            )
            self.assertEqual(
                structured_request_hash(self.client, self.edges.SYSTEM_PROMPT, prompt), golden
            )
        options = self.edges._allowed_relation_options(
            nodes[1], self.edges.select_candidate_nodes(nodes, 1)
        )
        self.assertEqual(options[0]["relation_ref"], "R0001")
        self.assertEqual(self.edges.RELATION_OPTION_TRANSPORT_VERSION, "relation-option-refs-v1")

    def test_default_run_is_canonical_per_source_with_the_july_repair_prompt(self) -> None:
        calls: list[tuple[list[str], str, str]] = []

        def fake(_client, _model, batch, _nodes, _effort, _verbose, prompt, attempt, _max):
            (item,) = batch
            source = item["node"]["segment_id"]
            calls.append(([source], item["payload"], prompt))
            refs = ["R9999"] if source == "s2" and attempt == 1 else []
            return self.edges.BatchRelationsDraft(
                sources=[self.edges.NodeRelationsDraft(source=source, relation_refs=refs)]
            )

        with patch.object(self.edges, "call_model", side_effect=fake):
            result = self.edges.build_semantic_edges(
                self.client, "mock-model", self.graph, None, None, "low", False, max_attempts=2
            )
        # s5 (OUT) no se consulta; s2 se repara una vez: 4 fuentes + 1 reparación
        self.assertEqual([c[0] for c in calls], [["s1"], ["s2"], ["s2"], ["s3"], ["s4"]])
        self.assertTrue(all(payload == "full" for _, payload, _ in calls))
        hashes = [structured_request_hash(self.client, self.edges.SYSTEM_PROMPT, c[2]) for c in calls]
        self.assertEqual(hashes[1], GOLDEN_PROMPT_HASH_S2)
        self.assertEqual(hashes[2], GOLDEN_REPAIR_HASH_S2)
        self.assertEqual(hashes[3], GOLDEN_PROMPT_HASH_S3)
        validation = result["validation"]["semantic_edges"]
        self.assertEqual(validation["estimator_variant"], "canonical")
        self.assertEqual(validation["payload"], "full")
        self.assertEqual(validation["transport"], "relation-option-refs-v1")
        self.assertEqual(validation["prompt_hashes_by_source"]["s2"], GOLDEN_PROMPT_HASH_S2)
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual(audit["repaired_sources"], ["s2"])
        self.assertEqual(audit["skipped_empty_inventory"], ["s5"])
        self.assertEqual(validate_record(result)["status"], "valid")

    def test_full_payload_rejects_batches_and_windows(self) -> None:
        for kwargs in ({"sources_per_request": 2}, {"candidate_window": 2}):
            with self.assertRaises(ValueError):
                self.edges.build_semantic_edges(
                    self.client, "mock-model", deepcopy(self.graph), None, None, "low", False, **kwargs
                )

    def test_canonical_call_requests_the_single_node_response_model(self) -> None:
        nodes = self.graph["nodes"]
        item = self.edges.build_source_item(nodes[1], self.edges.select_candidate_nodes(nodes, 1))
        seen: dict = {}

        def fake_request(_client, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                parsed=self.edges.NodeRelationsDraft(source="s2", relation_refs=[])
            )

        with patch.object(self.edges, "request_structured_response", side_effect=fake_request):
            draft = self.edges.call_model(self.client, "mock-model", [item], nodes, "low", False)
        self.assertIs(seen["response_model"], self.edges.NodeRelationsDraft)
        self.assertEqual(seen["system_prompt"], self.edges.SYSTEM_PROMPT)
        self.assertEqual(
            structured_request_hash(self.client, seen["system_prompt"], seen["user_prompt"]),
            GOLDEN_PROMPT_HASH_S2,
        )
        self.assertEqual([entry.source for entry in draft.sources], ["s2"])


class BatchPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = load_stage("edge_batching_stage_prompt", "05_aristas_semanticas_gcot.py")

    def test_compact_prompt_carries_one_inventory_per_source(self) -> None:
        graph = graph_with_types(["METHOD", "PROP", "CHECK", "PROP", "OUT"])
        nodes = graph["nodes"]
        items = [
            self.edges.build_source_item(
                nodes[index], self.edges.select_candidate_nodes(nodes, index), "compact"
            )
            for index in (0, 1, 2)
        ]
        prompt = self.edges.build_batch_prompt(items, nodes)
        # el contexto completo aparece una vez; los offsets y campos por fuente no viajan
        self.assertEqual(prompt.count('"segment_id":"s5"'), 1)
        self.assertNotIn("start_char", prompt)
        self.assertNotIn("relative_position", prompt)
        # inventario denso por tipo, con alias prefijados por la fuente
        self.assertIn('"LEADS_TO":[["s1:R0001","s2"]', prompt)
        self.assertIn('"CHECKS":[["s3:R0001","s1"],["s3:R0002","s2"]]', prompt)
        self.assertNotIn('"source":', prompt.split("All nodes of the trace")[0])

    def test_window_restricts_candidates(self) -> None:
        graph = graph_with_types(["PROP"] * 7)
        nodes = graph["nodes"]
        near = self.edges.select_candidate_nodes(nodes, 3, candidate_window=1)
        self.assertEqual([n["segment_id"] for n in near], ["s3", "s5"])
        self.assertEqual(len(self.edges.select_candidate_nodes(nodes, 3)), 6)
        with self.assertRaises(ValueError):
            self.edges.select_candidate_nodes(nodes, 3, candidate_window=0)


class BatchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = load_stage("edge_batching_stage_validation", "05_aristas_semanticas_gcot.py")
        graph = graph_with_types(["PROP", "PROP", "PROP"])
        self.nodes = graph["nodes"]
        self.batch = [
            self.edges.build_source_item(
                self.nodes[i], self.edges.select_candidate_nodes(self.nodes, i), "compact"
            )
            for i in (0, 1)
        ]

    def _draft(self, entries):
        return self.edges.BatchRelationsDraft(
            sources=[self.edges.NodeRelationsDraft(source=s, relation_refs=r) for s, r in entries]
        )

    def test_batch_contract_is_per_source(self) -> None:
        ok = self._draft([("s1", ["s1:R0001"]), ("s2", [])])
        self.assertEqual(self.edges.validate_batch_relations(self.batch, ok), [])
        reordered = self._draft([("s2", []), ("s1", [])])
        self.assertEqual(self.edges.validate_batch_relations(self.batch, reordered), [])
        by_source, notes = self.edges.validate_batch_by_source(self.batch, self._draft([("s1", [])]))
        self.assertEqual(by_source["s1"], [])
        self.assertTrue(any("Falta" in e for e in by_source["s2"]))
        by_source, notes = self.edges.validate_batch_by_source(
            self.batch, self._draft([("s1", []), ("s2", []), ("s3", [])])
        )
        self.assertEqual(by_source, {"s1": [], "s2": []})
        self.assertTrue(any("no solicitadas" in n for n in notes))
        by_source, _ = self.edges.validate_batch_by_source(self.batch, self._draft([("s1", []), ("s1", []), ("s2", [])]))
        self.assertTrue(any("repetidas" in e for e in by_source["s1"]))
        self.assertEqual(by_source["s2"], [])
        # alias de otra fuente del lote: error específico, no uno genérico
        by_source, _ = self.edges.validate_batch_by_source(self.batch, self._draft([("s1", ["s2:R0001"]), ("s2", [])]))
        self.assertTrue(any("pertenece a otra fuente" in e for e in by_source["s1"]))
        by_source, _ = self.edges.validate_batch_by_source(self.batch, self._draft([("s1", ["s9:R0001"]), ("s2", [])]))
        self.assertTrue(any("inexistente" in e for e in by_source["s1"]))
        # un alias plano del transporte canónico no pertenece al inventario compacto
        by_source, _ = self.edges.validate_batch_by_source(self.batch, self._draft([("s1", ["R0001"]), ("s2", [])]))
        self.assertTrue(any("inexistente" in e for e in by_source["s1"]))


class BatchInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = load_stage("edge_batching_stage_inference", "05_aristas_semanticas_gcot.py")

    def _fake(self, decide):
        """decide(source_id, options, attempt) -> list[relation_ref]"""

        def fake(_client, _model, batch, _nodes, _effort, _verbose, _prompt, attempt, _max):
            return self.edges.BatchRelationsDraft(
                sources=[
                    self.edges.NodeRelationsDraft(
                        source=item["node"]["segment_id"],
                        relation_refs=decide(item["node"]["segment_id"], item["options"], attempt),
                    )
                    for item in batch
                ]
            )

        return fake

    @staticmethod
    def _ref(options, target, edge_type):
        return next(o["relation_ref"] for o in options if o["target"] == target and o["edge_type"] == edge_type)

    def test_batches_share_requests_and_repair_only_invalid_sources(self) -> None:
        graph = graph_with_types(["METHOD", "PROP", "PROP", "PROP", "OUT"])
        calls: list[list[str]] = []
        prompts: list[str] = []

        def decide(source_id, options, attempt):
            if source_id == "s1" and attempt == 1:
                return ["s2:R0001"]  # alias de otra fuente del lote
            if source_id == "s1":
                return [self._ref(options, "s2", "LEADS_TO")]
            if source_id == "s4":
                return [self._ref(options, "s5", "SUPPORTS")]
            return []

        fake = self._fake(decide)

        def recording(*args, **kwargs):
            calls.append([item["node"]["segment_id"] for item in args[2]])
            prompts.append(args[6])
            return fake(*args, **kwargs)

        client = SimpleNamespace(provider="mock")
        with patch.object(self.edges, "call_model", side_effect=recording):
            result = self.edges.build_semantic_edges(
                client, "mock-model", graph, None, None, "low", False,
                max_attempts=2, sources_per_request=2, payload="compact",
            )
        # s5 (OUT) no se consulta: 4 fuentes -> 2 lotes; solo s1 se repara
        self.assertEqual(calls, [["s1", "s2"], ["s1"], ["s3", "s4"]])
        self.assertIn("Validation errors by source", prompts[1])
        self.assertIn("pertenece a otra fuente", prompts[1])
        semantic = [(e["source"], e["edge_type"], e["target"]) for e in result["edges"] if e["edge_type"] != "NEXT"]
        self.assertEqual(semantic, [("s1", "LEADS_TO", "s2"), ("s4", "SUPPORTS", "s5")])
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual([b["sources"] for b in audit["request_batches"]], [["s1", "s2"], ["s3", "s4"]])
        self.assertEqual(audit["sources_per_request"], 2)
        self.assertEqual(audit["candidate_policy"], "all_other_nodes")
        # agrupar fuentes es una variante explícita, no el estimador canónico
        self.assertEqual(audit["estimator_variant"], "compact_batched")
        self.assertEqual(audit["transport"], "relation-option-refs-v2-batched")
        self.assertEqual(audit["attempts_by_source"], {"s1": 2, "s2": 1, "s3": 1, "s4": 1, "s5": 0})
        self.assertEqual(audit["repaired_sources"], ["s1"])
        self.assertEqual(audit["cross_source_alias_errors"], 1)
        self.assertEqual(audit["skipped_empty_inventory"], ["s5"])
        # las fuentes de un lote comparten el hash del prompt base
        self.assertEqual(audit["prompt_hashes_by_source"]["s1"], audit["prompt_hashes_by_source"]["s2"])
        self.assertEqual(set(audit["prompt_hashes_by_source"]), {"s1", "s2", "s3", "s4"})
        self.assertEqual(result["validation"]["semantic_edges"]["request_batch_count"], 2)
        self.assertEqual(validate_record(result)["status"], "valid")

    def test_exhausted_source_fails_closed(self) -> None:
        graph = graph_with_types(["PROP", "PROP", "PROP"])
        client = SimpleNamespace(provider="mock")
        bad = self._fake(lambda s, o, a: ["s1:R9999"] if s == "s1" else [])
        with patch.object(self.edges, "call_model", side_effect=bad):
            with self.assertRaises(ValueError) as ctx:
                self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False,
                    max_attempts=2, sources_per_request=3, payload="compact",
                )
        self.assertIn("['s1']", str(ctx.exception))
        with patch.object(self.edges, "call_model", side_effect=bad):
            with self.assertRaises(ValueError) as ctx:
                self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False, max_attempts=2,
                )
        self.assertIn("Salida inválida para s1", str(ctx.exception))

    def test_window_is_recorded_and_limits_options_only(self) -> None:
        graph = graph_with_types(["PROP"] * 6)
        seen_prompts: list[str] = []

        def recording(*args, **kwargs):
            seen_prompts.append(args[6])
            return self._fake(lambda s, o, a: [])(*args, **kwargs)

        client = SimpleNamespace(provider="mock")
        with patch.object(self.edges, "call_model", side_effect=recording):
            result = self.edges.build_semantic_edges(
                client, "mock-model", graph, None, None, "low", False,
                sources_per_request=3, candidate_window=1, payload="compact",
            )
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual(audit["candidate_policy"], "window:1")
        self.assertEqual(audit["estimator_variant"], "compact_batched_windowed")
        self.assertEqual(audit["excluded_by_window_by_source"]["s1"], 4)
        self.assertEqual(audit["excluded_by_window_by_source"]["s3"], 3)
        # el contexto completo sigue viajando aunque la ventana limite los destinos
        self.assertTrue(all('"segment_id":"s6"' in prompt for prompt in seen_prompts))
        self.assertEqual(
            result["validation"]["semantic_edges"]["estimator_variant"], "compact_batched_windowed"
        )
        self.assertEqual(validate_record(result)["status"], "valid")

    def test_consensus_votes_per_edge_inside_batches(self) -> None:
        graph = graph_with_types(["PROP", "PROP", "PROP"])
        draws: dict[str, int] = {}

        def decide(source_id, options, attempt):
            draws[source_id] = draws.get(source_id, 0) + 1
            if source_id == "s1" and draws[source_id] in (1, 3):
                return [self._ref(options, "s2", "SUPPORTS")]
            if source_id == "s2" and draws[source_id] == 2:
                return [self._ref(options, "s3", "LEADS_TO")]
            return []

        client = SimpleNamespace(provider="mock")
        with patch.object(self.edges, "call_model", side_effect=self._fake(decide)) as calls:
            result = self.edges.build_semantic_edges(
                client, "mock-model", graph, None, None, "low", False,
                samples=3, sources_per_request=6, payload="compact",
            )
        self.assertEqual(calls.call_count, 3)
        semantic = [(e["source"], e["edge_type"], e["target"]) for e in result["edges"] if e["edge_type"] != "NEXT"]
        self.assertEqual(semantic, [("s1", "SUPPORTS", "s2")])
        kept = next(e for e in result["edges"] if e["edge_type"] == "SUPPORTS")
        self.assertEqual(kept["consensus"]["support"], 2)
        audit = result["audit"]["semantic_edges"][-1]
        self.assertEqual(audit["minority_relations_by_source"]["s2"][0]["target"], "s3")

    def test_resume_reuses_cached_sources_and_queries_only_pending(self) -> None:
        graph = graph_with_types(["PROP", "PROP", "PROP", "PROP"])

        def interrupted(_client, _model, batch, *_rest):
            ids = [item["node"]["segment_id"] for item in batch]
            if "s3" in ids:
                raise TimeoutError("simulated timeout")
            return self.edges.BatchRelationsDraft(
                sources=[self.edges.NodeRelationsDraft(source=s, relation_refs=[]) for s in ids]
            )

        client = SimpleNamespace(provider="mock")
        common = dict(max_attempts=1, sources_per_request=2, payload="compact")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with patch.object(self.edges, "call_model", side_effect=interrupted):
                with self.assertRaises(TimeoutError):
                    self.edges.build_semantic_edges(
                        client, "mock-model", deepcopy(graph), None, None, "low", False,
                        checkpoint_dir=checkpoint_dir, **common,
                    )
            with patch.object(self.edges, "call_model", side_effect=self._fake(lambda s, o, a: [])) as resumed:
                result = self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False,
                    checkpoint_dir=checkpoint_dir, resume_checkpoint=True, **common,
                )
            self.assertEqual(resumed.call_count, 1)
            self.assertEqual(
                result["validation"]["semantic_edges"]["checkpoint"]["reused_source_ids"], ["s1", "s2"]
            )
            # un lote parcialmente en caché consulta solo las fuentes pendientes
            path = checkpoint_path_for_trace(checkpoint_dir, "05_aristas_semanticas", graph["trace_id"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["items"]["s4"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            calls: list[list[str]] = []

            def recording(*args, **kwargs):
                calls.append([item["node"]["segment_id"] for item in args[2]])
                return self._fake(lambda s, o, a: [])(*args, **kwargs)

            with patch.object(self.edges, "call_model", side_effect=recording):
                result = self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False,
                    checkpoint_dir=checkpoint_dir, resume_checkpoint=True, **common,
                )
            self.assertEqual(calls, [["s4"]])
            audit = result["audit"]["semantic_edges"][-1]
            self.assertEqual(audit["request_batches"][1]["queried_sources"], ["s4"])
            self.assertEqual(
                result["validation"]["semantic_edges"]["checkpoint"]["reused_source_ids"], ["s1", "s2", "s3"]
            )
            # otra carga u otro tamaño de lote cambian la identidad: nada se reutiliza
            with patch.object(self.edges, "call_model", side_effect=self._fake(lambda s, o, a: [])) as rebatched:
                result = self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False,
                    max_attempts=1, checkpoint_dir=checkpoint_dir, sources_per_request=3,
                    payload="compact", resume_checkpoint=True,
                )
            self.assertEqual(rebatched.call_count, 2)
            self.assertEqual(result["validation"]["semantic_edges"]["checkpoint"]["load_status"], "identity_mismatch")
            with patch.object(self.edges, "call_model", side_effect=self._fake(lambda s, o, a: [])) as canonical:
                result = self.edges.build_semantic_edges(
                    client, "mock-model", deepcopy(graph), None, None, "low", False,
                    max_attempts=1, checkpoint_dir=checkpoint_dir, resume_checkpoint=True,
                )
            self.assertEqual(canonical.call_count, 4)
            self.assertEqual(result["validation"]["semantic_edges"]["checkpoint"]["load_status"], "identity_mismatch")


if __name__ == "__main__":
    unittest.main()
