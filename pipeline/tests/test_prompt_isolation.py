"""Ningún prompt LLM debe recibir el identificador real de la traza ni los
metadatos del registro fuente (enunciado, respuesta esperada, dataset,
generador). El método es agnóstico al generador: los estimadores solo ven la
traza, sus segmentos y el grafo, con un alias opaco de traza.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from tests.test_objetivo3_contract import base_graph  # noqa: E402
from util.llm_provider import LLMClient, StructuredResponse  # noqa: E402
from util.objective3_llm import (  # noqa: E402
    EpisodeBatchDraft,
    RelationBatchDraft,
    infer_episode_relations,
    infer_episodes,
)
from util.sgotgraph_objective3 import (  # noqa: E402
    build_objective3_record_from_candidates,
)
from util.trace_alias import prompt_trace_alias  # noqa: E402

REAL_ID = "gsm8k_deepseek_r1_0001"
FORBIDDEN = ("gsm8k", "deepseek", REAL_ID, "POISON_PROMPT", "POISON_ANSWER", "POISON_OUTPUT")
SOURCE_RECORD = {
    "id": REAL_ID,
    "dataset": "GSM8K",
    "model": "DeepSeek-R1",
    "prompt": "POISON_PROMPT",
    "expected_final_answer": "POISON_ANSWER",
    "model_output": "POISON_OUTPUT",
}


def load_stage(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_clean(test: unittest.TestCase, *prompts: str) -> None:
    for prompt in prompts:
        for token in FORBIDDEN:
            test.assertNotIn(token, prompt)


class PromptIsolationTests(unittest.TestCase):
    def test_alias_is_opaque_and_stable(self) -> None:
        alias = prompt_trace_alias(REAL_ID)
        self.assertTrue(alias.startswith("trace_"))
        self.assertEqual(alias, prompt_trace_alias(REAL_ID))
        self.assertNotIn("gsm8k", alias)
        self.assertNotEqual(alias, prompt_trace_alias("medqa_deepseek_r1_0001"))

    def test_segmentation_prompts_carry_only_the_alias(self) -> None:
        stage = load_stage("isolation_stage_01", "01_segmentacion_gcot.py")
        captured: list[str] = []

        def fake_call(_client, _model, _trace_id, prompt, _verbose, _kind, _effort):
            captured.append(prompt)
            units = stage.build_literal_units("Step one. Step two.")
            return stage.ParsedSegmentationResponse(
                draft=stage.SegmentationDraft(
                    trace_id="whatever",
                    segments=[
                        stage.SegmentRangeDraft(
                            start_unit_id=units[0]["unit_id"], end_unit_id=units[-1]["unit_id"]
                        )
                    ],
                ),
                provider="mock",
            )

        record = {**SOURCE_RECORD, "reasoning_trace": "Step one. Step two."}
        with patch.object(stage, "call_model", side_effect=fake_call):
            result = stage.segment_trace(
                SimpleNamespace(provider="mock"), "mock-model", record,
                max_attempts=1, verbose=False, reasoning_effort="minimal",
            )
        self.assertEqual(result["trace_id"], REAL_ID)
        assert_clean(self, stage.SYSTEM_PROMPT, *captured)
        self.assertIn(prompt_trace_alias(REAL_ID), captured[0])

    def test_refinement_prompts_carry_only_the_alias(self) -> None:
        stage = load_stage("isolation_stage_011", "01_1_refinamiento_segmentacion_gcot.py")
        segment = {"segment_id": "s1", "text": "Step one. Step two.", "start_char": 0, "end_char": 19}
        prompt = stage.build_prompt(REAL_ID, segment)
        draft = stage.RefinementDraft(segment_id="s1", decision="KEEP", subsegment_ranges=[])
        repair = stage.build_repair_prompt(REAL_ID, segment, draft, ["error"])
        assert_clean(self, stage.SYSTEM_PROMPT, prompt, repair)
        self.assertIn(prompt_trace_alias(REAL_ID), prompt)

    def test_typing_and_edge_prompts_never_see_source_metadata(self) -> None:
        typing = load_stage("isolation_stage_02", "02_tipado_gcot.py")
        edges = load_stage("isolation_stage_05", "05_aristas_semanticas_gcot.py")
        graph = base_graph()
        graph["trace_id"] = REAL_ID
        graph["source_record"] = dict(SOURCE_RECORD)
        typing_prompt = typing.build_prompt(graph)
        candidates = edges.select_candidate_nodes(graph["nodes"], 0)
        edge_prompt = edges.build_prompt(graph["nodes"][0], candidates)
        compact_item = edges.build_source_item(graph["nodes"][0], candidates, "compact")
        compact_prompt = edges.build_batch_prompt([compact_item], graph["nodes"])
        assert_clean(
            self,
            typing.SYSTEM_PROMPT,
            typing_prompt,
            edges.SYSTEM_PROMPT,
            edge_prompt,
            edges.COMPACT_SYSTEM_PROMPT,
            compact_prompt,
        )

    def test_episode_and_relation_prompts_carry_only_the_alias(self) -> None:
        graph = base_graph()
        graph["trace_id"] = REAL_ID
        graph["source_record"] = dict(SOURCE_RECORD)
        captured: list[tuple[str, str]] = []

        def fake_episodes(*args, **kwargs):
            captured.append((kwargs["system_prompt"], kwargs["user_prompt"]))
            episode_type = next(
                item
                for item in ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")
                if f"episode_type='{item}'" in kwargs["user_prompt"]
            )
            return StructuredResponse(
                parsed=EpisodeBatchDraft(
                    trace_id=prompt_trace_alias(REAL_ID), episode_type=episode_type, episodes=[]
                ),
                provider="openai",
                model="mock-model",
            )

        client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
        with patch("util.objective3_llm.request_structured_response", side_effect=fake_episodes):
            record = infer_episodes(client, model="mock-model", sgot_graph=graph, reasoning_effort="low")
        self.assertEqual(record["validation"]["status"], "VALID")
        self.assertEqual(record["trace_id"], REAL_ID)
        for system_prompt, user_prompt in captured:
            assert_clean(self, system_prompt, user_prompt)
            self.assertIn(prompt_trace_alias(REAL_ID), user_prompt)

        relation_record = build_objective3_record_from_candidates(
            graph, [], default_provenance={"origin": "llm_inferred", "stage": "06_episodes"}
        )
        captured.clear()

        def fake_relations(*args, **kwargs):
            captured.append((kwargs["system_prompt"], kwargs["user_prompt"]))
            return StructuredResponse(
                parsed=RelationBatchDraft(trace_id=prompt_trace_alias(REAL_ID), relations=[]),
                provider="openai",
                model="mock-model",
            )

        with patch("util.objective3_llm.request_structured_response", side_effect=fake_relations):
            related = infer_episode_relations(
                client, model="mock-model", record=relation_record, reasoning_effort="low"
            )
        self.assertEqual(related["trace_id"], REAL_ID)
        self.assertTrue(captured)
        for system_prompt, user_prompt in captured:
            assert_clean(self, system_prompt, user_prompt)


if __name__ == "__main__":
    unittest.main()
