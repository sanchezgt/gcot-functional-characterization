"""Huella del instrumento de medida: todo lo que determina qué se le pide al analizador LLM.

`instrument_manifest.json` fija, para cada etapa que llama al LLM (01, 01.1, 02, 05, 06, 07):

- la huella de cada texto de prompt (sistema, plantilla, reparación, especificaciones de tipo);
- la huella de la solicitud ya ensamblada sobre entradas sintéticas fijas (serialización incluida);
- lo que `prompt_hash` NO cubre en los artefactos: modelo, esfuerzo de razonamiento, intentos,
  muestras de consenso y esquema de respuesta; y los valores canónicos de transporte y consenso.

Si esta prueba falla, el instrumento cambió. Un cambio intencional exige regenerar el manifiesto, subiendo `instrument_version`:

    python tests/test_instrument_fingerprint.py --write

No hay llamadas a la API ni dependencia del entorno (.env y variables OPENAI_* se neutralizan).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from tests.test_edge_batching import graph_with_types  # noqa: E402
from tests.test_objetivo3_contract import base_graph  # noqa: E402
from util import (  # noqa: E402
    consensus,
    edge_batching,
    llm_provider,
    objective3_hierarchical,
    objective3_llm,
    objective3_stability,
    sgotgraph_objective3,
    sgotgraph_schema,
)
from util.llm_provider import LLMClient, StructuredResponse, structured_request_hash  # noqa: E402
from util.trace_alias import prompt_trace_alias  # noqa: E402

MANIFEST_PATH = PIPELINE_ROOT / "instrument_manifest.json"
INSTRUMENT_VERSION = "sgotgraph-instrument-2026.14"
CLIENT = SimpleNamespace(provider="openai")
TRACE_ID = "fp-trace"
TRACE = "Step one. Step two. Step three."
ENVIRONMENT_OVERRIDES = (
    "LLM_PROVIDER",
    "OPENAI_MODEL",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_REFINEMENT_REASONING_EFFORT",
    "OPENAI_TYPING_REASONING_EFFORT",
    "OPENAI_EDGE_REASONING_EFFORT",
)
STAGE_FILES = {
    "01": "01_segmentacion_gcot.py",
    "01.1": "01_1_refinamiento_segmentacion_gcot.py",
    "02": "02_tipado_gcot.py",
    "05": "05_aristas_semanticas_gcot.py",
    "06": "06_episodios_funcionales_gcot.py",
    "07": "07_relaciones_episodios_gcot.py",
}
EPISODE_TYPES = ("ALTERNATIVE", "CORRECTION", "PROGRESSION", "SUPPORT", "VERIFICATION")


def load_stage(key: str):
    name = "fingerprint_stage_" + key.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, PIPELINE_ROOT / STAGE_FILES[key])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(model) -> str:
    return text_hash(json.dumps(model.model_json_schema(), sort_keys=True, ensure_ascii=False))


def request_hash(system_prompt: str, user_prompt: str) -> str:
    return structured_request_hash(CLIENT, system_prompt, user_prompt)


def cli_defaults(stage) -> dict:
    """Valores por defecto REALES de argparse, sin .env ni variables de entorno que los alteren."""
    clean = {k: v for k, v in os.environ.items() if k not in ENVIRONMENT_OVERRIDES}
    with (
        patch.dict(os.environ, clean, clear=True),
        patch.object(sys, "argv", ["prog", "in.jsonl", "-o", "out.json"]),
        patch.object(stage, "load_dotenv", lambda *args, **kwargs: None),
    ):
        args = stage.parse_args()
    keys = ("model", "reasoning_effort", "max_attempts", "samples")
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def segmentation_requests(stage) -> dict:
    captured: list[str] = []
    units = stage.build_literal_units(TRACE)

    def fake_call(_client, _model, _trace_id, prompt, _verbose, _kind, _effort):
        captured.append(prompt)
        # La primera respuesta no cubre todas las unidades: fuerza el prompt de reparación.
        last = units[0] if len(captured) == 1 else units[-1]
        return stage.ParsedSegmentationResponse(
            draft=stage.SegmentationDraft(
                trace_id="ignored",
                segments=[
                    stage.SegmentRangeDraft(
                        start_unit_id=units[0]["unit_id"], end_unit_id=last["unit_id"]
                    )
                ],
            ),
            provider="mock",
        )

    with patch.object(stage, "call_model", side_effect=fake_call):
        stage.segment_trace(
            CLIENT, "fp-model", {"id": TRACE_ID, "reasoning_trace": TRACE},
            max_attempts=2, verbose=False, reasoning_effort="minimal",
        )
    assert len(captured) == 2, captured
    return {
        "base": request_hash(stage.SYSTEM_PROMPT, captured[0]),
        "repair": request_hash(stage.SYSTEM_PROMPT, captured[1]),
    }


def episode_and_relation_requests() -> dict:
    graph = base_graph()
    graph["trace_id"] = TRACE_ID
    alias = prompt_trace_alias(TRACE_ID)
    client = LLMClient(provider="openai", client=None)  # type: ignore[arg-type]
    captured: list[tuple[str, str]] = []

    def fake_episodes(*_args, **kwargs):
        captured.append((kwargs["system_prompt"], kwargs["user_prompt"]))
        episode_type = next(
            item for item in EPISODE_TYPES if f"episode_type='{item}'" in kwargs["user_prompt"]
        )
        return StructuredResponse(
            parsed=objective3_llm.EpisodeBatchDraft(
                trace_id=alias, episode_type=episode_type, episodes=[]
            ),
            provider="openai",
            model="fp-model",
        )

    with patch("util.objective3_llm.request_structured_response", side_effect=fake_episodes):
        objective3_llm.infer_episodes(
            client, model="fp-model", sgot_graph=graph, reasoning_effort="low"
        )
    episodes = {}
    for system_prompt, user_prompt in captured:
        episode_type = next(
            item for item in EPISODE_TYPES if f"episode_type='{item}'" in user_prompt
        )
        episodes[episode_type] = request_hash(system_prompt, user_prompt)
    assert sorted(episodes) == sorted(EPISODE_TYPES), sorted(episodes)

    record = sgotgraph_objective3.build_objective3_record_from_candidates(
        graph, [], default_provenance={"origin": "llm_inferred", "stage": "06_episodes"}
    )
    captured.clear()

    def fake_relations(*_args, **kwargs):
        captured.append((kwargs["system_prompt"], kwargs["user_prompt"]))
        return StructuredResponse(
            parsed=objective3_llm.RelationBatchDraft(trace_id=alias, relations=[]),
            provider="openai",
            model="fp-model",
        )

    with patch("util.objective3_llm.request_structured_response", side_effect=fake_relations):
        objective3_llm.infer_episode_relations(
            client, model="fp-model", record=record, reasoning_effort="low"
        )
    assert len(captured) == 1, len(captured)
    return {"episodes": episodes, "relations": request_hash(*captured[0])}


def compute_manifest() -> dict:
    s01, s011, s02, s05, s06, s07 = (load_stage(key) for key in STAGE_FILES)
    graph = graph_with_types(["METHOD", "PROP", "CHECK", "PROP", "OUT"], trace_id=TRACE_ID)
    nodes = graph["nodes"]
    candidates = s05.select_candidate_nodes(nodes, 1)
    compact_item = s05.build_source_item(nodes[1], candidates, "compact")
    segment = {"segment_id": "s1", "text": "Step one. Step two.", "start_char": 0, "end_char": 19}
    refinement_draft = s011.RefinementDraft(segment_id="s1", decision="KEEP", subsegment_ranges=[])
    objective3 = episode_and_relation_requests()
    specifications = json.dumps(
        objective3_llm.EPISODE_SPECIFICATIONS, sort_keys=True, ensure_ascii=False
    )

    return {
        "instrument_version": INSTRUMENT_VERSION,
        "note": (
            "Generado por tests/test_instrument_fingerprint.py --write. No editar a mano. "
            "Un cambio aquí es un cambio del instrumento de medida."
        ),
        "contracts": {
            "objective2": [sgotgraph_schema.SCHEMA_VERSION, sgotgraph_schema.PIPELINE_VERSION],
            "objective3": [sgotgraph_objective3.SCHEMA_VERSION, sgotgraph_objective3.PIPELINE_VERSION],
        },
        "canonical_settings": {
            "consensus_samples": consensus.DEFAULT_CONSENSUS_SAMPLES,
            "max_extra_samples": consensus.DEFAULT_MAX_EXTRA_SAMPLES,
            "episode_iou_threshold": objective3_stability.DEFAULT_EPISODE_IOU_THRESHOLD,
            "edge_payload": edge_batching.DEFAULT_PAYLOAD,
            "edge_sources_per_request": edge_batching.DEFAULT_SOURCES_PER_REQUEST,
            "edge_max_options_per_request": edge_batching.DEFAULT_MAX_OPTIONS_PER_REQUEST,
            "edge_transport": s05.RELATION_OPTION_TRANSPORT_VERSION,
            "block_core_halo_shift": [
                objective3_hierarchical.DEFAULT_BLOCK_CORE_NODES,
                objective3_hierarchical.DEFAULT_BLOCK_HALO_NODES,
                objective3_hierarchical.DEFAULT_BLOCK_SHIFT_NODES,
            ],
            "request_timeout_seconds": llm_provider.DEFAULT_REQUEST_TIMEOUT_SECONDS,
            "transport_max_retries": llm_provider.DEFAULT_TRANSPORT_MAX_RETRIES,
            "sampling_controls_sent": "none (no temperature, no seed)",
        },
        "stages": {
            "01": {
                "cli_defaults": cli_defaults(s01),
                "response_model": schema_hash(s01.SegmentationDraft),
                "texts": {
                    "system": text_hash(s01.SYSTEM_PROMPT),
                    "user_template": text_hash(s01.USER_PROMPT_TEMPLATE),
                    "repair_template": text_hash(s01.REPAIR_PROMPT_TEMPLATE),
                    "literal_unitization": [
                        s01.LITERAL_UNITIZATION_VERSION, s01.DEFAULT_LITERAL_UNIT_MAX_CHARS
                    ],
                },
                "requests": segmentation_requests(s01),
            },
            "01.1": {
                "cli_defaults": cli_defaults(s011),
                "response_model": schema_hash(s011.RefinementDraft),
                "texts": {
                    "system": text_hash(s011.SYSTEM_PROMPT),
                    "user_template": text_hash(s011.USER_PROMPT_TEMPLATE),
                    "repair_template": text_hash(s011.REPAIR_PROMPT_TEMPLATE),
                    "literal_unitization": s011.REFINEMENT_LITERAL_UNITIZATION_VERSION,
                },
                "requests": {
                    "base": request_hash(s011.SYSTEM_PROMPT, s011.build_prompt(TRACE_ID, segment)),
                    "repair": request_hash(
                        s011.SYSTEM_PROMPT,
                        s011.build_repair_prompt(TRACE_ID, segment, refinement_draft, ["error"]),
                    ),
                },
            },
            "02": {
                "cli_defaults": cli_defaults(s02),
                "response_model": schema_hash(s02.TypingDraft),
                "texts": {
                    "system": text_hash(s02.SYSTEM_PROMPT),
                    "user_template": text_hash(s02.USER_PROMPT_TEMPLATE),
                },
                "requests": {"base": request_hash(s02.SYSTEM_PROMPT, s02.build_prompt(graph))},
            },
            "05": {
                "cli_defaults": cli_defaults(s05),
                "response_model": schema_hash(s05.NodeRelationsDraft),
                "texts": {
                    "system": text_hash(s05.SYSTEM_PROMPT),
                    "user_template": text_hash(s05.USER_PROMPT_TEMPLATE),
                    "repair_suffix": text_hash(s05.REPAIR_PROMPT_SUFFIX),
                },
                "requests": {
                    "base": request_hash(s05.SYSTEM_PROMPT, s05.build_prompt(nodes[1], candidates))
                },
                "non_canonical_compact": {
                    "response_model": schema_hash(s05.BatchRelationsDraft),
                    "transport": s05.COMPACT_TRANSPORT_VERSION,
                    "system": text_hash(s05.COMPACT_SYSTEM_PROMPT),
                    "user_template": text_hash(s05.COMPACT_USER_PROMPT_TEMPLATE),
                    "repair_suffix": text_hash(s05.COMPACT_REPAIR_PROMPT_SUFFIX),
                    "base": request_hash(
                        s05.COMPACT_SYSTEM_PROMPT, s05.build_batch_prompt([compact_item], nodes)
                    ),
                },
            },
            "06": {
                "cli_defaults": cli_defaults(s06),
                "response_model": schema_hash(objective3_llm.EpisodeBatchDraft),
                "texts": {
                    "system": text_hash(objective3_llm.EPISODE_SYSTEM_PROMPT),
                    "user_template": text_hash(objective3_llm.EPISODE_USER_TEMPLATE),
                    "repair_suffix": text_hash(objective3_llm.EPISODE_REPAIR_SUFFIX),
                    "type_specifications": text_hash(specifications),
                },
                "requests": objective3["episodes"],
                "reconciliation": {
                    "response_model": schema_hash(objective3_llm.EpisodeReconciliationBatchDraft),
                    "system": text_hash(objective3_llm.RECONCILIATION_SYSTEM_PROMPT),
                    "user_template": text_hash(objective3_llm.RECONCILIATION_USER_TEMPLATE),
                    "repair_suffix": text_hash(objective3_llm.RECONCILIATION_REPAIR_SUFFIX),
                },
                "non_canonical_hierarchical": {
                    "response_model": schema_hash(objective3_hierarchical.EpisodeSynthesisBatchDraft),
                    "local_system": text_hash(objective3_hierarchical.LOCAL_EPISODE_SYSTEM_PROMPT),
                    "local_user_template": text_hash(
                        objective3_hierarchical.LOCAL_EPISODE_USER_TEMPLATE
                    ),
                    "synthesis_system": text_hash(objective3_hierarchical.SYNTHESIS_SYSTEM_PROMPT),
                    "synthesis_user_template": text_hash(
                        objective3_hierarchical.SYNTHESIS_USER_TEMPLATE
                    ),
                    "synthesis_repair_suffix": text_hash(
                        objective3_hierarchical.SYNTHESIS_REPAIR_SUFFIX
                    ),
                },
            },
            "07": {
                "cli_defaults": cli_defaults(s07),
                "response_model": schema_hash(objective3_llm.RelationBatchDraft),
                "texts": {
                    "system": text_hash(objective3_llm.RELATION_SYSTEM_PROMPT),
                    "user_template": text_hash(objective3_llm.RELATION_USER_TEMPLATE),
                    "repair_suffix": text_hash(objective3_llm.RELATION_REPAIR_SUFFIX),
                },
                "requests": {"base": objective3["relations"]},
            },
        },
    }


def flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


class InstrumentFingerprintTests(unittest.TestCase):
    def test_instrument_matches_the_frozen_manifest(self) -> None:
        frozen = flatten(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
        current = flatten(compute_manifest())
        changed = sorted(
            key for key in set(frozen) | set(current) if frozen.get(key) != current.get(key)
        )
        self.assertEqual(
            changed,
            [],
            "El instrumento de medida cambió en: "
            + ", ".join(changed)
            + ". Si es intencional: `python tests/test_instrument_fingerprint.py --write` subiendo instrument_version.",
        )

    def test_fingerprint_is_deterministic(self) -> None:
        self.assertEqual(compute_manifest(), compute_manifest())

    def test_canonical_stage05_request_matches_the_july_golden_hash(self) -> None:
        # Misma entrada sintética que tests/test_edge_batching.py: la huella debe coincidir con la de referencia.
        from tests.test_edge_batching import GOLDEN_PROMPT_HASH_S2

        self.assertEqual(compute_manifest()["stages"]["05"]["requests"]["base"], GOLDEN_PROMPT_HASH_S2)

    def test_artifact_source_hash_covers_every_pipeline_module(self) -> None:
        listed = set(sgotgraph_objective3.IMPLEMENTATION_FILES)
        for relative_path in sorted(listed):
            self.assertTrue((PIPELINE_ROOT / relative_path).is_file(), relative_path)
        expected = {f"util/{path.name}" for path in (PIPELINE_ROOT / "util").glob("*.py")}
        expected |= {path.name for path in PIPELINE_ROOT.glob("0*_gcot.py")}
        expected.discard("util/__init__.py")
        self.assertEqual(sorted(expected - listed), [], "módulos fuera de validation.runtime.source_hash")


if __name__ == "__main__":
    if "--write" in sys.argv:
        MANIFEST_PATH.write_text(
            json.dumps(compute_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("escrito:", MANIFEST_PATH)
    else:
        unittest.main()
