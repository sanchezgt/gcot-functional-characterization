"""Opciones de transporte y de consenso que el orquestador del objetivo 2 pasa a cada etapa."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util import run_pipeline_completo_gcot, run_pipeline_gcot, run_pipeline_objetivo3  # noqa: E402


def make_args(**overrides) -> SimpleNamespace:
    base = dict(
        trace_id=None,
        index=None,
        limit=None,
        provider="openai",
        model="model",
        base_url=None,
        edge_max_attempts=3,
        consensus_samples=3,
        request_timeout_seconds=180.0,
        transport_max_retries=2,
        checkpoint_dir=None,
        resume_internal=False,
        pretty=False,
        verbose=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def command(script: str, args: SimpleNamespace) -> list[str]:
    return run_pipeline_gcot.build_stage_command(
        script, Path("in.json"), Path("out.json"), args, include_selection=False
    )


def option(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


class RunnerTransportOptionsTests(unittest.TestCase):
    def test_default_edge_transport_is_canonical(self) -> None:
        cmd = command("05_aristas_semanticas_gcot.py", make_args())
        self.assertEqual(option(cmd, "--payload"), "full")
        self.assertEqual(option(cmd, "--sources-per-request"), "1")
        self.assertNotIn("--candidate-window", cmd)
        self.assertEqual(option(cmd, "--samples"), "3")

    def test_non_canonical_edge_transport_is_forwarded(self) -> None:
        cmd = command(
            "05_aristas_semanticas_gcot.py",
            make_args(edge_payload="compact", edge_sources_per_request=6, edge_candidate_window=8),
        )
        self.assertEqual(option(cmd, "--payload"), "compact")
        self.assertEqual(option(cmd, "--sources-per-request"), "6")
        self.assertEqual(option(cmd, "--candidate-window"), "8")

    def test_segmentation_can_use_its_own_sample_count(self) -> None:
        args = make_args(consensus_samples=3, segmentation_consensus_samples=5)
        self.assertEqual(option(command("01_segmentacion_gcot.py", args), "--samples"), "5")
        self.assertEqual(
            option(command("01_1_refinamiento_segmentacion_gcot.py", args), "--samples"), "5"
        )
        self.assertEqual(option(command("02_tipado_gcot.py", args), "--samples"), "3")
        self.assertEqual(option(command("05_aristas_semanticas_gcot.py", args), "--samples"), "3")
        # sin valor propio, la segmentación usa el número global
        self.assertEqual(option(command("01_segmentacion_gcot.py", make_args()), "--samples"), "3")
        self.assertNotIn("--samples", command("04_backbone_next_gcot.py", args))

    def test_parser_defaults_are_the_canonical_instrument(self) -> None:
        # Se comprueban los valores por defecto REALES de argparse, no los respaldos de build_stage_command.
        for runner, argv in (
            (run_pipeline_gcot, ["prog", "in.jsonl", "-o", "out"]),
            (run_pipeline_completo_gcot, ["prog", "in.jsonl", "-o", "out"]),
        ):
            with self.subTest(runner=runner.__name__), patch.object(sys, "argv", argv):
                args = runner.parse_args()
                self.assertEqual(args.edge_payload, "full")
                self.assertEqual(args.edge_sources_per_request, 1)
                self.assertEqual(args.edge_batch_offset, 0)
                self.assertIsNone(args.edge_candidate_window)
                self.assertEqual(args.consensus_samples, 3)
                self.assertIsNone(args.segmentation_consensus_samples)

    def test_hierarchical_episode_method_requires_a_single_sample(self) -> None:
        # La etapa 06 rechaza jerárquico + consenso; los orquestadores deben decirlo antes de gastar en 01–05.
        for runner in (run_pipeline_completo_gcot, run_pipeline_objetivo3):
            with self.subTest(runner=runner.__name__):
                with self.assertRaisesRegex(ValueError, "requiere --consensus-samples 1"):
                    runner.normalize_episode_method_args(
                        SimpleNamespace(episode_method="hierarchical", consensus_samples=3)
                    )
                runner.normalize_episode_method_args(
                    SimpleNamespace(episode_method="hierarchical", consensus_samples=1)
                )
                runner.normalize_episode_method_args(
                    SimpleNamespace(episode_method="global", consensus_samples=3)
                )

    def test_batch_identity_covers_transport_and_consensus_options(self) -> None:
        # Es la propiedad que impide reanudar un lote mezclando instrumentos.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.jsonl"
            source.write_text('{"id":"c1","reasoning_trace":"One."}\n', encoding="utf-8")

            def identity(**overrides):
                args = SimpleNamespace(
                    input_path=source, provider="openai", model="model", base_url=None,
                    edge_max_attempts=3, semantic_max_attempts=3, request_timeout_seconds=180.0,
                    transport_max_retries=2, **overrides,
                )
                run_pipeline_completo_gcot.normalize_episode_method_args(args)
                return run_pipeline_completo_gcot.batch_run_configuration(
                    args, Path(tmp), {}, [{"id": "c1"}]
                )

            base = identity()
            for key, canonical in (
                ("consensus_samples", 3), ("segmentation_consensus_samples", None),
                ("edge_payload", "full"), ("edge_sources_per_request", 1),
                ("edge_batch_offset", 0), ("edge_candidate_window", None), ("episode_method", "global"),
            ):
                self.assertEqual(base[key], canonical, key)
            for overrides in (
                {"edge_payload": "compact"},
                {"edge_payload": "compact", "edge_sources_per_request": 6},
                {"edge_payload": "compact", "edge_candidate_window": 8},
                {"consensus_samples": 5},
                {"segmentation_consensus_samples": 5},
            ):
                self.assertNotEqual(identity(**overrides), base, overrides)


if __name__ == "__main__":
    unittest.main()
