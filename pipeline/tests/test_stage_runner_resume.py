from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util import run_pipeline_gcot, run_pipeline_objetivo3


def write_mock_stage_output(command: list[str]) -> None:
    output = Path(command[command.index("-o") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"mock":"valid"}\n', encoding="utf-8")


class StageRunnerResumeTests(unittest.TestCase):
    def test_objective2_resume_restores_every_compatible_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.json"
            input_path.write_text(
                '{"id":"case","reasoning_trace":"Reasoning."}\n',
                encoding="utf-8",
            )
            checkpoint_dir = root / "checkpoints"

            def args(output_dir: Path, resume: bool):
                return SimpleNamespace(
                    input_path=input_path,
                    output_dir=output_dir,
                    trace_id="case",
                    index=None,
                    limit=None,
                    provider="openai",
                    model="model",
                    api_key=None,
                    base_url=None,
                    edge_max_attempts=3,
                    request_timeout_seconds=180.0,
                    transport_max_retries=2,
                    checkpoint_dir=checkpoint_dir,
                    resume_internal=resume,
                    pretty=True,
                    verbose=False,
                )

            with (
                patch.object(
                    run_pipeline_gcot,
                    "parse_args",
                    return_value=args(root / "attempt_1", False),
                ),
                patch.object(
                    run_pipeline_gcot,
                    "run_command",
                    side_effect=lambda command, *_args: (
                        write_mock_stage_output(command)
                    ),
                ) as first_run,
            ):
                self.assertEqual(run_pipeline_gcot.main(), 0)
            self.assertEqual(first_run.call_count, 6)

            with (
                patch.object(
                    run_pipeline_gcot,
                    "parse_args",
                    return_value=args(root / "attempt_2", True),
                ),
                patch.object(
                    run_pipeline_gcot,
                    "run_command",
                ) as resumed_run,
            ):
                self.assertEqual(run_pipeline_gcot.main(), 0)
            resumed_run.assert_not_called()
            self.assertTrue(
                (
                    root
                    / "attempt_2"
                    / "00_sgotgraph_objetivo2.json"
                ).exists()
            )
            report = json.loads(
                (
                    root / "attempt_2" / "resume_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["restored_stages"]), 6)
            self.assertEqual(report["executed_stages"], [])

    def test_objective3_resume_restores_completed_llm_stages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "objective2.json"
            input_path.write_text(
                '{"trace_id":"case"}\n',
                encoding="utf-8",
            )
            checkpoint_dir = root / "checkpoints"

            def args(output_dir: Path, resume: bool):
                return SimpleNamespace(
                    input_path=input_path,
                    output_dir=output_dir,
                    trace_id=None,
                    index=None,
                    limit=None,
                    provider="openai",
                    model="model",
                    api_key=None,
                    base_url=None,
                    request_timeout_seconds=180.0,
                    transport_max_retries=2,
                    checkpoint_dir=checkpoint_dir,
                    resume_internal=resume,
                    episode_reasoning_effort="low",
                    relation_reasoning_effort="low",
                    semantic_max_attempts=3,
                    episode_method="global",
                    block_core_nodes=32,
                    block_halo_nodes=8,
                    block_shift_nodes=0,
                    pretty=True,
                    verbose=False,
                )

            with (
                patch.object(
                    run_pipeline_objetivo3,
                    "parse_args",
                    return_value=args(root / "attempt_1", False),
                ),
                patch.object(
                    run_pipeline_objetivo3,
                    "run",
                    side_effect=lambda command, *_args: (
                        write_mock_stage_output(command)
                    ),
                ) as first_run,
            ):
                self.assertEqual(run_pipeline_objetivo3.main(), 0)
            self.assertEqual(first_run.call_count, 4)

            with (
                patch.object(
                    run_pipeline_objetivo3,
                    "parse_args",
                    return_value=args(root / "attempt_2", True),
                ),
                patch.object(
                    run_pipeline_objetivo3,
                    "run",
                    side_effect=lambda command, *_args: (
                        write_mock_stage_output(command)
                    ),
                ) as resumed_run,
            ):
                self.assertEqual(run_pipeline_objetivo3.main(), 0)
            self.assertEqual(resumed_run.call_count, 1)
            self.assertIn(
                "09_validacion_objetivo3_gcot.py",
                resumed_run.call_args.args[0],
            )
            report = json.loads(
                (
                    root / "attempt_2" / "resume_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["restored_stages"],
                [
                    "06_episodios",
                    "07_relaciones_episodios",
                    "08_rutas_perfiles",
                ],
            )
            self.assertEqual(
                report["executed_stages"],
                ["09_validacion_objetivo3"],
            )

    def test_episode_method_is_part_of_stage06_cache_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "objective2.json"
            input_path.write_text(
                '{"trace_id":"case"}\n',
                encoding="utf-8",
            )
            checkpoint_dir = root / "checkpoints"

            def args(
                output_dir: Path,
                resume: bool,
                episode_method: str,
            ):
                return SimpleNamespace(
                    input_path=input_path,
                    output_dir=output_dir,
                    trace_id=None,
                    index=None,
                    limit=None,
                    provider="openai",
                    model="model",
                    api_key=None,
                    base_url=None,
                    request_timeout_seconds=180.0,
                    transport_max_retries=2,
                    checkpoint_dir=checkpoint_dir,
                    resume_internal=resume,
                    episode_reasoning_effort="low",
                    relation_reasoning_effort="low",
                    semantic_max_attempts=3,
                    episode_method=episode_method,
                    consensus_samples=1 if episode_method == "hierarchical" else 3,
                    block_core_nodes=32,
                    block_halo_nodes=8,
                    block_shift_nodes=0,
                    pretty=True,
                    verbose=False,
                )

            with (
                patch.object(
                    run_pipeline_objetivo3,
                    "parse_args",
                    return_value=args(root / "global", False, "global"),
                ),
                patch.object(
                    run_pipeline_objetivo3,
                    "run",
                    side_effect=lambda command, *_args: (
                        write_mock_stage_output(command)
                    ),
                ),
            ):
                self.assertEqual(run_pipeline_objetivo3.main(), 0)

            with (
                patch.object(
                    run_pipeline_objetivo3,
                    "parse_args",
                    return_value=args(
                        root / "hierarchical",
                        True,
                        "hierarchical",
                    ),
                ),
                patch.object(
                    run_pipeline_objetivo3,
                    "run",
                    side_effect=lambda command, *_args: (
                        write_mock_stage_output(command)
                    ),
                ) as changed_run,
            ):
                self.assertEqual(run_pipeline_objetivo3.main(), 0)

            self.assertEqual(changed_run.call_count, 2)
            stage06_command = changed_run.call_args_list[0].args[0]
            self.assertIn("06_episodios_funcionales_gcot.py", stage06_command)
            self.assertEqual(
                stage06_command[
                    stage06_command.index("--episode-method") + 1
                ],
                "hierarchical",
            )
            self.assertIn(
                "09_validacion_objetivo3_gcot.py",
                changed_run.call_args_list[1].args[0],
            )
            report = json.loads(
                (
                    root
                    / "hierarchical"
                    / "resume_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["restored_stages"],
                ["07_relaciones_episodios", "08_rutas_perfiles"],
            )
            self.assertEqual(
                report["executed_stages"],
                ["06_episodios", "09_validacion_objetivo3"],
            )


if __name__ == "__main__":
    unittest.main()
