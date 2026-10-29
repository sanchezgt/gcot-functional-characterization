from __future__ import annotations

import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util import run_pipeline_completo_gcot


class CompletePipelineRunnerTests(unittest.TestCase):
    def test_failure_classifier_distinguishes_literal_refinement(self) -> None:
        log_text = (
            "Error: No se pudo validar el refinamiento de case/s1 tras "
            "3 intentos: KEEP debe cubrir desde la primera hasta la última "
            "unidad literal.\n"
            "Error: Command '['python', 'runner.py'] returned non-zero exit "
            "status 1.\n"
        )

        self.assertEqual(
            run_pipeline_completo_gcot.classify_failure(log_text),
            "literal_refinement_failure",
        )
        self.assertTrue(
            run_pipeline_completo_gcot.last_error_line(log_text).startswith(
                "Error: No se pudo validar el refinamiento"
            )
        )

    def test_failure_classifier_distinguishes_provider_timeout(
        self,
    ) -> None:
        self.assertEqual(
            run_pipeline_completo_gcot.classify_failure(
                "openai.APITimeoutError: Request timed out"
            ),
            "provider_timeout",
        )

    def test_case_command_enables_internal_resume_without_api_key(
        self,
    ) -> None:
        args = SimpleNamespace(
            input_path=Path("/tmp/input.jsonl"),
            provider="openai",
            model="model",
            base_url=None,
            api_key="must-not-appear",
            edge_max_attempts=3,
            semantic_max_attempts=3,
            episode_method="hierarchical",
            consensus_samples=1,
            block_core_nodes=32,
            block_halo_nodes=8,
            block_shift_nodes=16,
            request_timeout_seconds=90.0,
            transport_max_retries=1,
            checkpoint_dir=None,
            resume=True,
            pretty=True,
        )
        attempt_dir = Path("/tmp/case/attempt_002")

        command = run_pipeline_completo_gcot.build_case_command(
            args,
            "case-id",
            attempt_dir,
        )

        self.assertIn("--resume", command)
        self.assertEqual(
            command[command.index("--checkpoint-dir") + 1],
            str(attempt_dir.parent / "checkpoints"),
        )
        self.assertEqual(
            command[command.index("--request-timeout-seconds") + 1],
            "90.0",
        )
        self.assertEqual(
            command[command.index("--episode-method") + 1],
            "hierarchical",
        )
        self.assertEqual(
            command[command.index("--block-core-nodes") + 1],
            "32",
        )
        self.assertEqual(
            command[command.index("--block-halo-nodes") + 1],
            "8",
        )
        self.assertEqual(
            command[command.index("--block-shift-nodes") + 1],
            "16",
        )
        self.assertNotIn("must-not-appear", command)

    def test_semantic_attempts_are_forwarded_only_to_objective3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '{"id":"case_1","reasoning_trace":"Reasoning."}\n',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input_path=input_path,
                output_dir=root / "output",
                trace_id=None,
                index=0,
                limit=None,
                provider="openai",
                model="gpt-6-luna",
                api_key=None,
                base_url=None,
                edge_max_attempts=5,
                semantic_max_attempts=7,
                episode_method="hierarchical",
                consensus_samples=1,
                block_core_nodes=32,
                block_halo_nodes=8,
                block_shift_nodes=16,
                pretty=True,
                verbose=False,
            )

            with (
                patch.object(
                    run_pipeline_completo_gcot,
                    "parse_args",
                    return_value=args,
                ),
                patch.object(
                    run_pipeline_completo_gcot.subprocess,
                    "run",
                ) as mocked_run,
            ):
                self.assertEqual(run_pipeline_completo_gcot.main(), 0)

            self.assertEqual(mocked_run.call_count, 2)
            objective2_command = mocked_run.call_args_list[0].args[0]
            objective3_command = mocked_run.call_args_list[1].args[0]

            self.assertNotIn("--semantic-max-attempts", objective2_command)
            edge_attempts_index = objective2_command.index(
                "--edge-max-attempts"
            )
            self.assertEqual(
                objective2_command[edge_attempts_index + 1],
                "5",
            )
            attempts_index = objective3_command.index(
                "--semantic-max-attempts"
            )
            self.assertEqual(objective3_command[attempts_index + 1], "7")
            self.assertEqual(
                objective3_command[
                    objective3_command.index("--episode-method") + 1
                ],
                "hierarchical",
            )
            self.assertEqual(
                objective3_command[
                    objective3_command.index("--block-shift-nodes") + 1
                ],
                "16",
            )

    def test_batch_continues_after_an_isolated_case_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        '{"id":"case_1","reasoning_trace":"One."}',
                        '{"id":"case_2","reasoning_trace":"Two."}',
                        '{"id":"case_3","reasoning_trace":"Three."}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            args = SimpleNamespace(
                input_path=input_path,
                output_dir=output_dir,
                trace_id=None,
                index=None,
                limit=None,
                provider="openai",
                model="model",
                api_key=None,
                base_url=None,
                edge_max_attempts=3,
                semantic_max_attempts=3,
                episode_method="hierarchical",
                consensus_samples=1,
                block_core_nodes=32,
                block_halo_nodes=8,
                block_shift_nodes=0,
                case_concurrency=2,
                resume=False,
                pretty=True,
                verbose=False,
            )

            def execute_case(
                _args,
                _root,
                _env,
                trace_id,
                _attempt_number,
                _attempt_dir,
            ):
                if trace_id == "case_2":
                    return {
                        "status": "FAILED",
                        "return_code": 1,
                        "failure_category": "semantic_contract_failure",
                        "error_summary": "invalid edge",
                    }
                return {
                    "status": "COMPLETED",
                    "return_code": 0,
                    "failure_category": None,
                    "error_summary": None,
                }

            with (
                patch.object(
                    run_pipeline_completo_gcot,
                    "parse_args",
                    return_value=args,
                ),
                patch.object(
                    run_pipeline_completo_gcot,
                    "execute_isolated_case",
                    side_effect=execute_case,
                ) as mocked_execute,
            ):
                self.assertEqual(run_pipeline_completo_gcot.main(), 1)

            self.assertEqual(mocked_execute.call_count, 3)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["summary"]["completed"], 2)
            self.assertEqual(manifest["summary"]["failed"], 1)
            self.assertEqual(manifest["episode_method"], "hierarchical")
            self.assertEqual(manifest["block_core_nodes"], 32)
            self.assertEqual(manifest["block_halo_nodes"], 8)
            self.assertEqual(manifest["block_shift_nodes"], 0)
            self.assertEqual(
                [case["status"] for case in manifest["cases"]],
                ["COMPLETED", "FAILED", "COMPLETED"],
            )

    def test_resume_skips_complete_case_and_retries_failed_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '{"id":"case_1","reasoning_trace":"One."}\n'
                '{"id":"case_2","reasoning_trace":"Two."}\n',
                encoding="utf-8",
            )
            output_dir = root / "output"
            cases_root = output_dir / "cases"

            complete_case = (
                cases_root
                / run_pipeline_completo_gcot.safe_case_directory_name(
                    "case_1"
                )
                / "attempt_001"
            )
            (complete_case / "objective3").mkdir(parents=True)
            (complete_case / "objective3" / "00_sgotgraph_objetivo3.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (complete_case / "status.json").write_text(
                '{"status":"COMPLETED"}\n',
                encoding="utf-8",
            )

            failed_case = (
                cases_root
                / run_pipeline_completo_gcot.safe_case_directory_name(
                    "case_2"
                )
                / "attempt_001"
            )
            failed_case.mkdir(parents=True)
            (failed_case / "status.json").write_text(
                '{"status":"FAILED"}\n',
                encoding="utf-8",
            )

            args = SimpleNamespace(
                input_path=input_path,
                output_dir=output_dir,
                trace_id=None,
                index=None,
                limit=None,
                provider="openai",
                model="model",
                api_key=None,
                base_url=None,
                edge_max_attempts=3,
                semantic_max_attempts=3,
                episode_method="global",
                block_core_nodes=32,
                block_halo_nodes=8,
                block_shift_nodes=0,
                request_timeout_seconds=180.0,
                transport_max_retries=2,
                checkpoint_dir=None,
                case_concurrency=1,
                resume=True,
                pretty=True,
                verbose=False,
            )
            selected = run_pipeline_completo_gcot.load_source_records(
                input_path
            )
            environment = (
                run_pipeline_completo_gcot.build_subprocess_env(args)
            )
            configuration = (
                run_pipeline_completo_gcot.batch_run_configuration(
                    args,
                    PIPELINE_ROOT,
                    environment,
                    selected,
                )
            )
            run_pipeline_completo_gcot.write_json_atomic(
                output_dir / "manifest.json",
                {
                    "schema_version": (
                        run_pipeline_completo_gcot.BATCH_MANIFEST_VERSION
                    ),
                    "configuration": configuration,
                    "configuration_hash": (
                        run_pipeline_completo_gcot.canonical_payload_hash(
                            configuration
                        )
                    ),
                },
            )

            def execute_case(
                _args,
                _root,
                _env,
                trace_id,
                attempt_number,
                _attempt_dir,
            ):
                self.assertEqual(trace_id, "case_2")
                self.assertEqual(attempt_number, 2)
                return {
                    "status": "COMPLETED",
                    "return_code": 0,
                    "failure_category": None,
                    "error_summary": None,
                }

            with (
                patch.object(
                    run_pipeline_completo_gcot,
                    "parse_args",
                    return_value=args,
                ),
                patch.object(
                    run_pipeline_completo_gcot,
                    "execute_isolated_case",
                    side_effect=execute_case,
                ) as mocked_execute,
            ):
                self.assertEqual(run_pipeline_completo_gcot.main(), 0)

            mocked_execute.assert_called_once()
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [case["status"] for case in manifest["cases"]],
                ["SKIPPED_COMPLETE", "COMPLETED"],
            )
            self.assertEqual(manifest["cases"][1]["attempt"], 2)

    def test_output_rejects_a_different_episode_instrument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '{"id":"case_1","reasoning_trace":"One."}\n'
                '{"id":"case_2","reasoning_trace":"Two."}\n',
                encoding="utf-8",
            )
            output_dir = root / "output"

            def args(method: str, resume: bool):
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
                    edge_max_attempts=3,
                    semantic_max_attempts=3,
                    episode_method=method,
                    consensus_samples=1 if method == "hierarchical" else 3,
                    block_core_nodes=32,
                    block_halo_nodes=8,
                    block_shift_nodes=0,
                    request_timeout_seconds=180.0,
                    transport_max_retries=2,
                    checkpoint_dir=None,
                    case_concurrency=1,
                    resume=resume,
                    pretty=True,
                    verbose=False,
                )

            completed = {
                "status": "COMPLETED",
                "return_code": 0,
                "failure_category": None,
                "error_summary": None,
            }
            with (
                patch.object(
                    run_pipeline_completo_gcot,
                    "parse_args",
                    return_value=args("global", False),
                ),
                patch.object(
                    run_pipeline_completo_gcot,
                    "execute_isolated_case",
                    return_value=completed,
                ),
            ):
                self.assertEqual(run_pipeline_completo_gcot.main(), 0)

            for resume in (False, True):
                with self.subTest(resume=resume):
                    with (
                        patch.object(
                            run_pipeline_completo_gcot,
                            "parse_args",
                            return_value=args("hierarchical", resume),
                        ),
                        patch.object(
                            run_pipeline_completo_gcot,
                            "execute_isolated_case",
                        ) as execute_case,
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "no coincide con el manifiesto",
                        ):
                            run_pipeline_completo_gcot.main()
                    execute_case.assert_not_called()

    def test_queued_cases_remain_pending_until_a_worker_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '{"id":"case_1","reasoning_trace":"One."}\n'
                '{"id":"case_2","reasoning_trace":"Two."}\n'
                '{"id":"case_3","reasoning_trace":"Three."}\n',
                encoding="utf-8",
            )
            output_dir = root / "output"
            args = SimpleNamespace(
                input_path=input_path,
                output_dir=output_dir,
                trace_id=None,
                index=None,
                limit=None,
                provider="openai",
                model="model",
                api_key=None,
                base_url=None,
                edge_max_attempts=3,
                semantic_max_attempts=3,
                case_concurrency=1,
                resume=False,
                pretty=True,
                verbose=False,
            )
            first_snapshot: list[str] = []

            def execute_case(
                _args,
                _root,
                _env,
                _trace_id,
                _attempt_number,
                _attempt_dir,
            ):
                if not first_snapshot:
                    manifest = json.loads(
                        (output_dir / "manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    first_snapshot.extend(
                        case["status"] for case in manifest["cases"]
                    )
                return {
                    "status": "COMPLETED",
                    "return_code": 0,
                    "failure_category": None,
                    "error_summary": None,
                }

            with (
                patch.object(
                    run_pipeline_completo_gcot,
                    "parse_args",
                    return_value=args,
                ),
                patch.object(
                    run_pipeline_completo_gcot,
                    "execute_isolated_case",
                    side_effect=execute_case,
                ),
            ):
                self.assertEqual(run_pipeline_completo_gcot.main(), 0)

            self.assertEqual(first_snapshot.count("RUNNING"), 1)
            self.assertEqual(first_snapshot.count("PENDING"), 2)


if __name__ == "__main__":
    unittest.main()
