from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import BaseModel


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.llm_checkpoint import ItemCheckpoint
from util.llm_provider import (
    LLMClient,
    build_llm_client,
    request_structured_response,
)
from util.run_pipeline_completo_gcot import summarize_llm_metrics
from util.stage_cache import (
    build_stage_signature,
    restore_stage_output,
    store_stage_output,
)


class ExampleDraft(BaseModel):
    value: int


class OperationalLLMControlsTests(unittest.TestCase):
    def test_client_receives_explicit_timeout_and_bounded_retries(
        self,
    ) -> None:
        with patch("util.llm_provider.OpenAI") as openai:
            client = build_llm_client(
                "openai",
                "test-key",
                request_timeout_seconds=12.5,
                transport_max_retries=1,
            )

        openai.assert_called_once_with(
            api_key="test-key",
            timeout=12.5,
            max_retries=1,
        )
        self.assertEqual(client.request_timeout_seconds, 12.5)
        self.assertEqual(client.transport_max_retries, 1)

    def test_success_and_error_requests_are_recorded_without_prompts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "requests.jsonl"
            response = SimpleNamespace(
                output_parsed=ExampleDraft(value=7),
                model="resolved-model",
                id="response-id",
                system_fingerprint="fingerprint",
                usage=SimpleNamespace(
                    input_tokens=11,
                    output_tokens=5,
                    total_tokens=16,
                ),
            )
            parse = MagicMock(
                side_effect=[response, TimeoutError("provider stalled")]
            )
            client = LLMClient(
                provider="openai",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            )

            with patch.dict(
                os.environ,
                {
                    "LLM_REQUEST_METRICS_PATH": str(metrics_path),
                    "LLM_STAGE_NAME": "05_aristas",
                },
                clear=False,
            ):
                result = request_structured_response(
                    client,
                    model="requested-model",
                    system_prompt="secret system content",
                    user_prompt="secret user content",
                    response_model=ExampleDraft,
                    reasoning_effort="low",
                )
                with self.assertRaises(TimeoutError):
                    request_structured_response(
                        client,
                        model="requested-model",
                        system_prompt="secret system content",
                        user_prompt="secret user content",
                        response_model=ExampleDraft,
                        reasoning_effort="low",
                    )

            self.assertEqual(result.parsed.value, 7)
            events = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [event["status"] for event in events],
                ["success", "error"],
            )
            self.assertEqual(
                events[0]["token_usage"]["total_tokens"],
                16,
            )
            serialized = metrics_path.read_text(encoding="utf-8")
            self.assertNotIn("secret system content", serialized)
            self.assertNotIn("secret user content", serialized)

            summary = summarize_llm_metrics(metrics_path)
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(summary["successful_requests"], 1)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["total_tokens"], 16)
            self.assertEqual(
                summary["requests_by_stage"],
                {"05_aristas": 2},
            )
            self.assertEqual(
                summary["by_stage"]["05_aristas"]["total_tokens"],
                16,
            )
            self.assertEqual(
                summary["by_stage"]["05_aristas"][
                    "failed_requests"
                ],
                1,
            )
            self.assertEqual(
                summary["errors_by_type"],
                {"TimeoutError": 1},
            )

    def test_checkpoint_reuses_only_an_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            first = ItemCheckpoint(
                path,
                identity={"input": "A", "model": "M"},
                resume=False,
            )
            first.put("item-1", {"decision": "KEEP"})

            exact = ItemCheckpoint(
                path,
                identity={"input": "A", "model": "M"},
                resume=True,
            )
            self.assertEqual(
                exact.get("item-1"),
                {"decision": "KEEP"},
            )
            self.assertEqual(exact.load_status, "resumed")

            incompatible = ItemCheckpoint(
                path,
                identity={"input": "B", "model": "M"},
                resume=True,
            )
            self.assertIsNone(incompatible.get("item-1"))
            self.assertEqual(
                incompatible.load_status,
                "identity_mismatch",
            )

    def test_stage_output_cache_restores_only_an_exact_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_dir = root / "checkpoints"
            output = root / "stage.json"
            output.write_text('{"value": 1}\n', encoding="utf-8")
            signature = {
                "input_hash": "sha256:input",
                "implementation_hash": "sha256:implementation",
                "command": ["stage"],
            }
            store_stage_output(
                checkpoint_dir=checkpoint_dir,
                namespace="objective2",
                stage_name="stage",
                output_path=output,
                signature=signature,
            )
            output.unlink()

            self.assertTrue(
                restore_stage_output(
                    checkpoint_dir=checkpoint_dir,
                    namespace="objective2",
                    stage_name="stage",
                    output_path=output,
                    signature=signature,
                )
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"value": 1},
            )
            self.assertFalse(
                restore_stage_output(
                    checkpoint_dir=checkpoint_dir,
                    namespace="objective2",
                    stage_name="stage",
                    output_path=output,
                    signature={**signature, "input_hash": "sha256:other"},
                )
            )

    def test_stage_signature_ignores_only_operational_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = root / "stage.py"
            input_path = root / "input.json"
            output_path = root / "output.json"
            script.write_text("VALUE = 1\n", encoding="utf-8")
            input_path.write_text('{"input": 1}\n', encoding="utf-8")
            output_path.write_text('{"output": 1}\n', encoding="utf-8")

            base_command = [
                sys.executable,
                "-B",
                script.name,
                str(input_path),
                "-o",
                str(output_path),
                "--model",
                "model-a",
            ]
            common_arguments = {
                "root": root,
                "namespace": "objective3",
                "stage_name": "06_episodios",
                "script_name": script.name,
                "input_path": input_path,
                "output_path": output_path,
                "env": {"OPENAI_MODEL": "model-a"},
            }

            base_signature = build_stage_signature(
                **common_arguments,
                command=base_command,
            )
            operational_signature = build_stage_signature(
                **common_arguments,
                command=[
                    *base_command,
                    "--pretty",
                    "--verbose",
                    "--resume",
                    "--resume-checkpoint",
                    "--resume-internal",
                    "--checkpoint-dir",
                    str(root / "checkpoints"),
                ],
            )
            scientific_change_signature = build_stage_signature(
                **common_arguments,
                command=[
                    *base_command[:-1],
                    "model-b",
                ],
            )

            self.assertEqual(base_signature, operational_signature)
            self.assertNotEqual(
                base_signature,
                scientific_change_signature,
            )


if __name__ == "__main__":
    unittest.main()
