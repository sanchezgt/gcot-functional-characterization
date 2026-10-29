#!/usr/bin/env python3
"""Runner automático 01–09 con aislamiento y reanudación por traza."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import re
import subprocess
import sys
from threading import Lock
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.llm_provider import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TRANSPORT_MAX_RETRIES,
    load_dotenv,
)
from util.consensus import DEFAULT_CONSENSUS_SAMPLES
from util.edge_batching import (
    DEFAULT_PAYLOAD,
    DEFAULT_SOURCES_PER_REQUEST,
    PAYLOADS,
    validate_transport_combination,
)
from util.objective3_hierarchical import (
    DEFAULT_BLOCK_CORE_NODES,
    DEFAULT_BLOCK_HALO_NODES,
    DEFAULT_BLOCK_SHIFT_NODES,
)
from util.stage_cache import implementation_tree_hash


BATCH_MANIFEST_VERSION = "pipeline-batch-manifest-1.2"
CASE_STATUS_VERSION = "pipeline-case-status-1.1"
BATCH_CONFIGURATION_ENV_NAMES = {
    "LLM_PROVIDER",
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_TRANSPORT_MAX_RETRIES",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_REFINEMENT_REASONING_EFFORT",
    "OPENAI_TYPING_REASONING_EFFORT",
    "OPENAI_EDGE_REASONING_EFFORT",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_THINKING",
    "DEEPSEEK_MAX_TOKENS",
}


def parse_args() -> argparse.Namespace:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta SGoTGraph 01–05 y la caracterización funcional 06–09. "
            "Los lotes se aíslan automáticamente por traza."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--trace-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek"],
        default=os.getenv("LLM_PROVIDER", "openai"),
    )
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "LLM_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        ),
        help="Tiempo máximo por solicitud LLM. Default: 180 segundos.",
    )
    parser.add_argument(
        "--transport-max-retries",
        type=int,
        default=int(
            os.getenv(
                "LLM_TRANSPORT_MAX_RETRIES",
                str(DEFAULT_TRANSPORT_MAX_RETRIES),
            )
        ),
        help="Reintentos de transporte por solicitud. Default: 2.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Directorio de checkpoints internos. En lotes se asigna uno "
            "persistente por traza."
        ),
    )
    parser.add_argument(
        "--edge-max-attempts",
        type=int,
        default=3,
        help="Intentos máximos de validación/reparación en la etapa 05. Default: 3.",
    )
    parser.add_argument(
        "--semantic-max-attempts",
        type=int,
        default=3,
        help="Intentos máximos de validación/reparación en 06 y 07. Default: 3.",
    )
    parser.add_argument(
        "--consensus-samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por decisión en 01, 01.1, 02, 05 y 06 "
            "(episodios, solo con --episode-method global); se conserva la "
            "mayoría estricta. 1 desactiva el consenso. "
            f"Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--segmentation-consensus-samples",
        type=int,
        default=None,
        help=(
            "Muestras de consenso solo para la segmentación (01 y 01.1). Si "
            "se omite, se usa --consensus-samples."
        ),
    )
    parser.add_argument(
        "--edge-payload",
        choices=list(PAYLOADS),
        default=DEFAULT_PAYLOAD,
        help=(
            "Carga del prompt de la etapa 05. full (predeterminado, canónico) "
            "o compact (NO canónico; necesario para lotes y ventana)."
        ),
    )
    parser.add_argument(
        "--edge-sources-per-request",
        type=int,
        default=DEFAULT_SOURCES_PER_REQUEST,
        help=(
            "Nodos fuente consecutivos por solicitud en la etapa 05. "
            f"Default: {DEFAULT_SOURCES_PER_REQUEST}."
        ),
    )
    parser.add_argument(
        "--edge-batch-offset",
        type=int,
        default=0,
        help="Desplazamiento de las fronteras de lote en la etapa 05 (medición). Default: 0.",
    )
    parser.add_argument(
        "--edge-candidate-window",
        type=int,
        default=None,
        help=(
            "Opcional y NO canónico: ventana de candidatos en la etapa 05 "
            "(restringe las relaciones expresables). Default: sin ventana."
        ),
    )
    parser.add_argument(
        "--episode-method",
        choices=["global", "hierarchical"],
        default="global",
        help=(
            "Organización de contexto para identificar episodios en 06. "
            "Default: global."
        ),
    )
    parser.add_argument(
        "--block-core-nodes",
        type=int,
        default=DEFAULT_BLOCK_CORE_NODES,
        help="Nodos propietarios por bloque jerárquico. Default: 32.",
    )
    parser.add_argument(
        "--block-halo-nodes",
        type=int,
        default=DEFAULT_BLOCK_HALO_NODES,
        help="Nodos de contexto por lado del bloque. Default: 8.",
    )
    parser.add_argument(
        "--block-shift-nodes",
        type=int,
        default=DEFAULT_BLOCK_SHIFT_NODES,
        help="Desplazamiento de fronteras jerárquicas. Default: 0.",
    )
    parser.add_argument(
        "--case-concurrency",
        type=int,
        default=1,
        help=(
            "Trazas aisladas que se ejecutan simultáneamente en modo lote. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Omite trazas completas y reutiliza checkpoints internos "
            "compatibles de las pendientes o fallidas."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_episode_method_args(args: argparse.Namespace) -> None:
    """Completa y valida la configuración científica de la etapa 06."""

    args.episode_method = getattr(args, "episode_method", "global")
    args.consensus_samples = getattr(
        args,
        "consensus_samples",
        DEFAULT_CONSENSUS_SAMPLES,
    )
    args.edge_sources_per_request = getattr(
        args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST
    )
    args.edge_batch_offset = getattr(args, "edge_batch_offset", 0)
    args.edge_candidate_window = getattr(args, "edge_candidate_window", None)
    if args.edge_sources_per_request <= 0:
        raise ValueError("--edge-sources-per-request debe ser mayor que 0.")
    if not 0 <= args.edge_batch_offset < args.edge_sources_per_request:
        raise ValueError("--edge-batch-offset debe estar entre 0 y --edge-sources-per-request - 1.")
    if args.edge_candidate_window is not None and args.edge_candidate_window <= 0:
        raise ValueError("--edge-candidate-window debe ser mayor que 0.")
    args.segmentation_consensus_samples = getattr(
        args, "segmentation_consensus_samples", None
    )
    if (
        args.segmentation_consensus_samples is not None
        and args.segmentation_consensus_samples <= 0
    ):
        raise ValueError("--segmentation-consensus-samples debe ser mayor que 0.")
    args.edge_payload = getattr(args, "edge_payload", DEFAULT_PAYLOAD)
    validate_transport_combination(
        args.edge_payload,
        args.edge_sources_per_request,
        args.edge_candidate_window,
    )
    args.block_core_nodes = getattr(
        args,
        "block_core_nodes",
        DEFAULT_BLOCK_CORE_NODES,
    )
    args.block_halo_nodes = getattr(
        args,
        "block_halo_nodes",
        DEFAULT_BLOCK_HALO_NODES,
    )
    args.block_shift_nodes = getattr(
        args,
        "block_shift_nodes",
        DEFAULT_BLOCK_SHIFT_NODES,
    )
    if args.episode_method not in {"global", "hierarchical"}:
        raise ValueError(
            "--episode-method debe ser global o hierarchical."
        )
    if args.episode_method != "global" and args.consensus_samples != 1:
        # La etapa 06 rechaza esa combinación; sin esta comprobación el fallo
        # llega después de haber ejecutado y facturado las etapas 01–05.
        raise ValueError(
            "--episode-method hierarchical requiere --consensus-samples 1: "
            "el consenso multimuestra de la etapa 06 solo existe para el "
            "método global."
        )
    if args.block_core_nodes <= 0:
        raise ValueError("--block-core-nodes debe ser mayor que 0.")
    if args.block_halo_nodes < 0:
        raise ValueError("--block-halo-nodes no puede ser negativo.")
    if (
        args.block_shift_nodes < 0
        or args.block_shift_nodes >= args.block_core_nodes
    ):
        raise ValueError(
            "--block-shift-nodes debe estar entre 0 y "
            "--block-core-nodes - 1."
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def batch_run_configuration(
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identidad sin secretos del instrumento y la selección de un lote."""

    return {
        "input_path": str(args.input_path.resolve()),
        "input_sha256": file_sha256(args.input_path.resolve()),
        "selected_trace_ids": [record["id"] for record in selected],
        "provider": args.provider,
        "model_argument": args.model,
        "base_url_argument": args.base_url,
        "edge_max_attempts": args.edge_max_attempts,
        "semantic_max_attempts": args.semantic_max_attempts,
        "consensus_samples": getattr(args, "consensus_samples", DEFAULT_CONSENSUS_SAMPLES),
        "edge_sources_per_request": getattr(
            args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST
        ),
        "segmentation_consensus_samples": getattr(
            args, "segmentation_consensus_samples", None
        ),
        "edge_payload": getattr(args, "edge_payload", DEFAULT_PAYLOAD),
        "edge_batch_offset": getattr(args, "edge_batch_offset", 0),
        "edge_candidate_window": getattr(args, "edge_candidate_window", None),
        "episode_method": args.episode_method,
        "block_core_nodes": args.block_core_nodes,
        "block_halo_nodes": args.block_halo_nodes,
        "block_shift_nodes": args.block_shift_nodes,
        "request_timeout_seconds": args.request_timeout_seconds,
        "transport_max_retries": args.transport_max_retries,
        "configuration_environment": {
            name: env[name]
            for name in sorted(BATCH_CONFIGURATION_ENV_NAMES)
            if name in env
        },
        "implementation_hash": implementation_tree_hash(root),
    }


def ensure_batch_manifest_compatible(
    manifest_path: Path,
    cases_root: Path,
    configuration: dict[str, Any],
) -> None:
    """Impide mezclar instrumentos o selecciones en un directorio de lote."""

    prior_attempts_exist = cases_root.exists() and any(
        path.is_dir() for path in cases_root.rglob("attempt_*")
    )
    if not manifest_path.exists():
        if prior_attempts_exist:
            raise ValueError(
                "No se puede continuar: existen intentos previos pero falta "
                "el manifiesto que identifica su configuración."
            )
        return
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "No se puede continuar: el manifiesto previo no es JSON válido."
        ) from exc
    expected_hash = canonical_payload_hash(configuration)
    if (
        previous.get("schema_version") != BATCH_MANIFEST_VERSION
        or previous.get("configuration") != configuration
        or previous.get("configuration_hash") != expected_hash
    ):
        raise ValueError(
            "No se puede usar este directorio porque la entrada, selección, "
            "implementación o configuración no coincide con el manifiesto "
            "previo. Use un directorio de salida nuevo."
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_llm_metrics(metrics_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": "llm-request-summary-v1",
        "metrics_path": str(metrics_path.resolve()),
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "duration_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "requests_without_token_usage": 0,
        "errors_by_type": {},
        "requests_by_stage": {},
        "requests_by_model": {},
        "by_stage": {},
    }
    if not metrics_path.exists():
        return summary

    for line in metrics_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        stage = event.get("stage")
        if not isinstance(stage, str) or not stage:
            stage = "unspecified"
        stage_summary = summary["by_stage"].setdefault(
            stage,
            {
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "duration_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "requests_without_token_usage": 0,
                "errors_by_type": {},
            },
        )
        summary["requests"] += 1
        stage_summary["requests"] += 1
        status = event.get("status")
        if status == "success":
            summary["successful_requests"] += 1
            stage_summary["successful_requests"] += 1
        else:
            summary["failed_requests"] += 1
            stage_summary["failed_requests"] += 1
            error_type = event.get("error_type", "unknown")
            errors = summary["errors_by_type"]
            errors[error_type] = errors.get(error_type, 0) + 1
            stage_errors = stage_summary["errors_by_type"]
            stage_errors[error_type] = (
                stage_errors.get(error_type, 0) + 1
            )

        duration = event.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(
            duration, bool
        ):
            summary["duration_seconds"] += float(duration)
            stage_summary["duration_seconds"] += float(duration)

        model = event.get("resolved_model") or event.get("requested_model")
        if isinstance(model, str):
            models = summary["requests_by_model"]
            models[model] = models.get(model, 0) + 1
        stages = summary["requests_by_stage"]
        stages[stage] = stages.get(stage, 0) + 1

        usage = event.get("token_usage")
        if not isinstance(usage, dict) or not usage:
            summary["requests_without_token_usage"] += 1
            stage_summary["requests_without_token_usage"] += 1
            continue
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                summary[key] += value
                stage_summary[key] += value

    summary["duration_seconds"] = round(
        summary["duration_seconds"],
        6,
    )
    for stage_summary in summary["by_stage"].values():
        stage_summary["duration_seconds"] = round(
            stage_summary["duration_seconds"],
            6,
        )
    return summary


def write_llm_metrics_summary(
    output_dir: Path,
    metrics_path: Path,
) -> Path:
    summary_path = output_dir / "llm_metrics_summary.json"
    write_json_atomic(
        summary_path,
        summarize_llm_metrics(metrics_path),
    )
    return summary_path


def load_source_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No se encontraron registros en {path}.")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list) and all(
            isinstance(item, dict) for item in data
        ):
            records = data
        else:
            raise ValueError(
                "El JSON debe contener un objeto o una lista de objetos."
            )
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Línea {line_number}: JSON inválido: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Línea {line_number}: cada registro debe ser un objeto."
                )
            records.append(value)
    if not records:
        raise ValueError(f"No se encontraron registros en {path}.")
    return records


def select_source_records(
    records: list[dict[str, Any]],
    trace_id: str | None,
    index: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if trace_id is not None and index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")
    selected = records
    if trace_id is not None:
        selected = [record for record in records if record.get("id") == trace_id]
        if not selected:
            raise ValueError(f"No existe un registro con id={trace_id!r}.")
    elif index is not None:
        if index < 0 or index >= len(records):
            raise ValueError(f"--index debe estar entre 0 y {len(records) - 1}.")
        selected = [records[index]]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit debe ser mayor que 0.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("La selección quedó vacía.")

    seen: set[str] = set()
    for position, record in enumerate(selected):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                f"Registro seleccionado {position}: falta un id válido."
            )
        if record_id in seen:
            raise ValueError(f"id duplicado en la selección: {record_id!r}.")
        seen.add(record_id)
    return selected


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = args.provider
    if args.api_key:
        env[
            "OPENAI_API_KEY"
            if args.provider == "openai"
            else "DEEPSEEK_API_KEY"
        ] = args.api_key
    if args.model:
        env[
            "OPENAI_MODEL"
            if args.provider == "openai"
            else "DEEPSEEK_MODEL"
        ] = args.model
    if args.base_url:
        env[
            "OPENAI_BASE_URL"
            if args.provider == "openai"
            else "DEEPSEEK_BASE_URL"
        ] = args.base_url
    env["LLM_REQUEST_TIMEOUT_SECONDS"] = str(
        args.request_timeout_seconds
    )
    env["LLM_TRANSPORT_MAX_RETRIES"] = str(
        args.transport_max_retries
    )
    return env


def run_single_pipeline(
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
) -> int:
    normalize_episode_method_args(args)
    output_dir = args.output_dir.resolve()
    objective2_dir = output_dir / "objective2"
    objective3_dir = output_dir / "objective3"
    objective2_dir.mkdir(parents=True, exist_ok=True)
    objective3_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else output_dir / "checkpoints"
    )
    metrics_path = output_dir / "llm_requests.jsonl"
    if not args.resume and metrics_path.exists():
        metrics_path.unlink()
    run_env = env.copy()
    run_env["LLM_REQUEST_METRICS_PATH"] = str(metrics_path)

    objective2_cmd = [
        sys.executable,
        "-B",
        "util/run_pipeline_gcot.py",
        str(args.input_path.resolve()),
        "-o",
        str(objective2_dir),
        "--provider",
        args.provider,
        "--edge-max-attempts",
        str(args.edge_max_attempts),
        "--consensus-samples",
        str(getattr(args, "consensus_samples", DEFAULT_CONSENSUS_SAMPLES)),
        "--edge-sources-per-request",
        str(getattr(args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST)),
        *(
            ["--segmentation-consensus-samples", str(args.segmentation_consensus_samples)]
            if getattr(args, "segmentation_consensus_samples", None) is not None
            else []
        ),
        "--edge-payload",
        getattr(args, "edge_payload", DEFAULT_PAYLOAD),
        "--edge-batch-offset",
        str(getattr(args, "edge_batch_offset", 0)),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--transport-max-retries",
        str(args.transport_max_retries),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if getattr(args, "edge_candidate_window", None) is not None:
        objective2_cmd.extend(
            ["--edge-candidate-window", str(args.edge_candidate_window)]
        )
    if args.resume:
        objective2_cmd.append("--resume-internal")
    if args.model:
        objective2_cmd.extend(["--model", args.model])
    if args.base_url:
        objective2_cmd.extend(["--base-url", args.base_url])
    if args.trace_id:
        objective2_cmd.extend(["--trace-id", args.trace_id])
    if args.index is not None:
        objective2_cmd.extend(["--index", str(args.index)])
    if args.limit is not None:
        objective2_cmd.extend(["--limit", str(args.limit)])
    elif (
        args.input_path.suffix.lower() != ".json"
        and args.trace_id is None
        and args.index is None
    ):
        # ``main`` entra aquí sin selector únicamente cuando el archivo
        # contiene un solo registro. Fuerza salida JSON, no JSONL.
        objective2_cmd.extend(["--limit", "1"])
    if args.pretty:
        objective2_cmd.append("--pretty")
    if args.verbose:
        objective2_cmd.append("--verbose")
        print("[pipeline completo] Ejecutando objetivo 2.", file=sys.stderr)
    try:
        subprocess.run(
            objective2_cmd,
            cwd=root,
            check=True,
            env=run_env,
        )
    except Exception:
        write_llm_metrics_summary(output_dir, metrics_path)
        raise

    objective2_artifact = objective2_dir / "00_sgotgraph_objetivo2.json"
    objective3_cmd = [
        sys.executable,
        "-B",
        "util/run_pipeline_objetivo3.py",
        str(objective2_artifact),
        "-o",
        str(objective3_dir),
        "--provider",
        args.provider,
        "--semantic-max-attempts",
        str(args.semantic_max_attempts),
        "--consensus-samples",
        str(getattr(args, "consensus_samples", DEFAULT_CONSENSUS_SAMPLES)),
        "--episode-method",
        args.episode_method,
        "--block-core-nodes",
        str(args.block_core_nodes),
        "--block-halo-nodes",
        str(args.block_halo_nodes),
        "--block-shift-nodes",
        str(args.block_shift_nodes),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--transport-max-retries",
        str(args.transport_max_retries),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.resume:
        objective3_cmd.append("--resume-internal")
    if args.model:
        objective3_cmd.extend(["--model", args.model])
    if args.base_url:
        objective3_cmd.extend(["--base-url", args.base_url])
    if args.pretty:
        objective3_cmd.append("--pretty")
    if args.verbose:
        objective3_cmd.append("--verbose")
        print("[pipeline completo] Ejecutando objetivo 3.", file=sys.stderr)
    try:
        subprocess.run(
            objective3_cmd,
            cwd=root,
            check=True,
            env=run_env,
        )
    finally:
        write_llm_metrics_summary(output_dir, metrics_path)
    return 0


def safe_case_directory_name(trace_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", trace_id).strip("._-")
    readable = readable[:80] or "trace"
    digest = sha256(trace_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}--{digest}"


def attempt_directories(case_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in case_dir.glob("attempt_*")
            if path.is_dir() and path.name[8:].isdigit()
        ),
        key=lambda path: int(path.name[8:]),
    )


def next_attempt_number(case_dir: Path) -> int:
    attempts = attempt_directories(case_dir)
    return int(attempts[-1].name[8:]) + 1 if attempts else 1


def completed_attempt(case_dir: Path) -> Path | None:
    for attempt_dir in reversed(attempt_directories(case_dir)):
        status_path = attempt_dir / "status.json"
        artifact = (
            attempt_dir
            / "objective3"
            / "00_sgotgraph_objetivo3.json"
        )
        if not status_path.exists() or not artifact.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "COMPLETED":
            return attempt_dir
    return None


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if any(
        marker in lowered
        for marker in (
            "apitimeouterror",
            "readtimeout",
            "connecttimeout",
            "request timed out",
            "timed out",
        )
    ):
        return "provider_timeout"
    if "invalid_prompt" in log_text:
        return "provider_policy_rejection"
    if "No se pudo validar la segmentacion" in log_text:
        return "literal_segmentation_failure"
    if "No se pudo validar el refinamiento" in log_text:
        if any(
            marker in log_text
            for marker in (
                "reconstruye exactamente",
                "unidad literal",
                "unidades literales",
                "KEEP debe cubrir",
            )
        ):
            return "literal_refinement_failure"
        return "segmentation_refinement_failure"
    if "Salida inválida para" in log_text:
        return "semantic_contract_failure"
    if "Validación:" in log_text and "INVALID" in log_text:
        return "final_validation_failure"
    return "pipeline_failure"


def last_error_line(log_text: str) -> str | None:
    fallback: str | None = None
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("Error:"):
            fallback = fallback or stripped[:1000]
            if not stripped.startswith("Error: Command "):
                return stripped[:1000]
    return fallback


def build_case_command(
    args: argparse.Namespace,
    trace_id: str,
    attempt_dir: Path,
) -> list[str]:
    normalize_episode_method_args(args)
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        / safe_case_directory_name(trace_id)
        if args.checkpoint_dir is not None
        else attempt_dir.parent / "checkpoints"
    )
    command = [
        sys.executable,
        "-B",
        "util/run_pipeline_completo_gcot.py",
        str(args.input_path.resolve()),
        "-o",
        str(attempt_dir),
        "--trace-id",
        trace_id,
        "--provider",
        args.provider,
        "--edge-max-attempts",
        str(args.edge_max_attempts),
        "--semantic-max-attempts",
        str(args.semantic_max_attempts),
        "--consensus-samples",
        str(getattr(args, "consensus_samples", DEFAULT_CONSENSUS_SAMPLES)),
        "--edge-sources-per-request",
        str(getattr(args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST)),
        *(
            ["--segmentation-consensus-samples", str(args.segmentation_consensus_samples)]
            if getattr(args, "segmentation_consensus_samples", None) is not None
            else []
        ),
        "--edge-payload",
        getattr(args, "edge_payload", DEFAULT_PAYLOAD),
        "--edge-batch-offset",
        str(getattr(args, "edge_batch_offset", 0)),
        *(
            ["--edge-candidate-window", str(args.edge_candidate_window)]
            if getattr(args, "edge_candidate_window", None) is not None
            else []
        ),
        "--episode-method",
        args.episode_method,
        "--block-core-nodes",
        str(args.block_core_nodes),
        "--block-halo-nodes",
        str(args.block_halo_nodes),
        "--block-shift-nodes",
        str(args.block_shift_nodes),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--transport-max-retries",
        str(args.transport_max_retries),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--verbose",
    ]
    if args.resume:
        command.append("--resume")
    if args.model:
        command.extend(["--model", args.model])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.pretty:
        command.append("--pretty")
    return command


def execute_isolated_case(
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    trace_id: str,
    attempt_number: int,
    attempt_dir: Path,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "execution.log"
    status_path = attempt_dir / "status.json"
    started_at = utc_now()
    status: dict[str, Any] = {
        "schema_version": CASE_STATUS_VERSION,
        "trace_id": trace_id,
        "attempt": attempt_number,
        "status": "RUNNING",
        "started_at": started_at,
        "completed_at": None,
        "return_code": None,
        "failure_category": None,
        "error_summary": None,
    }
    write_json_atomic(status_path, status)

    command = build_case_command(args, trace_id, attempt_dir)
    return_code = 1
    execution_error: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            return_code = completed.returncode
    except OSError as exc:
        execution_error = str(exc)
        log_path.write_text(
            f"Error al iniciar la ejecución aislada: {exc}\n",
            encoding="utf-8",
        )

    artifact = (
        attempt_dir / "objective3" / "00_sgotgraph_objetivo3.json"
    )
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    completed_ok = return_code == 0 and artifact.exists()
    if return_code == 0 and not artifact.exists():
        execution_error = "La ejecución terminó sin artefacto canónico."
    metrics_summary_path = attempt_dir / "llm_metrics_summary.json"
    metrics_summary: dict[str, Any] | None = None
    if metrics_summary_path.exists():
        try:
            loaded_summary = json.loads(
                metrics_summary_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            loaded_summary = None
        if isinstance(loaded_summary, dict):
            metrics_summary = loaded_summary
    resume_reports: dict[str, Any] = {}
    for objective_name in ("objective2", "objective3"):
        report_path = attempt_dir / objective_name / "resume_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            resume_reports[objective_name] = report

    status.update(
        {
            "status": "COMPLETED" if completed_ok else "FAILED",
            "completed_at": utc_now(),
            "return_code": return_code,
            "failure_category": (
                None if completed_ok else classify_failure(log_text)
            ),
            "error_summary": (
                None
                if completed_ok
                else execution_error or last_error_line(log_text)
            ),
            "canonical_artifact": (
                str(artifact.resolve()) if completed_ok else None
            ),
            "llm_metrics_summary_path": (
                str(metrics_summary_path.resolve())
                if metrics_summary_path.exists()
                else None
            ),
            "llm_metrics_summary": metrics_summary,
            "resume_reports": resume_reports,
        }
    )
    write_json_atomic(status_path, status)
    return status


def update_manifest_summary(manifest: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for case in manifest["cases"]:
        status = case["status"]
        counts[status] = counts.get(status, 0) + 1
    manifest["summary"] = {
        "selected": len(manifest["cases"]),
        "completed": counts.get("COMPLETED", 0)
        + counts.get("SKIPPED_COMPLETE", 0),
        "failed": counts.get("FAILED", 0),
        "pending": counts.get("PENDING", 0) + counts.get("RUNNING", 0),
        "status_counts": counts,
    }
    manifest["updated_at"] = utc_now()


def run_isolated_batch(
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    selected: list[dict[str, Any]],
) -> int:
    if args.case_concurrency <= 0:
        raise ValueError("--case-concurrency debe ser mayor que 0.")

    output_dir = args.output_dir.resolve()
    cases_root = output_dir / "cases"
    manifest_path = output_dir / "manifest.json"
    configuration = batch_run_configuration(args, root, env, selected)
    ensure_batch_manifest_compatible(
        manifest_path,
        cases_root,
        configuration,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": BATCH_MANIFEST_VERSION,
        "configuration": configuration,
        "configuration_hash": canonical_payload_hash(configuration),
        "input_path": str(args.input_path.resolve()),
        "provider": args.provider,
        "model": args.model,
        "edge_max_attempts": args.edge_max_attempts,
        "semantic_max_attempts": args.semantic_max_attempts,
        "consensus_samples": getattr(args, "consensus_samples", DEFAULT_CONSENSUS_SAMPLES),
        "edge_sources_per_request": getattr(
            args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST
        ),
        "segmentation_consensus_samples": getattr(
            args, "segmentation_consensus_samples", None
        ),
        "edge_payload": getattr(args, "edge_payload", DEFAULT_PAYLOAD),
        "edge_batch_offset": getattr(args, "edge_batch_offset", 0),
        "edge_candidate_window": getattr(args, "edge_candidate_window", None),
        "episode_method": args.episode_method,
        "block_core_nodes": args.block_core_nodes,
        "block_halo_nodes": args.block_halo_nodes,
        "block_shift_nodes": args.block_shift_nodes,
        "request_timeout_seconds": args.request_timeout_seconds,
        "transport_max_retries": args.transport_max_retries,
        "internal_checkpoint_stages": [
            "01_1_refinamiento",
            "05_aristas_semanticas",
        ],
        "case_concurrency": args.case_concurrency,
        "resume": args.resume,
        "started_at": utc_now(),
        "updated_at": None,
        "cases": [],
        "summary": {},
    }
    pending: list[tuple[int, str, int, Path]] = []

    for selection_index, record in enumerate(selected):
        trace_id = record["id"]
        case_dir = cases_root / safe_case_directory_name(trace_id)
        prior_complete = completed_attempt(case_dir) if args.resume else None
        if prior_complete is not None:
            entry = {
                "selection_index": selection_index,
                "trace_id": trace_id,
                "status": "SKIPPED_COMPLETE",
                "attempt": int(prior_complete.name[8:]),
                "attempt_dir": str(prior_complete.relative_to(output_dir)),
                "failure_category": None,
                "error_summary": None,
            }
        else:
            attempt_number = next_attempt_number(case_dir)
            attempt_dir = case_dir / f"attempt_{attempt_number:03d}"
            entry = {
                "selection_index": selection_index,
                "trace_id": trace_id,
                "status": "PENDING",
                "attempt": attempt_number,
                "attempt_dir": str(attempt_dir.relative_to(output_dir)),
                "failure_category": None,
                "error_summary": None,
            }
            pending.append(
                (selection_index, trace_id, attempt_number, attempt_dir)
            )
        manifest["cases"].append(entry)

    update_manifest_summary(manifest)
    write_json_atomic(manifest_path, manifest)

    max_workers = min(args.case_concurrency, max(1, len(pending)))
    manifest_lock = Lock()

    def execute_pending_case(
        selection_index: int,
        trace_id: str,
        attempt_number: int,
        attempt_dir: Path,
    ) -> dict[str, Any]:
        with manifest_lock:
            manifest["cases"][selection_index]["status"] = "RUNNING"
            update_manifest_summary(manifest)
            write_json_atomic(manifest_path, manifest)
        return execute_isolated_case(
            args,
            root,
            env,
            trace_id,
            attempt_number,
            attempt_dir,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case: dict[Any, tuple[int, str]] = {}
        for selection_index, trace_id, attempt_number, attempt_dir in pending:
            future = executor.submit(
                execute_pending_case,
                selection_index,
                trace_id,
                attempt_number,
                attempt_dir,
            )
            future_to_case[future] = (selection_index, trace_id)

        for future in as_completed(future_to_case):
            selection_index, trace_id = future_to_case[future]
            try:
                case_status = future.result()
            except Exception as exc:
                case_status = {
                    "status": "FAILED",
                    "return_code": 1,
                    "failure_category": "runner_internal_failure",
                    # el tipo evita resúmenes vacíos cuando la excepción no trae mensaje
                    "error_summary": f"{type(exc).__name__}: {exc}",
                }
            with manifest_lock:
                entry = manifest["cases"][selection_index]
                entry.update(
                    {
                        "status": case_status["status"],
                        "return_code": case_status.get("return_code"),
                        "failure_category": case_status.get(
                            "failure_category"
                        ),
                        "error_summary": case_status.get("error_summary"),
                    }
                )
                if args.verbose:
                    print(
                        f"[pipeline lote] {trace_id}: {entry['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
                update_manifest_summary(manifest)
                write_json_atomic(manifest_path, manifest)

    return 1 if manifest["summary"]["failed"] else 0


def main() -> int:
    args = parse_args()
    args.request_timeout_seconds = getattr(
        args,
        "request_timeout_seconds",
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    args.transport_max_retries = getattr(
        args,
        "transport_max_retries",
        DEFAULT_TRANSPORT_MAX_RETRIES,
    )
    args.checkpoint_dir = getattr(args, "checkpoint_dir", None)
    args.resume = getattr(args, "resume", False)
    normalize_episode_method_args(args)
    if args.trace_id is not None and args.index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")
    if args.edge_max_attempts <= 0:
        raise ValueError("--edge-max-attempts debe ser mayor que 0.")
    if args.semantic_max_attempts <= 0:
        raise ValueError("--semantic-max-attempts debe ser mayor que 0.")
    if args.consensus_samples <= 0:
        raise ValueError("--consensus-samples debe ser mayor que 0.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds debe ser mayor que 0.")
    if args.transport_max_retries < 0:
        raise ValueError("--transport-max-retries no puede ser negativo.")

    root = Path(__file__).resolve().parent.parent
    records = load_source_records(args.input_path.resolve())
    selected = select_source_records(
        records,
        args.trace_id,
        args.index,
        args.limit,
    )
    env = build_subprocess_env(args)

    single_selection = (
        args.trace_id is not None
        or args.index is not None
        or args.limit == 1
        or len(selected) == 1
    )
    if single_selection:
        return run_single_pipeline(args, root, env)
    return run_isolated_batch(args, root, env, selected)


def cli() -> int:
    try:
        return main()
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
