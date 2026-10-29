#!/usr/bin/env python3
"""Runner canónico del pipeline SGoTGraph para el objetivo específico 2."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.consensus import DEFAULT_CONSENSUS_SAMPLES
from util.edge_batching import (
    DEFAULT_PAYLOAD,
    DEFAULT_SOURCES_PER_REQUEST,
    PAYLOADS,
    validate_transport_combination,
)
from util.llm_provider import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TRANSPORT_MAX_RETRIES,
    load_dotenv,
)
from util.stage_cache import (
    build_stage_signature,
    restore_stage_output,
    store_stage_output,
)


STAGES: list[tuple[str, str]] = [
    ("01_segmentacion_gcot.py", "01_segmentacion"),
    ("01_1_refinamiento_segmentacion_gcot.py", "01_1_refinamiento"),
    ("02_tipado_gcot.py", "02_tipado"),
    ("03_validacion_tipado_gcot.py", "03_validacion_tipado"),
    ("04_backbone_next_gcot.py", "04_backbone"),
    ("05_aristas_semanticas_gcot.py", "05_aristas"),
]

VERBOSE_STAGE_SCRIPTS = {
    "01_segmentacion_gcot.py",
    "01_1_refinamiento_segmentacion_gcot.py",
    "02_tipado_gcot.py",
    "05_aristas_semanticas_gcot.py",
}

LLM_STAGE_SCRIPTS = set(VERBOSE_STAGE_SCRIPTS)
SEGMENTATION_STAGE_SCRIPTS = {
    "01_segmentacion_gcot.py",
    "01_1_refinamiento_segmentacion_gcot.py",
}
CONSENSUS_STAGE_SCRIPTS = {
    "01_segmentacion_gcot.py",
    "01_1_refinamiento_segmentacion_gcot.py",
    "02_tipado_gcot.py",
    "05_aristas_semanticas_gcot.py",
}


def parse_args() -> argparse.Namespace:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline SGoTGraph 01-05 y genera el artefacto canónico "
            "del objetivo específico 2."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Archivo fuente JSON o JSONL con las gCoTs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio donde se guardarán las etapas y el artefacto canónico final.",
    )
    parser.add_argument("--trace-id", help="Procesa solo el registro con este id.")
    parser.add_argument("--index", type=int, help="Procesa solo el registro en esta posicion 0-based.")
    parser.add_argument("--limit", type=int, help="Procesa solo los primeros N registros seleccionados.")
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="Proveedor LLM para las etapas con IA. Default: LLM_PROVIDER u openai.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Modelo LLM para las etapas con IA. Si se omite, cada etapa usa "
            "el default del proveedor seleccionado."
        ),
    )
    parser.add_argument(
        "--api-key",
        help=(
            "API key del proveedor. No se propaga por linea de comandos; se inyecta "
            "como OPENAI_API_KEY o DEEPSEEK_API_KEY en los subprocesos."
        ),
    )
    parser.add_argument(
        "--base-url",
        help="Base URL opcional del proveedor. Util para endpoints compatibles.",
    )
    parser.add_argument(
        "--edge-max-attempts",
        type=int,
        default=3,
        help=(
            "Intentos máximos por nodo para validar/reparar aristas "
            "semánticas en la etapa 05. Default: 3."
        ),
    )
    parser.add_argument(
        "--consensus-samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por decisión en 01, 01.1, 02 y 05; "
            "se conserva la mayoría estricta. 1 desactiva el consenso. "
            f"Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--segmentation-consensus-samples",
        type=int,
        default=None,
        help=(
            "Muestras de consenso solo para la segmentación (01 y 01.1). Si "
            "se omite, se usa --consensus-samples. La estabilidad de las "
            "etapas posteriores sigue a la de la segmentación."
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
            "Nodos fuente consecutivos por solicitud en la etapa 05 (la "
            f"aceptación sigue siendo por fuente). Default: {DEFAULT_SOURCES_PER_REQUEST}."
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
            "Opcional y NO canónico: ventana de candidatos (distancia de "
            "posición) en la etapa 05; restringe las relaciones expresables y "
            "marca el artefacto como variante 'windowed'. Default: sin ventana."
        ),
    )
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
        help="Directorio compartido para checkpoints internos de 01.1 y 05.",
    )
    parser.add_argument(
        "--resume-internal",
        action="store_true",
        help="Reutiliza checkpoints internos compatibles de 01.1 y 05.",
    )
    parser.add_argument("--pretty", action="store_true", help="Usa salida indentada para corridas JSON de un solo caso.")
    parser.add_argument("--verbose", action="store_true", help="Muestra progreso de las etapas.")
    return parser.parse_args()


def is_single_selection(args: argparse.Namespace, input_path: Path) -> bool:
    if input_path.suffix.lower() == ".json":
        return True
    if args.trace_id is not None or args.index is not None:
        return True
    if args.limit == 1:
        return True
    return False


def output_suffix_for(input_path: Path, args: argparse.Namespace) -> str:
    return ".json" if is_single_selection(args, input_path) else ".jsonl"


def stage_output_path(output_dir: Path, stage_name: str, suffix: str) -> Path:
    return output_dir / f"{stage_name}{suffix}"


def build_stage_command(
    script_name: str,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    include_selection: bool,
) -> list[str]:
    cmd = [sys.executable, "-B", script_name, str(input_path), "-o", str(output_path)]
    if include_selection and args.trace_id is not None:
        cmd.extend(["--trace-id", args.trace_id])
    if include_selection and args.index is not None:
        cmd.extend(["--index", str(args.index)])
    if include_selection and args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if script_name in LLM_STAGE_SCRIPTS:
        cmd.extend(["--provider", args.provider])
        cmd.extend(
            [
                "--request-timeout-seconds",
                str(args.request_timeout_seconds),
                "--transport-max-retries",
                str(args.transport_max_retries),
            ]
        )
        if args.model:
            cmd.extend(["--model", args.model])
        if args.base_url:
            cmd.extend(["--base-url", args.base_url])
    if script_name == "05_aristas_semanticas_gcot.py":
        cmd.extend(["--max-attempts", str(args.edge_max_attempts)])
        cmd.extend(["--payload", getattr(args, "edge_payload", DEFAULT_PAYLOAD)])
        cmd.extend(
            [
                "--sources-per-request",
                str(getattr(args, "edge_sources_per_request", DEFAULT_SOURCES_PER_REQUEST)),
                "--batch-offset",
                str(getattr(args, "edge_batch_offset", 0)),
            ]
        )
        window = getattr(args, "edge_candidate_window", None)
        if window is not None:
            cmd.extend(["--candidate-window", str(window)])
    if script_name in CONSENSUS_STAGE_SCRIPTS:
        samples = args.consensus_samples
        segmentation_samples = getattr(args, "segmentation_consensus_samples", None)
        if script_name in SEGMENTATION_STAGE_SCRIPTS and segmentation_samples is not None:
            samples = segmentation_samples
        cmd.extend(["--samples", str(samples)])
    if (
        script_name
        in {
            "01_1_refinamiento_segmentacion_gcot.py",
            "05_aristas_semanticas_gcot.py",
        }
        and args.checkpoint_dir is not None
    ):
        cmd.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
        if args.resume_internal:
            cmd.append("--resume-checkpoint")
    if args.pretty:
        cmd.append("--pretty")
    if args.verbose and script_name in VERBOSE_STAGE_SCRIPTS:
        cmd.append("--verbose")
    return cmd


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = args.provider

    if args.api_key:
        key_name = "OPENAI_API_KEY" if args.provider == "openai" else "DEEPSEEK_API_KEY"
        env[key_name] = args.api_key

    if args.model:
        model_name = "OPENAI_MODEL" if args.provider == "openai" else "DEEPSEEK_MODEL"
        env[model_name] = args.model

    if args.base_url:
        base_url_name = "OPENAI_BASE_URL" if args.provider == "openai" else "DEEPSEEK_BASE_URL"
        env[base_url_name] = args.base_url

    env["LLM_REQUEST_TIMEOUT_SECONDS"] = str(
        args.request_timeout_seconds
    )
    env["LLM_TRANSPORT_MAX_RETRIES"] = str(
        args.transport_max_retries
    )

    return env


def run_command(cmd: list[str], workdir: Path, verbose: bool, env: dict[str, str] | None = None) -> None:
    if verbose:
        print(f"[run_pipeline] Ejecutando: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=workdir, check=True, env=env)


def main() -> int:
    args = parse_args()
    if args.trace_id is not None and args.index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds debe ser mayor que 0.")
    if args.transport_max_retries < 0:
        raise ValueError("--transport-max-retries no puede ser negativo.")
    args.consensus_samples = getattr(
        args,
        "consensus_samples",
        DEFAULT_CONSENSUS_SAMPLES,
    )
    if args.consensus_samples <= 0:
        raise ValueError("--consensus-samples debe ser mayor que 0.")
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
    if args.resume_internal and args.checkpoint_dir is None:
        raise ValueError("--resume-internal requiere --checkpoint-dir.")

    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = output_suffix_for(input_path, args)
    stage_outputs: list[Path] = []
    restored_stages: list[str] = []
    started_stages: list[str] = []
    executed_stages: list[str] = []
    current_input = input_path

    def write_resume_report() -> None:
        (output_dir / "resume_report.json").write_text(
            json.dumps(
                {
                    "schema_version": "stage-resume-report-v1",
                    "resume_requested": args.resume_internal,
                    "restored_stages": restored_stages,
                    "started_stages": started_stages,
                    "executed_stages": executed_stages,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    write_resume_report()

    repo_root = Path(__file__).resolve().parent.parent
    subprocess_env = build_subprocess_env(args)

    for idx, (script_name, stage_name) in enumerate(STAGES):
        output_path = stage_output_path(output_dir, stage_name, suffix)
        cmd = build_stage_command(
            script_name,
            current_input,
            output_path,
            args,
            include_selection=(idx == 0),
        )
        stage_env = subprocess_env.copy()
        stage_env["LLM_STAGE_NAME"] = stage_name
        signature = build_stage_signature(
            root=repo_root,
            namespace="objective2",
            stage_name=stage_name,
            script_name=script_name,
            input_path=current_input,
            output_path=output_path,
            command=cmd,
            env=stage_env,
        )
        restored = (
            args.resume_internal
            and restore_stage_output(
                checkpoint_dir=args.checkpoint_dir,
                namespace="objective2",
                stage_name=stage_name,
                output_path=output_path,
                signature=signature,
            )
        )
        if restored:
            restored_stages.append(stage_name)
            if args.verbose:
                print(
                    f"[run_pipeline] Reanudando {stage_name} desde "
                    "una salida completa compatible.",
                    file=sys.stderr,
                )
        else:
            started_stages.append(stage_name)
            write_resume_report()
            run_command(
                cmd,
                repo_root,
                args.verbose,
                stage_env,
            )
            store_stage_output(
                checkpoint_dir=args.checkpoint_dir,
                namespace="objective2",
                stage_name=stage_name,
                output_path=output_path,
                signature=signature,
            )
            executed_stages.append(stage_name)
        write_resume_report()
        stage_outputs.append(output_path)
        current_input = output_path

    canonical_path = output_dir / f"00_sgotgraph_objetivo2{suffix}"
    shutil.copyfile(stage_outputs[-1], canonical_path)
    write_resume_report()

    if args.verbose:
        print(
            f"[run_pipeline] Corrida completa lista en {output_dir}",
            file=sys.stderr,
        )
        print(
            f"[run_pipeline] Artefacto canónico objetivo 2: {canonical_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
