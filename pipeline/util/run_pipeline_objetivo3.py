#!/usr/bin/env python3
"""Runner automático de las etapas 06–09 del objetivo 3."""

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
from util.llm_provider import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TRANSPORT_MAX_RETRIES,
    load_dotenv,
)
from util.objective3_hierarchical import (
    DEFAULT_BLOCK_CORE_NODES,
    DEFAULT_BLOCK_HALO_NODES,
    DEFAULT_BLOCK_SHIFT_NODES,
)
from util.stage_cache import (
    build_stage_signature,
    restore_stage_output,
    store_stage_output,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(
        description="Ejecuta episodios, relaciones, rutas/perfiles y validación del objetivo 3."
    )
    parser.add_argument(
        "input_path", type=Path, help="Artefacto canónico del objetivo 2."
    )
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
        help="Directorio compartido para salidas completas de etapas.",
    )
    parser.add_argument(
        "--resume-internal",
        action="store_true",
        help="Reutiliza salidas completas compatibles de etapas anteriores.",
    )
    parser.add_argument(
        "--episode-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="low",
    )
    parser.add_argument(
        "--relation-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="low",
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
            "Muestras LLM independientes por tipo de episodio en 06; se "
            "conserva la mayoría estricta. 1 desactiva el consenso. "
            f"Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--episode-method",
        choices=["global", "hierarchical"],
        default="global",
    )
    parser.add_argument(
        "--block-core-nodes",
        type=int,
        default=DEFAULT_BLOCK_CORE_NODES,
    )
    parser.add_argument(
        "--block-halo-nodes",
        type=int,
        default=DEFAULT_BLOCK_HALO_NODES,
    )
    parser.add_argument(
        "--block-shift-nodes",
        type=int,
        default=DEFAULT_BLOCK_SHIFT_NODES,
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_single(args: argparse.Namespace) -> bool:
    return (
        args.input_path.suffix.lower() == ".json"
        or args.trace_id is not None
        or args.index is not None
        or args.limit == 1
    )


def normalize_episode_method_args(args: argparse.Namespace) -> None:
    """Completa y valida los parámetros científicos de la etapa 06."""

    args.episode_method = getattr(args, "episode_method", "global")
    args.consensus_samples = getattr(
        args,
        "consensus_samples",
        DEFAULT_CONSENSUS_SAMPLES,
    )
    if args.consensus_samples <= 0:
        raise ValueError("--consensus-samples debe ser mayor que 0.")
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


def run(
    cmd: list[str],
    cwd: Path,
    verbose: bool,
    env: dict[str, str],
) -> None:
    if verbose:
        print(f"[objective3] Ejecutando: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> int:
    args = parse_args()
    normalize_episode_method_args(args)
    if args.trace_id is not None and args.index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds debe ser mayor que 0.")
    if args.transport_max_retries < 0:
        raise ValueError("--transport-max-retries no puede ser negativo.")
    if args.resume_internal and args.checkpoint_dir is None:
        raise ValueError("--resume-internal requiere --checkpoint-dir.")

    root = Path(__file__).resolve().parent.parent
    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".json" if is_single(args) else ".jsonl"

    stage06 = output_dir / f"06_episodios{suffix}"
    stage07 = output_dir / f"07_relaciones_episodios{suffix}"
    stage08 = output_dir / f"08_rutas_perfiles{suffix}"
    semantic_audit_dir = output_dir / "semantic_attempts"
    stage06_semantic_audit = (
        semantic_audit_dir / "06_episodios.jsonl"
    )
    stage07_semantic_audit = (
        semantic_audit_dir / "07_relaciones_episodios.jsonl"
    )
    canonical = output_dir / f"00_sgotgraph_objetivo3{suffix}"
    validation = output_dir / "09_validacion_objetivo3.json"
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
    restored_stages: list[str] = []
    started_stages: list[str] = []
    executed_stages: list[str] = []

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

    def run_cached_stage(
        *,
        stage_name: str,
        script_name: str,
        stage_input: Path,
        stage_output: Path,
        command: list[str],
    ) -> None:
        stage_env = env.copy()
        stage_env["LLM_STAGE_NAME"] = stage_name
        signature = build_stage_signature(
            root=root,
            namespace="objective3",
            stage_name=stage_name,
            script_name=script_name,
            input_path=stage_input,
            output_path=stage_output,
            command=command,
            env=stage_env,
        )
        restored = (
            args.resume_internal
            and restore_stage_output(
                checkpoint_dir=args.checkpoint_dir,
                namespace="objective3",
                stage_name=stage_name,
                output_path=stage_output,
                signature=signature,
            )
        )
        if restored:
            restored_stages.append(stage_name)
            write_resume_report()
            if args.verbose:
                print(
                    f"[objective3] Reanudando {stage_name} desde una "
                    "salida completa compatible.",
                    file=sys.stderr,
                )
            return
        started_stages.append(stage_name)
        write_resume_report()
        run(command, root, args.verbose, stage_env)
        store_stage_output(
            checkpoint_dir=args.checkpoint_dir,
            namespace="objective3",
            stage_name=stage_name,
            output_path=stage_output,
            signature=signature,
        )
        executed_stages.append(stage_name)
        write_resume_report()

    cmd06 = [
        sys.executable,
        "-B",
        "06_episodios_funcionales_gcot.py",
        str(input_path),
        "-o",
        str(stage06),
        "--provider",
        args.provider,
        "--reasoning-effort",
        args.episode_reasoning_effort,
        "--max-attempts",
        str(args.semantic_max_attempts),
        "--samples",
        str(args.consensus_samples),
        "--episode-method",
        args.episode_method,
        "--block-core-nodes",
        str(args.block_core_nodes),
        "--block-halo-nodes",
        str(args.block_halo_nodes),
        "--block-shift-nodes",
        str(args.block_shift_nodes),
        "--semantic-audit-path",
        str(stage06_semantic_audit),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--transport-max-retries",
        str(args.transport_max_retries),
    ]
    if args.model:
        cmd06.extend(["--model", args.model])
    if args.base_url:
        cmd06.extend(["--base-url", args.base_url])
    if args.trace_id:
        cmd06.extend(["--trace-id", args.trace_id])
    if args.index is not None:
        cmd06.extend(["--index", str(args.index)])
    if args.limit is not None:
        cmd06.extend(["--limit", str(args.limit)])
    if args.pretty:
        cmd06.append("--pretty")
    if args.verbose:
        cmd06.append("--verbose")
    run_cached_stage(
        stage_name="06_episodios",
        script_name="06_episodios_funcionales_gcot.py",
        stage_input=input_path,
        stage_output=stage06,
        command=cmd06,
    )

    cmd07 = [
        sys.executable,
        "-B",
        "07_relaciones_episodios_gcot.py",
        str(stage06),
        "-o",
        str(stage07),
        "--provider",
        args.provider,
        "--reasoning-effort",
        args.relation_reasoning_effort,
        "--max-attempts",
        str(args.semantic_max_attempts),
        "--semantic-audit-path",
        str(stage07_semantic_audit),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--transport-max-retries",
        str(args.transport_max_retries),
    ]
    if args.model:
        cmd07.extend(["--model", args.model])
    if args.base_url:
        cmd07.extend(["--base-url", args.base_url])
    if args.pretty:
        cmd07.append("--pretty")
    if args.verbose:
        cmd07.append("--verbose")
    run_cached_stage(
        stage_name="07_relaciones_episodios",
        script_name="07_relaciones_episodios_gcot.py",
        stage_input=stage06,
        stage_output=stage07,
        command=cmd07,
    )

    cmd08 = [
        sys.executable,
        "-B",
        "08_rutas_perfiles_gcot.py",
        str(stage07),
        "-o",
        str(stage08),
    ]
    if args.pretty:
        cmd08.append("--pretty")
    run_cached_stage(
        stage_name="08_rutas_perfiles",
        script_name="08_rutas_perfiles_gcot.py",
        stage_input=stage07,
        stage_output=stage08,
        command=cmd08,
    )
    shutil.copyfile(stage08, canonical)

    started_stages.append("09_validacion_objetivo3")
    write_resume_report()
    run(
        [
            sys.executable,
            "-B",
            "09_validacion_objetivo3_gcot.py",
            str(canonical),
            "-o",
            str(validation),
        ],
        root,
        args.verbose,
        env,
    )
    executed_stages.append("09_validacion_objetivo3")
    write_resume_report()

    if args.verbose:
        print(f"[objective3] Artefacto canónico: {canonical}", file=sys.stderr)
        print(f"[objective3] Validación: {validation}", file=sys.stderr)
    return 0


def cli() -> int:
    try:
        return main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
