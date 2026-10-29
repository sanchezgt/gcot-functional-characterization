#!/usr/bin/env python3
"""Etapa 06: identifica y canoniza episodios funcionales sobre SGoTGraph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from util.llm_provider import (
    add_llm_provider_args,
    build_llm_client,
    finalize_llm_args,
    load_dotenv,
)
from util.objective3_hierarchical import (
    DEFAULT_BLOCK_CORE_NODES,
    DEFAULT_BLOCK_HALO_NODES,
    DEFAULT_BLOCK_SHIFT_NODES,
    infer_episodes_hierarchical,
)
from util.consensus import DEFAULT_CONSENSUS_SAMPLES
from util.objective3_consensus import DEFAULT_EPISODE_IOU_THRESHOLD
from util.objective3_llm import DEFAULT_OBJECTIVE3_MODEL, infer_episodes
from util.objective3_semantic_audit import (
    initialize_semantic_attempt_journal,
)
from util.objective3_io import (
    load_records,
    records_by_trace_id,
    select_records,
    write_records,
)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Identifica automáticamente episodios funcionales mediante un LLM."
    )
    parser.add_argument(
        "input_path", type=Path, help="SGoTGraph objetivo 2 en JSON o JSONL."
    )
    parser.add_argument("-o", "--output", type=Path, help="Salida JSON o JSONL.")
    parser.add_argument("--trace-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    add_llm_provider_args(parser, DEFAULT_OBJECTIVE3_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Esfuerzo de razonamiento del estimador de episodios. Default: low.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Intentos máximos por tipo ante una salida inválida. Default: 3.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por tipo de episodio; se conservan "
            "los episodios presentes en una mayoría estricta. 1 reproduce el "
            f"estimador de una sola muestra. Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--consensus-iou-threshold",
        type=float,
        default=DEFAULT_EPISODE_IOU_THRESHOLD,
        help=(
            "IoU mínima de miembros para considerar que dos muestras proponen "
            f"el mismo episodio. Default: {DEFAULT_EPISODE_IOU_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--episode-method",
        choices=["global", "hierarchical"],
        default="global",
        help=(
            "Organización de contexto para identificar episodios. "
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
        "--semantic-audit-path",
        type=Path,
        help=(
            "Diario JSONL durable de cada intento semántico. Si se omite "
            "y existe --output, se crea junto a la salida."
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return finalize_llm_args(parser.parse_args(), DEFAULT_OBJECTIVE3_MODEL)


def main() -> int:
    try:
        args = parse_args()
        base_records = select_records(
            load_records(args.input_path), args.trace_id, args.index, args.limit
        )
        records_by_trace_id(base_records, "Grafo base")
        client = build_llm_client(
            args.provider,
            args.api_key,
            args.base_url,
            args.request_timeout_seconds,
            args.transport_max_retries,
        )
        semantic_audit_path = args.semantic_audit_path
        if semantic_audit_path is None and args.output is not None:
            semantic_audit_path = args.output.with_name(
                f"{args.output.stem}.semantic_attempts.jsonl"
            )
        initialize_semantic_attempt_journal(semantic_audit_path)
        if args.episode_method != "global" and args.samples != 1:
            raise ValueError(
                "El consenso multimuestra (--samples > 1) solo está "
                "disponible para --episode-method global."
            )
        results = []
        for base_graph in base_records:
            if args.episode_method == "global":
                result = infer_episodes(
                    client,
                    model=args.model,
                    sgot_graph=base_graph,
                    reasoning_effort=args.reasoning_effort,
                    verbose=args.verbose,
                    max_attempts=args.max_attempts,
                    semantic_audit_path=semantic_audit_path,
                    samples=args.samples,
                    consensus_iou_threshold=args.consensus_iou_threshold,
                )
            else:
                result = infer_episodes_hierarchical(
                    client,
                    model=args.model,
                    sgot_graph=base_graph,
                    reasoning_effort=args.reasoning_effort,
                    verbose=args.verbose,
                    max_attempts=args.max_attempts,
                    core_nodes=args.block_core_nodes,
                    halo_nodes=args.block_halo_nodes,
                    shift_nodes=args.block_shift_nodes,
                    semantic_audit_path=semantic_audit_path,
                )
            results.append(result)
        write_records(results, args.output, args.pretty)
        return int(
            any(
                result["validation"]["episodes"]["status"] == "INVALID"
                for result in results
            )
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
