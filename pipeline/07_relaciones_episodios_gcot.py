#!/usr/bin/env python3
"""Etapa 07: infiere relaciones funcionales y deriva el orden temporal."""

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
from util.objective3_llm import (
    DEFAULT_OBJECTIVE3_MODEL,
    infer_episode_relations,
)
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
        description=(
            "Infiere relaciones funcionales entre todos los episodios mediante "
            "un LLM y deriva IMMEDIATELY_PRECEDES_ONSET determinísticamente."
        )
    )
    parser.add_argument("input_path", type=Path, help="Salida de la etapa 06.")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--trace-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    add_llm_provider_args(parser, DEFAULT_OBJECTIVE3_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Esfuerzo de razonamiento del estimador de relaciones. Default: low.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Intentos máximos ante una salida relacional inválida. Default: 3.",
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
        records = select_records(
            load_records(args.input_path), args.trace_id, args.index, args.limit
        )
        records_by_trace_id(records, "Registro")
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
        results = []
        for record in records:
            result = infer_episode_relations(
                client,
                model=args.model,
                record=record,
                reasoning_effort=args.reasoning_effort,
                verbose=args.verbose,
                max_attempts=args.max_attempts,
                semantic_audit_path=semantic_audit_path,
            )
            results.append(result)
        write_records(results, args.output, args.pretty)
        return int(
            any(
                result["validation"]["relations"]["status"] == "INVALID"
                for result in results
            )
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
