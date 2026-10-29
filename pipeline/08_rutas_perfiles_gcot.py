#!/usr/bin/env python3
"""Etapa 08: deriva líneas, rutas, perfiles y configuraciones funcionales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from util.objective3_io import (
    load_records,
    records_by_trace_id,
    select_records,
    write_records,
)
from util.sgotgraph_objective3 import derive_routes_and_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compone rutas focales con vecindad funcional inmediata y deriva "
            "líneas de progresión, perfil y configuraciones sin inferencia semántica."
        )
    )
    parser.add_argument("input_path", type=Path, help="Salida de la etapa 07.")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--trace-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        records = select_records(
            load_records(args.input_path), args.trace_id, args.index, args.limit
        )
        records_by_trace_id(records, "Registro")
        results = [derive_routes_and_profile(record) for record in records]
        write_records(results, args.output, args.pretty)
        return int(
            any(
                result["validation"]["complete"]["status"] == "INVALID"
                for result in results
            )
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
