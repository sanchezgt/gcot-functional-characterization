#!/usr/bin/env python3
"""Validador de artefactos completos del objetivo 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.objective3_io import load_records, records_by_trace_id
from util.sgotgraph_objective3 import validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida episodios, relaciones, rutas y perfiles del objetivo 3."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--stage",
        choices=["episodes", "relations", "complete"],
        default="complete",
    )
    return parser.parse_args()


def build_report(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    batch_errors: list[str] = []
    try:
        records_by_trace_id(records, "Registro")
    except ValueError as exc:
        batch_errors.append(str(exc))

    results = []
    for record in records:
        validation = validate_record(record, stage=stage)
        results.append({"trace_id": record.get("trace_id"), **validation})
    counts = {
        "valid": sum(item["status"] == "VALID" for item in results),
        "invalid": sum(item["status"] == "INVALID" for item in results),
    }
    return {
        "records": len(results),
        **counts,
        "batch_errors": batch_errors,
        "results": results,
    }


def main() -> int:
    try:
        args = parse_args()
        report = build_report(load_records(args.input_path), args.stage)
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        return 1 if report["invalid"] or report["batch_errors"] else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
