#!/usr/bin/env python3
"""Valida artefactos SGoTGraph contra el contrato del objetivo 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.sgotgraph_schema import validate_record


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        return [payload] if isinstance(payload, dict) else list(payload)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    reports = []
    for record in read_records(args.input_path):
        report = validate_record(record)
        reports.append({"trace_id": record.get("trace_id"), **report})

    summary = {
        "records": len(reports),
        "valid": sum(item["status"] == "valid" for item in reports),
        "invalid": sum(item["status"] == "invalid" for item in reports),
        "with_warnings": sum(bool(item.get("warnings")) for item in reports),
        "results": reports,
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
