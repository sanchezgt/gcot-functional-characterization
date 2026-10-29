#!/usr/bin/env python3
"""Compara dos conjuntos de ejecuciones del objetivo 3 por trace_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.objective3_io import load_records
from util.objective3_stability import (
    DEFAULT_EPISODE_IOU_THRESHOLD,
    compare_objective3_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Alinea episodios y calcula estabilidad de relaciones, "
            "configuraciones y perfiles entre dos ejecuciones."
        )
    )
    parser.add_argument("left_path", type=Path)
    parser.add_argument("right_path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--episode-iou-threshold",
        type=float,
        default=DEFAULT_EPISODE_IOU_THRESHOLD,
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _index(records: list[dict], label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        trace_id = record.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"{label} contiene un trace_id inválido.")
        if trace_id in result:
            raise ValueError(f"{label} repite trace_id={trace_id!r}.")
        result[trace_id] = record
    return result


def main() -> int:
    args = parse_args()
    left = _index(load_records(args.left_path), "left_path")
    right = _index(load_records(args.right_path), "right_path")
    if set(left) != set(right):
        raise ValueError(
            "Los dos conjuntos deben contener exactamente los mismos trace_id."
        )
    comparisons = [
        compare_objective3_records(
            left[trace_id],
            right[trace_id],
            threshold=args.episode_iou_threshold,
        )
        for trace_id in sorted(left)
    ]
    result = {
        "schema_version": "objective3-stability-comparison-set-v1",
        "episode_iou_threshold": args.episode_iou_threshold,
        "trace_count": len(comparisons),
        "comparisons": comparisons,
    }
    text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


def cli() -> int:
    try:
        return main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
