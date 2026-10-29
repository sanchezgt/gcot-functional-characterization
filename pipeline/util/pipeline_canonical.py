"""Construcción y actualización del registro canónico SGoTGraph."""

from __future__ import annotations

from typing import Any

from util.sgotgraph_schema import PIPELINE_VERSION, SCHEMA_VERSION


CANONICAL_KEYS = {
    "schema_version",
    "pipeline_version",
    "trace_id",
    "source_record",
    "trace_text",
    "segments",
    "nodes",
    "edges",
    "validation",
    "audit",
}


def _source_record(record: dict[str, Any]) -> dict[str, Any]:
    existing = record.get("source_record")
    if isinstance(existing, dict):
        return dict(existing)
    excluded = CANONICAL_KEYS | {"reasoning_trace", "routes"}
    return {key: value for key, value in record.items() if key not in excluded}


def ensure_pipeline_record(record: dict[str, Any]) -> dict[str, Any]:
    trace_id = record.get("trace_id") or record.get("id")
    trace_text = record.get("trace_text")
    if trace_text is None:
        trace_text = record.get("reasoning_trace")

    segments = list(record.get("segments", []))
    nodes = list(record.get("nodes", []))
    if not segments and nodes:
        segments = [
            {
                "segment_id": node["segment_id"],
                "text": node["text"],
                "start_char": node["start_char"],
                "end_char": node["end_char"],
                **({"provenance": node["provenance"]} if "provenance" in node else {}),
            }
            for node in nodes
            if all(key in node for key in ("segment_id", "text", "start_char", "end_char"))
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "trace_id": trace_id,
        "source_record": _source_record(record),
        "trace_text": trace_text,
        "segments": segments,
        "nodes": nodes,
        "edges": list(record.get("edges", [])),
        "validation": dict(record.get("validation", {})),
        "audit": dict(record.get("audit", {})),
    }


def set_validation_stage(record: dict[str, Any], stage: str, value: dict[str, Any]) -> dict[str, Any]:
    pipeline = ensure_pipeline_record(record)
    validation = dict(pipeline.get("validation", {}))
    validation[stage] = value
    pipeline["validation"] = validation
    return pipeline
