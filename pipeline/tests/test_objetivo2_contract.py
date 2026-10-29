from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.sgotgraph_schema import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    make_provenance,
    validate_record,
)


def valid_record() -> dict:
    trace = "Compute 1+1. Answer: 2."
    segment_provenance = make_provenance("llm_proposed", "segmentation")
    node_provenance = make_provenance("llm_proposed", "typing")
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "trace_id": "case_1",
        "source_record": {"problem_id": "p1"},
        "trace_text": trace,
        "segments": [
            {"segment_id": "s1", "text": "Compute 1+1. ", "start_char": 0, "end_char": 13, "provenance": segment_provenance},
            {"segment_id": "s2", "text": "Answer: 2.", "start_char": 13, "end_char": 23, "provenance": segment_provenance},
        ],
        "nodes": [
            {"segment_id": "s1", "text": "Compute 1+1. ", "start_char": 0, "end_char": 13, "position": 1, "node_type": "METHOD", "provenance": node_provenance},
            {"segment_id": "s2", "text": "Answer: 2.", "start_char": 13, "end_char": 23, "position": 2, "node_type": "OUT", "provenance": node_provenance},
        ],
        "edges": [
            {"edge_id": "e_next_1", "source": "s1", "target": "s2", "edge_type": "NEXT", "provenance": make_provenance("deterministic", "backbone")},
            {"edge_id": "e_sem_1", "source": "s1", "target": "s2", "edge_type": "LEADS_TO", "provenance": node_provenance},
        ],
        "validation": {},
        "audit": {},
    }


class ContractTests(unittest.TestCase):
    def test_valid_record(self) -> None:
        self.assertEqual(validate_record(valid_record())["status"], "valid")

    def test_lossless_reconstruction_is_hard_invariant(self) -> None:
        record = valid_record()
        record["segments"][0]["text"] = "Changed text "
        report = validate_record(record)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("reconstruye" in error or "no coincide" in error for error in report["hard_errors"]))

    def test_missing_anchor_is_a_nonblocking_warning_without_injection(self) -> None:
        record = valid_record()
        record["trace_text"] = "Check this."
        provenance = make_provenance("llm_proposed", "typing")
        record["segments"] = [{"segment_id": "s1", "text": "Check this.", "start_char": 0, "end_char": 11, "provenance": provenance}]
        record["nodes"] = [{"segment_id": "s1", "text": "Check this.", "start_char": 0, "end_char": 11, "position": 1, "node_type": "CHECK", "provenance": provenance}]
        record["edges"] = []
        report = validate_record(record)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(record["edges"], [])
        self.assertTrue(
            any(
                item["code"] == "MISSING_SEMANTIC_ANCHOR"
                for item in report["warnings"]
            )
        )

    def test_backward_leads_to_is_invalid(self) -> None:
        record = valid_record()
        record["edges"][1]["source"] = "s2"
        record["edges"][1]["target"] = "s1"
        self.assertEqual(validate_record(record)["status"], "invalid")

    def test_segments_without_nodes_are_invalid_final_graph(self) -> None:
        record = valid_record()
        record["nodes"] = []
        record["edges"] = []
        report = validate_record(record)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("nodo por segmento" in error for error in report["hard_errors"]))


if __name__ == "__main__":
    unittest.main()
