from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def load_segmentation_stage():
    spec = importlib.util.spec_from_file_location(
        "literal_unit_segmentation_stage",
        PIPELINE_ROOT / "01_segmentacion_gcot.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_refinement_stage():
    spec = importlib.util.spec_from_file_location(
        "literal_unit_refinement_stage",
        PIPELINE_ROOT / "01_1_refinamiento_segmentacion_gcot.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiteralUnitSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage = load_segmentation_stage()

    def test_literal_units_are_lossless_and_contiguous(self) -> None:
        trace = "\nState S, so apply T.  Then verify.\nFinal answer.  "
        units = self.stage.build_literal_units(trace, max_chars=12)

        self.assertEqual("".join(unit["text"] for unit in units), trace)
        self.assertEqual(units[0]["start_char"], 0)
        self.assertEqual(units[-1]["end_char"], len(trace))
        self.assertEqual(
            [unit["unit_id"] for unit in units],
            [f"u{index}" for index in range(1, len(units) + 1)],
        )
        for left, right in zip(units, units[1:]):
            self.assertEqual(left["end_char"], right["start_char"])

    def test_ranges_reconstruct_exact_original_segments(self) -> None:
        trace = "State S, so apply T. Final answer."
        units = self.stage.build_literal_units(trace)
        self.assertGreaterEqual(len(units), 2)
        ranges = [
            self.stage.SegmentRangeDraft(
                start_unit_id=units[0]["unit_id"],
                end_unit_id=units[0]["unit_id"],
            ),
            self.stage.SegmentRangeDraft(
                start_unit_id=units[1]["unit_id"],
                end_unit_id=units[-1]["unit_id"],
            ),
        ]

        segments = self.stage.build_segment_objects(trace, units, ranges)

        self.assertEqual(
            "".join(segment["text"] for segment in segments),
            trace,
        )
        self.assertEqual(segments[0]["start_char"], 0)
        self.assertEqual(segments[-1]["end_char"], len(trace))
        self.assertEqual(
            segments[0]["end_char"],
            segments[1]["start_char"],
        )

    def test_ranges_reject_gap_overlap_reverse_and_unknown_ids(self) -> None:
        trace = "One. Two. Three."
        units = self.stage.build_literal_units(trace)
        self.assertEqual(len(units), 3)
        invalid_cases = {
            "gap": [
                self.stage.SegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u1",
                ),
                self.stage.SegmentRangeDraft(
                    start_unit_id="u3",
                    end_unit_id="u3",
                ),
            ],
            "overlap": [
                self.stage.SegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u2",
                ),
                self.stage.SegmentRangeDraft(
                    start_unit_id="u2",
                    end_unit_id="u3",
                ),
            ],
            "reverse": [
                self.stage.SegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u2",
                ),
                self.stage.SegmentRangeDraft(
                    start_unit_id="u3",
                    end_unit_id="u2",
                ),
            ],
            "unknown": [
                self.stage.SegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u999",
                )
            ],
        }

        for name, ranges in invalid_cases.items():
            with self.subTest(name=name):
                self.assertTrue(
                    self.stage.validate_segment_ranges(trace, units, ranges)
                )

    def test_segment_trace_never_needs_model_to_copy_long_text(self) -> None:
        trace = " ".join(
            f"Step {index}: compute a result, then inspect it."
            for index in range(1, 301)
        )
        units = self.stage.build_literal_units(trace)
        response = self.stage.ParsedSegmentationResponse(
            draft=self.stage.SegmentationDraft(
                trace_id="long-case",
                segments=[
                    self.stage.SegmentRangeDraft(
                        start_unit_id=units[0]["unit_id"],
                        end_unit_id=units[-1]["unit_id"],
                    )
                ],
            ),
            provider="mock",
            model="mock-model",
            reasoning_effort="low",
        )
        client = SimpleNamespace(provider="mock")

        with patch.object(self.stage, "call_model", return_value=response):
            result = self.stage.segment_trace(
                client,
                "mock-model",
                {"id": "long-case", "reasoning_trace": trace},
                max_attempts=1,
                verbose=False,
                reasoning_effort="low",
            )

        self.assertEqual(result["segments"][0]["text"], trace)
        self.assertEqual(
            result["validation"]["segmentation"]["transport"],
            self.stage.LITERAL_UNITIZATION_VERSION,
        )
        self.assertEqual(
            result["validation"]["segmentation"]["literal_unit_count"],
            len(units),
        )


class LiteralUnitRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage = load_refinement_stage()

    def test_refinement_units_are_lossless_and_contiguous(self) -> None:
        text = "  State S.\nApply T, then verify.  "
        units = self.stage.build_refinement_literal_units(text)

        self.assertEqual("".join(unit["text"] for unit in units), text)
        self.assertEqual(units[0]["start_char"], 0)
        self.assertEqual(units[-1]["end_char"], len(text))
        for left, right in zip(units, units[1:]):
            self.assertEqual(left["end_char"], right["start_char"])

    def test_refinement_ranges_reject_gap_overlap_and_unknown_ids(self) -> None:
        text = "One two three."
        units = self.stage.build_refinement_literal_units(text)
        invalid_cases = [
            [
                self.stage.SubsegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u1",
                ),
                self.stage.SubsegmentRangeDraft(
                    start_unit_id="u3",
                    end_unit_id="u3",
                ),
            ],
            [
                self.stage.SubsegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u2",
                ),
                self.stage.SubsegmentRangeDraft(
                    start_unit_id="u2",
                    end_unit_id="u3",
                ),
            ],
            [
                self.stage.SubsegmentRangeDraft(
                    start_unit_id="u1",
                    end_unit_id="u999",
                )
            ],
        ]

        for ranges in invalid_cases:
            with self.subTest(ranges=ranges):
                with self.assertRaises(ValueError):
                    self.stage.build_refined_texts(text, units, ranges)

    def test_long_keep_never_requires_model_to_copy_text(self) -> None:
        text = (
            'We are given: "'
            + " ".join(
                f"clinical observation {index}."
                for index in range(1, 301)
            )
            + '\\n\\n"'
        )
        units = self.stage.build_refinement_literal_units(text)
        record = {
            "trace_id": "long-refinement",
            "trace_text": text,
            "segments": [
                {
                    "segment_id": "s1",
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                }
            ],
            "nodes": [],
            "edges": [],
            "validation": {},
            "audit": {},
        }
        response = self.stage.RefinementDraft(
            segment_id="s1",
            decision="KEEP",
            subsegment_ranges=[
                self.stage.SubsegmentRangeDraft(
                    start_unit_id=units[0]["unit_id"],
                    end_unit_id=units[-1]["unit_id"],
                )
            ],
        )
        client = SimpleNamespace(provider="mock")

        with patch.object(self.stage, "call_model", return_value=response):
            result = self.stage.refine_record(
                client,
                "mock-model",
                record,
                reasoning_effort="low",
                min_length=0,
                max_subsegments=1,
                verbose=False,
                max_attempts=1,
            )

        self.assertEqual(result["segments"][0]["text"], text)
        self.assertEqual(
            result["validation"]["segmentation_refinement"]["transport"],
            self.stage.REFINEMENT_LITERAL_UNITIZATION_VERSION,
        )
        self.assertNotIn("subsegments", response.model_dump())


if __name__ == "__main__":
    unittest.main()
