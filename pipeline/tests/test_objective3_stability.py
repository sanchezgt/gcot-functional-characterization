from __future__ import annotations

from collections import Counter
import sys
import unittest
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from util.objective3_stability import (
    align_episodes,
    compare_objective3_records,
    icc_2_1,
    interval_set_overlap,
    multiset_jaccard,
)


def episode(
    episode_id: str,
    start: int,
    end: int,
    *,
    outcome: str = "ADVANCES",
) -> dict:
    return {
        "episode_id": episode_id,
        "episode_type": "PROGRESSION",
        "members": [
            {
                "start_char": start,
                "end_char": end,
                "text": "x" * (end - start),
            }
        ],
        "anchors": [],
        "entry_anchor": None,
        "exit_anchor": None,
        "outcome": outcome,
        "ambiguity_codes": [],
        "evidence": {
            "text_spans": [],
            "witness_node_ids": [],
            "witness_edge_ids": [],
        },
        "provenance": {
            "origin": "llm_inferred",
            "stage": "test",
        },
    }


def record(
    episodes: list[dict],
    *,
    relations: list[dict] | None = None,
    configurations: list[str] | None = None,
) -> dict:
    return {
        "trace_id": "trace",
        "sgot_graph": {"trace_text": "x" * 30},
        "episodes": episodes,
        "episode_relations": relations or [],
        "configurations": [
            {"configuration_signature": value}
            for value in (configurations or [])
        ],
        "profile": {
            "episode_count": len(episodes),
            "coverage": {"overall": {"ratio": 0.5}},
        },
    }


class Objective3StabilityTests(unittest.TestCase):
    def test_interval_iou_uses_union_of_discontinuous_members(self) -> None:
        overlap = interval_set_overlap(
            [(0, 5), (10, 15)],
            [(3, 12)],
        )
        self.assertEqual(overlap["intersection_chars"], 4)
        self.assertEqual(overlap["union_chars"], 15)
        self.assertAlmostEqual(overlap["iou"], 4 / 15)

    def test_matching_maximizes_cardinality_before_total_iou(self) -> None:
        left = [
            episode("left_a", 0, 10),
            episode("left_b", 3, 10),
        ]
        right = [
            episode("right_x", 0, 10),
            episode("right_y", 0, 5),
        ]

        result = align_episodes(left, right, threshold=0.5)

        self.assertEqual(result["matched_episode_count"], 2)
        self.assertEqual(result["detection_stability"], 1.0)
        pairs = {
            (
                item["left_episode_id"],
                item["right_episode_id"],
            )
            for item in result["matches"]
        }
        self.assertEqual(
            pairs,
            {
                ("left_a", "right_y"),
                ("left_b", "right_x"),
            },
        )

    def test_record_comparison_projects_relations_and_multisets(self) -> None:
        left_episodes = [
            episode("left_a", 0, 10),
            episode("left_b", 3, 10),
        ]
        right_episodes = [
            episode("right_x", 0, 10),
            episode("right_y", 0, 5),
        ]
        left_relation = {
            "relation_type": "CONTINUES",
            "source_episode_id": "left_a",
            "target_episode_id": "left_b",
        }
        right_relation = {
            "relation_type": "CONTINUES",
            "source_episode_id": "right_y",
            "target_episode_id": "right_x",
        }

        result = compare_objective3_records(
            record(
                left_episodes,
                relations=[left_relation],
                configurations=["a", "a", "b"],
            ),
            record(
                right_episodes,
                relations=[right_relation],
                configurations=["a", "c"],
            ),
            threshold=0.5,
        )

        self.assertEqual(
            result["functional_relations"]["jaccard"],
            1.0,
        )
        self.assertAlmostEqual(
            result["configurations"]["multiset_jaccard"],
            0.25,
        )
        self.assertIn(
            "episode_count",
            result["profile"]["absolute_differences"],
        )

    def test_relation_with_unmatched_endpoint_is_not_silently_discarded(
        self,
    ) -> None:
        left_episodes = [
            episode("left_matched", 0, 10),
            episode("left_unmatched", 20, 25),
        ]
        right_episodes = [episode("right_matched", 0, 10)]
        left_relation = {
            "relation_type": "CONTINUES",
            "source_episode_id": "left_unmatched",
            "target_episode_id": "left_matched",
        }

        result = compare_objective3_records(
            record(left_episodes, relations=[left_relation]),
            record(right_episodes),
            threshold=0.5,
        )

        self.assertEqual(result["functional_relations"]["jaccard"], 0.0)
        self.assertIsNone(
            result["functional_relations"][
                "comparable_endpoint_jaccard"
            ]
        )
        self.assertEqual(
            result["functional_relations"][
                "left_excluded_unmatched_endpoint_count"
            ],
            1,
        )

    def test_empty_episode_sets_are_noninformative(self) -> None:
        result = align_episodes([], [], threshold=0.5)
        self.assertIsNone(result["detection_stability"])
        self.assertTrue(result["empty_both_noninformative"])

    def test_multiset_jaccard_preserves_multiplicity(self) -> None:
        self.assertAlmostEqual(
            multiset_jaccard(
                Counter({"a": 2, "b": 1}),
                Counter({"a": 1, "c": 1}),
            ),
            0.25,
        )
        self.assertIsNone(multiset_jaccard(Counter(), Counter()))

    def test_icc_2_1_handles_perfect_agreement_and_degenerate_input(
        self,
    ) -> None:
        self.assertAlmostEqual(
            icc_2_1([[1, 1, 1], [2, 2, 2]]),
            1.0,
        )
        self.assertIsNone(icc_2_1([[1, 1, 1], [1, 1, 1]]))
        self.assertIsNone(icc_2_1([[1], [2]]))


if __name__ == "__main__":
    unittest.main()
