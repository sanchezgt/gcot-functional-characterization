"""Métricas reproducibles de estabilidad para registros del objetivo 3.

Los episodios se emparejan únicamente dentro del mismo tipo. Primero se
maximiza el número de pares que alcanzan el umbral de IoU y, entre todos los
emparejamientos de cardinalidad máxima, se maximiza la suma de IoU. Esta
desambiguación evita que dos soluciones de igual peso produzcan denominadores
distintos en la concordancia de detección.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any

from util.sgotgraph_objective3 import (
    EPISODE_TYPES,
    FUNCTIONAL_RELATION_TYPES,
)


STABILITY_SCHEMA_VERSION = "objective3-stability-v1"
DEFAULT_EPISODE_IOU_THRESHOLD = 0.5
IOU_SCORE_SCALE = 10**12


def _valid_interval(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start_char")
    end = value.get("end_char")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or end <= start
    ):
        return None
    return start, end


def merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_set_overlap(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> dict[str, Any]:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_chars = sum(end - start for start, end in left_merged)
    right_chars = sum(end - start for start, end in right_merged)
    intersection = 0
    left_index = 0
    right_index = 0
    while (
        left_index < len(left_merged)
        and right_index < len(right_merged)
    ):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        intersection += max(
            0,
            min(left_end, right_end) - max(left_start, right_start),
        )
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    union = left_chars + right_chars - intersection
    return {
        "intersection_chars": intersection,
        "union_chars": union,
        "iou": intersection / union if union else 0.0,
    }


def _episode_member_intervals(
    episode: dict[str, Any],
) -> list[tuple[int, int]]:
    return [
        interval
        for interval in (
            _valid_interval(span)
            for span in episode.get("members", [])
        )
        if interval is not None
    ]


def episode_member_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return interval_set_overlap(
        _episode_member_intervals(left),
        _episode_member_intervals(right),
    )


def _maximum_cardinality(
    adjacency: list[list[int]],
    right_count: int,
) -> int:
    matched_left_by_right = [-1] * right_count

    def augment(left_index: int, seen: set[int]) -> bool:
        for right_index in adjacency[left_index]:
            if right_index in seen:
                continue
            seen.add(right_index)
            previous_left = matched_left_by_right[right_index]
            if previous_left == -1 or augment(previous_left, seen):
                matched_left_by_right[right_index] = left_index
                return True
        return False

    cardinality = 0
    for left_index in range(len(adjacency)):
        if augment(left_index, set()):
            cardinality += 1
    return cardinality


@dataclass
class _FlowEdge:
    target: int
    reverse_index: int
    capacity: int
    cost: int


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: int,
) -> _FlowEdge:
    forward = _FlowEdge(
        target=target,
        reverse_index=len(graph[target]),
        capacity=capacity,
        cost=cost,
    )
    reverse = _FlowEdge(
        target=source,
        reverse_index=len(graph[source]),
        capacity=0,
        cost=-cost,
    )
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _maximum_cardinality_weight_matching(
    weights: list[list[float]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    left_count = len(weights)
    right_count = len(weights[0]) if weights else 0
    if not left_count or not right_count:
        return []
    adjacency = [
        [
            right_index
            for right_index, weight in enumerate(row)
            if weight >= threshold
        ]
        for row in weights
    ]
    cardinality = _maximum_cardinality(adjacency, right_count)
    if cardinality == 0:
        return []

    source = 0
    left_offset = 1
    right_offset = left_offset + left_count
    sink = right_offset + right_count
    graph: list[list[_FlowEdge]] = [
        [] for _ in range(sink + 1)
    ]
    for left_index in range(left_count):
        _add_flow_edge(
            graph,
            source,
            left_offset + left_index,
            1,
            0,
        )
    for right_index in range(right_count):
        _add_flow_edge(
            graph,
            right_offset + right_index,
            sink,
            1,
            0,
        )
    pair_edges: dict[tuple[int, int], _FlowEdge] = {}
    for left_index, right_indices in enumerate(adjacency):
        for right_index in sorted(right_indices):
            score = round(
                weights[left_index][right_index] * IOU_SCORE_SCALE
            )
            pair_edges[(left_index, right_index)] = _add_flow_edge(
                graph,
                left_offset + left_index,
                right_offset + right_index,
                1,
                -score,
            )

    node_count = len(graph)
    for _ in range(cardinality):
        infinity = 10**30
        distance = [infinity] * node_count
        parent: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0
        for _pass in range(node_count - 1):
            changed = False
            for node in range(node_count):
                if distance[node] == infinity:
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distance[node] + edge.cost
                    if candidate_distance < distance[edge.target]:
                        distance[edge.target] = candidate_distance
                        parent[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            raise ValueError(
                "No fue posible materializar la cardinalidad máxima."
            )
        node = sink
        while node != source:
            parent_node, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[parent_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = parent_node

    return [
        (left_index, right_index, weights[left_index][right_index])
        for (left_index, right_index), edge in sorted(pair_edges.items())
        if edge.capacity == 0
    ]


def _anchor_set(episode: dict[str, Any]) -> set[tuple[str, int, int]]:
    raw_values = [
        *(
            episode.get("anchors", [])
            if isinstance(episode.get("anchors"), list)
            else []
        ),
        episode.get("entry_anchor"),
        episode.get("exit_anchor"),
    ]
    anchors: set[tuple[str, int, int]] = set()
    for value in raw_values:
        interval = _valid_interval(value)
        if interval is None or not isinstance(value, dict):
            continue
        role = value.get("anchor_role")
        if isinstance(role, str):
            anchors.add((role, interval[0], interval[1]))
    return anchors


def _set_jaccard(
    left: set[Any],
    right: set[Any],
) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def align_episodes(
    left_episodes: list[dict[str, Any]],
    right_episodes: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_EPISODE_IOU_THRESHOLD,
) -> dict[str, Any]:
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")
    matches: list[dict[str, Any]] = []
    per_type: dict[str, dict[str, Any]] = {}
    observed_types = sorted(
        {
            str(episode.get("episode_type"))
            for episode in [*left_episodes, *right_episodes]
            if episode.get("episode_type") in EPISODE_TYPES
        }
    )
    for episode_type in observed_types:
        left_values = [
            episode
            for episode in left_episodes
            if episode.get("episode_type") == episode_type
        ]
        right_values = [
            episode
            for episode in right_episodes
            if episode.get("episode_type") == episode_type
        ]
        overlap_matrix = [
            [
                episode_member_overlap(left, right)
                for right in right_values
            ]
            for left in left_values
        ]
        weights = [
            [value["iou"] for value in row]
            for row in overlap_matrix
        ]
        selected = _maximum_cardinality_weight_matching(
            weights,
            threshold=threshold,
        )
        type_matches: list[dict[str, Any]] = []
        for left_index, right_index, weight in selected:
            left = left_values[left_index]
            right = right_values[right_index]
            overlap = overlap_matrix[left_index][right_index]
            left_anchors = _anchor_set(left)
            right_anchors = _anchor_set(right)
            type_matches.append(
                {
                    "episode_type": episode_type,
                    "left_episode_id": left.get("episode_id"),
                    "right_episode_id": right.get("episode_id"),
                    "member_iou": weight,
                    "member_intersection_chars": overlap[
                        "intersection_chars"
                    ],
                    "member_union_chars": overlap["union_chars"],
                    "outcome_agreement": (
                        left.get("outcome") == right.get("outcome")
                    ),
                    "anchor_exact_agreement": (
                        left_anchors == right_anchors
                    ),
                    "anchor_informative": bool(
                        left_anchors or right_anchors
                    ),
                    "anchor_jaccard": _set_jaccard(
                        left_anchors,
                        right_anchors,
                    ),
                }
            )
        type_matches.sort(
            key=lambda item: (
                str(item["left_episode_id"]),
                str(item["right_episode_id"]),
            )
        )
        matches.extend(type_matches)
        left_count = len(left_values)
        right_count = len(right_values)
        match_count = len(type_matches)
        per_type[episode_type] = {
            "left_episode_count": left_count,
            "right_episode_count": right_count,
            "matched_episode_count": match_count,
            "detection_stability": (
                2 * match_count / (left_count + right_count)
                if left_count + right_count
                else None
            ),
            "mean_member_iou": (
                sum(item["member_iou"] for item in type_matches)
                / match_count
                if match_count
                else None
            ),
        }

    left_count = len(left_episodes)
    right_count = len(right_episodes)
    match_count = len(matches)
    informative_anchor_matches = [
        item for item in matches if item["anchor_informative"]
    ]
    anchor_jaccards = [
        item["anchor_jaccard"]
        for item in informative_anchor_matches
        if item["anchor_jaccard"] is not None
    ]
    return {
        "threshold": threshold,
        "matching_objective": (
            "maximum_cardinality_then_maximum_total_member_iou"
        ),
        "left_episode_count": left_count,
        "right_episode_count": right_count,
        "matched_episode_count": match_count,
        "detection_stability": (
            2 * match_count / (left_count + right_count)
            if left_count + right_count
            else None
        ),
        "empty_both_noninformative": (
            left_count == 0 and right_count == 0
        ),
        "mean_member_iou": (
            sum(item["member_iou"] for item in matches) / match_count
            if match_count
            else None
        ),
        "outcome_agreement_rate": (
            sum(item["outcome_agreement"] for item in matches)
            / match_count
            if match_count
            else None
        ),
        "anchor_exact_agreement_rate": (
            sum(item["anchor_exact_agreement"] for item in matches)
            / match_count
            if match_count
            else None
        ),
        "anchor_informative_match_count": len(
            informative_anchor_matches
        ),
        "anchor_exact_agreement_rate_informative": (
            sum(
                item["anchor_exact_agreement"]
                for item in informative_anchor_matches
            )
            / len(informative_anchor_matches)
            if informative_anchor_matches
            else None
        ),
        "mean_anchor_jaccard_informative": (
            sum(anchor_jaccards) / len(anchor_jaccards)
            if anchor_jaccards
            else None
        ),
        "matches": matches,
        "per_type": per_type,
    }


def _functional_relation_set(
    record: dict[str, Any],
    episode_projection: dict[str, str],
    *,
    unmatched_namespace: str | None = None,
) -> tuple[
    set[tuple[str, str, str]],
    set[tuple[str, str, str]],
    int,
]:
    comparable: set[tuple[str, str, str]] = set()
    alignment_aware: set[tuple[str, str, str]] = set()
    total = 0
    for relation in record.get("episode_relations", []):
        if (
            not isinstance(relation, dict)
            or relation.get("relation_type")
            not in FUNCTIONAL_RELATION_TYPES
        ):
            continue
        total += 1
        source = episode_projection.get(
            str(relation.get("source_episode_id"))
        )
        target = episode_projection.get(
            str(relation.get("target_episode_id"))
        )
        relation_type = str(relation["relation_type"])
        if source is not None and target is not None:
            comparable.add((source, relation_type, target))
            alignment_aware.add((source, relation_type, target))
            continue
        if unmatched_namespace is not None:
            raw_source = str(relation.get("source_episode_id"))
            raw_target = str(relation.get("target_episode_id"))
            alignment_aware.add(
                (
                    source
                    or f"{unmatched_namespace}:{raw_source}",
                    relation_type,
                    target
                    or f"{unmatched_namespace}:{raw_target}",
                )
            )
    return comparable, alignment_aware, total


def _relation_stability(
    left: dict[str, Any],
    right: dict[str, Any],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    left_projection: dict[str, str] = {}
    right_projection: dict[str, str] = {}
    for index, match in enumerate(matches, start=1):
        projected_id = f"M{index:04d}"
        left_id = match.get("left_episode_id")
        right_id = match.get("right_episode_id")
        if isinstance(left_id, str):
            left_projection[left_id] = projected_id
        if isinstance(right_id, str):
            right_projection[right_id] = projected_id
    (
        left_comparable,
        left_alignment_aware,
        left_total,
    ) = _functional_relation_set(
        left,
        left_projection,
        unmatched_namespace="LEFT_UNMATCHED",
    )
    (
        right_comparable,
        right_alignment_aware,
        right_total,
    ) = _functional_relation_set(
        right,
        right_projection,
        unmatched_namespace="RIGHT_UNMATCHED",
    )
    comparable_union = left_comparable | right_comparable
    alignment_aware_union = (
        left_alignment_aware | right_alignment_aware
    )
    return {
        "left_functional_relation_count": left_total,
        "right_functional_relation_count": right_total,
        "left_comparable_relation_count": len(left_comparable),
        "right_comparable_relation_count": len(right_comparable),
        "left_excluded_unmatched_endpoint_count": (
            left_total - len(left_comparable)
        ),
        "right_excluded_unmatched_endpoint_count": (
            right_total - len(right_comparable)
        ),
        "jaccard": (
            len(left_alignment_aware & right_alignment_aware)
            / len(alignment_aware_union)
            if alignment_aware_union
            else None
        ),
        "comparable_endpoint_jaccard": (
            len(left_comparable & right_comparable)
            / len(comparable_union)
            if comparable_union
            else None
        ),
        "empty_relation_sets_noninformative": (
            not alignment_aware_union
        ),
        "empty_comparable_sets_noninformative": (
            not comparable_union
        ),
    }


def multiset_jaccard(
    left: Counter[str],
    right: Counter[str],
) -> float | None:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    if denominator == 0:
        return None
    numerator = sum(min(left[key], right[key]) for key in keys)
    return numerator / denominator


def _configuration_stability(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_values = Counter(
        str(item.get("configuration_signature"))
        for item in left.get("configurations", [])
        if isinstance(item, dict)
        and isinstance(item.get("configuration_signature"), str)
    )
    right_values = Counter(
        str(item.get("configuration_signature"))
        for item in right.get("configurations", [])
        if isinstance(item, dict)
        and isinstance(item.get("configuration_signature"), str)
    )
    return {
        "left_configuration_count": sum(left_values.values()),
        "right_configuration_count": sum(right_values.values()),
        "multiset_jaccard": multiset_jaccard(
            left_values,
            right_values,
        ),
        "empty_both_noninformative": not left_values and not right_values,
    }


def _flatten_numeric(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, bool):
        return result
    if isinstance(value, (int, float)) and isfinite(float(value)):
        result[prefix] = float(value)
        return result
    if not isinstance(value, dict):
        return result
    for key in sorted(value):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        result.update(
            _flatten_numeric(value[key], prefix=child_prefix)
        )
    return result


def _profile_differences(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_values = _flatten_numeric(left.get("profile"))
    right_values = _flatten_numeric(right.get("profile"))
    common = sorted(set(left_values) & set(right_values))
    return {
        "common_numeric_attribute_count": len(common),
        "absolute_differences": {
            key: abs(left_values[key] - right_values[key])
            for key in common
        },
        "left_only_attributes": sorted(
            set(left_values) - set(right_values)
        ),
        "right_only_attributes": sorted(
            set(right_values) - set(left_values)
        ),
    }


def compare_objective3_records(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    threshold: float = DEFAULT_EPISODE_IOU_THRESHOLD,
) -> dict[str, Any]:
    left_trace_id = left.get("trace_id")
    right_trace_id = right.get("trace_id")
    if left_trace_id != right_trace_id:
        raise ValueError(
            "Los registros deben corresponder al mismo trace_id."
        )
    left_text = left.get("sgot_graph", {}).get("trace_text")
    right_text = right.get("sgot_graph", {}).get("trace_text")
    if not isinstance(left_text, str) or left_text != right_text:
        raise ValueError(
            "Los registros deben usar exactamente la misma traza base."
        )
    episode_alignment = align_episodes(
        [
            item
            for item in left.get("episodes", [])
            if isinstance(item, dict)
        ],
        [
            item
            for item in right.get("episodes", [])
            if isinstance(item, dict)
        ],
        threshold=threshold,
    )
    return {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "trace_id": left_trace_id,
        "episode_alignment": episode_alignment,
        "functional_relations": _relation_stability(
            left,
            right,
            episode_alignment["matches"],
        ),
        "configurations": _configuration_stability(left, right),
        "profile": _profile_differences(left, right),
    }


def icc_2_1(matrix: list[list[float]]) -> float | None:
    """Calcula ICC(2,1) de acuerdo absoluto para una matriz completa."""

    if len(matrix) < 2:
        return None
    rater_count = len(matrix[0]) if matrix else 0
    if rater_count < 2 or any(
        len(row) != rater_count for row in matrix
    ):
        return None
    values = [
        [float(value) for value in row]
        for row in matrix
    ]
    if any(
        not isfinite(value)
        for row in values
        for value in row
    ):
        return None
    target_count = len(values)
    grand_mean = sum(map(sum, values)) / (
        target_count * rater_count
    )
    target_means = [
        sum(row) / rater_count for row in values
    ]
    rater_means = [
        sum(values[target][rater] for target in range(target_count))
        / target_count
        for rater in range(rater_count)
    ]
    mean_square_targets = (
        rater_count
        * sum((value - grand_mean) ** 2 for value in target_means)
        / (target_count - 1)
    )
    mean_square_raters = (
        target_count
        * sum((value - grand_mean) ** 2 for value in rater_means)
        / (rater_count - 1)
    )
    residual_sum = sum(
        (
            values[target][rater]
            - target_means[target]
            - rater_means[rater]
            + grand_mean
        )
        ** 2
        for target in range(target_count)
        for rater in range(rater_count)
    )
    mean_square_error = residual_sum / (
        (target_count - 1) * (rater_count - 1)
    )
    denominator = (
        mean_square_targets
        + (rater_count - 1) * mean_square_error
        + (
            rater_count
            * (mean_square_raters - mean_square_error)
            / target_count
        )
    )
    if denominator == 0:
        return None
    return (mean_square_targets - mean_square_error) / denominator


__all__ = [
    "DEFAULT_EPISODE_IOU_THRESHOLD",
    "STABILITY_SCHEMA_VERSION",
    "align_episodes",
    "compare_objective3_records",
    "episode_member_overlap",
    "icc_2_1",
    "interval_set_overlap",
    "merge_intervals",
    "multiset_jaccard",
]
