"""Consenso multimuestra de episodios funcionales (etapa 06).

Cada estimador por tipo se muestrea ``k`` veces con el mismo prompt. Los
candidatos de muestras distintas se emparejan por IoU de miembros con el mismo
emparejador del comparador de estabilidad (cardinalidad máxima y después IoU
total) y forman grupos. Sobrevive el grupo presente en una mayoría estricta de
las muestras. Lo representa su medoide: el candidato con mayor IoU media
respecto de los demás miembros del grupo (empate: la muestra más temprana).
El desenlace es el mayoritario del grupo.

La variación observada entre muestras se registra con los códigos de
ambigüedad que el contrato ya define: ``BOUNDARY_UNCERTAIN`` cuando los
miembros difieren, ``ANCHOR_UNCERTAIN`` cuando las anclas difieren y
``OUTCOME_UNCERTAIN`` cuando el desenlace no es unánime. El código no decide
límites, anclas ni desenlaces: cuenta propuestas del LLM y conserva una de
ellas literalmente.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from util.consensus import (
    consensus_record,
    majority_label,
    majority_threshold,
    validate_sample_count,
)
from util.objective3_stability import (
    DEFAULT_EPISODE_IOU_THRESHOLD,  # un solo umbral para el consenso de 06 y para el comparador de corridas
    _anchor_set,
    _maximum_cardinality_weight_matching,
    episode_member_overlap,
)
from util.sgotgraph_objective3 import canonical_json

EPISODE_CONSENSUS_METHOD_VERSION = "episode-consensus-iou-majority-v1"

BOUNDARY_UNCERTAIN = "BOUNDARY_UNCERTAIN"
ANCHOR_UNCERTAIN = "ANCHOR_UNCERTAIN"
OUTCOME_UNCERTAIN = "OUTCOME_UNCERTAIN"


def validate_iou_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("El umbral de IoU debe ser numérico.")
    if not 0 < value <= 1:
        raise ValueError("El umbral de IoU debe estar en (0, 1].")
    return float(value)


def _structural_identity(episode: dict[str, Any]) -> str:
    return canonical_json(
        [
            episode.get("episode_type"),
            [
                (span.get("start_char"), span.get("end_char"))
                for span in episode.get("members", [])
                if isinstance(span, dict)
            ],
            sorted(_anchor_set(episode)),
        ]
    )


def _dedupe_sample(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Elimina candidatos estructuralmente idénticos dentro de una muestra."""

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for episode in sample:
        identity = _structural_identity(episode)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(episode)
    return unique


def _pairwise_iou(items: list[dict[str, Any]]) -> list[list[float]]:
    size = len(items)
    matrix = [[1.0] * size for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            iou = episode_member_overlap(items[left], items[right])["iou"]
            matrix[left][right] = iou
            matrix[right][left] = iou
    return matrix


def cluster_episode_samples(
    samples: list[list[dict[str, Any]]],
    *,
    iou_threshold: float = DEFAULT_EPISODE_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Agrupa candidatos de un mismo tipo a través de ``k`` muestras.

    Devuelve un grupo por candidato distinto observado, con los ítems
    ``(índice de muestra, episodio)`` que lo componen. Cada muestra aporta a lo
    sumo un candidato por grupo porque el emparejamiento es uno a uno.
    """

    validate_sample_count(len(samples), "samples")
    threshold = validate_iou_threshold(iou_threshold)
    clusters: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        episodes = _dedupe_sample(list(sample))
        matched: set[int] = set()
        if clusters and episodes:
            weights = [
                [
                    episode_member_overlap(episode, cluster["representative"])["iou"]
                    for cluster in clusters
                ]
                for episode in episodes
            ]
            for episode_index, cluster_index, _ in _maximum_cardinality_weight_matching(
                weights,
                threshold=threshold,
            ):
                clusters[cluster_index]["items"].append(
                    (sample_index, episodes[episode_index])
                )
                matched.add(episode_index)
        for episode_index, episode in enumerate(episodes):
            if episode_index in matched:
                continue
            clusters.append(
                {
                    "items": [(sample_index, episode)],
                    "representative": episode,
                }
            )
    return clusters


def summarize_cluster(cluster: dict[str, Any], samples: int) -> dict[str, Any]:
    """Medoide, soporte, IoU media, voto de desenlace y acuerdo de anclas."""

    items = cluster["items"]
    episodes = [episode for _, episode in items]
    matrix = _pairwise_iou(episodes)
    size = len(episodes)
    if size == 1:
        mean_by_item = [1.0]
        member_iou_mean = 1.0
    else:
        mean_by_item = [
            sum(matrix[index][other] for other in range(size) if other != index)
            / (size - 1)
            for index in range(size)
        ]
        member_iou_mean = sum(
            matrix[left][right]
            for left in range(size)
            for right in range(left + 1, size)
        ) / (size * (size - 1) / 2)
    best = max(mean_by_item)
    medoid_index = next(
        index for index, value in enumerate(mean_by_item) if value == best
    )
    outcome_vote = majority_label([episode.get("outcome") for episode in episodes])
    anchor_sets = {frozenset(_anchor_set(episode)) for episode in episodes}
    return {
        "support": len({sample_index for sample_index, _ in items}),
        "samples": samples,
        "sample_indices": [sample_index for sample_index, _ in items],
        "medoid_sample": items[medoid_index][0],
        "medoid": episodes[medoid_index],
        "member_iou_mean": member_iou_mean,
        "outcome_vote": outcome_vote,
        "anchor_agreement": len(anchor_sets) == 1,
    }


def consensus_episodes(
    samples: list[list[dict[str, Any]]],
    *,
    iou_threshold: float = DEFAULT_EPISODE_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Episodios mayoritarios de un tipo a partir de ``k`` muestras válidas.

    Cada episodio conservado es una copia literal de su medoide con el
    desenlace mayoritario, los códigos de ambigüedad derivados de la variación
    entre muestras y un registro ``consensus``. Los grupos minoritarios se
    devuelven en la auditoría, no en la salida.
    """

    sample_count = len(samples)
    threshold = majority_threshold(sample_count)
    clusters = cluster_episode_samples(samples, iou_threshold=iou_threshold)
    summaries = [summarize_cluster(cluster, sample_count) for cluster in clusters]
    kept: list[dict[str, Any]] = []
    kept_identities: set[str] = set()
    audit_clusters: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda item: (-item["support"], item["sample_indices"][0]),
    ):
        medoid = summary["medoid"]
        vote = summary["outcome_vote"]
        derived: list[str] = []
        if summary["member_iou_mean"] < 1.0:
            derived.append(BOUNDARY_UNCERTAIN)
        if not summary["anchor_agreement"]:
            derived.append(ANCHOR_UNCERTAIN)
        if not vote["unanimous"]:
            derived.append(OUTCOME_UNCERTAIN)
        status = "kept" if summary["support"] >= threshold else "minority"
        identity = _structural_identity(medoid)
        if status == "kept" and identity in kept_identities:
            status = "duplicate_of_kept"
        audit_clusters.append(
            {
                "status": status,
                "support": summary["support"],
                "samples": sample_count,
                "sample_indices": summary["sample_indices"],
                "medoid_sample": summary["medoid_sample"],
                "member_iou_mean": round(summary["member_iou_mean"], 4),
                "outcome_votes": vote["votes"],
                "anchor_agreement": summary["anchor_agreement"],
                "members": [
                    {"start_char": span["start_char"], "end_char": span["end_char"]}
                    for span in medoid.get("members", [])
                ],
            }
        )
        if status != "kept":
            continue
        kept_identities.add(identity)
        episode = deepcopy(medoid)
        episode["outcome"] = vote["label"]
        episode["ambiguity_codes"] = sorted(
            {
                *(
                    code
                    for code in episode.get("ambiguity_codes", [])
                    if isinstance(code, str)
                ),
                *derived,
            }
        )
        episode["consensus"] = consensus_record(
            sample_count,
            summary["support"],
            votes=vote["votes"],
            tie_break=vote["tie_break"],
            member_iou_mean=summary["member_iou_mean"],
        )
        kept.append(episode)
    return {
        "episodes": kept,
        "audit": {
            "method": EPISODE_CONSENSUS_METHOD_VERSION,
            "samples": sample_count,
            "iou_threshold": float(iou_threshold),
            "majority_threshold": threshold,
            "candidate_counts_by_sample": [len(sample) for sample in samples],
            "cluster_count": len(clusters),
            "kept_count": len(kept),
            "minority_count": sum(
                1 for item in audit_clusters if item["status"] == "minority"
            ),
            "clusters": audit_clusters,
        },
    }
