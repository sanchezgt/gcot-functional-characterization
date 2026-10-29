"""Consenso multimuestra sobre decisiones LLM.

Un estimador semántico se muestrea ``k`` veces de forma independiente con el
mismo prompt y el código conserva únicamente la decisión mayoritaria. La
agregación es aritmética: no aplica prioridad entre categorías ni completa
salidas. Cuando un empate no puede resolverse con muestras adicionales, se
usa el orden de muestreo (nunca el tipo) y el desempate queda registrado.

El objetivo del consenso es reducir la varianza del estimador con entrada
idéntica; no sustituye la validación semántica.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, Iterable, Sequence


CONSENSUS_METHOD_VERSION = "consensus-majority-v1"
DEFAULT_CONSENSUS_SAMPLES = 3
DEFAULT_MAX_EXTRA_SAMPLES = 2

AMBIGUITY_UNSTABLE = "UNSTABLE_ACROSS_SAMPLES"
AMBIGUITY_TIE = "CONSENSUS_TIE"
TIE_BREAK_FIRST_SAMPLE = "first_sample"

CONSENSUS_FIELDS = {
    "method",
    "samples",
    "support",
    "agreement",
    "unanimous",
    "votes",
    "tie_break",
    "member_iou_mean",
}


def validate_sample_count(samples: Any, name: str = "samples") -> int:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError(f"{name} debe ser un entero mayor que 0.")
    return samples


def majority_threshold(samples: int) -> int:
    """Votos necesarios para una mayoría estricta (> k/2)."""

    return validate_sample_count(samples) // 2 + 1


def has_tie(labels: Sequence[Hashable]) -> bool:
    if not labels:
        raise ValueError("No hay etiquetas para evaluar el empate.")
    counts = Counter(labels)
    top = max(counts.values())
    return sum(1 for count in counts.values() if count == top) > 1


def majority_label(labels: Sequence[Hashable]) -> dict[str, Any]:
    """Etiqueta mayoritaria de una secuencia ordenada de muestras.

    Ante un empate entre las etiquetas más votadas se devuelve la que apareció
    primero en el orden de muestreo y se marca ``tie=True`` con
    ``tie_break="first_sample"``. Ningún criterio depende del valor de la
    etiqueta.
    """

    if not labels:
        raise ValueError("No hay etiquetas para calcular la mayoría.")
    counts = Counter(labels)
    top = max(counts.values())
    winners = {label for label, count in counts.items() if count == top}
    tie = len(winners) > 1
    label = next(value for value in labels if value in winners)
    return {
        "label": label,
        "votes": {str(key): count for key, count in counts.items()},
        "samples": len(labels),
        "support": top,
        "agreement": top / len(labels),
        "unanimous": len(counts) == 1,
        "tie": tie,
        "tie_break": TIE_BREAK_FIRST_SAMPLE if tie else None,
    }


def majority_set(sample_sets: Sequence[Iterable[Hashable]]) -> dict[str, Any]:
    """Elementos presentes en una mayoría estricta de las muestras.

    Devuelve los elementos conservados en orden de primera aparición, el
    soporte de cada elemento observado y el umbral aplicado.
    """

    if not sample_sets:
        raise ValueError("No hay muestras para calcular el consenso.")
    samples = len(sample_sets)
    threshold = majority_threshold(samples)
    support: dict[Hashable, int] = {}
    order: list[Hashable] = []
    for sample in sample_sets:
        seen: set[Hashable] = set()
        for element in sample:
            if element in seen:
                continue
            seen.add(element)
            if element not in support:
                support[element] = 0
                order.append(element)
            support[element] += 1
    kept = [element for element in order if support[element] >= threshold]
    return {
        "kept": kept,
        "support": support,
        "samples": samples,
        "threshold": threshold,
    }


def consensus_record(
    samples: int,
    support: int,
    *,
    votes: dict[str, int] | None = None,
    tie_break: str | None = None,
    member_iou_mean: float | None = None,
) -> dict[str, Any]:
    """Registro serializable del consenso de una decisión.

    ``member_iou_mean`` solo aplica a decisiones sobre intervalos (episodios):
    IoU media de los miembros entre las muestras que respaldan la decisión.
    """

    validate_sample_count(samples)
    if isinstance(support, bool) or not isinstance(support, int):
        raise ValueError("support debe ser un entero.")
    if support <= 0 or support > samples:
        raise ValueError("support debe estar entre 1 y samples.")
    record: dict[str, Any] = {
        "method": CONSENSUS_METHOD_VERSION,
        "samples": samples,
        "support": support,
        "agreement": support / samples,
        "unanimous": support == samples,
    }
    if votes is not None:
        record["votes"] = dict(votes)
    if tie_break is not None:
        record["tie_break"] = tie_break
    if member_iou_mean is not None:
        if (
            isinstance(member_iou_mean, bool)
            or not isinstance(member_iou_mean, (int, float))
            or not 0 <= member_iou_mean <= 1
        ):
            raise ValueError("member_iou_mean debe estar en [0, 1].")
        record["member_iou_mean"] = round(float(member_iou_mean), 6)
    return record


def validate_consensus_record(value: Any, label: str) -> list[str]:
    """Errores de contrato de un registro ``consensus``."""

    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label}: consensus debe ser un objeto."]
    unknown = set(value) - CONSENSUS_FIELDS
    if unknown:
        errors.append(
            f"{label}: consensus contiene campos no permitidos: "
            f"{sorted(str(field) for field in unknown)}."
        )
    samples = value.get("samples")
    support = value.get("support")
    if not isinstance(value.get("method"), str) or not value.get("method"):
        errors.append(f"{label}: consensus.method es obligatorio.")
    valid_samples = (
        isinstance(samples, int) and not isinstance(samples, bool) and samples > 0
    )
    if not valid_samples:
        errors.append(f"{label}: consensus.samples debe ser un entero positivo.")
    if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
        errors.append(f"{label}: consensus.support debe ser un entero positivo.")
    elif valid_samples and support > samples:
        errors.append(f"{label}: consensus.support no puede superar samples.")
    agreement = value.get("agreement")
    if (
        isinstance(agreement, bool)
        or not isinstance(agreement, (int, float))
        or not 0 < agreement <= 1
    ):
        errors.append(f"{label}: consensus.agreement debe estar en (0, 1].")
    if "unanimous" in value and not isinstance(value["unanimous"], bool):
        errors.append(f"{label}: consensus.unanimous debe ser booleano.")
    votes = value.get("votes")
    if votes is not None and (
        not isinstance(votes, dict)
        or not all(
            isinstance(key, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            for key, count in votes.items()
        )
    ):
        errors.append(
            f"{label}: consensus.votes debe mapear cadenas a enteros positivos."
        )
    if "tie_break" in value and value["tie_break"] not in (
        None,
        TIE_BREAK_FIRST_SAMPLE,
    ):
        errors.append(f"{label}: consensus.tie_break inválido.")
    if "member_iou_mean" in value:
        iou = value["member_iou_mean"]
        if (
            isinstance(iou, bool)
            or not isinstance(iou, (int, float))
            or not 0 <= iou <= 1
        ):
            errors.append(f"{label}: consensus.member_iou_mean debe estar en [0, 1].")
    return errors
