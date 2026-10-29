"""Transporte de la etapa 05: carga completa por fuente (canónica) y variantes.

El estimador canónico consulta un nodo fuente por solicitud con la carga
completa original (todos los demás nodos con offsets y posición relativa,
inventario plano de alias ``R0001..``). Las mediciones de septiembre de 2026
mostraron que las alternativas de transporte cambian lo que el estimador
decide, así que todas son variantes explícitas y no canónicas, registradas
en el artefacto (``estimator_variant``), en la identidad del checkpoint y en
la auditoría:

- ``compact``: carga compacta (sin offsets ni campos relativos, inventario
  denso por tipo, alias con prefijo de fuente). Reduce a la mitad los tokens
  de entrada, pero su grafo de consenso coincide con el canónico en ≈0,60
  frente a ≈0,73 entre dos ejecuciones del canónico.
- ``batched``: varias fuentes por solicitud (requiere ``compact``). Menor
  densidad, supresión de relaciones lejanas, concentración de destinos en el
  propio lote y sesgo por posición, ya desde dos fuentes por solicitud.
- ``windowed``: ventana de candidatos (requiere ``compact``); restringe las
  relaciones expresables.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar

DEFAULT_SOURCES_PER_REQUEST = 1
PAYLOAD_FULL = "full"
PAYLOAD_COMPACT = "compact"
PAYLOADS = (PAYLOAD_FULL, PAYLOAD_COMPACT)
DEFAULT_PAYLOAD = PAYLOAD_FULL
DEFAULT_MAX_OPTIONS_PER_REQUEST = 1500
CANDIDATE_POLICY_ALL = "all_other_nodes"
ESTIMATOR_VARIANT_CANONICAL = "canonical"
T = TypeVar("T")


def validate_sources_per_request(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sources_per_request debe ser un entero mayor que 0.")
    return value


def validate_batch_offset(value: Any, sources_per_request: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("batch_offset debe ser un entero no negativo.")
    if value >= sources_per_request:
        raise ValueError("batch_offset debe ser menor que sources_per_request.")
    return value


def validate_candidate_window(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("candidate_window debe ser None o un entero mayor que 0.")
    return value


def validate_max_options(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_options_per_request debe ser un entero mayor que 0.")
    return value


def candidate_policy_label(candidate_window: int | None) -> str:
    if candidate_window is None:
        return CANDIDATE_POLICY_ALL
    return f"window:{candidate_window}"


def validate_payload(value: Any) -> str:
    if value not in PAYLOADS:
        raise ValueError(f"payload debe ser uno de {PAYLOADS}.")
    return value


def validate_transport_combination(
    payload: str,
    sources_per_request: int,
    candidate_window: int | None,
) -> None:
    """La carga completa solo admite una fuente por solicitud y ninguna ventana."""

    validate_payload(payload)
    validate_sources_per_request(sources_per_request)
    validate_candidate_window(candidate_window)
    if payload == PAYLOAD_FULL and sources_per_request > 1:
        raise ValueError(
            "Agrupar fuentes en lotes requiere la carga compacta "
            "(payload=compact); la carga completa canónica consulta una "
            "fuente por solicitud."
        )
    if payload == PAYLOAD_FULL and candidate_window is not None:
        raise ValueError(
            "La ventana de candidatos requiere la carga compacta "
            "(payload=compact); el estimador canónico consulta todos los "
            "demás nodos."
        )


def estimator_variant(
    candidate_window: int | None,
    sources_per_request: int = DEFAULT_SOURCES_PER_REQUEST,
    payload: str = DEFAULT_PAYLOAD,
) -> str:
    """Etiqueta de variante: canónica solo con carga completa, una fuente y sin ventana."""

    parts: list[str] = []
    if validate_payload(payload) == PAYLOAD_COMPACT:
        parts.append("compact")
    if validate_sources_per_request(sources_per_request) > 1:
        parts.append("batched")
    if validate_candidate_window(candidate_window) is not None:
        parts.append("windowed")
    return "_".join(parts) if parts else ESTIMATOR_VARIANT_CANONICAL


def chunk_sources(
    items: Sequence[T],
    size: int,
    *,
    offset: int = 0,
    max_options: int | None = None,
    weight: Callable[[T], int] | None = None,
) -> list[list[T]]:
    """Bloques consecutivos, deterministas y exhaustivos de fuentes.

    ``size`` es el máximo de fuentes por bloque; ``offset`` acorta el primer
    bloque a ``size - offset`` fuentes para desplazar las fronteras (útil para
    medir el sesgo de frontera). ``max_options`` limita la suma de ``weight``
    por bloque (tope de opciones por solicitud); un bloque nunca queda vacío.
    """

    validate_sources_per_request(size)
    validate_batch_offset(offset, size)
    if max_options is not None:
        validate_max_options(max_options)
    batches: list[list[T]] = []
    current: list[T] = []
    current_weight = 0
    limit = size - offset
    for item in items:
        item_weight = weight(item) if weight is not None else 0
        exceeds_options = (
            max_options is not None
            and current
            and current_weight + item_weight > max_options
        )
        if current and (len(current) >= limit or exceeds_options):
            batches.append(current)
            current = []
            current_weight = 0
            limit = size
        current.append(item)
        current_weight += item_weight
    if current:
        batches.append(current)
    return batches
