"""Alias opaco de traza para los prompts.

El identificador de un registro (por ejemplo ``gsm8k_deepseek_r1_0001``)
revela el dataset y el modelo generador. El método es agnóstico al
generador, así que ningún prompt debe recibir ese identificador: los
estimadores reciben un alias determinista sin contenido y el artefacto
conserva el ``trace_id`` real fuera del prompt.
"""

from __future__ import annotations

from hashlib import sha256

TRACE_ALIAS_PREFIX = "trace_"
TRACE_ALIAS_DIGEST_CHARS = 12


def prompt_trace_alias(trace_id: str) -> str:
    """Alias estable y opaco para usar en lugar de ``trace_id`` en un prompt."""

    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("trace_id debe ser una cadena no vacía.")
    digest = sha256(trace_id.encode("utf-8")).hexdigest()
    return TRACE_ALIAS_PREFIX + digest[:TRACE_ALIAS_DIGEST_CHARS]
