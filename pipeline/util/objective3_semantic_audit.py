"""Registro durable de los intentos semánticos del objetivo 3.

El diario JSONL es un artefacto operacional separado del registro canónico.
Se escribe después de cada respuesta estructurada —válida o inválida— y
también cuando una solicitud falla antes de producir una respuesta analizable.
Así, un fallo terminal no elimina la evidencia de los intentos anteriores.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


SEMANTIC_ATTEMPT_SCHEMA_VERSION = "objective3-semantic-attempt-v1"


def initialize_semantic_attempt_journal(
    path: Path | str | None,
) -> Path | None:
    """Crea un diario vacío para una nueva ejecución.

    La inicialización es deliberadamente explícita: las funciones de
    inferencia solo anexan registros, de modo que varias trazas de una misma
    ejecución puedan compartir el archivo sin borrar intentos anteriores.
    """

    if path is None:
        return None
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("", encoding="utf-8")
    return journal_path


def _attempt_id(trace_id: str, event: dict[str, Any]) -> str:
    identity = {
        "trace_id": trace_id,
        "stage": event.get("stage"),
        "phase": event.get("phase"),
        "episode_type": event.get("episode_type"),
        "block_id": event.get("block_id"),
        "attempt": event.get("attempt"),
        "prompt_hash": event.get("prompt_hash"),
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"attempt_{sha256(encoded).hexdigest()}"


def append_semantic_attempt(
    path: Path | str | None,
    *,
    trace_id: str,
    event: dict[str, Any],
    response_payload: Any | None,
) -> str | None:
    """Anexa y sincroniza un intento completo en una única escritura."""

    if path is None:
        return None
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_id = _attempt_id(trace_id, event)
    record = {
        "schema_version": SEMANTIC_ATTEMPT_SCHEMA_VERSION,
        "trace_id": trace_id,
        "attempt_id": attempt_id,
        "event": event,
        "response_payload": response_payload,
    }
    line = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        journal_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return attempt_id


__all__ = [
    "SEMANTIC_ATTEMPT_SCHEMA_VERSION",
    "append_semantic_attempt",
    "initialize_semantic_attempt_journal",
]
