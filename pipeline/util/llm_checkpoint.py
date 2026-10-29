"""Persistencia operacional de decisiones LLM aceptadas.

Los checkpoints no modifican una decisión ni completan salidas parciales. Solo
permiten reutilizar una salida que ya pasó el mismo contrato determinístico y
cuya identidad de ejecución coincide exactamente con la corrida actual.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = "llm-item-checkpoint-v1"


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def checkpoint_path_for_trace(
    checkpoint_dir: Path | None,
    stage: str,
    trace_id: str,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", trace_id).strip("._-")
    readable = readable[:80] or "trace"
    digest = sha256(trace_id.encode("utf-8")).hexdigest()[:10]
    return checkpoint_dir / stage / f"{readable}--{digest}.json"


class ItemCheckpoint:
    """Checkpoint atómico por ítems con invalidación estricta por identidad."""

    def __init__(
        self,
        path: Path | None,
        *,
        identity: dict[str, Any],
        resume: bool,
    ) -> None:
        self.path = path
        self.identity = deepcopy(identity)
        self.identity_hash = canonical_json_hash(identity)
        self.items: dict[str, dict[str, Any]] = {}
        self.reused_item_ids: list[str] = []
        self.load_status = "disabled" if path is None else "fresh"

        if path is None:
            return

        if resume and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.load_status = "invalid_file"
            else:
                stored_items = payload.get("items")
                if (
                    payload.get("schema_version")
                    == CHECKPOINT_SCHEMA_VERSION
                    and payload.get("identity") == self.identity
                    and payload.get("identity_hash") == self.identity_hash
                    and isinstance(stored_items, dict)
                    and all(
                        isinstance(item_id, str)
                        and isinstance(item, dict)
                        for item_id, item in stored_items.items()
                    )
                ):
                    self.items = deepcopy(stored_items)
                    self.load_status = "resumed"
                else:
                    self.load_status = "identity_mismatch"

        self._write()

    def get(self, item_id: str) -> dict[str, Any] | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        return deepcopy(item)

    def mark_reused(self, item_id: str) -> None:
        if item_id not in self.reused_item_ids:
            self.reused_item_ids.append(item_id)

    def put(self, item_id: str, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.items[item_id] = deepcopy(payload)
        self._write()

    def discard(self, item_id: str) -> None:
        if item_id not in self.items:
            return
        del self.items[item_id]
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "identity": self.identity,
                    "identity_hash": self.identity_hash,
                    "items": self.items,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
