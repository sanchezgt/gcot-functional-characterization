"""Caché estricta de salidas completas de etapas del pipeline."""

from __future__ import annotations

from hashlib import sha256
import json
import shutil
from pathlib import Path
from typing import Any

from util.llm_checkpoint import canonical_json_hash


STAGE_CACHE_SCHEMA_VERSION = "pipeline-stage-cache-v1"

_OPERATIONAL_FLAGS = {
    "--pretty",
    "--resume",
    "--resume-checkpoint",
    "--resume-internal",
    "--verbose",
}

_OPERATIONAL_OPTIONS_WITH_VALUE = {
    "--checkpoint-dir",
    "--semantic-audit-path",
}

_CONFIG_ENV_NAMES = {
    "LLM_PROVIDER",
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_TRANSPORT_MAX_RETRIES",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_REFINEMENT_REASONING_EFFORT",
    "OPENAI_TYPING_REASONING_EFFORT",
    "OPENAI_EDGE_REASONING_EFFORT",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_THINKING",
    "DEEPSEEK_MAX_TOKENS",
}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def implementation_tree_hash(root: Path) -> str:
    digest = sha256()
    files = sorted(
        path
        for path in root.rglob("*.py")
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "tests" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def build_stage_signature(
    *,
    root: Path,
    namespace: str,
    stage_name: str,
    script_name: str,
    input_path: Path,
    output_path: Path,
    command: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    normalized_command = []
    input_text = str(input_path)
    output_text = str(output_path)
    skip_next = False
    for token in command[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in _OPERATIONAL_OPTIONS_WITH_VALUE:
            skip_next = True
            continue
        if token in _OPERATIONAL_FLAGS:
            continue
        if token == input_text:
            normalized_command.append("<INPUT>")
        elif token == output_text:
            normalized_command.append("<OUTPUT>")
        else:
            normalized_command.append(token)
    return {
        "namespace": namespace,
        "stage_name": stage_name,
        "script_name": script_name,
        "input_hash": file_hash(input_path),
        "implementation_hash": implementation_tree_hash(root),
        "command": normalized_command,
        "environment": {
            name: env[name]
            for name in sorted(_CONFIG_ENV_NAMES)
            if name in env
        },
    }


def _cache_paths(
    checkpoint_dir: Path,
    namespace: str,
    stage_name: str,
    suffix: str,
) -> tuple[Path, Path]:
    cache_root = checkpoint_dir / "stage_outputs"
    basename = f"{namespace}__{stage_name}{suffix}"
    return cache_root / basename, cache_root / f"{basename}.meta.json"


def _is_valid_json_artifact(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not text:
        return False
    try:
        if path.suffix.lower() == ".json":
            json.loads(text)
        else:
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
    except json.JSONDecodeError:
        return False
    return True


def restore_stage_output(
    *,
    checkpoint_dir: Path | None,
    namespace: str,
    stage_name: str,
    output_path: Path,
    signature: dict[str, Any],
) -> bool:
    if checkpoint_dir is None:
        return False
    cache_path, metadata_path = _cache_paths(
        checkpoint_dir,
        namespace,
        stage_name,
        output_path.suffix,
    )
    if not cache_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("schema_version") != STAGE_CACHE_SCHEMA_VERSION
        or metadata.get("signature") != signature
        or metadata.get("signature_hash")
        != canonical_json_hash(signature)
        or metadata.get("output_hash") != file_hash(cache_path)
        or not _is_valid_json_artifact(cache_path)
    ):
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    shutil.copyfile(cache_path, temporary)
    temporary.replace(output_path)
    return True


def store_stage_output(
    *,
    checkpoint_dir: Path | None,
    namespace: str,
    stage_name: str,
    output_path: Path,
    signature: dict[str, Any],
) -> None:
    if checkpoint_dir is None:
        return
    if not _is_valid_json_artifact(output_path):
        raise ValueError(
            f"La etapa {stage_name} no produjo un artefacto JSON válido."
        )
    cache_path, metadata_path = _cache_paths(
        checkpoint_dir,
        namespace,
        stage_name,
        output_path.suffix,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(f".{cache_path.name}.tmp")
    shutil.copyfile(output_path, temporary_cache)
    temporary_cache.replace(cache_path)
    metadata = {
        "schema_version": STAGE_CACHE_SCHEMA_VERSION,
        "signature": signature,
        "signature_hash": canonical_json_hash(signature),
        "output_hash": file_hash(cache_path),
    }
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.tmp"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
