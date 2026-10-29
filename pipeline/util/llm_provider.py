from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError


SUPPORTED_PROVIDERS = {"openai", "deepseek"}
DEFAULT_PROVIDER = "openai"
# Modelo del anotador para OpenAI: única definición; las etapas 01, 01.1, 02, 05, 06 y 07 la importan.
# Cambiarlo es cambiar el instrumento de medida (ver tests/test_instrument_fingerprint.py).
# gpt-5-mini-2025-08-07, usado hasta la versión 2026.09, ya no está disponible en la API.
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_THINKING = "enabled"
DEFAULT_DEEPSEEK_MAX_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_TRANSPORT_MAX_RETRIES = 2

T = TypeVar("T", bound=BaseModel)
_METRICS_LOCK = Lock()


@dataclass
class LLMClient:
    provider: str
    client: OpenAI
    request_timeout_seconds: float | None = None
    transport_max_retries: int | None = None


@dataclass
class StructuredResponse:
    parsed: BaseModel
    provider: str
    model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    duration_seconds: float | None = None
    token_usage: dict[str, int] | None = None


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Carga pares simples ``CLAVE=VALOR`` sin sobrescribir el entorno."""

    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalize_provider(provider: str | None) -> str:
    normalized = (provider or DEFAULT_PROVIDER).strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Proveedor LLM invalido {provider!r}. Usa uno de: {supported}.")
    return normalized


def add_llm_provider_args(parser: argparse.ArgumentParser, openai_default_model: str) -> None:
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER),
        choices=sorted(SUPPORTED_PROVIDERS),
        help="Proveedor LLM a usar: openai o deepseek. Default: LLM_PROVIDER u openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Modelo LLM a usar. Default: OPENAI_MODEL o "
            f"{openai_default_model} para OpenAI; DEEPSEEK_MODEL o "
            f"{DEFAULT_DEEPSEEK_MODEL} para DeepSeek."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key del proveedor. Si no se indica, usa OPENAI_API_KEY "
            "o DEEPSEEK_API_KEY segun --provider."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL opcional del proveedor. Para DeepSeek usa DEEPSEEK_BASE_URL "
            f"o {DEFAULT_DEEPSEEK_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=os.getenv(
            "LLM_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
        ),
        help=(
            "Tiempo máximo de espera por solicitud al proveedor. Default: "
            "LLM_REQUEST_TIMEOUT_SECONDS o 180 segundos."
        ),
    )
    parser.add_argument(
        "--transport-max-retries",
        type=int,
        default=os.getenv(
            "LLM_TRANSPORT_MAX_RETRIES",
            str(DEFAULT_TRANSPORT_MAX_RETRIES),
        ),
        help=(
            "Reintentos de transporte administrados por el cliente antes de "
            "propagar el error. Default: LLM_TRANSPORT_MAX_RETRIES o 2."
        ),
    )


def finalize_llm_args(args: argparse.Namespace, openai_default_model: str) -> argparse.Namespace:
    provider = normalize_provider(args.provider)
    args.provider = provider

    if not args.model:
        if provider == "openai":
            args.model = os.getenv("OPENAI_MODEL", openai_default_model)
        else:
            args.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    if not args.api_key:
        env_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        args.api_key = os.getenv(env_name)

    if not args.base_url:
        if provider == "openai":
            args.base_url = os.getenv("OPENAI_BASE_URL") or None
        else:
            args.base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)

    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds debe ser mayor que 0.")
    if args.transport_max_retries < 0:
        raise ValueError("--transport-max-retries no puede ser negativo.")

    return args


def build_llm_client(
    provider: str,
    api_key: str | None,
    base_url: str | None = None,
    request_timeout_seconds: float | None = None,
    transport_max_retries: int | None = None,
) -> LLMClient:
    provider = normalize_provider(provider)
    if not api_key:
        env_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        raise ValueError(f"No encontre la API key. Pasa --api-key o define {env_name}.")

    if request_timeout_seconds is None:
        request_timeout_seconds = float(
            os.getenv(
                "LLM_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        )
    if transport_max_retries is None:
        transport_max_retries = int(
            os.getenv(
                "LLM_TRANSPORT_MAX_RETRIES",
                str(DEFAULT_TRANSPORT_MAX_RETRIES),
            )
        )
    if request_timeout_seconds <= 0:
        raise ValueError("El timeout de solicitudes debe ser mayor que 0.")
    if transport_max_retries < 0:
        raise ValueError("Los reintentos de transporte no pueden ser negativos.")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": request_timeout_seconds,
        "max_retries": transport_max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return LLMClient(
        provider=provider,
        client=OpenAI(**kwargs),
        request_timeout_seconds=request_timeout_seconds,
        transport_max_retries=transport_max_retries,
    )


def extract_openai_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    raise ValueError("La respuesta del modelo no contiene output_text.")


def extract_deepseek_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("La respuesta de DeepSeek no contiene choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content:
        return content

    raise ValueError(
        "La respuesta de DeepSeek no contiene content. "
        "Si ocurre ocasionalmente, reintenta o aumenta DEEPSEEK_MAX_TOKENS."
    )


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_json_fences(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Primer objeto JSON completo; lo que siga (comentarios del modelo,
        # incluso con llaves) se ignora.
        start = stripped.find("{")
        if start == -1:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(stripped, start)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta JSON debe ser un objeto.")
    return parsed


def deepseek_max_tokens() -> int:
    raw_value = os.getenv("DEEPSEEK_MAX_TOKENS", str(DEFAULT_DEEPSEEK_MAX_TOKENS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("DEEPSEEK_MAX_TOKENS debe ser un entero.") from exc
    if value <= 0:
        raise ValueError("DEEPSEEK_MAX_TOKENS debe ser mayor que 0.")
    return value


def deepseek_extra_body(reasoning_effort: str | None) -> dict[str, Any]:
    thinking = os.getenv("DEEPSEEK_THINKING", DEFAULT_DEEPSEEK_THINKING).strip().lower()
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("DEEPSEEK_THINKING debe ser 'enabled' o 'disabled'.")

    extra_body: dict[str, Any] = {"thinking": {"type": thinking}}
    if thinking == "enabled" and reasoning_effort:
        if reasoning_effort in {"low", "medium", "high"}:
            extra_body["reasoning_effort"] = "high"
        elif reasoning_effort == "max":
            extra_body["reasoning_effort"] = "max"
    return extra_body


def provider_instruction(system_prompt: str) -> str:
    return (
        f"{system_prompt.rstrip()}\n\n"
        "Provider compatibility instruction: return exactly one valid JSON object. "
        "Do not wrap the JSON in markdown or add explanatory text."
    )


def structured_request_hash(
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Huella canónica de los mensajes textuales que recibe el proveedor."""

    effective_system = (
        provider_instruction(system_prompt)
        if llm_client.provider == "deepseek"
        else system_prompt
    )
    payload = [
        {"role": "system", "content": effective_system},
        {"role": "user", "content": user_prompt},
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def _integer_attribute(value: Any, *names: str) -> int | None:
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        if isinstance(value, dict):
            candidate = value.get(name)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    return None


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result: dict[str, int] = {}
    input_tokens = _integer_attribute(
        usage,
        "input_tokens",
        "prompt_tokens",
    )
    output_tokens = _integer_attribute(
        usage,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _integer_attribute(usage, "total_tokens")
    if input_tokens is not None:
        result["input_tokens"] = input_tokens
    if output_tokens is not None:
        result["output_tokens"] = output_tokens
    if total_tokens is not None:
        result["total_tokens"] = total_tokens

    output_details = getattr(usage, "output_tokens_details", None)
    if output_details is None:
        output_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = _integer_attribute(
        output_details,
        "reasoning_tokens",
    )
    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens

    input_details = getattr(usage, "input_tokens_details", None)
    if input_details is None:
        input_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = _integer_attribute(input_details, "cached_tokens")
    if cached_tokens is not None:
        result["cached_input_tokens"] = cached_tokens
    return result


def _append_request_metric(payload: dict[str, Any]) -> None:
    raw_path = os.getenv("LLM_REQUEST_METRICS_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    try:
        with _METRICS_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
    except OSError:
        # La telemetría operacional nunca puede alterar el resultado científico.
        return


def request_structured_response(
    llm_client: LLMClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    reasoning_effort: str | None = None,
) -> StructuredResponse:
    started_at = datetime.now(timezone.utc)
    started_clock = monotonic()
    request_hash = structured_request_hash(
        llm_client,
        system_prompt,
        user_prompt,
    )
    response: Any = None
    try:
        if llm_client.provider == "openai":
            kwargs: dict[str, Any] = {
                "model": model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "text_format": response_model,
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            try:
                response = llm_client.client.responses.parse(**kwargs)
                parsed = getattr(response, "output_parsed", None)
            except ValidationError:
                # El modelo devolvió texto después del objeto JSON (visto con
                # trazas largas) y el SDK rechaza la respuesta entera. Se repite
                # la misma solicitud sin el parseo del SDK y se extrae el objeto
                # JSON de forma tolerante; el prompt y el esquema no cambian.
                kwargs.pop("text_format")
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": False,
                    }
                }
                response = llm_client.client.responses.create(**kwargs)
                parsed = None
            if parsed is None:
                parsed = response_model.model_validate(
                    parse_json_object(
                        extract_openai_response_text(response)
                    )
                )
        else:
            response = llm_client.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": provider_instruction(system_prompt),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=deepseek_max_tokens(),
                extra_body=deepseek_extra_body(reasoning_effort),
            )
            parsed = response_model.model_validate(
                parse_json_object(
                    extract_deepseek_message_content(response)
                )
            )
    except Exception as exc:
        duration_seconds = monotonic() - started_clock
        _append_request_metric(
            {
                "schema_version": "llm-request-metric-v1",
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration_seconds, 6),
                "status": "error",
                "stage": os.getenv("LLM_STAGE_NAME"),
                "provider": llm_client.provider,
                "requested_model": model,
                "response_model": response_model.__name__,
                "reasoning_effort": reasoning_effort,
                "request_hash": request_hash,
                "error_type": type(exc).__name__,
            }
        )
        raise

    duration_seconds = monotonic() - started_clock
    token_usage = _response_token_usage(response)
    resolved_model = getattr(response, "model", None) or model
    response_id = getattr(response, "id", None)
    system_fingerprint = getattr(response, "system_fingerprint", None)
    _append_request_metric(
        {
            "schema_version": "llm-request-metric-v1",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 6),
            "status": "success",
            "stage": os.getenv("LLM_STAGE_NAME"),
            "provider": llm_client.provider,
            "requested_model": model,
            "resolved_model": resolved_model,
            "response_model": response_model.__name__,
            "reasoning_effort": reasoning_effort,
            "request_hash": request_hash,
            "response_id": response_id,
            "system_fingerprint": system_fingerprint,
            "token_usage": token_usage,
        }
    )
    return StructuredResponse(
        parsed=parsed,
        provider=llm_client.provider,
        model=resolved_model,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        duration_seconds=duration_seconds,
        token_usage=token_usage,
    )
