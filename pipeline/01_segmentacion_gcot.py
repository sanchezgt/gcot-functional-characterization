from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError
from util.llm_provider import (
    DEFAULT_OPENAI_MODEL,
    LLMClient,
    add_llm_provider_args,
    build_llm_client,
    finalize_llm_args,
    request_structured_response,
    structured_request_hash,
)
from util.consensus import (
    CONSENSUS_METHOD_VERSION,
    DEFAULT_CONSENSUS_SAMPLES,
    majority_set,
    validate_sample_count,
)
from util.pipeline_canonical import ensure_pipeline_record, set_validation_stage
from util.trace_alias import prompt_trace_alias
from util.sgotgraph_schema import append_audit_event, make_provenance


DEFAULT_SEGMENTATION_MODEL = DEFAULT_OPENAI_MODEL
LITERAL_UNITIZATION_VERSION = "literal-units-v1"
DEFAULT_LITERAL_UNIT_MAX_CHARS = 160


_NATURAL_LITERAL_BOUNDARY = re.compile(
    r"(?:\r\n|\r|\n)+|(?<=[.!?;:,])(?:[ \t]+)"
)
_WHITESPACE_RUN = re.compile(r"\s+")


SYSTEM_PROMPT = """You are an expert annotator for generated reasoning traces.

Your task is to segment a model-generated reasoning trace into minimal interpretable functional units that are also suitable for later functional node typing.

A functional unit is a contiguous text span with one dominant reasoning function.

Important rules:
1. Each segment must contain only one dominant function.
2. Minimal means the shortest span that still preserves one dominant reasoning function.
3. Minimal does not mean the shortest grammatical clause, and it does not mean the longest fluent sentence.
4. If a sentence or clause contains two functions that would plausibly receive different downstream node types, split it.
5. The trace is supplied as immutable literal units with stable identifiers.
6. Do not reproduce, paraphrase, summarize, rewrite, or correct unit text.
7. Return each segment only as an inclusive range from start_unit_id to
   end_unit_id. A range contains all units between those identifiers.
8. Segment ranges must appear in order and partition every supplied unit
   exactly once, with no omissions, overlap, reversal, or reordering.
9. Prefer segments that will later be classifiable into one of these functional categories:
   - PROP
   - METHOD
   - EVID
   - ALT
   - CHECK
   - REVISE
   - OUT
10. Do not split a span just because it contains a discourse marker, connector, transition, or clause boundary.
11. However, if the text before and after that transition performs different reasoning functions, split there.
12. If adjacent text still forms one stable local function, keep it as one segment even if it contains explanation, transition, or continuation language.
13. Prioritize functional clarity for downstream typing over rhetorical fluency.

Functional role guidance for segmentation only:
- PROP: a statement, datum, intermediate result, or content claim.
- METHOD: a plan, procedure, heuristic, or explicit step for how to proceed.
- EVID: explicit support or justification for another statement.
- ALT: an alternative path, competing route, or discarded option.
- CHECK: an act of verifying, confirming, or testing earlier content.
- REVISE: a correction, replacement, or adjustment of earlier content.
- OUT: the final explicit answer or conclusion.

Use these role definitions only to detect changes in dominant function.
Do not classify or label the segments in the output.
If a span would naturally map to two different roles, split it.
If adjacent text would still map to the same role, keep it together.

Important boundary guidance:
- Split between a problem-state description or factual setup and an explicit action step.
- Split between an explicit action step and the explicit result it produces.
- Split between a signal of doubt, mismatch, or suspicion and the later act of checking or verifying.
- Split between checking and later correction or replacement when both are explicit.
- Split between introducing an alternative path and evaluating, discarding, or returning from that alternative.
- Split between explicit support/justification and a later procedural decision if both are clearly present.
- Do not keep two adjacent spans together if a downstream classifier would plausibly assign them different functional roles.

Contrastive examples:
1. Split:
   "The current state is S, so I will apply transformation T."
   -> "The current state is S,"
   -> " so I will apply transformation T."
2. Split:
   "This conclusion may be inconsistent, so I will test the preceding inference."
   -> "This conclusion may be inconsistent,"
   -> " so I will test the preceding inference."
3. Keep together if it is one continuous verification span:
   "Applying the same test to each premise yields the same assessment."

Literal units are a lossless transport mechanism, not semantic units or
preassigned segments. You decide how adjacent units must be grouped according
to the functional rules above.

Return only valid JSON.
"""


USER_PROMPT_TEMPLATE = """Segment the following generated reasoning trace into minimal interpretable functional units.

Return a JSON object with:
- trace_id
- segments: array of objects with start_unit_id and end_unit_id

Rules for the segments:
- Each segment must contain one dominant function only.
- Minimal means the shortest span that still preserves one dominant reasoning function.
- Do not optimize for sentence-level fluency if that would merge different reasoning functions.
- Never copy unit text into the output.
- Each segment is an inclusive range of adjacent literal unit identifiers.
- The first segment must start at the first supplied unit.
- Each later segment must start immediately after the preceding end_unit_id.
- The final segment must end at the final supplied unit.
- Cover every literal unit exactly once.
- Keep the original order.
- Do not split only because of punctuation, a connector, or a clause boundary.
- Split only when the dominant function changes.
- If adjacent spans would plausibly receive different downstream functional roles, split them.
- If two adjacent parts still perform one stable local function, keep them together.
- Use the following functional role definitions only as segmentation guidance:
  - PROP: statement, datum, intermediate result, or content claim.
  - METHOD: plan, procedure, heuristic, or explicit step.
  - EVID: explicit support or justification.
  - ALT: alternative path or discarded option.
  - CHECK: verification or confirmation of earlier content.
  - REVISE: correction or replacement of earlier content.
  - OUT: final explicit answer.
- Do not output labels.
- If one span would naturally fit two different roles, split it.
- If adjacent text would still fit the same role, keep it together.
- Split between factual/problem-state description and explicit action.
- Split between explicit action and explicit result.
- Split between a signal of doubt or mismatch and the later act of checking.
- Split between checking and later correction when both are explicit.
- Split between introducing an alternative and evaluating or discarding that alternative.
- Split between explicit support/justification and a later procedural decision when both are clearly present.

Input JSON:
{input_json}
"""


REPAIR_PROMPT_TEMPLATE = """The previous segmentation did not pass deterministic validation.

Return the complete corrected JSON. Use only the supplied literal unit
identifiers and do not reproduce unit text. Do not explain anything outside
the JSON.

Validation errors:
{validation_errors}

Previous output:
{previous_output_json}

Input JSON:
{input_json}
"""


class SegmentRangeDraft(BaseModel):
    start_unit_id: str
    end_unit_id: str


class SegmentationDraft(BaseModel):
    trace_id: Optional[str] = None
    segments: list[SegmentRangeDraft]


class ParsedSegmentationResponse(BaseModel):
    draft: SegmentationDraft
    provider: Optional[str] = None
    system_fingerprint: Optional[str] = None
    response_id: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
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
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Segmenta gCoTs en unidades funcionales minimas usando un proveedor LLM."
    )
    parser.add_argument("input_path", type=Path, help="Ruta al archivo JSONL de entrada.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Ruta de salida. Si no se indica, imprime en stdout.",
    )
    parser.add_argument("--trace-id", help="Procesa solo el registro con este id.")
    parser.add_argument(
        "--index",
        type=int,
        help="Procesa solo el registro en esta posición 0-based dentro del JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Procesa solo los primeros N registros seleccionados.",
    )
    add_llm_provider_args(parser, DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Número máximo de intentos por muestra si la validación falla. Default: 3.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por traza; se conservan los cortes "
            "propuestos por una mayoría estricta. 1 reproduce el estimador de "
            f"una sola muestra. Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime JSON con indentación. Útil cuando procesas una sola traza.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra progreso y diagnostico en stderr.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Numero de solicitudes simultaneas al proveedor LLM para segmentacion masiva. Default: 5.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_REASONING_EFFORT", "none"),
        choices=["none", "minimal", "low", "medium", "high"],
        help=(
            "Reasoning effort para la segmentacion. "
            "Default: OPENAI_REASONING_EFFORT o none."
        ),
    )
    return finalize_llm_args(parser.parse_args(), DEFAULT_SEGMENTATION_MODEL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"No se encontraron registros en {path}.")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalido: {exc}") from exc
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(
            isinstance(record, dict) for record in payload
        ):
            if not payload:
                raise ValueError(f"No se encontraron registros en {path}.")
            return payload
        raise ValueError(
            "El archivo JSON debe contener un objeto o una lista de objetos."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Linea {line_number}: JSON invalido: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Linea {line_number}: cada registro debe ser un objeto JSON.")
            records.append(record)
    if not records:
        raise ValueError(f"No se encontraron registros en {path}.")
    return records


def select_records(
    records: list[dict[str, Any]],
    trace_id: str | None,
    index: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if trace_id is not None and index is not None:
        raise ValueError("Usa solo uno entre --trace-id y --index.")

    selected = records
    if trace_id is not None:
        selected = [record for record in records if record.get("id") == trace_id]
        if not selected:
            raise ValueError(f"No existe un registro con id={trace_id!r}.")
    elif index is not None:
        if index < 0 or index >= len(records):
            raise ValueError(f"--index debe estar entre 0 y {len(records) - 1}.")
        selected = [records[index]]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit debe ser mayor que 0.")
        selected = selected[:limit]

    if not selected:
        raise ValueError("La seleccion quedo vacia.")
    return selected


def call_model(
    client: LLMClient,
    model: str,
    trace_id: str,
    prompt: str,
    verbose: bool,
    action: str,
    reasoning_effort: str,
) -> ParsedSegmentationResponse:
    log(f"[{trace_id}] Enviando solicitud de {action} a {client.provider}:{model}.", verbose)
    response = request_structured_response(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=SegmentationDraft,
        reasoning_effort=reasoning_effort,
    )
    draft = response.parsed
    draft.trace_id = draft.trace_id or trace_id
    log(f"[{trace_id}] Respuesta de {action} recibida.", verbose)
    return ParsedSegmentationResponse(
        draft=draft,
        provider=response.provider,
        system_fingerprint=response.system_fingerprint,
        response_id=response.response_id,
        model=response.model,
        reasoning_effort=reasoning_effort,
    )


def _natural_boundary_positions(trace: str) -> list[int]:
    return [match.end() for match in _NATURAL_LITERAL_BOUNDARY.finditer(trace)]


def _last_whitespace_boundary(
    trace: str,
    start_char: int,
    max_end_char: int,
) -> int | None:
    boundary: int | None = None
    for match in _WHITESPACE_RUN.finditer(trace, start_char, max_end_char):
        if match.end() > start_char:
            boundary = match.end()
    return boundary


def _next_whitespace_boundary(
    trace: str,
    start_char: int,
) -> int | None:
    match = _WHITESPACE_RUN.search(trace, start_char)
    return match.end() if match is not None else None


def build_literal_units(
    trace: str,
    max_chars: int = DEFAULT_LITERAL_UNIT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Build a lossless, non-semantic transport partition of ``trace``.

    Natural punctuation and line boundaries keep the inventory readable.
    ``max_chars`` only limits unusually long pieces, using whitespace as the
    preferred mechanical split. No functional label or boundary is inferred.
    """

    if trace == "":
        raise ValueError("La traza no puede estar vacia.")
    if max_chars <= 0:
        raise ValueError("max_chars debe ser mayor que 0.")

    natural_boundaries = _natural_boundary_positions(trace)
    units: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(trace):
        remaining = trace[cursor:]
        if remaining.strip() == "" and units:
            units[-1]["end_char"] = len(trace)
            units[-1]["text"] = trace[
                units[-1]["start_char"]:units[-1]["end_char"]
            ]
            cursor = len(trace)
            break

        max_end = min(cursor + max_chars, len(trace))
        natural_end = next(
            (
                boundary
                for boundary in natural_boundaries
                if cursor < boundary <= max_end
                and trace[cursor:boundary].strip() != ""
            ),
            None,
        )

        if natural_end is not None:
            end_char = natural_end
        elif max_end == len(trace):
            end_char = len(trace)
        else:
            end_char = _last_whitespace_boundary(trace, cursor, max_end)
            if end_char is None or trace[cursor:end_char].strip() == "":
                end_char = _next_whitespace_boundary(trace, max_end)
            if end_char is None:
                end_char = len(trace)

        if end_char <= cursor:
            raise ValueError(
                "La unitizacion literal no pudo avanzar desde "
                f"start_char={cursor}."
            )

        units.append(
            {
                "unit_id": f"u{len(units) + 1}",
                "text": trace[cursor:end_char],
                "start_char": cursor,
                "end_char": end_char,
            }
        )
        cursor = end_char

    if not units:
        units.append(
            {
                "unit_id": "u1",
                "text": trace,
                "start_char": 0,
                "end_char": len(trace),
            }
        )

    reconstructed = "".join(unit["text"] for unit in units)
    if reconstructed != trace:
        raise ValueError(
            "La unitizacion literal no reconstruye exactamente la traza."
        )
    return units


def literal_units_for_prompt(
    literal_units: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "unit_id": unit["unit_id"],
            "text": unit["text"],
        }
        for unit in literal_units
    ]


def build_segment_objects(
    trace: str,
    literal_units: list[dict[str, Any]],
    segment_ranges: list[SegmentRangeDraft],
) -> list[dict[str, Any]]:
    if not literal_units:
        raise ValueError("No existen unidades literales.")
    if not segment_ranges:
        raise ValueError("No se devolvieron segmentos.")

    unit_index_by_id = {
        unit["unit_id"]: index for index, unit in enumerate(literal_units)
    }
    next_expected_index = 0
    built_segments: list[dict[str, Any]] = []

    for segment_index, segment_range in enumerate(segment_ranges, start=1):
        if next_expected_index >= len(literal_units):
            raise ValueError(
                f"Segmento {segment_index}: se devolvieron rangos despues de "
                "cubrir la ultima unidad literal."
            )
        start_index = unit_index_by_id.get(segment_range.start_unit_id)
        end_index = unit_index_by_id.get(segment_range.end_unit_id)
        if start_index is None:
            raise ValueError(
                f"Segmento {segment_index}: start_unit_id inexistente "
                f"{segment_range.start_unit_id!r}."
            )
        if end_index is None:
            raise ValueError(
                f"Segmento {segment_index}: end_unit_id inexistente "
                f"{segment_range.end_unit_id!r}."
            )
        if end_index < start_index:
            raise ValueError(
                f"Segmento {segment_index}: rango invertido "
                f"{segment_range.start_unit_id}->{segment_range.end_unit_id}."
            )
        if start_index != next_expected_index:
            expected_id = literal_units[next_expected_index]["unit_id"]
            raise ValueError(
                f"Segmento {segment_index}: se esperaba iniciar en "
                f"{expected_id!r}, no en {segment_range.start_unit_id!r}."
            )

        start_char = literal_units[start_index]["start_char"]
        end_char = literal_units[end_index]["end_char"]
        built_segments.append(
            {
                "segment_id": f"s{segment_index}",
                "text": trace[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        next_expected_index = end_index + 1

    if next_expected_index != len(literal_units):
        expected_id = literal_units[next_expected_index]["unit_id"]
        raise ValueError(
            "La cobertura final no incluye todas las unidades: "
            f"la siguiente unidad esperada es {expected_id!r}."
        )

    return built_segments


def validate_segment_ranges(
    trace: str,
    literal_units: list[dict[str, Any]],
    segment_ranges: list[SegmentRangeDraft],
) -> list[str]:
    try:
        build_segment_objects(trace, literal_units, segment_ranges)
        return []
    except ValueError as exc:
        return [str(exc)]


def sample_segmentation(
    client: LLMClient,
    model: str,
    trace_id: str,
    reasoning_trace: str,
    literal_units: list[dict[str, Any]],
    input_json: str,
    max_attempts: int,
    verbose: bool,
    reasoning_effort: str,
    sample_index: int,
    sample_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Obtiene una muestra válida de rangos de segmentación.

    Una salida que incumple el contrato se devuelve al LLM con sus errores
    hasta ``max_attempts`` veces. El código nunca corrige los rangos.
    """

    validation_errors: list[str] = []
    draft: SegmentationDraft | None = None
    for attempt in range(1, max_attempts + 1):
        log(
            f"[{trace_id}] Muestra {sample_index}: intento {attempt}/{max_attempts}.",
            verbose,
        )
        if attempt == 1:
            prompt = USER_PROMPT_TEMPLATE.format(input_json=input_json)
            response_meta = call_model(
                client,
                model,
                trace_id,
                prompt,
                verbose,
                "segmentacion",
                reasoning_effort,
            )
        else:
            prompt = REPAIR_PROMPT_TEMPLATE.format(
                validation_errors="\n".join(f"- {error}" for error in validation_errors),
                previous_output_json=json.dumps(
                    draft.model_dump() if draft is not None else {},
                    ensure_ascii=False,
                    indent=2,
                ),
                input_json=input_json,
            )
            response_meta = call_model(
                client,
                model,
                trace_id,
                prompt,
                verbose,
                "reparacion",
                reasoning_effort,
            )

        draft = response_meta.draft
        if draft.trace_id != trace_id:
            draft.trace_id = trace_id
        validation_errors = validate_segment_ranges(
            reasoning_trace,
            literal_units,
            draft.segments,
        )
        sample_events.append(
            {
                "sample": sample_index,
                "attempt": attempt,
                "validation_status": "valid" if not validation_errors else "invalid",
                "validation_errors": validation_errors,
                "segment_count": len(draft.segments),
            }
        )
        if not validation_errors:
            return {
                "draft": draft,
                "response_meta": response_meta,
                "prompt": prompt,
                "attempts": attempt,
            }
        log(f"[{trace_id}] Validacion fallida: {' | '.join(validation_errors)}", verbose)

    raise ValueError(
        f"No se pudo validar la segmentacion de {trace_id!r} (muestra {sample_index}) "
        f"tras {max_attempts} intentos. Errores: {' | '.join(validation_errors)}"
    )


def consensus_segment_ranges(
    literal_units: list[dict[str, Any]],
    drafts: list[SegmentationDraft],
) -> tuple[list[SegmentRangeDraft], dict[str, int], int]:
    """Rangos por voto de cortes.

    Cada muestra válida equivale a un conjunto de cortes: se corta tras la
    unidad final de cada segmento salvo el último. Se conserva un corte cuando
    una mayoría estricta de las muestras lo propone. Cualquier conjunto de
    cortes produce una segmentación válida, así que el código no decide
    fronteras: solo cuenta las propuestas por el LLM.
    """

    index_by_id = {unit["unit_id"]: index for index, unit in enumerate(literal_units)}
    cut_sets = [
        [index_by_id[segment_range.end_unit_id] for segment_range in draft.segments[:-1]]
        for draft in drafts
    ]
    vote = majority_set(cut_sets)
    cuts = sorted(vote["kept"])
    ranges: list[SegmentRangeDraft] = []
    start = 0
    for cut in [*cuts, len(literal_units) - 1]:
        ranges.append(
            SegmentRangeDraft(
                start_unit_id=literal_units[start]["unit_id"],
                end_unit_id=literal_units[cut]["unit_id"],
            )
        )
        start = cut + 1
    cut_support = {
        literal_units[index]["unit_id"]: support
        for index, support in sorted(vote["support"].items())
    }
    return ranges, cut_support, vote["threshold"]


def segment_trace(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    max_attempts: int,
    verbose: bool,
    reasoning_effort: str,
    samples: int = 1,
) -> dict[str, Any]:
    """Segmenta una traza por consenso de ``samples`` muestras del estimador.

    Con ``samples=1`` se conserva el estimador de una sola muestra. Con más
    muestras, cada corte sobrevive solo si lo propone una mayoría estricta.
    """

    trace_id = record.get("id")
    reasoning_trace = record.get("reasoning_trace")

    if not trace_id:
        raise ValueError("Cada registro debe tener el campo 'id'.")
    if not isinstance(reasoning_trace, str) or reasoning_trace == "":
        raise ValueError(f"El registro {trace_id!r} no tiene un 'reasoning_trace' valido.")

    log(
        f"[{trace_id}] Iniciando segmentacion. Longitud de traza: {len(reasoning_trace)} caracteres.",
        verbose,
    )

    literal_units = build_literal_units(reasoning_trace)
    input_json = json.dumps(
        {
            # Alias opaco: el prompt no recibe el identificador real, que
            # revela dataset y generador.
            "trace_id": prompt_trace_alias(trace_id),
            "literal_unitization": LITERAL_UNITIZATION_VERSION,
            "literal_units": literal_units_for_prompt(literal_units),
        },
        ensure_ascii=False,
        indent=2,
    )
    log(
        f"[{trace_id}] Inventario literal: {len(literal_units)} unidades.",
        verbose,
    )

    validate_sample_count(samples)
    sample_events: list[dict[str, Any]] = []
    sampled = [
        sample_segmentation(
            client,
            model,
            trace_id,
            reasoning_trace,
            literal_units,
            input_json,
            max_attempts,
            verbose,
            reasoning_effort,
            sample_index,
            sample_events,
        )
        for sample_index in range(1, samples + 1)
    ]
    drafts = [sample["draft"] for sample in sampled]
    cut_support: dict[str, int] | None = None
    cut_threshold = 1
    if samples == 1:
        segment_ranges = list(drafts[0].segments)
    else:
        segment_ranges, cut_support, cut_threshold = consensus_segment_ranges(
            literal_units,
            drafts,
        )
    built_segments = build_segment_objects(
        reasoning_trace,
        literal_units,
        segment_ranges,
    )
    primary = sampled[0]
    response_meta = primary["response_meta"]
    prompt_fingerprint = structured_request_hash(
        client,
        SYSTEM_PROMPT,
        primary["prompt"],
    )
    for segment in built_segments:
        segment["provenance"] = make_provenance(
            "llm_proposed",
            "segmentation",
            provider=response_meta.provider,
            model=response_meta.model or model,
            prompt_hash=prompt_fingerprint,
            response_id=response_meta.response_id,
            system_fingerprint=response_meta.system_fingerprint,
            reasoning_effort=response_meta.reasoning_effort
            or reasoning_effort,
        )
    log(
        f"[{trace_id}] Validacion estructural OK con {len(built_segments)} segmentos "
        f"({samples} muestra(s)).",
        verbose,
    )
    consensus_method = CONSENSUS_METHOD_VERSION if samples > 1 else None
    pipeline = ensure_pipeline_record(record)
    pipeline["trace_id"] = trace_id
    pipeline["trace_text"] = reasoning_trace
    pipeline["segments"] = built_segments
    pipeline["nodes"] = []
    pipeline["edges"] = []
    append_audit_event(
        pipeline,
        "segmentation",
        {
            "decision": "accepted",
            "attempt": primary["attempts"],
            "segment_count": len(built_segments),
            "literal_unitization": LITERAL_UNITIZATION_VERSION,
            "literal_unit_count": len(literal_units),
            "literal_unit_max_chars": DEFAULT_LITERAL_UNIT_MAX_CHARS,
            "samples": samples,
            "consensus_method": consensus_method,
            "attempts_by_sample": [sample["attempts"] for sample in sampled],
            "segment_count_by_sample": [len(draft.segments) for draft in drafts],
            "cut_support": cut_support,
            "cut_threshold": cut_threshold,
            "sample_events": sample_events,
        },
    )
    return set_validation_stage(
        pipeline,
        "segmentation",
        {
            "status": "valid",
            "attempts": primary["attempts"],
            "provider": response_meta.provider,
            "model": response_meta.model or model,
            "reasoning_effort": response_meta.reasoning_effort or reasoning_effort,
            "system_fingerprint": response_meta.system_fingerprint,
            "response_id": response_meta.response_id,
            "prompt_hash": prompt_fingerprint,
            "transport": LITERAL_UNITIZATION_VERSION,
            "literal_unit_count": len(literal_units),
            "samples": samples,
            "consensus_method": consensus_method,
        },
    )

def segment_trace_with_new_client(
    provider: str,
    api_key: str,
    base_url: str | None,
    model: str,
    record: dict[str, Any],
    max_attempts: int,
    verbose: bool,
    reasoning_effort: str,
    request_timeout_seconds: float | None = None,
    transport_max_retries: int | None = None,
    samples: int = 1,
) -> dict[str, Any]:
    client = build_llm_client(
        provider,
        api_key,
        base_url,
        request_timeout_seconds,
        transport_max_retries,
    )
    return segment_trace(
        client=client,
        model=model,
        record=record,
        max_attempts=max_attempts,
        verbose=verbose,
        reasoning_effort=reasoning_effort,
        samples=samples,
    )


def segment_records(
    records: list[dict[str, Any]],
    provider: str,
    api_key: str,
    base_url: str | None,
    model: str,
    max_attempts: int,
    verbose: bool,
    concurrency: int,
    reasoning_effort: str,
    request_timeout_seconds: float | None = None,
    transport_max_retries: int | None = None,
    samples: int = 1,
) -> list[dict[str, Any]]:
    if concurrency <= 0:
        raise ValueError("--concurrency debe ser mayor que 0.")

    if len(records) <= 1 or concurrency == 1:
        client = build_llm_client(
            provider,
            api_key,
            base_url,
            request_timeout_seconds,
            transport_max_retries,
        )
        return [
            segment_trace(
                client=client,
                model=model,
                record=record,
                max_attempts=max_attempts,
                verbose=verbose,
                reasoning_effort=reasoning_effort,
                samples=samples,
            )
            for record in records
        ]

    max_workers = min(concurrency, len(records))
    log(
        f"Ejecutando segmentacion concurrente con {max_workers} workers para {len(records)} trazas.",
        verbose,
    )

    ordered_results: list[dict[str, Any] | None] = [None] * len(records)
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                segment_trace_with_new_client,
                provider,
                api_key,
                base_url,
                model,
                record,
                max_attempts,
                verbose,
                reasoning_effort,
                request_timeout_seconds,
                transport_max_retries,
                samples,
            ): index
            for index, record in enumerate(records)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            trace_id = records[index].get("id", f"index_{index}")
            try:
                ordered_results[index] = future.result()
                log(f"[{trace_id}] Segmentacion completada.", verbose)
            except Exception as exc:
                failures.append(f"{trace_id}: {exc}")
                log(f"[{trace_id}] Error en segmentacion: {exc}", verbose)

    if failures:
        raise ValueError(
            "Fallaron una o mas trazas durante la segmentacion concurrente: "
            + " | ".join(failures)
        )

    return [result for result in ordered_results if result is not None]


def write_output(results: list[dict[str, Any]], output_path: Path | None, pretty: bool) -> None:
    single_output = len(results) == 1

    if output_path is None:
        if single_output or pretty:
            payload = results[0] if single_output else results
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return

        for result in results:
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix.lower() == ".json":
            payload = results[0] if single_output else results
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2 if pretty else None,
            )
            handle.write("\n")
            return

        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    configure_stdio()
    args = parse_args()

    try:
        log(f"Cargando archivo de entrada: {args.input_path}", args.verbose)
        records = load_jsonl(args.input_path)
        log(f"Registros leidos: {len(records)}", args.verbose)
        selected = select_records(records, args.trace_id, args.index, args.limit)
        log(f"Registros seleccionados: {len(selected)}", args.verbose)
        log(
            (
                f"Segmentacion lista para ejecutarse con {args.provider}:{args.model}, "
                f"reasoning_effort {args.reasoning_effort}, "
                f"concurrencia {args.concurrency}."
            ),
            args.verbose,
        )
        results = segment_records(
            records=selected,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_attempts=args.max_attempts,
            verbose=args.verbose,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            samples=args.samples,
            request_timeout_seconds=args.request_timeout_seconds,
            transport_max_retries=args.transport_max_retries,
        )
        log("Escribiendo salida.", args.verbose)
        write_output(results, args.output, args.pretty)
        log("Proceso completado.", args.verbose)
        return 0
    except (ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
