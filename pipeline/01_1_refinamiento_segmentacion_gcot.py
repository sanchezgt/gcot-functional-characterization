from __future__ import annotations

import argparse
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
from util.llm_checkpoint import (
    ItemCheckpoint,
    canonical_json_hash,
    checkpoint_path_for_trace,
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
from util.stage_cache import implementation_tree_hash


DEFAULT_REFINEMENT_MODEL = DEFAULT_OPENAI_MODEL
REFINEMENT_LITERAL_UNITIZATION_VERSION = "refinement-literal-units-v1"


_WHITESPACE_RUN = re.compile(r"\s+")


SYSTEM_PROMPT = """You refine already valid reasoning-trace segments.

You are not segmenting the whole trace again. You only inspect one existing segment and decide whether it should remain as one unit or be split into smaller contiguous subsegments.

Decision rules:
1. Return KEEP if the segment already expresses one dominant reasoning function.
2. Return SPLIT only if the segment clearly mixes two or more dominant reasoning functions.
3. Split especially when the segment moves from:
   - factual or problem-state description -> explicit action step
   - explicit action step -> explicit result
   - doubt, suspicion, or mismatch -> act of checking
   - checking -> correction or replacement
   - alternative proposal -> evaluation, discard, or return
   - explicit support -> later procedural decision
4. The segment is supplied as immutable literal units with local stable
   identifiers.
5. Never reproduce, paraphrase, normalize, or correct unit text.
6. For KEEP, return one inclusive range covering every supplied unit.
7. For SPLIT, return two or more ordered inclusive ranges that partition every
   supplied unit exactly once.
8. Return only valid JSON.
"""


USER_PROMPT_TEMPLATE = """Inspect the following segment and decide whether it should be kept as one unit or split.

Return a JSON object with:
- segment_id
- decision: KEEP or SPLIT
- subsegment_ranges: array of objects with start_unit_id and end_unit_id

Rules:
- KEEP if the segment has one dominant reasoning function.
- SPLIT only if it clearly mixes multiple dominant reasoning functions.
- Never copy literal text into the output.
- Each range is inclusive and contains all units between its two identifiers.
- Ranges must be ordered, contiguous, non-overlapping, and cover every supplied
  unit exactly once.
- KEEP requires one range from the first through the final unit.
- SPLIT requires at least two ranges.

Input JSON:
{input_json}
"""


REPAIR_PROMPT_TEMPLATE = """The previous refinement did not satisfy the
deterministic literal contract. Return a corrected JSON object and preserve the
semantic decision unless the previous decision itself violated the rules.

Validation errors:
{validation_errors}

Previous output:
{previous_output_json}

Input JSON:
{input_json}
"""


class SubsegmentRangeDraft(BaseModel):
    start_unit_id: str
    end_unit_id: str


class RefinementDraft(BaseModel):
    segment_id: Optional[str] = None
    decision: str
    subsegment_ranges: list[SubsegmentRangeDraft]


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
        description="Refina segmentos candidatos a estar fusionados tras la segmentacion base."
    )
    parser.add_argument("input_path", type=Path, help="Ruta al archivo JSON o JSONL segmentado.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Ruta de salida. Si no se indica, imprime en stdout.",
    )
    parser.add_argument("--trace-id", help="Procesa solo el registro con este id.")
    parser.add_argument("--index", type=int, help="Procesa solo el registro en esta posicion 0-based.")
    parser.add_argument("--limit", type=int, help="Procesa solo los primeros N registros seleccionados.")
    add_llm_provider_args(parser, DEFAULT_REFINEMENT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_REFINEMENT_REASONING_EFFORT", "low"),
        choices=["none", "minimal", "low", "medium", "high"],
        help=(
            "Reasoning effort del refinador. "
            "Default: OPENAI_REFINEMENT_REASONING_EFFORT o low."
        ),
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=35,
        help=(
            "Compatibilidad con corridas anteriores; ya no filtra segmentos. "
            "Todos los segmentos se revisan semánticamente."
        ),
    )
    parser.add_argument(
        "--max-subsegments",
        type=int,
        default=3,
        help=(
            "Compatibilidad con corridas anteriores; ya no limita una división "
            "semánticamente justificada."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Intentos máximos por muestra y segmento ante una salida inválida. Default: 3.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por segmento; un corte sobrevive solo "
            "con mayoría estricta. 1 reproduce el estimador de una sola "
            f"muestra. Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Directorio opcional para persistir decisiones aceptadas por "
            "segmento."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        action="store_true",
        help=(
            "Reutiliza decisiones del checkpoint solo si su identidad y "
            "validación coinciden exactamente."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Imprime JSON con indentacion.")
    parser.add_argument("--verbose", action="store_true", help="Muestra progreso y diagnostico en stderr.")
    return finalize_llm_args(parser.parse_args(), DEFAULT_REFINEMENT_MODEL)


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No se encontraron registros en {path}.")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("El archivo JSON debe contener un objeto o una lista.")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Linea {line_number}: JSON invalido: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Linea {line_number}: cada registro debe ser un objeto JSON.")
        records.append(row)
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
        selected = [record for record in records if record.get("trace_id") == trace_id]
        if not selected:
            raise ValueError(f"No existe un registro con trace_id={trace_id!r}.")
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


def derive_trace_text(pipeline: dict[str, Any]) -> str:
    trace_text = pipeline.get("trace_text")
    if isinstance(trace_text, str) and trace_text != "":
        return trace_text

    segments = pipeline.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("No se pudo derivar trace_text porque no hay segmentos.")

    concatenated = "".join(segment.get("text", "") for segment in segments)
    if not concatenated:
        raise ValueError("No se pudo derivar trace_text a partir de segmentos vacios.")
    pipeline["trace_text"] = concatenated
    return concatenated


def is_candidate_segment(segment: dict[str, Any], min_length: int) -> bool:
    """Todos los segmentos válidos se someten a la decisión del estimador.

    ``min_length`` se conserva en la firma para que las llamadas históricas no
    fallen, pero no participa en la selección.
    """

    text = segment.get("text", "")
    return isinstance(text, str) and text != ""


def build_refinement_literal_units(text: str) -> list[dict[str, Any]]:
    """Particiona un segmento en unidades literales locales y sin semántica.

    Cada corrida de espacio en blanco ofrece un límite mecánico. Los espacios
    permanecen unidos al contenido precedente; cualquier espacio inicial se
    conserva en la primera unidad con contenido. El inventario solo transporta
    texto: no propone una división funcional.
    """

    if text == "":
        raise ValueError("El segmento no puede estar vacío.")

    units: list[dict[str, Any]] = []
    cursor = 0
    for match in _WHITESPACE_RUN.finditer(text):
        end_char = match.end()
        if text[cursor:end_char].strip() == "":
            continue
        units.append(
            {
                "unit_id": f"u{len(units) + 1}",
                "text": text[cursor:end_char],
                "start_char": cursor,
                "end_char": end_char,
            }
        )
        cursor = end_char

    if cursor < len(text):
        units.append(
            {
                "unit_id": f"u{len(units) + 1}",
                "text": text[cursor:],
                "start_char": cursor,
                "end_char": len(text),
            }
        )
    if not units:
        units.append(
            {
                "unit_id": "u1",
                "text": text,
                "start_char": 0,
                "end_char": len(text),
            }
        )

    if "".join(unit["text"] for unit in units) != text:
        raise ValueError(
            "La unitización literal del refinamiento no reconstruye el "
            "segmento."
        )
    for left, right in zip(units, units[1:]):
        if left["end_char"] != right["start_char"]:
            raise ValueError(
                "Las unidades literales del refinamiento no son contiguas."
            )
    return units


def literal_units_for_prompt(
    literal_units: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"unit_id": unit["unit_id"], "text": unit["text"]}
        for unit in literal_units
    ]


def build_refined_texts(
    original_text: str,
    literal_units: list[dict[str, Any]],
    ranges: list[SubsegmentRangeDraft],
) -> list[str]:
    if not literal_units:
        raise ValueError("No existen unidades literales.")
    if not ranges:
        raise ValueError("subsegment_ranges no puede estar vacío.")

    unit_index_by_id = {
        unit["unit_id"]: index for index, unit in enumerate(literal_units)
    }
    next_expected_index = 0
    texts: list[str] = []

    for range_index, subsegment_range in enumerate(ranges, start=1):
        if next_expected_index >= len(literal_units):
            raise ValueError(
                f"Rango {range_index}: hay rangos después de cubrir la "
                "última unidad."
            )
        start_index = unit_index_by_id.get(subsegment_range.start_unit_id)
        end_index = unit_index_by_id.get(subsegment_range.end_unit_id)
        if start_index is None:
            raise ValueError(
                f"Rango {range_index}: start_unit_id inexistente "
                f"{subsegment_range.start_unit_id!r}."
            )
        if end_index is None:
            raise ValueError(
                f"Rango {range_index}: end_unit_id inexistente "
                f"{subsegment_range.end_unit_id!r}."
            )
        if end_index < start_index:
            raise ValueError(
                f"Rango {range_index}: rango invertido "
                f"{subsegment_range.start_unit_id}->"
                f"{subsegment_range.end_unit_id}."
            )
        if start_index != next_expected_index:
            expected_id = literal_units[next_expected_index]["unit_id"]
            raise ValueError(
                f"Rango {range_index}: se esperaba iniciar en "
                f"{expected_id!r}, no en "
                f"{subsegment_range.start_unit_id!r}."
            )

        start_char = literal_units[start_index]["start_char"]
        end_char = literal_units[end_index]["end_char"]
        texts.append(original_text[start_char:end_char])
        next_expected_index = end_index + 1

    if next_expected_index != len(literal_units):
        expected_id = literal_units[next_expected_index]["unit_id"]
        raise ValueError(
            "La cobertura final no incluye todas las unidades; la siguiente "
            f"unidad esperada es {expected_id!r}."
        )
    if "".join(texts) != original_text:
        raise ValueError(
            "Los rangos no reconstruyen exactamente el segmento original."
        )
    return texts


def build_prompt(trace_id: str, segment: dict[str, Any]) -> str:
    literal_units = build_refinement_literal_units(segment["text"])
    input_json = json.dumps(
        {
            "trace_id": prompt_trace_alias(trace_id),
            "segment_id": segment["segment_id"],
            "literal_unitization": REFINEMENT_LITERAL_UNITIZATION_VERSION,
            "literal_units": literal_units_for_prompt(literal_units),
        },
        ensure_ascii=False,
        indent=2,
    )
    return USER_PROMPT_TEMPLATE.format(
        input_json=input_json,
    )


def build_repair_prompt(
    trace_id: str,
    segment: dict[str, Any],
    draft: RefinementDraft,
    validation_errors: list[str],
) -> str:
    literal_units = build_refinement_literal_units(segment["text"])
    input_json = json.dumps(
        {
            "trace_id": prompt_trace_alias(trace_id),
            "segment_id": segment["segment_id"],
            "literal_unitization": REFINEMENT_LITERAL_UNITIZATION_VERSION,
            "literal_units": literal_units_for_prompt(literal_units),
        },
        ensure_ascii=False,
        indent=2,
    )
    return REPAIR_PROMPT_TEMPLATE.format(
        validation_errors="\n".join(
            f"- {error}" for error in validation_errors
        ),
        previous_output_json=json.dumps(
            draft.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
        input_json=input_json,
    )


def call_model(
    client: LLMClient,
    model: str,
    trace_id: str,
    segment: dict[str, Any],
    reasoning_effort: str,
    verbose: bool,
    *,
    repair_draft: RefinementDraft | None = None,
    validation_errors: list[str] | None = None,
) -> RefinementDraft:
    prompt = (
        build_repair_prompt(
            trace_id,
            segment,
            repair_draft,
            validation_errors or [],
        )
        if repair_draft is not None
        else build_prompt(trace_id, segment)
    )
    log(
        f"[{trace_id}/{segment['segment_id']}] Enviando solicitud de refinamiento a {client.provider}:{model}.",
        verbose,
    )
    response = request_structured_response(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=RefinementDraft,
        reasoning_effort=reasoning_effort,
    )
    parsed = response.parsed
    if not parsed.segment_id:
        parsed.segment_id = segment["segment_id"]
    log(
        f"[{trace_id}/{segment['segment_id']}] Respuesta de refinamiento recibida.",
        verbose,
    )
    return parsed


def validate_refinement(
    segment: dict[str, Any],
    draft: RefinementDraft,
    max_subsegments: int,
) -> list[str]:
    errors: list[str] = []
    original_text = segment["text"]
    literal_units = build_refinement_literal_units(original_text)

    if draft.segment_id != segment["segment_id"]:
        errors.append("segment_id no coincide.")

    if draft.decision not in {"KEEP", "SPLIT"}:
        errors.append("decision debe ser KEEP o SPLIT.")

    if not draft.subsegment_ranges:
        errors.append("subsegment_ranges no puede estar vacío.")
        return errors

    try:
        build_refined_texts(
            original_text,
            literal_units,
            draft.subsegment_ranges,
        )
    except ValueError as exc:
        errors.append(str(exc))

    if draft.decision == "KEEP":
        if len(draft.subsegment_ranges) != 1:
            errors.append("KEEP debe devolver exactamente un rango.")
        elif literal_units:
            only_range = draft.subsegment_ranges[0]
            if (
                only_range.start_unit_id != literal_units[0]["unit_id"]
                or only_range.end_unit_id != literal_units[-1]["unit_id"]
            ):
                errors.append(
                    "KEEP debe cubrir desde la primera hasta la última "
                    "unidad literal."
                )

    if draft.decision == "SPLIT":
        if len(draft.subsegment_ranges) < 2:
            errors.append("SPLIT debe devolver al menos dos rangos.")

    return errors


def rebuild_segments(trace_text: str, segment_texts: list[str]) -> list[dict[str, Any]]:
    cursor = 0
    rebuilt: list[dict[str, Any]] = []

    for index, text in enumerate(segment_texts, start=1):
        end_char = cursor + len(text)
        original = trace_text[cursor:end_char]
        if original != text:
            raise ValueError(
                f"No se pudo reconstruir offsets globales en el segmento {index}: "
                "el texto no coincide exactamente con trace_text."
            )
        rebuilt.append(
            {
                "segment_id": f"s{index}",
                "text": text,
                "start_char": cursor,
                "end_char": end_char,
            }
        )
        cursor = end_char

    if cursor != len(trace_text):
        raise ValueError(
            f"La reconstruccion global no cubre todo trace_text: {cursor} vs {len(trace_text)}."
        )
    return rebuilt


def _validate_cached_refinement_sample(
    cached: Any,
    segment: dict[str, Any],
    max_subsegments: int,
) -> dict[str, Any]:
    segment_id = segment["segment_id"]
    draft = RefinementDraft.model_validate(cached["draft"])
    prompt_hash = cached["prompt_hash"]
    attempts = cached["attempts"]
    request_events = cached["request_events"]
    if not isinstance(prompt_hash, str) or not prompt_hash.startswith(
        "sha256:"
    ):
        raise ValueError("prompt_hash inválido.")
    if not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("attempts inválido.")
    if (
        not isinstance(request_events, list)
        or len(request_events) != attempts
        or not all(isinstance(event, dict) for event in request_events)
    ):
        raise ValueError("request_events inválido.")
    final_event = request_events[-1]
    if (
        final_event.get("segment_id") != segment_id
        or final_event.get("attempt") != attempts
        or final_event.get("prompt_hash") != prompt_hash
    ):
        raise ValueError("El evento final no coincide con la decisión.")
    validation_errors = validate_refinement(
        segment,
        draft,
        max_subsegments,
    )
    if validation_errors:
        raise ValueError(" | ".join(validation_errors))
    return {
        "draft": draft,
        "prompt_hash": prompt_hash,
        "attempts": attempts,
        "request_events": request_events,
    }


def load_cached_refinement(
    checkpoint: ItemCheckpoint,
    segment: dict[str, Any],
    max_subsegments: int,
    samples: int = 1,
) -> list[dict[str, Any]] | None:
    """Recupera las muestras aceptadas de un segmento.

    Solo se reutiliza un ítem con exactamente ``samples`` muestras que
    revalidan contra el contrato actual; cualquier discrepancia lo descarta.
    """

    segment_id = segment["segment_id"]
    cached = checkpoint.get(segment_id)
    if cached is None:
        return None
    try:
        if cached["consensus_samples"] != samples:
            raise ValueError("El número de muestras del checkpoint difiere.")
        stored = cached["samples"]
        if not isinstance(stored, list) or len(stored) != samples:
            raise ValueError("samples inválido.")
        results = [
            _validate_cached_refinement_sample(sample, segment, max_subsegments)
            for sample in stored
        ]
    except (KeyError, TypeError, ValueError, ValidationError):
        checkpoint.discard(segment_id)
        return None
    checkpoint.mark_reused(segment_id)
    return results


def sample_refinement(
    client: LLMClient,
    model: str,
    trace_id: str,
    segment: dict[str, Any],
    reasoning_effort: str,
    verbose: bool,
    max_attempts: int,
    max_subsegments: int,
    sample_index: int,
) -> dict[str, Any]:
    """Obtiene una muestra válida de la decisión KEEP/SPLIT de un segmento.

    Una salida inválida se devuelve al LLM con sus errores hasta
    ``max_attempts`` veces. El código nunca corrige los rangos.
    """

    draft: RefinementDraft | None = None
    validation_errors: list[str] = []
    request_events: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        request_prompt = (
            build_repair_prompt(
                trace_id,
                segment,
                draft,
                validation_errors,
            )
            if draft is not None
            else build_prompt(trace_id, segment)
        )
        prompt_hash = structured_request_hash(
            client,
            SYSTEM_PROMPT,
            request_prompt,
        )
        request_events.append(
            {
                "segment_id": segment["segment_id"],
                "sample": sample_index,
                "attempt": attempt,
                "prompt_hash": prompt_hash,
            }
        )
        if draft is None:
            draft = call_model(
                client,
                model,
                trace_id,
                segment,
                reasoning_effort,
                verbose,
            )
        else:
            draft = call_model(
                client,
                model,
                trace_id,
                segment,
                reasoning_effort,
                verbose,
                repair_draft=draft,
                validation_errors=validation_errors,
            )
        validation_errors = validate_refinement(
            segment,
            draft,
            max_subsegments,
        )
        if not validation_errors:
            return {
                "draft": draft,
                "prompt_hash": prompt_hash,
                "attempts": attempt,
                "request_events": request_events,
            }
        log(
            (
                f"[{trace_id}/{segment['segment_id']}] Refinamiento invalido "
                f"(muestra {sample_index}) en intento {attempt}/{max_attempts}. "
                f"decision={draft.decision} "
                f"subsegment_ranges="
                f"{draft.subsegment_ranges!r}"
            ),
            verbose,
        )
    raise ValueError(
        f"No se pudo validar el refinamiento de "
        f"{trace_id}/{segment['segment_id']} (muestra {sample_index}) tras "
        f"{max_attempts} intentos: {' | '.join(validation_errors)}"
    )


def consensus_subsegment_ranges(
    literal_units: list[dict[str, Any]],
    drafts: list[RefinementDraft],
) -> tuple[list[SubsegmentRangeDraft], dict[str, int], int]:
    """Rangos por voto de cortes dentro de un segmento.

    Un corte tras una unidad sobrevive solo si lo propone una mayoría estricta
    de las muestras; sin cortes mayoritarios el segmento se conserva entero.
    El código cuenta propuestas del LLM y no introduce fronteras propias.
    """

    index_by_id = {unit["unit_id"]: index for index, unit in enumerate(literal_units)}
    cut_sets = [
        [index_by_id[item.end_unit_id] for item in draft.subsegment_ranges[:-1]]
        for draft in drafts
    ]
    vote = majority_set(cut_sets)
    cuts = sorted(vote["kept"])
    ranges: list[SubsegmentRangeDraft] = []
    start = 0
    for cut in [*cuts, len(literal_units) - 1]:
        ranges.append(
            SubsegmentRangeDraft(
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

def refine_record(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    reasoning_effort: str,
    min_length: int,
    max_subsegments: int,
    verbose: bool,
    max_attempts: int = 3,
    checkpoint_dir: Path | None = None,
    resume_checkpoint: bool = False,
    samples: int = 1,
) -> dict[str, Any]:
    """Refina cada segmento por consenso de ``samples`` muestras KEEP/SPLIT.

    Con ``samples=1`` se conserva el estimador de una sola muestra.
    """

    pipeline = ensure_pipeline_record(record)
    _ = (min_length, max_subsegments)
    trace_id = pipeline.get("trace_id")
    if not trace_id:
        raise ValueError("Cada registro debe tener trace_id.")

    trace_text = derive_trace_text(pipeline)
    segments = pipeline.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"El registro {trace_id!r} no tiene segmentos validos.")
    if max_attempts <= 0:
        raise ValueError("--max-attempts debe ser mayor que 0.")
    validate_sample_count(samples)

    checkpoint = ItemCheckpoint(
        checkpoint_path_for_trace(
            checkpoint_dir,
            "01_1_refinamiento",
            trace_id,
        ),
        identity={
            "stage": "01_1_refinamiento",
            "implementation_hash": implementation_tree_hash(
                Path(__file__).resolve().parent
            ),
            "input_hash": canonical_json_hash(
                {
                    "trace_id": trace_id,
                    "trace_text": trace_text,
                    "segments": segments,
                }
            ),
            "provider": client.provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "consensus_samples": samples,
            "transport": REFINEMENT_LITERAL_UNITIZATION_VERSION,
        },
        resume=resume_checkpoint,
    )

    candidate_ids = [
        segment["segment_id"]
        for segment in segments
        if is_candidate_segment(segment, min_length)
    ]
    log(
        f"[{trace_id}] Refinamiento: {len(candidate_ids)} segmentos candidatos de {len(segments)}.",
        verbose,
    )

    refined_texts: list[str] = []
    reviewed_segments = 0
    kept_segments = 0
    split_segments = 0
    changed_segment_ids: list[str] = []
    request_hashes: dict[str, str] = {}
    request_events: list[dict[str, Any]] = []
    attempts_by_segment: dict[str, int] = {}
    literal_unit_counts_by_segment: dict[str, int] = {}
    samples_by_segment: dict[str, int] = {}
    decisions_by_sample_by_segment: dict[str, list[str]] = {}
    cut_support_by_segment: dict[str, dict[str, Any]] = {}

    for segment in segments:
        if segment["segment_id"] not in candidate_ids:
            refined_texts.append(segment["text"])
            continue

        reviewed_segments += 1
        literal_units = build_refinement_literal_units(segment["text"])
        literal_unit_counts_by_segment[segment["segment_id"]] = len(
            literal_units
        )
        segment_id = segment["segment_id"]
        sample_results = load_cached_refinement(
            checkpoint,
            segment,
            max_subsegments,
            samples,
        )
        if sample_results is not None:
            log(
                f"[{trace_id}/{segment_id}] Refinamiento recuperado del "
                f"checkpoint ({len(sample_results)} muestra(s)).",
                verbose,
            )
        else:
            sample_results = [
                sample_refinement(
                    client,
                    model,
                    trace_id,
                    segment,
                    reasoning_effort,
                    verbose,
                    max_attempts,
                    max_subsegments,
                    sample_index,
                )
                for sample_index in range(1, samples + 1)
            ]
            checkpoint.put(
                segment_id,
                {
                    "consensus_samples": samples,
                    "samples": [
                        {
                            "draft": result["draft"].model_dump(),
                            "prompt_hash": result["prompt_hash"],
                            "attempts": result["attempts"],
                            "request_events": result["request_events"],
                        }
                        for result in sample_results
                    ],
                },
            )
        for result in sample_results:
            request_events.extend(result["request_events"])
        attempts_by_segment[segment_id] = sum(
            result["attempts"] for result in sample_results
        )
        samples_by_segment[segment_id] = len(sample_results)
        decisions_by_sample_by_segment[segment_id] = [
            result["draft"].decision for result in sample_results
        ]
        if len(sample_results) == 1:
            request_hashes[segment_id] = sample_results[0]["prompt_hash"]
            ranges = list(sample_results[0]["draft"].subsegment_ranges)
        else:
            request_hashes[segment_id] = structured_request_hash(
                client,
                SYSTEM_PROMPT,
                build_prompt(trace_id, segment),
            )
            ranges, cut_support, cut_threshold = consensus_subsegment_ranges(
                literal_units,
                [result["draft"] for result in sample_results],
            )
            cut_support_by_segment[segment_id] = {
                "support": cut_support,
                "threshold": cut_threshold,
            }

        if len(ranges) == 1:
            kept_segments += 1
        else:
            split_segments += 1
            changed_segment_ids.append(segment_id)

        refined_texts.extend(
            build_refined_texts(
                segment["text"],
                literal_units,
                ranges,
            )
        )

    rebuilt_segments = rebuild_segments(trace_text, refined_texts)
    old_by_span = {
        (segment["start_char"], segment["end_char"], segment["text"]): segment
        for segment in segments
    }
    for segment in rebuilt_segments:
        previous = old_by_span.get(
            (segment["start_char"], segment["end_char"], segment["text"])
        )
        if previous is not None and "provenance" in previous:
            segment["provenance"] = previous["provenance"]
        else:
            source_segment = next(
                item
                for item in segments
                if item["start_char"] <= segment["start_char"]
                and segment["end_char"] <= item["end_char"]
            )
            segment["provenance"] = make_provenance(
                "llm_proposed",
                "segmentation_refinement",
                provider=client.provider,
                model=model,
                prompt_hash=request_hashes[source_segment["segment_id"]],
                reasoning_effort=reasoning_effort,
            )
    pipeline["segments"] = rebuilt_segments
    append_audit_event(
        pipeline,
        "segmentation_refinement",
        {
            "decision": "accepted",
            "selection_policy": "all_segments",
            "transport": REFINEMENT_LITERAL_UNITIZATION_VERSION,
            "reviewed_segment_ids": candidate_ids,
            "changed_segment_ids": changed_segment_ids,
            "literal_unit_counts_by_segment": (
                literal_unit_counts_by_segment
            ),
            "requests": request_events,
            "samples": samples,
            "consensus_method": CONSENSUS_METHOD_VERSION if samples > 1 else None,
            "samples_by_segment": samples_by_segment,
            "decisions_by_sample_by_segment": decisions_by_sample_by_segment,
            "cut_support_by_segment": cut_support_by_segment,
            "checkpoint": {
                "enabled": checkpoint.path is not None,
                "load_status": checkpoint.load_status,
                "reused_segment_ids": checkpoint.reused_item_ids,
            },
        },
    )

    return set_validation_stage(
        pipeline,
        "segmentation_refinement",
        {
            "status": "valid",
            "provider": client.provider,
            "reviewed_segments": reviewed_segments,
            "split_segments": split_segments,
            "kept_segments": kept_segments,
            "changed_segment_ids": changed_segment_ids,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "transport": REFINEMENT_LITERAL_UNITIZATION_VERSION,
            "literal_unit_counts_by_segment": (
                literal_unit_counts_by_segment
            ),
            "attempts_by_segment": attempts_by_segment,
            "samples": samples,
            "consensus_method": CONSENSUS_METHOD_VERSION if samples > 1 else None,
            "checkpoint": {
                "enabled": checkpoint.path is not None,
                "load_status": checkpoint.load_status,
                "reused_segment_ids": checkpoint.reused_item_ids,
            },
            "prompt_hashes": {
                segment_id: request_hashes[segment_id]
                for segment_id in candidate_ids
            },
        },
    )


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
        if args.resume_checkpoint and args.checkpoint_dir is None:
            raise ValueError(
                "--resume-checkpoint requiere --checkpoint-dir."
            )
        log(f"Cargando archivo de entrada: {args.input_path}", args.verbose)
        records = load_records(args.input_path)
        log(f"Registros leidos: {len(records)}", args.verbose)
        selected = select_records(records, args.trace_id, args.index, args.limit)
        log(f"Registros seleccionados: {len(selected)}", args.verbose)
        client = build_llm_client(
            args.provider,
            args.api_key,
            args.base_url,
            args.request_timeout_seconds,
            args.transport_max_retries,
        )
        log(
            f"Cliente {args.provider} inicializado con modelo {args.model} y reasoning_effort {args.reasoning_effort}.",
            args.verbose,
        )
        results = [
            refine_record(
                client=client,
                model=args.model,
                record=record,
                reasoning_effort=args.reasoning_effort,
                min_length=args.min_length,
                max_subsegments=args.max_subsegments,
                verbose=args.verbose,
                max_attempts=args.max_attempts,
                checkpoint_dir=args.checkpoint_dir,
                resume_checkpoint=args.resume_checkpoint,
                samples=args.samples,
            )
            for record in selected
        ]
        log("Escribiendo salida.", args.verbose)
        write_output(results, args.output, args.pretty)
        log("Proceso completado.", args.verbose)
        return 0
    except (ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
