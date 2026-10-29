from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError
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
    AMBIGUITY_TIE,
    AMBIGUITY_UNSTABLE,
    CONSENSUS_METHOD_VERSION,
    DEFAULT_CONSENSUS_SAMPLES,
    DEFAULT_MAX_EXTRA_SAMPLES,
    consensus_record,
    has_tie,
    majority_label,
    validate_sample_count,
)
from util.pipeline_canonical import ensure_pipeline_record, set_validation_stage
from util.sgotgraph_schema import append_audit_event, make_provenance


DEFAULT_TYPING_MODEL = DEFAULT_OPENAI_MODEL
DEFAULT_TYPING_MAX_ATTEMPTS = 3


SYSTEM_PROMPT = """You are an expert annotator for structured reasoning traces.

Your task is to classify already segmented reasoning-trace units into the SGoTGraph node ontology.

You are not segmenting. The segments are already fixed and must be preserved exactly.

Node ontology. Each segment receives exactly one dominant functional type:
- PROP: datum, claim, expressed inference, or intermediate result.
  Exclusion: it is not mainly a procedure (METHOD), a justification of other
  content (EVID), an evaluation (CHECK), a correction (REVISE), a competing
  route (ALT), or the delivery of the final answer (OUT).
- METHOD: plan, rule, operation, or procedure that indicates how to proceed.
  Exclusion: it is not the result produced by applying the procedure (that
  result is PROP).
- EVID: content whose dominant function is to explicitly justify other
  content. Exclusion: mere topical compatibility with other content does not
  make a segment evidence.
- ALT: route, interpretation, or hypothesis that competes with other content.
  Exclusion: an explicit replacement of an error is REVISE, not ALT.
- CHECK: act of checking, contrasting, or evaluating previous content.
  Exclusion: the correction that results from a check is REVISE.
- REVISE: explicit correction, replacement, retraction, or adjustment of
  previous content. Exclusion: doubt without a replacement is CHECK.
- OUT: explicit delivery of the final answer or conclusion of the trace.
  Exclusion: a partial result is PROP.

Boundary guidance, consistent with the definitions above:
- METHOD vs PROP: METHOD specifies what operation, transformation, or
  reasoning step should be carried out; PROP states information, an
  intermediate result, or a claim, even when it is mathematically relevant.
- CHECK vs PROP: CHECK performs or explicitly initiates an evaluation of
  earlier content; PROP merely states a fact, observation, or claim.
- EVID vs PROP: EVID is offered as a reason, ground, or support for other
  identifiable content; PROP asserts content on its own.
- OUT vs PROP: OUT is the explicit final delivery of the answer or
  conclusion; a result that is partial or not delivered as final is PROP.
- REVISE is used only when the segment mainly corrects, replaces, retracts,
  or adjusts earlier content. Do not force REVISE on segments that mostly
  introduce new information or support.

Rules:
1. Preserve every segment exactly as given.
2. Return one node classification for each segment_id.
3. Use only the ontology values listed above.
4. Determine the dominant function from the complete trace context and the
   operational definitions; no category has a general priority over another.
5. If a textually indivisible segment genuinely preserves more than one
   plausible dominant function, choose the best-supported type and record the
   ambiguity with one or more of these codes:
   - AMBIGUOUS_DOMINANT_FUNCTION: two or more types are comparably supported.
   - INSEPARABLE_MIXED_FUNCTION: the segment mixes functions that cannot be
     separated textually.
   - INSUFFICIENT_CONTEXT: the trace does not give enough context to decide.
   Otherwise return an empty list of ambiguity codes.
6. Do not use task correctness, external answers, keywords, or language-specific
   expressions as classification rules.
7. Return only valid JSON.
"""


USER_PROMPT_TEMPLATE = """Classify the following already segmented reasoning trace units into the SGoTGraph node ontology.

Return a JSON object with:
- nodes: array of objects

Each node object must contain only:
- segment_id
- node_type
- ambiguity_codes

Important:
- Do not modify the segments.
- Use the given segment ids exactly.
- Classify each segment independently but using the full trace context.

Segments:
{segments_json}
"""


VALID_NODE_TYPES = {"PROP", "METHOD", "EVID", "ALT", "CHECK", "REVISE", "OUT"}
VALID_AMBIGUITY_CODES = {
    "AMBIGUOUS_DOMINANT_FUNCTION",
    "INSEPARABLE_MIXED_FUNCTION",
    "INSUFFICIENT_CONTEXT",
}


class NodeDraft(BaseModel):
    segment_id: str
    node_type: str
    ambiguity_codes: list[str] = Field(default_factory=list)


class TypingDraft(BaseModel):
    trace_id: Optional[str] = None
    nodes: list[NodeDraft]


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
        description="Tipa segmentos de gCoT segun la ontologia de SGoTGraph."
    )
    parser.add_argument("input_path", type=Path, help="Ruta a la segmentacion JSON o JSONL.")
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
        help="Procesa solo el registro en esta posicion 0-based.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Procesa solo los primeros N registros seleccionados.",
    )
    add_llm_provider_args(parser, DEFAULT_TYPING_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("OPENAI_TYPING_REASONING_EFFORT", "low"),
        choices=["none", "minimal", "low", "medium", "high"],
        help="Esfuerzo de razonamiento del estimador de tipos. Default: low.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CONSENSUS_SAMPLES,
        help=(
            "Muestras LLM independientes por traza; cada nodo conserva el "
            "tipo mayoritario. 1 reproduce el estimador de una sola muestra. "
            f"Default: {DEFAULT_CONSENSUS_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_TYPING_MAX_ATTEMPTS,
        help=(
            "Intentos máximos por muestra ante una salida que incumple el "
            f"contrato. Default: {DEFAULT_TYPING_MAX_ATTEMPTS}."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime JSON con indentacion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra progreso y diagnostico en stderr.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Si la salida existe, salta trace_id ya procesados y agrega nuevos resultados.",
    )
    return finalize_llm_args(parser.parse_args(), DEFAULT_TYPING_MODEL)


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


def validate_typing(record: dict[str, Any], draft: TypingDraft) -> list[str]:
    errors: list[str] = []
    trace_id = record.get("trace_id")
    segments = record.get("segments")

    if draft.trace_id != trace_id:
        errors.append("trace_id no coincide.")

    if not isinstance(segments, list) or not segments:
        errors.append("La segmentacion de entrada no contiene segmentos validos.")
        return errors

    expected_ids = [segment.get("segment_id") for segment in segments]
    returned_ids = [node.segment_id for node in draft.nodes]
    if returned_ids != expected_ids:
        errors.append("Los segment_id devueltos no coinciden exactamente con el orden esperado.")

    for node in draft.nodes:
        if node.node_type not in VALID_NODE_TYPES:
            errors.append(f"{node.segment_id}: node_type invalido.")
        invalid_codes = sorted(
            set(node.ambiguity_codes) - VALID_AMBIGUITY_CODES
        )
        if invalid_codes:
            errors.append(
                f"{node.segment_id}: ambiguity_codes invalidos: {invalid_codes}."
            )

    return errors


def build_prompt(record: dict[str, Any]) -> str:
    segments_payload = [
        {
            "segment_id": segment["segment_id"],
            "text": segment["text"],
            "start_char": segment["start_char"],
            "end_char": segment["end_char"],
        }
        for segment in record["segments"]
    ]
    return USER_PROMPT_TEMPLATE.format(
        segments_json=json.dumps(segments_payload, ensure_ascii=False, indent=2)
    )


def call_model(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    verbose: bool,
    reasoning_effort: str = "low",
) -> TypingDraft:
    trace_id = record["trace_id"]
    prompt = build_prompt(record)

    log(f"[{trace_id}] Enviando solicitud de tipado a {client.provider}:{model}.", verbose)
    response = request_structured_response(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=TypingDraft,
        reasoning_effort=reasoning_effort,
    )
    log(f"[{trace_id}] Respuesta de tipado recibida.", verbose)
    return response.parsed


def sample_typing_draft(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    verbose: bool,
    reasoning_effort: str,
    sample_index: int,
    max_attempts: int,
    sample_events: list[dict[str, Any]],
) -> TypingDraft:
    """Obtiene una muestra válida del estimador de tipos.

    Una salida que incumple el contrato se registra y se vuelve a muestrear
    con el mismo prompt; el código nunca corrige ni completa la clasificación.
    """

    trace_id = record["trace_id"]
    validation_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        draft = call_model(client, model, record, verbose, reasoning_effort)
        if draft.trace_id != trace_id:
            draft.trace_id = trace_id
        validation_errors = validate_typing(record, draft)
        sample_events.append(
            {
                "sample": sample_index,
                "attempt": attempt,
                "validation_status": "valid" if not validation_errors else "invalid",
                "validation_errors": validation_errors,
            }
        )
        if not validation_errors:
            return draft
        log(
            f"[{trace_id}] Muestra {sample_index}, intento {attempt}: salida "
            "invalida: " + " | ".join(validation_errors),
            verbose,
        )
    raise ValueError(
        f"No se pudo validar el tipado de {trace_id!r} (muestra {sample_index}) "
        f"tras {max_attempts} intentos: {' | '.join(validation_errors)}"
    )


def type_record(
    client: LLMClient,
    model: str,
    record: dict[str, Any],
    verbose: bool,
    reasoning_effort: str = "low",
    samples: int = 1,
    max_attempts: int = DEFAULT_TYPING_MAX_ATTEMPTS,
    max_extra_samples: int = DEFAULT_MAX_EXTRA_SAMPLES,
) -> dict[str, Any]:
    """Tipa los segmentos de un registro por consenso de ``samples`` muestras.

    Cada muestra es una clasificación completa e independiente de la traza.
    El tipo de cada nodo es el mayoritario entre las muestras. Un empate se
    intenta resolver con hasta ``max_extra_samples`` muestras adicionales; si
    persiste, se conserva el tipo que apareció primero en el orden de muestreo
    y el nodo queda marcado como ambiguo. Con ``samples=1`` el resultado es el
    estimador de una sola muestra.
    """

    trace_id = record.get("trace_id")
    if not trace_id:
        raise ValueError("Cada registro debe tener trace_id.")
    if not isinstance(record.get("segments"), list) or not record["segments"]:
        raise ValueError(f"El registro {trace_id!r} no tiene segmentos validos.")
    validate_sample_count(samples)
    if max_attempts <= 0:
        raise ValueError("max_attempts debe ser mayor que 0.")
    if max_extra_samples < 0:
        raise ValueError("max_extra_samples no puede ser negativo.")

    segment_count = len(record["segments"])
    log(
        f"[{trace_id}] Iniciando tipado para {segment_count} segmentos "
        f"con {samples} muestra(s).",
        verbose,
    )
    sample_events: list[dict[str, Any]] = []
    drafts: list[TypingDraft] = [
        sample_typing_draft(
            client,
            model,
            record,
            verbose,
            reasoning_effort,
            sample_index,
            max_attempts,
            sample_events,
        )
        for sample_index in range(1, samples + 1)
    ]

    def tied_positions() -> list[int]:
        return [
            index
            for index in range(segment_count)
            if has_tie([draft.nodes[index].node_type for draft in drafts])
        ]

    extra_samples = 0
    while tied_positions() and extra_samples < max_extra_samples:
        extra_samples += 1
        log(
            f"[{trace_id}] Empate en {len(tied_positions())} nodo(s); "
            f"muestra adicional {extra_samples}/{max_extra_samples}.",
            verbose,
        )
        drafts.append(
            sample_typing_draft(
                client,
                model,
                record,
                verbose,
                reasoning_effort,
                samples + extra_samples,
                max_attempts,
                sample_events,
            )
        )

    prompt_fingerprint = structured_request_hash(
        client,
        SYSTEM_PROMPT,
        build_prompt(record),
    )
    nodes = []
    unstable_node_count = 0
    tie_node_count = 0
    for position, segment in enumerate(record["segments"], start=1):
        index = position - 1
        vote = majority_label([draft.nodes[index].node_type for draft in drafts])
        llm_codes: set[str] = set()
        for draft in drafts:
            llm_codes.update(draft.nodes[index].ambiguity_codes)
        derived_codes: list[str] = []
        if not vote["unanimous"]:
            derived_codes.append(AMBIGUITY_UNSTABLE)
            unstable_node_count += 1
        if vote["tie"]:
            derived_codes.append(AMBIGUITY_TIE)
            tie_node_count += 1
        ambiguity_codes = sorted(llm_codes | set(derived_codes))
        node: dict[str, Any] = {
            "segment_id": segment["segment_id"],
            "text": segment["text"],
            "start_char": segment["start_char"],
            "end_char": segment["end_char"],
            "position": position,
            "node_type": vote["label"],
            "review_status": "ambiguous" if ambiguity_codes else "valid",
            "ambiguity_codes": ambiguity_codes,
            "provenance": make_provenance(
                "llm_proposed",
                "typing",
                provider=client.provider,
                model=model,
                prompt_hash=prompt_fingerprint,
                reasoning_effort=reasoning_effort,
            ),
        }
        if len(drafts) > 1:
            node["consensus"] = consensus_record(
                len(drafts),
                vote["support"],
                votes=vote["votes"],
                tie_break=vote["tie_break"],
            )
        nodes.append(node)

    log(
        f"[{trace_id}] Tipado validado correctamente con {len(drafts)} muestra(s); "
        f"{unstable_node_count} nodo(s) no unanime(s).",
        verbose,
    )
    pipeline = ensure_pipeline_record(record)
    pipeline["nodes"] = nodes
    consensus_method = CONSENSUS_METHOD_VERSION if len(drafts) > 1 else None
    append_audit_event(
        pipeline,
        "typing",
        {
            "decision": "accepted",
            "node_count": len(nodes),
            "ambiguous_node_count": sum(
                bool(node["ambiguity_codes"]) for node in nodes
            ),
            "requested_samples": samples,
            "samples": len(drafts),
            "extra_samples": extra_samples,
            "consensus_method": consensus_method,
            "unstable_node_count": unstable_node_count,
            "tie_node_count": tie_node_count,
            "sample_events": sample_events,
        },
    )
    return set_validation_stage(
        pipeline,
        "typing",
        {
            "status": "valid",
            "provider": client.provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_hash": prompt_fingerprint,
            "samples": len(drafts),
            "max_attempts": max_attempts,
            "consensus_method": consensus_method,
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


def load_existing_trace_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            trace_id = data.get("trace_id")
            return {trace_id} if trace_id else set()
        if isinstance(data, list):
            return {
                item.get("trace_id")
                for item in data
                if isinstance(item, dict) and item.get("trace_id")
            }
        return set()

    trace_ids: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("trace_id"):
            trace_ids.add(row["trace_id"])
    return trace_ids


def append_jsonl_output(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    configure_stdio()
    args = parse_args()

    try:
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
        log(f"Cliente {args.provider} inicializado con modelo {args.model}.", args.verbose)
        if args.resume and args.output is None:
            raise ValueError("--resume requiere --output.")

        existing_ids: set[str] = set()
        if args.resume and args.output is not None:
            existing_ids = load_existing_trace_ids(args.output)
            if existing_ids:
                log(
                    f"Se encontraron {len(existing_ids)} trace_id ya procesados en {args.output}.",
                    args.verbose,
                )

        pending = [
            record for record in selected
            if record.get("trace_id") not in existing_ids
        ]
        log(f"Registros pendientes de tipado: {len(pending)}", args.verbose)

        results: list[dict[str, Any]] = []
        for record in pending:
            typed = type_record(
                client=client,
                model=args.model,
                record=record,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
                samples=args.samples,
                max_attempts=args.max_attempts,
            )
            results.append(typed)
            if args.resume and args.output is not None and args.output.suffix.lower() != ".json":
                append_jsonl_output([typed], args.output)

        if not (args.resume and args.output is not None and args.output.suffix.lower() != ".json"):
            log("Escribiendo salida.", args.verbose)
            write_output(results, args.output, args.pretty)
        log("Proceso completado.", args.verbose)
        return 0
    except (ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
