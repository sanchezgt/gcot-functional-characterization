# SGoTGraph pipeline

The pipeline turns a generated chain of thought into a validated artifact with two layers: a graph of typed text segments (structural layer) and the functional episodes, relations, routes and profile derived on top of it (functional layer). A language model makes the semantic decisions; the code validates each of them against the contract of the representation and derives everything else deterministically. Reference answers and correctness labels never enter the construction.

## Stages

| Stage | Operation | Type |
|---|---|---|
| 01 | Grouping of literal units (sentences or lines) into segments | LLM + deterministic reconstruction |
| 01.1 | Subdivision of segments that mix functions (`KEEP`/`SPLIT` over literal ranges) | LLM + deterministic reconstruction |
| 02 | Node typing (proposition, method, evidence, alternative, check, revision, output) | LLM |
| 03 | Validation of the typing | Deterministic |
| 04 | Reading-order backbone (`NEXT` edges) | Deterministic |
| 05 | Semantic edges (*supports*, *leads to*, *checks*, *revises*, *is alternative to*), chosen among the options the contract admits for each source node | LLM + validation and retry |
| 06 | Identification of episodes per type, with anchor and outcome, and reconciliation of overlaps | LLM + deterministic organization and validation |
| 07 | Relations between episodes | LLM |
| 08 | Routes, profile and route configurations | Deterministic |
| 09 | Validation of the functional artifact | Deterministic |

In stages 01 and 01.1 the model never returns text: it selects ranges of literal unit identifiers and the code recovers the exact spans, so the concatenation of the segments reproduces the input character by character. In stage 05 the model selects among source–type–target combinations that are already admissible under the contract; a source node with no admissible option (for example an output node) is not queried.

## Consensus, blinding and frozen instrument

- Stages 01, 01.1, 02, 05 and 06 sample the model `k` times with the same prompt and keep the majority decision (`--consensus-samples`, default 3). Episode candidates from different samples are paired by intersection over union of their members (≥ 0.5) and represented by the most central candidate. Stage 06 reconciliation and stage 07 use one sample. Every node, edge and episode produced with `k > 1` carries a `consensus` object; decisions discarded by the vote are kept in `audit`.
- Prompts receive an opaque alias of the chain instead of its identifier and never contain the source metadata (dataset, generating model, reference answer). `tests/test_prompt_isolation.py` checks this.
- `instrument_manifest.json` fixes the fingerprint of every prompt and assembled request, the model, the reasoning effort, the number of attempts and samples and the response schemas (`sgotgraph-instrument-2026.14`). `tests/test_instrument_fingerprint.py` fails if any of them changes; after an intentional change the manifest is regenerated with `python tests/test_instrument_fingerprint.py --write` and the version is raised.

## Requirements

- Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/).
- An API key for the annotator model. The annotator used in the article is `gpt-6-luna` (OpenAI), defined as `DEFAULT_OPENAI_MODEL` in `util/llm_provider.py`; `--model` or the `OPENAI_MODEL` variable override it for a run.

All commands below are run from this folder.

## Installation and credential

```bash
uv sync --frozen
cp .env.example .env    # then set OPENAI_API_KEY
```

Never put the key in inputs, outputs or shared files.

## Tests (no API calls)

```bash
uv run python -B -m unittest discover -s tests -p 'test_*.py' -v
```

## Input

A JSONL file with one object per line. Only two fields are required:

```json
{"id": "chain_0001", "reasoning_trace": "Full text of the generated chain of thought."}
```

`id` must be unique in the file and `reasoning_trace` must be the complete, unmodified chain. Other fields (problem, dataset, generating model, answers) are kept as metadata outside the prompts; `../RQ2_RQ3/cadenas_aceptadas.jsonl` and `../RQ1/escenarios/escenarios.jsonl` follow the extended format used in the article.

## Running the configuration used in the article

```bash
uv run python util/run_pipeline_completo_gcot.py INPUT.jsonl -o OUTPUT_DIR \
  --provider openai \
  --edge-max-attempts 3 --semantic-max-attempts 3 \
  --consensus-samples 3 \
  --request-timeout-seconds 300 --transport-max-retries 2 \
  --edge-payload compact --edge-sources-per-request 6 --edge-candidate-window 24 \
  --pretty
```

The three `--edge-*` options define how stage 05 queries the model: a compact representation of the graph, six source nodes per request and candidate targets within 24 nodes of the source. Without them, stage 05 queries one source node per request with the full graph as context. Both corpora of the article were annotated with the options above.

Useful options: `--index N` or `--trace-id ID` to process a single record; `--resume` to continue an interrupted batch (completed stages are not requested again); `--help` for the full list.

## Output

For each chain, `OUTPUT_DIR/cases/<id>--<hash>/attempt_001/` contains the per-stage files of the structural layer (`objective2/`, with `00_sgotgraph_objetivo2.json` as the consolidated graph) and of the functional layer (`objective3/`, with `00_sgotgraph_objetivo3.json` as the complete artifact: graph, episodes, relations, routes, profile, configurations, validation, provenance and audit), plus the request log and metrics. `OUTPUT_DIR/manifest.json` records the batch configuration and the fingerprint of the input. The schemas of both artifacts are in `schemas/`.
