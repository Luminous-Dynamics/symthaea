# Benchmark Repro Guide (Reviewer Path)

This guide is the single entry-point for running Symthaea benchmarks in a
reviewer-friendly, reproducible way.

## Principles

- Local-only: no remote model APIs.
- Cyborg Mind: HDC primitives do the reasoning; the LLM is translation-only.
- Deterministic environments: OSWorld runs on NixOS to remove entropy.
- Action budgets enforced via ActionIR policies.

## Quick Map

- External suites registry: `benchmarks/manifest.json`
- External harness root: `benchmarks/external/`
- Internal Criterion benches: `benches/` (see `benches/BENCHMARK_GUIDE.md`)

## Local LLM (Translator) Setup

Symthaea uses a local Ollama backend by default.

1) Start Ollama and pull a quantized model (examples):

```
ollama serve
ollama pull phi4:mini-q4_K_M
# or: ollama pull llama3:8b-instruct-q4_K_M
```

2) Configure the translator model:

```
export SYMTHAEA_LLM_MODEL=phi4:mini-q4_K_M
```

You can source a prefilled profile:

```
set -a
source benchmarks/external/config/cyborg_mind.env
set +a
```

The translator prompt is defined in `src/language/llm_organ.rs` as
`TRANSLATION_SYSTEM_PROMPT`. This enforces "translate, do not reason".

## Action Budgets

Action budgets are enforced in `src/action/mod.rs` via `PolicyBundle`.
For benchmark runs, use a restrictive policy and explicit allowlists.

Environment switch for real execution:

```
export SYMTHAEA_ALLOW_REAL_EXEC=1
```

If you do not set this, ActionIR runs in simulation mode.

## External Benchmarks (New)

Run the harness from the `symthaea/` root:

```
python benchmarks/external/run_external.py --list
python benchmarks/external/run_external.py --bench arc-agi-2
```

Each benchmark has:

- `fetch.sh` to obtain datasets or tools
- `run.sh` to execute and write normalized results
- `data/READY` as the "dataset ready" marker

See `benchmarks/external/README.md` for details per suite.

### Run All (Reviewer)

```
benchmarks/external/run_all.sh
```

If you want to override the env file or skip READY checks:

```
export SYMTHAEA_BENCH_ENV=benchmarks/external/config/cyborg_mind.env
export SYMTHAEA_BENCH_SKIP_READY=1
benchmarks/external/run_all.sh
```

### Agent Hook (Optional)

For GAIA tool-use evaluation, you can provide a custom agent via:

```
export SYMTHAEA_AGENT_CMD="/path/to/agent"
```

See `benchmarks/external/agent_protocol.md`.

### Cyborg Mind (HDC+CfC/LTC) Controls

```
export SYMTHAEA_AGENT_PROFILE=standard
export SYMTHAEA_AGENT_TEMPORAL_BACKEND=cfc
export SYMTHAEA_AGENT_CYCLES=3
export SYMTHAEA_AGENT_CONTEXT_LIMIT=4000
# export SYMTHAEA_AGENT_GENESIS=cyborg-mind-v1
```

## Internal Benchmarks (Existing)

```
cargo bench --bench quick
cargo bench --bench standard
cargo bench --bench consciousness
```

See `benches/BENCHMARK_GUIDE.md` for the full suite.

## Result Files

External benchmark results go to:

```
benchmarks/external/results/*.json
```

Use the schema in `benchmarks/external/results/README.md`.

## Baseline Runs

To establish a baseline after datasets are fetched:

1) Run each external suite via `run_external.py`.
2) Copy results to `benchmarks/external/results/baseline-YYYYMMDD.json`.
3) Record model + policy settings used in the result metadata.
