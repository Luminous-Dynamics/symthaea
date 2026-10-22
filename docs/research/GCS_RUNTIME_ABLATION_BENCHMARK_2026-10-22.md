# GCS Runtime Ablation Pilot (2026-10-22)

## Purpose

Measure whether Geodesic Code Synthesis (GCS) topology-derived advisory prompts help the coding agent, without conflating the result with a different compiler feature set. The agent binary is compiled once with `geodesic_synthesis`; `CodingAgentConfig::enable_gcs_advisories` toggles only whether GCS diagnostics are collected/injected. Both arms retain the same compiler and independent external behavioral-test oracle.

This is a pilot instrument, not an SOTA claim. It starts with eight small deterministic Rust tasks. The next step, after the protocol is validated, is to extend the same runtime toggle to the full oracle-hardened task bank and held-out repository-repair tasks.

## Current methodology

- **Paired intervention:** each task/repeat runs once with GCS advisories off and once on. A pair enters the win/loss/tie treatment comparison only when both arms have complete oracles; unresolved pairs are counted separately, not treated as ties.
- **Order control:** treatment order alternates by task index plus repeat (`off_then_on` / `on_then_off`) to reduce simple order effects.
- **Oracle preflight (schema v2):** every fixed task declares its expected test count explicitly in `TaskCase`; that count is bound into the task digest rather than inferred by scanning source text. The reference must pass exactly that many tests; an empty patch must be rejected by a real, non-simulated Rust compiler diagnostic; and every semantic mutant must compile and fail at least one test while accounting for the full declared count. Missing mutants, surviving mutants, compile errors in a mutant, simulated execution, or incomplete tests abort before agent trials.
- **Fail-closed scoring:** success requires real execution, successful compilation, no runtime/test-runner error, exactly the expected number of observed test functions, all expected tests passing, and zero failures. Oracle completeness is separate: `CodeExecutor` may record a nonzero test-binary exit in `runtime_error` when an assertion fails. If the full expected test count is present, the result is a complete behavioral failure, not an unresolved oracle; missing counts or simulated execution remain unresolved.
- **Source-bound execution:** before oracle controls or agent trials, the producer requires `GIT_COMMIT_SHA` to be a full 40-character SHA matching `git rev-parse HEAD`, and requires a clean working tree. It refuses to create an unauditable receipt from a dirty or ambiguous checkout.
- **Backend identity and observed routing:** schema v4 records both `selected_generation_tiers` (the dispatcher's selected tier before fallback) and `backend_attempts` (the implementation actually invoked, including failed calls and fallback paths). This distinction is necessary because a selected CloudLLM call may fail and then execute through Ollama or a simulated backend. The per-arm summary separately counts selected-tier decisions, actual attempts by tier and backend name, successful/failed attempts, optional internal-verifier acceptance/rejection/unknown outcomes, external-LLM attempts, simulated attempts, and runs with no backend calls; the independent auditor recomputes these counts from per-run records. `success` means the backend returned a non-empty output; `verification_passed` is a separate optional field and is not inferred from output presence. Each backend attempt records its own wall-clock duration, including failures before fallback; the auditor recomputes per-arm attempt-latency total and nearest-rank p50/p95 from the attempt records. It also stores an internal estimated-energy unit for the effective implementation per attempt and recomputes that total from run records. For example, a Simulated implementation has a zero per-attempt estimate even when reached after a CloudLLM selection. This is not provider billing, measured joules, or a substitute for the dispatcher's own budget charge; failed cloud + fallback dispatches can charge both selected invocation costs. Agent end-to-end latency, attempt latency, and external oracle verification latency remain separate measurements. These are application-reported telemetry, not provider attestation.
- **Reproducible receipt:** schema v4 includes the resolved source revision, requested backend/model label, selected tier decisions, actual backend-attempt records, ordered task-manifest digest and task content (description, expected symbol, explicit expected test count, reference source, test fixture, semantic mutants), per-task source/test digests, mutant preflight outcomes, generated-source digests, repeat/order, expected/observed test counts, completion status, pass result, runtime, iterations, and energy. Schema-v1/v2/v3 receipts are not interchangeable with v4. Treat these pilot fixtures as public; this receipt design is not a substitute for hidden tests.
- **Separate outcomes:** compilation rate, complete-oracle rate, behavioral success rate, per-task paired wins/losses, route-use counters, mean and nearest-rank p50/p95 agent/verification latency, and energy are reported independently. Tail quantiles are descriptive summaries; they are not confidence intervals and the small pilot cannot establish statistical significance.

## Pre-register the task manifest

Before any compiler controls or model trials, print the current frozen manifest without running the benchmark:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --print-manifest
```

Record `source_revision` and `task_manifest_blake3` in an independently reviewed preregistration or committed protocol record. This command requires the same clean source checkout, but performs no reference, mutant, empty-patch or agent trial execution.

## Run

Run from the pinned Nix/Rust environment with the repository's toolchain and dependencies available:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- \
  --repeats 3 --json-out /tmp/symthaea-gcs-ablation.json
```

Optional backends:

The local backend's actual constructor default is `gemma4:e2b`; set `SYMTHAEA_LLM_MODEL` to pin another model tag. The cloud backend requires a non-empty `ANTHROPIC_API_KEY`; `ANTHROPIC_MODEL` defaults to `claude-sonnet-4-20250514`, matching the backend constructor. The receipt records the requested model, selected tier decisions, and actual backend invocations. A selected tier is not proof of which implementation supplied the returned output; fallback can cross that boundary. The auditor validates actual backend names against known tiers (`Ollama` → `LocalLLM`, `Anthropic` → `CloudLLM`, `Simulated` → `Simulated`) and checks actual attempts against the configured dispatcher, while allowing simulated fallback and reporting it rather than excluding it. This is application-reported telemetry, not cryptographic attestation of the provider or model revision.

```bash
# Ollama must be running with the configured coding model.
cargo run --release --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --llm --repeats 3

# Requires the expected cloud credentials/configuration.
cargo run --release --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --cloud --repeats 3
```

The default pilot is 8 tasks x 3 repeats x 2 modes = 48 agent runs, plus reference, empty-patch, and ten semantic-mutant controls. Increase repeats only after checking run-to-run variance. Output is written to the selected `--json-out` path (or a default temp path).

## Pre-registered interpretation

Primary outcome: paired behavioral pass@1 difference. Report both absolute pass rates and task-level transitions. `GCS on wins`, `GCS off wins`, and ties must sum to the paired task-trial count. The pilot's net percentage-point delta is descriptive; this 8-task suite is too small to justify statistical significance or field-wide claims.

Secondary outcomes: complete-oracle rate, real compilation rate, median/p95 agent time, mean energy, iterations, and source output digests. Any gain must be measured against a pinned model/backend, resource budget, and exact task manifest. Compare multiple model seeds/repeats and bootstrap confidence intervals on the larger held-out suite before considering a performance conclusion.

## Expansion gates

1. Run reference-pass and real empty-patch-compile-rejection controls; every semantic mutant must compile and be rejected by at least one test.
2. Run formatting and compile/test checks on the exact PR head.
3. Execute the pilot in both native and a configured LLM mode; archive receipts, even if the result is null or negative.
4. Re-run after each benchmark/task/verifier change; version the task manifest and explain result breaks.
5. Scale to at least 100 curated Rust tasks, then to 200+ held-out repository-level tasks with fail-to-pass and pass-to-pass regression tests, timeouts, dependency/network controls, and an independent verifier.
6. Compare with SWE-Bench Pro / Terminal-Bench only using their published protocol and keep task-family/language differences explicit. Their scores are not directly comparable to this native Rust micro-task pilot.

## External methodological grounding

- SWE-Bench Pro V2 uses 642 repository-level tasks in 11 repositories, graded with hidden fail-to-pass and pass-to-pass tests in pristine containers: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro
- Terminal-Bench 4 is a continuous benchmark that calibrates task resources and fixes/removes problematic tasks across releases: https://www.tbench.ai/news/terminal-bench-4-0
- Independent Epoch AI review of Terminal-Bench 4.0.0 identified issues in many public tasks, including verifier reward hacks, leakage, and false negatives: https://epoch.ai/benchmarks/terminal-bench-4/review
- OpenAI's 2026 coding benchmark audit argues that defective or contaminated tasks can distort performance conclusions: https://openai.com/index/separating-signal-from-noise-coding-evaluations/

## Validation status

The authoring environment does not have `rustc` or `cargo`; no build, test, oracle run, or benchmark result is claimed. Execute these checks on the exact head:

```bash
cargo fmt --check
cargo test --features 'code_generation geodesic_synthesis' --lib coding_agent::tests::test_gcs_advisories_default_on_and_runtime_toggle_exists
cargo run --release --example coding_gcs_ablation_benchmark --features 'code_generation geodesic_synthesis' -- --repeats 1 --json-out /tmp/gcs-ablation-smoke.json
```

Inspect the resulting receipt, verify task/reference/test digests independently, then run the planned repeated trials. Queued or absent CI results are not passes.