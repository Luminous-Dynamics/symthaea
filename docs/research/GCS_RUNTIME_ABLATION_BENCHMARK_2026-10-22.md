# GCS Runtime Ablation Pilot (2026-10-22)

## Purpose

Measure whether Geodesic Code Synthesis (GCS) topology-derived advisory prompts help the coding agent, without conflating the result with a different compiler feature set. The agent binary is compiled once with `geodesic_synthesis`; `CodingAgentConfig::enable_gcs_advisories` toggles only whether GCS diagnostics are collected/injected. Both arms retain the same compiler and independent external behavioral-test oracle.

This is a pilot instrument, not an SOTA claim. It starts with eight small deterministic Rust tasks. The next step, after the protocol is validated, is to extend the same runtime toggle to the full oracle-hardened task bank and held-out repository-repair tasks.

## Current methodology

- **Paired intervention:** each task/repeat runs once with GCS advisories off and once on.
- **Order control:** treatment order alternates by task index plus repeat (`off_then_on` / `on_then_off`) to reduce simple order effects.
- **Oracle preflight:** every task must have at least one declared `#[test]` function; a hand-written reference must pass all expected tests, and an empty patch must fail before agent trials begin. A failed control aborts the run.
- **Fail-closed scoring:** success requires real execution, successful compilation, no runtime/test-runner error, exactly the expected number of observed test functions, all expected tests passing, and zero failures. Missing code or an incomplete oracle execution is not success.
- **Reproducible receipt:** the JSON includes source revision (when `GIT_COMMIT_SHA` is supplied), model mode, BLAKE3 task-manifest digest, per-task reference/test digests, generated-source digests, repeat/order, expected/observed test counts, completion status, pass result, runtime, iterations, and energy.
- **Separate outcomes:** compilation rate, complete-oracle rate, behavioral success rate, per-task paired wins/losses, and time/energy are reported independently.

## Run

Run from the pinned Nix/Rust environment with the repository's toolchain and dependencies available:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- \
  --repeats 3 --json-out /tmp/symthaea-gcs-ablation.json
```

Optional backends:

```bash
# Ollama must be running with the configured coding model.
cargo run --release --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --llm --repeats 3

# Requires the expected cloud credentials/configuration.
cargo run --release --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --cloud --repeats 3
```

The default pilot is 8 tasks x 3 repeats x 2 modes = 48 agent runs, plus reference and empty-patch controls. Increase repeats only after checking run-to-run variance. Output is written to the selected `--json-out` path (or a default temp path).

## Pre-registered interpretation

Primary outcome: paired behavioral pass@1 difference. Report both absolute pass rates and task-level transitions. `GCS on wins`, `GCS off wins`, and ties must sum to the paired task-trial count. The pilot's net percentage-point delta is descriptive; this 8-task suite is too small to justify statistical significance or field-wide claims.

Secondary outcomes: complete-oracle rate, real compilation rate, median/p95 agent time, mean energy, iterations, and source output digests. Any gain must be measured against a pinned model/backend, resource budget, and exact task manifest. Compare multiple model seeds/repeats and bootstrap confidence intervals on the larger held-out suite before considering a performance conclusion.

## Expansion gates

1. Run the reference/empty-patch controls and confirm every fixture is sound.
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