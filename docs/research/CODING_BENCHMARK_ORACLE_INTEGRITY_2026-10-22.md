# Coding Benchmark Oracle Integrity (2026-10-22)

## Status and claim boundary

This change hardens a measurement path. It is **not** a measured improvement in code-generation ability and makes no state-of-the-art claim. The repository has not been compiled or executed in the environment that prepared this change; exact-head Rust tests and benchmark runs remain required.

## Finding

The prior `TaskResult::is_correct` predicate accepted a compiled result when both test counts were zero. Its `test_source_empty()` helper inferred that no tests had been specified from the execution result rather than from the task fixture. Therefore a failed-to-launch test process, an empty test run, or an accidental compile-only path could be scored as correct.

The documented `cargo run --example coding_agent_benchmark` target was also absent from the explicit Cargo example registrations even though the root manifest sets `autoexamples = false`.

## Change

- Explicitly register `examples/coding_agent_benchmark.rs` as a Cargo example behind `code_generation`.
- Count the expected `#[test]` cases in each static task fixture.
- Report expected and observed tests separately per task.
- Emit an explicit `oracle_status`: `missing_oracle`, `not_run_compile_failure`, `no_tests_executed`, `test_count_mismatch`, `failed`, or `passed`.
- Count a task as correct only if it compiles, has a non-empty predeclared oracle, executes exactly the expected test count, and has zero failing tests.
- Add unit tests for missing, zero-run, partial, fully passing, and failing oracles.

This deliberately uses a conservative policy: compile-only work can contribute to compilation metrics but cannot silently contribute to correctness. Static inspection of the present suite found **11 hard tasks with empty test fixtures**; they remain useful compilation probes but will be marked `missing_oracle` and excluded from the assessed correctness numerator and denominator until independent behavioral tests are authored. A compilation failure on a task with a defined oracle is a scored task failure. If compilation succeeds but the test runner executes zero or only a subset of the expected tests, the attempt is marked unverified and excluded from the assessed denominator—not silently passed or treated as an agent failure. A complete run with failing assertions is a scored task failure. The per-task JSON exposes `assessed`, `unverified`, and `oracle_status` so these cases remain distinguishable. The static fixture format is the source of the expected-test count; if the fixture format changes, the expected-count parser must change with it.

## Local verification protocol

Run from the repository root on the exact PR head:

```bash
cargo test --example coding_agent_benchmark --features code_generation
cargo run --release --example coding_agent_benchmark --features code_generation
```

Keep these gates separate:

1. **Compilation:** the benchmark target builds.
2. **Oracle-integrity unit tests:** all fail-closed regression cases pass.
3. **Fixture coverage:** every task intended to claim correctness declares a positive number of expected tests.
4. **Execution coverage:** each run reports expected tests, executed tests, and oracle status; no zero-run or count-mismatch task counts as correct.
5. **Capability:** only the actual held-out test outcomes count as task resolution. Do not infer capability improvement from a benchmark registration fix.

Do not promote the benchmark or change quality policy based on a single green-looking aggregate. Inspect the per-task JSON, validate that the external oracle really ran, and bind results to the exact commit and build configuration before comparing runs.

## Paradigm shift: measure verified engineering value

A coding agent's output volume, compile rate, structural similarity, and claimed test pass rate are different quantities. The key outcome should be **correct, independently test-verified tasks per fixed compute/cost budget**, with compile rate, oracle coverage, regressions, latency, retries, and cost reported alongside it.

For follow-on experiments, preregister the tasks and acceptance criteria; freeze agent/model/tool versions and budgets; preserve a private held-out task set; use multiple paired runs for stochastic systems; report uncertainty; and retain raw per-task receipts. Do not train on the held-out tasks or reuse their reference patches.

## Industry comparison plan

Use more than one evaluation tier:

- **Rust-native regression suite:** fast, repeatable, and suitable for every commit. It measures Symthaea's local mechanics but is not an industry leaderboard.
- **Private repository-level Rust tasks:** new, held-out bugs and feature requests authored after the evaluated system's training cutoff where possible. Grade against independent tests, hidden regression cases, and reviewer-defined behavioral constraints.
- **SWE-Bench Pro V2:** a later public, repository-level comparator. Its September 22, 2026 release describes 642 tasks across 11 repositories. Treat it as a candidate comparison set, not automatically valid ground truth; confirm the exact release, task IDs, execution harness, and quality audit before comparing scores.
- **SWE-Bench Live Rust split:** useful for fresh, language-specific issue resolution, but treat leaderboard scores as time- and harness-specific rather than directly interchangeable with other suites.
- **Terminal-Bench:** a complementary systems/terminal-agent testbed that covers work beyond editing one patch. Its maintainers recommend repeatedly checking oracle task behavior in the target sandbox, which is an appropriate minimum practice for our own fixtures.

**Important benchmark caveat:** OpenAI's July 8, 2026 audit of the 731-task public SWE-Bench Pro split found the automated review flagged 200 tasks (27.4%) and a five-engineer annotation campaign identified 249 (34.1%) as broken/problematic; the article estimates about 30% broken and retracts its earlier recommendation to adopt Pro. This audit does not by itself prove the later 642-task Pro V2 release has the same defect rate. It does mean we must not infer quality from a benchmark name or version number: audit the exact task IDs we use and keep a private, independently validated Rust suite as the primary capability gate. SWE-bench Verified also has contamination risks and should not be the sole frontier claim.

References:

- OpenAI, *Separating signal from noise in coding evaluations* (2026-07-08): https://openai.com/index/separating-signal-from-noise-coding-evaluations/
- OpenAI, *Why SWE-bench Verified no longer measures frontier coding capabilities* (2026-02-23): https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Scale AI, *SWE-Bench Pro V2* (2026-09-22): https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- Terminal-Bench: https://github.com/harbor-framework/terminal-bench
- SWE-Bench Live: https://swe-bench-live.github.io/
- SWE-Bench Pro paper, ICML 2026: https://proceedings.mlr.press/v306/deng26f.html

## Proposed acceptance targets for the next phase

These are targets, not current results:

- 100% of tasks counted as correct have a non-empty oracle and a complete, successful test run.
- Zero tasks with `missing_oracle`, `no_tests_executed`, or `test_count_mismatch` are counted as correct.
- Report compilation, oracle-complete, and correctness rates separately.
- Compare native-only, local-model, and hosted-model configurations under matched tasks and budgets.
- Require a statistically defensible paired improvement on held-out tasks, with no increase in regression or unverified-success rate, before claiming a coding-capability gain.
- Report wall-clock latency, retries, token/tool cost where available, and performance per dollar; optimize the quality/cost Pareto frontier rather than raw pass rate alone.

The next implementation tranche should add provenance-bound benchmark receipts (commit SHA, fixture digest, model/scaffold version, execution settings), mutation tests that prove the oracle catches intentionally wrong solutions, and repository-level tasks. This PR intentionally stops short of claiming those capabilities already exist.
