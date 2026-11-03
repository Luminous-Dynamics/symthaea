# Benchmark Run Receipt v1

**Status:** first Rust implementation and synthetic fixture; dedicated workflow pending  
**Issue:** [BENCH-RECEIPT-001 / #10029](https://github.com/Luminous-Dynamics/symthaea/issues/10029)  
**Schema:** `schemas/benchmark-run-receipt-v1.schema.json`

## Purpose

A benchmark score is a claim with a denominator, eligible task universe, execution budget, system identity, target environment and independent grader. It must not be a scalar detached from how it was produced.

The receipt separates three layers:

1. **Official metric:** the benchmark owner's exact metric, formula version, unit, numerator/denominator where applicable, and grader-output digest.
2. **Coverage and accounting:** eligible, excluded, attempted, not attempted, in progress, completed, graded, passed, failed, refused, abstained, unsupported, timed out, tool failed and ungraded.
3. **Derived internal metrics:** coverage-adjusted yield, conditional success, selective risk, cost per pass and other explicitly non-official calculations.

A synthetic receipt is never an external benchmark result.

## Core invariants

For one receipt, the Rust semantic validator enforces:

```
universe = eligible + excluded
eligible = attempted + not_attempted
attempted = in_progress + completed
completed = passed + failed + refused + abstained
          + unsupported + timed_out + tool_failed + ungraded
graded = passed + failed
```

Every eligible task has exactly one task-result row, including eligible tasks not attempted. Excluded tasks must have been predeclared with a reason and timestamp, and must not appear as task results. Duplicate task IDs are rejected. A task with a terminal pass/fail must carry a matching grader record; a binary pass cannot have a false boolean grade or vice versa. Unknown and unavailable states are not silently coerced into pass/fail.

The receipt computes and checks two canonical task-set commitments:

- `eligible_task_set_sha256`: SHA-256 of RFC 8785 canonical JSON for the lexically sorted array of eligible task IDs.
- `task_universe_sha256`: SHA-256 of RFC 8785 canonical JSON for the task-universe rows sorted by task ID.

The raw receipt-file SHA-256 and canonical semantic receipt SHA-256 are separate outputs. Hashes establish byte/semantic consistency only; they do not authenticate a producer or grader. A signed receipt needs an independently trusted signer/key and revocation/replay policy, which this first CLI does not implement.

## Budget accounting

Budget caps and actual usage are represented as wall time, CPU time, peak memory, model calls, input/output tokens and tool calls. Null actual usage requires an explicit reason in `unavailable_usage`. A measured value and an unavailable reason for the same dimension are contradictory. Any measured usage above a declared cap rejects the receipt.

No budget value is inferred from missing usage. A null budget cap means unbounded/not declared, not zero.

## Official and derived metrics

The `official_score` object always records the benchmark's original formula, unit, source output digest and official/not-official flag. If the benchmark has no valid score (for example, zero graded tasks), use null score/numerator/denominator and a nonempty unavailability reason. Do not fabricate 0%, 100%, or a denominator of one.

All `derived_metrics` must declare `is_official: false`. Examples include:

- eligible coverage = attempted / eligible;
- graded coverage = graded / eligible;
- coverage-adjusted task yield = passed / eligible;
- conditional success = passed / graded;
- cost per passed task;
- selective risk versus coverage.

If a denominator is zero, the derived value is null with a reason. These derived values must never be labelled as the benchmark owner's score.

## Matched-arm comparisons

`validate_comparable_arms` checks that receipts share:
- benchmark identity, version, revision, dataset digest, access class and scoring formula;
- target scope and runner identity;
- budget caps;
- eligible task-set digest and full task-universe digest;
- comparison group and protocol digest;
- grading mode.

Per-run grader output digests may differ because each arm produces its own result. A comparison receipt cannot silently substitute a different eligible task universe or loosen a resource budget.

## Usage

From the repository root:

```bash
cargo fmt --manifest-path tools/benchmark-run-receipt-harness/Cargo.toml -- --check
cargo test --manifest-path tools/benchmark-run-receipt-harness/Cargo.toml --locked

cargo run --manifest-path tools/benchmark-run-receipt-harness/Cargo.toml --locked -- \
  examples/benchmark-run-receipt-v1.synthetic.json
```

Validate a single receipt:

```bash
benchmark-run-receipt-validate run.json
```

Validate a matched comparison:

```bash
benchmark-run-receipt-validate arm-a.json --compare arm-b.json arm-c.json
```

The CLI prints the raw-file digest and canonical semantic digest separately. It does not certify answer correctness; the benchmark's independent grader remains the correctness oracle.

## Current implementation boundary

The initial Rust implementation validates the v1 receipt contract, accounting equations, task coverage, task digests, grade/disposition agreement, resource caps, official/derived metric separation, and matched comparison groups. It rejects duplicate raw JSON keys and bounds raw inputs to 16 MiB.

Not yet implemented:
- detached signature verification and trusted signer/key policy;
- JSON Schema 2020-12 runtime execution inside the CLI (the committed schema is a language-neutral structural contract; the Rust validator independently performs stricter semantic checks);
- benchmark-specific scoring formula execution;
- aggregation across signed run receipts or a leaderboard service;
- automatic data ingestion from SRE-Bench, AgentRE-Bench, Decompile-Bench, MobSF or MASTG.

## External baseline use

For SRE-Bench, preserve the benchmark's official score and graded denominator. The October 2026 public-facing snapshot reports GPT-6-Astra at 5.55/6 on 181 graded binaries out of 262, with 149/181 fully solved (82.3%). The 82.3% value is conditional on the graded subset; 149/262 (56.9%) is a derived full-corpus fraction, not the official headline. Public-facing safeguarded results must remain separate from the unconstrained setting.

Official source: https://daplab.cs.columbia.edu/benchmarks/sre-bench/

## Claim ceiling

A valid receipt means its accounting, identities, task digests and declared budgets are internally consistent. It does not prove producer authenticity, grader correctness, task-answer truth, fair task construction, or general superiority over another system.
