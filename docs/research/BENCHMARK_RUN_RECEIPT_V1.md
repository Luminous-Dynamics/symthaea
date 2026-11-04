# Benchmark Run Receipt v1

A language-neutral, provenance-bound benchmark run receipt plus a Rust semantic validator.

## What a receipt establishes

A valid receipt establishes that the supplied record is structurally valid under the Draft 2020-12 schema, that its task rows reconcile with the denominator, budgets and score metadata are internally consistent, and that its canonical payload hash matches.

It does **not** establish that the grader is honest, the system answer is correct, a producer is authentic, or a signature is trusted. Correctness still depends on an independent grader/oracle. Authenticity requires an independently trusted signature or attestation layer.

## Contract files

- `schemas/benchmark-run-receipt-v1.schema.json` — JSON Schema Draft 2020-12.
- `examples/benchmark-run-receipt-v1.synthetic.json` — synthetic five-task fixture. It is not an external benchmark result.
- `tools/benchmark-receipt-harness/src/lib.rs` — Rust semantic validator.
- `tools/benchmark-receipt-harness/src/main.rs` — CLI.
- `tools/benchmark-receipt-harness/validate_schema.py` — CI-only JSON Schema audit, with external references disabled.
- `tools/benchmark-receipt-harness/Cargo.lock` — isolated pinned Rust dependency graph.

## Accounting model

A receipt enumerates every eligible task and assigns exactly one disposition:

- passed
- failed
- refused
- abstained
- unsupported
- timed_out
- tool_failed
- ungraded
- not_attempted

The v1 semantic validator enforces:

- eligible = attempted + not_attempted;
- attempted = the sum of all attempted terminal dispositions;
- not_attempted equals the number of not_attempted rows;
- graded = passed + failed;
- every eligible task ID is unique and has an input digest;
- refused, abstained, unsupported and timed-out tasks remain in the eligible denominator;
- every graded task has a grader-result digest and score;
- only preregistered exclusions are allowed;
- measured resource usage does not exceed declared budget caps;
- official scores require an official grader, the pinned scoring revision and an artifact digest for the grader output;
- derived metrics are explicitly non-official and known coverage/yield values must reconcile to the task accounting;
- matched arms bind the same eligible task set, budget profile, environment cohort and scoring revision;
- duplicate JSON keys and oversized raw receipts are rejected.

The sample has five eligible tasks: two passes, one refusal, one unsupported result and one not attempted. Thus it has 4/5 eligible coverage and 2/5 coverage-adjusted yield. The refusal and unsupported case are not silently removed from the denominator.

## Official versus derived scores

Never relabel a derived measure as the benchmark owner's official metric.

- Official score: reproduce the owner's formula, unit, numerator/denominator, scoring revision and grader output digest.
- Coverage: attempted / eligible.
- Graded coverage: graded / eligible.
- Coverage-adjusted task yield: passed / eligible.

When a denominator is zero, emit null with a reason rather than divide by zero. The official score and derived coverage measures must remain separate.

## Canonical receipt digest

`integrity.receipt_sha256` is SHA-256 over the RFC 8785/JCS canonical JSON receipt with only that field removed. The exact raw file SHA-256 is calculated separately by the validator and is not substituted for the semantic receipt hash.

The self-hash is an integrity commitment, **not** a signature. The CLI always reports `authenticity_verified: false`; a detached signature/trust verifier must remain a separate component.

## Run

```bash
cargo fmt --manifest-path tools/benchmark-receipt-harness/Cargo.toml -- --check
cargo test --manifest-path tools/benchmark-receipt-harness/Cargo.toml --locked
cargo run --manifest-path tools/benchmark-receipt-harness/Cargo.toml --locked --bin benchmark_receipt_validate -- \
  --receipt examples/benchmark-run-receipt-v1.synthetic.json \
  --out benchmark-receipt-validation.json
```

Validate the JSON Schema independently in CI:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/benchmark-receipt-harness/requirements-ci.txt
.venv/bin/python tools/benchmark-receipt-harness/validate_schema.py \
  --schema schemas/benchmark-run-receipt-v1.schema.json \
  --instance examples/benchmark-run-receipt-v1.synthetic.json
```

## Matched A/B/C comparisons

The Rust library exposes `validate_matched_receipts` for multi-arm comparisons. Every run must have a unique run ID and arm, and every receipt must bind the same group, benchmark, eligible task-set digest, budget-cap digest, environment/cohort digest and scoring revision. It is not enough for two receipts to merely label themselves "matched".

## Current limitations

- v1 official score representation supports a numerator/denominator ratio. Benchmarks with more complex scoring rules need an explicit versioned formula adapter rather than being forced into the ratio.
- Semantic v1 task outcomes use binary grading for passed/failed; non-binary graded outcomes require an explicit future schema version/terminal-outcome extension.
- The receipt validator does not verify detached signatures or producer identity.
- Cross-run A/B/C comparability is checked only when callers supply the full receipt set to `validate_matched_receipts`.
- This harness is a first implementation; CI results determine whether it can be treated as qualified.

## Claim ceiling

A valid receipt makes accounting and integrity reproducible. It does not prove answer correctness, source authenticity, real-device execution, causal validity or benchmark-owner certification.
