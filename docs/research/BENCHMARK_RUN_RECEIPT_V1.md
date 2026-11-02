# Benchmark Run Receipt v1

This contract is intended to make every REA/Symthaea benchmark result reproducible and auditable. A benchmark score is never stored as an isolated number.

## Files

- `schemas/benchmark-run-receipt-v1.schema.json`: JSON Schema 2020-12 structural contract.
- `examples/benchmark-run-receipt-v1.synthetic.json`: synthetic schema example with one pass, one refusal and one not-attempted task. It is not an external benchmark result.
- [BENCH-RECEIPT-001 / #10029](https://github.com/Luminous-Dynamics/symthaea/issues/10029): Rust semantic validator and integration requirements.

## Three score layers

1. **Official metric**: preserve the benchmark owner's metric, unit, formula version, numerator, denominator and grader output digest exactly.
2. **Coverage/accounting**: disclose the complete eligible universe and the disposition of every task.
3. **Derived measures**: coverage-adjusted yield, cost-normalized yield, calibration and selective risk are explicitly marked `derived_non_official`. Never relabel them as the official score.

A synthetic run is structurally prevented from declaring an official score.

## Task accounting equations

The schema validates field types and shape. A semantic validator must additionally enforce:

```text
universe = eligible + excluded
eligible = attempted + not_attempted
attempted = completed + in_progress
completed = passed + failed + refused + abstained
          + unsupported + timed_out + tool_failed + ungraded
graded = passed + failed                 # binary pass/fail protocols
task_outcomes.length = eligible
excluded_tasks.length = excluded
```

The eight terminal-disposition counts must equal the corresponding task rows. Task IDs must be unique across eligible and excluded rows. Every task row must have a 64-character input digest. Attempted rows cannot be relabelled as not attempted; refusals, abstentions, unsupported cases, timeouts and tool failures remain visible.

For non-binary scoring protocols, the run must declare a separately versioned grading equation. Do not apply the binary equation to a partial-credit/continuous grader.

## Budgets and resource accounting

Each budget cap is bounded, unbounded or not applicable with a reason. Each actual resource measurement is measured or explicitly unavailable/not applicable with a reason. A missing measurement is never interpreted as zero.

The semantic validator checks every measured actual against its corresponding bounded cap. Resource names have fixed units in the schema (milliseconds, bytes, calls, tokens, micro-USD and microjoules).

## Matched comparison rules

Two arms are comparable only when they bind the same:
- eligible task-set digest and benchmark/split/scoring protocol revision;
- target cohort and instrumentation contract;
- budget-protocol digest;
- benchmark grader and grader version/digest.

Differences must produce `comparable=false` and explicit reasons. A one-arm run is never a matched A/B comparison.

## Receipt integrity and trust

`integrity.payload_sha256_excludes_integrity` is SHA-256 over the RFC 8785 JCS encoding of the entire receipt object **with the top-level `integrity` member removed**. This avoids a self-referential digest. A detached signature, when present, signs that payload digest.

The digest provides content integrity only. A self-hash does not authenticate the runner, grader, model, tool or benchmark operator. Authenticity requires a separately trusted signature/attestation and trust policy.

## Derived metrics

Recommended formulas:
- eligible coverage = attempted / eligible;
- graded coverage = graded / eligible;
- coverage-adjusted task yield = passed / eligible;
- official binary success rate = official passed / official graded denominator;
- cost per passed task and cost per eligible task;
- selective risk versus coverage.

For zero denominator, emit `value: null` and a nonempty `zero_denominator_reason`; never emit NaN/Infinity or silently substitute zero.

## Claim ceiling

A valid receipt proves only that the run's accounting and provenance fields satisfy the validator contract. It does not prove that a task result is correct; the independent benchmark grader/oracle establishes that. It does not prove general system superiority. Every result remains limited to the benchmark version, dataset/split, target cohort, model/tool versions and budgets recorded in the receipt.
