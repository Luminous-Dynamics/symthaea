# Factorial result ingestion contract

The companion binary analyze-results turns composer result rows into coverage,
per-seed effect, and cross-seed summary reports. It is intentionally dependency-free
and builds as part of this standalone Cargo workspace.

## Input CSV

Required columns:

| Column | Contract |
|---|---|
| case_id | Exact ID emitted by emit-cases; unknown or mutated IDs reject |
| metric | Metric name declared on the command line; ASCII letters, digits, _, - |
| status | Exactly observed, not_applicable, or failed |
| value | Required finite scalar for observed; must be empty for other statuses |
| artifact_sha256 | Required 64-character hexadecimal digest for observed; optional for non-observed outcomes |
| note | Optional diagnostic text, retained in coverage output |

note is the only optional column. Column order is not significant. Quoted CSV
fields, commas, embedded newlines, escaped quotes, and CRLF are supported.

An observed measurement is not accepted without an artifact digest. The importer
validates the digest's shape; it does not recompute the digest from an audio or
score artifact, authenticate who produced it, or prove that the digest names the
correct artifact. The future composer adapter must compute and bind that evidence.

## Run

Command:

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- discovery observations.csv note_density,mean_pitch coverage.csv effects.csv summary.csv

The cohort selector is discovery, confirmation, or both. The metric list is
explicit and comma-separated: this is necessary to expose a metric that is wholly
absent from the input, which cannot be inferred from the rows that happen to exist.

All three reports are written even when outcomes are incomplete. In that case
the process exits non-zero after writing the reports. This makes it possible to
inspect the complete denominator without treating an incomplete run as a pass.

## Output semantics

- coverage.csv: one row for every expected style/seed/metric block, including
  blocks with no input rows. It reports 81 expected cases, observed,
  not-applicable, failed and missing counts, a disposition, the ordered list of
  supplied artifact digests, and diagnostic notes.
- effects.csv: 80 orthogonal contrasts per complete block only. A block is
  complete only when all 81 cases have exactly one finite observed value for
  that metric. Incomplete blocks emit no effect estimates.
- summary.csv: cross-seed mean, sample standard deviation, minimum and maximum
  for each contrast, grouped separately by cohort, style and metric. A
  one-seed standard deviation is blank, not zero.

No missing, failed, or not-applicable result is imputed as zero. A not-applicable
metric is not a successful zero response. Duplicate case/metric rows, unexpected
case IDs, undeclared metrics, malformed digests, non-finite values, and malformed
CSV reject the input.

The report contains descriptive orthogonal coefficients and seed dispersion. It
does not produce p-values, confidence intervals, musical-quality judgments,
listener preference, or a causal claim beyond the registered deterministic
intervention design.
