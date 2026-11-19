# Factorial result ingestion contract

The companion binary analyze-results turns composer result rows into coverage,
per-seed effect, cross-seed summary, and exact axis-invariance reports. It is
intentionally dependency-free and builds as part of this standalone Cargo workspace.

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

    cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- discovery observations.csv note_density,mean_pitch coverage.csv effects.csv summary.csv axis-invariance.csv

The cohort selector is discovery, confirmation, or both. The metric list is
explicit and comma-separated: this is necessary to expose a metric that is wholly
absent from the input, which cannot be inferred from the rows that happen to exist.

All four reports are written even when outcomes are incomplete. The process exits
non-zero when a block has failed, missing, or mixed outcomes, after writing the
reports. A block with 81 explicit not-applicable outcomes remains a
not_applicable coverage row and does not itself fail the run; it produces no
effect estimate. This permits grammars with no such plan field to be represented
without fabricating a zero.

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
- axis-invariance.csv: per style/seed/axis counts across 27 slices. Each slice
  holds the other three factors fixed and compares the full score SHA-256 at the
  low, centre and high levels of the target axis. It reports complete,
  identical, changed and unavailable slice counts, the declared support status,
  and a disposition. A declared unsupported axis with 27/27 identical slices is
  negative_control_pass; any exact change is unexpected_change. A declared
  supported axis is reported as response_observed or flat_supported_axis, not
  automatically scored as a win.

No missing, failed, or not-applicable result is imputed as zero. A not-applicable
metric is not a successful zero response. Duplicate case/metric rows, unexpected
case IDs, undeclared metrics, malformed digests, non-finite values, and malformed
CSV reject the input.

The report contains descriptive orthogonal coefficients and seed dispersion. It
does not produce p-values, confidence intervals, musical-quality judgments,
listener preference, or a causal claim beyond the registered deterministic
intervention design.



## Identity-only metrics

Metric names beginning with `identity_` are coverage-audited but are not sent
through the scalar contrast estimator. They must still have complete 81/81
observed rows to receive a complete coverage disposition. They produce no rows
in `effects.csv` or `summary.csv`. Use this mode for exact fingerprints,
category identifiers, or other values whose numeric encoding is not a
meaningful interval scale. The coverage report records `analysis_mode=identity_only`;
ordinary scalar channels use `effect_estimated`.

Examples from the composer adapter are:
- `identity_score_meter_beats`
- `identity_score_key_tonic_pc`
- `identity_exact_fingerprint_high32`
- `identity_exact_fingerprint_low32`

Tonality is represented by three ordinary 0/1 indicator metrics (major, minor,
modal), not by taking an artificial average over category codes.


## Exact axis-invariance control

The axis report uses the artifact digest repeated across metrics for each
case. The importer requires those digests to agree; inconsistent hashes across
metrics reject the input. It also requires the declared support-axis string in
the observation notes to be consistent within each style/seed block.

Each factor has 27 matched slices: all combinations of the other three factors.
For each slice, the analyzer compares the three complete score digests across
the target factor's low/centre/high levels. A slice is identical only when all
three digests match byte-for-byte; otherwise it is changed. Missing artifact
digests make the slice unavailable, never identical.

This creates an exact negative-control check for styles that declare an axis
unsupported. It does not prove that an unsupported control has no effect on
rendered audio or listener experience, and a digest change on a supported axis
does not prove the musical change is desirable or correctly targeted. Scalar
response surfaces and human/audio qualification remain separate.

A wholly not-applicable metric block is preserved as not_applicable, not
imputed, and not counted as a failed execution. Mixed observed/not-applicable
outcomes, failed rows, and missing rows remain incomplete and suppress all
effect estimates for that block.
