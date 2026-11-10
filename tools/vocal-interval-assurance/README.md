# Vocal interval assurance (VIA-001)

A dependency-free Rust contract for conservative decisions over uncertain acoustic measurements.

## Why this exists

An acoustic analyzer may report a point estimate inside a permitted range even though its uncertainty extends beyond that range. Accepting the point estimate alone can turn an unresolved measurement into a false pass.

VIA-001 uses three outcomes:

- **VERIFIED_COMPLIANT** — the complete uncertainty interval satisfies the requested target or collateral constraint.
- **VERIFIED_NONCOMPLIANT** — the complete interval is disjoint from the permitted region.
- **INDETERMINATE** — the interval overlaps a decision boundary. This is never a pass; obtain better evidence or report the uncertainty.

## Decision rules

For a target response region [a, b] and an observed uncertainty interval [l, u]:

- Compliant when l >= a and u <= b.
- Noncompliant when u < a or l > b.
- Indeterminate otherwise.

A target interval must exclude zero, so a no-op cannot satisfy a target-response rule.

For a collateral limit ±e:

- Compliant when l >= -e and u <= e.
- Noncompliant when u < -e or l > e.
- Indeterminate otherwise.

At the control level, a definite noncompliance dominates; absent a definite failure, any indeterminate metric makes the whole trial indeterminate. Only all-compliant metrics produce a verified-compliant result.

## Provenance and fail-closed behavior

Every metric observation binds:

- stable metric ID and unit;
- finite, ordered lower and upper delta bounds;
- interval-family ID and family-wise coverage declaration;
- uncertainty method ID and calibration receipt ID;
- source-case ID and trial ID;
- frozen input-manifest digest.

One evaluation requires exactly the target plus every declared collateral metric, each exactly once. All observations must bind the same source case, trial, input manifest, interval family and family-wise coverage. Unexpected, missing, duplicated, stale, incompatible or incomplete observations return an error and cannot produce a compliant report.

The evaluator preserves the per-metric decision and the upstream uncertainty/calibration references. It does not calculate an interval or certify the statistical truth of an upstream coverage claim.

## Statistical requirements for interval producers

The upstream analyzer is responsible for the interval method. A credible confirmatory protocol should:

1. Keep calibration and confirmatory source cases disjoint.
2. Use an independent sampling unit such as singer × song × recording session, rather than treating highly correlated frames from one recording as independent.
3. Account for simultaneous coverage across the target and every collateral metric, using a justified family-wise method or a joint/max-statistic bootstrap.
4. Freeze the uncertainty method, calibration receipt, analyzer artifact and input manifest before confirmatory scoring.
5. Report compliant, noncompliant and indeterminate counts separately. Publish the indeterminate fraction rather than dropping unresolved cases.

The family-wise coverage value is a producer declaration carried for traceability. This crate checks that it is present, consistent and meets the requested minimum; it cannot establish that the interval-generation procedure achieves that coverage.

## Integration

- Observation adapters in #11402 can emit the interval records after independent acoustic analysis.
- VCRB-001 #11401 supplies target and collateral metric definitions.
- VNOC-001 #11420 calibrates no-op tolerances; it is not itself an interval estimator.
- Held-out vocal qualification remains a separate requirement in #11425.
- The outer benchmark can aggregate the returned trial reports with summarize_decisions.

## Run

```sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-interval-assurance/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-interval-assurance/Cargo.toml
```

## Claim boundary

Synthetic tests establish only the evaluator's interval-containment, three-way decision, provenance and aggregation behavior. They do not qualify an acoustic analyzer, renderer, voicebank, uncertainty method, dataset, naturalness or SOTA performance.
