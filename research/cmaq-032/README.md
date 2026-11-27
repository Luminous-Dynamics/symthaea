# CMAQ-032 — Recoverability calibration to anytime-monitor bridge

CMAQ-032 closes the data-contract gap between:

- **CMAQ-030:** one-sided calibration of dangerous optimism in the (k)-th largest recovery-option margin across distinct failure domains; and
- **CMAQ-031:** immutable-reference, anytime-valid predictive-rank monitoring of independent cluster-block scores.

## What is implemented

CMAQ-030 now retains its exact per-cluster optimism scores in `CalibrationResult.cluster_scores`. Each score carries the independent cluster ID, score units, reference block-manifest digest, exact candidate-margin-block digest, and reference-margin receipt digest.

The bridge crate depends on the actual CMAQ-030 and CMAQ-031 crates. `build_calibration_rank_reference` takes:

1. the real CMAQ-030 `CalibrationResult`;
2. a `CalibrationBinding` containing profile, manifest and scope identities plus the exact calibration spec; and
3. one `ClusterRankAttestation` for every calibration cluster.

It creates a CMAQ-031 `CalibrationRankReference` without recomputing or relabelling scores. It rejects missing/extra/duplicate cluster identities, duplicate tie keys, invalid digest syntax, manifest/spec mismatches, and any attested score or source-binding mismatch. The returned reference can be passed directly to `AnytimeCalibrationMonitor::new`.

## Trust boundary — deliberately limited

The bridge is a typed consistency checker, **not a cryptographic evidence verifier**. It checks that receipt identifiers are SHA-256-shaped and that the attested fields exactly match the score inventory. It does not:

- verify signatures, certificates, trust roots, revocation or authority;
- prove that an aggregate-score receipt cryptographically covers the score and all source inputs;
- prove that a tie-break key came from an independent cryptographically secure RNG;
- prove independence of environment/mechanism clusters; or
- turn exchangeability into an assumption-free statement under arbitrary distribution shift.

Those properties must be established by a separately governed ingestion and attestation boundary. In particular, random tie-break receipts must be independently generated and authenticated under the same rule for calibration and deployment. A deterministic lexicographic fallback is not an acceptable substitute when ties are possible.

## Exact-head checks

The dedicated workflow runs format, tests and strict Clippy for all three local crates. Passing these checks establishes Rust build and contract conformance only. It does not establish that the upstream attestor exists, that signatures are verified, that calibration clusters are independent, or that the monitor has adequate power on a real deployment stream.

## Benchmark programme

The next empirical gate must evaluate the whole path, not only helper functions:

| Dimension | Required measurement |
|---|---|
| Handoff correctness | 100% exact score/cluster/provenance correspondence; zero acceptance of missing, duplicated, altered or extra attestations |
| Null behaviour | At least 10,000 independent null replications at several stream horizons; empirical false-alarm rate and 95% binomial confidence interval against the pre-registered (alpha) |
| Drift power | Detection probability and median / 90th / 95th percentile detection delay under upper-tail, lower-tail, dispersion, center-concentration and mixed shifts |
| Calibration size | Curves over independent calibration cluster counts, including the finite-sample rank floor |
| Recovery continuity | Alarm-to-fallback time, recovery-option availability, and whether a separately qualified mode exists before the current profile is quarantined |
| Runtime cost | p50 / p95 / p99 monitor update latency, memory, allocations and sustained throughput on pinned hardware |

Comparators should include: fixed-threshold monitoring, rolling-window empirical breach rates, a confidence-sequence risk monitor, a single-feature predictive-rank martingale, and the four-feature fixed-bet portfolio here. A naive product of repeated fixed-reference conformal p-values may be retained only as an explicitly invalid negative control: repeated reuse makes those ranks dependent.

The principal target is not simply “detect more shifts.” It is **detect meaningful calibration failures early while respecting the pre-registered anytime false-alarm budget and preserving safe fallback availability**. Detection power is bounded at fixed calibration size; larger calibration samples and clearly stated shift assumptions must be part of every result.

## Research position

The predictive-rank martingale is established prior work, not CMAQ-032 novelty. See:

- Kuang & Xia, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536) (2026).
- Fang, [Anytime Valid Conformal Prediction on Streaming Time Series](https://arxiv.org/abs/2611.20905) (2026).
- Chang & Ahmed, [Barrier Function Conformal Safety Clearance Certification with CVaR](https://arxiv.org/abs/2608.26533) (2026).

The contribution here is the typed, fail-closed handoff from a decision-relevant recoverability-margin calibration to a fixed-reference online monitor, with provenance retained across the boundary.
