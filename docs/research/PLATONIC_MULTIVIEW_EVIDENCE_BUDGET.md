# Multi-View Evidence Budget Benchmark

**Status:** executable experiment committed; exact-head validation pending  
**Implementation:** crates/core/symthaea-core/examples/platonic_multiview_evidence_budget.rs

## Paradigm shift

Move from “classify this one vector” to “choose the next action under an evidence budget.” A system may answer now, fuse additional views, return a candidate set, or abstain. This experiment quantifies the first two choices and includes an exact-vector semantic-twin control to mark the information-theoretic limit. It does not implement a production active sensor scheduler.

## Protocol

- Five fixed seeds; 32 registered bipolar prototypes at 4,096 dimensions.
- Three independent per-view noise rates (10%, 20%, 30%) and fusion budgets of 1, 3, and 9 observations. Views are independently corrupted from one latent pattern; multi-view cases use a coordinate-wise majority bundle.
- **Known:** query views come from an exact registered latent prototype.
- **Near unknown:** a latent vector is first changed by 2% of coordinates and is not entered in memory; subsequent views are noisy samples of this separate latent vector.
- **Exact-vector twin:** reuses the identical fused query and scores from a known case, while changing only the hidden semantic identity to “unregistered.” Any vector-only acceptance rule must produce the same decision for both.
- The runner emits a fixed grid of score thresholds (0.70 through 1.00) at a fixed top-two margin. Every threshold is reported; none is selected using evaluation results. Rows stay separated by seed, observation noise, view count, and query kind.

The experiment separates two effects: repeated independent views may recover a stable latent vector and distinguish it from a nearby unknown; they cannot distinguish two semantic labels attached to the exact same observable vector. Only an independently authenticated source of identity evidence can resolve that second case.

## Metrics

Per operating point: top-1 correctness for known queries, accepted coverage, accepted-query error risk, a descriptive Wilson 95% upper endpoint, unknown false-accept count/rate and upper endpoint, mean best similarity, top-two margin, majority-fusion time, exact-scan time, stored vector payload, and observed-view payload. Wilson intervals are naive binomial diagnostics because rows share identities and repeated views; they are not cluster-robust or finite-sample deployment certificates. Threshold sweeping on the same output is diagnostic only; future claims require independent calibration, certification and evaluation data.

## Guardrails and falsifiable targets

These are targets, not achieved results:

1. For known noisy queries, more independent views should improve top-1 recall or accepted coverage at matched held-out risk on at least the declared low/moderate-noise regimes.
2. For 2%-corrupted near-unknown latent patterns, multi-view fusion should increase separability of known versus unknown score distributions where signal-to-noise permits it. If it only makes false accepts more confident, that negative outcome must remain visible.
3. For exact-vector twins, acceptance decisions must match 100% for all thresholds and view counts. Any mismatch is a benchmark bug.
4. Report marginal results by seed/noise/view count; no pooled score may hide a failure mode.
5. Require at least five additional independent seeds and larger query counts before using the proposed 5% upper-risk / 70% coverage target as an acceptance claim. Account for identity-level clustering and use a certificate appropriate to the sampling design.
6. Count observations as a resource. Compare decision utility and risk at equal observation budgets before claiming an active-acquisition advantage.

## Industry comparisons

This is a memory/retrieval reliability control, not an ANN database leaderboard. For the latter, use exact search as a ground-truth oracle and compare public ANN baselines using Recall@K vs QPS/latency curves, p95/p99 latency, index/build time, and measured total memory on identical hardware and corpus. The Big ANN Benchmarks methodology separates in-memory and out-of-core tracks: https://github.com/harsha-simhadri/big-ann-benchmarks.

For model comparisons, HDC/VSA must be tested against conventional classifiers on common public tasks rather than only against other VSA variants. The 2025 comparative review covers seven established HDC datasets and 121 UCI classification tasks: https://doi.org/10.1007/s10462-025-11181-2. Selective classification should also be compared with published classwise risk/coverage methods such as one-sided prediction: https://proceedings.mlr.press/v130/gangrade21a.html.

## Interpretation limit

Even an excellent multi-view result does not prove identity. Evidence-source trust, sensor independence assumptions, sensor spoofing, data leakage, temporal correlation and registry update semantics remain separate validation requirements.
