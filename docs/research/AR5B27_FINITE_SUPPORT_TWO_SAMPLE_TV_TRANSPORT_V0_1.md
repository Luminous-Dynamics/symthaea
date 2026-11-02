# AR-5B27 — Finite-Support Two-Sample TV Transport v0.1

**Status:** implementation draft; Rust tests authored but not executed.  
**Tracking:** [AR-5B27 / Issue #10008](https://github.com/Luminous-Dynamics/symthaea/issues/10008)  
**Module:** **crates/domains/symthaea-architecture-engineering/src/finite_tv_two_sample_transport.rs**  
**Branch:** **research/ar5b27-two-sample-tv-transport-v0.1-20261102**  
**Dependency:** AR-5B21 finite-support calibration / PR #9925.

## 1. Research question

How can a confidence interval calibrated under one law be qualified for a target law when the two laws are not known to be identical?

This v0.1 adds a conservative distribution-level certificate for two independently sampled IID complete-path populations on the same frozen finite categorical support. It does not pretend that calibration/confirmation sample independence implies equal distributions. It also does not treat a digest or drift-monitor non-alarm as proof of equality.

Let calibration law \(P\) and target law \(Q\) live on a common support with \(K\) categories. Let \(\widehat P_n\) and \(\widehat Q_m\) be the two empirical laws. The triangle inequality gives:

\[
\operatorname{TV}(P,Q)
\le
\operatorname{TV}(\widehat P_n,\widehat Q_m)
+
\operatorname{TV}(P,\widehat P_n)
+
\operatorname{TV}(Q,\widehat Q_m).
\]

For a finite categorical law, the Weissman-style bound used by AR-5B21 is:

\[
\Pr\!\left(\operatorname{TV}(\widehat P_n,P) >
\sqrt{\frac{K\ln 2+\ln(1/\delta_P)}{2n}}\right)
\le \delta_P
\]

and analogously for \(Q\). By a union bound, the two-sample TV radius

\[
r_{P,Q}
=
\min\!\left\{
1,\,
\operatorname{TV}(\widehat P_n,\widehat Q_m)
+
\sqrt{\frac{K\ln 2+\ln(1/\delta_P)}{2n}}
+
\sqrt{\frac{K\ln 2+\ln(1/\delta_Q)}{2m}}
\right\}
\]

covers \(\operatorname{TV}(P,Q)\) with probability at least
\(1-\delta_P-\delta_Q\), under the declared IID complete-path assumptions.

For any measurable \(g\) with known range \([a,b]\),

\[
\left|\mathbb E_Q[g]-\mathbb E_P[g]\right|
\le (b-a)\operatorname{TV}(P,Q).
\]

Therefore a valid calibration-law interval \([L_P,U_P]\) expands to:

\[
L_Q=\max\{a,L_P-(b-a)r_{P,Q}\}, \qquad
U_Q=\min\{b,U_P+(b-a)r_{P,Q}\}.
\]

The transport event has failure probability \(\delta_P+\delta_Q\). The calibration-interval failure probability remains a separate event; callers composing a null-side guarantee must add every relevant event (including transport) to the null failure budget before conditional alpha spending. Do not silently reuse one delta for two distinct claims.

## 2. Implemented request and report

### Request

The typed request binds:

- immutable transport ID and functional-contract SHA-256;
- calibration and target design identities;
- support and category-order SHA-256;
- calibration and target sample artifact digests;
- observed category counts for both laws;
- separate failure allocations \(\delta_P\), \(\delta_Q\);
- IID-complete-path declarations for both samples;
- independent calibration/target sample declaration;
- pre-outcome-only target reference sample declaration.

The target reference sample is required to be pre-outcome-only. Reusing confirmatory outcomes to estimate transport and then treating the estimate as independent would require a joint or sequential theorem not supplied by this v1.

### Report

The deterministic report contains:

- empirical calibration/target TV distance;
- each side's Weissman concentration slack;
- calibration and target sample sizes and data digests;
- both design identities, support digest and functional contract digest;
- each separate failure allocation and the union-bound transport failure probability;
- total TV radius;
- **INFORMATIVE** or **VACUOUS** status;
- canonical report SHA-256.

A radius of one is vacuous for normalized bounded-function expectation transport. The code reports that state explicitly rather than presenting it as a useful qualification.

### Functional adapter

The function transport_bounded_expectation_interval verifies the transport report and expands a calibration confidence interval using the registered range width. It clips to the functional's valid support and rejects malformed, out-of-range, overflowing, or inverted values.

The output reports the transport failure probability only. It does not claim to include the calibration functional's failure budget; the caller must compose that separately.

## 3. Fail-closed assumptions and limits

V1 requires:

1. The two reference samples are IID complete-path draws within their respective populations.
2. The calibration and target reference samples are independent.
3. Both samples use the exact same finite support and category order.
4. The support contains between 1 and 100,000 categories.
5. Both sample sizes are positive, checked for integer overflow and at most \(2^{53}\) for exact count-to-f64 conversion.
6. The target reference sample is pre-outcome-only and not the confirmatory outcome stream.
7. The registered functional contract supplies a valid finite range \([a,b]\).

These conditions are represented as declarations and digest bindings. The module does **not** authenticate the source data, prove IID sampling, prove sample independence, prove that the support catalogue is complete, or establish that the two design identities correspond to a valid causal estimand.

The radius is distribution-level and therefore applies to all bounded functionals on the common support, including registered functions omitted from a functional list. This is stronger in scope, but often looser than task-specific functional transport. It can be vacuous for large \(K\) or small samples. In those cases the correct result is **VACUOUS**, not “no shift”.

Unsupported in v0.1:

- unbounded or unknown density ratios;
- target support outside the calibration support;
- unobserved or open-ended categorical path support;
- serial/cluster dependence or non-IID data;
- target samples derived from confirmatory outcomes;
- covariate-only or conditional shift without a justified mapping to the complete-path law;
- a guarantee that a non-alarming drift detector excludes arbitrary changes.

## 4. Tests authored

Ten module tests are authored:

1. Two-sample empirical TV plus both Weissman slacks.
2. One-category exact zero TV radius.
3. Bounded expectation interval expansion by \((b-a)r_{P,Q}\).
4. Support mismatch, target non-IID, and target outcome reuse.
5. Invalid error allocations and empty samples.
6. High-dimensional vacuous-radius handling.
7. Tampered report and out-of-range calibration interval.
8. Non-independent calibration/target sample rejection.
9. Non-IID calibration sample rejection.
10. Overflowing functional-range rejection.

**Tests are authored, not run.** No cargo test, cargo check, rustfmt, Clippy, CI, or performance result is claimed.

## 5. Formula-level simulation pilot (not Rust execution)

An independent NumPy formula pilot was run with:

- \(K=8\);
- calibration law uniform over eight categories;
- target law \((0.10,0.10,0.15,0.15,0.10,0.10,0.15,0.15)\);
- true TV distance \(0.10\);
- \(n=m=3{,}000\);
- \(\delta_P=\delta_Q=0.005\);
- 30,000 independent two-sample replications;
- deterministic seed 20261102.

Observed:
- coverage: 30,000 / 30,000 = 100%;
- exact 95% Clopper–Pearson lower confidence limit: approximately 99.9877%;
- mean empirical TV: approximately 0.10019;
- mean total TV radius: approximately 0.18521;
- radius p05 / median / p95: approximately 0.16436 / 0.18502 / 0.20602.

This supports the formula pilot only. It does not validate the Rust implementation, prove source provenance, establish the declared sampling assumptions, or imply the bound is efficient. The observed 100% coverage in a finite simulation is not a proof.

## 6. Industry and research positioning

The certificate complements rather than replaces operational experimentation systems and sequential drift monitors. Staged rollout and metric-triggered rollback are useful for managing deployments, but a non-alarm from a monitor does not prove equality of full path distributions. The finite-support two-sample certificate instead provides an explicit, conservative upper bound on a declared calibration-to-target TV discrepancy, with a named failure event and an explicit vacuity status.

Task-specific transport can be substantially tighter than distribution-level TV when only a few expectations matter. The next research comparison should therefore include:

1. same-target-law-by-design;
2. this finite-support two-sample TV bound;
3. bounded density-ratio transport with explicit positivity and ratio bounds;
4. direct two-sample functional discrepancy bounds;
5. online martingale certificates for time-series/serial dependence, only under their own theorem assumptions.

A 2026 UAI time-series forecasting certificate uses martingale concentration to handle serial dependence and nonstationarity; it is a comparator for temporal risk calibration, not an automatic certificate for arbitrary decision guardrails. A separate 2026 confidence-calibration result studies expectation consistency under covariate shift; its estimand-specific theorem must not be generalized to arbitrary outcomes or harms.

## 7. Next integration work

- Add a typed transport-to-DFCE composition adapter that explicitly unions calibration and transport failure budgets before conditional alpha spending.
- Compare distribution-level TV to direct functional transport on matched decision tasks; report width, power, utility regret, false qualification, and calibration cost.
- Integrate #8795 drift monitoring as a freshness alarm, not a proof of no shift.
- Derive bounded density-ratio transport only where support/positivity and ratio assumptions can be justified.
- Include temporal/cluster dependence as a separate regime with a theorem-supported martingale or cluster-robust certificate.

The objective is not to make every changed target law pass. It is to produce a useful, correctly scoped certificate when assumptions and evidence permit—and a clear non-qualification when they do not.

## 8. Example commands

From the repository root:

~~~bash
cargo test --locked -p symthaea-architecture-engineering finite_tv_two_sample_transport
~~~

Run against the exact PR head once a compatible Rust toolchain is available.
