# SEMI-OBSERVATION-MONTE-CARLO-001 — deterministic joint missingness stress v0.1

**Issue:** [#13957](https://github.com/Luminous-Dynamics/symthaea/issues/13957)  
**Implementation:** tests/observation_process_mnar_simulation.rs  
**Status:** Deterministic integration stress harness authored; exact-head formatting, Clippy, test and reference validation pending.

## Purpose

This is a synthetic counterexample generator for the four-stage observation-process ledger. It holds population truth in the test oracle and exposes only the appropriate evidence to the observation ledger. It does not model a real fab or estimate its probabilities.

The harness composes:
- randomized physical selection with per-stratum inclusion probability;
- K1 qualification MCAR, MAR by observed stratum and MNAR by true outcome;
- arbitrary bounded actual feedback delay;
- finite-capacity CA-OCRC proxy tracking;
- independent Bernoulli audit sampling of the nonqualified population.

## Deterministic random streams

A SplitMix64 test-only generator uses separate deterministic seed streams for:
1. latent population outcomes;
2. physical acquisition selection / acquisition success;
3. K1 qualification;
4. actual physical feedback delay;
5. CA-OCRC proxy-delay tracking;
6. independent audit inclusion.

The six seeds are checked for uniqueness and their manifest is hashed into the independent-audit randomization provenance. Same seed + same regime produces byte-identical input/report digests.

This separation is essential: using one PRNG stream for qualification and audit can accidentally couple the audit sample to the operational missingness mechanism.

## Frozen synthetic regimes

### MCAR
Physical selection is uniform across strata. K1 qualification failure is independent of the true binary loss. This is the reference missingness regime.

### MAR by observed stratum
Physical acquisition is risk-prioritized by an observable low/high tool-chamber stratum. K1 qualification failure also depends on that recorded stratum, but not on the latent outcome after conditioning on the stratum.

### MNAR by outcome
Physical acquisition remains stratum-aware, but K1 qualification failure is directly more likely when the latent loss is high. This creates a deliberately biased qualified-only sample and a false-reassurance test.

The true loss probability differs by stratum (low vs high), so pooled selection/qualification mechanisms can shift the observed population distribution.

## Finite-capacity feedback process

The proxy delay distribution is uniform on {-1,0,1,2,3}. For actual delay d in {0,1,2,3}, the exact survival probability is:

\[
F(d)=P(\bar d \ge d)=\{0.8,0.6,0.4,0.2\}.
\]

Thus the underload parameter is:

\[
\ell_F = F(1)+F(2)+F(3)=1.2 \le K=2.
\]

Each measured outcome receives an independently sampled actual delay and proxy delay. The test-only queue:
- retains admitted slots through the inclusive feedback/proxy-expiry round;
- evicts expired slots before the next round's admission;
- uses PROXY_NOT_TRACKED for proxy delay -1;
- uses CAPACITY_DENIED when K slots are occupied;
- uses TRACKED_RECEIVED when actual delay is within proxy horizon;
- uses PROXY_EXPIRED when actual delay exceeds proxy horizon.

The ledger independently reconstructs peak occupancy and checks it never exceeds K. Unmeasured population rounds carry NOT_APPLICABLE tracking state, not a fake proxy loss.

## Independent audit

The audit is a Bernoulli sample of nonqualified population rows with pi=0.40, driven by the audit-only PRNG stream. Its reference-method digest differs from the operational gauge. The audit never modifies K1 qualification or the online CA-OCRC controller.

The test evaluates the existing Horvitz–Thompson population estimate and conservative Hoeffding interval. It separately computes ground-truth risk using all latent outcomes.

## Metrics and current acceptance checks

Across 64 seed replicates for each regime, the integration test requires:
- 100% containment by the worst-case partial-identification interval;
- at least 90% empirical coverage for the audit Hoeffding interval (a smoke-test threshold with Monte Carlo uncertainty; the preregistered theoretical level remains 95%);
- mean audit-estimate bias below 0.025 in absolute population-risk units;
- reconstructed tracking occupancy no greater than K;
- at least one capacity-denied event in each regime.

A separate MNAR test requires:
- mean qualified-subset risk to understate true population risk by at least 0.02;
- mean independent-audit estimate to be within 0.025 of the true risk.

A deterministic-replay test checks identical input and report digests for repeated seed bundles.

These are synthetic software targets. The 90% coverage smoke gate is not a weakening of the desired 95% statistical benchmark; it accounts for finite Monte Carlo replication count. The measured Monte Carlo coverage and its binomial uncertainty should be reported, not just a boolean pass.

## Limitations and next steps

- Binary 0/1 loss is used in V0.1; add continuous bounded losses and interval scores later.
- The stratum is a synthetic observed covariate, not a real tool/chamber causal label.
- The proxy-delay model is intentionally simple and exactly known, not fitted to a fab delay distribution.
- The randomized selection probabilities are simulation inputs, not empirical fab propensities.
- The audit is independent conditional on the nonqualified population frame; it does not establish reference-lab independence in practice.

Next: expand to confidence-calibrated Monte Carlo reporting, long-tailed bounded delays, zero-positivity strata, calibration-common-cause failures and coupled-RNG negative controls.
