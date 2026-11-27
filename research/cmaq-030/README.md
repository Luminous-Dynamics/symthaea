# CMAQ-030 — Recoverability-Margin Conformal Calibration

**Status:** dependency-free Rust evaluator and deterministic tests authored; exact-head formatting, tests and strict Clippy pending.  
**Relationship:** decision-functional alternative to CMAQ-029's coordinate-wise L-infinity trajectory envelope; feeds CMAQ-027 commitment and CMAQ-028 recovery-option portfolios.  
**Claim ceiling:** marginal finite-sample coverage of the one-sided optimism of a pre-registered recovery-margin block, conditional on exchangeability of independent cluster scores and an independently verified reference-margin pipeline. No hard physical-safety or arbitrary OOD guarantee.

## Research hypothesis

CMAQ-029 calibrates the maximum coordinate error across a complete HDC trajectory block. That is general-purpose but may be unnecessarily conservative for a specific assurance decision.

CMAQ-030 calibrates the dangerous error direction in the quantity the runtime actually uses: the margin of the kth independent recovery option.

At a candidate prefix, each qualified recovery profile has an entry margin. Profiles sharing a failure domain count once; the domain's best profile margin is retained. Sorting distinct failure-domain margins in descending order gives the kth-largest margin for a runtime requiring k independent recovery modes.

- Positive margin: the reachable set remains inside that recovery option's certified entry region.
- Non-positive margin: the option is not admissible at that prefix.
- kth-largest margin: the narrowest margin among the best k distinct failure domains.

For predicted and independently verified reference margins, the one-sided cluster score is:

`S_c = max over cells max(0, predicted_k_margin - reference_k_margin)`

Only over-optimism is penalized. Pessimistic predictions are not treated as dangerous errors.

The finite-sample quantile q yields the lower margin:

`M_lower_k = M_predicted_k - q`

If this lower bound is positive over every cell in the frozen block, the candidate may commit within the calibrated scope. If any cell is non-positive, the block is not admitted; the runtime must replan, switch, probe, or return UNKNOWN according to its higher-level policy.

## Connection to 2026 prior work

This is deliberately compared against direct safety-margin calibration rather than presented as a first:

- Chang and Ahmed (2026), Barrier Function Conformal Safety Clearance Certification with CVaR for Driving Trajectory Selection, calibrate the gap between planned barrier margin and realized clearance after planner selection on 300 frozen nuPlan sessions. Their published page reports 93.3–96.7% exact-clearance coverage at a 90% target and an 87.3% rate of nonnegative clearance certificates in one comparison.
- CERT-Replan (CoRL 2026) uses horizon-indexed conformal uncertainty and calibrated barrier risk as an event trigger for replacing a local planning mode.
- CAFT (ICML 2024) provides simultaneous trajectory-level bands.
- CMAQ-029 provides cluster-level coordinate-wise trajectory envelopes.

CMAQ-030's hypothesis is narrower: calibrating the kth independent recovery-option margin may be more decision-relevant and less conservative than calibrating every state coordinate when the downstream decision is whether a specific recoverability portfolio remains available.

This is a testable hypothesis, not an achieved advantage.

## Independent recovery-option semantics

For each cell, the candidate and reference blocks must use the same profile IDs and failure-domain mapping. Multiple profiles sharing one failure-domain digest count once; within one domain the best margin is retained.

The profile set and certificate manifest are frozen by digest. A changed recovery portfolio, certificate, failure-domain mapping, plant, context, metric, action/query selection policy, or horizon invalidates the calibrator.

The candidate receives predicted option margins and profile/failure-domain identities from already qualified recovery-profile metadata. It does not receive reference margins, family labels or calibration residuals. The reference side is evaluator-only and must be produced by an independent reachability/certificate verifier or authenticated replay with exact state evidence.

Each evaluator-only reference block is explicitly bound to the digest of the exact candidate margin block whose prediction error it scores. Candidate and reference blocks must also use identical cell identities and profile/failure-domain manifests. A correct residual attached to a different trajectory is not valid calibration evidence.

## Cluster-block calibration

One calibration unit is one independent mechanism/environment cluster. Its score is the maximum one-sided optimism over all pre-registered rollouts, action/query cells and horizon prefixes.

For n calibration clusters and miscoverage alpha = a/b:

`k = ceil((n + 1) * (b - a) / b)`

The kth ordered cluster score is q. If k > n, return InsufficientCalibrationClusters; do not fabricate a finite margin. At 95% nominal target, 19 independent clusters are the mathematical minimum to obtain a finite non-randomized rank, not a scientifically sufficient study size.

Training, scale-validation, calibration and held-out evaluation clusters are disjoint. Within-cluster episodes do not increase n.

## Deterministic witnesses

1. Multiple profile IDs sharing a failure domain count as one independent option.
2. One shared domain cannot satisfy k=2, even with many profiles.
3. Nineteen calibration clusters at 95% return the expected upper optimism radius; eighteen are insufficient.
4. Underprediction produces zero one-sided optimism penalty.
5. One overoptimistic prefix controls the cluster-block maximum.
6. A positive lower kth margin permits commitment within the exact block.
7. A non-positive lower margin blocks commitment and identifies the first failing prefix.
8. Unknown previous execution status blocks all new commitment.
9. A changed recovery portfolio or profile manifest invalidates the envelope.
10. Candidate and reference profile/failure-domain manifests must match.
11. Training/calibration/evaluation leakage and duplicate cluster IDs are rejected.
12. Evaluator reference margin receipt is bound to the exact candidate margin-block digest.

## Matched benchmark matrix

Compare at identical candidate trajectories, profile sets, independent clusters, action/query portfolio, horizon and miscoverage target:

- CMAQ-029 coordinate-wise L-infinity cluster trajectory envelope;
- CAFT simultaneous trajectory conformal;
- Chang/Ahmed post-selection barrier-clearance conformal;
- CERT-Replan horizon-indexed barrier-risk/event-trigger method;
- uncalibrated point portfolio margin;
- CMAQ-027 single-profile recoverability;
- CMAQ-028 k-domain recovery portfolio.

Report:
- cluster-level simultaneous coverage of the kth recovery margin;
- false positive portfolio-admission rate;
- lower margin and conservatism;
- percentage of prefixes retaining at least k independent domains;
- unnecessary switch rate and high-performance utilization;
- single-domain fault survival;
- recovery time/energy and task utility;
- calibration/inference latency and all-in memory.

## Claim boundary

The guarantee is marginal for a new independent cluster block under exchangeability of one-sided optimism scores and a frozen candidate/reference pipeline. It assumes the reference margin and recovery-option identities come from independently authenticated recovery-profile evidence. This crate does not prove that the reference margin is physically sound, establish statistical independence of failure domains, authenticate certificates, or guarantee distribution-shift coverage.

A positive calibrated lower margin cannot override a hard certificate-scope mismatch, a violated physical invariant, or an unknown prior execution outcome.
