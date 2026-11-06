# SPECTRA Paired CRN Contrast Diagnostics v0.1

**Status:** deterministic pairing validator and empirical variance diagnostic; Rust tests declared, not executed.
**Research parent:** #10787
**Method ID:** spectra-paired-crn-contrast-variance-diagnostic-v1

## Motivation

SPECTRA's CRN allocator can request every action in a scenario at the same replication index with a shared seed. This creates an opportunity to compare matched action outcomes directly.

For two actions a,b under scenario s and replication r:

d[r] = Y[a,s,r] - Y[b,s,r]

If a qualified shared additive shock g[s,r] is truly common to both actions:

Y[a,s,r] = μ[a,s] + g[s,r] + ε[a,s,r]

Y[b,s,r] = μ[b,s] + g[s,r] + ε[b,s,r]

then d[r] = μ[a,s] - μ[b,s] + ε[a,s,r] - ε[b,s,r], and the common shock cancels.

**A matching seed is necessary for this paired experiment but does not prove the additive common-shock decomposition.** The caller must provide a separately qualified coupling contract. Arm-specific reactions to the seed, nonmatching random streams, missing pair rows or a wrong simulator model invalidate this interpretation.

## New October 2026 research track

Zierahn, Koolen, Agrawal, Katsimerou & van der Hoeven, *Best Arm Identification for Bandits with Shifting Means*, arXiv:2610.10488 (7 October 2026):
https://arxiv.org/abs/2610.10488

Their model permits an adversarially varying common mean shift while pairwise arm gaps remain stable. The paper shows GLRT stopping, including Track-and-Stop, can fail in that setting and proposes ISM, with delta-correctness and a matching-up-to-constants worst-case lower bound under its bounded-mean/sub-Gaussian assumptions.

This PR is not an implementation of ISM and does not inherit its theorem. It is a CRN simulator diagnostic for a stronger, explicitly paired observation design. The benchmark must keep the two settings distinct.

## Contract and validation

A frozen contract declares:
- policy version and coupling-contract digest;
- explicit shared-coupling-qualified flag;
- complete action and scenario universe;
- common declared utility range.

Raw observations retain:
- action ID;
- scenario ID;
- zero-based replication index;
- coupling identity;
- raw target utility.

The module fails closed unless:
- at least two frozen actions and one scenario are present;
- IDs are unique and the utility range is finite and ordered;
- every scenario × replication index contains exactly one observation for every frozen action;
- all actions in a matched scenario × replication share the same coupling identity;
- a coupling identity is not reused for two replication indices in a scenario;
- replication indices are contiguous from zero;
- every scenario has at least two paired replications;
- every observation is finite and inside the frozen utility range.

The coupling ID is a matching field, not a cryptographic attestation. The contract digest and qualification flag are caller-supplied scope declarations; provenance authenticity remains outside this helper.

## Diagnostic outputs

For each scenario and each pair a < b:
- paired mean difference;
- marginal sample variance for each action;
- sample variance of paired differences;
- independent-variance approximation Var(a)+Var(b);
- paired-to-independent variance ratio;
- empirical pair correlation;
- descriptive sqrt(Var(d)/n).

A ratio below one indicates that the observed pairing reduced empirical variance relative to the independence approximation. A ratio above one means pairing was harmful for that pair. The empirical variance is descriptive, not an inferential confidence interval. The standard-error field is explicitly labelled diagnostic.

The code uses online Welford moments for univariate and bivariate statistics and canonical order for actions, scenarios, pairs and replication indices.

## Rust implementation

File: tools/spectra-uvsd-fit/src/paired_contrast.rs

- Complete pairing and coupling-identity validation.
- Rejects missing, duplicate, noncontiguous and reused coupling observations.
- Computes all pairwise contrasts for each scenario.
- Calculates paired and independence-approximation variances and empirical correlation.
- Emits a deterministic receipt and supports independent recomputation replay.
- Does not alter inference, allocator priorities, stopping, confidence bounds or production score receipts.

## Tests declared

- A shared additive per-replication shock cancels, with near-zero paired difference variance.
- Common offsets varying by replication leave pairwise contrast moments invariant (within floating-point tolerance).
- Candidate and input-row permutations preserve the receipt.
- Missing action, mismatched coupling identity, reused coupling identity and replication-index gaps fail closed.
- Unqualified coupling, missing digest and out-of-range utility fail closed.
- Replay accepts exact diagnostics and rejects changed moments.

The Python mirror exercised a three-action/six-replication case. The three paired-to-independent variance ratios were approximately 2.9e-33, 4.2e-33 and 5.6e-33 for the synthetic perfect-common-shock fixture; common-shift invariance, input-order invariance and invalid-pairing controls passed. This is algebraic smoke, not Rust execution, inferential coverage or a performance result.

Rust tests are declarations only. cargo, rustc and rustfmt are unavailable in the current environment; compile, tests, format, clippy and CI remain unverified.

## Follow-on statistical work

Before any paired contrast can replace marginal action intervals:
1. Specify the actual simulator/generator filtration and shared-shock model.
2. Prove a confidence sequence for the paired difference under adaptive pair selection and the allowed dependence.
3. Allocate error probability simultaneously over every frozen pair × scenario claim, time and any selected objective.
4. Preserve source/action/scenario/replication/coupling IDs in replayable raw receipts.
5. Test arm-specific perturbations and "same seed, no common shock" negative controls.
6. Compare paired CS, marginal CS and ISM only in matched assumptions.

## Benchmark

- static stationary BAI;
- static common intercept;
- per-replication common additive shifts with stable gaps;
- adversarial shifting means as defined by Zierahn et al.;
- smooth common drift and abrupt common shifts;
- arm-specific drift (negative control);
- same-seed but action-dependent noise transformations;
- partial pairing and missing rows.

Compare ISM, Track-and-Stop/GLRT (negative where its assumptions fail), uniform, SPECTRA independent-cell anytime bounds, and eventually paired-contrast CS after proof.

Measure fixed-confidence false-winner/anytime false-stop, samples-to-correct-decision versus K/Δ/σ²/U, paired variance reduction, unsupported-pairing rejection, cumulative regret-to-stop, and p50/p95/p99 allocation/replay runtime.

No SOTA or confidence claim is made by this PR.
