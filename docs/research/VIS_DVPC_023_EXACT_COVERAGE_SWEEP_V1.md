# VIS-DVPC-023: Exact coverage sweep and target-shift calibration controls v1

**Status:** deterministic, exact-distribution mechanism benchmark. It is not a production calibration certificate or a real-trace performance claim.

## Why this work exists

VIS-DVPC-022 implements an authorization contract that uses a one-sided Clopper-Pearson upper bound for Bernoulli risk and a Hoeffding lower bound for bounded decision value. Correct formulas are necessary, not sufficient: we need evidence that the implementation achieves its stated coverage under known assumptions, and explicit negative controls showing what breaks when source evidence is incorrectly treated as target evidence.

This work calculates coverage from the exact binomial probability mass function. It avoids Monte Carlo noise for the included Bernoulli family. It does not establish coverage for arbitrary non-Bernoulli distributions, dependent observations, adaptive re-use, or shifted target populations.

## Research anchors

- [Anytime-Valid Inference for Online Ranking of LLMs (ICML/PMLR 2026)](https://proceedings.mlr.press/v306/gu26l.html) develops e-process-based online tests with family-wise error control under adaptive sampling and repeated monitoring. Our sweep is fixed-sample; it must not claim anytime validity.
- [Anytime-Valid Evidence for Prespecified Predictive Corrections (2026)](https://arxiv.org/abs/2608.08174) studies source-to-target predictive corrections and emphasizes that evidence for a correction does not, by itself, identify the mechanism of a shift. The DVPC negative control makes the same boundary operational: a source-domain bound is not automatically target-calibrated.
- [Autonomous Science Bench 0.1](https://www.autonomous-science-bench.ai/) evaluates closed-loop discovery and held-out learning as separate gates under budgeted campaigns. Its pilot includes six tasks and three trials per task and cautions that small score differences are not yet meaningful; our benchmark similarly separates mechanism checks from claims about real system capability.
- [AutoSciBench (2026)](https://arxiv.org/abs/2610.05140) studies automatically generated and adaptively refined scientific-agent tasks to reduce benchmark saturation and shortcuts. The next step after this frozen grid is to grow the grid and challenge families rather than overfit to a handful of fixtures.

## Exact coverage definition

For risk observations, with n independent Bernoulli trials, true rate p and observed failure count K, the one-sided Clopper-Pearson upper bound U(K) is found from the binomial tail equation:

P at p=U(K) of [X <= K] = alpha_risk.

The exact grid coverage is the sum of BinomialPMF(k; n, p) over all k for which U(k) >= p.

For bounded value observations in [0,1], the lower confidence bound is:

LCB = sample_mean - sqrt(log(1/alpha_value) / (2n)).

On the included Bernoulli subclass, its exact coverage is computed by summing the probabilities of all success counts for which LCB <= true_mean.

The fixture uses alpha_risk = 0.025 and alpha_value = 0.025. Their sum is at most a declared joint alpha of 0.05, so the Bonferroni/union-bound lower guarantee for the paired claims is at least 95%, provided the marginal procedures and sampling assumptions hold. The computed grid minimum is reported separately; do not confuse it with the theorem's assumption-bound guarantee.

## Frozen input and outputs

- tests/fixtures/vision-dvpc-023-grid.tsv freezes 117 (n, p) input points.
- Sample sizes: 20, 40, 58, 59, 71, 72, 100, 120 and 250.
- Truth probabilities: 0, 0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95, 0.99, 0.999 and 1.
- The scorer emits 1,521 paired coverage cells: each sample size crossed with risk truth and value truth from the 13-point probability grid.
- Each cell reports the exact risk coverage, exact value coverage and a conservative joint lower bound.
- A separate shift control estimates, exactly under the source distribution, the probability that a source-calibrated risk UCB from n=72, p_source=0.01 would even cover p_target=0.20. The result is labelled TARGET_CALIBRATED_FALSE. It is a deliberately non-transportable illustration, not a way to repair or authorize the shifted model.

The fixture digest is verified before compilation. The standalone scorer uses no third-party dependencies.

## Acceptance gates

1. Every risk grid cell has coverage at least 1 - alpha_risk within numerical tolerance.
2. Every value grid cell has coverage at least 1 - alpha_value within numerical tolerance.
3. Every paired cell's conservative joint coverage lower bound is at least 0.95.
4. Zero-failure risk boundary is exact: 71 observations are insufficient for a 5% cap at alpha 0.025; 72 suffice.
5. Source-to-target negative control demonstrates that nominal source coverage does not transfer automatically to a material target shift.
6. Probability mass sums to one, edge probabilities 0 and 1 behave correctly, malformed grids fail closed, and output is bit-for-bit deterministic for the fixed build and platform.
7. Frozen input bytes match their digest and the focused Rust tests and scorer run to completion.

## Boundaries and next step

The exact calculation covers this declared Bernoulli family and frozen grid. It does not establish validity for dependent samples, arbitrary bounded-valued distributions, nonstationary processes, optional stopping, or many simultaneous slices and repeated campaigns. For these, preregister the alpha allocation or use an appropriate sequential method, and test coverage over a broader generator with independent confirmation scenarios.

A source/target shift should result in denial or insufficient evidence unless a declared transport/calibration method has its own valid target evidence. A digest proves byte identity, not independence, authenticity, or scientific truth.

Do not claim SOTA, real-world calibration, transportability, safety, or generalization from this synthetic exact-distribution sweep. Its immediate claim is narrower and useful: the bound implementation's coverage mechanics are checked against known truth over a reproducible grid.
