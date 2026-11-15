# AOBENCH-013C: reproducible Bernoulli Monte Carlo calibration

**Date:** 2026-11-15  
**Parent:** [AOBENCH-013 / #12197](https://github.com/Luminous-Dynamics/symthaea/issues/12197)  
**Per-episode evaluator:** [PR #12216](https://github.com/Luminous-Dynamics/symthaea/pull/12216)  
**Suite scorer:** [PR #12219](https://github.com/Luminous-Dynamics/symthaea/pull/12219)  
**Implementation issue:** [#12248](https://github.com/Luminous-Dynamics/symthaea/issues/12248)

## Purpose

A time-uniform confidence-sequence formula is a mathematical guarantee under declared assumptions. A benchmark still needs a reproducible harness to measure its finite-sample behavior, policy stopping, power, false-positive rate, coverage and cost under concrete scenarios.

This increment adds a deterministic synthetic generator and one reference policy for the narrow Bernoulli v0. It does not create another inference method.

## Reproducibility contract

- PRNG: local `splitmix64-v1`, pinned in source.
- Policy: `balanced-anytime-v1`, pinned in source.
- Root seed: frozen in the manifest.
- Episode seed: derived from root seed, stable FNV-1a hash of stratum ID and episode index; does not depend on stratum vector order.
- Manifest digest: SHA-256 over a canonical sorted representation of scenario, stratum probabilities, episode counts, seed and version IDs.
- Episode record: ID, derived seed, complete agent-visible trace and evaluator report.
- Output: stratified summaries; no pooled headline rate that hides a difficult effect-size stratum.

The PRNG is deterministic simulation machinery only. It is not cryptographic and must never be used for real treatment assignment.

## Policy boundary

The balanced policy interleaves one control and one treatment observation, updates only from observations it has already generated, and consults only the public Bernoulli scenario plus the recomputed interim confidence boundary. Hidden arm probabilities are used only by the outcome generator and by the evaluator after the trace is closed.

Synthetic trials have verified assignment, delivery, isolation and observation status. Negative evidence fixtures remain the responsibility of the per-episode evaluator tests; this Monte Carlo reference policy is not an authentication service.

## Frozen scenario strata

A useful initial calibration manifest should keep these separate:

| Stratum | Example probabilities | Purpose |
|---|---|---|
| Null | 0.50 vs 0.50 | Directional false-positive rate |
| Equivalence interior | 0.50 vs 0.53 | No-material-effect decision |
| Just inside margin | 0.50 vs 0.59, with `delta=0.10` | Margin discrimination |
| Just outside margin | 0.50 vs 0.61 | Hard low-power / near-boundary behavior |
| Moderate benefit | 0.30 vs 0.50 | Power at a clear material effect |
| Large benefit | 0.10 vs 0.90 | Positive-control stopping |
| Moderate harm | 0.50 vs 0.30 | Negative-direction power |
| Large harm | 0.90 vs 0.10 | Negative-control stopping |

Do not summarize the whole suite without reporting these strata separately. In particular, 80% power cannot be demanded at the exact decision boundary without specifying a separated design alternative and sufficient sample budget. Define the minimum practically relevant effect `delta` and power alternatives such as `delta + epsilon`, `1.5*delta` and `2*delta` separately.

## Example invocation

The API accepts a frozen scenario and a set of named probability strata. Example only:

```rust
use symthaea_organizational_closure::{
    BernoulliSequentialScenario, BernoulliSimulationManifest,
    BernoulliSimulationStratum, run_bernoulli_simulation_suite,
};

let scenario = BernoulliSequentialScenario::new(
    0.05,       // family error for two arms
    0.10,       // minimum material risk difference
    12_000,     // total attempted samples
    6_000,      // maximum attempted samples per arm
    1, 12_000,  // per-sample cost and total cost ceiling
    1, 12_000,  // per-sample risk and total risk ceiling
);
let manifest = BernoulliSimulationManifest::new(
    0xA0B0_C0D0_E0F0_1234,
    scenario,
    vec![
        BernoulliSimulationStratum::new("null", 0.50, 0.50, 2_000),
        BernoulliSimulationStratum::new("benefit", 0.30, 0.50, 2_000),
        BernoulliSimulationStratum::new("harm", 0.50, 0.30, 2_000),
    ],
);
let result = run_bernoulli_simulation_suite(&manifest)?;
```

The returned records preserve every trace and report. The suite scorer reports null false-positive rate, qualified power on materially non-null effects, final and all-look coverage, correct-claim rate, process qualification and mean resource use, each with separate denominators and 95% Wilson intervals.

## How much Monte Carlo is enough?

A rough independent-episode planning approximation for a two-sided 95% proportion interval with half-width `e` is:

```text
n ≈ 1.96^2 * p * (1-p) / e^2
```

At a rate near 5% or 95%, an absolute half-width near one percentage point takes roughly 1,825 independent episodes. A half-width near half a percentage point takes roughly 7,300. Use exact Wilson interval planning in the frozen preregistration; the approximation is for scale only.

Report actual counts and intervals. Do not stop collecting episodes merely because a favorable estimate appears. Freeze sample counts and seeds before confirmatory runs. This harness produces per-stratum Wilson intervals; family-clustered uncertainty and independent seed-block replication remain follow-on requirements.

## Comparison with AOBENCH-017

[AOBENCH-017 / #12242](https://github.com/Luminous-Dynamics/symthaea/issues/12242) is the broader bounded-outcome, family-wise inference programme. Its proposed per-arm alpha spending `alpha_arm/[n(n+1)]` differs from the Bernoulli v0's `6*alpha_arm/(pi^2*n^2)`; both schedules sum to the arm-level error budget. They have different finite-sample widths.

The next method comparison should include:
- the current Bernoulli Hoeffding schedule;
- the telescoping `1/[n(n+1)]` schedule;
- Howard-style stitched / mixture confidence sequences;
- empirical-Bernstein or betting confidence sequences where their assumptions match;
- e-process / likelihood-ratio methods;
- naive repeated fixed-time intervals as an intentionally invalid negative control.

Do not call one method more powerful until the same generated episodes, stopping policy, outcome support, alpha allocation and uncertainty analysis are used. The literature shows that empirical-Bernstein / betting confidence sequences can be materially tighter than simple Hoeffding sequences for bounded data, while preserving time-uniform guarantees under their stated assumptions. <https://doi.org/10.1093/jrsssb/qkad009>

## Limitations

- Only iid Bernoulli outcomes within arms.
- The policy is a simple balanced reference, not an adaptive model-based experiment designer.
- No empirical type-I, power or coverage results are asserted by adding the harness.
- No support for multiple endpoints, family-wise query allocation, Gaussian/count data, drift, interference, noncompliance estimands or informative missingness.
- No evidence receipt is authenticated by this simulation package.
- No real-world causal validity or external generalization claim follows.

Exact-head nested-workspace compile/test/Clippy, Monte Carlo calibration, and comparison against the broader AOBENCH-017 methods remain pending.
