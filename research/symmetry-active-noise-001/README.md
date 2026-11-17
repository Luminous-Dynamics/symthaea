# Symmetry active-experiment noise reference v0.1

First executable uncertainty-safe replay slice for [SYMM-ACTIVE-NOISE-001 #12667](https://github.com/Luminous-Dynamics/symthaea/issues/12667), composed with deterministic minimax planner [PR #12670](https://github.com/Luminous-Dynamics/symthaea/pull/12670).

## The key distinction

A hypothesis may be eliminated only when its complete declared outcome model cannot explain the observation record. Uncertainty can keep more hypotheses alive; it cannot create extra certainty.

V0 implements two separate non-probabilistic regimes:

1. **Bounded per-probe outcomes.** Each hypothesis/probe pair has a finite allowed-outcome set. An observation set retains the hypothesis when it overlaps the allowed set. Repeated observations of the same probe constrain one per-probe outcome through set intersection; they are not treated as independent resamples.
2. **Campaign-persistent shared bias.** Each hypothesis predicts one outcome for every probe under each possible global error state. A hypothesis survives only if one and the same error-state index explains the entire observation history. The nuisance state cannot silently change between probes. This is a structured, correlated-noise subcase, not the full general persistent-noise ODT/ASRN model.

An empty surviving set is reported as model inconsistency, not as proof that no physical symmetry exists.

## Zero-noise bridge

A deterministic problem from the exact active-experiment crate can be lifted into singleton allowed-outcome sets. Its survivor set must match deterministic observation replay exactly. The bridge is tested on a frozen fixture.

## Adversarial controls

Tests cover overlapping bounded predictions, repeated same-probe observations, globally shared persistent error state, observations outside every model, unknown/inadmissible probes, malformed outcome tables, duplicate/empty observation sets, mutated model contracts and forged receipts.

Receipts bind canonical bytes for the complete model and ordered observation history, every intermediate survivor set, and final compatibility. Canonical bytes are versioned but are not cryptographic digests or signatures.

## Reproduction

    cargo fmt --manifest-path research/symmetry-active-noise-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-active-noise-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-active-noise-001/Cargo.toml --all-targets -- -D warnings

## Competitive positioning

This implementation is not a new noisy decision-tree algorithm. Jia et al. (JMLR 2024) already study cost-sensitive optimal decision trees and adaptive submodular ranking with noisy outcomes, including persistent noise where repeating a test gives the same noisy output: https://www.jmlr.org/papers/v25/23-1484.html

The IJCAI 2026 active-feature-acquisition survey frames sequential acquisition as a POMDP and distinguishes embedded, model-based, model-free and hybrid policies: https://ijcai.org/proceedings/2026/860

AFABench (KDD 2026) offers a standardized evaluation framework covering static, myopic and reinforcement-learning acquisition policies: https://doi.org/10.1145/3770855.3817493

The next comparison step is a faithful implementation/adapter for the JMLR noisy ODT/ASRN assumptions and an AFABench-style cost/coverage report. Do not claim superior performance until the same frozen instances, outcomes, priors/noise assumptions, costs and stop rules are used.

## Claim boundary

Finite discrete outcome sets only; no continuous numeric intervals, likelihoods, priors, posterior calibration, independent transient-noise repetitions, POMDP policy, physical actuation, safety authorization, causal inference, or symmetry proof. The receipt establishes model compatibility only.


## Numeric interval extension

The Rust reference also includes an exact fixed-point interval path:
- `FixedIntervalV1` stores closed signed integer bounds at an explicitly declared unit scale. It uses no hidden epsilon and all arithmetic is checked.
- `BoundedIntervalProblemV1` retains a hypothesis only when every observed interval overlaps its declared prediction interval. Repeated readings of one probe intersect the same evidence interval. Once that intersection is empty, the campaign remains inconsistent and still records every subsequent observation in its replay snapshots.
- `SharedAdditiveBiasProblemV1` models one continuous, campaign-wide additive nuisance value. Each observation induces a feasible interval for that bias; a hypothesis survives only if the intersection across all probes and repeated readings is nonempty. It cannot assign a different bias to each probe.

Tests include interval-overlap discrimination, repeated-reading contradiction, one shared bias across probes, fixed-point overflow rejection, and observationally equivalent hypotheses that must remain ambiguous. The numeric scale is part of the caller's problem definition; these intervals are not probabilities or confidence intervals.
