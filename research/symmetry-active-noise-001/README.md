# Symmetry active-experiment noise reference v0.1

First executable uncertainty-safe replay slice for [SYMM-ACTIVE-NOISE-001 #12667](https://github.com/Luminous-Dynamics/symthaea/issues/12667), composed with deterministic minimax planner [PR #12670](https://github.com/Luminous-Dynamics/symthaea/pull/12670).

## The key distinction

A hypothesis may be eliminated only when its complete declared outcome model cannot explain the observation record. Uncertainty can keep more hypotheses alive; it cannot create extra certainty.

V0 now has two categorical and two numeric, non-probabilistic observation models:

1. **Bounded categorical outcomes.** Each hypothesis/probe pair has a finite allowed-outcome set. An observation set retains the hypothesis when it overlaps the allowed set. Repeated observations of the same probe constrain one per-probe outcome through set intersection; they are not treated as independent resamples.
2. **Campaign-persistent categorical state.** Each hypothesis predicts one outcome for every probe under each possible global error state. A hypothesis survives only if one and the same error-state index explains the entire observation history. The nuisance state cannot silently change between probes. This is a structured, correlated-noise subcase, not the full general persistent-noise ODT/ASRN model.
3. **Bounded fixed-point numeric intervals.** Closed signed-integer intervals are used with a required unit/scale identifier. A hypothesis survives only when its declared interval overlaps every observed interval. Repeated readings of a probe intersect the same evidence interval. Arithmetic overflow and reversed intervals fail closed.
4. **Campaign-persistent additive numeric bias.** Each hypothesis has nominal predictions and one shared additive bias interval for the whole campaign. Every observation narrows the same feasible bias interval; it cannot use one offset per probe or a fresh offset per repeated reading.

An empty surviving set is reported as model inconsistency, not as proof that no physical symmetry exists.

## Zero-noise bridge

A deterministic problem from the exact active-experiment crate can be lifted into singleton allowed-outcome sets. Its survivor set must match deterministic observation replay exactly. The bridge is tested on a frozen fixture.

## Identifiability frontier and adversarial controls

Before a policy is selected, pairwise reports determine whether the complete admissible probe family can separate each candidate pair. Bounded categorical pairs return disjoint-outcome probes or explicit ambiguity. Shared-persistent pairs return one cheapest probe witness per global state or an identical-signature collision. This is a full-family identifiability diagnostic, not a guarantee that one fixed probe separates every candidate.

Tests cover overlapping categorical and numeric predictions, repeated same-probe intersections, globally shared categorical error states, one shared numeric bias, observations outside every model, unknown/inadmissible probes, malformed units and outcome tables, duplicate/empty observation sets, interval overflow, mutated model contracts, forged receipts, and observationally equivalent hypotheses.

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

Exact finite categorical sets and bounded fixed-point numeric intervals only. No likelihoods, priors, posterior calibration, independent transient-noise repetitions, POMDP policy, physical actuation, safety authorization, causal inference, or proof of physical symmetry. The receipt establishes compatibility with the declared model only; it does not establish the correctness of that model or the physical system.


## Numeric interval extension

The Rust reference also includes an exact fixed-point interval path:
- `FixedIntervalV1` stores closed signed integer bounds at an explicitly declared unit scale. It uses no hidden epsilon and all arithmetic is checked.
- `BoundedIntervalProblemV1` retains a hypothesis only when every observed interval overlaps its declared prediction interval. Repeated readings of one probe intersect the same evidence interval. Once that intersection is empty, the campaign remains inconsistent and still records every subsequent observation in its replay snapshots.
- `SharedAdditiveBiasProblemV1` models one continuous, campaign-wide additive nuisance value. Each observation induces a feasible interval for that bias; a hypothesis survives only if the intersection across all probes and repeated readings is nonempty. It cannot assign a different bias to each probe.

Tests include interval-overlap discrimination, repeated-reading contradiction, one shared bias across probes, fixed-point overflow rejection, and observationally equivalent hypotheses that must remain ambiguous. The required quantity_unit_id binds the integer endpoint scale into canonical problem bytes (for example, millimetre-integer); changing the unit changes the receipt contract. These intervals are not probabilities or confidence intervals.


## Identifiability frontier (stacked v0.1 increment)

Before selecting a policy, the library can now test whether a hypothesis pair is distinguishable at all under the declared probe family.

- **Bounded per-probe model:** a single admissible probe whose allowed outcome sets are disjoint is a constructive separation witness. If every admissible probe overlaps, the pair is observationally indistinguishable under this probe family.
- **Shared persistent-state model:** for every possible global campaign state, the pair must differ on at least one admissible probe. The report records the cheapest witness probe for each state. If any state gives the same complete signature, it emits that collision and reports unresolved identifiability.

This is an upper-bound diagnostic using the complete available probe family, not a policy claiming that one experiment identifies all candidates. It prevents an optimizer from being blamed for failing an impossible instance, and makes probe-family design a first-class benchmark variable.


## Numeric identifiability witnesses

The numeric models expose the same pairwise identifiability question:

- `bounded_interval_pair_identifiability` reports admissible probes whose predicted intervals are disjoint, or an explicit unresolved pair when all admissible intervals overlap.
- `shared_bias_pair_identifiability` reports probes where nominal predictions differ. Because one common additive bias shifts both hypotheses equally, that difference is invariant to the nuisance value. If all admissible nominal predictions agree, the pair is unresolved under that probe family.

These are pairwise, full-probe-family diagnostics. They do not imply a single globally optimal experiment policy or guarantee identification of every candidate at once.


## Latest focused qualification

At exact code head `1dca4ffecf7de3490c65b4c0541aedb43ab12897`, the pinned Rust 1.96 workflow passed formatting, **22 tests (0 failures)** and strict Clippy. The later README-only correction does not change the tested source. Repository-wide CI/workflow checks are separate; the draft-runner admission guard remains an inherited failure on the stacked maintenance base.
