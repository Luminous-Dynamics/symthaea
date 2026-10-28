# SCI-XENV-001: Cross-Environment Law Invariance Under Interventions

**Status:** proposal / protocol specification  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8705  
**Evidence class:** design proposal; no experiment results claimed  
**Version:** 0.1.0

## 1. Research question

Can a scientific-discovery system recover a mechanism that transports across environments while distinguishing (a) context-specific parameters, (b) spurious correlations, (c) genuine changes in the governing mechanism, and (d) worlds where available evidence cannot identify a unique law?

The target is not merely a low-residual expression. It is a structural hypothesis with a declared domain, assumptions, allowed environment-specific quantities, and predictions that survive prespecified changes in context and interventions.

This protocol composes with the shared scientific-discovery evaluator and PhysicsIR rather than creating another general evaluator or physical model authority.

## 2. Motivation and prior work

Recent research makes progressive and multi-experiment law discovery an active frontier:

- **AI-Newton**, accepted by *Physical Review Research* on 9 October 2026, studies concept-driven general physical laws from raw multi-experiment mechanics data: https://doi.org/10.1103/dvvc-4rsr
- **Chain of Symbolic Regression (CoSR)**, accepted on 6 October 2026, studies progressive symbolic knowledge units and broader law generalization: https://doi.org/10.1103/f2kp-pp4q
- **Science-Gym** joins data collection, experiment design and equation discovery in one benchmark loop: https://doi.org/10.1007/s10994-025-06914-x
- **ERBench** targets equation recovery and stresses robustness to dimensionality, sample size, sampling distribution and sampling domain: https://arxiv.org/abs/2606.09276
- A 2026 survey identifies search complexity, overfitting and high-dimensional scaling among the field's continuing challenges: https://doi.org/10.1007/s11831-026-10681-w

These publications motivate the benchmark; their reported numbers are not transferable to this protocol without matched implementations, datasets, information access and budgets.

## 3. Scope and composition

This benchmark is a distinct layer over existing work:

- **#8487**: umbrella scientific-discovery recovery protocol.
- **#6879 PhysicsIR**: canonical model, units, dimensions, assumptions, coordinate frames and provenance.
- **#8286 active probe design**: intervention contracts, acquisition strategies, query costs and exact small-instance oracles.
- **#8588 regime transitions**: cases where the true governing structure changes.
- **#8607 unknown noise**: cases where observation-process misspecification is distinct from law misspecification.
- **#8667 dimensional invariance**: unit/representation transformations.
- **#8671 progressive discovery**: intermediate-knowledge validity and negative transfer.

Do not copy their evaluators or create a second causal discovery engine. In this protocol, intervention selection is an experimental condition and reusable policy adapter, not a new causal inference implementation.

## 4. Benchmark tracks

### Track A — Shared mechanism, context-specific parameters

Generate multiple environments sharing a structural mechanism while allowing declared quantities such as initial conditions, boundary conditions, calibration offsets, scale parameters or nuisance distributions to vary.

The evaluator labels which quantities should remain invariant and which legitimately vary. A system must not get credit for forcing all parameters to be equal, nor for treating every environment as a separate unrelated fit.

### Track B — Spurious pooled association

Construct worlds where a compact pooled expression predicts well under discovery correlations but fails when a covariate distribution or confounder changes. Include cases in which an observational relation reverses or vanishes after a context shift.

The score distinguishes prediction on observed environments from recovery of a mechanism that transports. A pooled low error is not enough.

### Track C — Intervention transport

Expose only explicitly permitted interventions and their observed outcomes to the policy. Score predictions on interventions and contexts withheld from discovery. Use distinct action types for observation, hard intervention, and mechanism replacement; they must not be conflated.

Intervention outcomes support mechanism discrimination only when the candidate worlds are distinguishable under the available action set. An intervention's cost and feasibility are public scenario inputs; its hidden outcome is not.

### Track D — Genuine mechanism change

Include worlds where the generating structure truly changes across environments. A policy that always reports a shared law must incur a visible false-invariance penalty. Reuse the regime-transition fixture family from #8588 when compatible.

### Track E — Non-identifiable transport

Create pairs of worlds that are observationally equivalent under all permitted data in a scenario. The correct response is an equivalence-class/set-valued answer, a justified next experiment, or explicit unresolved status. A false confident singleton is a primary failure, not a stylistic defect.

### Track F — Representation and units

Apply transformations covered by #8667 and PhysicsIR where relevant. Compare semantic/equivalence-class outcomes rather than formula-string equality. Dimensional compatibility is necessary where applicable, but is never a substitute for empirical or causal validity.

## 5. Synthetic world design

Start with a deterministic, evaluator-sealed generator. Each world manifest records a schema version, structural-family ID, environment configuration IDs, intervention/action manifest, sample design, measurement-process configuration, seed, and content digest. The policy-facing manifest must omit truth labels and any target-derived metadata.

Include multiple families rather than repeatedly sampling the same simple polynomial:

1. Dimensionally valid mechanics-like laws with contextual parameters.
2. Additive and multiplicative mechanism families.
3. Nonlinear interaction terms.
4. Stable mechanisms with nuisance-only shifts.
5. Confounding and correlation changes.
6. Real mechanism-change negatives.
7. No-effect intervention controls.
8. Observationally equivalent / intervention-insufficient worlds.

Separate exploratory debugging worlds from confirmatory worlds by structural family and environment design. A final environment must not be a random row split of an environment already visible during training.

The first slice should be deliberately bounded and executable. Arbitrary PDEs, complex simulator-to-real transfer, and empirical claims across all sciences are outside v0.1.

## 6. Policy information boundary

The discovery policy may observe only:

- public variable semantics, units, domains and scenario constraints;
- public environment/context descriptors;
- candidate actions and declared per-action costs;
- observations from interventions it has actually executed;
- its own candidate equations, scores and selection history;
- protocol-authorized fit/calibration data.

The policy may not access the generating equation, hidden mechanism or regime labels, latent confounders, evaluator eligibility labels, unobserved intervention outcomes, held-out-environment outcomes, or sealed final confirmation data.

No evaluator field may be passed to the policy to control equation search, grammar expansion, parameter fitting, acquisition, stopping or abstention. Check this by interface design and adversarial negative tests, not by relying on a written promise.

## 7. Baselines and fairness

Register pinned versions and configurations for the subset of tasks each method supports:

1. Per-environment symbolic regression.
2. Pooled symbolic regression.
3. Shared-structure / environment-specific-parameter regression.
4. Sparse regression/SINDy when its assumptions match the generator.
5. Invariant prediction or causal baselines only on tasks where their formal assumptions hold.
6. Fixed, random, information-gain, cost-normalized information-gain and exact small-instance oracle intervention policies, reusing #8286 adapters.
7. Progressive-discovery baselines and multi-experiment concept-driven methods, if their runnable interfaces can be compared fairly.
8. Negative controls: no-intervention, intentionally misspecified, and always-invariant policies.

Equalize the initial observations, candidate-action set, maximum query/intervention budget, observation semantics, and resource-accounting rules. Report separately the number and type of observations, interventions, candidate evaluations, solver calls, wall time, memory and compute cost. Record failed runs and abstentions.

A published result is not a reproduced baseline. When a method cannot be run under the same contract, classify it as related work and do not place it on the same leaderboard.

## 8. Metrics

### Primary outcomes

- Structural recovery by functional/mechanism equivalence class.
- Held-out-environment prediction error.
- Held-out intervention-effect prediction error.
- False-invariance rate when the governing structure changes.
- False confident-transport rate when a candidate fails in a new environment.
- Correct unresolved/set-valued response rate on non-identifiable worlds.
- Query/intervention cost to the first independently validated transportable hypothesis.

### Secondary outcomes

- In-domain interpolation and true-range extrapolation, reported separately.
- Per-environment error and worst-environment regret.
- Context-specific parameter recovery and uncertainty, only when identifiable.
- Credible-set or interval coverage and size, where the method returns these.
- Abstention, unsupported-confidence and out-of-support rates.
- Query selection regret versus the exact small-world oracle.
- Resource cost: candidate attempts, compute/wall time, memory, energy if measured, and human review time if measured.
- Intermediate model validity and negative transfer, composed with #8671.

Publish every family and environment denominator, paired-seed intervals, distributions, exclusions and failure reasons. A single composite score may be reported only as a secondary summary alongside the full vector and its declared weights.

## 9. Preregistered controls and gates

Before confirmatory runs, freeze all family splits, world seeds, action budgets, scoring rules, eligibility definitions, abstention semantics, stopping criteria and exclusions.

Required controls:

- **Positive invariant control:** the same structural mechanism really does transport.
- **Nuisance-shift control:** context-specific quantities vary but the shared mechanism does not.
- **Spurious-association control:** pooled observational fit does not survive the declared shift.
- **Mechanism-change control:** the true law changes in one environment.
- **No-effect intervention control:** intervention machinery does not manufacture an effect.
- **Non-identifiability control:** available observations/actions are insufficient to select a unique mechanism.
- **Leakage control:** deliberately inject an evaluator-only field through a test adapter and assert that the policy interface rejects it.
- **Determinism control:** same versioned inputs reproduce the same world and manifest digest.
- **Representation control:** equivalent expressions and supported unit transforms preserve semantic scores.

Use at least 60 held-out worlds across six or more structural families and 20 paired seeds per world for confirmatory comparative results, harmonizing with the wider programme's target where feasible. Increase replication if a preregistered precision/power analysis requires it. This is a target for the eventual comparison, not a statement that such runs have occurred.

Do not set a performance-improvement percentage before measuring a matched baseline. First establish a reproducible reference and show the Pareto frontier across structural recovery, transport, false invariance, non-identifiability handling, cost and runtime.

## 10. Reproducibility receipts

Each run must bind the protocol version, source/model commit, grammar/operator registry, structural-family and seed manifest, environment/action manifest, public observation digest, policy output digest, evaluator version and score artifact digest. Keep the policy receipt separate from evaluator-only truth and score receipts. Use standard SHA-256 and existing repository receipt infrastructure; do not design custom cryptographic schemes.

Receipts establish artifact linkage and replay inputs, not that the physical world is true, that source execution was authenticated, or that a law is scientifically valid. State those trust boundaries explicitly.

## 11. Acceptance and claims policy

The benchmark is ready for a confirmatory comparison only when:

- generator known-answer and negative controls pass independent checks;
- no policy-facing API exposes evaluator-only fields;
- the same budget and information contract is used for comparable baselines;
- non-identifiable, unresolvable, missing, and failed outcomes remain explicit in denominators;
- final confirmation uses untouched environments and/or interventions;
- exact-head executable checks and reproducibility receipts exist.

Do not claim the system has discovered a causal mechanism because it found a low-residual equation. Do not claim cross-domain scientific truth from synthetic data. Physical-science claims require independent data, domain expertise, calibrated measurement, and appropriate validation.

## 12. Expected contribution

The intended contribution is a diagnostic benchmark for a common failure mode: a symbolic model may fit several experiments while failing to distinguish a reusable mechanism from context-bound correlation. The protocol makes the distinction measurable and creates a clean bridge between equation discovery, active experiment selection, uncertainty, dimensional semantics, and held-out regime testing—without prematurely claiming that any one algorithm wins.
