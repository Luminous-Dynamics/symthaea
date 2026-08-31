# Affective Emergence v0.2 — Observational Regulatory Affect Plan

Status: **design-only / blocked on Native Interoception v0.1 qualification**

Parent evidence lineage: `research/affective-emergence-v0.1-native-interoception`.

This document does not authorize code execution, causal affect integration, cognitive-loop wiring, neuromodulation, memory weighting, action-selection changes, or emotion-language mapping. v0.2 begins only after one exact v0.1 head satisfies its qualification contract and matching evidence artifacts exist.

## Scientific question

Does a deterministic artificial regulator exhibit robust, label-free observables that distinguish improvement from deterioration in its predicted future regulatory condition, above simpler reactive or stimulus-only explanations?

The initial target is **not** emotion. The target is a reproducible latent regulatory signal whose structure could later justify testing a functional-affect interpretation.

## Why compare models instead of choosing one formula

A classic free-energy account proposes valence as the negative rate of change of free energy (Joffily & Coricelli, 2013, PLoS Computational Biology, doi:10.1371/journal.pcbi.1003094). Recent work continues to connect emotion regulation with retrospective and prospective free-energy minimization and allostatic control (Caria & Pezzulo, 2026, Neuroscience & Biobehavioral Reviews, doi:10.1016/j.neubiorev.2026.106639).

However, empirical work also cautions that affect does not universally reduce to a simple progress-prediction-error computation. Therefore v0.2 must preregister multiple candidate observables and null baselines rather than retrospectively selecting whichever signal looks most emotion-like.

Recent interoceptive work also supports keeping affect measurement separate from action at first: metabolic interoceptive rewards can alter affect without a correspondingly clear effect on action (Fleming et al., 2026, Biological Psychology, doi:10.1016/j.biopsycho.2025.109187).

## Architecture boundary

Preferred implementation after v0.1 qualifies:

`crates/domains/symthaea-affect-observatory`

Dependency direction:

`affect-observatory -> symthaea-interoception`

Forbidden dependency direction:

`symthaea-interoception -> affect-observatory`

The observatory must consume immutable evidence artifacts (`StudyExecutionTrace`, registered forecasts, blinded study artifacts) and must not receive mutable references to the live native model.

The v0.2 public API must expose no commands, drives, policy recommendations, neuromodulator values, memory weights, or action-selection outputs. It is telemetry-only.

No `EmotionCategory`, `Fear`, `Joy`, `Sadness`, `Grief`, `Anger`, or equivalent named-emotion primitive is allowed in the causal or metric API.

## Candidate family A — prospective regulatory improvement

Let `D_t` be the dynamics-aware discounted allostatic debt computed from the state at step `t` under a preregistered forecast configuration and declared future-drive assumption.

Primary candidate:

`prospective_debt_improvement_t = D_{t-1} - D_t`

Interpretation is intentionally neutral:

- positive: predicted future regulatory burden improved;
- negative: predicted future regulatory burden worsened;
- near zero: predicted burden changed little.

This is the closest v0.2 analogue to the negative derivative family of valence theories, but it must remain a **candidate observable**, not a definition of emotion or valence.

## Candidate family B — reactive improvement baseline

Let `H_t` be current weighted homeostatic deviation.

`reactive_improvement_t = H_{t-1} - H_t`

This baseline asks whether prospective forecasting contributes anything beyond simply noticing that the current state improved or worsened.

A v0.2 claim about anticipatory regulatory organization fails if Candidate A cannot outperform or qualitatively distinguish itself from this baseline in preregistered anticipation tests.

## Candidate family C — kinematic forecast baseline

Let `K_t` be discounted allostatic debt from the existing kinematic velocity extrapolator.

`kinematic_improvement_t = K_{t-1} - K_t`

This control tests whether the dynamics-aware forecast adds structure beyond linear extrapolation of measured velocity.

## Candidate family D — forecast-surprise observable

Where a prior forecast can be compared with subsequently realized regulatory burden, define a signed forecast residual:

`forecast_residual = realized_burden - previously_predicted_burden`

This is not treated as valence. It is included because expectation-update accounts of affect are plausible competitors. The analysis must test whether regulatory improvement and forecast surprise explain distinct variance/conditions rather than silently collapsing them into one signal.

## Null / nuisance baselines

At minimum preregister:

1. **current-state level** — `-H_t`; a static good/bad-state scalar;
2. **drive magnitude** — norm/aggregate magnitude of the imposed external drive;
3. **absolute state velocity** — magnitude of native channel velocity irrespective of sign;
4. **phase-shuffled derivative** — candidate derivatives computed after preregistered within-arm temporal shuffling;
5. **sign-inverted candidate** — falsification control for directional hypotheses.

A candidate should not be interpreted as meaningful merely because it correlates with stimulus intensity or static regulatory health.

## Regulatory urgency — separate from improvement

Do not call this `arousal` in v0.2.

Preregister a separate **regulatory urgency** family:

- `breach_imminence`: monotonic transform of `first_breach_step` when a breach is forecast;
- `forecast_peak_deviation`;
- `unique_breached_channels`;
- `absolute_debt_change = |D_t - D_{t-1}|`.

The purpose is to test whether direction-of-change and urgency are empirically separable dimensions.

A high-urgency improving trajectory and a high-urgency deteriorating trajectory should be representable without changing the sign convention of the improvement candidate.

## Explicitly deferred: controllability / dominance

Do not derive a control/dominance variable in v0.2.

A defensible controllability signal requires an explicit set of alternative available actions/policies and counterfactual forecasts of their regulatory consequences. The v0.1 native regulator does not yet provide that action-space semantics.

A later tranche may define something like:

`counterfactual_regulatory_efficacy = debt(no_action) - min_a debt(action_a)`

but only once the available action set and counterfactual model are themselves evidence-bearing and ablatable.

## Minimal experimental worlds

### OAR-001 — stationary neutral

Start inside preferred bands with zero drive.

Expected:

- reactive improvement approximately zero;
- prospective improvement approximately zero;
- urgency approximately zero;
- no candidate should manufacture oscillatory affect-like structure.

Failure here blocks interpretation.

### OAR-002 — matched current state, opposite trajectory

Construct two arms with the same current homeostatic report but opposite recent/future regulatory trajectories.

Expected:

- static state-level baseline cannot distinguish arms;
- trajectory candidates distinguish improving from deteriorating arms;
- prospective candidate must retain the declared sign under parameter sensitivity.

### OAR-003 — anticipatory divergence

Use matched current state/history but different preregistered future-drive assumptions in the dynamics-aware forecast.

Expected:

- reactive baseline remains matched at the comparison instant;
- prospective candidate differentiates the future-risk conditions;
- removing prospective information abolishes that differentiation.

This is the key allostatic-vs-homeostatic test.

### OAR-004 — recovery after perturbation

Apply a bounded intervention, then remove external load.

Expected:

- deterioration signal during worsening phase;
- sign reversal during recovery;
- convergence toward near-zero change after stabilization;
- no requirement for long-lived persistence or mood-like behavior.

### OAR-005 — equal drive, different internal context

Apply the same external drive to two validated starting states with different regulatory margins.

Expected:

- drive-magnitude baseline is identical;
- candidate regulatory observables can differ because internal consequences differ.

This is required before claiming the observable is more than stimulus coding.

### OAR-006 — forecast-model comparison

Run the same states through kinematic and dynamics-aware forecast bases.

Expected outcome is **not** preregistered as “dynamics-aware must win” universally. Instead preregister conditions where the native dynamics should matter (e.g. recovery outside preferred ranges) and conditions where the two methods should agree approximately.

Unexpected disagreement in simple linear regimes is a model-validation failure, not an affect result.

## Discovery / confirmation split

Use two non-overlapping stages.

### Exploratory stage

Allowed:

- inspect candidate time-series shape;
- estimate viable effect-size thresholds;
- identify numerical pathologies;
- refine forecast horizons and sensitivity regions.

Forbidden:

- later relabel exploratory data as confirmatory.

Any changed formula, threshold, horizon, baseline, exclusion rule, or analysis procedure requires a new prospective confirmatory study identity.

### Confirmatory stage

Must use `EvidenceRunClass::Confirmatory` and the qualified study-level evidence path inherited from v0.1.

The blinded metric artifact must be frozen before semantic-arm unblinding.

## Primary confirmatory hypotheses — proposed, not yet locked

Do not lock these until v0.1 qualifies and exploratory pilots establish numerically sane threshold ranges.

Candidate set:

- **H1 neutrality:** stationary-neutral arms remain within a preregistered near-zero envelope for prospective and reactive change candidates.
- **H2 directional ordering:** improving trajectory > matched neutral > deteriorating trajectory for prospective-debt improvement.
- **H3 anticipatory specificity:** under matched current homeostasis, prospective-debt improvement separates future-risk arms while reactive improvement remains within a declared equivalence tolerance.
- **H4 non-stimulus specificity:** equal-drive/different-context arms produce different prospective regulatory observables despite identical drive magnitude.
- **H5 prospective ablation:** removing future-state information selectively removes anticipatory differentiation without deleting ordinary homeostatic deviation.
- **H6 model robustness:** qualitative H2/H3 ordering survives a preregistered parameter region rather than one default point.
- **H7 baseline competition:** the primary prospective candidate must provide preregistered discriminative structure not reproduced by static state level, drive magnitude, or phase-shuffled controls.

A null on H3 or H5 is especially important: it would argue against treating the proposed signal as specifically allostatic/anticipatory.

## Multiple-candidate discipline

Do not crown a winning observable based only on in-sample effect size.

Before confirmatory work, preregister:

- primary candidate;
- secondary candidates;
- null baselines;
- exact metrics;
- effect-size/equivalence thresholds;
- parameter region;
- ranking rule or model-comparison criterion;
- treatment of ties and contradictory findings.

If exploratory data are used to choose the primary candidate, all confirmatory data must be newly generated.

## Parameter sensitivity

At minimum vary, prospectively:

- recovery rate;
- forecast horizon;
- discount factor;
- step duration within stable constraints;
- preferred-band width;
- viable-band width;
- channel importance weighting;
- drive magnitude;
- perturbation duration.

Report qualitative stability regions and failure regions. Do not report only the best-performing parameter point.

## Evidence artifacts proposed for v0.2

The observatory should produce immutable, versioned artifacts such as:

- `ObservationalAffectMetricDefinitionManifest`;
- `CandidateTimeSeriesArtifact` keyed only by blinded arm code during primary analysis;
- `CandidateComparisonReport`;
- `SensitivitySurfaceSummary`;
- `NullControlReport`;
- `ObservationalAffectQualificationReceipt`.

All derived artifacts should bind:

- exact v0.1 source/model-semantics identity;
- exact v0.2 observatory source identity;
- study preregistration SHA-256;
- execution trace SHA-256;
- exclusion-decision SHA-256;
- candidate-definition schema/version;
- analysis version;
- raw input artifact hashes.

## No causal feedback invariant

The first v0.2 implementation must mechanically demonstrate that candidate observables cannot alter the native execution trajectory.

Preferred gate:

Run the exact same locked v0.1 study twice:

1. without the observatory;
2. while the observatory computes every candidate online/read-only or from the completed trace.

Require the complete native `StudyExecutionTrace` to remain byte-for-byte / structurally identical.

If observation changes execution, v0.2 fails its foundational isolation gate.

## Graduation criteria

v0.2 can graduate to a later **causal functional-affect** experiment only if:

1. v0.1 exact-head qualification is complete;
2. observational calculations are causally read-only;
3. stationary-neutral behavior does not manufacture signal;
4. the primary candidate survives preregistered directional, anticipatory, context, and ablation tests;
5. null/stimulus baselines do not explain the same structure adequately under the preregistered comparison rule;
6. the result survives the declared sensitivity region;
7. exploratory and confirmatory data remain separate;
8. exclusions and null hypotheses are preserved in the evidence lineage;
9. independent replay reproduces the exact metric artifacts.

Passing these criteria would justify a narrow claim such as:

> Symthaea exhibits a reproducible, label-free regulatory observable that tracks changes in predicted future viability and contains anticipatory information beyond current homeostatic deviation and stimulus magnitude.

It would **not** establish emotion, subjective valence, feeling, sentience, or consciousness.

## Deferred later questions

Only after observational qualification should later tranches test:

- causal access to attention or learning;
- neuromodulatory coupling;
- memory consolidation effects;
- policy selection;
- counterfactual controllability;
- persistence/mood-like dynamics;
- autobiographical dependence;
- social/attachment phenomena;
- learned mappings from latent dynamics to human emotion concepts.
