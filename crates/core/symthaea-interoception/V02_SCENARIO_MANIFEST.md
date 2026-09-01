# Affective Emergence v0.2 — Scenario Manifest and Holdout Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines deterministic scenario identity, discovery/confirmation separation, and anti-leakage scenario families for Observational Regulatory Affect v0.2.

## Why scenario identity must be independent

A candidate definition, a scenario, and an analysis rule answer different questions and must have different cryptographic identities.

- candidate identity: what quantity is computed?
- scenario identity: what world/state/input sequence is presented?
- analysis identity: how are candidate outcomes judged?

Changing any one starts a new corresponding evidence identity.

## Proposed ScenarioManifest

Each deterministic scenario should bind at minimum:

- scenario schema/version;
- stable scenario ID;
- cohort class: `Discovery`, `ConfirmatoryHoldout`, or `DiagnosticOnly`;
- exact v0.1 model-semantics version;
- initial native state;
- native dynamics configuration;
- ordered executed drive schedule;
- ordered intervention schedule;
- declared observation/prediction cut points;
- which current/past drive information is visible at each cut point;
- any legitimate predictive cue available through each cut point;
- semantic scenario family;
- paired/twin group ID where applicable;
- declared nuisance-matching fields;
- expected mechanical invariants;
- explicit invalidation/exclusion conditions;
- source generator ID/version when generated;
- generator seed/index where applicable;
- canonical SHA-256.

Semantic arm meaning must remain outside blinded primary execution artifacts.

## Discovery and confirmatory holdout separation

The candidate formula, thresholds, scenario generator version, holdout indices, baseline set, and decision rules must be frozen before confirmatory scenarios are inspected through the candidate pipeline.

A scenario used to:

- choose a formula;
- choose a horizon/discount;
- tune a threshold;
- discover a numerical pathology;
- select a baseline;
- alter an exclusion criterion;

is a discovery scenario and cannot later become confirmatory evidence.

Confirmatory holdout identity should be locked prospectively as either:

1. an explicit list of complete scenario manifests/digests; or
2. a frozen generator version plus immutable seed/index set whose materialized manifests are deterministic.

## No overlap by content, not just by ID

Discovery and confirmatory cohorts must be checked for semantic/content overlap.

Changing only `scenario_id` does not create a novel holdout case.

At minimum compute a `scenario_content_sha256` over all outcome-relevant scenario fields excluding labels whose only purpose is naming/cohort bookkeeping.

The confirmatory cohort must not contain a content digest already used in discovery.

Near-duplicate policy should also be declared. For example, a confirmatory case that differs from a discovery case only in an irrelevant blind code is not independent evidence.

## Required paired scenario families

### S1 — neutral stability

Inside preferred bands, zero drive, no intervention.

Use to test absence of manufactured signal and numerical drift.

### S2 — equal current burden, different observed velocity/history

Match current homeostatic burden while giving different already-observed trajectories.

Use to compare current-state baselines against prefix-causal trajectory forecasts.

### S3 — equal current state, different currently observed drive

At the comparison point, current homeostasis is matched but the already-observed current load differs.

Use current-drive-persistence forecasts while retaining drive magnitude as an explicit nuisance baseline.

### S4 — equal external drive, different internal margin

Apply identical drive to states with different validated viability margins.

Use to show whether candidate values encode regulatory consequence rather than only stimulus magnitude.

### S5 — deterministic recovery

Perturb outside the preferred band, remove load, then allow native recovery.

Use to validate zero-input native forecast self-consistency and sign changes in realized regulatory change.

### S6 — crossed-sign cases

Construct cases required by `V02_TEMPORAL_ALIGNMENT.md`:

- worsening but better than expected;
- improving but worse than expected;
- unchanged current burden but revised future outlook;
- current change with self-consistent unchanged overlapping forecast.

These prevent R1/R2/R3/R4 from collapsing into one generic scalar.

### S7 — identical-prefix / divergent-future twins

Two scenarios must have byte-identical information available through cut point `t` and may diverge only after `t`.

Every `OnlinePrefixCausal` candidate must be exactly identical through `t`.

This is a direct future-information leakage test. A difference before the divergence is a hard failure.

### S8 — future-mutation adversarial twins

Take a completed scenario and mutate arbitrary future drives/interventions/states after a chosen cut point while preserving the prefix supplied to the online candidate.

The candidate value at the cut point must remain unchanged.

Property-test across multiple valid scenarios and cut points.

### S9 — forecast-policy agreement regimes

Construct regimes where kinematic, zero-input native, and current-drive-persistence forecasts are expected to agree approximately or exactly.

Unexpected disagreement is a forecast-model validation failure.

### S10 — forecast-policy disagreement regimes

Construct regimes where native restorative dynamics or ongoing observed drive should cause preregistered differences between policy classes.

No universal winner is assumed.

## Nuisance matching

Scenario pairs should declare which nuisance variables are intentionally matched, such as:

- current weighted homeostatic deviation;
- peak deviation;
- drive magnitude;
- perturbation magnitude;
- elapsed steps;
- number of active deviated channels;
- current state velocity magnitude;
- initial regulatory margin.

A candidate should not receive credit for discriminating paired scenarios when the same discrimination is trivially available from an undeclared nuisance difference.

## Scenario cut-point identity

Many v0.2 hypotheses compare candidate values at particular times.

The cut point must therefore be prospective evidence, not selected after viewing time series.

Each primary comparison should bind either:

- an exact step index;
- a deterministic event rule defined without candidate outputs (e.g. first executed step after a declared intervention);
- or a prospectively defined window and aggregation rule.

Selecting the maximum candidate excursion after the run is exploratory unless that selection rule was itself preregistered.

## Deterministic holdout evaluation

Because the substrate is deterministic, do not claim independent random replication merely because there are many scenario manifests.

Recommended confirmatory summaries include:

- number/fraction of holdout scenarios satisfying a directional relation;
- worst-case signed margin;
- minimum effect margin across a declared core subset;
- equivalence success/failure for neutral controls;
- paired candidate-minus-baseline margin per scenario;
- explicit count/list of failure scenarios;
- coverage of the declared parameter/scenario region.

If a generator is used, its scenario space should be described sufficiently to make coverage claims meaningful.

## Scenario exclusions

A scenario should be excluded only under a preregistered mechanical criterion, such as malformed input, invariant failure, non-finite derived value, or execution/replay mismatch.

A surprising or hypothesis-inconsistent outcome is not an exclusion criterion.

Every scenario in a locked confirmatory cohort must end as one of:

- included;
- excluded with evidence-bearing preregistered reason;
- indeterminate with evidence-bearing reason.

No scenario may disappear from the cohort accounting.

## Cohort manifest

A later `ScenarioCohortManifest` should bind:

- cohort ID/version;
- cohort class;
- ordered scenario digests;
- generator/version and seed/index set if used;
- content-overlap audit against discovery cohort;
- scenario count;
- planned cut points/comparison windows;
- candidate-definition digest set allowed for the cohort;
- analysis-definition digest;
- canonical SHA-256.

A confirmatory evidence capsule should bind the exact cohort manifest digest.

## Claim boundary

A held-out deterministic cohort can demonstrate robustness across prospectively selected cases and parameter regions. It does not by itself establish statistical population generalization, biological emotion, subjective feeling, sentience, or consciousness.