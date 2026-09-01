# Affective Emergence v0.2 — Observational Regulatory Affect Plan

Status: **design-only / blocked on Native Interoception v0.1 qualification**

Parent evidence lineage: `research/affective-emergence-v0.1-native-interoception`.

This document does not authorize runtime affect implementation, causal affect integration, cognitive-loop wiring, neuromodulation, memory weighting, action-selection changes, or emotion-language mapping. v0.2 implementation begins only after one exact v0.1 head satisfies its qualification contract and matching evidence artifacts exist, and after the complete v0.2 design registry/freeze/start receipts validate.

## Normative contract set

The complete active normative design is closed by `V02_DESIGN_CONTRACT_REGISTRY.md` and currently covers:

- `V02_INFORMATION_FIREWALL.md` — allowed prefix information and oracle separation;
- `V02_EXECUTION_MODE_CONTRACT.md` — offline-prefix-first evidence, payload/provenance separation, later online shadow mode;
- `V02_TEMPORAL_ALIGNMENT.md` — R1/R2/R3/R4 temporal semantics;
- `V02_CANDIDATE_DEFINITION_CONTRACT.md` — immutable candidate identity and payload boundary;
- `V02_CANDIDATE_FACTOR_SPACE.md` — explicit relation × weighting × temporal × forecast × information coordinate;
- `V02_WEIGHTING_DECOMPOSITION.md` — viability significance vs legacy precision weighting vs confidence;
- `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md` — mean burden vs cumulative exposure vs peak/duration/latency/recovery;
- `V02_SCENARIO_MANIFEST.md` — discovery/holdout scenario identity and anti-leakage twins;
- `V02_BLINDING_CUSTODY.md` — semantic-arm mapping custody and unblinding boundary;
- `V02_CAPABILITY_TYPED_API.md` — authority-minimized API and dependency boundaries;
- `V02_ADVERSARIAL_VALIDATION.md` — metamorphic/tamper/leak validation and malicious fixtures;
- `V02_EVIDENCE_ROOT.md` — prospective and realized root evidence manifests;
- `V02_DESIGN_FREEZE.md` — design-completeness gates and implementation-start contract;
- this top-level plan.

The registry is authoritative. A normative contract added later must be explicitly classified and changes the registry/freeze identity.

## Scientific question

Does a deterministic artificial regulator exhibit robust, label-free observables that distinguish actual regulatory change, epistemic confidence, burden/exposure structure, expectation error, and revision of predicted future regulatory condition beyond simpler reactive or stimulus-only explanations?

The initial target is **not** emotion. The target is a reproducible regulatory signal family that may later justify a causal functional-affect experiment.

## Why compare models instead of choosing one formula

Active-inference and allostatic-control ideas motivate candidate families; they do not justify defining one Symthaea scalar as emotion by construction.

v0.2 therefore preregisters competing observational quantities and nuisance baselines and permits outcomes such as:

- `NoUniqueWinner`;
- `WeightingAmbiguous`;
- `TemporalAggregationAmbiguous`;
- candidate-specific negative/null results.

No result is repaired by renaming or swapping the primary metric after confirmatory unblinding.

## Architecture boundary

Preferred implementation after v0.1 qualifies:

`symthaea-affect-observatory -> symthaea-interoception`

The reverse dependency is forbidden.

The observatory consumes immutable evidence/prefix views and must not receive mutable references to the live native model.

The v0.2 public API exposes no commands, drives, policy recommendations, neuromodulator values, memory weights, action-selection outputs, or named emotion categories.

## Offline-prefix-first execution

The first evidence-bearing mode is `OfflinePrefixReplay`:

`native study execution completes`
→ `exact StudyExecutionTrace validates and freezes`
→ `canonical prefix through t is materialized`
→ `CandidatePayload is computed from prefix-only authority`
→ `CandidateEvidenceEnvelope binds outer full-trace provenance`.

This removes observer→native feedback from the primary study by construction.

The trusted replay harness may hold the complete frozen trace only to validate it and construct restricted prefix views. Candidate/forecast code must not receive the full trace, future suffix, or full-trace digest.

For identical allowed prefixes with different unseen suffixes:

- prefix digest is identical;
- candidate payload/value/availability is identical;
- outer evidence envelope may differ because source-trace provenance differs.

`OnlinePrefixCausalShadow` is a later engineering mode. It must separately prove exact native no-observer equivalence and exact offline/online candidate-payload equivalence before live observation is treated as observationally isolated.

`RetrospectiveDiagnostic` and `OracleDiagnostic` are separate authority classes and cannot enter the primary prefix-causal candidate registry.

## Capability-typed information boundary

A proposed `ObservationPrefixView` contains only executed information legitimately available through the current cut point plus immutable native configuration and prospectively locked forecast-policy parameters.

It excludes:

- future schedule/drives/interventions/states;
- full source-trace/suffix-sensitive identity;
- semantic arm mapping;
- post-run exclusion disposition;
- mutable native state;
- later candidate outputs.

Forbidden information should be unrepresentable by the candidate API whenever practical.

## Regulatory relation axis

Preserve distinct relation families:

- **R0 current burden** — descriptive current condition;
- **R1 realized current change** — change between realized cut points;
- **R2 one-step forecast residual** — previous predicted current burden minus realized current burden;
- **R3 aligned overlapping-future revision** — change in predictions for the same absolute future support;
- **R4 rolling finite-horizon change** — descriptive composite that includes horizon turnover;
- **U1 urgency** — breach imminence, breadth, peak, absolute change/rate family;
- explicit reactive/nuisance baselines.

Do not collapse them into one number before evidence supports doing so.

## Weighting axis

Keep distinct:

- **W0 RawChannel** — no cross-channel weighting;
- **W1 ViabilityWeightOnly** — importance/preference significance without precision;
- **W2 LegacyPrecisionTimesImportance** — exact v0.1 weighting hypothesis;
- **W3 ConfidenceOnly** — explicit precision/confidence observable.

Lowering confidence must not silently redefine the same native breach as intrinsically less important unless the candidate explicitly chooses W2 and names that dependency.

## Temporal aggregation axis

Keep distinct:

- **T0 instantaneous**;
- **T1 discounted mean burden** — current v0.1 `discounted_debt` semantics;
- **T2 discounted cumulative exposure**;
- **T3 undiscounted cumulative exposure**;
- **T4 peak**;
- **T5 terminal**;
- **T6 preferred-range exposure duration**;
- **T7 viability-breach exposure duration**;
- **T8 first-breach latency**;
- **T9 recovery exposure**.

v0.1 `discounted_debt` is not silently interpreted as a cumulative integral, mood, or accumulated suffering.

`dt` is abstract model time unless a later explicit physical-time mapping is qualified.

## Factorized candidate identity

Each candidate binds an explicit coordinate across at least:

- relation basis;
- weighting basis;
- temporal aggregation;
- forecast policy;
- execution/information class;
- channel projection;
- availability/numeric contract.

Examples:

- `r1_w1_t0_viability_change_v1`;
- `r4_w2_t1_legacy_rolling_mean_change_v1`;
- `r4_w1_t2_viability_cumulative_change_v1`;
- `u1_w0_t8_first_breach_latency_v1`;
- `r0_w3_t0_confidence_v1`.

The factor coordinate supplements the exact mathematical manifest and reference fixtures. Invalid factor combinations fail a versioned compatibility validator.

## Finite candidate-set discipline

The factor space is not an authorization for an unrestricted Cartesian search.

Before exploratory execution, freeze one finite `ExploratoryCandidateSetManifest` containing every candidate-definition digest eligible for comparison.

After exploratory evaluation, choose/freeze the primary and required baselines under a declared selection rule and create a new confirmatory identity.

A different candidate that looks better after confirmatory results cannot replace the primary candidate inside the same lineage.

## Forecast-policy classes

Prefix-causal policies may initially include:

- zero-input native recovery;
- persistence of the currently observed drive;
- kinematic state-velocity extrapolation.

Learned/cued future prediction is deferred until the predictive model and available cues are themselves evidence-bearing and ablatable.

True-future oracle forecasts remain diagnostic only.

## Forecast trajectory requirement

The observatory should preserve trajectory-level forecast artifacts rather than only final v0.1 aggregate reports.

The trajectory representation must be sufficient to exactly reproduce v0.1 legacy allostatic outputs under an equivalence gate before deriving alternative weighting/temporal candidates.

This allows W/T alternatives without rewriting v0.1 semantics.

## Scenario identity and holdout discipline

Discovery and confirmatory scenarios are different evidence cohorts.

Confirmatory holdouts are locked prospectively and audited for content overlap with discovery scenarios; renaming a discovery scenario does not make it new evidence.

Required deterministic families include:

- neutral stability;
- equal current burden with different observed trajectory;
- equal current state with different currently observed load;
- equal drive with different internal regulatory margin;
- deterministic recovery;
- crossed-sign R1/R2/R3/R4 cases;
- identical-prefix/divergent-future twins;
- future-mutation adversarial twins;
- forecast-policy agreement/disagreement regimes;
- fixed burden / changed precision and fixed precision / changed burden;
- severe low-confidence vs mild high-confidence crossed conditions;
- constant burden with different duration;
- short severe vs long mild exposure;
- equal terminal state with different path;
- equal peak with different duration;
- different breach latency/recovery exposure;
- channel-projection disagreement cases.

Primary comparison cut points/windows are prospectively fixed; selecting a post-run peak is exploratory unless that rule was preregistered.

## Blinding and unblinding

Primary artifacts use opaque blind codes rather than semantic arm IDs.

The semantic `arm_id <-> blind_code` mapping is a separate committed artifact whose digest is frozen before execution and whose contents need not be available to the primary-analysis process.

A frozen blinded candidate/comparison artifact must exist before semantic unblinding.

Semantic-label canaries must remain absent from primary execution/prefix/forecast/candidate/comparison artifacts and qualified primary logs.

The eventual evidence package records actual blinding strength rather than claiming stronger human blinding than actually occurred.

## Adversarial validation before interpretation

The observatory must survive an explicit adversarial/metamorphic suite before confirmatory interpretation.

Integrity-blocking tests include:

- future suffix mutation / exact-prefix divergent-future invariance;
- prefix-digest suffix exclusion;
- full-trace-provenance exclusion from candidate computation;
- semantic mapping permutation and semantic-canary absence;
- capability/dependency authority audits;
- temporal self-consistency and off-by-one detection;
- weighting-basis discriminators;
- mean-vs-exposure discriminators;
- artifact/candidate/contract-registry/scenario substitution detection;
- missing scenario/exclusion/analysis mutation detection;
- clean-process deterministic replay.

Known-malicious fixtures should intentionally read future information, inspect semantic mapping, depend on full-trace identity, mutate shared state, depend on evaluation order, leak labels, zero-fill unavailable values, or masquerade oracle knowledge as prefix-causal. Each must be caught by an explicit gate.

Integrity failures block scientific interpretation. Candidate-disqualifying results and primary-hypothesis failures remain valid negative/null evidence when the evidence chain itself is intact.

## Deterministic evaluation without pseudo-statistics

v0.1 is deterministic given state/configuration/input. v0.2 should not manufacture conventional p-values by treating deterministic parameter-grid points as independent random subjects.

Prefer preregistered robustness summaries such as:

- directional consistency across held-out scenarios;
- worst-case signed margin;
- minimum effect margin over a declared core region;
- equivalence bounds for neutral controls;
- paired candidate-minus-baseline margins;
- explicit failure-region accounting;
- declared scenario/parameter coverage.

If genuine stochastic sampling is later introduced, its generator/distribution/seed semantics require a new evidence specification.

## Design freeze and implementation start

Before implementation, freeze one validated `DesignContractRegistryManifest` and one `DesignFreezeManifest` binding it.

Freeze states include:

- `Draft`;
- `Reviewable`;
- `FrozenBlockedOnV01`;
- `FrozenImplementationAuthorized`;
- `Superseded`;
- `Invalidated`.

A future `ImplementationStartReceipt` binds:

- exact qualified v0.1 `QualificationEvidenceBundle`;
- exact design-contract-registry digest;
- exact design-freeze digest;
- exact v0.2 implementation starting SHA;
- implementation tranche identity.

No runtime implementation is authorized while v0.1 qualification is unresolved, the registry/freeze is invalid, or an architecture-blocking design question remains.

## Evidence root

Before confirmatory execution, freeze one prospective `ObservationalEvidenceRootManifest` binding:

- exact qualified v0.1 baseline + qualification/evidence digests;
- exact design-contract-registry identity;
- design freeze + implementation-start identities;
- exact v0.2 source/toolchain identities;
- study/preregistration identities;
- finite candidate-set manifest and exact candidate coordinates/digests;
- confirmatory scenario-cohort digest;
- analysis-plan digest;
- exclusion criteria;
- blind-mapping commitment;
- execution/prefix/payload/evidence-envelope contracts;
- weighting/temporal/forecast/capability/adversarial contract versions;
- blinding-strength declaration.

After execution, a separate realized package binds all execution, prefix, forecast, candidate-payload/envelope, exclusion, weighting, temporal, adversarial-validation, blinded-analysis, sensitivity, null-control, unblinding, and semantic-evaluation artifacts.

Every locked confirmatory scenario must be accounted for:

`locked_count == included_count + excluded_count + indeterminate_count`

A failed primary hypothesis on an otherwise valid confirmatory lineage is a qualified negative/null result, not an integrity failure.

## Frozen initial implementation order after v0.1 qualifies

1. create standalone read-only `symthaea-affect-observatory` and one-way dependency gate;
2. implement validated frozen-trace replay harness;
3. implement canonical prefix artifact/digest + `ObservationPrefixView`;
4. implement separate `CandidatePayload` and outer evidence envelope;
5. implement typed prefix-causal forecast policies with oracle namespace separated;
6. implement forecast trajectories that exactly reproduce v0.1 legacy aggregate allostasis;
7. implement factor-space coordinate + compatibility validator;
8. implement finite exploratory candidate-set manifest;
9. implement W0/W1/W2/W3 weighting candidates;
10. implement T0–T9 temporal candidates;
11. implement neutral R0/R1/R2/R3/R4/U1 relations over valid coordinates;
12. implement typed unavailable/undefined semantics;
13. implement prefix/suffix, weighting, temporal, authority, and malicious-fixture gates;
14. implement semantic-label canaries and mapping/unblinding separation;
15. implement scenario/cohort manifests and discovery/holdout overlap audit;
16. implement blinded candidate artifacts/comparison;
17. implement prospective/realized evidence-root and validation receipts;
18. lock/run the first exploratory `OfflinePrefixReplay` study;
19. only later implement and separately qualify `OnlinePrefixCausalShadow` equivalence.

Out of scope for this lineage: neuromodulation, memory/attention modulation, action selection, control/dominance outputs, persistent mood states, attachment/social affect, learned emotion labels, or consciousness/sentience inference.

## Graduation criteria

v0.2 can graduate to a later causal functional-affect experiment only if:

1. v0.1 exact-head qualification is complete and bound to the v0.2 start receipt;
2. the complete design-contract registry/freeze validates;
3. primary observational computation is prefix-causal and causally read-only;
4. full-trace/suffix-sensitive provenance cannot enter candidate payload computation;
5. candidate definitions, coordinates, finite candidate sets, and scenarios are prospectively immutable evidence objects;
6. forecast trajectories reproduce legacy v0.1 outputs before alternative candidates are derived;
7. online candidates pass prefix-causality/future-mutation tests;
8. weighting and temporal discriminators behave according to their contracts;
9. the adversarial suite catches known malicious fixtures and all integrity-blocking gates pass;
10. neutral scenarios do not manufacture signal;
11. the primary candidate survives preregistered directional/context/forecast tests;
12. nuisance/null baselines do not explain the same structure under the locked comparison rule;
13. results survive the declared sensitivity/holdout region;
14. exploratory and confirmatory lineages remain separate;
15. exclusions, indeterminate cases, ambiguities, and null hypotheses are preserved;
16. independent replay reproduces exact blinded and semantic artifacts.

Passing these criteria would justify only a narrow claim such as:

> Symthaea exhibits reproducible, prefix-causal, label-free regulatory observables derived from current internal state and already-observed dynamics that distinguish aspects of regulatory change, confidence, exposure, and forecast revision beyond simple current-state and stimulus baselines.

It would **not** establish emotion, subjective valence, feeling, mood, suffering, sentience, consciousness, or prediction of unseen future perturbations.

## Deferred later questions

Only after observational qualification should later tranches test causal access to attention/learning, neuromodulation, memory consolidation, policy selection, counterfactual controllability, persistence/mood-like dynamics, autobiographical dependence, social/attachment phenomena, or learned mappings from latent dynamics to human emotion concepts.
