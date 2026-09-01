# Affective Emergence v0.2 — Observational Regulatory Affect Plan

Status: **design-only / blocked on Native Interoception v0.1 qualification**

Parent evidence lineage: `research/affective-emergence-v0.1-native-interoception`.

This document does not authorize runtime affect implementation, causal affect integration, cognitive-loop wiring, neuromodulation, memory weighting, action-selection changes, or emotion-language mapping. v0.2 implementation begins only after one exact v0.1 head satisfies its qualification contract and matching evidence artifacts exist, and after the design/start receipts defined below are valid.

The companion contracts are:

- `V02_INFORMATION_FIREWALL.md` — prefix-causal information and oracle separation;
- `V02_TEMPORAL_ALIGNMENT.md` — R1/R2/R3/R4 temporal semantics;
- `V02_CANDIDATE_DEFINITION_CONTRACT.md` — immutable candidate identity;
- `V02_SCENARIO_MANIFEST.md` — discovery/holdout scenario identity and anti-leakage twins;
- `V02_BLINDING_CUSTODY.md` — semantic-arm mapping custody and unblinding boundary;
- `V02_CAPABILITY_TYPED_API.md` — authority-minimized API and dependency boundaries;
- `V02_ADVERSARIAL_VALIDATION.md` — metamorphic/tamper/leak validation matrix and malicious fixtures;
- `V02_DESIGN_FREEZE.md` — design-completeness gates and implementation-start contract;
- `V02_EVIDENCE_ROOT.md` — prospective and realized root evidence manifests.

If this overview conflicts with a stricter companion contract, the stricter contract governs. The intended design-freeze process should eliminate such conflicts before implementation is authorized.

## Scientific question

Does a deterministic artificial regulator exhibit robust, label-free observables that distinguish actual regulatory change, expectation error, and revision of predicted future regulatory condition beyond simpler reactive or stimulus-only explanations?

The initial target is **not** emotion. The target is a reproducible regulatory signal family that may later justify a causal functional-affect experiment.

## Why compare models instead of choosing one formula

A classic free-energy account proposes valence as the negative rate of change of free energy. More recent active-inference work connects emotion regulation with allostatic control and prospective/retrospective inference. These ideas motivate candidate families; they do not justify defining one Symthaea scalar as emotion by construction.

v0.2 therefore preregisters competing observational quantities and nuisance baselines and permits `NoUniqueWinner` as a valid result.

## Architecture boundary

Preferred implementation after v0.1 qualifies:

`symthaea-affect-observatory -> symthaea-interoception`

The reverse dependency is forbidden.

The observatory consumes immutable evidence/prefix views and must not receive mutable references to the live native model.

The v0.2 public API exposes no commands, drives, policy recommendations, neuromodulator values, memory weights, action-selection outputs, or named emotion categories.

## Capability-typed information boundary

Qualified online computation should receive only the authority needed to compute from information available through the current cut point.

A proposed `ObservationPrefixView`-style input contains executed information through `t` and immutable native configuration but no future schedule, future states, semantic mapping, exclusion disposition, or mutable native state.

Oracle, retrospective, and unblinding authority live in separate types/modules. Do not use a runtime `oracle=true`/`allow_future=true`/`unblind=true` flag in one universal API.

Measurement outputs expose observations and provenance only and cannot be converted into native drives/actions/cognitive control signals.

## Prefix-causal information firewall

Every online candidate at time `t` may use only information available through `t` plus prospectively locked forecast-policy parameters.

It may not read the experiment's known future schedule, future interventions/states, post-run exclusions, or semantic arm identity.

For identical prefixes, the online candidate must be identical even if the unseen future is changed.

True-future schedule knowledge is an explicit `OracleDiagnostic`, never a primary endogenous candidate.

## Regulatory quantities remain separate

Do not collapse the following into one number before evidence supports doing so:

- **R1 realized current change:** `H_(t-1) - H_t`;
- **R2 one-step forecast residual:** previous predicted current burden minus realized current burden;
- **R3 aligned overlapping-future revision:** change in predictions for the same absolute future times;
- **R4 rolling finite-horizon debt change:** descriptive composite that includes horizon turnover;
- **regulatory urgency:** breach imminence, peak forecast deviation, breadth of threatened channels, or absolute change magnitude.

The first implementation preserves these as separate neutral candidate families.

## Forecast-policy classes

Prefix-causal policies may include:

- zero-input native recovery;
- persistence of the currently observed drive;
- kinematic state-velocity extrapolation.

Learned/cued future prediction is deferred until a predictive model and its available cues are themselves evidence-bearing and ablatable.

True-future oracle forecasts remain diagnostic only.

## Candidate identity

A candidate's identity includes formula, sign, temporal indices, availability class, forecast policy, horizon, discount, temporal alignment, weighting/normalization, numerical rules, undefined-value semantics, implementation identity, reference fixtures, and the v0.1 semantic lineage.

Any result-changing change produces a new candidate identity.

Candidate IDs stay interpretation-neutral during v0.2 (`r1_*`, `r2_*`, `r3_*`, `r4_*`, `u1_*`).

## Scenario identity and holdout discipline

Discovery and confirmatory scenarios are different evidence cohorts.

Confirmatory holdouts are locked prospectively and audited for content overlap with discovery scenarios; renaming a discovery scenario does not make it new evidence.

Required deterministic families include:

- neutral stability;
- equal current burden with different observed trajectory;
- equal current state with different currently observed load;
- equal drive with different internal regulatory margin;
- deterministic recovery;
- crossed-sign R1/R2/R3 cases;
- identical-prefix/divergent-future twins;
- future-mutation adversarial twins;
- forecast-policy agreement/disagreement regimes.

Primary comparison cut points/windows are also prospectively fixed; selecting a post-run peak is exploratory unless that rule was preregistered.

## Blinding and unblinding

Primary artifacts use opaque blind codes rather than semantic arm IDs.

The semantic `arm_id <-> blind_code` mapping is a separate committed artifact whose digest is frozen before execution and whose contents need not be available to the primary-analysis process.

A frozen blinded candidate/comparison artifact must exist before semantic unblinding.

Semantic-label canaries must remain absent from primary execution/forecast/candidate/comparison artifacts and qualified primary logs.

The eventual evidence package records actual blinding strength (for example artifact-only versus independent analyst/custodian) rather than claiming stronger human blinding than actually occurred.

## Adversarial validation before interpretation

The observatory must survive an explicit adversarial/metamorphic suite before confirmatory interpretation.

Integrity-blocking tests include:

- future suffix mutation / exact-prefix divergent-future invariance;
- semantic mapping permutation and semantic-canary absence;
- observer/no-observer native execution bisimulation;
- oracle removal/separation;
- temporal self-consistency and off-by-one detection;
- artifact/candidate/scenario substitution detection;
- missing scenario/exclusion/analysis mutation detection;
- clean-process deterministic replay.

The validation suite also contains deliberately malicious fixtures that read future information, inspect semantic mapping, mutate shared state, depend on evaluation order, leak semantic labels, zero-fill unavailable values, or masquerade oracle knowledge as online prediction. Each known-bad fixture must be caught by an explicit gate.

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

## No-feedback invariant

The first v0.2 implementation is telemetry-only.

Run the same locked v0.1 study with and without observation and require the complete native `StudyExecutionTrace` to remain identical.

Repeat under candidate-order permutations/all-candidates/one-candidate conditions where relevant to detect hidden mutable shared state.

If observation changes native execution, the foundational isolation gate fails.

## Design freeze and implementation start

Planning must not remain indefinitely editable while code/data generation begins.

`V02_DESIGN_FREEZE.md` defines design states including:

- `Draft`;
- `Reviewable`;
- `FrozenBlockedOnV01`;
- `FrozenImplementationAuthorized`;
- `Superseded`;
- `Invalidated`.

A future machine-readable `DesignFreezeManifest` binds all scientific/epistemic contracts. Once v0.1 qualification passes, an `ImplementationStartReceipt` binds the exact qualified v0.1 source/evidence lineage, design-freeze digest, v0.2 starting SHA, and implementation-tranche version.

No runtime implementation is authorized while an unresolved architecture-blocking design question remains or v0.1 qualification is unresolved.

## Evidence root

Before confirmatory execution, freeze one prospective `ObservationalEvidenceRootManifest` binding:

- exact qualified v0.1 baseline + qualification/evidence digests;
- design-freeze + implementation-start identities;
- exact v0.2 source identity;
- study/preregistration identities;
- primary/secondary/null candidate digests;
- confirmatory scenario-cohort digest;
- analysis-plan digest;
- exclusion criteria;
- blind-mapping commitment;
- information-firewall/temporal-alignment/candidate/scenario/capability/adversarial/blinding contract versions;
- blinding-strength declaration;
- no-feedback and semantic-canary gate definitions;
- toolchain/dependency identities.

After execution, a separate realized package binds all execution, exclusion, forecast, candidate, capability/leak/adversarial validation, blinded-analysis, sensitivity, null-control, unblinding, and semantic-evaluation artifacts.

Every locked confirmatory scenario must be accounted for:

`locked_count == included_count + excluded_count + indeterminate_count`

A failed primary hypothesis on an otherwise valid confirmatory lineage is a qualified negative/null result, not an integrity failure.

## Frozen initial implementation order after v0.1 qualifies

1. create standalone read-only `symthaea-affect-observatory` and one-way dependency gate;
2. implement `ObservationPrefixView` and capability types;
3. implement typed prefix-causal forecast policy interfaces with oracle namespace separated;
4. add trajectory-level forecast artifacts that exactly reproduce v0.1 aggregate allostasis;
5. implement neutral R1/R2/R3/R4 candidates separately;
6. implement typed unavailable/undefined semantics and immutable candidate-definition manifests/reference fixtures;
7. implement prefix-causality, future-mutation, malicious-fixture, and dependency-boundary adversarial gates;
8. implement no-feedback observer/native bisimulation and candidate-order independence;
9. implement semantic-label canaries and mapping/unblinding separation;
10. implement scenario/cohort manifests and discovery/holdout content-overlap audit;
11. implement blinded candidate artifacts/comparison;
12. implement prospective/realized evidence-root and adversarial-validation receipts;
13. only then lock/run the first exploratory v0.2 study.

Out of scope for this lineage: neuromodulation, memory/attention modulation, action selection, control/dominance outputs, mood-like persistence, attachment/social affect, learned emotion labels, or consciousness/sentience inference.

## Graduation criteria

v0.2 can graduate to a later causal functional-affect experiment only if:

1. v0.1 exact-head qualification is complete and bound to the v0.2 start receipt;
2. observational calculations are causally read-only;
3. capability/source-dependency gates prevent future/semantic/control authority in online code;
4. candidate definitions and scenarios are prospectively immutable evidence objects;
5. online candidates pass prefix-causality and future-mutation invariance;
6. the adversarial suite catches known malicious fixtures and all integrity-blocking gates pass;
7. neutral scenarios do not manufacture signal;
8. the primary candidate survives preregistered directional/context/forecast tests;
9. nuisance/null baselines do not explain the same structure under the locked comparison rule;
10. results survive the declared sensitivity/holdout region;
11. exploratory and confirmatory lineages remain separate;
12. exclusions, indeterminate cases, and null hypotheses are preserved;
13. independent replay reproduces exact blinded and semantic artifacts.

Passing these criteria would justify only a narrow claim such as:

> Symthaea exhibits reproducible, prefix-causal, label-free regulatory observables derived from current internal state and already-observed dynamics that distinguish aspects of regulatory change and forecast revision beyond simple current-state and stimulus baselines.

It would **not** establish emotion, subjective valence, feeling, sentience, consciousness, or prediction of unseen future perturbations.

## Deferred later questions

Only after observational qualification should later tranches test causal access to attention/learning, neuromodulation, memory consolidation, policy selection, counterfactual controllability, persistence/mood-like dynamics, autobiographical dependence, social/attachment phenomena, or learned mappings from latent dynamics to human emotion concepts.