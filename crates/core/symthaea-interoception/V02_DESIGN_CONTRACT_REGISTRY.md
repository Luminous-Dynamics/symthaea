# Affective Emergence v0.2 — Design Contract Registry

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines the canonical closure over the normative v0.2 design surface. The design freeze and later evidence roots bind one exact registry identity so a scientific contract, machine-readable registry, or validation surface cannot be omitted silently.

## 1. Principle

Create one validated `DesignContractRegistryManifest` containing the complete ordered set of active normative design identities.

Changing, omitting, replacing, or reclassifying any normative item changes the registry and therefore the design-freeze identity.

Content digest is authoritative; paths are audit aids.

## 2. Registry entry model

Each entry binds at least:

- stable `contract_role`;
- repository-relative path;
- content SHA-256;
- schema/prose-contract version;
- normative status;
- exact design source commit;
- optional supersession identity;
- architecture/evidence blocking class.

The registry itself has deterministic ordering, schema/version, exact source commit, and a canonical digest. Its own final digest is not embedded inside itself.

## 3. Required normative roles

The first v0.2 lineage requires exactly one active normative identity for each applicable role:

- `ObservationalAffectPlan`;
- `InformationFirewall`;
- `TemporalAlignment`;
- `CandidateDefinition`;
- `CandidateFactorSpace`;
- `MinimalExploratoryCandidateSet`;
- `ExploratoryCandidateRegistry`;
- `WeightingDecomposition`;
- `ChannelAggregation`;
- `AllostaticExposureDecomposition`;
- `HistoryStateSufficiency`;
- `CalibrationAndPreprocessing`;
- `ObservatoryStateLifecycle`;
- `IdentifiabilityAndDiscrimination`;
- `FunctionalEvaluationAndPromotion`;
- `FunctionalTargetRegistry`;
- `CausalContrasts`;
- `ExecutionMode`;
- `ScenarioManifest`;
- `MinimalExploratoryScenarioBattery`;
- `ExploratoryScenarioRegistry`;
- `ScenarioConstructibilityAndParameterFreeze`;
- `BlindingCustody`;
- `CapabilityTypedApi`;
- `AdversarialValidation`;
- `MinimalMaliciousFixtureMatrix`;
- `MaliciousFixtureRegistry`;
- `EvidenceRoot`;
- `DesignFreeze`.

The registry-contract specification is bound separately by the freeze to avoid a self-hash cycle.

## 4. Normative status and closure

Statuses:

- `Normative` — implementation/evidence must conform;
- `Supporting` — explanatory only;
- `Superseded` — retained for history but cannot authorize new evidence;
- `Invalidated` — known unusable.

Before `FrozenBlockedOnV01` or `FrozenImplementationAuthorized`, validation proves:

1. each required role has exactly one active normative identity;
2. every path resolves to the recorded digest in the exact frozen tree;
3. no active normative `V02_*` design artifact is omitted;
4. no entry points outside the exact design source commit;
5. supersession references are acyclic and resolvable;
6. no required identity is superseded or invalidated;
7. deterministic ordering is reproduced exactly;
8. prose contracts and machine-readable registries agree on role IDs, counts, authority classes, and scientific meaning.

A prose/JSON disagreement is an integrity failure, not a preference about which representation is authoritative.

## 5. Freeze and evidence-root binding

Dependency direction:

`registry-contract digest`
→ `DesignContractRegistryManifest`
→ registry digest
→ `DesignFreezeManifest`
→ prospective evidence root.

The prospective `ObservationalEvidenceRootManifest` redundantly binds the same registry digest. Freeze/registry/root cross-pairing is invalid.

## 6. Candidate scientific closure

Candidate identity is more than a formula or factor coordinate.

It binds:

- `R × W × A × T × F × I × H` coordinate;
- preprocessing/calibration identity;
- evaluator-state/cache authority;
- numeric/undefined semantics;
- exact formula/source/reference fixtures;
- v0.1 source/model-semantics lineage.

Different value-changing preprocessing, evaluator authority, history window, forecast policy, aggregation, weighting, or temporal semantics creates a different scientific candidate unless prospective equivalence proves output preservation over the locked valid domain.

## 7. Finite E00–E11 candidate closure

`V02_MINIMAL_EXPLORATORY_CANDIDATE_SET.md` and `V02_EXPLORATORY_CANDIDATE_REGISTRY.json` jointly freeze exactly twelve first-lineage roles E00–E11.

The JSON registry is machine-readable identity; the prose contract supplies rationale and interpretation boundaries. They must agree exactly on:

- role IDs;
- candidate IDs;
- relation/weighting/aggregation/temporal/forecast/history classes;
- primary `ObservedDrivePersistence` forecast policy;
- diagnostic `NativeZeroInputRecovery` and `KinematicVelocity` policies;
- no fitted preprocessing;
- `NoneAcrossEvaluationCoordinates` evaluator state;
- at-most-three nonredundant classes eligible for later confirmatory-design consideration.

No additional exploratory candidate may be introduced after output inspection without a new registry/freeze lineage.

## 8. Finite X00–X11 scenario closure

`V02_MINIMAL_EXPLORATORY_SCENARIO_BATTERY.md` and `V02_EXPLORATORY_SCENARIO_REGISTRY.json` jointly freeze twelve scenario-family identities X00–X11 and their discrimination obligations.

The preferred first materialization is no more than 24 primary arms before malicious fixtures and forecast-policy diagnostics.

The abstract registry deliberately does **not** freeze guessed numeric parameters. Exact arms are materialized under `V02_SCENARIO_CONSTRUCTIBILITY_AND_PARAMETER_FREEZE.md` only after the implementation can prove the frozen mechanical contrast is constructible.

Candidate-dependent cut points, future-dependent grouping, semantic-condition-dependent construction, or discovery/calibration/confirmatory overlap violations are invalid.

## 9. Scenario constructibility and numeric parameter freeze

Parameter search is an engineering step, not hidden candidate optimization.

Allowed search objectives are only prospectively frozen native/mechanical X-family constraints. E00–E11 comparative scores, Y0–Y3 outcomes, affect interpretation, semantic attractiveness, and post-hoc plot appearance are forbidden parameter-selection inputs.

If multiple constructions satisfy a family, use a prospectively frozen deterministic tie-break rule independent of candidate output.

Once all required X families are constructible, freeze one exact `ExploratoryScenarioParameterManifest` before candidate comparison. Numeric changes after that point start a new exploratory scenario lineage.

`NotConstructibleUnderCurrentSubstrate` is a valid preserved result.

## 10. Functional Y0–Y3 target closure

`V02_FUNCTIONAL_EVALUATION_AND_PROMOTION.md` and `V02_FUNCTIONAL_TARGET_REGISTRY.json` jointly freeze four retrospective evaluation targets:

- Y0 next-step realized viability change;
- Y1 next-16 realized cumulative viability exposure;
- Y2 realized first viability-breach latency within 16 steps;
- Y3 realized terminal viability burden at +16.

Candidate payloads freeze before suffix-derived Y targets are attached. Y targets are never candidate inputs, preprocessing authority, cut-point selectors, or scenario-parameter-search objectives.

Evaluation is multi-target. There is no post-hoc universal affect score. Near-tautological candidate/target pairs are flagged rather than treated as broad explanatory evidence. Zero winners is valid.

## 11. M01–M14 malicious-fixture closure

`V02_MINIMAL_MALICIOUS_FIXTURE_MATRIX.md` and `V02_MALICIOUS_FIXTURE_REGISTRY.json` jointly freeze fourteen known-bad roles M01–M14.

Every implemented malicious fixture must be rejected by its expected primary gate with the expected failure class, while the nearest valid paired control passes.

A known-malicious fixture that passes is an integrity failure of the validation architecture. A gate that rejects both malicious and valid control is reported as over-restrictive and does not count as success.

New normative authority boundaries must map to an existing malicious fixture or prospectively add a new fixture and superseding suite identity before freeze.

## 12. Information, history, preprocessing, and evaluator-state closure

Initial primary evidence is offline prefix replay over a completed immutable native trace.

Candidate computation cannot receive:

- future suffixes;
- semantic arm mapping;
- suffix-sensitive full-trace digest in its payload;
- mutable native state;
- confirmatory-cohort adaptive statistics;
- Y0–Y3 future outcomes.

H1 history is same-scenario immutable-prefix history only. It is external observatory information, not native memory or mood.

Cross-scenario/candidate persistent evaluator state is forbidden in the first lineage. Required metamorphic tests include scenario/candidate order, cold/warm process, cache hit/miss, serial/parallel, batch/chunk, and incremental/from-scratch equivalence.

## 13. Identifiability and causal closure

Every primary-vs-baseline superiority claim requires a prospectively registered discriminator. Observationally equivalent candidates remain an equivalence class; parsimony may prefer the simpler representative only after equivalence is established.

Mechanistic language additionally requires a frozen causal contrast and passing manipulation check. Numerical matching alone is not causal control.

H1 gains beyond H0 require matched-current-state history discriminators and are interpreted only as external historical information gain.

## 14. Cross-coverage closure

Before exploratory execution, validation proves:

1. every E00–E11 required comparison is covered by X00–X11;
2. every implemented integrity/authority boundary maps to M01–M14 or a prospectively superseding suite;
3. every promotion comparison references frozen candidate, scenario/cut point, Y target, and simpler-baseline obligation;
4. no Y target enters candidate, preprocessing, cut-point, or parameter-search dependencies;
5. prose and JSON identities are mutually consistent;
6. every exact numeric scenario arm traces back to one frozen X-family constructibility obligation.

Large scenario or test counts cannot substitute for this closure.

## 15. Change severity

- Class I: supporting explanation with no normative identity change;
- Class II: candidate/scenario/malicious/target registries, parameterization, weighting, aggregation, temporal/history semantics, preprocessing, evaluator lifecycle, identifiability, causal contrasts, analysis, or evidence semantics;
- Class III: future-information authority, primary execution mode, feedback/control authority, causal outputs, or native persisted affect/memory.

Any active normative change after freeze supersedes the old registry/freeze lineage.

## 16. Required implementation gates

Future implementation mechanically tests at least:

- stable registry round trip/digest;
- missing/duplicate/omitted normative roles rejected;
- path/digest/source/supersession mismatch rejected;
- prose/JSON E/X/M/Y disagreement rejected;
- E00–E11 count/identity drift rejected;
- X00–X11 family/subcase drift rejected;
- missing discriminator coverage rejected;
- candidate-dependent cut points rejected;
- candidate/Y-dependent scenario-parameter selection rejected;
- M01–M14 known-bad acceptance rejected;
- paired valid-control failure reported separately;
- suffix target entering candidate/preprocessing rejected;
- shortlist over three nonredundant classes rejected unless a prospectively superseding discrimination stage exists;
- fitted confirmatory preprocessing rejected;
- evaluator order/cache/state dependence rejected;
- native-memory claims rejected when persistence exists only externally;
- causal claims rejected without valid contrasts/manipulation checks.

## 17. Current active normative membership

Prose contracts:

- `V02_OBSERVATIONAL_AFFECT_PLAN.md`
- `V02_INFORMATION_FIREWALL.md`
- `V02_TEMPORAL_ALIGNMENT.md`
- `V02_CANDIDATE_DEFINITION_CONTRACT.md`
- `V02_CANDIDATE_FACTOR_SPACE.md`
- `V02_MINIMAL_EXPLORATORY_CANDIDATE_SET.md`
- `V02_WEIGHTING_DECOMPOSITION.md`
- `V02_CHANNEL_AGGREGATION_CONTRACT.md`
- `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md`
- `V02_HISTORY_STATE_SUFFICIENCY.md`
- `V02_CALIBRATION_AND_PREPROCESSING.md`
- `V02_OBSERVATORY_STATE_LIFECYCLE.md`
- `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md`
- `V02_FUNCTIONAL_EVALUATION_AND_PROMOTION.md`
- `V02_CAUSAL_CONTRASTS.md`
- `V02_EXECUTION_MODE_CONTRACT.md`
- `V02_SCENARIO_MANIFEST.md`
- `V02_MINIMAL_EXPLORATORY_SCENARIO_BATTERY.md`
- `V02_SCENARIO_CONSTRUCTIBILITY_AND_PARAMETER_FREEZE.md`
- `V02_BLINDING_CUSTODY.md`
- `V02_CAPABILITY_TYPED_API.md`
- `V02_ADVERSARIAL_VALIDATION.md`
- `V02_MINIMAL_MALICIOUS_FIXTURE_MATRIX.md`
- `V02_EVIDENCE_ROOT.md`
- `V02_DESIGN_FREEZE.md`

Machine-readable normative registries:

- `V02_EXPLORATORY_CANDIDATE_REGISTRY.json`
- `V02_EXPLORATORY_SCENARIO_REGISTRY.json`
- `V02_MALICIOUS_FIXTURE_REGISTRY.json`
- `V02_FUNCTIONAL_TARGET_REGISTRY.json`

Supporting/non-normative history/future notes do not satisfy normative roles.

## 18. Claim boundary

A closed design registry establishes that one explicit, finite, machine-auditable design governed an evidence lineage. It does not establish affect, emotion, subjective valence, native mood, suffering, sentience, or consciousness.
