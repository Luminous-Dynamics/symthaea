# Affective Emergence v0.2 — Design Contract Registry

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines one canonical machine-readable registry for the normative v0.2 design documents. Its purpose is to prevent the `DesignFreezeManifest` and later evidence roots from silently omitting a newly added scientific contract.

## 1. Principle

Create one validated `DesignContractRegistryManifest` containing the complete ordered set of normative design-contract identities. The design freeze binds that registry digest as the authoritative closure over the design surface.

If a normative contract is added, removed, replaced, or reclassified, the registry digest changes and therefore the design-freeze identity changes.

## 2. Proposed registry schema

A future `DesignContractRegistryManifest` contains at minimum:

- registry schema/version;
- exact v0.2 design source commit;
- ordered `contracts: Vec<DesignContractEntry>`;
- canonical registry SHA-256.

Each `DesignContractEntry` binds stable role, repository-relative path, content SHA-256, contract version, normative status, optional supersession identity, and optional architecture-blocking flag.

Paths aid auditability but are not sufficient identity; content digest is authoritative.

## 3. Required normative roles

Initial required roles include at least:

- `ObservationalAffectPlan`;
- `InformationFirewall`;
- `TemporalAlignment`;
- `CandidateDefinition`;
- `CandidateFactorSpace`;
- `MinimalExploratoryCandidateSet`;
- `WeightingDecomposition`;
- `ChannelAggregation`;
- `AllostaticExposureDecomposition`;
- `HistoryStateSufficiency`;
- `CalibrationAndPreprocessing`;
- `ObservatoryStateLifecycle`;
- `IdentifiabilityAndDiscrimination`;
- `CausalContrasts`;
- `ExecutionMode`;
- `ScenarioManifest`;
- `MinimalExploratoryScenarioBattery`;
- `BlindingCustody`;
- `CapabilityTypedApi`;
- `AdversarialValidation`;
- `MinimalMaliciousFixtureMatrix`;
- `EvidenceRoot`;
- `DesignFreeze`.

The registry-contract specification itself is bound separately by the design freeze as `design_contract_registry_contract_digest`, avoiding a self-hash cycle.

## 4. Normative status and closure

Suggested status enum:

- `Normative` — implementation/evidence must conform;
- `Supporting` — explanatory material, not independently architecture-defining;
- `Superseded` — retained for history but cannot authorize new evidence;
- `Invalidated` — known unusable contract.

Before `FrozenBlockedOnV01` or `FrozenImplementationAuthorized`, validation proves:

1. every required role exists exactly once as active normative content;
2. every listed path resolves to the recorded digest in the exact frozen source tree;
3. no active normative `V02_*` contract file is omitted;
4. no registry entry points outside the exact design source commit;
5. supersession references are acyclic/resolvable;
6. no required contract is superseded/invalidated;
7. architecture-blocking contracts are included;
8. ordering is deterministic.

The omitted-file gate prevents a new normative contract from silently disappearing from the freeze.

## 5. Canonical ordering and self-reference

Order by stable `contract_role`, then path where a role is legitimately multi-valued. Filesystem enumeration order must not become scientific identity.

Use the dependency direction:

`V02_DESIGN_CONTRACT_REGISTRY.md digest`
→ registry schema/validation
→ `DesignContractRegistryManifest`
→ registry SHA-256
→ `DesignFreezeManifest`.

Do not include the realized registry's own final SHA-256 inside itself.

## 6. Freeze and evidence-root binding

`DesignFreezeManifest` treats the registry as authoritative. Repeated individual contract digests are audit anchors only and must equal registry entries.

The prospective `ObservationalEvidenceRootManifest` binds the same exact registry digest. Freeze/registry or root/registry cross-pairing is an integrity failure.

## 7. Candidate runtime-identity closure

Candidate identity includes more than `R × W × A × T × F × I × H`.

It also binds preprocessing/calibration identity, calibration cohort/fitted parameters when applicable, evaluator-isolation identity, persistent-state/cache policy, implementation/reference fixtures, and exact source/model lineage.

Different value-changing preprocessing or evaluator authority means a different scientific candidate/evidence identity unless prospective mechanical equivalence proves output preservation over the locked valid domain.

## 8. Minimal exploratory candidate-set closure

`V02_MINIMAL_EXPLORATORY_CANDIDATE_SET.md` is normative for the first exploratory lineage.

A canonical `ExploratoryCandidateSetManifest` binds exactly E00–E11, their candidate-definition digests, pairwise discrimination obligations, primary `ObservedDrivePersistence` forecast policy, zero fitted preprocessing, evaluator-isolation identity, and prospective selection/parsimony rule.

The factor space is not permission to add candidates after seeing data.

Extra candidates, missing roles, oracle/retrospective primary candidates, fitted/adaptive preprocessing, cross-evaluation mutable state, or missing discrimination obligations invalidate the set.

## 9. Minimal exploratory scenario-battery closure

`V02_MINIMAL_EXPLORATORY_SCENARIO_BATTERY.md` is normative for the first exploratory lineage.

A canonical `ExploratoryScenarioBatteryManifest` binds X00–X11, exact concrete arm/scenario digests, matched groups, cut points/windows, causal contrasts where applicable, and the E00–E11 discriminator coverage matrix.

The preferred primary-arm budget is no more than 24 arms before malicious fixtures and forecast-policy sensitivity diagnostics. Exceeding that budget requires a prospectively documented reason and a superseding battery identity before outputs are inspected.

A large scenario count cannot compensate for missing discriminator coverage.

Candidate-dependent cut points, future-dependent grouping, unregistered extra primary scenario families, or discovery/calibration/confirmatory overlap violations invalidate the battery.

## 10. History/state-sufficiency closure

Initial v0.2 distinguishes H0 current native state, H1 externally replayed prefix history, retrospective/oracle diagnostics, and future separately qualified native persisted memory.

A candidate differing across matched-current-state histories proves only that prior trace history helps the external observatory unless a later native mechanism carries a qualified sufficient statistic.

Restart/state-sufficiency must show identical future native execution from identical complete native state/configuration plus identical future inputs.

## 11. Observatory-state closure

H1 same-scenario history does not authorize cross-scenario process memory.

Primary `OfflinePrefixReplay` evaluation requires scenario/candidate order invariance, cold/warm process equivalence, serial/parallel equivalence, cache hit/miss equivalence, and from-scratch/incremental prefix equivalence when incremental evaluation exists.

Hidden mutable state or order dependence is an `IntegrityFailure`.

## 12. Calibration/holdout closure

Every value-changing transform has a frozen preprocessing manifest. Fitted parameters come only from prospectively identified discovery/calibration or external-reference data.

Confirmatory cases cannot refit scaling, thresholds, clipping, smoothing, normalization, or other parameters. Individual confirmatory values must be invariant to unrelated cohort size/order/outcomes.

Adaptive confirmatory fitting is an `IntegrityFailure`.

## 13. Identifiability closure

A valid `CandidateDiscriminationManifest` binds the finite candidate set, scenario/cut-point set, primary-vs-baseline obligations, equivalence tolerances, discriminator coverage, and prospective parsimony/model-selection rule.

A candidate cannot be promoted beyond a baseline when the locked design lacks a discriminator capable of separating them.

H1 gains beyond H0 require matched-current-state history discriminators.

## 14. Causal-contrast closure

Mechanistic hypotheses bind `CausalContrastManifest` identities declaring manipulated fields, pre-treatment equalities, allowed mediators, forbidden changes, supported discrimination obligation, and contrast class.

A causal claim requires a passing manipulation-check artifact. Otherwise report descriptive scenario differences.

## 15. Malicious-fixture closure

`V02_MINIMAL_MALICIOUS_FIXTURE_MATRIX.md` is normative for validation of the first observatory implementation.

Before exploratory outputs become scientifically interpretable, a canonical malicious-fixture report must cover M01–M14 or a prospectively superseding suite.

Each malicious case must be rejected by its expected primary gate with the expected failure class, and its nearest valid paired control must pass.

The suite covers future/suffix leakage, semantic leakage, confirmatory preprocessing leakage, cross-scenario state, cache-key poisoning, unavailable-as-zero, temporal misalignment, relation substitution, weighting/aggregation substitution, oracle escalation, observer feedback, and artifact substitution.

A known-malicious fixture that passes is an integrity failure of the validation architecture.

## 16. Cross-coverage closure

Before exploratory execution, validation must prove all three finite surfaces are mutually closed:

1. every required E00–E11 pairwise discrimination obligation is covered by at least one X00–X11 family/cut point;
2. every implemented integrity/authority boundary is covered by at least one M01–M14 malicious fixture or an explicitly registered successor;
3. every X-family and M-fixture maps back to a normative contract and expected result/failure class.

This prevents candidate, scenario, and validation plans from drifting independently.

## 17. Change severity

- purely supporting explanation may be Class I when no normative identity changes;
- changing candidate set, scenario battery, malicious suite, preprocessing, evaluator lifecycle, weighting, aggregation, temporal/history semantics, identifiability, causal contrasts, analysis, or evidence is Class II;
- changing future-information authority, primary execution mode, feedback authority, causal outputs, or introducing native persisted affect/memory is Class III.

Adding any active normative contract after freeze changes the registry and supersedes the old freeze.

## 18. Review and CI gates

Future implementation mechanically tests:

- stable registry round trip/digest;
- missing/duplicate/omitted normative roles rejected;
- path/digest/source/supersession mismatch rejected;
- freeze/registry/root mismatch rejected;
- missing E/X/M finite-surface contract rejected;
- E00–E11 role/count/identity drift rejected;
- missing candidate discriminator coverage rejected;
- X00–X11 required-family drift rejected;
- candidate-dependent cut points rejected;
- known-malicious M-fixture unexpectedly accepted rejected;
- paired valid control unexpectedly rejected reported separately;
- preprocessing/evaluator identity undeclared rejected;
- fitted confirmatory preprocessing rejected;
- order-dependent evaluator rejected;
- native-memory claim rejected when persistence exists only externally;
- causal claim rejected without valid contrast/manipulation check.

## 19. Current intended normative membership

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
- `V02_CAUSAL_CONTRASTS.md`
- `V02_EXECUTION_MODE_CONTRACT.md`
- `V02_SCENARIO_MANIFEST.md`
- `V02_MINIMAL_EXPLORATORY_SCENARIO_BATTERY.md`
- `V02_BLINDING_CUSTODY.md`
- `V02_CAPABILITY_TYPED_API.md`
- `V02_ADVERSARIAL_VALIDATION.md`
- `V02_MINIMAL_MALICIOUS_FIXTURE_MATRIX.md`
- `V02_EVIDENCE_ROOT.md`
- `V02_DESIGN_FREEZE.md`

Supporting/non-normative notes such as history claim summaries or future native-persistence notes are classified separately and do not satisfy normative roles.

Future additions require explicit classification and a new registry/freeze identity.

## 20. Claim boundary

A closed registry establishes that one explicit design surface governed an evidence lineage. The finite candidate, scenario, and malicious-fixture surfaces plus calibration, evaluator-isolation, history, identifiability, and causal contracts constrain what can legitimately be inferred.

They do not establish affect, emotion, subjective valence, native mood, sentience, or consciousness.