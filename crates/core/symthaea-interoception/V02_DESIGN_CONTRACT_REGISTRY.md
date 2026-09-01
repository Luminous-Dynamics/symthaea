# Affective Emergence v0.2 — Design Contract Registry

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines one canonical machine-readable registry for the normative v0.2 design documents. Its purpose is to prevent the `DesignFreezeManifest` and later evidence roots from silently omitting a newly added scientific contract.

## 1. Principle

A design freeze should not depend on a hand-maintained scattering of optional digest fields.

Create one validated `DesignContractRegistryManifest` containing the complete ordered set of normative design-contract identities. The design freeze binds that registry digest as the authoritative closure over the design surface.

If a normative contract is added, removed, replaced, or reclassified, the registry digest changes and therefore the design-freeze identity changes.

## 2. Proposed registry schema

A future `DesignContractRegistryManifest` should contain at minimum:

- registry schema/version;
- exact v0.2 design source commit;
- ordered `contracts: Vec<DesignContractEntry>`;
- canonical registry SHA-256.

Each `DesignContractEntry` binds:

- stable `contract_role` enum;
- repository-relative path;
- content SHA-256;
- contract schema/version or prose-contract version;
- normative status;
- optional supersedes/superseded-by identity;
- optional architecture-blocking flag.

Paths aid auditability but are not sufficient identity; content digest is authoritative.

## 3. Required normative roles

Initial required roles include at least:

- `ObservationalAffectPlan`;
- `InformationFirewall`;
- `TemporalAlignment`;
- `CandidateDefinition`;
- `CandidateFactorSpace`;
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
- `BlindingCustody`;
- `CapabilityTypedApi`;
- `AdversarialValidation`;
- `EvidenceRoot`;
- `DesignFreeze`.

The registry-contract specification itself is bound separately by the design freeze as `design_contract_registry_contract_digest`, avoiding a self-hash cycle.

## 4. Normative status

Suggested status enum:

- `Normative` — implementation/evidence must conform;
- `Supporting` — explanatory material, not independently architecture-defining;
- `Superseded` — retained for history but cannot authorize new evidence;
- `Invalidated` — known unusable contract.

Only active `Normative` entries satisfy required roles.

A singular required role with zero or multiple active normative entries is a validation failure.

## 5. Closure rule

Before `FrozenBlockedOnV01` or `FrozenImplementationAuthorized`, registry validation must prove:

1. every required role exists exactly once as active normative content;
2. every listed path resolves to the exact recorded digest in the frozen source tree;
3. no active normative `V02_*` contract file is omitted;
4. no registry entry points outside the exact v0.2 design source commit;
5. supersession references are acyclic and resolvable;
6. no required normative contract is `Superseded` or `Invalidated`;
7. architecture-blocking contracts are explicitly included;
8. canonical ordering is deterministic.

The omitted-file gate matters: checking only a known hard-coded role list would let a newly added normative contract disappear from the freeze by accident.

## 6. Canonical ordering

Prefer deterministic ordering by stable `contract_role`, then path where a role is legitimately multi-valued.

Filesystem enumeration order must not become scientific identity.

## 7. Self-reference avoidance

Do not include the realized registry's own final SHA-256 inside itself.

Use:

`V02_DESIGN_CONTRACT_REGISTRY.md digest`
→ registry schema/validation
→ `DesignContractRegistryManifest`
→ registry SHA-256
→ `DesignFreezeManifest`.

The freeze therefore binds both the registry-specification digest and the realized registry digest.

## 8. Design-freeze binding

`DesignFreezeManifest` treats the registry as authoritative.

Selected individual contract digests may be repeated for audit convenience, but they must equal corresponding registry entries and cannot substitute for closure.

A freeze is invalid if:

- an individual digest disagrees with the registry;
- the registry omits a normative contract;
- registry source commit differs from the design source commit;
- registry contract/version differs from the freeze declaration;
- a required contract is superseded or invalidated.

## 9. Evidence-root binding

The prospective `ObservationalEvidenceRootManifest` also binds the same exact registry digest.

Recommended redundancy:

- evidence root records `design_freeze_sha256`;
- evidence root records `design_contract_registry_sha256`;
- validator requires equality with the registry embedded by the freeze.

This makes freeze/root cross-pairing detectable immediately.

## 10. Candidate runtime-identity closure

A candidate's scientific identity is not exhausted by its mathematical factor coordinate.

The prospective candidate/evidence identity must also bind:

- preprocessing/calibration manifest digest or explicit `None`;
- calibration cohort/fitted-parameter identities when applicable;
- evaluator-isolation manifest digest;
- allowed evaluator persistent-state class;
- cache/state policy;
- implementation/reference-fixture identity.

Therefore two candidates with the same `R × W × A × T × F × I × H` coordinate but different value-changing preprocessing, fitted parameters, evaluator state lifecycle, or cache authority are different scientific candidate/evidence identities unless a prospective mechanical equivalence proof shows the difference is output-preserving over the locked valid domain.

## 11. History/state-sufficiency closure

A prefix-causal replay process may compute history-sensitive statistics not represented by the native regulator.

Initial v0.2 distinguishes:

- `H0CurrentNativeStateOnly`;
- `H1ReplayedPrefixHistory`;
- retrospective/oracle diagnostic history;
- future separately qualified native persisted memory.

A candidate that differs across matched-current-state histories proves only that prior trace history adds information to the external observatory unless a later native memory mechanism carries a qualified sufficient statistic.

The restart/state-sufficiency gate must verify that identical complete native state/configuration plus identical future inputs produces identical future native execution under the v0.1 deterministic contract.

## 12. Observatory-state closure

H1 within-scenario history does not authorize cross-scenario process memory.

Primary `OfflinePrefixReplay` candidate evaluation must satisfy `V02_OBSERVATORY_STATE_LIFECYCLE.md`:

- scenario-local evaluator lifecycle;
- no cross-arm/scenario adaptive state;
- candidate/scenario order invariance;
- cold/warm process equivalence;
- serial/parallel equivalence;
- cache hit/miss equivalence;
- from-scratch vs incremental prefix equivalence where incremental evaluation exists.

A hidden mutable-state or evaluation-order failure is an `IntegrityFailure`.

## 13. Calibration/holdout closure

A prefix-causal formula can still leak confirmatory information through fitted preprocessing parameters.

Primary confirmatory candidate computation therefore satisfies `V02_CALIBRATION_AND_PREPROCESSING.md`:

- every value-changing transform has a frozen manifest;
- fitted parameters come only from prospectively identified discovery/calibration or external-reference data;
- confirmatory cohort values cannot refit scaling, thresholds, clipping, smoothing, or normalization;
- individual confirmatory candidate values are independent of cohort size/order and other confirmatory outcomes;
- calibration and confirmatory content overlap is audited;
- preprocessing sensitivity is frozen prospectively when required.

Adaptive confirmatory fitting is an `IntegrityFailure`.

## 14. Identifiability closure

A complete registry is not enough if locked scenarios cannot distinguish locked candidates.

Before confirmatory freeze, require a valid `CandidateDiscriminationManifest` under `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md` binding:

- finite candidate set;
- scenario/cut-point set;
- required primary-vs-baseline pairwise obligations;
- equivalence tolerances;
- discriminator coverage;
- prospective parsimony/model-selection rule.

A candidate may not be promoted as superior to a baseline when the design lacks a discriminator capable of separating them.

History-sensitive H1 candidates claiming information beyond current state require matched-current-state H1-vs-H0 discriminators.

## 15. Causal-contrast closure

Mechanistic language requires more than numerical matching.

For every primary hypothesis framed as a response to a manipulation, bind one or more `CausalContrastManifest` identities under `V02_CAUSAL_CONTRASTS.md`.

Each contrast declares manipulated fields, pre-treatment equalities, allowed mediators, forbidden changes, supported discrimination obligation, and contrast class (`total-path`, `direct-path`, `mediator-specific`, or `diagnostic`).

A realized causal claim requires a passing manipulation-check artifact. Otherwise report descriptive scenario differences.

## 16. Change severity

Registry changes inherit the severity of the contract change they represent.

- purely supporting explanation may be Class I when no normative identity changes;
- changing candidate, preprocessing, calibration, evaluator lifecycle, weighting, aggregation, temporal integration, history access, identifiability, causal contrast, scenario, analysis, or evidence semantics is Class II;
- changing future-information authority, primary execution mode, feedback authority, causal outputs, or introducing native persisted affect/memory is Class III.

Adding any active normative contract after freeze changes the registry and supersedes the old freeze.

## 17. Review and CI gates

Future implementation should mechanically test:

- stable registry round trip/digest;
- content changes alter entry digests;
- missing/duplicate required roles rejected;
- omitted normative file rejected;
- path/digest/source mismatch rejected;
- supersession cycle rejected;
- freeze/registry/root mismatch rejected;
- missing aggregation/history/calibration/evaluator/identifiability/causal contracts rejected;
- candidate rejected when preprocessing or evaluator-state identity is undeclared;
- fitted confirmatory preprocessing rejected;
- order-dependent evaluator rejected;
- history-sensitive candidate rejected when history basis is undeclared;
- native-memory claim rejected when persistence exists only in external replay/evaluator state;
- primary-vs-baseline claim rejected without a discriminator;
- causal claim rejected without a valid contrast/manipulation check.

Known-malicious fixtures should intentionally violate these boundaries and be caught.

## 18. Current intended normative membership

The active design set currently includes:

- `V02_OBSERVATIONAL_AFFECT_PLAN.md`
- `V02_INFORMATION_FIREWALL.md`
- `V02_TEMPORAL_ALIGNMENT.md`
- `V02_CANDIDATE_DEFINITION_CONTRACT.md`
- `V02_CANDIDATE_FACTOR_SPACE.md`
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
- `V02_BLINDING_CUSTODY.md`
- `V02_CAPABILITY_TYPED_API.md`
- `V02_ADVERSARIAL_VALIDATION.md`
- `V02_EVIDENCE_ROOT.md`
- `V02_DESIGN_FREEZE.md`

This list describes the current design stage, not a permanently fixed schema. Future additions require explicit classification and a new registry/freeze identity.

## 19. Claim boundary

A closed registry establishes that one explicit design surface governed an evidence lineage. The additional calibration, evaluator-isolation, history, identifiability, and causal contracts constrain what can legitimately be inferred from the resulting artifacts.

They do not establish affect, emotion, subjective valence, native mood, sentience, or consciousness.