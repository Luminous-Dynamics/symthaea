# Affective Emergence v0.2 — Observational Evidence Root

Status: **design-only / blocked on Native Interoception v0.1 qualification**

The v0.2 design has several independently versioned identities: native baseline, complete design-contract registry, design freeze, study protocol, factorized candidate definitions, scenarios, analysis rules, blind mapping, exclusion decisions, execution traces, validation contracts, and derived artifacts. This document defines the proposed root-of-evidence object that binds them into one auditable lineage.

## Principle

No single SHA-256 can prove scientific validity, but a root manifest can make silent substitution detectable.

The evidence root should answer:

> Exactly which qualified baseline, complete frozen design, prospective study, factorized candidate set, scenario cohort, analysis rules, execution/information boundary, adversarial-validation contract, and blinding commitment does this result claim to belong to?

If any identity changes, the root identity changes.

## Complete-design dependency

The prospective root must bind the same exact `DesignContractRegistryManifest` as the referenced `DesignFreezeManifest`.

Recommended redundant binding:

- `design_freeze_sha256`;
- `design_contract_registry_sha256`;
- `design_contract_registry_contract_digest`.

Validation requires that the root's registry digest exactly equals the registry embedded in the validated design freeze. A valid freeze paired with a different valid registry is an integrity failure.

This redundancy makes the complete scientific design surface directly inspectable by a reproducer rather than requiring it to infer registry identity indirectly.

## Proposed ObservationalEvidenceRootManifest

Before confirmatory execution, construct and freeze a prospective root containing at minimum:

- root schema/version;
- evidence run class (`Confirmatory` for confirmatory inference);
- exact v0.1 source commit;
- v0.1 model-semantics version;
- v0.1 `QualificationEvidenceBundle` SHA-256 as the authoritative baseline-promotion identity;
- v0.1 qualification-receipt SHA-256 for component-level auditability;
- v0.1 evidence-capsule SHA-256 for component-level auditability;
- v0.2 `DesignFreezeManifest` SHA-256;
- v0.2 `DesignContractRegistryManifest` SHA-256;
- `V02_DESIGN_CONTRACT_REGISTRY.md` content digest/version;
- v0.2 `ImplementationStartReceipt` SHA-256;
- exact v0.2 observatory implementation source commit;
- primary execution mode (`OfflinePrefixReplay` for the initial lineage);
- replay-harness contract/version;
- prefix-view / prefix-digest schema identities;
- candidate-payload / provenance-envelope schema identities;
- study-preregistration SHA-256;
- study-level preregistration SHA-256;
- finite exploratory/confirmatory candidate-set manifest digest as applicable;
- ordered primary/secondary candidate-definition digests;
- explicit primary candidate digest;
- each candidate's factor-space coordinate digest/identity;
- ordered nuisance/null baseline candidate digests;
- candidate factor-space compatibility-contract version;
- scenario-cohort manifest SHA-256;
- analysis-plan SHA-256;
- exclusion-criteria definition digest;
- arm-identity-mapping commitment SHA-256;
- weighting-decomposition contract/version;
- allostatic-exposure-decomposition contract/version;
- execution-mode contract/version;
- information-firewall contract/version;
- temporal-alignment contract/version;
- candidate-definition contract/version;
- candidate-factor-space contract/version;
- scenario/holdout contract/version;
- capability-typed API contract/version;
- adversarial-validation contract/version;
- blinding-custody contract/version;
- blinding-strength declaration;
- prefix-causality gate definition/version;
- offline observer/native isolation rule and later online-shadow equivalence definition/version;
- semantic-label-canary gate definition/version;
- toolchain/dependency identities required by the v0.2 implementation;
- canonical SHA-256 of the entire validated root.

This prospective root is frozen before confirmatory execution.

The component qualification-receipt and evidence-capsule digests are retained for inspection, but they do not independently authorize promotion. The bound `QualificationEvidenceBundle` is the authoritative proof that those components belong to the same exact v0.1 source/model-semantics lineage.

## Design-freeze dependency

The prospective evidence root may not claim `LockedConfirmatory` unless it binds:

- a valid complete design-contract registry;
- a valid v0.2 design freeze binding exactly that registry;
- a valid implementation-start receipt binding that same freeze/registry;
- one exact qualified v0.1 baseline bundle.

The design freeze establishes which scientific/epistemic contracts governed implementation. The implementation-start receipt proves that runtime work began from the declared qualified v0.1 baseline and complete frozen v0.2 design.

A later normative design change that is Class II/III under `V02_DESIGN_FREEZE.md` changes the registry and requires a new freeze/implementation identity before confirmatory use.

## Candidate-coordinate closure

Every confirmatory candidate definition must bind its explicit factor-space coordinate from `V02_CANDIDATE_FACTOR_SPACE.md`.

At minimum the coordinate identifies:

- regulatory relation basis;
- weighting basis;
- temporal aggregation;
- forecast policy where applicable;
- information class;
- channel projection where applicable;
- availability/numeric contract.

The root binds a **finite, prospectively closed candidate set**. The factor space is not an authorization to compute an unrestricted Cartesian product after seeing results.

The exact candidate coordinate and exact formula manifest are both authoritative. A coordinate cannot substitute for the formula, and a formula cannot hide a changed coordinate.

## AnalysisPlan identity

The root must bind an independently canonical `AnalysisPlan` rather than relying on prose in a notebook or paper.

It should include:

- primary hypotheses;
- exact candidate/baseline IDs and coordinates;
- scenario subset/cohort references;
- comparison cut points/windows;
- directional relations;
- minimum-effect thresholds;
- equivalence tolerances;
- parameter/sensitivity region;
- candidate ranking/model-comparison rule;
- treatment of ties and `NoUniqueWinner`;
- treatment of weighting/temporal ambiguity;
- treatment of indeterminate results;
- boundary-turnover diagnostic rule;
- required prefix-causality tests;
- required burden-vs-precision discriminators;
- required temporal-aggregation discriminators;
- required neutrality tests;
- required offline execution-mode validation and any later online-shadow equivalence tests;
- required semantic-label-canary test;
- required adversarial-validation suite/version;
- deterministic robustness summary rules;
- exclusion handling;
- unblinding procedure/version.

Changing any of these after confirmatory execution starts creates a new analysis identity and a new confirmatory lineage.

## Prospective vs realized evidence

Do not mix prospective identity with realized results.

### Prospective root

Contains what was promised before confirmatory data:

- qualified baseline bundle identity plus component audit identities;
- complete contract-registry identity;
- frozen design/implementation-start identity;
- study/protocol identity;
- finite factorized candidate set;
- scenarios;
- analysis;
- execution/information/capability boundaries;
- weighting and temporal semantics;
- adversarial-validation contract;
- blinding commitment;
- exclusions;
- environment/toolchain identity.

### Realized evidence package

After execution, produce a result package that binds the prospective root plus:

- exact native study execution digest(s);
- canonical prefix artifact/digest identities for each analyzed cut point;
- candidate payload digests;
- outer candidate evidence-envelope digests;
- exact exclusion decision receipt digest(s);
- exclusion-evidence registry digest(s) when that later schema is part of the qualified study lineage;
- forecast trajectory artifact digests;
- weighting-decomposition report digest;
- temporal-exposure-decomposition report digest;
- candidate time-series artifact digests;
- prefix-causality validation report digest;
- suffix-sensitive-provenance separation validation digest;
- offline execution-mode/isolation report digest;
- later online-shadow equivalence report digest if online evidence is included;
- semantic-label-canary report digest;
- `ObservatoryAdversarialValidationReport` digest;
- blinded candidate-comparison report digest;
- sensitivity/robustness report digest;
- null-control report digest;
- unblinding receipt digest;
- semantic hypothesis evaluation digest;
- all excluded/indeterminate scenario artifact digests;
- complete artifact hash list.

The realized package has its own canonical SHA-256.

## Payload vs provenance closure

For `OfflinePrefixReplay`, the root must preserve the distinction between:

- prefix-causal `CandidatePayload`;
- outer `CandidateEvidenceEnvelope` containing full source-trace provenance.

Given suffix-divergent source traces with byte-identical allowed prefixes:

- prefix digest must be identical;
- candidate payload digest/value/availability must be identical;
- outer evidence-envelope/source-trace digests may differ and are expected to differ.

A root or analysis pipeline that supplies full-trace identity to candidate computation violates the information boundary and is an integrity failure.

## v0.1 qualification dependency

A confirmatory v0.2 prospective root is invalid unless the referenced v0.1 `QualificationEvidenceBundle` validates, reports qualified, and binds the exact v0.1 source commit/model-semantics lineage named by the v0.2 root.

The root must also require that its component v0.1 qualification-receipt and evidence-capsule digests match the corresponding embedded artifacts inside that bundle. Supplying two independently valid but cross-paired artifacts from different v0.1 heads is an integrity failure.

The root may be drafted before v0.1 qualifies, but it must carry a status such as `BlockedOnBaselineQualification` and cannot be promoted to `LockedConfirmatory` until the bundle dependency is satisfied.

Any new v0.1 source commit supersedes an older bundle for implementation-start or confirmatory purposes unless a newly validated bundle explicitly binds the new source head.

## Candidate-set closure

The prospective root contains the complete candidate/baseline set eligible for confirmatory interpretation.

After execution begins:

- no new primary candidate may be added;
- no factor coordinate may be changed;
- no weighting or temporal basis may be swapped because it looks more intuitive;
- no baseline may be removed because it explains the result too well;
- no failed candidate may be deleted from the package;
- a new candidate may be computed only as explicitly exploratory post-hoc analysis and must not alter the locked confirmatory root.

## Scenario-set closure

The root binds the complete confirmatory holdout cohort before candidate outputs are inspected.

Every scenario digest in the cohort must appear in final accounting as included, excluded, or indeterminate.

The realized evidence package must report:

`locked_count == included_count + excluded_count + indeterminate_count`

A mismatch is a hard integrity failure.

The scenario set must contain the prospectively required discriminators for relation semantics, burden-vs-precision separation, temporal aggregation, future-suffix invariance, forecast-policy disagreement, and relevant channel projections.

## Capability-boundary closure

The prospective root binds the capability-typed API and execution-mode contracts used by qualified computation.

Confirmatory evidence is invalid if the implementation exposes an undeclared authority path that allows prefix-causal candidates to access:

- future protocol/state information;
- full-trace or suffix-sensitive provenance identity;
- semantic arm mapping;
- mutable native execution state;
- oracle diagnostics through runtime escalation;
- control/drive/cognitive output authority.

Capability-boundary violations are integrity failures even when numerical results appear plausible.

## Adversarial-validation closure

Before semantic hypothesis evaluation can become qualified evidence, the realized package must bind an adversarial-validation report matching the locked suite/version.

All integrity-blocking adversarial gates must pass. Candidate-disqualifying gates may fail only by producing the corresponding candidate failure status; they may not be reclassified away after unblinding.

The suite must include known-malicious fixtures demonstrating that it catches at least:

- future/suffix leakage;
- full-trace provenance masquerading as prefix information;
- semantic leakage;
- hidden mutable state;
- oracle masquerading as prefix-causal;
- unavailable-as-zero behavior;
- weighting conflation;
- temporal mean/exposure conflation;
- artifact/scenario/analysis substitution.

## Artifact dependency graph

Recommended dependency direction:

`v0.1 qualification receipt + evidence capsule`

→ `v0.1 QualificationEvidenceBundle`

→ `v0.2 DesignContractRegistryManifest`

→ `v0.2 design freeze + implementation start`

→ `v0.2 prospective evidence root`

→ `native study execution`

→ `immutable prefix artifacts`

→ `forecast trajectories`

→ `factorized CandidatePayloads`

→ `outer candidate evidence envelopes`

→ `exclusion decisions / exclusion evidence registry when applicable`

→ `prefix-causality + isolation + semantic-leak + weighting/temporal + adversarial validation`

→ `frozen blinded comparison`

→ `unblinding receipt`

→ `semantic hypothesis evaluation`

→ `realized evidence package`

No downstream artifact may silently alter an upstream identity.

## Fail-closed promotion state

A later typed status should distinguish at least:

- `DraftDesign`;
- `BlockedOnBaselineQualification`;
- `LockedExploratory`;
- `LockedConfirmatory`;
- `ExecutedAwaitingValidation`;
- `Excluded`;
- `Indeterminate`;
- `QualifiedNegativeOrNullResult`;
- `QualifiedSupportedResult`;
- `IntegrityFailure`.

There should be no generic `Success` state.

A failed primary hypothesis on an otherwise valid confirmatory study is a **qualified negative/null result**, not an integrity failure.

An integrity failure means the evidence chain itself cannot support inference.

Examples include mismatched v0.1 bundle/component identities, contract-registry/freeze/root mismatch, future-information leakage, suffix-sensitive provenance in candidate computation, observational feedback into native execution, semantic-label leakage into blinded primary artifacts, capability escalation, execution replay mismatch, invalid temporal alignment, missing locked scenarios, artifact/hash substitution, or failure of an integrity-blocking adversarial gate.

## Reproduction contract

An independent reproducer should be able to start from the prospective root and verify:

1. exact source/dependency/toolchain identities;
2. v0.1 `QualificationEvidenceBundle` and its embedded qualification/evidence components;
3. complete design-contract registry and its exact relation to the design freeze;
4. design-freeze and implementation-start identities;
5. candidate definitions and factor-space coordinates;
6. finite candidate-set closure;
7. scenario cohort materialization;
8. native study execution replay;
9. prefix artifact/digest construction;
10. forecast trajectories and exact legacy-v0.1 aggregate equivalence;
11. weighting and temporal alternative candidates;
12. exclusion decisions and registry-bound evidence when applicable;
13. capability boundary/source-dependency gates;
14. prefix-causality, provenance-separation, isolation, semantic-canary, and adversarial gates;
15. blinded comparison artifact;
16. unblinding transformation;
17. semantic hypothesis outcomes;
18. final realized evidence digest.

Any discrepancy must identify the first dependency edge that fails instead of merely reporting a different final number.

## Human-readable receipt

In addition to machine-readable JSON, generate a concise deterministic text receipt summarizing:

- root digest;
- source commits;
- v0.1 qualification/evidence bundle digest;
- design-contract-registry digest;
- design-freeze digest;
- v0.1 qualification state;
- run class and execution mode;
- primary candidate ID and factor coordinate;
- candidate/baseline counts;
- scenario counts and disposition accounting;
- blinding strength;
- capability-boundary status;
- prefix-causality/provenance-separation/isolation/semantic-canary/adversarial status;
- primary hypothesis outcomes;
- integrity-gate status;
- realized evidence package digest.

The human-readable receipt is a view of machine-readable evidence, not an independently editable source of truth.

## Claim boundary

A valid evidence root can show that one complete fixed prospective research design led reproducibly to one fixed set of results without silent substitution and with declared information, weighting, temporal, and authority boundaries. It does not by itself establish that any regulatory candidate is emotion, subjective valence, mood, suffering, sentience, or consciousness.
