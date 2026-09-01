# Affective Emergence v0.2 — Observational Evidence Root

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines the prospective and realized evidence roots that bind one complete observational-affect lineage.

## 1. Principle

No digest proves scientific validity, but a complete root makes silent substitution, data leakage, and cross-lineage pairing detectable.

A result should answer:

> Exactly which qualified native baseline, frozen design, candidate definitions, preprocessing/calibration parameters, evaluator-state policy, scenario cohort, analysis plan, information boundary, blinding commitment, and validation suite produced this result?

If any evidence-critical identity changes, the root identity changes.

## 2. Complete-design dependency

The prospective root binds the same exact `DesignContractRegistryManifest` as the referenced `DesignFreezeManifest`.

Record:

- `design_freeze_sha256`;
- `design_contract_registry_sha256`;
- registry-contract specification digest/version.

Validator requires exact equality between root and freeze registry identities.

## 3. Proposed ObservationalEvidenceRootManifest

Before confirmatory execution, freeze at minimum:

- root schema/version and run class;
- exact v0.1 source commit/model-semantics version;
- authoritative v0.1 `QualificationEvidenceBundle` digest;
- component qualification/evidence-capsule digests for audit;
- exact design-freeze and contract-registry digests;
- exact `ImplementationStartReceipt` digest;
- exact v0.2 observatory source/toolchain/dependency identities;
- primary execution mode (`OfflinePrefixReplay` initially);
- replay/prefix/payload/evidence-envelope schema identities;
- study/preregistration identities;
- finite candidate-set manifest digest;
- ordered primary/secondary/baseline candidate-definition digests;
- candidate factor coordinates;
- each candidate's preprocessing-manifest digest or explicit `None`;
- calibration-cohort and fitted-parameter artifact digests where applicable;
- evaluator-isolation-manifest digest;
- allowed evaluator persistent-state/cache class;
- scenario/cohort manifest digest;
- candidate-discrimination manifest digest;
- causal-contrast manifest digest set;
- analysis-plan digest;
- exclusion-criteria definition digest;
- arm-mapping commitment digest;
- information/capability/history/weighting/aggregation/temporal/calibration/evaluator/adversarial contract versions;
- blinding-strength declaration;
- required integrity-gate versions;
- canonical root SHA-256.

This root freezes before confirmatory execution.

## 4. Baseline qualification dependency

A confirmatory root is invalid unless the referenced v0.1 `QualificationEvidenceBundle`:

- validates structurally;
- reports qualified;
- binds the exact v0.1 source/model-semantics identities recorded by the root;
- contains the matching qualification receipt/evidence capsule.

Cross-paired valid components are an integrity failure.

A drafted root may remain `BlockedOnBaselineQualification`, but cannot become `LockedConfirmatory` until this dependency passes.

## 5. Candidate identity closure

The root binds a finite, prospectively closed candidate set.

Every candidate identity includes:

- mathematical factor coordinate (`R × W × A × T × F × I × H` as applicable);
- exact formula/fixtures/source identity;
- exact preprocessing/calibration identity;
- exact evaluator-isolation/state/cache identity;
- native lineage and frozen design identity.

A formula cannot hide changed preprocessing or evaluator authority. Same coordinate + different value-changing preprocessing or evaluator-state policy is a different scientific candidate unless equivalence is prospectively proven.

No new primary candidate, factor coordinate, preprocessing variant, or baseline may be inserted after confirmatory execution begins.

## 6. Calibration / preprocessing closure

For every confirmatory primary/baseline candidate, the root binds all fitted preprocessing parameters **before** execution.

Allowed parameter sources are:

- structural constants;
- prospectively identified discovery/calibration cohort;
- independently identified external reference.

The root additionally binds:

- calibration cohort manifest;
- fitted-parameter artifact;
- calibration/confirmatory content-overlap audit;
- preprocessing sensitivity-set identity when required.

Confirmatory cohort outcomes cannot change individual candidate preprocessing.

Adding/removing/reordering a confirmatory scenario must not change another scenario's candidate payload.

## 7. Evaluator-state closure

The root binds one `EvaluatorIsolationManifest` under `V02_OBSERVATORY_STATE_LIFECYCLE.md`.

Initial primary evidence requires persistent-state class:

`NoneAcrossEvaluationCoordinates`.

The realized package must show candidate artifacts are invariant to:

- scenario order;
- candidate order;
- duplicate evaluation/removal;
- cold vs warm process;
- allowed cache enabled/disabled;
- serial vs parallel execution;
- allowed batch-size/chunk changes;
- incremental vs from-scratch prefix evaluation where incremental H1 computation exists.

Cross-scenario mutable state or order dependence is an `IntegrityFailure`.

## 8. History/state-sufficiency closure

The root records each candidate's history-access basis.

H1 replay history means external observatory information, not native memory.

For claims that H1 adds information beyond H0, bind matched-current-state discriminators.

The realized package also binds the native restart/state-sufficiency report proving identical complete native state/configuration + identical future input produces identical future native execution under the v0.1 deterministic contract.

Any future H2 native persisted memory requires a new qualified architecture lineage.

## 9. Scenario / calibration / holdout closure

Scenario roles are explicit:

- discovery;
- calibration;
- confirmatory holdout;
- diagnostic.

A scenario used to fit preprocessing or choose a formula/threshold is not an untouched confirmatory holdout.

The root binds content-overlap audits, cut points/windows, causal-contrast identities, and required pairwise discrimination obligations.

Every locked confirmatory scenario is finally accounted for as:

- included;
- excluded with evidence;
- indeterminate with evidence.

Required equality:

`locked_count == included_count + excluded_count + indeterminate_count`.

## 10. Analysis-plan identity

The canonical `AnalysisPlan` binds at minimum:

- primary hypotheses;
- exact candidate/baseline/preprocessing identities;
- scenario/cut-point references;
- directional relations/minimum-effect thresholds;
- equivalence tolerances;
- sensitivity region;
- candidate ranking/parsimony rule;
- treatment of ties, ambiguity, preprocessing/calibration sensitivity, and indeterminate outcomes;
- required discrimination/causal-contrast checks;
- required prefix/history/restart/isolation/leakage gates;
- deterministic robustness summaries;
- exclusion/unblinding procedures.

Changing these after confirmatory execution starts creates a new analysis/root lineage.

## 11. Prospective vs realized evidence

### Prospective root

Contains what was promised before confirmatory data:

- qualified baseline;
- complete design registry/freeze/start receipt;
- candidate definitions + preprocessing/calibration + evaluator-isolation identities;
- finite candidate set;
- scenario/calibration/holdout cohorts;
- discrimination/causal contrasts;
- analysis;
- capability/information/history boundaries;
- blinding;
- toolchain/environment.

### Realized package

After execution, bind the prospective root plus:

- native execution digests;
- prefix artifact/digest identities;
- forecast trajectory artifacts;
- candidate payloads and outer evidence envelopes;
- exclusion decisions/evidence;
- restart/state-sufficiency report;
- preprocessing reproduction and holdout-leakage audit;
- evaluator isolation/order/cache/concurrency reports;
- weighting/aggregation/temporal decomposition reports;
- candidate fingerprints/equivalence classes;
- discrimination/manipulation-check reports;
- prefix/suffix/provenance-separation validation;
- semantic-label canary;
- adversarial-validation report;
- blinded comparison;
- sensitivity/null-control reports;
- unblinding receipt;
- semantic hypothesis evaluation;
- excluded/indeterminate artifacts;
- complete artifact hash list;
- realized package SHA-256.

## 12. Payload vs provenance closure

For suffix-divergent traces with identical allowed prefix:

- prefix digest is identical;
- candidate payload is identical;
- outer evidence/source-trace provenance may differ.

Full-trace or suffix-sensitive identity entering candidate computation is an integrity failure.

## 13. Population analysis occurs after individual artifacts freeze

Individual candidate payloads may not depend on confirmatory cohort mean/variance, ranking, exclusions, or other candidate outputs.

Population summaries and model comparison occur only after all locked individual blinded candidate artifacts have been frozen.

This prevents cohort-level preprocessing/evaluator state from contaminating per-scenario evidence.

## 14. Identifiability closure

A primary result cannot be `QualifiedSupportedResult` for superiority over baseline B when the locked design did not make candidate C vs B identifiable.

The root binds:

- discrimination manifest;
- equivalence tolerance;
- scenario/cut-point discriminator coverage;
- parsimony rule.

Realized evidence binds fingerprints, pairwise results, equivalence classes, and any `InsufficientDiscrimination` findings.

## 15. Causal-contrast closure

Mechanistic claims require the prospective root to bind causal contrasts and the realized package to bind passing manipulation checks.

Without these, report descriptive scenario differences only.

Nuisance matching must not silently condition away intended mediators.

## 16. Capability / causal-isolation closure

Primary candidate code may not access:

- unseen future state/schedule;
- full-trace provenance;
- semantic arm mapping;
- mutable native execution state;
- oracle authority;
- action/drive/cognitive output authority;
- cross-scenario evaluator memory;
- confirmatory cohort statistics used to adapt individual values.

Violations block inference even when numerical results look compelling.

## 17. Adversarial-validation closure

The realized package binds the exact locked adversarial suite and known-malicious fixtures.

Required attacks include at least:

- future/suffix leakage;
- full-trace provenance leakage;
- semantic leakage;
- native-observer feedback;
- hidden global/thread-local evaluator state;
- evaluation-order dependence;
- cache-key/value dependence on forbidden information;
- incremental reset failure;
- full-confirmatory-cohort z-score/scaler leakage;
- adaptive clipping/threshold fitting;
- unavailable-as-zero behavior;
- weighting/denominator/mean-exposure conflation;
- artifact/scenario/analysis substitution.

Integrity-blocking failures produce `IntegrityFailure`; candidate-theory failures remain negative/null evidence.

## 18. Artifact dependency graph

Recommended direction:

`v0.1 qualification components`
→ `QualificationEvidenceBundle`
→ `DesignContractRegistryManifest`
→ `DesignFreezeManifest`
→ `ImplementationStartReceipt`
→ `prospective evidence root`
→ `native executions`
→ `prefix artifacts / forecast trajectories`
→ `preprocessing parameters + evaluator context`
→ `CandidatePayloads`
→ `CandidateEvidenceEnvelopes`
→ `validation / exclusion / discrimination / causal reports`
→ `frozen blinded comparison`
→ `unblinding receipt`
→ `semantic hypothesis evaluation`
→ `realized evidence package`.

No downstream artifact may silently rewrite an upstream identity.

## 19. Promotion states

Distinguish at least:

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

There is no generic `Success` state.

A failed hypothesis under an intact lineage is qualified negative/null evidence. A preprocessing leak, evaluator-order effect, future leak, missing scenario, replay mismatch, or cross-lineage substitution is an integrity failure.

## 20. Reproduction contract

An independent reproducer should be able to verify, in dependency order:

1. source/toolchain identities;
2. v0.1 qualified bundle;
3. registry/freeze/start receipt;
4. candidate definitions/factor coordinates;
5. preprocessing/calibration parameters from their source cohort;
6. evaluator isolation/state/cache policy;
7. scenario/cohort materialization and overlap audits;
8. native execution replay;
9. prefix construction;
10. forecast trajectories and legacy-v0.1 equivalence;
11. candidate payloads in clean processes;
12. restart/state-sufficiency, preprocessing-leakage, evaluator-isolation, discrimination, causal, semantic, and adversarial gates;
13. blinded comparison;
14. unblinding transformation;
15. semantic hypothesis outcomes;
16. final realized digest.

A discrepancy reports the earliest failed dependency edge.

## 21. Human-readable receipt

Generate a deterministic view summarizing:

- root/realized digests;
- source commits;
- v0.1 qualification bundle/state;
- design registry/freeze/start identities;
- run/execution class;
- primary candidate and preprocessing/calibration identity;
- evaluator persistent-state class;
- candidate/baseline/scenario counts;
- blinding strength;
- prefix/restart/preprocessing/isolation/discrimination/causal/adversarial gate states;
- scenario disposition accounting;
- hypothesis outcomes;
- integrity status.

The text receipt is a view, not an editable source of truth.

## 22. Claim boundary

A valid root can show that one fixed, qualification-bound, leakage-controlled observational design reproducibly produced one fixed result set. It does not establish emotion, subjective valence, native mood, native memory, suffering, sentience, or consciousness.