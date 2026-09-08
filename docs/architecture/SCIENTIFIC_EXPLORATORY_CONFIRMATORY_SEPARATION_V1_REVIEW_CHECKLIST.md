# SCI-005 Review Checklist — Exploratory / Confirmatory Separation v1

**Status:** review aid only; non-authorizing; non-qualifying.

## A. Fundamental separation

- [ ] Exploratory evidence is distinct from confirmatory evidence.
- [ ] Frozen confirmatory contract is distinct from prospective evidence eligibility.
- [ ] Hidden data is distinct from unexposed information.
- [ ] Unexposed information is distinct from independent evidence.
- [ ] Confirmatory failure remains evidence.
- [ ] Exploratory evidence remains scientifically useful rather than being treated as low quality by definition.
- [ ] Confirmatory evidence remains distinct from scientific truth and action authority.

## B. Contextual exposure model

- [ ] Exposure binds exact artifact identity.
- [ ] Exposure binds exact information projection rather than assuming whole-file visibility.
- [ ] Exposure binds decision-process identity.
- [ ] Exposure binds target/target-family context where relevant.
- [ ] Exposure binds scientific use.
- [ ] Exposure history is append-only.
- [ ] Revoking future access cannot erase historical exposure.
- [ ] Unknown/incomplete exposure fails closed for strong eligibility where material.

## C. Information projections

- [ ] Features/covariates can remain distinct from labels/outcomes.
- [ ] Score-only feedback can remain distinct from full-label reveal.
- [ ] Aggregate feedback can remain distinct from case-level reveal.
- [ ] Derived representations can carry outcome information.
- [ ] Domain profiles decide which projections are target-relevant.
- [ ] Unknown projection is not silently treated as harmless.

## D. Decision-process scope

- [ ] Human analyst exposure is tracked where relevant.
- [ ] Model/training-pipeline exposure is tracked where relevant.
- [ ] Hyperparameter/model-selection agents can count as decision processes.
- [ ] Retrieval systems/model weights/learned grammar can carry prior information.
- [ ] The architecture does not assume “the current operator never saw it” means the system is uncontaminated.

## E. Role/use separation

- [ ] Hypothesis generation is distinct from fitting/tuning/calibration/evaluation.
- [ ] Calibration data cannot silently become untouched holdout evidence.
- [ ] Model-selection feedback is represented as exposure/use.
- [ ] Threshold/metric/stopping-rule selection is represented as exposure/use where material.
- [ ] Publication/reporting selection can be represented as outcome-aware use.

## F. No direct role conversion

- [ ] No convenience conversion exists from exploratory result to confirmatory evidence.
- [ ] A legitimate confirmatory path requires a new prospective contract plus eligible evidence.
- [ ] Exploratory evidence can remain in hypothesis/prior/design provenance without becoming new confirmatory likelihood evidence.
- [ ] Same evidence reused in prior and likelihood cannot be counted as independent support.

## G. Scoped data reuse

- [ ] The system does not impose a universal “one use ever” rule.
- [ ] Covariate-only exposure can be distinguished from outcome exposure when scientifically justified.
- [ ] A genuinely distinct target/use can receive an independent eligibility assessment.
- [ ] Target overlap/equivalence uncertainty fails closed rather than being assumed disjoint.
- [ ] Reuse preserves all historical exposure/dependency lineage.

## H. Representation laundering

- [ ] Copying/renaming files does not create fresh evidence.
- [ ] Reformatting/compressing/reordering data does not reset exposure.
- [ ] Derived summaries/embeddings/models can retain contamination ancestry.
- [ ] Different SCI-002 content identities are not treated as evidence of informational independence.
- [ ] Serialization/deserialization cannot reset exposure history.

## I. Hidden benchmark lifecycle

- [ ] Hiddenness is modeled as lifecycle/context rather than permanent boolean.
- [ ] Relevant reveal makes the benchmark spent for the declared scope.
- [ ] Spent generation cannot return to sealed-fresh for the same scope.
- [ ] Fresh benchmark generation requires substantive scientific freshness, not Git/path renaming.
- [ ] Historical/spent benchmark remains useful development/historical evidence under explicit role.

## J. Repeated-query leakage

- [ ] Score/leaderboard feedback can count as exposure.
- [ ] Query count/budget can be represented where material.
- [ ] Score precision/subgroup/per-case feedback can be represented.
- [ ] Autonomous optimization against hidden scores cannot be treated as no exposure by default.
- [ ] Final one-shot holdback generation can remain distinct when prospectively protected.

## K. Model/tool/historical contamination

- [ ] Model training cutoff can be relevant.
- [ ] Fine-tuning/retrieval/embedding/tool corpora can be relevant.
- [ ] Learned grammar/macros can carry target knowledge.
- [ ] Historical replay visible-corpus cutoff alone is not sufficient.
- [ ] Prior benchmark runs/results can contaminate later historical evaluation.

## L. Confirmatory failure lifecycle

- [ ] Null/negative result remains evidence for the completed campaign.
- [ ] Using that result to tune a successor creates new development lineage.
- [ ] Tuned successor requires new prospective contract for new confirmation.
- [ ] Spent evidence cannot automatically confirm the tuned successor for the affected scope.

## M. Cross-validation/resampling

- [ ] Cross-validation is not classified as automatically exploratory or confirmatory.
- [ ] Prospectively frozen fold/partition policy can support scoped confirmation.
- [ ] Nested tuning/evaluation roles can remain distinct.
- [ ] Post-hoc fold/model/metric selection cannot inherit confirmatory authority.
- [ ] Claim scope matches the declared resampling design.

## N. Adaptive campaigns

- [ ] SCI-004 precommitted adaptive policy may react to allowed observations.
- [ ] Exposure ledger accumulates each revealed step.
- [ ] Final result is interpreted as one adaptive campaign, not a collection of independent one-shot tests.
- [ ] Manual actions outside frozen adaptation policy become protocol deviations.

## O. Replication

- [ ] Prospective cleanliness and dependency independence remain separate questions.
- [ ] `ReplicationAttempt` does not imply independence.
- [ ] SCI-006 will own dependency/replication topology.
- [ ] A prospectively clean replication may still share major dependencies.

## P. Positive eligibility

- [ ] Positive eligibility is a derived scoped witness, not artifact field.
- [ ] Exact SCI-004 contract/preregistration identity is required.
- [ ] Exact artifact/projection/process/use context is required.
- [ ] Relevant exposure/custody/dependency evidence is required by policy.
- [ ] Private positive wrapper is not directly deserializable into live authority.
- [ ] Archived eligibility is audit material, not self-restoring current authority.

## Q. Non-positive outcomes

- [ ] Eligibility failure is not reduced to one `false` if the reason is scientifically relevant.
- [ ] Prior outcome exposure can be distinguished from tuning use.
- [ ] Target-overlap uncertainty can be distinguished from known contamination.
- [ ] Custody/chronology/dependency incompleteness remain explicit.

## R. First implementation gate

The first shared implementation should record exposure only:

- `InformationProjectionV1`;
- `DecisionProcessIdentityV1`;
- `EvidenceUseV1`;
- `ExposureEventV1`;
- `ExposureLedgerV1`.

Reject first-tranche introduction of:

- positive prospective-eligibility issuance;
- confirmatory evidence issuance;
- measurement qualification;
- replication authority;
- action authority.

## Review question

> Does SCI-005 block exploratory/tuning/previously revealed information from being relabeled as fresh confirmation while still supporting cumulative science, scoped reuse, cross-validation, and prospectively constrained adaptive experiments?
