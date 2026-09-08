# SCI-004 Review Checklist — Scientific Experiment Contract v1

**Status:** review aid only; non-authorizing; non-qualifying.

Use this checklist to review `SCIENTIFIC_EXPERIMENT_CONTRACT_V1.md` without letting contract immutability, preregistration chronology, execution, measurement, adjudication, replication, or action authority collapse into one state.

## A. Prospective boundary

- [ ] Draft design is distinct from frozen contract.
- [ ] Frozen contract is distinct from verified preregistration chronology.
- [ ] Preregistration chronology is distinct from experiment execution.
- [ ] Execution is distinct from valid measurement.
- [ ] Measurement is distinct from claimed outcome success.
- [ ] Outcome success is distinct from scientific truth.
- [ ] Confirmatory scientific evidence is distinct from action authority.
- [ ] A post-outcome contract cannot regain naive prospective status merely by receiving a timestamp.

## B. Study class

- [ ] Exploratory and Confirmatory are explicit distinct intended uses.
- [ ] ReplicationAttempt does not imply independent replication.
- [ ] BenchmarkQualification does not imply external scientific validity.
- [ ] MethodValidation is not silently treated as validation of every downstream application.
- [ ] Study-class labels do not mint authority by themselves.

## C. Target identity

- [ ] Exact proposition/hypothesis/estimand/claim target is bound.
- [ ] Human-readable title is not the sole target identity.
- [ ] Target substitution after reveal creates a new generation and loses predecessor prospective authority.
- [ ] Causal estimand semantics remain domain-owned where applicable.

## D. Input/data eligibility

- [ ] Input/source/vintage/snapshot eligibility is frozen or governed by a frozen policy.
- [ ] Exact membership is retained where membership itself is scientifically material.
- [ ] Training/fitting/calibration/evaluation/holdout roles remain distinguishable where applicable.
- [ ] Generic SCI-004 does not force train/test terminology onto domains that do not use it.
- [ ] Actual downstream execution can be checked against the frozen role policy.
- [ ] Evaluation data used in fitting cannot silently retain fresh confirmatory status.

## E. Intervention/control design

- [ ] Intervention/treatment/control semantics are frozen before outcome-aware use.
- [ ] Assignment/randomization/adaptation rules are explicit where applicable.
- [ ] A declaration that randomization was intended is distinct from evidence that randomization executed correctly.
- [ ] Scientific intervention specification does not become physical effect authority.

## F. Measurement specification

- [ ] Theoretical variable/construct is distinct from measurement specification.
- [ ] Source/instrument/method/units/operationalization are bound where material.
- [ ] Transformation/preprocessing/aggregation is bound where material.
- [ ] Measurement window is explicit where material.
- [ ] Uncertainty model is distinct from source/measurement validity.
- [ ] Contract validity does not prove measurement validity.

## G. Outcome-observation policy

- [ ] The contract freezes what realized outcome counts before reveal.
- [ ] Mutable/revised sources bind source + series/measure + release/vintage policy where applicable.
- [ ] Transform identity is frozen where applicable.
- [ ] Missing-outcome disposition is prospective.
- [ ] Outcome-selection policy is not supplied after reveal by the same result being judged.
- [ ] SCI-004 preserves the Futures theorem that resolution linkage is not preregistered outcome semantics.

## H. Analysis plan

- [ ] Analysis method label is distinct from exact implementation when implementation is material.
- [ ] Exact estimator/model/algorithm policy is bound where relevant.
- [ ] Hyperparameter-selection/adaptation policy is prospective where it affects inference.
- [ ] Statistical test/scoring/loss semantics are explicit.
- [ ] Causal identification strategy remains separate from estimator implementation.
- [ ] SCI-003 execution identity can bind actual analysis execution later.

## I. Metrics/criteria

- [ ] Metric definition is distinct from metric value.
- [ ] Decision threshold/rule is distinct from scientific disposition.
- [ ] Result labels remain scoped, e.g. `MeetsAllDeclaredBounds`, not universal `Validated`.
- [ ] No default threshold silently becomes scientific authority across domains.

## J. Multiplicity

- [ ] Primary/secondary endpoints or hypothesis families are explicit where multiplicity matters.
- [ ] Multiplicity policy is frozen before outcome-aware selection.
- [ ] SCI-004 does not prescribe one universal correction method.
- [ ] Subgroup/model/metric selection cannot silently expand the confirmatory search space post hoc.

## K. Stopping/sample-size

- [ ] Fixed or adaptive stopping rule is explicit where stopping affects inference.
- [ ] Sequential/adaptive designs are allowed when the adaptation rule itself is precommitted.
- [ ] Stopping after a favorable intermediate result outside the frozen policy is a protocol deviation.
- [ ] Resource ceilings/futility rules are explicit where they alter interpretation.

## L. Adaptive scientific planning

- [ ] Candidate experiment/action space is prospectively bounded when adaptive selection is confirmatory.
- [ ] Selection/planner implementation identity is frozen or governed by an exact policy.
- [ ] Information/falsification objective is declared before adaptive use.
- [ ] Cost/risk/constraint policy is explicit where material.
- [ ] Update and stopping rules are precommitted.
- [ ] Intelligent adaptive science remains allowed without granting unrestricted outcome-aware discretion.
- [ ] Changing the planner/adaptation algorithm creates a new contract generation when decision-relevant.

## M. Missingness/exclusions

- [ ] Missing-data disposition is frozen before missingness becomes outcome-dependent.
- [ ] Exclusion eligibility is predeclared where exclusions affect inference.
- [ ] Actual exclusions become evidence-bearing downstream records.
- [ ] Inconvenient missing/failed cases cannot silently disappear.
- [ ] A post-hoc reason not represented in the frozen policy cannot be described as preregistered.

## N. Blinding/custody/reveal

- [ ] Reveal boundary is explicit when prospective status depends on hidden outcomes/labels.
- [ ] Contract declaration is distinct from evidence that custody/blinding actually held.
- [ ] Historical availability is distinct from live possession/custody.
- [ ] Model/tool access to hidden/post-cutoff knowledge is considered contamination, not only human analyst access.

## O. Exposure/contamination

- [ ] Prior outcome exposure can be represented or detected by later layers.
- [ ] Previously revealed benchmark data cannot become fresh hidden data by relabeling.
- [ ] Model training, retrieval corpora, embeddings, learned grammar, and tools can participate in contamination lineage.
- [ ] Contract timestamp alone cannot erase prior exposure.

## P. Amendments/versioning

- [ ] Contracts are immutable after issuance.
- [ ] Pre-reveal amendments create new generations and require new chronology evidence.
- [ ] Decision-relevant post-reveal amendments do not inherit predecessor confirmatory authority.
- [ ] Historical generations remain auditable.
- [ ] Post-reveal reanalysis can remain scientifically useful under an explicitly exploratory/scoped status.

## Q. Preregistration receipt

- [ ] Contract identity is distinct from chronology evidence.
- [ ] `preregistered: bool` cannot mint authority.
- [ ] Receipt binds the exact contract generation.
- [ ] Receipt identifies the relevant reveal/outcome boundary.
- [ ] Declared timestamps are distinct from verified chronology.
- [ ] External timestamping alone is not treated as proof of no prior exposure.
- [ ] Persistence/deserialization cannot recreate current chronology authority without revalidation.

## R. SCI-003 execution binding

- [ ] Contract names exact admissible execution profile/capsule semantics.
- [ ] Later attempt receipts bind the exact contract generation.
- [ ] Neighboring configuration/run substitution fails.
- [ ] If a configuration family is permitted, the frozen admissibility rule and exact realized selection are retained.

## S. Reporting/deviations

- [ ] Null/negative/indeterminate results remain reportable lineage.
- [ ] Failed/incomplete executions remain lineage.
- [ ] Protocol deviations are first-class artifacts, not silent contract edits.
- [ ] Deviation existence is distinct from automatic invalidation; later adjudication decides its effect.

## T. Replication/safety/authority

- [ ] Same protocol does not imply independent replication.
- [ ] Replication dependency belongs to SCI-006/later analysis.
- [ ] Preregistered safety criteria are distinct from discharged safety evidence.
- [ ] Safety evidence is distinct from safety/effect authority.
- [ ] Experiment contract cannot directly grant deployment/action/self-modification authority.

## U. First implementation gate

The first shared Rust tranche should remain:

```text
StudyClassV1
ExperimentContractV1
ExperimentContractIdentityV1
```

and establish only:

```text
contract semantics frozen
```

Reject first-tranche expansion into:

- chronology verification;
- solver/tool execution;
- measurement qualification;
- universal result adjudication;
- replication authority;
- safety authority;
- action authority.

## Review question

> Does SCI-004 freeze all decision-relevant semantics that could otherwise be chosen after outcome exposure while preserving adaptive science and keeping chronology, execution, measurement, adjudication, replication, safety, and action authority as independent layers?
