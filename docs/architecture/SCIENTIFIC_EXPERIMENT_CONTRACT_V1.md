# Scientific Experiment Contract v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Series:** SCI-004, stacked on SCI-003.

**Parent:** `architecture/scientific-execution-capsule-v1@699e18efe7fca700d329019b986cd96dde6abb9e`

## 1. Purpose

SCI-004 defines the prospective experiment boundary needed to distinguish genuine confirmatory evidence from exploratory analysis performed after outcomes are visible.

Symthaea already contains strong domain-native preregistration patterns:

- Physical Agency freezes exact outcome claims before strict simulation execution and separately freezes safety obligations before the run;
- Matter nuclear calibration separates fit, calibration, and structural holdout sets and requires externally anchored preregistered metric bounds without claiming local chronology proof;
- Futures freezes source/series/release/vintage/transform/missing-outcome semantics before reveal;
- economics diagnostics distinguish identification assumptions from diagnostic specifications/results and bind preregistered thresholds/implementation identity;
- research/evaluation work repeatedly treats null, negative, indeterminate, failed, and missing outcomes as evidence rather than silently deleting them.

The common kernel needs a contract that preserves those mechanics without pretending one statistical design applies to every science.

---

## 2. Core theorem

SCI-004 preserves:

```text
hypothesis
    != experiment contract

experiment contract
    != verified preregistration chronology

preregistered design
    != executed experiment

executed experiment
    != valid measurement

valid measurement
    != successful outcome

successful outcome
    != scientific truth

exploratory analysis after reveal
    != prospective confirmatory evidence

preregistration
    != methodological quality

confirmatory scientific evidence
    != action authority
```

The central rule is:

> A confirmatory result may receive prospective status only from a contract whose decision-relevant semantics were frozen before the relevant outcome information became available to the decision process.

---

## 3. Contract layers

SCI-004 distinguishes four objects.

### 3.1 Draft experiment design

Mutable planning object.

May be edited freely and has no prospective authority.

### 3.2 Frozen experiment contract

Immutable content-addressed experiment specification.

Conceptually:

```text
ExperimentContractV1 {
    contract_profile
    study_class
    target_proposition_or_hypothesis_set
    admissible_inputs
    sampling_or_case_selection_design
    intervention_or_treatment_policy
    control_or_comparator_policy
    measurement_specifications
    outcome_observation_policy
    analysis_plan
    estimator_or_model_policy
    uncertainty_policy
    metrics
    decision_criteria
    multiplicity_policy
    stopping_rule
    missing_data_policy
    exclusion_policy
    reveal_or_blinding_policy
    adaptation_policy
    required_execution_profiles
    reporting_policy
}
```

The exact fields required are profile/domain controlled.

### 3.3 Preregistration receipt

Evidence that one exact frozen contract existed before a declared reveal/outcome boundary.

```text
FrozenExperimentContract
    + chronology/custody evidence
    -> PreregistrationReceiptV1
```

The frozen contract itself does **not** prove chronology.

### 3.4 Experiment execution/adjudication lineage

SCI-003 execution receipts and later measurement/adjudication objects consume the exact contract identity.

```text
PreregistrationReceipt
    + exact executions
    + exact observations
    + exact analysis
    -> adjudication input
```

Adjudication belongs to later layers; the contract does not decide its own success.

---

## 4. Study class

The contract must explicitly distinguish at least:

```text
Exploratory
Confirmatory
ReplicationAttempt
BenchmarkQualification
MethodValidation
```

These labels describe intended evidentiary use; they do not by themselves grant authority.

### 4.1 Exploratory

May adapt freely to observed data/outcomes, subject to provenance rules.

Exploratory work can generate new hypotheses and future experiment contracts.

### 4.2 Confirmatory

Decision-relevant semantics must be frozen before the relevant outcome/reveal boundary.

### 4.3 Replication attempt

Requires a separate replication/dependency analysis later. Calling a contract `ReplicationAttempt` does not establish independence.

### 4.4 Benchmark qualification

May freeze software/protocol acceptance criteria without implying external scientific truth.

---

## 5. Target binding

The contract must identify exactly what is being tested.

Depending on the domain:

```text
scientific proposition identity
hypothesis set
causal estimand
forecast target
model-performance claim
conservation-law candidate
simulation outcome claim
method-validation claim
```

A human-readable title is not sufficient.

Target identity should eventually use SCI-002 semantic artifact/proposition identities.

Changing a target after reveal creates a new contract generation and cannot inherit prospective status from the predecessor.

---

## 6. Input/data eligibility

The contract must freeze or bind the policy that determines which data/cases may enter the experiment.

Possible coordinates:

```text
source identities
source vintages
input snapshot identities
eligibility criteria
sampling frame
case selection
training/calibration/evaluation role assignment
holdout policy
historical cutoff
custody/reveal state requirements
```

The generic kernel must not assume every experiment uses train/test splits.

The domain profile owns the exact design.

### 6.1 Membership must be explicit enough to audit

Where exact membership is scientifically important, retain exact artifact/case identities rather than only a prose rule.

### 6.2 Data-role leakage

The contract should be able to distinguish roles such as:

```text
Exploration
Training
Fitting
Calibration
Validation
ConfirmatoryEvaluation
StructuralHoldout
NegativeControl
```

but these are optional vocabulary, not one universal required partition.

A later evidence layer must verify that actual use respected the frozen roles.

---

## 7. Intervention/treatment/control semantics

For experiments involving interventions, freeze the intervention assignment/generation policy before outcome observation.

Potential fields:

```text
intervention identity
control/comparator identity
assignment rule
randomization procedure
blocking/stratification
perturbation schedule
allowed dose/intensity range
adaptive assignment policy
```

SCI-004 does not provide physical effect authority. It records scientific design only.

### 7.1 Randomization declaration is not verified randomization

```text
contract says randomized
    != randomization executed correctly
```

Actual assignment/execution evidence belongs downstream.

---

## 8. Measurement specification

Outcome variables must not be invented or redefined after seeing results.

Bind, where applicable:

```text
theoretical variable / construct
measurement source
instrument/method
units
operationalization
transformation/preprocessing
aggregation
measurement window
measurement uncertainty model
quality/missingness handling
```

This preserves the economics theorem:

```text
theory variable != measurement specification != qualified observation
```

Measurement validity remains a later evidentiary question.

---

## 9. Outcome observation policy

SCI-004 generalizes the Futures result that scoring rules alone are insufficient; the contract must also define **what realized outcome will be observed**.

Where external mutable/revised sources are used, the policy may need to freeze:

```text
source identity
series/measure identity
release-selection rule
vintage/revision policy
transformation identity
missing-outcome disposition
```

Examples of typed release selection:

```text
FirstPublished
Ordinal(n)
FixedVintage(id)
```

The generic kernel does not require these exact selectors outside relevant domains.

The principle is normative:

> Outcome selection semantics must not be chosen after observing which candidate outcome is favorable.

---

## 10. Analysis plan

The contract must identify how observations become reported statistics/results.

Possible coordinates:

```text
analysis implementation identity
estimator/model family
hyperparameter policy
feature construction
normalization/transformation
aggregation
statistical test
loss/scoring rule
confidence/credible interval method
bootstrap/resampling policy
causal identification strategy
sensitivity analyses
```

The exact implementation may be frozen directly or selected through a preregistered deterministic/adaptive policy.

### 10.1 Specification vs implementation

```text
analysis method label
    != exact analysis implementation
```

Where implementation details materially affect results, bind the implementation artifact and SCI-003 execution profile.

---

## 11. Metrics and decision criteria

The contract should distinguish:

```text
metric definition
observed metric value
decision threshold/rule
scientific disposition
```

A threshold is not a scientific truth oracle.

Possible outcome labels should remain scoped, e.g.:

```text
MeetsAllDeclaredBounds
ViolatesAtLeastOneDeclaredBound
IndeterminateUnderDeclaredCriteria
```

rather than automatically:

```text
Validated
ScientificallyTrue
```

This preserves the Matter calibration boundary.

---

## 12. Multiplicity

If multiple hypotheses, metrics, subgroup analyses, model variants, or interim tests can generate confirmatory claims, the contract should freeze the multiplicity policy before reveal.

Possible approaches include:

```text
predeclared primary endpoint
family-wise correction
false-discovery control
hierarchical testing
closed testing
no multiplicity correction with explicitly limited claim scope
```

SCI-004 does not prescribe one universal statistical correction.

It requires the relevant policy to be explicit when multiplicity affects inference.

---

## 13. Stopping and sample-size rules

A confirmatory design must specify how/when data collection or execution stops if stopping can affect the reported result.

Examples:

```text
fixed sample size
fixed number of seeds/replicates
fixed time horizon
sequential boundary
Bayesian stopping rule
resource ceiling
futility rule
adaptive information criterion
```

### 13.1 Adaptive stopping is allowed

Preregistration does not require a static experiment.

An adaptive/sequential design is prospective when the **adaptation rule itself** is frozen before outcomes guide adaptation.

```text
precommitted adaptive policy
    != post-hoc researcher discretion
```

---

## 14. Adaptive experiment selection

This is especially important for Symthaea because active-learning/scientific-agent systems select experiments iteratively.

A confirmatory adaptive campaign may freeze:

```text
candidate experiment/action space
selection algorithm identity
information/falsification objective
constraints
cost/risk policy
update rule
stopping policy
reveal boundaries
```

It does **not** need to enumerate every future selected experiment in advance.

The contract must make it possible to distinguish:

```text
precommitted adaptive search
    != outcome-aware manual goalpost movement
```

This allows future SCI-009 experiment planning to remain scientifically prospective.

---

## 15. Missing data and exclusions

Missingness/exclusion semantics must be fixed before they can become outcome-dependent.

Possible states/policies:

```text
RemainMissing
ResolveMissingWithoutScore
ImputeUnderDeclaredMethod
ExcludeOnlyUnderPredeclaredCriterion
CountAsFailure
AbortExperiment
```

A missing or inconvenient outcome must not create a post-hoc option to silently delete the case.

### 15.1 Evidence-bearing exclusions

Actual exclusions should produce auditable records identifying:

```text
case/input
predeclared criterion
observed reason/evidence
decision
```

The contract defines eligibility; later receipts prove how exclusions were applied.

---

## 16. Blinding, custody, and reveal

A confirmatory contract may need to bind a reveal/custody policy.

Possible concepts:

```text
outcome hidden until freeze
labels hidden until prediction commitment
evaluation bytes held by independent custodian
semantic labels blinded during metric extraction
model/tool cutoff frozen before historical replay
```

The local contract cannot prove that blinding/custody actually occurred.

Separate evidence must establish chronology and access state.

---

## 17. Prior exposure and contamination

Prospective status depends on what information was available to the decision process, not merely on when a JSON file was created.

Potential contamination sources include:

```text
analyst previously saw outcomes
model trained on evaluation examples
hidden benchmark reused after prior reveal
learned grammar contains target information
retrieval/model/tool contains post-cutoff knowledge
manual tuning used confirmatory failures
```

SCI-004 does not fully solve contamination analysis, but the contract must retain enough model/tool/data identity to let later dependency/custody layers evaluate it.

A contract created after meaningful outcome exposure cannot regain naive prospective authority merely by receiving a later timestamp.

---

## 18. Amendments and versioning

Scientific plans legitimately change. SCI-004 must represent amendments without rewriting history.

### 18.1 Before reveal

A pre-reveal amendment creates a new immutable contract generation linked to its predecessor.

Its prospective status depends on a new chronology receipt proving the amended contract existed before the relevant reveal boundary.

### 18.2 After reveal

A decision-relevant post-reveal amendment cannot silently retain the predecessor's confirmatory authority.

Possible conservative outcomes:

```text
ExploratoryReanalysis
ConfirmatoryScopeReduced
NewIndependentDataRequired
ProspectiveStatusNotEstablished
```

The exact adjudication belongs to a later policy layer.

### 18.3 No in-place mutation

```text
amendment != mutate historical contract bytes
```

All generations remain auditable.

---

## 19. Preregistration chronology

### 19.1 Frozen object is not chronology proof

A content-addressed contract establishes what the contract is, not when it was frozen relative to outcomes.

```text
immutable contract identity
    != preregistered-before-outcome
```

### 19.2 Preregistration receipt

A future `PreregistrationReceiptV1` should bind:

```text
exact contract identity
registration event/artifact identity
trusted chronology/custody source
relevant reveal boundary identity
registered-at / available-at semantics
verifier/qualification identity
```

The receipt must distinguish declared time from independently verified temporal ordering.

### 19.3 External timestamp is not automatically sufficient

A timestamp may prove that some bytes existed by a time without proving:

- authors had not already seen outcomes;
- the contract is semantically complete;
- the relevant data were inaccessible;
- the timestamp source is authentic/current.

Those remain separate evidence dimensions.

---

## 20. Execution binding

The contract must specify which SCI-003 execution profiles/capsules are admissible for its confirmatory path.

A later execution receipt must bind back to the exact contract generation.

```text
execution satisfying neighboring config
    != execution of this contract
```

Where a contract allows a family of admissible configurations, the admissibility rule itself must be frozen and the exact selected configuration retained in execution lineage.

---

## 21. Reporting policy

To reduce outcome-dependent publication/reporting selection, the contract may freeze a reporting policy such as:

```text
report all primary outcomes
retain null/negative/indeterminate outcomes
retain failed/incomplete executions
report preregistered exclusions
report protocol deviations
```

The shared kernel should not require public publication, but it should make selective disappearance of unfavorable outcomes detectable within a campaign lineage.

---

## 22. Protocol deviations

Execution can diverge from the contract.

A deviation should be a first-class artifact:

```text
ProtocolDeviationReceiptV1 {
    contract_identity
    affected execution/case
    deviation kind
    observed facts
    timing relative to reveal
    corrective action
}
```

A deviation does not automatically invalidate all evidence; a later adjudicator decides its effect.

But deviations must not be silently repaired by editing the historical contract.

---

## 23. Replication relationship

A replication attempt may reuse the exact proposition/measurement/analysis contract intentionally.

That reuse can improve comparability while simultaneously creating shared dependencies.

Therefore:

```text
same protocol
    != independent replication
```

SCI-006 dependency/replication analysis must later determine which dependencies are shared or disjoint.

---

## 24. Safety and action authority

Safety criteria may be preregistered, as Physical Agency demonstrates, but SCI-004 does not turn scientific safety evidence into physical execution authority.

The generic boundary remains:

```text
preregistered safety-obligation contract
    != discharged safety evidence
    != safety authorization
    != effect capability
```

Domain safety/action systems remain outside the Scientific Method Kernel authority layer.

---

## 25. Proposed shared vocabulary

Names are illustrative; semantics are normative.

```text
StudyClassV1
ExperimentContractProfileV1
ExperimentContractV1
OutcomeObservationPolicyV1
AnalysisPlanV1
DecisionCriteriaV1
MultiplicityPolicyV1
StoppingRuleV1
MissingDataPolicyV1
ExclusionPolicyV1
RevealPolicyV1
AdaptationPolicyV1
ExperimentContractAmendmentV1
PreregistrationReceiptV1
ProtocolDeviationReceiptV1
```

Do not require every domain to populate one enormous monolithic struct.

A profile may compose smaller typed subcontracts while preserving one canonical overall contract identity.

---

## 26. First implementation tranche

The first Rust tranche should be **non-executing and non-chronology-authorizing**.

Suggested slice:

```text
StudyClassV1
ExperimentContractV1
ExperimentContractIdentityV1
```

with:

1. SCI-002 canonical content identity;
2. exact target identity;
3. exact required sub-policy identities;
4. exact SCI-003 execution-profile references;
5. immutable/private-field issued frozen contract;
6. no mutable result fields;
7. no `preregistered: bool` authority;
8. no experiment execution;
9. no scientific outcome adjudication.

A frozen contract may truthfully say only:

```text
contract semantics frozen
```

not:

```text
preregistration chronology verified
```

---

## 27. Second implementation tranche

Pilot `PreregistrationReceiptV1` with one narrow existing domain where chronology semantics are already explicit.

Good candidates:

- a Futures outcome-observation policy;
- a Physical Agency outcome-claim contract;
- a nuclear declared-bounds contract with external registration evidence.

The pilot must preserve the stronger domain-native theorem and keep local contract validity separate from chronology verification.

---

## 28. Third implementation tranche

Add one confirmatory execution/adjudication join:

```text
exact preregistration receipt
+ exact SCI-003 execution receipt
+ exact observation/measurement evidence
+ exact analysis implementation
    -> scoped adjudication input
```

Do not implement a universal `ExperimentPassed` boolean.

The output should preserve per-criterion outcomes, missing/indeterminate states, and protocol deviations.

---

## 29. Adversarial requirements

Future implementation should include at least:

### Prospective boundary

- contract created after outcome reveal cannot mint prospective receipt;
- post-reveal target substitution fails;
- post-reveal metric substitution fails;
- post-reveal threshold substitution fails;
- post-reveal source/vintage/release substitution fails;
- post-reveal analysis implementation substitution fails;
- post-reveal missing/exclusion policy substitution fails.

### Data leakage

- calibration/evaluation membership substitution fails where exact membership is frozen;
- hidden evaluation case used in fitting is detectable as role/dependency violation;
- previously revealed benchmark cannot be treated as fresh hidden confirmatory data without explicit new lineage.

### Adaptive designs

- precommitted adaptive policy may select different future experiments based on observed history;
- arbitrary manual selection outside the frozen policy cannot inherit confirmatory status;
- changing the adaptation algorithm creates a new contract generation.

### Missing/exclusions/stopping

- inconvenient missing outcome cannot be silently dropped when policy says retain unresolved;
- post-hoc exclusion reason not present in the contract cannot count as preregistered exclusion;
- stopping after a favorable intermediate result outside the frozen stopping rule is a protocol deviation.

### Persistence/authority

- serialized frozen contract cannot deserialize into verified preregistration chronology;
- caller-supplied `preregistered=true`/`confirmatory=true` cannot mint chronology authority;
- contract cannot directly become scientific truth, replication, safety, or action authority.

---

## 30. Relationship to SCI-002

Every immutable contract/subcontract should have canonical SCI-002 identity.

Human-readable protocol names are navigation only.

Historical contract generations remain immutable and content-addressed.

---

## 31. Relationship to SCI-003

SCI-004 binds admissible execution profiles/capsules but does not itself establish that execution occurred.

SCI-003 attempt/verified receipts must retain exact contract identity in downstream confirmatory lineages.

---

## 32. Relationship to SCI-005

SCI-005 should enforce exploratory/confirmatory separation at the type/capability level.

SCI-004 defines the prospective contract semantics that SCI-005 will use to prevent convenience conversion:

```text
ExploratoryResult
    -/-> ConfirmatoryEvidence
```

without a new valid preregistered contract and genuinely prospective evidence.

---

## 33. Relationship to SCI-009

SCI-009 adaptive experiment planning should be compatible with SCI-004 by freezing the planner/selection policy and action space prospectively.

This avoids a false choice between:

```text
rigorous preregistration
```

and

```text
intelligent adaptive science
```

The system may adapt aggressively as long as the adaptation semantics and evidentiary interpretation are prospectively constrained.

---

## 34. Relationship to historical discovery benchmarks

Historical replay needs additional cutoff/exposure semantics beyond SCI-004.

A model or grammar containing post-cutoff knowledge can contaminate a historically frozen visible corpus.

SCI-004 should retain exact model/tool/input identities so SCI-015 can later enforce historical eligibility.

---

## 35. Dependency order

```text
SCI-001 audit
    -> SCI-002 artifact identity
    -> SCI-003 execution capsule
    -> SCI-004 experiment contract
    -> SCI-005 exploratory/confirmatory capability separation
    -> later measurement/adjudication/dependency layers
```

The contract architecture may be reviewed before all domain-native prerequisite PRs qualify, but no implementation inherits their evidence status.

---

## 36. Review boundary

Review SCI-004 on this question:

> Does this contract freeze enough decision-relevant scientific semantics before outcome exposure to support honest confirmatory/adaptive experiments, while keeping chronology proof, execution, measurement validity, adjudication, replication, safety, and action authority separate?

A positive architecture review does not establish that any existing experiment was preregistered or scientifically valid.
