# Scientific Claim / Falsifier Contract v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Parent:** SCI-006 Scientific Evidence Dependency Graph v1.

## 1. Purpose

SCI-006 establishes how scientific contributions may share ancestry, scientific dependencies, and target compatibility constraints. The next missing boundary is not yet a Theory Atlas disposition engine. It is the prospective object that makes a claim meaningfully vulnerable to evidence.

Symthaea already contains several strong local precedents:

- Economic Science requires at least one predeclared falsification criterion for every empirical prediction;
- Physical Agency freezes outcome predicates before strict simulation execution;
- Matter retains falsifiers and claim limitations across cross-scale lineage instead of silently dropping them;
- the defeater-aware scientific argument work distinguishes ordinary opposition, defeaters, falsifier outcomes, and disposition;
- SCI-004 already freezes experiment semantics before outcome exposure.

SCI-007 extracts the common falsifier mechanics while leaving proposition semantics, measurement semantics, scientific disposition, and domain-specific consequence rules where they belong.

---

## 2. Core theorem

The shared kernel must preserve:

```text
scientific proposition
    != prediction
    != falsifier specification
    != experiment contract
    != observation
    != falsifier evaluation
    != scientific disposition
    != truth
    != action authority
```

and:

```text
not falsified
    != verified true

observed disagreement
    != preregistered falsification

falsifier triggered
    != universal refutation

failed experiment
    != falsifier not triggered

inconclusive result
    != support
```

A falsifier is therefore a **prospectively specified relation between an exact scientific target, an admissible observation/evidence lineage, and an exact contradictory condition**. It is not merely an opposing sentence.

---

## 3. SCI-007 scope versus existing Theory Atlas work

SCI-007 deliberately does **not** duplicate the broader defeater-aware argument graph.

The intended separation is:

```text
SCI-007
    owns prospective falsifier specification + exact falsifier outcome lineage

#701 / later Theory Atlas argument graph
    owns how support/opposition/falsifier outcomes/defeaters coexist

later disposition policy
    owns whether a triggered falsifier yields a scoped refutation state
```

Thus:

```text
FalsifierOutcomeV1::Triggered
    !=
ScientificDisposition::RefutedWithinDeclaredFalsifierScope
```

The latter transition is policy/domain owned and must retain the exact falsifier outcome that justified it.

---

## 4. Scientific target identity

A falsifier must target an exact immutable scientific proposition identity, or an exact domain target object that can later be adapted to the immutable proposition contract.

Human-readable wording is insufficient.

The target must distinguish scientifically material differences such as, where applicable:

```text
population / domain / regime
intervention or contrast
outcome construct
causal estimand
horizon / temporal scope
mathematical relation
mechanism claim
boundary conditions
```

SCI-007 does not define those domain semantics.

Required theorem:

```text
similar wording != same falsification target
```

---

## 5. Falsifier specification

Illustrative common envelope:

```text
FalsifierSpecificationV1 {
    falsifier_id
    target_proposition_id
    scope_profile_id
    contradictory_observation_policy_id
    required_measurement_specification_ids
    required_experiment_contract_family_id?
    required_execution_profile_ids[]
    analysis / evaluator identity
    decision criterion
    uncertainty handling policy
    missingness / non-evaluable policy
    multiplicity / family relation?
    applicability conditions
    limitations
}
```

This is an envelope, not a universal statistical schema.

The domain owns the actual contradictory predicate. Examples may include:

```text
measured value exceeds declared bound
interval entirely outside predicted range
invariant residual exceeds declared tolerance
specified intervention reverses predicted direction
required phenomenon absent under exact admissible conditions
counterexample exists inside declared mathematical domain
```

The kernel must not reinterpret one domain's predicate in another domain.

---

## 6. Falsifier families and predictions

A scientific claim may imply multiple predictions, and one prediction may admit multiple falsifiers.

SCI-007 should retain this topology explicitly:

```text
Proposition P
    -> Prediction p1
        -> Falsifier f1
        -> Falsifier f2
    -> Prediction p2
        -> Falsifier f3
```

The economics rule "at least one falsifier per empirical prediction" is a strong domain precedent, but SCI-007 does not universally require every proposition class to be empirically falsifiable in the same way.

For example:

- mathematical statements may use proof/counterexample semantics;
- descriptive historical claims may use source/evidence contradiction semantics;
- normative propositions are not scientific empirical claims merely because they can be phrased as predictions.

The shared kernel should therefore carry claim/prediction/falsifier topology without imposing one philosophy-of-science rule on every statement class.

---

## 7. Prospective binding

A falsifier intended to carry confirmatory force must be bound prospectively through SCI-004 / SCI-005 semantics.

The important chain is:

```text
target proposition
    -> frozen falsifier specification
    -> verified preregistration chronology
    -> prospective eligible evidence
    -> execution / observation
    -> falsifier evaluation
```

Post-hoc discovery that an observed result contradicts a claim may still be scientifically important, but it is not retroactively a preregistered falsifier.

Required theorem:

```text
post-hoc contradiction != preregistered falsification event
```

The result may enter the argument graph as ordinary opposition, an exploratory falsification candidate, or a newly generated falsifier for a future prospective test.

---

## 8. Exact applicability conditions

A falsifier must declare the conditions under which its result is interpretable.

Illustrative conditions include:

```text
measurement validity class
instrument/calibration requirements
input/state range
population/sample eligibility
intervention fidelity
solver capability
boundary conditions
exchangeability / identification assumptions
minimum observation completeness
required uncertainty representation
```

A result outside those conditions is not automatically `NotTriggered`.

It may be:

```text
NotApplicable
NotEvaluable
Inconclusive
ProtocolDeviation
```

This prevents invalid experiments from laundering themselves into evidence that a theory survived.

---

## 9. Outcome vocabulary

The generic falsifier evaluation should preserve at least:

```text
Triggered
NotTriggered
Inconclusive
NotEvaluable
NotApplicable
ExecutionFailed
MeasurementInvalid
ProtocolDeviation
```

Domains may add more precise subtypes.

### Triggered

The exact observed/evaluated evidence satisfies the preregistered contradictory condition inside its declared applicability scope.

This still means only:

```text
falsifier condition triggered under this exact scope
```

not universal falsehood.

### NotTriggered

The experiment/evidence was evaluable under the declared falsifier and did not meet its contradictory condition.

This means only:

```text
this falsifier did not trigger
```

not that the proposition is true or even strongly supported.

### Inconclusive

The evidence was relevant but the uncertainty/result did not permit the preregistered condition to be decided.

### NotEvaluable / NotApplicable

The required measurement, intervention, domain, data completeness, or other applicability contract was not satisfied.

These states must not be silently converted into `NotTriggered`.

---

## 10. Falsifier evaluation receipt

A future result should retain the exact reasoning inputs, not only a status enum.

Illustrative envelope:

```text
FalsifierEvaluationReceiptV1 {
    falsifier_specification_identity
    target_proposition_identity
    experiment_contract_identity?
    execution_receipt_identity?
    measurement_evidence_identities[]
    exact evidence / observation identities
    evaluator implementation identity
    evaluator execution identity
    applicability assessment
    criterion inputs
    uncertainty inputs
    outcome
    deviations
    limitations
}
```

A positive `Triggered` result should be verifier-owned/private-construction in the eventual implementation path; deserializing a stored record must not recreate current evaluation authority without revalidation.

---

## 11. Counterexample semantics

For formal/mathematical claims, a valid counterexample may be particularly strong, but identity and scope still matter.

The shared kernel should preserve:

```text
candidate counterexample
    != verified counterexample
    != proof target identity
    != universal domain coverage
```

A counterexample verifier must bind:

- exact theorem/proposition target;
- exact domain assumptions;
- exact candidate value/object;
- exact verifier method/execution;
- exact result.

A counterexample to a stronger neighboring proposition must not silently refute a weaker/different target.

---

## 12. Falsifier vs defeater

Falsifiers and defeaters are different scientific objects.

A falsifier asks whether an exact prediction/claim fails under an exact test.

A defeater may attack the inference path itself, for example:

```text
measurement invalidity
sampling failure
execution mismatch
source provenance defect
causal identification failure
unmodeled confounding
calibration failure
post-hoc analysis
non-independence
```

An undercutting defeater can block the evidentiary force of a falsifier without establishing that the proposition is true.

Example:

```text
falsifier condition numerically triggered
    + instrument calibration invalid
    -> triggered-looking observation exists
    -> evidentiary force contested / undercut
    -> not automatic clean refutation
```

SCI-007 records the falsifier evaluation. The argument/disposition layer owns the defeater relation.

---

## 13. Multiple falsifiers

Multiple falsifiers do not become a vote.

Possible states include:

```text
f1 Triggered
f2 NotTriggered
f3 Inconclusive
```

The kernel should retain all three.

It must not implicitly compute:

```text
2/3 survived -> 66% confidence
```

or:

```text
one trigger automatically deletes all supporting evidence
```

A later domain policy decides how a specific falsifier family affects scientific disposition.

---

## 14. Falsifier dependency and independence

Falsification attempts themselves have SCI-006 dependencies.

Two apparent falsifications may share:

```text
same dataset
same instrument/calibration
same preprocessing code
same model/estimator
same verifier
same learned grammar
same experimental apparatus
same hidden benchmark
```

Therefore:

```text
multiple triggered falsifiers != independent falsifications
```

SCI-006 dependency topology must remain attached to falsifier evidence when the Theory Atlas later reasons about robustness/replication.

---

## 15. Falsification and uncertainty

SCI-007 does not define uncertainty semantics; SCI-008 will.

But a falsifier must bind an explicit uncertainty-handling policy rather than silently reducing observations to point values.

Examples:

```text
interval entirely outside claimed range
posterior mass below predeclared threshold
exact-by-construction mismatch
replicate distribution exceeds tolerance
```

The criterion must say what happens when uncertainty straddles the threshold.

Defaulting a threshold-straddling interval to `Triggered` or `NotTriggered` is not generic-kernel behavior.

---

## 16. Negative and null results

SCI-007 makes null/inconclusive outcomes first-class.

A failed falsification campaign remains scientifically valuable because it constrains what was tested and what remains unresolved.

The record should retain:

```text
what falsifier was attempted
what scope was tested
what evidence was produced
what failed or remained uncertain
what next discriminating experiment is implied
```

This is more useful than merely incrementing or decrementing confidence.

---

## 17. Relationship to SCI-013 falsification campaigns

SCI-007 defines the atomic unit.

SCI-013 may later define a campaign over many attacks such as:

```text
preregistered direct falsifiers
negative controls
placebo interventions
label permutations
alternative preprocessing
alternative estimators
sensitivity analyses
boundary attacks
OOD tests
seed sweeps
alternative causal graphs
simpler baseline explanations
```

A campaign should retain every atomic SCI-007 result and its SCI-006 dependencies.

The campaign result must not reduce the entire attack history to one scalar confidence increment.

---

## 18. Interaction with SCI-009 experiment planning

A future experiment planner should be able to ask:

```text
which candidate experiment most discriminates among live hypotheses?
which exact falsifier has highest expected falsification value?
which unresolved defeater can this experiment resolve?
```

But planner selection does not modify falsifier semantics after seeing the outcome.

For adaptive confirmatory science, SCI-004 must prospectively freeze the planner/action space/update/stopping policy.

---

## 19. First implementation slice

The first shared Rust tranche should remain non-evaluating and non-authorizing:

```text
FalsifierSpecificationV1
FalsifierApplicabilityProfileV1
FalsifierOutcomeClassV1
```

with exact SCI-002 identities and references to SCI-004 contract/profile identities where applicable.

No public constructor should mint a positive `Triggered` scientific capability merely from a caller-supplied status field.

The second tranche should implement one narrow evaluator pilot in a domain with strong existing semantics, for example:

- Economic Science prediction criterion;
- a Matter declared bound;
- a Conjecture Engine exact counterexample/sample-check case;
- a Physical Agency preregistered simulation outcome.

The pilot must not weaken the domain's existing theorem.

---

## 20. Dependency order

```text
SCI-001 Scientific Method Kernel audit
    -> SCI-002 artifact identity
    -> SCI-003 execution capsule
    -> SCI-004 experiment contract
    -> SCI-005 exploratory/confirmatory separation
    -> SCI-006 evidence dependency graph
    -> SCI-007 falsifier specification + outcome lineage
    -> SCI-008 uncertainty-bearing observations
    -> SCI-009 experiment design
    -> SCI-013 falsification campaign
    -> SCI-014 argument/disposition/Atlas integration
```

#701 remains the stronger design reference for defeater-aware argument/disposition semantics and should consume SCI-007 results later rather than being replaced by this layer.

#729 remains the stronger design reference for immutable proposition semantic identity.

---

## 21. Deliberate non-claims

SCI-007 does not:

- decide which scientific propositions must be falsifiable;
- define one universal contradictory predicate language;
- prove preregistration chronology;
- validate measurement semantics;
- define uncertainty models;
- establish experimental validity;
- qualify evidence independence;
- define universal refutation rules;
- implement a Theory Atlas disposition engine;
- assign truth probabilities;
- grant safety, governance, recommendation, or action authority.

---

## 22. Review boundary

Review SCI-007 only on:

> Does this contract make falsification an exact, prospective, evidence-bearing scientific object—distinguishing target, applicability, observation, evaluator, outcome, dependency, defeaters, and disposition—without turning failure-to-falsify into truth or a triggered test into universal refutation?
