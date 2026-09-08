# SCI-007 Review Checklist — Scientific Claim / Falsifier Contract v1

Use this checklist to review the SCI-007 architecture contract only. It does not qualify any falsifier, experiment, or scientific claim.

## A. Target identity

- [ ] Every falsifier targets one exact scientific proposition/target identity.
- [ ] Human-readable wording is not used as scientific target identity.
- [ ] Target-family or ontology similarity does not imply evidence-transfer authority.
- [ ] The contract does not redefine domain-owned proposition semantics.
- [ ] A falsifier aimed at a neighboring proposition cannot silently refute the target proposition.

## B. Prediction / falsifier separation

- [ ] Proposition, prediction, and falsifier are distinct objects.
- [ ] Multiple predictions and multiple falsifiers can be retained without flattening topology.
- [ ] The economics one-falsifier-per-prediction rule is treated as a strong precedent, not a universal rule for every statement class.
- [ ] Normative propositions are not converted into empirical claims by attaching a falsifier-shaped object.

## C. Prospective semantics

- [ ] Confirmatory falsifier force requires SCI-004/SCI-005 prospective semantics.
- [ ] Post-hoc contradiction remains scientifically useful but is not retroactively preregistered.
- [ ] A later timestamp cannot erase prior exposure to the outcome.
- [ ] Decision-relevant falsifier amendments create new lineage rather than mutating predecessor semantics.

## D. Applicability

- [ ] Falsifier applicability conditions are explicit.
- [ ] Measurement validity requirements can be bound.
- [ ] Population/domain/regime/input-range restrictions can be bound.
- [ ] Intervention fidelity or solver capability can be bound when relevant.
- [ ] Required uncertainty representation can be bound.
- [ ] Out-of-scope evidence cannot silently become `NotTriggered`.

## E. Outcome vocabulary

- [ ] `Triggered` is distinct from scientific refutation/disposition.
- [ ] `NotTriggered` is distinct from support/truth.
- [ ] `Inconclusive` remains first-class.
- [ ] `NotEvaluable` remains first-class.
- [ ] `NotApplicable` remains first-class.
- [ ] execution failure remains distinct from falsifier failure-to-trigger.
- [ ] invalid measurement remains distinct from falsifier failure-to-trigger.
- [ ] protocol deviation remains visible.

## F. Evaluation lineage

- [ ] A future evaluation receipt binds exact falsifier identity.
- [ ] It binds exact target identity.
- [ ] It can bind exact experiment-contract identity.
- [ ] It can bind exact execution receipt/capsule lineage.
- [ ] It binds exact evidence/observation identities.
- [ ] It binds evaluator implementation identity.
- [ ] It binds evaluator execution identity.
- [ ] It retains applicability assessment and criterion inputs.
- [ ] It retains uncertainty inputs/policy.
- [ ] It retains deviations and limitations.

## G. Authority construction

- [ ] Caller-supplied `triggered: bool` or equivalent cannot be the future positive-authority root.
- [ ] Positive evaluation capability should be verifier-owned/private-construction.
- [ ] Serialized records do not automatically recreate current evaluation authority.
- [ ] The contract does not grant disposition/action authority.

## H. Falsifier vs defeater

- [ ] Falsifier outcomes remain distinct from rebutting defeaters.
- [ ] Falsifier outcomes remain distinct from undercutting defeaters.
- [ ] Falsifier outcomes remain distinct from scope defeaters.
- [ ] A measurement/provenance/execution defect can undercut a triggered-looking result without proving the proposition true.
- [ ] Defeater resolution stays in the argument/disposition layer rather than SCI-007.

## I. Dependency / replication

- [ ] Falsifier attempts carry SCI-006 dependency inventory/ancestry.
- [ ] Multiple triggered falsifiers do not imply independent falsification.
- [ ] Same instrument/calibration can remain a dependency.
- [ ] Same dataset/preprocessing/model/verifier can remain a dependency.
- [ ] Shared hidden benchmark/learned grammar can remain a dependency.
- [ ] No falsification count becomes a replication score.

## J. Multiple falsifiers

- [ ] Mixed outcomes are retained rather than majority-voted.
- [ ] One triggered falsifier does not automatically delete supporting evidence.
- [ ] Several non-triggered falsifiers do not automatically establish truth.
- [ ] No hidden confidence arithmetic is implied by falsifier counts.

## K. Uncertainty boundary

- [ ] SCI-007 binds an uncertainty-handling policy but does not define universal uncertainty semantics.
- [ ] Threshold-straddling uncertainty does not silently default to triggered/not-triggered.
- [ ] Point estimates do not automatically override richer interval/distribution evidence.
- [ ] SCI-008 remains the owner of generic uncertainty-bearing observation semantics.

## L. Counterexamples / formal claims

- [ ] Candidate counterexample is distinct from verified counterexample.
- [ ] Formal target identity is bound.
- [ ] Domain assumptions are bound.
- [ ] Counterexample object/value identity is bound.
- [ ] Verifier method/execution is bound.
- [ ] A counterexample to a stronger/different proposition cannot silently refute a neighboring target.

## M. Null / failed / negative evidence

- [ ] Null outcomes remain in the scientific record.
- [ ] Failed executions remain in the scientific record.
- [ ] Inconclusive falsifier attempts remain in the scientific record.
- [ ] Missing/non-evaluable outcomes remain in the scientific record.
- [ ] Such outcomes may inform future experiment design without being rewritten as support/opposition.

## N. Relation to existing Symthaea architecture

- [ ] #701 remains the owner/reference for defeater-aware argument/disposition semantics.
- [ ] #729 remains the owner/reference for immutable proposition semantic identity.
- [ ] #507 Economic Science falsifier semantics are not weakened.
- [ ] #777 Matter falsifier/limitation propagation is not weakened.
- [ ] SCI-004 preregistration semantics are preserved.
- [ ] SCI-005 exposure/freshness semantics are preserved.
- [ ] SCI-006 dependency semantics are preserved.

## O. First implementation slice

- [ ] First shared Rust slice is non-evaluating/non-authorizing.
- [ ] It can start with `FalsifierSpecificationV1`.
- [ ] It can start with `FalsifierApplicabilityProfileV1`.
- [ ] It can start with `FalsifierOutcomeClassV1`.
- [ ] The first evaluator pilot is narrow and domain-backed.
- [ ] No qualification transfers from the domain pilot to the generic layer.

## P. Anti-shortcuts

Reject the architecture if it introduces or implies any of the following shortcuts:

- [ ] `not_falsified = true` as truth authority.
- [ ] `triggered = true` as universal refutation authority.
- [ ] `falsifier_count` as evidence strength.
- [ ] majority vote across falsifiers.
- [ ] post-hoc contradiction relabeled as preregistered.
- [ ] experiment failure treated as theory survival.
- [ ] invalid measurement treated as theory survival.
- [ ] missing data treated as theory survival.
- [ ] independent replication inferred from multiple falsifier objects.
- [ ] scientific falsifier output becoming execution/governance/action authority.

## Q. Review question

The PR is ready for architectural review only if the answer to the following is yes:

> Does SCI-007 make a falsifier an exact prospective scientific object whose target, applicability, evidence, execution, evaluator, outcome, dependencies, and limitations remain auditable—while keeping failure-to-falsify, refutation, defeaters, scientific disposition, truth, and action authority explicitly separate?
