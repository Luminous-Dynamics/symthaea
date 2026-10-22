# Calibrated Decision Contract v1

**Status:** Frozen architecture contract
**Authority:** Architecture only; no calibration, competence, autonomy, or scientific claim

## Purpose

Symthaea already contains world-grounded prediction, Brier/ECE tracking, proper scoring rules, FEP policy selection, probabilistic belief representations, factor-graph inference, and authority gates. This contract freezes the semantic boundaries required before those mechanisms are consolidated into a calibrated fast-decision substrate.

The goal is not to create another confidence scalar. The goal is to make every probability-bearing decision auditable: what question was asked, which producer emitted the distribution, what population it is calibrated against, what transformed it, what later resolved the outcome, and what authority—if any—may consume it.

## Non-substitution invariants

```text
policy probability        != outcome probability
outcome probability       != epistemic status
probability concentration != calibration quality
calibration quality       != competence
competence                 != action authority
simulation calibration    != real-world calibration
model output               != observation
generated evidence         != empirical evidence
```

No adapter may silently substitute one category for another.

## Probability semantics

A value may be called a calibrated probability only when all of the following are explicit:

- the exact question/outcome schema;
- the producing model/head identity;
- the calibration procedure identity;
- the calibration cohort/domain;
- the prediction horizon when applicable;
- the resolver/evidence class;
- the deployment/distribution identity or declared unknown status;
- held-out calibration evidence.

HDC similarity, FEP softmax policy mass, heuristic confidence, epistemic labels, evidence authority, and ranking scores are not probabilities merely because they lie in `[0, 1]`.

## Calibration-key boundary

Calibration evidence is scoped to an explicit cohort identity conceptually containing:

```text
CalibrationKey
  decision_schema_digest
  producer_digest
  calibrator_digest
  outcome_space_digest
  domain_or_task_family
  prediction_horizon
  resolver_class
  modality
  deployment_distribution
```

A mature cohort may not grant measured calibration to another cohort solely because both are produced by Symthaea or share a numerical output range.

## Transformation rule

Any transformation that changes a probability distribution creates a new probability producer unless the transform is separately established to preserve the relevant calibration property.

Examples include:

- factor-graph reconciliation;
- ensemble weighting;
- memory or contextual adjustment;
- Bayesian/posterior updates;
- temperature or other calibration maps;
- policy/risk transformations.

Preferred ordering is therefore:

```text
raw evidence/logits
  -> structural coherence
  -> final predictive producer
  -> calibration
  -> decision policy
```

If probability mass is changed after calibration, the transformed output requires its own evaluation before it may inherit any calibration claim.

## Calibration objective

Expected Calibration Error is a diagnostic, not the learning objective. Training and model selection should use a strictly proper scoring rule appropriate to the outcome space, while diagnostics report calibration and discrimination separately.

At minimum, evaluation should retain:

- Brier or multiclass Brier score where applicable;
- log score / negative log likelihood where applicable;
- calibration/reliability error;
- resolution/discrimination;
- sample count and uncertainty interval;
- abstention/coverage behavior;
- domain/cohort identity;
- IID versus distribution-shift status.

## FEP boundary

FEP action-selection probability answers roughly:

```text
How likely is the current policy to select action a?
```

A calibrated decision model answers:

```text
Given state s and action a, how likely is outcome o within horizon h?
```

These are different distributions and must remain different types or explicitly tagged semantics.

Conceptually:

```text
calibrated outcome model -> expected consequences -> FEP/policy selection -> authority gate -> action
```

The decision model predicts consequences. FEP may choose among policies. Authority independently determines whether an action is permitted.

## Unknown and abstention

Unknown mass is first-class. Weak evidence must not be silently renormalized into artificial certainty.

A decision substrate must support explicit abstention/defer states and must not reinterpret an empty or unsupported prediction as a normalized confident distribution.

## Structural coherence

Atomic semantic judgments may be produced independently for speed, but relationships between them are enforced only when explicitly declared by schema.

Examples of explicit relations include:

- complement;
- mutually exclusive;
- exhaustive;
- implication;
- conditional dependence;
- domain-specific factor.

Unrelated questions must not be forced into one simplex merely because they share labels or subjects.

## Autonomy boundary

Calibration is necessary but never sufficient for autonomous action.

Autonomy decisions must remain separately sensitive to at least:

- matching-domain/cohort calibration evidence;
- sample sufficiency;
- distribution-shift/OOD status;
- action risk and reversibility;
- evidence/authority provenance;
- applicable policy or human-approval requirements.

High confidence never creates permission.

## Planned implementation order

1. Repair persistent MAGI calibration integrity and restart semantics.
2. Bind autonomy to matching calibration cohorts rather than global averages.
3. Remove or rename ungrounded scalar values currently labeled `confidence` where they are not probabilities.
4. Add a dependency-light typed decision schema and outcome receipts.
5. Reuse/consolidate proper-scoring and calibration analytics.
6. Add explicit calibrator identities.
7. Add structurally coherent joint-decision transforms.
8. Add HDC/CfC decision producers only after the substrate is frozen and measurable.
9. Integrate calibrated outcome prediction into FEP without conflating policy and outcome probabilities.
10. Train sequential decision learning only after supervised calibrated prediction clears frozen baselines.

## Nonclaims

This document does not establish that Symthaea is calibrated, competent, autonomous, conscious, safe, scientifically correct, or superior to any external model. It defines architecture and failure boundaries only.
