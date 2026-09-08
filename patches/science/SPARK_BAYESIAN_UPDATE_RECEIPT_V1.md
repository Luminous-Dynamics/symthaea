# Spark Bayesian Update Receipt v1

Status: queue-neutral design and qualification contract only.

Related: #930, #912, #927, #928, #906, SCI-002, SCI-006.

## Problem

Current `HypothesisBelief::update(design, observed)` mutates belief from value-level inputs but does not retain observation identity or an update receipt.

The same observation can therefore be applied repeatedly and gain evidentiary weight each time.

## Core non-equivalences

```text
observation value
    !=
observation identity

same evidence replay
    !=
independent replication

posterior value
    !=
posterior evidence lineage
```

## Target input reference

Conceptually:

```text
ObservationEvidenceRefV1 {
    observation_artifact_ref,
    measurement_spec_ref,
    acquisition_or_execution_ref,
    dependency_refs,
}
```

Canonical authoritative identity must wait for SCI-002 rather than use ad-hoc hashes.

## Target receipt

Conceptually:

```text
BayesianUpdateReceiptV1 {
    prior_belief_ref,
    observation_evidence_ref,
    likelihood_model_profile,
    evaluated_likelihoods,
    posterior_belief_ref,
    dependency_assessment_ref?,
    update_status,
}
```

Possible statuses:

```text
Applied
DuplicateEvidenceIdentity
ObservationInvalid
PredictionUnavailable
LikelihoodModelUnavailable
DependencyPolicyBlocked
NumericalFailure
```

## Idempotence theorem

Within one belief lineage:

```text
apply evidence E once -> posterior P1
replay exact evidence identity E -> no second likelihood multiplication
```

The duplicate attempt remains auditable but contributes no new evidence weight.

## Do not deduplicate on values

Two independent observations may have exactly equal values.

The same observation may also be transformed/re-encoded while retaining one scientific identity.

Therefore `(rate, energy)` equality is not the idempotence key.

## Dependency boundary

Exact duplicate identity is a simple known dependency case.

Two distinct observation identities are not thereby independent. Shared instrument, batch, source, calibration, preprocessing, or acquisition lineage remains SCI-006 dependency structure.

## Required negative controls

1. one observation identity changes belief once;
2. exact identity replay does not change it again;
3. a distinct observation identity with identical numeric values remains distinguishable from replay;
4. invalid observation cannot receive `Applied` (#912);
5. duplicate receipt replay cannot mutate state;
6. posterior state can identify its exact update ancestry;
7. planning snapshot binds the update-history lineage rather than only the final probability vector.

## Ordering with other Spark contracts

```text
raw observation
 -> observation validation (#912)
 -> evidence identity / dependency reference
 -> likelihood evaluation (#868/#857 direction)
 -> BayesianUpdateReceiptV1
 -> stable posterior evolution (#928)
 -> validated canonical belief (#927/#904)
 -> PlanningSnapshotV1 (#906)
```

## Authority boundary

An applied update receipt means only that one declared likelihood update was applied to one validated evidence contribution under one model profile.

It does not establish calibration, independence, scientific truth, experiment success, or action authorization.
