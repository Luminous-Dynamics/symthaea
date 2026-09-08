# Spark Observation Validation v1 — design / qualification contract

Status: staged architecture/correctness contract only. No product implementation. No scientific or action authority.

Tracks: #912

Base audited: `2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`.

## Current defect

`ObservedOutcome` exposes raw floating-point rate/energy values directly to class matching and Bayesian update.

Non-finite values do not produce an explicit invalid-measurement state. Instead they can fail ordinary comparison predicates, become `matching_class == None`, and therefore enter the likelihood path as if a valid surprising observation matched none of the predicted classes.

That can alter posterior belief.

## Required theorem

```text
InvalidObservation
    != ValidObservationNoClassMatch
    != Counterevidence
```

A numeric payload being representable as `f64` is not sufficient for scientific evidentiary use.

## Minimal validation profile

A Spark-local v1 should at least distinguish:

```text
Valid
NonFiniteRate
NonFiniteEnergy
```

before `OutcomeClasses::matching_class()` or any Bayesian update.

Future SCI-008 profiles may add instrument/domain-specific validation such as:

- out-of-domain sensor value;
- saturation;
- censoring;
- detector failure;
- calibration invalidity;
- missing required channel;
- malformed artifact lineage.

Those richer semantics must not be retroactively claimed by this minimal non-finite check.

## Missingness is not NaN

Preserve:

```text
neutron_energy_mev = None
    -> channel absent/unavailable under current signature type

neutron_energy_mev = Some(NaN)
    -> invalid numeric payload
```

Do not canonicalize these to the same state.

## Consumption-boundary validation

Because `ObservedOutcome` has public fields and derives `Deserialize`, constructor-only validation is not representation closure.

Evidence-producing code must validate immediately before likelihood evaluation.

Conceptually:

```text
ValidatedObservedOutcomeV1::try_from(raw)
    -> Result<ValidatedObservedOutcomeV1, ObservationInvalidReasonV1>
```

and:

```text
update_validated(design, validated_observation)
```

A compatibility `update(raw)` can eventually delegate to validation after consumer audit, but invalidity must become observable rather than silently mapped to a posterior update.

## Bayesian update receipt direction

A later non-authorizing receipt can distinguish:

```text
BayesianUpdateReceiptV1 {
    observation_ref,
    prediction_model_ref,
    likelihood_profile,
    prior_belief_ref,
    posterior_belief,
    update_status,
}
```

with statuses such as:

- `Updated`
- `ObservationInvalid`
- `PredictionModelInvalid`
- `LikelihoodUnavailable`

Receipt existence does not make the underlying probabilities calibrated science.

## Required current-main negative controls

Use a design with at least two outcome classes and a prior where no-match likelihoods change relative weights.

Before repair, exercise:

1. rate = `NaN`;
2. rate = `+INF`;
3. rate = `-INF`;
4. energy = `Some(NaN)`;
5. energy = `Some(+INF)`.

Require the qualification harness to record which inputs currently reach no-match/update semantics and whether posterior values change or execution panics.

Preserve exact observed current behavior as failure evidence; do not assume every invalid value fails in the same way.

After repair require:

1. every non-finite raw observation is rejected before class matching;
2. belief remains unchanged;
3. no evidentiary no-match result is emitted;
4. invalid reason is machine-readable;
5. finite valid-but-surprising observations can still produce no-class-match semantics under the declared likelihood model;
6. `None` channel absence remains distinct from invalid numeric content;
7. deserialize/public-construction bypasses are caught at consumption.

## Relationship to adjacent work

```text
#868  missing prediction != invented likelihood
#909  missing prediction != invented observation
#912  invalid observation != counterevidence
```

Together these establish the fail-closed frontier required before a richer SCI-008/SCI-009 observation likelihood can be trusted.

## Non-claims

This contract does not define universal physical measurement ranges, calibrated uncertainty, censoring semantics, correct likelihoods, or experiment authorization.
