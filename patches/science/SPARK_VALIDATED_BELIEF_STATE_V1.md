# Spark Validated Belief State v1

Status: queue-neutral design and qualification contract only.

This artifact does not modify product Rust and does not establish qualification.

Related: #927, #904, #906.

## Problem

`HypothesisBelief::new()` validates a five-hypothesis positive finite weight set, but `HypothesisBelief` also derives `Deserialize`. Constructor validation is therefore not representation closure.

A planner/evidence boundary must not assume that every deserialized `HypothesisBelief` satisfies constructor invariants.

## Core non-equivalences

```text
Rust type inhabitant
    !=
validated scientific belief

constructor-valid
    !=
deserialization-valid

transport representation
    !=
planner-admissible state
```

## Target v1 semantic object

Conceptually:

```text
ValidatedHypothesisBeliefV1 {
    hypothesis_order_profile,
    normalization_profile,
    inference_representation_profile,
    entries_in_canonical_order,
}
```

Validation should establish at least:

1. every required hypothesis appears exactly once;
2. no unknown/duplicate semantic entry exists;
3. all numerical state required by the selected representation is finite/valid;
4. total mass / normalization semantics are valid;
5. canonical order follows `ALL_HYPOTHESES` under the compatibility profile;
6. planner tie-break semantics are separately declared (#904/#907).

## Raw transport vs validated state

Prefer an explicit boundary such as:

```text
RawHypothesisBeliefV1
    -> validate / canonicalize
ValidatedHypothesisBeliefV1
```

or a custom deserializer that can establish the same invariant.

Strong planner/evidence paths should consume the validated form, not rely on field privacy.

## Weight vs probability wire semantics

A schema must say whether the numerical entries are:

```text
positive unnormalized weights
```

or:

```text
normalized probabilities
```

Do not silently accept both under one unversioned representation.

## Required negative controls

- missing hypothesis rejected;
- duplicate hypothesis rejected;
- NaN rejected;
- +/-INF rejected;
- negative numerical state rejected;
- zero-total/invalid normalization rejected;
- valid permuted state canonicalizes consistently after #904;
- validation failure cannot reach MAP, entropy, EIG, rollout, or planning snapshot as a valid state.

## Authority boundary

Validation proves only structural/numerical admissibility of a planner state.

It does not establish that:

- priors are scientifically correct;
- probabilities are calibrated;
- likelihoods are valid;
- the planner objective is optimal;
- an experiment is authorized.

## SCI integration

Future authoritative identity belongs under SCI-002 canonical scientific artifact identity.

Do not use ordinary serde bytes or Debug output as scientific belief identity.
