# Promotion Webhook Local Temporal Sequence v1

This tranche adds a separate local monotonic ordering primitive to the bounded temporal effect model.

## Why wall-clock time is insufficient

Local wall-clock timestamps can move backward or jump forward.

A monotonic sequence can establish ordering without pretending to provide absolute elapsed time.

## Sequence contract

LocalTemporalSequenceV1 records:

    reservation sequence
    dispatch sequence
    observation sequence

Required order:

    reservation_seq <= dispatch_seq <= observation_seq

Values must be positive.

## Interaction with wall-clock timing

Temporal admissibility requires both:

    usable provider/local clock relation with explicit skew bound
    +
    valid local monotonic sequence

The sequence establishes local ordering.

The clock relation bounds cross-domain wall-clock comparison.

Neither establishes provider truthfulness or operation causality.

## Fail-closed cases

Without local sequence evidence:

    local-monotonic-sequence-missing

For invalid ordering or non-positive sequence values:

    invalid-local-monotonic-sequence

In both cases the timing result is not temporally admissible.

## Claim ceiling

This establishes only explicit local monotonic ordering as a prerequisite to the bounded temporal effect claim.

It does not establish monotonic wall-clock time, synchronized clocks, provider truthfulness, causal attribution, provider-side topology CAS, governance legitimacy, production atomicity, or promotion success.

Related: #7169, #7171, #7176, #7177.