# Promotion Webhook Effect Timing v1

This tranche introduces explicit temporal evidence without pretending that provider and local clocks are the same clock.

## Separate time domains

The model retains:

    provider event occurrence time
    provider delivery time, when available
    local reservation time
    local dispatch-intent time
    local observation time
    clock relation evidence

These timestamps are not collapsed into one generic event time.

GitHub documents that webhook deliveries may arrive out of order and recommends timestamps when event ordering matters. The timing model therefore uses event occurrence time rather than webhook arrival order as its semantic input.

## Clock relation

ClockRelationV1 records:

    provider clock domain
    local clock domain
    maximum permitted skew
    explicit verification state

A clock relation is unusable unless it is explicitly verified and its skew bound is non-negative.

Therefore:

    provider timestamp present
        !=
    cross-domain temporal proof

Without a usable clock relation, the temporal state is cross-domain-time-unbounded.

## Conservative admissibility

With maximum skew K, the reference model admits an event only when both bounds are provable:

    provider_event - K >= local_dispatch
    provider_event + K <= local_observation

An event definitely before dispatch is rejected.

An event definitely after local observation is rejected.

An event whose uncertainty overlaps the dispatch or observation boundary is cross-domain-time-uncertain and is not admissible.

Local reservation time must precede or equal local dispatch time.

Provider delivery time, when supplied, must not precede provider event time.

## Historical-event protection

A historical merge that occurred before a new reservation/dispatch window remains temporally inadmissible even when its webhook is delivered later.

This prevents delayed delivery or redelivery from turning an old effect into a new operation's effect merely because the evidence arrived after reservation.

## Stack interaction

PromotionStackEffectTimingSetV1 requires one timing record per reserved stack entry and rejects the complete set when any member is temporally inadmissible.

Thus an exact effect set remains subject to an independent temporal qualification boundary.

## Missing and malformed time

The reference model distinguishes missing provider event time, missing local reservation/dispatch/observation time, malformed timestamps, invalid negative values, and invalid local/provider time order.

It does not invent a timestamp when the provider omits one.

## Causality boundary

Temporal admissibility strengthens effect observation by bounding when the observed effect could have occurred.

It still does not establish that the local async operation caused the effect.

## Claim ceiling

This establishes deterministic temporal-admissibility semantics with explicit clock-domain uncertainty in the provider-free reference model.

It does not establish synchronized clocks, provider truthfulness, causal attribution, provider-side topology CAS, governance legitimacy, production atomicity, or promotion success.

Related: #7169, #7171, #7152, #7154, #7156, #7166, #7168.