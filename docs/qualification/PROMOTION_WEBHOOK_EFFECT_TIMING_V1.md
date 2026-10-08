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

## Exact effect-source binding

Temporal admissibility is not sufficient by itself to identify which merged effect the timing belongs to.

Each stack timing record is therefore bound to the exact `PromotionStackEffectEvidenceV1` identity digest containing the effect head, merge commit, delivery ID, payload digest, hook, event type, repository, and source-authentication class. `PromotionStackEffectTimingSetV1.validates_complete()` requires the timing identities to match the effect-evidence set positionally under the exact operation identity.

A temporally admissible timestamp attached to another PR's evidence, another delivery, another payload, or another merge result is rejected rather than being composable by PR number alone.

## Local timeline monotonicity

The local evidence path also requires:

    reservation <= dispatch <= observation

A local wall-clock rollback between dispatch and observation is invalid temporal evidence and cannot be rescued by a provider timestamp or skew bound.


## Timestamp occurrence semantics

A parsed provider timestamp is not automatically an exact event instant. The timing model requires explicit occurrence semantics:

- `truncated` means the reported value is the lower edge of its represented timestamp bucket; the admissible occurrence interval extends to the end of that reported precision bucket.
- `exact` is accepted only for a timestamp represented at 1 ms resolution.
- unsupported fractional precision or unknown occurrence semantics are not admissible.

Therefore a second-resolution provider timestamp is modeled as an interval rather than a point. Temporal admission requires the entire provider occurrence interval, expanded by clock uncertainty, to fit between local dispatch and observation; overlap remains `cross-domain-time-uncertain`.

## Clock-relation validity and drift

A usable cross-domain clock relation now binds:

- provider and local clock-domain identities;
- baseline maximum skew;
- relation verification state;
- local verification time;
- local expiry time;
- maximum drift in parts per million.

The effective uncertainty grows monotonically from the verification time according to the declared drift bound. A relation used after its validity window, before its verification instant, or with invalid bounds is not usable for temporal admission.

The model therefore does not treat a one-time skew measurement as an indefinitely valid synchronized-clock fact.

## Timestamp policy identity

The occurrence semantics are carried by `ProviderTimestampPolicyV1`, not selected as an unbound per-event string. The policy binds:

- provider identity;
- exact source field (`pull_request.merged_at`);
- occurrence semantics;
- maximum accepted reported resolution;
- policy generation.

The timing record retains the policy identity digest and rejects a tampered policy object or an interval that exceeds the policy's maximum accepted resolution. A different timestamp interpretation therefore creates a different policy identity rather than silently changing the meaning of the same payload.

## Stack-level temporal policy pin

`PromotionStackEffectTimingSetV1` now carries one `timestamp_policy_identity_digest` and requires every timing record in the complete set to reference that same policy identity.

This prevents a complete stack from mixing, for example, a newer exact-resolution interpretation for one PR with an older/coarser interpretation for another PR. A policy generation change therefore creates a new temporal evidence identity instead of silently coexisting inside one stack result.

## Monotonic local-time witness

Wall-clock ordering is no longer sufficient. Each local reservation, dispatch, and observation point also carries a monotonic-clock reading from the local runtime.

The reference model requires both:

    wall:      reservation <= dispatch <= observation
    monotonic: reservation <= dispatch <= observation

Missing monotonic readings and any monotonic rollback fail closed. A wall-clock sequence that appears valid while the monotonic sequence regresses is therefore not temporally admissible.

This still does not prove the correctness of the local clock source; it only prevents a local wall-clock-only timestamp sequence from being treated as the sole temporal ordering witness.

## Stack-level clock-relation pin

`PromotionStackEffectTimingSetV1` also carries one `clock_relation_identity_digest` and requires every timing record to resolve to that same relation identity.

This prevents one stack result from silently mixing different skew baselines, validity windows, or drift bounds. A clock-relation change is therefore a new temporal evidence generation rather than a local reinterpretation of only one stack member.

## Timing-record integrity

`ProviderWebhookEffectTimingV1.identity_digest()` commits the temporal inputs that can affect classification: provider event interval, provider delivery time, local wall-clock points, local monotonic points, timestamp-policy identity, and clock-relation identity.

`PromotionStackEffectTimingV1` carries that digest and the complete stack validator recomputes it. Therefore a changed interval, clock relation, local timestamp, or policy cannot be silently represented as the same timing evidence record.

## Stacked webhook topology evidence

GitHub's stacked-PR webhooks can carry stack metadata including stack number, size, position, and the stack base ref/SHA. The model preserves this as `ProviderWebhookStackMetadataV1` and can validate it against the exact operation identity.

This is deliberately an observation boundary, not provider-side CAS. Matching webhook stack metadata says the authenticated payload describes the expected topology at that delivery; it does not prove that GitHub atomically reserved or merged that topology.

## Directional clock relation

The cross-domain relation uses the explicit convention:

    offset = local_clock - provider_clock

`ClockRelationV1` now carries lower and upper offset bounds independently. Symmetric absolute skew is not assumed.

At evaluation time, drift expands the interval conservatively:

    effective_lower = lower_bound - drift
    effective_upper = upper_bound + drift

The provider event occurrence interval is translated into a local-time interval using these directional bounds before it is compared with local dispatch and observation.

The relation also binds provider/local clock identities, verification method, verification evidence digest, measurement time, and expiry. A relation is unusable when the lower bound exceeds the upper bound, required verification evidence is absent, or the evaluation time is outside the validity window.

Negative offsets are valid and do not mean an invalid relation; only inverted bounds are invalid.