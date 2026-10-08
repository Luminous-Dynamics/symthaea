# Promotion Webhook Effect Monotonicity v1

GitHub documents that webhook deliveries can arrive out of order. Therefore the qualification effect reducer must not depend on delivery order or use a latest-delivery-wins rule. GitHub also uses the delivery ID to identify a delivery and preserves that identity across redelivery attempts.

## Terminal effect state

The per-PR reducer is additionally bound to the exact `PromotionOperationIdentityV1` digest that created the reservation. This prevents an effect-state record from being silently reused across a new operation whose PR/head happens to remain identical while another authority-bearing dimension (such as stack topology, base tip, trust-root generation, or governance generation) changes.

For each reserved stack entry, the reducer has the monotonic states:

    Unobserved -> EffectObserved
    EffectObserved -> EffectObserved
    EffectObserved -> Conflict

The reducer never transitions a valid EffectObserved state back to Unobserved.

## Admission

An authenticated merged pull-request observation may transition Unobserved to EffectObserved.

The retained state records the compact effect and the source delivery IDs supporting that effect.

## Repetition

An equivalent observation through another authenticated delivery is a compatible repeat. It does not create a second effect.

An exact repeated delivery ID is classified as duplicate-delivery and is idempotent.

## Conflict

If a later authenticated merged observation disagrees on the expected head or merge commit, the reducer enters Conflict rather than choosing one observation.

This is fail-closed evidence handling: contradictory provider evidence does not become a guessed truth. Once `Conflict` is entered, it is absorbing: later compatible, duplicate, non-effect, or untrusted arrivals cannot reclassify the state as `EffectObserved`.

## Out-of-order non-effects

A non-effect webhook delivery is represented at this effect layer as no eligible merge effect.

After EffectObserved, such a delivery leaves the state unchanged.

This means an older opened event or closed-but-unmerged event cannot downgrade a terminal merged-effect observation.

## Untrusted and unrelated input

Untrusted observations are rejected without changing state.

Observations for unrelated PRs are ignored without changing state.

Therefore transport arrival order does not alter the terminal effect claim.

## Causality boundary

Monotonic effect admission establishes only stable observation state.

It does not establish that a particular async promotion operation caused the effect, nor does it establish provider-side topology CAS.

## Claim ceiling

This establishes deterministic monotonic effect-state semantics under arbitrary webhook delivery order in the provider-free reference model.

It does not establish provider event ordering, provider truthfulness, causal attribution, provider-side topology CAS, governance legitimacy, production atomicity, or promotion success.

Related: #7150, #7152, #7154, #7156, #7157, #7166.