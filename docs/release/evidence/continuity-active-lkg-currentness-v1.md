# Active LKG currentness v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-crash-recovered-lkg-v1`

Parent exact head at branch creation:

`a4b6e1528ad8519ba5bf5d389c2349214231907e`

Exact code subject frozen before this evidence-only commit:

`29a13388e43cfbe5ac4e995a715ae1aab328a7dc`

This subject closes the distinction between:

```text
"selection A was once valid"
!=
"a fresh rollback-resistant attestation names A as the current active-LKG head"
```

The currentness claim reuses the already-defined rollback-resistant platform/root profile used by journal anchoring, but has a separate claim domain, canonical wire domain, authentication purpose, monotonic anchor sequence, and qualified identity.

Every claim binds the exact active-selection id and generation, exact predecessor-selection id, checkpoint id/generation, subject, realization, selection time, a non-zero fresh caller challenge digest, boot identity/counter, monotonic counter, anchor time, and raw platform evidence digest.

## Legal progression

After the initial generation-one anchor, exactly two progression shapes are legal:

1. **same-selection refresh** — the exact same active selection/checkpoint/realization is freshly re-attested under the next anchor sequence, exact previous currentness id, a new freshness challenge, and rollback-resistant counter progression;
2. **selection advance** — selection generation advances by exactly one, the new selection names the previously attested active selection as its exact predecessor, checkpoint generation strictly advances, and selection time strictly advances.

Generation rollback, generation skipping, same-generation identity drift, wrong predecessor, profile/root substitution, subject substitution, anchor-sequence discontinuity, boot-counter rollback, same-boot identity drift, non-advancing same-boot monotonic counter, wall-clock rollback, and freshness-challenge replay fail closed.

## Freshness and replay

A valid old attestation is not intended to be timeless currentness. The caller supplies a fresh challenge digest and qualification requires that exact challenge. Production adapters must produce/authenticate the claim from the current protected platform state in response to that challenge; possession of an older signed claim must not satisfy a new challenge.

The canonical authentication payload is `canonical_active_lkg_currentness_claim_bytes()`. Serde bytes are not the trust contract.

## Scope

This proves which LKG selection a fresh rollback-resistant platform attestation names as current. It does not grant recovery execution, retry, checkpoint promotion, active-selection advancement, bootstrap admission, or external side-effect compensation.

No test or CI pass is claimed here. Exact-head CI and all stacked parent qualification remain required.
