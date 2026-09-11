# Continuity Proven No-External-Effects V1 — Frozen Subject

## Stack base

This tranche is stacked directly on draft PR #1497 exact head:

`77293a06395215351c47e493ec543bb4bd6a7892`

## Exact code subject

The code subject frozen before this evidence-only commit is:

`35d60627c9c0ff33a51749f99cf12bdd744f8487`

## Theorem

A physical transition may use the no-effects execution path only when all of the following refer to the same exact transition and backend world:

1. exact trusted commit eligibility;
2. exact active known-good A checkpoint/selection;
3. owner-authorized `NoExternalEffectsDeclarationV1`;
4. exact backend ID, implementation digest and generation;
5. independent verifier qualification that the reachable external-effect set is the canonical empty set under the exact coverage profile;
6. coverage analysis evaluated at the exact trusted commit time;
7. one-use A -> B execution intent durably reconstructed as spent;
8. exact rollback-resistant post-intent journal anchor;
9. same-root protected no-effects execution commitment over the exact envelope and anchor.

Only then may `ReadyProvenNoEffectsExecutionV1` reach a physical backend.

## Public execution boundary

The generic execution coordinator is private at the crate boundary and `prepare_known_good_execution()` is no longer publicly re-exported.

The intended physical entry points are now:

- effectful: `prepare_coverage_qualified_effect_execution(...)`;
- proven effect-free: `no_effects_execution::prepare_proven_no_effects_execution(...)`.

The no-effects path does not interpret an empty obligation vector as evidence. It requires a distinct owner constraint plus independent verifier proof of the domain-separated canonical empty effect set.

## Exact-time rule

`QualifiedNoExternalEffectsCoverageV1::analyzed_at_unix_ms()` must equal the exact trusted commit timestamp. The protected commitment is anchored at that same timestamp by the generic known-good coordinator.

V1 intentionally provides no grace interval for a previously established empty-effect result.

## Protected world

`NoEffectsExecutionCommitmentClaimV1` binds under the exact same journal-anchor profile/root:

- exact qualified post-intent journal anchor;
- exact trusted commit epoch;
- exact no-effects execution envelope;
- exact known-good execution intent and attempt;
- exact subject;
- exact no-effects declaration;
- exact no-effects owner authorization;
- exact no-effects coverage proof/challenge;
- exact coverage-manifest digest;
- exact backend ID, implementation digest and generation.

## Fail-closed cases

The path denies execution on:

- `EffectsFound` or `Unknown` coverage outcomes;
- non-canonical empty-set evidence;
- stale/different-time no-effects analysis;
- backend substitution;
- owner-authorization substitution;
- coverage-profile/verifier substitution as reflected in the qualified coverage identity;
- A/B/subject/context drift;
- wrong journal anchor or trusted epoch;
- unspent/missing durable intent;
- result already present before mutation release;
- commitment/envelope mismatch.

## Non-claims

This frozen subject does **not** establish:

- CI/compiler/test qualification;
- that a coverage profile is philosophically omniscient outside its exact declared external boundary/model/taxonomy;
- physical execution success;
- post-transition identity or health;
- LKG promotion;
- bootstrap admission;
- restart rebind for the serialized no-effects envelope.

All physical mutation remains disabled by policy until exact-head executable evidence and all stacked parents qualify.