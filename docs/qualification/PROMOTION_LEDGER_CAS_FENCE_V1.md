# Promotion Ledger CAS Fence v1

## Purpose

This note defines the missing publication primitive identified by #7059 and #7066.

The core distinction is:

historical qualification
!=
current promotion eligibility
!=
repository merge authority

Historical receipts stay immutable. Current qualification usability continues to be represented by QualificationClaimDispositionV1. PromotionEligibilityLeaseV1 remains a narrow decision-time projection. The new ledger is only the serialization/currentness substrate for active promotion decisions.

## Atomicity boundary

A lease reference and a trust-root reference cannot safely form an atomic pair when they are updated independently.

The forbidden construction is:

GitHub-specific refinement: GitHub's GraphQL `updateRefs` mutation can atomically update multiple refs using per-ref `beforeOid` preconditions. Therefore, a single ledger ref is not the only possible atomic substrate on GitHub. It remains the preferred v1 substrate here because it collapses the currentness theorem into one authoritative object, avoids cross-object composition rules, and gives the lab a REST-compatible primitive that does not depend on a multi-ref transaction surface.


~~~text
read trust_root_ref = A
read lease_ref = A
...
write lease_ref = success
~~~

because the trust-root ref can move to B between the reads and the write.

The authoritative state therefore lives in one domain-scoped Git ref:

~~~text
refs/qualification/promotions/<domain>/current
~~~

The referenced commit contains both:

- the current trusted-root identity/generation;
- the current governance/policy snapshot;
- current qualification dispositions consumed by promotion;
- active promotion lease projections;
- invalidation/supersession transitions;
- the previous ledger-head identity.

## Linearization point

Every trusted ledger writer follows:

1. read the current ledger head L;
2. independently construct one successor commit whose parent is exactly L;
3. re-check the transition inputs against the proposed successor;
4. advance the ledger ref without force, so only a fast-forward successor of the observed head can become active;
5. treat any ref-update conflict/non-fast-forward rejection as STALE / RECHECK_REQUIRED;
6. re-read the ref and require that the observed head equals the exact successor SHA before reporting publication success.

The ref update is the linearization point. On GitHub, an implementation using GraphQL `updateRefs` may instead use its documented multi-ref transaction semantics when there is a concrete reason to retain separate refs; the proof must then bind every participating ref with `beforeOid` and treat the whole mutation as the linearization point.

For two candidates B and S both based on A:

~~~text
A -> B      succeeds
A -> S      conflicts
~~~

S may still exist as an immutable Git object, but it is not active authority because the current ledger ref does not reach it.

## Convergence with existing release lineage

The repository already implements an append-only lineage pattern in `crates/domains/symthaea-fabrication-kernel/src/release_lineage.rs`. Its events bind:

- monotonically increasing sequence;
- previous active promotion digest;
- previous event digest;
- resulting active promotion digest;
- authority digest;
- an event digest computed over the complete transition payload.

Validation rejects sequence discontinuity, time regression, previous-event mismatch, previous-active mismatch, and event-digest mismatch. This is the desired application-level shape for the future promotion ledger transition record: the Git parent provides provider-level history, while the encoded transition should independently bind the semantic predecessor and transition payload so a semantically malformed fast-forward successor cannot be mistaken for a valid state transition.

The CAS lab intentionally does not implement that domain schema yet; its claim remains limited to provider-level ref serialization.

## Trust-root race closure

A trust-root invalidation is itself a ledger transition.

Example:

~~~text
ledger(A)
  |
  +--> evaluator-success(B)
  |
  +--> root-invalidation(I)
~~~

Exactly one successor can win from A.

If invalidation wins first:

~~~text
A -> I
old evaluator A -> S
~~~

the stale evaluator cannot advance the ledger to S.

If evaluator wins first:

~~~text
A -> B
root changes / invalidator reads B
B -> I
~~~

the invalidator produces a newer successor that explicitly makes the earlier lease stale/superseded.

## Domain scope

Do not create one monolithic global promotion register unless repository scale requires it.

A ledger domain must contain every state transition that is required to share one atomic currentness theorem.

Suitable domains can include:

- HDC qualification/promotion;
- Broca trust-anchor promotion;
- physical deployment promotion.

A promotion lease must never claim fencing from a trust root outside its ledger domain unless an actual transactional service supplies the cross-object atomicity.

## Required lease bindings

PromotionEligibilityLeaseV1 should consume, rather than replace, QualificationClaimDispositionV1.

At minimum it binds:

- repository identity;
- exact qualification subject;
- exact qualification run and run attempt;
- qualification artifact identity, digest, and lifecycle;
- exact disposition identity and disposition generation;
- verifier workflow/source identity;
- trust-root generation and exact root identity;
- independent policy/admission identity;
- governance authority snapshot;
- current base/default-branch tip where required;
- final reconciliation observation;
- decision-time validity/restriction semantics.

The ledger additionally commits its own predecessor head and transition identity.

## Status projection rule

GitHub commit status/check output is a projection/cache for human and repository integration purposes.

It is not the sole promotion authority.

A status may say success for a commit while a newer ledger successor has marked the corresponding lease stale. Promotion consumers must therefore validate the current ledger head and the exact active lease, not merely inspect the latest status context.

## Rollback rule

Rollback of the ledger ref to an older reachable commit is not a valid currentness transition.

Normal publication must be monotonic in ledger history:

~~~text
L_n -> L_(n+1)
~~~

and the successor must retain previous_ledger_head = L_n.

Emergency recovery, if ever needed, is a distinct governance/reconciliation protocol and must not silently reintroduce an old promotion decision.

## Executable provider test

The companion workflow .github/workflows/qual-promotion-ledger-cas-lab.yml is intentionally manual and default-branch-rooted.

It:

- does not checkout PR code;
- creates a temporary Git ref;
- constructs two successor commits from the same parent;
- verifies that one winner can advance the ref;
- verifies that the stale successor cannot overwrite the winner without force;
- verifies that the active ref remains on the winner;
- races two same-parent successors concurrently and requires exactly one publication winner;
- advances the winner to a newer invalidation successor;
- verifies rollback/non-fast-forward rejection;
- requires successful cleanup of every temporary ref and verifies the refs are gone;
- deletes temporary refs on failure as a best-effort containment path.

The workflow establishes provider-level behavior only. It does not make the temporary test ref itself an application authority.

## Adversarial corpus

The application-level qualification suite should additionally model:

1. old success -> trust-root change -> old evaluator completion;
2. policy-root change during evaluator execution;
3. governance snapshot change during evaluator execution;
4. two evaluators from one ledger head;
5. invalidation first vs evaluator first;
6. stale candidate commit existing but unreachable from current ledger;
7. run-attempt replacement;
8. artifact expiration/deletion before repromotion;
9. subject force-push;
10. base-branch movement;
11. malformed predecessor identity;
12. duplicated/ambiguous active lease identity;
13. ledger rollback attempt;
14. privileged alternate writer path;
15. recovery from ambiguous publication result.

## Provider race and cleanup boundary

The lab also performs a synthetic application-integrity preflight: each successor candidate carries a canonical transition record containing its claimed semantic predecessor and a digest of that transition payload. A valid candidate must have its encoded predecessor equal the actual Git commit parent. A self-consistent candidate with a deliberately wrong semantic predecessor is rejected before any ref update occurs. This exercises the #7098 boundary without claiming that the production `PromotionEligibilityLeaseV1` ledger schema has been implemented.

The provider-level lab is intentionally stronger than a sequential demonstration. Its acceptance condition includes a true concurrent same-parent race, where exactly one of two valid successors may advance the ref. The losing successor may remain a valid immutable Git object but is not active because it is unreachable from the current ledger ref.

Successful completion also requires deletion of the temporary refs followed by an explicit 404 observation. Cleanup is therefore part of the positive lab predicate rather than an unverified best-effort side effect.
Temporary ref names and synthetic candidate commit identities are scoped to both the GitHub run ID and run attempt. This prevents a rerun from sharing the same temporary namespace or deterministic candidate SHAs with an earlier attempt, and makes cleanup ownership checks materially stronger.

GitHub's REST reference API defines non-forced reference updates as fast-forward updates and documents conflict responses. GitHub's GraphQL `updateRefs` additionally provides an atomic multi-ref mutation with `beforeOid` preconditions. This lab deliberately exercises the single-ref REST-compatible primitive; it does not establish an application-level authorization theorem.

The lab also contains a deliberate force-writer negative control. It advances a separate temporary ref, force-resets that ref to its historical predecessor, and then demonstrates that a normal non-force successor can be accepted from the resurrected predecessor. This is an explicit boundary finding: `force=false` in the evaluator is not sufficient to establish monotonic currentness when an alternate privileged writer can force-reset the same ref. The production authority theorem must therefore include force-writer exclusion (including delete/recreate paths) or use another non-rollbackable currentness anchor.
This requirement also applies when the implementation chooses GitHub GraphQL `updateRefs`: its atomic transaction and `beforeOid` predicates do not themselves provide an anti-ABA history guarantee. A privileged `A -> B -> A` force/reset sequence can restore the expected OID before a later transaction checks `beforeOid = A`. Cross-ref atomicity and non-rollbackable currentness are therefore separate properties.

## Claim ceiling

A passing CAS lab establishes:

~~~text
GitHub Git-ref publication can serve as a serialization point for a
domain-scoped promotion ledger when all authoritative transitions
contend on the same ref and non-fast-forward updates are rejected.
~~~

It does not establish:

- scientific correctness;
- truth of evidence;
- governance legitimacy;
- artifact durability;
- absence of privileged alternate paths;
- merge authority;
- atomicity across multiple independent Git refs.

Related: #7059, #7045, #7066, #1424.