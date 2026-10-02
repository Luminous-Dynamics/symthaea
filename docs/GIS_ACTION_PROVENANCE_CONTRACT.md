# GIS Action Provenance Contract

## Purpose

Extend the epistemic provenance chain from:

Evidence → Frame → Conclusion

to:

Evidence → Frame → Conclusion → Action.

A frame revision is not a declaration that a conclusion was false and is not, by itself, a declaration that a previously executed action was wrong. It creates a provenance obligation: future actions whose epistemic prerequisites are affected must be re-evaluated according to their risk and authorization policy.

## Contract

For an action A:

- A declares the conclusions it depends on.
- Each dependency declares why it matters: inference, causal, ontology, evidence, or assumption.
- A declares a risk tier.
- Historical authorization records the frame, conclusions, evidence, policy, and decision that existed when the action was authorized.
- A later frame revision may create a re-evaluation witness without rewriting that historical record.

The core invariant is:

> Historical authorization is evidence of what authorized an action then; it is not automatically current authorization for executing the action again now.

### Current support is not the conclusion lifecycle

A conclusion's lifecycle state is not a generic trust score. In particular, `Active` means the conclusion has not been reopened or superseded; it does not establish that its support is fresh, conflict-free, or provenance-complete.

Current High/Critical authorization therefore evaluates these dimensions separately: frame binding, explicit dependency coverage, lifecycle state, freshness, conflict state, and provenance completeness. `CurrentConclusionSupport` encodes those orthogonal support dimensions and its authorization predicate requires `Active`, fresh, non-conflicted, provenance-complete support. A current-support assessment must fail closed when any required dimension is unresolved. This preserves the distinction between historical epistemic state and current authorization while avoiding the false equivalence `Active == authoritative`.

## Revision propagation

The deterministic path is:

    Frame F1
      ↓ revision
    Frame F2
      ↓ impact mask
    affected conclusions
      ↓ typed action dependencies
    affected actions
      ↓ risk policy
    Ready | RequiresReevaluation | Deferred

Typed dependency reasons prevent graph proximity from being treated as semantic dependence.

An ontology-only revision should affect ontology-dependent actions without automatically gating an unrelated causal action. A causal-model revision should affect causal interventions. An evidence-boundary revision should affect actions whose support depends on that boundary.

The implementation now exposes a direct bridge that first asks the conclusion graph to reopen affected claims, then applies typed action dependencies and risk policy. Typed conclusion edges propagate only when their dependency kind is affected by the revision. Legacy untyped conclusion dependencies remain conservative because their semantic basis is unavailable. The action record exposes fail-closed current-authorization checks for High/Critical actions. Presence of an identifier is not treated as epistemic support: the execution boundary can require every declared prerequisite to be `Active` in the authoritative conclusion store. Reopened, superseded, qualified, or missing conclusions therefore cannot silently authorize a new high-risk decision. Historical `record_decision` remains available for recording already-authorized historical facts; current execution should use the current-authorization path.

## Historical execution

Executed actions are immutable historical facts. A later revision may establish that a future repetition requires fresh authorization, but it must not rewrite:

- the frame that existed at execution time;
- the evidence available at execution time;
- the conclusion state used at execution time;
- the authorization policy;
- the recorded decision.

This keeps correction compatible with auditability.

## Risk policy

Informational and low-risk actions may remain executable under policy when the changed provenance does not require current support.

High-risk and critical actions should require current epistemic support when an affected prerequisite changes. Missing or stale provenance should fail closed for those actions.

This is intentionally a policy boundary rather than a confidence threshold.

## Re-evaluation witness

A gated action should retain an explicit witness containing:

- prior frame;
- revised frame;
- affected conclusion IDs;
- typed dependency reasons;
- the policy reason for re-evaluation.

This makes the gate explainable without trusting a free-form model rationale as authorization.

## Adversarial requirements

The current module includes focused tests for high-risk gating, ontology-only typed propagation through the conclusion/action bridge, and missing-prerequisite deferral. These tests are source-level additions; they have not yet been run in a repository build. The full conformance suite should deterministically test:

1. ontology-only revision gates ontology-dependent high-risk actions;
2. causal revision gates causal interventions;
3. evidence-boundary revision gates evidence-dependent actions;
4. current high-risk execution rejects missing, reopened, qualified, superseded, stale, conflicted, provenance-incomplete, cross-frame, or incompletely witnessed prerequisites;
5. unrelated actions remain unaffected;
6. superseded conclusions never resurrect old action authorization;
7. missing/stale provenance fails closed for high-risk actions;
8. executed history remains immutable;
9. cycles terminate deterministically;
10. re-evaluation is distinct from falsification;
11. historical authorization is distinct from current authorization.

## Research alignment

Recent work independently converges on several of these separations. Typed provenance research distinguishes stored material from supported claims and uses protected decision witnesses. Recent agent-authorization work separates per-action authorization from standing capability and mutable runtime state. Work on tool-using agents likewise separates action induction from execution authorization and warns against promoting historical context into current authority.

GIS should therefore treat action provenance as a dependency-bearing temporal contract, not another scalar confidence score.

### Authorization witness binding

Current support is necessary but not sufficient for an effectful authorization. The authorization witness must also bind the decision to the exact action identity/digest, current frame, support assessment digest, policy identity, and issuance state. A support assessment that was valid for one action must not be reusable as permission for another action merely because the underlying conclusions are unchanged.

The intended temporal boundary is therefore:

    Support Assessment
      ↓ exact-action + frame + policy binding
    Authorization Witness
      ↓ enforcement
    Execution Receipt

A current authorization witness is still not sufficient for replay-resistant execution. The execution boundary now models a separate durable authorization lease:

    Canonical Action + Support + Policy + Authority Epoch
      ↓
    Authorization Witness
      ↓ atomic prepare
    Authorization Lease
      ↓ effect / observation
    Execution Receipt

The lease carries an explicit execution budget and serialized state. Revocation and expiry are terminal states: neither can be reversed by replaying an earlier witness, and neither replenishes the execution budget. Prepared prevents two concurrent attempts from consuming the same reservation; Exhausted prevents another presentation of the same authorization instance from silently replenishing its consumed budget; and Indeterminate blocks blind retry when the effect boundary cannot establish whether the effect happened. A genuinely new authorization is represented by a new explicit authorization instance, rather than by minting a fresh presentation identifier for the old instance. Reconciliation is an explicit state transition rather than a second authorization. In a distributed deployment, these transitions require a shared durable atomic consumption domain; an in-memory state machine demonstrates the invariant but does not by itself provide cross-process replay protection.

This boundary also keeps execution evidence non-authorizing: an ExecutionReceipt records the action digest, attempt, authority epoch, and observed outcome, but does not contain the support/policy/frame fields required to authorize another execution.


This follows the broader evidence/authorization separation seen in recent agent-security work: authorization evidence should be request-specific, and execution evidence should remain a separate record of what actually happened.


## Canonical action identity and effect-boundary fencing

Authorization must bind to a digest derived from the immutable executable action contract, not to an arbitrary caller-supplied digest string. GIS now derives a domain-separated SHA-256 action digest from the action ID, description, risk tier, and canonicalized typed dependencies. Lifecycle/history fields are deliberately excluded so recording execution does not mutate the identity of the action being authorized.

Changing any executable action field changes the digest and invalidates the prior authorization witness. Reordering semantic dependencies does not change the digest. This makes action mutation an explicit authorization boundary rather than an implicit caller obligation.

Revocation and expiry are also fenced at the effect boundary. A lease in Ready may be revoked or expired; a lease already Prepared cannot be silently converted to a terminal cancellation state. If an attempt has crossed far enough that its effect is uncertain, the lease must enter Indeterminate and be reconciled. This prevents a control-plane revocation from falsely asserting that no effect could have occurred.

These rules align with current agent-authorization research: exact action hashing, a shared authorization instance/consumption key, terminal state transitions, and durable atomic consumption are being treated as distinct requirements rather than properties of a signed token alone. The relevant IETF work is still an Internet-Draft, not a final standard.


## Authorization instance and semantic replay

The authorization instance is the durable identity of one issuance of authority. It is intentionally distinct from the canonical action identity:

    Canonical Action Digest
      +
    Authorization Instance
      ↓
    Durable Authorization Lease

A fresh token, retry ID, wrapper, session, or presentation identifier MUST NOT create a new spendable authorization instance by itself. Re-presenting the same authorization instance remains subject to its existing budget and terminal state. If policy explicitly grants the same canonical action again, issuance creates a new authorization instance and persists it before execution admission. This makes semantic replay distinguishable from legitimate re-authorization.

The durable store uses the authorization instance as the lease and receipt consumption key. Existing action-keyed state is migrated by preserving the historical action ID as its initial authorization instance, so the schema hardening does not silently discard prior consumption history.

## Durable shared consumption domain

The in-memory authorization lease is now complemented by a SQLite-backed shared consumption domain. The durable store persists the lease state, remaining execution budget, attempt identity, and immutable execution observations.

Reservation and consumption transitions execute inside SQLite write transactions. The store uses WAL mode and FULL synchronous durability. Concurrent processes therefore contend on one authoritative state machine rather than independently maintaining local budgets. Reopening the database preserves terminal consumption, and repeated commits for the same authorization-instance/attempt return the recorded receipt instead of consuming authority again.

This still does not make an external side effect transactionally atomic with the authorization database. The prepare/effect/commit gap remains an explicit failure boundary: an uncertain external effect becomes Indeterminate, and reconciliation is required before another attempt can be admitted.

SQLite is appropriate as a durable local/shared-node implementation where writer concurrency is bounded; deployments requiring many concurrent writers or multiple independent servers should use an equivalent client/server transactional domain rather than treating SQLite as a universal distributed-consensus layer.
