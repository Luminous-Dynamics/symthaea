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


## Exact effect-boundary binding

For actions that cross an external effect sink, the immutable executable contract may carry an `ActionEffectBinding` containing:

- executor-observed target identity;
- intended audience/environment;
- exact adapter/finality-sink identity.

The effect binding is included in the canonical action digest. Consequently, changing the target, audience, or adapter changes the authorized action identity rather than silently rebinding an already-authorized action at execution time.

The effectful execution API requires the expected effect binding to match the action before admission. The authorization witness still binds to the resulting canonical action digest, so the binding is enforced transitively through the existing authorization boundary.

This is deliberately stronger than checking effect fields earlier in a workflow and trusting a later reconstruction. Current reconstruction-aware agent-security research identifies that pattern as a residual authorization risk: the representation inspected for approval can differ from the object ultimately consumed by the sink. Current IETF work similarly calls for a frozen observed action, exact target identity, durable pre-dispatch state, and effect-boundary verification. These documents are Internet-Drafts and research, not final standards.

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


## Durable dispatch boundary and crash recovery

The durable execution lifecycle now makes the external-effect boundary explicit:

    AUTHORIZATION
      ↓
    PREPARED
      ↓ durable commit
    DISPATCH_PENDING
      ↓ provider entry
    INVOKED
      ↓ authoritative outcome
    EXECUTED | FAILED
      └────────→ INDETERMINATE → authenticated reconciliation

`Prepared` is a reservation, not permission to enter an external effect sink. An executor must durably transition the same authorization instance and attempt to `DispatchPending` before provider entry. The transition is serialized in the shared consumption domain and is fenced by the exact authorization instance plus attempt identity.

The execution identities remain deliberately non-interchangeable:

- **Authorization instance** identifies the durable grant of authority.
- **Attempt ID** identifies one executor attempt and may change across recovery/replacement.
- **Provider idempotency key** identifies the downstream replay-control value for the same native grant and effect target. It is derived deterministically from the native replay identity + target identity + canonical action digest, under a versioned domain separator, and intentionally excludes the operation ID and attempt ID.

Consequently, a crash or executor replacement must not manufacture a new provider idempotency key merely because it has a new attempt ID. A fresh native grant receives a distinct replay identity and therefore a distinct provider key. The provider key is a downstream replay-control identity, not a substitute for authorization and not evidence that the effect occurred.

### Frozen durable dispatch record

For effectful execution, the pre-dispatch transition is now backed by a first-class immutable dispatch record. It freezes, in one durable row, the authorization instance, attempt ID, action ID/digest, provider idempotency key, target identity, audience/environment, adapter/finality sink, and boundary identity. `mark_invoked_bound` re-reads and verifies those exact fields before allowing provider-entry evidence to advance the lease.

The boundary identity scopes attempt ownership but is deliberately excluded from the shared action identity. This prevents two boundary instances sharing one durable store from claiming each other's attempt records while preserving one common same-action fence. A stale executor that presents a different target, audience, adapter, action digest, provider key, or boundary identity is rejected rather than being allowed to reinterpret the authorization after dispatch preparation.

The legacy attempt-only transition remains available for non-effectful compatibility paths; external effect adapters should use the bound dispatch-record path. An action carrying an `ActionEffectBinding` is explicitly rejected by `prepare_for_execution`; effectful actions must therefore enter through `prepare_for_execution_bound` and the canonical pinned-native-authorization dispatch path. The durable record is not itself authorization and `Invoked` is not success evidence: it is frozen provider-entry evidence tied to the exact pre-dispatch contract.

`DispatchPending` is therefore an evidence boundary, not an execution receipt. `Invoked` is the subsequent durable provider-entry marker; it is likewise not an execution receipt. It proves that the local executor durably recorded its intent immediately before the effect boundary; it does not prove that the provider accepted the effect.

On restart, `DispatchPending` and `Invoked` non-terminal reservations are conservatively recovered to `Indeterminate`. A `Prepared` reservation is a distinct pre-entry case: when an authorized recovery operation proves that no DispatchPending record exists, `recover_pre_dispatch_attempt` atomically records a not-entered marker and returns the reservation to `Ready` without consuming budget. `Invoked` records durable evidence that provider entry began; it still does not establish that the protected effect succeeded. An attempt that has crossed the dispatch fence remains occupied and cannot be retried until explicit reconciliation establishes a terminal outcome. Reconciliation consumes the existing authorization budget; it does not create a new authorization.

The store therefore does not claim atomicity between SQLite and an external provider. The safety property is narrower and auditable: authority is durably reserved before dispatch, dispatch intent is durably recorded before provider entry, ambiguous outcomes are preserved rather than guessed, and replay is blocked until authenticated reconciliation. This matches current distributed-systems analysis that a crashed executor cannot infer sink acceptance from its own database alone, and current agent-effect boundary work that requires a durable pre-dispatch state and an explicit indeterminate path. The cited IETF material is an Internet-Draft, not a final standard.

### Boundary-scoped attempt ownership and recovery

Attempt ownership is now durable before the effect boundary rather than being inferred only from a dispatch row. The prepare_for_execution_bound API persists the execution boundary alongside the Prepared reservation, so a crash before DispatchPending still leaves an attributable owner. Bound attempt identifiers are unique within the shared durable state domain, preventing the same attempt identifier from being reused by another boundary or authorization instance.

For an effectful attempt, the preferred lifecycle is:

    prepare_for_execution_bound
      ↓
    mark_dispatch_pending_bound_from_pinned_native_authority
      ↓
    mark_invoked_bound
      ↓
    commit_bound
      └────────→ Indeterminate → reconcile_indeterminate_bound

recover_incomplete_attempt_for_boundary is the preferred exact recovery primitive. It requires both the boundary identity and one exact attempt identifier, so recovery cannot implicitly claim a sibling attempt. A batch recover_incomplete_attempts_for_boundary helper also exists for executor startup maintenance; it filters to one boundary but should not substitute for an exact recovery authorization. Both forms validate that any DispatchPending/Invoked attempt has a matching frozen dispatch record before converting the attempt to Indeterminate. The legacy recover_incomplete_attempts path is intentionally limited to unscoped legacy attempts; it does not sweep boundary-owned attempts.

The legacy attempt-only transition APIs are also fenced from boundary-owned attempts. Once an attempt has a durable boundary owner, mark_dispatch_pending, mark_invoked, commit, and reconcile_indeterminate cannot bypass the bound path. This prevents an older or less-specific caller from claiming an attempt merely because it knows the authorization instance and attempt identifier.

Boundary identity remains a recovery/ownership field, not an action-identity field. It is persisted on attempt-bearing durable records so recovery operations can be scoped to one execution boundary, while the canonical action digest and downstream provider idempotency identity remain common across boundaries for the same authorized action. This follows the current Action Evidence Boundary Internet-Draft's distinction between shared action identity and boundary-scoped attempt ownership, including its requirement that recovery of one attempt cannot close or release records belonging to another.


### Provider-verifier boundary

Terminal provider outcomes are now a separate authority boundary from the durable dispatch record.

A DispatchPending or Invoked record proves only local lifecycle facts. A raw Succeeded/Failed enum returned by an adapter is not terminal evidence by itself. Effectful terminal commitment and reconciliation therefore require a relying-party-configured ProviderEvidenceVerifier.

The verifier is passed:

- the exact frozen DurableDispatchRecord;
- evidence explicitly classified as TerminalOutcome;
- the exact action ID/digest;
- exact attempt ID;
- provider idempotency key;
- target identity;
- audience/environment;
- adapter/finality sink;
- boundary identity.

The verifier configuration is itself relying-party-pinned and auditable: verifier implementation/profile identifier and digest, trust-anchor-set digest, and accepted evidence-profile/schema digest are persisted with the terminal evidence. The verifier is invoked with an explicit `TerminalOutcome` purpose; a verifier result that does not affirm that purpose is not terminal authority. This makes `verified` an inspectable statement about a configured verification procedure, rather than an opaque Boolean. The verifier must return an explicit affirmative VerifiedProviderOutcome, including verifier identity and a verification digest. The durable store re-checks the returned binding before committing the terminal state and persists the evidence/verifier digests alongside the terminal receipt.

A PreEntryLookup is a different evidence kind and is rejected by terminal commitment/reconciliation. It cannot be converted into Failed for an attempt that reached DispatchPending.

The legacy outcome-only commit_bound and reconcile_indeterminate_bound paths are fenced for terminal outcomes. They may no longer turn a locally observed enum into provider truth. The intended effectful path is:

    provider evidence
      ↓
    ProviderEvidenceVerifier
      ↓ explicit terminal affirmation
    exact frozen dispatch record
      ↓
    EXECUTED | FAILED

This remains an adapter boundary, not a universal proof of physical truth. The verifier's trust anchors, provider authentication, freshness rules, cancellation semantics, and semantic interpretation remain relying-party configuration. The store records which verifier/revision produced the accepted terminal attestation rather than pretending SQLite itself authenticated the provider.

### Execution identity lattice

The effect boundary now records the execution identities that must not be conflated:

- **Authorization instance** identifies the durable grant of authority.
- **Operation ID** identifies the logical operation being attempted.
- **Native replay identity** identifies the one native grant of authority and is supplied by the native authorization path.
- **Attempt ID** identifies one executor attempt.
- **Action digest** identifies the frozen material action.
- **Provider idempotency key** identifies the downstream replay-control value.

The bound dispatch API requires the operation ID and native replay identity explicitly and rejects either when absent. Neither is derived from the attempt ID, provider idempotency key, wrapper, boundary label, or other local retry metadata. The durable dispatch row and terminal provider evidence carry both values, and the verifier must affirm both against the exact frozen record.

The store also rejects reuse of a non-empty operation ID or native replay identity within its durable dispatch domain. This makes identity collisions visible at the persistence boundary rather than allowing a later attempt to reinterpret an existing operation or native grant.

For an effectful attempt, the store also enforces a transactional same-action in-flight fence over the exact effecting target identity and canonical action digest. A second authorization with a fresh native replay identity cannot enter DispatchPending while another attempt for the same material target/action remains DispatchPending, Invoked, or Indeterminate. The check is performed inside the same immediate SQLite write transaction as dispatch-row creation, so it is not a best-effort preflight race. Terminal Succeeded or Failed states release this in-flight fence; Indeterminate remains occupied until authenticated reconciliation.

This layer deliberately does **not** claim that the native replay identity has been derived correctly merely because a caller supplied a string. The native authorization adapter remains responsible for authenticating the native authority and deriving its replay identity from the pinned authority namespace and native authorization identifier. The store's role is to require, freeze, bind, and durably fence that value once it crosses the execution boundary.

The effect boundary now exposes a canonical native-authorization entry point that derives the native replay identity from the pinned authority namespace and native authorization identifier before constructing the durable dispatch record. That derivation witness—authority namespace, native authorization identifier, and derivation digest—is persisted with the dispatch record and propagated into terminal provider evidence in the same durable lifecycle. The former free-form replay-identity entry points remain only as deprecated hard-fail shims: they cannot create an effectful dispatch record without durable native derivation provenance.

The durable SQLite authorization store is also pinned to one relying-party control domain. Production callers can open it with an explicit relying-party identifier; the compatibility open path uses the legacy-local domain. The same-action in-flight fence is therefore evaluated as relying party + effecting target identity + action digest, while the relying-party binding itself remains outside the canonical action digest. A store cannot later be reopened under a different relying party, preventing accidental cross-domain fence reuse. The native authority pin set is likewise durable and write-once: an accepted issuer is resolved to one authority namespace inside the effect boundary before native replay identity derivation. Unpinned issuers and attempts to change an established mapping fail closed. The same-action fence is independent of native replay identity: fresh native authority cannot bypass an occupied action key. An EXECUTED/SUCCEEDED result keeps that action key closed; a FAILED result releases it for a later attempt only when fresh native authority and a new operation lifecycle are provided.

Provider idempotency derivation is now native-replay based. The effectful path derives the provider key from the native replay identity plus the exact effecting target and canonical action digest under a versioned domain separator. The operation identifier, authorization presentation identifier, attempt identifier, wrapper/handoff digest, session, task, trace, challenge, and provider-supplied identifier are excluded. This means executor retries and operation-label changes cannot manufacture a fresh downstream effect identity. The former authorization-instance derivation API is deprecated and hard-fenced; it is not a usable effectful compatibility path.

### Pre-entry recovery versus Indeterminate reconciliation

The recovery boundary is intentionally split in two. A Prepared attempt that has not crossed DispatchPending can be recovered as a pre-entry stop only through the exact recovery witness and an atomic transition that records the not-entered marker. That transition prevents the stranded executor from dispatching the old attempt after recovery. A DispatchPending or Invoked attempt cannot use this release path; it remains occupied and follows the Indeterminate reconciliation path.

`RecoveryAuthorizationWitness` binds a recovery operation to the authorization instance, exact attempt, exact boundary, action digest, authority epoch, and issuance metadata. The store enforces those bindings; the authority/authentication layer remains responsible for authenticating the issuer and policy. This keeps recovery authorization distinct from the execution receipt and prevents a recovery operation for one attempt from being applied to another.

