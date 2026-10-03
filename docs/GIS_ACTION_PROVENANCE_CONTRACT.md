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

Effectful authorization now also carries a pinned temporal validity contract. `AuthorizationClockPolicy` is persisted by digest in the relying-party store domain and covers maximum authorization age, allowed clock skew, whether an explicit expiry is required, and the explicit clock-source identity. The current project default is 48 hours maximum age, 5 minutes allowed skew, mandatory expiry, and `system-utc-wall-clock-v1` backed by the process's `SystemTime` UTC source. Deployments can open the store with an explicit policy, and an existing durable store rejects reopening under a different policy digest or clock-policy configuration. Before `prepare_for_execution_bound` commits the effectful reservation, the witness issuance/expiry window is parsed, normalized, checked against the trusted system UTC clock, and persisted on the lease. `DispatchPending` rechecks that frozen window before writing the durable dispatch record. Immediately before provider entry, `mark_invoked_bound` rechecks it again; if it has expired or otherwise become invalid, the dispatch is atomically closed as `not_entered`, the authorization instance becomes terminally `Expired`, and a durable `not_entered_validity` marker is recorded. The provider is never entered on that path, and a later witness cannot extend the same authorization instance with a new validity window. Terminal settlement and later reconciliation validate the frozen temporal provenance structurally but do not reject an already-entered effect merely because the authorization has subsequently expired. This prevents two boundary instances sharing one durable store from claiming each other's attempt records while preserving one common same-action fence. A stale executor that presents a different target, audience, adapter, action digest, provider key, or boundary identity is rejected rather than being allowed to reinterpret the authorization after dispatch preparation.

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

#### Boundary-scoped attempt provenance

Every bound attempt now has a deterministic `attempt_scope_digest` derived from `(boundary_id, attempt_id)` under a dedicated domain separator. The value is persisted alongside every attempt-keyed durable record: the lease, provider-status checks, recovery markers, dispatch record, terminal evidence, and execution receipts. Raw `attempt_id` values remain for compatibility and human/audit correlation, but the derived scope is the durable namespace binding used to detect cross-boundary substitution.

The effect-bound preparation path creates the scope before DispatchPending, and the dispatch record carries the same derived scope into terminal evidence and receipts. Recovery markers and crash-recovery receipts receive the same scope so pre-entry recovery and indeterminate reconciliation cannot silently move to a different boundary namespace.

Startup migration is fail-closed. Historical status rows recover their boundary only from the authoritative dispatch row for the same authorization/attempt pair; the migration then derives the scope. For every attempt-bearing table, a missing scope is backfilled from the persisted boundary, while a present non-empty scope must exactly match the deterministic derivation or store initialization fails. No caller-supplied boundary is accepted as migration repair input.

This is an implementation hardening of AEB-07, not a claim that the draft mandates this particular SHA-256 encoding. The current AEB-07 text requires recovery authorization to identify exactly one attempt, requires attempt identity to be unique and scoped across boundaries sharing durable state, and requires the boundary identifier to enter the derivation of every attempt-keyed record. citeturn728301search0turn728301search2


### Provider-verifier boundary

Terminal provider outcomes are a separate authority boundary from the durable dispatch record.

A DispatchPending or Invoked record proves only local lifecycle facts. A raw Succeeded/Failed enum returned by an adapter is not terminal evidence by itself. Effectful terminal commitment and reconciliation therefore require a relying-party-configured ProviderEvidenceVerifier.

The terminal verifier configuration is pinned as a complete provenance tuple: relying-party ID, verifier/profile ID, exact verifier revision, verifier implementation ID, verifier implementation digest, verifier configuration digest, trust-anchor-set digest, and accepted evidence-profile/schema digest. The exact revision and implementation tuple is persisted with terminal evidence and revalidated inside the authoritative settlement/reconciliation transaction. Presented verifier metadata cannot become its own trust root or silently select a new implementation.

The provider-status verifier is held to the same distinction. Its relying-party pin now includes verifier ID, exact revision, implementation ID, implementation digest, and configuration digest. Admission and pre-entry both re-check that pin at the authoritative dispatch transaction after verifier execution, so a verifier cannot mutate its own trusted implementation between verification and provider entry. The status pin is write-once; historical two-field pins may be explicitly completed with the three previously absent fields while existing values must match exactly.

A PreEntryLookup is a different evidence kind and is rejected by terminal commitment/reconciliation. The verifier is invoked with an explicit TerminalOutcome purpose and the exact frozen dispatch record; a verifier result that does not affirm that purpose or the relying-party-selected configuration is not terminal authority.

This is an implementation hardening of AEB-07, not a claim that the draft mandates this exact storage schema. The current AEB-07 text explicitly requires the relying party to pin the native verifier revision and the verifier implementation identifier/digest, and requires those inputs to remain relying-party-selected rather than introduced by presented data. citeturn835719view0

### Execution identity lattice

The effect boundary now records the execution identities that must not be conflated:
Before an effect-bound dispatch can reach the native status verifier, the material effect-binding identities must all be concrete: target identity, audience/environment identity, and adapter identity cannot be empty. This check is structural rather than semantic—provider-specific meaning remains the responsibility of the pinned native/status/adapter layers—but an empty material identity is never admitted as a valid frozen effect contract.


- **Authorization instance** identifies the durable grant of authority.
- **Operation ID** identifies the logical operation being attempted.
- **Native replay identity** identifies the one native grant of authority and is supplied by the native authorization path.
- **Attempt ID** identifies one executor attempt.
- **Action digest** identifies the frozen material action.
- **Provider idempotency key** identifies the downstream replay-control value.

The bound dispatch API requires the operation ID and native replay identity explicitly and rejects either when absent. Neither is derived from the attempt ID, provider idempotency key, wrapper, boundary label, or other local retry metadata. The durable dispatch row and terminal provider evidence carry both values, plus the persisted native issuer and native-authority pin-set reference for effectful native handoffs; the verifier must affirm the execution/evidence tuple against the exact frozen record.

The effect-bound attempt namespace is also derived rather than merely labeled: the store computes an `attempt_scope_digest` from the exact boundary ID and attempt ID under a dedicated domain separator. That scope is persisted across the lease, receipt, status-check, recovery-marker, terminal-evidence, and dispatch records. Existing durable rows are backfilled only from authenticated stored boundary ownership; a present-but-mismatched scope digest fails store initialization, and a dispatch scope mismatch fails validation before provider entry or terminal settlement. This is an implementation-level strengthening of AEB-07's requirement that attempt ownership remain unique across boundaries and that recovery remain attached to exactly one attempt. citeturn835719view0


The store also rejects reuse of a non-empty operation ID or native replay identity within its durable dispatch domain. This makes identity collisions visible at the persistence boundary rather than allowing a later attempt to reinterpret an existing operation or native grant.

For an effectful attempt, the store also enforces a transactional same-action in-flight fence over the exact effecting target identity and canonical action digest. A second authorization with a fresh native replay identity cannot enter DispatchPending while another attempt for the same material target/action remains DispatchPending, Invoked, or Indeterminate. The check is performed inside the same immediate SQLite write transaction as dispatch-row creation, so it is not a best-effort preflight race. Terminal Succeeded or Failed states release this in-flight fence; Indeterminate remains occupied until authenticated reconciliation.

This layer deliberately does **not** claim that the native replay identity has been derived correctly merely because a caller supplied a string. The native authorization adapter remains responsible for authenticating the native authority and deriving its replay identity from the pinned authority namespace and native authorization identifier. The store's role is to require, freeze, bind, and durably fence that value once it crosses the execution boundary.

The effect boundary now exposes a canonical native-authorization entry point that derives the native replay identity from the pinned authority namespace and native authorization identifier before constructing the durable dispatch record.

The temporal contract is part of the frozen dispatch evidence, alongside native replay and pin-set provenance. The store uses an explicit `TrustedAuthorizationClock` interface; the default `SystemUtcClock` is only one implementation. The relying party's clock source identifier is durably pinned on the store and included in the clock-policy digest, and a reopen under a different source identity fails closed. This keeps time validation testable and prevents an unpinned ambient clock implementation from silently changing the meaning of a historical validity policy. An expired effectful grant is terminal rather than reusable; legitimate re-authorization therefore requires a new authorization instance. The boundary therefore does not treat a transient `expires_at` field as a best-effort hint; it turns the accepted window and clock policy into durable evidence that later restart/reconciliation logic can verify. That derivation witness—authority namespace, native authorization identifier, and derivation digest—is persisted with the dispatch record and propagated into terminal provider evidence in the same durable lifecycle. The former free-form replay-identity entry points remain only as deprecated hard-fail shims: they cannot create an effectful dispatch record without durable native derivation provenance.

The durable SQLite authorization store is also pinned to one relying-party control domain. Native authority pins are normalized for issuer-collision checking: URI schemes are lowercased; for HTTP(S), host casing is canonicalized, a trailing host dot is removed, default HTTP(S) ports are removed, trailing URL path slashes are removed, and the AEB-07 `https:a.example` form is treated as the equivalent authority URL with `//`. Non-URI or otherwise unparseable issuer values retain their exact representation rather than receiving invented normalization. Equivalent issuer spellings must resolve to one authority namespace; a normalized collision with a different namespace fails closed.

Each effectful dispatch now freezes the identity and digest of the exact **native-authority pin set** used at admission. The canonical dispatch also records the issuer spelling that was resolved, and later lifecycle checks normalize that issuer and require it to resolve to the same pinned authority namespace. The durable store retains the canonical issuer→authority-namespace snapshot keyed by a stable relying-party-scoped pin-set identity and versioned digest (sha256:v2:...); historical v1 content-only digests remain verifiable for older durable evidence. Terminal evidence carries the same issuer and pin-set reference. Adding a later issuer pin creates a new v2 pin-set digest without rewriting earlier attempts, so reconciliation can recover the exact historical native-authority mapping rather than silently evaluating an old attempt under today's pin set. Other verifier, clock, status-source, or gateway configuration remains a separate relying-party concern. Production callers can open it with an explicit relying-party identifier; the compatibility open path uses the legacy-local domain. The same-action in-flight fence is therefore evaluated as relying party + effecting target identity + action digest, while the relying-party binding itself remains outside the canonical action digest. A store cannot later be reopened under a different relying party, preventing accidental cross-domain fence reuse. The native authority pin set is likewise durable and write-once: an accepted issuer is resolved to one authority namespace inside the effect boundary before native replay identity derivation. Unpinned issuers and attempts to change an established mapping fail closed. The same-action fence is independent of native replay identity: fresh native authority cannot bypass an occupied action key. An EXECUTED/SUCCEEDED result keeps that action key closed; a FAILED result releases it for a later attempt only when fresh native authority and a new operation lifecycle are provided.

### Provider status / revocation boundary

Effectful admission now uses a relying-party supplied `ProviderStatusVerifier` to obtain authenticated current-status evidence for the native authorization's status identifier. The verifier binds the lookup to the frozen native issuer, authority namespace, native authorization identifier, action digest, effecting target, audience, and adapter, and returns the status source digest plus an observed/valid-until window. The store independently validates the returned evidence for required fields, source continuity, and freshness against its pinned clock policy. The accepted `status_source_digest` is itself a relying-party write-once pin; the status verifier cannot establish, replace, or broaden the trusted status source by returning a different digest. The relying party also pins the status verifier implementation/configuration identity and digest. The canonical admission and pre-entry paths refuse a runtime verifier whose declared configuration does not match that durable pin, so dependency injection cannot silently substitute a different verifier while retaining the same status source.

The status check occurs before the canonical dispatch record is created. If admission status is revoked, stale, unavailable, unauthenticated, or otherwise fails verification, the already-prepared reservation is atomically released and recorded as `not_entered_status`; no `DISPATCH_PENDING` record is created.

Immediately before provider entry, the boundary invokes the same verifier again with `ProviderStatusVerificationPurpose::PreEntry` and the exact frozen native identities and status source digest. A failed or stale result atomically closes the `DISPATCH_PENDING` attempt as `not_entered`, releases its lease reservation, and records `not_entered_status`; it does not become `FAILED` or `INDETERMINATE` because provider entry has not begun. Successful admission and pre-entry status evidence are retained in the append-only `authorization_status_checks` table. This separates status authority from provider terminal-outcome evidence while preserving the exact status source and freshness context used at each checkpoint.
Provider idempotency derivation is now native-replay based. The effectful path derives the provider key from the native replay identity plus the exact effecting target and canonical action digest under a versioned domain separator. The operation identifier, authorization presentation identifier, attempt identifier, wrapper/handoff digest, session, task, trace, challenge, and provider-supplied identifier are excluded. This means executor retries and operation-label changes cannot manufacture a fresh downstream effect identity. The former authorization-instance derivation API is deprecated and hard-fenced; it is not a usable effectful compatibility path.

### Pre-entry recovery versus Indeterminate reconciliation

The recovery boundary is intentionally split in two. A Prepared attempt that has not crossed DispatchPending can be recovered as a pre-entry stop only through the exact recovery witness and an atomic transition that records the not-entered marker. That transition prevents the stranded executor from dispatching the old attempt after recovery. A DispatchPending or Invoked attempt cannot use this release path; it remains occupied and follows the Indeterminate reconciliation path.

`RecoveryAuthorizationWitness` binds a recovery operation to the authorization instance, exact attempt, exact boundary, action digest, authority epoch, and issuance metadata. When the durable lease carries an explicit operation ID, the recovery witness must also carry the same non-empty operation ID; an empty or mismatched operation identity is rejected. The store enforces those bindings; the authority/authentication layer remains responsible for authenticating the issuer and policy. This keeps recovery authorization distinct from the execution receipt and prevents a recovery operation for one attempt from being applied to another.


### Operation identity is frozen at preparation

The effectful lifecycle now treats the logical operation identifier as an attempt-bound authorization input, not caller metadata that can be replaced at the `DispatchPending` transition. `prepare_for_execution_bound_with_operation` persists the operation identity before dispatch; both the precondition check and the authoritative `DispatchPending` transaction require the exact persisted value. A caller cannot prepare operation A and enter the provider as operation B while retaining the same action, native replay, target, and status evidence.

### Terminal settlement rejects record splicing

Terminal settlement and reconciliation now compare the caller-supplied durable dispatch record against the authoritative persisted dispatch row across the complete immutable identity surface: operation ID, native replay identity and derivation inputs, issuer/namespace/native authorization ID, relying-party domain, action identity/digest, provider idempotency key, target, audience, adapter, and boundary. A record that combines fields from different attempts is rejected before lease settlement.

This closes a distinct class of provenance failure from operation-ID substitution: a valid attempt cannot be paired with a forged target, audience, adapter, or operation while retaining a valid action digest. The persisted row remains the source of truth; terminal provider evidence is accepted only after the record itself is proven to be the exact durable attempt.

### Terminal verifier configuration is relying-party pinned

Terminal provider evidence is now accepted only under a write-once relying-party pin for the complete verifier configuration: relying-party identity, verifier implementation/profile identifier, verifier configuration digest, trust-anchor digest, and evidence-profile digest. Missing pin metadata or a verifier result that differs from the pinned tuple is rejected before terminal settlement or authenticated reconciliation. The pin is itself write-once: re-pinning a different verifier identity, configuration digest, trust-anchor digest, or evidence-profile digest fails closed.

The verifier's returned configuration is therefore evidence about what verifier claims it used, not a presenter-controlled trust root. The durable store selects the accepted configuration and requires an exact match before consuming the authorization. This follows the AEB-07 requirement that verifier revisions, trust anchors, and related validation inputs be relying-party-selected rather than introduced by presented data. citeturn102081search0turn102081search2


### Immutable attempt provenance commitment

Each strict effectful dispatch now carries a deterministic attempt_binding_digest committed at DispatchPending. The digest is domain-separated and length-prefixed over the immutable attempt provenance tuple: authorization instance, attempt and operation identities, native replay identity and derivation witness, issuer/authority namespace, relying-party identity, action identity/digest, provider idempotency key, effect target/audience/adapter, admission status evidence, and the frozen authorization validity window and native-authority pin-set snapshot.

Lifecycle state is deliberately excluded from the commitment because state transitions are governed by the transactional state machine. Before provider entry and again before terminal settlement or reconciliation, the durable store recomputes the commitment from the persisted row and requires exact equality with the caller's dispatch record. This means database-side mutation of status evidence, validity provenance, target, operation, replay derivation, or other committed material fields cannot be repaired merely by presenting a matching caller object.

This is an implementation-level provenance commitment rather than a new AEB standard requirement; it strengthens the AEB-07 rule that an operation record must bind the native replay identity and action key to the executor-owned action and operation identifier, and that reconciliation must remain attached to the original attempt. citeturn914749search0turn914749search1

### Adapter revision and implementation pin

Strict effectful admission now requires the adapter identifier named by the frozen effect to resolve to a relying-party-pinned adapter revision and implementation digest. The pin is durable and write-once. The dispatch record and terminal evidence carry the exact selected revision and implementation digest, and the attempt provenance commitment covers both values.

Settlement additionally re-reads the adapter pin and requires it to match the frozen record. The same persisted-pin check runs before provider entry, before the boundary performs the pre-entry status verification. A later adapter replacement, database-side pin mutation, or presenter-selected revision therefore cannot silently reinterpret an already-authorized attempt. This aligns with AEB-07's requirement that the relying party pin every adapter revision and, in the native compilation contract, the verifier/adapter implementation identity and digest. citeturn281056view0

### Bound receipts preserve the native provider replay key

Bound terminal receipts now persist the exact provider idempotency key used by the native-replay-derived dispatch. Re-reading or idempotently replaying a terminal receipt therefore returns that same key rather than reconstructing the legacy authorization-instance-derived key.

Historical unbound receipts may retain the legacy reconstruction fallback for compatibility. Strict effect-bound receipts do not: their persisted provider key is part of the durable attempt lineage and is preserved across terminal retries and recovery paths. This prevents a read-after-settlement operation from silently presenting a different replay identity than the one used at provider entry. citeturn366187view0
