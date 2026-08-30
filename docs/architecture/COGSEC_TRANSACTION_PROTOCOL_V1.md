# CogSec Mutation Transaction Protocol v1

Status: design contract for the constitutional-core tranche. Runtime integration is not yet implemented.

## Purpose

Authorization and mutation are separate events. A decision that was valid when a request was evaluated may become invalid before the protected state is actually changed because resource state, policy, identity, revocation, trust, or consequence context changed.

CogSec therefore uses state-bound, one-use mutation transactions rather than generic authorization booleans.

## Transaction phases

1. **Prepare** — build an exact mutation proposal from untrusted/ordinary cognition.
2. **Evaluate** — the reference monitor evaluates the proposal against trusted facts and canonical policy.
3. **Authorize** — on `Allow`, the monitor mints a one-use permit bound to the exact proposal and observed state.
4. **Precommit revalidation** — the protected state owner verifies that every bound state/epoch is still current.
5. **Commit** — the protected state owner consumes the permit while applying the exact mutation.
6. **Receipt** — record the committed/aborted/uncertain outcome with commitments, not raw private cognition by default.

A permit is not a promise that a mutation will eventually happen. It is evidence that one exact transition was authorized against one exact state snapshot.

## Required permit bindings

A live permit must bind at least:

- request identity;
- mutation kind;
- resource identity;
- exact mutation/effect digest;
- protected-resource state root;
- policy root;
- policy epoch;
- authorization epoch;
- revocation epoch;
- logical sequence/anti-replay value.

Deployment-specific adapters may additionally bind identity key epochs, capability-chain roots, consequence budgets, environment roots, expiry, human-consent transcripts, or hardware-attestation roots.

## Precommit invariant

Immediately before a protected state write, the state owner must compare the live permit against authoritative current state.

Commit fails closed when any security-relevant binding changed, including:

- resource state root;
- policy root or epoch;
- authorization epoch;
- revocation epoch;
- resource identity;
- mutation digest;
- mutation kind;
- sequence/replay state.

The correct outcome is `RequireRevalidation`, not silent reuse of the stale permit.

## One-use semantics

A live permit should behave approximately like a linear capability:

- no public constructor;
- no `Default`;
- no serde deserialization into a live permit;
- no ordinary `Clone`;
- consumed by commit;
- exact resource/effect binding;
- replay state advances on successful commit or explicit terminal abort according to policy.

A serialized signed capability envelope is an authorization input. It is not a live commit permit.

A historical receipt is evidence that a prior decision/commit occurred. It is not reusable authority.

## Atomicity classes

CogSec must not overclaim rollback semantics. Mutations fall into different atomicity classes.

### A. Reversible internal state

Examples: an in-memory goal set, staged working-memory structure, local accounting object.

Preferred semantics:

`clone/stage -> validate -> commit`

or an equivalent explicit prepare/commit API.

A rejected mutation leaves pre-existing protected state unchanged except for explicitly independent append-only audit/error accounting.

### B. Persistent local state

Examples: episodic memory, semantic memory, local model checkpoint, local policy database.

Preferred semantics:

`prepare durable candidate -> fs/db transaction or append-only event -> atomic pointer/root promotion`

The new state root becomes visible only after validation succeeds. Crash recovery can determine whether promotion occurred.

### C. External/irreversible effect

Examples: sending money, deleting remote infrastructure, transmitting sensitive data, operating an actuator, sending a message to an external system.

CogSec cannot truthfully promise rollback after the effect reaches the external world.

Preferred semantics:

`prepare -> authorize -> precommit revalidate -> execute once with idempotency/effect binding -> reconcile outcome -> receipt`

For such actions, all feasible checks happen **before** execution. If the external outcome is uncertain after a crash/timeout, the system enters `UnknownExternalOutcome` and reconciles using authoritative external receipts/status. It must not blindly retry a consequential action.

## Crash-recovery states

A transaction journal may use states such as:

- `Prepared`;
- `Denied`;
- `Quarantined`;
- `AuthorizationRequired`;
- `Permitted`;
- `Committing`;
- `Committed`;
- `Aborted`;
- `UnknownExternalOutcome`;
- `Reconciled`.

`Permitted` alone does not mean the mutation happened.

After restart:

- reversible/local transactions may be safely aborted or completed according to their durable commit marker;
- external `Committing` transactions require reconciliation before retry;
- stale policy/authorization/revocation epochs force revalidation;
- historical permits are never reconstructed from receipts.

## Nested and cascading mutations

One approved mutation must not implicitly authorize every downstream effect it triggers.

Derived privileged mutations carry a parent transaction/request reference and are re-evaluated under their own mutation kind, resource, consequence, and required capability.

Example:

`GoalActivation` does not automatically authorize `ToolInvocation`.

`ToolInvocation` does not automatically authorize `ExternalAction`.

`PersistentMemoryCommit` does not automatically authorize `SemanticPromotion`.

`LearningPromotion` does not automatically authorize `SecurityPolicyChange`.

This prevents authority laundering through a chain of individually plausible internal transitions.

## Consequence accumulation

Transaction policy should evaluate both the requested mutation and cumulative session/plan consequences. Many individually moderate operations can compose into a high-consequence effect.

The transaction context should therefore be able to bind or reference:

- parent transaction;
- action/plan sequence root;
- cumulative consequence budget;
- cumulative confidentiality release;
- cumulative influence budget.

Crossing a configured budget requires new authorization even if each individual step would otherwise be allowed.

## Delegation attenuation

A delegated authorization may only narrow its parent:

- equal or smaller mutation set;
- equal or narrower resources;
- equal or lower consequence ceiling;
- equal or shorter validity;
- equal or stricter policy constraints.

A child capability may never widen parent authority. This is a formal/property-test target.

## Error accounting

Error/audit accounting is security-relevant state but may be intentionally independent of successful mutation state.

If policy specifies that a failed backend/action attempt increments an append-only error counter, that counter may commit even while the proposed cognitive mutation is aborted. The distinction must be explicit and tested.

This mirrors the useful correctness invariant already developed in Symthaea's experimental SCIP transactional LLM lineage: rejected surfaces preserve prior accepted state, while explicitly independent backend-error accounting may still advance.

## External adapter contract

Xenia or another authorization adapter may verify signatures, identities, delegation chains, consent transcripts, freshness, and revocation. It returns trusted authorization facts/capability facts to CogSec.

The adapter does not directly mutate cognition.

CogSec does not perform cryptographic verification itself in the logical core.

The protected state owner does not trust authorization metadata embedded in the untrusted mutation payload.

## Qualification obligations

At minimum, tests should demonstrate:

1. state-root change after authorization prevents commit;
2. policy-root/epoch change after authorization prevents commit;
3. revocation-epoch change after authorization prevents commit;
4. mutation-digest substitution prevents commit;
5. resource substitution prevents commit;
6. a consumed permit cannot be replayed through the normal API;
7. rejected transactions preserve non-zero pre-existing protected state;
8. independent error/audit accounting changes only the fields explicitly allowed by contract;
9. crash recovery never blindly repeats an external consequential action with uncertain outcome;
10. child delegation never exceeds parent scope/consequence/lifetime;
11. chained privileged effects require their own permits;
12. cumulative consequence/influence/declassification budgets cannot be bypassed by splitting one operation into many smaller operations.

## Non-claims

This protocol does not claim ACID semantics for arbitrary remote systems, rollback of external model/provider internals, or exactly-once behavior where the external system supplies no idempotency/reconciliation primitive. Those limits must remain visible in the resulting assurance case.
