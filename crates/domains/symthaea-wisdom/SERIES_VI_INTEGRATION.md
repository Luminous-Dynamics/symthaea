# Symthaea Wisdom — Series VI Production Integration

Series VI turns the v5 trust model into an executable local service boundary. It does not claim that a database transaction can atomically commit an external side effect. Instead, it makes the unavoidable uncertainty visible and recoverable.

## Patch sequence

### 39. Staged action execution

`ActionExecutionCoordinator` now exposes two explicit stages:

1. `prepare_execution()` evaluates structural ethics, issues and consumes the capability permit, allocates the attempt, and appends `ActionExecutionStarted`.
2. `complete_execution()` accepts the exact prepared attempt and appends the terminal result.

Production callers must durably commit stage 1 before invoking an external executor. The original `authorize_and_execute()` remains as a compatibility helper, but its documentation now states that it is only process-local.

### 40. Transactional storage adapter

`LedgerStorageFrame` is a canonical checksummed envelope around one `EvidenceLedger` and its declared `LedgerRevision`.

`AtomicLedgerBackend` is the production adapter contract. A database implementation must provide:

- monotonic writer fencing,
- durable frame loading,
- atomic lease-and-revision compare/exchange,
- durability before reporting `Committed`.

`BackendLedgerStore<B>` converts that contract into the existing `DurableLedgerStore` interface. Corruption, revision mismatch, oversized frames, and invalid evidence fail before becoming runtime state.

### 41. Permit authority recovery

`recover_permit_journal()` reconstructs:

- issued permits,
- consumed nonce history,
- explicit revocations,
- expiry and policy invalidation,
- outstanding single-use authority,
- the next nonce.

Contradictory terminal events, binding changes, consumption without issuance, and evicted history fail closed. `WisdomState::recover_action_permits()` installs the recovered gate.

### 42. Operational admission permits

A ready `OperationalStartupReport` can issue an opaque `OperationalStartupPermit`. It binds:

- the exact durable ledger revision,
- operational-state fingerprint,
- ethics-policy fingerprint,
- trust-registry fingerprint,
- runtime-source-set fingerprint,
- validation time.

The permit is short-lived and must be revalidated immediately before accepting work. Changes after preflight invalidate admission.

### 43. Crash-honest runtime service

`WisdomRuntimeService<S>` owns:

- one `CoordinatedLedgerWriter`,
- reconstructed `WisdomState`,
- recovered `ActionExecutionCoordinator`,
- recovered per-source runtime cursors,
- the exact trust registry and source identities admitted at startup.

Local observations and authenticated runtime events are committed as complete ledger successors. Any fencing or revision uncertainty permanently deactivates the service until a fresh bootstrap.

Action dispatch follows this sequence:

1. prepare authority and append start;
2. commit the start;
3. invoke the executor;
4. append completion;
5. commit completion.

If step 5 fails, the service returns the executor result inside `CompletionPersistenceFailed`, deactivates itself, and leaves a durable in-doubt action for external reconciliation.

### 44. Persistent trust registry

`TrustRegistry::to_canonical_bytes()` and `from_canonical_bytes()` provide deterministic, checksummed persistence. Decoding rejects:

- bad magic or version,
- excessive lengths or key count,
- invalid UTF-8,
- invalid role/status markers,
- inverted key windows,
- duplicates,
- non-canonical ordering,
- trailing bytes,
- checksum mismatch.

### 45. Crash-injection invariant

The service tests inject failure on the terminal commit after a successful executor call. The durable ledger retains `ActionExecutionStarted`, contains no completion, and `recover_execution_journal()` reports one in-doubt execution.

## Minimal wiring

```rust
let store = BackendLedgerStore::new(production_backend);

let startup = validate_operational_startup(
    &preflight_state,
    startup_evidence,
    OperationalStartupRequirements::default(),
);
let admission = startup.admission_permit()?;

let mut service = WisdomRuntimeService::bootstrap(
    store,
    "symthaea-runtime-1",
    wisdom_config,
    ethics_policy,
    trust_registry,
    runtime_sources,
    admission,
    now_millis,
    DEFAULT_STARTUP_ADMISSION_MAX_AGE_MILLIS,
)?;

let outcome = service.authorize_and_execute_durable(
    action_request,
    &mut executor,
    started_at_millis,
    completed_at_millis,
    "scheduler:action-dispatch",
)?;
```

## Required production behavior

- Never call the executor before the start commit succeeds.
- Never automatically retry an in-doubt action.
- Reconcile uncertainty from a receipt issued by the actual side-effecting system.
- Treat a deactivated service as fenced; construct a new service from the durable head.
- Persist the trust registry as canonical bytes and compare its fingerprint with startup and deployment expectations.
- Keep runtime-source verification on every event, not only at startup.

## Explicit remaining boundary

The first `WisdomRuntimeService::bootstrap()` intentionally rejects ledgers with evicted records. v5 proves retention and archive continuity, but Series VI does not yet define a signed authority checkpoint containing permit state, execution duplicate-prevention state, and every runtime cursor. Until that checkpoint exists, admitting work from a bounded head would require guessing missing authority history.

A production deployment may still use bounded evidence for audit and release, but the live service must bootstrap from complete authority history or remain stopped.
