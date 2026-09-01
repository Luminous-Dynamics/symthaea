# AI Assurance Effect-Entry Evidence v1.3

Status: design-frozen successor to effect-entry v1.2. No runtime source change is authorized by this document alone.

Base branch: `research/ai-assurance-effect-entry-v1.2`.

Fresh design branch: `research/ai-assurance-effect-entry-evidence-v1.3r1`.

## Why v1.3 exists

Effect-entry v1.2 gives acquisition and revocation one deterministic ordering point. A successful `EffectEntryPermit::enter(...)` constructs an `EffectEntryReceipt` immediately before invoking arbitrary adapter code, but returns that receipt only after the callback returns.

That leaves one evidence-ordering gap:

> admission may have won and an external effect may have begun, yet panic, abort, process loss, or another abrupt boundary can prevent the caller from ever receiving the admission receipt.

The in-memory activity counters are repaired on Rust unwind, but counter repair is not the same claim as durable evidence that adapter entry was admitted.

v1.3 closes only that gap. It does not widen authority, change the acquire-vs-revoke linearization point, or claim durable crash recovery by itself.

## Core invariant

For the strict host path, evidence of effect admission must be available to trusted host code **before arbitrary adapter code can begin**.

The intended ordering is:

1. exact v1.2 ticket acquisition linearizes;
2. the affine permit transitions from `outstanding` to `in_flight` under the domain lock;
3. an immutable `EffectEntryReceipt` is constructed;
4. trusted host code receives and may persist that receipt;
5. only then may arbitrary adapter code cross its declared point of no return;
6. dropping/unwinding the in-flight guard repairs activity accounting.

The domain mutex remains released before adapter code runs.

## Intended API

The preferred source refinement is additive:

- `EffectEntryPermit::begin(self) -> Result<EffectInFlight, EffectEntryError>`
- `EffectInFlight::receipt(&self) -> EffectEntryReceipt`
- `EffectInFlight::run(self, effect) -> R`

`EffectInFlight` is public, affine, and neither `Clone` nor `Copy`.

`EffectEntryPermit::enter(...)` may remain as a compatibility convenience implemented as `begin -> receipt -> run`, but its documentation must state that callers requiring panic/crash-surviving evidence should use the explicit `begin` path and persist the receipt before calling `run`.

No caller-supplied verifier, clock, epoch, or authority root is introduced.

## State accounting

`begin` must atomically, under the existing short domain mutex:

- require one outstanding permit;
- checked-increment `in_flight_effects`;
- decrement `outstanding_permits`;
- capture the resulting activity snapshot for the receipt;
- detach the original permit's Drop cleanup so the same unit cannot be decremented twice.

After `begin` returns, the domain must report that unit as in flight until the `EffectInFlight` value is dropped.

`EffectInFlight::Drop` must decrement `in_flight_effects` exactly once, including during Rust panic unwinding.

Dropping an `EffectInFlight` without ever running adapter code is valid accounting: admission was won, but this primitive alone must not reinterpret that as proof an external effect occurred. The existing effect-attempt/observation layers own that distinction.

## Evidence semantics

`EffectEntryReceipt` remains evidence that effect **admission** linearized successfully, not proof that the external operation succeeded or even began.

The receipt must continue to preserve:

- permit id;
- ticket id;
- effect-entry domain id;
- admitted epoch;
- exact `EffectAdmissionCommitment`;
- acquisition sequence;
- activity snapshot at transition to in-flight.

The field currently described as activity immediately before callback execution should be documented more precisely as activity when the permit transitioned in flight. Adapter execution may intentionally occur later, after the host persists evidence.

## Persistence boundary

v1.3 makes pre-adapter evidence **available**. It does not define one universal persistence backend.

A concrete host may require one of several policies, for example:

- append + fsync to a local evidence journal;
- durable database transaction;
- Xenia/HAL-signed receipt store;
- process-external supervisor acknowledgement;
- replicated evidence-plane write;
- weaker in-memory-only persistence for low-risk operations.

The required persistence class should be policy-selected by action/effect class. The core must not falsely call a memory copy crash-durable evidence.

A future strict facade should make the selected persistence acknowledgement structurally necessary before invoking `EffectInFlight::run` for risk classes that require durable admission evidence.

## Failure taxonomy

Keep these cases distinct:

### Acquisition rejected

No admission receipt exists. Adapter entry is not authorized by this effect-entry domain.

### Begin rejected

The permit could not transition into in-flight accounting. Adapter code must not run.

### Begun but adapter not invoked

An admission receipt exists and the guard may have been persisted, but external-effect occurrence is not inferred. The effect-attempt layer should record `ProvenNoEffect` only when adapter-specific evidence justifies that stronger statement.

### Adapter panic/unwind

The previously exposed receipt remains available to the host/persistence layer. In-memory in-flight accounting is repaired by guard Drop. Panic does not imply `NoEffect`.

### Process abort/crash after persistence

The durable receipt can establish that admission won before the crash, subject to the persistence system's own guarantees. In-memory activity accounting is lost with the process and must not be reconstructed as authoritative durable state without a separate recovery protocol.

### Process abort/crash before persistence acknowledgement

Outcome remains evidence-incomplete/unknown. v1.3 must not claim to solve this window.

## Revocation interaction

The v1.2 linearization rule is unchanged:

- revocation first -> acquisition fails;
- acquisition first -> one permit is already admitted.

`begin` does not create a second revocation race or a second authority decision. It is an evidence/accounting transition for a permit whose admission already won.

After revocation latches the domain stopped, an already-acquired permit may still `begin`; it represents exactly the already-admitted unit recorded by the revocation receipt. Resume remains blocked until outstanding plus in-flight admitted work becomes quiescent.

## Adversarial qualification requirements

The v1.3 source implementation must add tests proving at least:

1. `begin` exposes the exact admission receipt before arbitrary adapter code runs;
2. after `begin`, activity is `outstanding=0, in_flight=1` for the single permit;
3. dropping an unrun `EffectInFlight` restores quiescence;
4. normal `run` restores quiescence;
5. panic during `run` restores quiescence while a receipt copied/persisted before the call remains available;
6. revocation can complete while `run` blocks, proving no domain mutex is held across adapter code;
7. revocation after acquisition but before `begin` still allows only that already-admitted permit and blocks resume until the guard settles;
8. the same permit cannot `begin` twice at compile time;
9. the same in-flight guard cannot `run` twice at compile time;
10. `EffectEntryReceipt` preserves the same commitment/domain/epoch/acquisition sequence as v1.2;
11. compatibility `enter` preserves existing observable v1.2 behavior;
12. no new unlocked dependency or unrelated workspace change is required.

## Qualification gate

Do not treat v1.3 as compiler-qualified merely because this design is committed.

Implementation should begin from the exact v1.2 tree that passes or yields actionable focused qualification evidence. The implementation commit must then pass the read-only focused lane with Rust 1.96.0:

1. `cargo fmt --package symthaea-ai-assurance -- --check`
2. `cargo metadata --locked --format-version 1`
3. `cargo test --locked -p symthaea-ai-assurance`
4. compile-fail doctests included by the package test run
5. `cargo clippy --locked -p symthaea-ai-assurance --all-targets -- -D warnings`
6. no tracked assurance or lockfile mutation during qualification

If the v1.2 focused lane reports a source defect, repair the owning lower layer first and advance this branch from that repaired tree before implementing v1.3.

## Abandoned writer provenance

An earlier branch, `research/ai-assurance-effect-entry-evidence-v1.3`, briefly added and then removed a write-capable one-shot GitHub Actions workflow intended to synthesize this source change. The source implementation did not land; the branch returned to the old v1.2 source tree.

This successor intentionally does not revive that workflow. CI must qualify committed source, not manufacture the source it is supposed to qualify.

## Non-claims

v1.3 does not by itself provide:

- durable evidence storage;
- process-crash recovery of in-flight accounting;
- distributed transaction atomicity;
- composite multi-domain admission;
- adapter-specific cancellation after point of no return;
- proof that `adapter_semantics_digest` truthfully describes the implementation;
- proof that an admission receipt means the external effect occurred;
- end-to-end MAGI integration;
- general AI alignment or correctness.

## Successor work after v1.3 qualification

The next architectural tranche should address **degradation and recovery authority** rather than silently falling back when a required trust dependency is unavailable.

That successor should distinguish at least `Unavailable`, `Revoked`, policy-authorized `Degraded`, and explicitly `Recovered` states; bind recovery to a fresh epoch/configuration commitment; forbid automatic weakening of trust roots or enforcement classes; and compose degradation/recovery ordering with effect-entry admission.
