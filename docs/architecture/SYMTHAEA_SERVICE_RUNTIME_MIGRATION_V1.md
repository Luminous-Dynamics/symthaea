# Symthaea Service Runtime Migration v1

Status: architecture/migration contract only. This document does not claim that the service has already migrated, that latency targets are met, or that any runtime source tranche has passed executable qualification.

## 1. Purpose

The service daemon currently stores the concrete cognitive facade behind a shared asynchronous mutex:

```text
ServiceState
  └── Mutex<Symthaea>
```

A normal query acquires that mutex and holds it across the complete asynchronous `Symthaea::process()` call. Status/introspection/metrics requests, the background consciousness loop, sleep/save paths, and voice cognition turns also acquire the same mutable facade.

This creates a responsiveness coupling:

```text
long cognitive turn
      ↓
Symthaea mutex held
      ↓
status / introspection / other mutation paths wait
```

INT-003A introduces a bounded single-owner command core. This document freezes the concrete daemon migration so that adopting that owner does not merely rename the mutex or route read-only UI traffic through the same slow command queue.

## 2. Governing invariant

After the migration is complete:

> Exactly one task owns mutable `Symthaea` state after startup initialization.

No service connection, UI, voice transport, background ticker, HTTP handler, WebSocket handler, or shell adapter receives `&mut Symthaea` or owns another `Mutex<Symthaea>`.

The intended topology is:

```text
                               ┌──────────────────────┐
commands ─────────────────────▶│   SymthaeaRuntime    │
                               │ sole mutable owner   │
                               └──────────┬───────────┘
                                          │
                          post-command observation
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 │                        │                        │
          cognitive state          semantic events          runtime activity
           latest-wins              ordered/retained         latest-wins
                 │                        │                        │
      status / TUI / web           streamed responses        busy/idle UI
```

Sensory traffic and durable evidence remain separate planes.

## 3. Startup boundary

Construction work that is inherently single-threaded remains outside the owner until the facade is ready:

1. resume/create `Symthaea`;
2. attach configured consciousness database;
3. enable experience bridge if requested;
4. perform any other startup-only configuration that requires mutable facade access;
5. choose/inject the canonical `RuntimeId`;
6. move the fully initialized facade into the runtime owner task.

After step 6, startup code must not retain another mutable facade handle.

Runtime identity is injected explicitly. It must not derive semantic order from wall-clock time. Replay/simulation may intentionally inject an identity chosen by the replay controller.

## 4. Request classification

### 4.1 Mutable cognition commands

These operations belong on the owner command lane because they may mutate the cognitive facade:

- `Query` → `Symthaea::process()`;
- `Sleep` → `Symthaea::sleep()`;
- `Save` / pause-state persistence when it requires mutable facade access;
- shutdown persistence;
- the cognition step inside `VoiceTurn`;
- the cognition step inside `VoiceTranscribe`;
- any background/autonomous cognitive cycle that currently acquires `state.symthaea.lock()`;
- future mutation commands explicitly admitted by the runtime protocol.

They are serialized by the sole owner.

### 4.2 Snapshot reads

These must **not** become owner commands merely because they currently read through the mutex:

- `Status`;
- `Introspect`;
- `StreamMetrics` / status-like metrics snapshots;
- UI/TUI cognitive state displays;
- WebSocket latest-state consumers;
- read-only values used to annotate UI such as partnership/consciousness summary.

They read a published immutable snapshot. Therefore a slow cognitive command cannot prevent the interface from reading the most recently published state.

### 4.3 Service-local operations

These remain outside cognitive ownership entirely unless their semantics later change:

- `Ping`;
- protocol metadata;
- audit-log queries;
- authentication and request-size checks;
- connection/socket/HTTP lifecycle;
- placeholder/static semantic-search helpers that do not touch mutable cognition;
- pure command classification/validation logic that needs no cognitive mutation.

### 4.4 Voice transport

Voice locks and cognition ownership must never be nested across long operations.

The intended flow is:

```text
listen / STT
   │          (voice/STT ownership only)
   ▼
semantic transcript
   │
   ▼
submit cognition command
   │          (runtime owner only)
   ▼
assistant semantic response
   │
   ▼
speak / TTS
              (voice ownership only)
```

Do not hold a voice mutex while awaiting a runtime-owner ticket. Do not hold runtime ownership while performing blocking TTS playback.

## 5. Runtime activity is not cognitive state

A state plane contains the latest completed cognitive observation. During a long command that observation may be several seconds old but still valid as the last completed state.

The UI also needs immediate knowledge that work is in progress. That is a separate control-plane value:

```text
RuntimeActivity
  Idle
  Processing {
      owner_command_seq,
      kind,
  }
  ShuttingDown
```

A command admission may publish `Processing` immediately. Completion returns activity to `Idle` after the post-command snapshot/event publication boundary.

This avoids two bad alternatives:

1. blocking status until cognition completes; or
2. fabricating a newer Phi/coherence/consciousness value just to make the UI look active.

Runtime activity sequence is control metadata, not `EventSeq`.

## 6. Post-command observation boundary

The owner must define one explicit observation point after every admitted mutating command, including error returns where the engine may have partially changed before reporting the error.

Conceptually:

```text
owner receives command N
        ↓
publish Processing(N)
        ↓
handler mutates engine
        ↓
construct immutable cognitive snapshot
        ↓
publish state snapshot
        ↓
emit semantic completion/error event(s)
        ↓
publish Idle
        ↓
resolve caller ticket
```

The exact ordering of state, semantic event, activity, and ticket completion must be frozen before external clients rely on it. The recommended v1 ordering above makes the observable runtime state current before the requesting client is told the command has completed.

If state/event publication itself fails, failure must be explicit. A command must not silently claim a fully observed completion while all observation paths failed.

## 7. Snapshot contents

The first service snapshot should contain only values already available without expensive new cognition:

- consciousness/introspection summary required by existing `Status`/`Introspect` responses;
- memory counts;
- partnership state needed by query response/UI;
- last completed experience-bridge cycle metadata reference when available;
- runtime-local counters such as completed owner commands only where clearly labeled as runtime metadata.

Avoid placing large mental-movie frame payloads or audio into this state object. Those belong to sensory/artifact planes.

The canonical internal snapshot should eventually be wrapped in `RuntimeState<T>` with authoritative `RuntimeCursor` once the owner has canonical runtime identity and semantic sequence allocation. Legacy protocol-v1 adapters may down-project that state, but must not fabricate a `Live` cursor on their own.

## 8. Semantic event allocation

`OwnerCommandSeq` and `EventSeq` are deliberately different.

```text
OwnerCommandSeq
  = bounded-owner control/admission correlation

EventSeq / RuntimeCursor
  = authoritative semantic runtime ordering
```

One owner command may emit zero, one, or many semantic events. Streaming generation is the obvious example:

```text
command 41
  ├── ResponseStarted   event 900
  ├── ResponseDelta     event 901
  ├── ResponseDelta     event 902
  └── ResponseFinished  event 903
```

Never reuse owner-command numbering as semantic-event numbering.

## 9. Background cognition

The existing background consciousness loop must not remain a second mutable owner.

Migration rule:

```text
background timer
      ↓
try/submit typed BackgroundCycle command
      ↓
canonical owner
```

If the bounded owner mailbox is full, the background policy must be explicit. It may coalesce/skip a non-critical periodic tick if the contract permits that, but it must not create a second direct mutation path to avoid waiting.

Continuous cognition is still deferred until replay/resource/budget gates are established. This migration only removes multi-owner mutation; it does not authorize a higher autonomous tick rate.

## 10. Backpressure

Foreground command admission must be bounded.

A full owner mailbox is not a reason to allocate another queue. The caller receives an explicit overload outcome and may:

- reject with a retryable busy response;
- coalesce a specifically declared background tick;
- defer at a higher layer with its own bounded policy.

Read-only state remains available during command-mailbox overload.

This is the key responsiveness property:

> command pressure can reduce command throughput without making the nervous system disappear.

## 11. Cancellation

Dropping an awaiting HTTP/socket future is not command cancellation.

Once accepted by INT-003A, a command executes unless the runtime protocol explicitly cancels it. Future cancellation therefore needs typed identity such as `TurnId` / `UtteranceId` and a cancellation registry/token observed by cancellable subsystems.

This matters especially for:

- streaming model generation;
- voice synthesis;
- long research/tool operations;
- future autonomous tasks.

Cancellation does not create or modify action authority.

## 12. Action authority boundary

The runtime owner serializes mutable cognition. It does not grant permission to execute external actions.

The lifecycle remains:

```text
cognition proposes
      ↓
deterministic policy/capability evaluation
      ↓
operator authorization when required
      ↓
exact authorized execution
      ↓
result observation
      ↓
cognition perceives result
```

Phi/Psi/coherence/arousal or any other cognitive telemetry may influence deliberation or request stronger review. They must never mint capabilities.

## 13. Concrete migration tranches

### INT-003A — owner core

Already drafted separately:

- bounded owner mailbox;
- sole mutable engine task;
- non-blocking admission;
- command ticket completion;
- no snapshot/event semantics yet.

### INT-003A1 — service command/snapshot types

Add a small service-runtime module defining:

- typed mutable command enum;
- immutable service cognitive snapshot;
- runtime activity type;
- conversion helpers from snapshot to existing service `Status`/`Introspect` wire responses.

No daemon ownership change yet.

### INT-003A2 — construct and hand off ownership

In `ServiceState::new`:

- complete database/experience-bridge initialization first;
- inject runtime identity;
- move `Symthaea` into the owner;
- store owner handle plus state/activity receivers instead of adding any second facade copy.

Keep old request paths temporarily only if needed for a parity transition; there must never be two mutable facade instances.

### INT-003A3 — migrate Query + read-only status

- `Query` submits to owner and awaits its ticket;
- `Status`, `Introspect`, and `StreamMetrics` read snapshots without entering the command mailbox;
- prove a deliberately blocked long query does not block snapshot reads;
- preserve existing response fields unless provenance requires an explicit correction.

### INT-003A4 — migrate lifecycle mutation

Move:

- Sleep;
- Save;
- shutdown persistence;
- background consciousness tick

onto typed owner commands.

Then remove every remaining direct service/background `symthaea.lock()` use.

### INT-003A5 — migrate voice cognition

- listen/transcription outside owner;
- cognition command through owner;
- speech playback outside owner;
- prove no voice lock is held while awaiting cognition;
- connect TurnId/UtteranceId cancellation when the voice-session tranche is ready.

### INT-003A6 — remove legacy mutex ownership

Exit gate:

- no `Mutex<Symthaea>` in the service daemon;
- no direct mutable facade access after runtime handoff;
- all mutation paths enumerate through the owner command protocol;
- read-only interface state does not require owner-mailbox admission.

## 14. Required negative controls

Before calling the service migration qualified, exercise at least:

### Long-think liveness

Artificially block a cognitive command for several seconds.

Expected:

- latest snapshot remains readable;
- runtime activity reports Processing;
- cancellation/control surfaces remain responsive;
- no fake new cognitive value is emitted.

### Mailbox saturation

Fill the owner mailbox while one command is blocked.

Expected:

- additional mutation admission fails explicitly;
- memory remains bounded;
- status/latest state remains readable;
- no second mutation path is created.

### Voice lock ordering

Hold/delay TTS or STT.

Expected:

- cognitive owner is not held by playback/listening;
- unrelated status/UI remains responsive.

### Background collision

Trigger a background tick while a long foreground command is running.

Expected:

- tick obeys the same bounded owner policy;
- no direct facade mutation occurs outside the owner.

### Observation failure

Force state/event publication failure after a command.

Expected:

- failure is surfaced according to the frozen completion contract;
- the runtime never silently reports a fully observed completion when observation was lost.

## 15. Non-goals

This migration does not:

- introduce autonomous continuous cognition by itself;
- merge voice/TUI/Bevy transports;
- make WebSocket telemetry the semantic event authority;
- put large sensory payloads on the state/event planes;
- replace the existing action capability model;
- equate transport command sequence with semantic `EventSeq`;
- claim that cognitive metrics establish consciousness as a scientific fact.

## 16. Exit architecture

The service should end this migration looking approximately like:

```text
socket / HTTP / TUI / voice adapters
                │
        ┌───────┴────────┐
        │                │
 mutable commands    read-only state
        │                │
        ▼                ▼
RuntimeOwnerHandle    StateReceiver
        │                │
        ▼                │
 one Symthaea owner      │
        │                │
        ├── snapshot ────┘
        ├── semantic events
        └── sensory/artifact references
```

That is the service-level interpretation of the broader interaction theorem:

> one running Symthaea, one mutable cognitive authority, many replaceable real-time interfaces.
