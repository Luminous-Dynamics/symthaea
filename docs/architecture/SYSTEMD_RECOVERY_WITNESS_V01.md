# Systemd Recovery Witness v0.1

Status: **draft / unqualified**  
Branch: `agency/systemd-recovery-witness-v0.1`  
Stack: `#279 -> #291 -> #292 -> this tranche`

## Purpose

Prove one consequential vertical slice of bounded Symthaea agency before adding a broad integration surface.

The witness is deliberately narrow:

> Given an already-inspected unhealthy `.service` unit, exercise exactly one pre-authorized restart against exactly that host/unit/state, reserve authority before dispatch, persist the reservation before actuation, independently verify the result, and never retry an unknown external outcome automatically.

This is not a general shell, system administration API, or claim of autonomous production readiness.

## Trusted split

```text
Symthaea cognition / Nixward diagnosis
              |
              | RestartPlan (proposal only)
              v
      symthaea-authority
              |
              | exact one-use grant
              v
   symthaea-action-runtime
              |
              | reserved use/risk
              v
 symthaea-action-checkpoint
              |
              | durability acknowledgement
              v
   symthaea-system-broker
              |
              | typed RestartService
              v
        systemd / host
              |
              v
 independent ServiceObservation
```

The cognition layer does not appear in the authority decision API. There is no Phi/confidence parameter in the broker mutation path.

## v0.1 semantic surface

The broker accepts only:

- one validated `HostId`;
- one conservative ASCII `.service` `ServiceUnit`;
- read-only service observation;
- one typed `service.restart` effect.

There is intentionally no:

- `exec(String)`;
- shell string;
- arbitrary executable name;
- arbitrary `systemctl` subcommand;
- unit glob;
- `.socket`, `.mount`, `.timer`, target, slice, or scope mutation;
- configuration edit;
- package installation;
- firewall operation;
- implicit root/local authority.

The concrete `SystemctlBackend` invokes `std::process::Command` directly with fixed subcommands and the validated unit as a distinct argument. It does not invoke a shell.

## Required grant shape

The v0.1 witness intentionally rejects broad grants.

A restart grant must bind exactly:

- subject == plan actor;
- audience == exact broker executor identity;
- task == exact plan task (including `None` equality);
- one exact resource: `host://<host>/systemd/unit/<unit>.service`;
- one exact operation: `service.restart`;
- exact `RestartPlan` digest;
- exact inspected `ServiceObservation` digest;
- current authority epoch;
- `max_uses == 1`;
- at least one mutation risk unit;
- no applicable negative authority fact.

Caller-supplied `GrantUseState` is ignored. The broker evaluates the grant using the actual `GrantAccount` state.

## Mutation ordering

The intended effect-entry ordering is:

1. validate exact grant/plan bindings;
2. independently re-observe the service;
3. reject if the authorized world digest is stale;
4. reject if the service is already healthy (minimal-intervention rule);
5. reserve one use + one mutation unit in `GrantAccount`;
6. construct and persist/acknowledge a grant-bound checkpoint;
7. re-observe **again** after persistence;
8. if that pre-dispatch observation fails, release the never-dispatched reservation and persist the release;
9. if state changed while authority was being persisted, release, persist, and reject as stale;
10. dispatch the typed restart;
11. classify dispatch as `Applied`, `NotDispatched`, or `OutcomeUnknown`;
12. commit known-applied, release proven-not-dispatched, or retain an unknown reservation;
13. persist the post-dispatch accounting state;
14. independently observe service state again;
15. if dispatch was unknown, reconcile to applied only when a changed non-empty systemd `InvocationID` proves a new invocation;
16. persist any reconciliation transition;
17. return a privacy-minimized `RecoveryReceipt`.

## Three-way dispatch semantics

A failure return is not enough information to decide whether authority should remain charged.

### `Applied`

The backend established that the restart effect was accepted/applied. One use is committed.

### `NotDispatched`

The backend established that the external dispatch never occurred. The reservation may be released.

The concrete `SystemctlBackend` uses this only when process creation itself fails before `systemctl` can run.

### `OutcomeUnknown`

A process/effect may have occurred, but the outcome cannot be established. The reservation remains charged.

The concrete `SystemctlBackend` treats a non-zero `systemctl restart` child exit as `OutcomeUnknown` because systemd may already have partially stopped or started the unit before returning failure.

```text
OutcomeUnknown => reservation remains charged
NotDispatched  => reservation may be released
```

No automatic retry is authorized while an unknown reservation remains charged.

A healthy post-observation may establish that the service is currently healthy. It does not by itself prove that the uncertain restart caused that state. For uncertain dispatch, v0.1 only upgrades effect accounting to `ReconciledApplied` when the systemd `InvocationID` changed.

The witness has no automatic `reconcile_not_applied` path because ordinary service observation generally cannot prove that no restart effect occurred.

## Checkpoint ordering invariant

No mutation is dispatched before the reservation checkpoint has been acknowledged by the supplied `CheckpointStore`.

The trait boundary does **not** prove what "durable" means. A production store must define it explicitly, e.g.:

- local fsync + independently retained checkpoint head;
- Xenia append-only authority/evidence ledger;
- TPM-backed monotonic/anchor state;
- supervisor-owned append-only log;
- authenticated remote witness.

A checkpoint store that lies about durability invalidates crash-safety claims but does not widen the semantic grant itself.

## Latching persistence containment

Checkpoint uncertainty is not treated as an ordinary recoverable application error.

If checkpoint construction, checkpoint hashing, persistence, or acknowledged-head verification fails, the broker enters a latching `contained` state. Further mutation attempts on that broker instance fail with `ContainmentRequired`.

```text
checkpoint uncertainty
        => containment
        => no further mutation
        => reconstruct from externally trusted state
```

This prevents a dangerous pattern in which a caller repeatedly retries after a persistence failure while being unsure which authority state is actually durable.

A failed initial persistence happens before dispatch, so the in-memory reservation may be released for local accounting. That release is **not** treated as a durable fact when persistence itself is uncertain; the broker remains contained and cannot use the released capacity until reconstructed from an externally trusted checkpoint/head.

## Restore invariant

`SystemdRecoveryBroker::from_checkpoint` accepts a checkpoint only when:

- the checkpoint payload validates against the exact external grant;
- its computed head equals the externally supplied trusted head.

Successful trusted reconstruction is a new broker instance and clears the runtime containment latch because authority state has been re-established from explicit trusted evidence.

Rolling back both the checkpoint and the external trusted head remains outside the protection of a local hash chain. The head must therefore live in an independently protected authority domain for strong anti-rollback claims.

## Durable receipt surface

`RecoveryReceipt` retains:

- execution/reservation IDs;
- exact grant commitment;
- exact plan commitment;
- before/after world commitments;
- effect-accounting outcome;
- independent health-verification result;
- checkpoint head;
- committed/reserved use accounting.

It does not retain journal text, environment variables, shell output, raw stderr, secrets, or arbitrary command strings. Backend diagnostics are represented by domain-separated digests.

## Adversarial qualification matrix

The focused deterministic suite must cover at least:

| Attack/failure | Required result |
|---|---|
| wrong subject | deny before backend call |
| wrong executor/audience | deny before backend call |
| wrong task | deny before backend call |
| second service added to grant | deny as broad scope |
| second operation added to grant | deny as broad scope |
| plan digest substitution | deny |
| world digest substitution | deny |
| stale world before reservation | deny/no charge |
| state changes during checkpoint persistence | release/no dispatch |
| pre-dispatch observation fails after reservation | release/no dispatch |
| already healthy service | no restart |
| revoked grant | deny before backend call |
| stale authority epoch | deny |
| caller lies about use counters | ignored; real account wins |
| checkpoint persistence fails | no restart + containment latch |
| checkpoint store acknowledges wrong head | no restart + containment latch |
| attempt after persistence containment | deny `ContainmentRequired` |
| trusted-head mismatch on restore | deny restore |
| backend proves process never spawned | `NotDispatched`; release capacity |
| restart known applied | commit one use |
| restart response unknown + InvocationID changes | reconcile applied |
| restart response unknown + InvocationID unchanged | remain reserved/unknown |
| second attempt after commit | deny use exhausted |
| second attempt while unknown | deny use exhausted |
| `.socket` / path / whitespace / shell-like unit | parser rejects |
| malicious journal text | not an authority input to this crate |
| Phi/confidence = any value | impossible: no such authority input |
| generic shell fallback | impossible through this crate's public effect surface |

## Production backend boundary

`SystemctlBackend` is a local implementation only. It binds itself to one exact `HostId` and queries:

- `ActiveState`;
- `SubState`;
- `InvocationID`.

It executes only typed `systemctl restart -- <validated>.service`.

A successful process exit is treated as known applied. Failure to spawn `systemctl` is `NotDispatched`. A non-zero child exit is conservatively `OutcomeUnknown` because systemd may have partially stopped/started the unit before returning failure.

## Non-claims

v0.1 does not establish:

- journal diagnosis correctness;
- prompt-injection resistance of an upstream language model;
- cryptographic grant signatures (the authority crate is semantic/crypto-independent);
- authenticated workload identity;
- production checkpoint durability;
- TPM/measured-boot attestation;
- kernel/root compromise resistance;
- D-Bus policy hardening;
- OS sandbox compilation (Landlock/seccomp/systemd sandboxing);
- physical-host deployment qualification;
- that `InvocationID` proves user intent;
- that current health proves causal success of an uncertain restart;
- general NixOS mutation authority.

## Next exit gate

Do not broaden to MCP/GitHub/general app mutation from this witness until:

1. `symthaea-authority`, action runtime, and checkpoint stack have compiler/test evidence;
2. this crate passes format/check/Clippy/tests on the exact stacked head;
3. the hostile matrix above is represented by deterministic tests;
4. a concrete checkpoint store defines its durability boundary;
5. a Nixward observer can create the before-snapshot/plan without journal content being able to influence authority;
6. Xenia (or another authenticated authority layer) can bind a real approval signature to the exact grant/plan commitments.

Only then should the same Agency Kernel semantics be reused for broader system mutations.
