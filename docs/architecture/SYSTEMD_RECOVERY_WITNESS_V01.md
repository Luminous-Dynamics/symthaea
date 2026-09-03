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
8. reject/release if state changed while authority was being persisted;
9. dispatch the typed restart;
10. commit a known-applied result, or mark an uncertain result `OutcomeUnknown`;
11. persist the post-dispatch accounting state;
12. independently observe service state again;
13. if dispatch was unknown, reconcile to applied only when a changed non-empty systemd `InvocationID` proves a new invocation;
14. persist any reconciliation transition;
15. return a privacy-minimized `RecoveryReceipt`.

## Unknown outcome invariant

A transport/process result that does not prove the effect outcome is **not** equivalent to "nothing happened".

```text
OutcomeUnknown => reservation remains charged
```

No automatic retry is authorized while that reservation remains charged.

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

## Restore invariant

`SystemdRecoveryBroker::from_checkpoint` accepts a checkpoint only when:

- the checkpoint payload validates against the exact external grant;
- its computed head equals the externally supplied trusted head.

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
| already healthy service | no restart |
| revoked grant | deny before backend call |
| stale authority epoch | deny |
| caller lies about use counters | ignored; real account wins |
| checkpoint persistence fails | no restart |
| checkpoint store acknowledges wrong head | no restart |
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

A successful process exit is treated as known applied. A nonzero child exit is conservatively `OutcomeUnknown` because systemd may have partially stopped/started the unit before returning failure.

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
