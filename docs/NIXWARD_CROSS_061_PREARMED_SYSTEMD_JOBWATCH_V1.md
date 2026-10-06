# Nixward CROSS-061 — Pre-Armed systemd JobRemoved Watcher V1

## Purpose

CROSS-061 removes an avoidable evidence-loss race in the systemd lifecycle path.

The governed sequence is:

```
watcher armed
  -> lifecycle effect dispatched
  -> exact Job object path returned
  -> exact live Job identity captured
  -> exact JobRemoved terminal signal observed
  -> post-state sampled
```

The watcher is an observation capability only. It does not authorize, dispatch, or mint an execution decision.

## Arming contract

`NixSystemdReadOnlyObserverV1::arm_job_removed_watcher()`:

1. captures the current unique D-Bus owner of `org.freedesktop.systemd1`;
2. creates a signal stream for the exact Manager object/interface and `JobRemoved` member;
3. relies on zbus's registered signal match rule before returning the watcher;
4. re-reads the manager owner and rejects an owner transition during arming.

zbus's signal stream registration is therefore completed before the caller receives the watcher. The watcher is the object that crosses the pre-dispatch boundary.

## One-shot and identity binding

`NixSystemdJobRemovedWatcherV1` owns its stream and is consumed by `await_job_removed()`.

It contains no public stream/proxy escape hatch and cannot be cloned. Consuming it prevents accidental reuse across unrelated jobs.

The watcher records the systemd manager's unique owner as an observation epoch. The exact live `NixSystemdJobHandleV1` must carry the same owner, otherwise the wait is rejected.

## Terminal signal qualification

The accepted signal must satisfy all of:

- sender equals the captured systemd manager unique owner;
- native `JobRemoved` body decodes as `uoss`;
- job ID equals the captured Job ID;
- job object path equals the captured Job object path;
- canonical unit equals the captured unit.

Mismatched signals are ignored. Timeout, stream termination, malformed bodies, sender mismatch, and watcher/Job owner mismatch are fail-closed errors.

## Residual ceiling

Pre-arming removes the original race in which a sufficiently fast job could finish between Job capture and watcher subscription.

It does **not** guarantee that a Job object remains readable long enough to capture `JobType` after dispatch. A very short-lived job may still disappear before the post-dispatch `capture_job()` call. In that case the governed protocol remains conservative: the exact JobRemoved tuple can be present in the pre-armed stream, but the required live JobType provenance is unavailable, so the caller must not upgrade the effect to a stronger claim.

This is deliberate. CROSS-061 is not a claim of atomic execution or CAS semantics.

## Relationship to later tranches

- CROSS-060 binds service-definition identity into authorization.
- CROSS-062 replaces scalar stability counts with an actual repeated-observation sequence.
- CROSS-063 carries the systemd manager incarnation into durable job evidence/receipts.
- CROSS-064 supplies the typed native lifecycle mutation transport needed to dispatch without shell parsing.

## Qualification expectation

The implementation must remain subject to exact-head workflow evidence. Queued, pending, cancelled, mergeable, or static analysis states are not PASS evidence.
