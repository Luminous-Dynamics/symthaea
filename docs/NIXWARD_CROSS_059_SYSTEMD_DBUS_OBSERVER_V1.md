# Nixward CROSS-059: systemd D-Bus observer V1

## Purpose

CROSS-059 introduces an independent, read-only systemd D-Bus observation capability for governed Nixward service effects.

Its job is to collect host evidence. It does not decide whether an effect is authorized, does not execute an effect, and does not turn a semantic recommendation into authority.

The intended boundary remains:

```
Observation -> typed evidence -> post-state assessment
Authorization -> execution authorization -> typed effect
```

These paths may bind to one another through CROSS-058 receipts, but they are not interchangeable capabilities.

## Transport boundary

`NixSystemdReadOnlyObserverV1` owns the D-Bus `Connection` privately and exposes only narrowly scoped operations:

- `Manager.GetUnit` for canonical unit-object resolution.
- `org.freedesktop.DBus.Properties.GetAll` for Unit properties.
- `org.freedesktop.DBus.Properties.GetAll` for Service properties, including `Result`.
- `Properties.GetAll` on a live Job object for job identity and job type.
- Manager `JobRemoved` signal observation.
- Manager `GetUnitByInvocationID` for invocation identity re-resolution.

The observer does not expose a raw `zbus::Proxy`, connection handle, or mutation-capable wrapper.

The Cargo feature `systemd-observer` is separate from the legacy `libsystemd` C dependency. The observer therefore has a smaller transport dependency surface than the older systemd integration.

## Native systemd contracts

The observer intentionally type-checks against native D-Bus shapes rather than treating the wire values as arbitrary strings:

| Evidence | Native D-Bus contract |
| --- | --- |
| Unit `InvocationID` | `ay` |
| Manager `GetUnitByInvocationID` | `ay -> o` |
| Job `Id` | `u` |
| Job `Unit` | `(so)` |
| Job `JobType` | `s` |
| Manager `JobRemoved` | `uoss` |
| Unit `StateChangeTimestampMonotonic` | `t` |
| Unit `FragmentPath` | `s` |
| Unit `DropInPaths` | `as` |
| Service `Result` | `s` |

Unknown state vocabulary, malformed object paths, wrong D-Bus types, and identity mismatches fail closed.

## Job correlation

A `JobRemoved` signal does not contain the original JobType. Therefore the observer captures the exact live Job object first and records:

- Job.Id;
- Job.Unit;
- Job.JobType;
- Job object path;
- resolved Unit object path.

Completion then matches the `JobRemoved` tuple by:

```
(id, object_path, canonical_unit)
```

The captured JobType is carried only after that exact tuple match.

This avoids synthesizing a job type from the requested operation after completion.

## Invocation identity

Invocation identity is handled as a native 16-byte byte array.

The observer:

1. rejects IDs that are not exactly 16 bytes;
2. rejects an all-zero ID as unavailable;
3. resolves `GetUnitByInvocationID`;
4. re-reads the returned Unit object's `Id`;
5. re-reads its `InvocationID`;
6. requires both unit identity and exact invocation-byte equality.

An unavailable or inconsistent invocation ID remains unproven rather than being guessed from another field.

## Definition identity

The observer reads the systemd Unit's `FragmentPath` and `DropInPaths` and passes them to CROSS-058's definition-identity type.

That identity is intentionally a **source identity**, not a content hash and not a cryptographic attestation.

CROSS-060 remains responsible for putting the authorized definition commitment into the authority context itself.

## Monotonic observation time

Observer capture timestamps use Linux `CLOCK_MONOTONIC` through the safe `nix` API.

This is deliberately separate from wall-clock time. Wall-clock timestamps are not suitable for proving elapsed stability windows because they can jump.

systemd's `StateChangeTimestampMonotonic` is kept as systemd-supplied monotonic provenance; the observer's own capture clock is recorded separately.

## Fast-job race

The current `await_job_removed` API subscribes after the caller supplies a captured Job handle. A very fast job can therefore complete before the signal subscription is armed.

The conservative behavior is fail closed: missing completion evidence is not treated as success.

The next transport tranche is to arm a JobRemoved watcher before mutation dispatch, then bind the later Job handle to that pre-armed watcher. See CROSS-061.

## Stability claim ceiling

CROSS-059 does not by itself upgrade a post-state observation to a durable `Proven` stability claim.

CROSS-058 currently records a required monotonic window, last state-change timestamp, and sample count. It does not yet carry the full sequence of independently captured snapshots or event hashes.

The intended next closure is a repeated snapshot/event protocol in which every sample binds the same unit object, generation, definition identity, and state digest.

## Security boundary

CROSS-059 provides an observation capability, not an authority capability.

In particular it does not:

- authorize an effect;
- call systemd mutation methods;
- accept a free-form command as service semantics;
- claim atomic CAS against a concurrent external mutation;
- cryptographically attest the observer process;
- cryptographically hash unit-definition contents.

The qualification lane must therefore keep the following claims separate:

```
observed != stable != proven != authorized
```

## Qualification

The focused Nixward qualification now:

- formats the observer source;
- compiles the Nixward library;
- runs the observer's focused tests under `--features systemd-observer`;
- runs the source-boundary scanner;
- verifies the observer is present in the governed authority-file set;
- mechanically rejects observer mutation calls and execution/authorization imports.

Exact-head GitHub Actions success remains the qualification requirement. Queued, cancelled, mergeable, or static-review states are not PASS.
