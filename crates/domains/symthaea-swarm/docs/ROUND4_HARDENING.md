# Round-four transport hardening

Round four closes ambiguity and operator-safety gaps above the round-three Iroh
data plane.

## Duplicate-safe reliable operations

`send_reliable_idempotent` signs a caller-provided operation UUID into direct
protocol v2. The receiver reserves the operation before queue admission and
records it only after admission succeeds.

A retry with a new packet sequence but the same operation UUID receives an
accepted duplicate receipt without queueing the action again. The retained
record also contains a BLAKE3 fingerprint of the signed protocol version, lane,
and payload. Reusing an operation UUID for different bytes fails with
`OperationConflict` rather than silently treating a different command as the
same action.

Queue admission and operation-state transition occur under one local critical
section with no await point, so task cancellation cannot leave an admitted
operation permanently marked as pending. A transient `OperationInProgress`
response remains part of the wire contract for forward-compatible handlers and
should be retried with bounded backoff.

The operation window is bounded. Applications must still retain durable domain
idempotency records when effects must survive process restart.

## Enrollment persistence and endpoint rollover

`DirectEnrollmentBook` is a versioned, bounded persistence format for pinned
Iroh endpoint identities. `DirectPeerRolloverProof` requires signatures from
both the old and replacement endpoint keys, has a bounded expiry, and closes the
superseded connection when applied.

This proves key continuity. It does not replace organizational authorization,
revocation publication, or human recovery procedures.

## Real-time clock and authority safety

`RealtimeAdmissionGuard` now retains a monotonic high-water tick per world.
Freshness checks use that high-water mark even when a caller clock regresses
within its configured tolerance. Larger regressions fail closed.

Authority, revocation, and safety control frames cannot execute at a future
tick. Future controller leases can instead be staged: the old controller
remains active until the new lease's `valid_from_tick`, at which point the guard
activates the higher epoch and clears per-kind tick windows.

Expired or revoked subjects can be explicitly pruned. Subjects with staged
leases or latched safety stops cannot be silently removed.

## Readiness profiles

`TransportReadinessReport` evaluates cumulative health snapshots. The supplied
profiles distinguish:

- gossip control-plane readiness;
- pinned real-time data-plane readiness with datagram support.

Durable queue exhaustion, missing required peers, unpinned real-time admission,
invalid direct traffic, conflicting idempotency keys, or missing datagram
capability cause fail-closed readiness. Loss-tolerant drops and rate limiting
remain visible warnings.

## Deterministic fault witnesses

`fault::DeterministicFaultInjector` schedules bounded loss, delay, jitter,
duplication, reordering, and corruption from a fixed seed. It is an adapter-level
test utility and does not claim to alter Iroh internals.

Run the pure witness with:

```text
cargo run --example fault_profile_witness
```

The native direct witness now sends the same reliable operation twice and fails
unless the second send is acknowledged as a duplicate without a second remote
application event.
