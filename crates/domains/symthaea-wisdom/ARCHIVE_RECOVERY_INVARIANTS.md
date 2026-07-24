# Archive Recovery Invariants

This document records the release-blocking invariants introduced by Series VIII.

| Threat or failure | Required enforcement | Hard failure |
|---|---|---|
| Truncated or malformed archive transport | Strict canonical decoders for retention checkpoints and archive segments | Decode rejected before signature or replay |
| Archive signing-key rotation | Key-resolving verifier plus time-scoped trust authorization | Unknown, expired, retired, or revoked identity rejected |
| Segment reordering or fork | Monotonic index and predecessor fingerprint | Chain verification rejected |
| Creation-time rollback | Nondecreasing signed creation times | Chain verification rejected |
| Divergent overlapping snapshots | Sequence-indexed merge with exact record equality | Duplicate record mismatch rejected |
| Missing historical prefix | Complete-history policy requires sequence zero | Operational restoration rejected |
| Gap between archive and retained head | Signed retention overlap and contiguous sequence checks | Restoration rejected |
| Archive replay memory exhaustion | Explicit maximum reconstructed-record budget | Restoration rejected before unbounded materialization |
| Signed but false operational checkpoint | Replay archived prefix and compare state fingerprint | Checkpoint claim rejected |
| Authority checkpoint paired with another archive | Rotation bundle binds retention-checkpoint fingerprint and ledger revision | Bundle rejected |
| Archive persisted but authority artifact lost | Persist canonical complete rotation bundle before eviction | Rotation must not commit |
| State changes after startup validation | Admission permit binds revision, state, policy, trust, and sources | Admission validation rejected |
| Archive proves history but not authority | Separate authority recovery remains mandatory | Service remains inactive |
| Inherited invalid test contract | Enum-reference validation checks named variants | Verification fails before packaging |

## Mandatory rotation invariant

Historical evidence must never be evicted until all artifacts required to recover
both operational state and executable authority have been durably persisted and
re-read successfully.

Formally, for rotation head `H`:

- archive segment ledger revision = `H`;
- authority checkpoint base revision = `H`;
- authority checkpoint retention fingerprint = archive retention fingerprint;
- authority and archive operational-state fingerprints are equal;
- the predecessor fingerprint extends the current archive head;
- the canonical complete bundle is durable before live retention advances.

## Mandatory startup invariant

A runtime may accept work only when all of the following describe the same current
head:

- durable ledger revision;
- archive-backed operational replay;
- authority recovery report;
- structural ethics policy;
- trust-registry fingerprint;
- runtime-source identity set;
- startup admission permit.

Any mismatch is a denial, not a warning.

## Verification limits

Static syntax and structural checks cannot replace Rust compilation. Before merge,
the workspace must still run formatting, compilation, unit and integration tests,
Clippy, and the normal Nix verification lanes.
