# Checkpoint Operational Repair Campaign

Schema: `symthaea.checkpoint-operational-repair-campaign.v1`

This campaign is the Series 13 promotion boundary for bounded protocol work,
auditable malformed requests, request-rate enforcement, descriptor-pinned
service directories, reconciled audit-export publication, and canonical
monotonic-service policy and key identity. It supplements rather than replaces
the confidentiality, persistence, recovery, rollback, operational-trust, and
supervision/attestation campaigns.

## Frozen implementation under test

- Checkpoint key-agent protocol V1 with bounded O(1) replay membership.
- External monotonic protocol V1/V2 with bounded O(1) replay membership.
- Bounded key-inventory response contract.
- Durable malformed-request key-agent audit.
- Fixed-window request budgets for both local service protocols.
- Descriptor-pinned supervised socket directory lifecycle V2.
- Audit-export durability and reconciliation contract V2.
- Canonical monotonic policy-manifest schema V1.
- Explicit monotonic attestation-key identity schema V2.
- Operational evidence schema V3.

Record the exact Git commit, Cargo.lock, Rust toolchain, kernel, filesystem,
mount options, effective UID/GID, service supervisor, service executable/image
digest, runtime configuration digest, peer-policy digest, rollback-policy
digest, protocol versions, rate-limit settings, replay-window settings, and
attestation-key custody.

## Lane A — bounded replay and inventory work

Exercise both Unix protocols with a full replay window, repeated identifiers,
all-zero identifiers, and maximum-size key inventories.

Required gates:

- Replay membership remains bounded and does not linearly scan the full replay
  history for each request.
- The oldest identifier is evicted only after the configured capacity is
  exceeded.
- An identifier still inside the window is rejected as replay.
- The all-zero request identifier is rejected by both protocols.
- Key-inventory responses at the limit succeed.
- The server rejects key inventories above `MAX_CHECKPOINT_KEY_CANDIDATES` before
  serializing them, and the client independently rejects an over-limit list inside
  the already bounded protocol frame.

Record request latency distributions at empty, half-full, and full windows. The
campaign does not claim constant wall-clock time across operating systems, but
must show no replay-window-length-proportional growth.

## Lane B — malformed-request audit

Send empty, truncated, oversized, length-mismatched, and undecodable key-agent
frames through the audited service entry point.

Required gates:

- Every malformed request produces exactly one durable `InvalidRequest` audit
  record before the connection is closed.
- The record carries kernel-derived peer identity where available.
- A nonzero request binding is deterministically derived from the malformed
  bytes without treating attacker-supplied fields as trusted metadata.
- No key identifier, plaintext key, or successful operation is fabricated.
- Audit persistence failure remains fail closed.
- Malformed requests never produce key-bearing responses.

## Lane C — service request budgets

Configure small deterministic fixed-window budgets for both the key agent and
monotonic service. Exercise allowed requests up to the limit and at least two
requests beyond it.

Required gates:

- Requests inside the configured budget retain their original semantics.
- The first over-budget request is rejected as `RateLimited`.
- Key-agent rate-limit decisions are durably distinguishable from authorization
  denials.
- No provider key lookup, checkpoint mutation, or monotonic advance occurs after
  rate-limit rejection.
- The next window resets only according to the configured monotonic elapsed-time
  source; wall-clock rollback must not create an extra budget.
- A zero request budget and zero window duration are rejected at construction.

Fixed-window limiting is a local availability control, not a distributed denial-
of-service defense and not a replacement for supervisor-level connection limits.

## Lane D — descriptor-pinned supervised sockets

Create a supervised listener, rename or replace its original service-directory
pathname, and continue identity checks and shutdown cleanup.

Required gates:

- The listener pins the originally opened service directory.
- Socket identity checks resolve through the pinned directory capability on
  Linux.
- Replacing the original directory pathname cannot redirect cleanup.
- Shutdown removes only the device/inode pair originally created.
- A replacement socket, file, symlink, or directory is never removed.
- Missing descriptor-relative support fails closed rather than falling back to
  an unpinned pathname.

## Lane E — audit-export durability reconciliation

Inject failure after no-overwrite publication but before directory
synchronization.

Required gates:

- The export call returns an explicit indeterminate-durability receipt rather
  than a generic failure.
- The receipt includes the expected artifact digest.
- Publication, verification, and reconciliation pin the export parent directory
  for the duration of each operation on Linux.
- `reconcile_export` opens the final artifact with no-follow semantics and
  requires the exact digest.
- A matching final artifact reconciles to a verified published export.
- Missing, replaced, truncated, or modified artifacts fail reconciliation.
- Retrying publication to the same name remains forbidden.

This lane demonstrates deterministic handled-fault semantics. It does not claim
survival across real device power loss until filesystem-specific testing is run.

## Lane F — canonical policy and attestation-key identity

Construct the monotonic service attestor and client policy from the same
validated `CheckpointMonotonicPolicyManifest`.

The manifest must bind:

- Service executable or measured-image digest.
- Runtime configuration digest.
- Peer-authorization policy digest.
- Rollback-state policy digest.
- Monotonic protocol version.
- Claimed protection level.

Required gates:

- Canonical encoding is deterministic.
- Any manifest field change changes the policy digest.
- Attestor and verifier constructors derive the same digest and minimum level
  from the manifest.
- The proof binds an explicit nonzero attestation-key identifier.
- Wrong key ID, wrong proof key, wrong manifest, downgraded protection level,
  zero boot identity, or changed request challenge fails closed.
- Independently provisioned key IDs are recorded separately from secret key
  bytes; compatibility-derived identifiers are identified as legacy behavior.

A software manifest and keyed proof do not become a hardware quote merely by
using an image digest. Hardware-rooted measurement remains a deployment claim.

## Lane G — operational evidence integrity

Generate `symthaea.checkpoint-operational-evidence.v3` directly from measured
results.

Required separate gates:

- Pinned supervised service directory.
- Durable malformed-request audit.
- Key-agent rate-limit coverage and rejection.
- Monotonic-service rate-limit coverage and rejection.
- Indeterminate audit-export publication coverage and successful reconciliation.
- Canonical monotonic policy-manifest verification.
- Explicit monotonic attestation-key identifier verification.

Older V1 and V2 reports remain recognizable but cannot satisfy Series 13 gates.
Missing required lanes are `not_exercised`, never pass.

## Promotion rule

Promotion requires every required gate in lanes A through G plus all existing
checkpoint confidentiality, persistence, recovery, rollback, operational-trust,
and supervision/attestation gates. Engineering runs may use a same-domain
monotonic service, but production rollback promotion still requires an
independently administered monotonic domain.

## Explicit non-claims

Series 13 does not claim:

- Hardware-rooted measured boot or TPM/HSM attestation.
- Distributed or per-principal denial-of-service protection.
- Trusted wall-clock time.
- Safe destructive audit compaction.
- Real power-loss survival on untested filesystems and devices.
- That bounded in-memory replay state survives service restart.
- That descriptor-relative Linux behavior is portable to every Unix platform.
- That compatibility-derived attestation-key identifiers are unlinkable.

## Series 14 companion campaign

Run `CHECKPOINT_DURABLE_REPLAY_ARCHIVE_CAMPAIGN.md` for restart-durable replay,
active-connection admission, replay publication faults, independently keyed
audit archive receipts, and exact-live-head compaction proofs. Series 13
in-memory replay and request-rate results do not satisfy those gates.

The finalized Series 14 lane also caps active-connection configuration, binds receipt publication to an externally supplied repository identity, validates private replay/archive filesystem objects, and races independently opened replay guards against one request identifier.

## Series 15 companion campaign

Run `CHECKPOINT_COMPACTION_CRASH_CAMPAIGN.md` for the multi-process admission and
retention-bound audit-compaction claims. Series 13 bounded-work and descriptor
repair results remain prerequisites but do not establish crash-safe compaction.
