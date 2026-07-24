# Checkpoint Operational Trust Campaign

Schema: `symthaea.checkpoint-operational-trust-campaign.v3`

This campaign is the continuing operational promotion boundary through Series 13. It evaluates operational
trust separately from PCM and SI-observation recovery parity. A bit-identical
checkpoint waveform does not imply safe key release, auditable authorization,
serialized freshness state, or an independently monotonic trust domain.

## Frozen implementation under test

- Opaque encrypted checkpoint envelope V3.
- Independently assigned checkpoint key identifiers.
- Unix checkpoint key-agent protocol V1 with bounded replay and request-budget controls.
- Kernel `SO_PEERCRED` peer authorization on Linux/Android.
- Authenticated, hash-chained key-agent audit schema V1.
- Pinned-root checkpoint, audit, and durable rollback stores.
- Cross-process advisory locking for durable rollback updates.
- Unix external-monotonic protocol V1 for ordinary clients and V2 for attested clients.
- Operational evidence schema V3.
- Secure checkpoint-service listener lifecycle V2 with descriptor-pinned service directories on Linux.
- Authenticated audit export/retention contracts V1 plus publication-reconciliation contract V2.
- Challenge-bound monotonic service attestation V2 with canonical policy manifest and explicit key identity.
- Migration-plan schema V1 and keyed migration-approval schema V1.

Record the tested Git commit, Cargo.lock, Rust toolchain, kernel, filesystem,
mount options, `/proc` availability, key-agent executable digest, monotonic
service executable digest, and the custody domains for audit and rollback keys.

## Lane A — peer credentials plus bearer authentication

Run the key agent in a separate process under a dedicated service account.
Exercise an allowlisted peer, a peer with the wrong UID, a peer with only the
wrong GID, and a valid peer presenting a wrong bearer token.

Required gates:

- Kernel peer credentials are read from the connected socket, not supplied by
  the request body.
- A permitted UID or GID and a valid nonzero PID are required by policy.
- A bearer token remains independently required.
- Wrong UID/GID, wrong token, missing token, and unsupported peer-credential
  platforms fail before any key material is written.
- Replayed request identifiers are rejected within the configured bounded
  replay window.
- Oversized, empty, truncated, and response-ID-mismatched frames fail closed.

## Lane B — durable key-release audit

Run active-key discovery, encrypt, decrypt, legacy inventory, legacy migration,
backend denial, peer denial, replay denial, and malformed-request cases through
the audited service entry point.

Required gates:

- Every processed request produces exactly one authenticated audit record.
- Records form a continuous sequence and previous-record digest chain.
- Complete-chain verification passes after process restart.
- The raw utterance identifier does not appear in the durable log; only a keyed
  binding is stored.
- Wrong audit key, modified body, removed middle record, reordered records, and
  truncated tail fail verification.
- The audit file and lock file are mode `0600`; the directory is `0700`.
- Injected audit persistence failure prevents a key-bearing response.
- Audit file and directory synchronization complete before key release.

## Lane C — cooperating-process rollback serialization

Run at least 100,000 updates from multiple processes and multiple protector
objects sharing one rollback-state directory. Include simultaneous attempts to
commit different digests at the same sequence.

Required gates:

- Exactly one competing same-sequence writer commits.
- Every loser observes `ForkDetected`, `RollbackDetected`, or `SequenceGap` as
  appropriate; no loser silently overwrites the winner.
- Exact current-position re-observation remains idempotent.
- Process restarts preserve the committed position.
- Wrong state key and modified state bytes fail authentication.
- Root pathname replacement cannot redirect state updates.
- Interrupted temporary writes do not replace the last committed state.
- Advisory-lock interruption is retried or reported; lock failure never falls
  back to an unlocked update.

This lane earns `SameTrustDomain`. Kernel advisory locks serialize cooperating
writers but do not constrain privileged or deliberately non-cooperating writers.

## Lane D — external monotonic service

Run the Unix monotonic adapter against the actual TPM/HSM/remote append-only
service intended for deployment. The client must discover the service-reported
protection level during construction; caller-supplied level claims are not
accepted.

Required positive coverage:

- Protection-level handshake.
- Current-position lookup.
- Atomic verify-and-advance.
- Idempotent exact re-observation.
- Restart of client, service process, and application host where applicable.

Required negative controls:

- Wrong peer UID/GID and wrong bearer token.
- Replayed request ID.
- Older sequence, same-sequence fork, and sequence gap.
- Service timeout, disconnect, malformed response, and request-ID mismatch.
- A service reporting `ProcessLocal` or a level below the configured minimum.

Promotion requires `IndependentMonotonic`. The local example intentionally uses
`SameTrustDomain` and is an engineering lane, not promotion evidence.

## Lane E — descriptor-confined storage roots

On Linux, pin checkpoint, audit, and rollback roots and then repeatedly rename,
replace, and symlink their original pathnames while operations run.

Required gates:

- No operation is redirected into a replacement pathname.
- Final artifact and state opens use no-follow semantics.
- Temporary, final, audit, lock, and state files remain in the pinned directory.
- Root synchronization applies to the pinned directory descriptor.
- Missing or inaccessible `/proc/self/fd` fails closed.
- Symlinks and non-regular filesystem objects are rejected.

Non-Linux systems require a platform-specific capability-directory and peer-
credential implementation before receiving these claims.

## Lane F — migration planning and approval

Create complete V1, V2, and V3 source chains, rotate the active key, export a
portable plan, approve it with a separately held migration-approval key, and
execute only after independent review.

Required positive coverage:

- Plan serialization and checksum round trip.
- Keyed approval round trip.
- Source format, source digest, source key ID, target key, sequence, and
  predecessor continuity are rechecked at execution.
- Complete target chain opens under rebuilt predecessor digests.

Required negative controls:

- Wrong approval key.
- Modified plan bytes with unchanged tag.
- Recomputed unkeyed checksum without keyed approval.
- Stale source digest or source format.
- Active target-key change after approval.
- Existing target artifact.
- Duplicate, crossing, or source-equals-target names.
- Partial-chain migration attempt.

## Lane G — report integrity

Generate `symthaea.checkpoint-operational-evidence.v3` directly from measured
campaign results.

Required gates:

- Peer credentials and unauthorized-peer negative are separately represented.
- Audit coverage, audit-chain validity, and audit-failure behavior are separate.
- Competing writer count and winner count are recorded.
- Fork and gap negatives are separate.
- External service request coverage and reported protection level are recorded.
- Socket privacy, inode-guarded cleanup, audit export, retention refusal, service
  identity, policy digest, and boot identity are separately recorded.
- Missing required lanes are `not_exercised`, never pass.
- More than one competing rollback winner is an unconditional failure.

## Promotion rule

Promotion requires every required gate in lanes A through G, plus the existing
checkpoint confidentiality, persistence, rollback, PCM, and SI-observation
campaigns. `not_exercised` is never equivalent to pass.

## Explicit non-claims

Series 13 does not claim:

- That a bearer token alone replaces kernel peer credentials or sandboxing.
- That the client process never receives plaintext key material.
- That the keyed audit log encrypts peer UID/GID/PID, operation, key ID, or
  checkpoint sequence metadata.
- That same-filesystem rollback state defeats privileged snapshot rollback.
- That advisory locks constrain privileged or non-cooperating writers.
- That the local monotonic example provides independent monotonicity.
- That `/proc/self/fd`, `SO_PEERCRED`, or `flock` semantics are portable beyond
  the specifically tested platforms.
- That audit-key, rollback-key, bearer-token, or migration-approval-key custody
  is solved by this crate.


## Series 12–13 companion campaigns

Run `CHECKPOINT_SUPERVISION_ATTESTATION_CAMPAIGN.md` for the Series 12 socket
lifecycle, audit export/retention, service-attestation, and deterministic
publication-fault lanes. Run `CHECKPOINT_OPERATIONAL_REPAIR_CAMPAIGN.md` for
the Series 13 bounded-work, malformed-audit, rate-limit, descriptor-pin,
reconciliation, canonical-policy, and attestation-key-identity lanes.

## Series 14 companion campaign

Run `CHECKPOINT_DURABLE_REPLAY_ARCHIVE_CAMPAIGN.md` for operational-evidence V4
and the restart-durable replay, connection-admission, independent-retention, and
compaction-proof lanes. Earlier operational reports remain historical evidence
but cannot satisfy the new gates.

The finalized Series 14 lane also caps active-connection configuration, binds receipt publication to an externally supplied repository identity, validates private replay/archive filesystem objects, and races independently opened replay guards against one request identifier.

## Series 15 companion campaign

Run `CHECKPOINT_COMPACTION_CRASH_CAMPAIGN.md` for operational-evidence V5 and the
shared-admission, retention-commitment, compaction-execution, reconciliation,
and process-crash lanes. Earlier operational reports cannot infer these gates.
