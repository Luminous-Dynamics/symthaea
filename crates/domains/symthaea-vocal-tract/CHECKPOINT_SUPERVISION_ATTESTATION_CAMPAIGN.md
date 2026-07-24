# Checkpoint Supervision, Audit Export, and Attestation Campaign

Schema: `symthaea.checkpoint-supervision-attestation-campaign.v1`

This campaign is the Series 12 promotion boundary for local service lifecycle,
audit export, retention planning, monotonic-service identity, and deterministic
publication-fault handling. It supplements the existing checkpoint
confidentiality, persistence, recovery, metadata, and operational campaigns.

## Frozen implementation under test

- Secure checkpoint-agent listener lifecycle V2.
- Authenticated checkpoint-key audit export V1.
- Fail-closed audit retention decision V1.
- Monotonic service attestation V2; legacy V1 remains recognizable but cannot satisfy Series 13 identity gates.
- Operational evidence schema V3.
- Deterministic checkpoint fault points V1 plus export publication reconciliation V2.

Record the exact Git commit, Cargo.lock, Rust toolchain, target triple, kernel,
filesystem, mount options, effective UID/GID, service supervisor, socket parent
path, audit-export destination, monotonic service identity, monotonic policy
digest, attestation-key custody, and boot-identity source.

## Lane A — supervised socket lifecycle

Run both key-agent and monotonic-agent listeners through
`SecureCheckpointAgentListener` under their production service accounts.

Required positive coverage:

- The service directory is a real directory owned by the effective UID.
- Directory mode is `0700` and socket mode is `0600` before accepting clients.
- A same-owner stale Unix socket can be replaced.
- The listener records the device and inode of the socket it created.
- Normal shutdown removes only that exact socket.

Required negative controls:

- Symlink, regular file, FIFO, and foreign-owner objects at the socket path.
- Symlink or non-directory service parent.
- Injected failure after bind but before mode hardening.
- Socket pathname replacement before listener drop.

Promotion requires that an injected post-bind failure leaves no socket and that
listener drop never removes a replacement object.

## Lane B — authenticated audit export

Generate allowed, denied, malformed, unavailable-provider, and replay audit
records. Export the verified live chain to a separately retained destination.

Required gates:

- The live audit chain verifies before export.
- Export uses a distinct authentication domain from live record authentication.
- Export binds a random export ID, source-log byte count, first/last sequence,
  head digest, and complete record set.
- Export publication is no-overwrite and mode `0600`.
- Export verification succeeds with the correct audit key after process restart.
- Wrong key, modified body, modified tag, reordered record, removed record,
  changed head digest, and changed sequence bounds fail closed.
- A second export to the same final pathname is refused.

## Lane C — retention planning

Evaluate retention at thresholds immediately below and above both the record and
byte limits.

Required gates:

- Threshold evaluation verifies the live chain first.
- The decision reports live count, live bytes, and current head digest.
- Crossing either threshold requires an authenticated export.
- The crate never silently truncates the append-only log.
- `destructive_retention_permitted` remains false until an externally managed,
  separately evidenced archival/rotation protocol exists.

This lane does not claim off-host durability merely because a local export file
exists.

## Lane D — monotonic-service attestation

Run the external monotonic service with a separately provisioned attestation
key. The client preregisters service ID, policy digest, and minimum protection
level.

Required positive coverage:

- Attestation is challenge-bound to the request ID.
- The response binds schema, service ID, boot ID, policy digest, and protection
  level under a distinct keyed BLAKE3 domain.
- The boot ID is nonzero and changes across a controlled service restart.
- The expected service ID and policy digest match exactly.
- The attested protection level meets the configured minimum.

Required negative controls:

- Wrong attestation key.
- Wrong expected service ID.
- Wrong expected policy digest.
- Zero boot ID.
- Modified protection level.
- Replayed attestation under a different request ID.
- Unattested legacy protection-level response when attestation is required.

A keyed software attestation proves possession of the configured service key;
it is not a TPM quote unless the deployment attestation key is hardware-backed
and the policy digest covers the measured service image and configuration.

## Lane E — deterministic publication faults

Run `checkpoint_supervision_fault_campaign` and additional scripted schedules.

Required fault points:

- Before socket bind.
- After socket bind but before mode hardening.
- Before audit-export write.
- After export-file synchronization but before publication.
- After no-overwrite publication but before directory synchronization.

Required gates:

- Pre-publication failures leave no final export artifact.
- Temporary files are removed after handled failures.
- A post-publication/pre-directory-sync failure is reported as an indeterminate
  durability result and never silently retried to the same final pathname.
- Existing final artifacts are never overwritten.
- Fault schedules trigger only at preregistered points.

## Lane F — evidence integrity

Generate `symthaea.checkpoint-operational-evidence.v3` directly from measured
results.

Required separate gates:

- Private supervised socket mode.
- Inode-guarded socket cleanup.
- Authenticated audit-export coverage.
- Independent audit-export verification.
- Retention refusal.
- Monotonic service-identity attestation.
- Monotonic policy-digest attestation.
- Nonzero boot identity.

Missing required lanes are `not_exercised`, never pass.

## Promotion rule

Promotion requires every required gate in lanes A through F plus all existing
Series 8–11 confidentiality, recovery, rollback, key-agent, and external
monotonic-service requirements. The local campaign may lower the monotonic
minimum to `SameTrustDomain` only for engineering; production promotion still
requires an independently administered monotonic domain.

## Explicit non-claims

Series 12 does not claim:

- That socket permissions replace process sandboxing or peer authorization.
- That a local audit export is durably retained off-host.
- That audit history can yet be safely compacted or deleted by this crate.
- That keyed software attestation is equivalent to a hardware quote.
- That wall-clock time is trusted or included in service attestation.
- That handled fault injection reproduces power-loss behavior of every
  filesystem and storage device.


## Series 13 continuation

Run `CHECKPOINT_OPERATIONAL_REPAIR_CAMPAIGN.md` for bounded replay and inventory
work, malformed-request auditing, key-agent and monotonic request budgets,
descriptor-pinned service-directory behavior, indeterminate export
reconciliation, canonical policy manifests, and explicit attestation-key
identity. These are separate V3 evidence gates and are not implied by a passing
Series 12 report.
