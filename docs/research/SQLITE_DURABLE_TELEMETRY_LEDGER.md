# SQLite-Backed Telemetry Delivery Ledger: Prototype and Guardrails

**Status:** Research prototype added to Symthaea PR #7425.  
**Scope:** Durable event metadata, retry/ACK state and expiry tombstones for intermittent links.  
**Not claimed:** Production readiness, encrypted-at-rest storage, authenticated source identity, signed acknowledgements, a complete replay-defense protocol, or satellite transport interoperability.

## Why this increment

A process-local queue loses delivery and terminal state when the daemon restarts. A delayed link amplifies the consequences: the sender cannot know whether an unacknowledged batch reached the receiver, so the correct baseline is at-least-once delivery plus idempotent receiver behavior—not pretending to provide exactly-once network delivery.

The new module, tools/energy_telemetry/durable_queue.py, uses Python's standard sqlite3 module and a persistent SQLite database file. It explicitly rejects the :memory: path because each public operation opens a new connection. It is distinct from the in-memory reference policy in store_forward.py.

## Persisted state

The ledger stores event metadata and an envelope digest. Metadata-only enqueue rows contain no application payload. The protected enqueue path stores the complete canonical signed envelope BLOB, where event metadata and application payload are AEAD-encrypted by the envelope creator—not plaintext sensor values:

- event ID and domain;
- source-local sequence number;
- observation and expiry time;
- maximum actionable age, priority and quality;
- payload digest and estimated wire size;
- state (pending/delivered/expired), attempt count, accepted time and terminal time.

The schema enforces basic state constraints, UTC timestamps are normalized to a fixed-width ISO representation, and status transitions occur inside explicit write transactions using BEGIN IMMEDIATE. The database is tagged with a SQLite application_id; an existing database with a foreign application ID or unrelated unowned tables is rejected before switching its journal mode or creating telemetry tables. After ownership is verified, the prototype restricts the main database and existing WAL/SHM sidecars to POSIX mode 0600 where supported; the parent directory must still be private to prevent exposure before sidecar chmod and to protect backups. Queue byte/record/time-skew policy is persisted and mismatched instances fail closed. Metadata-only delivery can use the prototype's unsigned digest ACK. Protected-envelope delivery requires `acknowledge_protected`, which verifies a receiver-signed Ed25519 ACK, requires the ACK's exact event/digest binding and rejects acceptance timestamps outside allowed clock skew or at/after expiry. An exact repeated valid ACK for an already delivered protected event is idempotent. Terminal rows remain as replay tombstones.

## Durability posture

The prototype configures WAL journaling and synchronous=FULL. SQLite documents FULL synchronous mode as using the VFS sync operation before continuing, and describes FULL in WAL mode as providing ACID durability, subject to the truthful behavior of the filesystem, storage device and synchronization layer: https://www.sqlite.org/pragma.html

This is a reasoned local persistence configuration, not proof against all hardware faults, corrupted filesystems, buggy storage firmware, malicious modification or loss of the storage device. The project needs deployment tests on its actual target filesystem and power-loss/reboot recovery tests before relying on it.

## Fail-closed capacity policy

- A configurable pending-byte budget prevents the queue from silently evicting pending events.
- A configurable total-record ceiling includes terminal rows. When that ceiling is hit, insertion fails closed. The prototype does **not** silently delete replay tombstones to reclaim storage.
- Production requires an explicit archival/retention design: define a trusted event identity and authentication lifetime, the maximum accepted observation age, how tombstones move to durable archival storage, and what proof remains after pruning. Removing tombstones without such a policy can reopen event-replay ambiguity.
- Database files and WAL/SHM sidecars may contain operationally sensitive timing and event metadata. Production must use restrictive file permissions, device-level or application-level storage encryption as appropriate, safe backup/restore handling and secure disposal.

## Security boundaries that remain outside the prototype

1. **Authenticity:** the queue validates shapes and state only. The network adapter must authenticate the sender and bind a canonical envelope (including the metadata, digest, expiry and policy/version context) to a vetted signature or MAC.
2. **ACK authenticity:** protected queue records must use `acknowledge_protected`, which verifies a canonical Ed25519 receiver-signed positive ACK against an explicitly supplied trusted-key map. The legacy unsigned ACK path refuses protected records. The new `ReceiverInbox` commits exact ciphertext and the event-ID/digest replay record before generating its signed ACK, so a retry after a commit-but-lost-ACK is idempotent. This prototype-level receiver flow is not yet proof of target-filesystem crash/power-loss behavior or downstream processing transaction atomicity.
3. **Payload protection:** the protected API persists AEAD-encrypted envelope bytes, validates their SHA-256 digest/size binding, and replays the exact bytes. The queue itself does not encrypt or verify signatures; the caller must use the envelope adapter and the receiver must verify. The database is not encrypted by this module, so event metadata remains readable at rest.
4. **Schema migration:** schema migration advances version 1/2 to version 3 with PRAGMA user_version and adds an optional `envelope_bytes` BLOB; unsupported future versions fail closed. Future changes need versioned, transactional migrations and downgrade/upgrade tests.
5. **Privacy budgeting:** delivery retries must not double-charge an identical private release. The separate `DurablePrivacyLedger` prototype now transactionally reserves unique release identity, HMAC-tagged participant membership, mechanism, adjacency, declared contribution cap/policy, and cumulative epsilon/delta spend. It does not generate noise or prove upstream clipping; see [DURABLE_DP_RELEASE_ACCOUNTING.md](DURABLE_DP_RELEASE_ACCOUNTING.md).
6. **Power/control:** a queue ACK is never actuator authorization. Any later actuator command path needs separate credentials, local interlocks, bounded capabilities and a safety case.

## Test plan added

The accompanying test module, tests/test_energy_telemetry_durable_queue.py, uses temporary local database files, test-only keys and synthetic event metadata. Thirty-one deterministic tests are added; they cover:
- enqueue and pending-byte recovery after creating a fresh process-like queue instance;
- re-transmission after restart and persisted attempt counts;
- ACK and duplicate ACK behavior across queue instances;
- conflicting ID reuse and persistent terminal tombstones;
- expiry persistence and rejection of ACK after expiry;
- queue byte budget and total-row fail-closed limits;
- concurrent duplicate enqueues, concurrent byte-capacity enforcement and concurrent idempotent ACKs;
- invalid inputs and SQLite integrity_check;
- protected envelope persistence/reopen, exact-byte retransmission, hash/size mismatch rejection and separation of metadata-only rows from protected-envelope delivery;
- requirement for receiver-signed ACKs on protected rows, rejection of forged/unknown/future-dated ACKs, and rejection of a signed ACK for the wrong envelope digest.

The companion `tests/test_energy_telemetry_authenticated_envelope.py` has 22 test cases, `tests/test_energy_telemetry_receiver_inbox.py` adds 17 receiver-inbox cases, and `tests/test_energy_telemetry_pipeline_integration.py` adds 3 end-to-end tests for normal signed delivery, a lost ACK with idempotent retry, and a tampered transmission. These and the queue tests have been added to the dedicated GitHub Actions workflow. At the time this document was authored, the exact-head CI run was queued/pending; these tests are not declared passed until that specific run concludes successfully.

## Standards and security research

- SQLite's official PRAGMA guidance explains WAL and synchronous-mode durability trade-offs: https://www.sqlite.org/pragma.html
- IETF RFC 9171 specifies Bundle Protocol Version 7 for store-carry-forward networking. This SQLite queue is an application-layer policy component, not a BPv7 implementation: https://www.rfc-editor.org/rfc/rfc9171.html
- NIST IR 8259 Rev. 1, published April 2026, updates foundational cybersecurity activities for IoT product manufacturers. It emphasizes designing security capabilities and ongoing support early in device development: https://csrc.nist.gov/pubs/ir/8259/r1/final
- NIST IR 8259A describes a baseline of device technical cybersecurity capabilities such as device identification, configuration, data protection, access control and security-state awareness: https://csrc.nist.gov/pubs/ir/8259/a/final

## Next gates before deployment

- Execute the exact-head CI suite and preserve the successful run link and commit SHA.
- Run multi-process contention and deterministic kill/reopen tests against a temporary DB; assert transaction atomicity for selected batch attempts and ACK transitions.
- Test database full, read-only path, locked DB, malformed DB, unsupported schema version, backup and restore.
- Validate sender queue, receiver inbox and DurablePrivacyLedger with exact-head CI, target-filesystem crash/power-loss and backup/restore tests; bind key IDs to authorized identities; and separately verify the DP output generator and contribution clipping.
- Define and review a bounded terminal-record archive policy before field operation.
- Benchmark CPU, flash write amplification, energy use and queue throughput on target edge hardware.
- Keep event payloads out of Mycelix evidence records; pass only authorized, minimized receipts or aggregate envelopes.
