# SQLite-Backed Telemetry Delivery Ledger: Prototype and Guardrails

**Status:** Research prototype added to Symthaea PR #7425.  
**Scope:** Durable event metadata, retry/ACK state and expiry tombstones for intermittent links.  
**Not claimed:** Production readiness, encrypted-at-rest storage, authenticated source identity, signed acknowledgements, a complete replay-defense protocol, or satellite transport interoperability.

## Why this increment

A process-local queue loses delivery and terminal state when the daemon restarts. A delayed link amplifies the consequences: the sender cannot know whether an unacknowledged batch reached the receiver, so the correct baseline is at-least-once delivery plus idempotent receiver behavior—not pretending to provide exactly-once network delivery.

The new module, tools/energy_telemetry/durable_queue.py, uses Python's standard sqlite3 module and a persistent SQLite database file. It explicitly rejects the :memory: path because each public operation opens a new connection. It is distinct from the in-memory reference policy in store_forward.py.

## Persisted state

The ledger stores envelope metadata and a payload digest, not actual sensor payloads or household readings:

- event ID and domain;
- source-local sequence number;
- observation and expiry time;
- maximum actionable age, priority and quality;
- payload digest and estimated wire size;
- state (pending/delivered/expired), attempt count, accepted time and terminal time.

The schema enforces basic state constraints, UTC timestamps are normalized to a fixed-width ISO representation, and status transitions occur inside explicit write transactions using BEGIN IMMEDIATE. A delivery ACK requires that the event was selected for delivery, has not expired, and has the expected digest. An exact repeated ACK for an already delivered event is idempotent. Terminal rows remain as replay tombstones.

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
2. **ACK authenticity:** a digest match is not a proof that the authorized receiver acknowledged delivery. ACKs need authenticated identity, replay protection and receiver-side idempotency.
3. **Payload protection:** this module stores no payload bytes and performs no encryption. The caller must encrypt transit; production should not place actual values in the digest field or mistake a digest for anonymization.
4. **Schema migration:** schema migration advances version 1 to version 2 with PRAGMA user_version; unsupported future versions fail closed. Future changes need versioned, transactional migrations and downgrade/upgrade tests.
5. **Privacy budgeting:** delivery retries must not double-charge a retransmission of an identical private release. The separate DP accountant must transactionally reserve unique release identity, participant contribution bounds, mechanism, adjacency definition, and cumulative privacy spend. It is still a separate, not-yet-production component.
6. **Power/control:** a queue ACK is never actuator authorization. Any later actuator command path needs separate credentials, local interlocks, bounded capabilities and a safety case.

## Test plan added

The accompanying test module, tests/test_energy_telemetry_durable_queue.py, uses temporary local database files and synthetic event metadata. Nineteen deterministic tests are added; they cover:
- enqueue and pending-byte recovery after creating a fresh process-like queue instance;
- re-transmission after restart and persisted attempt counts;
- ACK and duplicate ACK behavior across queue instances;
- conflicting ID reuse and persistent terminal tombstones;
- expiry persistence and rejection of ACK after expiry;
- queue byte budget and total-row fail-closed limits;
- concurrent duplicate enqueues, concurrent byte-capacity enforcement and concurrent idempotent ACKs;
- invalid inputs and SQLite integrity_check.

These tests have been added to the dedicated GitHub Actions workflow. At the time this document was authored, the exact-head CI run was queued/pending; these tests are not declared passed until that specific run concludes successfully.

## Standards and security research

- SQLite's official PRAGMA guidance explains WAL and synchronous-mode durability trade-offs: https://www.sqlite.org/pragma.html
- IETF RFC 9171 specifies Bundle Protocol Version 7 for store-carry-forward networking. This SQLite queue is an application-layer policy component, not a BPv7 implementation: https://www.rfc-editor.org/rfc/rfc9171.html
- NIST IR 8259 Rev. 1, published April 2026, updates foundational cybersecurity activities for IoT product manufacturers. It emphasizes designing security capabilities and ongoing support early in device development: https://csrc.nist.gov/pubs/ir/8259/r1/final
- NIST IR 8259A describes a baseline of device technical cybersecurity capabilities such as device identification, configuration, data protection, access control and security-state awareness: https://csrc.nist.gov/pubs/ir/8259/a/final

## Next gates before deployment

- Execute the exact-head CI suite and preserve the successful run link and commit SHA.
- Run multi-process contention and deterministic kill/reopen tests against a temporary DB; assert transaction atomicity for selected batch attempts and ACK transitions.
- Test database full, read-only path, locked DB, malformed DB, unsupported schema version, backup and restore.
- Add authenticated canonical message/ACK validation at the adapter boundary.
- Define and review a bounded terminal-record archive policy before field operation.
- Benchmark CPU, flash write amplification, energy use and queue throughput on target edge hardware.
- Keep event payloads out of Mycelix evidence records; pass only authorized, minimized receipts or aggregate envelopes.
