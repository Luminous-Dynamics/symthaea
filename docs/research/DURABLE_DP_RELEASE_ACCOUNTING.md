# Durable Differential-Privacy Release Accounting: Reservation and Output Binding

**Status:** Research prototype for persistent privacy-budget accounting. It is not a noise generator and does not by itself establish that any output is differentially private. New tests are committed but remain unverified until exact-head CI completes.  
**Implementation:** `tools/energy_telemetry/durable_privacy_ledger.py`  
**Tests:** `tests/test_energy_telemetry_durable_privacy_ledger.py`

## Why persist the privacy budget?

The earlier `ReleasePolicyGuard` kept participant-level budget and overlap state in memory. A process restart would lose it, potentially allowing the same participant to contribute to additional research/public releases after the previous budget appeared to reset. The new `DurablePrivacyLedger` persists each reservation and cumulative participant spend in SQLite transactions.

This ledger is separate from sender delivery accounting. Network retries and repeated ACKs must not spend privacy budget twice, while a semantically new statistic or release must make a new reservation even if its values happen to resemble an earlier output.

## Reservation contract

`reserve_release` requires:
- a random 128-bit-looking release ID (32 lowercase hex characters), query-descriptor SHA-256, UTC time window, recipient class and privacy-policy identifier;
- a unique local participant-token set of 20–10,000 members under the default floor and hard resource ceiling;
- a named contribution-bound policy, an integer Wh contribution cap, adjacency definition and mechanism ID;
- epsilon and delta values within the prototype's per-release caps.

Only `approved_research` and `public_reporting` recipients can reserve through this DP ledger. Operational aggregates for trusted grid operations remain a separate, explicitly authorized flow. The 10,000-participant ceiling bounds caller-controlled iterable materialization and per-participant SQLite work; oversized input fails before participant hashing and before a reservation transaction. It is a resource limit, not a privacy threshold. Do not bypass it by splitting one intended query into smaller public releases; large-cohort support needs a reviewed bounded/streaming design and composition analysis.

Within a `BEGIN IMMEDIATE` transaction, the ledger:
1. Checks whether the release ID already exists. An exact reservation retry is idempotent and does not spend the budget again. Reusing the release ID with any different query or policy field fails closed.
2. Rejects any overlapping UTC release window for a participant already present in another reservation. This conservative policy reduces simple time-window differencing; it is not a complete defence against arbitrary correlated queries or collusion.
3. Checks each participant's remaining epsilon/delta budget.
4. Writes the reservation, participant membership tags and cumulative spend atomically.

The default cumulative per-participant budgets are epsilon 2.0 and delta 2e-6. The prototype caps a single release at epsilon 1.0 and delta 1e-6. Composition is the basic additive (sequential) sum of per-release epsilon and delta, not an advanced-composition accountant. Epsilon is stored as integer nano-epsilon units and delta as integer pico-delta units, rounding each reservation upward to avoid under-accounting caused by floating-point arithmetic.

## Participant privacy in the ledger

Raw participant tokens are not stored. The caller provides a stable, secret 32-byte-or-longer HMAC key; participant tags are HMAC-SHA-256 of a domain-separated local token. The ledger stores only these tags and a digest of the participant-tag set in each receipt. The key itself is not persisted. A derived key-check value is stored so a different key cannot silently reopen the same database and create a fresh apparent budget namespace.

Important limits:
- Tags remain linkable within this ledger, so the database still contains sensitive pseudonymous membership data.
- Losing, rotating or replacing the HMAC key without a reviewed migration must fail closed; changing the key changes participant tags and could otherwise reset budgets.
- The database file and WAL are not encrypted by this module. Use a private parent directory, restrictive file permissions and appropriate local storage/backup encryption.
- Participant-token canonicalization and mapping to actual subjects must be consistent and controlled by the caller.

## Immutable output binding

Reserving a release consumes its budget immediately, even if the process crashes before output generation. This is intentionally conservative.

After a reviewed mechanism has generated the output, and only once its declared UTC release window has ended, the release service must call `QualifiedOutputVerifier.verify_and_commit` with the exact output bytes, a canonical signed mechanism-attestation receipt and its Ed25519 signature. The verifier uses a deployment-pinned public key and deployment-pinned qualification reference, library/version, mechanism identity, feature flags, privacy unit, adjacency, contribution-policy ID, privacy-policy ID and maximum contribution cap; the request cannot nominate its own trusted key or policy.

The verifier binds the signature to the exact reservation receipt hash, query hash, UTC window, recipient, mechanism, adjacency, contribution cap, policy IDs, epsilon/delta, cohort size and output digest. It also checks the declared integer-sum sensitivity against the configured `[0, L]` contribution cap. The output bytes, canonical attestation, attestation digest, Ed25519 signature and signer fingerprint are now persisted in the same SQLite transaction. An output cannot be committed first and its attestation added later. Repeating the same attested commit is idempotent only when both the output and signed attestation bytes are identical; a changed or missing attestation on retry fails closed.

On restart, `QualifiedOutputVerifier.load_committed_output` reloads the stored output and retained signature evidence, checks the evidence digest, re-verifies the signature under the deployment-pinned trust root and revalidates all reservation bindings before returning the bytes. The raw storage/retrieval helpers are underscore-prefixed internal methods, not public release authorization methods. Python does not enforce access control on underscore methods, so deployment architecture must keep them out of ordinary release-call paths. A valid signature authenticates the attester's assertion; it does not independently prove the mechanism ran or that the attester is correct. The attester therefore needs independent mechanism/clipping verification, and the signing key and trust policy need protected provisioning, rotation/revocation and audit.

The ledger's `receipt_sha256` is a stable hash of reservation metadata and participant-set commitment, not a digital signature. Mycelix could later attach an authenticated, purpose-bound receipt if key ownership and signer policy are specified.

## What this does not prove

This code does **not**:
- generate Laplace/Gaussian noise or verify a mechanism implementation;
- prove the upstream data contributions are clipped to `contribution_limit_wh`;
- verify that the declared adjacency definition correctly describes the dataset;
- compute global sensitivity, select noise scale, verify utility or issue a formal DP theorem;
- provide secure aggregation or prevent inference from external side information;
- prove anonymity or POPIA compliance.

The `mechanism`, `adjacency`, contribution cap, contribution-policy ID, privacy-policy ID and query hash are recorded as policy context. A trusted aggregator must enforce the actual bound and a reviewed implementation must compute the correct noise for that bound/adjacency before output can be committed. A correct accountant cannot repair an incorrectly implemented DP mechanism.

## Durability and capacity

The ledger uses SQLite WAL, `synchronous=FULL`, explicit write transactions, a database application ID and a persisted configuration record. Changing the budget limits, cohort floor or HMAC key fails closed. The configured maximum reservation count includes completed and reserved entries; when it is reached, the ledger rejects new reservations instead of implicitly deleting history.

A production system still needs:
- target-filesystem crash/power-loss and backup/restore testing;
- a reviewed retention/archive design that preserves budget history and release IDs;
- authenticated authorization that binds query purpose and policy to the requesting service;
- tamper-evident or signed receipts if the database operator is in the threat model;
- an output service that enforces commit-before-publish and obtains first-time releases only through `QualifiedOutputVerifier.verify_and_commit` and restart recovery only through `QualifiedOutputVerifier.load_committed_output`;
- a migration plan for HMAC key rotation without resetting subject budgets.

## Deterministic test coverage

The test suite has 41 methods using synthetic tokens, fixed test-only keys, fixed timestamps and small synthetic output blobs. Twelve attestation-gate methods cover signature rejection, exact reservation/output binding, deployment-pinned library and qualification identity, allowed feature flags, canonical JSON/duplicate-key rejection, declared sensitivity, and absence of a public unqualified commit method. It covers restart recovery, idempotent reservation, release-ID conflicts, overlap suppression, disjoint cohorts, cumulative epsilon/delta exhaustion, cohort floor, policy-field validation, immutable output commit/replay, output digest checks, HMAC-key mismatch, record limits, concurrent duplicate reservations, extreme numeric configuration, commit timestamp ordering, bounded rejection of unbounded participant iterables and signed output-attestation verification, retained-attestation recovery after restart, stored-evidence tamper detection, malformed deployment-policy rejection, contribution caps above the deployment-qualified maximum and early rejection of oversized outputs.

These tests are added to the telemetry GitHub Actions workflow. They must not be described as passing until a completed exact-head run succeeds.
