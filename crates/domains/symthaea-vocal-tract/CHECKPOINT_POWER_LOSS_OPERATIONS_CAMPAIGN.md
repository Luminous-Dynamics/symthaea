# Checkpoint Power-Loss Operations Campaign

**Schema:** `symthaea.checkpoint-operational-evidence.v7`
**Series:** 17
**Status:** preregistered; no operational or physical result is claimed by this document

## Purpose

Series 16 defines what constitutes authenticated storage and sudden-power-loss evidence. Series 17 defines how a real campaign is assigned, executed, resumed, and audited without allowing a completed result to appear without an independently authenticated operational history.

This campaign is separate from the physical durability campaign. A valid Series 17 operations bundle cannot make a process-exit test count as a physical power cut, and a valid physical result cannot satisfy the Series 17 gates without its exact lease, journal, and execution receipt.

## Independent authority

The campaign freezes one nonzero `CheckpointPowerLossOperationsKeyId`. The corresponding secret is separate from:

- the storage-profile attestation key;
- the power-loss result authority key;
- checkpoint encryption keys;
- rollback and replay-state keys;
- monotonic-service attestation keys.

The operations authority authenticates leases, journal snapshots, execution receipts, concurrency negatives, and the merged operations-evidence artifact. It does not authenticate storage profiles or recovery outcomes.

## Lab manifests

At least two lab identities are preregistered. Each manifest binds:

- a nonzero lab identifier;
- organization and operator-group digests;
- the exact test-harness digest frozen by the physical campaign;
- the exact power-controller binding frozen by the physical campaign;
- a facility binding;
- a bounded validity interval.

A lab whose harness or controller binding differs from the physical campaign cannot receive a valid lease.

## Trial leases

Every planned result must have exactly one completed proof under a lease that binds:

- campaign and operations-plan digests;
- operations authority;
- trial and storage-profile identifiers;
- lab identity;
- nonzero lease identity;
- attempt number;
- issue and expiry times.

Requirements:

1. Attempts are numbered from one and bounded by the operations plan.
2. A lease may not exceed the configured duration or seven days.
3. No trial may be claimed, prepared, armed, or power-interrupted after lease expiry.
4. If a physical event was observed before expiry, recovery, classification, evidence sealing, and completion may occur after expiry.
5. An aborted attempt is terminal and requires a new lease with a higher attempt number.
6. A quarantined attempt is terminal and cannot be converted into a completed result.

## Journal state machine

The only accepted successful path is:

`Claimed → Prepared → Armed → PowerEventObserved → RecoveryStarted → RecoveryClassified → EvidenceSealed → Completed`

Allowed terminal deviations are:

- `Claimed|Prepared|Armed → Aborted`
- `PowerEventObserved|RecoveryStarted|RecoveryClassified|EvidenceSealed → Quarantined`

For physical-device trials, `RecoveryStarted` is invalid unless `PowerEventObserved` already exists in the authenticated chain.

Every entry binds:

- campaign, trial, and lease;
- zero-based sequence number;
- previous-entry digest;
- nonzero event-evidence digest;
- nonzero operator-session binding;
- monotonic observation time.

The journal is bounded to 64 entries. Terminal states cannot advance.

## Restart and concurrency lane

At least one completed trial must contain more than one operator-session binding, proving that the execution resumed after an operator process or host-session boundary without skipping state.

Every participating lab must execute a stale-writer negative with at least two competing writers. Required outcome:

- exactly one writer commits;
- the stale expected journal digest is rejected;
- the authenticated concurrency-test digest is nonzero;
- the test is recorded under that lab identity.

The durable journal store must use:

- a private effective-user-owned root;
- no-follow opens;
- descriptor-pinned resolution on Linux;
- a kernel advisory lock;
- authenticated journal snapshots;
- file synchronization before replacement;
- atomic replacement and directory synchronization;
- expected-digest compare-and-swap semantics.

## Result and artifact binding

A completed execution receipt binds:

- campaign and operations-plan digests;
- operations authority;
- exact lease and journal digests;
- exact canonical result digest;
- exact domain-separated digest of the sealed result-evidence artifact;
- lab and attempt;
- finalization time.

The journal's `EvidenceSealed` entry must contain the same sealed-artifact digest. The final evaluator must authenticate and open the same result-evidence bytes whose digest is present in every receipt and in the merged operations bundle.

## Multi-lab aggregation

Labs may produce authenticated partial operations bundles. Partial bundles may contain a strict subset of proofs and concurrency tests. They cannot pass final completeness gates.

The merger must:

- authenticate every partial bundle before parsing;
- reject duplicate trial proofs;
- reject duplicate concurrency tests for one lab;
- require one common sealed-result-artifact digest;
- deterministically order proofs by trial identifier;
- validate the complete merged bundle against every authenticated result.

## Promotion gates

Series 17 uses `CheckpointOperationalTrustRequirements::series_17_delta()` and requires:

1. verified independent operations authority;
2. at least 12 valid completed trial leases;
3. a complete monotonic journal for every authenticated result;
4. exact result and sealed-artifact receipt binding for every result;
5. at least two preregistered lab identities represented in completed proofs;
6. at least one multi-session resumed trial;
7. an authenticated stale-writer negative with exactly one commit.

Missing lanes are `not_exercised`, not passes. A partial campaign fails the lease, journal, and receipt coverage gates.

## Required negative controls

- wrong operations key;
- altered operations plan;
- unknown lab;
- harness or controller mismatch;
- zero lease ID;
- attempt zero or attempt above the configured maximum;
- lease issued outside lab validity;
- pre-cut transition after expiry;
- recovery before physical-event observation;
- skipped or repeated state;
- nonmonotonic timestamp;
- wrong previous-entry digest;
- stale expected journal digest;
- duplicate trial proof;
- duplicate lab concurrency test;
- result digest mismatch;
- result-artifact digest mismatch;
- incomplete partial bundle presented as final evidence.

## Runnable tools

- `checkpoint_series17_operator_journal` creates, advances, inspects, and resumes a durable journal using exact expected digests.
- `checkpoint_series17_operations_evaluator` opens the independently authenticated result and operations artifacts and evaluates only the Series 17 gates.

These tools do not generate a passing physical result and do not operate the power controller. Physical-event evidence remains external and must satisfy the Series 16 campaign.
