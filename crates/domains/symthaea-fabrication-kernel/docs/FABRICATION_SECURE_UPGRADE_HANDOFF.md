# Fabrication Secure Upgrade Handoff

Series 13 adds an explicit authority-transfer layer between fabrication-kernel releases. A new executable does not inherit machine, governance, or recovery authority merely because it was installed successfully.

## Why this layer exists

A fabrication gateway upgrade changes more than code. It can change:

- durable-state schemas;
- trust and quorum policies;
- expiry interpretation;
- replay identities;
- gateway and containment generations;
- recovery procedures;
- evidence-retention formats.

Without an explicit handoff, a release can be cryptographically authentic while still applying older state under newer semantics, rolling one authority domain backward, trusting a bad local clock, or deleting evidence that no longer fits the hot-state schema.

Series 13 treats upgrade authority as a capability earned from exact predecessor and successor evidence.

## Quorum-derived time

`ClockObservation` is signed by a lifecycle-authorized `ClockAuthority` key and binds:

- source identity;
- observed Unix time;
- uncertainty interval;
- clock epoch;
- signing algorithm and key identity.

`verify_clock_quorum` counts only cryptographically valid, lifecycle-eligible observations. The accepted intervals must overlap within policy, and optional algorithm diversity prevents one cryptographic implementation from becoming the only time root.

`ClockEpochTracker` rejects epoch rollback, same-epoch substitution, and consensus-time regression. The resulting `VerifiedClockWindow` is evidence, not a general claim that UTC itself is perfect.

## Cross-domain epoch seal

`AuthorityEpochVector` combines the monotonic positions of:

- trust snapshots;
- gateway membership;
- durable gateway state;
- release resilience state;
- containment state;
- transparency history;
- release lineage;
- incident history.

A successor vector must dominate every predecessor component. Advancing one generation cannot hide rollback in another domain.

## Policy migration proofs

`PolicyMigrationPlan` binds exact predecessor and successor policy identities. Every predecessor invariant must have one explicit disposition:

- `Retained`: same invariant digest remains present;
- `Strengthened`: the named invariant remains but changes identity;
- `Waived`: the invariant is removed only under an incident-bound, time-bounded waiver permitted by policy.

There is no implicit “not mentioned” state. `PolicyMigrationTracker` forms append-only per-domain lineage and blocks forks, activation-time regression, duplicate migrations, and expired waivers.

## Offline recovery authority

`RecoveryKeySet` describes an independently governed, generation-numbered offline recovery quorum. It includes distinct custodians and regions, allowed recovery scopes, validity windows, and signer thresholds.

`RecoveryActivationRequest` is bound to:

- one key-set digest and generation;
- one recovery scope;
- one target digest;
- one incident digest;
- one nonce;
- one short validity window;
- one human-readable reason.

The generic threshold ceremony must contain only signers named by the recovery key set. `RecoveryActivationTracker` prevents key-set rollback and nonce replay. This grants bounded break-glass authority; it does not make ordinary production signing keys into recovery keys.

## Evidence compaction

`EvidenceJournal` is a sequence- and time-ordered hash chain. `CompactedEvidence` removes old payloads from hot state while retaining:

- total record count;
- compacted prefix count;
- exact prefix chain head;
- a bounded tail of complete records;
- exact final chain head;
- predecessor checkpoint linkage.

The compacted tail must reconstruct the final chain head exactly. The prefix head still requires a trusted external anchor; a digest alone does not prove unavailable historical payloads.

`EvidenceCompactionAnchor` provides that external threshold-authorized checkpoint and binds count, prefix count, final head, issue time, and expiry. Its tracker rejects count rollback and same-count substitution.

## Secure handoff

`UpgradeEndpoint` binds each side of the transition to:

- software version;
- source-tree digest;
- executable digest;
- durable-state digest;
- replay-contract digest;
- authority epoch vector.

`UpgradeHandoffPlan` additionally binds:

- preparation, activation, and finalization windows;
- rollback target state;
- policy-migration set;
- clock-quorum evidence;
- compacted-evidence checkpoint;
- recovery-key-set identity;
- explicit rationale.

The successor epoch must dominate the predecessor. Every supplied migration must cover the handoff window. A threshold ceremony with purpose `upgrade-handoff` grants the capability.

## Append-only lifecycle

`UpgradeHandoffTracker` permits only:

```text
Prepared -> Activated -> Finalized
                    \-> RolledBack
Prepared/Activated  \-> Failed
```

Preparation is bound to the predecessor state, activation and finalization to the successor state, and rollback to the exact retained rollback target. Finalized, rolled-back, and failed states are terminal.

`FabricationUpgradeState` hash-links durable generations. A new handoff sequence can begin only after the previous handoff is terminal. Same-sequence handoff substitution, skipped generations, predecessor replacement, and commit-time regression fail closed.

## Replay and portable bundles

`UpgradeReplayContract` binds:

- source tree;
- handoff;
- authorized migration set and migration tracker;
- clock evidence;
- authority epoch;
- recovery key set;
- evidence compaction;
- upgrade lifecycle tracker;
- durable upgrade state.

`UpgradeEvidenceBundle` carries those artifacts in a bounded JSON envelope. Decoding verifies the replay digest and reconstructs every retained binding before returning the bundle.

## Operational rule

A production deployment should:

1. verify source and executable artifacts independently;
2. establish quorum-derived time;
3. validate all policy migrations and waiver expiry;
4. verify authority-epoch dominance;
5. anchor any compacted evidence;
6. persist `Prepared` state before replacing the running gateway;
7. activate only inside the authorized window;
8. persist `Activated` before granting machine authority;
9. finalize only after supervised observation and state verification;
10. otherwise roll back to the exact authorized target or enter terminal failure.

Series 13 does not claim that software evidence proves physical upgrade safety. Real deployments still require process supervision, compatibility tests, power-loss exercises, signer interoperability, and machine-level validation.
