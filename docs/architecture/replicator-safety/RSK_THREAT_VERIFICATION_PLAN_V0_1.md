# Replicator Safety Kernel — Threat Verification Plan v0.1

Status: **normative verification plan; authored design evidence, not executed evidence**

This document maps named RSK threats to deterministic, black-box, property, fault, formal, cryptographic, and governance evidence families.

It contains no physical replication mechanism.

---

## 1. Verification rule

A threat is not considered mitigated because a control is documented.

For every applicable threat, production admission requires one or more of:

- executable negative tests on the exact candidate implementation;
- generated property/state-machine evidence;
- fault injection;
- cryptographic/trust golden and negative vectors;
- formal analysis within explicitly documented bounds;
- externally verified governance/runtime evidence;
- explicit residual-risk acceptance where the threat is outside the ordinary guarantee.

Queued jobs and authored tests are not passing evidence.

---

## 2. Verification-family IDs

### VF-ACT — exact action/authority binding

Targets: T05, T06, T07, T10, T24.

Core assertions:

- action committed equals action evaluated/authorized;
- wrong subject/lineage/resource/capability/command target denies;
- failed mismatch leaves authoritative state unchanged;
- mutation ID is not consumed by a pre-mutation mismatch;
- stale cursor/head cannot commit.

Current candidate coverage:

- #1724 authors exact resource substitution tests;
- child/derivation commitment and governance-command binding remain open.

### VF-CAP — capability and ancestral budget attenuation

Targets: T08, T09.

Core assertions:

- descendant capabilities never expand above applicable ancestor/output-lineage ceiling;
- ancestor subject and lineage budget scopes survive descendant grants/generations;
- branch policy cannot widen ceilings or downgrade risk;
- restart/replay cannot reset consumption.

Reference-state representability: partial now; durable restart/rollback requires P7/P8.

### VF-QUORUM — policy/quorum integrity

Targets: T02, T03, T04, T17.

Core assertions:

- requester `required=0` cannot lower kernel/policy floor;
- policy can raise the floor;
- duplicate signer identity counts once/rejects;
- same required failure domain cannot satisfy independent quorum;
- stale/revoked/wrong-role signer rejects.

Current candidate coverage:

- #1728 authors semantic floor/self-restriction tests;
- identity/failure-domain/lifecycle verification requires #1668/#1673.

### VF-TIME — temporal integrity and continuity

Targets: T11, T12, T13, T23, T29.

Core assertions:

- commit before evaluation denies with no mutation;
- commit at evaluation boundary is semantically valid if all other facts pass;
- commit at/after expiry denies;
- rollback/jump/disagreement/continuity loss freezes authority;
- outage never extends snapshot validity.

Current candidate coverage:

- #1726 authors semantic half-open interval tests;
- trusted source/continuity remains #1669.

### VF-MON — runtime assurance integrity

Targets: T14, T15, T16, T29.

Core assertions:

- forged raw witness cannot enter production evaluator;
- stale/wrong-envelope/wrong-safety-case witness denies;
- configured independent negative path survives one monitor compromise;
- committed negative state dominates fully positive evidence;
- monitor/DKG outage freezes rather than widens authority.

Reference-state representability: negative precedence partially representable; verified witness/common-mode failure requires #1668.

### VF-LEDGER — canonical durable evidence and rollback resistance

Targets: T10, T18, T19, T21, T28.

Core assertions:

- canonical golden bytes stable;
- every authority-relevant field changes record/state digest;
- delete/reorder/substitute/predecessor mutation rejects;
- older valid checkpoint cannot silently restore authority;
- crash-before/after-append retries are exactly-once semantically;
- malformed/oversized decoder inputs fail within strict bounds.

Requires P7/P8 implementation from #1241 contract.

### VF-FORK — equivocation containment and recovery

Targets: T18, T20, T25, T26, T30.

Core assertions:

- same predecessor + distinct successors => forked/non-operational;
- no timestamp/last-write-wins/autonomy-selected winner;
- both branches retained as incident evidence;
- recovery requires external role/quorum and exact old-terminal binding;
- new epoch carries no active grant automatically.

Requires P9/P6 implementation.

### VF-TRANS — transparency/Mycelix non-authority

Targets: T20, T22, T23, T29.

Core assertions:

- valid receipt plus invalid local state still denies;
- receipt proves inclusion/provenance only;
- contradictory valid receipts surface fork;
- DKG unavailable before expiry consumes remaining lifetime only;
- DKG unavailable at/after expiry denies new authority.

Requires P10 integration; hard non-authority rule is already normative.

### VF-BUILD — admitted build/runtime identity

Targets: T27.

Core assertions:

- evidence capsule names exact source commit, lockfiles, toolchain/Nix environment, features/configuration, artifact digest, policy/schema versions;
- runtime identity matches admitted artifact/profile;
- changed artifact/config cannot reuse old admission evidence.

Requires #1682.

### VF-GOV — governance/recovery authorization

Targets: T24, T25, T30.

Core assertions:

- role matrix rejects unauthorized mutation kind/target/scope;
- grant/operator/revoker credentials cannot recover unless separately authorized;
- recovery threshold and trust-root changes are Class A governed;
- catastrophic threshold compromise appears in residual-risk record.

Requires #1670/P12.

---

## 3. Threat-by-threat candidate tests

| Test ID | Threat | Candidate executable check | Earliest layer |
|---|---|---|---|
| TV01 | T01 | unverified/raw grant cannot reach production evaluator | verified evidence (#1668) |
| TV02 | T02 | requester quorum requirement zero cannot lower R3 floor | pure authority core / #1728 |
| TV03 | T03 | duplicate signer identity cannot inflate quorum | verified quorum (#1668) |
| TV04 | T04 | two signers in same required failure domain insufficient | verified quorum/policy |
| TV05 | T05 | grant subject/lineage substitution denies | pure/verified authority |
| TV06 | T06 | lower/higher resource substitution denies with no mutation | ledger / #1724 |
| TV07 | T07 | child/output commitment substitution denies | exact-action layer |
| TV08 | T08 | descendant capability re-expansion denied | ledger/reference |
| TV09 | T09 | new grant/branch cannot escape ancestor budget | ledger/reference + durable replay |
| TV10 | T10 | stale token after intervening mutation denies | ledger/reference; durable head later |
| TV11 | T11 | revoked/expired signer evidence denies | verified evidence/time |
| TV12 | T12 | commit before evaluation denies/no mutation | ledger / #1726 |
| TV13 | T13 | time disagreement/forward jump freezes | trusted time (#1669) |
| TV14 | T14 | caller-constructed runtime witness rejected | verified witness (#1668) |
| TV15 | T15 | one/common-domain monitor compromise cannot satisfy configured assurance | runtime assurance profile |
| TV16 | T16 | quarantine/revocation/fork dominates otherwise green evidence | ledger + fork layer |
| TV17 | T17 | requester cannot lower policy; old policy cannot regain authority | #1728 + verified policy |
| TV18 | T18 | old valid checkpoint/journal cannot restore authority | durable replay/checkpoints |
| TV19 | T19 | event delete/reorder/substitute rejected | canonical journal |
| TV20 | T20 | same-predecessor conflicting successors => fork/non-operational | fork layer |
| TV21 | T21 | durable-before-ack retry applies semantic mutation once | transactional store |
| TV22 | T22 | valid transparency receipt alone cannot authorize | transparency integration |
| TV23 | T23 | outage cannot extend snapshot expiry | time + transparency integration |
| TV24 | T24 | command kind/target substitution rejects before mutation | governed mutation layer |
| TV25 | T25 | ordinary operator/revoker credential cannot recover | governance/recovery layer |
| TV26 | T26 | new epoch starts with no active grant carry-over | recovery layer/formal model |
| TV27 | T27 | mismatched runtime artifact cannot claim old admission evidence | build/runtime identity |
| TV28 | T28 | oversized/noncanonical evidence rejected within bounds | durable decoder/fuzz |
| TV29 | T29 | evidence-service outage freezes authority without widening/destructive implication | fault/partition tests |
| TV30 | T30 | recovery-threshold compromise recorded as explicit outside-guarantee assumption | governance/admission record |

---

## 4. Property/state-machine suite

Once the pure/ledger stack executes cleanly, introduce a generated reference state machine with operations such as:

- evaluate grant;
- commit descendant;
- register stricter lineage branch;
- quarantine/revoke subject or lineage;
- revoke grant;
- advance/lose/restore freshness state;
- change grant generation;
- attempt stale cursor/mutation replay;
- inject requester policy weakening;
- inject exact-action mismatch.

After every successful transition assert:

1. creation alone never creates authority;
2. descendant capability ceilings are subsets of all applicable ceilings;
3. subject/lineage counters never exceed applicable hard scopes;
4. negative ancestry implies no positive replication authorization;
5. failed mutation leaves authoritative state unchanged;
6. one cursor has at most one distinct successful successor;
7. stale generation cannot restore authority;
8. requester cannot lower constitutional/policy floor;
9. commit time never precedes evaluation in successful transitions;
10. exact action amount in committed event equals authorization-bound amount.

When durable evidence exists, extend operations with crash, replay, rollback, checkpoint, fork, partition, and recovery.

---

## 5. Formal suite

TLA+ v0.2 should explicitly model properties corresponding to at least:

- TV06 / `CommittedActionWasAuthorizedExactly`;
- TV08 / `CapabilityNeverExpandsAcrossDescent`;
- TV09 / `AncestralBudgetsNeverExceeded`;
- TV10 / `StaleAuthorizationCannotCommit`;
- TV12 / `CommitNeverPrecedesEvaluation`;
- TV16 / `NegativeStateDominatesPositiveAuthority`;
- TV17 / `RequesterCannotLowerPolicyFloor`;
- TV20 / `ForkedLedgerCannotAuthorize`;
- TV21 / `OneCursorHasAtMostOneDistinctSuccessfulSuccessor`;
- TV26 / `RecoveryStartsFreshEpochWithoutGrantCarryover`.

Model bounds and environmental assumptions must be retained with the evidence. A bounded check is not a universal proof.

---

## 6. Multi-threat fault/property bundles

Run at least the B01–B12 bundles in `RSK_THREAT_TO_ADMISSION_GATE_MAP_V0_1.md` once required layers exist.

Additionally randomize combinations of:

- one compromised/unavailable positive signer;
- one compromised/unavailable monitor;
- stale policy/evidence/time;
- rollback/fork/crash;
- subject/lineage negative state;
- stale cursor/token;
- policy weakening;
- duplicate identity/failure-domain collapse;
- runtime build mismatch.

The generator must bias toward boundary values: zero, one below threshold, exact threshold, one above threshold, expiry-1, expiry, evaluation-1, evaluation, maximum counters, and arithmetic-overflow edges.

---

## 7. Evidence retention

For every production-candidate verification run, retain or bind:

- exact Git commit/tree;
- exact Cargo.lock and Nix/toolchain identity;
- target/features/configuration;
- test/fault/model parameters and random seeds;
- TLA+/model checker version and bounds;
- artifacts/logs/digests;
- policy/schema/trust-root profile identifiers;
- final artifact/runtime digest where applicable.

This is required to prevent T27 from making otherwise valid test evidence irrelevant to the deployed implementation.

---

## 8. Current evidence status

As of authorship:

- reference deterministic/black-box tests exist as authored candidates across the stacked RSK PRs;
- #1724, #1726, and #1728 add candidate coverage for TV06, TV12, and TV02/TV17 respectively;
- focused GitHub RSK semantic runs are queued rather than executed;
- cryptographic verifier, durable replay, transactional store, fork recovery, verified time, and exact runtime admission are not implemented;
- therefore no production threat row is considered fully closed.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
