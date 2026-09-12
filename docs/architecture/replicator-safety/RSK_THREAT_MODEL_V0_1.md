# Replicator Safety Kernel — Threat Model v0.1

Status: **normative threat/compromise model for production-admission work; authored design evidence, not production admission**

This document defines the adversary model, protected assets, trust zones, compromise budgets, security invariants, and expected fail-safe outcomes for the Replicator Safety Kernel (RSK).

It contains **no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path**.

---

## 1. Security objective

RSK protects one core boundary:

> **Within the declared compromise budget, no attacker-controlled or uncertain component may create, widen, revive, replay, or silently transfer positive replication authority.**

RSK does not promise uninterrupted replication availability. Under uncertainty or attack, the safe result may be:

```text
allow unavailable -> deny/freeze new replication authority
confirmed negative -> quarantine/revoke affected scope
fork/equivocation -> non-operational affected authority scope
recovery           -> explicit external ceremony / fresh epoch
```

A denial/freeze is not itself permission for destructive action against an existing subject.

---

## 2. Scope and boundary

### 2.1 In scope

The threat model covers authority-relevant digital control state including:

- grants and grant generations;
- signer/approval evidence;
- quorum and failure-domain policy;
- subject and lineage identity;
- capability ceilings;
- descendant/population/depth/resource budgets;
- safety-case and containment bindings;
- runtime assurance evidence;
- policy and evidence freshness;
- time/continuity evidence;
- quarantine and revocation;
- durable event history and checkpoints;
- fork/equivocation state;
- recovery and epoch transitions;
- admitted source/toolchain/artifact/runtime identity;
- transparency/Mycelix evidence provenance.

### 2.2 Explicitly out of scope

RSK does not claim to prove:

- physical containment against unmodeled physical sabotage;
- correctness of any hazardous physical, biological, molecular, chemical, or manufacturing design;
- absence of side channels not represented by the consuming deployment safety case;
- safety after simultaneous compromise of every configured independent trust root;
- safety after compromise of the configured recovery threshold plus all durable evidence roots;
- correctness of human governance outside the declared roles/thresholds;
- that software evidence alone establishes environmental or biological safety.

A consuming system must maintain a separate physical/system safety case.

---

## 3. Protected assets

### A01 — Authority integrity

No action obtains positive replication authority unless the exact action satisfies the constitutional kernel, current trusted policy, verified evidence, inherited constraints, and current negative state.

### A02 — Authority non-inheritance

Creation/existence does not confer authority. A descendant starts with no positive replication authority.

### A03 — Capability attenuation

A descendant cannot regain capabilities outside every applicable ancestor/output-lineage ceiling.

### A04 — Budget monotonicity

Population, direct-child, depth, and resource consumption cannot be reset, rolled back, or escaped through replay, restart, new generation, branch creation, or descendant indirection.

### A05 — Lineage integrity

Subjects and lineages have unambiguous ancestry; unknown/cyclic/ambiguous lineage fails closed.

### A06 — Negative-state precedence

Quarantine, revocation, fork state, stale critical evidence, and containment failure cannot be canceled by positive grants or receipts.

### A07 — Temporal integrity

Expired, not-yet-valid, rolled-back, stale, or continuity-uncertain authority cannot become positive authority.

### A08 — Durable history integrity

Deletion, reorder, substitution, rollback, stale checkpoint, fork, or torn writes cannot silently produce an authoritative state.

### A09 — Policy integrity

The requester does not choose the policy used to authorize itself. Policy downgrade/supersession is externally governed and evidence-bound.

### A10 — Recovery integrity

Ordinary replication, operator, monitor, or revocation authority does not automatically imply recovery authority. Recovery cannot silently carry active grants across epochs.

### A11 — Runtime identity integrity

Executed authority semantics correspond to the exact admitted source/toolchain/features/configuration/artifact/runtime identity.

### A12 — Transparency non-authority

Transparency/Mycelix receipts and claims can strengthen provenance/auditability but cannot independently produce `Allow`.

---

## 4. Trust zones and actors

Roles are modeled separately even if a deployment later co-locates some of them. Co-location creates a common failure domain that must be declared.

### Z01 — Controlled subject / descendant

Assume fully malicious within its software/process privileges. It may forge inputs, replay requests, suppress cooperative evidence, attempt resource substitution, and attempt self-authorization.

### Z02 — Parent subject

May be compromised or buggy. Ancestral constraints must survive descendant/parent hostility.

### Z03 — Symthaea / high-capability reasoning plane

Useful for analysis, simulation, anomaly detection, explanation, and proposals. Explicitly untrusted for hard positive authority.

### Z04 — Constitutional authority kernel

Small deterministic decision core. Trusted for policy composition and deny-by-default semantics after production admission.

### Z05 — Runtime safety monitor(s)

Trusted only according to the deployment's verified monitor profile. Primarily negative authority: veto/freeze/quarantine/report.

### Z06 — Grant authority

Issues bounded positive grants. Must be externally verified; controlled subject cannot impersonate this role.

### Z07 — Approval/quorum authorities

Provide independent approvals under a policy that reasons about identities and practical failure domains, not raw counts alone.

### Z08 — Policy authority

Defines accepted algorithms, roles, risk floors, quorum/failure-domain requirements, freshness windows, schemas, and transition rules.

### Z09 — Negative/governance authority

May quarantine/revoke according to policy. Ordinary negative authority does not automatically imply recovery authority.

### Z10 — Recovery quorum

Distinct high-trust role capable of declaring fresh-epoch recovery under strict old-state binding and threshold policy.

### Z11 — Durable storage/CAS/replay layer

Trusted only after canonical encoding, hash-chain verification, atomic CAS, crash safety, anti-rollback, checkpoint equivalence, and fork detection are established.

### Z12 — Time/continuity source

Provides verified monotonicity/freshness evidence. Caller-provided wall clock is never sufficient production evidence.

### Z13 — Transparency / Mycelix epistemic plane

Supports signed claims, receipts, support/challenge/supersession relations, contradiction discovery, and selective disclosure. Not a positive-authority oracle.

### Z14 — Host OS / process / privilege boundary

May be part of the TCB depending on deployment. If it can forge verified inputs or rewrite admitted binaries, that dependency must be explicitly included in P0/P11/P12 evidence.

### Z15 — Cryptographic key custody / trust roots

Includes hardware/software keystores, root metadata, signer lifecycle, revocation sources, and operator access controlling keys.

---

## 5. Independence model

### 5.1 Independence is not identity count

Two signers/monitors are not independent merely because their identifiers or public keys differ.

A production evidence model must be able to represent relevant failure-domain dimensions such as:

- `key_root_domain` — common root/HSM/key custody;
- `admin_domain` — same administrator or IAM principal;
- `state_domain` — same writable trust/policy database;
- `process_host_domain` — same process/host/hypervisor where relevant;
- `implementation_domain` — same software lineage/common bug where profile requires diversity;
- `organization_domain` — same controlling organization where policy requires separation;
- `subject_control_domain` — controlled by the requesting subject or its descendants;
- `sensor_domain` — same physical/virtual sensor path;
- `network_power_domain` — shared infrastructure where common-mode failure matters.

### 5.2 Deployment profile rule

A deployment profile must declare which dimensions are required to differ for each quorum/monitor role.

R4/R5 promotion requires explicit common-mode analysis. A pair of checks backed by one mutable source of truth is redundancy, not independence.

### 5.3 Independence claims are verified evidence

The controlled subject may not self-declare that two keys/services are independent. Independence is validated against trusted policy/trust metadata.

---

## 6. Compromise-budget model

Each deployment must instantiate a concrete compromise budget. v0.1 defines minimum expectations.

### C0 — Untrusted autonomy compromise

Compromise of Z01/Z02/Z03 alone **must remain safe**. The attacker may cause denial or bad proposals but cannot create positive authority.

### C1 — Single positive-authority credential compromise

Compromise of one grant/approval credential **must not** satisfy a policy requiring multiple independent domains. Lower-consequence profiles that deliberately accept one authority must declare that authority as a trust root.

### C2 — Single monitor compromise

For profiles requiring multiple independent negative paths, one compromised monitor must not suppress all negative evidence. For a single-monitor lower-consequence profile, monitor compromise is explicitly a trust-root assumption, not hidden redundancy.

### C3 — Storage compromise

A malicious/stale storage backend may cause freeze/non-operation. It must not silently restore old authority after canonical chain/checkpoint/witness verification exists. Same-predecessor conflicting successors make the scope forked.

### C4 — Time-source compromise

One bad/unavailable time source may cause freeze. Production profiles must declare how continuity/uncertainty is established. Time disagreement cannot be resolved by the requesting subject choosing the favorable clock.

### C5 — Transparency/DKG compromise or outage

May reduce auditability/availability. It cannot mint authority or extend expiry. Conflicting independently valid evidence can force fork/non-operation but cannot automatically choose a winner.

### C6 — Ordinary governance-key compromise

A compromised quarantine/revocation/operator credential may add allowed negative state according to role policy, but must not clear negative state, create a fresh epoch, or widen ceilings unless separately authorized as recovery authority.

### C7 — Recovery-threshold compromise

Compromise of the configured recovery threshold is a catastrophic trust-root event outside ordinary RSK guarantees. It must be named in the deployment safety case and protected more strongly than ordinary grant authority.

### C8 — Build/runtime supply-chain compromise

If the running authority binary/configuration is not the admitted artifact, source-level evidence does not apply. P11/P12/issue #1682 must bind evidence to exact runtime identity.

---

## 7. Named threat catalog

Threat identifiers are stable references for tests, formal properties, ADRs, incident evidence, and production-admission records.

### T01 — Local positive-grant forgery

**Attack:** controlled code constructs a grant and labels itself externally independent.

**Protected assets:** A01, A02, A09.  
**Required response:** unverified value cannot enter production evaluator.  
**Primary gates:** P1, P11, P12.

### T02 — Approval-count forgery

**Attack:** requester supplies a favorable approval count or `required = 0`.

**Protected assets:** A01, A09.  
**Required response:** policy-owned floors dominate requester assertions; production consumes verified signer evidence.  
**Primary gates:** P1, P5, P11.

### T03 — Duplicate signer / Sybil quorum

**Attack:** one actor uses multiple keys/identities to satisfy a numeric quorum.

**Protected assets:** A01, A09.  
**Required response:** duplicate identity/failure-domain collapse rejected according to policy.  
**Primary gates:** P5, P11.

### T04 — Common-mode quorum compromise

**Attack:** nominally separate signers depend on the same admin/root/database/controlled process.

**Protected assets:** A01, A09.  
**Required response:** failure-domain policy prevents counting them as independent.  
**Primary gates:** P0, P5, P11.

### T05 — Grant subject/lineage substitution

**Attack:** move a valid grant to another subject or lineage.

**Protected assets:** A01, A05.  
**Required response:** exact subject/lineage binding; mismatch denies.  
**Primary gates:** P1, P2, P11.

### T06 — Action/resource substitution

**Attack:** evaluate one action/resource amount and commit another.

**Protected assets:** A01, A04.  
**Required response:** exact intent→authorization→commit equality; no mutation on mismatch.  
**Primary gates:** P2, P11.  
**Candidate:** #1724.

### T07 — Child/output commitment substitution

**Attack:** authorize one intended output but bind authority to a different child/output after execution.

**Protected assets:** A01, A05.  
**Required response:** child identity or deterministic derivation/input commitment must be bound where applicable.  
**Primary gates:** P2, P11.

### T08 — Capability re-expansion

**Attack:** descendant later receives a grant that restores capability removed by an ancestor/output-lineage ceiling.

**Protected assets:** A03.  
**Required response:** effective capabilities remain subset of every applicable ceiling.  
**Primary gates:** P2, P11.

### T09 — Ancestor-budget laundering

**Attack:** new grants/generations/branches attempt to reset population, depth, or resource limits.

**Protected assets:** A04.  
**Required response:** ancestral scopes and hard lineage ceilings remain binding; monotonic consumption.  
**Primary gates:** P2, P7, P8, P11.

### T10 — Stale authorization replay

**Attack:** replay an authorization after ledger state/negative state/grant generation changes.

**Protected assets:** A01, A04, A06, A08.  
**Required response:** exact cursor/head/generation binding and mutation idempotency.  
**Primary gates:** P2, P7, P8, P11.

### T11 — Revoked/expired signer replay

**Attack:** replay evidence signed by a now-revoked, retired, or expired key.

**Protected assets:** A01, A07, A09.  
**Required response:** verified lifecycle/revocation/freshness state.  
**Primary gates:** P1, P3, P5, P11.

### T12 — Clock rollback / pre-evaluation commit

**Attack:** present a time earlier than evaluation or before a negative/expiry boundary.

**Protected assets:** A07.  
**Required response:** `evaluated_at <= commit_time < expires_at` plus trusted continuity.  
**Primary gates:** P3, P11.  
**Candidate semantic floor:** #1726.

### T13 — Forward clock jump / forced expiry ambiguity

**Attack:** malicious or broken time source jumps ahead or disagrees enough to invalidate freshness assumptions.

**Protected assets:** A07, availability.  
**Required response:** uncertainty freezes new authority; no attacker-selected favorable source.  
**Primary gates:** P3, P11.

### T14 — Runtime witness forgery

**Attack:** caller constructs healthy/fresh/contained booleans or digests.

**Protected assets:** A01, A06.  
**Required response:** production consumes authenticated verified monitor evidence.  
**Primary gates:** P4, P11.

### T15 — Runtime monitor common-mode failure

**Attack:** multiple apparent monitors derive from one compromised mutable source or implementation.

**Protected assets:** A06.  
**Required response:** deployment profile defines monitor failure domains/common-mode analysis.  
**Primary gates:** P0, P4, P11.

### T16 — Negative-state suppression

**Attack:** autonomy or positive evidence attempts to ignore quarantine, revocation, containment drift, or fork state.

**Protected assets:** A06.  
**Required response:** negative predicates independently dominate positive evidence.  
**Primary gates:** P4, P6, P9, P11.

### T17 — Policy downgrade / self-policy

**Attack:** requester selects weaker quorum/risk/freshness/schema requirements.

**Protected assets:** A09.  
**Required response:** trusted policy source; requester may only self-restrict.  
**Primary gates:** P0, P1, P5, P11.  
**Candidate semantic separation:** #1728.

### T18 — Durable storage rollback

**Attack:** present an older but internally valid journal/checkpoint.

**Protected assets:** A04, A06, A08.  
**Required response:** monotonic durable head/checkpoint/witness continuity; stale state freezes authority.  
**Primary gates:** P7, P8, P9, P11.

### T19 — Journal deletion/reorder/substitution

**Attack:** tamper with event sequence, predecessor, mutation, or authority-relevant field.

**Protected assets:** A08.  
**Required response:** canonical encoding + domain-separated hashes + replay verification rejects/non-operational.  
**Primary gates:** P7, P11.

### T20 — Same-predecessor fork/equivocation

**Attack:** produce conflicting valid successors for one `(epoch, sequence, predecessor_hash)`.

**Protected assets:** A08, A10.  
**Required response:** no automatic winner; affected scope non-operational; preserve both branches; external fresh-epoch recovery.  
**Primary gates:** P9, P10, P11.

### T21 — Crash/ack ambiguity and double commit

**Attack/fault:** durable append succeeds but acknowledgement is lost; retry risks duplicate semantic mutation.

**Protected assets:** A04, A08.  
**Required response:** atomic CAS + unique mutation ID + replay-equivalent idempotency.  
**Primary gates:** P8, P11.

### T22 — Transparency receipt as permission

**Attack:** treat a valid receipt/inclusion proof as sufficient positive authority.

**Protected assets:** A01, A12.  
**Required response:** receipts validate provenance/inclusion only; all local hard predicates still required.  
**Primary gates:** P10, P11.

### T23 — DKG/transparency outage extends validity

**Attack/fault:** network unavailable, operator allows stale evidence indefinitely.

**Protected assets:** A07, A12.  
**Required response:** outage never extends pre-existing validity; after expiry new authority denies.  
**Primary gates:** P3, P10, P11.

### T24 — Governance command substitution

**Attack:** retarget quarantine/revocation/recovery/checkpoint command or change command kind after authorization.

**Protected assets:** A06, A10.  
**Required response:** exact command target/kind/scope binding and authenticated role.  
**Primary gates:** P2, P6, P11.

### T25 — Recovery-role confusion

**Attack:** ordinary operator/revoker/grant credential clears quarantine or creates new epoch.

**Protected assets:** A06, A10.  
**Required response:** structurally distinct recovery role/quorum and fresh-epoch ceremony.  
**Primary gates:** P6, P9, P11.

### T26 — Recovery grant carry-over

**Attack/fault:** old active grants automatically remain valid in a new epoch.

**Protected assets:** A01, A10.  
**Required response:** no implicit grant carry-over; all new positive authority reissued under new epoch/policy.  
**Primary gates:** P9, P11.

### T27 — Build/runtime substitution

**Attack:** run code/config/features/toolchain different from the artifact whose evidence passed.

**Protected assets:** A11 and all semantic assets transitively.  
**Required response:** admission evidence binds exact source→toolchain→artifact→runtime identity.  
**Primary gates:** P0, P11, P12.

### T28 — Evidence-parser resource exhaustion

**Attack:** malformed/oversized evidence causes allocation/CPU exhaustion or parser ambiguity.

**Protected assets:** availability, A08.  
**Required response:** strict size/depth/count bounds before allocation; canonical decode; bounded verification.  
**Primary gates:** P7, P11.

### T29 — Monitor/transparency/time denial-of-service

**Attack:** make evidence services unavailable to pressure operators into bypassing checks.

**Protected assets:** A01, A06, A07, A12.  
**Required response:** freeze new authority, preserve non-replicating service where separately safe, never widen/bypass expiry.  
**Primary gates:** P3, P4, P10, P11.

### T30 — Catastrophic recovery-root compromise

**Attack:** adversary controls configured recovery threshold and enough durable/trust-root state to produce apparently valid fresh-epoch recovery.

**Protected assets:** A10 and all assets transitively.  
**Required response:** explicitly outside ordinary RSK guarantee; deployment must harden custody, threshold, offline/physical controls, audit, and incident response.  
**Primary gates:** P0, P6, P9, P12.

---

## 8. Attack outcomes and required safe states

Threat handling uses four outcomes.

### S0 — Normal authority evaluation

All required evidence and continuity are valid.

### S1 — Authority freeze

Used for uncertainty/unavailability/staleness where compromise is not established. No new positive replication authority.

### S2 — Quarantine/revocation

Used for committed negative facts or confirmed compromise according to policy. Monotonic within epoch except through externally governed recovery.

### S3 — Forked/non-operational

Used for unresolved conflicting authoritative histories. No automatic winner. External fresh-epoch recovery required.

A threat handler must never map uncertainty directly to destructive action solely because authority cannot be proven.

---

## 9. Threat-composition requirements

Single-threat tests are insufficient. Production verification must include bounded combinations such as:

- T01 + T17: forged grant plus weakened policy;
- T03 + T04: duplicate signers inside one failure domain;
- T10 + T18: stale authorization against rolled-back storage;
- T11 + T12: revoked signer plus clock rollback;
- T14 + T15: forged witness plus common-mode monitor compromise;
- T16 + T22: negative local state plus valid transparency receipt;
- T18 + T20: rollback plus fork/equivocation;
- T21 + T10: durable-before-ack retry plus stale token replay;
- T23 + T29: DKG outage plus time-source outage;
- T24 + T25: governance command substitution plus recovery-role confusion;
- T27 + any semantic threat: unadmitted binary attempts to present old passing evidence.

Property/state-machine generation should vary these combinations rather than relying on hand-selected happy paths.

---

## 10. Formal-model obligations

TLA+ v0.2 / equivalent formal refinement must state which threat assumptions are represented and which remain environmental.

At minimum, model-checkable properties should correspond to:

- T06 `CommittedActionWasAuthorizedExactly`;
- T08 `CapabilityNeverExpandsAcrossDescent`;
- T09 `AncestralBudgetsNeverExceeded`;
- T10 `StaleAuthorizationCannotCommit`;
- T12 `CommitNeverPrecedesEvaluation`;
- T16 `NegativeStateDominatesPositiveAuthority`;
- T17 `RequesterCannotLowerPolicyFloor`;
- T18 `RollbackCannotRestoreAuthority` once durable epoch/head semantics are modeled;
- T20 `ForkedLedgerCannotAuthorize`;
- T21 `OneCursorHasAtMostOneDistinctSuccessfulSuccessor`;
- T25/T26 `RecoveryRequiresFreshEpochAndNoGrantCarryover`.

Cryptographic unforgeability, key custody, real failure-domain independence, and physical containment remain assumptions/interfaces rather than mathematical consequences of the state machine unless explicitly modeled.

---

## 11. External architectural references

These references support architectural separation and threat categories; they are not certification claims.

- NIST SP 800-207, **Zero Trust Architecture** — policy decision/enforcement separation and rejection of implicit trust.
- NASA, **A Verification Framework for Runtime Assurance of Autonomous UAS** — trusted runtime monitor/reversion path around untrusted advanced control.
- NASA, **Verification of Autonomous Systems** — runtime monitoring and Simplex-style assurance concepts.
- IETF RFC 9943, **An Architecture for Trustworthy and Transparent Digital Supply Chains** — signed statements, transparency services, verifiable logs, and receipts.
- The Update Framework (TUF) security model — rollback, freeze, fast-forward, role separation, signed metadata, and expiration.

---

## 12. Promotion rule

This threat model becomes production evidence only when the production-admission record demonstrates:

1. every applicable threat has a mapped control/gate;
2. every representable threat has executable negative/fault/property evidence;
3. environmental assumptions are explicitly inherited by the consuming safety case;
4. TLA+/state-machine assumptions use the same threat IDs;
5. the exact admitted runtime/build and trust-root configuration are bound;
6. residual risks and catastrophic trust-root assumptions are accepted by the designated external authority.

Until then:

**Production admission status: DENIED / NOT YET ELIGIBLE.**
