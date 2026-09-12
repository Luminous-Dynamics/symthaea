# Replicator Safety Kernel — Threat to Production-Admission Gate Map v0.1

Status: **normative traceability map; authored design evidence, not production admission**

This document links named threats in `RSK_THREAT_MODEL_V0_1.md` to the P0-P12 production-admission gates, expected verification evidence, and current candidate work.

It contains no physical replication mechanism.

---

## 1. Evidence classes

| Code | Evidence type |
|---|---|
| E-DESIGN | authored architecture/ADR/specification |
| E-UNIT | deterministic unit tests |
| E-BLACKBOX | public-API integration tests |
| E-PROP | generated property/state-machine histories |
| E-FAULT | crash/fork/rollback/partition/failure injection |
| E-FORMAL | TLA+ or equivalent explicit-state/formal analysis |
| E-CRYPTO | signature/trust/key-lifecycle golden/negative vectors |
| E-RUNTIME | exact source/toolchain/artifact/runtime identity evidence |
| E-GOV | protected-branch/reviewer/role/recovery governance evidence |

Authored tests count as E-DESIGN until executed on the exact candidate subject.

---

## 2. Threat traceability matrix

| Threat | Primary admission gates | Minimum candidate evidence | Current v0.1 state |
|---|---|---|---|
| T01 local positive-grant forgery | P1, P11, P12 | E-CRYPTO, E-BLACKBOX, E-PROP | open: #1668 |
| T02 approval-count forgery | P1, P5, P11 | E-BLACKBOX, E-CRYPTO, E-PROP | semantic floor candidate #1728; verified identities open |
| T03 duplicate signer/Sybil quorum | P5, P11 | E-CRYPTO, E-BLACKBOX, E-PROP | open: #1668 |
| T04 common-mode quorum compromise | P0, P5, P11 | E-DESIGN, E-BLACKBOX, E-PROP, failure-domain review | open: #1668/#1676 |
| T05 grant subject/lineage substitution | P1, P2, P11 | E-CRYPTO, E-BLACKBOX, E-PROP | reference semantics present; production verification open |
| T06 action/resource substitution | P2, P11 | E-UNIT, E-BLACKBOX, E-PROP | candidate #1724; CI pending |
| T07 child/output commitment substitution | P2, P11 | E-BLACKBOX, E-PROP | open under #1335 |
| T08 capability re-expansion | P2, P11 | E-UNIT, E-BLACKBOX, E-PROP, E-FORMAL | reference ledger candidate exists; executable evidence pending |
| T09 ancestor-budget laundering | P2, P7, P8, P11 | E-BLACKBOX, E-PROP, E-FAULT, E-FORMAL | reference ledger candidate exists; durable evidence open |
| T10 stale authorization replay | P2, P7, P8, P11 | E-BLACKBOX, E-PROP, E-FAULT | process-local cursor semantics candidate; durable CAS open |
| T11 revoked/expired signer replay | P1, P3, P5, P11 | E-CRYPTO, E-BLACKBOX, E-PROP | open: #1668/#1669 |
| T12 clock rollback/pre-evaluation commit | P3, P11 | E-BLACKBOX, E-PROP, E-FAULT | semantic monotonicity candidate #1726; trusted time open |
| T13 forward jump/time disagreement | P3, P11 | E-FAULT, E-PROP | open: #1669 |
| T14 runtime witness forgery | P4, P11 | E-CRYPTO, E-BLACKBOX | open: #1668 |
| T15 monitor common-mode failure | P0, P4, P11 | E-DESIGN, E-FAULT, common-mode analysis | open: #1668/#1676 |
| T16 negative-state suppression | P4, P6, P9, P11 | E-BLACKBOX, E-PROP, E-FAULT | reference negative precedence candidate; production roles open |
| T17 policy downgrade/self-policy | P0, P1, P5, P11 | E-BLACKBOX, E-PROP, E-CRYPTO | semantic separation candidate #1728; verified policy open #1673 |
| T18 durable storage rollback | P7, P8, P9, P11 | E-FAULT, E-PROP, E-RUNTIME | specified in #1241; implementation open |
| T19 journal delete/reorder/substitute | P7, P11 | golden vectors, E-PROP, E-FAULT | specified in #1241; implementation open |
| T20 same-predecessor fork/equivocation | P9, P10, P11 | E-FAULT, E-PROP, E-FORMAL | specified only |
| T21 crash/ack ambiguity/double commit | P8, P11 | E-FAULT, E-PROP | atomic durable store open |
| T22 transparency receipt as permission | P10, P11 | E-BLACKBOX, E-PROP | normative non-authority rule authored |
| T23 outage extends validity | P3, P10, P11 | E-FAULT, E-PROP | normative no-grace rule authored; runtime implementation open |
| T24 governance command substitution | P2, P6, P11 | E-BLACKBOX, E-PROP, E-CRYPTO | open: #1670 |
| T25 recovery-role confusion | P6, P9, P11 | E-CRYPTO, E-BLACKBOX, E-GOV | open: #1670 |
| T26 recovery grant carry-over | P9, P11 | E-BLACKBOX, E-PROP, E-FORMAL | specified only |
| T27 build/runtime substitution | P0, P11, P12 | E-RUNTIME, E-GOV | open: #1682 |
| T28 evidence-parser exhaustion | P7, P11 | fuzz/property bounds, E-FAULT | durable decoder not implemented |
| T29 safety-service denial-of-service | P3, P4, P10, P11 | E-FAULT, E-PROP | fail-closed policy authored; executable distributed evidence open |
| T30 catastrophic recovery-root compromise | P0, P6, P9, P12 | E-GOV, residual-risk acceptance | outside ordinary guarantee; must be explicit |

---

## 3. Admission-gate coverage view

### P0 — Scope and TCB freeze

Threats: T04, T15, T17, T27, T30.

Required additions to P0 evidence:

- exact failure-domain model for quorum and monitors;
- exact host/process/runtime identity dependency;
- exact recovery trust-root threshold;
- explicit components outside positive-authority TCB, including Symthaea and live Mycelix availability.

### P1 — Authentic positive authority

Threats: T01, T02, T05, T11, T17.

Required evidence:

- verified grant/policy types with private construction;
- canonical signed bytes/golden vectors;
- wrong signer/role/scope/generation/policy tests;
- stale/revoked lifecycle rejection.

### P2 — Exact intent → authorization → commit binding

Threats: T05, T06, T07, T09, T10, T24.

Required evidence:

- exact evaluated resource/action binding;
- exact subject/output-lineage/capability/generation/cursor/head binding;
- child or deterministic derivation commitment where applicable;
- exact governance command kind/target/scope binding.

Current candidate: #1724 closes the concrete resource substitution path only.

### P3 — Trusted monotonic time and freshness

Threats: T11, T12, T13, T23, T29.

Required evidence:

- monotonic continuity;
- uncertainty/disagreement handling;
- restart/rollback/forward-jump tests;
- outage cannot extend evidence lifetime.

Current candidate: #1726 establishes only the semantic lower/upper interval floor.

### P4 — Authenticated independent runtime assurance

Threats: T14, T15, T16, T29.

Required evidence:

- verified witness provenance/freshness;
- common-mode/failure-domain analysis;
- negative path cannot be suppressed by autonomy;
- monitor loss freezes new authority.

### P5 — Quorum quality

Threats: T02, T03, T04, T11, T17.

Required evidence:

- distinct verified identities;
- role/key-lifecycle/revocation;
- practical failure-domain metadata and policy;
- duplicate/common-domain collapse rejection.

Current candidate: #1728 separates semantic policy floors from requester assertions but does not verify signer identity or independence.

### P6 — Authenticated negative/governance mutations

Threats: T16, T24, T25, T30.

Required evidence:

- role matrix and private verifier boundary;
- exact command binding;
- ordinary negative authority cannot become recovery authority;
- recovery threshold documented as high-trust root.

### P7 — Canonical durable evidence and replay

Threats: T09, T10, T18, T19, T28.

Required evidence:

- canonical bounded encoding;
- predecessor chain;
- semantic replay;
- decoder resource bounds;
- mutation/reorder/delete/substitute rejection.

### P8 — Transactional persistence and crash safety

Threats: T09, T10, T18, T21.

Required evidence:

- atomic `(epoch, sequence, head_hash)` compare-and-append;
- crash before/after append;
- retry idempotency;
- stale CAS rejection;
- restart continuity.

### P9 — Fork containment and external recovery

Threats: T16, T18, T20, T25, T26, T30.

Required evidence:

- forked/non-operational state;
- no last-write-wins/timestamp winner;
- exact old-terminal-state recovery binding;
- fresh epoch;
- no active grant carry-over.

### P10 — Non-authoritative transparency/Mycelix

Threats: T20, T22, T23, T29.

Required evidence:

- receipt inclusion/provenance tests;
- valid receipt + invalid local state still denies;
- outage does not extend validity;
- contradictory witnesses surface fork but do not choose authority.

### P11 — Formal, adversarial, property verification

Threats: T01–T30 as applicable.

Required evidence:

- generated single- and multi-threat histories;
- formal mapping for model-representable threats;
- negative/fault tests for implementation-representable threats;
- bounded decoder fuzzing;
- exact evidence subject identity.

### P12 — Governance and merge enforcement

Threats: T01, T17, T27, T30 and any change that weakens P0–P11.

Required evidence:

- Class A ADR/change detection;
- protected-branch required checks;
- code owner/reviewer separation as applicable;
- exact production-admission record;
- recovery/trust-root changes themselves governed as Class A.

---

## 4. Multi-threat verification bundles

The following bundles should become named CI/fault-test suites once supporting implementations exist.

| Bundle | Threat combination | Expected result |
|---|---|---|
| B01 | T01 + T17 | forged grant + weak self-policy cannot authorize |
| B02 | T03 + T04 | multiple keys in one failure domain do not satisfy independent quorum |
| B03 | T10 + T18 | stale token against rolled-back store cannot commit |
| B04 | T11 + T12 | revoked signer plus clock rollback cannot revive authority |
| B05 | T14 + T15 | one/common-mode monitor compromise cannot satisfy configured independent assurance |
| B06 | T16 + T22 | valid receipt cannot override committed negative state |
| B07 | T18 + T20 | rollback/fork produces non-operational scope, not automatic winner |
| B08 | T21 + T10 | durable-before-ack retry is idempotent; stale competing successor denied |
| B09 | T23 + T29 | DKG/time/monitor outage eventually freezes authority; never extends expiry |
| B10 | T24 + T25 | ordinary governance command cannot be transformed into recovery |
| B11 | T26 + T20 | fresh-epoch recovery from fork carries no old active grant |
| B12 | T27 + any semantic pass | evidence for different runtime/build cannot claim production admission |

---

## 5. Closure rule for #1676

`#1676` is not closed by these documents alone.

Closure requires:

- executed Class A CI on exact candidate subject;
- compromise matrix accepted as the shared vocabulary for verified grants/quorum/monitors/recovery;
- representable threats covered by executable tests/fault injection;
- TLA+ v0.2 uses the same threat IDs for modeled assumptions/properties;
- all unmodeled environmental assumptions flow into the consuming safety case;
- production-admission record includes accepted M3/catastrophic trust-root assumptions.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
