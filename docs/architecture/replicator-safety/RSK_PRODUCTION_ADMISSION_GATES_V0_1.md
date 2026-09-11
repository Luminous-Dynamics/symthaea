# Replicator Safety Kernel — Production Admission Gates v0.1

Status: **normative safety gate; current Rust crates are reference semantics and are NOT production-admitted**

This document defines the minimum evidence required before any higher-level system may treat the Replicator Safety Kernel (RSK) as a production replication-authority boundary.

It contains **no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path**.

---

## 1. Admission rule

> **No production integration by accumulation. RSK becomes production-admitted only when every applicable gate below has explicit passing evidence.**

A crate compiling, a unit test passing, a transparency receipt existing, or an operator believing a system is safe is not by itself sufficient.

Until admission is recorded:

- `symthaea-replicator-safety` is a reference constitutional evaluator;
- `symthaea-replicator-ledger` is a reference in-memory state machine;
- raw grants, quorum counts, runtime witnesses, and caller-provided time are modeled claims rather than authenticated facts;
- no hazardous physical actuation path should rely on an RSK `Allow` as sufficient authority.

---

## 2. Gate P0 — scope and TCB freeze

**Requirement:** identify the exact production trusted computing base before implementation promotion.

Must enumerate:

- constitutional evaluator version/digest;
- lineage-ledger version/digest;
- grant verifier;
- quorum/trust verifier;
- runtime monitor verifier;
- trusted time/clock-continuity source;
- durable store and replay verifier;
- checkpoint/witness verifier;
- recovery/governance verifier;
- process and privilege boundaries between them.

**Pass evidence:** architecture record showing that Symthaea cognition, general-purpose LLM output, live Mycelix/DKG availability, and physical creation are not hidden positive-authority dependencies.

**Current status:** OPEN.

---

## 3. Gate P1 — authentic positive authority

**Requirement:** plain caller-constructed positive claims cannot enter the production authority decision.

Production should introduce verified wrappers such as:

```text
VerifiedReplicationGrant
VerifiedGrantIssuer
VerifiedSafetyCaseSnapshot
```

Exact names are not normative. Construction semantics are.

A verified grant must establish at least:

- signature authenticity;
- accepted algorithm/profile;
- signer/key trust root;
- issuer role;
- issuer not controlled by the requesting subject where external independence is required;
- exact subject ID;
- exact lineage ID;
- allowed capabilities;
- budget ceilings;
- grant generation;
- validity interval;
- safety-case digest;
- containment-envelope digest;
- policy version/epoch binding.

No public constructor may allow arbitrary code in the controlled authority path to label itself `ExternalIndependent` and thereby manufacture trusted positive authority.

**Pass evidence:** signature/trust tests, wrong-key/wrong-role tests, replay tests, trust-rotation tests, compromised-key revocation tests, and API review demonstrating unverified grant values cannot reach the production decision function.

**Current status:** OPEN — the v0.1 authority crate uses reference-value structs.

---

## 4. Gate P2 — exact intent → authorization → commit binding

**Requirement:** the opaque authorization consumed by a mutation must commit to every authority-relevant value that was evaluated.

At minimum bind:

- ledger epoch and pre-commit cursor/head;
- subject;
- intended output lineage;
- exact requested capability set;
- exact resource amount;
- grant ID and generation;
- risk class;
- safety-case digest;
- containment-envelope digest;
- validity interval;
- applicable budget ceilings;
- operation/mutation commitment where known before authorization.

If child identity is known before authorization, bind it directly. If it is not known until a later deterministic creation step, bind a pre-committed derivation/input digest and require the resulting child identity to match that commitment.

### Known v0.1 blocker

`BoundedReplicationAuthorization` currently does **not** contain the exact requested resource amount checked during `evaluate_replication_authority`, while `commit_descendant` accepts `resource_units` as a separate argument.

The ledger still enforces hard/ancestral ceilings, so this is not an unbounded-budget escape. It is nevertheless a Class A mismatch because the committed action can differ from the action evaluated.

Required fix:

```text
authorized.requested_resource_units == committed.resource_units
```

with mismatch denied before any mutation.

**Pass evidence:** positive exact-match tests plus negative tests for changed resource amount, changed output lineage, changed capabilities, changed operation commitment, stale cursor, and substituted child commitment.

**Current status:** OPEN / MERGE-BLOCKING FOR PRODUCTION ADMISSION.

---

## 5. Gate P3 — trusted monotonic time and freshness

**Requirement:** callers cannot choose the time against which authority expiry is evaluated.

Production must use verified time evidence providing enough continuity to establish:

```text
commit_time >= evaluation_time
commit_time < authorization_expiry
freshness_age <= applicable maximum
```

Clock rollback, time-source disagreement beyond policy tolerance, or lost continuity freezes new replication authority.

Acceptable architectures may include a monotonic authority epoch/counter combined with verified wall-clock evidence; the exact mechanism is deployment-specific.

### Known v0.1 blocker

`commit_descendant` currently accepts caller-supplied `now_unix_secs`. This is reference semantics only.

**Pass evidence:** rollback tests, forward-jump tests, restart continuity tests, expired-authority tests, time-source disagreement tests, and proof that the controlled subject cannot substitute arbitrary time.

**Current status:** OPEN.

---

## 6. Gate P4 — authenticated independent runtime assurance

**Requirement:** runtime monitor/containment facts used at commit are authenticated, fresh, and meaningfully independent of the autonomy requesting authority.

Production should consume a verified type such as:

```text
VerifiedRuntimeSafetyWitness
```

that establishes:

- monitor identity;
- evidence signature/MAC/attestation as applicable;
- freshness/sequence;
- containment-envelope binding;
- safety-case binding;
- monitor policy version;
- independence/failure-domain metadata required by the deployment profile.

The monitor's hard authority should be primarily negative: veto/freeze/quarantine/report. It must not become a general positive-grant mint.

### Known v0.1 blocker

`RuntimeSafetyWitness` is a plain public reference struct.

**Pass evidence:** forged-witness rejection, stale-witness rejection, wrong-envelope rejection, monitor compromise/failure injection, common-mode analysis, and tests that autonomy cannot suppress a committed negative fact.

**Current status:** OPEN.

---

## 7. Gate P5 — quorum quality, not quorum count

**Requirement:** production quorum evidence proves distinct authorized signers/failure domains rather than accepting a caller-supplied integer.

Evidence must be able to represent at least:

- signer identity;
- signature/key root;
- role;
- organization/trust domain where required;
- relevant implementation/administrative failure domain;
- subject-control relationship;
- signature validity interval and revocation state.

For R4/R5, multiple approvals that collapse to one practical failure domain must not automatically count as independent.

### Known v0.1 blocker

`QuorumEvidence` currently contains counts only and is reference semantics.

**Pass evidence:** duplicate-signer tests, same-root tests, same-admin-domain policy tests, revoked signer tests, insufficient-independent-domain tests, and valid independent-quorum tests.

**Current status:** OPEN.

---

## 8. Gate P6 — authenticated negative/governance mutations

**Requirement:** ledger mutation access is role-separated and authenticated.

Production must distinguish authority for at least:

- descendant commit;
- lineage-branch registration;
- subject quarantine;
- subject revocation;
- lineage quarantine;
- lineage revocation;
- grant revocation;
- checkpoint acceptance;
- fork declaration;
- epoch recovery.

A controlled subject must not be able to clear quarantine/revocation, select a fork winner, or widen inherited ceilings through an administrative API.

Local runtime monitors may be permitted to add negative facts according to policy. Recovery remains externally governed.

### Known v0.1 blocker

The in-memory ledger assumes trusted access to its mutation methods.

**Pass evidence:** role-matrix tests, unauthorized mutation tests, compromised ordinary grant key tests, recovery-key separation tests, and API/privilege-boundary review.

**Current status:** OPEN.

---

## 9. Gate P7 — canonical durable evidence and replay

**Requirement:** durable state is reconstructed from verified evidence, not trusted deserialization.

Must implement the durable-evidence contract, including:

- fixed canonical ID/digest bytes;
- frozen wire discriminants;
- strict bounds before allocation;
- domain-separated SHA-256 record hashing;
- predecessor binding;
- exact sequence continuity;
- unique mutation IDs;
- replay-from-genesis semantic verification;
- complete authority-relevant state digest;
- checkpoint/full-replay equivalence.

**Pass evidence:** golden vectors, mutated-byte rejection, predecessor substitution, deletion/reorder tests, unknown-enum/noncanonical decode rejection, generated valid-history replay equivalence, and invalid-history rejection.

**Current status:** SPECIFIED, NOT IMPLEMENTED.

---

## 10. Gate P8 — transactional persistence and crash safety

**Requirement:** durable mutation commit is atomic against `(epoch, sequence, head_hash)`.

Required semantics:

```text
compare_and_append(expected_epoch, expected_sequence, expected_head_hash, event)
```

must be one atomic storage operation or provide equivalent guarantees.

Must handle:

- crash before durable append;
- durable append before acknowledgement;
- retry with same mutation ID;
- retry with different mutation against stale cursor;
- torn/corrupt tail;
- checkpoint ordering;
- process concurrency;
- storage restart.

**Pass evidence:** fault-injection/crash tests showing exactly-once semantic mutation and no authority state advancement after a failed append.

**Current status:** OPEN — current cursor semantics are process-local reference behavior.

---

## 11. Gate P9 — fork/equivocation containment and external recovery

**Requirement:** conflicting valid successors do not resolve through last-write-wins, timestamps, or autonomy preference.

Conflict at the same `(epoch, sequence, predecessor_hash)` must:

1. mark the affected replication-authority scope forked/non-operational;
2. preserve all conflicting branches as incident evidence;
3. deny new replication authority;
4. require externally governed recovery;
5. create a fresh epoch whose genesis binds the old terminal evidence;
6. carry no active grant across the epoch implicitly.

Transparency/witness receipts may expose equivocation but do not choose authority.

**Pass evidence:** independent conflicting-log test, witness disagreement test, no-auto-winner test, fresh-epoch recovery test, and evidence-retention test.

**Current status:** SPECIFIED, NOT IMPLEMENTED.

---

## 12. Gate P10 — transparency/Mycelix remains non-authoritative

**Requirement:** external transparency and the Mycelix epistemic DKG strengthen provenance without becoming a hidden liveness dependency or positive authority oracle.

A receipt/checkpoint/claim can prove evidence provenance or inclusion. It cannot alone produce:

```text
Allow
```

The hard local authority predicates remain independently required.

During DKG/network outage:

- already-verified snapshots remain usable only until their pre-existing expiry;
- expiry is never extended because the network is unavailable;
- new negative local facts still dominate;
- after expiry, new replication authority fails closed.

**Pass evidence:** DKG unavailable before/after expiry tests, forged receipt rejection, valid receipt-but-invalid-local-state denial, contradiction/fork surfacing, and minimal-disclosure review.

**Current status:** SPECIFIED, NOT IMPLEMENTED.

---

## 13. Gate P11 — formal, adversarial, and property verification

**Requirement:** example tests are supplemented with generated histories and explicit formal-model correspondence.

Minimum verification program:

- execute the TLA+ model with documented bounds;
- property/state-machine generation over grants, generations, lineages, descendants, budgets, negative facts, stale cursors, and duplicate delivery;
- after each successful transition assert all global constitutional invariants;
- mutate valid histories and prove replay rejection/non-operation;
- differential tests between reference live state and durable replay state;
- crash/fork fault injection;
- fuzz bounded decoders once canonical durable encoding exists.

Core properties include:

```text
creation != authority
child ceiling subset-of all applicable ancestor ceilings
all subtree counters <= all applicable ancestor scopes
negative ancestry => no positive replication authority
failed mutation => state unchanged
one cursor => at most one distinct successful successor
stale generation cannot restore authority
in-epoch recovery cannot widen immutable ceilings
```

**Pass evidence:** retained CI artifacts/results with exact commit/toolchain/environment identity; no claim of proof beyond the tested/model-checked bounds.

**Current status:** PARTIAL — deterministic unit/black-box tests authored; TLA+/Rust execution still awaiting available runners/tooling at time of this document.

---

## 14. Gate P12 — governance and merge enforcement

**Requirement:** safety semantics cannot be changed through an ordinary unreviewed path.

RSK authority and ledger code are Class A. The focused RSK workflow must require a changed RSK Class A ADR for authority/ledger/governance-surface changes.

Before production admission, repository policy must also be externally verified to ensure applicable protected branches require the relevant RSK safety checks before merge.

Important distinction:

- repository rulesets query returned no configured rulesets at the time this gate was authored;
- branch-protection details were not readable through the available GitHub integration;
- therefore required-check enforcement is **not assumed** and must be verified by an authorized repository administrator before production admission.

RSK crates remain `publish = false` until this gate and all preceding gates pass.

**Pass evidence:** Class A ADR, successful focused RSK workflow on admission commit, verified protected-branch required-check configuration, code-owner/reviewer policy as applicable, and explicit production-admission record.

**Current status:** PARTIAL — Class A registration, ADR gate, focused CI, and `publish = false` are authored; protected-branch enforcement is not verified.

---

## 15. Admission matrix

| Gate | Requirement | Current v0.1 status | Production blocking? |
|---|---|---|---|
| P0 | TCB/scope freeze | Open | Yes |
| P1 | cryptographically verified grants | Open | Yes |
| P2 | exact evaluated-action binding | Open; known resource mismatch | Yes |
| P3 | trusted monotonic time | Open | Yes |
| P4 | verified independent runtime witness | Open | Yes |
| P5 | verified quorum/failure domains | Open | Yes |
| P6 | authenticated governance mutations | Open | Yes |
| P7 | canonical durable evidence/replay | Specified only | Yes |
| P8 | transactional/crash-safe persistence | Open | Yes |
| P9 | fork containment/fresh-epoch recovery | Specified only | Yes |
| P10 | non-authoritative transparency/DKG | Specified only | Yes |
| P11 | formal/property/fault verification | Partial | Yes |
| P12 | Class A + required-merge enforcement | Partial | Yes |

**Production admission status: DENIED / NOT YET ELIGIBLE.**

That status is a safety property, not a criticism of the reference model. The reference model's job is to make these semantics precise enough to verify before anyone depends on them physically.

---

## 16. Promotion record

When all gates pass, production admission should be a separate Class A ADR containing:

- exact commit SHA(s);
- exact Rust/Nix/toolchain identity;
- formal-model version and executed bounds;
- retained CI/property/fault evidence digests;
- trust-root and recovery-role configuration;
- durable-store implementation/profile;
- runtime-monitor profile/common-mode analysis;
- protected-branch/required-check evidence;
- residual risks and deployment constraints;
- explicit statement of which physical/system safety case consumes RSK and which hazards remain outside RSK's scope.

No earlier document or passing test should be interpreted as that production-admission record.
