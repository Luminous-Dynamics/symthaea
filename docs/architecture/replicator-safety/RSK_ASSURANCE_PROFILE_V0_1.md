# Replicator Safety Kernel — Assurance Hardening Profile v0.1

Status: **normative design profile for the RSK safety boundary**

This profile strengthens the Replicator Safety Kernel (RSK) around runtime assurance, fail-safe degradation, independent evidence, transparency, and recovery. It does not define or implement a physical replication mechanism.

The profile is intentionally asymmetric:

> **RSK may deny replication authority on uncertainty, but denial alone must not be interpreted as permission for destructive action.**

The purpose is to make bounded reproduction difficult to authorize incorrectly while keeping false positives recoverable through quarantine, evidence review, and externally governed epoch transition.

---

## 1. Safety objective

RSK exists to enforce one constitutional separation:

> **Creation, capability, and authority are different facts. Creation never implies authority.**

A system can physically exist while having no permission to create descendants. A descendant can possess technical capabilities while still having no replication authority. An external grant can authorize a bounded action only inside all inherited ceilings and current safety conditions.

Safety takes precedence over replication liveness:

```text
uncertain authority -> deny/freeze replication authority
negative evidence    -> quarantine/revoke according to policy
evidence conflict    -> forked/non-operational replication authority
recovery              -> explicit external ceremony / fresh epoch
```

None of these transitions inherently requires destruction of an existing system.

---

## 2. Trusted-computing-base decomposition

RSK should be split into five conceptual planes with intentionally unequal trust.

### 2.1 Constitutional authority kernel — trusted

Responsibilities:

- explicit subject/lineage grant checks;
- inherited capability ceilings;
- replication/resource/depth budgets;
- risk-class constraints;
- grant generation and expiry;
- quarantine/revocation precedence;
- deny-by-default decision.

Requirements:

- deterministic;
- small;
- no network dependency;
- no ML/LLM dependency;
- no physical-control implementation;
- no implicit recovery.

### 2.2 Runtime assurance monitor — trusted and independent

Responsibilities:

- monitoring freshness/health;
- containment-envelope status;
- policy/evidence freshness;
- local negative facts;
- bounded timing checks.

The monitor should be simpler than the autonomy it constrains. It must not rely on Symthaea's cognitive output as its sole source of truth.

### 2.3 Durable evidence plane — trusted for integrity, not policy invention

Responsibilities:

- canonical encoding;
- append-only predecessor binding;
- replay-from-genesis;
- checkpoints;
- CAS/transaction semantics;
- crash recovery;
- fork evidence.

Persistence may prove history. It may not mint or reinterpret authority.

### 2.4 Transparency / Mycelix epistemic plane — externally useful, not a hard local dependency

Responsibilities:

- publish immutable safety-case/checkpoint claims;
- preserve support/challenge/supersession relations;
- independent receipt/witness evidence;
- contradiction and equivocation discovery;
- selective disclosure where appropriate.

The local RSK kernel consumes previously verified bounded snapshots. Loss of DKG/network availability never converts unknown evidence into permission.

### 2.5 Symthaea/autonomy plane — explicitly untrusted for hard authority

Symthaea may:

- estimate risk;
- discover anomalies;
- propose safer policies;
- explain denials;
- prioritize investigations;
- simulate counterfactuals.

Symthaea must not:

- mint its own replication grant;
- widen inherited ceilings;
- suppress a hard negative fact;
- extend an expired safety-case snapshot;
- unquarantine/unrevoke itself;
- choose a winning ledger fork;
- bypass the runtime monitor.

---

## 3. Runtime-assurance independence

NASA runtime-assurance/Simplex work provides a useful architectural analogy: capable autonomy can remain outside the trusted safety core when an independently trusted monitor can detect unsafe conditions and force a bounded safe response.

RSK adapts that pattern specifically to **replication authority**, not general vehicle/process control.

### 3.1 Independence requirements

For a monitor to count as independent evidence, avoid common-mode dependence on the authority requester.

At minimum, the hard monitor must not share all of these with the autonomy it constrains:

- the same mutable decision state;
- the same unverified model output;
- the same authority signing key;
- the same policy cache without independent freshness evidence;
- the same writable event history;
- the same process-local boolean as the only sensor of containment.

For R4/R5 deployments, promotion should require an explicit common-mode-failure analysis. Two copies of the same check reading the same compromised state are redundancy, not independence.

### 3.2 Monitor authority is negative, not generative

The runtime monitor should primarily be able to:

- veto;
- freeze;
- quarantine;
- report evidence;
- require re-evaluation.

It should not possess a general capability to create positive replication grants. This keeps compromise of the monitor from turning a safety component into an authority mint.

### 3.3 Reversion state

The RSK analogue of a Simplex reversionary controller is a **non-replicating authority state**:

```text
advanced autonomy path -> replication authority may be evaluated
monitor trigger         -> replication authority becomes unavailable
safe reversion state    -> existing system remains bounded/non-replicating
```

Where a deployment needs additional physical fail-safe behavior, that belongs to the relevant device/process safety case, not to this abstract kernel.

---

## 4. Fail-safe degradation ladder

RSK should distinguish absence of positive evidence from confirmed malicious/unsafe state.

### Level 0 — normal

All required authority, lineage, budget, monitor, containment, and freshness predicates are current.

### Level 1 — authority freeze

Trigger examples:

- monitoring freshness uncertain;
- DKG snapshot expired;
- policy freshness cannot be established;
- clock evidence uncertain;
- transactional store temporarily unavailable.

Effect:

- deny **new replication authority**;
- preserve evidence;
- do not infer compromise merely from unavailability.

### Level 2 — quarantine

Trigger examples:

- containment drift;
- contradictory trusted evidence;
- lineage integrity anomaly;
- unresolved ledger fork;
- explicit external quarantine action.

Effect:

- deny new replication authority for the affected scope;
- preserve and surface the incident;
- require external review/recovery semantics.

### Level 3 — revocation

Trigger examples:

- confirmed compromised grant/key;
- confirmed invalid lineage authority;
- explicit governance revocation.

Effect:

- monotonic negative authority within the current epoch;
- descendants inherit the negative restriction through ancestry;
- ordinary local code cannot reverse it.

### Level 4 — new epoch recovery

Used only after an externally authorized recovery ceremony.

Effect:

- old evidence remains immutable;
- old authority does not implicitly cross the boundary;
- explicitly preserved subjects/lineages receive newly bounded ceilings;
- new grants are required.

---

## 5. Temporal safety contracts

Freshness must be represented as evidence, not a vague boolean convention.

Each deployment profile should define explicit maxima for:

- `max_monitor_age`;
- `max_containment_observation_age`;
- `max_policy_age` or exact policy version;
- `max_safety_case_age` / `valid_until`;
- `max_clock_uncertainty`;
- `max_checkpoint_age` where checkpoints participate in recovery;
- `max_external_receipt_age` only when receipts are required by that profile.

### 5.1 Monotonic time rule

Wall-clock time alone must not permit rollback of expiry/freshness.

A durable implementation should bind time-sensitive authority to a monotonic authority epoch/counter or independently verified clock-continuity evidence. Clock rollback or uncertainty must freeze new replication authority until continuity is restored.

### 5.2 No grace extension by outage

An outage may consume the remaining lifetime of an already verified snapshot. It may not extend that lifetime.

```text
verified until T + network outage -> usable only until T
network unavailable at/after T     -> deny new replication authority
```

---

## 6. Independent transparency without authority delegation

IETF SCITT's architecture is a useful model for separating signed claims, transparent registration, and independently verifiable receipts.

RSK should adopt the same separation of concerns:

- **issuer**: produces a signed RSK safety/checkpoint statement;
- **transparency service / witness**: records or witnesses it;
- **receipt**: proves registration/inclusion;
- **consumer**: verifies the receipt and applicable policy.

### 6.1 Receipt non-authority rule

A transparency receipt proves that a statement was registered under the transparency service's rules. It does **not**, by itself, grant replication authority.

The hard kernel must never implement:

```text
valid transparency receipt => allow replication
```

Instead:

```text
receipt/evidence -> may help validate a bounded safety-case snapshot
snapshot          -> one required input among many
hard predicates   -> still independently evaluated
```

### 6.2 Anti-equivocation witness rule

If two independently valid receipts/checkpoints show different successors for the same:

```text
(epoch, sequence, predecessor_hash)
```

then the state is forked.

Required response:

1. freeze new replication authority for the affected scope;
2. preserve both branches;
3. emit a high-severity incident claim;
4. prohibit automatic winner selection;
5. require external recovery into a fresh epoch.

### 6.3 Minimal disclosure

Transparency should normally publish digests, bounded summaries, and proofs—not detailed physical designs or sensitive operational data.

A public/replicated safety claim should be sufficient to prove:

- which policy/evidence state was relied upon;
- which ledger/checkpoint state it binds;
- validity interval;
- issuer/witness provenance;
- support/challenge/supersession relationships;

without requiring disclosure of a hazardous design payload.

---

## 7. Authority independence and quorum quality

A raw signer count is insufficient to establish independence.

Future RSK quorum evidence should distinguish at least:

- unique signer identities;
- trust domain / organization;
- key root;
- implementation/failure domain where relevant;
- policy role;
- whether a signer is controlled by the subject requesting authority.

### 7.1 High-consequence rule

R4/R5 already require at least two independent approvals. The stronger promotion criterion should be:

> **No R4/R5 quorum counts multiple approvals that collapse to one practical failure domain.**

Examples that should not automatically count as independent:

- two keys held by the same controlled process;
- two services backed by one writable database and one admin principal;
- two model instances deriving the decision from the same unverified state;
- two signatures produced under one compromised root key.

The exact organizational policy can vary by deployment, but the evidence format must be capable of representing this distinction.

---

## 8. Negative-fact propagation latency

For each deployment, define a maximum time/sequence distance within which a newly learned negative fact must dominate future authority.

Examples:

- grant revocation;
- subject quarantine;
- lineage quarantine;
- monitor-health failure;
- containment drift;
- fork detection;
- policy invalidation.

The safest v0.1 local rule is zero event-lag inside one ledger process: once the negative mutation commits, all later authority contexts observe it.

Distributed deployments must state their replication/partition assumptions explicitly. During a partition where freshness cannot be proven, new replication authority freezes rather than relying on stale positive state.

---

## 9. Safety versus denial-of-service

Fail-closed replication authority creates an intentional availability tradeoff. This needs to be explicit so operators do not later weaken the kernel to improve uptime.

### 9.1 Safe DoS posture

A denial-of-service against evidence, DKG, or monitoring may cause:

- inability to obtain new replication authority;
- expiry of existing bounded authority;
- operator investigation;

but must not cause:

- automatic widening of budgets;
- expiry bypass;
- implicit unquarantine;
- automatic fork resolution;
- destructive response solely because evidence is unavailable.

### 9.2 Separate service continuity from replication continuity

Where safe, deployments should design ordinary non-replicating service operation separately from replication authority. This reduces the incentive to weaken RSK just to keep unrelated functionality available.

---

## 10. Recovery-key separation

Recovery authority must be structurally separated from ordinary replication authority.

A subject that can reproduce must not be able to use the same credential or code path to:

- clear its quarantine;
- reverse its revocation;
- choose a fork;
- mint a new epoch;
- widen its own inherited ceiling.

Recommended properties for a future recovery ceremony:

- external quorum;
- keys/roles distinct from ordinary grants;
- fresh evidence bundle;
- explicit reason code;
- old epoch/head/state binding;
- new epoch ID;
- newly declared hard ceilings;
- no implicit carryover of active grants;
- durable incident linkage between epochs.

---

## 11. Promotion gates for durable RSK

No durable RSK implementation should graduate from draft until all of the following are demonstrated.

### G1 — constitutional tests

- deny-by-default authority core passes;
- R0 cannot authorize replication;
- parent grants do not transfer;
- capability intersection/attenuation holds.

### G2 — atomic ledger tests

- competing allows from one cursor cannot double-commit;
- replayed mutations cannot append twice;
- ancestor subject and lineage budgets remain binding;
- quarantine/revocation propagation holds;
- runtime drift between evaluation and commit fails closed.

### G3 — canonical evidence tests

- golden bytes are stable;
- unknown enum tags reject;
- non-canonical representations reject;
- every authority-relevant field changes its containing digest;
- event tamper/delete/reorder/substitute are detected.

### G4 — replay equivalence

For generated valid histories:

```text
live_state(history) == replay_from_genesis(history)
```

For any mutated invalid history:

```text
replay(history') -> reject/non-operational
```

### G5 — crash/fork tests

- crash-before-append;
- durable-before-ack retry;
- torn tail;
- stale CAS;
- same-predecessor equivocation;
- checkpoint rollback;
- checkpoint/full-replay mismatch.

### G6 — runtime assurance composition

- monitor loss freezes authority;
- safety monitor cannot mint grants;
- autonomy cannot suppress hard negatives;
- stale/uncertain time cannot revive expired authority;
- DKG outage cannot extend evidence validity.

### G7 — recovery ceremony

- quarantine/revocation cannot be locally cleared in-epoch;
- fork cannot be resolved by last-write-wins;
- new epoch binds old terminal evidence;
- no grant automatically crosses epochs.

---

## 12. Adversarial verification strategy

Testing should move beyond individual examples into generated state-machine histories.

Generate bounded sequences containing combinations of:

- grant generations;
- descendant commits;
- lineage branches;
- resource consumption;
- subject/lineage quarantine;
- subject/lineage revocation;
- grant revocation;
- monitoring freshness changes;
- policy/evidence expiry;
- concurrent stale cursors;
- duplicate mutation delivery.

After every successful transition assert global invariants:

1. no subject has authority by virtue of existence alone;
2. every descendant capability ceiling is a subset of every applicable ancestor ceiling;
3. no subtree counter exceeds any applicable ancestor scope;
4. no lineage counter exceeds any ancestor hard policy;
5. negative ancestry prevents positive replication authority;
6. ledger sequence is monotonic;
7. failed mutations leave state unchanged;
8. two distinct successful mutations never consume the same cursor;
9. stale generations cannot restore authority;
10. recovery-like widening is impossible inside the epoch.

This state-machine/property layer should become the main regression defense after the deterministic black-box cases are green.

---

## 13. External architecture references

These are architectural references, not claims that RSK is certified against them.

- NASA Technical Reports Server, **A Verification Framework for Runtime Assurance of Autonomous UAS** (2024), describing runtime assurance and Simplex separation between advanced and trusted reversionary control.
- NASA, **Verification of Autonomous Systems** (2025), overviewing runtime monitoring and verification of Simplex-style architectures.
- IETF **RFC 9943 — An Architecture for Trustworthy and Transparent Digital Supply Chains** (2026), defining signed statements, transparency services, verifiable data structures, and receipts.
- IETF SCITT working-group material on integrity, traceability, cryptographic receipts, independent verification, and selective disclosure.

RSK borrows the separation principles—small trusted monitor, append-only/verifiable evidence, independent receipts—while applying them specifically to bounded replication authority.

---

## 14. Next implementation tranches

After #1233/#1239 compile cleanly:

1. **canonical primitive bytes**
   - read-only 32-byte accessors/copies for opaque IDs and digests;
   - frozen `RiskClass`, grant-issuer, negative-state, and event-kind wire tags;
   - golden vectors.
2. **evidence journal**
   - canonical encoder/decoder with strict bounds;
   - SHA-256 domain-separated predecessor chain;
   - full verification report.
3. **replay verifier**
   - reconstruct from genesis;
   - verify semantic successor validity;
   - state digest equivalence.
4. **transactional store**
   - atomic `(epoch, sequence, head_hash)` CAS;
   - idempotent mutation retry;
   - crash/torn-write tests.
5. **checkpoint + transparency receipts**
   - signed/witnessed checkpoints;
   - receipt verification;
   - fork/equivocation detection;
   - receipts remain non-authoritative.
6. **Mycelix epistemic binding**
   - compact digest-only safety-case claims;
   - support/challenge/supersession edges;
   - no live-DKG dependency in the hard kernel.

Only after these layers are independently verified should any higher-level autonomous fabrication system be permitted to depend on RSK as a replication-authority safety boundary.
