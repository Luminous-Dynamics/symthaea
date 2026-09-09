# Replicator Safety Kernel — Atomic Lineage Ledger v0.1

Status: **draft implementation contract**

This document describes the process-local append-only lineage and budget ledger introduced by `symthaea-replicator-ledger`.

It is an authority/accounting component only. It contains **no physical replication mechanism, molecular implementation, biological implementation, fabrication recipe, or autonomous manufacturing path**.

## 1. Core rule

> **Creation does not confer authority, and descendant creation cannot create a path around an ancestor's authority ceiling.**

A committed descendant is a lineage/accounting fact. The child begins without replication authority and must later receive a fresh explicit external grant that passes the RSK constitutional evaluator.

## 2. Three nested ceilings

Every successful descendant commit is constrained by three independent layers.

### 2.1 Hard lineage policy

Each lineage has an immutable v0.1 policy containing:

- minimum risk class;
- capability ceiling;
- maximum direct children per subject;
- maximum total descendants in the lineage subtree;
- maximum absolute lineage depth;
- maximum resource units in the lineage subtree.

A child lineage may only be **no more permissive** than its parent lineage:

- risk class may stay equal or increase;
- capabilities may only narrow;
- every numeric ceiling may only stay equal or decrease.

Therefore branching cannot reset a population/resource/depth budget or downgrade risk classification.

### 2.2 Subject capability ceiling

Every subject has a capability ceiling that is **not authority**.

- the root starts with the root lineage's hard capability ceiling;
- a newly committed child receives the effective capability set under which it was created as its maximum future ceiling;
- later explicit grants for that child must remain within this ceiling.

This makes capability authority attenuating across reproduction. A fresh external grant cannot restore a capability that was already removed from an ancestral creation path.

No capability-ceiling widening operation exists in v0.1. A future widening protocol, if ever added, should require an explicit higher-order constitutional ceremony rather than ordinary grant issuance.

### 2.3 Grant-generation subject budget scope

When a subject first commits a child, the evaluated grant establishes a subject-subtree budget scope containing the grant identity/generation and its population, depth, and resource limits.

Future descendants increment the counters of **every ancestor subject**, not only the immediate parent. This prevents a grandchild from escaping an ancestor's total-descendant or resource ceiling by receiving a new grant.

Rules:

- a lower grant generation is stale and denied;
- the same generation must be semantically identical for all authority-relevant fields;
- a higher generation may supersede the subject budget scope, but remains bounded by immutable lineage hard ceilings and the subject capability ceiling;
- revoking a grant generation blocks both direct reuse and descendant operations whose ancestor scope depends on that grant.

## 3. Atomicity and replay resistance

Every mutable operation is bound to a `LedgerCursor { epoch, sequence }`.

A bound RSK allow decision captures the cursor used during evaluation. `commit_descendant` accepts it only if the ledger still has the same cursor.

Consequences:

- two different commits evaluated from one snapshot cannot both succeed;
- after the first commit advances the sequence, the second receives `CursorMismatch`;
- every mutation also carries a unique `MutationId` and exact replays receive `DuplicateMutation`;
- the ledger epoch prevents cursors from one ledger instance being accepted by another epoch.

Within the in-memory ledger this is atomic through exclusive `&mut self` mutation. A durable/distributed adapter must preserve the same semantics using compare-and-swap/transactional commit against the full cursor.

## 4. Lineage and subject revocation

Negative authority state is monotonic in v0.1.

Supported transitions:

- clear → quarantined;
- clear → revoked;
- quarantined → revoked.

There is deliberately no ordinary unquarantine/unrevoke method in this tranche.

Both subject and lineage negative state propagate through ancestry during authority preparation and commit:

- quarantining/revoking a subject blocks its descendant subject subtree;
- quarantining/revoking a lineage blocks descendant lineage branches;
- negative state is rechecked at commit time rather than trusted from the earlier authority snapshot.

## 5. Runtime-assurance recheck

The constitutional evaluator is not the last check.

`commit_descendant` also requires a current `RuntimeSafetyWitness` and fails closed when:

- monitoring is unhealthy;
- monitoring is stale;
- policy state is stale;
- safety evidence is stale;
- containment no longer matches;
- the safety-case digest differs from the evaluated authorization;
- the containment-envelope digest differs from the evaluated authorization.

This closes the ledger-side time-of-check/time-of-use gap. Authenticating the witness itself remains an integration responsibility for the trusted runtime-assurance/control-plane boundary.

## 6. Budget accounting semantics

For a proposed child of parent `P` in output lineage `L`:

1. direct-child count of `P` increments by one;
2. subject-subtree descendant count increments for `P` and every ancestor subject;
3. subject-subtree resource count increments for `P` and every ancestor subject;
4. lineage-subtree descendant count increments for `L` and every ancestor lineage;
5. lineage-subtree resource count increments for `L` and every ancestor lineage;
6. the child's absolute depth must fit every applicable subject scope and lineage policy.

All arithmetic is checked before state mutation. Overflow fails closed.

Counts are consumptive in v0.1. Revoking or losing a descendant does not automatically replenish a replication budget.

## 7. Branching semantics

A parent may create into:

- its current lineage; or
- an already-registered **immediate child lineage**.

A subject cannot skip directly into an unrelated or deeper lineage branch. This keeps parent/lineage ancestry explicit and auditable.

Registering a branch does not authorize creation into it. It only creates a stricter-or-equal policy namespace; physical creation still requires an explicit bound RSK grant.

## 8. What the ledger deliberately does not do

v0.1 does not provide:

- physical replication or fabrication;
- grant signature verification;
- runtime-witness attestation verification;
- a durable database;
- a cryptographic event hash chain;
- distributed consensus;
- budget replenishment;
- recovery/unquarantine ceremonies;
- capability-ceiling widening;
- autonomous policy amendment.

Those omissions are intentional. In particular, the current append-only event vector is a semantic reference implementation, not yet durable tamper-evident evidence.

## 9. Test matrix in the crate

The initial unit tests cover:

- stale cursor and duplicate mutation double-spend rejection;
- subject and lineage budget accounting;
- descendant capability attenuation;
- ancestor grant budget persistence across grandchildren;
- branch policy non-widening and risk non-downgrade;
- ancestor lineage quarantine propagation;
- ancestor subject revocation propagation into authority context;
- stale grant generations;
- same-generation authority mutation rejection;
- direct grant revocation and ancestor-scope grant revocation;
- runtime drift between evaluation and commit;
- lineage population ceilings shared across sibling subject subtrees.

## 10. Next safe tranche

The next implementation should make the semantic ledger durable without weakening these invariants:

1. canonical versioned event encoding;
2. cryptographic predecessor/hash binding;
3. replay-from-genesis state reconstruction and invariant verification;
4. transactional/CAS storage adapter;
5. signed checkpoint/export evidence;
6. compact safety-case snapshot binding suitable for Mycelix epistemic-DKG support/challenge edges;
7. fault-injection tests for crash-before-commit, crash-after-log-before-state, duplicate delivery, stale replica, forked store, corrupted event, and revoked witness.

The durable layer must treat the in-memory v0.1 state machine as the reference semantics rather than introducing a second authority interpretation.
