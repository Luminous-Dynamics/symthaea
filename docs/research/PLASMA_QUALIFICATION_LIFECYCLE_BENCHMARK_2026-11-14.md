# PLASMA-QUAL-LIFECYCLE-001 — event-reconciled Bond Qualification Passport

**Date:** 2026-11-14  
**Status:** S0 lifecycle-policy audit and synthetic known-answer fixture  
**Issue:** [#12052](https://github.com/Luminous-Dynamics/symthaea/issues/12052)  
**Composes with:** #12002 qualification transfer, #12026 method-to-claim scope, #12033 method-qualified transfer receipts, #6414 typed evidence DAG, #2150/#2227 calibration continuity, #5834 sensor diagnosability, #4344 material lineage, #11971 bond-life evidence and #11952 spatial support.

## Executive thesis

A Bond Qualification Passport cannot be a static PASS attached to a recipe name. Its claims remain admissible only while the evidence dependencies that support them remain current, in-scope and unretracted.

A supplier change, out-of-tolerance calibration, process drift, stale test method, source retraction or adverse field event can invalidate only some claims within a Passport. A later requalification can create a new generation, but it must never rewrite the prior record into a history that appears continuously qualified.

The paradigm shift is from **static qualification certificate** to **claim-level, event-reconciled qualification lifecycle**.

## What is new here—and what is not

The repository already has a general typed evidence-DAG/invalidation roadmap in [#6414](https://github.com/Luminous-Dynamics/symthaea/issues/6414), calibration continuity owners in #2150/#2227, sensor self-test/diagnosability in #5834, and material/sample lineage in #4344. This increment does not create a second provenance store, calibration registry, event ledger or generic dependency graph.

It defines the plasma-specific policy that consumes those existing evidence references:
- which event families affect which classes of bond qualification claims;
- how effective-time and event-ingestion time differ;
- how claim-specific dependencies constrain invalidation;
- how suspension, revocation, expiry and supersession differ;
- how a new generation replaces a previous one without mutating history.

The current implementation is an isolated policy auditor. It is not yet wired into #12006's transfer audit, a customer QMS, a MES/PLM system, or production release workflow.

## Paradigm shift 1 — a Passport is a set of claims, not one green boolean

The fixture models separate claims such as:
- initial bond-strength observation;
- wettability observation;
- bond-life / survival evidence;
- qualification-transfer scope.

Each claim has its own:
- evidence references;
- method-qualified observation receipts where applicable;
- dependency-closure reference and digest;
- explicit dependency keys;
- optional review deadline or validity end.

This permits targeted degradation. For example, expiration of a load-cell calibration can suspend the bond-strength claim that depends on it while leaving a contact-angle claim backed by a different calibration path current. A process drift event can affect both, if both depend on the changed process profile.

A qualification-level target event can still suspend every claim in that generation. A dependency-scoped event affects only claims whose explicit dependency sets intersect the event's affected dependency keys.

## Paradigm shift 2 — lifecycle states carry different meanings

The S0 output distinguishes:

| State | Meaning |
|---|---|
| `CURRENT` | No applicable invalidating trigger or validity expiry has been observed under the declared contract |
| `SUSPENDED` | Current admission is paused pending review, bridge evidence or investigation |
| `REVOKED` | A foundational source or premise was confirmed invalid/retracted |
| `EXPIRED` | A declared claim validity or review condition elapsed |
| `SUPERSEDED` | A newer qualified generation explicitly replaced this generation for future scope |
| `BLOCKED` | Required lineage, dependency, event, time or method evidence is incomplete |

These are not interchangeable. A suspected field failure suspends a dependent claim pending review; it does not automatically establish root cause or prove that an original test was false. A confirmed retraction of a foundational bond-test record revokes the dependent claim. A due date produces EXPIRED rather than pretending the underlying historical test no longer happened.

There is no transition that simply changes REVOKED, EXPIRED or SUSPENDED back to CURRENT. A later successful requalification is represented by a new generation with a predecessor reference and an explicit supersession event. The old generation remains inspectable with its old claim states and evidence ancestry.

## S0 event contract

Every event carries:
- unique immutable event ID;
- monotonic sequence and predecessor event ID;
- event type and evidence status (`suspected` or `confirmed`);
- subject and optional directly targeted generation;
- affected dependency keys;
- event time, ingestion time, effective-from and optional effective-until;
- time-source reference;
- source reference and SHA-256 field;
- accountable authority reference.

Events are ordered by declared sequence, not file or network arrival order. An exact duplicate event (same ID and exact payload/digest) is idempotent. Reuse of an event ID with a different payload is REJECTED. Sequence gaps, missing predecessors or unknown target generations are BLOCKED.

The lifecycle audit distinguishes:
- **event time:** when the event is asserted to have occurred;
- **ingestion time:** when the event entered the evidence stream;
- **effective interval:** the period of observations or product cohorts to which the event applies.

A late-arriving event can therefore affect claims whose evidence was generated during an earlier effective interval. A future-effective event does not invalidate a claim before its effective time.

S0 only checks timestamp consistency and that a time-source reference is present. It does not authenticate clock trust, estimate time uncertainty, or determine the physical cause of an event.

## Trigger policy

| Event | S0 policy |
|---|---|
| Calibration expired / out of tolerance | SUSPEND claims whose dependency closure includes that calibration |
| Sensor witness/self-test failure | SUSPEND claims whose required observation path depends on the failed witness |
| Process drift | SUSPEND process-dependent claims |
| Material / supplier / adhesive / head change | SUSPEND affected claims pending #12002 bridge or requalification |
| Test-method scope/edition change | SUSPEND claims depending on that method receipt |
| Suspected field failure | SUSPEND related reliability claims pending review |
| Confirmed source retraction | REVOKE claims depending on the retracted source |
| Review due / validity deadline elapsed | EXPIRE only the scoped claim |
| Qualified successor generation | SUPERSEDE predecessor claims for future scope |

A trigger with unknown dependency closure cannot default to whole-family PASS. An event outside a claim's dependency keys must not cause broad invalidation. When the dependency closure itself is not resolved, the full integrated system should BLOCK rather than assume the explicit key list is complete.

## Implementation

Branch: `research/plasma-qualification-lifecycle-v0-1`, stacked on `research/plasma-claim-types-v0-1` (draft PR #12028).

Files:
- `xtask/src/plasma_qualification_lifecycle.rs`
- `data/research/plasma_qualification_lifecycle_fixture_v0_1.json`
- `xtask/src/main.rs` (new command wiring)

Command:

```bash
cargo run -p xtask -- plasma-qualification-lifecycle-audit
```

The auditor:
- requires immutable qualification-generation identity, profile, authority and receipt references;
- validates method-measurement claim receipts using the shared method-scope evaluator from #12026;
- distinguishes other claim types (survival/reliability, transfer, process observation, surface state, material integrity, lifecycle recovery) without inventing a second method registry;
- validates claim-level dependency keys and closure references;
- checks generation predecessor/successor lineage and prevents branching successors;
- validates event identity, ordering, idempotency, source digest shape and time consistency;
- computes claim-level state using event-type policy, affected dependencies and effective-time overlap;
- preserves historical generations and requires a new generation for reinstatement.

The digest validator checks lowercase SHA-256 syntax only. It does not resolve, authenticate, retrieve or recompute the referenced closure or source artifact bytes. The generic typed evidence DAG in #6414 must resolve and qualify the actual dependency closure before this policy can be used for a production Passport.

## Frozen synthetic fixture

Fixture: `data/research/plasma_qualification_lifecycle_fixture_v0_1.json`

The fixture contains **28 synthetic known-answer scenarios**:

| Audit disposition | Cases |
|---|---:|
| PASS — event stream is structurally valid and expected claim states are emitted | 15 |
| BLOCKED — missing/unknown lineage, time or required evidence | 9 |
| REJECTED — contradictory IDs, timestamps, method scope or generation lineage | 4 |

Challenge families include:
- no-trigger current Passport;
- calibration expiry affecting one claim only;
- shared calibration drift affecting several claims;
- process drift affecting process-dependent claims;
- unrelated calibration event with no over-invalidation;
- exact source retraction causing claim revocation;
- suspected field failure causing scoped suspension;
- review deadline and validity-window expiry;
- successor generation superseding its predecessor;
- exact duplicate event replay;
- same event ID with conflicting digest;
- event sequence gap and absent predecessor;
- malformed dependency-closure digest;
- future-effective and pre-generation events;
- method/material/edition failure in a Passport's claim receipt;
- missing protocol digest, predecessor generation, successor, target generation or time-source reference;
- impossible event/ingestion ordering and reversed effective interval;
- generation-targeted event suspending every claim.

The current exact fixture was fetched back from GitHub and independently replayed in JavaScript: **28/28 expected dispositions and claim states matched** (15 PASS, 9 BLOCKED, 4 REJECTED).

## Baselines and SOTA comparison

A future blinded evaluation should compare the same event corpus and dependency graph against:
1. static certificate / recipe-name validity;
2. last-write-wins status records that overwrite prior state;
3. broad family-wide invalidation;
4. dependency-aware, event-sourced lifecycle replay with exact effective-time and claim-level scope.

Conventional QMS, MES, PLM and calibration systems already manage change control, calibration status, nonconformance and traceability. We should not claim they lack these functions. The differentiation to prove is a vendor-neutral evidence layer that links each downstream engineering claim to the exact test-method, specimen, process, calibration, material and lifecycle dependencies—and can replay the consequences of a new event without rewriting the historical record.

Industrial plasma providers such as Plasmatreat and Surfx already offer substantial process monitoring and logging. Luminous should consume those records and connect them to downstream claim admissibility and qualification lifecycle, not rebuild their generator controls.

Prospective software-contract targets:
- zero stale claims remain CURRENT on planted critical triggers;
- 100% recall of affected claims on known-truth dependency fixtures;
- zero unrelated false suspensions on exact known-truth negative controls;
- 100% preservation of historical generation and event ancestry;
- 100% detection of conflicting event-ID/digest reuse and missing sequence links;
- deterministic replay under event-array and generation-array permutation when explicit sequence order is unchanged;
- separate reporting of false suspension, missed suspension, revocation precision, affected-claim recall, invalidation latency, and review burden.

These are synthetic policy-audit goals, not measured reliability, safety integrity or real production quality outcomes. Zero synthetic false-current does not imply zero field risk.

## Further physical/operational promotion requirements

Before any production use, an accountable quality authority and the owners of #6414, calibration continuity, material lineage and product release would need to establish:
1. authentic event sources/signatures and trusted time semantics;
2. actual dependency-closure resolution and impact analysis;
3. exact product/lot and manufacturing effective-time mapping;
4. approved review/containment/requalification workflows;
5. independent validation against historical changes, calibration failures, source retractions and field nonconformances;
6. a documented human review and authority boundary.

This software must not automatically stop/restart plasma equipment, disposition product, recall lots, or authorize production release.

## Validation boundary

- Independent synthetic fixture replay: **28/28** exact expected dispositions and claim states matched.
- Eight Rust tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head CI are **not verified**.
- This branch does not modify `plasma_transfer.rs` or the existing #12006 transfer audit.
- The dependency closure and authority references are not authenticated here.
- No physical test, calibration, failure causality, material equivalence, bond lifetime, production containment, product qualification, safety, standards compliance, or restart/release authority is claimed.
