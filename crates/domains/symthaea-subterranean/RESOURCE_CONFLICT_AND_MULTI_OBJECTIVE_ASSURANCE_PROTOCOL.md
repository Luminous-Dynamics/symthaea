# Resource-Conflict and Multi-Objective Assurance Protocol

**Campaign:** XIX
**Date:** 2026-07-20
**Scope:** `symthaea-subterranean`

## Purpose

A subterranean platform frequently faces several individually legitimate demands at the same time: preserve immediate physical safety, retain enough energy to return, contain environmental harm, protect failing hardware, complete restoration, assist a distressed peer, maintain communications, and continue productive work. A safe system cannot resolve those conflicts through an opaque scalar reward or by allowing the most recent requester to consume the remaining reserve.

This protocol defines explicit normalized resource budgets, protected objective classes, deterministic arbitration, starvation and service-fairness evidence, same-frame command restriction, restart continuity, and release evidence. The protocol is safety-monotonic: resource arbitration may throttle work, require return, or hold for review. It cannot create actuator authority or weaken physical hazards, final-command invariants, operator restrictions, lifecycle restrictions, stewardship constraints, or decommissioning.

## Objective classes

The canonical objective set is:

- **Physical safety** — cooling, dewatering, sealing, stabilization, and other immediate recovery needs.
- **Return reserve** — energy, thermal headroom, time, and recovery capacity needed to preserve a feasible route home.
- **Environmental containment** — groundwater, contamination, settlement, exclusion-zone, and restoration containment needs.
- **Asset integrity** — service or withdrawal required to avoid modeled component failure.
- **Restoration** — previously incurred site obligations.
- **Peer assistance** — accepted and feasible rescue or team support.
- **Communications** — relay and mesh continuity obligations.
- **Mission work** — surveying, boring, sampling, and other discretionary production.

Objectives are classified as protected, obligatory, or discretionary. Protected objectives are admitted first. Obligatory and discretionary objectives may use only the remaining capacity after the protected reserve is withheld.

## Resource representation

Each objective demand uses a bounded four-dimensional vector:

- battery capacity;
- thermal headroom;
- mission time;
- recovery hardware capacity.

Values are normalized to `[0, 1]`. The representation is an operational admission model, not a substitute for calibrated physical energy, thermal, or reliability models. Invalid or non-finite inputs fail closed as an immediate physical-safety demand.

## Protected reserve

The runtime derives a reserve from live battery state, thermal margin, return-energy margin, and available recovery equipment. Discretionary work cannot consume that reserve. A protected objective that cannot be funded produces a review hold rather than allowing a lower-class objective to proceed.

Protected allocation occurs before urgency ordering. Urgency may prioritize objectives inside the same class, but cannot allow mission work to displace return, safety, or environmental containment.

## Deterministic arbitration

Active demands are ordered by:

1. objective class;
2. urgency plus retained starvation debt;
3. stable objective identity.

This ordering is deterministic. Equal-score conflicts do not depend on hash-map iteration, arrival order, or nondeterministic scheduling.

The supervisor emits one of four authority states:

- **Nominal** — all admitted demands fit within the available envelope.
- **Throttled** — reduced productive work may continue inside the funded envelope.
- **ReturnOnly** — productive work stops and only protected withdrawal and recovery remain.
- **HoldForReview** — movement and productive work stop while safety-preserving recovery actions remain available.

The command transformation can only reduce authority. It never creates track, cutter, auger, ballast, or recovery demand that was absent from the incoming command.

## Starvation accounting

Every active objective retains bounded service history:

- consecutive deferred steps;
- total deferred steps;
- total served steps;
- warning or critical starvation disposition;
- whether a protected objective was deferred.

Starvation debt influences priority only within the existing objective-class ordering. It cannot promote a discretionary objective above a protected objective and cannot create actuator authority. Critical debt requests replanning or review rather than silently forcing motion.

## Stakeholder service fairness

Where an objective represents an attributable stakeholder or service beneficiary, the ledger records requested and delivered service. It retains bounded service ratios, under-service evidence, and a Jain-style aggregate fairness index.

Fairness is diagnostic and constraining. It may throttle a mission that monopolizes shared capacity, but it cannot compel unsafe rescue, override consent, consume return reserve, or bypass environmental and lifecycle obligations.

## Runtime authority order

Relevant deployed ordering is:

1. physical state, hazards, return feasibility, and team constraints;
2. lifecycle, stewardship, epistemic, and temporal constraints;
3. resource-conflict assessment and mission override;
4. operator authority and physical recovery planning;
5. field envelope, maintenance, and actuator isolation;
6. formal transition assurance;
7. final-command invariant enforcement;
8. plant actuation and evidence retention.

A later stage may remove additional authority. No stage may recreate authority removed earlier.

## Persistence

Operational checkpoint schema version 12 persists:

- current resource-conflict authority and disposition;
- retained objective starvation debt;
- stakeholder service ledger;
- selected, deferred, and protected-deferred objectives;
- resource consumption and remaining envelope.

The entire checkpoint validates before live state changes. Older checkpoints default to no resource-conflict authority and are then subject to ordinary runtime reassessment.

## Evidence and explanations

Every final command may record:

- resource-conflict authority and disposition;
- selected and deferred objectives;
- protected objectives that could not be funded;
- consumed and remaining resource vectors;
- maximum capacity fraction;
- maximum deferred duration;
- service fairness index and underserved stakeholders;
- explicit arbitration reasons;
- the command before and after resource arbitration.

Counterfactual explanations may identify resource conflict as the reason more work or movement was not authorized. Explanation is observational and cannot alter the command.

## Release requirements

The canonical registry contains:

- `SUB-RES-001` — protected resource-budget preservation;
- `SUB-RES-002` — protected-objective priority;
- `SUB-RES-003` — bounded objective-starvation detection;
- `SUB-RES-004` — attributable service-fairness accounting;
- `SUB-RES-005` — checkpoint continuity without authority expansion.

Seven deterministic contracts cover protected-first admission, reserve preservation, deterministic tie-breaking, starvation monotonicity, fairness monotonicity, same-frame command restriction, and checkpoint continuity.

A self-consistent evidence bundle binds build identity, live supervisor state, recomputed validation results, and distinct externally authenticated, hardware-backed Safety Reviewer and Resource Governance Reviewer attestations. The built-in digest is deterministic for reproducibility only and is not cryptographic authentication.

## Explicit non-claims

This protocol does not claim:

- that normalized resources replace calibrated physical units;
- that the selected fairness metric represents social legitimacy;
- that every stakeholder or obligation has been identified;
- that urgency values are ethically or legally authoritative;
- that deterministic reviewer digests are cryptographic signatures;
- that resource feasibility proves mission safety;
- that simulation validates real battery, thermal, recovery, or environmental dynamics.

Production release requires calibrated resource models, externally governed objective definitions, independent stakeholder review, real workspace compilation, hardware-in-the-loop exhaustion and conflict campaigns, cryptographic evidence provenance, and field validation.
