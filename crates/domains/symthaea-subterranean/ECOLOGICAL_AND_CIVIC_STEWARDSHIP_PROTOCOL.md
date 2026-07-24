# Ecological and Civic Stewardship Protocol

**System:** `symthaea-subterranean`  
**Campaign:** XIV  
**Date:** 2026-07-20  
**Status:** deterministic software protocol; environmental, cultural, civic, and regulatory legitimacy remain external

## 1. Purpose

This protocol governs site-bound groundwater protection, contamination accounting,
subsidence limits, protected ecological and cultural zones, community mandate scope,
and restoration obligations for subterranean autonomy.

The stewardship layer is safety-monotonic. It may reduce productive work to bounded
probing, require return, require hold, or preserve emergency containment actions. It
cannot add actuator authority, override a physical emergency, restore failed hardware,
or determine that an external authority is legitimate.

## 2. Trust boundary

The crate accepts externally established facts and authority records, including:

- environmental baselines and receptor limits;
- protected ecological, cultural, sacred, civic, and utility zones;
- community mandates and independent approvals;
- discharge-quality observations;
- contamination measurements;
- restoration-completion evidence.

The crate validates identity, geometry, bounds, freshness, replay order, role
separation, and resulting command authority. It does **not** establish land title,
Indigenous or community representation, sacred-site legitimacy, environmental
permits, laboratory accreditation, or regulatory approval.

## 3. Authority ordering

The relevant command authority order is:

1. malformed observation and hardware-frame rejection;
2. physical hazard assessment and latching;
3. return reserve and escape feasibility;
4. operator, team, degraded-operation, and partition constraints;
5. survivability envelopes and actuator isolation;
6. lifecycle assurance and terminal decommissioning;
7. ecological and civic stewardship projection;
8. deterministic recovery planning;
9. final independent runtime invariants;
10. physical actuation and post-step accounting.

Stewardship therefore constrains a candidate command before it executes. Physical
emergency recovery may still dewater, cool, seal, stabilize, or retreat where needed
to avoid trapping the machine or worsening an immediate hazard.

## 4. Site environmental baseline

A baseline is bound to one `site_id`, one `deployment_id`, and one revision. It
contains finite limits for:

- attributable groundwater drawdown;
- attributable surface settlement;
- uncontained contaminant release;
- discharged water volume;
- bounded environmental and civic receptors.

A baseline with an empty identity, invalid limit, duplicate receptor, or unsupported
schema is rejected before installation. Installing a new stewardship supervisor is
atomic: an invalid replacement cannot partially overwrite the active policy.

## 5. Protected zones

Protected depth zones may represent cultural heritage, sacred sites, ecological
reserves, community exclusions, or critical utilities. Each zone carries an external
authority reference, bounded geometry, a buffer, and explicit work/access rules.

The command semantics are:

- outside the look-ahead horizon: ordinary stewardship evaluation;
- buffered approach: bounded probing only;
- work-prohibited entry: productive excavation removed;
- access-prohibited entry: deeper movement removed and inward escape preserved;
- expired or unauthenticated authority: productive work removed while return remains possible.

A protected-zone rule must never trap a machine by removing safe inward motion.

## 6. Groundwater stewardship

Groundwater accounting is cumulative. Dewatering contributes to modeled drawdown and,
unless an external closed-loop treatment adapter says otherwise, discharged volume.

Discharge requires current externally verified quality evidence. Missing, stale, or
unacceptable quality evidence creates a hold-and-contain disposition while preserving
the dewatering needed for immediate safety. Approaching drawdown, discharge, or
aquifer-risk limits removes productive authority; exhausted limits require return.

## 7. Contamination stewardship

The contamination ledger records conservative estimated and externally measured
uncontained mass by class. Later measurements may refine evidence but cannot erase a
larger conservative estimate. Unknown disturbed material requires containment rather
than being treated as clean by default.

Candidate excavation is projected before actuation. A command that would exceed a
release threshold loses productive authority on the same control frame. Actual
cumulative accounting is updated only from the post-arbitration command that truly
executed.

## 8. Subsidence stewardship

The settlement model accumulates excavated volume, unsupported equivalent volume,
installed support, and estimated surface settlement. Settlement is monotonic in the
reference model and cannot be reset by a later favorable frame.

Projected settlement can reduce work, require return, or require hold and
stabilization before the modeled site limit is crossed. The model is an internal
reference model, not a substitute for calibrated site geomechanics or survey data.

## 9. Community mandate

A community mandate is bound to one site and deployment, one epoch/sequence stream, a
validity interval, depth and disturbance limits, allowed activities, a public evidence
reference, and independent authenticated roles.

At minimum, a Community Representative and Environmental Observer must be distinct,
hardware-backed, and externally authenticated. Stale epochs and replayed sequences
are rejected. Expiry, revocation, missing authority, or scope excess removes
productive work while preserving return.

The crate verifies the structure and authority effect of the mandate. It cannot judge
whether the named representatives legitimately speak for a community.

## 10. Restoration obligations

Disturbance may create bounded obligations to seal bores, stabilize roofs, treat
water, remove spoil, restore habitat, or monitor groundwater. An obligation advances
through open, in-progress, awaiting-evidence, and complete states.

Completion requires both:

1. sufficient physical progress; and
2. a non-empty externally verified evidence reference.

Elapsed time, mission completion, or a cached boolean can never complete restoration.
Overdue obligations require return; near-due obligations prevent new productive work.

## 11. Persistence and evidence

Operational checkpoint schema v7 preserves the complete stewardship supervisor and
rejects malformed or identity-mismatched state before activation.

Each bounded operational evidence frame records:

- overall stewardship disposition and command authority;
- groundwater drawdown, discharge, and quality status;
- contamination mass, class, budget, and containment status;
- excavation, support, settlement, and settlement fraction;
- protected-zone kind, identity, distance, authority freshness, and access status;
- mandate identity and scope status;
- restoration progress and overdue obligations.

## 12. Release evidence

Seven canonical requirements (`SUB-EST-001` through `SUB-EST-006` and
`SUB-CIV-001`) are release blocking and linked to deterministic contracts.

The community-facing stewardship bundle binds:

- site and deployment identity;
- source tree and build identity;
- live stewardship supervisor and recomputed assessment;
- complete validation results;
- canonical release requirements;
- a public evidence reference;
- distinct authenticated Community Observer and Environmental Verifier attestations.

The included deterministic digest is for reproducibility tests only. Production must
inject cryptographic hashing and signature verification.

## 13. Explicit non-claims

This protocol does not establish:

- environmental-impact assessment approval;
- legal land access or mineral rights;
- community consent or representative legitimacy;
- Indigenous consultation or free, prior, and informed consent;
- sacred-site identification;
- laboratory accreditation or measurement uncertainty;
- calibrated hydrogeological, geochemical, or geomechanical prediction;
- remediation completeness;
- cryptographic authenticity;
- production readiness.

## 14. Production qualification still required

Before deployment, the real Rust 1.94 workspace must pass formatting, Clippy, full
tests, physical HIL, chamber testing, calibrated hydrogeological and geomechanical
models, accredited water and contamination measurements, independent community and
environmental review, cryptographic provenance, field trials, and every applicable
jurisdictional permitting and consultation process.
