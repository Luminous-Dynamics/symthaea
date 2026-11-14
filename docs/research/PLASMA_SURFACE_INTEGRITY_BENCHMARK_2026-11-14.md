# PLASMA-SURFACE-INTEGRITY-001 — constrained substrate preservation under plasma activation

**Date:** 2026-11-14  
**Issue:** [#12086](https://github.com/Luminous-Dynamics/symthaea/issues/12086)  
**Parent application benchmark:** [PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795)  
**Branch:** `research/plasma-surface-integrity-v0-1`  
**Stack:** #12074 → #12087 → this branch

## Executive result

The first S0 surface-integrity auditor and fixture are implemented. It formalizes an important manufacturing constraint:

> An activation process must meet the product's bondability requirement without violating predeclared limits on substrate function.

The decision is not based on a weighted scalar. Hard substrate constraints are conjunctive. A successful bond result or improved water contact angle cannot compensate for a barrier or mechanical property that exceeds its permitted degradation.

This is a focused child of #11795. The parent issue already owns bondability lifetime, untreated/incumbent/plasma benchmark arms, nominal/under/over-treatment controls and physical promotion requirements. This increment adds typed hard limits for barrier and bulk substrate properties.

## Why this changes the optimization problem

Three recent findings make an unconstrained “maximize activation” objective untenable:

1. A February 2026 LDPE/HDPE cold-argon plasma study reported improved wettability and surface free energy alongside increases of up to 60% in water-vapour permeability and 45% in oxygen permeability, with partial hydrophobic recovery. The numeric changes are specific to that study's film and conditions; they are not universal plasma effects.
2. The October 2026 PP/HIPS/PVC/PC/PET comparison reported that the lowest water contact angle did not consistently produce the highest bond strength. It also reported visible degradation in several polymer conditions under longer nitrogen exposure; argon had a wider process window in the tested systems.
3. A June 2026 acrylate-joint study found that plasma alone did not provide long-term water resistance in its tested system without an additional adhesion promoter.

The objective should therefore be **constrained surface-state engineering**, not minimum contact angle, maximum surface energy or maximum discharge dose.

## Standards and method scope

The new method registry entries are exact-edition and endpoint-specific:

| Method | Registered endpoint | Claim ceiling |
|---|---|---|
| ASTM D882-26 | Thin-film tensile properties | Tensile observation for plastic film / thin sheeting below 1.0 mm; not a universal safety or product-integrity certificate |
| ISO 15106-5:2015 | Water-vapour transmission rate | WVTR measurement using the pressure-sensor method for plastic films, sheeting and multilayer structures |
| ISO 15105-2:2025 | Gas-transmission rate | Equal-pressure gas-transmission observation for films, sheeting, laminates, co-extruded and flexible plastic-coated materials |

ASTM's D882-26 listing places specimens below 1.0 mm in this method and directs thicker specimens to the applicable alternative method. The S0 auditor additionally checks the measured substrate profile thickness; D882 is blocked at or above 1.0 mm.

ISO lists 15106-5:2015 as confirmed in 2025 and scoped to plastic film, sheeting and multilayer structures. ISO 15105-2:2025 is the published second edition for equal-pressure gas transmission across films, sheet, laminates, co-extruded and flexible coated material.

A current-standards caution: ISO 2528:2017 remains listed as confirmed, but a fourth edition dated 2026-11 is currently under publication and will replace it. The S0 registry does not pin that unpublished edition. A lab must obtain the controlled current text before specifying a method that may have changed.

Official references:
- ASTM D882-26: https://store.astm.org/d0882-26.html
- ISO 15106-5:2015: https://www.iso.org/standard/62026.html
- ISO 15105-2:2025: https://www.iso.org/standard/91001.html
- ISO 2528:2017 current listing: https://www.iso.org/standard/72382.html
- ISO 2528 fourth edition under publication: https://committee.iso.org/standard/88372.html

These methods support observations only within their stated sample preparation, specimen thickness, gas, temperature, humidity, apparatus and conditioning scope. They do not independently establish package safety or shelf life.

## S0 contract

### Application and substrate identity

The case binds:
- substrate grade, supplier, lot, product function, thickness and surface finish;
- substrate profile digest and frozen constraint-plan digest;
- declared minimum confidence;
- plasma article, head, gas species and purity;
- delivered power, gas flow, standoff, line speed and pass count;
- treatment-to-bond delay and maximum allowed delay;
- exposure profile digest, raw trace digest and time-source reference.

The current fixture is synthetic; these digest fields are syntactically checked and cross-bound, not authenticated against physical instrument files.

### Hard constraints

Every required property is evaluated as a **treated / baseline ratio**. Each constraint declares:
- endpoint and method-qualified baseline/treated measurement receipts;
- whether the application requires a maximum ratio or minimum retention ratio;
- the frozen application-specific limit;
- a lower and upper uncertainty bound from a linked analysis result;
- the design and sample manifest, independent unit, grouping, plan freeze and result digest.

For `maximum_ratio` (e.g. barrier transmission), the upper interval bound must be at or below the limit to pass. If the lower bound already exceeds the limit, the constraint is REJECTED. If the interval crosses the limit, the result is BLOCKED.

For `minimum_ratio` (e.g. tensile-property retention), the lower interval bound must be at or above the required floor to pass. If the upper bound is below the floor, the constraint is REJECTED. If the interval crosses the floor, it is BLOCKED.

The ratios and limits are examples of application-specific engineering contracts, not universal tolerances. The code does not compute the ratios or intervals; it checks the declared analysis receipt and evidence linkage.

### Independent experimental units

The S0 contract supports:
- `independent_arms`: baseline and treated arms must not share manufacturing clusters;
- `paired_by_cluster`: both arms must include the exact same cluster set, while specimen identities remain distinct.

Recognized units include material lot, film roll, production day, production line and manufacturing site. Multiple observations from one film roll are not treated as multiple rolls.

### No proxy override

Contact-angle evidence may be included as a diagnostic. It never contributes to surface-integrity PASS. At least one hard substrate-property constraint is mandatory. A contact-angle-only or visual-only “no damage” claim is BLOCKED.

The overall disposition is conjunctive:
- any hard property clearly outside its limit → REJECTED;
- missing/insufficient evidence or uncertainty crossing a limit → BLOCKED;
- all declared hard constraints structurally evidenced and within limits → PASS / `ELIGIBLE_WITHIN_DECLARED_LIMITS`.

“Eligible” means only within the limits declared by that application receipt. It is not a product release.

## Implementation

Files:
- `xtask/src/plasma_surface_integrity.rs`
- `data/research/plasma_surface_integrity_fixture_v0_1.json`
- `xtask/src/plasma_claim_scope.rs` (registry extension)
- `data/research/plasma_claim_scope_fixture_v0_4.json`
- `xtask/src/main.rs` (CLI wiring)

Command:

```bash
cargo run -p xtask -- plasma-surface-integrity-audit
```

The claim-scope registry now admits:
- ASTM D882-26 thin-film tensile observation;
- ISO 15106-5:2015 WVTR observation;
- ISO 15105-2:2025 gas-transmission observation.

It rejects promotion of these observations to product safety, food-contact safety, barrier-safety guarantees or packaging shelf life.

## Frozen challenge results

The surface-integrity fixture contains 18 synthetic known-answer cases:

| Disposition | Cases |
|---|---:|
| PASS | 3 |
| BLOCKED | 8 |
| REJECTED | 7 |
| **Total** | **18** |

It covers:
- barrier and tensile limits met;
- improved wettability and a claimed good bond with barrier degradation beyond the hard limit;
- an uncertainty interval crossing the barrier limit;
- contact-angle-only and visual-only cases;
- D882 specimen thickness outside scope;
- baseline/treated receipt versus specimen-manifest mismatch;
- treatment-to-bond delay beyond the declared limit;
- higher contact angle with substrate integrity still inside declared limits;
- tensile degradation below its hard retention floor despite a claimed bond pass;
- paired film-roll controls;
- independent-arm roll overlap;
- post-outcome design change;
- insufficient confidence;
- mismatched exposure digest;
- unrecognized method edition and unsupported material class.

The v0.4 method-scope fixture contains 61 cases (16 SUPPORTED, 14 BLOCKED, 31 REJECTED).

Latest exact-branch source and fixtures were fetched back and independently replayed in JavaScript:
- surface-integrity fixture: **18/18** expected dispositions;
- method-claim fixture: **61/61** expected dispositions.

This is synthetic contract replay. It is not a physical experiment or a Rust execution.

## Paradigm shift and benchmark programme

The relevant comparison is not only plasma generator telemetry. It is how process selection handles competing product outcomes.

Compare:
1. contact-angle-only optimization;
2. process-parameter optimization without a hard substrate-damage veto;
3. weighted scalar scoring where excellent bond performance can mask a catastrophic substrate loss;
4. constrained multi-property optimization with hard integrity limits and explicit uncertainty.

Once all hard limits are met, use Pareto selection over:
- bondability window and bond durability;
- barrier-property retention;
- tensile / elongation retention;
- energy per accepted bonded area;
- throughput and treatment-to-bond buffer;
- consumables, extraction and cooling burden;
- scrap and rework.

Never allow a weighted average to hide a predeclared hard-limit violation.

Prospective targets—not achieved results:
- zero eligible/PASS outcomes when any hard substrate limit is violated;
- 100% detection of planted over-treatment and treatment-to-bond-window violations;
- 100% detection of method, specimen, cluster, profile and exposure-digest mismatch;
- uncertainty intervals calibrated on grouped held-out lots/rolls, grades, thicknesses and plasma heads;
- report safe-window width, barrier retention, mechanical retention, bondability window, throughput, total system energy and false-accept separately;
- use independent lot/roll/day/site holdouts; no random coupon split as the only generalization test.

## Ownership / non-duplication

- #11795 owns the primary bondability-lifetime benchmark, baseline arms and physical promotion path.
- #11952 owns spatial support and registration.
- #12026/#12074 owns exact method-to-claim scope.
- #12070 owns preregistered cluster-aware design/result linkage.
- #12087 composes method/design evidence into transfer decisions.
- #11971 owns aging and survival analysis.
- #6414 owns the generic evidence DAG and invalidation closure.
- Statistics owns numerical interval estimation and model validity.

No duplicate statistics engine, calibration registry, material-lot authority, DOE engine or plasma controller is introduced.

## Transfer integration boundary

The surface-integrity audit is currently a separate application-qualification command. It is **not yet a mandatory prerequisite** inside `plasma-transfer-audit` or the Bond Qualification Passport. That separation is intentional for this increment: the current transfer fixture is a PP-GRADE-A rigid-plastic single-lap profile, while these substrate hard limits need an application-specific product-function profile and controlled baseline data. A follow-on integration should add an exact `surface_integrity_case_id` to the transfer bridge and bind its substrate/exposure digests to the same material lot and process profile before using the result in a transfer decision.

## Validation boundary

- Surface-integrity synthetic replay: **18/18**.
- Method-claim synthetic replay: **61/61**.
- Nine surface-integrity Rust tests and eleven claim-scope Rust tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- The audit does not calculate WVTR, OTR, tensile retention, uncertainty intervals or bond strength.
- Referenced artifact bytes, calibration, lab custody, clock trust, actual sample/roll identity and standard execution are not authenticated.
- No physical substrate integrity, actual safe activation window, bond durability, product qualification, safety, standard compliance or production release is established.


## 2026-11-14 — method applicability is now bound to the substrate profile

The substrate profile now contains an explicit `supported_method_material_classes` list. An individually valid WVTR, gas-transmission or tensile receipt is BLOCKED when the profile does not declare that method-material class as applicable. The D882 path also checks the actual substrate thickness is below 1.0 mm.

The expanded fixture now contains **18 cases** (3 PASS, 8 BLOCKED, 7 REJECTED), including a valid method receipt whose material class is omitted from the substrate profile. The exact updated fixture independently replays **18/18**. The claim-scope v0.4 registry independently replays **61/61** (16 SUPPORTED, 14 BLOCKED, 31 REJECTED).

The method-material class list and profile digest are structurally cross-bound; the underlying material specification, supplier/lot records and test artifacts are not authenticated by this S0 auditor.
