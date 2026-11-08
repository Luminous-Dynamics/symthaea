# Marine mass properties, weight growth and loading-condition ledger — V0.1

**Status:** mass-accounting contract and synthetic fixtures; no actual LOS-18 mass, displacement, centre of gravity, stability or seaworthiness is established.  
**Parent:** [MARINE-MASS-STATE-001 #11074](https://github.com/Luminous-Dynamics/symthaea/issues/11074)

## 1. Purpose and authority boundary

Create one revision-bound ledger for every mass item, tank/fluid state and loading condition in the LOS family. Reconcile mass and first moments to a named datum, preserve uncertainty, track weight growth, and emit a handoff to qualified hydrostatics/stability engineers.

This is not a hydrostatics solver, stability solver, inclining-test procedure, loading-manual approval, weight-control authority, component identity system or physical measurement ontology. It consumes component/as-built references, quantities and observations from their canonical owners.

## 2. Reuse canonical owners

- LOS mission and coupled mass/volume/energy requirement: #10926.
- Catamaran and comparative hull engineering: #7719.
- Hydrodynamics and displacement/CG observables: #8082.
- Cargo/tank payload and loading hard constraints: #7444.
- Symtropy shipyard DFM/build sequence: #2048.
- Symtropy component identity/as-built/inspection lineage: #3331.
- Marine instrumented test/campaign evidence: #11064.
- At-sea module removal context: #11049.
- Exact mission evidence join: #10988.

The mass-state profile must not create a parallel part/serial ledger, measurement/calibration ontology, hydrostatic model or shipyard scheduler.

## 3. Paradigm shift: weight is a lifecycle state

Follow each item through requirements, design estimate, supplier value, receipt, fabrication, installation, measured as-built, operation, consumption, removal, repair and replacement. Do not overwrite an estimate with a measured value; retain the state and evidence class.

The same component must not be counted twice because it appears in both an assembly BOM and a child list. If a parent assembly mass is used, its child detail rows are reference-only for that condition. If individual child masses are summed, the parent aggregate is reference-only.

Mass and centre of gravity are inseparable for stability use. Total mass can still be reported when an item's location is unknown, but CG eligibility becomes indeterminate. A location with no named datum/frame is not a usable coordinate.

## 4. Exact coordinate and identity contract

Each ledger binds:
- exact vessel design/as-built configuration digest and mission/condition identity;
- one named vessel datum and coordinate-frame revision;
- axis sign convention: longitudinal, transverse, vertical; vertical positive upward;
- each mass item identity, parent/child lineage, quantity and revision;
- mass and uncertainty in kg; item CG coordinates in m in that same datum/frame, with uncertainty;
- evidence class (synthetic, assumed, supplier-declared, engineering-estimate or measured) and source;
- explicit counted-mass versus reference-only role;
- tank-state and fluid-property references;
- loading-condition membership and temporary state transitions;
- mass budget, hard maximum and reserved weight-growth margin.

A frame change, as-built revision change, component substitution, module removal or tank-state change invalidates dependent load-case reports until recomputed.

## 5. Fluid/tank state is a moving mass distribution

A tank's empty shell is one mass item. Its contents are separate, condition-specific mass.

Each tank state binds exact tank geometry/revision, fluid type, usable capacity, fill fraction, volume, density and temperature/source, liquid mass, fluid CG location/uncertainty, transfer state and free-surface review status.

Do not compute fluid mass from percentage alone without capacity/geometry and density. Do not assume tank-fluid CG remains fixed as fill changes. Part-filled/slack tanks may require free-surface correction; this ledger records the need and a qualified reviewer reference, but does not calculate that correction.

## 6. Loading-condition corpus

The minimum LOS condition family should include:
- lightship/as-built;
- departure with crew, stores, fuel, fresh water, tender and payload;
- arrival with changed fluid and consumable levels;
- anchorage/living condition;
- full tanks;
- partial/slack tanks;
- equipment removed for service and stowed aboard;
- optional payload/tender/anchor changes;
- resupply and provisioning;
- weight-growth after installation;
- unknown high-mounted equipment;
- degraded states that alter mass only where the physical item state genuinely changes.

Damage stability, flooding progression and emergency stability are not approximated by this ledger. They require the applicable qualified analyses and procedures.

## 7. Arithmetic and uncertainty

For admitted counted items and tank fluids, the nominal mass/first moments are:

- Total mass: sum of all counted item and fluid masses.
- Longitudinal first moment: sum of mass × longitudinal coordinate.
- Transverse first moment: sum of mass × transverse coordinate.
- Vertical first moment: sum of mass × vertical coordinate.
- LCG/TCG/KG: corresponding first moment divided by total mass.

The implementation must preserve source unit, uncertainty and correlations. A conservative interval bound may be used for the first oracle; do not assume independent uncertainties or manufacture a tight confidence interval without covariance/evidence. The nominal equations are accounting, not hydrostatics.

If a counted item's mass is unknown, total mass is indeterminate. If mass is known but a counted item's position is unknown, mass can be reported but CG cannot. If a fluid mass or fluid CG is unknown, the same rule applies to that loading case.

Weight growth:
- growth used = as-built mass − frozen design baseline mass;
- compare against the approved growth reserve and hard maximum;
- exceeding either is an engineering change/re-analysis trigger, not a scheduler optimization opportunity.

## 8. Independent verifier and negative controls

A future Rust verifier must independently reconstruct mass totals and first moments, not trust a caller-supplied validity boolean or total-mass field.

Reject or mark indeterminate:
- counted parent plus counted child;
- duplicate item IDs or duplicate tank state for one tank/condition;
- wrong configuration/frame/datum;
- stale/superseded item revision;
- mass or location silently defaulted to zero;
- unknown density/geometry with claimed fluid mass;
- fluid mass inconsistent with volume × density beyond declared tolerance/uncertainty;
- unknown CG position where a CG claim is requested;
- partial tank with free-surface review pending but downstream stability marked eligible;
- module removed ashore but still included, or removed and stowed aboard without a new location;
- mass growth beyond reserve/hard limit while still marked within budget;
- synthetic/estimated data promoted to measured;
- load cases assembled from different vessel configuration digests.

## 9. Benchmarks — targets, not achieved results

- 100% of counted items have unique ID, revision, mass unit, evidence class and inclusion role.
- 100% of CG-eligible items have location in one exact datum/frame and uncertainty status.
- Zero duplicate mass counting in frozen hard-negative controls.
- 100% of load cases bind one vessel/as-built configuration and item-state snapshot.
- 100% of fluid/tank states preserve density/temperature/fill/geometry and free-surface status where relevant.
- 100% of mass growth is reported against design baseline, reserved margin and hard maximum.
- Independent oracle agreement on all supported synthetic mass and first-moment cases.
- Zero CG output when any counted position/frame is unknown.
- Zero stability eligibility when partial-tank/free-surface review is required but missing.
- Report mass, LCG/TCG/KG, payload/deadweight, tank mass, growth margin and module/service penalty separately.
- No percentage mass-saving target before a sourced design baseline and actual measured article exist.

## 10. Physical evidence ladder and standards

- ISO 12217-1:2022 remains published but ISO lists it at stage 90.92 (to be revised); ISO/DIS 12217-1.2 Edition 5 is in development/enquiry for non-sailing boats. Part 1 is principally applicable to mechanically/human powered craft from 6 m to 24 m hull length and includes habitable-multihull inversion/escape considerations. It assigns design category against design and maximum load; it is not a mass-ledger schema. https://www.iso.org/standard/79072.html
- ISO/DIS 12217-2.2 Edition 5 is the sailing-boat draft in enquiry. Do not apply Part 2 to a powered-only design merely because it has twin hulls. https://www.iso.org/standard/88606.html
- ITTC Recommended Procedure 7.5-02-07-04.7, Inclining Tests, Revision 01, effective 2024, describes draft measurements, angles and hull geometry with numerical processing to determine displacement mass and longitudinal/transverse/vertical CG. Actual inclining tests require competent naval-architecture planning, calibrated measurements and the applicable procedure; synthetic ledger arithmetic does not replace them. https://www.ittc.info/media/11946/75-02-07-047.pdf

The exact stability standard/class/flag pathway depends on final hull length, powered/sailing definition, use, operating waters and flag. Confirm applicability before design freeze and obtain licensed standards.

## 11. Claim ceiling

A schema or synthetic accounting PASS does not establish actual displacement, CG, free-surface correction, intact/damage stability, structural adequacy, seakeeping, design-category assignment, seaworthiness, class/flag/SAMSA compliance or permission to operate. Real claims require the exact physical vessel, measured mass/CG/tank data, uncertainty, qualified hydrostatics/stability analysis, professional review and competent authority acceptance.
