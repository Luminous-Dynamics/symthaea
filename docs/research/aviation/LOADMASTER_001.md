# LOADMASTER-001 — Evidence-bound aircraft loading

**Status:** research prototype / design contract, version 0.1  
**Owner system:** Symthaea  
**Simulation counterpart:** Symtropy  
**Spatial counterpart:** Sol-Atlas  
**Trust/provenance counterpart:** Mycelix  
**Operational status:** not approved for real aircraft loading, dispatch, air traffic control, or flight.

## 1. Paradigm shift

Treat loading as a **configuration-bound, evidence-carrying engineering problem**, not a spreadsheet with a single total-weight cell. The load plan must be evaluated against the exact aircraft variant, serial/configuration revision, approved weight-and-balance source, station definitions, and relevant operational phase. A number without provenance is not a safe number; a geometrically feasible placement without securement evidence is not a safe load; a passing local leg does not prove that the multimodal mission can be completed.

The initial Rust crate `symthaea-aircraft-loading` audits a single declared mass state: empty mass/moment + item masses at declared stations, station caps, maximum take-off mass, and one uniquely applicable declared CG envelope band. It accepts only `LoadingPhase::Takeoff`; Ramp, Taxi, and Landing states are rejected until phase-specific profiles and calculations exist. It returns `BlockedByDeclaredLimit`, `NotAssessable`, or `WithinDeclaredEnvelopeForHumanReview`. The final state is explicitly **human-review-only**, never a load release.

## 2. Initial audit contract

A profile declares:
- aircraft/configuration identity and exact revision;
- source locator and source SHA-256 digest syntax;
- an approval-evidence locator;
- empty mass and moment, maximum take-off mass;
- station ID, arm, and aggregate station-load cap;
- mass-dependent forward/aft CG envelope slices.

A scenario declares matching aircraft/configuration identity and a list of uniquely identified items, each with kind, station, positive mass, and (for cargo, occupants, and ballast) a securement-evidence locator. Inputs are SI units. Calculations reject non-finite values, overflow and detectable floating-point precision loss; the independent integration oracle covers 900 bounded integer mass/arm combinations and an exact inclusive-boundary case. Unknown station IDs and profile revisions fail the audit rather than being guessed.

A CG band's mass interval is inclusive. If no band matches, or more than one band matches, the CG test is **not assessable**; the engine must not pick a convenient band. Numeric exceedances take precedence in the disposition even if other evidence is missing, so a known hard failure is never hidden by an unknown.

## 3.1 Bounded uncertainty envelope (new research slice)

The crate now has a separate `analyze_loading_uncertainty` function, with explicit intervals for empty-aircraft mass/moment and each loaded item's mass/effective station arm. It returns outward-rounded bounds for total mass, total moment and center of gravity. It requires one uncertainty record per scenario item, rejects duplicate/missing/extra item identities, and requires the intervals to contain all nominal values.

The center-of-gravity interval is computed by interval division over independent total-moment and total-mass bounds. This intentionally ignores correlations between numerator and denominator, so the interval can be wider than the physically attainable values. That is acceptable for an uncertainty screen but could make the result uninformative; it must not be interpreted as CG-envelope approval. Comparing the result against mass-dependent approved CG bands and accounting for correlated measurement/calibration errors remain separate follow-on work.

An independent integration oracle enumerates all 64 endpoint combinations of the two-item synthetic fixture and checks the mass, moment and CG values all lie inside the returned bounds. Additional negative controls reject missing intervals and ranges that exclude the nominal profile/item values. These checks passed on exact code head `bd8a31dfabbb45c3805598a12c7949e4f559da39` in [workflow 39423836970](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39423836970); subsequent edits to this paragraph are documentation-only.

## 3. Safety and evidence boundary

Presence of a URL/reference and a well-formed SHA-256 string does not authenticate source bytes or establish that an authority approved the profile. The profile can be wrong, stale, or mismatched to the tail/configuration. A securement reference is only a locator; the engine does not verify the evidence or calculate tie-down forces.

Not implemented in v0.1:
- zero-fuel, ramp, taxi, take-off, landing, and fuel-burn phase-specific states;
- fuel transfer, passenger movement, cargo shifts, and loading-sequence stability;
- container/ULD geometry, cargo-bay collision/fit, local floor pressure/running loads, structural limits, and restraint/securement calculations;
- dangerous-goods compatibility and segregation, lithium-battery/fire-containment checks, live weather, runway performance, take-off trim, or aircraft performance;
- aircraft maintenance/airworthiness, dispatch release, crew decisions, or real-time ATC instructions.

The FAA explains that approved flight-manual limitations govern cargo location, method, and type; its weight-and-balance material describes mass/moment calculations and aircraft-specific CG limits. This prototype operationalizes only a small part of that reasoning and cannot replace the approved programme.

## 4. System architecture

| System | Responsibility | Contract boundary |
|---|---|---|
| **Symthaea** | Reason over declared aircraft, engineering, cargo, and maintenance evidence; surface contradictions, alternatives, and uncertainty. | The loadmaster crate is a deterministic arithmetic auditor, not an autonomous dispatcher. |
| **Symtropy** | Digital-twin scene for cargo-bay geometry, load sequence, restraints, maintenance state, manufacturing/process simulation, and replayable what-if scenarios. | Simulation results need model version, assumptions, uncertainty, and independent physical validation. |
| **Sol-Atlas** | Spatial graph of airports/airfields, runways, airspace constraints, weather/exposure, energy, ground links, and regional service dependencies. | Route/airport suitability is a planning assessment, never a clearance or substitute for ATC. |
| **Mycelix** | Signed provenance/credentials for manifests, inspection records, supplier lots, aircraft configurations, and operator handoffs; privacy-preserving audit receipts. | Put hashes/attestations on shared infrastructure only where appropriate; sensitive passenger, cargo, and security data remains access-controlled. |

This supports a shared **mission contract** across aircraft, road/rail/ship links, hubs, payload condition, handoffs, deadlines, and recovery. The existing Symtropy cargo and mission-chain work is the right integration point; avoid duplicating its generic multimodal contracts inside this crate.

## 5. Benchmark plan and release gates

### v0.1 numerical correctness
1. **Zero false-safe outcomes** across boundary-oriented fixtures and a separately implemented oracle.
2. 100% agreement on mass, moment, CG, station caps, and disposition for exhaustive bounded test grids.
3. 100% fail-closed behavior for mismatched revisions, unsupported loading phases, unknown stations, malformed source digests, invalid/non-finite mass, overflow, detectable floating-point precision loss, uncovered envelope bands, and ambiguous bands.
4. Deterministic report equality when item order changes.
5. Explicit units and provenance in every report; zero use of generic/default aircraft limits.

### v0.2 load-state completeness
- independent oracle for zero-fuel/ramp/taxi/take-off/landing mass and CG calculations;
- fuel-burn and passenger/cargo relocation cases, including all envelope boundaries;
- property-based generation, mutation tests, and negative controls for missing/forged/stale evidence;
- independent review of unit conversions and approved-data interpretation.

### v0.3 geometry and restraint
- 100% detection of deliberately overlapping package fixtures and out-of-bounds placements;
- explicit floor area, concentrated load, linear/running load, compartment, ULD, restraint-point, and securement constraints;
- unknown load geometry/restraint inputs must never become a passing plan.

### performance / SOTA comparison
Record a pinned-machine release-mode baseline and p50/p95 latency before setting an optimization claim. An initial engineering target is p95 below 10 ms for 10,000 simple items over 64 stations, excluding I/O; this is a testable software-performance target, not a safety or physical-validity metric. Report accuracy, boundary failures, determinism, memory use, and throughput separately. Never trade correctness for a higher throughput number.

Compare against approved operator weight-and-balance systems and independently reviewed calculation sheets, not merely against another AI. The differentiator should be explicit uncertainty, exact source/config binding, reproducible reports, independent numeric oracles, multimodal mission continuity, and fail-closed refusal—not claims that a generic AI can replace licensed operational control.

## 6. Immediate follow-on work

1. Add a phase-aware load-state model so empty/ramp/taxi/take-off/landing/fuel states cannot be mixed implicitly.
2. Add a signed, versioned profile import path; check approval/currency and aircraft-configuration identity outside the arithmetic kernel.
3. Add ULD/bay geometry plus floor-pressure and restraint evidence in Symtropy's digital twin.
4. Connect cargo condition, dangerous goods, cold chain, accessible boarding/egress, transfer hubs, and recovery to Symtropy's mission-chain contract.
5. Bind profile revisions, manifest snapshots, inspection credentials, and result receipts to Mycelix access-controlled provenance.
6. Let Sol-Atlas consume only reviewed constraints for spatial planning; it must not turn route recommendations into clearances.

## Sources

- FAA, [AC 120-27F — Aircraft Weight and Balance Control](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1035868).
- FAA, [Aircraft Weight and Balance Handbook, FAA-H-8083-1B](https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-1.pdf).
- FAA, [Cargo Safety resources](https://www.faa.gov/aircraft/safety/cargosafety/resources) and [Common Cargo Hazards](https://www.faa.gov/aircraft/safety/cargosafety/cargo_hazards).
- NASA Langley, [Formal Methods Research Program](https://shemesh.larc.nasa.gov/), including state-based separation assurance research.
- EASA, [Design Organisations / Part 21](https://www.easa.europa.eu/en/domains/aircraft-products/design-organisations) and [Production Organisations Approvals](https://www.easa.europa.eu/en/domains/aircraft-products/production-organisations-approvals).

These sources ground the engineering direction only. The crate is not reviewed or certified by the FAA, EASA, ICAO, an aircraft manufacturer, or an operator.
