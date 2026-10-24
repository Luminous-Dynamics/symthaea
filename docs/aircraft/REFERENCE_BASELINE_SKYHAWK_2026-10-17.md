# Conventional Reference Aircraft Data Card: Cessna Skyhawk

**Reviewed:** 2026-10-17  
**Purpose:** candidate reference dataset for mission-model verification planning.  
**Evidence level:** manufacturer-published specification summary; not an independent flight-test dataset.  
**Related work:** [AIRROAD-003 #7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729), [candidate mission contract](https://github.com/Luminous-Dynamics/symthaea/blob/research/aircraft-energy-architecture-2026-10-16/docs/aircraft/CANDIDATE_SPECIFIC_MISSION_MODEL_CONTRACT_2026-10-17.md), [aircraft-energy screening #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647).

## 1. Why this reference was selected

A conventional, certified light airplane provides a useful *scale and mission reference* for a proposed personal air-road system. The Cessna Skyhawk manufacturer page publishes dimensions, weights, occupant/baggage capacity, range, cruise speed, takeoff and landing distances, climb rate, and engine rated power. Those are useful anchors for requirements and later model validation design.

This card does **not** claim the Skyhawk is directly comparable to an electric roadable aircraft or that these summary values are sufficient to validate an aerodynamic or propulsion model. Its piston engine, fuel, airframe, mission profile, operating assumptions and certification basis differ. Use it to frame the baseline; build like-for-like profiles before cost or energy claims.

## 2. Manufacturer-published reference values

The table transcribes metric values published on the Cessna/Textron Aviation Skyhawk page as reviewed on 2026-10-17. Values are manufacturer-stated specifications; do not treat them as independent measurements.

| Quantity | Published value | Unit / interpretation |
|---|---:|---|
| Length | 8.3 | m |
| Height | 2.7 | m |
| Wingspan | 11.00 | m |
| Wing area | 16.17 | m² |
| Maximum occupants | 4 | people |
| Baggage weight capacity | 54 | kg; separate capacity field, not extra payload above the aircraft limit |
| Maximum ramp weight | 1,160 | kg |
| Maximum takeoff weight (MTOW) | 1,157 | kg |
| Maximum landing weight | 1,157 | kg |
| Usable fuel weight | 144 | kg |
| Usable fuel volume | 201 | L |
| Basic empty weight | 762 | kg |
| Useful load | 398 | kg |
| Maximum payload | 395 | kg |
| Payload with full fuel | 254 | kg |
| Maximum cruise speed | 124 KTAS / 230 | km/h |
| Maximum range | 640 nmi / 1,185 | km |
| Takeoff distance | 497 | m |
| Takeoff ground roll | 293 | m |
| Landing distance | 407 | m |
| Landing ground roll | 175 | m |
| Service ceiling | 4,267 | m |
| Maximum climb rate | 223 | m/min |
| Stall speed | 89 | km/h (published 48 KCAS) |
| Engine | Lycoming IO-360-L2A | piston engine |
| Rated power | 180 hp / approximately 134 | kW |
| Propeller | McCauley, two-blade, fixed pitch | manufacturer description |

**Sources:** Cessna/Textron Aviation, [Cessna Skyhawk — specifications](https://cessna.txtav.com/en/piston/cessna-skyhawk) and [official Skyhawk product card (PDF)](https://cessna.txtav.com/-/media/cessna/files/product-cards/piston/skyhawk_product_card.pdf) (accessed 2026-10-17). The product card states that performance figures assume standard conditions and zero wind, field performance assumes a level, hard-surface, dry runway, and maximum range is based on a ferry mission with one pilot at 45% power with IFR reserves. The manufacturer's values may change between page revisions; this card preserves the review date, not a claim that the figures are a controlled engineering dataset.

## 3. Derived arithmetic-only reference metrics

These ratios are reproducible arithmetic from the rounded manufacturer-published values above; they are **not additional manufacturer claims** and are not an aerodynamic or performance model.

| Metric | Calculation | Approximate value | Permitted interpretation |
|---|---|---:|---|
| Geometric aspect ratio | wingspan² / wing area = 11.00² / 16.17 | 7.483 | Geometry-derived reference ratio only |
| Wing loading at published MTOW | 1,157 kg / 16.17 m² | 71.55 kg/m² | Gross loading ratio; not lift curve or stall prediction |
| Weight loading at published MTOW | 1,157 kg × 9.80665 m/s² / 16.17 m² | 701.69 N/m² | Same loading with standard gravity |
| Engine nameplate power per MTOW | 134,000 W / 1,157 kg | 115.82 W/kg | Rated engine power divided by maximum takeoff mass; not bus power or phase-specific shaft power |
| MTOW minus basic empty weight | 1,157 − 762 | 395 kg | Arithmetical margin that matches the page's rounded maximum-payload value; not a component mass breakdown |
| Ramp weight minus basic empty weight | 1,160 − 762 | 398 kg | Arithmetical margin that matches rounded useful load; ramp and takeoff limits are different |

Reference constants and formula choices must remain in the fixture manifest. Do not use any one of these ratios as a target for the Roadster concept without defining a compatible configuration and mission.

## 4. Critical interpretation boundaries

### Mass definitions cannot be mixed

- Basic empty weight already includes an aircraft configuration and equipment; this card does not split it into fixed airframe, propulsion, energy system, and avionics sub-masses.
- Useful load is not synonymous with payload available after fuel. The manufacturer separately publishes useful load (398 kg), maximum payload (395 kg), and full-fuel payload (254 kg); retain these as distinct fields.
- Fuel mass/volume must not be counted a second time when comparing manufacturer figures. For our future mass-closure model, record one explicit boundary per mass component and avoid adding an aggregate published weight to its already-included subcomponents.
- MTOW and maximum ramp weight are distinct operating limits. Use MTOW for the mission takeoff-mass check; do not substitute ramp weight.

### Range, speed and field performance are not a single mission profile

A maximum range, maximum cruise speed, maximum climb rate, takeoff distance and stall speed are achieved under different configurations/conditions and cannot be combined into one “best of every column” route calculation. The manufacturer product card states that performance data use standard conditions and zero wind, field performance assumes a level, hard-surface, dry runway, and range is based on a ferry mission with one pilot at 45% power with IFR reserves. It does **not** establish one universal operational range, mission profile, or energy-consumption curve. Read the exact source revision before using a range figure for comparison.

Therefore:
- Do not derive a 100 km cruise energy trace directly by scaling maximum range.
- Do not infer fuel burn from rated engine power alone.
- Do not interpret maximum cruise speed as the speed over the ground on a particular route.
- Do not infer climb, turn, stall margin, obstacle clearance or landing reserve from these summary values alone.
- Keep the Cessna as a *reference card* until a full sourceable performance dataset and appropriate validation cases have been obtained.

## 5. How to use this in the Symthaea program

**Appropriate now**
- Set plausible size/occupant/mass-class context for trade studies.
- Establish unit/definition tests for metre/kilogram/speed/range conversions.
- Create a requirements trace with each manufacturer-published value tied to its source URL and review date.
- Document which values remain unavailable for a fully reproducible model.

**Not appropriate yet**
- Treat this card as an aero/propulsion training dataset.
- Generate phase-by-phase battery loads by assuming engine rated power is always required.
- Claim a new vehicle beats the Skyhawk on payload, range, operational cost, safety, repairability or efficiency without equivalent mission profiles and evidence.
- Treat agreement with the manufacturer summary as independent physical validation.

## 6. Minimum data to collect before model validation

For a defensible reference comparison, obtain a specific aircraft configuration and document source/revision for:
- aircraft serial/configuration or a precise model-year variant;
- weight and balance / mass properties with datum, equipment and payload;
- approved Pilot's Operating Handbook or Pilot Information Manual and applicable supplements;
- performance tables for pressure altitude, temperature, takeoff/landing configurations, climb, cruise, descent, wind and reserves;
- engine/propeller operating data and fuel flow versus power/RPM/altitude;
- instrument uncertainty and conditions for any independent flight-test data;
- restrictions/limitations and the validity domain of each datum.

If those are unavailable, report the model-validation gate as **blocked on data**, not passed. Choose an additional independent published reference or analytic benchmark rather than tuning a model to this one summary.

## 7. Validation plan linkage

This is the proposed M1 starting card under #7729, not yet the M1 acceptance pass. The acceptance test should:
1. preserve these manufacturer values in a versioned source manifest;
2. verify conversions and distinctions (MTOW vs ramp weight, useful load vs full-fuel payload, KTAS vs ground speed);
3. obtain a specific, source-controlled performance dataset with mission conditions;
4. compare predictions to independent data that were not used for calibration;
5. publish residual/error and uncertainty by operating condition.

NASA's VSPAERO conceptual-design work demonstrates a useful validation pattern: explicitly model a geometry and operating case, then compare outputs against published theoretical/experimental results. It does not imply one conceptual flow solver covers all flight regimes or validates an unbuilt roadable aircraft. See [NASA OpenVSP/VSPAERO basics](https://www.nasa.gov/reference/openvsp-vspaero-basics/) and NASA's [VSPAERO propeller-blown-wing evaluation](https://ntrs.nasa.gov/citations/20210017397).

## Nonclaims

This data card is not an aircraft specification created by Luminous Dynamics, not an endorsement by Cessna/Textron Aviation, not a complete simulation input, and not evidence that any proposed flying-car architecture is feasible, safe, economical, or certifiable.
