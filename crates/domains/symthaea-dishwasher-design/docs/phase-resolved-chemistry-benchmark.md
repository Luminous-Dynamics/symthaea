# InstantClean phase-resolved chemistry and detergent benchmark

Status: research programme, 3 November 2026. This document does not prescribe a detergent formula or dose. All quantities and release timings must come from formulation-owner safety documentation and independent physical calibration.

## Why this is now a core competitive problem

Two 2026 commercial developments make detergent chemistry a first-class appliance control input:

- **LG's 2026 European dishwasher platform** combines a one-hour wash-and-dry programme, QuadWash Pro micro-bubbles, Dynamic Heat Dry+, TrueSteam and AI SenseClean. LG says its turbidity sensor monitors prewash, main wash and rinse, adapting temperature, rinse count and detergent amount through up to 32 combinations.
  - Official release: https://www.lg.com/global/newsroom/news/home-appliance-solution/lg-electronics-to-showcase-new-dishwasher-lineup-at-eurocucina-2026/
  - Corporate release: https://www.lgcorp.com/media/release/30047
- **Henkel Somat Smartwash**, launched in Germany on 20 October 2026, adds automatic multi-component product dispensing. Henkel says its device combines detergent, rinse aid, salt and machine-care components, senses temperature and movement patterns, detects cycle phase and adjusts timing/quantity of component release. It is positioned as a cross-brand add-on for household dishwashers.
  - Official release: https://www.henkel.com/press-and-media/press-releases-and-kits/2026-10-20-henkel-launches-somat-smartwash-2258310

These are company-announced feature claims, not independent matched cleaning or resource results. Their presence means that "AI dosing" or "automatic detergent" is no longer a sufficient differentiator.

## Paradigm shift: chemistry as a controlled process, not a consumable

Treat the wash as a coupled, staged process:
1. Soil wetting and hydration.
2. Soil-specific chemical and enzymatic action.
3. Hydraulic shear / impingement and mechanical removal.
4. Separation of released soil from the load.
5. Rinse and chemical carryover control.
6. Drying / moisture management.

Each stage has distinct process conditions and potentially different chemical requirements. Optimizing one global detergent mass is not equivalent to optimizing component timing, phase, temperature, water chemistry and mechanical action together.

The January 2026 dishwasher cleaning-kinetics study reports different non-linear behavior for carbohydrate and protein soils. It identifies amylase as important for carbohydrate removal and notes that bleach reduced amylase effectiveness in its experiments; protein stain removal involved amylase/protease with bleaching also affecting stain colour. This is a formulation-specific experimental result, not a universal chemical rule or a dosing recommendation.
- DOI: https://doi.org/10.1016/j.ces.2025.122486
- Publisher summary: https://www.sciencedirect.com/science/article/pii/S0009250925013077

An October 2026 joint detergent/cycle optimization paper links detergent formulation and cycle conditions in an event-based mechanistic model and genetic-algorithm framework. Its reference case reports 31% shorter wash time, 59% shorter drying and 5% higher stain removal; underlying data are confidential, so this is a replication lead rather than a claim for InstantClean.
- DOI: https://doi.org/10.1016/j.compchemeng.2026.109771

## Research hypothesis

The defensible next step is an **evidence-bounded phase-resolved controller**. The controller does not invent chemistry. A formulation owner provides a versioned calibration profile that defines:
- which component may be released in which phase;
- validated per-release and per-cycle quantity ranges;
- supported soil/material classes;
- water-hardness and phase-specific temperature envelopes;
- component interactions and minimum separation intervals;
- the evidence digest for each calibration.

The appliance proposes only a schedule inside that envelope. If a sensor is missing, a formulation is unidentified, a release is outside its phase, or an incompatibility cannot be evaluated, reject the schedule and use the conservative validated programme.

## Existing implementation

The Symthaea design crate now contains `chemistry::validate_dose_plan`.

The public validation contract checks:
- exact formulation identity and calibration evidence SHA-256;
- unique component IDs and versioned component calibrations;
- phase windows ordered as prewash → main wash → intermediate rinse → final rinse → drying;
- phase-specific release temperature and fixed water hardness against calibrated ranges;
- soil class and surface material against supported scopes;
- stage permissions and release timing;
- per-event, per-component-cycle and total-formulation mass limits;
- interaction timing constraints between calibrated incompatible components;
- nonempty evidence for every interaction rule.

It returns a validation summary only. It does not choose a dose, estimate cleaning, certify chemistry safety, or claim that a validated plan cleans successfully. Cleaning/rinse acceptance still requires independent physical outcomes.

The illustrative tests use fake calibration data and arbitrary masses solely to test logic. They are not formulation guidance.

### Dose arithmetic and delivery timing implementation boundary

The proposed-plan validator now uses integer milligrams (`u64`) for commanded component mass, with checked arithmetic for component and total accumulators. It does not compare floating-point gram values at a chemical dose boundary. The per-release CSV stores integer mg; cycle-level totals may remain in grams for product energy/resource accounting.

Each release includes both a start time and delivery duration. The complete physical delivery must finish within the relevant phase window. Evidence-backed incompatibility separation is measured between the end of one component's release and the start of another, not merely between their start timestamps. A commanded schedule is still not proof that a dispenser delivered that mass; actual dose must be measured and archived separately.

The separate `check_reported_dose_delivery` function compares a requested dose event against reported actual telemetry using a calibrated under/over-delivery tolerance (basis points), timing tolerance, and integer-milligram arithmetic. It binds the measurement to the command's event ID, component ID, physical dispenser-channel ID and delivery-calibration ID. It checks calibration/measurement digest shape only. It does not authenticate the sensor or source records, prove that the dispenser physically delivered the measured quantity, or establish cleaning. The per-event archive therefore records commanded and actual start/duration separately, actual mass, dispenser channel, delivery calibration, measurement-evidence digest and calibration-bundle digest.

## Competitive benchmark matrix

Compare exact SKUs and exact programmes; capture market, label, detergent, options and firmware where available.

| Comparator | Already-present capability | InstantClean test |
|---|---|---|
| LG 2026 European range | One-hour wash/dry; QuadWash Pro micro-bubbles; AI SenseClean phase sensing and up to 32 combinations; Dynamic Heat Dry+ / TrueSteam | Matched one-hour normal-soil test; independent assay of each soil family; real energy/water; verify detergent amount and rinse carryover |
| Henkel Somat Smartwash + compatible standard dishwasher | Automatic multi-component dispensing; temperature/movement sensing; cycle-phase release; cross-brand positioning | Compare its default dose schedule against fixed dose and calibrated phase-aware schedules; measure each component, chemistry residue, cleaning and drying |
| Miele G8 Diamond (announced; DACH April 2027) | Camera/AI load and item recognition, AutoDos, variable hot-air drying | Add as a future competitor when exact model and physical product are available; capture camera recognition and adaptive recipe behavior; do not invent label values |
| V-ZUG OptiStart / heat-pump platform | Cold pre-wetting during delayed start; lower subsequent wash temperature claimed by manufacturer | Compare timed pre-soak with phase dosing for aged protein/starch soils and total standby/thermal energy |
| Conventional fixed-dose programme | Fixed detergent mass and fixed stage timing | Baseline for determining whether added complexity has measurable benefit |

### Johannesburg water-hardness baseline

Rand Water's report for supply to Johannesburg municipality (23 April 2024–22 April 2025) records mean total hardness of 74.51 mg/L as CaCO3, SD 9.51 mg/L, mean + 3 SD 103.03 mg/L, and an operational range of 20–200 mg/L across 336 results. These statistics do not specify the water at an individual residence.

Initial controlled water-hardness points: **20, 50, 75, 100, 150 and 200 mg/L as CaCO3**. At least three actual local inlet-water samples should also be measured, including any post-tank / filter / softener effects. Record hardness at the appliance inlet per run; never assign the municipal mean to every test.

Source: https://www.randwater.co.za/media/sans241/12month_Johannesburg_Apr2025.pdf

## Experimental matrix

### Soil and surface

Soil families: starch/carbohydrate, protein, fat/oil, tea/coffee/tannin and mixed soil only after single-soil calibration.

Surfaces/geometries: ceramic, glass, stainless steel, representative polymer, bottle interior and difficult-to-wet recesses.

### Chemical control conditions

1. Manufacturer-approved fixed dose and standard timing.
2. Existing auto-sensor mode from the incumbent.
3. Manufacturer's automatic multi-component dosing, where present.
4. Calibrated phase-resolved schedule from the same formulation.
5. Ablations: remove phase sensing, remove turbidity, remove conductivity, remove camera context, hold dose fixed, disable any optional pre-soak.

Do not compare different detergent formulations and attribute the result only to the controller. Either hold formulation constant for a control comparison or run a factorial study that separates formulation and control effects.

### Process factors

Water hardness, inlet temperature, phase-specific temperature trajectory, detergent batch and age, soil batch/age, load/occupancy, hydraulic flow and pressure, cycle duration, drying boundary, and ambient conditions.

Use randomized blocked runs and hold out complete soil batches and test days. Minimum five runs per condition is a pilot only; size confirmatory testing using pilot variance and a preregistered minimum effect.

## Endpoints

Cleaning and rinse are independent hard gates:
- residual soil mass or assay per soil family;
- surface-specific cleaning score under the same standardized method;
- detergent / rinse-aid carryover in final rinse;
- cross-transfer to clean witness surfaces;
- rewash rate and false-clean rate.

Drying and hygiene/sanitization are separate endpoints:
- residual moisture by material and item geometry;
- drying kWh and minutes at a matched moisture endpoint;
- microbial endpoints only where a defined validated test and claim are intended.

Resource and service outcomes:
- real wall kWh per verified successful clean;
- total fresh water including prewash, all rinse and purge;
- total mass of every dispensed component;
- first-item release, median/p95 time-to-release and party-mode throughput;
- standby/preheat energy, filter loading, waste discharge and service burden.

## Aspirational goals

These are research targets, not expected results:
- At least 10% less total wall energy **or** fresh water per verified successful clean than the best matched incumbent, while preserving cleaning, rinse and drying endpoints.
- At least 5% lower active chemical mass per successful clean where formulation-owner guidance and residue endpoints permit; chemical minimization is not allowed to reduce cleaning.
- No increase in false-clean / false-rinse pass rate; pre-register a one-sided upper confidence bound from independent held-out cycles.
- Fused sensing + chemistry control must materially beat the best existing single control method or deliver equivalent outcomes with lower total hardware/consumable/service cost.
- If LG's one-hour programme meets the same normal-soil endpoint with lower resource use, InstantClean must beat it or change its differentiated use case; merely matching the hour is not enough.

## Safety, compatibility and data rules

- Never mix components or invent compatibility rules based on ingredient category alone.
- Any component interaction/separation interval must be supported by the exact formulation owner's documented study.
- Never derive a dose from camera or turbidity alone.
- Do not use cleaning colour change as a substitute for residual soil or rinse-residue assay.
- A plan outside calibration scope is rejected, not clipped silently to a nearby value.
- The controller cannot override the deterministic safety supervisor or manufacturer instructions.
- Record formulation, component, calibration and algorithm versions with every run.
- Preserve negative results and all failed cycles; do not drop a run to improve average resource metrics.

## Immediate next steps

1. Acquire one LG one-hour dishwasher and one Somat Smartwash unit when regional availability permits; identify exact SKUs and programme details.
2. Obtain the actual detergent component documentation and permissible use instructions from the formulation owner. Do not reverse engineer or infer component masses.
3. Extend the experiment archive with a separate one-row-per-release event file so each component and phase is recorded without duplicating cycle-level rows.
4. Validate the Rust plan checker with synthetic fixtures, then compile/test exact head; do not call fixtures physical evidence.
5. Run matched fixed-dose vs automatic-dosing vs phase-aware control experiments only after independent endpoints and calibration are frozen.

Related issues: https://github.com/Luminous-Dynamics/symthaea/issues/10216 ; FTO/prior-art review https://github.com/Luminous-Dynamics/symthaea/issues/10219. See [detergent-control-prior-art.md](detergent-control-prior-art.md) before making novelty or commercialization claims.
