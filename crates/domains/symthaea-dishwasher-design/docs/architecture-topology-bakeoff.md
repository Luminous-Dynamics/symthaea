# InstantClean architecture topology bake-off

Status: research decision record, 2026-11-02. No topology is selected. The descriptions below are engineering hypotheses, not validated product performance or cost results.

## Decision to make

Select the lowest-complexity architecture that can deliver a genuinely useful one-item / micro-batch clean while preserving competitive full-load performance, independently verified cleaning and rinse quality, safe fluid separation, repairability and acceptable lifecycle cost.

Do not select two complete chambers merely because the concept sounds faster. Do not select a common tub merely because it looks cheaper. Compare the physical failure modes and the real resource boundary.

The current competitive landscape already contains independently operated drawer compartments and soil-sensing automatic cycles. InstantClean's differentiator must be a measured performance envelope and verified result, not the presence of compartments or a turbidity sensor by itself.

## Candidate topologies

### A — Dedicated QuickCell + FullLoad chamber

A separately enclosed single-item / micro-batch cell adjacent to a conventional 14–16 place-setting chamber. The QuickCell may have its own local sump or a fully isolated hydraulic branch; the FullLoad chamber keeps its normal batch washing geometry.

**Potential advantages**
- Strongest physical separation between a one-item process and full-load soil/debris.
- QuickCell can use geometry-specific fixtures, close-range jets or an internal vessel lance.
- Independent operating schedules and local thermal mass may support a fast ready-state process.
- FullLoad design can be optimized independently against the IEC loading method.

**Potential liabilities**
- Extra enclosure volume, doors, seals, valves, sensors, drainage paths and service access.
- Duplicated pumps/heaters or complex shared services may erase resource savings.
- More leak interfaces and dead legs; a local cell can still cross-contaminate through shared filtration, plumbing or aerosols if poorly isolated.
- Higher manufacturing cost, standby losses and repair burden.

**Disconfirming evidence**
Reject this topology if its one-item improvement disappears after including warm-up, standby, flush, dry, upstream water heating, filter maintenance and added standby loads; or if the same endpoint can be achieved by a simpler micro-zone.

### B — Single tub with an isolatable micro-zone / vertical cassette

A conventional primary tub contains a removable or retractable fixture/cassette and a bounded micro-zone. The zone may use a local spray manifold and a physically isolated drain/recirculation path. Vertical plate slots are a candidate fixture, not an assumed cleaning improvement.

**Potential advantages**
- Potentially shares the cabinet, door, filtration service access and some thermal/hydraulic components.
- Less duplicated outer structure and potentially easier retrofit/manufacture.
- Fixture can be reconfigured for plates, cups, bottles and cutlery.
- A cassette may improve loading ergonomics and repeatable item placement.

**Potential liabilities**
- A partial partition is not fluid isolation. Splash, drain-back, aerosol, shared sump and contaminated return paths can invalidate the micro-zone.
- Partitions and cassette rails can shadow spray, trap food, obstruct loading or complicate cleaning.
- Shared pump/temperature states can couple the fast cell to the full tub and degrade both.
- Vertical slots may create face-to-face shadows or uneven spray incidence; geometric visibility is not proof of deposited water or cleaning.

**Disconfirming evidence**
Reject the shared-zone option if tracer/dye, conductivity, soil carryover or microbiological validation shows unacceptable transfer; if hard-to-reach witness coupons fail; or if the zone cannot be drained and serviced without dismantling the primary tub.

### C — Drawer-derived modular architecture

A single or dual drawer platform with independently operable, accessible wash compartments. QuickCell functionality is achieved by a small dedicated insert or a very small drawer module rather than a new second full-size chamber.

**Potential advantages**
- Existing commercial precedent for independent drawer operation and half-load washing.
- Ergonomic loading at counter height; smaller accessible modules may be easier to service.
- A drawer can be a well-defined serviceable module with replaceable seals, pump and fixtures.
- Lower development uncertainty in user interaction than an entirely new cabinet geometry.

**Potential liabilities**
- Drawer-specific sealing, slides, cable/hoses and moving hydraulic connections are durability risks.
- An existing half-load drawer is not automatically a one-item process; its warm-up, water and energy cost must be measured at actual occupancy.
- Two drawers or a small drawer plus a full tub can still duplicate pumps, seals and controls.
- It may not provide the volume or nozzle proximity needed for bottle/cup internal cleaning.

**Disconfirming evidence**
Reject it as the QuickCell platform if measured one-item resource use and time-to-release are not materially better than a current drawer/half-load/soil-sensing reference at a matched verified endpoint.

## Non-negotiable eligibility gates

No weighted score may compensate for a failed hard gate:

1. Cleaning: every required soil / surface / geometry class meets a preregistered residual-soil threshold.
2. Rinse: detergent and soil carryover remain within a defined assay or standardized acceptance limit.
3. Cross-contamination: no unacceptable transfer between a QuickCell item, FullLoad load, wash-water circuit, drain and subsequent cycle under defined worst-case tests.
4. Safety: doors, leaks, overfill, temperature, pressure, electrical insulation, backflow and drainage meet the applicable safety requirements. IEC 60335-2-5:2025 and the applicable IEC 60335-1 / national adoption need clause-level review.
5. Claim integrity: cleaning, rinsing, drying, hygiene and sanitization are tested and described separately.
6. Serviceability: filters, seals, valves, pumps and spray components can be inspected and replaced without destructive disassembly.

A topology that fails any gate is ineligible, regardless of its energy or cycle-time result.

## Common measurement boundary

Measure all candidates with the same instruments and protocols. Include:

- Wall real energy (not voltage multiplied by current), including control electronics, pump, heat, drying, standby during the defined duty window and all warm-up / purge cycles.
- Incoming cold/hot water volume and upstream water-heating energy where relevant.
- Every drain / purge / flush volume; distinguish fresh water from recirculated process-water volume.
- Detergent mass, rinse-agent dose and any consumables.
- Item count, mass, material, surface area proxy, soil family, soil mass, age and rack position.
- Arrival-to-release time, ready-state time and cold-start time separately.
- Residual soil, rinse residue, drying result, surface damage and rewash requirement.
- Noise at declared positions, vibration, leakage, filter solids and service interventions.
- Sensor and model versions, calibrations, uncertainty, raw time series, geometry digest and firmware/analysis commit.

Report per-cycle, per-item and per-successful-clean values. A failed item stays in the denominator for resource accounting; never improve a result by dropping failed or unprocessed items.

## Test matrix

### Stage 1 — Bench topology coupons

Build transparent, low-pressure geometric rigs for each candidate. Test flat plate, mug/cylinder, bowl, bottle/tall vessel and cutlery fixtures. Use standardized fluorescent witness films or another calibrated deposition method to measure coverage and retained/drained water. The result is a deposition test, not a cleaning test.

At minimum, compare:
- Conventional rotating spray / static jets.
- Close-range targeted jet or lance.
- Pulsed operation at matched delivered water and pump input.
- Candidate vertical cassette positions versus conventional rack placement.

### Stage 2 — Hydraulic and contamination tests

Use measured pump curves, branch pressure/flow and real pump power. Include blocked nozzle, partial filter restriction, low inlet pressure, air ingress, drain-back, valve leakage and sensor disconnect.

Run tracer, conductivity and soil carryover tests through every relevant fluid boundary. Test the first post-high-soil cycle and the first cycle after a detergent-heavy load. Confirm that fresh final-rinse water cannot be silently contaminated by wash return.

### Stage 3 — Physical cleaning and rinse comparison

Use at least three market references:
- A high-efficiency 16-place full-size machine.
- A current low-water full-size machine.
- A current independently operated drawer model.

For one-item workloads, compare against the same references in their manufacturer-supported quick / auto / half-load modes. Do not compare QuickCell to the competitors' full Eco-cycle label as if the test were equivalent.

Randomize run order; block by soil preparation, water hardness and day; hold out complete days or soil batches. Pilot at least five independent runs per condition, estimate variance, then pre-register confirmatory sample size. Retain every failed run and use a negative-control design.

## Decision metrics and selection logic

First apply the hard gates. Among passing candidates, publish the Pareto frontier over:

- kWh per verified successful clean.
- L fresh water per verified successful clean.
- Median and p95 arrival-to-release time.
- Detergent mass per verified clean.
- Drying energy and drying endpoint.
- Noise and service interventions.
- Estimated manufacturing cost range and module replacement / repair time.
- Standby and idle-period losses.

Do not create an opaque weighted score. Show the non-dominated candidates and the assumptions used for any business-case ranking.

### Proposed go/no-go gates (aspirational)

- QuickCell should deliver at least a 20% improvement in one-item time-to-release **and** at least a 10% reduction in one of energy or fresh water versus the best matched incumbent mode, at non-inferior cleaning/rinse endpoints. These are screening targets to test, not expected outcomes.
- FullLoad remains independently competitive at 16 place settings; QuickCell cannot be allowed to cannibalize its capacity, cleaning or Eco performance.
- All cross-fluid tracer and sensor-fault controls must be detected; no early-success state is permitted when the observer is out of envelope.
- A topology must show a lifecycle/repair advantage or a measurable use-case advantage sufficient to justify any added pumps, valves, seals and interfaces.

If no topology beats the existing drawer and auto-cycle incumbents at the matched endpoint, do not force the hybrid architecture. Re-scope to a better spray/fixture retrofit, adaptive control, or a software/controls upgrade of a simpler appliance.

## Immediate prototype recommendation

Do not build all three full appliances. Build:
1. A low-cost transparent coupon rig for A/B/C geometry and hydraulic isolation.
2. A replaceable QuickCell fixture with a single plate/cup/bottle set.
3. A small instrumented pump/valve skid to characterize power, pressure, flow, leakage and fluid carryover.

Only after Stage 1 and Stage 2 show a viable zone should the team choose an enclosure topology for Stage 3 cleaning trials.

## References and boundaries

- IEC 60436:2025, corrected 2026-04: performance measurement method. Licensed text required for formal conformity claims: https://webstore.iec.ch/en/publication/80459
- IEC 60335-2-5:2025: household dishwasher safety requirements; use with applicable IEC 60335-1 and national adoption: https://webstore.iec.ch/en/publication/101702
- Fisher & Paykel DishDrawer commercial precedent: https://www.fisherpaykel.com/us/dishwashing/dishdrawers/
- ENERGY STAR dishwasher soil-sensing cycle context: https://www.energystar.gov/products/dishwashers

This document is a design and experiment decision framework. It does not assert that a chosen architecture, cleaning method or resource target has been validated.
