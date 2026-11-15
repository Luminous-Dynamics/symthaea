# PWC Industry and SOTA Benchmark Crosswalk
## 2026-11-15 source snapshot

**Purpose:** anchor PWC-SIT research against current manufacturer-disclosed products and adjacent research methods without pretending that unlike specification sheets are head-to-head test results.

**Product owner:** PWC-CO-DESIGN-001 #12095  
**Installed waterjet:** #12135  
**Pump-loop arithmetic:** #12231  
**Next installed-system extension:** #12232

> Manufacturer specifications and demonstrations are not independent comparative tests. The figures below are source-attributed discovery anchors. They are not a valid performance ranking, nor do they establish a Luminous product design.

## 1. Current product landscape

| Product | Public source-reported configuration | What it establishes | What it does not establish |
|---|---|---|---|
| 2026 Kawasaki Jet Ski Ultra 310LX | 228 kW / 310 PS at 8,000 rpm; 286 N·m at 6,000 rpm; 1,498 cc supercharged inline-four; 160 mm single-stage axial-flow pump; 8,407 N stated thrust; 3 seats; 80 L fuel (Kawasaki Australia page) | A high-power, supercharged, direct-drive ICE PWC with a published pump/thrust specification | Independent thrust test, delivered ride energy, pump map, intake losses, thermal endurance, or matched competitor result |
| Yamaha GP SVHO | 1,812 cc supercharged 4-cylinder; 160 mm axial-flow pump; 344 kg dry; 70 L fuel; 1–3 riders | A current high-performance ICE PWC package with public mass/fuel/pump anchors | Direct comparison with Kawasaki's curb/dry mass or power; pump efficiency or mission energy |
| Taiga Orca WX3 (2026 World Tour page) | 120 kW; 360 kg; 3-rider capacity; “2 hrs” play time; “1h 20m” continuous throttle; fast charge under 30 minutes | A marketed battery-electric PWC with published power, mass, capacity and usage claims | A controlled, independently reproducible range/energy comparison; battery capacity, precise drive cycle, initial/final SOC, water state, rider loading and temperature are not specified on this page |
| MIT electric-PWC retrofit thesis (1997 Yamaha donor) | 72 V, 10.4 Ah pack and 5 kW motor coupled to existing impeller shaft; reported dry test | A low-power electrical integration and retrofit proof of concept | Full-scale sea-trial endurance, commercial reliability, production-ready marine ingress protection or matched ICE performance |

Primary sources:
- Kawasaki Australia 2026 Ultra 310LX: https://www.kawasaki.com.au/en-au/watercraft/jet-ski/supercharged-sit-down/jet-ski-ultra-310/2026-jet-ski-ultra-310lx
- Yamaha global GP SVHO specifications: https://global.yamaha-motor.com/business/waverunner/products/gpsvho/spec.html
- Taiga Orca WX3 2026 World Tour: https://www.taigamotors.com/en/orca-wx3-2026-world-tour/
- MIT thesis, *Electrifying a Jet Ski: Designing & Manufacturing an Electric Drivetrain for a Personal Watercraft*: https://dspace.mit.edu/entities/publication/3fb93265-740c-4252-8535-70916fa8b9e6

### Important data hygiene

Do not rank the three products by the raw values above. The sources mix dry and curb mass, use different regional/model revisions, and publish different usage descriptions. Taiga's World Tour page describes “3 riders” capacity while also explaining a demo format in which riders take turns on a single-rider machine. This is a reminder to distinguish certified capacity, seating, demo configuration and payload in the benchmark schema.

Kawasaki Australia's figure of 516 kg is **curb weight** and 452 kg is **estimated dry weight**; these are not interchangeable. Yamaha's 344 kg is dry weight. Do not subtract or normalize these values without matching definitions and equipment.

## 2. Adjacent tug powertrain / control research (method comparator only)

The PWC programme should reuse robust experiment design and deployable-controller lessons from tug research without pooling product datasets.

- Gao, Wu & Tian, *Ocean Engineering*, 15 October 2026, event-triggered hybrid tug EMS: reports 415.1 kg equivalent fuel vs 447.2 kg for REMS, 225 updates and approximately 97% fewer updates, 0.0372 s per-step computation, and terminal SOC near 45% for a particular simulated tug sequence. This is **simulation evidence for one tug system**, not a PWC efficiency target or field result. DOI: https://doi.org/10.1016/j.oceaneng.2026.127569
- Li, Long & Tian, *Energies*, 9 February 2026, dual-delay TD3 hybrid tug EMS: reports study-specific performance reductions against rule-based / ECMS / DDPG on its seven-phase simulated port cycle, with a hardware-in-the-loop setup. The comparison remains architecture- and cycle-specific. https://www.mdpi.com/1996-1073/19/4/905
- Li, Tian & Long, *Machines*, 2 March 2026, hybrid methanol tug sizing: reports a 50.8% equivalent-fuel reduction for one typical one-hour simulated duty cycle with explicitly idealized component models and further operational/HIL validation still needed. It is not a PWC reference. https://www.mdpi.com/2075-1702/14/3/274

For PWC controls, the relevant lessons are:
1. separate plant architecture effects from controller effects;
2. compare a transparent tuned baseline with MPC/ECMS or an appropriate deployable policy;
3. include forecasts available at action time, not perfect future knowledge;
4. include minimum/terminal battery state, reserve, thermal derating, battery degradation, and mission completion;
5. measure real-time latency and safe fallback, not just simulation reward.

## 3. What the industry currently optimizes—and the gap

Public product pages emphasize peak power, engine displacement, pump diameter, weight, seating/fuel capacity, acceleration/handling features, and advertised playtime/range. Those data are useful for product screening, but they do not reveal a common installed-system efficiency map or net energy per matched ride mission.

The defensible opportunity is to compete at the **ride-service level**, not on peak kW alone:

- Same rider mass and distribution, payload and initial fuel/SOC.
- Same water temperature/density, depth, current, wind and wave state.
- Same course, control/throttle schedule, acceleration/turning requirements and duration.
- Same definition of energy boundary (fuel chemical energy, battery DC output, inverter/motor/shaft, pump hydraulic energy, useful thrust).
- Same start/end conditions, battery SOC window and turnaround/charge assumptions.
- Same instrumentation quality and full accounting of failed, censored, or incomplete runs.

A faster craft that burns or stores more energy, overheats, derates, has more maintenance downtime, or cannot repeat its mission is not automatically the better product.

## 4. A benchmark that can change the competition

### A. Standardized ride missions

Create separate profiles rather than one composite “ride score”:
1. low-speed manoeuvring / docking and idle;
2. acceleration and planing transition;
3. matched cruise / range;
4. repeated acceleration-turn-deceleration circuit;
5. rougher-water or wave-encounter envelope;
6. thermal soak and repeated high-load endurance;
7. degraded controlled mode (sensor loss, thermal derate, reduced propulsion output) under a safe test protocol.

For the first benchmark, freeze one 2–3 seat PWC-SIT configuration and two incumbent products with their exact model-year and regional revisions. Keep stand-up craft, surfboards, hydrofoils and tow-sport duty cycles separate.

### B. Required metrics

- delivered service: course completion, acceleration, turn response, speed-control error, rider-relevant vertical/lateral acceleration and recovery time;
- energy: fuel energy, electrical DC energy, charge losses, pump/shaft power and energy per completed standardized mission;
- propulsion: thrust/force vector, torque/RPM, flow/head, pump efficiency, inlet/system losses and operating-point stability;
- system: temperatures, derating, battery throughput/health, noise spectra, vibration, cavitation/ventilation indicators, water ingress, failure/downtime and maintenance labour;
- confidence: calibration, sensor/timebase quality, repeated-run variance, uncertainty intervals, domain coverage and all excluded runs.

Do not infer pump flow or thrust from a motor/engine rating. Use the pump-loop reference #12231 only for its stated boundary. Use #12232 for the distinct installed-system and propulsion-test terms.

### C. Pre-registered benchmark goals (proposed, not achieved)

- **Energy:** at least 10% lower net energy per matched ride mission versus the best credible measured incumbent, with non-overlapping uncertainty against a predeclared practical-equivalence region.
- **Mission parity:** no reduction in payload/capacity, course completion, acceleration, turning response, safe operating envelope or availability to manufacture an energy win.
- **Thermal/endurance:** no hidden power derating or battery/engine thermal excursions over the frozen repeated-mission profile.
- **Reliability/maintainability:** at least 20% reduction in scheduled service labour only after independently measured baseline and maintenance records exist.
- **Evidence:** 100% configuration, calibration, timebase and raw-trace lineage for every scored run; zero suppressed failures or unexplained data loss.
- **Safety/environment:** non-compensatory gates for flotation, off-throttle steering, propulsion cut-off, flooding, electrical isolation, emissions, noise and biosecurity. A high energy score cannot compensate for a failed gate.

These targets are deliberately beyond marketing comparison but remain goals. We should not set a claimed achieved saving until a rights-admitted, physically measured baseline and an independent trial protocol exist.

## 5. How to outperform current SOTA without an unsupported “best” claim

1. **Reconcile three test scales:** pump loop → inlet/system rig → installed craft. The current pump-loop calculator forbids extrapolating its value to whole-craft efficiency. The next issue #12232 defines the station-energy, inlet and hull-interaction join.
2. **Add transient installed performance:** test straight-line points and both-direction manoeuvres with synchronized hull 6-DOF, intake/pump operating point, nozzle position and rider loading. This addresses behaviours a steady pump map misses.
3. **Separate control from hardware:** factorial tests compare controller variants on the same plant and plant variants with the same controller. This prevents attributing battery/supercapacitor hardware gains to an algorithm.
4. **Use counterfactual evidence:** compare matched ride missions and preserve the whole difference ledger—fuel/electricity boundary, initial/final SOC, charge losses, thermal derating, maintenance, and availability.
5. **Select the next experiment by value of information:** prefer the smallest experiment that distinguishes competing causes (pump degradation vs intake distortion vs steering/nozzle vs sensor error) over a broad expensive campaign.
6. **Make abstention a feature:** if cavitation, ventilation, sensor quality, or hull/propulsor cause cannot be distinguished, report INDETERMINATE/OUT_OF_DOMAIN rather than an overconfident diagnosis.
7. **Build design-for-repair into the objective:** accessible intake/pump service, replaceable modules, corrosion protection, debris tolerance, parts availability and service hours are measured life-cycle performance—not afterthoughts.

## 6. Claim ceiling and next physical gate

This document is a dated source crosswalk and proposed benchmark. It does not establish a head-to-head winner, performance improvement, waterjet efficiency, product safety, regulatory compliance, or partner access.

Before a physical comparison, freeze the exact craft revisions, inspect the applicable ISO 13590 / national requirements with a competent authority, verify test-facility capabilities, agree IP/data rights, write the calibration and uncertainty plan, and establish a safe independent trial protocol.
