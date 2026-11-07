# Shipboard power-transient and multi-timescale assurance — V0.1

**Status:** architecture and evidence contract with synthetic transient fixtures; no validated marine electrical model or physical qualification is claimed.  
**Parent:** [MARINE-POWER-TRANSIENT-001 #11028](https://github.com/Luminous-Dynamics/symthaea/issues/11028)

## 1. Why this exists

The slow resource-service scheduler works in minutes/hours and passes aggregate loads to existing grid-physics scheduling. That is necessary but insufficient. A vessel can have adequate daily kWh and still experience a failed start, inverter current limiting, DC-bus sag, AC frequency/voltage excursion, converter interaction or nuisance protection trip when multiple services start together.

Recent research reinforces this gap:
- Zhang et al. (IEEE Transactions on Power Electronics, 1 Oct 2026) review simultaneous pulsed loads, negative incremental impedance, voltage instability, harmonic distortion and multi-timescale control conflicts in shipboard DC microgrids. DOI: https://doi.org/10.1109/TPEL.2026.3697148
- Wang et al. (Applied Energy 426, December 2026 issue; online ahead of issue) review ship EMS coupling to degradation, transient buffering, MVDC stability, HIL/PHIL and deployment evidence. DOI: https://doi.org/10.1016/j.apenergy.2026.128503
- EMSA's battery safety guidance (2023, updated 11 Nov 2025) is non-mandatory and identifies the role of class, standards and flag-state safety requirements. https://www.emsa.europa.eu/publications/item/5061-battery-energy-storage-systems-bess.html

- Kotsampopoulos, *Testing Ship Electric Propulsion and Shipboard Microgrids: Standards, Techniques and New Trends*, Energies 19(9), published 22 April 2026. Reviews laboratory testing, co-simulation, CHIL, PHIL and holistic multi-component testbeds; supports the staged HIL/PHIL ladder but does not provide LOS-18 equipment limits. https://doi.org/10.3390/en19092016
- Kondratyev et al., *Field-Validated Probabilistic Digital Twin of a Shipboard Microgrid for Reliability-Oriented Operation*, Energies 19(17), 4074, published 29 August 2026. Reports 360 thirty-minute field records, 12 operation/fault classes, 4 s samples and 1,080 paired evaluations. Useful supervisory-state/diagnostic precedent; its primary telemetry is operator-confidential, and 4 s cadence is not evidence for millisecond converter/protection dynamics. https://doi.org/10.3390/en19174074

Those sources include larger and naval ships. They motivate test questions, not directly transferable LOS-18 limits.

## 2. What current Luminous physics can and cannot say

The shared `symthaea-grid-physics` library offers useful battery SoC/efficiency/degradation, feeder, droop, islanding, trip-envelope, renewable-generation and slow storage scheduling models. The infrastructure adapter documents a two-node radial feeder, real-power-only flow, illustrative plant parameters, synthetic load and simplified voltage/frequency/thermal behavior.

Therefore:
- Use it for energy-accounting and schedule-policy questions within its stated limits.
- Do not claim it validates shipboard DC-bus stability, converter impedance, harmonic distortion, AC power quality, protection coordination, fault-current interruption or battery fire safety.
- Do not infer millisecond response from hourly or 15-minute load buckets.
- SMDC-ELECTRICAL #3846 owns facility three-phase/protection engineering; this work only defines the marine event/evidence contract and interfaces to qualified engineering.

## 3. Architecture hypotheses (do not preselect)

For the real LOS equipment list, compare:
- AC distribution with separate DC subsystems;
- DC distribution with appropriate conversion and local protection;
- zonal AC/DC architecture;
- battery-only transient support;
- battery + DC-link capacitance;
- battery + supercapacitor / hybrid storage only if event data show a need.

For LOS-18, MVDC is not presumed appropriate. Choose voltage and topology only after equipment selection, load survey, isolation/grounding review, fault/protection study, EMC, class/flag review and professional design. Added hardware weight, mass properties, cooling, fire separation, maintenance, spare support and lifecycle cost are part of the decision.

## 4. Multi-timescale contract

Every owner declares its native timescale and measurement bandwidth:
- mission/resource policy: minutes to hours;
- generator/source/battery dispatch: seconds to minutes;
- converter/bus dynamics: equipment/topology-specific millisecond and faster dynamics;
- local protection/BMS: manufacturer-defined and authority-reviewed response.

A schedule is eligible only if the exact event and operating point have owner-reviewed dynamic acceptance evidence. A slow scheduler can propose staggering compatible discretionary starts, but cannot alter safety logic, propulsion demand, certified protection, BMS limits or equipment interlocks.

Cross-rate aggregation must declare filtering, anti-aliasing, sample rate, synchronization uncertainty, event capture window and missing-data handling. Never treat an averaged load as proof of transient behavior.

## 5. Event corpus and evidence semantics

The companion JSON contains eight synthetic event records. They cover watermaker pump start, compressor start, induction cooking, thruster operation, propulsion load step, generator loss, shore-power loss and coincident service starts.

All amplitudes and timing values in this first fixture are **synthetic contract-test values only**. They are not equipment specifications, bus ratings, protective settings, current limits, or recommended LOS-18 design values.

Every event must bind:
- exact vessel configuration, mission and condition digests;
- exact equipment/revision and service owner;
- pre-event battery/source/load state;
- waveform or synthetic generation method, duration, rise/fall, repetition and coincidence group;
- time base, sample rate, calibration and time-synchronization uncertainty when measured;
- acceptance limits and their engineering owner;
- local protection/fallback owner;
- raw data digest, solver/tool version, evidence class and rights.

The current fixtures mark acceptance limits as **unqualified**. This is deliberate: absent OEM, standard, class or qualified-engineer limits, the event cannot yield a pass/fail power-quality result.

## 6. Benchmark metrics

For AC where applicable: voltage/frequency excursions and duration, phase imbalance, harmonic distortion/THD, power factor, source transient and local controller response.

For DC where applicable: bus min/max, ripple, sag/overshoot, recovery, source current peak/RMS/slew, converter current-limit operation, protection response and stability evidence.

Across all architectures: service continuity, restart/recovery, unserved service, battery C-rate/current/temperature/throughput/degradation interval, generator response/step-load acceptance, converter loss/thermal margin, added mass/volume/cost, maintenance burden and p50/p95/worst event outcome.

Never pool unlike architectures or event classes into one score. Safety/electrical acceptance is a hard gate, not a compensating weighted objective.

## 7. Baselines

Use identical event waveform, equipment, initial source state, bus architecture and acceptance limits:
1. OEM/local equipment control;
2. baseline generator+battery/inverter;
3. deterministic staged-start rule;
4. battery-only transient support;
5. battery plus DC-link capacitance;
6. supercapacitor/hybrid storage only where justified;
7. slow EMS scheduler with the *same* qualified transient acceptance oracle;
8. ideal source as a labelled theoretical bound only.

Compare MPC or Symthaea service recommendations only if they receive the same forecasts, state, constraints, compute budget and operating horizon. No control policy is allowed to bypass local protection.

## 8. Staged validation

1. **Requirements/datasheets:** actual equipment list, OEM start/current/ramp/power-quality limits, voltage, environmental and installation constraints.
2. **Analytical checks:** independent unit, energy, current and power-balance calculations; exact architecture and assumptions.
3. **Qualified solver:** AC/DC power quality, dynamics, impedance and protection using tools suitable for that architecture; document validity and uncertainty.
4. **HIL:** controller and software response with recorded/frozen traces.
5. **PHIL / bench:** only after an independently reviewed hardware boundary, current/voltage limits, isolation, emergency stop, protection and competent supervision.
6. **On-vessel commissioning:** separate design revision, class/flag/survey and operator acceptance; not implied by any prior stage.

## 9. Acceptance gates — targets, not achieved results

- Every event is bound to exact configuration, mission, conditions and equipment revision.
- No hourly aggregate is accepted as proof of millisecond or sub-second limits.
- No false-safe schedule recommendation where a required transient acceptance report is absent.
- Every critical event has a qualified limit, test method and local fallback owner before eligibility.
- Measured traces include sample rate, calibration, synchronization uncertainty, missing samples and raw digest.
- Synthetic event generation and canonical report order are deterministic where promised.
- Zero scheduler authority over protection, BMS, generator control or actuation.
- Report worst event and lifecycle burden; no assumed superiority of DC, supercapacitors or any topology.

## 10. Integration

The slow service scheduler #11008 consumes a qualified feasibility/disposition from this transient assurance owner; it does not calculate fast dynamics. Results bind into the exact-mission evidence join #10988. Vessel route/propulsion profiles remain with MARINE-DESIGN #7703 and marine operating-envelope coverage #8130. The battery/energy and service owners retain their own state/physics.

## Claim ceiling

This protocol and its synthetic fixtures establish only data-contract and campaign structure. They do not establish actual vessel power quality, electrical safety, EMC compliance, protection adequacy, battery fire safety, class/flag acceptance, seaworthiness or authority to operate. Each later result is limited to the exact architecture, equipment, event, environment, measurement chain and review scope.
