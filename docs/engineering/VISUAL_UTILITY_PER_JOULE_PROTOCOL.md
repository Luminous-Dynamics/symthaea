# Visual Utility per Joule Protocol

**Protocol:** ODRH-ENERGY-001 v0.1.0  
**Parent:** [ODRH-001 #8030](https://github.com/Luminous-Dynamics/symthaea/issues/8030)  
**Manufacturing owner:** [VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012)  
**Status:** proposed; no power or task-performance measurements claimed

## 1. Paradigm shift

Do not optimize a display only for peak brightness, resolution, refresh rate, or watts at one static patch. Measure **energy spent to deliver a defined successful visual task**, while preserving independent image-quality, latency, safety and reliability floors.

The chosen boundary must be explicit. At minimum, distinguish:
- **DUT input:** power at the monitor's declared wall-plug supply boundary;
- **display-engine rail:** power at a suitable low-voltage DC boundary, if that boundary can be measured safely and calibrated;
- **end-to-end system:** host + display + any required illumination, processing or sensor components included in the task.

These boundaries answer different questions and must never be mixed in a direct comparison. For a complete monitor powered from mains, the most appropriate existing reference is [IEC 62087-7:2018, Methods of measurement for power consumption — Part 7: Computer monitors](https://webstore.iec.ch/en/publication/27420). It covers relevant monitor technologies and static/moving content scenarios within its stated scope; its published scope excludes battery-powered monitors and network/wirelessly connected devices, and excludes certain specialized monitor categories. Consult the full standard for the prescribed measurement procedure and record the method/edition; this protocol does not reproduce it or claim compliance.

## 2. Define task utility before measurement

For each task, freeze:
- task goal and observable success/failure criterion;
- exact workload/content digest, frame-pacing/sequence and task duration or completion rule;
- DUT model/revision and configuration;
- display mode, resolution, refresh, pixel encoding/range, SDR/HDR state and OSD/profile settings;
- warm-up, ambient conditions, input/control path, and the task timing window;
- boundary and equipment included in energy accounting;
- any predeclared image-quality, temporal-response, text/readability, contrast or other task requirements that constitute hard floors.

Utility is not just the number of pixels emitted or frames rendered. The task must be completed under its declared success criterion and the required quality floors. A mode that consumes less power but fails the task is not an improvement.

## 3. Measure energy on a declared boundary

Retain time-stamped raw power observations. State the instrument, calibration status, uncertainty/range, sample interval, gaps, synchronization and integration rule. The Rust reference arithmetic is in [energy.rs](../../tools/open-display-platform-verify/src/energy.rs); its known-answer and negative tests validate computation on synthetic traces only, not a physical instrument.

For samples (time_i, power_i), a trapezoidal integration for the declared task window is:

```text
energy_joules = sum_i (
    (power_i + power_(i+1)) / 2
    * (time_(i+1) - time_i)
)
```

with power in watts, time in seconds and energy in joules. This simple expression is valid only for the retained adjacent valid samples and the defined interpolation assumptions. The implementation must define what happens at missing samples, dropped timestamps, interval boundaries, clock discontinuities and sensor overload. Do not silently bridge gaps outside the preregistered policy.

Record baseline/idle energy policy separately. Report raw task energy and any derived incremental-energy result as different metrics; never subtract idle or host-background power silently. If time boundaries differ between modes or success/failure cases, comparisons are unmatched until handled by the preregistered method.

## 4. Primary and companion metrics

Primary efficiency metric (when task outcomes are comparable):

**Energy per successful task = total declared-boundary joules / count of successful tasks**

Also report:
- number attempted, succeeded, failed, aborted and excluded;
- completion time and latency distribution;
- protected visual-quality outcomes and instrument capability status;
- power/time profile and energy-integral uncertainty;
- raw energy per run and per task, not only an average;
- module configuration, firmware, stimulus, method and analysis digests;
- repair/rework and lifecycle consequences where relevant.

If no attempt succeeds, energy per successful task is undefined; do not assign zero or infinity as an invented measurement. Report zero successes and block an efficiency ranking for that batch. If success rates differ, publish the joint energy/success results and confidence/uncertainty rather than treating energy alone as decisive.

## 5. Benchmark gates

### G0 — protocol integrity
- Every energy observation binds an explicit accounting boundary, workload/task identity, DUT configuration, instrument/calibration identity, sampling/interpolation policy, raw samples and reducer revision.
- Wall-plug, low-voltage rail and end-to-end system energy have distinct metric IDs.
- Missing or invalid samples are handled according to a frozen rule; no silent deletion or hidden extrapolation.
- A known-answer synthetic waveform validates the integration arithmetic and unit conversions.

### G1 — measurement capability
- Declare uncertainty terms and the minimum engineering effect for the decision.
- Apply the proposed ODRH decision-capability gate from the [metrology protocol](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md); inadequately resolved comparisons cannot rank candidates.
- When using an external AC power instrument for a mains-powered complete monitor, follow the applicable IEC procedure for the scope and record deviations; do not claim standards compliance merely because a meter is present.

### G2 — task utility
- 100% of candidates are tested on the same preregistered task set and success criteria.
- A candidate failing any hard quality/latency/reliability/safety floor is ineligible regardless of energy reduction.
- Report joules per success alongside success rate, duration, latency and uncertainty.
- Preserve failed tasks and zero-success cases.

### G3 — improvement
After a matched, independently reviewed baseline exists, investigate at least 10% lower energy per successful task with uncertainty supporting the threshold and no protected quality/task regression. The 10% threshold is an ODRH research target, not an achieved result or an IEC criterion.

## 6. Fail-closed cases

1. AC wall energy is compared directly with DC rail energy.
2. Host background energy is removed without a predeclared boundary and method.
3. A failed task is counted as successful or omitted from the denominator.
4. Missing power samples are bridged without a validated rule.
5. Candidate and baseline use different picture modes, patterns, APL, brightness or task criteria without a valid matching protocol.
6. A result meets energy target but fails a protected optical, temporal, thermal or reliability floor.
7. The measuring instrument lacks the range/bandwidth/calibration capability needed for the declared waveform.
8. A synthetic model or software test is presented as a physical watt/joule measurement.

## 7. Links

- [IEC 62087-7:2018 — computer monitor power measurement](https://webstore.iec.ch/en/publication/27420).
- [ICDM / IDMS](https://www.sid.org/Standards/ICDM/ICDM) for display measurement methods relevant to protected quality floors.
- [Display pattern robustness and uncertainty](DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md).
- [Display manufacturing route protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/vis-tech-001-program-2026-10-17/docs/engineering/DISPLAY_MANUFACTURING_ROUTE_PROTOCOL.md).
- [ODRH-MET-001 #8049](https://github.com/Luminous-Dynamics/symthaea/issues/8049).

The protocol remains proposed until a specific physical instrument, workload, task criterion, accounting boundary, baseline, and uncertainty budget are frozen and measured.
