# Resilient-systems benchmarks and acceptance gates

**Status:** proposed research benchmark protocol; targets below are project goals, not published standards or measured results  
**Companion:** [Feasibility portfolio](./RESILIENT_SYSTEMS_FEASIBILITY_PORTFOLIO.md) · [Machine-readable baseline](./resilient_systems_baseline.json) · [Rust screening module](../../crates/domains/symthaea-engineering/src/resilient_service_screening.rs)

## 1. Main paradigm: benchmark continuity, not a single resilience score

The central observable is a time series of essential-service fulfillment. For service (i) at step (t), let demand be (d_i(t)), deliverable capacity be (c_i(t)), and fulfillment be

[
f_i(t)=\min\left(1,\frac{c_i(t)}{d_i(t)}\right).
]

The service deficit is (u_i(t)=\max(0,d_i(t)-c_i(t))). Summed deficit over a scenario is

[
U=\sum_t\sum_i u_i(t)\,\Delta t.
]

(U) retains the caller's service units and time unit. Do not add litres to kilograms or healthcare events to kilowatt-hours without an explicitly justified normalization. Publish per-service traces alongside any aggregate.

This matches a broader finding in infrastructure-resilience research: resilience curves and summary metrics use several different performance measures, and the right metric depends on the system and decision ([Poulin & Kane, *Infrastructure resilience curves: Performance measures and summary metrics*](https://doi.org/10.1016/j.ress.2021.107926)). Work on cyber-physical infrastructure also emphasizes functionality loss, recovery time, minimum post-disturbance performance and risk rather than one isolated scalar ([2025 R-KPI framework](https://doi.org/10.1016/j.jnlssr.2024.12.005)). NIST's recovery guidance recommends defining measurable recovery metrics in advance while ensuring that measurement does not obstruct restoration ([NIST SP 800-184](https://doi.org/10.6028/NIST.SP.800-184)).

## 2. Required reported metrics

Every scenario report must include:

1. **Cumulative unmet service**, separately by service and in service-unit-steps.
2. **Minimum fulfillment** and the time of first/last breach of an explicitly declared service floor.
3. **Stable recovery time**: the first time from which every required service remains above its floor through the rest of the modeled horizon. State clearly when recovery is not observed before the horizon ends.
4. **Failure and recovery envelope**: which component failed, capacity reduction, start time, repair/recovery assumptions, common-mode scope, and dependencies.
5. **Uncertainty/evidence classification**: observed data, fitted parameter, model assumption, bounded interval or illustrative scenario.
6. **Distributional results** for any ensemble: median, 90th/95th percentile, worst specified case and confidence interval when statistical assumptions justify one.
7. **Burden and trade-offs**: life-cycle cost, power, material, maintenance labour, spare parts, emissions and affected population where the underlying data support them.

Also report scenario inputs, model commit SHA, configuration digest, run count, units, step size, random seed (if any), and failed scenarios. An absent value must remain unknown, not silently become zero.

## 3. Levels of benchmark readiness

### B0 — Schema and fail-closed checks

**Acceptance target:** 100% of defined malformed-input cases are rejected with an explicit error.

Test empty service sets, duplicate IDs, non-positive or non-finite demand, negative/non-finite capacity, unknown or duplicate dependencies, dependency cycles, invalid service floors, unknown event targets, out-of-horizon events, zero recovery duration and invalid capacity factors. Do not auto-repair inputs without recording the change.

### B1 — Deterministic reference fixtures

**Acceptance target:** repeated execution on the same code/configuration produces the same report; a nominal no-disturbance scenario has zero unmet service when capacity is sufficient; output contains a trace for every modeled service.

Maintain small hand-checkable cases where results have exact expectations. Keep units and time steps explicit.

### B2 — Metamorphic/negative-control tests

These are essential because no ground-truth field dataset may exist for many proposed concepts.

- **Redundancy control:** when the model says two independent equal sources can carry the demand and one is removed, the declared service floor is still met. This is a mathematical fixture—not proof of real-world independence.
- **Common-mode negative control:** removing both sources must cause a deficit if no substitute exists.
- **Repair monotonicity:** with all else fixed, a slower capacity recovery must not lower cumulative deficit.
- **Severity monotonicity:** lowering the starting capacity factor must not improve cumulative deficit.
- **Capacity monotonicity:** increasing nominal capacity must not worsen any service fulfillment in this model.
- **Dependency propagation:** removing a required upstream service must affect its downstream dependents according to the declared dependency contract.
- **Ordering invariance:** permuting an acyclic service list must not change service traces by identifier.
- **Fail-closed control:** malformed inputs never yield a seemingly valid report.

**Acceptance target:** every required property has a named test and a deliberate negative control. A test passing validates this implementation property, not a prediction about the physical world.

### B3 — Calibration against observations

Before using the simulator to predict an actual settlement or supply chain:

- Reproduce at least **three** observed outage/recovery traces covering distinct failure classes.
- Reserve at least one event or time window as hold-out data.
- Publish parameter-estimation procedures and avoid tuning on the held-out event.
- Report error in the measured native service units, error in recovery time, and uncertainty.
- Compare against a trivial baseline (e.g. no cascade, fixed restoration time, or last-value forecast) to show the model adds information.

**Proposed acceptance target:** pre-register error tolerances with domain owners before fitting. No universal percentage is specified here because the measurement noise and consequences differ by service. If the model does not beat the simple baseline on hold-out cases, do not use it for prediction.

This approach is consistent with recent water-energy microgrid work integrating optimization, outage scenarios and survival analysis; it demonstrates a possible method, not a transferable parameter set ([2025 water-energy microgrid study](https://doi.org/10.1016/j.nexus.2025.100590)). Recent supply-chain models similarly add temporal node state and recovery rather than considering only static topology ([Shi et al., 2026](https://doi.org/10.1016/j.ress.2025.111763); [Prasetia, 2026](https://link.springer.com/article/10.1007/s10479-026-07406-4)).

### B4 — Intervention comparison

Compare a declared reference configuration with interventions such as spare inventory, a second qualified source, local repair, modular replacement, improved sensing, load shedding, alternative transport or energy storage.

**Acceptance target:** an intervention is promoted only if it improves a predeclared set of service metrics across the required scenario suite without hiding a newly introduced critical dependency. Report the Pareto frontier, not just the best weighted score. Include intervention cost, upkeep, false alarms, repair skill and the risk that “redundant” sources fail together.

### B5 — Decision-grade use

Require external domain review, traceable input data, uncertainty and sensitivity analysis, representative validation, documented model limitations, and a named decision owner. Life-safety, structural certification, pressure-boundary qualification, healthcare, lunar-readiness or aviation approval require their own standards and physical evidence. This screening model is not a surrogate for those processes.

## 4. Initial benchmark suite

Every release of the screen should include these fixed cases:

| ID | Scenario | Expected result / falsifier |
|---|---|---|
| RS-01 | Nominal, sufficient capacity | Zero unmet service; no floor breach |
| RS-02 | Deliberate undersizing | Deficit shown even without an outage |
| RS-03 | Redundant capacity, one independent source lost | Service floor holds when modeled residual capacity is sufficient |
| RS-04 | Same sources lost by a common-mode event | Deficit appears; no false redundancy credit |
| RS-05 | Essential upstream power loss | Declared downstream dependencies are degraded |
| RS-06 | Slow versus fast restoration | Slow restoration does not improve cumulative deficit |
| RS-07 | Mild versus severe capacity loss | More severe loss does not improve cumulative deficit |
| RS-08 | Two independent simultaneous disruptions | Both effects are reflected; no event silently dropped |
| RS-09 | Unknown dependency | Explicit validation error |
| RS-10 | Dependency cycle | Explicit validation error |
| RS-11 | Invalid demand/capacity/floor/recovery parameters | Explicit validation error |
| RS-12 | Permuted acyclic service input order | Same per-service results |
| RS-13 | Horizon ends before recovery | Recovery stays unknown, not falsely reported as complete |
| RS-14 | Service returns above floor briefly, then falls below again | Stable recovery is after the final violation, not the first brief recovery |

This first Rust tranche implements a minimal version of the model and unit tests for many of these properties. The complete RS-01–RS-14 catalog is the target coverage; cases not yet implemented must stay marked as gaps.

## 5. Domain-specific benchmark targets

### 100 people for 730 days

- Close hourly/seasonal balances for electricity, heat, potable water, process water, calories, micronutrients, sanitation, medicines and maintenance.
- Publish reserve assumptions and a minimum service floor per service.
- Test 3-, 14- and 30-day power disruptions; water treatment outage/contamination; seasonal crop loss; transport isolation; and loss of a unique repair skill.
- Never claim self-sufficiency from annual averages alone; include ramp, storage, repair, degraded operation and restart.

### Supply network

- Include random single-node loss, the critical single node, sole qualified-source loss, regional/common-mode loss and a repair-skill outage.
- Compare static centrality against time-dependent service deficits and recovery.
- Test candidate mitigations on held-out historical disruptions.
- Report concentration and common upstream dependencies, not just supplier counts.

### Community microfactory

- Test a real catalog of 50–100 common parts and select the first 10 after full-cost and criticality screening.
- Measure lead time avoided, stockout reduction, accepted yield, rework, quality escapes, kWh, consumables and operator dependency.
- Exclude safety-critical parts unless process qualification, material provenance and required testing are established.

### Transport

- Use identical cargo, origin/destination, deadline and safety constraints.
- Include full delivery chains, empty return legs, utilization, weather closures, loading, permits, infrastructure, maintenance and recovery.
- Report cost per delivered tonne and deadline reliability alongside energy and emissions.

### Self-repairing structures

- Compare baseline inspection, modular replacement, sensing + repair and integrated self-healing approaches.
- Measure detection probability by defect type/location, false negatives, isolation time, residual capacity, recovered properties and repeated-cycle durability.
- Credit restored structural capacity only after representative tests of that property.

### Lunar industry

- State the customer mission and alternative imported solution.
- Include plant, power, storage, tooling, spares, rejects, wear and maintenance over the mission lifetime.
- Measure usable/qualified output—not nominal printed or excavated mass—and distinguish terrestrial simulator validation from lunar qualification.

### Minimum viable civilization

- Define floors for service access, health, rights, accessibility, food/water quality, education and ecological burdens alongside physical quantities.
- Report subgroup-level service loss as well as total deficit.
- Treat governance and rights as constraints requiring accountable human deliberation, not weights automatically optimized by a simulator.

## 6. Implementation limits of the current Rust model

The first implementation deliberately uses a small abstraction:

- service capacity and demand share a caller-defined unit;
- dependencies propagate the weakest upstream fulfillment fraction;
- disruption capacity factors multiply and recover linearly;
- output is deterministic for fixed inputs.

It does **not** yet model queues, inventory depletion, stochastic demand, repair crews, multiple resource types in one process, spatial effects, failure probabilities, confidence intervals, or real physical laws. Those must be added only with explicit contracts and reference fixtures. The initial model is useful for testing causal direction and scenario consistency; it is not yet calibrated for predicting a community's actual outage.

## 7. Research references

- Poulin, C. & Kane, M. B. (2021), [Infrastructure resilience curves: Performance measures and summary metrics](https://doi.org/10.1016/j.ress.2021.107926).
- NIST SP 800-184, [Guide for Cybersecurity Event Recovery](https://doi.org/10.6028/NIST.SP.800-184).
- [Resilience in Cyber-Physical Infrastructures: R-KPI prioritization](https://doi.org/10.1016/j.jnlssr.2024.12.005), 2025.
- [Modeling and operation of water-energy microgrids considering resilience assessment](https://doi.org/10.1016/j.nexus.2025.100590), 2025.
- Shi et al., [Temporal modeling and resilience analysis of supply chain networks under cascading failures](https://doi.org/10.1016/j.ress.2025.111763), 2026.
- Prasetia, [Agents of collaborative adaptation: modelling temporal resilience in supply chains](https://link.springer.com/article/10.1007/s10479-026-07406-4), published September 3, 2026.

---
*Integrity rule: simulated benchmarks, skipped CI, and successful source-level checks are not evidence of field reliability. Report each evidence class separately.*
