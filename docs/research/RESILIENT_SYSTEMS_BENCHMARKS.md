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
- **Resource conservation:** shared resource stocks remain within [0, storage capacity]; consumption is no more than stock plus actual replenishment.
- **Stockout control:** if replenishment stops and a required resource is exhausted, affected services cannot keep receiving uncharged outputs.

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

## 4. Proposed performance and reproducibility budgets

These are **unmeasured engineering targets**, not results or guaranteed requirements. Establish a named reference host and actual benchmark harness before treating them as release gates. The resource-constrained model has separate, looser targets below because it also tracks inventories and per-service consumption.

- **Small reference case:** 100 services, up to 200 dependency edges, 20 disruption records and 730 time steps; target p95 runtime below 100 ms in a release build.
- **Batch case:** 1,000 scenarios of that size; target completion below 60 seconds on the same host.
- **Large screening case:** 1,000 services, up to 2,000 dependency edges and 730 steps; target one deterministic run below 2 seconds.
- **Reproducibility:** the same Rust toolchain, binary, inputs and configuration must produce identical per-service trajectories; any intentional numerical-method change must update a versioned reference fixture.
- **Traceability:** record CPU, OS, `rustc -Vv`, build profile, commit SHA, configuration digest, service count, dependency-edge count, disruption count, horizon and timing statistic.

For the finite-resource model, proposed targets are: 100 services, 20 resources, up to 500 service-resource requirements, 200 dependency edges and 730 time steps below 250 ms per scenario; 1,000 such scenarios below 120 seconds; and a 1,000-service / 2,000-requirement / 730-step case below 5 seconds. All are unmeasured planning targets. If either implementation misses a budget, profile it before weakening model detail. Do not reduce time resolution, discard per-service/resource traces, skip validation or drop scenarios merely to meet a speed target. These targets should be revised using measured data from a documented reference machine before becoming normative.

## Resource-coupling benchmark extension

The next paradigm step is from independent service curves to **finite shared resource dynamics**. This reduces a common modeling failure: a system appears resilient because every service has nominal capacity, while real operations compete for the same finite energy, water, fuel, materials, storage, or repair resources.

A first deterministic implementation is available in [`resource_constrained_screening.rs`](../../crates/domains/symthaea-engineering/src/resource_constrained_screening.rs). It treats supplies as explicit stocks with bounded replenishment and storage, lets a declared service priority policy allocate shared stocks, and reports each service and resource trajectory. The priority policy is intentionally visible: lower-priority services can be starved. It is a research abstraction, not a claim that fixed priorities are ethically or operationally optimal.

Add the following reference fixtures before using it as a planning model:

- **RC-01 stock conservation:** inventory remains non-negative and never exceeds storage capacity.
- **RC-02 consumption conservation:** resource consumed equals the sum of delivered output times declared per-unit use, and cannot exceed available stock.
- **RC-03 finite stockout:** with no replenishment, a service requiring the resource must become resource-limited once stock is exhausted.
- **RC-04 supply interruption:** a supply event affects replenishment, not previously stored inventory; recovery follows the explicit event rule.
- **RC-05 dependency + resource bottleneck:** upstream service availability and finite materials can both limit downstream delivery.
- **RC-06 explicit allocation policy:** competing services obey the declared priority and a stable identifier tie-break; report per-service impacts so aggregate metrics do not hide starvation.
- **RC-07 storage bound:** excess replenishment is curtailed at the declared storage limit and does not accumulate off-ledger.
- **RC-08 resource substitution (future):** a declared substitute resource is credited only when its conversion, quality, qualification and switching delay are separately represented.

Acceptance targets: resource conservation error within a predeclared floating-point tolerance for valid inputs; zero negative stock; zero storage-capacity overshoot; no hidden replenishment; deterministic traces; and a negative control showing service shortfall after stocks and substitute supplies are removed. Any normalization across unlike service units must be justified and documented. Prefer vectors of per-service deficit and recovery to an unqualified cross-service sum.

Recent evidence supports explicitly modeling time, resource interruptions and recovery rather than static component counts. A 2025 open-access island-microgrid study models load and energy management under resource interruptions ([Scientific Reports](https://doi.org/10.1038/s41598-025-99974-x)); a 2026 study compares centralized and decentralized reconfiguration under physical microgrid vulnerability ([Scientific Reports, July 20, 2026](https://www.nature.com/articles/s41598-026-63123-9)); and a 2026 supply-chain study combines disruption-propagation simulation with calibration and optimization ([Computers & Industrial Engineering, July 2026](https://doi.org/10.1016/j.cie.2026.112063)). These inform model structure; they do not validate our parameters or code.

## 5. Initial benchmark suite

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

## 6. Domain-specific benchmark targets

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

## 7. Implementation limits of the current Rust models

The service-continuity implementation deliberately uses a small abstraction:

- service capacity and demand share a caller-defined unit;
- dependencies propagate the weakest upstream fulfillment fraction;
- disruption capacity factors multiply and recover linearly;
- output is deterministic for fixed inputs.

It does **not** yet model queues, inventory depletion, stochastic demand, repair crews, multiple resource types in one process, spatial effects, failure probabilities, confidence intervals, or real physical laws. Those must be added only with explicit contracts and reference fixtures. The initial model is useful for testing causal direction and scenario consistency; it is not yet calibrated for predicting a community's actual outage.

## Repair-crew scheduling extension

The third Rust layer is [`repair_crew_scheduling.rs`](../../crates/domains/symthaea-engineering/src/repair_crew_scheduling.rs). It compares priority-first, shortest-job-first, and greedy weighted-service-recovery policies while enforcing skill eligibility, precedence, crew exclusivity, outage start times, and finite spare-part reservations.

This fills an important literature-backed gap. Xu et al.'s 2026 review describes repair-resource scheduling as difficult precisely because disruptions, network interdependencies, uncertain field conditions, and stakeholder priorities interact ([Risk Analysis, first published 12 March 2026](https://doi.org/10.1111/risa.70220)). Pan and Boyles' 2026 volume paper models two-way road/power interdependency, including the effect of power failures on crew travel and the effect of transport delays on restoration ([doi:10.1080/10286608.2025.2602872](https://doi.org/10.1080/10286608.2025.2602872)). Robust multicrew road-restoration scheduling also explicitly addresses uncertain repair durations ([Computers & Operations Research, April 2026](https://doi.org/10.1016/j.cor.2025.107379)). These results motivate our next layers; they do not show that this small heuristic is optimal.

### Required repair-scheduling tests

- **RCREW-01 — Feasible skill assignment:** every job is assigned only to a crew with the declared skill.
- **RCREW-02 — Crew exclusivity:** a crew never has overlapping jobs.
- **RCREW-03 — Precedence:** a dependent repair cannot start before all prerequisites finish.
- **RCREW-04 — Parts conservation:** reserved parts never exceed initial stock and remaining stock never goes negative.
- **RCREW-05 — Horizon accounting:** a job finishing after the horizon is counted as unfinished within horizon; loss is bounded by the modeled horizon.
- **RCREW-06 — Determinism:** identical inputs and policy produce identical schedules.
- **RCREW-07 — Policy witness:** a disclosed synthetic two-job case gives priority-first weighted loss 1,110 service-unit-steps and weighted-recovery loss 111 (90% lower): one duration-10 / weight-1 job has priority 0, while a duration-1 / weight-100 job has priority 1. This is a transparent fixture calculated independently of the Rust implementation, not empirical evidence or a universal superiority claim.
- **RCREW-08 — Infeasibility honesty:** cycles, unknown skills, missing parts and insufficient spares fail explicitly.
- **RCREW-09 — Preserve scarce skills:** when two crews can do the same job, prefer the less-flexible crew on an otherwise tied decision so a versatile crew remains available for a job with fewer alternatives. Report whether the tie-break actually improves completion or weighted downtime.

### Scheduling benchmarks and open limitations

1. Compare all policies on identical job, crew, parts, outage and dependency inputs. Report total weighted service-unit-steps lost, time each service is restored, completion share within horizon, spare usage, fairness/distributional outcomes, and infeasible jobs.
2. The Rust tests now include an independent permutation-enumeration oracle for one disclosed two-job/one-crew fixture. Extend this to a published corpus of small cases (initially no more than 8 jobs), respecting skill and precedence feasibility, and compare heuristic objective gap with the exact optimum. Target **within 10% of optimum on at least 95% of the corpus**, but only promote that as an acceptance gate after the corpus and objective are reviewed. This remains an aspiration, not a result.
3. For 10–100 jobs, use deterministic scenario suites and compare heuristic outcomes with priority-first, shortest-job-first and a valid lower bound where available. Publish worst-case as well as average gap; do not label a heuristic optimal.
4. Next iterations should add uncertain repair durations, travel/access times and power/transport interdependency. The model currently has none of these, no online replanning, no shared tool constraint, and no policy-level fairness guarantee.
5. The weighted-service objective is caller-defined and can encode harmful priorities. Keep service-level floors and group impacts as explicit constraints; never treat a scalar weight as an ethical decision.\n6. Initial performance target (unmeasured): 100 jobs, 20 crews, 50 part types, and 200 precedence edges below 250 ms p95 on a documented release-build host; a 1,000-job/100-crew stress case below 10 s. Profile before setting release gates, and retain feasibility checks and full traces.

## 8. Research references

- Poulin, C. & Kane, M. B. (2021), [Infrastructure resilience curves: Performance measures and summary metrics](https://doi.org/10.1016/j.ress.2021.107926).
- NIST SP 800-184, [Guide for Cybersecurity Event Recovery](https://doi.org/10.6028/NIST.SP.800-184).
- [Resilience in Cyber-Physical Infrastructures: R-KPI prioritization](https://doi.org/10.1016/j.jnlssr.2024.12.005), 2025.
- [Modeling and operation of water-energy microgrids considering resilience assessment](https://doi.org/10.1016/j.nexus.2025.100590), 2025.
- Shi et al., [Temporal modeling and resilience analysis of supply chain networks under cascading failures](https://doi.org/10.1016/j.ress.2025.111763), 2026.
- Prasetia, [Agents of collaborative adaptation: modelling temporal resilience in supply chains](https://link.springer.com/article/10.1007/s10479-026-07406-4), published September 3, 2026.\n- Xu, M. et al. (2026), [Scheduling Repair Resources for Post-Disaster Critical Infrastructure Systems: A Review of Models and Algorithms](https://doi.org/10.1111/risa.70220).\n- Pan, S. & Boyles, S. (2026), [Optimising repair sequences for interdependent infrastructure resilience](https://doi.org/10.1080/10286608.2025.2602872).\n- [The Robust Multicrew Scheduling and Routing Problem in road restoration](https://doi.org/10.1016/j.cor.2025.107379), April 2026.\n- Hosseina, Moghaddam & Hassannia (2025), [Optimizing energy and load management in island microgrids for enhancing resilience against resource interruptions](https://doi.org/10.1038/s41598-025-99974-x).\n- Pourramezani et al. (2026), [Physical vulnerability-oriented reconfiguration of microgrids: centralized vs. decentralized control](https://www.nature.com/articles/s41598-026-63123-9).\n- [Simulation-driven optimization for resilient and sustainable supply chain operations](https://doi.org/10.1016/j.cie.2026.112063), July 2026.

---
*Integrity rule: simulated benchmarks, skipped CI, and successful source-level checks are not evidence of field reliability. Report each evidence class separately.*
