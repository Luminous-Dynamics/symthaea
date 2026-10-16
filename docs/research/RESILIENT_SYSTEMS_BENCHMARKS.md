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
4. Next iterations should add uncertain repair durations, travel/access times and power/transport interdependency. The fixed scheduler itself has no online replanning, shared tool constraint or policy-level fairness guarantee; a separate rolling-horizon module now provides a first event-driven dispatcher but does not model travel or resupply.
5. The weighted-service objective is caller-defined and can encode harmful priorities. Keep service-level floors and group impacts as explicit constraints; never treat a scalar weight as an ethical decision.\n6. Initial performance target (unmeasured): 100 jobs, 20 crews, 50 part types, and 200 precedence edges below 250 ms p95 on a documented release-build host; a 1,000-job/100-crew stress case below 10 s. Profile before setting release gates, and retain feasibility checks and full traces.

## Duration-uncertainty evaluation

The fourth module, [`robust_repair_evaluation.rs`](../../crates/domains/symthaea-engineering/src/robust_repair_evaluation.rs), adds **out-of-sample evaluation** of the three existing repair policies. Each initial schedule is built using nominal durations. Its dispatch order and crew assignments are then held fixed while that same plan is replayed with alternative realized-duration scenarios. Realized durations can delay work through crew contention and prerequisite constraints, but cannot change the initial decision using future knowledge.

This explicit information boundary matters. Stochastic scheduling has long recognized the need for nonanticipativity: decisions must be based on information available when they are made, or the computed plan can be unattainable in practice ([nonanticipativity in stochastic project scheduling](https://doi.org/10.1016/0360-8352(96)00119-2)). Risk-averse restoration research also compares expected performance with worst-case and CVaR-based outcomes under uncertain repair and travel times ([Alkhaleel, Liao & Sullivan](https://doi.org/10.1016/j.ejor.2021.04.025)). The April 2026 robust multicrew road-restoration paper uses an explicit uncertainty set for uncertain repairs ([Moreno et al., 2026](https://doi.org/10.1016/j.cor.2025.107379)).

### What is evaluated

- Expected weighted service-unit-steps lost under explicitly supplied scenario weights.
- Worst-case loss across the supplied scenario set.
- Upper-tail CVaR at a user-declared `alpha`, averaging the worst (1-\alpha) weighted fraction of outcomes.
- Per-scenario repair start/finish times, realized duration, crew assignment, completion share and loss.
- A fixed-plan replay where nominal decisions are preserved; future durations cannot rearrange the initial dispatch order.

Scenario weights must sum to one, but they are **not probabilities** unless justified by data. With no defensible probability model, use named stress scenarios and interpret expected loss only as a weighted scenario score. Always report worst-case and per-scenario results alongside CVaR.

### Required tests

- **DUR-01 — Nonanticipativity:** the nominal job/crew decisions are identical across duration scenarios; only resulting times and consequences differ.
- **DUR-02 — Tail risk:** a rare slow-repair scenario raises worst-case/CVaR even when expected loss looks modest.
- **DUR-03 — Monotonicity:** lengthening a realized repair cannot reduce its loss in the fixed-plan replay.
- **DUR-04 — Dependency delay:** a slower prerequisite delays dependent restoration.
- **DUR-05 — Valid scenario weights:** invalid, missing, repeated or unknown duration records fail closed.
- **DUR-06 — Determinism:** identical scenarios and nominal inputs reproduce identical policy-risk summaries.
- **DUR-08 — Crew readiness:** scenario replay preserves a crew's declared initial availability time.
- **DUR-07 — Risk-report completeness:** expected, worst-case and CVaR results all accompany per-scenario traces; no summary silently replaces the details.

### Targets and limits

- Keep **nominal plan selection** separate from **scenario replay**; prohibit future information leaking into the starting plan.
- Include at least 3 duration scenarios per tiny fixture (ordinary, adverse and severe) when constructing a published stress set; use calibrated scenario weights only when historical data supports them.
- Compare all three policies on identical scenarios and report expected loss, worst-case loss, CVaR, and scenario-by-scenario regret relative to the best *nonanticipative* plan found by an independent small-instance oracle.
- Next, implement true rolling-horizon replanning as a separate policy class that may use only observations revealed up to the current step. Compare its regret, service-floor violations and action stability against the fixed-plan baseline.
- Current robust-duration coverage is a starting point, not a validated uncertainty model. Scenario generation, uncertainty calibration and travel uncertainty remain future work. Adaptive/online dispatch is now implemented in a separate module and paired with the fixed-plan evaluator; a published corpus, holdout split and confidence intervals remain open. The proposed, unmeasured batch target is 100 jobs, 20 crews and 20 duration scenarios under 1 second on a documented release-build host; a 500-job/50-crew/100-scenario stress case targets under 15 seconds. Validate the harness and host before adopting either as a release gate.

## 8. Research references

- Poulin, C. & Kane, M. B. (2021), [Infrastructure resilience curves: Performance measures and summary metrics](https://doi.org/10.1016/j.ress.2021.107926).
- NIST SP 800-184, [Guide for Cybersecurity Event Recovery](https://doi.org/10.6028/NIST.SP.800-184).
- [Resilience in Cyber-Physical Infrastructures: R-KPI prioritization](https://doi.org/10.1016/j.jnlssr.2024.12.005), 2025.
- [Modeling and operation of water-energy microgrids considering resilience assessment](https://doi.org/10.1016/j.nexus.2025.100590), 2025.
- Shi et al., [Temporal modeling and resilience analysis of supply chain networks under cascading failures](https://doi.org/10.1016/j.ress.2025.111763), 2026.
- Prasetia, [Agents of collaborative adaptation: modelling temporal resilience in supply chains](https://link.springer.com/article/10.1007/s10479-026-07406-4), published September 3, 2026.\n- Xu, M. et al. (2026), [Scheduling Repair Resources for Post-Disaster Critical Infrastructure Systems: A Review of Models and Algorithms](https://doi.org/10.1111/risa.70220).\n- Pan, S. & Boyles, S. (2026), [Optimising repair sequences for interdependent infrastructure resilience](https://doi.org/10.1080/10286608.2025.2602872).\n- [The Robust Multicrew Scheduling and Routing Problem in road restoration](https://doi.org/10.1016/j.cor.2025.107379), April 2026.\n- [The role of nonanticipativity constraints in stochastic project scheduling](https://doi.org/10.1016/0360-8352(96)00119-2) — decisions must use only information available at decision time.\n- Hosseina, Moghaddam & Hassannia (2025), [Optimizing energy and load management in island microgrids for enhancing resilience against resource interruptions](https://doi.org/10.1038/s41598-025-99974-x).\n- Pourramezani et al. (2026), [Physical vulnerability-oriented reconfiguration of microgrids: centralized vs. decentralized control](https://www.nature.com/articles/s41598-026-63123-9).\n- [Simulation-driven optimization for resilient and sustainable supply chain operations](https://doi.org/10.1016/j.cie.2026.112063), July 2026.

---
*Integrity rule: simulated benchmarks, skipped CI, and successful source-level checks are not evidence of field reliability. Report each evidence class separately.*


## 8. Event-triggered rolling-horizon repair dispatch

A fifth Rust module, [rolling_horizon_repair.rs](../../crates/domains/symthaea-engineering/src/rolling_horizon_repair.rs), adds **event-triggered recourse** to the fixed-plan baseline. At each observed completion/release event it rebuilds the eligible set from completed prerequisites, available qualified crews, outage release times, and remaining spare stock. It then dispatches one repair at a time per free crew. Parts are consumed only when a repair starts.

The information boundary is deliberately narrow: nominal duration ranks pending jobs; the realized duration is looked up only after the job has been selected, and only schedules the job's completion event. The report records decision time, eligible job IDs, completed jobs observed, available crews, dispatch choice, remaining stock, and service-unit-step loss. This is a simulation contract—not cryptographic proof of nonanticipativity and not field validation. The paired fixed-versus-adaptive scenario harness is implemented separately in [repair_policy_comparison.rs](../../crates/domains/symthaea-engineering/src/repair_policy_comparison.rs); empirical calibration, a versioned holdout corpus, and statistical confidence intervals remain future work.

The paradigm shift is from treating the schedule as a commitment that must remain fixed to treating it as a **revisable policy operating on observed state**. A scenario can therefore change the next decision through completion timing, newly released jobs, dependency availability, or stock depletion, without granting the policy future repair durations. The fixed robust replay stays important as a paired baseline; adaptive dispatch must not be assumed superior in every case.

The method is consistent with rolling-horizon restoration research that updates the plan as information about damaged components and field conditions changes. Wei et al. describe adjustable restoration windows with repair and travel times on standard feeder cases ([2024 preprint](https://arxiv.org/abs/2404.03197)). Research on repair-sequence scheduling likewise emphasizes resource-constrained sequence quality and comparison with alternative heuristics ([Xu et al., 2019](https://doi.org/10.1111/mice.12435)). These papers motivate the experiment; they do not validate this implementation.

### RH reference fixtures

- **RH-01 — Nonanticipative first action:** when two cases have identical observable state but different unrealized durations, the current decision must be identical.
- **RH-02 — Replan after observed delay:** a long realized repair can cause a newly released high-priority job to move ahead of a lower-priority pending job.
- **RH-03 — Completion-gated dependency:** dependent work cannot start before its prerequisite is observed complete.
- **RH-04 — Crew legality:** skill eligibility and single-job-at-a-time constraints hold at every decision.
- **RH-05 — Stock conservation:** parts are charged at dispatch, never pre-reserved for work that never starts, and stock never becomes negative.
- **RH-06 — Release and availability:** neither jobs nor crews are used before their declared release/availability time.
- **RH-07 — Horizon boundaries:** a repair finishing exactly at the horizon counts as complete; a later finish remains in progress; unstarted work is explicit.
- **RH-08 — Determinism and auditability:** identical inputs reproduce the same event trace and report.
- **RH-09 — Paired policy comparison:** fixed-plan replay and rolling-horizon policy consume identical realized-duration scenarios. Report per-case deltas, median, mean, upper-tail CVaR and worst specified case.
- **RH-10 — Counterexamples retained:** publish cases where adaptation loses as well as wins; do not claim adaptive dispatch dominates every scenario.
- **RH-11 — Oracle separation:** an optional clairvoyant/ perfect-information solution is an upper bound only, never described as an implementable policy.

### Proposed acceptance and performance targets

These are targets, not measured results or universal standards.

1. **Zero information-boundary violations** in all defined RH-01 paired cases. The dispatcher has a first-decision information-boundary test and the separate paired evaluator is implemented; a larger published paired corpus and held-out validation remain future work.
2. **Small-instance quality:** generalize the exact oracle to a public corpus of at most eight jobs and aim to be within 10% of exact optimum on at least 95% of feasible cases. Label every instance where the heuristic exceeds the gap; do not tune on the holdout corpus.
3. **Decision value:** on a preregistered, paired scenario suite, report the distribution of adaptive-versus-fixed service-loss deltas. Treat a median reduction of at least 10% as an aspirational research target, not a release gate; also publish any floor violation or tail-risk regression.
4. **Proposed runtime budget:** 100 jobs, 20 crews and 20 duration scenarios in under 1 second for the complete release-build batch; stress case 500 jobs, 50 crews and 100 scenarios in under 15 seconds. Both are unmeasured targets pending a named reference host.
5. **Evidence gate:** before predictive use, add access/travel times, resupply events and observed-duration calibration; retain deterministic negative controls and compare against a simple baseline on held-out events.

The initial adaptive fixture is synthetic. On its one-crew, three-job trace, fixed nominal dispatch order is A→C→B. With A taking 10 rather than its nominal 1 step and B becoming eligible at step 5, event-triggered priority dispatch reorders the still-pending work to A→B→C. The fixed-order realized loss is 922 versus 724 for adaptive dispatch on the declared weights: a 21.5% reduction for this one designed witness. This is a reproducible scenario claim only; it is not an average improvement claim, nor evidence of real-world predictive value.

Current source-level coverage is 11 authored unit tests for this module. None of the Rust tests in this portfolio have been compiled or run in the current environment. The paired evaluator now runs a deterministic batch comparison, but it does not yet estimate statistical confidence intervals or calibrate scenarios against operational records.


## 9. Paired fixed-plan versus adaptive evaluation

The paired evaluator, [repair_policy_comparison.rs](../../crates/domains/symthaea-engineering/src/repair_policy_comparison.rs), connects the frozen-plan robust replay and rolling-horizon dispatcher. For each of the three repair policies, it sends the same nominal jobs, crew pool, spare stocks, horizon and realized-duration scenarios through both methods. Each comparison retains fixed-plan schedule, adaptive event trace and per-case loss delta.

Use the sign convention **adaptive loss minus fixed-plan loss**: negative is better for the adaptive run; positive is worse. Report expected loss, worst specified loss, upper-tail CVaR, expected delta, relative expected reduction when defined, counts of improved/worsened/tied scenarios, the sum of scenario weights for wins/losses, and the trace for every paired case. Scenario weight totals are descriptive weighted mass—not empirical probabilities unless backed by evidence.

### PC acceptance gates

- **PC-01 — Exact pairing:** fixed and adaptive records have the same scenario IDs, weights, horizon, and job-duration realization.
- **PC-02 — Arithmetic integrity:** paired delta equals adaptive loss minus fixed-plan loss for every scenario; expected delta equals the weighted sum of paired deltas.
- **PC-03 — Risk visibility:** fixed and adaptive expected loss, worst-case loss and CVaR appear side by side.
- **PC-04 — No winner-only reporting:** retain improved, worsened and tied cases with the decision trace. The aggregate cannot imply universal dominance.
- **PC-05 — Reproducibility:** identical inputs reproduce an identical comparison report.
- **PC-06 — Honest baseline:** the fixed strategy remains the existing nominal-plan replay, not a weakened straw man; a clairvoyant optimizer, if ever added, is separately labeled as an unattainable upper bound.

### Targets and next research gate

The paired harness now exists and has six authored tests, but none have been executed in this environment. The synthetic A→C→B versus A→B→C witness is a proof-of-concept fixture, not a benchmark distribution. Next, publish a deterministic corpus with small exact-optimality cases, paired scenario IDs and a holdout split; include both adaptive wins and regressions. Then add uncertainty calibration, travel/access and resupply only behind dedicated fixtures.

Statistical targets remain proposed and unmeasured: 100 jobs × 20 crews × 20 scenarios in under 1 second for a release-build batch, and 500 jobs × 50 crews × 100 scenarios in under 15 seconds on a documented reference host. Report both compute cost and policy quality; do not relax information boundaries or remove traces to meet runtime.

## 10. Exact tiny-instance oracle and public fixture corpus

The six-fixture, machine-readable corpus is [repair_scheduling_oracle_corpus.json](./repair_scheduling_oracle_corpus.json). The independent Rust oracle is [repair_small_instance_oracle.rs](../../crates/domains/symthaea-engineering/src/repair_small_instance_oracle.rs). It exhaustively evaluates all permutations for a deliberately limited problem class: up to eight jobs, one hypothetical crew qualified for every job, all outages starting at time zero, no precedence constraints and no spare parts. The objective is weighted completion time. Scope violations fail closed rather than being silently ignored. The module also exposes an optimality-gap evaluator for the priority-first, shortest-job-first and weighted-recovery policies, returning each policy's schedule order, loss, absolute gap and relative gap from the exhaustive optimum.

This gives us an exact answer on instances small enough to enumerate, while larger instances need separate lower/upper bounds or best-known solutions. The oracle is not intended for operational schedules with travel, constraints or uncertainty. ORACLE-01–ORACLE-05 have hand-checkable expected outcomes. ORACLE-06 was independently exhaustively enumerated in JavaScript across all 40,320 permutations, giving loss 406 and order b→f→d→h→c→g→e→a. The corresponding Rust test pins that expected reference but has not been executed in this environment; do not turn an authored expected result into a Rust test-pass claim.

### Current fixture corpus

| ID | Purpose | Expected exact reference |
|---|---|---|
| ORACLE-01 | Expose priority-first objective counterexample | optimum 111; priority-first 1,110 |
| ORACLE-02 | Three-job weighted completion | order B→C→A; objective 28 |
| ORACLE-03 | Four-job policy-quality case | order A→B→C→D; objective 113 |
| ORACLE-04 | Equal-ratio tie-break | order a→z; objective 14 |
| ORACLE-05 | Single-job boundary | objective 15 |
| ORACLE-06 | Maximum supported exhaustive search | loss 406; order b→f→d→h→c→g→e→a; 40,320 permutations; independently enumerated in JS, Rust test pending |

The general principle is an **oracle ladder**: exhaustive exact enumeration for tiny inputs; exact MILP/CP-SAT or branch-and-bound for medium validation cases; best-known bounds plus runtime/quality reporting for larger instances; then paired rolling-horizon tests on observed restoration traces. Published road-restoration instances are available, including a systematic network-instance database for repair/restoration studies ([Almeida et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9679711/)) and benchmark studies of crew scheduling/routing ([Moreno et al., 2024](https://doi.org/10.1287/trsc.2023.0227)). The new module is not yet a solver for those richer problems. Also, this narrow single-machine weighted-completion objective is already solved optimally by Smith's weighted-shortest-processing-time rule: the oracle is a correctness check and a gap benchmark for alternative heuristics, not a novel scheduling result ([scheduling overview](https://doi.org/10.1016/j.orl.2013.08.001)).

Fresh research reinforces the direction toward feedback-driven restoration: a 2026 rolling-horizon co-optimization study models changing repair conditions, crews, network operation and distributed generation, reporting results on modified IEEE 33- and 123-node feeders ([ScienceDirect record](https://www.sciencedirect.com/science/article/abs/pii/S0951832026013347)). Its reported values are study-specific and must not be used as expected results for this simpler simulator.

### New target

- **Oracle correctness:** 100% agreement with exhaustive objective values on the published fixtures; each permutation count must equal n!.
- **Heuristic quality:** compute per-instance absolute and relative optimality gaps for all three greedy policies; publish full instance-level rows, not only aggregate averages. ORACLE-01 expects priority-first loss 1,110 (absolute gap 999; relative gap 9.0, or 900%), while shortest-job-first and weighted-recovery both match loss 111.
- **Determinism:** stable objective and lexicographic tie-break under input permutation.
- **Benchmark hygiene:** the hand-built corpus is a correctness suite, not a statistical holdout. Build separate versioned training/tuning and holdout seeds before making general policy-quality claims.

## 11. Exact multi-crew repair oracle

The new [multi-crew oracle](../../crates/domains/symthaea-engineering/src/multi_crew_repair_oracle.rs) enumerates precedence-ready job choices and skill-qualified crew assignments on tiny instances (maximum six jobs, three crews). It computes an exact minimum for weighted service loss / weighted completion under its declared restrictions and reports optimality gaps for priority-first, shortest-job-first, and weighted-recovery list schedulers.

The first [multi-crew corpus](./multi_crew_repair_oracle_corpus.json) has three independent arithmetic fixtures: (MC-01) skill-constrained parallel scheduling, optimum 41; (MC-02) a cross-crew power→pump prerequisite, optimum 52; and (MC-03) preserving a multi-skilled crew for a plumbing-only repair, optimum 42. The source-level unit tests assert each objective, key start/finish times and dispatch-sequence counts. These cases are synthetic correctness fixtures, not field evidence.

This extension aligns with established multi-skilled resource-constrained project scheduling (MS-RCPSP), a mature field with published instance classes and empirical datasets. Snauwaert and Vanhoucke classify MS-RCPSP variants and compare generated instances with published benchmarks and seven empirical software/railway cases ([EJOR 2023](https://doi.org/10.1016/j.ejor.2022.05.049)). The iMOPSE multi-objective benchmark reports true Pareto fronts for six small instances and approximate fronts for 36 larger instances ([Applied Soft Computing 2022](https://doi.org/10.1016/j.asoc.2022.109253)). A review of multi-skilling scheduling covers 160 studies and identifies skills, objective choice and resource flexibility as major model dimensions ([Computers & Industrial Engineering 2021](https://doi.org/10.1016/j.cie.2020.107004)). Those are candidate external validation sources; the current module does not yet import or implement their full data formats.

### Multi-crew acceptance targets

- **Exact reference:** 100% agreement with MC-01–MC-03, including skills, precedence, non-overlap, weighted objective and deterministic tie-breaking.
- **Instance size:** stay within the explicitly bounded 6-job / 3-crew oracle; do not raise bounds without measuring the combinatorial growth and retaining an execution budget.
- **Policy gaps:** report exact optimum, heuristic loss, absolute gap and relative gap for each policy; a negative gap beyond tolerance is a correctness failure.
- **External reproducibility:** port a small subset of a published MS-RCPSP or iMOPSE benchmark with clear mapping assumptions, preserve the original source identifiers, and compare optimal values/bounds only where the objective and constraints match exactly.
- **Next expansion:** add finite spare stocks or repair-material constraints as a separate oracle dimension with a fresh negative-control corpus. Do not silently apply the present exact results to constraints this module excludes.


## 12. Independent repair-schedule certificate verification

**Paradigm shift: separate the schedule producer from the schedule checker.** The new Rust module `repair_schedule_certificate.rs` accepts a proposed job/crew timeline and a claimed weighted-completion loss. It independently verifies job coverage and uniqueness, identifier validity, skill eligibility, duration consistency, crew non-overlap, precedence, the currently supported model assumptions, and the objective arithmetic. It intentionally does not call the exhaustive search or heuristic scheduler.

The checker is deliberately narrower than a full operational validator: zero outage start, crews available at step zero, positive integer durations, one required skill per repair, and no spare-parts requirements. Unsupported cases fail closed. A successful check validates feasibility and the arithmetic of the reported objective; **it does not prove that the schedule is globally optimal**. Optimality still depends on the exhaustive oracle's search argument or a separate proof/lower bound.

### Certificate-verifier acceptance targets

- **Negative controls:** 100% rejection of the explicitly enumerated invalid certificate fixtures, including missing/duplicate jobs, unknown identifiers, unqualified crew assignment, overlapping jobs, precedence violation, incorrect durations, unsupported constraints, invalid numbers, and false objective claims.
- **Mutation campaign:** at least 10,000 deterministic schedule mutations across valid tiny fixtures; zero invalid mutated schedule may be accepted. Record seed, mutation operators, corpus version, verifier revision, rejected/accepted counts, and minimized counterexamples.
- **Differential check:** for every generated tiny oracle result, run the independent verifier on the returned schedule and compare its recomputed objective to the oracle report. Any disagreement blocks the result.
- **Determinism:** same inputs and certificate produce the same accept/reject outcome and computed objective; reorder input jobs and crews and retain the same semantic result.
- **Scope guard:** any expansion to parts, travel, shifts, release times, or uncertainty must add explicit certificate fields/checks and negative controls before it is used by the planner.

This verifier is a small, independently implemented trust boundary for later solvers, not an operational safety certification. Its source includes a deterministic 10,000-case mutation campaign covering ten invalidation operators; the campaign is authored but has not been executed in the current environment.

## External benchmark route: multi-skill scheduling

Do not claim the three synthetic MC fixtures establish external validity. Snauwaert and Vanhoucke publish a classification and multiple datasets for multi-skilled resource-constrained project scheduling, including comparison with seven empirical software/railway cases ([EJOR 2023](https://doi.org/10.1016/j.ejor.2022.05.049)). The iMOPSE work publishes a 36-instance benchmark and true Pareto fronts for six smaller cases across five objectives ([Applied Soft Computing 2022](https://doi.org/10.1016/j.asoc.2022.109253)); its objectives and model are not automatically equivalent to this repair-loss objective.

The next reproducible benchmark step is a versioned adapter with preserved upstream IDs, source release, file hashes, and documented mapping/exclusion rules. The official [MSLIB repository](https://github.com/MarioVanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library) catalogs large multi-skilled scheduling sets (including MSLIB1–4 and an empirical MSLIB5 set) and best-known solutions; its [release history](https://github.com/MarioVanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library/releases) currently lists v2.2. The [iMOPSE source repository](https://github.com/imopse/imopse) supplies an open-source optimizer and MS-RCPSP instances, while the associated study reports true Pareto fronts for six small cases and a 36-instance large benchmark ([Applied Soft Computing 2022](https://doi.org/10.1016/j.asoc.2022.109253)).

Do not compare numbers with incompatible objective functions. The current exact oracle minimizes weighted completion loss, while mainstream project-scheduling benchmarks often score project makespan, cost, or multiple resource-use objectives. The adapter must first map only instances whose precedence, skill and resource assumptions are supported, and then report benchmark-native makespan/objectives separately from the repair-loss objective. Preserve source IDs and checksums; disclose excluded constraints; separate tuning from held-out instances. Report feasibility rate, best-known gap only for matching objectives, runtime distributions, candidate counts, skill-flexibility sensitivity, schedule stability under perturbations, and Pareto quality for service loss versus labor/cost or skill overuse. The current PR does not yet import or run either external dataset.
