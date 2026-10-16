# Living City Policy Robustness Benchmark v0.2

**Status:** Research benchmark contract. Proposed target for the active `research/urban-systems-lab-v0-1` branch and draft PR #7538. This file specifies experiments; it is not a validated urban simulator, engineering standard, planning approval, or evidence that a real city meets any threshold.

**Scope:** Compare intervention *policies* across matched future scenarios, not merely score a single proposed city at one point estimate. This is deliberately complementary to the interval/cohort work in issue #7607: #7607 defines what counts as robust feasibility; this contract defines how to test decision policies against counterfactuals, distribution shift, delayed feedback and credible baselines.

## 1. Paradigm shift

Move from:

```text
one forecast -> one design -> one feasibility score
```

to:

```versioned evidence + locally ratified floors
  -> explicit candidate policies
  -> paired scenario families with partial observability and action delay
  -> domain solvers / validated reduced-order models
  -> hard-floor and cohort gate
  -> Pareto comparison, regret, service-loss and recovery profiles
  -> retained evidence + post-deployment forecast-vs-observation updates
```

The object being qualified is a policy under specified information and authority constraints—not a model's ability to narrate a plausible plan. A policy cannot inspect the hidden future, bypass the observation interface, rewrite hard floors, or use an oracle state that operators would not have at decision time.

## 2. Evaluation contract

Every experiment must pin:

- study-area and jurisdiction identifiers; spatial/temporal boundary; scenario-family and profile versions;
- initial conditions and their provenance, model versions, parameter units, calibration/validation scope and source digests;
- policy identity/version, permitted observations, observation noise/missingness, action set, implementation delay, cost/resource limits and rollback behavior;
- applicable resident/cohort roster version and locally authorized hard floors;
- random seed or deterministic replay key, train/validation/held-out split, aggregation and metric definitions;
- exact output, verifier identity/version, and the reason a case is pass, fail, blocked or not run.

A scenario probability may be used only when a defensible, evidence-backed probability model exists. Otherwise report a named stress envelope and worst-case/quantile-within-that-set results; do not turn equally weighted test cases into fabricated real-world probabilities.

## 3. Required policy baselines

Run every candidate against the *same* exogenous worlds, starting states, observation streams and resource budgets. Use paired seeds where stochastic models are involved.

1. **No-intervention / do-minimum:** dated status quo or business-as-usual, with its assumptions visible. Never treat a zero-effort policy as an automatically safe default.
2. **Reactive greedy:** the best locally available immediate gain per declared resource unit. This exposes short-term optimization that neglects dependencies and recovery.
3. **Cost-first:** minimize declared lifecycle cost subject to the same non-negotiable admissibility and safety gates.
4. **Floor-first rule baseline:** transparent deterministic policy designed to protect the most urgent currently-observed floor shortfall first, with fixed tie-breaking.
5. **Candidate Symthaea policy:** the proposed planner/controller with exact observation limits, allowed actions and intervention delays.
6. **Hindsight oracle (upper bound only):** an explicitly non-deployable policy with future information. Report its gap as a bound on avoidable regret; never compare it as if it were a valid operational competitor.
7. **Random feasible policy:** optional diagnostic control, constrained to the same admissible action set and budget. It is not a serious deployment baseline.

Report the Pareto frontier and per-case results. Do not reduce safety, access, displacement, cost and ecology to one weighted scalar. Do not allow gains in one group or metric to compensate for a mandatory floor violation.

## 4. Scenario matrix

Keep the current LCB-0.1 seven-case registry as the seed suite. Extend it incrementally, preserving existing cases and adding versioned families for:

| Family | Required perturbation | What it reveals |
|---|---|---|
| Baseline and single hazards | Existing baseline, heat, drought, 72-hour grid outage | Basic feasibility and single-domain response |
| Compound hazards | Existing heat+outage, drought+energy shortfall, flood+communications failure | Shared dependencies and correlated failures |
| Severity envelope | At least three pre-registered severities per supported hazard; label the units and local basis | Cliff edges, graceful degradation and where the model is outside its validity range |
| Dependency/common-cause | Simultaneous loss of shared power, communications, access, supplier or control path | Hidden single points of failure and correlated service loss |
| Observability degradation | Stale telemetry, sensor bias, missing sensors, delayed reports and contradictory measurements | Whether a policy mistakes blindness for safety |
| Implementation friction | Action latency, partial completion, unavailable crews/materials, resource caps and a failed rollback | Whether a paper plan survives delivery constraints |
| Distribution shift | Held-out event combinations, altered dependency topology, shifted demand and parameter ranges beyond calibration | Generalization and appropriate abstention |
| Equity/cohort stress | Same city aggregate, but burden/access shifted to one affected neighbourhood or population cohort | Aggregate metrics concealing concentrated harm |
| Recovery and rebound | Recovery resource bottlenecks, repeated events before full recovery, deferred maintenance | Whether apparent resilience merely postpones failure |
| Adversarial/evaluator controls | Hidden future unavailable to policy; tampered metric, stale evidence, omitted cohort and invalid unit controls | Leakage, specification gaming and verifier integrity |

Not every family applies to every physical model. Applicability must be declared before a run; an unsupported domain is **blocked/out of scope**, not a pass. A case must not be removed after observing that a candidate fails it.

## 5. Primary measurements

Report values per case, per cohort where authorized, and in a clearly defined aggregate. Preserve units and denominators.

- **Hard-floor violations:** count and duration of violations for every mandatory city/cohort floor. A promoted candidate must have zero violations across its required qualification envelope under the pre-registered conservative-bound semantics; otherwise it fails or remains blocked.
- **Essential-service loss:** unmet person-hours (or another explicitly justified denominator) for water, cooling/heat refuge, emergency access, critical energy and other locally designated essential services. Avoid double-counting people affected by multiple services unless the composite estimand explicitly requires it.
- **Worst continuous outage and recovery time:** time from disruption to service restoration and time to stable recovery. Define the service threshold and hysteresis before tests.
- **Tail and worst-case loss:** report maximum over the declared stress envelope. Use CVaR, percentiles or expected loss only when the probability/sample interpretation is justified; label finite-sample estimates as such.
- **Regret:** objective-vector distance or decision loss versus the best feasible policy on the same realized world, plus a separately labelled gap to the hindsight oracle. Define units/normalization in advance and keep hard-floor failures outside compensatory regret.
- **Shift degradation:** paired change from in-distribution to held-out scenarios for service loss, recovery, cost and violations. Report the full distribution and worst case, not just a mean.
- **Resource efficiency:** lifecycle cost, energy/material/crew use, action count and computation/latency under the same resource envelope.
- **Evidence and governance:** missing/stale source rate, unaccounted transformations, unbound metric/interval/cohort records, unauthorized observation/action attempts and review/appeal status.
- **Calibration, only where applicable:** coverage and sharpness for statistical intervals against held-out observations; otherwise report scenario-envelope containment separately. A stress envelope is not a confidence interval.

Metric choices, cohort definitions, severity ranges, tolerance, and primary-vs-secondary endpoints must be registered *before* results are inspected. Publish all mandatory cases including failures and blocked cases.

## 6. Initial qualification targets

These are proposed engineering targets, **not external standards or demonstrated performance**. A site-specific pilot must ratify applicable values with engineers, affected communities, operators and responsible authorities.

### Gate A — integrity and reproducibility

- 100% of required cases have a record: pass, fail, blocked or not run; missing records block suite eligibility.
- 100% of tampered digests, invalid units, missing mandatory cohorts, unknown policy actions and evidence/path mismatches in the frozen negative-control set are rejected.
- Same pinned code, inputs and deterministic seed reproduce canonical decision and evidence bytes exactly, or the report explicitly declares nondeterministic fields and excludes them from the canonical digest.
- The policy sees only observations available at that simulated time; a hidden-future leakage test must fail closed.

### Gate B — hard constraints

- Zero mandatory hard-floor violations for a policy described as *qualified within the declared envelope*. One failure cannot be offset by another case's performance.
- Unknown, unbounded or unsupported uncertainty on a mandatory floor blocks promotion; do not use a point estimate as a substitute.
- A city-wide average that passes while one applicable cohort fails must fail the aggregate qualification gate.
- No model result grants land-use, spending, eligibility, data-collection or infrastructure-control authority.

### Gate C — decision quality (research target)

- On preregistered synthetic stress cases, the candidate must show a measurable improvement over the strongest admissible transparent baseline on at least one primary service-loss/recovery objective, without worsening a mandatory floor or hiding a trade-off.
- Track a provisional target of **at least 20% lower worst-case essential-service person-hours than the floor-first baseline** on a preregistered held-out synthetic suite. Treat this as an ambitious experiment target; if the result is worse, publish it and diagnose the failure rather than alter the suite or threshold.
- The candidate should reduce policy regret relative to the strongest deployable baseline on held-out cases; separately report its gap to the non-deployable hindsight oracle.
- Report performance under equal resources and equal information. Higher compute, extra sensors or privileged knowledge must be separately accounted for.
- Do not claim a universal percentage improvement from synthetic tests; the target only justifies moving to the next controlled experiment.

### Gate D — external validity

- Differential checks against an accepted reference solver or published benchmark where a suitable model exists; fixed tolerances derived from the decision consequence and agreed before comparison.
- Calibration and external/field observations for the actual model's intended domain before using it for consequential engineering claims.
- Prospective forecast-vs-observation records after implementation, retaining model versions and failed predictions. A simulation pass does not imply legal compliance, professional sign-off or commissioning acceptance.

## 7. Experimental design and anti-gaming rules

- Freeze scenario generator version, candidate policies, primary endpoints, evaluation budget, seed set, train/validation/held-out split and stopping rule before comparison.
- Use matched exogenous conditions for every baseline; disclose policy-specific observation or compute differences.
- Hold out *families*, not just random rows derived from the same template. Include at least one leave-one-compound-family-out evaluation when data permit.
- Include negative controls where the intuitive answer is deliberately wrong: e.g. high mean service availability but a long catastrophic outage; city-wide pass with one cohort failing; stale sensors that report normal conditions after a fault; low-cost intervention that shifts risk to a common dependency.
- Compare against ablations: without dependency graph; without uncertainty; without cohort constraints; without action delay; without evidence-quality gates; with shuffled or privileged observations. Removing safety gates is allowed only in isolated offline controls, never on a real infrastructure action path.
- Preserve all attempted runs and failures. Never weaken floors, drop difficult cases, tune on the hidden holdout or redefine the outcome after seeing results.
- For any claimed improvement, publish effect size, uncertainty/replicates where meaningful, worst-case regressions and the exact benchmark version. Small synthetic test counts support engineering debugging, not population-level conclusions.

## 8. Minimal receipt schema

A future implementation should record at least:

```text
PolicyRunV1 {
  suite_version, case_id, case_family, held_out_status,
  scenario_profile_digest, model_and_solver_digests,
  policy_digest, policy_observation_contract_digest,
  seed_or_replay_key, initial_state_digest,
  action_trace_digest, observation_trace_digest,
  metrics_digest, cohort_roster_digest, uncertainty_method_digest,
  hard_floor_result, per_case_result, verifier_id, verifier_version,
  runtime_and_resource_usage, blockers, limitations
}
```

The concrete Rust schema should use strict versioned types and canonical serialization consistent with the existing Living City evidence package, not generic untyped metadata. It should compose with the run receipt and offline verifier rather than create a parallel integrity stack.

## 9. Domain-model separation and validation ladder

The city should be a **federation of independently testable domain models**, coordinated by a typed Rust experiment/evidence layer—not a single home-grown simulator that approximates every physical discipline. Symthaea can orchestrate candidate policies, observations, actions and receipts; each solver remains responsible for its own state equations, units, discretization, numerical error and applicability limits.

| Domain slice | Independent reference ecosystem | First qualification targets | Important boundary |
|---|---|---|---|
| Electrical distribution | EPRI OpenDSS example/test cases and IEEE PES distribution feeders | Node-phase voltages, branch flows/losses, convergence and event response for the exact supported feeder model | The current four analytical regressions only cover balanced radial linearized DistFlow; they are not an OpenDSS differential test |
| Buildings and local energy control | DOE/LBNL BOPTEST building-control benchmark; CityLearn community-energy coordination environments and fixed scenario datasets | Comfort/service, energy, cost, peak demand, controller actions, delay and fault response under identical boundary conditions | Reproduce each selected environment/version and its scorecard; do not equate a reinforcement-learning reward with occupant wellbeing or city-wide feasibility |
| Drinking-water distribution | US EPA EPANET | Pressure/head, flow, tank level, water-quality transport and extended-period service on supported networks | Water-quality, contaminant and hydraulic results need their own units, boundary conditions and validation; graph reachability alone is not hydraulics |
| Stormwater/drainage | US EPA SWMM | Runoff hydrographs, conduit/node state, surcharge/flooding and retention/infiltration behavior for published or locally qualified cases | A hazard label or dependency edge does not predict flood depth, runoff volume or inundation |
| Mobility and emergency access | SUMO, using a named scenario with calibration evidence | Travel-time and queue distributions, throughput, disruptions, accessible-route reachability and emergency response under the declared demand | A generic traffic network is not validated local mobility; calibration data and held-out conditions must be explicit |

Official entry points: [OpenDSS test cases](https://opendss.epri.com/ExamplesDocsandTestCases.html), [BOPTEST](https://www.energy.gov/cmei/buildings/boptest-building-operations-testing-framework), [CityLearn v3](https://www.citylearn.net/index.html), [EPANET](https://www.epa.gov/water-research/epanet), [EPA SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm), and [SUMO documentation](https://sumo.dlr.de/docs/). These are benchmark/reference candidates, not claims that this branch already integrates them.

### Adapter contract

Every domain adapter must declare solver/version and licence/basis; exact model/fixture identity and digest; spatial boundary and coordinate/units contract; initial state, boundary conditions and timestep; valid parameter range; observation/action surface; error/convergence state; calibration data and validation cases; output quantities and accepted tolerance; and whether the result is measured, simulated, derived or missing. The adapter emits a domain receipt bound to exact input/output bytes and all conversions.

Keep production orchestration and evidence handling Rust-first. A compatible solver may be run as a separately versioned executable/container or consume/produce frozen interchange files; a Python-based reference client must not silently become a production dependency just because the upstream benchmark uses Python. The interface boundary itself must be tested, versioned and licence-reviewed.

### Coupling ladder

1. **Contract parity:** prove units, identifiers, time semantics, missing-value behavior and boundary-condition conversions on hand-calculated fixtures.
2. **Domain reproduction:** reproduce a published case or compare against an independent reference solver with tolerance registered before seeing results.
3. **Domain falsification:** test conservation/balance where physically applicable, sensitivity, edge conditions, solver failure and out-of-range inputs.
4. **Event-coupling fixtures:** only after each linked domain passes its own gate, test interactions such as grid outage -> pump/cooling/traffic-signal loss, or flood -> blocked routes plus degraded communications. The adapter declares event ordering, latency, stale-data policy and time resolution.
5. **Held-out coupled scenarios:** evaluate common-cause failures and severity/profile shifts absent from tuning cases; preserve policy-blind hidden cases.
6. **Prospective shadow validation:** bind predictions to subsequent observations before claiming local predictive validity. Shadow mode must not actuate critical infrastructure.

No coupled city result is eligible for a stronger claim than its least-validated critical component and interface. If a required model/solver is unsupported or fails convergence, mark the affected decision blocked; do not substitute a graph heuristic or the last successful value and call it physical simulation.

## 10. Implementation sequence

### Executable synthetic prototype added

The research branch now contains `crates/domains/symthaea-scenarios/src/living_city_policy_benchmark.rs`, exported from the scenario crate. It currently provides:

- A deterministic, one-abstract-service discrete-time harness with separate hidden actual supply/demand and a policy-facing observation record.
- Four transparent baselines: no-intervention, reactive-greedy, cost-first and floor-first.
- A fixed four-case synthetic smoke suite—healthy baseline, forecasted shortfall, stale-telemetry outage and recurrent outage—producing 16 policy/case runs against all four baseline policies.
- Stale/missing sensor handling, a delayed and duration-limited backup intervention, an intervention budget, per-step deployment traces, service-floor breach accounting, unmet-service/cost totals, and delay to the first restored-service step. That restoration metric explicitly does not imply sustained recovery; later floor breaches remain in the trace.
- Paired policy comparison rejects mismatched scenario IDs, step counts, full configs, or scenario/config digests and leaves percentage reduction undefined when baseline service loss is zero.
- A fail-closed policy-suite assessor returns `qualified`, `failed` or `blocked`: missing/duplicate/extra cases, wrong policy identity, mismatched input digests and replay errors block; a valid hard-floor violation fails; only complete replay-verified cases with zero floor breaches qualify for this declared synthetic suite.
- A candidate-versus-baseline robustness assessment records per-case outcomes, worst-case loss change, equal-weight suite-total reduction (explicitly not an expected value), and exact candidate/baseline run SHA-256 digests; its canonical report has its own reproducible digest.
- Canonical JSON/SHA-256 digests for the exact scenario, configuration, run and replay receipt; the replay verifier recomputes a run to reject changed inputs, action traces or metrics. These digests are integrity/replay links, not signatures or independent physical validation.
- A 27-test authored synthetic suite covering deterministic smoke-suite pairing, forecast-vs-reactive behavior on a frozen fixture, receipt/replay verification, rejection of altered metrics and changed scenario/config inputs, digest changes, suite completeness/integrity, duplicate and extra cases, wrong-policy identities, a valid floor breach, mismatched paired hidden truth, zero-baseline-loss handling, immediate deficits, stale/missing observation controls, delayed actions, hidden-future truth isolation, paired comparison preconditions, boundary arithmetic, duplicate policies, invalid forecast ratios and repeatability.

**Verification ceiling:** this is an executable prototype in source, but its tests have not been compiled or run in the current authoring environment because `cargo`, `rustc` and `rustfmt` are unavailable. A queued workflow is not a pass. The harness binds canonical JSON/SHA-256 digests for the scenario, configuration, run and replay receipt, and its verifier reruns the deterministic case to catch changed inputs, action traces or metrics. These are integrity/replay links, not a cryptographic signature or independent physical validation. The harness still contains no calibrated physical model, no stochastic hazard probabilities, no cohort roster, no persisted package tied into the existing offline verifier, and no integrated Symthaea candidate policy. It cannot support a real-city performance or safety claim.

### Remaining sequence



1. **Freeze this protocol** and link its version into the Living City research brief; mark every numeric target as provisional until locally authorized.
2. Persist the new suite-assessment report and per-case run digests in the existing offline evidence bundle; add explicit stable-recovery and regret metrics without treating first restoration as sustained recovery.
3. Expand service-loss, stable recovery and resource accounting with hand-calculated fixtures and negative controls; add budget exhaustion, repeated intervention and unsupported-observation cases. Keep physical dynamics behind domain-model interfaces.
4. Add held-out severity/family tests and observation/actuation latency; confirm there is no future-state leakage.
5. Join the #7607 conservative-bound and cohort gate. An integrity-valid package with infeasible outcomes remains infeasible.
6. Validate each physical domain against a suitable independent solver / published case suite before making domain-specific claims; the current grid-reference tests are linearized analytical regression only.
7. Run controlled comparisons; publish results even when the candidate loses. Only then consider a shadow-mode deployment with no autonomous authority.

## 11. Research anchors

- Azadi et al., *What Have Urban Digital Twins Contributed to Urban Planning and Decision Making? From a Systematic Literature Review Toward a Socio-Technical Research and Development Agenda*, Smart Cities 8(1), 32 (2025). The review examines 88 peer-reviewed articles and highlights the persistent gap between technical ambitions and socio-technical implementation: https://doi.org/10.3390/smartcities8010032
- IEEE 1729-2025, *IEEE Recommended Practice for Electric Power Distribution System Analysis*. Its stated scope includes steady-state, event-based, probabilistic, stochastic and dynamic analysis; it is a useful power-domain benchmark anchor, not a universal urban benchmark: https://standards.ieee.org/ieee/1729/11171/
- NIST AMS 100-85, *On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices*: https://doi.org/10.6028/NIST.AMS.100-85
- EPRI OpenDSS examples and test cases: https://opendss.epri.com/ExamplesDocsandTestCases.html
- DOE/LBNL BOPTEST building-control benchmark and standardized KPIs: https://www.energy.gov/cmei/buildings/boptest-building-operations-testing-framework
- CityLearn v3 shared scenario interfaces and community-energy KPIs: https://www.citylearn.net/index.html
- US EPA EPANET water distribution and extended-period simulation: https://www.epa.gov/water-research/epanet
- US EPA SWMM urban stormwater/drainage modeling: https://www.epa.gov/water-research/storm-water-management-model-swmm
- SUMO microscopic/intermodal mobility simulation and named scenarios: https://sumo.dlr.de/docs/

## 12. Evidence ceiling / non-claims

This contract defines what to test next; it does not implement or execute the benchmark. The existing Living City vertical slice presently has synthetic fixtures, a constrained point-estimate evaluator, an evidence-bound package, a compound-hazard contract, and four linearized analytical grid regressions. It does **not** yet have a calibrated coupled city model, stochastic policy evaluation harness, validated cohort roster, calibrated interval method, validated hydraulic/thermal/mobility models, or real-world evidence that a policy improves outcomes.

No external result in this research note is a validation result for Symthaea. No generated plan is a substitute for licensed engineering analysis, statutory approval, resident governance or commissioning.
