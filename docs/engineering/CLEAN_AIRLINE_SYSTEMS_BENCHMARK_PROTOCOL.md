# Clean Airline Systems Benchmark Protocol

**Status:** Research protocol / synthetic-first; not a validated airline model.  
**Owner:** CLEAN-AVIATION-001 (#8930).  
**Scope:** Integrate aircraft mission feasibility, airline recovery, airport energy/fuel supply, lifecycle accounting and resilience without duplicating domain solvers.

## 1. Research question

Can an airline and its airport-energy system jointly select feasible aircraft/route assignments, charging and fueling schedules, electricity/fuel procurement, storage dispatch and disruption-recovery actions to reduce lifecycle climate impact and total system cost **without reducing the declared transport service or relaxing any hard safety/operational constraint**?

The system boundary is the delivered transport service, not an aircraft or fuel in isolation:

`energy source → conversion / production → airport infrastructure → aircraft and fuel assignment → passenger service → recovery and resilience → lifecycle accounting`

A result is only comparable when candidates use the same route set, payload/service demand, reserve policy, safety/operational limits, weather and disruption scenarios, accounting boundary, input evidence rules and compute budget.

## 2. Scope and ownership boundaries

This is a cross-layer integration benchmark, not a new aircraft solver or dispatch authority.

- **Aircraft and energy feasibility:** compose AIRCRAFT-ENERGY-001 (#7646), fixed-wing engineering (#6249) and mission-matched propulsor co-design (#7809). Those owners retain propulsion, installed mass, thermal, power and mission-model semantics.
- **Flight retiming / recovery:** compose AVIATION-ENTERPRISE-016 (#8902), draft PR #8904, and rolling-horizon recourse/tail risk #8603. Preserve the exact target SHA and each solver's claim ceiling.
- **Forecast inputs:** use #8862 for public-data forecast-to-decision evaluation. Keep forecast skill, decision value and causal impact separate.
- **Energy-hub model:** own only cross-layer flow, capacity, timing, provenance and accounting. It must not pretend to be a qualified grid, battery, hydrogen process, reactor, fusion, fuel-production, aerodynamics or thermal solver.
- **Digital twin and evidence:** Symtropy can host the synthetic/physical system model, Sol-Atlas can represent resource and infrastructure scenarios, and Mycelix can preserve provenance/currentness/approval references. Provenance records do not establish that their claims are true, physically valid or airworthy.

Do not add subsystem logic merely because a field is available in a central schema. Refer to the source contract and surface missing or unsupported states.

## 3. Candidate architecture families

Each family is a *scenario candidate*, not an assumption of certification or commercial availability.

1. **Incumbent reference:** existing aircraft and fossil Jet-A supply, using a frozen comparable schedule.
2. **Existing aircraft plus approved lower-carbon fuel:** declare the exact fuel pathway, blend/compatibility basis, lifecycle boundary, delivered volume and price. “SAF” is not a single lifecycle intensity or supply guarantee.
3. **Battery-electric regional:** include only routes that pass candidate-specific payload-range, power, reserve, temperature, end-of-life, turnaround, charger and certification-status gates.
4. **Hydrogen-electric regional:** account for hydrogen production, compression/liquefaction where applicable, storage, boil-off/losses, distribution, conversion, onboard storage mass/volume and route feasibility.
5. **Hybrid/turboelectric:** use mission- and configuration-specific aircraft/powertrain results; distinguish demonstration from commercial service and account for installed mass, conversion losses, heat rejection and fuel consumption.
6. **Synthetic aviation fuel / e-SAF:** account for electricity, hydrogen, carbon source, process yield, compression, transport, storage and actual fuel delivery. Do not credit hypothetical supply.
7. **Offboard nuclear- or fusion-supplied electricity:** treat as a declared source in the ground grid or energy hub. Require net delivered energy, lifecycle intensity, availability, price, grid connection and commissioning/contract evidence appropriate to the scenario.
8. **SMR/fusion-backed hydrogen or e-SAF:** scenario-level industrial production pathway only, with all conversion stages and economics declared.
9. **Onboard fission or fusion:** separate research-only feasibility screens, never operationally feasible by assumption. No aircraft-feasibility claim without credible complete-plant evidence for mass, volume, net electrical output, conversion, cooling/heat rejection, shielding/containment (as applicable), maintenance, safety, and an external approval pathway. If such evidence is absent, report `Unknown/ScenarioOnly`, not zero mass, zero emissions, zero cost or firm power.

Do not compare energy-source fuel energy density alone. Compare **usable energy delivered to the mission or airport service boundary**, with the complete conversion chain and constraints.

## 4. Frozen model contract

Each scenario manifest must pin the following.

### 4.1 Transport service and operations

- route/flight IDs and stable identities;
- time basis, airports, flight windows and declared weather/context;
- passenger demand, required payload and service obligations;
- aircraft family, exact configuration/generation and feasible mission envelope;
- available fleet count and maintenance/crew/cycle constraints in scope;
- departure/arrival windows, turnaround, connections, cancellations and recovery actions;
- reserve/diversion policy and all energy/power/operational hard limits;
- certification/evidence status of each aircraft candidate, as a constraint on scenario use rather than a prose footnote.

### 4.2 Airport energy and fuel

- interval length/time zone and chronology;
- electricity demand and any measured/synthetic load split;
- source-specific supply capacity, price, lifecycle intensity and evidence status;
- renewable profile and curtailment policy, grid limits/import/export;
- stationary storage energy and power limits, state-of-charge bounds, efficiencies, degradation and reserve;
- charger count/power/occupancy and charging efficiency;
- hydrogen/e-SAF/SAF delivery and storage inventory, conversion yield, loss, and replenishment lead time;
- firm-supply contract/availability assumptions, outages and backup resources;
- facility capital/maintenance cost and lifetime if included; construction must not disappear because the optimizer starts at commissioning.

### 4.3 Lifecycle and cost boundary

Keep explicit and non-overlapping:
- well-to-wake GHG emissions by gas and declared CO2e method/time horizon;
- direct aircraft combustion emissions;
- upstream electricity generation, transmission, conversion and fuel-production emissions;
- feedstock/carbon sourcing and transport;
- infrastructure embodied emissions, only if included consistently across candidates;
- non-CO2 aviation climate effects (including contrail-related effects) in a **separate reported axis** unless a declared and justified method supports a combined metric;
- fuel, electricity, water/process resources where material, maintenance, infrastructure capex/opex and replacement reserves;
- subsidies/tax credits as separate accounting views, not double-counted as physical efficiency gains.

A lifecycle factor must carry source URL/identifier, version/date, unit, geography, pathway, validity range, and uncertainty. Missing factors remain missing/unknown; they are never silently imputed as zero.

### 4.4 Evidence and currentness

Every material input binds:
- stable ID, units, source/provenance and content digest where available;
- evidence class: measured, supplier-reported, literature-derived, calibrated model, uncalibrated model or synthetic assumption;
- configuration and data generation;
- applicability bounds, uncertainty and known omissions;
- currentness/expiry conditions and the required independent check.

Changing an aircraft, battery, fuel pathway, electricity source, model, time grid, or lifecycle factor must invalidate precisely the affected dependent conclusions. Do not let a traceability record promote a claim's evidence class.

## 5. Objective: preserve the Pareto frontier

Do not optimize one opaque "green airline score". Report a vector at minimum:

- lifecycle kg CO2e per completed passenger-km;
- total lifecycle CO2e in the planning horizon;
- non-CO2 indicators separately, with their method and uncertainty;
- total system cost and cost per completed passenger-km;
- cost per available-seat-km (CASM) and/or revenue-independent operational cost under a declared formula;
- completed passengers/trips, cancellations, delay distribution and connection failures;
- payload/seat capacity actually delivered;
- energy use by carrier and source, grid import, renewable curtailment, peak power, storage reserve and fuel inventory;
- resilience under declared outages, including unmet critical airport load and time to recover;
- maintenance burden, fleet utilization and candidate evidence maturity;
- solve time, memory, time-to-first-feasible, work counters and known bound/gap.

Hard requirements—safety, aircraft mission feasibility, minimum service, reserves, legal/certification status and declared operational limits—must be feasibility gates, not weighted preferences. For Pareto comparisons, publish the constraint set and stakeholder preference that selects a point from the frontier.

## 6. Benchmark phases

### P0 — independent accounting oracles

Before optimization, create hand-checkable fixtures for:
1. unit conversions and interval alignment;
2. electricity and fuel inventory balances;
3. charge/discharge efficiency and state-of-charge conservation;
4. energy-to-hydrogen/e-SAF production yield and losses;
5. grid import/export and capacity saturation;
6. source-specific lifecycle emissions and non-overlapping upstream/downstream boundaries;
7. cost roll-up and capital annualization;
8. passenger-km, seat-km and completed-service denominators;
9. reserve/end-of-life/temperature derating;
10. time-zone and event-order edge cases;
11. overflow, negative values, duplicate identities, missing factor and out-of-domain input;
12. independent reproduction of every displayed aggregate from the manifest.

Known answers should be computable by a second, simple implementation or explicit arithmetic. The main optimizer must not be its own oracle.

### P1 — tiny synthetic whole-network oracle

Freeze a small problem with a handful of flights, aircraft families, time slots, chargers, energy sources, and disruptions. Exhaustive enumeration should be tractable and become ground truth for the optimizer comparisons.

Required synthetic controls:
- nominal operations with a unique known optimum;
- no-op energy hub where both policies are equivalent;
- charging bottleneck / competing charging windows;
- renewable surplus followed by low renewable availability;
- storage efficiency loss and end-of-life derating;
- grid import cap/outage;
- fuel shortage or delayed delivery;
- aircraft unavailable after a modeled maintenance event;
- simultaneous schedule disruption and energy shortage;
- hard constraint conflict for which the correct result is “no feasible plan”;
- input permutation that should not change the canonical solution;
- malformed/missing lifecycle factor;
- hypothetical SMR/fusion plant without net-power/cost/lifecycle evidence, which must stay unresolved;
- zero-severity/zero-disruption control.

All invented parameters must be marked synthetic. Synthetic scenario frequencies are not real-world risk probabilities.

### P2 — optimization bake-off

Compare on the identical frozen manifest:
- incumbent schedule and current policy;
- deterministic greedy baseline;
- exact exhaustive enumeration on tractable cases;
- an exact MILP/CP-SAT or other independently defined reference where the problem formulation supports it;
- bounded branch-and-bound / heuristic search;
- robust or recourse policy under explicitly declared distributions/ambiguity sets.

Every returned plan must be independently re-evaluated against source manifests and hard constraints. Record objective vectors—not just the solver's reported scalar—and distinguish:
- optimality proven;
- feasible but uncertified;
- no feasible plan found within budget;
- unsupported input / no evaluable result.

Report P50/P95/P99 runtime on repeated matched trials, time-to-first-feasible, node/action/bound counters, memory, timeout/budget state, and optimality gap where a valid bound exists. `queued`, `skipped`, `not run`, or missing artifacts are not PASS.

### P3 — matched historical replay

Use time-ordered historical data and freeze training/calibration/test partitions before scoring. Prevent post-event features, label leakage, and event-cluster leakage. Keep these claims separate:
1. forecast accuracy;
2. improvement in the optimizer's chosen action;
3. replayed operational value;
4. causal savings after real-world confounding and policy effects.

Report coverage, missingness, drift, calibration, prediction intervals, subgroup/airport/season stratification and out-of-support behavior. No retuning on final holdout. An offline replay is not an experiment or a causal estimate.

### P4 — robustness and stress campaign

Cross operational disruptions with energy disruptions:
- aircraft grounding / maintenance spillover;
- severe weather and connection-protection demand;
- late inbound arrival, turnaround and cancellation;
- peak electricity price or demand;
- grid import limit and outage;
- solar/wind underproduction and correlated weather;
- charger outage and queueing;
- battery degradation/temperature derating;
- hydrogen/e-SAF delivery or process interruption;
- simultaneous multi-flight and multi-asset failures.

Use paired random seeds when comparing policies. Report full tails and loss exceedance/shortfall where defined, not only the mean. Robustness should be compared against its cost: conservative plans can waste capacity. Shift scenarios outside the supported set must raise an explicit alarm rather than silently extrapolate.

### P5 — physical, procurement and deployment evidence

Only after earlier gates: reconcile with meter/load data, actual supplier quotes, fuel-pathway certificates, contracts, airport permits, equipment capability, aircraft certification and local regulatory requirements. Establish physical metrology and independent review. A software benchmark never certifies an aircraft, a reactor, a fuel, a charging installation, or airport resilience.

## 7. Target thresholds and falsification rules

These are **proposed research targets**, not industry standards or demonstrated results.

- 100% traceability for material inputs and emitted claims.
- 100% independent arithmetic agreement on frozen P0 known-answer cases.
- Zero hard-constraint violations in any result labelled feasible.
- Zero silent defaults, extrapolation or unit substitutions for missing/unsupported inputs.
- Deterministic byte-stable canonical outputs for pinned deterministic runs, including tie-breaking.
- Exact-oracle agreement on every tractable P1 case.
- Report improvements only against the matched baseline at equivalent delivered transport service and hard-constraint profile.
- Stretch hypothesis: at least 10% lower lifecycle CO2e per completed passenger-km **and** at least 5% lower total operating/system cost on a held-out scenario set. Claim this only if the improvements exceed declared uncertainty, do not reduce completed service or payload, and do not shift emissions outside the chosen boundary.
- For recovery/optimization, publish P50/P95/P99, time-to-first-feasible, feasibility rate, verified objective vector, valid bound/gap, and work counters. No runner status is implied by a test definition.

Falsify or mark unresolved if: the solution relies on unsupported aircraft capability; a lifecycle pathway is missing; physical energy balances do not close; service demand changes between candidates; a hard constraint fails; robust improvement is smaller than uncertainty; results depend on post-event information; or an onboard nuclear/fusion concept lacks complete-plant evidence.

## 8. Industry reality check (dated sources)

1. **SAF scale / availability:** IATA's June 2026 estimate is around 2.4 million tonnes of SAF production in 2026, about 0.8% of jet-fuel use; it reports a gap between installed capacity and actual production/use. The implication is not “SAF cannot scale”; it is that installed nameplate capacity must not be confused with affordable, contractually deliverable volume.  
   Source: https://www.iata.org/en/pressroom/2026-releases/06-06-saf-production-volumes-still-disappointing

2. **Electric aircraft readiness:** the US Government Accountability Office reported on 27 May 2026 that the FAA had not certified a manned electric aircraft for commercial operations as of March 2026 and was evaluating designs case by case. This is US-specific and time-bounded; certification state must be rechecked for a specific aircraft and jurisdiction.  
   Source: https://www.gao.gov/products/gao-26-107816

3. **Hybrid-electric progress:** NASA reported on 20 July 2026 that the NASA/GE megawatt-class hybrid-electric system flew on a modified Saab 340B and above 30,000 ft. This is a flight-test milestone, not a commercially certified aircraft, a full-electric long-haul demonstration, or proof of route economics.  
   Source: https://www.nasa.gov/directorates/rtmd/nasa-ge-hybrid-electric-flight/

4. **Operational recovery optimization:** the 2026 INFORMS case study of Lufthansa Group / SWISS reports EUR 12 million and 14 kilotonnes CO2 saved to date, with EUR 30 million and 50 kilotonnes CO2 annual targets across the group. This is a deployed enterprise optimization reference. It is not a comparable score for our synthetic test and should motivate real operational measures alongside emissions, not be copied as an expected gain.  
   Source: https://pubsonline.informs.org/doi/10.1287/inte.2025.0296

5. **Fusion maturity:** the US DOE's 9 June 2026 roadmap targets commercial fusion power development in the mid-2030s but conditions delivery on unresolved science, engineering, infrastructure and funding progress. Treat it as a contingent future energy scenario—not assured baseload supply for an airline.  
   Source: https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate

Comparisons are contextual, because these references use different scope and evidence. No claim of outperforming industry is justified without a matched, independently reproducible benchmark.

## 9. Recommended engineering sequence

1. This protocol and source register.
2. Frozen JSON scenario manifest and independent P0 accounting/oracle tests.
3. Tiny P1 exact oracle plus greedy baseline.
4. Adapter interfaces to aircraft energy (#7646/#7809) and retiming/recovery (#8603/#8902), preserving exact digests and input claims.
5. Optimizer bake-off with independent verifier and budget/coverage semantics.
6. Held-out historical replay and drift/out-of-support tests.
7. Energy-hub and correlated disruption stress campaign.
8. Physical input reconciliation / procurement and regulatory review, one bounded airport pilot at a time.

Do not make a broad integration refactor before the manifest, quantity semantics, independent oracle and ownership interfaces are pinned.

## 10. Nonclaims

This protocol does not show that a clean airline exists, that electric/hydrogen aircraft are currently deployable for the chosen routes, that SMRs or fusion will be available on a specific schedule, that synthetic SAF is price-competitive, or that any policy beats the industry. A future PASS applies only to the exact frozen software contract and fixtures. Operational use requires independently validated models, qualified aircraft and equipment, legal approvals, human operational authority, and measured outcomes.


## 11. P2 executable slice — time-indexed airport energy hub (2026-10-29)

The separate aviation_energy_hub module is the first time-indexed extension. It is synthetic-first and intentionally bounded to at most eight ordered intervals and 32 total delivered-energy quanta.

### V1 state and decisions

Per interval the model binds:
- fixed renewable generation (all generation cost and lifecycle emissions are counted, even if curtailed);
- charging/service demand and an explicit charger-capacity limit;
- grid import capacity, cost and declared lifecycle CO2e factor;
- renewable evidence reference/class and grid evidence reference/class;
- direct renewable use, grid import, battery discharge, renewable surplus charged, curtailment, and exact state-of-charge before/after.

Storage begins empty in V1. Integer-quantum efficiencies are conservative: charging conversion floors stored energy; discharge conversion rounds up the state of charge required. Simultaneous charge/discharge and grid-to-storage charging are disallowed. The terminal reserve is a hard constraint.

### Independent verification

The plan verifier independently checks:
- canonical SHA-256 scenario identity and plan identity;
- exact interval count/order;
- demand balance, charger/grid/storage power limits and source balance;
- no simultaneous charge and discharge;
- curtailment arithmetic and exact storage state transitions;
- minimum terminal reserve;
- recomputed source ledger, cost and accounted lifecycle CO2e.

Search returns a Pareto frontier minimizing cost, minimizing the supplied lifecycle CO2e metric, and maximizing final stored energy. It returns no frontier if its node budget is exhausted. A deterministic local renewable-first / grid-second / battery-last policy is included only as a deliberately myopic baseline.

### Explicit omissions

V1 is not an AC/DC power-flow solver and does not model voltage/frequency stability, transmission losses upstream of the declared energy bus, reactive power, grid-forming behavior, hydrogen/e-SAF production, grid-to-battery charging, aircraft-to-grid export, degradation over lifetime, uncertainty distributions, flight schedule recourse, physical aircraft feasibility, certification, or non-CO2 aviation effects. Renewable generation is a fixed input, not a forecast or a measured power series. All test factors are synthetic.

### Proposed P2 gates

- Exact agreement with independently hand-enumerated tiny fixtures.
- No exact result when a search budget expires.
- Zero demand, capacity, energy-state or terminal-reserve violations in returned plans.
- Byte-stable manifest digests under unchanged ordered input; digest changes on any material field change.
- Report cost, lifecycle CO2e, grid imports, renewable generated/used/curtailed, storage losses and final reserve separately.
- Demonstrate at least one case where a myopic baseline fails or is dominated by exact time-coupled scheduling.
- Do not claim a measured cost or CO2e improvement until inputs are calibrated, the model is independently tested and a matched held-out evaluation is available.

This software slice does not establish actual airport resilience, renewable availability, a carbon factor, a physical storage rating, or operational authority.


## 12. Literature-to-benchmark crosswalk (reviewed 2026-10-29)

These studies define relevant comparators, but they do not share a common benchmark/data boundary. Their headline gains must not be compared as if they were the same experiment.

### Flight-aware electric-aircraft charging under uncertainty

A 2026 study on airport-specific time-of-use pricing and demand charges jointly models flight timetables, turnaround, energy-demand uncertainty, deterministic scheduling, sample-average approximation, and robust optimization. It uses flight data from six major airports in China's Greater Bay Area and reports up to 5.77% charging-cost savings for its robust model under high-demand uncertainty relative to its stated deterministic baseline. Source: https://www.sciencedirect.com/science/article/pii/S0967070X26002398

**Reproduction target:** first reconstruct its deterministic comparator and its stated uncertainty/robustness experiment if the published details and compatible data permit. Record the exact cost boundary, tariffs, peak charges, allowed aircraft flexibility, charging constraints, uncertainty set and out-of-sample split. A result from our tiny synthetic fixture cannot be used to claim a 5.77% improvement or a better SOTA result.

### Airport microgrid, aircraft-to-grid and fixed storage

A 2026 IEEE paper models a solar-powered airport microgrid with aircraft-side batteries, fixed storage, grid imports/exports, EV parking demand, state-of-energy constraints, and time-varying tariffs. Its abstract reports a cost decrease from 83,508.30 to 74,867.88 in its stated currency units—about 10.35%—when coordinated aircraft-to-grid and fixed storage are added to its baseline. Source: https://ieeexplore.ieee.org/abstract/document/11578850

**Reproduction target:** match its objective, units, tariff horizon, storage constraints, aircraft-to-grid assumptions and baseline before comparison. Our current V1 excludes aircraft-to-grid, grid-to-storage charging, grid-export revenue and AC/DC power-flow constraints, so it is not yet an implementation of that model.

### Airport ground-equipment fleet and charging

A July 2026 Transport Policy paper jointly optimizes electric ground service equipment fleet sizing, charging and scheduling with on-site photovoltaic generation using a two-stage framework, MILP and Benders decomposition. Source: https://www.sciencedirect.com/science/article/abs/pii/S0967070X26001332

**Reproduction target:** treat ground-service-equipment electrification as a sibling subsystem, not automatically as aircraft charging. An eventual integrated energy hub can compose this demand only after its vehicle tasks, chargers, shift constraints and source data have their own verifier and model boundary.

### Airline schedule recovery in production

The 2026 INFORMS Lufthansa Group/SWISS case documents a deployed production system coordinating aircraft, passengers, and crew, and distinguishes fast disruption recovery from later schedule reoptimization. It reports operational savings under the authors' internal measurement procedure. Source: https://pubsonline.informs.org/doi/10.1287/inte.2025.0296

**Integration hypothesis:** use the existing Symthaea recovery/retiming owner (#8603/#8902) plus the energy hub to ask whether energy-aware decisions improve the *same* operational recovery result after adding grid limits, storage, and energy-price/emissions uncertainty. Treat schedule-recovery savings and energy-scheduling savings as potentially overlapping; do not simply add them together.

### Benchmark ladder required before a SOTA claim

1. **Protocol parity:** reproduce each study's stated objective, boundary, assumptions and baseline. If the original dataset or exact formulation is unavailable, label the exercise "related synthetic comparison," not reproduction.
2. **Independent oracle:** validate tiny cases against exhaustive enumeration and an independently implemented feasibility/accounting checker.
3. **Baseline parity:** identical data, chronology, constraints, objective boundary, machine class, runtime budget and stopping rule.
4. **Held-out comparison:** freeze tuning/calibration partitions; evaluate on untouched dates, airports or scenario families; use paired disruptions/seeds where applicable.
5. **Integrated value:** compare the same base optimizer with and without airport-energy coupling, and separately with and without robust recourse. This estimates incremental replay value, not causal deployed savings.
6. **Deployment evidence:** before operational claims, require metered energy data, supplier/fuel evidence, approved aircraft capabilities, lawful controls and a controlled field evaluation.

### Proposed competitive thresholds

- **Gate A — reproducibility:** 100% agreement with all published known-answer examples we can recover and every tractable internal exhaustive oracle case.
- **Gate B — parity:** reproduce each target paper's reported metric within its stated rounding and disclosed uncertainty before tuning for superiority.
- **Gate C — improvement:** seek at least one percentage point of additional relative cost reduction over a matched published baseline on a held-out benchmark, while preserving every hard constraint and reporting lifecycle CO2e and terminal energy reserve separately. This is a stretch hypothesis, not a prediction; a smaller gain may be the honest result.
- **Gate D — robust value:** report mean, P95/P99 cost/regret, infeasibility and unmet-demand rates, worst declared scenario, reserve violations (required zero), runtime and memory. Show the cost of robustness rather than treating worst-case protection as free.
- **Gate E — climate value:** report lifecycle CO2e per completed passenger-km, total CO2e and non-CO2 indicators separately. A lower cost alone is not a clean-airline result.

If exact parity is impossible because the source material omits essential variables, preserve that limitation and move to a clearly labelled synthetic benchmark. Never reverse-engineer undocumented assumptions solely to reproduce a headline gain.


## 13. P3 robustness slice — weighted stress replay versus perfect-information recourse (2026-10-29)

The new aviation-energy-hub-robustness module compares a fixed nominal schedule against a declared, weighted set of time-indexed airport-energy scenarios.

### Contract

- Every scenario must have a valid canonical SHA-256 content digest.
- The scenario set has a separate canonical digest over stable scenario IDs, exact integer weights and member digests. Scenario input ordering does not change this digest.
- Integer scenario weights must be positive and sum to exactly 1,000,000 ppm. They are design weights, not real-world probabilities unless independently calibrated.
- Scenarios must use the same currency, energy quantum, planning period, lifecycle boundary, delivered transport-service denominators, and ordered interval grid. Renewable generation, grid prices/capacities, and energy availability may differ.
- The nominal plan is independently verified before it becomes the replay baseline.
- **Static replay:** apply the same action sequence to each scenario, with only the scenario identity rebound. A plan that violates supply, demand, storage, or reserve constraints is recorded as infeasible for that scenario.
- **Perfect-information recourse reference:** independently solve each scenario and select its least-cost feasible Pareto point, subject to the exact solver budget. Because it knows the entire realized scenario, this is an optimistic reference—not a causal, deployable policy.
- Failed exact searches due to node budgets are recorded as **unresolved**, not as proof that no feasible plan exists. Actual infeasibility, unresolved weight, and feasible weight must sum back to the original 1,000,000 ppm.

### Required reporting

- static replay feasible / infeasible weight;
- perfect-information recourse feasible / definitely infeasible / unresolved weight;
- per-scenario failure reason and independently verified objective vector;
- conditional cost and CO2e summaries with their feasible-weight denominator;
- worst feasible cost, worst accounted lifecycle CO2e and storage-reserve outcome;
- scenario-set digest, per-scenario digest, solver budget and exact run identity.

Conditional metrics must never conceal an infeasible or unresolved fraction. Do not assign an invented zero cost to a failed schedule or average only successful cases without displaying the missing weight.

### Falsifier

The synthetic stress test freezes a two-interval nominal plan. One stress case reverses the relative grid prices; another reduces first-interval renewable availability. The nominal action sequence is still feasible in the price-reversal case but is invalid under the renewable shortfall. Perfect-information recourse finds a feasible plan in both cases under the test contract. The result is a synthetic demonstration of schedule brittleness and recourse headroom, not a measured real-world savings claim.

### What remains before online robust dispatch

This slice does not implement a nonanticipative policy, rolling-horizon replanning, online observations, probability calibration, confidence bounds, distributionally robust optimization, physical grid constraints, battery degradation or aircraft schedule recovery. Next work must define observation times and information available at each decision, then compare rolling-horizon recourse against this perfect-information reference and a frozen static baseline on identical paired stress sequences.


## 14. Next competitive frontier — flexibility portfolio, policy learning and stakeholder fairness

Additional current research (reviewed 2026-10-29) suggests we should benchmark energy flexibility as a portfolio, not merely a stationary battery.

### Landside vehicle-to-grid (V2G)

A 2026 American Control Conference study jointly schedules electric aircraft apron charging with bidirectional charging of landside EVs. Its model combines a linear program with an aggregate PDE-based EV charging representation and real grid prices, but uses synthetic parking-garage occupancy. It reports up to 32% aircraft-charging cost savings versus its no-V2G baseline, conditional on schedule and EV participation. Source: https://ieeexplore.ieee.org/document/11616483/

**Research implication:** a later energy hub should treat landside EVs, fixed storage and parked aircraft as distinct flexibility resources with individual availability windows, user departure requirements, energy reserves, battery wear costs and export constraints. Do not treat all parked batteries as freely dispatchable. The current V1 explicitly does not implement V2G or grid export.

### Physics-constrained reinforcement learning

A September 2026 IEEE Transactions on Smart Grid paper evaluates physics-based reinforcement learning for an airport thermo-electric microgrid, including e-plane charging, against a classical optimization method on an actual Canadian airport microgrid. The abstract reports near-optimal simulation results and improved convergence stability versus conventional reward-based approaches, but does not provide enough detail in its abstract alone for a numerical apples-to-apples claim. Source: https://ieeexplore.ieee.org/abstract/document/11701290/

**Research implication:** Symthaea can test learned forecasts, warm starts or policies, but the exact/independent optimizer and feasibility verifier must remain the ground-truth gate on tractable cases. Compare convergence/sample efficiency, worst-case feasibility, policy drift, tail regret, runtime and transfer across airports. Do not let an RL reward function override hard operational or safety constraints.

### Multi-stakeholder disruption recovery and fairness

A September 2026 study proposes airline/network-manager collaborative irregular-flight recovery with a fairness-protection mechanism and reports experiments using real operational data. Source: https://www.sciencedirect.com/science/article/abs/pii/S1366554526003169

**Research implication:** whole-airline optimization needs separate fairness and burden-distribution axes, such as delay/cancellation burden across passenger cohorts, airports and collaborating operators. A single airline cost score could merely transfer disruptions to another carrier or network manager. Define the fairness measure and protected constraints before tuning; do not conceal them inside a weighted “social benefit” score.

### Additions to the benchmark vector

Alongside cost, CO2e, service completion, grid peak, curtailment and terminal reserve, later phases should separately report:
- landed/airside/landside battery flexibility available by time and the portion actually dispatchable under user/mission commitments;
- battery degradation/warranty cost and reserve/SOC violations (the latter must remain zero);
- grid peak/demand-charge impact, exports, local congestion and value of flexibility;
- P50/P95/P99 delay, cancellation and recovery regret;
- distribution of disruption burden across protected cohorts or network stakeholders;
- robust feasible-scenario weight versus unresolved weight and scenario-support coverage;
- learned-policy training samples, wall time, seeds, memory, constraint violations and out-of-airport transfer performance.

These are research dimensions to implement, not metrics the current synthetic slice already measures.


## 15. P3 information-time gate — nonanticipativity for scenario policies (2026-10-29)

The next verifier checks that a scenario-indexed policy does not use future information.

### Policy contract

Each scenario policy plan carries:
- the exact scenario-bound plan and verified action sequence;
- one information-set ID per decision stage;
- a reference to the declared information/observation contract.

The verifier independently validates each plan, then requires equal actions at stage \(t\) for every scenario pair assigned to the same information set at stage \(t\). If two scenarios share an information set at stage \(t\), their declared information history must also agree at every previous stage. This prevents malformed scenario trees that “merge” histories that were previously distinguishable.

A canonical policy digest binds each scenario ID, scenario digest carried in the plan, all action fields, information-set IDs and information-contract reference. It is content identity, not a proof that caller-supplied information labels correspond to reality.

### Synthetic falsifier

The nominal energy schedule and a full-horizon optimizer for a future price-reversal scenario can choose different first-interval actions, even though the future price reversal is not yet observed at that decision. When both scenarios share the same first-stage information set, the verifier reports a nonanticipativity violation. The frozen nominal plan, rebound only to the alternate scenario identity and independently rechecked, is nonanticipative because it commits the same action before the future prices diverge.

This tests a critical distinction:
- **perfect-information recourse** is an optimistic reference that knows the whole future;
- **nonanticipative policy** can act only on information available at each decision;
- **rolling-horizon policy** must repeatedly replan after new observations while carrying the physical state forward.

### Claim ceiling

This verifier validates consistency with the supplied information partition. It does not validate the observation contract's realism or itself optimize a nonanticipative policy. A next implementation must combine state carryover, observable forecasts, actual realization, rolling replanning and paired held-out evaluation. Report first-action violations as hard failures; do not claim policy robustness from perfect-information results alone.


## 16. P3 executable slice — nonanticipative rolling-horizon dispatch (2026-10-29)

The rolling module advances beyond checking scenario labels: it runs a policy that observes the actual current interval, plans against forecasts for future intervals only, commits the first action, verifies that action against current observed conditions, advances state of charge, and replans.

### Information-time contract

For each stage, the input binds:
- a monotonically increasing decision timestamp;
- the timestamp when the current observation became available;
- the future-forecast issue and availability timestamps;
- a forecast vector for future interval IDs only; the vector cannot overwrite the current observed interval;
- a digest over forecast issue/availability times and material future interval fields;
- a planning-scenario digest over current observations, forecast suffix, and current SOC.

Observed data and forecast data remain separate. A future forecast is usable only if it was issued and available no later than the decision time. The runner uses the realized scenario to retrieve current-stage observations and stable interval IDs, but it does not use later realized prices, renewable output or loads in the suffix optimization. The episode digest is retrospective audit identity and includes the pinned realized scenario; it must not be exposed as a policy feature at earlier decision times.

### State and execution contract

- The initial state of charge and a non-empty opening-inventory ledger reference are explicit and bound into the energy-hub scenario digest. Positive opening SOC cannot use the no-opening-inventory sentinel.
- State of charge carried from the previous committed interval becomes the initial state of the next suffix optimization. Its inventory-ledger reference becomes a canonical digest over the verified committed prefix, actions, and realized source-cost/emissions summaries.
- At each stage, the exact solver returns a Pareto frontier and the reference policy selects its first cost-ordered point; this is one deterministic baseline policy, not a proof that cost is the only objective.
- The selected action is the first action from a complete plan independently verified by the exact solver against the current observation and forecast suffix. A separate whole-horizon service ledger is not constructed for the one-step check, because a zero-load interval may validly generate/charge energy without delivering load until later.
- Per-stage reporting uses a distinct interval-local flow ledger (current generation/import cost and CO2e, primary source energy, and load-delivered energy), rather than dividing one interval's values by the entire episode passenger-km denominator.
- Every committed action remains immutable; later observations only affect later actions.
- After all stages, the assembled plan is independently verified against the complete realized scenario.
- Search-budget exhaustion, definite no-plan and other solver failures produce different run statuses and never a completed run with a fabricated zero cost.
- The input opening SOC is an externally reconciled inventory: upstream cost and lifecycle emissions for this opening inventory must have been attributed to the prior charging/source ledger. Battery discharge records load delivery without charging those upstream impacts again.

### Load-delivery and energy-accounting correction

The energy-hub ledger now distinguishes generation from load delivery:
- all declared renewable generation still incurs its declared cost and lifecycle CO2e, even when some is curtailed;
- the renewable source record marks only direct renewable use as load-delivered energy; charging input is not counted as load delivery;
- battery discharge is separately represented as load delivery, with its upstream cost/emissions attributed at the original charging/source event;
- use per-source generation/import/curtailment/storage metrics when assessing primary energy. The aggregate source-energy field includes a secondary storage-output flow and is not a substitute for primary source-energy accounting.

The opening SOC source and its upstream ledger must be reconciled before a whole-lifecycle claim. The module does not invent a zero lifecycle factor for the original source; the zero marginal attribution at discharge prevents re-booking impacts already assigned at charge.

### Synthetic future-leak falsifier

A two-interval fixture starts with one unit of previously accounted stored energy. At the first decision, the observed current grid price is 1; the future grid price forecast is 10, while the realized future price is 0.1 but is not yet available. The rolling policy uses grid at the first interval and preserves SOC based on the issued forecast, then discharges the stored unit after observing the cheap future interval. A clairvoyant full-horizon solver instead discharges immediately and imports cheap grid energy later. The test asserts that the rolling first action follows the forecast, not the hidden future realized price. These values are synthetic.

### Limitations and next gates

This is a bounded deterministic replay/reference policy, not a live controller. It does not yet include probabilistic forecast calibration, forecast ensembles, nonanticipative stochastic programming, CVaR/distributionally robust objectives, time-varying grid demand charges, battery degradation, aircraft missions, physical power-flow, flight schedule recovery, live data connectors or real airport authorization.

Required next gates:
- independent state-transition and energy-accounting oracle parity for tiny cases;
- deliberate future-leak negative control;
- source/observation timestamps and forecast digests retained in every result;
- paired comparison against frozen static plan, greedy baseline and perfect-information reference;
- report forecast error and the value/cost of recourse, not just cost;
- zero hard-constraint and terminal-reserve violations in every completed run;
- separately report infeasible and unresolved weight, runtime, P50/P95/P99, and work counters.

No real airport saving, robust probability claim, or SOTA result follows from the synthetic runner.


## 17. P3 paired policy bake-off — static versus rolling versus clairvoyant (2026-10-29)

The paired benchmark evaluator runs four policies over one weighted, fixed scenario panel:

1. **Static plan:** actions from a separately verified frozen baseline are replayed unchanged against every realized scenario; each replay is independently re-verified.
2. **Greedy baseline:** the existing deterministic renewable-first/grid-second/battery-last policy is evaluated on each realized scenario.
3. **Rolling horizon:** the timestamped forecast-as-of-decision policy is replayed on that scenario, carrying state of charge and source-flow accounting forward.
4. **Perfect-information exact:** the exact energy-hub solver sees the full realized scenario. This is an optimistic reference, not an implementable policy.

### Paired evaluation contract

- Scenario weights are positive integer ppm and sum exactly to 1,000,000. Treat them as design weights unless empirically calibrated.
- Every scenario binds a canonical digest and must share currency, energy quantum, service/lifecycle accounting denominator, physical storage/opening-inventory configuration, and ordered interval grid with the frozen baseline.
- A single baseline plan is verified against its source scenario, then its action sequence is rebound to each realized scenario and independently verified there.
- Static, greedy, rolling and exact results carry independent feasibility verification. Search-budget exhaustion is unresolved, not infeasible. An actual no-plan result is separate from unresolved solver failure.
- Each policy reports feasible, infeasible and unresolved weight; conditional mean cost/CO2e is accompanied by its feasible-weight denominator. Missing scenarios are never assigned zero cost.
- Paired delta statistics are calculated only on scenarios where both compared policies have verified feasible plans. The report includes rolling cost regret relative to the perfect-information reference, static-minus-rolling cost and greedy-minus-rolling cost.
- If rolling's cost is lower than the exact perfect-information minimum beyond tolerance on any scenario with the same objective and constraints, the bake-off fails closed with a comparator-dominance error; inspect accounting, scenario identity or solver correctness before reporting a gain.
- The canonical benchmark digest binds the frozen baseline identity/actions, weighted scenario/forecast episode digests, policy statuses and evaluated output metrics/actions.

### Required interpretation

Feasibility gains and cost gains are separate. If the frozen schedule is infeasible under a stress case but rolling dispatch succeeds, report the avoided infeasibility by scenario/weight; do not hide that case by calculating cost only on successful static cases. Conversely, if rolling remains more expensive than the clairvoyant reference under an inaccurate forecast, report the regret as the observable cost of forecast error and delayed information.

A paired synthetic fixture holds identical service and storage configuration. One realized scenario has cheap future grid energy; a second has a first-interval grid-capacity outage. The frozen plan becomes infeasible in the outage while rolling recourse uses the opening inventory and then buys later grid energy. The exact full-information cost is a lower bound; rolling regret is computed only on the jointly feasible subset. All inputs are synthetic.

### Claim ceiling

This evaluator is a small synthetic benchmark harness. It does not imply calibrated scenario probabilities, real-world dispatch savings, causal operational impact, aircraft feasibility or SOTA outperformance. The next benchmark stage must reproduce a published charging/microgrid baseline on matched data/formulation and then compare paired held-out scenarios with identical inputs and compute budgets.
