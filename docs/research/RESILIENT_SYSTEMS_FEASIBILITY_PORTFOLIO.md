# Resilient Systems Feasibility Portfolio
**Status:** research synthesis and first-pass screening; not a validated engineering design, cost estimate, life-support design, or investment recommendation  
**As of:** 2026-10-16  
**Proposed home:** Symthaea engineering research  
**Related work:** [Vacuum-airship metamaterial screening](https://github.com/Luminous-Dynamics/symthaea/blob/research/vacuum-airship-screening-v1/docs/research/VACUUM_AIRSHIP_METAMATERIAL_SCREENING.md)

## Executive decision

Treat the seven ideas as one research program: **design for continuity of essential capability, not merely peak performance of individual machines**. The portfolio covers self-repairing structures; heavy-cargo transport; a 100-person, two-year settlement; an initial lunar industry; community microfactories; supply networks under node loss; and a humane “minimum viable civilization.”

The strongest near-term approach is a *shared evidence-and-simulation framework*, followed by separate domain studies. Reusing one set of service definitions, dependency graphs, uncertainty rules, and failure scenarios avoids seven incompatible spreadsheets and allows meaningful comparisons across technologies.

| Study | Best first hypothesis to test | First serious falsifier |
|---|---|---|
| Self-repairing structures | Detection, isolation, accessible repair, and redundant load paths may beat material “healing” alone on lifetime safety and cost. | Claimed recovery does not restore the specific function and safety margin required after representative damage and repeated cycles. |
| Heavy-cargo transport | A hybrid network will beat a universal “best vehicle”: rail/water for suitable trunk routes, road for flexible last mile, air for urgency, and airships only where their access/handling advantages outweigh weather and ground-operation penalties. | Whole-mission cost, payload utilization, reliability, or weather downtime erases the niche advantage. |
| 100 people for 730 days | A capability ledger, recovery-time analysis, and realistic provisioning can expose the real mass, power, water, food, and repair bottlenecks before designing buildings. | Seasonal and common-mode losses cause essential services to fall below agreed minimums for longer than the settlement can tolerate. |
| First lunar industry | Sell a mission service first—such as robotic site preparation, regolith moving/sintering, or qualified repair parts—before assuming a commodity export economy. | Delivered equipment, power, spares, and operating burden outweigh the imported mass/cost and mission risk the service avoids. |
| Community microfactory | A mixed repair-and-manufacturing cell for a carefully selected catalog can reduce downtime and critical dependency on distant spares. | Demand is too sparse, quality assurance too expensive, or feedstock/power/skills remain single points of failure. |
| Supply network under node loss | Time-dependent, multilayer dependency analysis will find hidden common-mode bottlenecks missed by supplier counts and static centrality. | The model cannot predict known disruptions or improvement is only a cosmetic score gain without lower service deficits. |
| Minimum viable civilization | Optimize a humane capability floor and recovery capacity across energy, water, food, health, repair, communications, and governance—not just calories, population, or self-sufficiency. | The proposed system meets physical survival minima but fails on health, inclusion, rights, ecological burden, or conflict resolution. |

These are hypotheses, not conclusions. No vehicle, material, settlement, or industrial process is declared feasible merely because a screening model can calculate its inputs.

## The paradigm shift: from object optimization to capability continuity

Conventional feasibility studies often compare peak output, specific strength, unit cost, or steady-state efficiency. Those quantities matter, but they miss the way systems fail: a cheap pump is useless without a compatible seal; a high-yield crop is not food security if its nutrient supply depends on one imported chemical; nominally redundant factories may share one substation; a self-healing skin does not restore a pressure boundary unless the recovered property has been measured.

The proposed unit of analysis is the **essential service delivered over time**.

### A shared model for all seven studies

Represent a system as a time-dependent graph with:

- **Services:** water delivered to specification, safe thermal conditions, food energy and micronutrients, cargo delivered, structural integrity, health capability, communications, production, sanitation, and other explicitly selected outcomes.
- **Assets and dependencies:** equipment, people/skills, energy, feedstock, software, transport, standards, external suppliers, and institutions.
- **State and inventory:** availability, capacity, degradation, stocks, lead times, repair time, substitutability, and confidence in each estimate.
- **Evidence:** source, date, conditions, specimen/scale, uncertainty, independent reviewer, and whether a result is observed, simulated, extrapolated, or assumed.
- **Disturbances:** random faults, targeted loss of a critical node, correlated regional events, seasonal changes, demand shocks, prolonged outages, contamination, and delayed repairs.
- **Policies:** isolation, load shedding, substitute routing, reserve use, repair prioritization, rationing rules, and restoration.

A common control loop is:

1. **Prevent:** simplify, protect, diversify, and design out avoidable failure modes.
2. **Detect:** identify degradation and the confidence of the diagnosis.
3. **Isolate:** contain faults and prevent cascading damage.
4. **Continue safely:** use redundancy, graceful degradation, reserve inventory, and explicit service priorities.
5. **Repair and recover:** restore a verified capability, not just a green status light.
6. **Prove and learn:** record the failure, repair, test result, residual risk, and whether the prior model was wrong.

### Metrics that should replace a single “resilience score”

Report a vector, retaining units and trade-offs:

- Essential service deficit: demand not met to specification over time, separated by service and population.
- Probability and duration of falling below the service floor; time to first violation and time to recover.
- Performance under at least one critical component loss and under correlated/common-mode loss.
- Median and tail outcomes (for example, 50th and 95th percentile loss and recovery time), with uncertainty and scenario weights stated.
- Lifecycle cost, energy, emissions, material throughput, maintenance labour, and spare-part burden.
- Local capability retained during isolation; external dependencies that cannot be substituted; ecological and distributional effects.
- Evidence quality and the range over which a result has actually been validated.

Do not collapse this vector into one weighted number until the decision-maker can inspect the weights and the Pareto frontier. A system that costs less but creates a higher chance of catastrophic service loss is not automatically better.

## Study 1 — Self-repairing structures: make repair a verified system property

### Research finding

Self-healing materials are a real field, including intrinsic reversible-bond systems, encapsulated agents, vascular networks, biological mechanisms, and self-healing cementitious approaches. Their effectiveness depends on the material, environment, damage, and target property; repeatability and representative field validation remain important barriers. Reviews explicitly distinguish promising laboratory results from design-practice readiness. See the 2025 [comprehensive review of self-healing cementitious composites](https://doi.org/10.1016/j.istruc.2024.108038), the 2025 [perspective on integrating self-healing into reinforced-concrete design](https://doi.org/10.1002/suco.202401045), and the 2025 review of [bio-based self-healing composites](https://doi.org/10.1186/s42252-025-00065-x).

### Paradigm shift

Move from “the material heals” to **damage-management architecture**:

- **Material layer:** suppress microcracks or restore a narrowly specified function where evidence supports it.
- **Sensing layer:** detect damage using redundant, calibrated methods; quantify false negatives and false alarms.
- **Structural layer:** maintain alternate load paths and prevent one crack from immediately becoming a system failure.
- **Service layer:** isolate, unload, patch, replace, or bypass the damaged module safely.
- **Proof layer:** verify recovered strength, stiffness, leak tightness, fatigue life, and/or environmental barrier function—only the properties actually needed.

The novel design target is the full loop: detect → classify → isolate → repair → inspect → requalify → return to service. Self-healing should be an additional defence, not a substitute for inspection, redundancy, or a safety case.

### First-pass experiment

Compare four matched designs using the same mission, geometry, loading, and material baseline:

A. conventional design and scheduled inspection;  
B. structural redundancy plus modular, replaceable joints;  
C. damage sensing plus accessible manual/robotic repair;  
D. B + C + a self-healing constituent.

Inject representative damage types and locations, including a defect that the chosen sensor is weak at detecting. Test environment, manufacturing defects, progressive/fatigue damage, repair-cycle repeatability, and common-mode sensor failure. Use blind hold-out damage cases rather than only cases used to tune the monitor.

Measure detection probability by defect size/location, time to isolate, residual load capacity before repair, recovered properties after repair, uncertainty, mass/energy overhead, consumables, and lifetime maintenance time. Count recovered structural capacity as **zero** until a matched test supports a positive value. A seal restored to leak-tightness is not proof that strength has recovered.

### Go/no-go gates

Proceed only if representative tests show a material benefit over the simpler alternatives after added mass, inspection, maintenance, and repair complexity are counted. Do not claim flight, pressure-vessel, or human-safety readiness from coupons or simulations alone.

## Study 2 — Heavy-cargo transport: compare missions, not vehicle fandom

### Baseline evidence and framing

Transport modes have very different energy and greenhouse-gas profiles, but the result depends on route, load factor, terrain, urgency, handling, and the full logistics chain. The IEA reports that transport accounts for around 30% of global energy demand and that road transport dominates domestic transport energy use in its 2025 analysis ([IEA, Energy Efficiency 2025](https://www.iea.org/reports/energy-efficiency-2025/transport)). The European Environment Agency finds large differences in freight greenhouse-gas intensity by mode, with rail and waterborne modes important low-carbon options where they fit the mission ([EEA, rail and waterborne transport](https://www.eea.europa.eu/en/analysis/publications/rail-and-waterborne-best-for-low-carbon-motorised-transport)). These broad averages are not route-specific performance guarantees.

### Paradigm shift

Choose the network that delivers the cargo reliably, rather than asking which vehicle wins in isolation. Include origin pickup, transshipment, loading/unloading, staging, return legs, weather closures, permits, crew, insurance, infrastructure, deadheading, maintenance, and inventory cost from end to end.

Compare at least:

- Rail and inland/coastal water for compatible trunk movements.
- Electric or other road freight for distribution, access, and flexible routes.
- Conventional aviation for urgency and hard-to-reach destinations where its energy and cost premium is justified.
- Conventional helium/hot-air airships as separate candidates only where low ground pressure, large cargo, or runway-free access matters.
- Vacuum-airship concepts as a speculative, high-risk branch. The accompanying Symthaea study is a structural and mass-balance screen, not evidence that a pressure-resistant envelope is buildable or flight-qualified.
- Hybrid chains, such as rail trunk + road last mile, or ship/rail + aerial lift for the final inaccessible segment.

### Fair-comparison protocol

Fix the same cargo mass, dimensions, origin-destination pair, delivery window, safety requirement, and service level. Evaluate light, bulky, dense, oversized, and remote cargo separately. Run optimistic, base, and adverse conditions, plus delayed/failed-dispatch cases. A technology that wins only with 100% utilization or perfect weather should say so.

Core outputs: delivered cost per tonne, energy and life-cycle emissions per tonne-kilometre, door-to-door time distribution, probability of meeting deadline, serviceable payload fraction, fleet utilization, ground labour, infrastructure cost, and recovery after vehicle/route outage.

### Initial result

The correct prior is **portfolio specialization**, not a universal winner. Rail/water deserve first comparison for large predictable flows when infrastructure exists. Road is often required for last mile and flexible access. Air is for valuable time savings. An airship may occupy a narrower but worthwhile niche on routes where roads/runways are absent and the cargo is bulky or oversized. Whether that niche pays depends on operational utilization, weather, ground handling, and life-cycle safety—not buoyancy alone.

### Falsification gate

Reject any mode recommendation based solely on cruise energy, nominal payload, or brochure range. The result must survive realistic loading, handling, standby, weather, return-trip, and repair assumptions.

## Study 3 — 100 people for two years: a quantitative service-floor baseline

This first arithmetic screen is deliberately transparent. It does not assert that one fixed set of consumption assumptions applies to every climate, culture, diet, building, or technology stack.

### Shared denominator

For 100 people and 730 days:

**100 × 730 = 73,000 person-days.**

That shared denominator makes food, water, healthcare, maintenance, and replacement plans comparable.

### Food

Use an initial design range of 2,300–3,035 kcal per person-day, with the upper value anchored to a NASA habitability requirement for a particular crew design case, not a universal human requirement ([NASA habitability functions](https://www.nasa.gov/reference/7-0-habitability-functions-vol-2/)). This is **167.9–221.6 million kcal** over two years.

If provisioned edible food averages 3,000–4,000 kcal/kg, the arithmetic-only edible mass is approximately **42–74 metric tonnes**. This is not a complete food logistics mass: packaging, spoilage, dietary diversity, micronutrients, medical diets, processing, cooking, and emergency reserves are excluded. A farm or greenhouse must earn credit for calories and nutrients actually delivered through the entire year, not just fresh produce or nominal growing capacity.

A viable agriculture plan must close or provision nitrogen, phosphorus, potassium, trace elements, seed, pollination, water, lighting, disease control, and waste treatment. Leafy greens can improve diet and morale without supplying much of the settlement’s total calories.

### Electricity

Use 5, 10, and 20 kWh per person-day as **illustrative planning scenarios** for delivered electricity, not as a universal standard and not including all heating or industrial process energy.

| Electricity scenario | Site total per day | Total over 730 days |
|---:|---:|---:|
| 5 kWh/person-day | 0.5 MWh/day | 365 MWh |
| 10 kWh/person-day | 1.0 MWh/day | 730 MWh |
| 20 kWh/person-day | 2.0 MWh/day | 1,460 MWh |

Do separate hourly power sizing from stored energy. A 1 MW peak load is not equivalent to 1 MWh/day, and storage must include conversion losses, reserves, ageing, temperature, and inverter constraints. Thermal loads are location-dependent and should be modeled separately. The resilient design question is not “How much solar?” but “Which essential loads remain served after a long low-generation interval, a failed inverter, a fuel-delivery delay, or a battery-string outage?”

### Water

Use 50–100 L per person-day as a **gross service-throughput scenario** for a settlement with sanitation and ordinary non-potable uses; actual requirements vary with climate, hygiene, industry, and the service standard. That yields 5–10 m³/day, or **3,650–7,300 m³ gross throughput** over 730 days.

A purely illustrative 90% recovery assumption would reduce idealized make-up water to 365–730 m³ over the period. NASA reports roughly 90% water recovery for its space-station system, but that figure is not a design guarantee for a ground settlement or for agriculture ([NASA ECLSS](https://www.nasa.gov/reference/environmental-control-and-life-support-systems-eclss/)). Real make-up demand must also account for rejects, contamination, leakage, irrigation losses, cleaning, chemical quality, and reserve requirements. Potable, process, fire, and irrigation water should be tracked separately.

### Healthcare, sanitation, communications, and repair

For each service, specify a minimum operating mode and an outage tolerance. The first version should include emergency triage/stabilization, diagnostics, cold-chain needs, chronic medication continuity, infection prevention, accessible sanitation, waste handling, communications within the settlement and externally, fire response, and the skills needed to maintain them. Avoid pretending that a settlement can locally manufacture complex pharmaceuticals, semiconductors, or every specialized medical device.

Build a capability ledger for every essential asset: function, failure modes, compatible substitutes, critical spares, repair skills, repair tools, expected lead time, maintenance schedule, and single-point dependencies. Carry replacement capacity for frequent/simple failures; stock life-limiting and long-lead parts; establish mutual aid or external resupply for specialized capabilities.

### Experiments that matter

Simulate and tabletop-test loss of grid power for 3, 14, and 30 days; water-treatment train outage; contamination; crop failure across a season; communications isolation; loss of the only trained specialist; transport interruption; and simultaneous failures. Require the system to state which services are preserved, which degrade, who decides priorities, and how restoration is verified.

### Go/no-go gates

Do not call the settlement self-sufficient because annual average generation matches annual average demand. It must clear a time-resolved seasonal balance and service-floor test, including repair labour and the parts/skills needed to restore the system.

## Study 4 — The smallest useful lunar industry: sell a mission service first

### Current technical boundary

NASA’s lunar surface work identifies continuous power, in-situ resources, construction/site preparation, autonomy, and dust mitigation as foundational needs ([NASA Lunar Surface Technology](https://www.nasa.gov/lunar-surface-technology/)). NASA’s advanced-manufacturing work reports a laser-based regolith-simulant process at TRL 4 and is still working toward harsher-environment demonstrations; this is valuable progress, not proof of an economically productive factory on the Moon ([NASA NTRS: ISRU with advanced manufacturing](https://ntrs.nasa.gov/citations/20250010678)). NASA’s integrated lunar-power analysis emphasizes that energy-storage mass and local lighting conditions can materially affect architecture ([NASA NTRS: Integrated Lunar Power Strategy Considerations](https://ntrs.nasa.gov/citations/20250010948)).

### Paradigm shift

Do not start with “mine the Moon and export commodities.” Start with **which mission bottleneck can local material and autonomy remove, at a lower whole-mission burden than importing a finished solution?**

Candidate first services, to be compared rather than preselected:

1. **Robotic site preparation:** regolith moving, berms, dust control, road or landing-surface preparation, and sensor-based site characterization. This may create direct mission value without first solving high-purity chemistry.
2. **Qualified local repair parts and tooling:** fixtures, covers, brackets, tools, and then progressively more demanding components whose material/process/inspection requirements can be proven.
3. **Oxygen or other extracted resources:** potentially important, but the chain includes feedstock characterization, power, extraction yield, separation, purity, storage, transport, maintenance, and end-use compatibility.
4. **Construction feedstock:** useful only after environmental exposure, mechanical properties, production rates, quality assurance, and installation methods are validated.

### Two break-even tests

A **mass break-even** is not simply the mass of local product versus imported product. It compares the lifetime mass delivered for the whole local production system—excavator/miner, power, thermal control, tooling, spares, consumables, storage, and replacement plant—against the imported mass and operational burden it avoids.

For an explicit mission count (N), define:
- imported baseline mass/cost per mission;
- local plant delivered mass and lifecycle support burden;
- local output rate, usable yield, reject rate, and qualified service life;
- extra power generation/storage and maintenance burden;
- probability the local service is unavailable when needed.

Only then calculate mass and cost break-even. A local product that arrives after the construction window or cannot be qualified is not a substitute.

### First demonstrator

Pick one measurable operation—such as placing a defined berm volume or manufacturing a defined non-critical part—using a specified regolith simulant and dust/thermal environment. Record productivity, kWh/kg, wear, jam rate, maintenance minutes, spare mass, output quality, and unattended uptime. Then evaluate the break-even over low, base, and high mission demand. Do not claim lunar readiness from terrestrial demonstrations alone.

### Go/no-go gates

The first industry should solve an identifiable mission need and demonstrate sustained autonomous operation plus realistic maintenance. Avoid choosing a glamorous commodity before securing power, logistics, quality, and a credible customer.

## Study 5 — Community microfactory: a repair commons, not a room full of printers

### Research finding

A 2025 systematic review of 77 studies finds that additive manufacturing can support on-demand spare parts, smaller inventories, shorter lead times, and flexible production, while identifying gaps in material sustainability, quality control, design reliability, intellectual property, and regulation ([systematic review of AM in spare-parts management](https://doi.org/10.1016/j.clet.2025.101029)). Decentralized production is not automatically environmentally preferable; energy, material, transport, and utilization need case-specific comparison ([Peron et al., 2025](https://doi.org/10.1109/TEM.2025.3540938)).

### Paradigm shift

Define a microfactory as a **local repair-and-production capability** comprising diagnosis, metrology, disassembly, CAD/reverse engineering, repair, additive and subtractive manufacturing, joining, electronics repair, inspection, records, trained operators, and stocked feedstock. A printer alone is not a factory.

Start with a bounded catalog supporting water pumps, irrigation, agricultural equipment, bicycles and utility vehicles, simple electrical enclosures, household hardware, workshop tooling, and non-critical fittings. Classify parts by criticality and qualification burden:

- **Tier 1:** non-safety-critical jigs, covers, holders, spacers, fixtures, and simple replacements after fit/function checks.
- **Tier 2:** working mechanical/electrical parts requiring specified strength, heat, chemical, dimensional, or electrical tests.
- **Tier 3:** pressure boundaries, lifting parts, high-speed rotating components, medical/aviation safety-critical parts, complex semiconductors, and other parts requiring formal process qualification or specialist supply. Do not silently print these from a photograph and treat them as equivalent.

### Business and resilience test

For each candidate part, compare central purchasing with local repair/production using: annual demand, stockout frequency and cost, critical downtime avoided, freight/lead time, inventory carrying cost, local energy/material/labour, machine utilization, setup time, scrap/rework, metrology/qualification, licensing, and capital depreciation.

A useful admission rule is: produce locally when the expected avoided downtime/stockout and total landed-cost advantage exceed the full local lifecycle cost **and** the process can meet documented acceptance criteria. Keep a central source or safer substitute where local production fails the quality gate.

### First experiment

Collect a real or transparently synthetic catalog of 50–100 common maintenance parts from a defined community. Categorize each by criticality, annual frequency, failure consequence, lead time, material/process, inspection, and reverse-engineering rights. Select the first 10 parts only after comparing local and centralized lifecycle scenarios. Test repeated production batches—not just the best specimen—and retain material/process/inspection records.

### Go/no-go gates

The pilot must show actual reductions in time to repair or stockout loss, not merely the number of printed parts. Measure quality escape rates, scrap, power use, feedstock dependence, and whether one operator or software tool is a hidden single point of failure.

## Study 6 — Supply networks after a critical node disappears

### Research basis

Supply-network research has long shown that cascading failure can occur in interdependent networks; newer work stresses time-dependent recovery and heterogeneous node behavior. A February 2026 study proposes adaptive risk-recovery strategies and a 2026 *Reliability Engineering & System Safety* study analyzes temporal network evolution and recovery, reinforcing the case for studying dynamics, not topology alone ([Hu et al., 2026](https://doi.org/10.1007/s10479-026-07070-8); [Shi et al., 2026](https://doi.org/10.1016/j.ress.2025.111763)). These models are not ready-made proof that any specific network is robust; they guide scenario construction and validation.

### Paradigm shift

From a supplier map to a **multilayer, time-dependent dependency model**. A node may be a supplier, factory, port, power substation, communications service, specialist, software signing authority, or finance/logistics provider. Edges should carry quantities, compatibility, qualification constraints, lead times, and the possibility of a shared upstream dependency.

Supplier diversity is not true redundancy if two nominally independent vendors rely on the same mine, cloud, shipping lane, energy source, firmware, or skilled technician.

### Benchmark disruptions

Use a controlled 24-month scenario horizon with repeatable disruption seeds, including:

- random single-node outage;
- targeted loss of the highest-throughput or highest-betweenness node;
- loss of the sole qualified source for a critical part;
- a regional shock that removes multiple apparently separate suppliers;
- power or communications outage affecting physical suppliers simultaneously;
- demand collapse (underload) and surge (overload);
- prolonged shipping interruption plus low inventory;
- repair crew/skill unavailability and slow qualification of a substitute.

Do not tune only to catastrophic events: routine delayed deliveries and small failures can accumulate into larger service deficits.

### Metrics and interventions

Report service deficits by time and service class, time to first violation, cumulative unmet demand, time to recover, lost throughput, spoilage/obsolescence, and recovery tail risk. Compare interventions individually and in combination: independent second sources, local inventory, repair capacity, interoperable designs, alternative routes, priority allocation, shared emergency stocks, and reduced single-source coupling.

Simulate static centrality as a baseline, then show when time-dependent capability-aware analysis changes the intervention ranking. Validate on known historical disruptions before presenting synthetic stress-test results as predictive.

### Go/no-go gates

An intervention counts as effective only if it lowers measured or simulated service deficits under disclosed scenarios without shifting equivalent risk to an unmodeled layer. Publish failed cases and assumptions; never report “network resilient” as an unqualified binary label.

## Study 7 — Minimum viable civilization: a humane, recoverable capability floor

This is the synthesis study, not an excuse to equate civilization with survival rations or isolated self-sufficiency. A society can produce calories and still fail through preventable disease, exclusion, unsafe shelter, coercion, ecological collapse, or inability to resolve conflict.

### Define “viable” before optimizing

Set explicit floors across:
- safe and accessible drinking water and sanitation;
- sufficient, diverse nutrition and culturally acceptable food;
- shelter, thermal safety, and fire protection;
- reliable energy for essential loads;
- basic healthcare, medicine continuity, and emergency response;
- communications, education, knowledge preservation, and trusted records;
- repair/manufacturing, tools, and critical skills;
- logistics and credible external support where local capability is not economical;
- legitimate decision-making, accountability, conflict resolution, privacy, and protection of individual rights;
- ecological limits and fair distribution of burdens and benefits.

Values and governance need human deliberation; a simulation can make consequences visible but cannot decide whose rights may be traded away.

### Paradigm shift

Move from maximizing aggregate efficiency or “percent self-sufficient” to finding a **Pareto frontier of dignified capability, autonomy, external dependence, environmental impact, and recovery capacity**. Some imports increase resilience if they diversify supply; forced autarky can increase risk. Local redundancy is worthwhile for high-consequence essentials, not necessarily for every commodity.

The two-year settlement is a useful initial scenario. The broader model should test multiple population sizes, climates, settlement types, and levels of trade. It must include the institutions and maintenance work that keep capabilities available, not just physical assets.

### First outputs

A living capability ledger; time-resolved mass/energy/material balances; service floors and outage tolerances; an inventory of single points of failure; a skill/maintenance roster with succession; emergency allocation rules; ecological budgets; and recovery tests. Publish which assumptions are normative rather than empirical.

### Go/no-go gates

Reject any configuration that meets an aggregate average by allowing a subgroup to fall below its minimum, or that depends on unacknowledged unpaid labour, impossible specialist staffing, hidden imports, or an unrepairable single point of failure.

## Cross-study findings: the most promising paradigm shifts

### 1. Build for recovery, not the fiction of zero failures

Every real system will experience damage, delays, defects, and surprises. The design question becomes: can the system detect, isolate, continue safely, recover, and prove it is safe again? This shifts value toward modularity, observability, standard interfaces, substitute capability, and repair.

### 2. Optimize the end-to-end service, not the impressive component

Vehicle payload, material strength, energy density, printer speed, crop yield, and extraction efficiency are intermediate quantities. The objective is cargo delivered, structure safely functional, food actually consumed, water delivered to quality, or a critical part restored on time.

### 3. Treat time, uncertainty, and correlated failure as first-class inputs

Annual averages hide dark periods and seasonal harvests. Supplier counts hide shared dependencies. A single probability hides a heavy tail. Use event histories, scenario ensembles, bounds, confidence intervals where supported, and explicit unknowns. Do not manufacture precise probabilities when data are absent.

### 4. A repairable capability is more valuable than a stockpile of opaque parts

Stock is essential, but storage ages, inventories become obsolete, and rare faults exceed catalogued spares. Pair stock with repair skills, tooling, diagnostic data, open interfaces where possible, and locally producible consumables. For critical components, a digital design file is not an equivalent replacement until process and output quality are qualified.

### 5. Evidence must travel with the design and part

For each material, part, model, or alternative supplier, retain versioned specifications, provenance, environment, process parameters, test data, uncertainty, inspection, repair history, and authorization boundaries. A cryptographically intact record does not make its underlying claim true; it makes the claim traceable and its tampering detectable.

### 6. Use hybrid centralization

Central production can improve precision, utilization, and quality. Distributed production can reduce lead time and single-route exposure. The optimum is likely layered: local diagnosis and routine repair, regional microfactories for selected parts, specialist central plants for rare/high-specification processes, and alternative qualified sources for life-critical needs.

### 7. Make the objective humane and multi-dimensional

Maximizing redundancy can waste resources; maximizing efficiency can destroy slack. A single resilience score can hide who bears the losses. Keep the constraint floors explicit, show the Pareto frontier, and report subgroup impacts alongside aggregate metrics.

## Unified execution plan

### Phase A — Define and normalize (now)

For each study, publish the mission boundary, system diagram, required services, input data, assumption register, failure scenarios, metrics, stop criteria, and source list. Keep source observations separate from extrapolations and engineering assumptions.

### Phase B — Build deterministic screening and sensitivity tests

Start with transparent arithmetic and small, auditable models. Sweep uncertain inputs instead of providing one preferred number. Include negative controls—for example, remove the supposed redundant source, set repair time to infinity, or introduce a shared dependency—and ensure the model’s result worsens as expected.

### Phase C — Add time-resolved simulation

Represent inventory, service demand, degradation, lead times, outages, and restoration. Seed stochastic experiments deterministically. Report the code revision, parameter file, seed, trial count, distribution summaries, and scenario failures. Keep deterministic benchmark cases to detect model regressions.

### Phase D — Validate against observations

Compare transport assumptions with actual route/dispatch data; repair assumptions with repeated specimens and field maintenance; settlement assumptions with metered services and seasonal yields; lunar manufacturing with realistic dust/thermal/vacuum tests and a customer mission; microfactory assumptions with part catalogs and production batches; network models with known disruptions; and capability-floor decisions with community/stakeholder review.

### Phase E — Decision and publication

Publish a decision record for each study: what evidence supports the claim, what failed, who/what the result applies to, confidence and limitations, remaining critical unknowns, and the next cheapest experiment that could change the decision.

## Immediate recommended order

1. **Complete the 100-person / two-year service ledger first.** It creates a concrete common benchmark with arithmetic that can be independently checked.
2. **Build the supply-network loss and recovery simulator around that ledger.** It reveals the dependencies that determine whether the settlement works during a disruption.
3. **Select the community microfactory’s 10-part pilot catalog from actual critical dependencies.** This turns abstract resilience into measurable repair-time reduction.
4. **Run the transport comparison on a fixed cargo mission.** Use the actual origin, destination, cargo envelope, service window, and alternative chain; compare all modes on the same boundary.
5. **Run self-repair component experiments in parallel** where a specific structure/use case can be defined and safely tested.
6. **Gate the first lunar-industry concept on a named mission/customer and delivered-system break-even.** Do not extrapolate ground production rates directly to lunar operations.
7. **Integrate results into the minimum-viable-civilization Pareto analysis.** It should consume results from the other studies rather than inventing separate assumptions.

The order can change when evidence or a concrete customer makes a different experiment more decision-relevant. There is no need to wait for every study before publishing validated components.

## Limitations and integrity rules

- This is a first research synthesis, not a literature-complete systematic review.
- Numerical figures in the settlement section are arithmetic from explicit scenario assumptions; they are not measured operating data or universal requirements.
- Literature results from concrete, composites, supply networks, AM, and lunar simulants are not automatically transferable to airship structures, field factories, a real settlement, or lunar operations.
- Simulated results are not experiments; a successful compile is not scientific validation; a skipped workflow is not a passing check.
- No commercial forecast, flightworthiness, lunar-readiness, safety certification, or “self-sufficient civilization” claim follows from this document.
- Research should prefer published sources, reproducible assumptions, open evaluation, and safe staged tests over marketing claims.

## Selected source register

1. [NASA — Environmental Control and Life Support Systems (ECLSS)](https://www.nasa.gov/reference/environmental-control-and-life-support-systems-eclss/) — space-station water, air, and oxygen subsystems; approximately 90% water recovery in that specific system.
2. [NASA — Habitability functions, food caloric content](https://www.nasa.gov/reference/7-0-habitability-functions-vol-2/) — a 3,035 kcal/day example requirement for a specified crew-design context.
3. [IEA — Energy Efficiency 2025: Transport](https://www.iea.org/reports/energy-efficiency-2025/transport) — transport energy context and modal framing.
4. [European Environment Agency — Rail and waterborne transport](https://www.eea.europa.eu/en/analysis/publications/rail-and-waterborne-best-for-low-carbon-motorised-transport) — European freight GHG mode comparisons and caveats.
5. [A comprehensive review: Self-healing methods and cementitious composites](https://doi.org/10.1016/j.istruct.2024.108038) — methods and limitations of autonomous/autogenous healing.
6. [Perspectives on incorporating self-healing in reinforced-concrete design](https://doi.org/10.1002/suco.202401045) — integrating material evidence into design practice.
7. [Biobased self-healing functional composites](https://doi.org/10.1186/s42252-025-00065-x) — bio-based matrices and healing approaches.
8. [NASA — Lunar Surface Technology](https://www.nasa.gov/lunar-surface-technology/) — lunar power, ISRU, construction, autonomy, and dust mitigation needs.
9. [NASA NTRS — ISRU with Advanced Manufacturing](https://ntrs.nasa.gov/citations/20250010678) — regolith-simulant manufacturing research and stated technology maturity.
10. [NASA NTRS — Integrated Lunar Power Strategy Considerations](https://ntrs.nasa.gov/citations/20250010948) — lunar power and storage-mass architectural considerations.
11. [The role of additive manufacturing in spare-parts management: systematic review](https://doi.org/10.1016/j.clet.2025.101029) — 77-study synthesis and open quality/sustainability gaps.
12. [Peron et al. — Is decentralized AM always environmentally preferable?](https://doi.org/10.1109/TEM.2025.3540938) — reason to perform case-specific life-cycle comparisons.
13. [Hu et al. (2026) — Adaptive and risk recovery strategies for supply-chain cascading failures](https://doi.org/10.1007/s10479-026-07070-8) — heterogeneous failure and adaptive recovery.
14. [Shi et al. (2026) — Temporal modeling and resilience analysis of supply chains](https://doi.org/10.1016/j.ress.2025.111763) — time-dependent disruptions and recovery.

---
*Research status: hypotheses and first-pass arithmetic only. Each study must progress through explicit evidence gates before any readiness or deployment claim.*


## Implementation update — deterministic service continuity screen

A first Rust implementation now exists at [`symthaea_engineering::resilient_service_screening`](../../crates/domains/symthaea-engineering/src/resilient_service_screening.rs). It models essential services, directed dependencies, capacity disruptions and linear recovery, and reports service traces, cumulative unmet service, service-floor violations and stable recovery.

The abstraction is deliberately narrow: dependencies transmit the weakest upstream fulfillment fraction; simultaneous capacity factors multiply; there are no inferred failure probabilities. It supports deterministic arithmetic and causal-direction tests, not prediction of real-world outage likelihood.

See the [resilience benchmarks and acceptance gates](./RESILIENT_SYSTEMS_BENCHMARKS.md) for the B0–B5 ladder, RS-01–RS-14 scenario catalog, metamorphic controls and calibration requirements.

### New acceptance targets

- **Correctness:** nominal sufficient-capacity case has zero deficit; malformed scenarios fail explicitly.
- **Redundancy honesty:** independent source loss may retain the floor only when modeled residual capacity is sufficient; common-mode loss must remove that credit.
- **Monotonicity:** greater disruption severity and slower recovery must not reduce simulated unmet service; extra nominal capacity must not worsen fulfillment.
- **Recovery honesty:** report stable recovery only after the final violation; if recovery is outside the modeled horizon, report it as unknown.
- **Validation:** calibrate against observed service outage/recovery traces before making real-world predictions. Proposed target is at least three observed cases, with at least one held-out event and a simple baseline comparison.

These are model-quality gates, not safety certification. Local environment lacked `cargo` and `rustc`, so the added Rust tests are authored but have not been executed locally. The current-head Actions status must be reported separately; skipped workflows are not passes.


## Implementation update — finite shared-resource dynamics

The branch now also includes [`resource_constrained_screening`](../../crates/domains/symthaea-engineering/src/resource_constrained_screening.rs), a deterministic Rust model where essential services consume finite shared inventories. Resources have initial stock, storage caps, replenishment per step, and supply-disruption/recovery events. Services have demand, capacity, dependencies, per-unit resource requirements and an explicit priority.

This is a deliberate paradigm expansion: capacity is not enough if the energy, water, fuel, chemical, feedstock or other resource needed to deliver that capacity is unavailable. The module records per-service fulfillment and unmet demand as well as per-resource inventory, replenishment and consumption. It intentionally does not sum unlike service units into one “resilience” number.

The policy and limitations are material: a deterministic priority order can starve lower-priority services; dependencies propagate the weakest upstream fulfillment fraction; only resource replenishment is interrupted by a resource-supply event; switching to substitutes, repair crew constraints and physical process laws are not yet modeled. These need explicit additional models and fixtures before the results can be used to predict a real settlement or supply network.

The initial module now contains 18 authored unit tests for finite stock, bounded replenishment, supply interruption, priority allocation, deterministic tie-breaking, dependency propagation, resource conservation, multi-resource bottlenecks, input validation and determinism. They are authored but have not been compiled or run in the current tool environment.


## Implementation update — repair crews, precedence and finite spares

A third Rust module, [`repair_crew_scheduling`](../../crates/domains/symthaea-engineering/src/repair_crew_scheduling.rs), now schedules repair jobs with explicit crew skills, task durations, priority, outage start time, prerequisite jobs and finite spare-part requirements. It compares three policies: priority-first, shortest-job-first, and a greedy weighted-service-recovery ratio.

The objective measures caller-weighted service-unit-steps lost from outage until repair completion, capped at the analysis horizon. It is intended for policy comparison on identical scenarios, not as an objective ethical ranking. The implementation is a deterministic list-scheduling heuristic, not a global optimizer.

The initial module now has 17 authored unit tests covering policy comparison, skill eligibility, crew exclusivity, dependency ordering, finite spares, horizon accounting, deterministic output, cycle rejection and invalid inputs. The tests have not been compiled or run in the current environment. The module includes a test-only permutation-enumeration oracle for a tiny, single-crew/no-prerequisite case; on the disclosed weighted two-job fixture it identifies weighted loss 111 as the exact optimum, which the weighted heuristic matches. The next research target is to generalize that oracle to a public small-case corpus, then add uncertain repair time and route/access dependencies. The oracle test is authored but has not been run in Rust.


### Policy-comparison witness (synthetic; not empirical)

An independent arithmetic check was performed on a one-crew, two-job fixture. Priority-first repairs a duration-10, service-weight-1 task before a duration-1, service-weight-100 task, yielding (10\times1 + 11\times100 = 1{,}110) weighted service-unit-steps lost. The weighted-recovery policy reverses the order and yields (1\times100 + 11\times1 = 111), a 90% reduction for this explicitly weighted objective.

This demonstrates why policies should be benchmarked against each other on a disclosed objective and why the objective matters. It does **not** prove the heuristic is globally optimal or that service weights capture ethical priorities. This was a separate arithmetic oracle, not execution of the Rust test.


#### New scheduling heuristic: preserve flexible crews

A new fixture demonstrates that when a mechanical repair can be handled by either a specialist or a multi-skilled technician, while an electrical repair can only be handled by the multi-skilled technician, a deterministic least-flexible-crew tie-break schedules both jobs at step 0. This is a scheduling heuristic, not a global optimality guarantee; benchmarks should compare it against stable-ID-only assignment and other baselines.


## Implementation update — nonanticipative duration stress evaluation

A fourth module, [`robust_repair_evaluation`](../../crates/domains/symthaea-engineering/src/robust_repair_evaluation.rs), evaluates priority-first, shortest-job-first and weighted-recovery repair policies under alternative realized-duration scenarios.

For each policy, the initial schedule is selected from nominal inputs only. The initial dispatch order and crew assignments are then kept fixed during scenario replay. Realized repair times can delay later jobs sharing a crew and jobs with prerequisites, but they cannot influence the original schedule. The report includes weighted expected loss, worst-case loss, upper-tail CVaR, and per-scenario traces.

This creates an explicit boundary between a schedule that looks good given assumed durations and one that is robust under disclosed alternatives. Scenario weights must sum to one, but are not automatically empirical probabilities. The next layer should compare fixed-plan replay with a separate rolling-horizon policy that makes decisions only from information revealed by that step.

The initial module contains ten authored unit tests for a CVaR tail case, frozen nominal job order, repair-duration monotonicity, dependency delays, scenario-weight validation, complete per-job scenario coverage, invalid alpha/duration inputs, crew-ready times and determinism. These tests are authored but have not been compiled or run in the available environment.


## Implementation update — event-triggered rolling-horizon repair

The fifth engineering model, [rolling_horizon_repair.rs](../../crates/domains/symthaea-engineering/src/rolling_horizon_repair.rs), dispatches repairs at observable event times. Instead of freezing the initial order, it re-evaluates released and precedence-ready jobs after actual completions, while enforcing skill eligibility, crew exclusivity and currently available spare stock. Its decision records expose the jobs and crews visible at each decision.

This is a shift from static resilience plans to **closed-loop recovery policies**. It allows the simulation to respond to delayed repairs and newly eligible work without giving the policy future realized durations: pending work is ranked by nominal durations, and the scenario's actual duration is used only after the dispatch choice to schedule the completion event. A paired synthetic witness changes fixed order A→C→B to adaptive order A→B→C and reduces weighted loss from 922 to 724. That is a fixture-specific result, not a general superiority or field-validation claim.

The next critical step is to publish a versioned paired scenario corpus, generalize the exact tiny-instance oracle, add a held-out split and report both improvements and regressions. The paired batch evaluator already provides per-case deltas plus expected/worst/CVaR loss and retains both traces. Full travel/access, crew shifts, replenishment, statistical uncertainty estimates, and calibration to observed restoration records remain out of scope. See the [RH benchmark fixtures and targets](./RESILIENT_SYSTEMS_BENCHMARKS.md#8-event-triggered-rolling-horizon-repair-dispatch).


## Implementation update — paired fixed-versus-adaptive policy evaluation

A sixth Rust module, [repair_policy_comparison.rs](../../crates/domains/symthaea-engineering/src/repair_policy_comparison.rs), now evaluates fixed nominal-plan replay and event-triggered repair dispatch against the same scenarios. It reports the fixed and adaptive expected loss, worst-case loss and upper-tail CVaR, per-case loss deltas, win/loss/tie counts, and both the fixed schedule and adaptive event trace.

This closes the first methodological loop: resilience planning is no longer only about a single chosen schedule or a single synthetic win. The evaluator can show where adaptation helps, where it regresses and how the tail changes, while retaining scenario identity. Scenario weights remain caller-supplied and are not empirical probabilities by default.

Six source-level tests now exercise the paired synthetic witness, scenario retention for all three policies, separate CVaR values, invalid-weight rejection, determinism and delta arithmetic. They remain authored but unexecuted. The next research gate is a published paired scenario corpus with an exact small-case oracle and a holdout split; actual operations data are still needed before prediction claims.

## Implementation update — exhaustive small-instance oracle and fixture corpus

A seventh engineering module, [repair_small_instance_oracle.rs](../../crates/domains/symthaea-engineering/src/repair_small_instance_oracle.rs), exhaustively enumerates all job permutations for up to eight independent repairs on a single qualified crew. It rejects unsupported prerequisites, spare-part requirements, nonzero outage starts, invalid weights and oversized instances. This creates a stricter ladder for measuring heuristic quality against an exact answer instead of a hand-picked comparison.

The public [JSON corpus](./repair_scheduling_oracle_corpus.json) contains six cases: a 1,110-vs-111 priority-policy counterexample, three-/four-job weighted-completion cases, a deterministic tie-break, a one-job boundary, and an eight-job / 40,320-permutation stress case. The first five expected results are hand-checkable. The eight-job case's expected result (loss 406; order b→f→d→h→c→g→e→a) was independently enumerated in JavaScript over all 40,320 permutations; the Rust assertion remains unexecuted.

Eleven source-level tests cover the oracle and policy-gap evaluator, increasing the portfolio total to 93 authored Rust tests. They are not compiled or executed in this tool environment. The exact oracle is narrowly scoped and must not be presented as a general optimizer; next work should bring the same correctness ladder to multi-crew feasibility, dependencies, finite stock and adaptive policies.

## Implementation update — tiny exact multi-crew repair oracle

The next layer is [multi_crew_repair_oracle.rs](../../crates/domains/symthaea-engineering/src/multi_crew_repair_oracle.rs): an exhaustive dispatch-sequence and skill-qualified crew-assignment oracle for up to six jobs and three crews. It honors acyclic repair prerequisites and crew exclusivity, places selected tasks at the earliest feasible time, and computes weighted completion loss. It also benchmarks the existing priority-first, shortest-job-first and weighted-recovery policies against the exact result.

The public [multi-crew fixture corpus](./multi_crew_repair_oracle_corpus.json) covers skill-constrained parallel work (loss 41), a power→pump prerequisite across two crews (loss 52), and preserving a multi-skilled crew for the only plumbing-qualified job (loss 42). All are deterministic synthetic correctness fixtures; none are field-calibrated. The 8 source-level tests are authored but unexecuted. The next step is to import a small published multi-skilled scheduling benchmark and document any semantic differences, then add finite-stock constraints only with their own exact fixtures.

Research grounding: Snauwaert and Vanhoucke provide a classification and benchmark instances for multi-skilled resource-constrained project scheduling ([EJOR 2023](https://doi.org/10.1016/j.ejor.2022.05.049)); Myszkowski and Laszczyk publish multi-objective MS-RCPSP benchmark fronts ([Applied Soft Computing 2022](https://doi.org/10.1016/j.asoc.2022.109253)). These provide external comparison targets, not validation of this implementation.


## Independent schedule-verification boundary

The new `repair_schedule_certificate.rs` module treats each planner output as an untrusted proposal and independently checks its job coverage, skills, durations, crew conflicts, precedence and objective arithmetic. It is a feasibility/objective checker, not a proof of global optimality or a field-safety certification. The next validation phase is to run every tiny-oracle result through this checker and to build a reproducible adapter for published multi-skill scheduling datasets before claiming external benchmark performance.
