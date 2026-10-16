# Living City / Settlement Systems Architecture and Benchmark Agenda v0.1

**Status:** Research and implementation agenda. These schemas, scenarios and targets are proposals, not a qualified urban simulator, space-habitat digital twin, engineering standard or deployment evidence.

**Related work:** Living City PRs #7538 and #7664; compound hazards/service dependencies #7591; context-adaptive cities #7686; Earth-to-space settlement systems #7687; policy robustness #7618; adaptive pathways #7636.

## Thesis

Build a reusable settlement-systems framework that evaluates decisions across interdependent services, uncertain futures and differing community priorities. Reuse semantic and evidence-handling machinery across domains, but do not reuse physical equations, safety limits or validation claims without domain-specific justification.

The current Living City evaluator is a synthetic single-essential-service policy benchmark. It is a seed for deterministic policy comparison, replay and fail-closed evidence handling—not a model of a complete city. The next stage is to define a multi-service contract and reproducible scenario corpus, then qualify each physical domain independently.

## Paradigm shifts

### 1. From point forecasts to robust decisions across plausible futures

Replace the single forecast, single design and single score pattern with a counterfactual workflow:

- Versioned evidence and locally authorized constraints.
- Candidate intervention policies evaluated against paired scenario families with partial observability, delayed interventions and interacting hazards.
- Domain solvers with explicit validation boundaries.
- Hard-floor and affected-group checks before policy ranking.
- Pareto outcomes, scenario-wise regret, service loss and recovery profiles.
- Retained evidence and post-event forecast-versus-observation review.

A forecast can be useful without being authoritative. Judge a policy by how it behaves when a forecast is wrong, observations are stale, interventions are delayed or hazards interact.

### 2. From a monolithic digital twin to a typed settlement graph plus domain solvers

Represent services, resources, infrastructure, dependencies, interventions and evidence as typed records. A dependency graph is a topology/logic contract; it is not a substitute for hydraulic, electrical, thermal, atmospheric or transport dynamics.

A dependency edge should state its meaning, direction, units where relevant, capacity model reference, failure behavior, delay model reference, provenance, generation and currentness. Do not infer ownership, capacity, authority or physical behavior from a connection alone.

### 3. From a universal city template to community-governed context profiles

A context profile can declare climate, ecology, locally approved service floors, stated preferences, uncertainty, evidence provenance and who may authorize changes. Never infer a resident's priorities from ethnicity, nationality, religion, disability or another identity proxy. Gather sensitive context only when necessary, lawful, consented where required, and governed by data minimization.

Different residents may disagree. Preserve the disagreement and its distributional consequences; do not hide it inside an average.

### 4. From a global score to non-substitutable floors plus a vector of outcomes

Keep mandatory safety, rights and essential-service constraints outside soft optimization weights. A benefit in one category must not compensate for a protected floor breach elsewhere.

Report the outcome vector and Pareto trade-offs. Show averages, worst cases, lower-tail performance and per-scenario results. Do not calculate probability-weighted expected loss or CVaR unless scenario probabilities are justified for that application.

### 5. From efficiency alone to repairability, recoverability and graceful degradation

The lowest immediate cost is not always the best policy. Model spare capacity, maintenance access, repair time, skilled-worker availability, manual/offline modes, modularity, reversibility, restoration stability and common-cause failures. Measure time to stable recovery, not just time to first restoration.

### 6. From one Earth model to portable settlement semantics with domain-specific physics

Cities, remote Earth stations, orbital habitats and planetary bases share resource allocation, maintenance, logistics, uncertainty, human needs and governance problems. They do not share identical physical equations or acceptable risk thresholds. Keep common semantics small and explicit; attach each physical/safety model through a versioned domain adapter with its own units, assumptions, validation evidence and claim ceiling.

## Proposed semantic contracts

These are logical contract families for design, not a commitment to one serialized format before review.

- **SettlementProfileV1:** domain, context profile, scope, system boundary, time horizon, units/coordinate/time conventions, assumptions, unknowns, authorized profile owner and revision identity.
- **ServiceContractV1:** service identity, beneficiary/coverage scope, essentiality class, authorized floor, service-quality measure, unit, availability semantics, degradation behavior, evidence and authority reference.
- **ResourceLedgerV1:** resource identity and generation, quantity/unit, source, sink, available capacity, reserve, quality/contamination state, transfer loss, time basis and accounting boundary.
- **DependencyEdgeV1:** source/target component generations, dependency kind, direction, declared capacity or solver reference, latency, failure semantics, common-cause group, provenance and uncertainty.
- **HazardScenarioV1:** initiating event, duration, affected system generations, hazard profile, observation quality, compound interactions, action/repair delays, scenario family and frozen scenario digest.
- **InterventionPolicyV1:** admissible actions, prerequisites, action latency, reversibility, resource demand, cost basis, fallback action, authority ceiling and explicit prohibitions.
- **EvidenceRefV1:** source identity, version/digest, producer, measurement or model status, observation time, validity window, uncertainty, transformation lineage and independent verifier reference.
- **CommunityConstraintV1:** locally authorized constraint, affected scope, approval provenance, owner, review/expiry trigger, appeal/correction path and non-waivable designation.
- **AuthorityBoundaryV1:** who may observe, recommend, approve, execute and stop an action. Recommendations are not actuation authorization.
- **OutcomeVectorV1:** separate service, exposure, affordability, accessibility, ecological, distributional, repairability, resource and governance outcomes with known/unknown markers. No default scalar social-welfare score.

Contracts must reject mixed units, stale inputs where freshness is required, ambiguous generations, missing required provenance, hidden defaults, incomplete cohorts where cohorts are part of the claim, and attempts to promote simulated output into measured or qualified evidence.

## Minimum scenario programme

All initial fixtures should be synthetic, deterministic, versioned and clearly labeled. A future real-place profile requires appropriate local review, privacy safeguards and source provenance.

### Earth city and district

1. Healthy baseline with non-binding service demand.
2. Hot/dry period with high cooling demand, constrained water and electricity reserve.
3. Intense rainfall/flood with transport, drainage and power disruptions.
4. Compound heat + grid outage + healthcare cooling demand.
5. Water-quality incident combined with telemetry delay and uneven alternative access.
6. Power disruption affecting water pumping, communications, refrigeration and a care facility.
7. Transit outage with shift-worker access and step-free route constraints.
8. Housing-cost shock with displacement pressure and changing local service demand.
9. Informal/peri-urban network with partial telemetry, distributed repair capability and incremental infrastructure.
10. Low-connectivity period when cloud services are unavailable but local services must continue.
11. Migration/population shift that changes demand without assuming historical averages persist.
12. Staged investment with false, missed and late triggers, plus a changed hazard regime.

### Remote Earth settlement

13. Delayed resupply and an unavailable critical spare.
14. Severe weather isolates a settlement while energy demand peaks.
15. Local staff/crew capacity drops during simultaneous maintenance tasks.
16. Communications loss with contradictory sensor inputs and degraded but safe local operation required.

### Space habitat: separate physical domain

17. Life-support subsystem degradation with uncertain sensor readings.
18. Power loss or storage degradation affecting life-support loads.
19. Water-recovery yield or purity problem with explicit quarantine and reprocessing assumptions.
20. Pressure, fire/smoke, thermal or radiation-protection event under a domain-authorized safety profile.
21. Dust contamination or external servicing failure for a lunar/Martian surface profile.
22. Cargo delay plus missing spare plus communication latency.
23. Multiple faults sharing a cause, reduced crew capacity and no timely Earth intervention.

Space fixtures are benchmark designs until their parameters and safety envelopes are grounded in appropriate engineering references. A generic city service floor must never stand in for a life-support limit.

## Policy baseline family

Compare policies on the same frozen fixtures and evidence inputs:

- **No intervention:** control/reference baseline.
- **Reactive:** act after an observed threshold breach; specify information latency.
- **Greedy immediate benefit:** choose the largest immediate benefit per declared cost/resource unit.
- **Floor-first:** prioritize avoiding explicitly protected service-floor violations.
- **Reserve-first:** protect declared reserves before discretionary improvements.
- **Scheduled maintenance:** preventive maintenance with explicit schedule and resource demand.
- **Adaptive/reversible:** use predeclared trigger rules and preserve options where feasible.
- **Perfect-information oracle:** optional upper-bound diagnostic only, never presented as an implementable policy.

Freeze policy ordering, action budgets, resource constraints and tie-break rules before evaluation. Do not give a candidate privileged inputs. Keep held-out scenario families separate from tuning scenarios. If a learned policy is introduced, maintain transparent baselines and frozen negative controls.

## Benchmark metrics and gates

### Integrity and semantic correctness

- 100% coverage of required frozen scenarios and declared policy/scenario comparisons, or an explicit blocked result.
- 100% of retained runs bound to exact policy, scenario, configuration, input-evidence and output digests.
- Deterministic replay equality for the deterministic harness.
- All authored negative controls execute and pass; skipped tests do not count as passes.
- Missing, duplicate, extra, wrong-generation, stale-required or unit-incompatible inputs fail closed.
- Attempts to convert unknown evidence to a favorable value, skip an affected group or silently weaken a floor are rejected.

### Outcome and robustness

- Count, duration and severity of floor breaches per service and scenario.
- Worst-case and lower-tail service loss; report scenario-wise values without inventing probabilities.
- Time to detect, isolate, first restore and **stable recovery**.
- Minimum reserve margin and time-to-critical threshold where a domain defines them.
- Resource consumption, transfer losses, waste, repair/spare demand and critical dependency blast radius.
- Distribution of benefits, burdens, exposure and access across explicitly justified affected groups.
- Affordability and complete-journey accessibility where relevant; time and access to care, food, water, sanitation, energy, cooling/shelter, transit and civic participation.
- False/missed action-trigger rates, trigger latency, regret versus frozen baselines, option value and lock-in cost.
- Robustness to missing/stale telemetry, demand change, model-form uncertainty, cost shocks, cross-sector dependencies and changes in authorized criteria.

Do not collapse these into one number. Every metric needs units, directionality, scope, aggregation method, missing-data handling and provenance.

### Initial acceptance thresholds

1. **Hard-invariant campaign:** zero violations in the frozen synthetic test envelope. This is a contract target, not evidence of real-world safety.
2. **Evidence completeness:** every required case is fully measured/replayed or explicitly **Blocked**; no silent omission.
3. **Negative controls:** 100% of required controls execute with expected rejection/blocked outcomes.
4. **Reproducibility:** the same frozen deterministic inputs produce byte-identical canonical payloads and the same digest in the declared execution environment.
5. **Policy efficacy:** first establish the baseline distribution and uncertainty sensitivity; only then register domain-specific improvement targets. The existing 20% worst-case-loss reduction remains a provisional research hypothesis, not an achieved result or universal city target.
6. **External model validity:** no pass based only on internal consistency or self-comparison. Require independent references, pre-registered tolerances, retained results and negative controls before physical claims.
7. **Community validity:** synthetic representational coverage is not cultural legitimacy. A claim about a real community requires local review and an explicit process for correction, disagreement and removal of nonessential personal data.
8. **Space qualification:** synthetic safety invariants are only the first rung; separate hardware-in-loop, integrated analogue, environmental, human-factors and program qualification evidence is necessary.

## Build and research sequence

1. **Protect and qualify the seed:** preserve the synthetic single-service boundary, run exact-head verification and keep queued/skipped status distinct from pass.
2. **Freeze semantic contracts and an adversarial corpus:** author a small schema/corpus first. Review unit conventions, generation semantics, missingness, provenance and policy authority.
3. **Construct a tiny multi-service toy network:** include at least power, water and a critical-service load, with explicit stocks/flows/capacities, one dependency chain and a compound outage. Label every parameter synthetic. This is a semantics test, not calibrated physics.
4. **Add profile and distributional tests:** heat/water scarcity, flood/power interruption, disability/access constraints, uneven service coverage, profile staleness, conflicting preferences and missing affected-group evidence.
5. **Add independent reference adapters:** only for domains with versioned tools/data, matching assumptions and explicit validation plans. Candidate domains include electrical distribution, building energy/thermal comfort, urban drainage/flooding, water distribution, transport and public-health exposure.
6. **Run adaptive-pathway experiments:** separate hazard uncertainty, demand change, cost, observation quality, cross-sector dependencies and changes to authorized criteria. A planner may propose a change but cannot silently rewrite a protected floor.
7. **Create a space-specific profile adapter:** begin with resource-accounting and fault-injection semantics; integrate qualified life-support/power/thermal models only under independent safety authority.
8. **Only then consider decision support for a real place:** require data governance, community authorization, operational ownership, fail-safe fallback and forecast-versus-observation review.

## Reference research

- Azadi et al., *What Have Urban Digital Twins Contributed to Urban Planning and Decision Making? From a Systematic Literature Review Toward a Socio-Technical Research and Development Agenda*, Smart Cities 8(1), 2025. https://doi.org/10.3390/smartcities8010032
- *Beyond connected digital twins – Can digital twins really deliver sustainable cities?*, Sustainable Cities and Society 131 (2025), 106596. https://doi.org/10.1016/j.scs.2025.106596
- UN-Habitat, *Spatial Inclusion in Cities: Regional Trends Report* (2025). https://unhabitat.org/spatial-inclusion-in-cities-regional-trends-report
- UN-Habitat, *Urban Regeneration for Climate Resilience* (2025). https://unhabitat.org/urban-regeneration-for-climate-resilience
- NASA Moon to Mars Architecture – Components. https://www.nasa.gov/moontomarsarchitecture-components/
- NASA Moon to Mars Architecture – Strategy and Objectives. https://www.nasa.gov/moontomarsarchitecture-strategyandobjectives/
- NASA Environmental Control and Life Support Systems (ECLSS). https://www.nasa.gov/reference/environmental-control-and-life-support-systems-eclss/
- *A multi-dimensional framework for adaptive planning under deep uncertainty and interacting risks*, Environmental Science & Policy 179 (2026), 104382. https://doi.org/10.1016/j.envsci.2026.104382
- *Dynamic Adaptive Pathways Planning with Changing Criteria for Cross-Sectoral Climate Adaptation*, Procedia Computer Science 288 (2026), 11–20. https://doi.org/10.1016/j.procs.2026.08.180

## Claim ceiling

This agenda specifies proposed contracts, scenarios and measurements. It does not establish that any policy works, that a real city is safe/equitable, that a habitat is viable, or that the shared layer is implemented. Synthetic replay establishes repeatability of a declared computation; hashes establish integrity linkage, not source truth, authorization, independent validation or safety.
