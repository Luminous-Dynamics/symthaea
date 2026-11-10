# NUCL-TEA-001 — Reproducible techno-economic comparison for port energy and marine nuclear systems

**Status:** research specification / synthetic-reference design  
**Date frozen:** 2026-11-10  
**Parent:** [Symthaea issue #11343](https://github.com/Luminous-Dynamics/symthaea/issues/11343)  
**Related work:** [Port nuclear readiness #11344](https://github.com/Luminous-Dynamics/symthaea/issues/11344) · [Nuclear passenger-vessel feasibility #11345](https://github.com/Luminous-Dynamics/symthaea/issues/11345) · [Civil SMR facility integration #6135](https://github.com/Luminous-Dynamics/symthaea/issues/6135) · [Cargo/floating nuclear feasibility #7445](https://github.com/Luminous-Dynamics/symthaea/issues/7445) · [SMR campus #6148](https://github.com/Luminous-Dynamics/symthaea/issues/6148)

This document specifies a transparent decision-support model. It is **not** a reactor design, safety analysis, naval architecture approval, site assessment, regulatory submission, cost estimate for a real project, or investment recommendation.

## 1. Strategic thesis

The comparison unit is the **delivered service over the complete asset lifecycle**, not the reactor's headline cost per kilowatt.

For a port-city, the service bundle can include electricity, dependable capacity, shore power, useful heat, water, cargo throughput and resilience. For vessels, it is transport service delivered over a route and schedule: tonne-nautical-miles for cargo or passenger-days for passenger ships.

The model must answer both:
- When does an architecture outperform a credible alternative under a declared set of conditions?
- What missing evidence or failed condition makes the architecture infeasible, uneconomic or not comparable?

It must be allowed to return `GO`, `CONDITIONAL`, `NO-GO`, `UNKNOWN` or `NOT_COMPARABLE`. A weighted score must never convert a failed legal, safety, access, insurance or emergency-readiness gate into a viable project.

## 2. Competitive landscape and honest comparison

### 2.1 External reference points

| Reference | What it contributes | What we must not infer |
|---|---|---|
| Lloyd's Register / LucidCatalyst, *Power Shift: Nuclear Power Pathways for Future Containership Fleets* (14 July 2026) | A quantitative, scenario-specific nuclear containership case, including the assumptions behind speed, annual voyages, cargo capacity and fuel-space effects. | Its modeled maximum gains are not achieved fleet performance or universal cost competitiveness. Reproduce the case and stress its assumptions. |
| Lloyd's Register / Port of Rotterdam / CORE POWER / Maersk, port-call feasibility study (11 June 2026) | A real port-call feasibility and regulatory-readiness investigation. | A desktop feasibility study is not a port approval, nuclear licence or final safety case. |
| Lloyd's Register / British Ports Association, *Advanced Nuclear Technology Use Within Port Environments* (22 October 2026) | Port-energy opportunities and a staged readiness concept; stakeholder, energy-demand, grid, workforce and authority questions. | A readiness label is not an authorization or evidence of site suitability. |
| IMO Nuclear Code / SOLAS Chapter VIII work | The international maritime regulatory-revision trajectory, with a workplan targeting 2030 milestones. | Future workplan milestones are not requirements already adopted or in force. |
| OECD NEA / EPRI, *The Costs of Generating Electricity 2025* | A comparison framework for electricity-generation economics, including SMR cost questions and system context. | A single generic LCOE does not decide a country- or port-specific system investment. |
| IUMI nuclear-propulsion policy update (7 May 2026) | The insurance and liability gap that must be examined for commercial maritime nuclear applications. | Conventional marine insurance can simply be assumed to cover nuclear risk. |
| 2026 peer-reviewed containership propulsion economics study | A published scenario-based levelized-cost-of-transport comparison. | A favorable result under selected speed and capital-cost assumptions is not a general result for all routes or vessels. |

Primary sources:
- [Lloyd's Register — Power Shift](https://www.lr.org/en/knowledge/case-studies/power-shift-nuclear-power-pathways-for-future-containership-fleets/)
- [Lloyd's Register — Rotterdam port-call feasibility](https://www.lr.org/en/knowledge/press-room/press-listing/press-release/2026/rotterdam-study-sets-out-pathway-for-nuclear-powered-commercial-ship-port-calls/)
- [Lloyd's Register / British Ports Association — Advanced nuclear technology in ports](https://www.lr.org/en/knowledge/press-room/press-listing/press-release/2026/ports-encouraged-to-prepare-now-for-advanced-nuclear-in-maritime-and-clean-energy/)
- [IMO — nuclear ship regulatory work](https://www.imo.org/en/mediacentre/pages/whatsnew-2419.aspx)
- [OECD NEA — Costs of Generating Electricity 2025](https://www.oecd-nea.org/jcms/pl_121713/the-costs-of-generating-electricity-2025)
- [IUMI — Nuclear propulsion](https://iumi.com/policy/nuclear-propulsion/)
- [2026 containership propulsion economics study](https://www.sciencedirect.com/science/article/pii/S0149197026000776)

### 2.2 Industry software baselines

The engineering assurance fabric must be compared against the relevant capabilities of established tools—not against a caricature of them.

- **BIM/CAD/CDE and infrastructure twins:** Autodesk Revit/Forma, Bentley iTwin, Rhino/Grasshopper and openBIM IFC/IDS workflows.
- **Plant/product/factory lifecycle and digital manufacturing:** Siemens' digital manufacturing and lifecycle portfolio.
- **Physics and multiphysics:** established specialist solvers in thermal-fluid, structural, materials, electrical-grid, naval-hydrodynamic and control domains.
- **Measurement and quality:** QIF-based metrology, MTConnect, SPC/MSA and established quality-management systems.
- **Operational integration and ontology:** platforms such as Palantir Foundry/Ontology and existing industrial integration systems.
- **Nuclear engineering and assurance:** supplier-owned licensed design bases, qualified engineering analysis tools, regulator-defined requirements and independent verification organizations.

The comparison must use the actual version, modules, connectors, configuration, data rights and export scope tested. Inaccessible proprietary behavior is `UNASSESSED`, not a competitor failure.

The relevant Luminous competitive crosswalks already exist:
- [Sol-Atlas matched assurance benchmark #502](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/502)
- [Sol-Atlas AEC design-to-operations industry crosswalk #600](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/600)
- [Symtropy composed manufacturing twin VVUQ #3121](https://github.com/Luminous-Dynamics/symtropy/issues/3121)

This programme consumes those comparison protocols rather than creating a parallel generic vendor scorecard.

## 3. Proposed Luminous division of engineering responsibility

| System | Proper role in this programme | Explicit non-role |
|---|---|---|
| **Symthaea** | Requirements decomposition, engineering knowledge, scenario orchestration, change-impact reasoning, assumptions and evidence comparison, uncertainty communication, decision-support proposals. | Not a nuclear regulator, licensed operator, safety authority or autonomous plant operator. |
| **Symtropy** | Deterministic synthetic execution, physical rehearsal within declared model scope, event/replay evidence, counterfactuals, numerical-integrity reporting. | Not a neutronics, thermal-hydraulics, CFD, FEM, grid, naval-hydrodynamics or nuclear-safety solver by virtue of being a physics engine. |
| **Sol-Atlas** | Spatial/geometric context, site and port layers, vessel/plant envelopes, reference frames, corridor/access reasoning, design-to-as-built comparison and spatial provenance. | Not a substitute for certified CAD, BIM, geotechnical, structural, survey or hydrographic authority. |
| **Mycelix** | Cross-organization provenance, supplier/component records, inspection and custody evidence, revision lineage, credentials and selective disclosure. | Not the nuclear operator, safety control system, safeguards authority, emergency command or source-of-truth for external approvals. |

### 3.1 Composed digital-thread boundary

Use a typed, immutable evidence chain:

```
need / service requirement
  -> design option + configuration
  -> model and context-of-use identity
  -> simulation / calculation proposal
  -> independent model and interface qualification
  -> physical measurement / inspection
  -> manufacturing and installation record
  -> commissioning evidence
  -> authorized operation / maintenance record
  -> change, requalification and decommissioning
```

Preserve these distinct states:

```
as-designed != as-manufactured != as-installed
             != as-tested != as-operated != inferred != synthetic
```

No simulation output can silently become a physical observation. No AI-generated proposal can become an approved design change. No component-level qualification automatically qualifies the whole system or its interfaces.

### 3.2 Multi-fidelity rather than one universal simulator

Use specialist tools where they are strongest. Compose them through explicit adapters and contracts, for example FMI/SSP where appropriate, while preserving:
- exact model and solver identity/version;
- declared context of use and validity domain;
- units, coordinate frames, clocks and coupling direction;
- boundary conditions and assumptions;
- numerical and measurement uncertainty;
- known model-form discrepancy;
- evidence for calibration and validation;
- unsupported fields and lossy transformations.

A faster or deterministic model is useful for screening and rehearsal; it is not automatically a more credible physical model.

## 4. Reference-case portfolio

All first-pass cases are synthetic and openly labelled. They are intentionally not actual facility designs or vessel performance claims.

### 4.1 Port-city energy system

Compare the same service demand under:
A. Existing grid and existing port systems.
B. Grid + renewables + storage + flexibility.
C. Grid + renewables/storage + firm generation options.
D. Shore-based SMR outside the port, supplying the port/industrial system through an engineered grid connection.
E. Floating power unit supplying an industrial/port network.
F. Non-nuclear resilience and generation alternatives with comparable service boundaries.

Demand domains remain separate:
- port and city electrical demand;
- shore-power load and berth schedule;
- crane, yard and cargo-handling demand;
- cold-chain and warehouse load;
- water demand, treatment and wastewater;
- industrial process heat by temperature/quality class;
- hydrogen/e-fuels only when a real customer, supply chain and lifecycle boundary are present.

The model must capture hourly/seasonal profiles, peaks, ramps, reserve, grid constraints, backup, storage duration, losses and unserved service. Annual energy balance alone is not sufficient.

### 4.2 Cargo vessel

Compare a common route and cargo-service requirement under:
A. Conventional reference vessel and fuel.
B. Battery/shore-power or hybrid option where the route and energy density make it physically credible.
C. Alternative fuels (e.g. methanol or LNG) only with evidenced fuel availability and lifecycle assumptions.
D. Nuclear-electric vessel using an **externally defined, licensed supplier reactor-module interface**.

Required outputs include annual delivered cargo capacity, route/service speed, voyage frequency, utilization, cargo mass/volume displaced, port network reach, refuelling/drydock/maintenance outage, whole-life cost and lifecycle GHG.

Reproduce the assumptions behind the LR containership study before testing variants. Do not hard-code its modeled speed or cargo gains as defaults.

### 4.3 Passenger / cruise vessel

Use the separate [NUCL-MAR-PAX-001 #11345](https://github.com/Luminous-Dynamics/symthaea/issues/11345) profile. Compare small expedition, mid-size cruise and large resort-style service archetypes. Distinguish propulsion energy from hotel loads, occupancy, passenger-days, itinerary and destination-port access, water, waste, staffing, evacuation interfaces, passenger/crew protection requirements, insurance and servicing/drydock.

A nuclear vessel that cannot visit a commercially viable set of destination ports is not a commercially viable route, regardless of theoretical propulsion performance.

### 4.4 Floating power unit

Compare:
A. Shore-based supply through grid/subsea infrastructure.
B. Floating nuclear power unit with shore connection.
C. Non-nuclear floating generation/storage alternatives.

Model station-keeping and marine structural requirements only through qualified specialist inputs. Keep mooring, grid cable, coastal hazard, heat rejection, service access, emergency support and decommissioning as distinct interfaces.

## 5. Economic calculation contract

### 5.1 Discounting convention

Every case declares:
- real or nominal prices;
- real or nominal discount rate;
- currency and base year;
- inflation escalation method;
- FX source/date when conversion is needed;
- analysis horizon and asset-life assumptions;
- residual-value and decommissioning treatment.

Do not mix nominal cash flows with a real discount rate, or values expressed in different currency years without an explicit conversion.

For a constant real discount rate (r), the present value of cash flow (C_t) is:

```
PV(C) = sum[t = 0..N] C_t / (1 + r)^t
```

### 5.2 Electricity and energy services

For a defined plant/system boundary:

```
LCOE = PV(all included lifecycle generation costs)
       / PV(net electricity delivered)
```

The numerator must disclose included capital, financing, fixed/variable O&M, fuel and fuel-cycle, grid connection, backup/reserve, waste and decommissioning costs. The denominator is net electricity delivered at the declared delivery point, not gross reactor output.

Report both:
- plant/generator-level metrics where comparable;
- system-level delivered service costs, including transmission, storage, reserve and reliability requirements.

Never compare plant-gate generation cost for one option with delivered customer cost for another.

### 5.3 Multi-output port systems

Calculate the whole-system NPV first. Report electricity, water and useful heat as separate service outputs, each with:
- quantity, quality and delivery point;
- time profile and availability;
- conversion, pumping, treatment and distribution costs;
- customer or offtake evidence;
- uncertainty and failure/fallback assumptions.

Do not add full electricity, heat and water benefits as if the same energy were independently created three times. If costs must be allocated to coproducts, publish the allocation rule (e.g. economic or exergy allocation), show the unallocated system result, and provide a sensitivity analysis.

Water cost is per cubic metre at a stated quality and delivery point and includes treatment, intake, pumping, brine/waste handling and distribution where applicable. Heat cost is per useful MWh-thermal by temperature/quality class at a stated customer boundary.

### 5.4 Cargo and passenger transport service

For cargo:

```
Levelized cost of transport =
  PV(all vessel and route lifecycle costs)
  / PV(delivered tonne-nautical-miles)
```

For passenger shipping:

```
Cost per passenger-day =
  PV(all vessel and route lifecycle costs)
  / PV(delivered passenger-days)
```

The numerator includes vessel CAPEX and financing, operating crew, fuel/fuel-cycle, maintenance, insurance/liability, port charges, emergency readiness, inspections, drydock, refuelling/servicing outages, waste/decommissioning and any route-specific costs. The denominator uses delivered service, not nominal capacity.

Report cost per voyage and annual service capacity alongside the levelized metric. If cargo or passenger space is displaced, model the opportunity cost explicitly.

### 5.5 Economic outputs

Do not compress the answer into a single weighted number. Report:
- NPV / lifecycle cost;
- LCOE and delivered service cost where applicable;
- cost per m³ of water at declared quality;
- cost per useful MWh-thermal by temperature class;
- cargo cost per tonne-nautical-mile;
- passenger cost per passenger-day;
- annual throughput / passenger capacity;
- availability and service continuity;
- unserved energy/service under declared scenarios;
- operating and lifecycle greenhouse-gas emissions;
- land/sea footprint and material/resource requirements where evidence exists.

Show Pareto fronts and break-even surfaces. State which outputs are incomparable rather than inventing conversion factors.

## 6. Hard feasibility gates before economic ranking

The following are categorical gates. They are not small penalties in a weighted score:

1. Applicable legal and regulatory pathway identified.
2. Responsible licensee/operator and authority boundaries identified.
3. Site and external-hazard evidence available for the intended stage.
4. Port admission and coastal/flag-state pathway credible for every required route segment.
5. Classification-society pathway identified.
6. Insurance and nuclear-liability route credible.
7. Emergency preparedness and competent response pathway defined.
8. Fuel-cycle, servicing, waste and decommissioning arrangements identified.
9. Qualified workforce, training, inspection and maintenance access identified.
10. Grid, cooling, water, land/sea footprint and connection arrangements plausible.
11. Community and workforce engagement is substantive, accessible and traceable.

A missing item is `UNKNOWN`, never favorable by default. A failed mandatory gate makes that case infeasible under the current scenario, irrespective of its computed NPV.

Readiness is a vector by application: for example a port can be more ready for shore-power demand response than for nuclear ship calls, or more ready for an off-site energy supply than a floating plant. Do not issue a single green badge for the whole port.

## 7. Evidence and input-quality manifest

Every decision-material input must carry:

| Field | Meaning |
|---|---|
| `input_id` | Stable identifier; not a mutable display label. |
| `value`, `unit` | Exact value and typed unit. |
| `source_class` | `OBSERVED`, `REGULATORY`, `INDUSTRY_MODEL`, `VENDOR_TARGET`, `PUBLISHED_ESTIMATE`, `ASSUMPTION` or `SYNTHETIC`. |
| `source_uri`, `source_digest` | Source locator and content identity where available. |
| `published_at`, `accessed_at` | Source publication and retrieval dates. |
| `geography`, `context` | Jurisdiction, project/route/system scope and applicability. |
| `currency`, `currency_year` | Required for monetary input. |
| `maturity` | Observed operating data, demonstrated project, vendor design, modeled scenario or research assumption. |
| `uncertainty` | Range/distribution with justification; use categorical unknown where it cannot be quantified. |
| `transformation` | Unit conversion, scaling, normalization or calculation applied. |
| `owner`, `review_status` | Responsible source owner and review disposition. |
| `validity_window` | Period for which the input may be relied upon. |

A cryptographic digest establishes byte identity, not source truth, authority or applicability. A vendor target must remain a vendor target after transformations. A synthetic fixture must never be relabelled as an observation.

## 8. Uncertainty, sensitivity and robustness

Start with transparent deterministic intervals and one-variable / multi-variable sensitivity. Only introduce probabilistic distributions when the source evidence supports them.

At minimum vary:
- overnight CAPEX and owner costs;
- financing rate and capital structure;
- construction duration and delay;
- module production volume / factory learning assumptions;
- plant or vessel availability;
- utilization and service speed;
- fuel, fuel-cycle and energy procurement price;
- maintenance, staffing, inspection and drydock duration;
- insurance/liability and port/site access;
- grid, storage, backup, transmission and water costs;
- carbon-policy assumptions;
- waste, replacement and decommissioning costs.

Report break-even surfaces and robust decision regions. Show when the winner changes. Do not imply that a probability distribution is empirical merely because a Monte Carlo engine can sample it.

For every result identify:
- dominant drivers;
- unresolved evidence;
- variables that reverse the result;
- evidence that would most efficiently reduce uncertainty;
- assumptions shared across options (common-mode dependence);
- downside / regret if the selected option is wrong.

## 9. “Better than SOTA” benchmark protocol

A performance comparison is admissible only after its workload, inputs, outputs, version and budgets are matched.

### 9.1 Benchmark tiers

**Tier A — semantic correctness and assurance**
- Exact product/asset/route/configuration identity.
- Correct units, currency-year and reference frames.
- All material dependencies represented.
- No silent defaults or hidden semantic loss.
- Correct provenance and source maturity.
- Correct hard-gate dispositions.
- Independent arithmetic oracle agreement.

**Tier B — engineering model credibility**
- Compare specialist model output against accepted analytical solutions, measurement campaigns or independently qualified reference data.
- Freeze context of use, validity domain, calibration/validation split and measurement uncertainty.
- Compare per-observable error, bias, coverage, residual structure and uncertainty calibration—not one aggregate “physics accuracy” score.
- A solver completing or replaying deterministically is not evidence of physical accuracy.

**Tier C — lifecycle and manufacturing continuity**
- Design-to-BOM-to-lot-to-process-to-inspection-to-installation-to-commissioning traceability.
- Exact revision and component identity preservation.
- Detection of stale, revoked, missing or contradictory evidence.
- Selective revalidation after a change.
- Complete as-designed/as-built/as-tested/as-operated distinctions.
- Offline/partition reconciliation and access-control behavior for cross-enterprise evidence.

**Tier D — productivity and economics**
- Engineering hours per qualified change.
- Time from source change to affected-evidence identification.
- Cost and duration forecast calibration against actual projects.
- Manufacturing rework/scrap and first-pass yield where a real pilot provides data.
- Schedule variance, outage and maintenance forecast calibration.
- p50/p95/p99 computation and review latency after correctness gates pass.

### 9.2 Initial non-compensable targets

| Measure | Initial target |
|---|---:|
| Decision-material inputs with provenance, units and scope | 100% |
| Silent defaults for missing material inputs | 0 |
| Double-counted energy/heat/water/service benefits | 0 |
| Infeasible hard-gate cases promoted by ranking | 0 |
| False identity merges in exact-identity corpus | 0 |
| Unsupported or stale dependencies silently accepted | 0 |
| Independent arithmetic oracle mismatches on exact fixtures | 0 |
| Non-deterministic output for identical frozen inputs | 0 |
| Synthetic cases counted as physical validation samples | 0 |
| Safety/licensing claims minted by the software | 0 |

These are acceptance targets for software semantics, not evidence that the targets have been met.

### 9.3 Comparative productivity targets

Do not set a percentage improvement over Siemens, Bentley, Ansys, Autodesk, Hexagon or Palantir until a matched trial exists. First freeze:
- exact product/version and licensed modules;
- source datasets, formats and original digests;
- data scope available to every contender;
- hardware, operating environment, compute/network/time budget;
- workflow/operator experience and training;
- output quality rubric and independent gold reference;
- missing/inaccessible functionality as `UNASSESSED`;
- test split, denominator and failure/timeout rules.

After semantic correctness and coverage pass, measure p50/p95/p99 runtime, memory, review hours, setup effort, change-impact latency, data loss and total workflow cost. Report both capability and cost. Never call a different workload a win.

## 10. Hostile / falsification corpus

At minimum exercise:

1. Construction delay plus WACC shock reverses the economic ranking.
2. One option omits fuel-cycle, insurance, waste or decommissioning cost.
3. A vendor target price is relabelled as a demonstrated price.
4. A modeled cargo capacity gain is imported without its speed, voyage and route assumptions.
5. Reactor or shielding weight/volume penalty is omitted from ship capacity.
6. One required route port does not admit a nuclear vessel.
7. Grid import, shore power, reactor generation or storage benefit is double counted.
8. Heat is credited to a customer whose temperature, distance or timing is incompatible.
9. Water output is credited without treatment, brine, pumping or delivery cost.
10. Annual energy matches but peak load, ramp, reserve or reliability fails.
11. Real and nominal currency/discount conventions are mixed.
12. Decommissioning or major replacement is omitted.
13. Critical input missing and treated as zero or a favorable default.
14. Sensitivity interval narrows without additional evidence.
15. Two model components share a boundary condition or data source but are treated as independent.
16. Source or model revision changes while the report is treated as current.
17. A port readiness document count is treated as regulatory authorization.
18. A simulation result is promoted into physical measurement or commissioning evidence.
19. A synthetic fixture is included in an empirical validation denominator.
20. A cost-efficient design fails emergency, legal, port-access or operator-readiness gates.

Each case has a frozen manifest, expected disposition, independent oracle and reproducible result. A case not exercised is `NOT_RUN`, not PASS.

## 11. Initial reference benchmark suite

| ID | Workload | Principal question | Authority / evidence boundary |
|---|---|---|---|
| `PORT-BASE-01` | Synthetic port hourly service demand | Do architecture comparisons preserve peaks, timing and service classes? | Synthetic only. |
| `PORT-ARCH-02` | Grid/renewables/storage vs shore-based supply vs floating supply | Are cost, reliability and constraints compared on the same service boundary? | Synthetic + public source inputs, each labelled. |
| `CARGO-LR-01` | Reproduction of the LR 15,000 TEU case | Can published modeled assumptions/results be independently reconstructed? | Literature reproduction, not real-vessel validation. |
| `CARGO-ROUTE-02` | Same route/service with conventional, alternative fuel, hybrid and nuclear-electric options | Which variables change break-even and feasible route coverage? | No nuclear route approval implied. |
| `PAX-CLASS-01` | Expedition, mid-size and large cruise service profiles | How do hotel loads, passenger-day service and destination access change economics? | Synthetic/reference only until real data is admitted. |
| `FLOAT-POWER-01` | Shore-based supply vs floating unit | Which connection, mooring, service and lifecycle assumptions dominate? | No site/marine certification. |
| `EVIDENCE-HOSTILE-01` | Missing, stale, conflicting, revoked and misclassified inputs | Does the pipeline fail closed and retain contradictions? | Independent semantic oracle. |
| `CHANGE-IMPACT-01` | Supplier interface/design revision | Are only affected downstream claims reopened, and are historical claims retained? | Exact synthetic lifecycle evidence. |

## 12. Staged execution

1. **Freeze the data contract** — units, source class, currency year, context, uncertainty and authority fields.
2. **Freeze reference manifests** — synthetic port, cargo, passenger and floating-power cases.
3. **Implement an independent arithmetic oracle** — separate implementation from the production calculation path.
4. **Run analytical and hostile fixtures** — exact reproducibility, all required dispositions and no favorable defaults.
5. **Reproduce public literature cases** — report reconstruction deltas and unresolved assumptions.
6. **Connect spatial and model projections** — Sol-Atlas spatial identity and Symtropy deterministic rehearsal, without changing canonical physical state.
7. **Connect lifecycle evidence** — Mycelix component/supplier/inspection records through exact source refs and authority scopes.
8. **Run matched industry trials** — use Sol-Atlas #502 and Symtropy #3121 protocols, only when tool access and data rights permit.
9. **Seek external engineering review** — qualified naval architects, power-system engineers, nuclear engineers, port operators, insurers and regulators review the relevant scopes.
10. **Only then consider physical pilot or project engineering** — via licensed, appropriately authorized project partners.

## 13. Explicit no-claims and safety boundary

The work in this repository may establish software contract correctness, synthetic scenario reproducibility, calculation correctness for declared inputs, provenance continuity and bounded integration semantics.

It does **not** establish:
- nuclear reactor or fuel design correctness;
- criticality, radiation protection or nuclear safety;
- structural, geotechnical, CFD, thermal-hydraulic, naval or electrical design qualification;
- site suitability, port acceptance, emergency-plan approval or regulatory compliance;
- insurance availability or public consent;
- physical plant/vessel performance or real-world economic competitiveness;
- permission to manufacture, construct, commission or operate.

Nuclear island, fuel, safety, security, safeguards and operations remain under the competent licensed supplier/operator and relevant authorities. Open engineering work must not disclose sensitive security or safeguards details. No Luminous service, blockchain, AI model, simulation, network, or cloud dependency may become a nuclear safety function by implication.

## 14. Advancement criteria

The first programme milestone is achieved only when:
- the four reference case manifests are frozen and versioned;
- every material input is typed, sourced or explicitly synthetic;
- the independent arithmetic oracle reproduces every analytical fixture exactly;
- all hostile cases have expected dispositions and executable checks;
- hard-gate failures cannot be overridden by weighted cost ranking;
- reports expose dominant assumptions and decision reversals;
- industry comparisons are either matched or explicitly `NOT_COMPARABLE`;
- no unsupported physical, safety, licensing or SOTA claim is made.

**The intended differentiator:** one auditable chain joining service requirements, spatial/engineering configuration, specialist model evidence, manufacturing and inspection provenance, lifecycle economics and decision authority—while preserving the limits of each source and model.
