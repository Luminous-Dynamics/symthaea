# TWO-WHEELER-ENG-001 — Co-designing bicycles, e-bikes, motorcycles and microfactories

**Status:** Research/benchmark contract plus initial Rust verifier source, v0.1 — proposed targets only  
**Tracker:** https://github.com/Luminous-Dynamics/symthaea/issues/8044  
**Review date:** 2026-10-18  
**Evidence status:** Literature/standards review, architecture proposal, frozen expected-output fixtures and an authored Rust verifier. The verifier has not been compiled or executed in this environment. No vehicle simulation, prototype test, production run, or CI result is claimed.

## Executive thesis

The paradigm shift is to optimize for **safe, useful mobility delivered over the service life**, not a single vehicle specification. The vehicle, factory, repair workflow, regional parts supply, and end-of-life path should be treated as one coupled design problem—but remain separately measurable and independently qualified.

A lightweight bicycle is the first anchor because it lets us exercise the requirements, geometry, materials, manufacturing, repair and evidence workflow with lower energy and consequence than a powered vehicle. The programme then expands to cargo/utility bicycles, e-bikes, and finally a motorcycle profile. This sequencing is a risk-reduction choice, not a claim that bicycle results transfer directly to motorcycles.

## 1. Existing owners: compose, do not duplicate

| Concern | Canonical owner to reuse | Boundary |
|---|---|---|
| Shared transport semantics | Symthaea [#6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244) | Mission/configuration/evidence joins; no universal vehicle physics |
| Road/off-road dynamics | Symthaea [#6247](https://github.com/Luminous-Dynamics/symthaea/issues/6247) | Tire/contact, traction, steering, braking, grade, payload/CG and bounded qualification profiles |
| Mobility hardware manufacturing | Symthaea [#5985](https://github.com/Luminous-Dynamics/symthaea/issues/5985) | As-built, component/process, functional and maintenance evidence |
| Physical qualification ladder | Symthaea [#6254](https://github.com/Luminous-Dynamics/symthaea/issues/6254) | Simulation, bench, controlled physical trials and outside authority remain distinct |
| Factory scheduling | Symthaea [#7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677) / [PR #7678](https://github.com/Luminous-Dynamics/symthaea/pull/7678) | Reuse the capability-constrained operations research scheduler; do not create a second scheduler |
| Personal/family vehicle and ownership-cost study | Symthaea [PR #7751](https://github.com/Luminous-Dynamics/symthaea/pull/7751) | Reuse repairability, lifetime-cost and evidence-envelope direction |
| Whole-life simulation and repair graph | Symtropy [PR #1762](https://github.com/Luminous-Dynamics/symtropy/pull/1762), [#1601](https://github.com/Luminous-Dynamics/symtropy/issues/1601), [#1773](https://github.com/Luminous-Dynamics/symtropy/issues/1773), [#1780](https://github.com/Luminous-Dynamics/symtropy/issues/1780) | Scenario simulation, provenance, serviceability, supplier resilience and Pareto comparisons |
| Asset dossier and maintenance events | Mycelix [PR #5131](https://github.com/Luminous-Dynamics/mycelix/pull/5131) / [#5132](https://github.com/Luminous-Dynamics/mycelix/issues/5132) | Operational history and actor-attributed events; does not certify a design |

Before adding a Rust type or solver, record the exact owner/path and identify a reproducible missing contract. A specialization is justified only by a mode-specific requirement that the shared profile cannot express without ambiguity.

## 2. Paradigm shifts worth testing

### 2.1 From product optimization to mobility-service optimization

Do not ask only whether a candidate is lighter, faster or cheaper at purchase. Evaluate equivalent mission utility and completed kilometres across a declared duty cycle, accounting for downtime, repairs, energy, consumables, parts availability and replacement. A bike that is inexpensive but often immobilized may provide less real value than a slightly more expensive, repairable one.

### 2.2 From one “best design” to a constrained Pareto portfolio

Keep legal/safety requirements and minimum mission capability as hard constraints. Among eligible candidates, expose the trade-offs among mass, durability, fit, useful payload, rider effort, energy/fuel, lifecycle cost, repair time, production yield, supply resilience and environmental burden. Do not let one weighted score trade away a failed braking or structural requirement.

### 2.3 Co-design the vehicle and its production route

Every candidate geometry/material must be evaluated with feasible operations, tooling, tolerances, operator skills, inspection access, scrap/rework, assembly sequence and maintenance access. A design that cannot be inspected or reproducibly made at the proposed volume is not a winning design. Use the existing factory scheduler for feasible routes, then a separate cost/yield model for economics; a greedy scheduler finding no route is not proof that every route is infeasible.

### 2.4 Engineer for repair, inspection and continued service

Standardize interfaces where they genuinely improve supplier choice and repair, not merely to maximize part reuse. Design access for normal service tasks; give replaceable items unambiguous revisions; and require post-repair verification. Keep the design, the written task, the performed repair, the measured post-repair state and independent approval as different evidence.

### 2.5 Treat model uncertainty as an input to design

Use validated physics models and sensitivity analysis to screen candidates, then select physical experiments that discriminate between the leading hypotheses. Publish model residuals by observable and operating envelope. Do not accept a simulation-only safety claim or let generated training/evaluation data leak between discovery and holdout sets.

### 2.6 Replace “one global product” with a common core plus regional variants

Share engineering contracts and service interfaces, but vary tires, corrosion protection, dust/water sealing, gearing, payload, lighting, powertrain and service plans as supported by regional evidence. Evaluate road condition, climate, energy availability, repair skills, lead times, affordability and local rules. “Locally assembled” must not be reported as “locally renewable” if critical tooling, parts or materials still have single external dependencies.

### 2.7 Treat the factory as a reconfigurable capability network

Compare conventional batch production with modular workcells, shared fixtures, flexible assembly, distributed fabrication, and supplier-supported remanufacture. Do not assume automation is always cheaper: test yield-adjusted costs at 1, 10, 25, 100 and 1,000 units with setup, inspection, training, capital recovery, downtime, scrap and rework included.

## 3. Product-family sequence

| Stage | Profile | Main questions | Promotion gate |
|---|---|---|---|
| A0 | Conventional commuter/utility bicycle | Fit, structure, fatigue, rolling resistance, braking, service access and baseline cost | Requirements and applicable safety/test plan reviewed |
| A1 | Cargo/utility bicycle | Payload/CG, wheel/brake/bearing life, loaded handling, load attachments and difficult-road duty | Profile-specific load and stability cases; do not inherit an out-of-scope standard silently |
| A2 | Pedal-assist e-bike | A0/A1 plus battery, BMS, drive, wiring, charger, thermal/environment and software configuration | Separate electrical-system and mechanical gates, plus jurisdiction-specific classification |
| A3 | Commuter/utility motorcycle | Motorcycle dynamics, braking, tires, engine/e-drive, heat, serviceability and E/E hazards | Mode-specific system safety, physical validation, type-approval and legal gates |
| A4 | Factory/product-line scale-up | Quality, throughput, supply resilience, costs and controlled variant introduction | Observed pilot batches and independent quality records |

The first software benchmark should cover A0 and use a synthetic, parameterized duty cycle clearly labelled as synthetic. Do not invent real-world routes, rider distributions, prices, manufacturing yields or life expectancy. Select real comparator variants and freeze their sources before publishing a market comparison.

## 4. Benchmark scorecard

All quantitative improvements below are **provisional research targets**, not achieved results, predictions, accepted industry standards or safety thresholds. Freeze and, where needed, revise them after a documented baseline campaign. The benchmark must retain negative results and uncertainty.

| Dimension | Measurement protocol | Proposed target or invariant |
|---|---|---|
| Evidence integrity | Independent verifier checks exact configuration, producer/model revision, input/output digests, evidence class, units, validity scope, uncertainty and source rights | Zero false accepts across the frozen negative-control suite |
| Structural/fatigue | Apply relevant standard-derived load cases to the correct bicycle class; compare simulation and physical fixture evidence separately | No candidate qualifies structurally from optimization output alone; any required test failure blocks promotion |
| Braking/handling | Declare tire generation, surface/friction, wet/dry state, load/CG, grade, temperature, instrumentation and operator protocol | No unsupported stopping-distance, stability or roadworthiness claims; quantify uncertainty and repeatability |
| Design-space efficiency | Frozen seed and parameter bounds; compare search method against random/simple baseline under equal solver budget | Report valid designs found, constraint-violation rate, Pareto coverage, solver cost and reproducibility; no universal target until baseline exists |
| Fit and human utility | Fit/adjustment range, contact/position measurements and independently reviewed, accessible user trials | Report subgroup coverage and adverse fit outcomes; do not infer comfort or accessibility solely from geometry |
| Energy | Instrument human power and rolling/route conditions for bicycles; separately meter electrical or fuel consumption for powered profiles | Report Wh/km or fuel/km plus rider work where relevant; compare only matched duty cycles and payloads |
| Repairability | Freeze at least 10 common tasks; capture diagnosis, hands-on time, tools, parts waits, verification, rework, elapsed downtime; aim for 3 technicians × 3 trials/task where feasible | Target at least 25% reduction in median and P90 task times against a matched, safe baseline, verified in workshop trials |
| Factory economics | Time studies and cost ledger for setup, material, labor, tooling/capital, utilities, yield, rework, inspection, packing, warranty and delivery | Explore 1/10/25/100/1,000-unit batches; candidate target ≥30% lower assembly touch labor at 25+ units without lowering acceptance quality |
| Quality | Track first-pass yield, defect escape, rework and traceability by variant and process revision | Candidate stable-pilot target ≥95% first-pass yield; critical defects still block acceptance at any yield |
| Lifetime economics | Compare equivalent service missions and ownership horizons; show purchase, consumables, repair, downtime and residual/recovery assumptions separately | Investigate ≥15% lower modeled lifecycle cost per completed 1,000 km as a stretch hypothesis; require quote- or measurement-backed inputs before claiming advantage |
| Supply resilience | Seed critical supplier, tool, energy, fuel and parts outages; calculate feasible substitutes and resulting lost service | Unknown dependencies stay unknown; no local-production claim from assembly location alone |
| Environmental burden | Apply a stated ISO 14040-aligned goal/scope, functional unit, boundaries and sensitivity analysis | Publish lifecycle assumptions and sensitivity; do not use “green” labels from material mass alone |
| Model validity | Hold out independent physical runs; compare residuals by load, surface, speed, variant and environment | A ≤10% error goal may be set only for individually named observables after data are adequate; no aggregate error figure |
| Factory scheduler | Validate emitted schedules independently for capability, precedence, availability, no overlap, required inspection and resource validity | Zero accepted invalid schedules in adversarial controls; scheduler feasibility is not physical process qualification |

Do not set numeric goals for stopping distance, crash protection, frame life, range or road legality inside this generic benchmark. Those depend on exact vehicle class, configuration, mission, standards, measurement method and external authorities.

## 5. Initial case matrix and adversarial controls

Freeze stable scenario IDs before evaluating design candidates. Include positive controls and deliberately invalid cases:

1. Nominal commuter-bicycle configuration with complete declared inputs.
2. Frame geometry changes but material/fatigue evidence remains tied to the old configuration.
3. A tire or brake generation changes without invalidating old test receipts.
4. A one-time brake stop exists but required repeatability, wet-condition or degradation evidence is absent.
5. The model returns a value outside its declared operating envelope and the result is still labelled qualified.
6. Missing material properties, calibration, price, tool capability or parts lead time are silently imputed as zero.
7. A high-payload candidate omits changed centre of gravity or the relevant load case.
8. An e-bike battery/BMS/charger revision changes while the previous electrical evidence is reused.
9. A repair task is marked complete without post-repair verification or with the wrong component revision.
10. An operation is assigned to an unavailable/unqualified worker or tool, or overlaps another operation's reservation.
11. The factory schedule omits a required inspection to improve makespan.
12. Scrap, rework, warranty, setup or equipment amortization is omitted from a candidate's cost while the baseline includes it.
13. A supplier outage invalidates the only critical replacement part and the model still claims uninterrupted service.
14. Local frame fabrication is presented as fully local production despite imported critical bearings, brake components, cells or control electronics.
15. Candidate A wins a scalar score despite violating a hard safety/legal constraint.
16. A run digest or result changes while the recorded configuration and input receipt remain unchanged.
17. Synthetic or simulated evidence is upgraded to physical/field/regulatory evidence.
18. One successful prototype or one repair is promoted into a fleet reliability claim.

The independent verifier must recompute decisions from the input fixture and policy. A digest binds bytes to an identity; it does not establish source authenticity or physical truth.

## 6. Evidence schema and experiment procedure

Every run should bind, either directly or by immutable reference:

- scenario ID and schema/version;
- vehicle category, exact configuration and BOM revision;
- user/mission profile, payload, duty cycle and environment;
- tire/brake/powertrain/battery generations as applicable;
- model, solver and source revision, units/frames, random seeds and parameter bounds;
- manufacturing route, machine/tool/worker capabilities, production volume and cost assumptions;
- repair-task graph and replaced component revisions where applicable;
- raw source/rights, provenance, input/output digests and evidence class;
- missingness reason, quantified uncertainty, applicability envelope and claim ceiling;
- independent-verification outcome, expected failure reason for negative controls, and raw outputs.

Evidence classes must remain explicit: `Source`, `Synthetic`, `Simulated`, `WorkshopMeasured`, `InstrumentedPhysical`, `FieldObserved`, and `RegulatoryReceipt`. These labels are descriptive, not a single ranking; evidence is applicable only to the configuration, observable and scope it actually supports.

The initial independent-decision prototype is wired into `xtask`. From the workspace root, the authored commands are:

```sh
cargo test -p xtask twowheeler_benchmark
cargo run -p xtask -- two-wheeler-benchmark \
  --input docs/research/data/two_wheeler_a0_benchmark_v0_1.json \
  --out reports/two_wheeler_a0_verification.json
```

These commands have **not** been run in the current environment. A verifier pass would establish agreement with these synthetic decision rules only, not physical engineering validity.

Suggested execution loop:

1. Freeze mission requirements and comparable baseline variants.
2. Audit owners and bind the applicable external requirements for the target market.
3. Generate candidates within physical, manufacturing and service constraints.
4. Use validated solvers to screen; preserve raw failures and out-of-envelope cases.
5. Use multi-objective sensitivity/robustness analysis; select discriminating physical experiments rather than only the most attractive model prediction.
6. Build low-energy coupons/fixtures and one serviceable bicycle demonstrator only after the required design and test review.
7. Measure and compare against the frozen baseline; use held-out measurements for model validation.
8. Pilot the factory cell in small batches; record actual labor, yield, scrap, rework and downtime.
9. Promote to e-bike and motorcycle profiles only with their separate safety, electrical, regulatory and physical evidence.
10. Publish uncertainty, failed cases and remaining gaps, not just the winning candidate.

## 7. Global and regional applicability

Maintain a shared core contract and explicit regional profiles. At minimum, make these variables first-class: rider sizing/accessibility, payload, surface and weather, dust/corrosion/water exposure, road condition, annual distance, energy/fuel availability, charging, repair skills/tools, parts lead time, taxes/duties, local suppliers, end-of-life infrastructure, and regulatory evidence.

South Africa is a useful initial context, but the source/date of any price, parts listing, regulatory interpretation or policy assumption must be recorded. A catalog listing is not a supplier quote, an advertised starting price is not a delivered build cost, and local assembly is not proof of local manufacturing eligibility or economic closure. Each additional region requires its own source and applicability review.

## 8. Standards and research references

Check the official catalogue and relevant national adoption before every compliance decision. The sources below establish scope/reference points; they do not mean the present programme is compliant.

- [ISO 4210-2:2023 — bicycle safety/performance requirements for specified bicycle classes](https://www.iso.org/standard/78077.html). Applicability is class-specific; delivery/cargo, recumbent, tandem and other specialized types may fall outside its scope.
- [ISO 4210-6:2023 — frame and fork test methods](https://committee.iso.org/standard/78081.html).
- [ASTM F2711-19 — bicycle frame structural-performance test methods](https://store.astm.org/standards/f2711).
- [ISO 8710:2010 — motorcycle brake tests and measurement methods](https://www.iso.org/standard/50875.html); ISO says this edition was confirmed in 2021.
- [ISO 26262-12:2018 — functional safety adaptation for motorcycles](https://www.iso.org/standard/69605.html) and [ISO/DIS 26262-12, Edition 2](https://www.iso.org/standard/90031.html). At this review date the new DIS is in enquiry; recheck status before design freeze.
- [UL 2849 — electrical systems for e-bikes](https://www.shopulstandards.com/ProductDetail.aspx?ShowFreeviewModal=1&productId=UL2849_1_S_20200102); applies to the described e-bike electrical system and is not a global substitute for applicable law or mechanical standards.
- [ISO 14040:2006 — lifecycle-assessment principles and framework](https://www.iso.org/standard/37456.html), reviewed and confirmed current by ISO in 2022.
- [NIST Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing): emphasizes trustworthy model validation and quantified uncertainty.
- [NIST, Digital Twins Workshops Summary Report, published 2026-07-21](https://www.nist.gov/publications/digital-twins-workshops-summary-report): identifies interoperability, verification/validation/uncertainty, cybersecurity and workforce-readiness as ongoing manufacturing digital-twin challenges.

## 9. Claim ceiling and next implementation slice

This document specifies a benchmark programme; it does not provide a finished bicycle/motorcycle design or demonstrate improvement. A benchmark pass may prove only the exact software contract and frozen synthetic scenarios that were executed. It does not prove structural strength, braking performance, user fit, physical durability, factory throughput, commercial economics, environmental superiority, roadworthiness, type approval, or permission to operate.

**Current wheel-demand tranche:** [wheel-demand benchmark protocol](TWO_WHEELER_WHEEL_DEMAND_BENCHMARK_2026-10-18.md), [13-case pointwise oracle](data/two_wheeler_wheel_demand_v0_1.json), [6-case signed-work trace fixture](data/two_wheeler_trace_energy_v0_1.json), and [published South African validation summary](data/published_sa_electric_motorcycle_validation_summary_v0_1.json). The 2026 South African full-powertrain twin is treated as an existing baseline; we are not claiming to reproduce it without raw telemetry and component maps.

**Initial fixtures and verifier sources:** [`two_wheeler_a0_benchmark_v0_1.json`](data/two_wheeler_a0_benchmark_v0_1.json) defines the bicycle-side evidence controls; [the powertrain frontier protocol](TWO_WHEELER_POWERTRAIN_FRONTIER_2026-10-18.md) and [candidate register](data/two_wheeler_powertrain_candidates_v0_1.json) add electric, hybrid, plasma-assisted combustion, space plasma and Spark Engine/LCF distinctions. The Rust `xtask two-wheeler-benchmark` and `xtask powertrain-energy-ledger` commands recompute declared synthetic decision rules. The powertrain ledger fixture contains one positive and eight negative controls. These sources are present but **have not been compiled or executed here**. Next: run exact-head Cargo tests, review every mismatch, add malformed-input/mutation controls, and bind to existing road-dynamics, repairability, manufacturing and factory-scheduler owners before expanding design search or adding physics solvers.
