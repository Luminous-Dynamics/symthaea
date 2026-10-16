# Living City Systems Lab: Symthaea Vertical Slice v0.1

**Status:** Active research specification plus an initial Rust evaluator on the branch research/urban-systems-lab-v0-1. The implementation's tests use synthetic fixture values only. It is not a calibrated city model, a planning approval engine, or evidence that a real city meets any target.

**Home:** Luminous-Dynamics/symthaea. This work intentionally does not target Terra Atlas or Terra Lumina; those are legacy projects for this effort. It does not create another standalone city repo.

## Executive decision

Yes: use Symthaea to help engineer better settlements, but use it as one part of a constrained, evidence-first urban design loop—not as an unconstrained AI urban planner.

The strongest seed already exists in the active codebase:

- **Infrastructure substrate:** crates/core/symthaea-infrastructure/ has a first-pass simulator, 28 state channels, a spatial metabolism grid, a controller, an active-inference agent, and a TownSympoiesis orchestrator.
- **Cross-domain scenario composition:** crates/domains/symthaea-scenarios/ already combines epidemiology, economics, control theory, and geodesy, and is intentionally light enough for fast property tests.
- **Ecological models:** crates/domains/symthaea-ecology/ supplies inspectable analytic population models, tipping-point baselines, trajectories, sensitivity diagnostics, and model caveats.
- **Symtropy:** provides a general physics/geometry/simulation substrate that can later replace coarse infrastructure heuristics where a suitable physical model exists.
- **Mycelix:** has domain concepts for housing, water, food, transport, mutual aid, finance, identity, and governance. Its own README cautions that most clusters remain pre-alpha; tests are not proof of production readiness.
- **Sol-Atlas:** is the appropriate spatial/planetary visualization and coordination surface, while the core scenario evaluator stays independent of rendering.

Useful starting code:
- [TownSympoiesis orchestrator](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-infrastructure/src/town_simpoiesis.rs)
- [Spatial metabolism grid](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-infrastructure/src/spatial_metabolism.rs)
- [Infrastructure simulator](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-infrastructure/src/simulator.rs)
- [Cross-domain scenarios](https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-scenarios)
- [Ecology model contract](https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-ecology)

## What I think about the older Resonatia / Resonant work

I could not locate a public repository with the exact name “Resonatia” or “Resonantia” in this pass. I did find the archived **Evolving Resonant Cocreationism** site repository, whose README describes it as an educational presentation of consciousness-first computing, the Seven Harmonies, human-centred design, and symbiotic intelligence.

My take: that body of work can be valuable as the **ethical and aesthetic constitution** of the city programme. It is not, by itself, a validated urban-planning methodology or a calibrated physical model. Keep what survives operationalization—dignity, flourishing, reciprocity, sovereignty, ecological limits, repair, participation—and convert each principle into:
1. an explicit constraint or decision right;
2. a measurable indicator where measurement is appropriate;
3. an appeal or correction path;
4. a falsifiable claim with a stated evidence level.

Do not encode a philosophy as one mysterious universal score. In particular, Φ or any consciousness-related research indicator should not determine a resident's entitlement, the value of their work, access to housing/water, or whose preferences count. Those are rights and governance questions, not something a consciousness proxy should silently price.

## Critical reading of the existing town prototype

The current implementation is a useful scaffold, but should not be presented as a city digital twin yet:

- The spatial grid creates real H3 cell identifiers, but its initial coordinate is explicitly mocked and the current aura renderer places circle proxies in a synthetic layout rather than rendering georeferenced city geometry.
- The grid uses global demand/availability inputs and simple role-based rules; it does not yet represent full network flow, street accessibility, water basins, household variation, supply-chain constraints, or calibrated cross-sector causality.
- The orchestrator hard-codes a 60 Hz grid assumption and illustrative parameters. Electrical frequency and grid constraints must be configuration/evidence, not universal constants.
- The TownSympoiesis economic ledger mints “Tend” using production, water clarity and a collective-Φ value. Separate physical production accounting from ethical/governance rules. Any community-credit scheme must be specified and authorized by its governance process; an unvalidated Φ score must not be treated as a monetary truth.
- The infrastructure state includes useful reliability and safety concepts, but its fast heuristic simulator must not be confused with the optional grid-physics simulator or with measured utility performance.

First task: make assumptions visible and replace mock inputs incrementally, rather than adding more visuals or a larger opaque agent.

## Paradigm shift: from “smart-city AI” to a reproducible living-systems design loop

1. **City as stocks, flows, and dependencies.** Connect energy, heat, water, food, mobility, materials, housing, care, ecology, jobs, and local wealth. A local intervention must disclose upstream inputs and downstream burdens.
2. **Scenario compiler, not a chatbot plan.** A proposal becomes a typed intervention portfolio with geography, cost and lifecycle, physical assumptions, delivery constraints, beneficiaries, possible harms, and a post-implementation measurement plan.
3. **Constraint-first optimization.** Protect essential needs and locally ratified rights floors first. Only designs that meet those floors are compared on Pareto fronts. Low carbon, fast delivery, or low cost cannot compensate for unsafe housing, water failure, severe access gaps, or avoidable displacement.
4. **Robust design, not a single forecast.** Run heat, drought, grid outage, flood, supply disruption, demographic change, and compound hazards. Report distributions and worst critical cases; do not hide failure behind mean performance.
5. **Counterfactual evaluation.** Compare against a dated baseline and a credible “do minimum / business as usual” scenario. State the causal hypothesis and what observation would falsify it.
6. **Cities as governed commons.** Separate technical feasibility from legitimate decisions. Residents and accountable local institutions set goals, validate trade-offs, own or govern assets where appropriate, and can inspect, contest, and correct evidence.
7. **Living calibration.** After implementation, compare predictions to observed outcomes, publish errors, update model cards, and retain failed interventions as learning rather than rewriting history.
8. **Open infrastructure over vendor lock-in.** City evidence and scenarios must be exportable, versioned, and replayable without relying on a proprietary dashboard.

## Target architecture and division of responsibility

### A. Evidence package

Every input needs a source, licence/legal basis, units, spatial and temporal scope, acquisition time, quality state, transformations, and uncertainty or coverage notes. Distinguish **measured**, **estimated**, **simulated**, and **missing**. Missing data must never silently become zero.

Candidate standards:
- OGC **CityGML 3.0** for a semantic model of 3D city objects and exchange across urban planning, disaster management, energy, transport, and environmental analysis.
- OGC **SensorThings API** for interoperable geospatial observation streams, sensor metadata, and measurement retrieval.
- **ISO 37120** for a comparable indicator vocabulary covering city services and quality of life. ISO currently lists the 2018 edition as published and “to be revised.”
- UN-Habitat **City Prosperity Initiative** for a multi-dimensional benchmark covering productivity, infrastructure, quality of life, equity and inclusion, environmental sustainability, and governance.

Use these as compatibility anchors, not as a reason to copy every indicator blindly. Adopt local units and standards as required, with explicit conversion and provenance.

### B. Canonical urban graph

Use stable entities and links for neighbourhoods, buildings/dwellings, street segments, transit stops, essential services, energy and water assets, habitat patches, hazards, operators, interventions, and governance bodies. Encode:
- physical flows (energy, water, heat, food, materials, waste and emissions);
- reachable access over real networks (not Euclidean radius alone);
- service dependencies and failure propagation;
- ownership, maintenance, authority, and decision rights;
- who benefits, who pays, who bears risk, and who could be displaced.

Keep personal movement or household detail out of the model unless necessary, lawful, and governed. Most accessibility and service-planning questions should be answerable with aggregated cohorts.

### C. Symthaea as the reasoning / experiment coordinator

Symthaea may propose hypotheses, generate candidate portfolios, choose informative experiments, detect inconsistent assumptions, and explain why a scenario failed. Its active-inference loop is not a substitute for the domain dynamics.

Enforce these boundaries:
- analytic and physical models own their equations, units, validity ranges, and error bounds;
- Symthaea cannot modify a hard floor or hide a failing metric to make a plan look better;
- any learned proposal is compared to simple baselines and a held-out benchmark suite;
- uncertainty and out-of-distribution status are visible, with abstention when evidence is inadequate;
- final public investment, land use, eligibility, policing, and allocation decisions remain with accountable humans and lawful institutions.

### D. Symtropy as physical simulation substrate

Bring in Symtropy only where the underlying mechanics are defined and tested: energy storage and flows, heat exchange, water conveyance, material handling, mobility/robotics, or spatial resource interactions. Do not turn every city variable into an arbitrary physical force. An urban social indicator is not a physical unit.

### E. Mycelix as the coordination and accountability layer

Use Mycelix for consented participation, provenance/attestation, proposals, contribution records, community governance, and shared resource coordination as those clusters mature. Treat its Commons, Civic, and Finance work as pre-alpha until independently verified at the relevant integration level. Do not place sensitive individual data on a public or broadly replicated ledger.

### F. Sol-Atlas as the spatial interface

Render scenarios and trade-offs on the geographic model: who gains access, which critical networks fail, how heat/cooling access shifts, where water stress accumulates, and which assumptions are uncertain. Visual layers must show data timestamps and confidence, and allow a user to distinguish source observation from simulation output.

**Next qualification target:** [Living City Policy Robustness Benchmark v0.2](https://github.com/Luminous-Dynamics/symthaea/blob/research/urban-systems-lab-v0-1/docs/research/LIVING_CITY_POLICY_ROBUSTNESS_BENCHMARK_V0_2.md) specifies matched policy baselines, held-out compound hazards, sensor/action delays, service-loss and recovery measures, and anti-leakage controls. Its implementation is tracked in [#7618](https://github.com/Luminous-Dynamics/symthaea/issues/7618); this is a specification, not an executed benchmark or a real-city validation claim.

## First benchmark release: LCB-0.1

The current branch adds a small pure-standard-library evaluator in the Symthaea scenario crate. It separates **locally configured hard floors** from a ten-metric Pareto comparison. It has no embedded universal city thresholds and no weighted “one number to rule them all.” Its tests use synthetic fixtures only.

**Executable policy-harness prototype:** [`living_city_policy_benchmark.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/research/urban-systems-lab-v0-1/crates/domains/symthaea-scenarios/src/living_city_policy_benchmark.rs) implements four deterministic baselines for a one-abstract-service synthetic environment; a four-case smoke suite produces 16 paired policy/case runs. It includes delayed backup actions, stale/missing telemetry, floor-breach and service-loss accounting, first-restoration timing (not stable recovery), canonical SHA-256-bound scenario/config/run receipts, deterministic replay verification, and a fail-closed suite assessor. The assessor distinguishes blocked evidence/coverage from valid floor failure and synthetic-suite qualification; its paired report binds candidate and baseline run digests. Twenty-seven tests are authored but remain unverified pending exact-head workflow evidence. The SHA-256 receipts are not authenticated signatures and are not yet persisted in the offline package verifier. This is not a physical model or a claim of real-city improvement. Implementation tracking: [#7618](https://github.com/Luminous-Dynamics/symthaea/issues/7618). Adaptive pathways, trigger robustness, and value-of-information research are tracked in [#7636](https://github.com/Luminous-Dynamics/symthaea/issues/7636).

### Feasibility floors (configure per pilot)

A real pilot should begin by having affected residents, service operators, engineers, ecologists, housing experts, and public-health experts ratify floors such as:
- essential-service access;
- the 90th-percentile housing-cost burden;
- critical-service reliability;
- safe-water service reliability;
- safe and accessible cooling during a defined heat event;
- involuntary-displacement risk.

LCB-0.1 currently implements these six categories. A scenario that fails any required floor is **infeasible**. Two infeasible scenarios are not silently ranked by a weighted score; fix the violations or explicitly document the unresolved governance choice.

### Outcome metrics for each feasible scenario

| Domain | Metric | Direction | Measurement contract |
|---|---|---|---|
| Access | Share of residents reaching each selected essential service via accessible walking/transit networks | Higher is better; floors should apply by cohort, not just city average | Include network topology, hours, crossings, grade, mobility needs, cost, and safety; report per service category |
| Housing | P90 housing cost as share of household income | Lower is better | Declare gross/net income basis, rent/mortgage and utilities treatment, geography, data vintage, and cohort |
| Critical infrastructure | Annual uptime of energy, water, communications, health, cooling, and other named critical services | Higher is better; hard floor | Define service-level and outage denominator; do not equate simulator channels with utility telemetry |
| Water | Reliability plus per-capita withdrawal | Reliability higher; withdrawal constrained by local basin and drought limits | Track potable/non-potable separation, water quality, seasonal availability, reuse safety, and basin context |
| Heat | Share of residents with accessible cooling/refuge and safe routes during a defined heat scenario | Higher is better; hard floor | Couple heat exposure to shading, building performance, mobility, hours, and local climate observations |
| Climate | Lifecycle greenhouse-gas emissions per resident-year | Lower is better | Include construction/embodied carbon, operations, transport, imported electricity/materials, and stated lifecycle boundary |
| Ecology | Habitat connectivity proxy with published method | Higher is better | Do not mistake tree count or green pixels for biodiversity; pair with native habitat, water, soil, and species indicators |
| Displacement | Modeled risk of involuntary displacement and observed housing loss | Lower is better; hard floor and post-project review | Include renter impacts, affordability preservation, anti-displacement measures and a comparison area |
| Cost | Lifecycle cost per resident-year | Lower is better among feasible designs | Include capex, operations, maintenance, replacement, financing, and price-year / discount assumptions |

Starter **hypotheses to test**, not universal standards: aim for at least 90% accessible essential-service coverage in every materially affected cohort; P90 housing-cost burden at or below 30%; ≥99.9% annual uptime for specifically defined critical services; ≥99.5% modeled safe-water demand served under the pilot's agreed design stress; and ≥95% accessible cooling/refuge access in the design heat event. These values must be adapted to local climate, legal obligations, infrastructure, data quality, and community decisions. The evaluator deliberately accepts explicit thresholds instead of imposing these values as defaults.

Climate and ecological targets should be defined as improvement against a transparent local baseline and a specified horizon. A first pilot might test a 50% lifecycle-emissions reduction against business-as-usual over ten years, but must publish embodied emissions, uncertainty, cost, and any burdens exported outside the boundary. Habitat and watershed goals need the same locality-specific treatment.

### Equity and robustness gates

In addition to citywide metrics:
- report each key access / reliability / cooling floor for materially affected neighbourhoods and cohorts;
- set and track an explicit maximum disparity (a pilot may test a gap of no more than five percentage points, subject to data quality and locally ratified groups);
- run single and compound stresses: heat + grid outage, drought + energy shortfall, flood + communications failure, and supply interruption;
- require the design to meet its critical floors for every designated safety-critical scenario, not merely on average;
- show distribution quantiles, failure count, recovery time, and uncertainty range across stochastic runs;
- report transfers of burden (e.g. emissions, water use, housing pressure, or heat risk displaced beyond the project boundary).

### Engineering and reproducibility benchmarks

| Benchmark | v0.1 acceptance rule | Why it matters |
|---|---|---|
| Input validity | Reject non-finite values, invalid ratios, and negative physical quantities | Prevent impossible data from generating a confident recommendation |
| Floor enforcement | Any required floor failure marks the candidate infeasible | Prevent compensatory “green scores” from hiding direct harm |
| Pareto logic | A scenario dominates only if it is no worse on every tracked objective and better on at least one; trade-offs remain visible | Preserve honest decision options |
| Infeasible comparison | Two infeasible scenarios remain labelled infeasible, not ranked by an arbitrary sum | Avoid disguising rights / safety violations as optimization |
| Deterministic replay | Same model commit, evidence hash, config, and random seed reproduce the same deterministic result | Enables audit and regression tests |
| Traceability | Every benchmark result points to input hashes, units, parameter set, model version, seed, and command | Prevent “queued != pass” and unrepeatable evidence |
| Predictive validity | Against held-out observed interventions, measure calibration and error by metric and cohort; compare to simple baselines | Prevent internal consistency from being mistaken for real-world accuracy |
| Robustness | Report all pre-registered hazard cases and preserve negative controls | Prevent cherry-picking favourable scenarios |
| Improvement loop | Compare predictions with post-implementation observations and publish misses | Make the model scientifically self-correcting |

Do not claim predictive accuracy until held-out data from real interventions are available. Synthetic tests establish code invariants, not city-scale validity.

## First experiment set

Keep the first benchmark small, deterministic, and interpretable. Use synthetic fixtures to validate the evaluator, then use a public, licence-compatible case with clear baselines before asserting real-world prediction.

1. **Baseline / business as usual:** current services and planned maintenance.
2. **Critical-service resilience:** microgrid/storage and load prioritization for named critical facilities.
3. **Heat-safe routes:** shade, cool roofs, passive cooling, cooling refuges, and accessible routes evaluated together.
4. **Water resilience:** leakage reduction, demand measures, safe non-potable reuse, storage, and drought stress.
5. **Access and housing:** mixed-use infill and network improvements paired with anti-displacement/affordability protections.
6. **Integrated portfolio:** compare the combined package against single interventions and identify cross-sector synergies, bottlenecks, and double-counted co-benefits.

For each portfolio, run at minimum: normal conditions, a design heat event, an agreed drought case, a 72-hour grid outage for designated critical assets, and at least one compound hazard. Those are benchmark scenarios, not claims about any specific city. Use per-scenario seeds and record them.

## Research protocol

1. **Pre-register:** chosen geography, time horizon, intervention, baseline, causal hypothesis, metric definitions, floors, exclusion criteria, hazard set, and stop/scale/revise rule.
2. **Model cards:** state model equations/source, units, calibration data, geographic and climate applicability, assumptions, known failure modes, and uncertainty.
3. **Baselines:** compare with no intervention, a simple rule-based plan, and—where meaningful—a standard optimization or domain-model baseline.
4. **Blind evaluation:** hold out locations, time periods, or interventions from tuning. Evaluate each major cohort and hazard.
5. **Independent review:** domain experts audit physical assumptions; residents review the fairness and burden distribution; a separate evaluator verifies the metric computation.
6. **Post-deployment:** collect observed outcomes, compare prediction intervals with outcomes, publish failures and corrective actions, and do not rewrite baselines after seeing results without a versioned amendment.

## Implementation sequence

### Landed on this research branch
- Add a pure-std living-city scenario evaluator to the existing scenario-composition crate.
- Implement ratio / non-negativity validation.
- Implement six configurable feasibility floors.
- Implement Pareto comparison across ten metrics without a weighted total.
- Add synthetic unit tests for floors, trade-offs, both-infeasible handling, and non-finite / out-of-range inputs.

### Next (before real data)
1. Add evidence manifests and a scenario-run receipt with stable hashes; missing source metadata must fail closed.
2. Add seeded canonical scenario fixtures and tests that exercise the cross-sector dependencies.
3. Replace mocked geography in the town prototype with a declared study boundary and real licensed source data.
4. Separate grid-frequency, energy, water, climate and financial assumptions into typed configuration with units and locale-specific settings.
5. Remove Φ from the monetary issuance formula; specify credits through a separate community-governed ledger with auditable production inputs and explicit legal/operational rules.
6. Use Symtropy where a validated physical model is needed; retain analytic domain models as reference oracles.
7. Connect Mycelix governance and Sol-Atlas presentation only after the evidence and scenario contracts are stable.

### Explicit non-goals for this phase
- No autonomous land-use or public-service allocation decisions.
- No single “city flourishing” or consciousness score.
- No claim that an LLM or Symthaea can replace urban planners, engineering sign-off, public-health review, or resident consent.
- No unverified claims of real-city improvements or performance.
- No waiting on a long-running CI job before inspecting, reviewing, or extending the research. Local deterministic evidence is useful; queued checks are not passes.

## Research references

- UN-Habitat, Urban Planning: https://unhabitat.org/topic/urban-planning
- UN-Habitat, Tools and Guides / City Prosperity Initiative indicators: https://unhabitat.org/tools-and-guides
- ISO 37120:2018, Sustainable cities and communities — Indicators for city services and quality of life: https://committee.iso.org/standard/68498.html
- OGC CityGML 3.0: https://www.ogc.org/standards/citygml/
- OGC SensorThings API: https://www.ogc.org/standards/sensorthings/
- US EPA, Reduce Heat Islands: https://www.epa.gov/green-infrastructure/reduce-heat-islands
- US EPA, Benefits of Trees and Vegetation: https://www.epa.gov/heatislands/benefits-trees-and-vegetation

## North star

Engineer cities so every neighbourhood can meet dignified human needs within ecological limits—and make each proposed intervention traceable, contestable, robust to shocks, and accountable for its actual outcomes.

## Updated implementation note — 2026-10-16

The follow-on draft PR now also contains:

- crates/domains/symthaea-scenarios/src/living_city_evidence.rs: typed provenance records and scenario-run receipt contracts, canonical length-prefixed bytes, digest-format validation, and fail-closed workflow states.
- crates/domains/symthaea-scenarios/src/living_city_benchmarks.rs: a versioned test registry with a baseline, individual heat/drought/grid-outage cases, and compound heat+outage, drought+energy-shortfall, and flood+communications-failure cases; profile-coverage and result-record validators.
- docs/research/LIVING_CITY_AUTOMATION_AND_VERIFICATION_V0_1.md: the complete engineering-compiler workflow and verification/promotion protocol.

The benchmark registry is not itself a physical simulator: local hazard magnitudes, site boundary conditions and evidence must be supplied in versioned profile bindings, then exercised by actual solvers. The receipt module now computes SHA-256 over canonical manifest/receipt bytes and verifies retained source/output bytes against declared digests, with metadata- and byte-corruption negative controls. An offline command at crates/domains/symthaea-scenarios/src/bin/living_city_verify.rs has also been added to read a run package from disk and stream-hash retained artifacts; it is not yet wired into orchestration/CI, does not authenticate a trusted signer, and has not been compiled or executed here. Rust tests have been added but have not been run in the available environment; skipped or in-progress PR-triggered workflows are not passing evidence. Exact latest PR head at this note's creation: d82e0e5142d8a64823e8514515a8083642f317e5.

This update does not change the original scientific limitation: synthetic fixtures can establish software invariants, but city-scale predictive validity requires independently checked reference cases and held-out observed data.

The follow-on branch also adds crates/domains/symthaea-scenarios/src/living_city_dependencies.rs, a deterministic static dependency-propagation layer. It exposes cascading outages and degraded services through explicit prerequisite links, with synthetic tests for grid-outage cascades, cycles, monotonicity and input-order invariance. It is not a physical solver: initiating failures must come from versioned hazard evidence or validated domain-model outputs. The follow-on implementation and reference-solver integration work is tracked at https://github.com/Luminous-Dynamics/symthaea/issues/7591.

The current branch adds an evidence-linked hazard profile bridge at crates/domains/symthaea-scenarios/src/living_city_hazard_profiles.rs. It connects stable benchmark IDs to named hazard parameters, manifest evidence IDs, initiating service failures, and the dependency-graph assessment. The profile's canonical digest is bound to the selected benchmark/profile revision, and evaluation blocks missing evidence, undeclared hazard types, duplicate parameters, unknown service nodes, stale bindings, or a manifest for another scenario.

This is still a pre-screen: a source item's presence in a manifest is not proof that its bytes are genuine, an explicit assumption is not a measurement, and a dependency cascade is not a physics simulation. Bundle schema v3 of the offline verifier loads the hazard profile, profile binding, service graph, and city-assessment package; rehashes retained source/output files; recomputes the dependency cascade and city-feasibility floors; and binds the profile, graph, city assessment, and grid-reference-suite identity into the receipt configuration digest. A city scenario that violates a configured hard floor is blocked from promotion even when every artifact hash is internally consistent. The governance-decision record is an explicit provenance requirement, not proof by itself that the decision represents lawful or genuine community authorization. The next milestones are wiring verifier output into scenario orchestration/promotion and validating the grid solver against an independent reference. See the detailed implementation and acceptance criteria in the engineering automation contract.

The active branch also adds a four-case analytical regression suite for Symthaea's existing linearized DistFlow feeder solver at crates/domains/symthaea-scenarios/src/living_city_grid_reference.rs: real-load voltage drop, reactive-only voltage drop, a two-segment radial feeder with downstream load aggregation, and DER-driven voltage rise. The offline verifier runs the suite and binds its suite identity/case IDs into the receipt configuration digest; a reference mismatch blocks workflow promotion. These are checks against the simplified model's own closed-form equations—not validation of nonlinear AC power flow, real outage performance, or city-scale predictions. The suite is added but still awaits compilation and execution on the exact PR head.

The newest vertical-slice increment adds crates/domains/symthaea-scenarios/src/living_city_assessment_package.rs. It ties all ten decision-bearing city metrics to named non-missing evidence items, commits the exact canonical metric payload and assessment package into the evidence manifest, and requires the six locally configured floors to cite a governance-decision item whose source reference matches the package's ratification reference. The CLI reevaluates the metrics/floors itself; it does not accept a generator-supplied feasibility boolean. A fully hashed but infeasible synthetic scenario is expected to return package-integrity verified while failing workflow promotion. This proves the non-compensatory gate behavior in the software contract only; it does not prove that metric derivations are physically valid or that a governance record is authentic.

A dedicated workflow at .github/workflows/living-city-contract.yml now runs pinned Rust formatting, the scenario crate's library tests, offline-verifier end-to-end/negative-control tests, and targeted Clippy on pushes to the Living City research branch and relevant pull-request changes. The general repository CI intentionally skips its heavyweight jobs on draft PRs, so this smaller path gives the work a direct validation route without waiting for the full workspace matrix. The workflow was just added; it is not yet evidence of a passing build.