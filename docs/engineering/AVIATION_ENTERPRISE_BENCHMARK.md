# Aviation Enterprise Closed-Loop Benchmark (AECB-001)

**Status:** research specification; not implemented or evaluated yet  
**Owner:** Symthaea engineering / aviation systems work  
**Purpose:** define a reproducible, fail-closed benchmark connecting airline operations, maintenance and spares, aircraft production capacity, and lifecycle economics through existing system owners.

## Executive hypothesis

Aviation optimizes a coupled physical enterprise, but decisions are distributed across flight operations, technical operations, procurement, OEM production, airport/ground handling, and finance. A locally optimal schedule can be globally poor: a flight recovery action can consume a maintenance window, strand a crew, exhaust a critical spare, or force production into a less valuable sequence.

**Hypothesis H1:** On frozen scenarios and identical constraints, an evidence-aware rolling-horizon coordinator that composes specialist solvers can reduce total disruption and lifecycle cost versus the strongest feasible siloed and integrated baselines, without relaxing any service, maintenance, qualification, or safety constraint.

This is a hypothesis, not an established performance claim. The benchmark should be designed to disprove it if the cross-lifecycle layer adds no value.

## State of the art and the gap we are testing

The industry already has serious digital and optimization capabilities; this work must compare against them, not a strawman.

- **Airbus Skywise:** Airbus described the April 2026 combination of Skywise digital solutions and Navblue flight-operations expertise as an effort to connect flight, technical, and ground operations across historically fragmented workflows. Its platform reports connection to nearly 12,000 aircraft. Sources: https://www.airbus.com/en/newsroom/stories/2026-04-with-skywise-airbus-is-re-imagining-the-digital-sky and https://www.skywise.com/en/digital-platform
- **Integrated schedule recovery:** an INFORMS Journal on Applied Analytics case reports that Lufthansa/Google's Operations Decision Support Suite jointly optimizes aircraft, passengers, and crew. The paper reports EUR 12 million and 14 kilotons of CO2 saved at Swiss, with EUR 30 million and 50 kilotons projected annually as deployment expands. These are case-specific reported outcomes, not directly transferable benchmark thresholds. Source: https://pubsonline.informs.org/doi/10.1287/inte.2025.0296
- **Predictive maintenance:** GE Aerospace's February 2025 SAS case combined flight-data analytics with technical-operations records to investigate E190 bleed-system and flight-control issues. GE's 2025 sustainability report separately reports internal results for engine monitoring. These establish a strong incumbent baseline for connected diagnostics, not a standardized public evaluation score. Sources: https://www.geaerospace.com/news/press-releases/ge-aerospace-and-sas-collaborate-predictive-maintenance-initiative-addressing-ata and https://www.geaerospace.com/sites/default/files/2025-sustainability-report.pdf
- **Supply-chain economics:** IATA estimated that aviation supply-chain constraints cost airlines more than USD 11 billion in 2025, including delayed fuel savings, additional maintenance, engine leasing, and surplus spares. This identifies a material systems problem, not an amount the proposed system can claim to recover. Source: https://www.iata.org/en/pressroom/2025-releases/2025-10-13-01/
- **Manufacturing digital thread:** NIST's Model-Based Enterprise work explicitly connects product, process, service, and logistics models so production capability and schedules can be reasoned about across existing systems. Source: https://www.nist.gov/programs-projects/model-based-enterprise-program
- **Aviation AI assurance:** FAA guidance emphasizes demonstrating safety within the aviation certification framework. EASA's proposed Concept Paper Issue 03 (published 3 June 2026 for consultation through 12 August 2026) explores further AI techniques and Level 3 advanced-automation use cases. The plan must distinguish research evidence from approved operational authority. Sources: https://www.faa.gov/aircraft/air_cert/step/roadmap_for_AI_safety_assurance and https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-proposed-issue-3

### Differentiation thesis

The target is not to rebuild Skywise, airline operations research, predictive-maintenance analytics, or an ERP. The testable differentiator is a **cross-lifecycle, configuration-bound decision-and-evidence loop** that composes specialized owners across airline service delivery and aircraft production, then checks whether a decision is valid against the physical and organizational constraints it affects.

No claim of industry novelty or superiority is permitted until a reproducible public benchmark and fair baselines support it.

## System boundary and existing owners

Compose existing work; do not create a parallel airline solver, maintenance evaluator, aircraft physics engine, manufacturing scheduler, or energy solver.

- **Symthaea:** reasoning, forecasting, scenario selection, proposed cross-domain plans, uncertainty and abstention.
- **Symtropy:** time-dependent digital-twin/scenario execution, resource and network consequences, disruptions, and coupled-system evaluation.
- **Sol-Atlas:** energy/resource feasibility and infrastructure constraints through an explicit versioned interface; it must not be presumed to own flight scheduling or aircraft physics.
- **Mycelix:** identity, permissions, provenance, revision binding, traceability, evidence receipts, and change-impact lineage.
- **Independent verifier:** recomputes constraints and objective from the emitted plan using a separate implementation. The proposer is never its own sole verifier.
- **Existing specialist owners to compose:**
  - Symthaea ENG-DESIGN-001 / issue #6005: https://github.com/Luminous-Dynamics/symthaea/issues/6005
  - Symthaea FACTORY-OPS-001 / issue #7677: https://github.com/Luminous-Dynamics/symthaea/issues/7677
  - Symthaea AIRMAINT-002 / issue #7771: https://github.com/Luminous-Dynamics/symthaea/issues/7771
  - Symthaea PROP-OPT-001 / issue #7809: https://github.com/Luminous-Dynamics/symthaea/issues/7809
  - Symthaea fixed-wing telemetry benchmark / issue #6259: https://github.com/Luminous-Dynamics/symthaea/issues/6259
  - Symthaea phase-aware loadmaster state chain / issue #8187: https://github.com/Luminous-Dynamics/symthaea/issues/8187
  - Symtropy AIRSPACE-OPS-001 / issue #2279: https://github.com/Luminous-Dynamics/symtropy/issues/2279
  - Symtropy AIRSPACE-OPS-002 / issue #2297: https://github.com/Luminous-Dynamics/symtropy/issues/2297
  - Mycelix manufacturing evidence / issue #3084 and process registry/routing / issue #3085: https://github.com/Luminous-Dynamics/mycelix/issues/3084 and https://github.com/Luminous-Dynamics/mycelix/issues/3085

The benchmark coordinator owns only cross-domain scenario packaging, interfaces, objective accounting, baseline orchestration, and independent evaluation.

## Minimum benchmark instance

Use a fully synthetic and published reference scenario first. Do not imply it represents a real airline or use proprietary airline, OEM, maintenance, or passenger records without explicit rights.

Each scenario contains:

1. a small fleet of typed aircraft configurations, maintenance state, route-compatible capability, and location;
2. a time-bounded flight schedule, passenger/connection counts, crew qualification/availability summaries, airport/turnaround limits, and weather/closure disruptions;
3. maintenance tasks with due windows, skill and tooling requirements, task dependencies, release/evidence prerequisites, and uncertainty ranges;
4. repairable/consumable spares, supplier lead-time distributions, inventory locations, and interchangeability/qualification constraints;
5. a small production line with capability-qualified operations, precedence constraints, resource windows, yields/rework, and delivery commitments;
6. cash, energy, parts, labor, and capacity costs with explicit units and source/version identifiers;
7. a frozen uncertainty model and random seeds.

Start with tiny instances whose feasibility can be enumerated exactly, then add medium cases for optimization quality, then larger stress cases for run-time and robustness. Use synthetic scenarios and public, reusable datasets only where licences and field definitions permit.

## Required comparisons

All methods receive identical data, constraints, action limits, forecast horizons, and compute budgets where applicable.

1. **Do-nothing / original plan:** useful as a reference, not as the only baseline.
2. **Deterministic greedy heuristics:** earliest-deadline, minimum-delay, and priority rules with explicit documented tie-breaking.
3. **Siloed specialists:** aircraft/crew/passenger recovery; maintenance-resource scheduling; inventory/spares planning; production-line scheduling run independently, with a fixed handoff policy.
4. **Conventional integrated optimizer:** MILP/CP-SAT or another established method where the selected instance admits a defensible formulation. Report optimality gaps or prove when exact optimality is not available.
5. **Rolling-horizon coordinator:** the proposed cross-domain planner that invokes existing owners and replans when new observations invalidate the current plan.
6. **Oracle / lower-bound reference:** exact optimum for tiny enumerated instances and mathematically valid lower bounds for larger ones; never label a heuristic result as optimal without proof.
7. **Ablations:** remove cross-domain coupling, evidence freshness checks, uncertainty handling, or re-planning one at a time to identify which mechanism contributes the gain.

Avoid unsupported comparisons to proprietary vendor products when their implementations and datasets are not reproducible. Public case-study numbers contextualize the field; they do not substitute for matched experiments.

## Primary objective and non-compensable constraints

Report the full vector of outcomes, not just one blended score.

### Primary objective

Minimize a declared, unit-consistent total cost over a fixed evaluation horizon, including:
- delay and cancellation costs;
- passenger re-accommodation and missed-connection costs;
- aircraft-on-ground time and maintenance disruption;
- parts expediting, inventory carrying, and shortage cost;
- production lateness, changeover, rework, scrap, and overtime;
- energy and emissions metrics when the scenario supports defensible accounting;
- planner compute time and re-planning instability.

Publish all objective weights and run sensitivity analysis. Also report the Pareto frontier for cost, service, resilience, emissions, and production throughput so weights cannot hide trade-offs.

### Hard constraints (never scored away)

- no flight is declared dispatchable without required, configuration-matched constraints and evidence;
- no crew assignment violates the declared qualification or duty/availability rules in the synthetic instance;
- no maintenance task is skipped, fabricated, or treated as released without its required prerequisite evidence;
- no production operation is assigned to an unqualified worker/resource or before its predecessors;
- no part is treated as interchangeable without an explicit, versioned qualification rule;
- no stale or superseded evidence silently remains current after a relevant configuration/assumption change;
- no unknown, missing, inconsistent, or out-of-domain critical input is coerced into a permissive value;
- if the independent verifier cannot establish feasibility, the result is INDETERMINATE/REJECTED, never PASS.

This is a research benchmark, not a flight-release, maintenance-release, dispatch, airworthiness, or certification system.

## Metrics and proposed targets

These are **proposed research acceptance thresholds**, not measured results or existing industry norms. Freeze them before comparative runs; revise only via a versioned benchmark change.

| Dimension | Measure | Initial acceptance target |
|---|---|---|
| Constraint integrity | Independent verifier acceptance of emitted plans | 100% of accepted plans pass independent checks |
| Negative controls | Deliberately invalid/stale/incomplete scenarios | 100% of known safety/qualification/evidence negatives are rejected or marked indeterminate |
| Tiny-instance optimality | Agreement with exact enumeration | 100% on the frozen tiny suite |
| Comparative value | Total cost vs strongest feasible baseline | At least 5% lower mean cost on held-out stress seeds, with a paired 95% confidence interval excluding zero, and no hard-gate regression |
| Service reliability | Flights completed, delay/cancellation and recovery time | Report per-scenario vectors; no claimed universal uplift. Establish target after baseline variance is measured |
| Maintenance resilience | AOG hours, overdue tasks, repeated disruption, spare fill rate | Improvement only if task/qualification/evidence completeness is non-inferior |
| Production performance | Throughput, makespan, late orders, rework and first-pass yield | Improvement with worker/tool qualification and all precedence/resource checks intact |
| Robustness | Cost and feasibility under shifted distributions, supplier shocks, data missingness and compound disruptions | Pre-register stress families; report worst-case and tail metrics, not only averages |
| Planner performance | Median/p95 decision latency, memory, CPU, solver gap/timeouts | Bounded budgets; timeout returns best verified feasible plan or indeterminate, not unverified output |
| Reproducibility | Repeatability across clean runs, manifest completeness | Fixed seeds/configuration/data digests reproduce equivalent outputs and metric summaries |

The 5% target is a hurdle for whether the additional integration complexity is worth pursuing on this benchmark. If the strongest baseline already solves an instance to optimality, the proposed system must match the optimum and demonstrate another measured benefit (such as faster feasible recovery, lower compute, or better robustness) rather than promise an impossible cost reduction.

Use at least 1,000 paired seeds for the first stochastic medium benchmark if runtime permits; otherwise select sample size through a pre-registered power/precision analysis. Hold out entire disruption families—not merely random rows—to test generalization. Report effect sizes, confidence intervals, denominators, failure counts, missingness, and all exclusions.

## Adversarial and fault-injection suite

Every benchmark release must include positive and negative controls:
- an aircraft becomes unavailable after plan generation;
- a crew assignment becomes invalid under a changed rule/revision;
- a critical spare is missing, delayed, or falsely marked interchangeable;
- a maintenance finding invalidates a planned return-to-service assumption;
- a factory tool or qualified inspector is unavailable;
- design/configuration revision changes and invalidates dependent test evidence;
- a supplier or telemetry source provides delayed, duplicated, contradictory, or malformed records;
- the forecast distribution shifts and the planner's confidence becomes miscalibrated;
- a downstream service is unavailable or returns a stale result;
- a malicious or compromised identity submits a plausible but unauthorized change;
- solver timeout, partial output, integer overflow, missing cost units, and non-finite values;
- verifier/proposer disagreement.

Each fault has an explicit expected outcome. Safety/evidence negatives must fail closed. Missing non-critical data may be tolerated only where the scenario schema declares a safe bounded fallback.

## Data contract and evidence record

Every run should emit a machine-readable, canonical manifest with:
- benchmark version and scenario-family ID;
- source/license and digest for each input dataset;
- scenario/configuration digest and code commit;
- method identity, solver/build version, parameters and compute budget;
- seed, simulated forecast and disruption identity;
- proposed plan digest and independent verifier implementation/version;
- per-constraint result, objective components, metrics and uncertainty;
- blocked/rejected/indeterminate reason codes;
- output digest and a parent-linked evidence receipt.

Use typed status values such as FEASIBLE_VERIFIED, INFEASIBLE, INDETERMINATE, REJECTED, and TIMED_OUT; never map a timeout or unknown to success. Mycelix can preserve provenance and authorization; it must not magically transform a signed but false claim into a true one.

## Implementation sequence

1. **Freeze scope and interface:** document shared input/output types and invariants; link existing canonical owners rather than cloning their logic.
2. **Tiny exact oracle:** build deterministic, hand-checkable fixtures with an independent verifier and negative controls.
3. **Baseline harness:** implement do-nothing, greedy, siloed and tiny-instance exact baselines; freeze seeds and metrics.
4. **Cross-domain handoff:** connect one flight-recovery decision to maintenance/spares and factory capacity; preserve provenance and invalidation.
5. **Rolling-horizon stress tests:** inject observations, failures, distribution shifts and solver timeouts.
6. **Energy/resource extension:** call Sol-Atlas through a versioned feasibility contract; do not bury resource constraints in an opaque weighted score.
7. **Public reproducibility:** publish command, environment, data manifest, raw outputs, summary script, baseline parameters and limitations.
8. **Only then evaluate field pilots:** use authorized real data, operator review, privacy/security controls and jurisdiction-appropriate safety processes. A simulator result is not field validation or regulatory approval.

## Claim ceiling

A passing benchmark supports only the exact scenario families, code/configuration digests, constraints, methods, seeds, and statistical protocol evaluated. It does **not** establish airline operational authority, certified maintenance, aircraft airworthiness, flight-control safety, OEM production approval, universal superiority, or a claim that an airline/manufacturer can be operated without people.

## Research decision rule

Proceed beyond the synthetic benchmark only if:
1. the verifier and negative controls pass;
2. cross-domain coupling beats the strongest feasible baseline on held-out disruption families, or demonstrates an equally valuable, predeclared gain in feasibility latency/robustness;
3. the benefit persists under weight sensitivity and scenario shifts;
4. gains can be attributed to specific mechanisms through ablations;
5. an engineering review confirms the work composes existing owners rather than creating a duplicate solver.

Otherwise, simplify the coordinator, improve its data contracts, or stop the integration experiment.
