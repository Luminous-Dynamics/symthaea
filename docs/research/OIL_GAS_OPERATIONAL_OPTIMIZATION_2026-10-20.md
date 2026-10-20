# Oil and Gas Industrial Optimization: Benchmark-First Research Program

**Status:** exploratory engineering plan, 2026-10-20  
**Claim ceiling:** no field validation, no production control authority, and no claim of state-of-the-art superiority.

## Executive thesis

Build a measurement-grounded industrial optimization layer that composes, rather than replaces, established domain engineering tools. The initial implementation is a small, deterministic methane-abatement portfolio benchmark. It is intentionally not a methane estimator, reservoir simulator, process simulator, or controller.

The longer-term target is a cross-asset decision system that links measured source inventories, site-level observations, physical process constraints, cost and resource budgets, geospatial logistics, uncertainty, and independently reviewable intervention evidence.

## What the existing state of practice already does

Do not compare a prototype to a straw-man industry baseline.

- The International Energy Agency's 2026 Global Methane Tracker estimates 124 Mt/year of methane emissions from fossil fuel operations in 2025, split into 45 Mt from oil, 43 Mt from coal, and 36 Mt from natural gas. It estimates that around 70% of fossil-fuel methane emissions can be abated with existing technology and more than 35 Mt could be avoided at no net cost under its 2025 energy-price assumptions. These are IEA estimates, not a single universally measured census. The IEA also notes major variation across countries and data availability.  
  Source: https://www.iea.org/reports/global-methane-tracker-2026/key-findings
- OGMP 2.0's 2026 report describes 163 member companies in 90 countries, covering 43% of global oil and gas production; 69% of their reported emissions were based on measurement-derived data. Its Level 5 approach reconciles source-level inventories with independent site-level measurement; site-level data complement, rather than replace, source-level information.  
  Sources: https://www.ogmpartnership.org/events/ogmp-20-2026-report-measurement-mitigation-better-methane-data-enabling-action and https://www.ogmpartnership.org/frequently-asked-questions
- DOE/NETL's open-source PARETO framework already optimizes produced-water infrastructure, treatment, distribution, and beneficial reuse. NETL's NEWTS data resource describes more than 400,000 energy-related wastewater samples. We should benchmark against and interoperate with PARETO, not claim that a new optimizer invents water-network optimization.  
  Sources: https://github.com/project-pareto/project-pareto and https://netl.doe.gov/centers-of-excellence/oil-gas
- Established vendors already provide connected drilling, production optimization, autonomous workflows, process simulation, and digital twins. SLB describes integrated digital operations across reservoir, drilling, production, and facility workflows. AspenTech markets Aspen HYSYS for process modeling and dynamic studies. Our prototype must be compared against the relevant installed workflow and its data quality, not marketing copy or generic dashboards.  
  Sources: https://www.slb.com/solutions/digital-operations and https://www.aspentech.com/en/products/engineering/aspen-hysys

## Paradigm shifts worth testing

### 1. From asset dashboards to conservation-closed process graphs

Represent each site as a typed graph of material, energy, water, equipment, and information flows. Every process boundary must enumerate feedstocks, reagents, recycle streams, working fluids, products, by-products, waste, utilities, and measurement uncertainty. A missing stream is an explicit gap, never silently zero.

Use Symthaea's existing industrial/process evidence and conservation work as foundations; do not create a competing mass-balance ontology.

### 2. From point estimates to uncertainty-aware decisions

Report lower, central, and upper scenarios separately. Do not call a wide interval a verified balance or a guaranteed abatement amount. Separate statistical confidence intervals, engineering scenarios, measured values, and model projections in the schema. The Rust prototype's scenario floor is an optimization input, not a confidence bound unless an independent method validates that interpretation.

### 3. From isolated interventions to portfolio and network optimization

Compare a transparent expected-abatement-per-dollar greedy baseline with an exact small-case optimizer that maximizes conservative scenario-floor abatement under a capex budget. Prevent obvious double-counting by source group. Future versions must model interaction effects, shared infrastructure, annual OPEX, gas-sale revenue, time value of money, lifecycle emissions, water, reliability, and constraints from the site's actual operating envelope.

### 4. From a single weighted score to a Pareto frontier

Do not hide trade-offs behind an arbitrary weighted sum. Preserve alternatives across capital cost, net operating cost, methane avoided, water use, energy, uptime, lifecycle impacts, and uncertainty. Disqualifying safety and environmental constraints are hard constraints, not objectives that can be compensated by profit.

### 5. From AI suggestions to evidence-gated decisions

Use models to propose and rank hypotheses, detect anomalies, or explore candidate portfolios. Do not let an unvalidated model directly change well, compressor, separator, flare, or pipeline control settings. Start with historical replay and shadow mode; require independent engineering review and approved operating procedures before any field trial.

### 6. From siloed site tools to auditable cross-project coordination

- **Symthaea:** uncertainty-aware inference, anomaly candidates, intervention search, conservation checks, and reproducible comparative experiments.
- **Symtropy:** physical/digital-twin experiments where its models are appropriate and independently validated; do not treat a rigid-body engine as a process simulator.
- **Sol-Atlas:** geospatial asset, resource, transport, hazard, and utility dependency context; preserve dataset lineage and spatial resolution.
- **Mycelix:** provenance, role-scoped approvals, work-order/measurement attestations, versioned evidence, and durable audit trails. An attestation proves that a stated record was signed under a defined policy; it does not prove the physical measurement is true.

## First implementation

The new stdlib-only Rust crate, symthaea-industrial-optimization, provides two explicitly limited portfolio strategies:

1. greedy_expected_per_dollar: an interpretable expected-value-per-capex baseline.
2. optimize_robust_portfolio: exhaustive search for small frozen cases, maximizing the sum of lower scenario estimates under the same budget.

Both reject malformed scenario ordering and require positive capex (so the ratio baseline remains defined), exclude candidates marked ineligible for planning, enforce one intervention per declared source group, and use deterministic output ordering. Embedded example values are invented mechanics fixtures, not field data or industry averages.

The exact solver is intentionally capped at 20 candidate interventions (at most 1,048,576 subsets). It does not infer emissions, establish that an intervention is safe, or model interactions between different source groups.

## Benchmark design and targets

All comparisons must freeze the data version, units, time basis, baseline workflow, random seed where applicable, model/configuration hashes, decision constraints, and evaluation script. Keep training/calibration cases separate from held-out evaluation cases. Report negative results and uncertainty.

| Workstream | Required baseline | Primary measures | Advancement gate |
|---|---|---|---|
| Methane portfolio | Operator/site intervention workflow plus transparent expected-abatement-per-dollar baseline | Measured tonnes CH4/year avoided, capex and net cost per tonne, time to verified repair, source overlap, estimation error | On frozen cases, beat the baseline on a pre-registered objective without violating any hard constraints; proposed initial research target: at least 10% more conservative abatement for the same capex on held-out cases |
| Measurement reconciliation | Source-level inventory plus independent site-level measurement, following OGMP 2.0 semantics | Source coverage, absolute/relative discrepancy, uncertainty, time to reconcile, unresolved sources | No claim of complete inventory while a material source or reconciliation gap remains |
| Produced water | Same cases and assumptions in DOE/NETL PARETO | Total cost, transport, treatment, reuse, residual/brine handling, chemistry constraints and energy | Demonstrate a feasible Pareto improvement or a clearly evidenced new capability against PARETO; match its baseline cases before claiming improvement |
| Process physics | Validated process/thermodynamic simulation or trusted measured case | Mass and energy closure, output quality/yield, throughput, model error, sensitivity | Every claimed improvement preserves physical closure and meets pre-registered reference tolerances; no unvalidated scale-up |
| Integrity analytics | Current inspection/condition-monitoring workflow and a naive/static rule baseline | Detection lead time, recall at a fixed false-alert burden, localization error, calibration, downtime | Improve an agreed metric on held-out labelled events while reporting missed detections and false positives |
| Cross-asset resilience | Current planning workflow | Service interruption, common-mode failures, recovery time, total lifecycle cost and environmental effects | Improve the pre-registered Pareto frontier without weakening safety, environmental, or evidence constraints |

The 10% target above is a proposed engineering hurdle, not a measured result, market benchmark, or forecast. It should be adjusted before evaluation only if the dataset and statistical power justify the change; the final target must be frozen before testing. A result on synthetic fixtures establishes optimizer behavior only.

## Adversarial cases that the benchmark must include

1. Two candidate actions claim the same metered emissions source; the optimizer must not add their abatement.
2. A candidate has high expected abatement but an inadequate or unsubstantiated lower scenario; show how the decision changes without treating the floor as statistical confidence.
3. A candidate is unqualified or explicitly ineligible; it must never be selected.
4. Cost or abatement arithmetic overflows; return an error or reject the combination rather than wrap.
5. Missing or non-finite real measurements, stale timestamps, unit mismatches, and gaps in source coverage; mark unresolved instead of converting them to zero.
6. More than 20 candidates; use a future scalable solver or explicit refusal, not unbounded exhaustive enumeration.
7. The same proposed repair changes production, energy, and emissions simultaneously; prevent duplicated benefits across separate objective ledgers.
8. A treatment option shifts contaminants into concentrate or waste; total material disposition remains visible.
9. One power, water, measurement, or gas-handling dependency creates a common-mode failure across multiple assets.
10. A scenario appears better only under one cost/price assumption; report sensitivity and break-even regions.
11. A model recommendation violates a permit, process envelope, or hard safety constraint; it remains infeasible even if economics are attractive.
12. A site-level methane estimate conflicts with source-level estimates; trigger reconciliation and uncertainty accounting rather than choosing whichever estimate is convenient.

## Next stages across the four projects

1. **Benchmark harness:** freeze synthetic portfolios, adversarial fixtures, deterministic outputs, and machine-readable results.
2. **Real evidence adapters:** read public IEA/OGMP guidance, vetted measurement exports, and properly licensed open datasets; retain units, timestamps, lineage, and quality flags.
3. **Produced-water comparison:** reproduce a documented PARETO case with common inputs before testing extensions.
4. **Spatial twin:** bind asset locations, pipeline routes, treatment options, utilities, hazards, and transport constraints in Sol-Atlas.
5. **Experiment provenance:** represent observations, model/configuration versions, approvals, intervention events, and post-intervention measurements in Mycelix.
6. **Field-readiness gate:** require qualified operator partners, process-safety review, cybersecurity assessment, instrument validation, and shadow-mode results. No autonomous control in this research tranche.

## What success means

The first success is not a claim that Symthaea is better than commercial systems. It is a reproducible benchmark where a reader can run the same frozen case, inspect all assumptions, reproduce the baseline, see the candidate result, and identify every unresolved dependency.

A state-of-the-art claim requires held-out, representative cases and an independently reviewed comparison against relevant operational and commercial baselines. Until then, the correct label is research prototype.
