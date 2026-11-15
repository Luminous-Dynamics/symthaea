# Atelier FactoryBench research corpus

- `factory-bench-corpus.v0.1.json` — four synthetic product operation graphs, layout alternatives and disruption scenarios.
- `factory-from-to-matrix.v0.1.json` — synthetic From-To flow matrix.
- `sota-comparison.v0.2.json` — machine-readable current research comparators, reported metrics, scope limits and reproduction tasks.
- `sota-comparison-v0.2.md` — research synthesis and staged product-to-factory co-optimization plan.

The v0.2 literature matrix expands the benchmark beyond facility layout to pattern-derived workload, fabric cut-order planning, machine/worker allocation, multi-line scheduling, supplier/partner allocation and event-based adaptation.

All performance figures from papers remain paper-reported, case-specific comparators. They are not Atelier results. Synthetic inputs are not production observations or capacity claims.


## Versioned metric dictionary

- `metric-dictionary.v0.1.0.json` — 21 versioned definitions spanning product work content, quality-adjusted labour efficiency, work-unit OEE, first-pass yield, defect rates, labour/machine productivity, lead time, OTIF, travel, changeover, time-to-first-conforming-unit, material yield, energy, disruption recovery, traceability, false-eligible assignments, fully-loaded cost and workforce workload.

Every metric defines its formula, numerator/denominator, unit, process/facility boundary, required metadata and caveats. Results must record evidence class, time interval, sample count, product/process/facility revisions, input digest, evidence references, exclusions and uncertainty.

### Comparison discipline

A comparison is admitted only when the relevant comparability keys match: garment and size mix, fabric/trim assumptions, order quantity, operation graph/standard-work revision, machine capability, staffing and skills, paid-time and shift definition, inspection and first-pass acceptance definition, downtime boundary, facility/partner scope, freight/queue boundary and measurement uncertainty.

Keep separate measures separate:
- defects per 100 units is not defective units per 100;
- OEE is per named work unit/asset, not a factory-wide average across heterogeneous equipment;
- changeover completion is not time to first conforming unit;
- material travel and person travel are distinct;
- simulator claims are not measured resource consumption;
- utilization is not worker wellbeing.

Missing values remain null with an explicit reason; they are never imputed as zero. Do not create a weighted composite that compensates for safety, egress, skill, capability, material admission or inspection failures.

### Standards status

The published KPI reference remains ISO 22400-2:2014 (with the applicable amendment). As of 15 November 2026, ISO/PRF 22400-2 Edition 2 entered the approval phase on 13 November 2026 but is not yet published. Keep KPI definition versions explicit and update the standard mapping after publication; do not claim conformity with an unpublished edition.

ISO 23247-5:2026 is published and provides a digital-thread architecture reference. NIST IR 8536 (September 2026) supports interoperability over uniformity across heterogeneous supply-chain records. This corpus uses those as design references, not conformity claims.


### Executable arithmetic companion

[draft PR #12364](https://github.com/Luminous-Dynamics/symthaea/pull/12364) adds dependency-free Rust calculators for the first eight KPI calculations/aggregations. The dictionary links the implemented functions directly to their metric definitions and marks them as authored but untested. The calculator does not perform source-data extraction, calibration, or measurement validation.


## Counterfactual evaluation and SOTA admission

- `counterfactual-evaluation-protocol.v0.1.0.md` defines frozen run manifests, separate synthetic/simulated/physical/independently measured lanes, paired scenario seeds, ablation and interaction tests, held-out garment/fabric/facility splits, independent feasibility checks, Pareto reporting, statistical reporting and strict industry-comparison admission.
- A claimed improvement must state exactly what was held constant, what changed, the evidence class, uncertainty and independent-verification status.
- Optimizer scores are not accepted until an independent feasibility verifier confirms the plan.
- The protocol explicitly prevents comparisons between unlike garment complexity, staffing, paid-time, quality, facility and cost boundaries.


## Modular reconfiguration and diagnosability

- [PR #12398](https://github.com/Luminous-Dynamics/symthaea/pull/12398) adds `reconfigbench-cases.v0.1.0.json` and the soft-first module transition evaluator defines 21 synthetic transition cases. The case matrix separates NoChange, AllocationOnly and PhysicalReconfiguration and includes site mismatch, expired evidence, process digest drift, interface incompatibility, footprint/keepout conflicts, coordinate frames, missing commissioning and partial owner data.
- `metric-dictionary.v0.1.0.json` now contains six additional reconfiguration KPIs: soft reallocation share, time to first independently accepted unit after a change, physical reconfiguration downtime, injected-fault diagnostic coverage, requalification-obligation count and qualified module reuse rate.
- The SOTA matrix adds a 247-paper RMS systematic review (2026) and a 112-company PCA-derived reconfigurability-index study (2026). These are research-gap and construct comparators, not direct textile production baselines.

### Research conclusion

Do not optimize physical modularity as an end in itself. Prefer soft allocation changes within the currently qualified layout; change tooling/process envelopes only within approved revisions; move/reconnect modules only when the expected product-quality, throughput or resilience benefit exceeds setup, downtime, validation and requalification burden.

The September 2026 apparel GA study is an important baseline because it reports additional-order rescheduling and machine reallocation inside a fixed modular structure without physical line reconfiguration. ReconfigBench therefore treats the share of demand changes handled without physical movement as a primary metric, but only when first-pass acceptance and qualified capability remain intact.

The May 2026 RMS review identifies diagnosability as under-studied and notes that the theory-to-practice relationship with resilience still needs experimental work. The June 2026 reconfigurability-index study gives adaptability and diagnosability prominent roles in its Portugal sample. Atelier's proposed contribution is to make those dimensions executable and auditable: inject known failures, localize them, track evidence invalidation and count the requalification consequences.

The current ReconfigBench cases are synthetic. They are not physical changeover-time, resilience or quality results.
