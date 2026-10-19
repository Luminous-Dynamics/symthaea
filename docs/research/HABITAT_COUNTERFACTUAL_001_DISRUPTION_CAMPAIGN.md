# HABITAT-COUNTERFACTUAL-001 — disruption campaign contract

**Parent:** [HABITAT-VERTICAL-001 #8051](https://github.com/Luminous-Dynamics/symthaea/issues/8051)  
**Factory/product contract:** [HABITAT-VERTICAL-001 contract](./HABITAT_VERTICAL_001_FACTORY_SUPPLY_CHAIN_CONTRACT.md)  
**Canonical production-value benchmark:** [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608)  
**Scheduler owner:** [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677)  
**Machine-readable definition:** [habitat-vertical-scenarios-v0.1.json](./fixtures/habitat-vertical-scenarios-v0.1.json)

## Paradigm shift: a factory is a service system under perturbation

A factory shouldn't be judged only by utilization, nominal throughput, or an attractive digital-twin visualization. The engineering question is whether it reliably produces accepted, useful articles through a changing supply chain—and how much service, time, work-in-process, material, energy, and cost are lost when assumptions break.

The next unit of work is therefore not a second scheduler or digital twin. It is a **frozen, paired disruption campaign** that can eventually drive the existing process/scheduling owners and independently verify their outputs. The campaign puts the eight disruptions from the base fixture into machine-readable interventions and fail-closed invariants, while explicitly identifying the missing adapters and state models that prevent scoring today.

This follows the measurement problem emphasized in NIST's manufacturing digital-twin work: model validity, interoperability, verification/validation, and uncertainty must be addressed, rather than assuming that a digital representation is trustworthy because it exists. Use existing supply-chain event standards, such as GS1 EPCIS, as interoperability mappings where useful; do not copy their whole identity or event ledger into this benchmark.

## What has been added

The fixture is a **scenario definition**, not an execution transcript. It freezes:

- all eight current disruption IDs, with exact set equality against the source fixture;
- a synthetic intervention and an expected safety/operational disposition for each case;
- hard invariants, required outputs, reportable metrics, and explicit execution blockers;
- a scorecard which keeps service loss, accepted output, lead time, backlog, cost, energy, recovery, genealogy correctness, hard-gate violations, and not-evaluable fields separate;
- baseline/candidate arm slots with unbound policy identities, so no unverified scheduler is silently treated as the baseline or the candidate;
- a proposed 10% lower lost-accepted-unit-hours hypothesis **only after** policies, runnable cases, metric boundaries, and the evaluation horizon are frozen.

Illustrative outage/delay magnitudes in the fixture are labelled synthetic parameters. They are not calibrated to a real factory and must not be reused as engineering limits.

## Fair counterfactual comparison

Before scoring, pin the exact policy and model revisions for both arms. Hold constant the demand, product/BOM revision, material-lot evidence, resource and worker calendars, currency/cost basis, utility dependencies, disruption event, and horizon. Change only the policy being studied. Where an adapter or input model differs, record it and do not call the comparison paired until the difference is resolved.

Required result states are **Better, Worse, Inconclusive, NotEvaluable, and Blocked**. Unknown or unavailable data is not zero; a blocked schedule is not a low-throughput observation; and a failed physical/quality guardrail cannot be compensated by gains in cost, lead time, or service.

If a future policy is stochastic, preregister the seed count and use paired common seeds. For deterministic runs, require exact replay under identical manifest and code identity. Publish scenario-wise paired deltas and uncertainty, where supported; never publish only a single aggregate that hides one catastrophic or non-evaluable case.

## Scenario-specific hard invariants

| Scenario | Required invariant |
|---|---|
| Supplier rejection | Rejected material cannot enter new accepted articles; lot containment returns all and only affected descendants. |
| Grade substitution | Shared marketing label cannot stand in for grade/process qualification. |
| Transport delay | An unarrived lot cannot satisfy material availability; replanning preserves every process and inspection constraint. |
| Equipment outage | The failed resource is unavailable during the outage; a substitute must possess the exact capability/qualification. |
| Power common mode | Production and inspection share the declared utility dependency; no acceptance bypasses unavailable inspection or required recalibration. |
| Skill absence | A machine capability does not imply worker availability or qualification. |
| Demand-mix change | Do not score a multi-variant schedule without compatible variant BOMs, process routes, demand, and setup/changeover model. |
| Repair-spare shortage | “Repairable by design” does not mean “repair completed”; spare, skill, capacity, and acceptance evidence are required. |

## Benchmark metrics and proposed targets

Report independently where the underlying model exists: accepted units by horizon, lost accepted-unit-hours, end-to-end lead time, backlog-unit-hours, cost per accepted unit, energy per accepted unit, recovery time, exact genealogy correctness, hard-gate violations, and every not-evaluable field with its reason.

The initial research hypothesis is **at least 10% lower lost accepted-unit-hours for the candidate across runnable cases relative to the frozen baseline**, with zero hard-gate violations. This target is provisional, not a known industry norm, not a pass/fail result encoded by synthetic arithmetic, and not currently testable: the baseline/candidate policies and household-goods scheduling adapter are not bound. Each case remains independently visible; the target cannot erase regression, unsafe behavior, or missing data.

The product economics, life-cycle resource accounting, and general baseline/ablation discipline remain owned by MFG-VALUE-001. Process qualification remains with MFG-PROC-000. Mycelix remains authoritative for applicable provenance and operational coordination records. This campaign consumes those contracts rather than creating parallel ledgers or engines.

## Research anchors

- [NIST, 2026 Roadmap on AI and ML for Smart Manufacturing](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) surveys industrial data integration, heterogeneous sensing/control, trustworthy and explainable operation, physics-informed and semantic AI, digital twins, supply-chain optimization, and sustainable manufacturing. The project implication is to benchmark the complete decision loop against conventional baselines, not an abstract model score.
- [NISTIR 8620, Digital Twins Workshops Summary Report (July 2026)](https://www.nist.gov/publications/digital-twins-workshops-summary-report) identifies interoperability, verification/validation/uncertainty quantification, cybersecurity, and workforce readiness as continuing priorities.
- [NIST Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing) describes quantified uncertainty and trustworthy validation as foundational to credible manufacturing twins.
- [GS1 EPCIS/CBV](https://www.gs1.org/standards/epcis) supplies an existing interoperable vocabulary for supply-chain events, status/location/custody, sensor observations and certification references; it is a mapping target, not a second local traceability ledger.
- [NIST's 2025 review of disassembly systems for circular product design](https://www.nist.gov/publications/review-disassembly-systems-circular-product-design) supports treating disassembly and circularity as whole-lifecycle system decisions rather than optimizing a detachable joint in isolation.

These sources motivate the design and test boundaries only. They do not validate this code, its synthetic intervention magnitudes, any real policy, or a physical factory.

## Current status and next gates

- Scenario contract: specified in the synthetic fixture.
- Rust structural qualifier: added with adversarial unit tests.
- Rust compilation/tests: not executed in the current environment; passing is not claimed.
- Policy identities and household-goods scheduler adapter: unbound.
- Disruption simulation, cost/lead-time improvement, real factory trials, human evaluation, and physical measurements: not run / none.
- Next: run formatting and targeted tests against the exact PR head; then bind the existing deterministic scheduler only after confirming its current interface and household-goods applicability; add an independent schedule verifier and immutable per-run manifest; score baseline and candidate on identical cases; only then consider replay or a measured pilot.

## Claim ceiling

A valid campaign definition proves only that a fixed set of synthetic experiments and their required gates are internally well-formed. It does not establish that the candidate can schedule, that an alternative supplier/resource exists, that the disruption can be recovered from, that a real factory is safe or resilient, or that any economic, environmental, quality, service, or repair target has been achieved.
