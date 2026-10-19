# HABITAT-RESILIENCE-OPTIONS-001 — option value of qualified factory flexibility

**Parent:** [HABITAT-VERTICAL-001 #8051](https://github.com/Luminous-Dynamics/symthaea/issues/8051)  
**Disruption contract:** [HABITAT-COUNTERFACTUAL-001](./HABITAT_COUNTERFACTUAL_001_DISRUPTION_CAMPAIGN.md)  
**Machine-readable challenge fixture:** [habitat-resilience-options-v0.1.json](./fixtures/habitat-resilience-options-v0.1.json)  
**Manufacturing outcome benchmark:** [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608)

## Paradigm shift: treat flexibility as a design option, not a slogan

A factory is not resilient because it has a second supplier, a spare machine, modular parts, or a backup battery written on a plan. Resilience depends on exact qualifications, capacity, lead time, utility dependencies, workforce skill, commissioning, and the ability to restore a defined level of accepted service.

A better design question is:

> What is the incremental cost of a *qualified* flexibility option, which disruptions does it mitigate, what residual service loss remains, and under which assumptions does the preferred design change?

This frames resilience investments as a decision under incomplete evidence. It does **not** turn uncertainty intervals into probabilities, assign fictional annual disruption rates, or monetize service loss without a defensible customer/business value model. Capital cost, lost accepted-unit-hours, recovery time and hard constraints stay separate.

## What the fixture contains

The fixture defines six synthetic option profiles:

1. Lean reference cell.
2. Qualified alternative supply and material grades.
3. Workcell/worker-skill redundancy.
4. Essential-utility continuity.
5. Modular/common components plus repair capability.
6. An integrated flexible microfactory portfolio.

They are crossed with all eight disruptions already declared by the source campaign, for 48 option-by-scenario outcomes. Each outcome supplies explicit bounds for lost accepted-unit-hours and recovery hours. The numbers were deliberately constructed as a **synthetic algorithmic challenge oracle**: mitigations are localized to the cases a strategy says it addresses, while unrelated cases retain the reference bounds. The values are invented test vectors, not estimates or forecasts. Do not cite them as engineering evidence.

Each option has prerequisite qualification gates. An improvement is conditional on all relevant gates; a declared but unqualified alternate supplier, a merely available-but-unqualified machine, or a modular design that hasn't passed product requalification earns no real-world resilience credit. The fixture does not claim any of those gates currently pass.

## How to evaluate it without inventing certainty

The next evaluator should compose the existing robust-frontier and robust-decision work, not implement another multi-objective or probability engine:

- [SE-OPT robust-frontier PR #7788](https://github.com/Luminous-Dynamics/symthaea/pull/7788) distinguishes robust dominance from merely possible dominance under interval-valued objectives.
- [ICCB-004 #8024](https://github.com/Luminous-Dynamics/symthaea/issues/8024) specifies non-probability-forcing payoff bounds and minimax-regret semantics when state probabilities are only bounded.
- [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608) owns fair baseline/candidate comparison and manufacturing-outcome claims.
- [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677) remains the scheduler owner; this fixture must not become a second scheduler.
- [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) and Mycelix [MFG-EVID-001 #3084](https://github.com/Luminous-Dynamics/mycelix/issues/3084) / [MFG-PROC-001 #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085) own process capability and manufacturing provenance contracts.

Only after those exact APIs/revisions are inspected and qualified should an evaluator classify candidate portfolios. Keep the result per objective and per scenario. Report robustly dominated, robustly nondominated, possible-only dominance, ambiguous tradeoff, blocked and not-evaluable states distinctly. A robust winner must never be manufactured from midpoint selection or an assumed probability distribution.

## What this can benchmark now—and what it cannot

The corpus can be used to test whether a future robust-frontier consumer:
- reads all 6 option identities and all 48 scenario outcomes exactly once;
- rejects missing, duplicate, mismatched or reversed intervals and unknown scenario IDs;
- preserves synthetic-only and qualification-conditional status;
- avoids probability, ROI and expected-annual-loss calculations when none were supplied;
- keeps hard feasibility as a gate rather than a compensable metric;
- produces invariant classifications when option/scenario input order is permuted;
- preserves why a candidate is blocked or not evaluable.

It cannot yet claim a preferred real portfolio. The current campaign doesn't have empirically measured loss distributions, supplier lead-time histories, accepted-output value, site utility measurements, variant-BOM changeover data, spare availability, or a calibrated uncertainty model. The fixture's computation status is intentionally **robust_frontier: not_run**, **minimax_regret: not_run**, policies unbound and empirical calibration none.

## Proposed benchmark targets

These are project acceptance hypotheses, not achieved results or universal industry norms:

- 100% of the 48 option-by-scenario cells present exactly once.
- Zero missed hard-gate violations and zero resilience benefit credited when prerequisites are unqualified or unknown.
- 100% deterministic classification under input permutation.
- 100% of reportable results preserve units, scenario identity, option identity, gate status and source revisions.
- No invented probability, monetized outage benefit, ROI or expected cost from these inputs alone.
- Robust frontier output exactly matches hand-checked synthetic known answers; any minimax-regret calculation matches an independent small-instance oracle once that owner interface is bound.
- In a later data-backed pilot, compare incremental flexibility investment with measured reductions in accepted-service loss and recovery time. Do not set a universal percentage target until a valid baseline, time horizon, service valuation and uncertainty interval are preregistered.

## Research grounding

NIST's [2026 AI/ML roadmap for smart manufacturing](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) emphasizes industrial data complexity, integration across heterogeneous systems, trustworthy operation, supply-chain/logistics optimization and sustainable manufacturing. NIST's [Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing) identifies verification, validation, quantified uncertainty and interoperability as foundations of trustworthy models. The September 2026 NIST paper on [manufacturing in a circular economy](https://www.nist.gov/publications/manufacturing-circular-economy-research-needs-design-systems-modeling-and-digital) argues for system-level modelling and digital-thread support when assessing circularity.

The engineering implication is not to make the model more elaborate at any cost. It is to make every claimed flexibility option, improvement, and exclusion auditable—and to identify which missing evidence could change a real decision.

## Claim ceiling

This fixture adds a synthetic decision challenge only. It does not qualify suppliers, machines, workers, backup utilities, modular interfaces or repair paths; it does not establish a factory layout, capital budget, risk forecast, resilience improvement, product safety, savings, or operational readiness. All real decisions remain blocked by missing evidence and the existing engineering/authorization owners.

## Upstream assumption tests

A resilience strategy should not be scored until the assumptions that confer its claimed mitigation have been examined. The follow-on [HABITAT-ASSUMPTION-FALSIFICATION-001](./HABITAT_ASSUMPTION_FALSIFICATION_001.md) binds nine assumptions to ten candidate tests. It explicitly does not claim that tests have been selected or that any cost, duration, probability, supplier capability, utility behavior, repair result, or option benefit has been measured. Use those definitions to identify which evidence must be admitted before an option's mitigation bounds can be treated as anything beyond synthetic challenge values.

