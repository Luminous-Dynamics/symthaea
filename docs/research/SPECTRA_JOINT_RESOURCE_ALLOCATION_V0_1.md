# SPECTRA Joint Resource Allocation — Research Specification v0.1

**Status:** Research specification; no joint allocator or empirical superiority claim.
**Issue:** [#10639 — co-allocate input data, simulations and validation](https://github.com/Luminous-Dynamics/symthaea/issues/10639)
**Related:** [#10615 — decision-directed simulation allocation](https://github.com/Luminous-Dynamics/symthaea/issues/10615), [#10618 — scenario-bank assurance](https://github.com/Luminous-Dynamics/symthaea/issues/10618), [#10601 — locked allocator comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10601)

## 1. Objective

Move beyond deciding only where to run more Monte Carlo replications.

SPECTRA should eventually compare the value and cost of:
- collecting new input data that changes the estimated target/scenario distribution;
- running more simulations under a frozen simulator/input snapshot;
- expanding or stress-testing the scenario bank;
- improving model/measurement adequacy;
- or stopping because none of the supported expenditures can change the decision enough to justify its cost.

This is a joint experimental-design and simulation-optimization problem. It is research-only until the estimands, resource contracts and inferential assumptions are separately qualified.

## 2. Literature basis

### Wang & Zhou (2025): input data and simulation as simultaneous budgets

[Data-Driven Ranking and Selection With Simultaneous Input Data Collection and Simulation](https://doi.org/10.1002/nav.22275), Naval Research Logistics 72(8), 1170–1183.

The paper treats multiple streaming input-data sources and simulation outputs as distinct resources. It formulates multistage budget allocation between input data collection and simulation, and gives consistency / asymptotic allocation results under its stated assumptions. Importantly, estimated input distributions change over time, so outputs generated at different stages can be non-IID.

### Li, Wan & Hong (2025): additive distributionally robust R&S

[Additive Distributionally Robust Ranking and Selection](https://arxiv.org/abs/2509.06147).

The paper studies additive allocation under finite distributional ambiguity. In its setting, only a critical subset of design × input-distribution scenarios is sampled infinitely often; some worst-case scenarios of non-best alternatives need not be among that subset. This challenges an assumption that every action's current worst-case scenario always deserves persistent simulation.

### Other comparators

- [Optimal Computing Budget Allocation for Data-Driven Ranking and Selection](https://doi.org/10.1287/ijoo.2024.0035), Wang & Zhou (2025), on streaming input uncertainty.
- [Optimal Simulation Budget Allocation Under Unknown Sampling Variance](https://arxiv.org/abs/2509.02138), Du, Ryzhov & Gao (2025), on allocation with unknown means and variances.

These results are comparators, not proofs that their procedures directly transfer to SPECTRA.

## 3. Do not merge four uncertainty types

| Type | What it means | What reduces it |
|---|---|---|
| Input-data uncertainty | Unknown target-population, observer, or exogenous input distribution | Representative input observations / calibration data |
| Monte Carlo uncertainty | Finite simulation replications under a frozen generator and input snapshot | More score simulations |
| Scenario-coverage uncertainty | Plausible mechanisms or interactions omitted from the finite scenario bank | New scenario discovery and external validation |
| Model/measurement uncertainty | Observation family, lapse/drift, missingness, timing and measurement assumptions | Held-out adequacy and measurement studies |

The #10599 confidence sequence quantifies Monte Carlo uncertainty for its declared score streams. It does not automatically quantify uncertainty in an estimated input distribution or a changing scenario bank.

## 4. Give scenario weights typed semantics

The word 'weight' must not conceal three different concepts.

1. **Prevalence weight:** estimated frequency of a regime in a target population. It may be updated from representative samples under an explicit sampling design.
2. **Normative importance weight:** a scientific or policy judgment about the consequences of a regime, irrespective of prevalence. Collecting more prevalence data must not silently change it.
3. **Ambiguity set:** an epistemic set of plausible distributions/weights used for worst-case or lower-tail evaluation. It is not automatically a probability distribution or posterior.

These require distinct typed contracts, provenance and update rules. A single generic floating-point weight vector is insufficient to represent all three.

## 5. Joint decision objective

Let b_data denote budgets for one or more input-data streams; b_sim denote simulation replication budgets by action × scenario; b_validation denote adequacy / scenario-discovery budgets; C_data, C_sim and C_validation denote measured resource costs; and J denote a predeclared terminal decision loss, such as robust regret or cost-to-valid-certificate.

Investigate minimizing expected or worst-case J under actual multi-resource constraints.

A candidate next action may:
- collect another input observation;
- update an input-distribution estimate;
- simulate under an existing frozen input snapshot;
- run a hidden-world stress search;
- collect a validation / calibration observation;
- or stop / abstain.

The controller's acquisition score is an efficiency policy, not an inferential guarantee. The #10599 score confidence sequence, #10585 epsilon certificate, #10582 model-adequacy veto and #10618 scenario-scope contract remain separate authorities.

Keep units distinct: participant time, specialist/data acquisition cost, simulation count, CPU/GPU time, memory, energy and wall-clock deadline. Prefer a vector-budget or constrained Pareto formulation over undocumented conversion weights.

## 6. Changing input distributions creates non-IID simulation streams

When new input data change the estimated input distribution, the simulator's conditional output law can change between stages. Pooling all stage outputs as one IID stream is not automatically valid.

For every score sample, bind:
- exact input-distribution snapshot / version;
- simulator and generator version;
- scenario, action and replication identity;
- source data manifest and frozen model parameters;
- sampling mode and PRNG identity.

Investigate:
- stage-specific estimation and aggregation;
- importance weighting, with explicit support / overlap diagnostics;
- stratification and control variates;
- fresh-snapshot replication streams;
- martingale or other valid estimators under predictable input-distribution updates.

Do not append outputs from different input-distribution versions to the current #10599 IID confidence sequence without a proved estimator and uncertainty method. A deterministic seed does not make non-identically distributed observations IID.

## 7. Critical-scenario allocation

Investigate whether the additive DRR&S critical-set structure applies to SPECTRA's exact robust objective.

Compare exhaustive action × scenario sampling with sparse allocation for:
- Maximin;
- weighted lower-tail CVaR;
- the actual ambiguity sets in #10618;
- positive / negative CRN correlation;
- ties and changing active worst scenarios;
- incumbent / challenger reversals.

A non-best action's worst scenario need not necessarily be a decision-critical scenario for the selection gap. But the asymptotic k + m - 1 result from the cited paper must not be assumed to transfer to SPECTRA's objective without a derivation.

Preserve a protected audit and sentinel floor. No scenario may be removed from the assurance claim just because it is not currently certificate-active. Receipts must state which cells are sampled, which are not, and why.

## 8. Locked benchmark

Add these arms to #10601 and #10618:

1. Fixed input distribution + uniform simulation allocation.
2. Plug-in input-distribution estimation followed by simulation.
3. Input-data-only refinement.
4. Simulation-only refinement.
5. Wang–Zhou-style simultaneous input-data and simulation budget allocation.
6. Decision-directed joint allocation using a calibrated prediction of decision-loss reduction per cost.
7. Exhaustive robust scenario simulation.
8. Additive / critical-scenario allocation when its assumptions apply.
9. Oracle allocation as an unattainable reference only.

Vary:
- one and multiple input streams with heterogeneous acquisition costs;
- balanced and imbalanced streaming rates;
- simulator runtime by action/scenario;
- input-distribution misspecification and target-covariate shift;
- support changes and poor overlap;
- high and low simulation variance;
- scenario-weight ambiguity and normative weights (kept distinct);
- critical-scenario changes as estimates update;
- fixed and changing scenario banks;
- short and long wall-clock deadlines.

Report:
- familywise PCS_epsilon, robust regret and false-robustness;
- input-distribution estimation risk;
- Monte Carlo coverage and interval width;
- cost-to-valid-decision by resource type;
- worst-decile and hidden-world regret;
- budget fractions for data, simulation, validation and audit;
- non-IID handling / support failures;
- exact sample and allocation replay.

Use untouched confirmatory seeds and preserve all failure / non-evaluable dispositions.

## 9. Proposed qualification targets — not results

- At equal total resource constraints, at least 15% lower median cost-to-valid-decision than the best single-resource allocation in preregistered regimes where input and simulation uncertainty are both material.
- No deterioration in Monte Carlo coverage or input-distribution uncertainty calibration.
- Zero pooling of non-identically distributed simulation outputs under an IID claim without a qualified estimator.
- 100% provenance binding of every score to its input snapshot and generator version.
- Critical-scenario sparsification preserves the same robust certificate and mandatory audit floor.
- All regimes where joint allocation loses remain visible.

These are research targets, not demonstrated performance.

## 10. Execution order

1. Add typed semantics for prevalence weights, normative weights and ambiguity sets to the scenario-assurance contract.
2. Define versioned input-data stream and resource-cost contracts.
3. Build a known-truth benchmark separating input uncertainty from simulation uncertainty.
4. Implement an offline reference campaign comparing single-resource and joint allocation.
5. Prove or reject re-use of confidence sequences when input distributions change.
6. Derive whether critical-scenario sparsification applies to SPECTRA's Maximin/CVaR objective.
7. Implement a joint allocator only after independent statistical review and locked, failure-inclusive results.

## Non-goals

- No live participant or external data collection.
- No automatic updating of normative weights from prevalence samples.
- No pooling across changing simulator/input versions under an IID assumption.
- No unproved transfer of scalar robust R&S critical-scenario results to SPECTRA.
- No replacement of the #10599 confidence sequence or robust certificate.