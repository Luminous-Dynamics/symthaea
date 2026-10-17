# SE-OPT: Evidence-Aware Robust Multi-Objective Frontier Analysis

**Date:** 2026-10-17  
**Status:** shared analysis primitive; not an optimizer, design-selection authority, or engineering qualification  
**Tracking:** [SE-OPT extension #7787](https://github.com/Luminous-Dynamics/symthaea/issues/7787)  
**Program owners:** [SE-OPT #3686](https://github.com/Luminous-Dynamics/symthaea/issues/3686), [ENG-OPT-001 #5635](https://github.com/Luminous-Dynamics/symthaea/issues/5635)  
**Initial consumer:** aircraft design and serviceability work; potential consumers include electro-acoustic design and other qualified engineering studies.

## 1. Do we need something better than Pareto?

We need something *beyond a conventional point-estimate Pareto filter*, not a replacement for Pareto optimality.

A standard Pareto filter is useful: it eliminates a candidate when another is no worse on every objective and strictly better on at least one. It does not, by itself, answer:

- whether a small apparent advantage is larger than measurement/model uncertainty;
- whether the candidate is feasible under hard constraints;
- whether missing data or failed evaluations have been hidden;
- how much a frontier changes when uncertainty is retained;
- which unresolved experiment would most change the engineering decision; or
- how the decision-maker values different trade-offs.

This implementation addresses the first four as a bounded, deterministic analysis tranche. It does not silently solve the latter two; they require separate preference and experiment-selection contracts.

The core distinction is:

- **Robust dominance:** one candidate is separated from another in a favorable direction across every declared objective interval.
- **Possible-only dominance:** the reported intervals permit one candidate to dominate another for some endpoint choices, but do not establish that it does.
- **Trade-off or unresolved relation:** interval evidence cannot establish either direction. That does not mean equivalence.

Interval overlap is not evidence that candidates are equal; it is a reason not to claim a robust winner.

## 2. Ownership and architecture

This work extends the existing zero-dependency `symthaea-operations-research` crate; it does not create another optimizer crate or an aircraft-only objective system.

It composes these established owners:

- SE-OPT #3686 owns the wider multidisciplinary optimization, uncertainty, sensitivity, calibration, and tool-adapter program.
- ENG-OPT-001 #5635 defines the reusable engineering design-space/objective/constraint contract.
- Domain-specific systems remain responsible for generating valid objective evidence and evaluating their hard constraints.
- The new `robust_frontier` module validates comparability, classifies interval relations, and reports the resulting frontier.
- Existing evidence-admission/qualification systems remain responsible for qualification claims. This module emits no `safe`, `approved`, `best`, or `qualified` state.

The module consumes data; it does not execute a solver, invent intervals, resolve evidence URLs, authenticate source artifacts, compute confidence intervals, or claim physical truth.

## 3. Input contract

The study declares an ordered objective profile. Each objective has:
- stable objective ID;
- direction (minimize/maximize);
- unit;
- reference-condition profile;
- interval semantics; and
- interval-method ID.

Each candidate contains:
- stable candidate ID;
- explicit status: feasible, infeasible, indeterminate, or evaluation failed;
- a source reference for the status assessment; and
- exactly one interval observation or a reasoned missing observation per objective.

Each observed interval carries finite lower/upper values, matching units/conditions/semantics/method, and an evidence reference. Feasible candidates must have complete intervals for every declared objective. Non-feasible candidates can retain missing observations with an explicit reason; they remain in report/coverage accounting and never enter the frontier.

A shared interval method/semantics and reference-condition ID are required across candidates for each objective. The implementation does not silently mix engineering bounds, confidence intervals, credible intervals, scenario envelopes, or measurement uncertainties as though they were the same thing.

The feasibility status is supplied by a domain evaluator and is not independently recomputed here. A status evidence reference provides traceability, not authentication.

## 4. Dominance mathematics

Normalize each objective into a loss convention. For a minimized objective with interval ([l,u]), its loss interval is ([l,u]). For a maximized objective, it becomes ([-u,-l]). Objectives remain separate; no weights or scalar quality score are introduced.

Let (A_j=[a_j^L,a_j^U]) and (B_j=[b_j^L,b_j^U]) be normalized loss intervals.

**Robust dominance:**

[
a_j^U leq b_j^L quad 	ext{for every objective }j
]

with strict inequality for at least one objective.

This establishes that even A's worst declared endpoint is no worse than B's best endpoint on every objective.

**Possible-only dominance:**

[
a_j^L leq b_j^U quad 	ext{for every objective }j
]

with strict inequality for at least one objective, provided robust dominance has not already been established.

This is a conservative interval possibility relation under independent per-objective endpoint choices. If objective intervals are correlated, endpoint combinations may not be jointly achievable. The current implementation neither derives a joint uncertainty set nor models correlation; possible-only relations must not be presented as probabilities.

## 5. Output contract

The report contains:

- **Robust frontier:** feasible candidates not demonstrably robustly dominated by another feasible candidate.
- **Unambiguous frontier diagnostic:** robust-frontier candidates with no possible-only incoming dominance under the independent interval relation.
- **Ambiguous frontier:** robust-frontier candidates that another candidate could possibly dominate for some interval values.
- **Robust dominance pairs** and **possible-only dominance pairs**, with explicit direction and stable IDs.
- **Midpoint-reference frontier:** ordinary Pareto analysis on interval midpoints, explicitly diagnostic only. A midpoint is not presumed to be a mean or most-likely value.
- **Midpoint disagreements:** candidates whose membership differs between the conservative interval frontier and midpoint-reference frontier.
- Per-objective interval coverage and interval-width summaries.
- Counts of feasible, infeasible, indeterminate, evaluation-failed, and missing metric observations.

Infeasible, indeterminate and failed evaluations remain visible but can never dominate or enter either frontier. If there is no feasible candidate, the result has an empty frontier; the empty set is not a pass or a selected design.

Candidate order is canonicalized by stable ID. Pair lists and ID lists are deterministic. The naive pairwise algorithm is (O(n^2m)) for (n) candidates and (m) objectives, and it rejects more than 2,000 candidates before pairwise work. Scaling beyond this cap requires a separately verified algorithm (e.g. a bounded epsilon-archive, verified batch frontier merge, or specialized high-dimensional method), not merely raising the constant.

## 6. Initial qualification corpus

The Rust module provides 16 authored unit-test declarations covering:

1. exact point intervals reproduce the ordinary Pareto frontier;
2. separated intervals establish robust dominance for minimize and maximize directions;
3. overlapping intervals produce possible-only relations rather than false robust dominance;
4. genuine trade-offs remain on the frontier;
5. infeasible, indeterminate, and failed candidates stay visible but are excluded;
6. missing metrics are reasoned and visible for non-feasible candidates;
7. a candidate declared feasible with a missing objective is rejected;
8. midpoint/interval frontier disagreements are surfaced;
9. empty feasible frontiers are reported without inventing a winner;
10. duplicate candidate IDs are rejected;
11. reversed, non-finite, and width-overflowing intervals fail closed;
12. metric units must match the declared profile;
13. candidate input permutations produce identical reports;
14. the candidate-count cap is checked before pairwise analysis;
15. empty objective plans fail closed; and
16. missing status provenance is rejected.

These test declarations require compilation and execution in a pinned Rust environment before they count as passed. Synthetic fixtures verify implementation semantics only; they do not establish physical calibration, interval coverage, or superiority of any design.

## 7. Proposed benchmarks and acceptance thresholds

These are qualification targets, not claimed results:

- **100%** hand-calculated known-answer agreement for robust dominance, possible-only dominance, maximize/minimize conversion, and point-interval Pareto equivalence.
- **Zero** robust-dominance claims unless A's worst normalized interval is no worse than B's best on every objective and strictly better on at least one.
- **100%** of non-feasible and evaluation-failed candidates retained in the report, with explicit status and metric coverage.
- **100%** deterministic report identity under candidate input permutation, given identical plan/records.
- **Zero** imputation of missing metrics, no hidden weighted score, and no frontier entry for non-feasible candidates.
- Report frontier sizes, possible-only relation counts, interval-width summaries, missingness, excluded candidates, and midpoint disagreements separately.
- Benchmark algorithm cost with increasing candidate/objective counts up to the cap; do not remove the cap until a replacement has independently verified complexity/correctness.
- Use independent endpoint-enumeration oracles for small interval fixtures and point-Pareto oracle comparisons. Such finite oracles validate the enumerated cases only.

When real engineering data are available, interval generation and coverage need their own protocol, with declared coverage meaning, repeated-measure structure, dependence/common-mode calibration, applicability domain, and held-out cases. The interval engine must not be used to manufacture calibrated uncertainty from arbitrary ranges.

## 8. What would actually take Symthaea beyond a frontier?

A frontier answers “which candidates are not dominated under this relation?” It does not answer “what should we learn next?” A broader program should proceed through distinct, auditable layers:

### Stage A — robust relation kernel (this PR)

Explicit uncertainty semantics, hard feasibility gate, robust/possible dominance, deterministic frontiers, comparison diagnostics, and negative controls.

### Stage B — sensitivity and decision stability

Vary declared assumptions and scenario sets; report the conditions where frontier membership changes. Do not infer an exact crossover from sparse samples, and do not call a sensitivity ranking causal evidence.

### Stage C — multi-fidelity evaluation

Use cheap analytic/surrogate screens to prioritize candidates for expensive numerical or physical studies, while preserving evaluation fidelity and exact tool/model identity. A surrogate's predicted objective is not an executed high-fidelity result. A better architecture should maximize the useful discrimination obtained per unit evaluation budget, not only the number of generated designs.

### Stage D — active experiment selection / value of information

Choose the next analysis or measurement according to expected reduction in decision-relevant uncertainty, expected improvement in the *robust* design set, experiment cost, and risk. Preserve alternative hypotheses when the evidence cannot discriminate them. This needs an explicit experiment owner and a separately qualified utility/acquisition method; do not bury stakeholder preference in a covert scalar.

### Stage E — preference-aware decisions (human or explicit policy)

Let the decision maker declare non-negotiable constraints and trade-off preferences, expose how the chosen solution changes as the preferences change, and retain the unranked frontier. Preference-guided selection can improve usefulness, but the preference model is an explicit policy—not a universal engineering truth.

### Stage F — independent qualification

Bind any selected candidate to the exact model/configuration, input artifacts, tool versions, assumptions, constraints, and evaluation history. Independent verification/validation and existing engineering authority remain outside the optimizer.

The near-term research goal should be fewer expensive wrong experiments, more reliable rejection of fragile “winners,” and transparent engineering decisions—not a claim that one new algorithm beats every existing optimizer.

## 9. Research basis

- [50 years of multi-objective optimization and decision-making](https://doi.org/10.1016/j.ejor.2025.06.012), European Journal of Operational Research, 2025: distinguishes mathematical and population-based optimization, including preference incorporation and choosing preferred solutions.
- [Active-learning multi-objective robust optimization under uncertainty](https://doi.org/10.1080/0305215X.2025.2495800), published online 2025: uses model uncertainty and targeted evaluation to improve the robust Pareto-region approximation.
- [Interactive multi-objective robust optimization under deep uncertainty](https://doi.org/10.1016/j.envsoft.2024.106233), Environmental Modelling & Software, 2025: supports exposing uncertainty and allowing decision makers to direct exploration.
- [NASA OpenMDAO](https://software.nasa.gov/software/LEW-18550-1): a multidisciplinary analysis and optimization framework whose scope includes integrating variable-fidelity discipline tools.
- [ENG-OPT-001 #5635](https://github.com/Luminous-Dynamics/symthaea/issues/5635), [speaker uncertainty-frontier #7724](https://github.com/Luminous-Dynamics/symthaea/issues/7724), and [CRISPR interval-frontier #7602](https://github.com/Luminous-Dynamics/symthaea/issues/7602): existing Symthaea problem owners motivate sharing only the domain-neutral interval-relation primitive; their domain semantics remain separate.

## 10. Nonclaims

This module is not a novel general-purpose optimizer, surrogate model, uncertainty generator, decision-maker, design approval, safety case, or certification. Interval dominance is established mathematics. The proposed contribution is a small shared Rust implementation with fail-closed data semantics and deterministic diagnostics that can be reused by qualified domain studies. Real-world value must be shown with independent analytical fixtures, actual compilation/test evidence, and domain-specific validation—not inferred from the fact that a frontier was computed.
