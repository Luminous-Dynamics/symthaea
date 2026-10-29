# AVIATION-ENTERPRISE-017 — Paired recovery evaluation across disruption ensembles

## Purpose

A recovery method should not be assessed on one convenient schedule. This protocol evaluates the greedy baseline and bounded delay-aware search on the same frozen set of synthetic disruption scenarios, computes paired candidate-minus-baseline costs, and evaluates that vector under the existing finite-support total-variation (TV) ambiguity-set implementation.

The adapter is intentionally thin: it reuses the retiming model/verifier and the existing distributionally robust cost-delta evaluator. It does not create new nominal probabilities, infer real-world probabilities, verify scenario realism, or guarantee behavior outside the supplied support.

## Core comparison

For each scenario \(s\):

1. Validate the frozen retiming scenario and caller-supplied nonempty input digest.
2. Produce a greedy plan and bounded-search plan under the same scenario model and objective.
3. Independently verify both plan outputs again by stable flight/aircraft IDs.
4. Record both objective costs and calculate \(\Delta_s=C_{search,s}-C_{greedy,s}\).
5. Preserve the search's optimality status, gap, node/work counters, delay minutes and cancellations.
6. Sort per-scenario outputs by scenario ID and pass \((p_s,\Delta_s)\) to the existing TV-robust evaluator.

A negative \(\Delta_s\) means search returned a lower-cost plan than greedy in that scenario; positive means worse. The robust output reports the nominal expected delta and worst expected delta over a declared TV ball. It also reports the worst individual scenario, so a good average cannot hide a local regression.

Nominal probabilities are provided in integer parts per million (ppm) and must sum exactly to 1,000,000. The TV radius is ppm of probability mass that may be moved, corresponding to an L1 distance at most twice that radius. The result is conditional on this finite support, the declared nominal masses, the model and the radius; it is not protection against omitted disruption modes or arbitrary distribution shift.

## First regression fixture

The first fault fixture contains one long, low-cancellation-penalty flight overlapping a shorter, high-penalty flight, with only one available aircraft. The time-ordered greedy baseline accepts the low-penalty flight and subsequently cancels the high-penalty flight. The search can instead cancel the first flight and operate the high-penalty one. This makes the local-greedy failure mode explicit and gives a known exact comparison.

A separate no-overlap control checks that the methods agree when the schedule creates no resource conflict. The current integration suite defines tests for the known cost gap, weighted TV-robust expectation, input-order invariance, duplicate-scenario rejection, probability-mass failure and no-feasible-within-budget failure.

## Acceptance gates and targets

Targets are not results until exact-head tests have executed.

- **Oracle correctness:** on tiny scenarios, the bounded search cost must match an independently implemented exhaustive raw-input oracle.
- **Greedy sanity:** the exact optimum should never cost more than a verified greedy plan under the same scenario/objective when optimality is proved.
- **Budget integrity:** no result may claim a plan or optimality if the search found no verified feasible plan; a partial search preserves its global lower bound.
- **Robust reporting:** exact ppm mass, stable scenario order, paired costs, and explicit worst individual regression are mandatory.
- **Input integrity:** duplicate IDs, empty digests, invalid model fields and invalid probability mass fail closed.
- **Work accounting:** report search cost, gap/status, expanded/generated nodes, candidate actions, lower-bound evaluations, availability-relaxation checks, delay minutes and cancellations alongside robust cost deltas.

For larger studies, report separately:

- mean/nominal cost improvement;
- TV-worst expected cost delta for several preregistered radii;
- worst single-scenario regression and share of scenarios improved/regressed/unchanged;
- regret against a tiny exact oracle wherever computable;
- p50/p95/p99 runtime and time to first verified feasible plan on a named hardware/toolchain profile;
- performance by disruption family, not just an aggregate average.

Do not choose the TV radius or nominal scenario probabilities after viewing the candidate results. The radius should be predeclared or calibrated from an independent held-out source, and the scenario distribution must be justified independently of the solver. Retain all adverse scenarios and null runs.

## Industry context and comparison discipline

Lufthansa Group's published operations recovery system coordinates aircraft, passenger and crew subsolvers, uses an initial heuristic phase and improvement phase, and combines large-neighborhood search with column generation. It is an operational benchmark for architecture, not a directly matched numerical baseline for this tiny synthetic delay-only model (Davies et al., 2026, DOI: https://doi.org/10.1287/inte.2025.0296).

Research on integrated flight/aircraft/crew recovery also models recovery as a sequential decision process and evaluates learned recovery policies (Transport Policy, 2025, DOI: https://doi.org/10.1016/j.tranpol.2024.11.011). This suggests a later simulator-based evaluation in which the policy can be re-run after each new disruption observation; the current adapter is a per-scenario static solver comparison, not that sequential setting.

The aircraft-recovery literature emphasizes scenario definitions and the cost of robustness: overly conservative solutions can waste aircraft utilization or ground resources. See the 2025 study on robust airline scheduling and turnaround under uncertainty (DOI: https://doi.org/10.1016/j.tre.2025.104440) and the review of airline disruption management published in 2025 (DOI: https://doi.org/10.1016/j.cor.2024.106915).

Two recent adjacent studies sharpen the evaluation design:

- Cha and Kwon's September 2026 *Data-driven optimization for airline service quality improvement under delay and cancellation risk* reports 97.95%–99.60% of hindsight-oracle value for a leakage-controlled forecast-based allocation policy on a locked 2025 carrier-month panel. It explicitly separates measurement consistency, ex-ante forecast value, optimization value, and causal effects, and notes that an uncertainty-aware robust extension can incur an ex-post decision cost. This is not the same problem as aircraft recovery, so the numbers are methodological context, not a target for this solver (DOI: https://doi.org/10.3934/jimo.2026188).
- A 19 October 2026 human-in-the-loop turnaround-coordination study tests 1,000 synthetic inbound cases across six regimes and reports monotonicity failures rather than hiding them. It explicitly says its synthetic regime frequencies are generator artifacts, not observed operational rates. This is a useful precedent for preregistered stress regimes, ablations, and retained negative results—not evidence of airline deployment (DOI: https://doi.org/10.3389/frai.2026.1977034).

These findings suggest a stronger benchmark principle: robustness must be evaluated together with realized policy cost, and every result must state whether its scenario population is observed, reconstructed, or synthetic. Do not assume a robust policy automatically has more realized value.

No comparison here establishes industry SOTA, operational savings, calibrated real-world probabilities, safety, or dispatch authority.

## Verification command

On the exact source head in a Rust-equipped environment:

    cargo fmt --check
    cargo test --locked -p symthaea-operations-research

Keep the exact source SHA, full stdout/stderr, exit code, toolchain, hardware, scenario/input digests, nominal masses and TV radius with every promoted result. A skipped workflow is not a pass.
