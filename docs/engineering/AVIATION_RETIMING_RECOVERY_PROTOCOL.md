# AVIATION-ENTERPRISE-016 — Delay-aware recovery protocol

## Purpose and evidence boundary

This is a second, separately versioned synthetic model intended to test recovery choices with a nontrivial cost trade-off. The earlier AECB-001 model remains a useful provenance/feasibility fixture, but its objective is only fixed source/maintenance cost and cancellation penalties. This model adds discrete departure delay and per-minute delay penalties so the optimizer must trade delayed flights against cancellation and resource conflicts.

It remains a small research model. It does not represent crew/passenger legality, airport capacity, airspace, weather, airline-specific policies, real disruption streams, or operational dispatch. Its output is not an operational recommendation.

## Model semantics

For each flight f, freeze:

- scheduled start s_f, duration d_f > 0;
- maximum integer delay m_f;
- delay cost per minute r_f >= 0;
- cancellation penalty c_f >= 0.

Each action is either:

1. assign the flight to a declared aircraft and choose x_f in {0,...,m_f}, or
2. explicitly cancel it.

An assigned flight occupies the interval [s_f + x_f, s_f + x_f + d_f). Every assigned flight must respect aircraft availability, and consecutive flights assigned to the same aircraft must respect its turnaround time. A cancelled flight carries no delay and incurs its cancellation penalty.

The toy objective is:

\[
C = \sum_{f \in \text{assigned}} r_f x_f
  + \sum_{f \in \text{cancelled}} c_f
\]

All times and objective calculations use checked integer arithmetic. No fractional cost scaling is implied.

## Why the lower bound is admissible

For a partial search node, the implementation sums the costs of committed actions and adds an independent optimistic minimum for every unresolved flight:

\[
L(n) = C_{\mathrm{committed}}(n) + \sum_{f \in \mathrm{unresolved}(n)} \ell_f
\]

For each unresolved flight, \(\ell_f\) is the minimum of its cancellation penalty and its least delay cost required to meet any declared aircraft's individual availability boundary (if that delay is within the flight's maximum). The relaxation ignores conflicts among unresolved flights and between unresolved and committed flights. Ignoring those conflicts can underestimate the true completion cost, but cannot overestimate it; every valid completion must pay at least the independent per-flight minimum. The lower bound is specific to this non-negative delay/cancellation objective. It cannot be re-used for an objective containing negative rewards, offsets, rebates, or future decisions that reduce already committed cost without a new proof.

The solver explores the full declared action universe: every aircraft and every integer delay from zero through the flight's maximum, plus cancellation. It checks partial availability/turnaround constraints before enqueueing a node. Stable priority/sequence ordering makes runs deterministic.

For the best independently verified feasible plan with cost U, and minimum frontier lower bound L, the global bound and gap are:

- LB = min(U, L);
- G = U - LB;
- the run may report OptimalWithinTolerance only if G <= epsilon, for a frozen absolute cost tolerance epsilon.

If a generated-node cap cannot fit all children for an expansion, the parent is returned to the frontier unchanged. This preserves coverage of its unexplored subtree rather than silently dropping children. A cap that ends the run before the requested gap is proved does not produce an optimality claim.

## Independent validation

The production verifier resolves stable flight and aircraft IDs, recomputes delayed intervals, checks availability and turnaround, and independently recomputes delay/cancellation cost. It rejects missing/duplicate/unknown IDs, stale scenario identity, over-maximum delay, nonzero delay on cancelled flights, time/cost overflow, constraint violations, and tampered objective totals.

The integration oracle independently enumerates each aircraft/delay/cancel choice directly from raw scenario fields. It does not call the production search's partial-feasibility or cost functions. Its partial-budget cases also compare the reported relaxed frontier lower bound to the independent full optimum and include an availability boundary designed to force a nonzero lower bound. Required future property tests include:

- zero-disruption/non-conflicting cases;
- same-start conflicts requiring a delayed assignment;
- maximum-delay and availability boundaries;
- alternative aircraft and turnaround sensitivity;
- zero/high delay rates and cancellation penalties;
- missing/duplicate/unknown IDs and tampered objective;
- generated-node cap and expansion-budget edge cases;
- schedule mutation and input-order determinism;
- zero false optimality claims when the generated-node cap interrupts expansion.

The current test suite defines these categories in a small set of test functions, including an oracle sweep across varied scenarios. Test definitions are not evidence of execution until the exact source revision has been compiled and run.

## Baselines and scorecard

The baseline portfolio now includes (1) a full Cartesian exhaustive oracle on tiny instances, independently implemented in test code; (2) a deterministic greedy repair that considers flights in scheduled-time/stable-ID order and selects the lowest-cost locally feasible delay/aircraft action, cancelling when cancellation is cheaper; and (3) the matched full-tree branch-and-bound search. The greedy baseline never revisits earlier assignments, so it is a transparent heuristic rather than a quality claim. All arms must use the same frozen input, discrete delay domain, feasible action space, cost function, and stopping budget.

Report separately:

- exact-oracle best cost, greedy best cost, and branch-and-bound best cost;
- best feasible total cost, delay minutes and cancellations;
- global lower bound, absolute gap, and terminal state;
- expanded/generated nodes, complete assignments examined, partial-feasibility prunes, objective-bound prunes, and peak frontier;
- time to first independently verified feasible plan (in the current future runner; node counts are available now);
- greedy candidate actions considered and how many were feasible;
- end-to-end latency p50/p95/p99 over repeated runs on a named hardware/toolchain profile;
- completion/timeout rate and exact-optimum agreement on tiny instances.

Do not infer a wall-clock win from fewer nodes alone. The generated-node cap is part of the resource budget and must be identical across compared policies. Retain null/failed runs.

## Industry context

Lufthansa Group's schedule-recovery case study describes aircraft/crew/passenger subsolvers combined with large-neighborhood search, a feasibility-preserving heuristic, column generation, and LP-relaxation-derived costs (INFORMS Journal on Applied Analytics, DOI: https://doi.org/10.1287/inte.2025.0296). MIT crew-recovery research combines MIP with supervised-learning-derived cuts and tunable search-space reduction on real schedules (MIT thesis, 2024: https://hdl.handle.net/1721.1/153670). A 2026 aircraft-recovery paper reports an average 66% end-to-end runtime reduction at comparable solution quality with recovery-aware features on five major Chinese airlines (DOI: https://doi.org/10.1016/j.trb.2026.103566).

These are useful research references, not comparable measurements for this synthetic model. The 66% figure is a possible future ambition only after the objective, constraints, runtime accounting, hardware profile, and optimality/quality tolerance are aligned. No industry-SOTA, runtime, airline-savings, operational-safety, dispatch-authority, or certification claim is made here.

## Commands to run before promoting the benchmark

    cargo fmt --check
    cargo test --locked -p symthaea-operations-research

Then preserve the exact source SHA, complete stdout/stderr and exit code, scenario/input digests, toolchain and hardware metadata. A skipped workflow is not a pass.
