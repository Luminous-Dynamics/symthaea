# PHOT-CTRL-002A: adaptive trajectory acquisition by ranking stability

Status: contract implementation draft; Rust execution and held-out optimization evaluation remain outstanding.

Issue: [PHOT-CTRL-002 #8720](https://github.com/Luminous-Dynamics/symthaea/issues/8720). Related implementation: [PR #8719](https://github.com/Luminous-Dynamics/symthaea/pull/8719).

## Why this is worth testing

A common evaluation protocol gives each optimization method a fixed number of expensive sampler calls. That is simple, but a hybrid controller may spend too much on easy instances and too little on difficult instances. An alternative is to allocate the sampling budget according to whether new trajectory batches continue to change the search-order recommendation.

This is deliberately a narrower claim than general adaptive experimental design. qReduMIS already repeatedly queries a quantum coprocessor while interleaving exact classical MIS reductions (Phys. Rev. Research 8, 033296, 2026: https://doi.org/10.1103/nh3b-1wv5). The present hypothesis is only that a predeclared downstream ranking-stability gate may save sampling cost without degrading downstream verified quality.

The gate is *not* a sequential hypothesis test, a confidence statement about optimal-set membership, or a correctness certificate. A wrong ranking can be stable. Every decision to fix a variable, prune a branch, or claim optimality remains in a separate exact/classical proof layer.

## Rust contract

The module trajectory_budget.rs defines:
- a frozen configuration with vertex count, top-k length, minimum attempted runs, minimum valid projections, minimum projection yield, required stable batches, and maximum attempted-run budget;
- a batch record with an ID, attempted count, valid-projection count, and an optional complete vertex ordering;
- derived cumulative yield (callers cannot supply a misleading ratio);
- strict input validation for duplicate IDs, invalid counts, out-of-range/repeated vertices, malformed ranking length, and cap overruns;
- a stability streak over the ordered top-k signature;
- a missing/unavailable recommendation that resets the stability streak;
- three distinct decision statuses: Continue, StableRanking, and BudgetExhausted.

StableRanking means the declared sampling thresholds are met and the recommendation has been identical for the required consecutive batches. It says nothing about whether that ranking is good. BudgetExhausted is distinct, even if a ranking happens to look stable but one of the minimum conditions is unmet.

The code never acts on the graph, changes a candidate, removes a vertex, or certifies an optimum. It returns only a sampling-control status plus an auditable snapshot.

## Predeclared comparison

For every identical, content-addressed problem instance compare:
1. fixed K trajectory batches;
2. the rank-stability early-stop gate;
3. no trajectory calls / degree-only ordering;
4. final-spin-frequency ordering;
5. time-resolved trajectory ordering;
6. shuffled-order and seeded-random controls.

The maximum permitted acquisition and downstream solver budgets must match. All calls—including failed, ambiguous, timed-out, and invalid-projection attempts—count toward the budget. Measure sampler time, controller time, transfer, downstream solver time, repair, verification, and instrumented energy where possible.

Use separate training and confirmation cohorts. Pin public QOBLIB MIS instance digests and use graph-level paired statistics; repetitions on one graph are not independent graph samples.

## Required metrics

- attempted / valid / ambiguous / failure / timeout counts;
- cumulative projection yield and number of batches used;
- stop status and the snapshot that triggered it;
- downstream verified objective/cardinality, feasibility, exact proof status, and target attainment;
- all-in wall time and energy boundary (measured or unknown);
- paired win/tie/loss counts and confidence intervals at the graph level.

## Tests and known-answer controls

The contract includes tests for:
- all minimum thresholds being required;
- rank changes resetting stability;
- unavailable ranking resetting stability and remaining in the attempted denominator;
- cap exhaustion remaining distinct from stable stopping;
- a deliberately arbitrary but stable ranking demonstrating that stability is not a correctness certificate;
- malformed batch IDs, counts, rankings, and cap overrun.

These are mechanism tests, not evidence that adaptive acquisition improves MIS solving. A positive experimental result must compare adaptive and fixed-K policies on the same held-out instances. If apparent savings disappear after charging acquisition, transfer, controller, solver, and verification costs—or if solution quality degrades—the heuristic should be stopped.

## Acceptance gates

1. Pass the pinned Rust build, formatting, and test checks at the exact PR head.
2. Replay every gate decision from the batch ledger and frozen configuration.
3. Run the fixed-K / adaptive arms against identical graph bytes and strong classical baselines.
4. Keep solver correctness independent from sampling decisions.
5. Publish negative or inconclusive results without tuning against the confirmation set.

## No claims

No hardware run, cost saving, solver speedup, SOTA result, calibrated uncertainty, physical energy advantage, or quantum advantage is claimed by this contract.
