# AR-5B2 — faithful Coral Discovery scheduler baseline v0.1

**Status:** published scheduler equations implemented as a separate baseline; tests authored but not compiled or executed.  
**Tracking issue:** [#9335](https://github.com/Luminous-Dynamics/symthaea/issues/9335)  
**Implementation:** crates/domains/symthaea-architecture-engineering/src/coral_baseline.rs  
**Primary source:** [Coral Discovery, arXiv:2610.33281](https://arxiv.org/abs/2610.33281), sections 3.1–3.3.

## Why a faithful baseline matters

Symthaea's DirectionBoard v0.1 and Coral Discovery share a research-direction allocation idea, but they are not the same algorithm. The Symthaea board currently uses lifetime local-record improvement divided by total evaluation cost, a configurable NEW admission-credit prior, and equal evaluation-count slots.

The published Coral scheduler uses a different credit definition, a sliding window, pseudo-count shrinkage, lifetime-count exploration, a first-k infancy estimator for NEW and agent claim caps. The comparison must preserve both methods as distinct policies.

## Published scheduler equations implemented

Defaults reproduced from the paper:

- window length W = 32;
- yield prior_n = 4 pseudo-evaluations;
- NEW infancy k = 3 evaluations;
- kappa = 0.05;
- four agent seats by default;
- maximum two claimed agents per direction, with one per direction in the first round.

For evaluation X in direction d, the baseline computes:

\`credit(X) = max(0, s(X) - max(s(parent), max(s(inspired-by)), best_d)) / abs(global_best)\`

Minimization objectives must be negated by the caller so every incoming score is a maximization utility. The first implementation treats the global best after including the current evaluated score as the normalization denominator. A zero/degenerate global best produces zero credit rather than division by zero.

For the current global window, let N be its number of evaluations, n_d the number of evaluations in direction d in that window, n_d_all the lifetime count for d, and m the mean credit across all evaluations in the window. Failed evaluations have no score and zero credit but remain in N and n_d.

\`yield_d = (sum(window credit for d) + prior_n * m) / (n_d + prior_n)\`

Then:

\`c = kappa * y_plus\`

\`potential_d = yield_d + c * sqrt(ln(N) / (n_d_all + 1))\`

where y_plus is the highest direction yield observed **at any point during the run**, not merely the highest yield in the current window. The implementation stores this historical maximum and never lets it decay when an old breakthrough leaves the window. The individual yields still use the current W-evaluation window. The lifetime count means a direction that has been neglected does not reset to zero observations.

For each direction opened after a global best existed, the NEW estimator looks at the best score in the first k evaluations compared with the global best at opening. Its infancy gain is truncated at zero, and its infancy evaluation count is min(k, lifetime evaluations).

\`y_NEW = (sum(infancy gains) + k * y_plus) / (sum(infancy counts) + k)\`

\`potential_NEW = y_NEW + c * sqrt(ln(N) / (sum(infancy counts) + 1))\`

Finally, the configured agent seats are distributed proportionally to each direction's and NEW's potential using deterministic largest-remainder rounding. If all potentials are zero, seats go to the global-best direction; when no scored direction exists yet, this implementation assigns the seats to NEW as an explicit bootstrap fallback. Direction claims are limited separately from the advisory wanted counts.

## Implemented interface

The module reconstructs each event's credit and global-best value from the ordered score inputs, checks that direction opening positions and opening global-best values match the relevant event-stream prefix, and verifies that the direction histories and attempt-ID index exactly reconcile with the global event stream. It also reconstructs the historical maximum window yield from every prefix and rejects a tampered exploration scale. The historical peak remains active even after a stalled window contains no new credit.

The module provides:
- versioned CoralBaselineConfig with the paper defaults;
- verifier-labelled CoralEvaluation records with parent and inspired-by scores;
- direction admission capturing the global best at opening;
- exact local-record credit and rolling-window yield;
- lifetime-count potential and first-k NEW estimate;
- deterministic wanted-agent allocation;
- separate first-round and steady-state direction claim caps;
- integrity reconciliation of direction records, global event stream, attempt IDs and global best.

Fourteen tests are authored for formula defaults, local record credit, parent/inspired thresholds, failed evaluation accounting, window expiry, historical exploration-scale retention, NEW-prior retention, NEW infancy, headcount conservation, zero-potential fallback, claim caps, per-evaluation yield semantics, and tamper rejection for credits/global-best receipts, direction opening and direction-history reconciliation.

## Important limitations / deviations

This is a faithful implementation of the published scheduler equations as represented in the paper, not the whole Coral Discovery system.

Not implemented here:
- the tool-less verifier that reads the full candidate diff and assigns mechanism/direction labels;
- semantic merging of provisional duplicate directions;
- agent round orchestration and the "finish your line before reading the board" protocol;
- active claim tracking, verbatim method duplicate checks, worktree checkout and insight recording;
- the exact timeout behavior and asynchronous agent/session management;
- a source-code/evaluator integration for any of Coral Discovery's nine experimental tasks.

The paper's verifier defaults to preserving the parent direction when uncertain and runs serially with a timeout; the scheduler module does not reproduce that verifier. Callers must supply direction labels from an independent verifier. If labels are planted by a synthetic generator, results qualify only formula and scheduler behavior, not semantic direction discovery.

The published scheduler measures credit per evaluation, not per heterogeneous compute unit. The module records evaluation_cost_units for audit but intentionally does not change the paper's yield equation. Compare equal-cost evals first; evaluate resource-aware variants as separate policy arms.

## Matched-budget experiment

The comparison campaign should run:
1. score-only winner-takes-all;
2. fixed equal-share / round-robin;
3. seeded random;
4. global UCB;
5. Symthaea DirectionBoard v0.1;
6. this Coral scheduler implementation;
7. sliding/discounted UCB;
8. MAP-Elites/CVT-MAP-Elites where behavior descriptors are appropriate;
9. an oracle allocator as a synthetic upper bound only.

For every policy, pin identical:
- direction labels and candidate proposal stream;
- task, recipe and evaluator versions;
- model, prompts, context and tool access;
- agent count and total number of evaluations;
- wall clock, CPU/GPU time, memory, token and energy budgets;
- failures, retries, crashes and timeouts in the total-cost denominator.

Report both a mechanism-matched scheduler comparison and a best-practical-system comparison. They answer different questions and must not be combined.

Primary endpoints:
- verified held-out capability at fixed total cost;
- simple regret and target hit probability vs budget;
- time/cost to first verified improvement;
- delayed-breakthrough rate;
- unique verifier-admitted direction count and duplicate rate;
- NEW precision and no-new-direction exploration tax;
- worst-family transfer and resource Pareto frontier.

Before confirmatory claims, use at least 100 paired landscape/payoff instances across six or more independent payoff families. Cluster uncertainty by landscape/payoff template, not by individual repeated evaluations.

## Claim ceiling

A passing test suite would establish formula and bookkeeping fidelity for the implemented scheduler. It would not establish that Coral Discovery or Symthaea is superior. That requires matched-budget runs with independently verified labels and exact evidence receipts. A synthetic payoff simulation is not a real agent replication.
