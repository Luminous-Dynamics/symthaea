# Fixed-prior minimax policy frontier — 2026-10-24

## Research question

Can an adaptive intervention policy be optimized against a set of candidate true-world priors while keeping each prior fixed for the entire experiment? The previous one-step minimax policy showed some mean improvements, while rectangular robust Bellman planning was overly conservative when scored against a single true prior that persists throughout an episode.

## Paradigm shift: optimize a policy tree, not a sequence of myopic scores

The new solver builds a finite-horizon contingent policy tree. Each candidate true prior contributes one expected terminal cross-entropy component to a risk vector. Candidate trees are combined observation by observation; Pareto-dominated risk vectors are pruned; the selected root tree minimizes worst-case expected cross-entropy across the original fixed candidate priors, with mean risk and deterministic risk-vector ordering as ties.

Crucially, the *identity of the true prior is fixed for the whole episode*. The policy is not given that identity. It receives the learner posterior and observable history; the solver accounts for each candidate prior's different predictive distribution and posterior update. This differs from rectangular ambiguity, which may choose a different worst-case scenario independently at each branch.

This is an exact policy-tree solution for a small finite model, not a general tractable method: frontier growth is part of the measurement.

## Full exact grid: horizons 1–2

The complete grid contains 1,080 rows: six candidate true priors × six learner priors × three scored objectives (graph, intervention-kernel assignment, joint hypothesis) × five policies × horizons one and two. The five policies are fixed-prior minimax frontier, one-step minimax, greedy joint information, round-robin, and uniform random. Thirty true/learner pairings are mismatched for each objective and horizon.

Mean / worst terminal cross-entropy (bits) over the 30 mismatched prior pairings at horizon two; lower is better:

| Score | Policy | Mean CE | Worst CE |
|---|---|---:|---:|
| Graph | **Fixed-prior minimax frontier** | **0.858556** | **1.609473** |
| Graph | One-step minimax | 0.858556 | 1.609473 |
| Graph | Greedy joint | 0.897715 | 1.708013 |
| Graph | Round-robin | 0.994745 | 1.866977 |
| Graph | Random | 1.079059 | 2.035729 |
| Kernel | **Fixed-prior minimax frontier** | **3.140296** | **6.075885** |
| Kernel | One-step minimax | 3.168323 | 6.075885 |
| Kernel | Greedy joint | 3.194939 | 6.177194 |
| Kernel | Round-robin | 3.147793 | 6.075885 |
| Kernel | Random | 3.163253 | 6.109654 |
| Joint | **Fixed-prior minimax frontier** | **4.185087** | **6.785929** |
| Joint | One-step minimax | 4.188293 | 6.785929 |
| Joint | Greedy joint | 4.205072 | 6.917406 |
| Joint | Round-robin | 4.263053 | 6.824400 |
| Joint | Random | 4.370710 | 6.943428 |

The fixed-prior frontier matches the one-step minimax on mean graph score in this model and modestly improves mean kernel and joint scores. It improves over greedy joint and random on the displayed mean graph/kernel/joint scores. This does not make it the best policy for every individual true-prior pair; the per-row exact data is available for inspection.

At horizon two, the maximum Pareto-frontier size observed during exact search was 4 for graph score, 450 for kernel score, and 12 for joint score, with 1,542 memoized states for each objective. This is a practical warning: the policy-tree frontier is tractable here, but kernel risk has a much larger non-dominated set and can dominate future runtime/memory.

## Horizon-three boundary slice

The six-prior, three-objective horizon-three run exceeded the local execution limit. It is **not** claimed as completed. Instead, a bounded exact slice used two candidate true priors (uniform and high-spill-biased), two learner priors (the same two), joint scoring, and horizons one through three: 60 rows.

For the two mismatched prior directions at horizon three, mean / worst terminal joint cross-entropy was:

| Policy | Mean CE | Worst CE |
|---|---:|---:|
| **Fixed-prior minimax frontier** | **3.644765** | **4.212759** |
| One-step minimax | 3.653549 | 4.212759 |
| Greedy joint | 3.694367 | 4.256389 |
| Round-robin | 3.807256 | 4.346340 |
| Random | 3.860326 | 4.413648 |

This is a narrow diagnostic (two mismatches), not a broad generalization result. It suggests the tree can exploit contingent second-step actions better than a one-step rule in this slice, but a pre-registered holdout is needed before making a performance claim.

## What this does and does not establish

- The implementation exactly evaluates one finite contingent policy tree against fixed candidate priors and validates that each reported scenario risk matches an independent rollout calculation.
- It makes the uncertainty semantics explicit: a persistent unknown prior differs from branchwise rectangular ambiguity.
- It demonstrates promising but small mean gains versus several baselines in this 16-hypothesis game.
- It does not test missing graph/mechanism support, real data, noisy target-subset telemetry, intervention costs, or action-induced task disturbance.
- Six prior candidates do not constitute a calibrated ambiguity region. No SOTA, novelty, or production-readiness claim is made.

## Scaling gate

Run the frontier with an increasing candidate-set size and horizon while recording maximum frontier size, memoized states, runtime, peak memory, and the gap to the best returned upper/lower bounds. Develop epsilon-dominance or convex-frontier approximations only after the exact solver supplies correctness oracles. For deployment, compare a certified/anytime approximation against fixed-prior minimax, one-step minimax, robust POMDP, and active causal discovery baselines.

## External comparison and prior art

Policy-tree search, multiobjective Pareto frontiers, robust POMDPs, and distributionally robust optimization are established concepts. Osogami's robust POMDP work uses dynamic programming/value iteration for uncertain transition parameters; Felekis et al. (ICML 2026) introduce Wasserstein ambiguity sets for distributionally robust causal abstractions. These provide relevant conceptual and methodological baselines—not evidence that this specific causal intervention policy is novel.

For external evaluation, ACDB (September 2026) separates skeleton recovery, directed-DAG recovery, and intervention efficiency under budgeted interventions. CausalArena emphasizes shifts across synthetic, semantic, formula-grounded and real-data SCM regimes. TCD-Arena (ICLR 2026) evaluates time-series methods under 33 assumption violations. None of those external suites was run here; our cross-entropy metrics are not numerically comparable to their F1/SHD scores.

- Robust POMDP (Osogami, ICML 2015): https://proceedings.mlr.press/v37/osogami15.html
- Distributionally Robust Causal Abstractions (Felekis et al., ICML 2026): https://proceedings.mlr.press/v306/felekis26a.html
- ACDB: https://arxiv.org/abs/2609.31675
- CausalArena: https://arxiv.org/abs/2609.11897
- TCD-Arena: https://tcd-arena.github.io/

## Reproducibility

- Source: validation/fixed_prior_minimax_policy_benchmark.py; Git blob 438301f07649d7ffebca6690990a4909ea28e054; SHA-256 676b1797209e46ecbb42284370063d99e60c0de207df27c6ea1dc2c984c1290a; 14,024 bytes.
- Tests: validation/test_fixed_prior_minimax_policy_benchmark.py; Git blob e316527b7b041c307a08461dc692a1174918bdde; SHA-256 b81fcd4cdc691718650143d59f25f2a5fba72cd8856309286bed15142ada5a9d; 10/10 dedicated tests pass.
- Full horizon 1–2 artifact: 1,080 rows, 415,371 bytes, SHA-256 2249fc06a87778e07f9d23ab18aa6e0cae39ec5815b8a5fca2b4450e47a856e4.
- Horizon-three boundary artifact: 60 rows, 18,593 bytes, SHA-256 c72f37975654d80e51489f4caf0852b3ae0c6e0396763da3f14a9c2856eb6111.
- Combined local regression: **61/61 tests passed** across eight benchmark modules under Python 3.13.5.
- Exact result JSONs are local-only. Source and test Git blob IDs match the executed files.
- No remote CI pass is claimed; skipped workflows are not passes.
