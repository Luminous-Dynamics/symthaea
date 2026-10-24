# Policy-Tree Column Generation: Finite-Model Benchmark (2026-10-24)

## Result

This exploratory lane replaces exhaustive Pareto-frontier enumeration in the *solver* with a restricted minimax master LP and a scalarized dynamic-programming (DP) pricing oracle. The exhaustive fixed-prior frontier remains an independent small-instance comparator. Each generated policy tree is independently re-scored against every candidate true-prior scenario before being added to the restricted master.

The local run covers 36 cases: six candidate priors × six learner priors × three objectives (graph, kernel, joint) × horizons 1–2. All 36 reached the specified primal/pricing certificate tolerance.

| Measure | Result |
|---|---:|
| Maximum primal upper bound minus pricing lower bound | 8.881784197001252e-16 bits |
| Maximum absolute difference vs exhaustive frontier + mixture LP | 8.881784197001252e-16 bits |
| Largest exact horizon-two kernel frontier | 450 policy trees |
| Column-generation trees retained in that case | 1 |
| Maximum number of trees in any horizon-two case | 2 |
| Overall mean runtime: column generation | 0.0731 s/row |
| Overall mean runtime: exhaustive frontier + LP | 0.0366 s/row |

The dominant kernel case compressed 450 frontier policies to one generated column and took about 0.120 s versus 0.386 s for exhaustive enumeration plus LP in this run. However, across all 36 rows column generation was about 2.56× slower on average. This is **not an overall speedup claim**: timing is environment-, cache-, and setup-dependent, and the current benchmark is a small finite toy problem.

## Method and certificate

At each iteration, solve a restricted minimax LP over discovered policy-tree risk vectors. Its dual scenario weights define a scalarized terminal-cross-entropy objective. An exact finite-horizon DP pricing oracle returns a minimum scalarized-risk policy tree. Independently re-score the returned tree under every candidate prior; add it only if new. Termination requires a closed primal-to-pricing gap within tolerance. The full Pareto frontier is used by the benchmark harness only as an independent oracle, not by the column-generation policy solver.

The candidate true-prior identity is fixed for the episode and hidden from the policy. Scores are expected terminal categorical cross-entropy in bits. LP primal/dual residuals and exact-tree re-scoring are checked.

## Provenance and qualification boundary

- Local result JSON: 36 rows, 74,511 bytes, SHA-256 `e48554121720fc480aeeed5f63cf7144e51d2fe8fdbee57731105bd21d9476f7`.
- The executed local benchmark source had SHA-256 `08371649907d5a3252036789a7f774f3f01b9ff57143467c0001d15c3d116409` and computed Git blob `1f2656c5f49a375043ffe42155df11952a27217a`.
- The committed branch implementation is Git blob `98cb554d3a474cb3c6c398049845519f211508bd`. It is a slightly revised rendition, **not byte-identical** to the source used to generate the result JSON. The committed test file is byte-identical to the locally tested test source (Git blob `e99ce5c1c5a994c4646e190470a8fdb156371e49`).
- Local Python 3.13.5 regression: 78/78 tests passed across nine modules. NumPy 2.3.5 / SciPy 1.17.0 are research-only. The local runtime had no checkout; the fixed-prior helper was reconstructed from the displayed source and was not byte-identical to repository blob `438301f07649d7ffebca6690990a4909ea28e054`. Therefore these tests validate the algorithmic prototype, **not exact-head integration**.
- No external causal-discovery suite was executed. No remote CI pass, SOTA result, novelty claim, or general scaling guarantee is claimed.

## Next experiments

1. Run exact-head integration tests in the actual repository environment and regenerate the artifact from the committed source blob.
2. Scale the candidate scenario count (6, 12, 24, 48), horizon (1–5), and graph/mechanism families while recording frontier size, DP states, peak memory, median/p95 runtime, and certificate gaps.
3. Add intervention costs, noisy/missing/delayed target telemetry, latent confounding, and task disturbance. Fail closed whenever the pricing oracle does not certify its bound.
4. Evaluate matched baselines on [ACDB](https://arxiv.org/abs/2609.31675), slice results across [CausalArena](https://arxiv.org/abs/2609.11897), and use [TCD-Arena](https://tcd-arena.github.io/) for assumption-violation stress tests. Report skeleton F1, directed precision/recall/F1, SHD, intervention efficiency, posterior log loss/Brier/calibration, cost, and regime-wise intervals. Toy cross-entropy bits are not comparable to those structural metrics.

## Relevant prior art

- Li, Ju, and Shroff, [How to Find the Exact Pareto Front for Multi-Objective MDPs? (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f8290ccc2905538be1a7f7914ccef629-Abstract-Conference.html). Its polytope/neighbor structure is motivation, not assumed to hold automatically in this partially observed causal-probing game.
- Choo et al., [Causal Discovery under Off-Target Interventions (AISTATS 2024)](https://proceedings.mlr.press/v238/choo24a.html), establishes stochastic off-target intervention subsets as prior art.


## Additional horizon-three pricing-only slice (2026-10-24)

Six additional cases were executed directly through column generation at horizon 3, without constructing the exact Pareto frontier: two learner priors (`uniform`, `graph_kernel_correlated`) × three objectives. **6/6 converged** with tolerance `1e-8`; maximum primal/pricing gap was `2.0295e-13` bits. The cases used 1–7 policy-tree columns and ran in 1.57–6.68 seconds on this machine. This is a pricing-certificate slice only; it is not a full horizon-three grid or an exhaustive-frontier comparison.

Local-only JSON: `policy_frontier_column_generation_horizon3_2026-10-24.json`, 3,084 bytes, SHA-256 `fd809bf5bdaaadfee6f15ae03c682099227259313dd48843c3cdd2868dd8a1a1`. It binds to local source Git blob `1f2656c5f49a375043ffe42155df11952a27217a`, not the compacted source blob committed to this branch. Exact-head integration is still required.
