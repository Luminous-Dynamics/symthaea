# Robust ambiguity-set policy benchmark — 2026-10-24

## Question

Can an intervention policy improve prediction quality when the learner's assumed prior is wrong, without being told the true prior? The previous lane showed that prior misspecification can cause a learner's graph predictions to get worse after collecting evidence. This lane tests a deployable policy candidate against fixed adaptive and non-adaptive baselines.

## Policy under test

For every available action, \`robust_minimax_one_step\` calculates the expected terminal cross-entropy after one more observation under each currently plausible candidate-world posterior, then chooses the action with the lowest **worst-case** score. Mean score and action index break ties. After observing the realized target subset and the three-bit outcome, the learner posterior and every non-zero-likelihood ambiguity scenario are updated. The action-selection function receives the learner posterior and the ambiguity set, but not the evaluator's true-prior label.

Baselines: greedy joint-information gain, round-robin, uniform random, and a learner-posterior Bellman entropy policy. The last is an entropy-planning baseline, not a true-prior-aware oracle. All policies use the same exact 16-hypothesis chain/fork model, target-subset telemetry, priors and probe budget.

## Exact evaluation grid

The full run contains 1,080 rows: six true-world priors × six learner priors × three scored objectives × five policies × horizons 1–2. At horizon two there are 30 mismatched and six matched prior pairings for each objective. A separate horizon-three boundary slice contains 180 rows for the four matched/mismatched pairings among uniform and high-spill-biased priors, with all three objectives and five policies at horizons 1–3.

Metrics are expected terminal proper categorical cross-entropy (bits) and Brier risk. The values are exactly integrated over observation branches in this finite model, not Monte Carlo estimates. Lower cross-entropy is better. These scores assess posterior prediction, not direct graph-recovery F1/SHD.

## Results: horizon 2, mismatched priors

Mean and worst-case terminal cross-entropy across the 30 mismatched true/learner-prior pairings:

| Objective | Policy | Mean CE (bits) | Worst CE (bits) | Mean CE reduction (bits) | Worst CE reduction (bits) |
|---|---|---:|---:|---:|---:|
| Graph | Robust minimax one-step | 0.858556 | 1.609473 | 0.361741 | 0.112542 |
| Graph | Greedy joint | 0.897715 | 1.708013 | 0.322582 | 0.000899 |
| Graph | Round-robin | 0.994745 | 1.866977 | 0.225552 | -0.037094 |
| Graph | Random | 1.079059 | 2.035729 | 0.141238 | -0.073213 |
| Kernel | Robust minimax one-step | 3.168323 | 6.075885 | 0.487364 | 0.317434 |
| Kernel | Greedy joint | 3.194939 | 6.177194 | 0.460749 | 0.063832 |
| Kernel | Round-robin | 3.147793 | 6.075885 | 0.507895 | 0.240136 |
| Kernel | Random | 3.163253 | 6.109654 | 0.492434 | 0.289955 |
| Joint | Robust minimax one-step | 4.188293 | 6.785929 | 0.853859 | 0.588478 |
| Joint | Greedy joint | 4.205072 | 6.917406 | 0.837079 | 0.606409 |
| Joint | Round-robin | 4.263053 | 6.824400 | 0.779099 | 0.537128 |
| Joint | Random | 4.370710 | 6.943428 | 0.671442 | 0.476258 |

The minimax candidate improves mean graph cross-entropy over greedy joint (0.85856 vs 0.89772 bits) and mean joint cross-entropy (4.18829 vs 4.20507) in this grid. It does **not** dominate across objectives: round-robin has lower mean kernel cross-entropy (3.14779 vs 3.16832). This is an objective trade-off, not a universal win.

Pairwise horizon-two results further qualify the picture. On the 30 mismatched pairs, robust minimax beats greedy joint on graph CE in 20, ties in 10, and loses in none; on joint CE it beats greedy joint in 21 and loses in nine. For kernel CE it beats greedy joint in 16 and loses in 14. It beats round-robin on joint CE across all 30 pairs, but loses to round-robin on kernel CE in 19 pairs (six wins, five ties). A single aggregate metric would hide this trade-off.

## Results: horizon 3 boundary slice

Only two mismatched prior pairings are represented, so this is a boundary check rather than a broad result. On joint CE, robust minimax has mean/worst terminal CE of 3.732584 / 4.300840 bits; greedy joint has 3.694367 / 4.256389. Robust minimax is worse here. For graph CE, it beats greedy joint (mean 0.633019 vs 0.678327); for kernel CE, round-robin beats robust minimax (2.981286 vs 3.003574). This is direct evidence that the one-step minimax objective is not universally optimal over longer horizons or every score.

## Interpretation and next paradigm shift

The most defensible conclusion is that **prior-robust intervention selection is promising but objective-dependent**, and a simple worst-case one-step action rule is not yet enough. A stronger next iteration should compare:

1. a multi-step robust Bellman policy over a declared ambiguity set;
2. calibrated prior mixtures / weights rather than an unweighted finite list of candidate priors;
3. a risk-sensitive objective that prices graph, kernel and joint errors according to downstream task impact;
4. held-out prior mixtures and held-out graph/mechanism families, including collider, nonlinear, temporal and confounded cases;
5. noisy or missing intervention telemetry, action costs, disturbance and stopping rules.

The ambiguity set in this test is just six strictly positive prior candidates. It is not a statistically calibrated uncertainty region, and the policy is a greedy one-step minimax heuristic rather than a multi-step robust-optimal policy. The true prior is one of the evaluated candidate priors; this experiment does not establish robustness to missing graph/mechanism support.

## Benchmarking against the field

Robust optimization is established prior art; no novelty claim is made for minimax or ambiguity-set optimization. ICML 2026 work on Distributionally Robust Causal Abstractions uses Wasserstein ambiguity sets and addresses environmental, structural and intervention-mapping misspecification. Our narrower question is adaptive intervention selection under prior uncertainty, which still requires a comparison with relevant baselines and proper external evaluation.

ACDB (September 2026) evaluates active causal discovery under budgeted interventions and separates skeleton recovery, directed-DAG recovery and intervention efficiency. Its current report lists PC with a greedy active-orientation heuristic at directed F1 42.7% / SHD 4.79, but those numbers **must not be compared numerically** with our toy-model cross-entropy values. TCD-Arena (ICLR 2026) tests time-series causal methods under 33 assumption violations; CausalRivers offers real-world hydrology time-series evaluation. Those suites motivate the next gates; none was executed in this lane.

- Distributionally Robust Causal Abstractions (ICML 2026): https://proceedings.mlr.press/v306/felekis26a.html
- ACDB: https://arxiv.org/abs/2609.31675
- TCD-Arena (ICLR 2026): https://tcd-arena.github.io/
- CausalRivers (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference.html

## Reproducibility and status

- Source: \`validation/robust_ambiguity_policy_benchmark.py\`; Git blob \`c511c9651de071458734e014137484d982c717dc\`; local SHA-256 \`890f2cd8598595bc37793d742d8af970b4e2cb4caa2d8daf8ac3f167cc0ba1c5\` (12,315 bytes).
- Tests: \`validation/test_robust_ambiguity_policy_benchmark.py\`; Git blob \`a45fd468130890974e694b0f52c70686434ac04f\`; local SHA-256 \`b7b798dd1d22b52e789b6bd586426a2acaf56df157b257f812ee195cc006061e\` (3,159 bytes); eight dedicated tests pass.
- Full horizon 1–2 result artifact: 1,080 rows, 675,036 bytes, SHA-256 \`6997bb7f001bd7505e182293f73c8f6e993a548896d0173c07b70759cc9755e9\`.
- Horizon-three boundary result artifact: 180 rows, 111,925 bytes, SHA-256 \`b26d170e8f157cdec13702fcd63b6062a1d0c3254bc644ca481cf39c4cd79490\`.
- Combined local regression suites: **51/51 pass** (Python 3.13).
- Exact JSON result artifacts are local-only; compact summary and source-bound receipt are committed to the research branch.
- The result is a finite exact toy-model evaluation. No external benchmark was run; no SOTA, novelty or production-readiness claim is made.
