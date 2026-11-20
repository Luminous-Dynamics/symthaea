# Objective-matched finite-horizon causal-probing oracle — 2026-10-23

## Why this follow-up exists

The first finite-horizon oracle optimized joint graph/action-kernel entropy. Measuring graph-only or kernel-only policies against that oracle is not a fair regret comparison: those policies optimize a different objective. This extension builds a separate exact Bellman oracle for each terminal objective and evaluates every policy against each score endpoint. The benchmark is still a reduced exact pilot, not external qualification or a state-of-the-art claim.

## Protocol

- **16 joint hypotheses:** chain or fork graph × eight assignments of three action-specific kernels, with each off-target probability either 0.1 or 0.6.
- **Actions:** attempt one of three targets. Its realized subset is either the intended singleton or the adjacent pair; all observations include the exact realized subset and the 3-bit SCM outcome.
- **Prior:** uniform over the 16 graph/kernel assignments.
- **Horizons:** 1, 2 and 3 interventions.
- **Policies:** round-robin, uniformly random action choice, greedy graph information, greedy kernel information, greedy joint information, coverage-first versions of each greedy objective, and exact Bellman oracle policies for each objective.
- **Three score endpoints:** graph entropy (initial 1 bit), joint action-kernel entropy (initial 3 bits), and joint graph/kernel entropy (initial 4 bits).
- **Oracle:** exact Bellman recursion for the selected terminal objective. Each policy is evaluated against the oracle for the same scored objective. Full posterior tuples, not rounded posterior summaries, are used as cache keys.
- **Output:** 99 rows (11 policies × 3 scored objectives × 3 horizons). The source/test files are bound by Git blob IDs in the evidence receipt.

## Horizon-three results

Regret is expected terminal entropy for the policy minus the exact oracle's expected terminal entropy for the **same** objective. Lower is better. Each objective has a separate oracle.

### Graph entropy objective

| Policy | Information gain (bits) | Regret (bits) |
|---|---:|---:|
| Greedy graph information | **0.412716** | **0.000000** |
| Exact graph oracle | 0.412716 | 0.000000 |
| Exact joint oracle | 0.373086 | 0.039630 |
| Greedy joint information | 0.345538 | 0.067178 |
| Round-robin | 0.200333 | 0.212382 |
| Uniform random | 0.189804 | 0.222912 |

### Kernel entropy objective

| Policy | Information gain (bits) | Regret (bits) |
|---|---:|---:|
| Round-robin | **0.642285** | **0.000000** |
| Greedy kernel information | **0.642285** | **0.000000** |
| Coverage-first policies | 0.642285 | 0.000000 |
| Exact kernel oracle | 0.642285 | 0.000000 |
| Greedy joint information | 0.595034 | 0.047251 |
| Uniform random | 0.590357 | 0.051928 |
| Exact joint oracle | 0.568567 | 0.073718 |
| Greedy graph information | 0.496755 | 0.145530 |

### Joint graph/kernel entropy objective

| Policy | Information gain (bits) | Regret (bits) |
|---|---:|---:|
| Exact joint oracle | **0.941652** | **0.000000** |
| Greedy joint information | 0.940572 | 0.001080 |
| Greedy graph information | 0.909471 | 0.032181 |
| Round-robin | 0.842618 | 0.099034 |
| Greedy kernel information | 0.842618 | 0.099034 |
| Uniform random | 0.780161 | 0.161491 |

“Coverage-first policies” tie with round-robin on the kernel-only objective in this reduced symmetric model. The complete exact JSON contains every policy/objective/horizon combination, including cross-objective results.

## Interpretation

1. **Regret must be objective-matched.** Graph-only and kernel-only strategies are not fairly judged by the joint-entropy optimum. This follow-up closes that comparison gap.
2. **No one policy wins every objective.** Greedy graph information is optimal for graph entropy at horizon three in this specific game; kernel entropy has a large tie among policies; greedy joint information is only 0.00108 bits behind the joint oracle.
3. **The exact oracle is a calibration instrument, not proof of universal superiority.** Results apply only to 16 hypotheses (two graph types and two spill probabilities independently assigned to three attempted targets) with perfect subset telemetry. Different priors, graph families, telemetry errors and objectives can change rankings.
4. **The wider adaptive pilot still matters.** In the 330-kernel finite-grid benchmark, greedy graph information concentrated probes on one target and had poor kernel estimation. These small cases show why the scorecard must preserve graph, kernel and joint outcomes separately.

## Research and industry comparison

Off-target stochastic interventions and action-dependent affected-subset distributions are established prior art (Choo, Shiragur & Uhler, AISTATS 2024). ACDB (September 2026) provides a budgeted-intervention evaluation ladder; its current report says PC with greedy active orientation has 42.7% directed F1 on its six-level ladder and emphasizes calibration issues. CausalArena (September 2026) reports substantial ranking shifts across SCM families and protocols, while TCD-Arena (ICLR 2026) stress-tests 33 assumption violations. Those contracts differ from this reduced exact game, so the results here are not directly comparable. No published baseline or external suite has been executed in this branch.

References:
- Choo et al., AISTATS 2024: https://proceedings.mlr.press/v238/choo24a.html
- ACDB, September 2026: https://arxiv.org/abs/2609.31675
- CausalArena, September 2026: https://arxiv.org/abs/2609.11897
- TCD-Arena, ICLR 2026: https://tcd-arena.github.io/

## Next qualification gates

- Extend the oracle to more graph families and richer action-kernel priors while retaining an exact small-instance control.
- Compare each selection policy to the exact oracle for the same terminal objective, over multiple priors and horizons.
- Add noisy/missing subset telemetry, off-grid mechanisms, heterogeneous intervention values, and equal cost/damage budgets.
- Run external baselines under matched semantics; report SHD/skeleton and directed F1, calibration/log loss/Brier, kernel L1/KL, intervention cost/disturbance, runtime and memory, plus family-level uncertainty intervals.

## Reproduction and evidence

From repository root, set PYTHONPATH=validation, run unit-test module test_objective_matched_oracle_benchmark, and execute objective_matched_oracle_benchmark.py --horizon 3 with --source-git-blob f101346c2f37cfad4e029ae280e9a002548f309c and an output path. Local suites for this and the three earlier benchmark modules pass **30/30 tests**. The 99-row artifact is 37,662 bytes, SHA-256 00e02e219779d4ef1cf779d32760746e0e564c3ff382966aa4d3beb47c9cedc5, bound to the source Git blob. Local tests are not remote CI evidence.