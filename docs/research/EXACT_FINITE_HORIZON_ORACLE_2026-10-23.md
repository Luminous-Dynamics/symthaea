# Exact finite-horizon causal-probing oracle — 2026-10-23

## Research question

Greedy active experiment policies use one-step expected information gain. How far can they fall behind planning over the full intervention horizon? The oracle reports expected terminal entropy and regret.

This is an exact but deliberately reduced pilot, not a real-world qualification or SOTA claim.

## Protocol

- 16 joint hypotheses: two graph families (chain, fork) × three action-specific kernel choices, each with off-target probability 0.1 or 0.6.
- Three candidate intervention targets; realized subset is either the attempted singleton or an adjacent pair.
- Observation is the exact realized subset and 3-bit outcome.
- Uniform prior, exact Bayesian updates, and exact Bellman recursion over every action and possible observation.
- Horizons of 1, 2, and 3 probes.
- Policies: round-robin, uniformly random, greedy graph information, greedy kernel information, greedy joint information, coverage-first then joint information, and exact Bellman oracle.
- Cache keys use full floating-point posterior tuples; no rounded-posterior merging.

## Results

Expected information gain is the 4-bit prior entropy minus expected terminal entropy. Regret is expected terminal joint entropy minus the exact oracle's terminal entropy. Lower regret is better.

| Horizon | Policy | Joint information gain (bits) | Regret (bits) |
|---:|---|---:|---:|
| 1 | Round-robin | 0.248610 | 0.138061 |
| 1 | Random | 0.283126 | 0.103546 |
| 1 | Greedy graph information | 0.386671 | 0.000000 |
| 1 | Greedy kernel information | 0.386671 | 0.000000 |
| 1 | Greedy joint information | 0.386671 | 0.000000 |
| 1 | Coverage then joint information | 0.386671 | 0.000000 |
| 1 | Exact Bellman oracle | 0.386671 | 0.000000 |
| 2 | Round-robin | 0.628523 | 0.061158 |
| 2 | Random | 0.542388 | 0.147293 |
| 2 | Greedy graph information | 0.680916 | 0.008766 |
| 2 | Greedy kernel information | 0.616443 | 0.073238 |
| 2 | Greedy joint information | 0.689681 | 0.000000 |
| 2 | Coverage then joint information | 0.628523 | 0.061158 |
| 2 | Exact Bellman oracle | 0.689681 | 0.000000 |
| 3 | Round-robin | 0.842618 | 0.099034 |
| 3 | Random | 0.780161 | 0.161491 |
| 3 | Greedy graph information | 0.909471 | 0.032181 |
| 3 | Greedy kernel information | 0.842618 | 0.099034 |
| 3 | Greedy joint information | 0.940572 | 0.001080 |
| 3 | Coverage then joint information | 0.842618 | 0.099034 |
| 3 | Exact Bellman oracle | **0.941652** | **0.000000** |

The main result is conditional: with this prior and this 16-hypothesis family, greedy joint information is 0.00108 bits behind the exact optimum at horizon three. Round-robin is 0.09903 bits behind and random selection is 0.16149 bits behind. At horizon two, greedy joint information matches the oracle to reported numerical precision. Greedy graph-only selection has 0.03218 bits regret at horizon three because it optimizes a different objective.

This does not show that greedy EIG is generally near-optimal. Previous larger-kernel pilot runs exposed concentration and graph/kernel trade-offs. This oracle lane exists to quantify regret on small cases where exhaustive planning is tractable.

## Next benchmarks

1. Expand to larger DAGs, denser graphs and heterogeneous mechanisms.
2. Define separate graph-only, kernel-only and joint objectives; compare each against an oracle optimizing the same objective.
3. Add imperfect/missing telemetry and equal intervention-cost or disturbance budgets.
4. Compare to published methods under matched action and scoring contracts; report graph SHD and directed F1, posterior log loss/Brier/calibration, kernel L1/KL, probe cost, task disturbance, runtime, memory and per-family uncertainty intervals.

## Prior art and external comparison

Choo, Shiragur and Uhler (AISTATS 2024) already study action-dependent distributions over arbitrary off-target subsets. ACDB (September 2026) evaluates budgeted active causal discovery; CausalArena (September 2026 preprint) focuses on SCM-family and protocol transfer; TCD-Arena (ICLR 2026) evaluates assumption violations. None of those published algorithms or benchmark suites was executed in this small oracle lane, so no SOTA comparison is claimed.

References:
- https://proceedings.mlr.press/v238/choo24a.html
- https://arxiv.org/abs/2609.31675
- https://arxiv.org/abs/2609.11897
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/7d90f1e2591685c286d5d4b014b8c0f8-Abstract-Conference.html

## Reproduction and evidence

Run from the repository root with Python: set PYTHONPATH=validation, run unittest module test_exact_finite_horizon_oracle, then run exact_finite_horizon_oracle.py --horizon 3 with --source-git-blob 20981e1e1b4db0886849f64ee4f689e68ecddf53 and --output validation/exact_finite_horizon_oracle_results_2026-10-23.json.

The oracle module passes 7 local tests. The oracle tests plus both preceding benchmark suites pass 22/22 locally. The exact 21-row result artifact is 8,988 bytes, SHA-256 7fb11a9b1b8e263e106e91ca0149b050d9024125fc4e740ad8fbb7b67b4b28d2. The source and test Git blob IDs match the executed local files. Local tests do not imply remote CI passes.