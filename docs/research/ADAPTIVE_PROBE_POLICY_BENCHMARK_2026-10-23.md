# Adaptive causal probe policy benchmark — 2026-10-23

## Research question

The previous target-subset study measured the value of recording which variables an intervention actually affects. The next unresolved question is whether adaptive selection can use that information better than non-adaptive allocation—and whether optimizing graph uncertainty alone misallocates a limited experiment budget.

This is an exact-posterior, seeded **pilot**, not an external benchmark leaderboard.

## Protocol

- Four three-node Boolean SCM graph families: chain, fork, collider, reverse chain.
- Each attempted target has an action-dependent distribution over all eight realized subsets. The learner uses the same explicit finite prior of 330 quarter-step categorical distributions as the previous benchmark.
- Two truth regimes: an in-grid kernel (0.5 on the intended singleton and 0.5 on an adjacent pair) and an out-of-grid kernel (0.40 singleton, 0.35 adjacent pair, 0.25 all-three subset).
- Observation channel: exact realized target subset plus the three-bit outcome. This isolates action selection under high-quality telemetry; telemetry quality has been ablated in the companion benchmark.
- Equal budget: eight intervention attempts per trial. Seven policies: round-robin, random, greedy graph information, greedy target-kernel information, greedy joint graph/kernel information, coverage-first then graph information, and coverage-first then joint information.
- Five replicates for each of four SCMs and each truth regime. Each policy × truth regime therefore has 20 trials. Total artifact: 280 policy-trial rows.
- The posterior is exact over the declared finite model class; the policy's one-step expected information is computed from that posterior. NumPy only accelerates entropy reductions when installed; a standard-library implementation remains available.

## Results

All entries average 20 trials per truth regime/policy. MAP accuracy is higher-is-better; graph log loss, Brier score, and target-kernel L1 are lower-is-better.

### In-grid kernel truth

| Policy | MAP accuracy | Graph log loss (bits) | Brier score | Mean kernel L1 | Mean action counts (0,1,2) |
|---|---:|---:|---:|---:|---|
| Round-robin | 75% | 0.436 | 0.190 | 0.616 | 3.00, 3.00, 2.00 |
| Random | 70% | 0.652 | 0.255 | 0.678 | 2.75, 2.80, 2.45 |
| Greedy graph information | 75% | 0.527 | 0.229 | 0.930 | 6.40, 1.00, 0.60 |
| Greedy kernel information | **80%** | 0.385 | 0.158 | 0.600 | 2.90, 2.60, 2.50 |
| Greedy joint information | 75% | 0.561 | 0.238 | **0.594** | 2.70, 2.70, 2.60 |
| Coverage then graph information | **80%** | **0.359** | **0.146** | 0.712 | 3.05, 3.55, 1.40 |
| Coverage then joint information | 75% | 0.561 | 0.238 | **0.594** | 2.70, 2.70, 2.60 |

### Out-of-grid kernel truth

| Policy | MAP accuracy | Graph log loss (bits) | Brier score | Mean kernel L1 | Mean action counts (0,1,2) |
|---|---:|---:|---:|---:|---|
| Round-robin | **95%** | 0.615 | 0.208 | 0.660 | 3.00, 3.00, 2.00 |
| Random | 75% | 0.622 | 0.249 | 0.652 | 2.45, 2.95, 2.60 |
| Greedy graph information | 75% | 0.695 | 0.296 | 0.807 | 4.90, 2.10, 1.00 |
| Greedy kernel information | 60% | 1.004 | 0.435 | 0.601 | 2.85, 2.80, 2.35 |
| Greedy joint information | 75% | 0.819 | 0.329 | **0.592** | 2.90, 2.65, 2.45 |
| Coverage then graph information | 75% | 0.689 | 0.295 | 0.709 | 3.25, 2.95, 1.80 |
| Coverage then joint information | 75% | 0.819 | 0.329 | **0.592** | 2.90, 2.65, 2.45 |

## The important result is the trade-off, not a single winner

1. **Pure graph information gain can collapse onto one action.** In-grid it spent an average 6.4 of 8 probes on target 0 and just 0.6 on target 2, with the worst in-grid mean kernel L1 error (0.930). The earlier 4-probe pilot also revealed this concentration, but this 8-probe run confirms it under a larger fixed horizon.
2. **Joint information improves kernel estimation but does not dominate graph recovery.** It had the best average target-kernel L1 in both regimes (0.594 in-grid; 0.592 out-of-grid), but round-robin led out-of-grid graph MAP accuracy at 95% versus 75% for joint information.
3. **Coverage constraints are a principled anti-collapse mechanism, not a universal performance guarantee.** Coverage-then-graph improved in-grid graph log loss from 0.527 to 0.359 and kernel L1 from 0.930 to 0.712 compared with unconstrained graph-greedy. On these seeds, unconstrained joint-greedy already visited targets broadly, so coverage-then-joint behaved identically in aggregate.
4. **Model mismatch changes policy rankings.** Greedy-kernel had good out-of-grid kernel L1 (0.601) but poor graph calibration (1.004 bits log loss, 60% MAP). Round-robin graph MAP accuracy was unusually high in this small out-of-grid sample. With only 20 trials per condition, these differences are exploratory and have no claim of statistical significance.

The benchmark therefore rejects the simplistic idea that “maximum graph EIG” is a sufficient experiment-design objective. It also rejects claiming a universally superior adaptive policy from one small task. The next candidate is a **predeclared multi-objective or constrained policy**: minimize graph log loss subject to a target-kernel error ceiling and mandatory coverage, then report the resulting graph/kernel Pareto frontier under held-out model families.

## Comparison with existing work

This benchmark is deliberately not a novelty claim. Choo, Shiragur and Uhler (AISTATS 2024) already study action-dependent distributions over arbitrary intervention target subsets and give algorithms for intervention-minimizing causal DAG search/verification. ACDB (September 2026) evaluates agents under budgeted interventions with skeleton/DAG recovery and intervention efficiency; its currently reported ladder has PC with a greedy active-orientation heuristic ahead of raw LLM agents. TCD-Arena (ICLR 2026) stress-tests causal discovery under 33 assumption violations. CausalArena (September 2026 preprint) emphasizes cross-family transfer and overlap between pretraining and test SCMs.

This local pilot has **not** executed the published Choo-style algorithms, ACDB, TCD-Arena, or CausalArena. The graph families, intervention counts, and telemetry contract differ, so there is no direct SOTA comparison yet.

- Choo et al., AISTATS 2024: https://proceedings.mlr.press/v238/choo24a.html
- ACDB, September 2026: https://arxiv.org/abs/2609.31675
- TCD-Arena, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7d90f1e2591685c286d5d4b014b8c0f8-Abstract-Conference.html
- CausalArena, September 2026 preprint: https://arxiv.org/abs/2609.11897

## Next qualification gates

- Use larger DAGs and mechanism/topology-family-held-out splits; increase repeats and publish uncertainty intervals across seeds.
- Add a genuine finite-horizon lookahead/oracle on tiny cases, and test greedy regret relative to that oracle.
- Evaluate equal intervention-cost and equal disturbance budgets, not only equal action counts.
- Compare action-selection policies against the Choo et al. setting under matching intervention semantics; separately map/adapt to ACDB/TCD-Arena/CausalArena where their APIs permit.
- Report graph SHD/skeleton and directed F1, calibration/log loss/Brier, target-kernel L1/KL, utility disturbance, interventions/cost, runtime, memory and failure/abstention behavior.
- Add noisy and missing telemetry; do not assume exact subset measurement is free or realistic.

## Reproduction and evidence

From repository root:

```bash
PYTHONPATH=validation python3 -m unittest -v test_adaptive_probe_policy_benchmark
PYTHONPATH=validation python3 adaptive_probe_policy_benchmark.py --replicates 5 --budget 8 --modes exact_target_subset --source-git-blob b8a40e8bdba0687a4792c9b57ed222889c50a92d --output validation/adaptive_probe_policy_results_exact_2026-10-23.json
```

The full 280-row local artifact is 184,055 bytes, SHA-256 `9d3221eda838d534a24215a55e738fadc9552115d3ceb299365b5414e40eafd5`. It is retained locally and is not committed to GitHub. The executable source and test blobs match the local byte-level Git object checks; seven local tests passed. This is local evidence only. Remote workflow status is reported separately in the receipt, and skipped jobs are not passes.
