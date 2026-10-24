# Action-dependent target-subset kernel benchmark — 2026-10-23

## Question

When an attempted causal intervention can realize an action-dependent random subset of affected variables, how much does observing the realized subset help with (a) causal-graph recovery and (b) estimating the intervention-delivery kernel itself?

This is a narrow, exact three-variable pilot. It does **not** claim a new stochastic-intervention model: Choo, Shiragur & Uhler (AISTATS 2024) already define action-dependent distributions over arbitrary subsets of vertices and study intervention-minimizing graph search/verification. The contribution here is an auditable evaluation fixture across telemetry levels and an out-of-grid kernel, which can later be adapted to established benchmarks.

## Exact protocol

- Four candidate Boolean SCM graphs: chain, fork, collider, reverse chain.
- Three Boolean variables; roots are independent fair Bernoulli variables. Each non-root computes XOR of its parents with independent 0.1 bit-flip noise.
- Three action choices: attempt to intervene on node `i` with requested value 1, for `i ∈ {0,1,2}`.
- Each action has its own distribution `D_i` over all eight subsets of the three variables. Every variable in the realized subset `S` is clamped to 1; unaffected variables follow their SCM equations. The empty subset and multi-node subsets are allowed.
- The learner uses a uniform finite prior over all 330 distributions whose eight probabilities are multiples of 0.25. This is an explicit discrete approximation, **not** the unrestricted continuous probability simplex.
- Two truth regimes: in-grid kernels (`0.5` on the intended singleton and `0.5` on an adjacent pair); out-of-grid kernels (`0.40`, `0.35`, `0.25` on the intended singleton, adjacent pair, and all-three subset).
- Three observation channels: (1) outcomes only; (2) outcome plus a synthetic target-only flag with 90% sensitivity/specificity; (3) outcome plus the exact realized subset.
- 24 deterministic seeds for every graph × kernel regime; 12 interventions per attempted target per seed. The same latent outcomes are paired across telemetry modes. Total: 192 simulated worlds and 576 posterior-fit rows.
- Exact enumeration for SCM outcome distributions and one-probe mutual information; exact Bayesian posterior updates over the declared finite graph/kernel hypothesis class.

## Exact one-probe expected information

Averaged over the three attempted targets, starting from the benchmark's uniform graph × kernel prior:

| Observation channel | Graph information (bits) | Action-kernel information (bits) | Joint information (bits) |
|---|---:|---:|---:|
| Outcomes only | 0.1886 | 0.3107 | 0.5349 |
| Noisy target-only flag | 0.1952 | 0.3997 | 0.6292 |
| Exact realized subset | 0.2664 | 1.5896 | 1.8560 |

In this model, exact subset telemetry gives about **5.1×** the one-probe action-kernel information of outcomes alone (1.5896 / 0.3107). The noisy flag adds information too, but its benefit is much smaller than directly measuring the realized subset. These are model-conditional information calculations, not real-world sensor measurements.

## Seeded repeated-intervention results

Each entry below averages 96 graph/kernel/seed trials. `MAP accuracy` is the fraction of trials whose maximum-posterior graph is correct. `Graph log loss` and Brier score retain calibration penalties that MAP accuracy discards. Kernel error is the mean L1 distance between the posterior-mean target-subset distribution and the true distribution, averaged across three actions; lower is better.

| Truth regime | Channel | Graph MAP accuracy | Graph log loss (bits) | Graph Brier | Kernel L1 error |
|---|---|---:|---:|---:|---:|
| In-grid | Outcomes only | 90.63% | 0.3083 | 0.1292 | 0.7901 |
| In-grid | Noisy flag | 92.71% | 0.1994 | 0.0922 | 0.6621 |
| In-grid | Exact subset | 98.96% | 0.0762 | 0.0312 | 0.2243 |
| Out-of-grid | Outcomes only | 88.54% | 0.3910 | 0.1763 | 0.6902 |
| Out-of-grid | Noisy flag | 92.71% | 0.2262 | 0.1002 | 0.5862 |
| Out-of-grid | Exact subset | 91.67% | 0.1982 | 0.0860 | 0.2045 |

The naive hard-do baseline (assume attempted target equals the only realized target; ignore telemetry) achieved 82.29% MAP accuracy / 1.4307 bits graph log loss in the in-grid truth regime, and 72.92% / 1.9467 bits in the out-of-grid regime. An oracle that knows the true action-specific subset distribution but sees only outcomes achieved 91.67% / 0.1822 bits in-grid and 89.58% / 0.2989 bits out-of-grid. That oracle is a reference upper-information baseline for the *outcome-only* lane, not a like-for-like competitor to methods with direct subset telemetry.

In the out-of-grid case, exact telemetry has lower mean graph log loss and Brier score than the noisy flag, while the noisy flag has slightly higher MAP accuracy. That ranking reversal is why the benchmark keeps discrimination, calibration, and kernel estimation separate rather than compressing them into one score.

## What this establishes — and does not

1. **Operational shift:** record the realized intervention as a latent/observed causal variable, not as an unexamined synonym for the requested `do(X_i=1)` action.
2. **Telemetry has measurable value in this toy system:** exact subset observations substantially reduce target-kernel error and improve graph posterior quality relative to outcomes alone and the hard-do assumption.
3. **Robustness must be evaluated out of model:** a 0.25-grid posterior is misspecified for the 0.40/0.35/0.25 truth. The resulting metric ranks differ; this is a useful diagnostic, not evidence that exact telemetry is universally inferior or that the finite learner is robust enough.
4. **No SOTA claim:** this pilot has not run ACDB, TCD-Arena, or the Choo et al. algorithms. Its four tiny Boolean graphs and synthetic 90% sensor are controlled fixtures only.
5. **Next qualification work:** compare equal-cost policies and Choo-style target-set models on larger seeded SCMs; learn continuous/Dirichlet action kernels; add interventions assigning variable-specific spill values; include noisy/missing/biased telemetry, confounding and mechanism changes; report graph SHD/F1, posterior calibration, target-set log score, kernel L1/KL, expected intervention cost, task disturbance, wall time and memory; and add a held-out family split.

## Reproduction

From the repository root, using Python 3.11+ and only the standard library:

```bash
PYTHONPATH=validation python3 -m unittest -v test_action_subset_kernel_benchmark
python3 validation/action_subset_kernel_benchmark.py --replicates 24 --samples-per-action 12 --source-git-blob ce19dccec3a81f4c6a18563de88152019b966f5d --output validation/action_subset_kernel_results_exact_2026-10-23.json
```

The full local result artifact is 479,709 bytes, SHA-256 `dcf7b1e4417d93b5b889b0fe4b5a4496a2122f15bf8580eb0bf9374f5e659051`. It records the source Git blob and all 576 rows. The exact source and test blobs were compared against the executed local files. Local tests passed 8/8. This does not imply any remote CI status; remote checks must be inspected on the PR's exact head.

## Related literature

- Choo, Shiragur & Uhler, “Causal Discovery under Off-Target Interventions,” AISTATS 2024: https://proceedings.mlr.press/v238/choo24a.html
- “Active Causal Discovery Benchmark: Evaluating LLM Agents Under Budgeted Interventions,” arXiv:2609.31675 (September 2026): https://arxiv.org/abs/2609.31675
- Stein et al., “TCD-Arena: Assessing Robustness of Time Series Causal Discovery Methods Against Assumption Violations,” ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7d90f1e2591685c286d5d4b014b8c0f8-Abstract-Conference.html
