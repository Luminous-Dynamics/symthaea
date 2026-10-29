# CASTING SOTA Baseline Reproduction Matrix (v1)

**Status:** protocol and source audit only. No published baseline is claimed reproduced and no comparative result is claimed.

## Why this matrix exists

CASTING-0036 is a synthetic prediction-transfer diagnostic. It should not be used to imply that a new optimizer beats modern methods. The next stage is a separately cost-matched optimizer bake-off. The benchmark must first reproduce known methods faithfully, publish run receipts, and distinguish code availability from algorithm-description availability.

## Baseline matrix

| Baseline | What it tests | Implementation status in this repository | Reproduction requirements |
|---|---|---|---|
| Low-fidelity-only / cold-start constrained BO | Whether transfer earns its cost at all | Local baseline implementations exist in the casting tool suite, but not yet a locked cross-paper bake-off | Frozen objective and constraint suite; fixed initialization; matched high-fidelity and total cost budgets; independent task/seed receipts |
| Pooled high-minus-low residual | Value of any historical residual signal without task conditioning | Implemented in CASTING-0036 synthetic transfer audit | Same historical archive and cost treatment as candidates |
| Descriptor + design KNN residual transfer | Transparent task-conditioned control | Implemented in CASTING-0036 synthetic transfer audit; not OOL-MFBO | Nested task/family metrics; shuffled descriptors, wrong-regime history, and missing/stale descriptor controls |
| MF-SCBO | Scalable constrained multi-fidelity BO with multiple fidelity levels and non-nested sampling | Public implementation identified: https://github.com/Nemo-Virtual-Lab/MF-SCBO ; MIT license indicated in its repository | Pin commit and environment; retain evaluated- and predicted-center variants; include SCBO and MF-MES comparators when practical; validate same objective sign, constraint sign, initialization cost, fidelity costs, and final high-fidelity audit semantics |
| Out-Of-The-Loop MFBO (Sutter et al., 2026) | Historical high-fidelity archive + descriptors when target high fidelity is unavailable during search | Paper identified: https://arxiv.org/abs/2608.04113 ; repository / artifact link not identified in the paper's HTML during this audit | Find and pin author artifact or contact authors; reproduce the published formulation, not a substitute KNN method; separate historical archive cost from per-target cost and do not let held-out high-fidelity outcomes enter the online loop |
| RESCUE causal-prior multi-objective MFBO (Hossen et al., 2026) | Causal multi-fidelity surrogate and causal hypervolume knowledge-gradient acquisition | Paper identified: https://arxiv.org/abs/2602.00788 ; repository / artifact link not identified in the paper's HTML during this audit | Find and pin author artifact or contact authors; do not label a generic causal baseline as RESCUE; freeze causal graph/prior selection and all tuning before target holdout unseal |
| Casting RLHS-BO + phase-field simulation (Pei et al., 2025) | Directly relevant small-data experimental/simulation casting optimization | Paper identified: https://www.nature.com/articles/s41524-025-01524-6 ; this is a domain-specific comparator, not necessarily plug-compatible with the synthetic suite | Map only shared objectives and constraints; report experiment count, initialization design, simulator/physical evidence and materials/process domain; do not compare scores across unrelated tasks |
| Commercial autonomous casting optimization (MAGMASOFT) | Industry capability and practical workflow comparison | Product/workflow reference: https://www.magmasoft.com/en/solutions/magmasoft/ ; no licensed local result data available to this benchmark | Direct numerical comparison only with permission and comparable cases, inputs, solver settings, quality metrics, and costs. Public product descriptions alone are not benchmark evidence |

### MF-SCBO environment and protocol notes

The source lock pins upstream commit `ca42fd6c5ec8c841e3f1ff9642509e12d104b581` and git-blob digests for the optimizer, requirements file, and MIT license. The public repository identifies BoTorch as its implementation framework and lists NumPy, SciPy, PyTorch, BoTorch and GPyTorch among its dependencies. Keep this stack as an isolated research environment. Do not pull its dependency graph into the Rust production path. Capture the exact repository commit, Python version, package lock / environment export, hardware metadata, RNG seeds, and raw stdout/stderr for every run.

Two compatibility details constrain a fair comparison. First, the optimizer accepts one shared constraint callable with no fidelity argument, so it cannot represent an unreliable low-fidelity constraint separately from a target-fidelity constraint; do not leak high-fidelity feasibility into supposedly cheap observations. Use a predeclared constraint that truly is invariant across fidelities for the faithful baseline, or mark the task as incompatible rather than pretending it is equivalent. Second, the source has no explicit local seed passed through its public initialization/optimization path, so seed the relevant global RNGs externally and verify repeated-run determinism. The source audit also flags a potential candidate-mask fallback issue: `torch.randint(0, dim - 1, ...)` excludes the last coordinate when an empty mask occurs (possible for dimensions above 20 under its mask probability). Keep the faithful upstream run unmodified, document this behavior, and if useful report a separately labeled corrected sensitivity fork rather than silently patching the comparator.

The MF-SCBO paper reports runs across dimensions from 6 to 100, several multi-fidelity constrained functions, and results as a function of both high-fidelity evaluations and total cost. It also reports that fidelity-cost ratios can reverse an apparent advantage. This is why CASTING must report both the high-fidelity count and full evaluation cost, not only an iteration count. The upstream paper and code artifacts are the authority for reproducing its published settings.

## Required benchmark protocol

1. **Freeze the test suite.** Use public analytic functions for initial comparator qualification, then casting-specific simulator tasks. The 32 synthetic transfer variants in CASTING-0036 are useful leakage and family-stratification controls, not substitutes for a broad optimizer suite.
2. **Use task-family holdout.** Do not random-split candidate points from a task. Hold out alloy/material family, geometry/topology, mold/process regime, and boundary-condition shifts. Report task-level and family-macro metrics; resample families for cross-family uncertainty.
3. **Pre-register budgets.** Match initialization cost, total evaluation cost, highest-fidelity calls, wall clock, batch size, stopping rule, and hyperparameter-search budget. Also report archive curation and amortization as separate line items.
4. **Preserve an anytime trace.** Every evaluation produces a row binding task ID, candidate, policy/source/config digest, fidelity, measured cost, objective vector, constraints, simulator exit status, and run receipt. A missing / failed run is not silently dropped.
5. **Score decision quality.** Report feasible hypervolume / feasible-quality AUC, cost-to-frozen-target, constraint violations, false-feasible risk and denominator, calibration, abstention coverage, OOD loss, worst-family performance, runtime, and memory. Use the same feasibility convention across all methods.
6. **Prevent leakage.** Target high-fidelity outcomes remain audit-only until policy and hyperparameters freeze. Use separate training, tuning, calibration, and final holdout splits. Record hashes of the source tree, task suite, run manifest, policies, environment, and evaluator.
7. **Keep evidence classes separate.** Synthetic, public analytic, simulator-backed, and physical results are separate evidence strata. Synthetic test count cannot qualify a physical casting process.
8. **Publish failures.** Include all seeds and failed/crashed/over-budget trials; do not rerun selectively or choose the strongest method after seeing test results without a preregistered selection rule.

## Preregistered promotion thresholds

Carry forward CASTING-0032's proposed targets as goals until they are formally frozen in the paired evaluator contract:

- At least 5% median cost-normalized feasible-quality AUC improvement over the strongest faithfully reproduced baseline, with the contract's uncertainty threshold satisfied.
- At least 70% independent task wins, ties counting as non-wins.
- No more than 5% median regression in any critical task family under its uncertainty gate.
- At least 20% lower cost to the same frozen target when both methods reach it.
- Every critical family's false-feasible upper confidence bound must satisfy the risk contract.
- No SOTA claim without a preregistered task suite, independent rerun, exact-head receipt, and simulator-backed evidence; no physical validity claim without prospective physical evidence.

These thresholds are research targets, not current results or universal safety limits.

## Research opportunity

The strongest plausible contribution is not another name for multi-fidelity BO. It is a **decision-centered, evidence-aware casting optimization protocol** that (i) learns when historical residuals transfer, (ii) chooses whether a cheap or expensive evaluation is worth its cost, (iii) abstains under unvalidated distribution shift, (iv) accounts for full-lifecycle archive and evaluation costs, and (v) makes the promotion decision independently auditable. Every component must survive ablation and strong baseline comparison.

## External sources

- OOL-MFBO (Aug 2026): https://arxiv.org/abs/2608.04113
- RESCUE causal-prior MFBO (Jan 2026): https://arxiv.org/abs/2602.00788
- MF-SCBO (Sep 2026) paper: https://arxiv.org/abs/2609.29941
- MF-SCBO public code: https://github.com/Nemo-Virtual-Lab/MF-SCBO
- Casting small-data RLHS-BO + phase-field study (2025): https://www.nature.com/articles/s41524-025-01524-6
- Casting AI/digital-twin review (2026): https://link.springer.com/article/10.1007/s43939-026-00685-5
- COCO performance assessment: https://numbbo.github.io/coco-doc/perf-assessment/
- NIST VVUQ / metrology harmonization report (Oct 2026): https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing


## Immutable MF-SCBO source lock

The research lock is recorded in `tools/casting-benchmark/baselines/mf_scbo_lock_v1.json` and validated by `test_baseline_lock_manifest.py`. It pins upstream commit `ca42fd6c5ec8c841e3f1ff9642509e12d104b581`, the main optimizer source blob, the exact requirements-file blob, and the MIT license blob. It deliberately states `source_pinned_not_reproduced`. A pure-stdlib test protects those pins and the key API semantic: `eval_objectives` is a scalar callable per fidelity, not a native multi-objective Pareto interface. Thus the direct benchmark must pre-register a scalarization, flip the feasibility-margin sign to the optimizer's `c(x) <= 0` convention, and keep the upstream Python environment isolated.


## Executable MF-SCBO comparison preflight

`tools/casting-benchmark/mfscbo_comparison_contract.py` adds a strict-schema preflight before any future upstream run. It validates the pinned source commit; requires a fixed suite digest; freezes normalization and scalarization before holdout unseal; requires at least five unique two-objective scalarization weights including both single-objective endpoints; enforces the margin-to-`c(x) <= 0` sign convention; rejects tasks with fidelity-dependent constraints because the upstream API takes one shared constraint callable; ensures full-pool audit truth never enters the search loop; and checks that the total budget covers the initial point design. Its pure-standard-library tests cover these rules and bind the contract, source lock, and evaluator digests in a receipt. CI runs the tests.

A passing preflight means **compatible contract validated, not run**. It does not install or execute MF-SCBO, reproduce the paper, or establish a performance result. Target-HF evaluations selected and paid for by MF-SCBO may occur inside its optimization loop; the full finite-pool audit truth remains reserved for post-run scoring. This distinction is essential when comparing it with methods such as OOL-MFBO that deliberately do not query target high fidelity during their active loop.
