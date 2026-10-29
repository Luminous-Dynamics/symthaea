# CASTING-0036 — Historical high-fidelity transfer benchmark v1

**Issue:** [CASTING-0036 / #8849](https://github.com/Luminous-Dynamics/symthaea/issues/8849)  
**Status:** deterministic synthetic research benchmark; not evidence of optimizer superiority or physical casting performance.

## Motivation from current literature

Historical transfer and causal multi-fidelity modeling are already active research areas; the contribution here is a casting-shaped, transparent diagnostic with deliberately held-out task families, shuffled-history controls, feasibility and Pareto diagnostics, and explicit archive amortization—not the general idea of transfer.

- Sutter et al., *Out-Of-The-Loop Multi-Fidelity Bayesian Optimization* (arXiv, 4 August 2026): https://arxiv.org/abs/2608.04113. This studies historical high-fidelity observations with task descriptors when target-task high fidelity is unavailable within the active loop.
- Hossen et al., *Multi-Objective Multi-Fidelity Bayesian Optimization with Causal Priors* / RESCUE (2026 preprint): https://arxiv.org/abs/2602.00788. This studies causal priors and intervention-aware acquisition; a causal-prior policy must be included as a comparator before making a novelty claim.
- Palazzolo et al., *MF-SCBO* (24 September 2026): https://arxiv.org/abs/2609.29941. This is a relevant constrained multi-fidelity optimization comparator.
- The 2026 casting / foundry digital-twin review highlights limited cross-alloy generalization, uncertainty quantification, shared benchmarks and production-scale validation: https://doi.org/10.1007/s43939-026-00685-5.

## Run

```sh
python3 tools/casting-benchmark/test_historical_transfer_benchmark.py
python3 tools/casting-benchmark/historical_transfer_benchmark.py \
  --history-points 48 --test-points 128 --nearest-k 16 \
  --reuse-count 50 --output history-transfer-receipt.json
```

All code uses the Python standard library. The run is deterministic under a fixed source version and argument set.

## Benchmark construction

The frozen suite has eight source tasks across four source families and four target tasks, each in a target family not present among the source families. Each task has three normalized descriptors standing in for composition, thermal properties and geometry. Task family is used only for split metadata and is not a prediction input.

A deterministic two-dimensional Halton design supplies non-overlapping source and target point ranges. For each source task and historical design point, the suite forms a paired low/high observation and archives the vector residual `high - low` for two minimization objectives and one constraint margin. No target-task high-fidelity response is passed to a prediction policy. Target truth is queried after all four predictions for the target point are fixed, solely by the audit scorer.

The low-fidelity proxy has smooth task- and design-dependent discrepancies. That is a controlled synthetic assumption, not a calibrated cast-process model. It is intentionally simple enough to audit: the benchmark is for testing transfer/leakage methodology before investing in an expensive simulator integration.

## Compared methods

1. `low_fidelity_only`: no history correction.
2. `pooled_residual_transfer`: add a global mean historical `high - low` residual, ignoring task descriptors and design location.
3. `descriptor_x_knn_transfer`: retrieve historical paired residuals by normalized task-descriptor and design-space distance; add the weighted residual to target low fidelity.
4. `shuffled_descriptor_control`: preserve historical residuals but rotate source-task descriptor assignments. This tests whether the apparent gain depends on meaningful descriptor-to-residual correspondence.

The kNN transfer is a transparent heuristic baseline, **not** an implementation of OOL-MFBO or RESCUE. These published methods should be reproduced from pinned source/dependencies before any SOTA claim.

## Metrics

For each held-out target task and policy, the receipt reports:
- MAE for both objectives and the constraint margin;
- feasibility classification accuracy and F1;
- false-feasible count and rate among predicted-feasible candidates;
- false-infeasible count and rate among truly feasible candidates;
- Pareto membership F1 between the predicted and true finite-point fronts;
- worst target-task objective and margin MAE;
- paired per-target-task objective-MAE improvement over low-fidelity-only.

The **independent unit is the target task**, not each candidate point. The default contains only four target tasks; 128 points inside each task are nested measurements, not 512 independent trials. The receipt intentionally does not claim statistically supported SOTA superiority from four task units.

## Full-cost treatment

Costs are illustrative normalized units: low fidelity costs one unit and high fidelity costs 25. Building the source archive includes one low and one high evaluation per archived point. The total archive cost is divided by a declared reuse count; the target low-fidelity sweep is then added. The receipt also reports a high-only target scan as a rough reference.

This is a sensitivity model, not an industry cost estimate. It does not yet price historical database curation, feature extraction, model fitting, storage, engineering labor, hardware differences, failures, or external audit. Real comparisons must measure or preregister those costs and test multiple reuse counts; archive preparation must never be treated as free by default.

## Required experiment extensions

- Sweep history sizes (0, 5, 10, 25, 50, 100+ high-fidelity paired points), number of source tasks, kNN neighbourhood and reuse count.
- Add interpolative and extrapolative targets and report both, without pooling them into one score.
- Hold out complete alloy and process families, not random rows from the same task.
- Compare cold-start baseline, nearest-task transfer, descriptor-conditioned residual transfer, and faithful implementations of OOL-MFBO, RESCUE and MF-SCBO.
- Test wrong / missing / stale descriptors, shuffled and intentionally misleading history, sign-reversed fidelity discrepancy, abrupt constraint-boundary changes, and histories drawn from the wrong process regime.
- Keep task descriptors and historical cutoffs frozen before target outcomes are opened. Use a separate calibration split if hyperparameters are selected.
- Bind receipts to source, suite, archive, policy, and environment digests; verify the original source artifacts rather than trusting digest strings alone.
- Integrate the successful transfer baseline into CASTING-0035's paired full-cost optimizer evaluation only after a transfer-prediction benefit is reproduced on an adequately sized suite.

## Claim ceiling

The current harness is a four-target, fully synthetic prediction-transfer diagnostic. It does not run an optimizer, establish calibrated uncertainty, prove descriptor availability, demonstrate a cost advantage after real archive preparation, outperform OOL-MFBO / RESCUE / MF-SCBO, establish physical feasibility, or qualify a foundry process. The shuffled-history control and explicit cost accounting make the hypothesis falsifiable; only preregistered held-out experiments can establish whether useful transfer occurs.
