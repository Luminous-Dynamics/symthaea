# CASTING-0035 — Paired matched-cost benchmark audit v1

**Issue:** [CASTING-0035 / #8846](https://github.com/Luminous-Dynamics/symthaea/issues/8846)  
**Status:** executable analysis contract; comparative results are only as sound as the supplied, independently audited run records.  
**Evidence classes:** synthetic, public analytical, simulator-backed, and physical manufacturing remain separate.

## Why this slice exists

COCO's established optimizer-benchmarking procedure emphasizes anytime performance and runtime/evaluation cost to reach quality targets, and treats termination as part of the algorithm rather than a harmless reporting detail:

- https://numbbo.github.io/coco-doc/perf-assessment/
- https://numbbo.github.io/coco-doc/experimental-setup/

Casting optimization adds constraints and shift robustness. A policy should not win by spending more, dropping a failed run, reporting thousands of dependent candidate × shift rows as independent samples, or averaging a serious boundary-shift failure away.

The paired evaluator makes a narrow contract executable: compare every policy on the same declared test units, scenario metadata, quality normalization, budget, and shift set; include incomplete and over-budget runs; calculate uncertainty by resampling the declared independent `unit_id` values only.

## CLI

```sh
python3 tools/casting-benchmark/paired_benchmark_audit.py \
  paired-runs.json --output paired-receipt.json
```

The manifest root has fixed fields: `schema_version`, `benchmark_id`, `protocol_sha256`, `normalization_id`, `baseline_policy`, `quality_target`, `sampling_unit`, `independence_note`, `bootstrap_replicates`, `bootstrap_seed`, and `runs`. The required schema is deliberately strict; unsupported keys fail rather than being silently ignored.

Each `unit_id` identifies one declared independent generated task-seed unit, and `sampling_unit` must literally be `unit_id`. Every policy must have exactly one run for each unit. Paired runs must match task family, constraint family, dimension, severity, seed and budget. They must share the same quality value at cost zero (the preregistered common initialization); otherwise a policy could receive a free head start. The policy roster must be identical for every unit; a failed, cancelled, or incomplete run still needs a record with `completed=false`, its last valid cost-quality observation, and actual cost accrued.

Each run binds provenance digests for its unique run receipt, policy source, policy configuration, runtime/environment and benchmark-suite manifest. A given policy's source, configuration, environment and suite digests must remain fixed across every paired unit; a policy cannot silently change between tasks. Run receipt digests must uniquely identify each policy × unit run. These fields are references; users must preserve and separately verify the artifacts behind them. The audit cannot prove a claimed digest actually points to the correct source code without those artifacts.

## Quality curve and cost accounting

The curve contains `{cost_units, quality}` points. Cost begins at zero and increases strictly. Quality uses the top-level frozen normalization identified by `normalization_id` and must be in `[0,1]`. A run's normalized quality AUC is trapezoidal integration over its **common budget**; after the last observed point, its last score is carried forward to the budget. This makes unused budget visible in the anytime score and prevents early termination from disappearing from the denominator.

Actual realized cost is reported separately. Any `cost_spent_units > budget_units` is retained and counted as an over-budget run, and later results do not extend the AUC beyond the shared budget. Full-cost definitions (optimizer fitting, failed calls, hardware, independent audit, physical/human review where measurable) must be preregistered in the protocol that the manifest's digest binds.

The target is crossed the first time the quality curve reaches the frozen target, with linear interpolation between adjacent points. A policy's target-hit rate uses **all** declared units. Median cost-to-target among successful runs is conditional; the comparison also reports target-hit-rate differences. Cost reductions are paired only on units where both candidate and baseline reach the target and where baseline cost is positive; the conditional denominator is stated.

## Shift and decision reporting

Each run supplies the same set of shift summaries, each containing normalized feasible quality, `false_feasible_count`, and `decision_count`. A non-abstaining decision must have an audited denominator in each shift; false-feasible events cannot exceed that denominator. An abstention has zero quality, zero decision count and zero false-feasible count for each shift. Abstention remains visible as its own count/rate so a policy is not rewarded for silently excluding difficult decisions.

The receipt reports:
- cost-normalized anytime AUC and paired candidate-minus-baseline differences;
- mean and median paired differences, win/tie/loss counts and 95% percentile bootstrap intervals for effect estimates / win rate;
- the fraction of declared units reaching a frozen quality target and the conditional cost-to-target comparison;
- mean shifted quality, worst-shift quality, per-shift effects and per-task-family effects;
- false-feasible count/rate with explicit decision denominator, abstention, incomplete and over-budget counts;
- raw input manifest and evaluator source digests.

For relative AUC improvement, runs with zero baseline AUC are excluded because the ratio is undefined; the excluded count is emitted beside the result. Both absolute AUC deltas and relative gains should be reported. Shift-wise estimates are paired at the independent unit and should not be interpreted as an IID collection of candidate × shift observations.

## Statistical interpretation

Bootstrap samples resample whole `unit_id` records and preserve each unit's policies and shift results. The manifest must state what makes those units independent. Code can verify pairing and IDs, but **cannot establish statistical independence from a caller-provided statement**. The percentile interval is an empirical uncertainty summary, not a frequentist theorem, confidence sequence, physical risk bound, or guarantee of future performance. With few independent units, call a comparison underpowered or inconclusive instead of treating a wide or degenerate interval as proof.

For stronger inference, preregister the unit-generation process and sample size / power assumptions; report raw paired-unit results; use an independent analyst or second implementation to verify receipts; and preserve all failures and out-of-budget outcomes. If conditions are nested (same task family / simulator parameter set / seed), the independent unit must be the appropriate top-level cluster rather than each derived record.

## Acceptance targets (not results)

Retain CASTING-0032's previously stated aspirational gates:
- at least 5% median improvement in cost-normalized feasible-quality AUC versus the strongest faithfully reproduced baseline;
- wins on at least 70% of preregistered task units under paired analysis;
- no more than 5% median regression in any critical task family;
- at least 20% lower full cost to the same frozen quality target where both approaches reach it.

Those targets must be interpreted alongside uncertainty, baseline-zero exclusions, target reach rates and shift-wise regressions. The evaluator does not automatically declare success from a point estimate. Freeze definitions and thresholds before unsealing test outcomes; publish every paired unit and failure.

## Hostile fixture requirements

The test module covers known-answer paired effects, deterministic bootstrap intervals, target reach/cost, shifts, family stratification, incomplete and over-budget runs, missing policy pairs, duplicate IDs, unmatched budgets/shifts, invalid curves, impossible false-feasible counts, provenance digests, duplicate JSON keys and receipt overwrite protection. Extend this suite with:
- a policy that looks best by final nominal score but loses anytime AUC;
- a low false-feasible rate achieved through abstention, reported alongside its abstention rate;
- a large number of correlated shift rows but only two independent units;
- a zero-AUC baseline for the relative-gain denominator;
- one critical-family regression despite a positive pooled average;
- independent raw-receipt verification and a run whose claimed source digest does not match its archived source.

## Literature / comparison context

- COCO experimental procedure and termination: https://numbbo.github.io/coco-doc/experimental-setup/
- COCO performance assessment: https://numbbo.github.io/coco-doc/perf-assessment/
- Xie et al., *Cost-aware Stopping for Bayesian Optimization*, ICML 2026: https://proceedings.mlr.press/v306/xie26c.html
- MF-SCBO, a recent multi-fidelity constrained BO comparator: https://arxiv.org/abs/2609.29941
- MF-JESMOC, constrained multi-objective multi-fidelity entropy search: https://doi.org/10.1016/j.neucom.2025.131674
- BoTorch multi-objective multi-fidelity BO tutorial: https://botorch.org/docs/v0.13.0/tutorials/Multi_objective_multi_fidelity_BO
- MAGMASOFT industry context: https://www.magmasoft.com/en/solutions/magmasoft/

This is a fair-comparison instrument, not yet a reproduced baseline, result, trained optimizer or physical manufacturing certificate.
