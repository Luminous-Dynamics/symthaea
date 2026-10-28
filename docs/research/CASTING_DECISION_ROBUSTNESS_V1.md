# CASTING Decision Robustness Audit v1

**Benchmark ID:** CASTING-DECISION-ROBUSTNESS-001  
**Status:** research harness and strict bake-off adapter implemented; no production shift manifest / physical evidence supplied  
**Scope:** two-objective constrained optimization with a fixed reference point  
**Evidence labels:** synthetic / simulator / physical must remain distinct

## Why this benchmark exists

A search policy can exploit defects in its design-time simulator. Improving the surrogate's in-domain fit or the nominal Pareto front does not guarantee that the selected decisions remain feasible or good when process parameters, boundary conditions, material properties, discretization, or model structure shift. This track measures the consequences of selection under explicit post-selection shifts rather than scoring only the optimizer's nominal trace.

It complements, rather than replaces, the compositional multi-fidelity bake-off and promotion_risk_gate.py. This evaluator computes descriptive outcomes. It never certifies false-feasible risk, proves split independence, or establishes physical validity.

## Frozen execution contract

1. Freeze the candidate pool, nominal predictions, selected-candidate flags, reference point, source revision, and selection trace **before** running any holdout shift. Record the frozen-selection SHA-256.
2. Freeze a holdout manifest with scenario-family definitions, parameter ranges, seeds, simulator revision, and evaluation budget. Record its SHA-256. Do not use these holdout observations to tune or reselect candidates for the same reported run.
3. Use `tools/casting-benchmark/adapt_bakeoff_to_robustness.py` to bind an unaltered compositional sweep, a frozen selection/prediction manifest and a sealed shift manifest. The adapter requires the exact sweep hash, selector, and candidate-pool digest to agree. Evaluate **every candidate in the declared pool under every shift**. Missing or duplicate candidate/shift rows fail closed. This makes finite-pool regret comparable and prevents selective omission of bad candidates.
4. Normalize both minimized objectives into the interval [0, reference_point[i]) using transformations frozen before evaluation. The reference point is shared across policies and shifts; do not tune it per run.
5. Score adapter output with `tools/casting-benchmark/decision_robustness_eval.py`. Archive all three source artifacts, converted input bytes, receipt, evaluator revision, environment, and source digests. The receipt hashes the exact converted input bytes and evaluator source bytes, and includes the source run's sweep digest, adapter source digest, candidate-pool digest, fidelity costs, and evaluation cost accounting.
6. Keep one row per task, selection, candidate, and shift in the raw archive. Preserve failed, timed-out, invalid, and refused evaluations as outcomes in the acquisition layer; do not omit failures from the accounting.
7. Report synthetic stress fixtures, public optimization benchmarks, engineering simulator results, and physical foundry observations as separate evidence classes.

The source selection manifest must contain `schema_version`, `benchmark_id`, `source_sweep_sha256`, `selector`, `candidate_pool_sha256`, `reference_point`, `selected_candidate_ids`, and one frozen nominal prediction for every candidate in the pool. The holdout manifest must contain the same selector and source/candidate-pool digests plus complete observations for every candidate in every named shift. Neither predictions nor shifted observations are inferred or invented by the adapter. The split fields in the input document are declarations, not cryptographic attestations of how data were collected. Reviewers must verify the hashes against controlled artifacts and independently check that the holdout really remained sealed.

## Required shift families

At minimum, preregister several distinct families, not just larger versions of training-time noise:

- **Parameter drift:** material or boundary parameters move outside the nominal calibration region.
- **Constraint-boundary drift:** feasibility margin changes near a safety or manufacturability boundary.
- **Structural discrepancy:** the model form changes, e.g. an omitted interaction, new discontinuity, or altered causal dependence.
- **Spatially localized defect:** a local region reverses ranking or feasibility, exposing simulator exploitation.
- **Numerical / discretization shift:** mesh, solver tolerance, convergence regime, or reduced-order approximation changes.
- **Noise / measurement regime shift:** variance becomes heteroscedastic or correlated and changes near critical regions.
- **Zero-shift control:** expected to reproduce the locked reference evaluator within declared numerical tolerances.

Every shift family needs predeclared parameter ranges, severity, deterministic seed policy, and simulator/evaluator digest. Do not treat a random seed as a substitute for independent physical or scenario coverage.

## Metrics emitted by the evaluator

- **Selected feasibility rate:** fraction of selected-candidate × shift pairs with observed constraint margin at least zero.
- **Selected false-feasible rate:** among selected pairs predicted feasible by the nominal margin, fraction observed infeasible under the shift.
- **Selected false-infeasible rate:** among selected pairs predicted infeasible by the nominal margin, fraction observed feasible under the shift.
- **Normalized objective prediction error:** absolute difference between nominal prediction and shifted observation, normalized by each frozen reference-point coordinate. Treat it as shift discrepancy, not an unbiased prediction-error estimate when the shift is intentional.
- **Finite-pool hypervolume retention:** shifted hypervolume of selected, actually feasible candidates divided by hypervolume of all actually feasible candidates in the evaluated finite pool.
- **Finite-pool normalized regret:** one minus the preceding retention, when the pool's feasible hypervolume is nonzero.
- **Worst-shift retention / regret:** exposes tail failures that a mean can hide.

Finite-pool regret is only regret relative to the declared evaluated pool, not the unknown global optimum. When no feasible pool reference exists, the regret is null, not zero. Metrics across candidate × shift rows are generally dependent; do not treat them as IID Bernoulli trials for a confidence claim without a justified sampling design. Use a predeclared cluster-aware uncertainty procedure over independent tasks, candidates, and shift families.

## Known-answer controls

The evaluator test module contains known-answer controls for false-feasible and false-infeasible decisions, finite-pool regret, all-selected zero regret, malformed/incomplete data, source-run identity, evaluator source digest binding, and no-feasible-pool behavior. The adapter test module checks exact sweep-byte binding, adapter-source digest binding, complete candidate and shift coverage, selector ambiguity, non-synthetic rejection, selected IDs, and evaluation cost accounting. There are 13 evaluator test methods and 12 adapter test methods authored; those counts describe test cases, not a run result.

These tests are mechanism checks, not evidence that the optimizer performs well on casting physics. They should run in the repository's locked environment before any benchmark claims.

## Proposed gates (targets, not results)

1. **Integrity gate:** all known-answer tests pass; missing, duplicate, non-finite, out-of-range, or malformed records fail closed.
2. **Zero-shift gate:** score equivalence to the locked baseline inside an explicitly justified numerical tolerance.
3. **Decision-quality gate:** report paired per-task finite-pool hypervolume retention and worst-shift regret at matched fidelity cost. Compare with all preregistered baselines; no aggregate rank without per-task results.
4. **Safety gate:** preregister a false-feasible risk target and evaluate it using a separate independent-validation protocol with sampling assumptions stated. The present evaluator is descriptive and does not certify risk.
5. **Generalization gate:** show performance across all held-out shift families and severities, including worst-case or lower-tail results; no selection of favorable shifts after unsealing.
6. **Physical credibility gate:** document provenance, process coverage, metrology uncertainty, and simulator-to-measurement discrepancy. No synthetic or simulator-only result justifies a physical foundry claim.

## Competitive framing

The relevant comparison is not only final nominal hypervolume. Compare cost-normalized anytime performance, true held-out feasibility, false-feasible decisions, shift-wise hypervolume retention, worst-shift regret, wall time, memory, and calibration / validation overhead. Include strong multi-fidelity constrained baselines (including MF-SCBO after code and environment locking), multi-objective multi-fidelity methods, and simple controls. Lock every baseline to an immutable source revision and environment before comparing.

The candidate research hypothesis is that **decision-impact-weighted fidelity allocation plus post-selection robustness auditing** can outperform methods that optimize the nominal surrogate while making robustness failures visible. This is an empirical hypothesis, not a novelty or superiority claim.

## CLI smoke test

From the repository root, run:

```sh
python3 tools/casting-benchmark/decision_robustness_eval.py \
  tools/casting-benchmark/fixtures/decision_robustness_toy.json \
  --output /tmp/casting-decision-robustness-receipt.json
python3 tools/casting-benchmark/test_decision_robustness_eval.py
```

The toy fixture intentionally selects a candidate that violates a constraint under one of two shifts while a better feasible candidate remains unselected. The receipt should report selected false-feasible count 1 of 2 predicted-feasible candidate/shift pairs, selected feasibility rate 0.5, worst-shift finite-pool retention 0, and worst-shift finite-pool regret 1.0. These are known-answer fixture values, not casting performance results. The fixture's digest strings are placeholders for schema tests only; replace them with verified artifact digests in a real experiment.

## Adapter contract and reproducible invocation

From the repository root after generating an unaltered compositional sweep JSON:

```sh
python3 tools/casting-benchmark/adapt_bakeoff_to_robustness.py \
  sweep.json frozen-selection-and-predictions.json sealed-holdout-shifts.json \
  --output /tmp/robustness-input.json
python3 tools/casting-benchmark/decision_robustness_eval.py \
  /tmp/robustness-input.json --output /tmp/robustness-receipt.json
python3 tools/casting-benchmark/test_decision_robustness_eval.py
python3 tools/casting-benchmark/test_bakeoff_robustness_adapter.py
```

The adapter only accepts the existing compositional synthetic sweep format today. The selection manifest must bind to the exact bytes of that sweep and its candidate-pool digest. The selected set is explicit and frozen; the full candidate prediction list must contain every ID from `0` through `design_count - 1`. The holdout manifest must repeat the selector and sweep/pool digests and provide every candidate exactly once for each shift. Missing values, duplicate candidate IDs, ambiguous run selectors, or digest mismatches abort conversion.

The nominal prediction table and shifted observations are separate upstream artifacts: this adapter is **not** a shift generator, a casting simulator, or a model that generates predictions. No adapter run has yet been made against an actual saved bake-off JSON plus independent shift manifests. The current-head test methods have not been executed after the adapter/schema integration. The repository CI now includes a standard-library-only job for both test modules and the toy CLI smoke test; a final result must be tied to the exact resulting head SHA. The schema and adapter tests remain synthetic known-answer tests until that end-to-end execution is archived.

## Current literature and comparator additions (reviewed 2026-10-28)

- **MF-SCBO** (Palazzolo, Binois, Giraldi; arXiv:2609.29941, submitted 24 September 2026): constrained multi-fidelity Bayesian optimization for high-dimensional black-box constraints, arbitrary fidelities and non-nested sampling. Public implementation: https://github.com/Nemo-Virtual-Lab/MF-SCBO. Treat it as a required candidate baseline, but lock an immutable commit, package environment and adapter before drawing comparisons. An author's benchmark result is not a result for this casting suite.
- **TS-MFBO** (Applied Soft Computing 201, 2026, article 115490; DOI https://doi.org/10.1016/j.asoc.2026.115490): two-stage multi-fidelity BO with landscape-aware initialization and dual-adaptive sampling. Add it to the comparator registry only after locating and freezing a reproducible implementation and configuration.
- **Data-driven uncertainty modeling for robust superalloy investment-casting design** (Materials Today Communications, 2026; DOI https://doi.org/10.1016/j.mtcomm.2026.114659): application-specific evidence that process variability belongs in casting design itself, not merely in a post-hoc metric. This is a manufacturing-method comparator, not a plug-in optimizer baseline; align alloy, process assumptions, units and outcome definitions before comparison.
- **Risk-based VVUQ / metrology integration** (NIST AMS 100-85, published 5 October 2026; DOI https://doi.org/10.6028/NIST.AMS.100-85): supports allocating validation effort by model influence and consequence, and keeping predictive uncertainty connected to measured physical quantities.

The review literature also identifies recurring barriers: case-specific datasets, weak cross-alloy and out-of-domain validation, limited UQ, expensive multiphysics simulation, and lack of shared/repeatable casting benchmarks (review DOI https://doi.org/10.1007/s43939-026-00685-5). The benchmark's most credible contribution could be a reusable protocol that makes such failure modes measurable, not a claim that an acquisition heuristic is novel.

### Comparator lock criteria

For every algorithm, archive source URL, immutable commit or release, license, dependency lock, hardware/runtime, acquisition settings, initialization budget, per-fidelity cost mapping, stopping criteria, objective/constraint adapter, and source/configuration/environment digests. Count tuning runs against an explicit tuning budget. If the original method cannot be reproduced under comparable conditions, label the comparator unavailable rather than silently replacing it with a similarly named heuristic.

### A falsifiable research hypothesis

At equal *total cost* (fidelity-weighted simulator cost plus policy/solver wall time), adaptive joint action selection plus post-selection shift auditing should improve the lower-tail feasible hypervolume retained under model shifts, while maintaining nominal performance within a predeclared non-inferiority margin. Falsify the hypothesis if it improves only average nominal hypervolume, if its worst-shift results regress beyond the declared margin, or if the overhead erases its evaluation-cost advantage.
