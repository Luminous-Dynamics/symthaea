# CASTING Decision Robustness Audit v1

**Benchmark ID:** CASTING-DECISION-ROBUSTNESS-001  
**Status:** research harness implemented; ready for controlled known-answer testing, not yet integrated with the main bake-off  
**Scope:** two-objective constrained optimization with a fixed reference point  
**Evidence labels:** synthetic / simulator / physical must remain distinct

## Why this benchmark exists

A search policy can exploit defects in its design-time simulator. Improving the surrogate's in-domain fit or the nominal Pareto front does not guarantee that the selected decisions remain feasible or good when process parameters, boundary conditions, material properties, discretization, or model structure shift. This track measures the consequences of selection under explicit post-selection shifts rather than scoring only the optimizer's nominal trace.

It complements, rather than replaces, the compositional multi-fidelity bake-off and promotion_risk_gate.py. This evaluator computes descriptive outcomes. It never certifies false-feasible risk, proves split independence, or establishes physical validity.

## Frozen execution contract

1. Freeze the candidate pool, nominal predictions, selected-candidate flags, reference point, source revision, and selection trace **before** running any holdout shift. Record the frozen-selection SHA-256.
2. Freeze a holdout manifest with scenario-family definitions, parameter ranges, seeds, simulator revision, and evaluation budget. Record its SHA-256. Do not use these holdout observations to tune or reselect candidates for the same reported run.
3. Evaluate **every candidate in the declared pool under every shift**. Missing or duplicate candidate/shift rows fail closed. This makes finite-pool regret comparable and prevents selective omission of bad candidates.
4. Normalize both minimized objectives into the interval [0, reference_point[i]) using transformations frozen before evaluation. The reference point is shared across policies and shifts; do not tune it per run.
5. Score raw inputs with tools/casting-benchmark/decision_robustness_eval.py. Archive input bytes, receipt, evaluator revision, environment, and source digests. The receipt hashes the exact input bytes.
6. Keep one row per task, selection, candidate, and shift in the raw archive. Preserve failed, timed-out, invalid, and refused evaluations as outcomes in the acquisition layer; do not omit failures from the accounting.
7. Report synthetic stress fixtures, public optimization benchmarks, engineering simulator results, and physical foundry observations as separate evidence classes.

The split fields in the input document are declarations, not cryptographic attestations of how data were collected. Reviewers must verify the hashes against controlled artifacts and independently check that the holdout really remained sealed.

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

The test module contains synthetic controls for: one selected design turning infeasible under one of two shifts; a better unselected feasible candidate producing nonzero finite-pool regret; selecting the entire pool producing zero finite-pool regret; false-infeasible detection; complete candidate × shift accounting; duplicate ID rejection; holdout-role rejection; input-byte digest binding; objective/reference-point validation; and no-feasible-pool behavior.

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
