# Scientific Law Discovery Under Unknown Noise and Adaptive Sampling

**Protocol v0.1 — preregistration proposal only. No method-performance or SOTA claim is made.**

Issue: [SCI-NOISE-001 (#8607)](https://github.com/Luminous-Dynamics/symthaea/issues/8607)  
Parent research: [truth-absent recovery (#8536)](https://github.com/Luminous-Dynamics/symthaea/issues/8536)  
Stacked exploratory implementation: [operator-held-out recovery PR #8584](https://github.com/Luminous-Dynamics/symthaea/pull/8584)

## 1. The missing qualification

The current integrated prequential-discovery pilot uses a known Gaussian noise model and a declared predictive scale. That makes it a useful controlled wiring test, but it does not establish reliable behavior when the observation process is unknown or misspecified. A monitor can alarm because the scientific law is wrong, because the noise model is wrong, or because dependence and adaptive selection violate the monitor's assumptions.

The benchmark must measure those failure modes separately. A robust residual score is not automatically a calibrated probability; a detector alarm is not a replacement law; and a low training residual is not a validated discovery.

## 2. Falsifiable questions

1. At a fixed false-alarm operating point, does a monitoring policy detect a structurally wrong law earlier than matched-budget baselines?
2. How much does a known-Gaussian PIT detector's clean-world false-alarm rate change under heteroscedasticity, heavy tails, contamination, dependence, censoring or drift when the underlying law is correct?
3. Does noise-aware monitoring improve *validated* law recovery, or does it merely cause more restarts and candidate proposals?
4. Under adaptive experiment selection, which calibration claims remain valid under the actual conditional observation process?
5. When the evidence cannot distinguish a law error from a noise-process change, does the system preserve the ambiguity or report a misleading diagnosis?

These questions are separate endpoints. There is no single aggregate score that can substitute for them.

## 3. Factorial world design

Cross two independently controlled axes:

- **Scientific structure:** target family, equation complexity, dimensional constraints, parameter scales, and whether the truth is in-grammar, truth-absent or operator-held-out.
- **Observation process:** the noise/measurement regimes below.

Keep four primary cells distinct:

| Law status | Noise status | What it isolates |
|---|---|---|
| Correct provisional law | Matched observation model | Clean calibration control |
| Correct provisional law | Misspecified observation model | False alarms from noise assumptions |
| Wrong provisional law | Matched observation model | Structural discrepancy detection |
| Wrong provisional law | Misspecified observation model | Combined, attribution-hard case |

Add explicitly non-identifiable worlds, in which more than one candidate is observationally equivalent on the permitted design. These must not be scored as ordinary law-recovery failures or removed silently. Report evaluator eligibility and performance conditional on eligibility separately.

### Versioned observation regimes

The machine-readable companion `SCIENTIFIC_DISCOVERY_NOISE_ROBUSTNESS_V0_1.json` defines a starter matrix.

- `gaussian_homoscedastic`: matched baseline.
- `gaussian_heteroscedastic`: scale varies with pre-outcome covariates under a frozen function.
- `student_t_heavy_tail`: fixed degrees of freedom and scale.
- `contamination_outliers`: a frozen sparse contamination probability and magnitude law.
- `ar1_residuals`: frozen serial dependence, tested with both fixed and adaptive designs.
- `censored_or_missing`: explicit generation and admission mechanism; missingness is not encoded as a zero residual.
- `scale_drift`: a frozen change in noise scale over predeclared time or operating-regime blocks.
- `combined_shift`: reserved for a later stress suite after the individual regimes are understood.

The policy receives only permitted design metadata, its past observations and its own pre-outcome forecasts. It must not receive the hidden law, noise-family label, latent noise draw, true residual, evaluator labels or evaluator holdouts.

## 4. Experimental protocol and anti-leakage rules

1. Freeze world families, noise mechanisms, seed lists, query budgets, model candidates, calibration partitions, decision thresholds, eligibility criteria and metric code before confirmatory execution.
2. Derive discovery designs before choosing evaluator-only law/noise labels. Use independent deterministic RNG streams for public design, target generation, noise, in-domain evaluation, marginal shifts, joint extrapolation and near-boundary tests.
3. Produce a timestamped or sequentially indexed predictive distribution before each measurement. Store the exact forecast, candidate/model digest, selected action, observable context and subsequent outcome. A post-outcome forecast reconstruction is invalid.
4. Pair policies on the same world, seed, design opportunities, noise realization where scientifically appropriate, and declared observation/computation budget. Report when coupling noise across policies is inappropriate for a policy-dependent intervention.
5. Separate discovery observations, method/calibration tuning data, fresh post-recovery policy-visible measurements, and evaluator-only score data. No evaluator score can affect candidate generation, fitting, ranking, stopping, monitor calibration or grammar expansion.
6. Run fixed-design and adaptive-design conditions separately. Do not assume IID, exchangeability or a martingale guarantee merely because a score is called PIT, rank, conformal or prequential. Document the precise conditional assumptions each comparator needs.
7. A selected measurement can be invalid, censored, missing or numerically unevaluable. Preserve each category as a separate outcome. Do not silently discard it and reduce the denominator.
8. Once confirmatory evaluation opens, do not change a gate or threshold in response to results. Any repair becomes a new version and uses a fresh evaluation manifest.

## 5. Comparators

Use a layered comparison rather than pretending every published algorithm solves the same task.

### Monitoring / observation-process comparators

- Current fixed-Gaussian PIT monitor, only in the settings where its forecast contract is satisfied.
- A robust predictive likelihood / scale estimate, with parameters fit using permitted discovery or calibration data only.
- Randomized predictive ranks or PIT diagnostics, but only where their conditional validity requirements are met.
- Calibration-split or conformal interval baselines for settings where the split/exchangeability assumptions are defensible; use block-aware splits for block-dependent observations.
- No-monitor and stop-as-unresolved controls.

### Discovery and experiment-selection comparators

- Existing nominal expected-information-gain selector, random selection, and fixed-grid/round-robin under matched budgets.
- Current residual-guided typed correction/recovery path.
- A robust/noise-aware selector only as an explicitly proposed treatment, not as a presumed improvement.
- On compatible static equation-recovery tracks, include PySR/SymbolicRegression.jl, AI Feynman where its assumptions fit, and pinned ERBench/SRBench protocol versions. RESTART, GenSR and EGG-SR are important structural-search comparators, but their published results are not transferable to this active, factorial protocol without matched implementations and budgets.

Do not compare a policy's best-of-many run with a baseline's single run. Publish all attempted runs and resource accounting.

## 6. Endpoints

Report each endpoint overall and by structural family × noise regime × design mode, with full denominators and uncertainty intervals.

**Scientific recovery**
- Initial-pool reference-equivalence hit rate.
- Exact symbolic or declared equivalence-class recovery.
- In-domain, marginal-shift, joint-extrapolation and near-boundary prediction errors as separate metrics.
- Fresh post-recovery validation count; false confident-singleton rate; unresolved, rejected, and not-evaluable rates.

**Observation-process robustness**
- Clean-world false-alarm rate and detection rate under true structural discrepancy.
- Detection delay in number of paid observations.
- Predeclared proper predictive score, such as negative log score, when predictive densities are available and numerically well-defined.
- Coverage and interval width for interval methods; report conditional coverage by regime as well as marginal coverage.
- PIT/rank diagnostics only where assumptions are met.
- Noise-regime attribution accuracy only where the evaluator verifies that the regime is identifiable from available evidence; otherwise report ambiguity.

**Efficiency and reproducibility**
- Measurements to validated recovery or explicit unresolved.
- Candidate evaluations/attempts, wall-clock time, CPU/GPU resource use, and any external-tool cost.
- Deterministic replay, manifest digests, model/config/toolchain identity, failure reasons, and all seed-level outcomes.

Do not hide a high false-alarm rate, increased abstention, or a weak regime behind a pooled average. Use paired intervals and account for clustering by world; seeds from one world are not independent scientific worlds.

## 7. Proposed acceptance gates (not results)

- Known-answer tests verify every noise generator, its moments or declared quantiles, dependency behavior, reproducibility and invalid-input handling.
- The observation generator is independently testable from candidate search. No search/monitor API receives evaluator-only truth.
- Clean-law/noise-mismatch controls quantify false alarms separately from wrong-law/matched-noise detection.
- The actual adaptive selection mechanism is covered by tests; invalid assumptions yield an explicit unsupported/not-evaluable outcome.
- Censored/missing/numerically invalid measurements cannot be converted into zero residuals or silently removed.
- A recovered singleton must survive fresh policy-visible measurements and independent evaluator-only same-domain and relevant OOD checks.
- Confirmatory scientific structure coverage: at least 60 held-out worlds across 6 or more structural families and 20 paired seeds per world. Noise regimes must have a predeclared precision/power target; if run as a full factorial, publish the complete per-cell denominators.
- False-alarm target: at most 5% on each predeclared clean-law/noise-mismatch cell **only when the 95% upper confidence bound also meets the declared limit**. Select an adequate number of independent worlds for this bound before unblinding; a small fixed-seed pilot is not sufficient.
- Report comparative superiority only if an agreed primary endpoint improves against the strongest compatible matched baseline without violating false-confidence and unresolved-rate guardrails.
- Exact-head executable tests and CI are required for implementation qualification. Queued, skipped or stale-head checks are not passes.

## 8. Scope boundary and research references

This protocol is not a claim that all observation shifts can be statistically distinguished from law misspecification. Some worlds are fundamentally non-identifiable from the allowed observations; the correct output there is an equivalence class or unresolved result.

External context:
- [ERBench (2026)](https://arxiv.org/abs/2606.09276) explains why IID test accuracy can mislead equation discovery and defines a hidden evaluation track.
- [RESTART (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html) motivates residual-guided structural refinement and a reusable repair library.
- [GenSR (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html) explores latent-space guidance for symbolic search.
- [EGG-SR (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html) uses symbolic equivalence to reduce redundant search.
- [2026 symbolic-regression survey](https://doi.org/10.1007/s11831-026-10681-w) identifies realistic noise, missing data, shift and uncertainty quantification as continuing benchmark concerns.

**Status:** protocol PR only. The scenario manifest is a versioned design artifact, not an executable generator, validated statistical guarantee, or performance report.
