# CASTING External Baseline Audit: MF-SCBO v1

**Related work:** [CASTING-0032 issue #8789](https://github.com/Luminous-Dynamics/symthaea/issues/8789), [draft PR #8790](https://github.com/Luminous-Dynamics/symthaea/pull/8790)  
**Audit date:** 2026-10-28  
**Purpose:** record pre-run comparison hazards; this is not a benchmark result.

## 1. Immutable upstream source

- Repository: https://github.com/Nemo-Virtual-Lab/MF-SCBO
- Audited commit: [ca42fd6c5ec8c841e3f1ff9642509e12d104b581](https://github.com/Nemo-Virtual-Lab/MF-SCBO/commit/ca42fd6c5ec8c841e3f1ff9642509e12d104b581)
- Upstream commit date returned by GitHub: 2026-09-29T12:10:24Z
- License at that revision: MIT; preserve attribution/license when redistributing or adapting.
- Optimizer source path: https://github.com/Nemo-Virtual-Lab/MF-SCBO/blob/ca42fd6c5ec8c841e3f1ff9642509e12d104b581/mfscbo/mfscbo_optimization.py
- Dependency declaration: https://github.com/Nemo-Virtual-Lab/MF-SCBO/blob/ca42fd6c5ec8c841e3f1ff9642509e12d104b581/requirements.txt

Commit immutability is necessary but not sufficient. The published requirements list names NumPy, SciPy, PyTorch, BoTorch, GPyTorch, torchvision and plotting/documentation packages without version pins. A credible run must resolve and archive a separate exact environment (including Python, OS/platform, accelerator/device, PyTorch/BoTorch/GPyTorch versions and deterministic settings) and its digest. Do not call an unpinned install reproducible merely because source is pinned.

## 2. Important comparability hazards found in the upstream source

### 2.1 It is a single-objective constrained optimizer

The upstream example passes one function per fidelity to the objective list; the implementation stores one scalar best objective and one best design. This is not a native multi-objective Pareto optimizer.

**Required handling:** do not score this implementation directly as if it returned a multi-objective Pareto front. Either:
- use it in a separately preregistered single-objective constrained casting track; or
- choose a scalarization before the experiment, freeze the scalarization/weights, and state that the result compares the scalarized problem only.

Use a true multi-objective multi-fidelity comparator for hypervolume/Pareto claims. Do not let post-hoc scalarization choices determine the winner.

### 2.2 Shared constraints across fidelity levels

The public interface accepts a constraint callable over design variables, not a fidelity-indexed constraint oracle. Code comments also explicitly assume constraints are shared across fidelities in the infeasible-state path.

**Required handling:** use MF-SCBO in the shared-constraint applicability track. In the fidelity-dependent-constraint stress track, label it out-of-assumption and do not aggregate it into the same score. This avoids judging an implementation against a problem outside its stated assumptions.

### 2.3 Diagnostic test mode can evaluate the high-fidelity objective for reporting

The public README example initializes the optimizer with test mode enabled and predicted centering. In the optimizer loop, that mode calls the highest-fidelity objective on the current best predicted candidate to populate the reported best objective. That is an extra high-fidelity evaluation relative to a prediction-only reported trace; reporting and evaluation accounting can therefore be contaminated if the mode is used naively.

**Required handling:** disable test mode for the primary policy run. Score candidates only through the sealed evaluator after the policy has finished, unless an explicitly allowed high-fidelity action was requested by the policy and charged in the ledger. If diagnostic mode is ever used in a separate replication, isolate it from policy inputs and count every oracle call. Never use a test-mode result as a supposedly free, policy-independent metric.

The implementation can also take an explicit high-fidelity corrective action in predicted-centering mode when its leave-one-out check indicates the center prediction error exceeds the declared interval. That is an algorithm action, not a free diagnostic; it must appear as an evaluation in the action trace and cost ledger.

### 2.4 The listed cost is not total run cost by itself

The optimizer accepts a per-fidelity cost vector and logs cumulative evaluation cost. The reported quantity does not by itself prove that acquisition/model fitting time, initialization/tuning, environment overhead, or independent shift auditing are accounted.

**Required handling:** archive the native fidelity-count trace and independently recompute fidelity-weighted cost. Also measure wall/CPU time and optimizer overhead where possible. Report:
1. oracle-only cost;
2. optimizer + oracle cost;
3. full end-to-end cost including necessary reference/audit/validation work.

Use identical fidelity cost assumptions across policies in the primary track and include sensitivity analysis for plausible ratios.

### 2.5 Environment and execution controls

The dependency declaration is unpinned and includes both runtime and documentation/plotting packages. Resolve exact versions and archive the lock/manifest plus platform and device data. Begin with a CPU, float64 reference execution if practical; any accelerated execution must record deterministic settings and hardware. Treat source reproducibility and execution reproducibility as separate gates.

## 3. Frozen baseline card to create before execution

Required fields:

- source repository, commit SHA, file digests, license;
- resolved lockfile and hash; Python, OS, CPU/GPU, PyTorch, BoTorch and GPyTorch versions;
- mode: predicted or evaluated centering; batch mode/size; restart flag; random/Sobol seed;
- initial points at each fidelity, bounds and normalization;
- fidelity cost vector and actual per-fidelity call counts;
- exact constraint track and semantics;
- single-objective objective definition or frozen scalarization;
- stopping budget and handling of failures/timeouts;
- policy-run outputs separated from evaluator-only high-fidelity truth;
- wall-clock, CPU time, model-fitting time, memory where measured;
- all tuning runs and the frozen configuration digest.

Missing fields mean the baseline is **not yet reproduced**, not that Symthaea has beaten it.

## 4. Minimal validation before comparative use

1. A tiny known-answer test verifies normalized bounds, objective direction and constraint sign convention.
2. A sentinel oracle records every high-fidelity call and proves that no evaluator-only result is exposed to acquisition or stopping logic.
3. The primary run disables test mode; policy-triggered high-fidelity corrective evaluations remain counted.
4. The trace count, native cost trace and independent recomputation agree exactly.
5. Two runs with the same locked source/environment/config/seed have reproducibility documented; any nondeterminism is measured rather than assumed absent.
6. Shared-constraint and fidelity-dependent-constraint tracks are reported separately.
7. A single-objective MF-SCBO result is never presented as direct evidence of multi-objective hypervolume superiority.

## 5. Decision

MF-SCBO is a strong candidate for the single-objective constrained, shared-constraint track once its environment and run configuration are locked. It is not a direct substitute for a native multi-objective baseline. The clean comparison path is therefore: MF-SCBO for its applicable track, plus a reproducible multi-objective multi-fidelity baseline for Pareto/hypervolume claims, all under matched total-cost budgets.

No external baseline execution has been run in this audit. It records source reconnaissance and controls required before execution, not measured performance.

## Sources

- MF-SCBO preprint: https://arxiv.org/abs/2609.29941
- Upstream repository and code at the pinned commit: https://github.com/Nemo-Virtual-Lab/MF-SCBO/tree/ca42fd6c5ec8c841e3f1ff9642509e12d104b581
- Cost-aware stopping (ICML/PMLR 306, 2026): https://proceedings.mlr.press/v306/xie26c.html
- COCO BBOB-constrained official archive: https://coco-platform.org/data-archive/index.html


## 6. Additional 2026 comparator: consequential improvement

A newly reviewed article is directly relevant: Aydogdu et al., *Consequential improvement acquisition function for efficient multi-fidelity Bayesian optimization*, published 27 March 2026 and assigned to the September 2026 issue, DOI https://doi.org/10.1007/s11081-026-10078-5. The method jointly selects the sample and fidelity while accounting for whether a low-fidelity improvement is consequential at high fidelity, and addresses evaluation cost/validity differences.

This is a stronger conceptual comparator than generic expected-improvement-per-cost. Consequently, do not claim that Symthaea's joint action/fidelity choice or cost-aware sampling is novel by itself. The candidate differentiation is the combination of multi-objective constrained decision quality, post-selection model-shift robustness, explicit abstention, and auditable end-to-end cost/trace semantics.

The article page says supporting data/code are available upon reasonable request; this audit found no clearly linked public implementation in the retrieved article page. Therefore:
- record CI acquisition as a **paper comparator / implementation sought**;
- do not mark it reproduced or beaten until an actual code artifact, revision/license, environment and runnable configuration are obtained;
- if no runnable artifact is available, report the comparator as unavailable rather than substituting a newly written imitation under the same name.

Source: https://link.springer.com/article/10.1007/s11081-026-10078-5

## 7. VOI-specific predictive-model gate

The exact discrete VOI calculator added to the research branch is mathematically exact only with respect to its supplied prior, loss table and observation likelihoods. This does not establish that these distributions match the actual process.

Before a model-driven VOI policy participates in a primary comparison, archive a calibration study using predictions made before observations are revealed. Include held-out predictive log score or Brier score, reliability diagnostics where applicable, and stratification near the constraint boundary and across shift families. If no suitable calibrated distribution exists, use the deterministic proxy track and label it accordingly. A correct decision-theoretic formula paired with a misspecified model can systematically purchase the wrong evaluations.

No calibrated predictive model or performance improvement is claimed by the current exact-discrete reference calculator.


## 8. Prequential forecast scoring prerequisite

The branch now includes `tools/casting-benchmark/forecast_validation.py`. It scores probability vectors made available before an outcome against the eventual observed outcome using multiclass Brier score, log loss, fixed-bin top-label reliability and regime-stratified reports. The outcome record binds to the canonical forecast digest; the scorer rejects any forecast bytes that no longer match the recorded digest.

This is an offline validator, not a timing authority. A valid digest proves byte agreement, not that the forecast was actually emitted before the outcome. A production evaluation must also preserve the policy's input/output boundary and independently archive forecast-time events; do not allow the optimizer to read any outcomes or holdout shifts before decisions are frozen.

The score report is descriptive. For repeated observations, define the independent unit and uncertainty method; do not treat correlated rows as independent. A low score on one synthetic regime does not imply physical calibration, and a high aggregate score must not hide boundary/shift degradation.

Implementation and unit tests are committed to the research branch and wired to the focused CI job. They have not yet been claimed passing pending an exact-head run.

## 9. Multi-objective comparator card: MF-JESMOC / MOBOCMF

- Paper: Fernández-Sánchez and Hernández-Lobato, *Joint entropy search for multi-objective Bayesian optimization with constraints and multiple fidelities*, Neurocomputing 657 (7 Dec 2025), article 131674, DOI https://doi.org/10.1016/j.neucom.2025.131674.
- Public repository: https://github.com/fernandezdaniel/MOBOCMF
- Pinned revision: [95293f6b59375ef9d4c71f61d4e42d788d116da8](https://github.com/fernandezdaniel/MOBOCMF/commit/95293f6b59375ef9d4c71f61d4e42d788d116da8), latest commit returned by GitHub's repository search, dated 25 April 2024.
- Main acquisition implementation: https://github.com/fernandezdaniel/MOBOCMF/blob/95293f6b59375ef9d4c71f61d4e42d788d116da8/mobocmf/acquisition_functions/JESMOC_MFDGP.py
- Public toy example: https://github.com/fernandezdaniel/MOBOCMF/blob/95293f6b59375ef9d4c71f61d4e42d788d116da8/examples/toy_synthetic_2D_JESMOCMF/toy_synthetic_2D_JESMOCMF.py
- The example configures two objectives, two constraints, two fidelities, per-evaluation cost units 1 and 10, and 60 Bayesian-optimization iterations. It is a genuinely relevant multi-objective constrained multifidelity baseline candidate, unlike single-objective MF-SCBO.

### Reproduction hazards

1. **Dependencies are not locked.** README lists only `botorch >= 0.9.0`; no root `requirements.txt` was found. Resolve and archive exact Python, PyTorch, BoTorch, GPyTorch, SciPy, NumPy and other runtime versions plus platform/device/determinism settings.
2. **License needs resolution.** `setup.py` declares `Apache License 2.0`, but a root `LICENSE` file could not be fetched at this revision. Do not redistribute copied implementation or assert license compliance until the exact license text/terms are verified. Source inspection and benchmarking discussions should keep this unresolved status explicit.
3. **Coupled versus decoupled evaluation.** The `coupled_acq` path sums objective and constraint acquisition terms, and the `_get_nextpoint_coupled` selection then divides by aggregate evaluation cost at each fidelity. In `decoupled_acq`, per-black-box cost division is commented out; a decoupled experiment would therefore require a separately reviewed and labelled correction. For faithful reproduction, start with coupled evaluation only.
4. **Keep diagnostic truth out of the policy.** The API exposes `eval_highest_fidelity`; the public example should be audited for this setting, and the primary baseline should use the normal fidelity-selection mode. Any high-fidelity call performed only to score a final result must be evaluator-only and charged to the common audit/total-cost ledger, never exposed to acquisition or stopping.
5. **Reproduce the same mathematical track.** Verify objective directions, constraint signs, feasible-set definition, objective normalizations and reference point against the paper before comparing hypervolume. Keep the exact method configuration and random seed fixed; count initial low/high-fidelity observations and all fitting/restarts.

### Fair benchmark placement

MF-JESMOC is the strongest currently identified open-source candidate for a native multi-objective constrained multi-fidelity comparator. It should be paired with an up-to-date maintained baseline such as BoTorch's published MF-HVKG/MOMF implementation, not used as the only comparator. The official BoTorch tutorial describes MF-HVKG as one-step lookahead on value of information about the Pareto set and MOMF as an alternative multi-objective multi-fidelity approach: https://botorch.org/docs/v0.13.0/tutorials/Multi_objective_multi_fidelity_BO.

Status: **candidate identified, source revision pinned, not reproduced**. No performance result or SOTA comparison is claimed. Do not close the reproduction gate until license, runtime environment, numerical conventions, fidelity costs, and policy/evaluator truth separation are archived.
