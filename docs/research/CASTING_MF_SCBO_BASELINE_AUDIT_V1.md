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
