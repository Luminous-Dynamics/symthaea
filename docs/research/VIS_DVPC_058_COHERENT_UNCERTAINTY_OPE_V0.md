# VIS-DVPC-058 — Coherent source-profile uncertainty for policy-portfolio OPE v0

**Status:** finite-scenario robust optimization benchmark on synthetic small contextual bandits. It is not a continuous uncertainty-set guarantee, external-data result, clinical validation, SOTA claim, or deployment certificate.

## Why this is the next step

VIS-DVPC-054–056 model heterogeneous and correlated annotation sources. VIS-DVPC-057 allocates a shared evidence budget to a portfolio of candidate policies. Those optimizers still select against one declared source profile. A pilot may underestimate the bias of an entire source family, the variance of its outputs, or a common rubric shock. Treating every cell's parameters as independently adversarial is usually too pessimistic and can describe no plausible data-generating process.

APRIL, *Auditing Predictions of Raters for LLM-generated Counterfactual Annotations* (ML4H 2026), motivates entropy-aware stopping for annotation acquisition: https://proceedings.mlr.press/v297/mandyam26a.html

The non-rectangular uncertainty literature in robust MDPs is a useful conceptual comparator: coherent model parameters should remain linked across states/cells instead of being independently selected at every cell: https://doi.org/10.1137/24M1631250

## Research question

Can one shared annotation allocation protect an entire policy portfolio when source bias, marginal variance and shared-factor covariance are uncertain but tied together by a globally coherent source-calibration process?

## Finite coherent uncertainty set

For each base dependence world from VIS-DVPC-056, v0 derives four globally coherent source profiles:

- `nominal`: the original profile;
- `low_noise`: all source biases scaled by 0.75, all source excess variances scaled by 0.65, and shared-factor variances scaled by 0.50;
- `high_dependence`: all source biases scaled by 1.25, all source excess variances scaled by 1.35, and shared-factor variances scaled by 1.75;
- `biased_pilot_missed_shock`: all source biases scaled by 1.10 plus one source-specific offset `[+0.035, -0.025, +0.020]` applied identically across every context-action cell; source excess variances are scaled by 1.15 and shared-factor variances by 1.60.

These are common source-family transformations across every cell, not cell-wise independent perturbations. Loadings, reward means, factual logger, target policies, source costs and capacities remain fixed in this first benchmark. Each generated profile is validated for finite parameters, PSD low-rank covariance and shared variance not exceeding source marginal variance.

The scenario set is finite and deterministic. Its exact contents are derived in `coherent_profile_scenarios` in the Rust source. The workflow receipt hashes that source and this protocol; no continuous-set guarantee is inferred from four scenarios.

## Robust objectives

Let `S` be the finite scenario set and `MSE_j(n,s)` be the exact DM+IS / annotated-DR MSE for portfolio policy j, allocation n and coherent profile s.

**Worst-profile prior-weighted risk:**

\[
J_{RP}(n)=\\max_{s\\in\\mathcal S}\\sum_{j=1}^{K}w_j MSE_j(n,s).
\]

**Worst-profile worst-policy risk:**

\[
J_{RM}(n)=\\max_{s\\in\\mathcal S}\\max_{j\\in[K]}MSE_j(n,s).
\]

Two independent exhaustive oracles enumerate the same feasible context allocation combinations and optimize each scalar objective over all four coherent profiles. The one allocation is shared across every target policy and every scenario. The oracle is globally optimal for the registered finite fixture only.

## Compared plans

- nominal prior-weighted exact portfolio allocation;
- nominal worst-policy exact portfolio allocation;
- independent-profile prior-weighted plugin allocation, evaluated under every correlated coherent profile;
- finite-scenario robust prior-weighted exact allocation;
- finite-scenario robust minimax exact allocation;
- finite-scenario robust minimax greedy baseline.

The greedy method is a baseline, not an optimizer certificate. No cell-wise rectangular worst-case plan is called robust; that would change the uncertainty set.

## Evaluation protocol

For each base dependence world, each plan is evaluated under all four profiles. Every profile row contains prior-weighted analytic MSE, worst-policy analytic MSE, per-policy analytic MSEs, common-sample Monte Carlo best-policy selection accuracy and expected simple regret, source counts, and gaps to both robust exact objectives.

The Monte Carlo evaluator draws one annotation-fitted reward model shared by all portfolio policies, then uses the same contexts, logged actions, factual rewards and propensities for all policies. Each profile/method receives an independent deterministic seed stream. These empirical selection metrics are decision diagnostics; they are not confidence intervals or simultaneous coverage guarantees.

## Output

- 12 `ROBUST_CONFIG` rows: 3 base dependence worlds × 4 coherent profiles;
- 72 `ROBUST_PROFILE_RESULT` rows: 3 base worlds × 6 allocation plans × 4 profiles;
- each result row has 20 TSV fields, including two exact robust-oracle gaps;
- exact source, profile-construction code, protocol and smoke/full outputs are bound by SHA-256 receipt.

## Acceptance gates

1. Every coherent scenario validates, with global source transforms applied consistently across all cells.
2. A singleton scenario set returns the same exact allocation/objective as the corresponding nominal portfolio oracle for both prior-weighted and worst-policy criteria.
3. Both robust exact objectives match independent brute-force enumeration over the entire finite feasible allocation set.
4. A robust exact allocation is no worse than the nominal independent-profile plugin on the same declared scenario set for its respective objective.
5. Source shared-factor covariance remains PSD and respects marginal-variance limits in every scenario.
6. All profile/method rows are deterministic, complete and schema-checked.
7. Best-policy regret and selection accuracy are reported under each profile; no simultaneous inference claim is made.
8. Annotations affect only the reward-model component; the factual logged reward/propensity residual and support gate are unchanged.

## Industry comparison and next steps

VIS-DVPC-057 is the shared multi-policy evidence baseline; VIS-DVPC-056 provides the correlated annotation model. VIS-DVPC-058 adds coherent profile uncertainty, which is distinct from a rectangular worst-case stress test. APRIL supplies an annotation stopping/entropy comparator. The 2026 non-rectangular robust-MDP literature motivates coherent scenario construction.

Next: estimate coherent profile regions from an independent overlap pilot; add a scenario-generation/separation oracle for continuous bias and low-rank covariance uncertainty; compare finite-scenario robust optimization with Bayesian model averaging and distributionally robust ambiguity sets; and validate an entropy-based stopping rule against actual policy MSE and best-policy regret. Report the uncertainty-set diameter and exact scenario contents with every result.

## Scope limits

v0 fixes source loadings, costs, capacities, reward means, behavior propensities and target policies. It uses four discrete global profile transforms and three synthetic dependence worlds. It does not prove coverage for unseen profiles, continuous robust optimality, adaptive sequential acquisition, nonstationary data, or causal identification under hidden confounding.