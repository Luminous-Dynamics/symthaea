# VIS-DVPC-056 — Correlated-source annotation and effective evidence v0

**Status:** synthetic covariance-aware stress benchmark layered on the exact-DP/MM implementation. This is not a public-data evaluation, SOTA claim, medical validation, or deployment certificate.

## Why this extension matters

VIS-DVPC-054/055 models source annotations as independent draws after conditioning on a fixed source bias and marginal variance. The 2026 multi-source OPE paper states its sources provide independent annotation draws. That makes its objective a valuable baseline, but not a safe default for panels that share model families, training corpora, prompts, reward rubrics, or institutional assumptions.

Kohli (June 2026), *Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels*, reports that nine frontier LLM judges from seven model families yielded about two independent votes' worth of information on the evaluated tasks; its reported accuracy was 8–22 percentage points below the independent-vote ideal. This is direct motivation to quantify evidence dependence, rather than equate source count with information count: https://machinelearning.apple.com/research/correlated-llm-evaluation-panels

The primary OPE reference remains Xiang et al. (2026), *Budgeted Multi-Source Counterfactual Annotation for Off-Policy Evaluation*: https://arxiv.org/abs/2610.10974. Its independent-source assumption is now made explicit as a switchable limiting case, not silently generalized.

## Error model

For source k, annotation j in context-action cell (x,a), model the error as:

\[
e_{k,j}(x,a)=\\epsilon_k(x,a)
+\\sum_{f=1}^{2}\\sqrt{\\tau_f^2}\\,l_{k,f}z_f(x,a)
+\\sqrt{\\sigma^2(x,a)+\\Delta_k(x,a)
-\\sum_f\\tau_f^2l_{k,f}^2}\\,\\eta_{k,j}(x,a).
\]

The z_f are shared latent shocks, drawn once per cell and shared across every annotation from every source with nonzero loading on factor f. The eta are independent residuals. Factual reward-model fitting observations remain independent Bernoulli samples. The source marginal error variance remains sigma-squared + Delta-k, while the covariance between two distinct annotations from sources k and l is:

\[
\\operatorname{Cov}(e_{k,j},e_{l,m})
=\\sum_f\\tau_f^2l_{k,f}l_{l,f},
\\qquad (k,j)\\ne(l,m).
\]

The covariance matrix is PSD by construction: it is a diagonal nonnegative residual covariance plus a sum of rank-one PSD shared-factor matrices. The validator also requires the shared marginal component for each source to be no larger than its total source marginal variance; otherwise the residual variance would be negative and the world is rejected.

For a cell with factual fit count nf and annotation counts nk, the pooled reward-model variance is:

\[
\\operatorname{Var}(\\widehat q_{xa})
=\\frac{n_f\\sigma_{xa}^2
+\\sum_kn_k(\\sigma_{xa}^2+\\Delta_k(x,a))
+\\sum_f\\tau_f^2\\left[
(\\sum_kn_kl_{k,f})^2-\\sum_kn_kl_{k,f}^2
\\right]}{(n_f+\\sum_kn_k)^2}.
\]

The final term sums off-diagonal covariance among distinct annotation observations, including repeated calls from the same source. If tau is zero, the independent VIS-DVPC-054 expression is recovered exactly. A mixed source panel can avoid the shared factor even when the source-specific marginal variances are identical; nominal annotation count alone will miss this.

The corrected cell variance feeds the same DM+IS/annotated-DR objective, exact DP, and MM surrogate. The factual IS residual remains the observed logged reward and exact behavior propensity. Annotations cannot replace it or establish action support.

## Frozen worlds and acquisition plans

The benchmark extends the same two-context/three-action/three-source finite worlds with three covariance patterns:

- `within_family_panel`: two sources share one latent model-family shock; a third source is independent.
- `shared_rubric_multi_family`: two shared latent factors combine broad rubric overlap and narrower family overlap.
- `high_common_mode`: all three sources inherit a common shock with different loadings.

Each scenario compares six plans:

1. `independent_profile_exact`: exact DP chooses under tau = 0, then is evaluated under the hidden correlated world.
2. `correlation_aware_greedy`: greedy marginal gain uses the true covariance profile.
3. `correlation_aware_exact`: exact DP uses the true covariance profile.
4. `robust_upper_bound_exact`: exact DP uses a declared upper covariance profile (1.5× factor standard deviations), then is evaluated under both the true profile and its assumed upper profile.
5. `mm_true_from_zero`: covariance-aware MM from no annotations.
6. `mm_true_from_greedy`: covariance-aware MM from max-gain greedy.

The `independent_profile_exact` plan represents a pilot that fails to observe shared error and sets correlation to zero. The robust plan minimizes risk at a conservative upper covariance profile. For the registered positive-loading factor worlds, this provides a simple uncertainty-set stress comparator. It is not a distributionally robust theorem for arbitrary covariance sets.

## Metrics

Every `CORR_RESULT` row reports: scenario and method, true and optimizer-assumed factor variances, exact correlated MSE, empirical bias/variance/MSE from an independently seeded Monte Carlo experiment, OPE-weighted reward-model variance inflation versus independence for the same allocation, exact-DP regret, source counts, worst-case MSE under the declared upper profile, and MM outer iterations.

The variance-inflation value is a design-effect diagnostic, not a literal annotator vote count and not a confidence interval. Interval coverage is deliberately not claimed: a separate inferential method and coverage calibration are required.

## Acceptance gates

- At tau = 0, cell reward-model variance reduces exactly to the independent VIS-DVPC-054 formula.
- The shared covariance matrix is PSD by construction; negative factor variances, non-finite loadings, and shared variance exceeding marginal variance fail closed.
- A repeated same-family panel exhibits a positive covariance penalty; a mix with an independent source removes that pair penalty.
- Analytic correlated OPE MSE matches independent Monte Carlo within a registered tolerance for each frozen covariance pattern.
- The robust upper-profile exact plan has no worse objective under its own declared upper profile than the plugin-independent plan.
- Exact-DP remains globally optimal on the finite fixtures; covariance-aware MM keeps a non-increasing original objective and reports its gap to exact DP.
- All six plans per scenario emit deterministic rows with explicit profile and true covariance values.
- No annotation replaces the factual residual or repairs zero logger support.

## Limits and next steps

v0 has two shared latent factors, fixed source loadings, cell-local shocks, known propensities, stationary Bernoulli rewards, fixed target policy and independent pilot/acquisition/evaluation streams. It does not model drifting loadings, shared shocks across different context-action cells, non-Gaussian dependence, source-selection bias in pilot measurement, or arbitrary covariance uncertainty sets.

Next: estimate factor structure from an independent overlap pilot; build uncertainty sets for factor variances and loadings; optimize worst-case regret; add multiway clustering across cells and time; and move from fixed batch allocation to sequential value-of-information acquisition. The paper's 0.05% mean / 0.26% maximum MM optimality gaps remain a matched-fixture stretch target, not a claim for these new synthetic worlds.