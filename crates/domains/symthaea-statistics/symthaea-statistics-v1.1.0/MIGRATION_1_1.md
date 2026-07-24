# Migration to 1.1

Version 1.1 is additive. It does not remove or rename the v1.0 stable prelude,
`API_LEVEL` remains `1`, and `NUMERICAL_CONTRACT` remains `validated-v1`.

## Nonlinear and multivariate comparison

Use `try_energy_distance` when any distributional difference is scientifically
relevant, and `try_distance_dependence` when nonlinear dependence may be missed
by Pearson or Spearman correlation. Seeded permutation variants use the
finite-Monte-Carlo add-one correction and retain their repetition counts.

`try_maximum_mean_discrepancy` exposes the Gaussian-kernel bandwidth and the
biased/unbiased estimator convention. `try_median_mmd` is a convenience path,
not a claim that the median heuristic is optimal for every domain.

`try_permanova` is one-factor Euclidean PERMANOVA. Labels are exchangeable only
under a scientifically valid permutation design; blocked, paired, or repeated
observations require a domain-specific permutation scheme.

## Rank and evidence synthesis

`try_kruskal_wallis` returns a tie-corrected omnibus statistic plus all Dunn
pairwise comparisons with Holm-adjusted p-values. The pairwise comparisons are
interpreted after the omnibus design is judged appropriate.

`try_meta_regression` always includes an intercept and expects moderator rows
without one. Random effects use a generalized DerSimonian-Laird moment estimate.
Prediction intervals describe heterogeneity in true effects and do not include
a future study's sampling error.

## Streaming and covariance

`P2Quantile` uses five adaptive markers and bounded memory. It is deterministic
but not mergeable. Exact retained quantiles and mergeable histogram summaries
remain preferable when their memory or resolution tradeoffs are acceptable.

`try_oas_covariance` uses the maximum-likelihood `1/n` empirical covariance and
shrinks toward a spherical target. It is intended for stable covariance and
Mahalanobis calculations, not as evidence that the true covariance is spherical.

## Bayesian and predictive additions

`NormalInverseGammaModel` jointly updates an unknown Gaussian mean and variance.
Its mean marginal and posterior predictive distributions are Student-t under the
stated conjugate model.

Multiclass scoring validates complete probability vectors. Top-label ECE is a
bin-dependent descriptive diagnostic; report the bin definition and proper
scores alongside it.

## Multiplicity

v1.1 adds Sidak, Hochberg, Benjamini-Yekutieli, and explicit-lambda Storey
q-values. Their dependence assumptions differ; select the correction from the
study design rather than from whichever produces the smallest adjusted value.
