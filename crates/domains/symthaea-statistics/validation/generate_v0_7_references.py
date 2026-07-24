#!/usr/bin/env python3
"""Generate independent numerical references for the v0.7 feature layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
import scipy.stats as st
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

out: dict[str, object] = {
    "versions": {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }
}

# Negative-binomial NB2 distribution.
mean = 7.5
dispersion = 0.35
shape = 1.0 / dispersion
probability = shape / (shape + mean)
count = 12
out["negative_binomial"] = {
    "log_pmf": float(st.nbinom.logpmf(count, shape, probability)),
    "pmf": float(st.nbinom.pmf(count, shape, probability)),
    "cdf": float(st.nbinom.cdf(count, shape, probability)),
    "sf": float(st.nbinom.sf(count, shape, probability)),
}

# Fixed-dispersion NB2 regression.
x = np.arange(16, dtype=float)[:, None]
y = np.array([1, 0, 3, 1, 5, 2, 7, 3, 10, 4, 14, 6, 19, 8, 26, 11], dtype=float)
design = sm.add_constant(x)
nb_fit = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=0.8)).fit()
out["negative_binomial_regression"] = {
    "coefficients": nb_fit.params.tolist(),
    "standard_errors": nb_fit.bse.tolist(),
    "log_likelihood": float(nb_fit.llf),
    "deviance": float(nb_fit.deviance),
    "null_deviance": float(nb_fit.null_deviance),
}

# Elastic net on the same standardized objective used by the Rust implementation.
x = np.array([[index - 20.0, (index * 17) % 11 - 5.0] for index in range(40)])
y = np.array([2.0 + 3.0 * row[0] for row in x])
means = x.mean(axis=0)
scales = np.sqrt(np.mean((x - means) ** 2, axis=0))
standardized = (x - means) / scales
elastic = ElasticNet(
    alpha=0.2,
    l1_ratio=1.0,
    fit_intercept=True,
    max_iter=100_000,
    tol=1.0e-12,
    selection="cyclic",
).fit(standardized, y)
raw_coefficients = elastic.coef_ / scales
raw_intercept = float(elastic.intercept_ - raw_coefficients @ means)
out["elastic_net"] = {
    "intercept": raw_intercept,
    "coefficients": raw_coefficients.tolist(),
    "iterations": int(elastic.n_iter_),
}

# Pairwise robust estimators.
rank_x = np.arange(6, dtype=float)
rank_y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 500.0])
theil = st.theilslopes(rank_y, rank_x, alpha=0.8)
kendall = st.kendalltau([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 4.0])
out["rank_robust"] = {
    "theil_sen_slope": float(theil.slope),
    "theil_sen_intercept": float(theil.intercept),
    "kendall_tau_b": float(kendall.statistic),
}

# Paired categorical tests.
cochran_data = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
    ]
)
cochran = cochrans_q(cochran_data)
out["paired_categorical"] = {
    "mcnemar_exact_p": float(mcnemar([[0, 1], [9, 0]], exact=True).pvalue),
    "cochran_q": float(cochran.statistic),
    "cochran_p": float(cochran.pvalue),
}

# Closed-form design-based references.
values = np.array([2.0, 4.0, 6.0])
probabilities = np.array([0.5, 0.5, 0.5])
weights = 1.0 / probabilities
hajek = float(np.sum(weights * values) / np.sum(weights))
out["survey"] = {
    "horvitz_thompson_total": float(np.sum(values / probabilities)),
    "hajek_mean": hajek,
    "kish_effective_sample_size": float(np.sum(weights) ** 2 / np.sum(weights**2)),
    "stratified_mean": 8.0,
}

# Aalen-Johansen and Nelson-Aalen hand-checked recurrence values.
out["survival_extensions"] = {
    "nelson_aalen_at_4": 1.0 / 4.0 + 1.0 / 2.0 + 1.0,
    "target_cumulative_incidence_at_3": 0.5,
}

path = Path(__file__).with_name("v0_7_reference_results.json")
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
