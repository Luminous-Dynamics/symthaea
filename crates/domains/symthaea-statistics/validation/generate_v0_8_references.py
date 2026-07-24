#!/usr/bin/env python3
"""Generate independent v0.8 reference values with NumPy/SciPy/scikit-learn."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
from scipy.special import gammaln
import sklearn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support


def main() -> None:
    output: dict[str, object] = {
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }
    }

    point = np.array([0.2, 0.3, 0.5])
    concentration = np.array([2.0, 3.0, 5.0])
    output["dirichlet"] = {
        "log_pdf": float(stats.dirichlet.logpdf(point, concentration)),
        "mean": (concentration / concentration.sum()).tolist(),
        "covariance": stats.dirichlet.cov(concentration).tolist(),
    }
    counts = np.array([3, 2, 1])
    probabilities = np.array([0.5, 0.3, 0.2])
    output["multinomial"] = {
        "log_pmf": float(stats.multinomial.logpmf(counts, counts.sum(), probabilities)),
    }
    posterior_alpha = np.array([3.0, 3.0, 3.0])
    future = np.array([1, 2, 0])
    future_total = future.sum()
    predictive_log_mass = (
        gammaln(future_total + 1)
        - gammaln(future + 1).sum()
        + gammaln(posterior_alpha.sum())
        - gammaln(posterior_alpha.sum() + future_total)
        + (gammaln(posterior_alpha + future) - gammaln(posterior_alpha)).sum()
    )
    output["dirichlet_multinomial"] = {
        "predictive_log_mass": float(predictive_log_mass),
    }

    actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    predicted = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1, 2])
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2], zero_division=np.nan
    )
    output["multiclass"] = {
        "matrix": matrix.tolist(),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_precision": float(np.nanmean(precision)),
        "macro_recall": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "kappa": float(cohen_kappa_score(actual, predicted)),
    }

    left_ratings = [0, 0, 1, 1, 2, 2, 3, 3]
    right_ratings = [0, 1, 1, 2, 2, 3, 3, 3]
    output["weighted_kappa"] = {
        "unweighted": float(cohen_kappa_score(left_ratings, right_ratings)),
        "linear": float(cohen_kappa_score(left_ratings, right_ratings, weights="linear")),
        "quadratic": float(cohen_kappa_score(left_ratings, right_ratings, weights="quadratic")),
    }

    left_measurements = np.array([10.1, 11.7, 9.8, 13.2, 12.5])
    right_measurements = np.array([9.9, 11.4, 10.0, 12.8, 12.2])
    differences = left_measurements - right_measurements
    sd = differences.std(ddof=1)
    bias = differences.mean()
    z = stats.norm.ppf(0.975)
    output["bland_altman"] = {
        "bias": float(bias),
        "standard_deviation": float(sd),
        "lower": float(bias - z * sd),
        "upper": float(bias + z * sd),
        "standard_error_bias": float(sd / np.sqrt(differences.size)),
    }

    concordance_left = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    concordance_right = np.array([1.1, 1.8, 3.2, 3.9, 5.5])
    covariance = np.mean(
        (concordance_left - concordance_left.mean())
        * (concordance_right - concordance_right.mean())
    )
    ccc = 2.0 * covariance / (
        concordance_left.var()
        + concordance_right.var()
        + (concordance_left.mean() - concordance_right.mean()) ** 2
    )
    output["concordance"] = {"coefficient": float(ccc)}

    reliability = np.array(
        [[1, 2, 1], [2, 3, 2], [3, 4, 4], [4, 5, 4], [5, 6, 5]], dtype=float
    )
    items = reliability.shape[1]
    alpha = items / (items - 1) * (
        1.0
        - reliability.var(axis=0, ddof=1).sum()
        / reliability.sum(axis=1).var(ddof=1)
    )
    correlation = np.corrcoef(reliability, rowvar=False)
    average_correlation = (correlation.sum() - items) / (items * (items - 1))
    standardized_alpha = items * average_correlation / (
        1.0 + (items - 1) * average_correlation
    )
    output["cronbach"] = {
        "raw": float(alpha),
        "standardized": float(standardized_alpha),
    }

    ratings = np.array(
        [[9, 8, 9], [6, 7, 6], [8, 8, 7], [5, 6, 5], [7, 7, 8]], dtype=float
    )
    n, k = ratings.shape
    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    ss_targets = k * ((row_means - grand) ** 2).sum()
    ss_raters = n * ((column_means - grand) ** 2).sum()
    ss_error = ((ratings - row_means[:, None] - column_means[None, :] + grand) ** 2).sum()
    ss_within = ((ratings - row_means[:, None]) ** 2).sum()
    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_within = ss_within / (n * (k - 1))
    output["icc"] = {
        "one_way_random": float(
            (ms_targets - ms_within) / (ms_targets + (k - 1) * ms_within)
        ),
        "two_way_random_agreement": float(
            (ms_targets - ms_error)
            / (ms_targets + (k - 1) * ms_error + k * (ms_raters - ms_error) / n)
        ),
        "two_way_mixed_consistency": float(
            (ms_targets - ms_error) / (ms_targets + (k - 1) * ms_error)
        ),
        "mean_square_targets": float(ms_targets),
        "mean_square_raters": float(ms_raters),
        "mean_square_error": float(ms_error),
        "mean_square_within": float(ms_within),
    }

    angles = np.deg2rad([350.0, 5.0, 10.0, 15.0, 355.0])
    sine = np.sin(angles).sum()
    cosine = np.cos(angles).sum()
    resultant = np.hypot(sine, cosine) / angles.size
    mean_direction = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    rayleigh_statistic = angles.size * resultant**2
    resultant_absolute = angles.size * resultant
    rayleigh_p = np.exp(
        np.sqrt(
            1.0
            + 4.0 * angles.size
            + 4.0 * (angles.size**2 - resultant_absolute**2)
        )
        - (1.0 + 2.0 * angles.size)
    )
    z_rayleigh = rayleigh_statistic
    rayleigh_p *= (
        1.0
        + (2.0 * z_rayleigh - z_rayleigh**2) / (4.0 * angles.size)
        - (
            24.0 * z_rayleigh
            - 132.0 * z_rayleigh**2
            + 76.0 * z_rayleigh**3
            - 9.0 * z_rayleigh**4
        )
        / (288.0 * angles.size**2)
    )
    output["circular"] = {
        "mean_direction": float(mean_direction),
        "mean_resultant_length": float(resultant),
        "variance": float(1.0 - resultant),
        "standard_deviation": float(np.sqrt(-2.0 * np.log(resultant))),
        "rayleigh_statistic": float(rayleigh_statistic),
        "rayleigh_p_value": float(rayleigh_p),
    }

    observations = np.array([1.0, 1.4, 0.9, 1.8])
    process_variance = 0.2
    observation_variance = 0.5
    predicted_mean = 0.0
    predicted_variance = 2.0
    steps: list[dict[str, float]] = []
    log_likelihood = 0.0
    for observation in observations:
        innovation = observation - predicted_mean
        innovation_variance = predicted_variance + observation_variance
        gain = predicted_variance / innovation_variance
        filtered_mean = predicted_mean + gain * innovation
        filtered_variance = (1.0 - gain) * predicted_variance
        log_likelihood += -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(innovation_variance)
            + innovation**2 / innovation_variance
        )
        steps.append(
            {
                "predicted_mean": float(predicted_mean),
                "predicted_variance": float(predicted_variance),
                "innovation": float(innovation),
                "innovation_variance": float(innovation_variance),
                "gain": float(gain),
                "filtered_mean": float(filtered_mean),
                "filtered_variance": float(filtered_variance),
            }
        )
        predicted_mean = filtered_mean
        predicted_variance = filtered_variance + process_variance
    smoothed_means = np.array([step["filtered_mean"] for step in steps])
    smoothed_variances = np.array([step["filtered_variance"] for step in steps])
    for index in range(len(steps) - 2, -1, -1):
        smoother_gain = steps[index]["filtered_variance"] / steps[index + 1]["predicted_variance"]
        smoothed_means[index] = steps[index]["filtered_mean"] + smoother_gain * (
            smoothed_means[index + 1] - steps[index + 1]["predicted_mean"]
        )
        smoothed_variances[index] = steps[index]["filtered_variance"] + smoother_gain**2 * (
            smoothed_variances[index + 1] - steps[index + 1]["predicted_variance"]
        )
    output["local_level"] = {
        "log_likelihood": float(log_likelihood),
        "steps": steps,
        "smoothed_means": smoothed_means.tolist(),
        "smoothed_variances": smoothed_variances.tolist(),
    }

    output["generalized_pareto"] = {}
    for shape in [0.2, 0.0, -0.2]:
        key = f"shape_{shape:+.1f}"
        output["generalized_pareto"][key] = {
            "log_pdf": float(stats.genpareto.logpdf(1.5, c=shape, scale=2.0)),
            "cdf": float(stats.genpareto.cdf(1.5, c=shape, scale=2.0)),
            "sf": float(stats.genpareto.sf(1.5, c=shape, scale=2.0)),
            "quantile_0_9": float(stats.genpareto.ppf(0.9, c=shape, scale=2.0)),
        }
    hill_values = np.array([1.0, 1.3, 1.7, 2.1, 3.0, 4.5, 7.0, 11.0])
    order_statistics = 3
    sorted_hill = np.sort(hill_values)
    threshold = sorted_hill[-order_statistics - 1]
    hill = np.log(sorted_hill[-order_statistics:] / threshold).mean()
    output["hill"] = {
        "threshold": float(threshold),
        "tail_index": float(hill),
        "pareto_exponent": float(1.0 / hill),
    }

    sample_left = np.array([[1, 2], [2, 1], [4, 5], [5, 4], [3, 3]], dtype=float)
    mean_null = np.array([0.0, 0.0])
    n_left, dimensions = sample_left.shape
    difference = sample_left.mean(axis=0) - mean_null
    covariance_left = np.cov(sample_left, rowvar=False, ddof=1)
    t_squared = n_left * difference @ np.linalg.solve(covariance_left, difference)
    f_statistic = (n_left - dimensions) * t_squared / (
        dimensions * (n_left - 1)
    )
    output["hotelling_one_sample"] = {
        "t_squared": float(t_squared),
        "f_statistic": float(f_statistic),
        "p_value": float(stats.f.sf(f_statistic, dimensions, n_left - dimensions)),
    }
    sample_right = np.array([[0, 1], [1, 0], [2, 2], [1, 2], [2, 1]], dtype=float)
    n_right = sample_right.shape[0]
    covariance_right = np.cov(sample_right, rowvar=False, ddof=1)
    pooled = (
        (n_left - 1) * covariance_left + (n_right - 1) * covariance_right
    ) / (n_left + n_right - 2)
    difference = sample_left.mean(axis=0) - sample_right.mean(axis=0)
    t_squared = (
        n_left
        * n_right
        / (n_left + n_right)
        * difference
        @ np.linalg.solve(pooled, difference)
    )
    denominator_df = n_left + n_right - dimensions - 1
    f_statistic = denominator_df * t_squared / (
        dimensions * (n_left + n_right - 2)
    )
    output["hotelling_two_sample"] = {
        "t_squared": float(t_squared),
        "f_statistic": float(f_statistic),
        "p_value": float(stats.f.sf(f_statistic, dimensions, denominator_df)),
    }

    destination = Path(__file__).with_name("v0_8_reference_results.json")
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
