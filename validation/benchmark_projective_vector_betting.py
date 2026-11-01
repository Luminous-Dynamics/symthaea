"""Reproducible single-intersection diagnostic for projective vector betting.

Optional dependency: numpy. This is NOT a closed-FDR or SupFDR benchmark.
Each scenario measures whether a specified intersection e-process crosses 1/alpha.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np


def covariance(d: int, rho: float) -> np.ndarray:
    if d < 2 or not (-1.0 / (d - 1) <= rho <= 1.0):
        raise ValueError("invalid dimension or equicorrelation")
    return (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))


def covariance_envelope(
    d: int, rho_min: float = 0.2, rho_max: float = 0.8, rho_reference: float = 0.5
) -> np.ndarray:
    if not rho_min <= rho_reference <= rho_max:
        raise ValueError("reference rho must lie inside declared interval")
    delta = max(
        0.0,
        (d - 1) * (rho_max - rho_reference),
        rho_reference - rho_min,
    )
    result = covariance(d, rho_reference) + delta * np.eye(d)
    for rho in np.linspace(rho_min, rho_max, 101):
        if np.linalg.eigvalsh(result - covariance(d, float(rho))).min() < -1e-9:
            raise ValueError("constructed envelope does not dominate declared grid")
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _crossing_row(
    *,
    rho: float,
    scenario: str,
    method: str,
    crossed: np.ndarray,
    alpha: float,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    successes, trials = int(crossed.sum()), int(crossed.size)
    low, high = wilson_interval(successes, trials)
    return {
        "rho": float(rho),
        "scenario": scenario,
        "method": method,
        "crossings": successes,
        "streams": trials,
        "rate": successes / trials,
        "wilson_95": [low, high],
        "alpha": alpha,
        "seed": seed,
        "horizon": horizon,
        "guarantee_scope": "single_intersection_anytime_crossing_diagnostic_only",
    }


def run_fixed_intersection(
    *,
    seed: int = 9751,
    streams: int = 20_000,
    dimension: int = 5,
    horizon: int = 60,
    alpha: float = 0.05,
    warmup: int = 5,
    threshold: float = 1.5,
    cap: float = 1.5,
    rho_values: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> list[dict[str, Any]]:
    if streams < 1 or dimension < 2 or horizon < warmup + 1:
        raise ValueError("invalid streams, dimension or horizon")
    if not 0.0 < alpha < 1.0 or threshold < 0 or cap < 0:
        raise ValueError("invalid alpha, threshold or cap")

    rng = np.random.default_rng(seed)
    proxy = covariance_envelope(dimension)
    proxy_diagonal = np.diag(proxy)
    results: list[dict[str, Any]] = []
    scenarios = (
        ("all_null", np.zeros(dimension)),
        ("sparse_last", np.asarray([0.0] * (dimension - 1) + [0.8])),
        ("dense_equal", np.full(dimension, 0.45)),
    )

    for rho in rho_values:
        true_covariance = covariance(dimension, rho)
        if np.linalg.eigvalsh(proxy - true_covariance).min() < -1e-9:
            raise ValueError("true covariance lies outside the declared envelope")

        for scenario, mean in scenarios:
            history = np.zeros((streams, dimension, horizon), dtype=np.float64)
            log_vector = np.zeros(streams)
            log_scalar = np.zeros(streams)
            log_coordinate = np.zeros((streams, dimension))
            crossing = {
                name: np.zeros(streams, dtype=bool)
                for name in ("vector", "scalar", "fixed_50_50", "coordinate_mixture")
            }

            for t in range(horizon):
                score = rng.multivariate_normal(mean, true_covariance, size=streams)
                if t >= warmup:
                    prior = history[:, :, :t]
                    coordinate_mean = prior.mean(axis=2)
                    # Each coordinate bet uses only that coordinate's pre-update history.
                    vector_bet = np.clip(
                        coordinate_mean - threshold / math.sqrt(t), 0.0, cap
                    )
                    # One scalar direction uses the previous per-round pooled mean.
                    pooled_mean = prior.mean(axis=(1, 2))
                    scalar_bet = np.clip(
                        pooled_mean - threshold / math.sqrt(t), 0.0, cap
                    )
                else:
                    vector_bet = np.zeros((streams, dimension))
                    scalar_bet = np.zeros(streams)

                vector_penalty = np.einsum(
                    "bi,ij,bj->b", vector_bet, proxy, vector_bet, optimize=True
                )
                log_vector += np.sum(vector_bet * score, axis=1) - 0.5 * vector_penalty

                # Coordinate e-processes form a fixed 1/d mixture, valid under
                # arbitrary cross-coordinate dependence if each marginal factor is valid.
                log_coordinate += (
                    vector_bet * score - 0.5 * vector_bet * vector_bet * proxy_diagonal
                )

                scalar_vector = np.broadcast_to(
                    scalar_bet[:, None], (streams, dimension)
                )
                scalar_penalty = np.einsum(
                    "bi,ij,bj->b", scalar_vector, proxy, scalar_vector, optimize=True
                )
                log_scalar += (
                    scalar_bet * score.sum(axis=1) - 0.5 * scalar_penalty
                )

                coordinate_max = log_coordinate.max(axis=1)
                log_coordinate_mixture = (
                    coordinate_max
                    + np.log(
                        np.exp(log_coordinate - coordinate_max[:, None]).mean(axis=1)
                    )
                )
                log_portfolio = (
                    np.logaddexp(log_vector, log_scalar) - math.log(2.0)
                )

                crossing["vector"] |= log_vector >= -math.log(alpha)
                crossing["scalar"] |= log_scalar >= -math.log(alpha)
                crossing["fixed_50_50"] |= log_portfolio >= -math.log(alpha)
                crossing["coordinate_mixture"] |= (
                    log_coordinate_mixture >= -math.log(alpha)
                )
                history[:, :, t] = score

            for method, did_cross in crossing.items():
                results.append(
                    _crossing_row(
                        rho=rho,
                        scenario=scenario,
                        method=method,
                        crossed=did_cross,
                        alpha=alpha,
                        seed=seed,
                        horizon=horizon,
                    )
                )
    return results


def run_predictable_arrival_null(
    *,
    seed: int = 9752,
    streams: int = 20_000,
    dimension: int = 5,
    horizon: int = 60,
    alpha: float = 0.05,
    warmup: int = 5,
    threshold: float = 1.5,
    cap: float = 1.5,
    rho_values: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> list[dict[str, Any]]:
    """Nested arrivals: next coordinate activates if previous active mean > 0.

    The activation decision uses only observations through t-1. This measures
    one all-null intersection process with predictable random active sets.
    """
    rng = np.random.default_rng(seed)
    proxy = covariance_envelope(dimension)
    results: list[dict[str, Any]] = []
    coordinate_ids = np.arange(dimension)[None, :]

    for rho in rho_values:
        true_covariance = covariance(dimension, rho)
        active_count = np.ones(streams, dtype=np.int64)
        observations_sum = np.zeros((streams, dimension))
        observations_count = np.zeros((streams, dimension), dtype=np.int64)
        log_capital = np.zeros(streams)

        for _t in range(horizon):
            last = active_count - 1
            last_count = observations_count[np.arange(streams), last]
            last_sum = observations_sum[np.arange(streams), last]
            last_mean = np.divide(
                last_sum,
                np.maximum(last_count, 1),
            )
            should_activate = (
                (active_count < dimension)
                & (last_count >= warmup)
                & (last_mean > 0.0)
            )
            active_count += should_activate.astype(np.int64)
            active = coordinate_ids < active_count[:, None]

            coordinate_mean = np.divide(
                observations_sum,
                np.maximum(observations_count, 1),
            )
            bet = np.clip(
                coordinate_mean
                - threshold
                / np.sqrt(np.maximum(observations_count, 1)),
                0.0,
                cap,
            )
            bet = np.where((observations_count >= warmup) & active, bet, 0.0)
            score = rng.multivariate_normal(
                np.zeros(dimension), true_covariance, size=streams
            )

            penalty = np.einsum("bi,ij,bj->b", bet, proxy, bet, optimize=True)
            log_capital += np.sum(bet * score, axis=1) - 0.5 * penalty
            observations_sum += np.where(active, score, 0.0)
            observations_count += active.astype(np.int64)

        crossing = log_capital >= -math.log(alpha)
        results.append(
            _crossing_row(
                rho=rho,
                scenario="all_null_predictable_nested_arrivals",
                method="vector",
                crossed=crossing,
                alpha=alpha,
                seed=seed,
                horizon=horizon,
            )
        )
    return results


def run_benchmark(
    seed: int = 9751,
    streams: int = 20_000,
    dimension: int = 5,
    horizon: int = 60,
) -> dict[str, Any]:
    fixed = run_fixed_intersection(
        seed=seed, streams=streams, dimension=dimension, horizon=horizon
    )
    arrivals = run_predictable_arrival_null(
        seed=seed + 1, streams=streams, dimension=dimension, horizon=horizon
    )
    return {
        "benchmark": "projective_vector_betting_intersection_diagnostic",
        "version": 1,
        "seed": seed,
        "streams_per_cell": streams,
        "dimension": dimension,
        "horizon": horizon,
        "alpha": 0.05,
        "warmup": 5,
        "threshold": 1.5,
        "bet_cap": 1.5,
        "covariance_family": "equicorrelation",
        "declared_rho_interval": [0.2, 0.8],
        "covariance_envelope": "Sigma(0.5) + 1.2 I",
        "methods": [
            "vector_coordinate_bet",
            "shared_scalar_bet",
            "fixed_50_50_vector_scalar_mixture",
            "fixed_equal_weight_coordinate_eprocess_mixture",
        ],
        "results": fixed + arrivals,
        "interpretation": (
            "Intersection e-process crossing diagnostics only. "
            "Not closed-FDR, stopped-FDR, FWER or SupFDR results. "
            "Guarantees require the stated conditional sub-Gaussian mgf envelope "
            "and predictable betting/arrival rules."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9751)
    parser.add_argument("--streams", type=int, default=20_000)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    payload = run_benchmark(
        seed=args.seed,
        streams=args.streams,
        dimension=args.dimension,
        horizon=args.horizon,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
