
"""Matched intersection-crossing benchmark for covariance-preconditioned betting.

Optional dependency: numpy. Includes Gaussian and Rademacher-factor sub-Gaussian
noise, static and predictable time-varying correlations, and a fixed-prior
mixture over learning rates. This is not a closed-FDR or SupFDR benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any

import numpy as np


def covariance(dimension: int, rho: float) -> np.ndarray:
    if dimension < 2 or not (-1.0 / (dimension - 1) <= rho <= 1.0):
        raise ValueError("invalid equicorrelation")
    return (1.0 - rho) * np.eye(dimension) + rho * np.ones((dimension, dimension))


def covariance_envelope(dimension: int) -> np.ndarray:
    """One global projective kernel for every finite universe dimension.

    V_d = (1-rho_min) I_d + rho_max 11^T, with [rho_min,rho_max]=[.2,.8].
    For any rho in the interval, V_d-Sigma_d(rho) =
    (rho-rho_min) I_d + (rho_max-rho) 11^T >= 0. Principal submatrices
    retain the exact same kernel as the universe is extended.
    """
    rho_min, rho_max = 0.2, 0.8
    proxy = (1.0 - rho_min) * np.eye(dimension) + rho_max * np.ones(
        (dimension, dimension)
    )
    for rho in np.linspace(rho_min, rho_max, 101):
        if np.linalg.eigvalsh(proxy - covariance(dimension, float(rho))).min() < -1e-9:
            raise ValueError("projective covariance envelope failed its declared grid check")
    return proxy


def _wilson(successes: int, trials: int) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = successes / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _cell_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _rho_at(mode: float | str, t: int, horizon: int) -> float:
    if isinstance(mode, (int, float)):
        return float(mode)
    if mode == "switch_up":
        return 0.2 if t < horizon // 2 else 0.8
    if mode == "switch_down":
        return 0.8 if t < horizon // 2 else 0.2
    if mode == "alternate":
        return 0.2 if (t // 5) % 2 == 0 else 0.8
    raise ValueError(f"unknown rho mode: {mode}")


def _draw(
    rng: np.random.Generator,
    mean: np.ndarray,
    rho: float,
    noise_model: str,
    streams: int,
) -> np.ndarray:
    if noise_model == "gaussian":
        return rng.multivariate_normal(mean, covariance(len(mean), rho), size=streams)
    if noise_model == "rademacher_factor":
        # Z_i - mean_i = sqrt(rho)*G_common + sqrt(1-rho)*G_i,
        # with independent Rademacher signs. Its covariance is Sigma(rho),
        # and log cosh(x) <= x^2/2 gives the same conditional mgf proxy.
        common = rng.choice(np.asarray([-1.0, 1.0]), size=(streams, 1))
        idiosyncratic = rng.choice(
            np.asarray([-1.0, 1.0]), size=(streams, len(mean))
        )
        return (
            mean[None, :]
            + math.sqrt(rho) * common
            + math.sqrt(1.0 - rho) * idiosyncratic
        )
    raise ValueError("noise_model must be gaussian or rademacher_factor")


def simulate_cell(
    *,
    master_seed: int,
    streams: int,
    dimension: int,
    horizon: int,
    noise_model: str,
    rho_mode: float | str,
    scenario: str,
    eta_values: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
    cap: float = 1.5,
    threshold: float = 1.5,
    warmup: int = 5,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    if streams < 1 or dimension < 2 or horizon <= warmup:
        raise ValueError("invalid streams, dimension or horizon")
    if not 0 < alpha < 1 or cap < 0 or threshold < 0:
        raise ValueError("invalid alpha, cap or threshold")
    if not eta_values or any(not math.isfinite(x) or x < 0 for x in eta_values):
        raise ValueError("eta_values must be a nonempty sequence of finite nonnegative values")
    if len(set(eta_values)) != len(eta_values):
        raise ValueError("eta_values must be unique")
    if 0.05 not in eta_values:
        raise ValueError("eta_values must include 0.05 for the fixed-rate comparator")

    means = {
        "all_null": np.zeros(dimension),
        "sparse_last": np.asarray([0.0] * (dimension - 1) + [0.8]),
        "dense_equal": np.full(dimension, 0.45),
        "heterogeneous": np.linspace(0.05, 0.65, dimension),
    }
    if scenario not in means:
        raise ValueError(f"unknown scenario: {scenario}")
    mean = means[scenario]
    proxy = covariance_envelope(dimension)
    proxy_diag = np.diag(proxy)
    etas = np.asarray(eta_values, dtype=np.float64)
    rng = np.random.default_rng(
        _cell_seed(
            master_seed,
            f"{noise_model}|{rho_mode}|{scenario}|{dimension}|{horizon}|{streams}",
        )
    )

    history = np.zeros((streams, dimension, horizon), dtype=np.float64)
    log_vector = np.zeros(streams)
    log_scalar = np.zeros(streams)
    log_coordinate = np.zeros((streams, dimension))
    log_eta = np.zeros((len(etas), streams))
    log_nat_single = np.zeros(streams)
    lambda_eta = np.zeros((len(etas), streams, dimension), dtype=np.float64)
    lambda_single = np.zeros((streams, dimension), dtype=np.float64)
    crossed = {
        key: np.zeros(streams, dtype=bool)
        for key in (
            "coordinate_local_vector",
            "shared_scalar",
            "coordinate_eprocess_mixture",
            "preconditioned_eta_0.05",
            "fixed_prior_eta_mixture",
            "fixed_50_50_vector_eta_mixture",
        )
    }

    for t in range(horizon):
        rho = _rho_at(rho_mode, t, horizon)
        if not 0.2 <= rho <= 0.8:
            raise ValueError("rho schedule escaped the declared envelope")
        score = _draw(rng, mean, rho, noise_model, streams)

        if t >= warmup:
            prior = history[:, :, :t]
            coordinate_mean = prior.mean(axis=2)
            vector_bet = np.clip(
                coordinate_mean - threshold / math.sqrt(t), 0.0, cap
            )
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

        log_coordinate += (
            vector_bet * score - 0.5 * vector_bet * vector_bet * proxy_diag
        )
        coordinate_max = log_coordinate.max(axis=1)
        log_coordinate_mix = coordinate_max + np.log(
            np.exp(log_coordinate - coordinate_max[:, None]).mean(axis=1)
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

        # The e-factors use the current, pre-observation lambda for every eta.
        eta_penalty = np.einsum(
            "kbi,ij,kbj->kb", lambda_eta, proxy, lambda_eta, optimize=True
        )
        log_eta += (
            np.sum(lambda_eta * score[None, :, :], axis=2)
            - 0.5 * eta_penalty
        )
        eta_gradient = score[None, :, :] - np.einsum(
            "kbi,ij->kbj", lambda_eta, proxy, optimize=True
        )
        lambda_eta = np.clip(
            lambda_eta
            + etas[:, None, None]
            * eta_gradient
            / proxy_diag[None, None, :],
            0.0,
            cap,
        )

        # A separately recorded eta=.05 expert, if present.
        if 0.05 in eta_values:
            j = eta_values.index(0.05)
            log_nat_single = log_eta[j].copy()

        eta_max = log_eta.max(axis=0)
        log_eta_mixture = eta_max + np.log(
            np.exp(log_eta - eta_max[None, :]).mean(axis=0)
        )
        log_vector_eta_portfolio = (
            np.logaddexp(log_vector, log_eta_mixture) - math.log(2.0)
        )

        crossed["coordinate_local_vector"] |= (
            log_vector >= -math.log(alpha)
        )
        crossed["shared_scalar"] |= log_scalar >= -math.log(alpha)
        crossed["coordinate_eprocess_mixture"] |= (
            log_coordinate_mix >= -math.log(alpha)
        )
        if 0.05 in eta_values:
            crossed["preconditioned_eta_0.05"] |= (
                log_nat_single >= -math.log(alpha)
            )
        crossed["fixed_prior_eta_mixture"] |= (
            log_eta_mixture >= -math.log(alpha)
        )
        crossed["fixed_50_50_vector_eta_mixture"] |= (
            log_vector_eta_portfolio >= -math.log(alpha)
        )
        history[:, :, t] = score

    output: list[dict[str, Any]] = []
    for method, did_cross in crossed.items():
        successes = int(did_cross.sum())
        output.append(
            {
                "noise_model": noise_model,
                "rho_mode": str(rho_mode),
                "scenario": scenario,
                "method": method,
                "crossings": successes,
                "streams": streams,
                "rate": successes / streams,
                "wilson_95": _wilson(successes, streams),
                "alpha": alpha,
                "seed": _cell_seed(
                    master_seed,
                    f"{noise_model}|{rho_mode}|{scenario}|{dimension}|{horizon}|{streams}",
                ),
                "horizon": horizon,
                "guarantee_scope": "single_intersection_crossing_diagnostic_only",
            }
        )
    return output


def run_benchmark(
    *,
    master_seed: int = 9756,
    streams: int = 20_000,
    dimension: int = 5,
    horizon: int = 60,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rho_modes: tuple[float | str, ...] = (0.2, 0.5, 0.8, "switch_up")
    noise_models = ("gaussian", "rademacher_factor")
    scenarios = ("all_null", "sparse_last", "dense_equal", "heterogeneous")
    for noise_model in noise_models:
        for rho_mode in rho_modes:
            for scenario in scenarios:
                rows.extend(
                    simulate_cell(
                        master_seed=master_seed,
                        streams=streams,
                        dimension=dimension,
                        horizon=horizon,
                        noise_model=noise_model,
                        rho_mode=rho_mode,
                        scenario=scenario,
                    )
                )
    return {
        "benchmark": "covariance_preconditioned_projective_betting",
        "version": 1,
        "master_seed": master_seed,
        "streams_per_cell": streams,
        "dimension": dimension,
        "horizon": horizon,
        "alpha": 0.05,
        "cap": 1.5,
        "warmup": 5,
        "bet_learning_rates": [0.025, 0.05, 0.10, 0.20],
        "prior_over_learning_rates": "uniform_fixed_ex_ante",
        "covariance_envelope": "Sigma(0.5) + 1.2 I",
        "declared_equicorrelation_interval": [0.2, 0.8],
        "noise_models": list(noise_models),
        "scenarios": list(scenarios),
        "rho_modes": [0.2, 0.5, 0.8, "switch_up"],
        "result_count": len(rows),
        "results": rows,
        "interpretation": (
            "Intersection-level threshold crossings only, not closed-FDR, "
            "stopped-FDR, FWER or SupFDR. The vector, scalar and preconditioned "
            "strategies are valid only under the declared conditional mgf "
            "envelope and predictable bet updates. Fixed-prior mixtures are "
            "used; no pointwise expert maximum is reported."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9756)
    parser.add_argument("--streams", type=int, default=20_000)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    result = run_benchmark(
        master_seed=args.seed,
        streams=args.streams,
        dimension=args.dimension,
        horizon=args.horizon,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
