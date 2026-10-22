#!/usr/bin/env python3
"""Independent Metropolis-Hastings theorem oracle for LQCD sampler work.

Standard library only. This is deliberately a compact U(1)-like toy target,
not an SU(3) production sampler. It freezes the acceptance, detailed-balance,
periodic-angle, and deterministic-replay semantics that later lattice samplers
must satisfy in their own domain.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

TAU = 2.0 * math.pi
MASK64 = (1 << 64) - 1


def wrap_angle(theta: float) -> float:
    wrapped = (theta + math.pi) % TAU - math.pi
    return -math.pi if wrapped == math.pi else wrapped


def angular_difference(a: float, b: float) -> float:
    return wrap_angle(a - b)


def log_target(theta: float, beta: float) -> float:
    if not (math.isfinite(theta) and math.isfinite(beta) and beta >= 0.0):
        raise ValueError("finite theta and beta >= 0 required")
    return beta * math.cos(theta)


def metropolis_acceptance(theta: float, proposal: float, beta: float) -> float:
    log_ratio = log_target(proposal, beta) - log_target(theta, beta)
    return 1.0 if log_ratio >= 0.0 else math.exp(log_ratio)


def symmetric_window_density(theta: float, proposal: float, half_width: float) -> float:
    if not math.isfinite(half_width) or not (0.0 < half_width <= math.pi):
        raise ValueError("half_width must be in (0, pi]")
    delta = abs(angular_difference(proposal, theta))
    return 1.0 / (2.0 * half_width) if delta <= half_width + 1.0e-15 else 0.0


def unnormalized_target(theta: float, beta: float) -> float:
    return math.exp(log_target(theta, beta))


def detailed_balance_residual(theta: float, proposal: float, beta: float, half_width: float) -> float:
    q_forward = symmetric_window_density(theta, proposal, half_width)
    q_reverse = symmetric_window_density(proposal, theta, half_width)
    lhs = unnormalized_target(theta, beta) * q_forward * metropolis_acceptance(theta, proposal, beta)
    rhs = unnormalized_target(proposal, beta) * q_reverse * metropolis_acceptance(proposal, theta, beta)
    return lhs - rhs


@dataclass
class Lcg64:
    state: int

    def next_u64(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & MASK64
        return self.state

    def uniform01(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


def run_chain(*, seed: int, beta: float, half_width: float, steps: int, initial_theta: float = 0.0):
    if steps < 0:
        raise ValueError("steps must be non-negative")
    rng = Lcg64(seed & MASK64)
    theta = wrap_angle(initial_theta)
    trajectory = [theta]
    accepted = 0
    for _ in range(steps):
        delta = (2.0 * rng.uniform01() - 1.0) * half_width
        proposal = wrap_angle(theta + delta)
        alpha = metropolis_acceptance(theta, proposal, beta)
        if rng.uniform01() < alpha:
            theta = proposal
            accepted += 1
        trajectory.append(theta)
    return trajectory, accepted


def discrete_ring_transition(beta: float, sites: int):
    if sites < 3:
        raise ValueError("sites >= 3 required")
    angles = [-math.pi + TAU * i / sites for i in range(sites)]
    weights = [unnormalized_target(theta, beta) for theta in angles]
    z = sum(weights)
    stationary = [w / z for w in weights]
    transition = [[0.0 for _ in range(sites)] for _ in range(sites)]
    for i, theta in enumerate(angles):
        for j in ((i - 1) % sites, (i + 1) % sites):
            proposal = angles[j]
            alpha = metropolis_acceptance(theta, proposal, beta)
            transition[i][j] += 0.5 * alpha
            transition[i][i] += 0.5 * (1.0 - alpha)
    return stationary, transition


def stationarity_residual(stationary, transition):
    n = len(stationary)
    projected = [sum(stationary[i] * transition[i][j] for i in range(n)) for j in range(n)]
    return max(abs(projected[j] - stationary[j]) for j in range(n))


def max_row_sum_residual(transition):
    return max(abs(sum(row) - 1.0) for row in transition)


def self_test():
    beta = 1.7
    half_width = 0.8
    pairs = [
        (0.1, 0.6),
        (2.9, -2.95),
        (-1.2, -0.7),
        (0.0, 0.79),
    ]
    residuals = [abs(detailed_balance_residual(a, b, beta, half_width)) for a, b in pairs]
    max_db = max(residuals)
    assert max_db < 1.0e-12, max_db

    q1 = symmetric_window_density(2.9, -2.95, half_width)
    q2 = symmetric_window_density(-2.95, 2.9, half_width)
    assert q1 == q2 and q1 > 0.0

    trajectory_a, accepted_a = run_chain(seed=0xC0FFEE, beta=beta, half_width=half_width, steps=64)
    trajectory_b, accepted_b = run_chain(seed=0xC0FFEE, beta=beta, half_width=half_width, steps=64)
    assert trajectory_a == trajectory_b
    assert accepted_a == accepted_b
    assert 0 < accepted_a < 64

    stationary, transition = discrete_ring_transition(beta=beta, sites=16)
    row_residual = max_row_sum_residual(transition)
    invariant_residual = stationarity_residual(stationary, transition)
    assert row_residual < 1.0e-15, row_residual
    assert invariant_residual < 1.0e-15, invariant_residual

    downhill = metropolis_acceptance(0.0, 1.0, beta)
    expected = math.exp(beta * (math.cos(1.0) - 1.0))
    assert abs(downhill - expected) < 1.0e-15
    assert 0.0 < downhill < 1.0
    assert metropolis_acceptance(1.0, 0.0, beta) == 1.0

    return {
        "status": "ok",
        "beta": beta,
        "half_width": half_width,
        "max_pairwise_detailed_balance_residual": max_db,
        "max_transition_row_sum_residual": row_residual,
        "stationarity_residual": invariant_residual,
        "replay_steps": 64,
        "accepted_moves": accepted_a,
        "final_theta": trajectory_a[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("this oracle currently exposes only --self-test")
    print(json.dumps(self_test(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
