#!/usr/bin/env python3
"""Finite-state conformal-risk recertification via augmented multisets.

Independent research implementation for finite observation alphabets and
losses linear in a conservativeness parameter lambda:
    loss_r(z, lambda) = g[r,z] * (1-lambda), lambda in [0,1].
Larger lambda is more conservative; all risks must hold jointly.

This implements exact finite-exchangeability validity, equal-share
recertification, and the finite-domain coordinate sweep. It is not a
distribution-shift or conditional-on-features guarantee.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Iterable
import numpy as np

SCHEMA = "symthaea-finite-state-crc-recertification-v1"


def count_vectors(total: int, types: int) -> list[tuple[int, ...]]:
    """All count-vector multisets of total objects over types categories."""
    if total < 0 or types < 1:
        raise ValueError("total must be >=0 and types must be >=1")
    if types == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for tail in count_vectors(total - first, types - 1):
            out.append((first,) + tail)
    return out


def add_type(counts: tuple[int, ...], type_index: int) -> tuple[int, ...]:
    result = list(counts)
    result[type_index] += 1
    return tuple(result)


def remove_type(counts: tuple[int, ...], type_index: int) -> tuple[int, ...]:
    result = list(counts)
    if result[type_index] <= 0:
        raise ValueError("cannot remove an absent type")
    result[type_index] -= 1
    return tuple(result)


class LinearFiniteCRC:
    """Exact CRC and recertification engine for finite-type linear losses."""

    def __init__(self, risk_coefficients: Iterable[Iterable[float]] | Iterable[float],
                 alpha: float | Iterable[float], n_calibration: int):
        g = np.asarray(risk_coefficients, dtype=np.float64)
        if g.ndim == 1:
            g = g[None, :]
        if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1:
            raise ValueError("risk_coefficients must be a nonempty risk-by-type matrix")
        if np.any(~np.isfinite(g)) or np.any(g < 0):
            raise ValueError("loss coefficients must be finite and nonnegative")
        a = np.asarray([alpha] if np.ndim(alpha) == 0 else alpha, dtype=np.float64)
        if a.shape != (g.shape[0],) or np.any(~np.isfinite(a)) or np.any((a <= 0) | (a >= 1)):
            raise ValueError("one alpha in (0,1) is required per risk")
        if n_calibration < 1:
            raise ValueError("n_calibration must be positive")
        self.g = g
        self.alphas = a
        self.n = int(n_calibration)
        self.k_types = int(g.shape[1])
        self.k_risks = int(g.shape[0])
        self.states = count_vectors(self.n, self.k_types)
        self.augmented_states = count_vectors(self.n + 1, self.k_types)
        self.budgets = (self.n + 1) * self.alphas

    def loss(self, risk: int, type_index: int, action: float) -> float:
        if not 0 <= risk < self.k_risks or not 0 <= type_index < self.k_types:
            raise IndexError("risk or type index out of bounds")
        if not math.isfinite(action) or not 0 <= action <= 1:
            raise ValueError("action lambda must lie in [0,1]")
        return float(self.g[risk, type_index] * (1.0 - action))

    def total_heldout_losses(self, controller: dict[tuple[int, ...], float],
                             augmented: tuple[int, ...]) -> np.ndarray:
        if sum(augmented) != self.n + 1:
            raise ValueError("augmented count vector must have n+1 observations")
        total = np.zeros(self.k_risks, dtype=np.float64)
        for z, multiplicity in enumerate(augmented):
            if multiplicity == 0:
                continue
            calibration = remove_type(augmented, z)
            if calibration not in controller:
                raise KeyError("controller missing a leave-one-out calibration state")
            action = controller[calibration]
            total += multiplicity * self.g[:, z] * (1.0 - action)
        return total

    def validity_report(self, controller: dict[tuple[int, ...], float],
                        tolerance: float = 1e-10) -> dict:
        if set(controller) != set(self.states):
            raise ValueError("controller must define every calibration count vector exactly once")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in controller.values()):
            raise ValueError("controller actions must be finite and in [0,1]")
        totals = {S: self.total_heldout_losses(controller, S) for S in self.augmented_states}
        max_excess = np.max(np.stack(list(totals.values())) - self.budgets[None, :], axis=0)
        return {
            "valid": bool(np.all(max_excess <= tolerance)),
            "max_budget_excess_by_risk": max_excess.tolist(),
            "augmented_totals": {",".join(map(str, S)): v.tolist() for S, v in totals.items()},
            "number_of_augmented_states": len(totals),
        }

    def envelope_crc(self) -> dict[tuple[int, ...], float]:
        """Joint envelope CRC, with U_r(lambda)=max_z loss_r(z,lambda)."""
        maxima = self.g.max(axis=1)
        controller = {}
        for D in self.states:
            calibration_coeff = self.g @ np.asarray(D, dtype=np.float64)
            denom = calibration_coeff + maxima
            bounds = np.where(denom > 0, 1.0 - self.budgets / np.maximum(denom, 1e-300), 0.0)
            controller[D] = float(np.clip(np.max(bounds), 0.0, 1.0))
        if not self.validity_report(controller)["valid"]:
            raise ArithmeticError("envelope CRC construction failed exact augmented-sample validity")
        return controller

    def standard_crc(self) -> dict[tuple[int, ...], float]:
        """Joint standard CRC with the global maximum loss as test-loss bound."""
        maxima = self.g.max(axis=1)
        controller = {}
        for D in self.states:
            calibration_coeff = self.g @ np.asarray(D, dtype=np.float64)
            bounds = []
            for r in range(self.k_risks):
                G = calibration_coeff[r]
                B = maxima[r]
                budget = self.budgets[r]
                if G <= 0:
                    if B > budget + 1e-12:
                        raise ValueError("standard CRC has no feasible action for this risk")
                    bounds.append(0.0)
                else:
                    required = 1.0 - (budget - B) / G
                    if required > 1.0 + 1e-12:
                        raise ValueError("standard CRC has no feasible action for this risk")
                    bounds.append(max(0.0, required))
            controller[D] = float(np.clip(max(bounds), 0.0, 1.0))
        if not self.validity_report(controller)["valid"]:
            raise ArithmeticError("standard CRC construction failed augmented-sample validity")
        return controller

    def _minimal_action(self, baseline_action: float,
                        allowances: list[tuple[int, int, float]]) -> float:
        """Exact inversion for linear losses; returns the least feasible lambda."""
        required = 0.0
        for risk, z, allowance in allowances:
            coefficient = self.g[risk, z]
            if coefficient > 0:
                required = max(required, baseline_action - allowance / coefficient)
        return float(np.clip(required, 0.0, baseline_action))

    def recertify_equal_share(self, controller: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
        """One valid equal-share recertification round."""
        if not self.validity_report(controller)["valid"]:
            raise ValueError("starting controller is not valid")
        updated = {}
        for D in self.states:
            baseline_action = controller[D]
            allowances = []
            for z in range(self.k_types):
                augmented = add_type(D, z)
                slack = self.budgets - self.total_heldout_losses(controller, augmented)
                if np.any(slack < -1e-8):
                    raise ArithmeticError("negative unused budget on an augmented sample")
                for risk in range(self.k_risks):
                    allowances.append((risk, z, max(0.0, float(slack[risk])) / (self.n + 1)))
            updated[D] = self._minimal_action(baseline_action, allowances)
        if not self.validity_report(updated)["valid"]:
            raise ArithmeticError("equal-share recertification broke exact validity")
        if any(updated[D] > controller[D] + 1e-12 for D in self.states):
            raise ArithmeticError("recertification became more conservative")
        return updated

    def recertify_until(self, controller: dict[tuple[int, ...], float],
                        max_rounds: int = 500, tolerance: float = 1e-10) -> tuple[dict, list[dict]]:
        if max_rounds < 1 or tolerance <= 0:
            raise ValueError("invalid round limit or tolerance")
        current = dict(controller)
        history = [dict(current)]
        for _ in range(max_rounds):
            updated = self.recertify_equal_share(current)
            history.append(updated)
            change = max(abs(updated[D] - current[D]) for D in self.states)
            current = updated
            if change <= tolerance:
                break
        return current, history

    def coordinate_sweep(self, controller: dict[tuple[int, ...], float],
                         state_order: Iterable[tuple[int, ...]] | None = None) -> dict:
        """One fixed-order finite-domain sweep; valid even without convex loss."""
        if not self.validity_report(controller)["valid"]:
            raise ValueError("starting controller is not valid")
        order = list(self.states if state_order is None else state_order)
        if len(order) != len(self.states) or set(order) != set(self.states):
            raise ValueError("state_order must be a permutation of every calibration multiset")
        updated = dict(controller)
        for D in order:
            baseline_action = updated[D]
            allowances = []
            for z in range(self.k_types):
                augmented = add_type(D, z)
                slack = self.budgets - self.total_heldout_losses(updated, augmented)
                if np.any(slack < -1e-8):
                    raise ArithmeticError("negative unused budget during coordinate sweep")
                multiplicity = augmented[z]
                for risk in range(self.k_risks):
                    allowances.append((risk, z, max(0.0, float(slack[risk])) / multiplicity))
            updated[D] = self._minimal_action(baseline_action, allowances)
            if not self.validity_report(updated)["valid"]:
                raise ArithmeticError("coordinate update broke exact augmented-sample validity")
        return updated

    def is_coordinatewise_admissible(self, controller: dict[tuple[int, ...], float],
                                     tolerance: float = 1e-8) -> bool:
        if not self.validity_report(controller)["valid"]:
            return False
        for D, baseline_action in controller.items():
            allowances = []
            for z in range(self.k_types):
                augmented = add_type(D, z)
                slack = self.budgets - self.total_heldout_losses(controller, augmented)
                for risk in range(self.k_risks):
                    allowances.append((risk, z, max(0.0, float(slack[risk])) / augmented[z]))
            minimum = self._minimal_action(baseline_action, allowances)
            if minimum < baseline_action - tolerance:
                return False
        return True

    @staticmethod
    def pointwise_minimum(*controllers: dict[tuple[int, ...], float]) -> dict:
        if len(controllers) < 2:
            raise ValueError("at least two controllers are required")
        keys = set(controllers[0])
        if any(set(c) != keys for c in controllers):
            raise ValueError("controllers must have identical state domains")
        return {D: min(c[D] for c in controllers) for D in keys}


def two_type_reference() -> dict:
    engine = LinearFiniteCRC([1.0, 0.5], 0.5, 2)
    envelope = engine.envelope_crc()
    standard = engine.standard_crc()
    equal_one = engine.recertify_equal_share(envelope)
    equal_two = engine.recertify_equal_share(equal_one)
    limit, history = engine.recertify_until(envelope, max_rounds=500, tolerance=1e-12)
    hh, hl, ll = (2, 0), (1, 1), (0, 2)
    alternate = engine.coordinate_sweep(envelope, [ll, hl, hh])
    minimum = engine.pointwise_minimum(limit, alternate)
    controllers = {
        "standard_crc": standard,
        "envelope_crc": envelope,
        "equal_share_round_1": equal_one,
        "equal_share_round_2": equal_two,
        "equal_share_limit": limit,
        "coordinate_sweep_alternate": alternate,
        "pointwise_minimum": minimum,
    }
    return {
        "engine": engine,
        "controllers": controllers,
        "validity": {name: engine.validity_report(c) for name, c in controllers.items()},
        "equal_share_history": history,
        "coordinatewise_admissible": {
            "limit": engine.is_coordinatewise_admissible(limit),
            "alternate": engine.is_coordinatewise_admissible(alternate),
        },
    }


def summarize_named(controller: dict[tuple[int, ...], float]) -> dict[str, float]:
    return {"HH": float(controller[(2, 0)]),
            "HL": float(controller[(1, 1)]),
            "LL": float(controller[(0, 2)])}


def run_pilot() -> dict:
    result = two_type_reference()
    controllers = result["controllers"]
    engine: LinearFiniteCRC = result["engine"]
    return {
        "schema": SCHEMA,
        "status": "LOCAL_EXACT_FINITE_STATE_REPRODUCTION_NOT_CIFAR_REPLICATION",
        "reference": {"paper": "Bai & Ramdas (2026), arXiv:2610.35821",
                      "section": "3.2-3.5 and Appendix B.5",
                      "url": "https://arxiv.org/abs/2610.35821"},
        "protocol": {
            "types": ["H", "L"], "n_calibration": 2, "alpha": 0.5,
            "loss": "loss(H,lambda)=1-lambda; loss(L,lambda)=0.5*(1-lambda)",
            "augmented_states": ["HHH", "HHL", "HLL", "LLL"],
            "validity_criterion": "sum_i loss(z_i,A(S_without_i)) <= (n+1)*alpha for every augmented multiset"},
        "controllers": {name: summarize_named(c) for name, c in controllers.items()},
        "validity": {
            name: {"valid": report["valid"], "augmented_totals": report["augmented_totals"]}
            for name, report in result["validity"].items()},
        "admissibility": result["coordinatewise_admissible"],
        "equal_share_rounds": [
            {"round": k, **summarize_named(A)}
            for k, A in enumerate(result["equal_share_history"][:6])],
        "table_sizes": {
            "types_2_n_2_calibration_states": len(engine.states),
            "types_2_n_2_augmented_states": len(engine.augmented_states),
            "types_3_n_50_calibration_states": math.comb(52, 2),
            "types_3_n_50_augmented_states": math.comb(53, 2)},
        "cifar100_reference_target": {
            "calibration_n": 50, "splits": 300, "test_images_per_split": 16,
            "alpha": [0.02, 0.05, 0.10, 0.15],
            "confidence_adaptive_at_alpha_0_10": {
                "envelope_crc_correct_specificity_percent": 67.572,
                "recertified_500_correct_specificity_percent": 68.235,
                "envelope_crc_mean_node_size": 19.40,
                "recertified_500_mean_node_size": 17.03},
            "status": "published reference values only; data/code/splits not yet reproduced here"},
        "limitations": [
            "exact engine currently covers finite types and losses linear in lambda",
            "two-type example is a mathematical reproduction, not a real-data experiment",
            "finite exchangeability validity does not imply validity under arbitrary distribution shift",
            "test-feature recertification must check every possible label; sampled labels alone do not certify",
            "CIFAR-100 cached ResNet-18 predictions and exact paired splits are not yet present in this branch"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_pilot()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    Path(args.output).write_bytes(raw)
    print(json.dumps({"output": args.output, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "validity": {k: v["valid"] for k, v in result["validity"].items()}}, indent=2))
