#!/usr/bin/env python3
"""Matched U(1) TEBD baseline using Quimb CircuitMPS.

This runner deliberately does not claim that Quimb's native norm estimate is
an exact dense-state error. For N<=12 it computes both a dense discrete-circuit
oracle (same ordered gate matrices) and a sparse continuous-time oracle
(exp(-i T sum_j H_j)). The two errors are reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply
import quimb
import quimb.tensor as qtn


def local_hamiltonian(term: dict[str, Any]) -> np.ndarray:
    """Return H in |00>,|01>,|10>,|11> order; first site is the MSB."""
    mu_l = float(term["mu_left"])
    mu_r = float(term["mu_right"])
    interaction = float(term["density_interaction"])
    hopping = complex(term["hopping"]["re"], term["hopping"]["im"])
    h = np.zeros((4, 4), dtype=np.complex128)
    h[1, 1] = mu_r
    h[2, 2] = mu_l
    h[3, 3] = mu_l + mu_r + interaction
    h[1, 2] = hopping
    h[2, 1] = hopping.conjugate()
    return h


def make_gate(term: dict[str, Any], tau: float) -> np.ndarray:
    return expm(-1j * tau * local_hamiltonian(term))


def apply_dense_two_site(
    state: np.ndarray, gate: np.ndarray, nsites: int, bond: int
) -> np.ndarray:
    """Apply a two-site gate in MSB-first site order without constructing U_full."""
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * nsites)
    rest = [i for i in range(nsites) if i not in (bond, bond + 1)]
    permutation = [bond, bond + 1, *rest]
    inverse = np.argsort(permutation)
    grouped = np.transpose(tensor, permutation).reshape(4, -1)
    updated = gate @ grouped
    return np.transpose(updated.reshape((2, 2) + (2,) * (nsites - 2)), inverse).reshape(-1)


def ordered_strang_gates(case: dict[str, Any]):
    """Yield exactly the even(dt/2), odd(dt), even(dt/2) gate sequence."""
    terms = {int(term["bond_index"]): term for term in case["bonds"]}
    dt = float(case["time_step"])
    for _ in range(int(case["steps"])):
        for parity, scale in ((0, 0.5), (1, 1.0), (0, 0.5)):
            for bond in sorted(index for index in terms if index % 2 == parity):
                yield bond, make_gate(terms[bond], dt * scale)


def initial_dense(case: dict[str, Any]) -> np.ndarray:
    n = int(case["site_count"])
    bits = case["initial_bits"]
    if len(bits) != n or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"invalid initial_bits for {case['id']}")
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state = np.zeros(1 << n, dtype=np.complex128)
    state[index] = 1.0
    return state


def dense_discrete_oracle(case: dict[str, Any], initial: np.ndarray) -> np.ndarray:
    state = initial.copy()
    for bond, gate in ordered_strang_gates(case):
        state = apply_dense_two_site(state, gate, int(case["site_count"]), bond)
    return state


def embed_two_site_operator(local: np.ndarray, nsites: int, bond: int) -> csr_matrix:
    identity = eye(2, dtype=np.complex128, format="csr")
    left = eye(1 << bond, dtype=np.complex128, format="csr")
    right = eye(1 << (nsites - bond - 2), dtype=np.complex128, format="csr")
    return kron(kron(left, csr_matrix(local), format="csr"), right, format="csr")


def continuous_hamiltonian(case: dict[str, Any]) -> csr_matrix:
    n = int(case["site_count"])
    dim = 1 << n
    total = csr_matrix((dim, dim), dtype=np.complex128)
    for term in case["bonds"]:
        bond = int(term["bond_index"])
        total += embed_two_site_operator(local_hamiltonian(term), n, bond)
    return total.tocsr()


def state_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    l2 = float(np.linalg.norm(actual - reference))
    norm_actual = float(np.vdot(actual, actual).real)
    norm_reference = float(np.vdot(reference, reference).real)
    overlap = np.vdot(reference, actual)
    fidelity = float(abs(overlap) ** 2 / max(norm_actual * norm_reference, np.finfo(float).tiny))
    return {
        "state_l2_error": l2,
        "fidelity": fidelity,
        "actual_norm_squared": norm_actual,
        "reference_norm_squared": norm_reference,
    }


def charge_metrics(state: np.ndarray, nsites: int, target_charge: int) -> dict[str, float]:
    probabilities = np.abs(state) ** 2
    leakage = 0.0
    mean_charge = 0.0
    charge_variance = 0.0
    for basis, probability in enumerate(probabilities):
        charge = int(basis).bit_count()
        mean_charge += charge * float(probability)
        if charge != target_charge:
            leakage += float(probability)
    for basis, probability in enumerate(probabilities):
        charge = int(basis).bit_count()
        charge_variance += (charge - mean_charge) ** 2 * float(probability)
    return {
        "total_charge_target": float(target_charge),
        "total_charge_expectation": mean_charge,
        "total_charge_variance": charge_variance,
        "charge_leakage_probability": leakage,
    }


def maybe_max_bond(circuit: Any) -> int | None:
    try:
        psi = circuit.psi
        value = getattr(psi, "max_bond", None)
        if callable(value):
            return int(value())
        if value is not None:
            return int(value)
        sizes = getattr(psi, "bond_sizes", None)
        if callable(sizes):
            return int(max(sizes(), default=1))
    except Exception:
        return None
    return None


def run_once(case: dict[str, Any], cutoff: float, cutoff_mode: str, max_bond: int):
    n = int(case["site_count"])
    bits = "".join(str(int(bit)) for bit in case["initial_bits"])
    psi0 = qtn.MPS_computational_state(bits)
    circuit = qtn.CircuitMPS(
        N=n,
        psi0=psi0,
        max_bond=max_bond,
        cutoff=cutoff,
        gate_opts={
            "max_bond": max_bond,
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
            "renorm": 0,
        },
        gate_contract="auto-mps",
        dtype="complex128",
    )
    gate_count = 0
    start = time.perf_counter()
    for bond, gate in ordered_strang_gates(case):
        circuit.apply_gate_raw(gate, (bond, bond + 1))
        gate_count += 1
    elapsed = time.perf_counter() - start

    state_dense = None
    dense_metrics = None
    continuous_metrics = None
    charge = None
    nsites = int(case["site_count"])
    if nsites <= 12 and case.get("oracle", {}).get("discrete_dense", False):
        state_dense = np.asarray(circuit.to_dense(), dtype=np.complex128).reshape(-1)
        initial = initial_dense(case)
        discrete = dense_discrete_oracle(case, initial)
        dense_metrics = state_metrics(state_dense, discrete)
        charge = charge_metrics(state_dense, nsites, int(case["total_charge"]))
        if case.get("oracle", {}).get("continuous_dense", False):
            total_h = continuous_hamiltonian(case)
            total_time = float(case["time_step"]) * int(case["steps"])
            continuous = expm_multiply((-1j * total_time) * total_h, initial)
            continuous_metrics = state_metrics(state_dense, continuous)

    norm_estimate = float(np.real(circuit.psi.H @ circuit.psi))
    try:
        quimb_fidelity = float(circuit.fidelity_estimate())
        quimb_error = float(circuit.error_estimate())
    except Exception:
        quimb_fidelity = None
        quimb_error = None

    result = {
        "implementation": "quimb.CircuitMPS",
        "quimb_version": getattr(quimb, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "case_id": case["id"],
        "site_count": nsites,
        "total_charge": int(case["total_charge"]),
        "time_step": float(case["time_step"]),
        "steps": int(case["steps"]),
        "strang_order": ["even_dt_over_2", "odd_dt", "even_dt_over_2"],
        "gate_count": gate_count,
        "cutoff": cutoff,
        "cutoff_mode": cutoff_mode,
        "max_bond": max_bond,
        "renormalization": "disabled_via_renorm_0",
        "runtime_seconds": elapsed,
        "gate_throughput_per_second": gate_count / elapsed if elapsed > 0 else None,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "norm_squared_from_network": norm_estimate,
        "quimb_norm_based_fidelity_estimate": quimb_fidelity,
        "quimb_norm_based_error_estimate": quimb_error,
        "observed_max_bond": maybe_max_bond(circuit),
        "dense_discrete_oracle": dense_metrics,
        "continuous_time_oracle": continuous_metrics,
        "charge_metrics": charge,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parents[2] / "u1_tebd" / "fixtures" / "u1_tebd_cases_v1.json")
    parser.add_argument("--case", default="u1-two-site-hopping-known-answer")
    parser.add_argument("--track", choices=("fixed_bond", "matched_error"), default="fixed_bond")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--cutoff-mode", choices=("rel", "rsum2"), default="rel")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    case = next((item for item in fixture["cases"] if item["id"] == args.case), None)
    if case is None:
        raise SystemExit(f"unknown case {args.case!r}")

    if args.track == "fixed_bond":
        cutoff = 0.0 if args.cutoff is None else args.cutoff
        max_bond = int(case.get("max_bond_dimension", 16) if args.max_bond is None else args.max_bond)
    else:
        cutoff = 1e-8 if args.cutoff is None else args.cutoff
        max_bond = 256 if args.max_bond is None else args.max_bond

    for _ in range(max(0, args.warmups)):
        run_once(case, cutoff, args.cutoff_mode, max_bond)
    records = [run_once(case, cutoff, args.cutoff_mode, max_bond) for _ in range(max(1, args.repeats))]
    times = [float(record["runtime_seconds"]) for record in records]
    summary = {
        "schema": "luminous.u1_tebd_benchmark_result.v1",
        "fixture_schema": fixture.get("schema"),
        "track": args.track,
        "case_id": case["id"],
        "repeats": len(records),
        "warmups": max(0, args.warmups),
        "runtime_median_seconds": statistics.median(times),
        "runtime_p95_seconds": float(np.percentile(times, 95)),
        "records": records,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "thread_env": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
            )},
        },
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
