#!/usr/bin/env python3
"""Independently compare Symthaea's VWN5 golden points against PySCF/Libxc.

Optional developer check; it is deliberately not imported by Rust builds.
Install PySCF in an isolated environment, then run:
    python scripts/qc/check_vwn5_pyscf_reference.py

This checks pointwise LDA correlation energy-per-electron and potential, not
molecular DFT energies, SCF convergence, or chemical accuracy.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any

try:
    import numpy as np
    import pyscf
    from pyscf.dft import libxc
except ImportError as exc:
    print(
        "PySCF is required for this optional reference check. Install it in an "
        f"isolated environment first. Import failed: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


# Same explicit density grid and golden values used by the Rust VWN5 unit test.
# Values come from the canonical VWN5 expression at its published unpolarized
# parameter set. This script checks those independently against PySCF's Libxc.
REFERENCES = (
    (1e-6, -0.004776617504447303, -0.006099907725695197),
    (1e-3, -0.02486479492898193, -0.0297181942740259),
    (1e-1, -0.05339728918594983, -0.06081203033126156),
    (1.0, -0.07159261230679066, -0.07993838317598562),
)
ABS_TOL = 3e-11


def first_derivative(vxc: Any) -> np.ndarray:
    """Extract d(epsilon_xc * rho)/d(rho) from PySCF's LDA result."""
    if not isinstance(vxc, (tuple, list)) or not vxc:
        raise RuntimeError(f"unexpected Libxc potential result: {type(vxc).__name__}")
    return np.asarray(vxc[0], dtype=float).reshape(-1)


def main() -> int:
    densities = np.asarray([entry[0] for entry in REFERENCES], dtype=float)

    # A blank exchange part requests pure VWN correlation in PySCF's parser.
    exc, vxc, _, _ = libxc.eval_xc(",vwn", densities, spin=0, deriv=1)
    observed_eps = np.asarray(exc, dtype=float).reshape(-1)
    observed_v = first_derivative(vxc)

    # Also check the compound alias used by Symthaea: Slater exchange + VWN5.
    full_exc, full_vxc, _, _ = libxc.eval_xc("lda,vwn", densities, spin=0, deriv=1)
    full_observed_eps = np.asarray(full_exc, dtype=float).reshape(-1)
    full_observed_v = first_derivative(full_vxc)

    records = []
    failed = False
    for i, (rho, expected_eps, expected_v) in enumerate(REFERENCES):
        eps = float(observed_eps[i])
        potential = float(observed_v[i])
        eps_error = abs(eps - expected_eps)
        v_error = abs(potential - expected_v)

        slater_eps = -0.75 * (3.0 * rho / math.pi) ** (1.0 / 3.0)
        expected_full_eps = expected_eps + slater_eps
        expected_full_v = expected_v + (4.0 / 3.0) * slater_eps
        full_eps_error = abs(float(full_observed_eps[i]) - expected_full_eps)
        full_v_error = abs(float(full_observed_v[i]) - expected_full_v)

        row_pass = max(eps_error, v_error, full_eps_error, full_v_error) <= ABS_TOL
        failed |= not row_pass
        records.append(
            {
                "rho": rho,
                "expected_vwn5_epsilon": expected_eps,
                "pyscf_vwn5_epsilon": eps,
                "epsilon_abs_error": eps_error,
                "expected_vwn5_potential": expected_v,
                "pyscf_vwn5_potential": potential,
                "potential_abs_error": v_error,
                "full_lda_epsilon_abs_error": full_eps_error,
                "full_lda_potential_abs_error": full_v_error,
                "pass": row_pass,
            }
        )

    result = {
        "schema": "symthaea.vwn5-pyscf-reference.v1",
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "backend": "PySCF Libxc",
        "correlation_functional": ",vwn",
        "compound_functional": "lda,vwn",
        "absolute_tolerance_hartree": ABS_TOL,
        "scope": "pointwise functional validation only; not a molecular DFT validation",
        "pass": not failed,
        "records": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
