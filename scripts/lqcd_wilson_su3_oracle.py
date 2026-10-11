#!/usr/bin/env python3
"""Independent Wilson-action oracle for a tiny periodic SU(3) lattice.

Purpose:
- freeze exact lattice indexing / plaquette / Wilson-action semantics;
- verify local gauge invariance independently of Symthaea Rust code;
- provide deterministic synthetic fixtures for parity tests.

This is deliberately NOT a production lattice-QCD simulator. It does not
sample an equilibrium ensemble, include fermions, estimate continuum physics,
or claim any physical prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence, Tuple

ComplexMatrix3 = List[List[complex]]
Site = Tuple[int, int, int, int]


def identity3() -> ComplexMatrix3:
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(3)] for i in range(3)]


def mat_mul(a: ComplexMatrix3, b: ComplexMatrix3) -> ComplexMatrix3:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def dagger(a: ComplexMatrix3) -> ComplexMatrix3:
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]


def trace(a: ComplexMatrix3) -> complex:
    return sum(a[i][i] for i in range(3))


def determinant3(a: ComplexMatrix3) -> complex:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def su3_diagonal(theta: float, phi: float) -> ComplexMatrix3:
    """A deterministic diagonal SU(3) element.

    diag(exp(i theta), exp(i phi), exp(-i(theta+phi)))
    """
    phases = (theta, phi, -(theta + phi))
    out = [[0.0j for _ in range(3)] for _ in range(3)]
    for i, angle in enumerate(phases):
        out[i][i] = complex(math.cos(angle), math.sin(angle))
    return out


def max_unitarity_error(a: ComplexMatrix3) -> float:
    aa = mat_mul(a, dagger(a))
    ident = identity3()
    return max(abs(aa[i][j] - ident[i][j]) for i in range(3) for j in range(3))


def assert_su3(a: ComplexMatrix3, tol: float = 1e-12) -> None:
    err = max_unitarity_error(a)
    det_err = abs(determinant3(a) - 1.0)
    if err > tol:
        raise ValueError(f"matrix is not unitary within tolerance: {err}")
    if det_err > tol:
        raise ValueError(f"matrix determinant is not 1 within tolerance: {det_err}")


class GaugeField:
    def __init__(self, dims: Sequence[int]) -> None:
        if len(dims) != 4 or any(int(n) <= 0 for n in dims):
            raise ValueError("dims must contain four positive extents")
        self.dims = tuple(int(n) for n in dims)
        self.links: Dict[Tuple[Site, int], ComplexMatrix3] = {
            (x, mu): identity3() for x in self.sites() for mu in range(4)
        }

    def sites(self) -> Iterable[Site]:
        return product(*(range(n) for n in self.dims))

    def shift(self, x: Site, mu: int, step: int = 1) -> Site:
        if not 0 <= mu < 4:
            raise ValueError("direction must be in 0..3")
        y = list(x)
        y[mu] = (y[mu] + step) % self.dims[mu]
        return tuple(y)  # type: ignore[return-value]

    def set_link(self, x: Site, mu: int, value: ComplexMatrix3) -> None:
        assert_su3(value)
        self.links[(x, mu)] = value

    def plaquette(self, x: Site, mu: int, nu: int) -> ComplexMatrix3:
        if mu == nu:
            raise ValueError("plaquette directions must differ")
        u_mu_x = self.links[(x, mu)]
        u_nu_x_plus_mu = self.links[(self.shift(x, mu), nu)]
        u_mu_x_plus_nu = self.links[(self.shift(x, nu), mu)]
        u_nu_x = self.links[(x, nu)]
        return mat_mul(
            mat_mul(mat_mul(u_mu_x, u_nu_x_plus_mu), dagger(u_mu_x_plus_nu)),
            dagger(u_nu_x),
        )

    def average_plaquette(self) -> float:
        values = [
            trace(self.plaquette(x, mu, nu)).real / 3.0
            for x in self.sites()
            for mu, nu in combinations(range(4), 2)
        ]
        return sum(values) / len(values)

    def wilson_action(self, beta: float) -> float:
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        return beta * sum(
            1.0 - trace(self.plaquette(x, mu, nu)).real / 3.0
            for x in self.sites()
            for mu, nu in combinations(range(4), 2)
        )

    def polyakov_loop(self, spatial: Tuple[int, int, int]) -> complex:
        x, y, z = spatial
        if not (
            0 <= x < self.dims[0]
            and 0 <= y < self.dims[1]
            and 0 <= z < self.dims[2]
        ):
            raise ValueError("spatial coordinate out of bounds")
        product_u = identity3()
        for t in range(self.dims[3]):
            product_u = mat_mul(product_u, self.links[((x, y, z, t), 3)])
        return trace(product_u) / 3.0

    def gauge_transform(self, local_gauge: Dict[Site, ComplexMatrix3]) -> "GaugeField":
        expected = set(self.sites())
        if set(local_gauge) != expected:
            raise ValueError("gauge transformation must provide exactly one SU(3) matrix per site")
        for g in local_gauge.values():
            assert_su3(g)

        transformed = GaugeField(self.dims)
        for x in self.sites():
            for mu in range(4):
                transformed.links[(x, mu)] = mat_mul(
                    mat_mul(local_gauge[x], self.links[(x, mu)]),
                    dagger(local_gauge[self.shift(x, mu)]),
                )
        return transformed


def deterministic_fixture() -> dict:
    dims = (2, 2, 1, 1)
    beta = 6.0
    field = GaugeField(dims)
    identity_action = field.wilson_action(beta)
    identity_plaquette = field.average_plaquette()
    identity_polyakov = field.polyakov_loop((0, 0, 0))

    field.set_link((0, 0, 0, 0), 0, su3_diagonal(0.3, -0.1))
    flux_action = field.wilson_action(beta)
    flux_plaquette = field.average_plaquette()

    gauges = {
        x: su3_diagonal(0.07 * sum(x), -0.03 * (1 + x[0]))
        for x in field.sites()
    }
    transformed = field.gauge_transform(gauges)
    transformed_action = transformed.wilson_action(beta)
    transformed_plaquette = transformed.average_plaquette()

    return {
        "schema": "symthaea-lqcd-wilson-oracle-v1",
        "dims": list(dims),
        "beta": beta,
        "identity": {
            "wilson_action": identity_action,
            "average_plaquette": identity_plaquette,
            "polyakov_loop_re": identity_polyakov.real,
            "polyakov_loop_im": identity_polyakov.imag,
        },
        "localized_diagonal_flux": {
            "wilson_action": flux_action,
            "average_plaquette": flux_plaquette,
        },
        "gauge_transformed_flux": {
            "wilson_action": transformed_action,
            "average_plaquette": transformed_plaquette,
        },
    }


def self_test() -> dict:
    fixture = deterministic_fixture()
    identity = fixture["identity"]
    flux = fixture["localized_diagonal_flux"]
    gauge = fixture["gauge_transformed_flux"]

    assert abs(identity["wilson_action"]) < 1e-14
    assert abs(identity["average_plaquette"] - 1.0) < 1e-14
    assert abs(identity["polyakov_loop_re"] - 1.0) < 1e-14
    assert abs(identity["polyakov_loop_im"]) < 1e-14
    assert flux["wilson_action"] > 0.0
    assert flux["average_plaquette"] < 1.0
    assert abs(flux["wilson_action"] - gauge["wilson_action"]) < 1e-12
    assert abs(flux["average_plaquette"] - gauge["average_plaquette"]) < 1e-12

    # Periodic boundary closure.
    field = GaugeField((2, 3, 4, 5))
    assert field.shift((1, 2, 3, 4), 0) == (0, 2, 3, 4)
    assert field.shift((1, 2, 3, 4), 3) == (1, 2, 3, 0)

    fixture["self_test"] = "ok"
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-fixture", action="store_true")
    args = parser.parse_args()

    fixture = self_test()
    if args.emit_fixture:
        print(json.dumps(fixture, indent=2, sort_keys=True))
    else:
        print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
