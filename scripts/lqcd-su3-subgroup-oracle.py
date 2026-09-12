#!/usr/bin/env python3
"""Independent SU(3) subgroup-proposal oracle for LQCD-011.

Standard-library only. Imports no Symthaea code. Freezes the algebraic and
reversibility semantics for a Cabibbo-Marinari-style Metropolis proposal using
the three embedded SU(2) subgroups of SU(3). This is not an ensemble generator.
"""

import argparse
import math

PAIRS = ((0, 1), (0, 2), (1, 2))
TOL = 1.0e-12


def identity():
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(3)] for i in range(3)]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def dagger(a):
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]


def trace(a):
    return a[0][0] + a[1][1] + a[2][2]


def det3(a):
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def max_matrix_error(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))


def validate_su3(a, tol=TOL):
    ue = max_matrix_error(mul(a, dagger(a)), identity())
    de = abs(det3(a) - 1.0)
    if not math.isfinite(ue) or not math.isfinite(de) or ue > tol or de > tol:
        raise ValueError(f"invalid SU(3): unitarity={ue}, determinant={de}")
    return ue, de


def embedded_su2(pair, axis, angle):
    if pair not in PAIRS or not math.isfinite(angle):
        raise ValueError("invalid subgroup proposal")
    norm = math.sqrt(sum(x * x for x in axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("proposal axis must have finite positive norm")
    nx, ny, nz = (x / norm for x in axis)
    c, s = math.cos(angle), math.sin(angle)
    u00 = c + 1j * s * nz
    u01 = s * ny + 1j * s * nx
    u10 = -s * ny + 1j * s * nx
    u11 = c - 1j * s * nz
    out = identity()
    i, j = pair
    out[i][i], out[i][j], out[j][i], out[j][j] = u00, u01, u10, u11
    validate_su3(out)
    return out


def shift(site, mu, step, dims):
    out = list(site)
    out[mu] = (out[mu] + step) % dims[mu]
    return tuple(out)


def site_index(site, dims):
    x, y, z, t = site
    return (((x * dims[1] + y) * dims[2] + z) * dims[3] + t)


def identity_field(dims):
    if len(dims) != 4 or any(n <= 0 for n in dims):
        raise ValueError("all four lattice extents must be positive")
    return [identity() for _ in range(math.prod(dims) * 4)]


def link(field, dims, site, mu):
    return field[site_index(site, dims) * 4 + mu]


def set_link(field, dims, site, mu, value):
    validate_su3(value)
    field[site_index(site, dims) * 4 + mu] = value


def plaquette(field, dims, site, mu, nu):
    x_mu = shift(site, mu, 1, dims)
    x_nu = shift(site, nu, 1, dims)
    return mul(
        mul(mul(link(field, dims, site, mu), link(field, dims, x_mu, nu)), dagger(link(field, dims, x_nu, mu))),
        dagger(link(field, dims, site, nu)),
    )


def wilson_action(field, dims, beta):
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    total = 0.0
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    site = (x, y, z, t)
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            total += beta * (1.0 - trace(plaquette(field, dims, site, mu, nu)).real / 3.0)
    return total


def affected_action(field, dims, site, mu, beta):
    total = 0.0
    for nu in range(4):
        if nu == mu:
            continue
        a, b = sorted((mu, nu))
        for base in (site, shift(site, nu, -1, dims)):
            total += beta * (1.0 - trace(plaquette(field, dims, base, a, b)).real / 3.0)
    return total


def metropolis_acceptance(delta_action):
    if not math.isfinite(delta_action):
        raise ValueError("non-finite action difference")
    return 1.0 if delta_action <= 0.0 else math.exp(-delta_action)


def self_test():
    axis, angle = (1.0, 2.0, 3.0), 0.2
    max_unitarity = max_det = max_inverse = 0.0
    for pair in PAIRS:
        proposal = embedded_su2(pair, axis, angle)
        inverse = embedded_su2(pair, axis, -angle)
        ue, de = validate_su3(proposal)
        max_unitarity = max(max_unitarity, ue)
        max_det = max(max_det, de)
        max_inverse = max(max_inverse, max_matrix_error(inverse, dagger(proposal)))
        validate_su3(mul(proposal, identity()))

    dims, beta, site, mu = (2, 2, 1, 1), 6.0, (0, 0, 0, 0), 0
    field = identity_field(dims)
    before_full = wilson_action(field, dims, beta)
    before_local = affected_action(field, dims, site, mu, beta)
    proposal = embedded_su2((0, 1), axis, angle)
    original = link(field, dims, site, mu)
    set_link(field, dims, site, mu, mul(proposal, original))
    after_full = wilson_action(field, dims, beta)
    after_local = affected_action(field, dims, site, mu, beta)
    delta_full, delta_local = after_full - before_full, after_local - before_local
    parity_error = abs(delta_full - delta_local)
    acceptance = metropolis_acceptance(delta_full)
    if abs(delta_full - 0.159467377270067) > TOL or parity_error > TOL:
        raise AssertionError((delta_full, delta_local))
    if abs(acceptance - 0.8525977810098941) > TOL:
        raise AssertionError(acceptance)
    reverse = embedded_su2((0, 1), axis, -angle)
    set_link(field, dims, site, mu, mul(reverse, link(field, dims, site, mu)))
    restoration_action_error = abs(wilson_action(field, dims, beta) - before_full)
    restoration_link_error = max_matrix_error(link(field, dims, site, mu), original)
    if restoration_action_error > TOL or restoration_link_error > TOL:
        raise AssertionError((restoration_action_error, restoration_link_error))
    print("ok")
    print(f"max_unitarity_error={max_unitarity:.17g}")
    print(f"max_determinant_error={max_det:.17g}")
    print(f"max_inverse_error={max_inverse:.17g}")
    print(f"delta_action_full={delta_full:.17g}")
    print(f"delta_action_local={delta_local:.17g}")
    print(f"local_delta_parity_error={parity_error:.17g}")
    print(f"forward_acceptance={acceptance:.17g}")
    print(f"restoration_action_error={restoration_action_error:.17g}")
    print(f"restoration_link_error={restoration_link_error:.17g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("only --self-test is supported; this is a qualification oracle")


if __name__ == "__main__":
    main()
