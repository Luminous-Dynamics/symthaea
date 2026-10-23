#!/usr/bin/env python3
"""Independent finite-probe oracle for SU(3) subgroup overrelaxation.

Standard-library only. Imports no Symthaea code. The oracle reconstructs the
local SU(2) quaternion force from five subgroup probes (+I, -I, i sigma_1,
i sigma_2, i sigma_3), then reflects the identity quaternion across the local
equal-action hyperplane. This intentionally avoids hard-coding staple
orientation conventions in the oracle.
"""
import argparse
import copy
import math

TOL = 2e-13
PAIRS = ((0, 1), (0, 2), (1, 2))


def ident3():
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


def validate_su3(a):
    unitarity = max_matrix_error(mul(a, dagger(a)), ident3())
    determinant = abs(det3(a) - 1.0)
    if not math.isfinite(unitarity) or not math.isfinite(determinant) or unitarity > 1e-12 or determinant > 1e-12:
        raise AssertionError((unitarity, determinant))


def embedded_quaternion(pair, q):
    a0, a1, a2, a3 = q
    if abs(sum(x * x for x in q) - 1.0) > 2e-12:
        raise AssertionError(("non-unit quaternion", q))
    out = ident3()
    i, j = pair
    out[i][i] = a0 + 1j * a3
    out[i][j] = a2 + 1j * a1
    out[j][i] = -a2 + 1j * a1
    out[j][j] = a0 - 1j * a3
    validate_su3(out)
    return out


def embedded_rotation(pair, axis, angle):
    norm = math.sqrt(sum(x * x for x in axis))
    x, y, z = (v / norm for v in axis)
    return embedded_quaternion(
        pair,
        (math.cos(angle), math.sin(angle) * x, math.sin(angle) * y, math.sin(angle) * z),
    )


def site_index(site, dims):
    x, y, z, t = site
    return (((x * dims[1] + y) * dims[2] + z) * dims[3] + t)


def shift(site, mu, step, dims):
    out = list(site)
    out[mu] = (out[mu] + step) % dims[mu]
    return tuple(out)


def identity_field(dims):
    return [ident3() for _ in range(math.prod(dims) * 4)]


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


def touching_keys(dims, site, mu):
    keys = set()
    for nu in range(4):
        if nu == mu:
            continue
        a, b = sorted((mu, nu))
        keys.add((site, a, b))
        keys.add((shift(site, nu, -1, dims), a, b))
    return sorted(keys)


def touching_trace_sum(field, dims, site, mu):
    return sum(trace(plaquette(field, dims, base, a, b)).real for base, a, b in touching_keys(dims, site, mu))


def probe_affine_quaternion_force(field, dims, site, mu, pair):
    original = copy.deepcopy(link(field, dims, site, mu))

    def evaluate(q):
        candidate = copy.deepcopy(field)
        set_link(candidate, dims, site, mu, mul(embedded_quaternion(pair, q), original))
        return touching_trace_sum(candidate, dims, site, mu)

    plus = evaluate((1.0, 0.0, 0.0, 0.0))
    minus = evaluate((-1.0, 0.0, 0.0, 0.0))
    constant = 0.5 * (plus + minus)
    q0 = 0.5 * (plus - minus)
    q1 = evaluate((0.0, 1.0, 0.0, 0.0)) - constant
    q2 = evaluate((0.0, 0.0, 1.0, 0.0)) - constant
    q3 = evaluate((0.0, 0.0, 0.0, 1.0)) - constant
    return constant, (q0, q1, q2, q3)


def reflection_quaternion(force):
    norm_sq = sum(x * x for x in force)
    if norm_sq <= 1e-28 or not math.isfinite(norm_sq):
        raise AssertionError(("degenerate force", force))
    q0 = force[0]
    out = [2.0 * q0 * x / norm_sq for x in force]
    out[0] -= 1.0
    if abs(sum(x * x for x in out) - 1.0) > 2e-12:
        raise AssertionError(("reflection not unit", out))
    return tuple(out)


def overrelax(field, dims, site, mu, pair):
    constant, force = probe_affine_quaternion_force(field, dims, site, mu, pair)
    reflection = reflection_quaternion(force)
    original = copy.deepcopy(link(field, dims, site, mu))
    set_link(field, dims, site, mu, mul(embedded_quaternion(pair, reflection), original))
    return constant, force, reflection


def self_test():
    dims = (2, 2, 2, 2)
    beta = 5.7
    field = identity_field(dims)
    fixtures = (
        ((0, 0, 0, 0), 0, (0, 1), (1.0, 2.0, 3.0), 0.31),
        ((1, 0, 1, 0), 2, (0, 2), (2.0, -1.0, 1.0), -0.27),
        ((0, 1, 0, 1), 3, (1, 2), (1.0, 1.0, -2.0), 0.22),
        ((1, 1, 1, 1), 1, (0, 1), (-2.0, 1.0, 1.0), 0.19),
    )
    for site, mu, pair, axis, angle in fixtures:
        set_link(field, dims, site, mu, mul(embedded_rotation(pair, axis, angle), link(field, dims, site, mu)))

    before = wilson_action(field, dims, beta)
    print("ok")
    print(f"fixture_action={before:.17g}")
    for pair in PAIRS:
        candidate = copy.deepcopy(field)
        original = copy.deepcopy(link(candidate, dims, (0, 0, 0, 0), 0))
        constant, force, reflection = overrelax(candidate, dims, (0, 0, 0, 0), 0, pair)
        after = wilson_action(candidate, dims, beta)
        action_error = abs(after - before)
        trace_error = abs(
            touching_trace_sum(candidate, dims, (0, 0, 0, 0), 0)
            - touching_trace_sum(field, dims, (0, 0, 0, 0), 0)
        )
        overrelax(candidate, dims, (0, 0, 0, 0), 0, pair)
        involution_link_error = max_matrix_error(link(candidate, dims, (0, 0, 0, 0), 0), original)
        involution_action_error = abs(wilson_action(candidate, dims, beta) - before)
        if max(action_error, trace_error, involution_link_error, involution_action_error) > TOL:
            raise AssertionError((pair, action_error, trace_error, involution_link_error, involution_action_error))
        label = f"{pair[0]}{pair[1]}"
        print(f"pair_{label}_constant={constant:.17g}")
        print("pair_%s_force=%s" % (label, ",".join(f"{x:.17g}" for x in force)))
        print("pair_%s_reflection=%s" % (label, ",".join(f"{x:.17g}" for x in reflection)))
        print(f"pair_{label}_action_error={action_error:.17g}")
        print(f"pair_{label}_trace_error={trace_error:.17g}")
        print(f"pair_{label}_involution_link_error={involution_link_error:.17g}")
        print(f"pair_{label}_involution_action_error={involution_action_error:.17g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported; this is a qualification oracle")
    self_test()


if __name__ == "__main__":
    main()
