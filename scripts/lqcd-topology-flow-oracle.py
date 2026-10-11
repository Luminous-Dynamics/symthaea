#!/usr/bin/env python3
"""Independent clover-topology + finite-difference Wilson-gradient-flow oracle.

Standard-library only. Imports no Symthaea/Rust code. The flow integrator is a
slow semantic reference: it estimates the Wilson-action gradient by central
finite differences along all eight Gell-Mann directions, then applies a
Lie-Euler descent step. It is not the production flow implementation.
"""

import math

DIMS = (2, 2, 2, 2)
FLOW_DT = 1.0e-3
GRAD_EPS = 2.0e-6
FIXTURE_OPS = [
    ((0, 0, 1, 0), 0, 3, -0.5819354339098058),
    ((0, 1, 1, 1), 0, 2, -0.32393958624710995),
    ((0, 1, 1, 1), 0, 6, -0.1250464331865352),
    ((1, 0, 1, 0), 1, 2, 0.19161237542499698),
    ((0, 1, 0, 1), 1, 2, 0.22983089548025992),
    ((1, 0, 1, 1), 1, 3, 0.4032003002590304),
    ((1, 1, 1, 1), 3, 5, 0.5114974521920254),
    ((1, 1, 1, 1), 2, 4, -0.09168927490735845),
    ((1, 1, 1, 0), 1, 6, -0.4726273806149346),
    ((0, 1, 0, 0), 2, 2, 0.3411282616087933),
    ((1, 1, 0, 0), 3, 3, -0.341627594608156),
    ((1, 0, 1, 1), 2, 5, -0.11361497865301007),
    ((1, 0, 1, 0), 1, 4, 0.003219526879812973),
    ((0, 1, 1, 1), 3, 1, 0.2092875716255429),
    ((0, 0, 1, 1), 1, 0, -0.2872940926296875),
    ((1, 0, 1, 1), 1, 3, -0.33032520366686874),
    ((1, 1, 1, 0), 3, 1, 0.5401678740668779),
    ((0, 0, 1, 0), 2, 2, 0.31304313667691264),
    ((1, 1, 0, 0), 0, 1, 0.2375215627437346),
    ((0, 1, 0, 0), 0, 0, 0.3958604638855575),
    ((1, 1, 0, 0), 0, 5, -0.42942950746005015),
    ((1, 1, 1, 1), 2, 6, 0.5455840065263183),
    ((1, 0, 0, 0), 2, 3, 0.017115673418346744),
    ((1, 0, 0, 1), 1, 2, 0.13107989509472173),
    ((1, 0, 1, 1), 1, 7, -0.4551702244963054),
    ((1, 0, 0, 0), 0, 0, -0.5395094927947152),
    ((1, 1, 1, 1), 2, 2, 0.3670444298654235),
    ((0, 0, 0, 0), 2, 6, -0.24861863368491577),
    ((1, 1, 1, 1), 2, 3, 0.335177509798799),
    ((0, 1, 1, 1), 2, 5, 0.5656866202369394),
    ((1, 1, 1, 1), 2, 2, 0.47986932117951764),
    ((1, 0, 0, 0), 3, 4, 0.013402168869181441),
    ((0, 0, 0, 1), 2, 4, 0.10603947222843657),
    ((0, 1, 0, 0), 2, 0, 0.23018975518126505),
    ((0, 0, 0, 0), 1, 0, 0.19239685997822842),
    ((1, 0, 1, 0), 0, 6, -0.08507387388380139),
    ((1, 1, 0, 0), 0, 2, -0.22404873035871248),
    ((1, 0, 1, 0), 3, 0, -0.35908049548949106),
    ((1, 1, 0, 0), 0, 6, 0.07694552778636243),
    ((1, 0, 0, 0), 0, 5, -0.12989873801836926),
]


def eye3():
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(3)] for i in range(3)]


def zero3():
    return [[0.0j for _ in range(3)] for _ in range(3)]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(3)] for i in range(3)]


def scale(c, a):
    return [[c * a[i][j] for j in range(3)] for i in range(3)]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def dagger(a):
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]


def trace(a):
    return a[0][0] + a[1][1] + a[2][2]


def determinant(a):
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def frobenius(a):
    return math.sqrt(sum(abs(a[i][j]) ** 2 for i in range(3) for j in range(3)))


def matrix_exp(a, terms=50):
    norm = frobenius(a)
    squarings = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = scale(1.0 / (2**squarings), a)
    out = eye3()
    term = eye3()
    for k in range(1, terms + 1):
        term = scale(1.0 / k, mul(term, x))
        out = add(out, term)
    for _ in range(squarings):
        out = mul(out, out)
    return out


I = 1.0j
SQRT3 = math.sqrt(3.0)
GELL_MANN = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -I, 0], [I, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -I], [0, 0, 0], [I, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -I], [0, I, 0]],
    [[1 / SQRT3, 0, 0], [0, 1 / SQRT3, 0], [0, 0, -2 / SQRT3]],
]
GELL_MANN = [[[complex(v) for v in row] for row in m] for m in GELL_MANN]


def su3_rotation(generator, theta):
    return matrix_exp(scale(1.0j * theta, GELL_MANN[generator]))


def sites(dims):
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    yield (x, y, z, t)


def site_index(site, dims):
    x, y, z, t = site
    return (((x * dims[1] + y) * dims[2] + z) * dims[3] + t)


def shift(site, mu, step, dims):
    out = list(site)
    out[mu] = (out[mu] + step) % dims[mu]
    return tuple(out)


def identity_field(dims):
    return [eye3() for _ in range(math.prod(dims) * 4)]


def field_copy(field):
    return [[row[:] for row in matrix] for matrix in field]


def link(field, dims, site, mu):
    return field[site_index(site, dims) * 4 + mu]


def set_link(field, dims, site, mu, value):
    field[site_index(site, dims) * 4 + mu] = value


def oriented_link(field, dims, site, signed_direction):
    if signed_direction > 0:
        mu = signed_direction - 1
        return link(field, dims, site, mu), shift(site, mu, +1, dims)
    mu = -signed_direction - 1
    previous = shift(site, mu, -1, dims)
    return dagger(link(field, dims, previous, mu)), previous


def transporter(field, dims, start, directions):
    value = eye3()
    site = start
    for direction in directions:
        edge, site = oriented_link(field, dims, site, direction)
        value = mul(value, edge)
    return value, site


def plaquette(field, dims, site, mu, nu):
    dirs = [mu + 1, nu + 1, -(mu + 1), -(nu + 1)]
    return transporter(field, dims, site, dirs)[0]


def wilson_action(field, dims, beta=1.0):
    value = 0.0
    for site in sites(dims):
        for mu in range(4):
            for nu in range(mu + 1, 4):
                value += beta * (1.0 - trace(plaquette(field, dims, site, mu, nu)).real / 3.0)
    return value


def clover_sum(field, dims, site, mu, nu):
    pmu = mu + 1
    pnu = nu + 1
    loops = [
        [pmu, pnu, -pmu, -pnu],
        [pnu, -pmu, -pnu, pmu],
        [-pmu, -pnu, pmu, pnu],
        [-pnu, pmu, pnu, -pmu],
    ]
    result = zero3()
    for loop in loops:
        result = add(result, transporter(field, dims, site, loop)[0])
    return result


def clover_field_strength(field, dims, site, mu, nu):
    """Hermitian traceless F_munu=(C-C^dagger)/(8i), lattice spacing a=1."""
    clover = clover_sum(field, dims, site, mu, nu)
    f = scale(1.0 / (8.0j), sub(clover, dagger(clover)))
    singlet = trace(f) / 3.0
    for color in range(3):
        f[color][color] -= singlet
    return f


def topological_density(field, dims, site):
    f01 = clover_field_strength(field, dims, site, 0, 1)
    f02 = clover_field_strength(field, dims, site, 0, 2)
    f03 = clover_field_strength(field, dims, site, 0, 3)
    f12 = clover_field_strength(field, dims, site, 1, 2)
    f13 = clover_field_strength(field, dims, site, 1, 3)
    f23 = clover_field_strength(field, dims, site, 2, 3)
    contraction = (
        trace(mul(f01, f23)).real
        - trace(mul(f02, f13)).real
        + trace(mul(f03, f12)).real
    )
    return contraction / (4.0 * math.pi * math.pi)


def clover_topological_charge(field, dims):
    return sum(topological_density(field, dims, site) for site in sites(dims))


def deterministic_fixture(dims):
    field = identity_field(dims)
    for site, mu, generator, theta in FIXTURE_OPS:
        updated = mul(su3_rotation(generator, theta), link(field, dims, site, mu))
        set_link(field, dims, site, mu, updated)
    return field


def gauge_transform(field, dims):
    omegas = {}
    for site in sites(dims):
        k = site_index(site, dims)
        omegas[site] = mul(
            su3_rotation(k % 8, 0.04 * (k + 1)),
            su3_rotation((k + 3) % 8, -0.02 * (k + 1)),
        )
    transformed = identity_field(dims)
    for site in sites(dims):
        for mu in range(4):
            forward = shift(site, mu, +1, dims)
            value = mul(mul(omegas[site], link(field, dims, site, mu)), dagger(omegas[forward]))
            set_link(transformed, dims, site, mu, value)
    return transformed


def finite_difference_gradient(field, dims, epsilon=GRAD_EPS):
    gradient = {}
    for site in sites(dims):
        for mu in range(4):
            original = link(field, dims, site, mu)
            components = []
            for generator in GELL_MANN:
                plus = field_copy(field)
                minus = field_copy(field)
                set_link(plus, dims, site, mu, mul(matrix_exp(scale(+1.0j * epsilon, generator)), original))
                set_link(minus, dims, site, mu, mul(matrix_exp(scale(-1.0j * epsilon, generator)), original))
                derivative = (wilson_action(plus, dims) - wilson_action(minus, dims)) / (2.0 * epsilon)
                components.append(derivative)
            gradient[(site, mu)] = components
    return gradient


def finite_difference_wilson_flow_step(field, dims, dt=FLOW_DT, epsilon=GRAD_EPS):
    gradient = finite_difference_gradient(field, dims, epsilon)
    updated = field_copy(field)
    for (site, mu), components in gradient.items():
        hermitian = zero3()
        for coefficient, generator in zip(components, GELL_MANN):
            hermitian = add(hermitian, scale(coefficient, generator))
        rotation = matrix_exp(scale(-1.0j * dt, hermitian))
        set_link(updated, dims, site, mu, mul(rotation, link(field, dims, site, mu)))
    return updated


def max_matrix_error(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))


def max_field_error(a, b):
    return max(max_matrix_error(x, y) for x, y in zip(a, b))


def assert_close(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        raise AssertionError((label, actual, expected, tolerance))


def self_test():
    identity = identity_field(DIMS)
    assert_close(clover_topological_charge(identity, DIMS), 0.0, 1.0e-15, "identity Q")

    fixture = deterministic_fixture(DIMS)
    action_before = wilson_action(fixture, DIMS)
    charge_before = clover_topological_charge(fixture, DIMS)
    assert_close(action_before, 8.2223611910686, 2.0e-12, "fixture action")
    assert_close(charge_before, -0.00041713340026960005, 2.0e-15, "fixture Q")

    transformed = gauge_transform(fixture, DIMS)
    assert_close(wilson_action(transformed, DIMS), action_before, 1.0e-11, "gauge action")
    assert_close(clover_topological_charge(transformed, DIMS), charge_before, 1.0e-15, "gauge Q")

    flowed = finite_difference_wilson_flow_step(fixture, DIMS)
    action_after = wilson_action(flowed, DIMS)
    charge_after = clover_topological_charge(flowed, DIMS)
    assert_close(action_after, 8.131857099790445, 2.0e-10, "flowed action")
    assert_close(charge_after, -0.0004119078363020288, 2.0e-14, "flowed Q")
    if not action_after < action_before:
        raise AssertionError("Wilson-action gradient step did not decrease the action")

    flowed_transformed = finite_difference_wilson_flow_step(transformed, DIMS)
    transformed_flowed = gauge_transform(flowed, DIMS)
    covariance_error = max_field_error(flowed_transformed, transformed_flowed)
    if covariance_error > 3.0e-12:
        raise AssertionError(("flow gauge covariance", covariance_error))

    max_det_error = max(abs(determinant(matrix) - 1.0) for matrix in flowed)
    max_unitarity_error = max(
        max_matrix_error(mul(dagger(matrix), matrix), eye3()) for matrix in flowed
    )
    if max_det_error > 2.0e-12 or max_unitarity_error > 2.0e-12:
        raise AssertionError(("SU3 preservation", max_det_error, max_unitarity_error))

    print("ok")
    print(f"fixture_action={action_before:.17g}")
    print(f"fixture_q={charge_before:.17g}")
    print(f"flowed_action={action_after:.17g}")
    print(f"flowed_q={charge_after:.17g}")
    print(f"flow_action_delta={action_after-action_before:.17g}")
    print(f"gauge_covariance_max_error={covariance_error:.17g}")
    print(f"flowed_max_det_error={max_det_error:.17g}")
    print(f"flowed_max_unitarity_error={max_unitarity_error:.17g}")


if __name__ == "__main__":
    self_test()
