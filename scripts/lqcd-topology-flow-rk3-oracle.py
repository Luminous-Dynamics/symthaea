#!/usr/bin/env python3
"""Independent Lie-group RK3 Wilson-flow step-halving oracle.

Standard-library only. Imports no Symthaea/Rust code. This subject qualifies the
third-order Runge-Kutta stage composition on a compact nontrivial periodic 2^4
SU(3) fixture using the analytic Wilson-action staple gradient.

It is an integration-order oracle, not a physical flow-scale result.
"""

import math

DIMS = (2, 2, 2, 2)
TOTAL_FLOW_TIME = 0.008
DT_VALUES = (0.004, 0.002, 0.001, 0.0005)

FIXTURE_OPS = [
    ((0, 0, 1, 0), 0, 3, -0.5819354339098058),
    ((0, 1, 1, 1), 0, 2, -0.32393958624710995),
    ((1, 0, 1, 0), 1, 2, 0.19161237542499698),
    ((1, 1, 1, 1), 3, 5, 0.5114974521920254),
    ((1, 1, 1, 0), 1, 6, -0.4726273806149346),
    ((1, 0, 1, 1), 1, 3, 0.4032003002590304),
    ((0, 1, 0, 0), 2, 2, 0.3411282616087933),
    ((1, 1, 0, 0), 0, 5, -0.42942950746005015),
]

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
GELL_MANN = [[[complex(v) for v in row] for row in matrix] for matrix in GELL_MANN]


def eye():
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(3)] for i in range(3)]


def zero():
    return [[0.0j for _ in range(3)] for _ in range(3)]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


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


def matrix_exp(a, terms=50):
    norm = math.sqrt(sum(abs(v) ** 2 for row in a for v in row))
    squarings = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = scale(1.0 / (2**squarings), a)
    out = eye()
    term = eye()
    for k in range(1, terms + 1):
        term = scale(1.0 / k, mul(term, x))
        out = add(out, term)
    for _ in range(squarings):
        out = mul(out, out)
    return out


def rotation(generator, theta):
    return matrix_exp(scale(1.0j * theta, GELL_MANN[generator]))


def sites():
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    yield (x, y, z, t)


def site_index(site):
    x, y, z, t = site
    return (((x * 2 + y) * 2 + z) * 2 + t)


def shift(site, mu, step):
    out = list(site)
    out[mu] = (out[mu] + step) % 2
    return tuple(out)


def identity_field():
    return [eye() for _ in range(16 * 4)]


def field_copy(field):
    return [[row[:] for row in matrix] for matrix in field]


def link(field, site, mu):
    return field[site_index(site) * 4 + mu]


def set_link(field, site, mu, value):
    field[site_index(site) * 4 + mu] = value


def fixture():
    field = identity_field()
    for site, mu, generator, theta in FIXTURE_OPS:
        set_link(field, site, mu, mul(rotation(generator, theta), link(field, site, mu)))
    return field


def plaquette(field, site, mu, nu):
    x_mu = shift(site, mu, 1)
    x_nu = shift(site, nu, 1)
    return mul(
        mul(
            mul(link(field, site, mu), link(field, x_mu, nu)),
            dagger(link(field, x_nu, mu)),
        ),
        dagger(link(field, site, nu)),
    )


def wilson_action(field):
    return sum(
        1.0 - trace(plaquette(field, site, mu, nu)).real / 3.0
        for site in sites()
        for mu in range(4)
        for nu in range(mu + 1, 4)
    )


def staple(field, site, mu):
    result = zero()
    x_plus_mu = shift(site, mu, 1)
    for nu in range(4):
        if nu == mu:
            continue
        x_plus_nu = shift(site, nu, 1)
        forward = mul(
            mul(link(field, x_plus_mu, nu), dagger(link(field, x_plus_nu, mu))),
            dagger(link(field, site, nu)),
        )
        result = add(result, forward)

        x_minus_nu = shift(site, nu, -1)
        x_minus_nu_plus_mu = shift(x_minus_nu, mu, 1)
        backward = mul(
            mul(
                dagger(link(field, x_minus_nu_plus_mu, nu)),
                dagger(link(field, x_minus_nu, mu)),
            ),
            link(field, x_minus_nu, nu),
        )
        result = add(result, backward)
    return result


def gradient(field):
    out = {}
    for site in sites():
        for mu in range(4):
            x = mul(link(field, site, mu), staple(field, site, mu))
            out[(site, mu)] = [
                trace(mul(generator, x)).imag / 3.0 for generator in GELL_MANN
            ]
    return out


def combine(*terms):
    result = [0.0] * 8
    for coefficient, components in terms:
        for a in range(8):
            result[a] += coefficient * components[a]
    return result


def apply_left_algebra(base, algebra):
    updated = field_copy(base)
    for (site, mu), components in algebra.items():
        hermitian = zero()
        for coefficient, generator in zip(components, GELL_MANN):
            hermitian = add(hermitian, scale(coefficient, generator))
        update = matrix_exp(scale(1.0j, hermitian))
        set_link(updated, site, mu, mul(update, link(base, site, mu)))
    return updated


def rk3_step(field, dt):
    g0 = gradient(field)
    z0 = {key: [-dt * value for value in values] for key, values in g0.items()}

    w1 = apply_left_algebra(
        field, {key: [0.25 * value for value in z0[key]] for key in z0}
    )

    g1 = gradient(w1)
    z1 = {key: [-dt * value for value in values] for key, values in g1.items()}
    w2 = apply_left_algebra(
        w1,
        {
            key: combine((8.0 / 9.0, z1[key]), (-17.0 / 36.0, z0[key]))
            for key in z0
        },
    )

    g2 = gradient(w2)
    z2 = {key: [-dt * value for value in values] for key, values in g2.items()}
    return apply_left_algebra(
        w2,
        {
            key: combine(
                (3.0 / 4.0, z2[key]),
                (-8.0 / 9.0, z1[key]),
                (17.0 / 36.0, z0[key]),
            )
            for key in z0
        },
    )


def oriented_link(field, site, direction):
    if direction > 0:
        mu = direction - 1
        return link(field, site, mu), shift(site, mu, 1)
    mu = -direction - 1
    previous = shift(site, mu, -1)
    return dagger(link(field, previous, mu)), previous


def transporter(field, start, directions):
    product = eye()
    site = start
    for direction in directions:
        edge, site = oriented_link(field, site, direction)
        product = mul(product, edge)
    return product


def clover_field_strength(field, site, mu, nu):
    pmu = mu + 1
    pnu = nu + 1
    loops = [
        [pmu, pnu, -pmu, -pnu],
        [pnu, -pmu, -pnu, pmu],
        [-pmu, -pnu, pmu, pnu],
        [-pnu, pmu, pnu, -pmu],
    ]
    clover = zero()
    for path in loops:
        clover = add(clover, transporter(field, site, path))
    clover_dagger = dagger(clover)
    antihermitian = [
        [clover[i][j] - clover_dagger[i][j] for j in range(3)] for i in range(3)
    ]
    field_strength = scale(1.0 / (8.0j), antihermitian)
    singlet = trace(field_strength) / 3.0
    for color in range(3):
        field_strength[color][color] -= singlet
    return field_strength


def topological_charge(field):
    charge = 0.0
    for site in sites():
        f01 = clover_field_strength(field, site, 0, 1)
        f02 = clover_field_strength(field, site, 0, 2)
        f03 = clover_field_strength(field, site, 0, 3)
        f12 = clover_field_strength(field, site, 1, 2)
        f13 = clover_field_strength(field, site, 1, 3)
        f23 = clover_field_strength(field, site, 2, 3)
        contraction = (
            trace(mul(f01, f23)).real
            - trace(mul(f02, f13)).real
            + trace(mul(f03, f12)).real
        )
        charge += contraction / (4.0 * math.pi * math.pi)
    return charge


def max_su3_errors(field):
    max_unitarity = 0.0
    max_determinant = 0.0
    identity = eye()
    for matrix in field:
        gram = mul(dagger(matrix), matrix)
        for i in range(3):
            for j in range(3):
                max_unitarity = max(
                    max_unitarity, abs(gram[i][j] - identity[i][j])
                )
        max_determinant = max(max_determinant, abs(determinant(matrix) - 1.0))
    return max_unitarity, max_determinant


def run(dt):
    steps_exact = TOTAL_FLOW_TIME / dt
    steps = round(steps_exact)
    if abs(steps_exact - steps) > 1.0e-12:
        raise AssertionError("dt must divide total flow time")
    field = fixture()
    action_history = [wilson_action(field)]
    for _ in range(steps):
        field = rk3_step(field, dt)
        action_history.append(wilson_action(field))
    if any(
        action_history[i + 1] > action_history[i] + 1.0e-12
        for i in range(len(action_history) - 1)
    ):
        raise AssertionError(("non-monotone action", dt, action_history))
    unitarity, determinant_error = max_su3_errors(field)
    return {
        "dt": dt,
        "steps": steps,
        "action": action_history[-1],
        "q": topological_charge(field),
        "unitarity": unitarity,
        "determinant": determinant_error,
    }


def self_test():
    results = [run(dt) for dt in DT_VALUES]
    expected = [
        (0.004, 2, 2.511434591009796, 1.02889044906806e-05),
        (0.002, 4, 2.5114347854282113, 1.0288905158564748e-05),
        (0.001, 8, 2.5114348093973327, 1.0288905241074472e-05),
        (0.0005, 16, 2.5114348123728787, 1.0288905251328004e-05),
    ]
    for result, target in zip(results, expected):
        dt, steps, action, q = target
        if result["dt"] != dt or result["steps"] != steps:
            raise AssertionError(("schedule", result, target))
        if abs(result["action"] - action) > 3.0e-13:
            raise AssertionError(("action", result["action"], action))
        if abs(result["q"] - q) > 3.0e-18:
            raise AssertionError(("q", result["q"], q))
        if result["unitarity"] > 1.0e-13 or result["determinant"] > 1.0e-13:
            raise AssertionError(("SU3 drift", result))

    action_diffs = [
        abs(results[i]["action"] - results[i + 1]["action"])
        for i in range(len(results) - 1)
    ]
    q_diffs = [
        abs(results[i]["q"] - results[i + 1]["q"])
        for i in range(len(results) - 1)
    ]
    action_ratios = [
        action_diffs[i] / action_diffs[i + 1] for i in range(len(action_diffs) - 1)
    ]
    q_ratios = [q_diffs[i] / q_diffs[i + 1] for i in range(len(q_diffs) - 1)]
    if not all(7.5 < ratio < 8.5 for ratio in action_ratios + q_ratios):
        raise AssertionError(("not third-order", action_ratios, q_ratios))

    print("ok")
    print(f"total_flow_time={TOTAL_FLOW_TIME:.17g}")
    for result in results:
        print(
            f"dt={result['dt']:.17g} steps={result['steps']} "
            f"action={result['action']:.17g} q={result['q']:.17g} "
            f"max_unitarity={result['unitarity']:.17g} "
            f"max_det_error={result['determinant']:.17g}"
        )
    print("action_step_halving_ratios=" + ",".join(f"{x:.17g}" for x in action_ratios))
    print("q_step_halving_ratios=" + ",".join(f"{x:.17g}" for x in q_ratios))


if __name__ == "__main__":
    self_test()
