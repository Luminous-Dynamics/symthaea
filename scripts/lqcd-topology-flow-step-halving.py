#!/usr/bin/env python3
"""Independent fixed-flow-time step-halving study for the LQCD-017D analytic Wilson flow.

Standard-library only. Imports no Symthaea/Rust code. This subject qualifies the
first-order Lie-Euler integration behavior on the frozen explicit 2^4 fixture;
it does not define a production flow-time scale.
"""

import math

DIMS = (2, 2, 2, 2)
TOTAL_FLOW_TIME = 0.004
DT_VALUES = (0.002, 0.001, 0.0005, 0.00025)

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

I = 1j
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
GELL_MANN = [[[complex(value) for value in row] for row in matrix] for matrix in GELL_MANN]


def identity():
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


def matrix_exp(a, terms=50):
    norm = math.sqrt(sum(abs(value) ** 2 for row in a for value in row))
    squarings = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = scale(1.0 / (2**squarings), a)
    out = identity()
    term = identity()
    for k in range(1, terms + 1):
        term = scale(1.0 / k, mul(term, x))
        out = add(out, term)
    for _ in range(squarings):
        out = mul(out, out)
    return out


def rotation(generator, theta):
    return matrix_exp(scale(1j * theta, GELL_MANN[generator]))


def sites():
    for x in range(DIMS[0]):
        for y in range(DIMS[1]):
            for z in range(DIMS[2]):
                for t in range(DIMS[3]):
                    yield (x, y, z, t)


def site_index(site):
    x, y, z, t = site
    return (((x * DIMS[1] + y) * DIMS[2] + z) * DIMS[3] + t)


def shift(site, mu, step):
    out = list(site)
    out[mu] = (out[mu] + step) % DIMS[mu]
    return tuple(out)


def identity_field():
    return [identity() for _ in range(math.prod(DIMS) * 4)]


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


def oriented_link(field, site, direction):
    if direction > 0:
        mu = direction - 1
        return link(field, site, mu), shift(site, mu, 1)
    mu = -direction - 1
    previous = shift(site, mu, -1)
    return dagger(link(field, previous, mu)), previous


def transporter(field, start, directions):
    value = identity()
    site = start
    for direction in directions:
        edge, site = oriented_link(field, site, direction)
        value = mul(value, edge)
    return value


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


def flow_step(field, dt):
    gradients = {}
    for site in sites():
        for mu in range(4):
            x = mul(link(field, site, mu), staple(field, site, mu))
            gradients[(site, mu)] = [
                trace(mul(generator, x)).imag / 3.0 for generator in GELL_MANN
            ]

    updated = field_copy(field)
    for (site, mu), components in gradients.items():
        hermitian = zero()
        for coefficient, generator in zip(components, GELL_MANN):
            hermitian = add(hermitian, scale(coefficient, generator))
        rotation_matrix = matrix_exp(scale(-1j * dt, hermitian))
        set_link(updated, site, mu, mul(rotation_matrix, link(field, site, mu)))
    return updated


def run(dt):
    steps_exact = TOTAL_FLOW_TIME / dt
    steps = round(steps_exact)
    if abs(steps - steps_exact) > 1.0e-12:
        raise ValueError("dt must divide the frozen total flow time exactly")
    field = fixture()
    action_history = [wilson_action(field)]
    for _ in range(steps):
        field = flow_step(field, dt)
        action_history.append(wilson_action(field))
    if any(
        action_history[index + 1] > action_history[index] + 1.0e-12
        for index in range(len(action_history) - 1)
    ):
        raise AssertionError(("non-monotonic action", dt, action_history))
    return {
        "dt": dt,
        "steps": steps,
        "action": action_history[-1],
        "q": topological_charge(field),
        "action_drop": action_history[-1] - action_history[0],
    }


def self_test():
    results = [run(dt) for dt in DT_VALUES]
    expected = [
        (0.002, 2, 7.865198193066977, -0.0003964908549886643),
        (0.001, 4, 7.86614167020082, -0.00039653225241561587),
        (0.0005, 8, 7.86661084741866, -0.0003965528746344892),
        (0.00025, 16, 7.866844800892616, -0.00039656316670853977),
    ]
    for result, target in zip(results, expected):
        dt, steps, action, q = target
        if result["dt"] != dt or result["steps"] != steps:
            raise AssertionError(("schedule mismatch", result, target))
        if abs(result["action"] - action) > 2.0e-12:
            raise AssertionError(("action mismatch", result["action"], action))
        if abs(result["q"] - q) > 2.0e-15:
            raise AssertionError(("Q mismatch", result["q"], q))

    action_differences = [
        abs(results[i]["action"] - results[i + 1]["action"])
        for i in range(len(results) - 1)
    ]
    q_differences = [
        abs(results[i]["q"] - results[i + 1]["q"])
        for i in range(len(results) - 1)
    ]
    action_ratios = [
        action_differences[i] / action_differences[i + 1]
        for i in range(len(action_differences) - 1)
    ]
    q_ratios = [
        q_differences[i] / q_differences[i + 1]
        for i in range(len(q_differences) - 1)
    ]

    if not all(1.9 < ratio < 2.1 for ratio in action_ratios + q_ratios):
        raise AssertionError(("not first-order step-halving behavior", action_ratios, q_ratios))

    print("ok")
    print(f"total_flow_time={TOTAL_FLOW_TIME:.17g}")
    for result in results:
        print(
            f"dt={result['dt']:.17g} steps={result['steps']} "
            f"action={result['action']:.17g} q={result['q']:.17g} "
            f"action_drop={result['action_drop']:.17g}"
        )
    print("action_step_halving_ratios=" + ",".join(f"{x:.17g}" for x in action_ratios))
    print("q_step_halving_ratios=" + ",".join(f"{x:.17g}" for x in q_ratios))


if __name__ == "__main__":
    self_test()
