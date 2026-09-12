#!/usr/bin/env python3
"""Independent tiny pure-SU(3) ensemble pilot for LQCD-016.

Standard-library only. Imports no Symthaea/Rust code. This is a qualification
oracle for algorithm/schedule semantics, not a physical lattice-QCD result.
"""

import argparse
import math
import statistics
import struct

MASK32 = 0xFFFF_FFFF
PAIRS = ((0, 1), (0, 2), (1, 2))
DOMAIN_TRANSITION = 1
DOMAIN_INITIALIZATION = 4
SEED = bytes([0x5A]) * 32
DIMS = (2, 2, 1, 2)
BETA = 5.7
ENSEMBLE_SLOT = 17


def rotl32(x, n):
    return ((x << n) & MASK32) | (x >> (32 - n))


def quarter_round(x, a, b, c, d):
    x[a] = (x[a] + x[b]) & MASK32
    x[d] ^= x[a]
    x[d] = rotl32(x[d], 16)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] ^= x[c]
    x[b] = rotl32(x[b], 12)
    x[a] = (x[a] + x[b]) & MASK32
    x[d] ^= x[a]
    x[d] = rotl32(x[d], 8)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] ^= x[c]
    x[b] = rotl32(x[b], 7)


def chacha8_block(seed, counter, stream):
    constants = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
    key = list(struct.unpack("<8I", seed))
    state = constants + key + [
        counter & MASK32,
        (counter >> 32) & MASK32,
        stream & MASK32,
        (stream >> 32) & MASK32,
    ]
    work = state.copy()
    for _ in range(4):
        quarter_round(work, 0, 4, 8, 12)
        quarter_round(work, 1, 5, 9, 13)
        quarter_round(work, 2, 6, 10, 14)
        quarter_round(work, 3, 7, 11, 15)
        quarter_round(work, 0, 5, 10, 15)
        quarter_round(work, 1, 6, 11, 12)
        quarter_round(work, 2, 7, 8, 13)
        quarter_round(work, 3, 4, 9, 14)
    return struct.pack("<16I", *[((work[i] + state[i]) & MASK32) for i in range(16)])


def pack_stream_id(domain, ensemble_slot, replica, rank=0):
    if not 0 <= ensemble_slot < (1 << 24):
        raise ValueError("ensemble_slot must fit 24 bits")
    return (domain << 56) | (ensemble_slot << 32) | (replica << 16) | rank


class ChaCha8Stream:
    def __init__(self, seed, stream):
        self.seed = seed
        self.stream = stream
        self.counter = 0
        self.buffer = b""
        self.cursor = 0

    def _bytes(self, count):
        out = bytearray()
        while len(out) < count:
            if self.cursor >= len(self.buffer):
                self.buffer = chacha8_block(self.seed, self.counter, self.stream)
                self.counter = (self.counter + 1) & ((1 << 64) - 1)
                self.cursor = 0
            take = min(count - len(out), len(self.buffer) - self.cursor)
            out.extend(self.buffer[self.cursor:self.cursor + take])
            self.cursor += take
        return bytes(out)

    def next_u64(self):
        return struct.unpack("<Q", self._bytes(8))[0]

    def open01(self):
        k = (self.next_u64() >> 12) + 1
        return k / ((1 << 52) + 1)


def identity():
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(3)] for i in range(3)]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def dagger(a):
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]


def trace(a):
    return a[0][0] + a[1][1] + a[2][2]


def embedded_su2(pair, axis, angle):
    norm = math.sqrt(sum(value * value for value in axis))
    nx, ny, nz = (value / norm for value in axis)
    c, s = math.cos(angle), math.sin(angle)
    block = (
        c + 1j * s * nz,
        s * ny + 1j * s * nx,
        -s * ny + 1j * s * nx,
        c - 1j * s * nz,
    )
    out = identity()
    i, j = pair
    out[i][i], out[i][j], out[j][i], out[j][j] = block
    return out


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
    return [identity() for _ in range(math.prod(dims) * 4)]


def link(field, dims, site, mu):
    return field[site_index(site, dims) * 4 + mu]


def set_link(field, dims, site, mu, value):
    field[site_index(site, dims) * 4 + mu] = value


def plaquette(field, dims, site, mu, nu):
    x_mu = shift(site, mu, 1, dims)
    x_nu = shift(site, nu, 1, dims)
    return mul(
        mul(
            mul(link(field, dims, site, mu), link(field, dims, x_mu, nu)),
            dagger(link(field, dims, x_nu, mu)),
        ),
        dagger(link(field, dims, site, nu)),
    )


def average_plaquette(field, dims):
    values = [
        trace(plaquette(field, dims, site, mu, nu)).real / 3.0
        for site in sites(dims)
        for mu in range(4)
        for nu in range(mu + 1, 4)
    ]
    return sum(values) / len(values)


def affected_plaquettes(site, mu, dims):
    result = set()
    for nu in range(4):
        if nu == mu:
            continue
        a, b = sorted((mu, nu))
        result.add((site, a, b))
        result.add((shift(site, nu, -1, dims), a, b))
    return sorted(result)


def affected_action(field, dims, site, mu, beta):
    return sum(
        beta * (1.0 - trace(plaquette(field, dims, base, a, b)).real / 3.0)
        for base, a, b in affected_plaquettes(site, mu, dims)
    )


def draw_rotation(source, pair, max_angle):
    uz, uphi, uangle = source.open01(), source.open01(), source.open01()
    z = 2.0 * uz - 1.0
    phi = 2.0 * math.pi * uphi
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    axis = (radial * math.cos(phi), radial * math.sin(phi), z)
    angle = max_angle * (2.0 * uangle - 1.0)
    return embedded_su2(pair, axis, angle)


def metropolis_sweep(field, dims, beta, max_angle, source):
    attempted = accepted = 0
    for site in sites(dims):
        for mu in range(4):
            for pair in PAIRS:
                before = affected_action(field, dims, site, mu, beta)
                old = link(field, dims, site, mu)
                set_link(field, dims, site, mu, mul(draw_rotation(source, pair, max_angle), old))
                delta = affected_action(field, dims, site, mu, beta) - before
                probability = 1.0 if delta <= 0.0 else math.exp(-delta)
                attempted += 1
                if source.open01() < probability:
                    accepted += 1
                else:
                    set_link(field, dims, site, mu, old)
    return accepted / attempted


def make_disordered_start(replica, rounds=2, max_angle=math.pi):
    field = identity_field(DIMS)
    source = ChaCha8Stream(
        SEED,
        pack_stream_id(DOMAIN_INITIALIZATION, ENSEMBLE_SLOT, replica),
    )
    for _ in range(rounds):
        for site in sites(DIMS):
            for mu in range(4):
                for pair in PAIRS:
                    old = link(field, DIMS, site, mu)
                    set_link(field, DIMS, site, mu, mul(draw_rotation(source, pair, max_angle), old))
    return field


def run_chain(start, replica, max_angle, burn_in, stride, measurements):
    field = identity_field(DIMS) if start == "cold" else make_disordered_start(replica)
    initial_plaquette = average_plaquette(field, DIMS)
    source = ChaCha8Stream(
        SEED,
        pack_stream_id(DOMAIN_TRANSITION, ENSEMBLE_SLOT, replica),
    )
    keep = {burn_in + stride * i for i in range(1, measurements + 1)}
    final_sweep = burn_in + stride * measurements
    plaquettes, acceptance = [], []
    for sweep in range(1, final_sweep + 1):
        rate = metropolis_sweep(field, DIMS, BETA, max_angle, source)
        if sweep in keep:
            plaquettes.append(average_plaquette(field, DIMS))
            acceptance.append(rate)
    return initial_plaquette, plaquettes, acceptance


def split_rhat(chains):
    count = len(chains[0])
    if count % 2:
        raise ValueError("split-R-hat fixture requires even chain length")
    if any(len(chain) != count for chain in chains):
        raise ValueError("chains must have equal lengths")
    n = count // 2
    split = [chain[:n] for chain in chains] + [chain[n:] for chain in chains]
    means = [statistics.mean(chain) for chain in split]
    variances = [statistics.variance(chain) for chain in split]
    m = len(split)
    within = statistics.mean(variances)
    between = n * sum((value - statistics.mean(means)) ** 2 for value in means) / (m - 1)
    estimate = ((n - 1) / n) * within + between / n
    return math.sqrt(estimate / within)


def pilot(max_angle, burn_in, stride, measurements):
    cold = run_chain("cold", 0, max_angle, burn_in, stride, measurements)
    disordered = run_chain("disordered", 1, max_angle, burn_in, stride, measurements)
    return {
        "max_angle": max_angle,
        "burn_in": burn_in,
        "stride": stride,
        "measurements": measurements,
        "cold_stream": pack_stream_id(DOMAIN_TRANSITION, ENSEMBLE_SLOT, 0),
        "disordered_stream": pack_stream_id(DOMAIN_TRANSITION, ENSEMBLE_SLOT, 1),
        "cold_initial": cold[0],
        "disordered_initial": disordered[0],
        "cold_mean": statistics.mean(cold[1]),
        "disordered_mean": statistics.mean(disordered[1]),
        "mean_difference": statistics.mean(cold[1]) - statistics.mean(disordered[1]),
        "cold_mean_acceptance": statistics.mean(cold[2]),
        "disordered_mean_acceptance": statistics.mean(disordered[2]),
        "split_rhat": split_rhat([cold[1], disordered[1]]),
    }


def self_test():
    zero_vector = bytes.fromhex(
        "3e00ef2f895f40d67f5bb8e81f09a5a1"
        "2c840ec3ce9a7f3b181be188ef711a1e"
        "984ce172b9216f419f445367456d5619"
        "314a42a3da86b001387bfdb80e0cfe42"
    )
    if chacha8_block(bytes(32), 0, 0) != zero_vector:
        raise AssertionError("ChaCha8 reference vector mismatch")

    short = pilot(0.18, 20, 2, 10)
    longer = pilot(0.5, 200, 10, 20)

    expected = {
        "short_cold_mean": 0.733691306815339,
        "short_disordered_mean": 0.2632210271828042,
        "short_rhat": 11.586746995780686,
        "long_cold_mean": 0.6023654725827776,
        "long_disordered_mean": 0.6392231534844031,
        "long_rhat": 1.2271025266120743,
    }
    actual = {
        "short_cold_mean": short["cold_mean"],
        "short_disordered_mean": short["disordered_mean"],
        "short_rhat": short["split_rhat"],
        "long_cold_mean": longer["cold_mean"],
        "long_disordered_mean": longer["disordered_mean"],
        "long_rhat": longer["split_rhat"],
    }
    for key, target in expected.items():
        if abs(actual[key] - target) > 1e-12:
            raise AssertionError((key, actual[key], target))

    print("ok")
    for name, result in (("short", short), ("longer", longer)):
        print(
            f"{name}: angle={result['max_angle']:.17g} burn={result['burn_in']} "
            f"stride={result['stride']} n={result['measurements']} "
            f"cold_stream=0x{result['cold_stream']:016x} "
            f"disordered_stream=0x{result['disordered_stream']:016x}"
        )
        print(
            f"{name}: cold_initial={result['cold_initial']:.17g} "
            f"disordered_initial={result['disordered_initial']:.17g}"
        )
        print(
            f"{name}: cold_mean={result['cold_mean']:.17g} "
            f"disordered_mean={result['disordered_mean']:.17g} "
            f"difference={result['mean_difference']:.17g}"
        )
        print(
            f"{name}: cold_acceptance={result['cold_mean_acceptance']:.17g} "
            f"disordered_acceptance={result['disordered_mean_acceptance']:.17g} "
            f"split_rhat={result['split_rhat']:.17g}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported; this is a qualification pilot")
    self_test()


if __name__ == "__main__":
    main()
