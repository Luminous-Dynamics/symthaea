#!/usr/bin/env python3
"""Independent SU(2) heat-bath target-density oracle.

This standard-library-only program samples the exact scalar SU(2) heat-bath
conditional density using a generic rejection sampler, not Kennedy-Pendleton.
It therefore provides an algorithmically independent distribution oracle for a
later optimized Kennedy-Pendleton implementation.
"""
import argparse
import math

MASK = (1 << 64) - 1
BINS = 20
QUAD_STEPS = 200_000
SAMPLES = 100_000


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def open01(self):
        k = (self.next_u64() >> 11) + 1
        return k / ((1 << 53) + 1)


def target_weight(a0, alpha):
    return math.sqrt(max(0.0, 1.0 - a0 * a0)) * math.exp(alpha * a0)


def reference_quadrature(alpha):
    # Midpoint quadrature avoids the integrable endpoint square-root cusp.
    h = 2.0 / QUAD_STEPS
    z = m1 = m2 = 0.0
    bins = [0.0] * BINS
    for i in range(QUAD_STEPS):
        a0 = -1.0 + (i + 0.5) * h
        w = target_weight(a0, alpha)
        z += w
        m1 += a0 * w
        m2 += a0 * a0 * w
        b = min(BINS - 1, int((a0 + 1.0) * 0.5 * BINS))
        bins[b] += w
    probs = [x / z for x in bins]
    return m1 / z, m2 / z, probs


def draw_a0(rng, alpha):
    attempts = 0
    while True:
        attempts += 1
        a0 = 2.0 * rng.open01() - 1.0
        # Envelope uses exp(alpha), so acceptance is <= 1 for alpha >= 0.
        acceptance = math.sqrt(max(0.0, 1.0 - a0 * a0)) * math.exp(alpha * (a0 - 1.0))
        if rng.open01() < acceptance:
            return a0, attempts


def draw_unit_vector(rng):
    z = 2.0 * rng.open01() - 1.0
    phi = 2.0 * math.pi * rng.open01()
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(phi), r * math.sin(phi), z)


def draw_quaternion(rng, alpha):
    a0, attempts = draw_a0(rng, alpha)
    radius = math.sqrt(max(0.0, 1.0 - a0 * a0))
    n = draw_unit_vector(rng)
    return (a0, radius * n[0], radius * n[1], radius * n[2]), attempts


def sample_fixture(alpha, seed):
    ref_mean, ref_second, expected_bins = reference_quadrature(alpha)
    rng = SplitMix64(seed)
    sum_a0 = sum_a02 = 0.0
    vector_sums = [0.0, 0.0, 0.0]
    counts = [0] * BINS
    attempts = 0
    max_norm_error = 0.0
    for _ in range(SAMPLES):
        q, used = draw_quaternion(rng, alpha)
        attempts += used
        a0, a1, a2, a3 = q
        sum_a0 += a0
        sum_a02 += a0 * a0
        vector_sums[0] += a1
        vector_sums[1] += a2
        vector_sums[2] += a3
        max_norm_error = max(max_norm_error, abs(sum(x * x for x in q) - 1.0))
        b = min(BINS - 1, int((a0 + 1.0) * 0.5 * BINS))
        counts[b] += 1

    mean = sum_a0 / SAMPLES
    second = sum_a02 / SAMPLES
    variance = max(0.0, second - mean * mean)
    mean_se = math.sqrt(variance / SAMPLES)
    second_values_var = max(0.0, second * (1.0 - second))
    second_se_bound = math.sqrt(second_values_var / SAMPLES)
    max_bin_z = 0.0
    for count, p in zip(counts, expected_bins):
        expected = SAMPLES * p
        sigma = math.sqrt(max(1e-300, SAMPLES * p * (1.0 - p)))
        max_bin_z = max(max_bin_z, abs(count - expected) / sigma)

    mean_z = abs(mean - ref_mean) / mean_se
    second_z_bound = abs(second - ref_second) / max(second_se_bound, 1e-15)
    vector_means = [x / SAMPLES for x in vector_sums]

    if max_norm_error > 2e-15:
        raise AssertionError(("unit quaternion", max_norm_error))
    if mean_z > 5.0 or second_z_bound > 5.0 or max_bin_z > 5.5:
        raise AssertionError((alpha, mean_z, second_z_bound, max_bin_z))
    if max(abs(x) for x in vector_means) > 0.01:
        raise AssertionError(("direction bias", alpha, vector_means))

    return {
        "alpha": alpha,
        "reference_mean": ref_mean,
        "sample_mean": mean,
        "mean_z": mean_z,
        "reference_second": ref_second,
        "sample_second": second,
        "second_z_bound": second_z_bound,
        "max_bin_z": max_bin_z,
        "mean_attempts": attempts / SAMPLES,
        "vector_means": vector_means,
        "max_norm_error": max_norm_error,
    }


def self_test():
    zero_mean, zero_second, _ = reference_quadrature(0.0)
    if abs(zero_mean) > 1e-12 or abs(zero_second - 0.25) > 2e-7:
        raise AssertionError((zero_mean, zero_second))

    fixtures = [
        sample_fixture(1.5, 0x1540_160B_0000_0001),
        sample_fixture(5.0, 0x1540_160B_0000_0002),
    ]
    print("ok")
    print(f"haar_reference_mean={zero_mean:.17g}")
    print(f"haar_reference_second={zero_second:.17g}")
    for f in fixtures:
        tag = str(f["alpha"]).replace(".", "p")
        print(f"alpha_{tag}_reference_mean={f['reference_mean']:.17g}")
        print(f"alpha_{tag}_sample_mean={f['sample_mean']:.17g}")
        print(f"alpha_{tag}_mean_z={f['mean_z']:.17g}")
        print(f"alpha_{tag}_reference_second={f['reference_second']:.17g}")
        print(f"alpha_{tag}_sample_second={f['sample_second']:.17g}")
        print(f"alpha_{tag}_second_z_bound={f['second_z_bound']:.17g}")
        print(f"alpha_{tag}_max_bin_z={f['max_bin_z']:.17g}")
        print(f"alpha_{tag}_mean_attempts={f['mean_attempts']:.17g}")
        print("alpha_%s_vector_means=%s" % (tag, ",".join(f"{x:.17g}" for x in f["vector_means"])))
        print(f"alpha_{tag}_max_norm_error={f['max_norm_error']:.17g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported; this is a qualification oracle")
    self_test()


if __name__ == "__main__":
    main()
