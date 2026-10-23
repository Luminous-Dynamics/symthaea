#!/usr/bin/env python3
"""Independent joint blocked-jackknife oracle for correlated gradient-flow scales.

Standard-library only. Each row is one retained configuration trajectory across
all flow times. Valid jackknife replicates delete the SAME complete row block at
all flow times. The deliberately broken negative control deletes different
blocks per column and exists only to prove why cross-flow covariance must be
preserved.
"""

import math

TIMES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BLOCK_SIZE = 3
T0_TARGET = 0.30
W0_TARGET = 0.32

# Frozen dimensionless F_i(t)=t^2 E_i(t) trajectories.
# 12 retained configurations = four contiguous blocks of three.
TRAJECTORIES = (
    (0.1120, 0.1921, 0.2542, 0.3343, 0.4064, 0.4865),
    (0.1140, 0.1901, 0.2562, 0.3323, 0.4084, 0.4845),
    (0.1160, 0.1881, 0.2582, 0.3303, 0.4104, 0.4825),
    (0.1160, 0.1987, 0.2634, 0.3461, 0.4208, 0.5035),
    (0.1180, 0.1967, 0.2654, 0.3441, 0.4228, 0.5015),
    (0.1200, 0.1947, 0.2674, 0.3421, 0.4248, 0.4995),
    (0.1198, 0.2056, 0.2734, 0.3592, 0.4370, 0.5228),
    (0.1218, 0.2036, 0.2754, 0.3572, 0.4390, 0.5208),
    (0.1238, 0.2016, 0.2774, 0.3552, 0.4410, 0.5188),
    (0.1242, 0.2116, 0.2810, 0.3684, 0.4478, 0.5352),
    (0.1262, 0.2096, 0.2830, 0.3664, 0.4498, 0.5332),
    (0.1282, 0.2076, 0.2850, 0.3644, 0.4518, 0.5312),
)


def validate_trajectories(times, rows, block_size):
    if len(times) < 4:
        raise ValueError("need at least four flow times for joint t0/w0 study")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(rows) < 2 * block_size:
        raise ValueError("need at least two complete blocks")
    if len(rows) % block_size != 0:
        raise ValueError("partial trailing block is forbidden")
    if any(not math.isfinite(t) or t <= 0.0 for t in times):
        raise ValueError("flow times must be positive and finite")
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("flow times must be strictly increasing")
    for row in rows:
        if len(row) != len(times):
            raise ValueError("trajectory width mismatch")
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("trajectory values must be finite and nonnegative")


def column_means(rows):
    width = len(rows[0])
    return tuple(sum(row[j] for row in rows) / len(rows) for j in range(width))


def unique_linear_crossing(xs, ys, target):
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target must be positive and finite")
    exact = [i for i, y in enumerate(ys) if y == target]
    if len(exact) == 1:
        return xs[exact[0]]
    if len(exact) > 1:
        raise ValueError("ambiguous exact crossings")
    hits = []
    for i in range(len(xs) - 1):
        left = ys[i] - target
        right = ys[i + 1] - target
        if left * right < 0.0:
            fraction = (target - ys[i]) / (ys[i + 1] - ys[i])
            hits.append(xs[i] + fraction * (xs[i + 1] - xs[i]))
    if len(hits) != 1:
        raise ValueError("crossing must be unique")
    return hits[0]


def t0_like(times, mean_dimensionless, target):
    return unique_linear_crossing(times, mean_dimensionless, target)


def w0_like(times, mean_dimensionless, target):
    response_times = []
    response = []
    for i in range(1, len(times) - 1):
        derivative = (
            (mean_dimensionless[i + 1] - mean_dimensionless[i - 1])
            / (times[i + 1] - times[i - 1])
        )
        value = times[i] * derivative
        if not math.isfinite(value):
            raise ValueError("non-finite derivative response")
        response_times.append(times[i])
        response.append(value)
    t_cross = unique_linear_crossing(response_times, response, target)
    if t_cross <= 0.0:
        raise ValueError("nonpositive w0^2")
    return math.sqrt(t_cross)


def jackknife_standard_error(replicates):
    if len(replicates) < 2:
        raise ValueError("need at least two jackknife replicates")
    mean = sum(replicates) / len(replicates)
    variance = (len(replicates) - 1) / len(replicates) * sum(
        (value - mean) ** 2 for value in replicates
    )
    return math.sqrt(variance), mean


def joint_blocked_jackknife(times, rows, block_size, statistic):
    validate_trajectories(times, rows, block_size)
    blocks = len(rows) // block_size
    full_mean = column_means(rows)
    central = statistic(times, full_mean)
    replicates = []
    for block in range(blocks):
        start = block * block_size
        end = start + block_size
        retained = rows[:start] + rows[end:]
        replicates.append(statistic(times, column_means(retained)))
    standard_error, replicate_mean = jackknife_standard_error(replicates)
    return central, tuple(replicates), replicate_mean, standard_error


def broken_independent_column_delete(times, rows, block_size, statistic):
    """INVALID negative control: destroys same-configuration covariance."""
    validate_trajectories(times, rows, block_size)
    blocks = len(rows) // block_size
    replicates = []
    for replicate in range(blocks):
        mixed_mean = []
        for column in range(len(times)):
            deleted_block = (replicate + column) % blocks
            values = [
                row[column]
                for index, row in enumerate(rows)
                if index // block_size != deleted_block
            ]
            mixed_mean.append(sum(values) / len(values))
        replicates.append(statistic(times, tuple(mixed_mean)))
    standard_error, _ = jackknife_standard_error(replicates)
    return tuple(replicates), standard_error


def assert_close(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        raise AssertionError((label, actual, expected))


def main():
    t0_stat = lambda times, mean: t0_like(times, mean, T0_TARGET)
    w0_stat = lambda times, mean: w0_like(times, mean, W0_TARGET)

    t0, t0_reps, t0_rep_mean, t0_se = joint_blocked_jackknife(
        TIMES, TRAJECTORIES, BLOCK_SIZE, t0_stat
    )
    w0, w0_reps, w0_rep_mean, w0_se = joint_blocked_jackknife(
        TIMES, TRAJECTORIES, BLOCK_SIZE, w0_stat
    )
    bad_t0_reps, bad_t0_se = broken_independent_column_delete(
        TIMES, TRAJECTORIES, BLOCK_SIZE, t0_stat
    )
    bad_w0_reps, bad_w0_se = broken_independent_column_delete(
        TIMES, TRAJECTORIES, BLOCK_SIZE, w0_stat
    )

    assert_close(t0, 0.3375, 2.0e-15, "t0 central")
    assert_close(t0_se, 0.008054420481627625, 2.0e-15, "t0 joint SE")
    assert_close(w0, 0.6324555320336759, 2.0e-15, "w0 central")
    assert_close(w0_se, 0.005889670897688516, 2.0e-15, "w0 joint SE")
    assert_close(bad_t0_se, 0.00543672129497043, 2.0e-15, "bad t0 SE")
    assert_close(bad_w0_se, 0.021883689599635547, 2.0e-15, "bad w0 SE")

    if not bad_t0_se < 0.8 * t0_se:
        raise AssertionError("negative control should materially underestimate t0 uncertainty")
    if not bad_w0_se > 3.0 * w0_se:
        raise AssertionError("negative control should materially overestimate w0 uncertainty")

    try:
        joint_blocked_jackknife(TIMES, TRAJECTORIES[:-1], BLOCK_SIZE, t0_stat)
    except ValueError:
        pass
    else:
        raise AssertionError("partial trailing blocks must fail closed")

    print("ok")
    print(f"configurations={len(TRAJECTORIES)}")
    print(f"block_size={BLOCK_SIZE}")
    print(f"blocks={len(TRAJECTORIES)//BLOCK_SIZE}")
    print("full_dimensionless_mean=" + ",".join(f"{x:.17g}" for x in column_means(TRAJECTORIES)))
    print(f"t0_like={t0:.17g}")
    print("t0_joint_replicates=" + ",".join(f"{x:.17g}" for x in t0_reps))
    print(f"t0_joint_replicate_mean={t0_rep_mean:.17g}")
    print(f"t0_joint_standard_error={t0_se:.17g}")
    print("t0_broken_independent_column_replicates=" + ",".join(f"{x:.17g}" for x in bad_t0_reps))
    print(f"t0_broken_independent_column_standard_error={bad_t0_se:.17g}")
    print(f"t0_broken_to_joint_se_ratio={bad_t0_se/t0_se:.17g}")
    print(f"w0_like={w0:.17g}")
    print("w0_joint_replicates=" + ",".join(f"{x:.17g}" for x in w0_reps))
    print(f"w0_joint_replicate_mean={w0_rep_mean:.17g}")
    print(f"w0_joint_standard_error={w0_se:.17g}")
    print("w0_broken_independent_column_replicates=" + ",".join(f"{x:.17g}" for x in bad_w0_reps))
    print(f"w0_broken_independent_column_standard_error={bad_w0_se:.17g}")
    print(f"w0_broken_to_joint_se_ratio={bad_w0_se/w0_se:.17g}")


if __name__ == "__main__":
    main()
