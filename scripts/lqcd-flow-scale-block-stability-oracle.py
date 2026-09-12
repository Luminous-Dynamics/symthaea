#!/usr/bin/env python3
"""Independent LQCD-018G block-size stability oracle.

Standard-library only. Imports no Symthaea/Rust code.
"""
from math import sqrt

TIMES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DIMENSIONLESS = [
    [0.1120,0.1921,0.2542,0.3343,0.4064,0.4865],
    [0.1140,0.1901,0.2562,0.3323,0.4084,0.4845],
    [0.1160,0.1881,0.2582,0.3303,0.4104,0.4825],
    [0.1160,0.1987,0.2634,0.3461,0.4208,0.5035],
    [0.1180,0.1967,0.2654,0.3441,0.4228,0.5015],
    [0.1200,0.1947,0.2674,0.3421,0.4248,0.4995],
    [0.1198,0.2056,0.2734,0.3592,0.4370,0.5228],
    [0.1218,0.2036,0.2754,0.3572,0.4390,0.5208],
    [0.1238,0.2016,0.2774,0.3552,0.4410,0.5188],
    [0.1242,0.2116,0.2810,0.3684,0.4478,0.5352],
    [0.1262,0.2096,0.2830,0.3664,0.4498,0.5332],
    [0.1282,0.2076,0.2850,0.3644,0.4518,0.5312],
]

def unique_crossing(xs, ys, target):
    exact = [i for i, y in enumerate(ys) if y == target]
    if len(exact) == 1:
        return xs[exact[0]]
    if len(exact) > 1:
        raise ValueError("ambiguous exact crossing")
    crossings = []
    for i in range(len(xs)-1):
        left = ys[i] - target
        right = ys[i+1] - target
        if left * right < 0.0:
            f = (target - ys[i]) / (ys[i+1] - ys[i])
            crossings.append(xs[i] + f * (xs[i+1] - xs[i]))
    if len(crossings) != 1:
        raise ValueError("crossing count != 1")
    return crossings[0]

def mean_curve(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(TIMES))]

def t0_like(rows, target=0.30):
    return unique_crossing(TIMES, mean_curve(rows), target)

def chain_blocks(chain_lengths, block_size):
    if block_size <= 0:
        raise ValueError("block_size")
    ranges = []
    offset = 0
    for length in chain_lengths:
        if length == 0 or length % block_size:
            raise ValueError("partial chain block")
        for local in range(0, length, block_size):
            ranges.append((offset + local, offset + local + block_size))
        offset += length
    return ranges

def jackknife(rows, chain_lengths, block_size):
    blocks = chain_blocks(chain_lengths, block_size)
    if len(blocks) < 2:
        raise ValueError("too few blocks")
    central = t0_like(rows)
    reps = []
    for start, end in blocks:
        retained = rows[:start] + rows[end:]
        reps.append(t0_like(retained))
    rep_mean = sum(reps) / len(reps)
    var = (len(reps)-1)/len(reps) * sum((x-rep_mean)**2 for x in reps)
    return central, reps, sqrt(var)

def symmetric_relative_change(a, b):
    d = abs(a) + abs(b)
    return 0.0 if d == 0.0 else 2.0 * abs(a-b) / d

def main():
    expected = {
        1: 0.004212888395644406,
        2: 0.00610455552751023,
        3: 0.008054420481627626,
        4: 0.009086301895189922,
        6: 0.012732198384543647,
    }
    ses = {}
    for block_size in [1,2,3,4,6]:
        central, _, se = jackknife(DIMENSIONLESS, [12], block_size)
        assert abs(central - 0.3375) < 4e-15
        assert abs(se - expected[block_size]) < 5e-15
        ses[block_size] = se

    plateau = [3,4,6]
    changes = [
        symmetric_relative_change(ses[a], ses[b])
        for a, b in zip(plateau, plateau[1:])
    ]
    max_change = max(changes)
    assert max_change > 0.33 and max_change < 0.35
    assert not (max_change <= 0.30)
    assert max_change <= 0.35

    rejected_cross_chain_candidate = False
    try:
        chain_blocks([6,6], 4)
    except ValueError:
        rejected_cross_chain_candidate = True
    assert rejected_cross_chain_candidate
    assert chain_blocks([6,6], 3) == [(0,3),(3,6),(6,9),(9,12)]

    print("subject=independent-standard-library")
    print(f"central_t0={t0_like(DIMENSIONLESS):.17g}")
    for b in [1,2,3,4,6]:
        print(f"block_{b}_se={ses[b]:.17g}")
    print("plateau_blocks=3,4,6")
    print(f"plateau_change_3_4={changes[0]:.17g}")
    print(f"plateau_change_4_6={changes[1]:.17g}")
    print(f"plateau_max_change={max_change:.17g}")
    print("policy_0.30=FAIL")
    print("policy_0.35=PASS")
    print("two_chain_block_4=REJECT_PARTIAL_CHAIN_BLOCK")
    print("two_chain_block_3=CHAIN_LOCAL")
    print("authority=synthetic-resampling-semantics-only")

if __name__ == "__main__":
    main()
