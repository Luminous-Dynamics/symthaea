#!/usr/bin/env python3
"""Independent deterministic keyed-reduction oracle for LQCD-021L.

This subject qualifies only canonical binary64 reduction semantics.  It imports
no Symthaea/Rust implementation.  Scientific callers must preserve atomic keyed
values through sharding; arbitrary shard-local partial sums are not authorized
as substitutes for those leaves.
"""
import hashlib
import json
import math
import random
import struct

ORACLE_ID = "lqcd_canonical_keyed_pairwise_f64_v1"


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def checked_add(a, b):
    out = a + b
    if not math.isfinite(out):
        raise ValueError("non-finite reduction result")
    return out


def canonical_keyed_sum(records):
    """Sort unique integer keys, then reduce values with a fixed pairwise tree."""
    if not records:
        raise ValueError("empty reduction")
    ordered = sorted(records, key=lambda item: item[0])
    keys = [key for key, _ in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    level = []
    for key, value in ordered:
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError("integer key required")
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("finite binary64 value required")
        level.append(value)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 == len(level):
                nxt.append(level[i])
            else:
                nxt.append(checked_add(level[i], level[i + 1]))
        level = nxt
    return level[0]


def canonical_keyed_mean(records):
    total = canonical_keyed_sum(records)
    out = total / float(len(records))
    if not math.isfinite(out):
        raise ValueError("non-finite mean")
    return out


def naive_sum(records):
    total = 0.0
    for _, value in records:
        total += value
    return total


def require_same_bits(a, b, label):
    if bits(a) != bits(b):
        raise AssertionError((label, a, b, bits(a), bits(b)))


def main():
    baseline = [
        (40, 3.0),
        (10, 1.0e16),
        (30, -1.0e16),
        (20, 1.0),
        (50, -0.25),
        (60, 0.5),
        (70, 0.125),
    ]
    canonical_sum = canonical_keyed_sum(baseline)
    canonical_mean = canonical_keyed_mean(baseline)

    # Input order must not matter because keys define the leaf order.
    rng = random.Random(0x21_0C_D)
    shuffled_bits = []
    for _ in range(64):
        candidate = list(baseline)
        rng.shuffle(candidate)
        current = canonical_keyed_sum(candidate)
        require_same_bits(current, canonical_sum, "permutation")
        shuffled_bits.append(bits(current))

    # Any valid shard partition/order must give the same result when shards
    # preserve atomic keyed leaves and the canonical reducer runs after union.
    shards = [
        [baseline[0], baseline[4]],
        [baseline[2], baseline[6], baseline[1]],
        [baseline[5]],
        [baseline[3]],
    ]
    shard_orders = [
        (0, 1, 2, 3),
        (3, 2, 1, 0),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
    ]
    shard_union_bits = []
    for order in shard_orders:
        union = []
        for shard_index in order:
            union.extend(shards[shard_index])
        current = canonical_keyed_sum(union)
        require_same_bits(current, canonical_sum, "shard-union")
        shard_union_bits.append(bits(current))

    # Negative control: ordinary insertion-order summation is not an adequate
    # scientific commitment for ill-conditioned inputs.
    naive_a = naive_sum([(0, 1.0e16), (1, 1.0), (2, -1.0e16)])
    naive_b = naive_sum([(0, 1.0e16), (2, -1.0e16), (1, 1.0)])
    if bits(naive_a) == bits(naive_b):
        raise AssertionError("negative control did not expose order sensitivity")

    # Odd leaf counts are part of the frozen tree convention: an unpaired final
    # leaf advances unchanged to the next level.
    odd = [(4, 0.5), (1, 0.25), (3, 1.0), (2, -0.125), (5, 0.0625)]
    odd_sum = canonical_keyed_sum(odd)

    try:
        canonical_keyed_sum([(1, 1.0), (1, 2.0)])
        raise AssertionError("duplicate key accepted")
    except ValueError as exc:
        if str(exc) != "duplicate key":
            raise

    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            canonical_keyed_sum([(1, bad)])
            raise AssertionError("non-finite value accepted")
        except ValueError as exc:
            if str(exc) != "finite binary64 value required":
                raise

    try:
        canonical_keyed_sum([])
        raise AssertionError("empty reduction accepted")
    except ValueError as exc:
        if str(exc) != "empty reduction":
            raise

    result = {
        "oracle_id": ORACLE_ID,
        "baseline": {
            "canonical_sum": canonical_sum,
            "canonical_sum_hex": canonical_sum.hex(),
            "canonical_sum_bits": f"0x{bits(canonical_sum):016x}",
            "canonical_mean": canonical_mean,
            "canonical_mean_hex": canonical_mean.hex(),
            "canonical_mean_bits": f"0x{bits(canonical_mean):016x}",
            "permutation_trials": len(shuffled_bits),
            "unique_permutation_result_bits": sorted(
                {f"0x{x:016x}" for x in shuffled_bits}
            ),
            "shard_union_result_bits": [f"0x{x:016x}" for x in shard_union_bits],
        },
        "odd_leaf_fixture": {
            "sum": odd_sum,
            "sum_hex": odd_sum.hex(),
            "sum_bits": f"0x{bits(odd_sum):016x}",
        },
        "negative_control": {
            "naive_a": naive_a,
            "naive_a_bits": f"0x{bits(naive_a):016x}",
            "naive_b": naive_b,
            "naive_b_bits": f"0x{bits(naive_b):016x}",
        },
        "contract": {
            "leaf_order": "ascending_unique_integer_key",
            "tree": "adjacent_pairwise_left_to_right_carry_odd_leaf",
            "shards_preserve_atomic_keyed_leaves": True,
            "non_finite": "reject",
            "duplicate_keys": "reject",
            "empty": "reject",
        },
    }
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)


if __name__ == "__main__":
    main()
