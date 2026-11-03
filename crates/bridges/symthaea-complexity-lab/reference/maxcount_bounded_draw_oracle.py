#!/usr/bin/env python3
"""Small exact oracle for rejection-sampled bounded MaxCount branch draws."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
U128_MAX = (1 << 128) - 1

def draw_below(bound: int, words):
    if not 1 <= bound <= U128_MAX:
        raise ValueError("bound out of range")
    if bound == 1:
        return 0, []
    mask = (1 << (bound - 1).bit_length()) - 1
    used = []
    for word in words:
        if not 0 <= word <= U128_MAX:
            raise ValueError("raw word is not u128")
        used.append(word)
        candidate = word & mask
        if candidate < bound:
            return candidate, used
    raise ValueError("word stream exhausted")

def verify(record, expected_bound):
    if record["upper_bound"] != expected_bound:
        raise ValueError("wrong upper bound")
    value, consumed = draw_below(expected_bound, record["random_words"])
    if consumed != record["random_words"] or value != record["accepted_draw"]:
        raise ValueError("transcript does not replay")
    return value

def exact_pair_check(bound):
    if bound == 1:
        return 1
    space = 1 << (bound - 1).bit_length()
    counts = Counter()
    for first in range(space):
        if first < bound:
            counts[first] += space
        else:
            for second in range(space):
                if second < bound:
                    counts[second] += 1
    assert len({counts[x] for x in range(bound)}) == 1
    return counts[0]

def run():
    pair_checks = sum((1 << (b - 1).bit_length()) ** 2 for b in range(1, 65))
    for b in range(1, 65):
        exact_pair_check(b)
    accepted, words = draw_below(3, [3, 2])
    assert (accepted, words) == (2, [3, 2])
    receipt = {"upper_bound": 3, "random_words": words, "accepted_draw": accepted}
    assert verify(receipt, 3) == 2
    rejects = 0
    for mutant in [dict(receipt, accepted_draw=1),
                   dict(receipt, random_words=[2, 1]),
                   dict(receipt, upper_bound=4)]:
        try:
            verify(mutant, 3)
        except ValueError:
            rejects += 1
    assert rejects == 3
    wide, wide_words = draw_below(U128_MAX, [U128_MAX, U128_MAX - 1])
    assert (wide, wide_words) == (U128_MAX - 1, [U128_MAX, U128_MAX - 1])
    assert draw_below(1, [])[0] == 0
    result = {
        "suite": "MaxCount unbiased bounded-draw transcript oracle",
        "bounds_exhaustively_checked": 64,
        "exact_two_word_path_checks": pair_checks,
        "all_bounded_draws_uniform": True,
        "transcript_replay": True,
        "u128_ceiling_rejection": True,
        "bound_one_consumes_no_randomness": True,
        "negative_controls_rejected": rejects,
        "claim_boundary": "Exact finite reference only. Removes modulo bias assuming ideal uniform raw words; does not prove entropy quality, compile Rust, or benchmark performance."
    }
    Path(__file__).with_name("maxcount_bounded_draw_results.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    run()
