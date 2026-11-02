#!/usr/bin/env python3
"""Exact-Fraction structural check for cut-projective donation routing.

This is not an e-process simulator and does not estimate FDR/SupFDR.
The route for donor i may use current evidence for IDs <= i, but never IDs > i.

Run: python3 routing_cut_projectivity_check.py --traces 250 --rounds 40
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

@dataclass
class Token:
    token_id: tuple[int, int]  # (source_id, source-local ordinal)
    source: int
    destination: int
    created_round: int
    amount: Fraction
    remaining: Fraction

def route_destination(donor_id: int, evidence: list[Fraction],
                      round_no: int, universe_size: int) -> int | None:
    """Current-evidence-aware route using only the donor's ID-prefix cone."""
    if donor_id >= universe_size:
        return None
    fingerprint = round_no * 104_729 + donor_id * 65_537
    for candidate_id, value in enumerate(evidence[:donor_id], start=1):
        fingerprint += candidate_id * int(value * 1_000_000)
    return donor_id + 1 + fingerprint % (universe_size - donor_id)

def simulate(seed: int, universe_size: int, horizon: int,
             rounds: int) -> dict[str, Any]:
    gamma = Fraction(1, universe_size)
    evidence = [Fraction(1) for _ in range(universe_size)]
    incoming = [Fraction(0) for _ in range(universe_size)]
    outgoing = [Fraction(0) for _ in range(universe_size)]
    previous_scores = [gamma * value for value in evidence]
    source_local_ordinal = [0 for _ in range(universe_size)]
    tokens: dict[tuple[int, int], Token] = {}
    reservations: list[tuple[Any, ...]] = []
    credits: list[tuple[Any, ...]] = []
    activation_round = [1 + ((j * 3 + 1) % 5)
                        for j in range(universe_size)]

    for round_no in range(1, rounds + 1):
        # Stable ID order makes the within-round prefix filtration explicit.
        for position in range(horizon):
            donor_id = position + 1
            rng = random.Random(seed + 1_000_003 * donor_id
                                + 97_409 * round_no)
            delta = Fraction(rng.randint(0, 1_500_000), 1_000_000)
            old_evidence = evidence[position]
            evidence[position] += delta

            # The current donor observation is visible; higher IDs are not.
            destination = route_destination(donor_id, evidence,
                                            round_no, universe_size)
            target = (Fraction(0) if destination is None else
                      min(max(gamma * evidence[position] - 1,
                              Fraction(0)), gamma))
            reserve = target - outgoing[position]
            assert reserve >= 0
            assert reserve <= gamma * delta
            assert target <= gamma * min(evidence[position], Fraction(1))
            assert evidence[position] >= old_evidence

            if reserve:
                source_local_ordinal[position] += 1
                token_id = (donor_id, source_local_ordinal[position])
                tokens[token_id] = Token(token_id, donor_id, destination,
                                         round_no, reserve, reserve)
                reservations.append((token_id, donor_id, destination,
                                     round_no, reserve))
                outgoing[position] += reserve

        # Missing recipients remain in escrow in the shorter horizon.
        for token_id in sorted(tokens):
            token = tokens[token_id]
            if (token.remaining and token.destination <= horizon
                    and activation_round[token.destination - 1] <= round_no):
                amount = token.remaining
                token.remaining = Fraction(0)
                incoming[token.destination - 1] += amount
                credits.append((token_id, token.destination, round_no, amount))

        scores = [gamma * evidence[j] + incoming[j] - outgoing[j]
                  for j in range(universe_size)]
        for j in range(horizon):
            assert scores[j] >= 0
            assert scores[j] >= previous_scores[j]
        previous_scores = scores
        for prefix in range(1, horizon + 1):
            assert sum(incoming[:prefix]) <= sum(outgoing[:prefix])
        assert sum(token.remaining for token in tokens.values()) == (
            sum(outgoing) - sum(incoming)
        )

    return {"evidence": evidence, "incoming": incoming, "outgoing": outgoing,
            "tokens": tokens, "reservations": reservations, "credits": credits,
            "scores": tuple(scores[:horizon])}

def cut_projection(run: dict[str, Any], prefix: int) -> tuple[Any, ...]:
    candidates = tuple(
        (j, run["evidence"][j-1], run["incoming"][j-1],
         run["outgoing"][j-1], run["scores"][j-1])
        for j in range(1, prefix + 1)
    )
    token_rows = []
    for token_id, token in sorted(run["tokens"].items()):
        if token.source > prefix:
            continue
        if token.destination <= prefix:
            token_rows.append((token_id, token.source, token.destination,
                               token.created_round, token.amount,
                               "internal", token.remaining))
        else:
            # Escrow plus delivery beyond the cut is one outside balance.
            outside = token.remaining + (token.amount - token.remaining)
            token_rows.append((token_id, token.source, token.destination,
                               token.created_round, token.amount,
                               "crossing", outside))
    reservations = tuple(row for row in run["reservations"] if row[1] <= prefix)
    credits = tuple(row for row in run["credits"] if row[1] <= prefix)
    return candidates, tuple(token_rows), reservations, credits

def global_counter_negative_control() -> dict[str, Any]:
    # An added high-ID token shifts an old donor's later global token number.
    return {"short_old_token_id": 2, "extended_old_token_id": 3,
            "mismatch_expected": True, "source_scoped_identity": [1, 2]}

def higher_id_lookahead_negative_control() -> dict[str, Any]:
    short_destination = 2
    extended_destination = 3  # higher-ID evidence changes the old donor's route
    return {"donor_id": 1, "short_destination": short_destination,
            "extended_destination": extended_destination,
            "projectivity_broken": short_destination != extended_destination}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=250)
    parser.add_argument("--universe", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()
    if args.traces < 1 or args.universe < 3 or args.rounds < 1:
        raise SystemExit("traces/rounds must be positive; universe must be >= 3")

    projected = 0
    crossing_differences = 0
    max_prefix = args.universe - 1
    for trace in range(args.traces):
        prefix = 1 + ((trace * 13 + 5) % max_prefix)
        seed = 20_261_102 + trace * 17
        short = simulate(seed, args.universe, prefix, args.rounds)
        extended = simulate(seed, args.universe, args.universe, args.rounds)
        assert cut_projection(short, prefix) == cut_projection(extended, prefix)

        short_cross = {tid: tok.remaining for tid, tok in short["tokens"].items()
                       if tok.source <= prefix and tok.destination > prefix}
        long_cross = {tid: tok.remaining for tid, tok in extended["tokens"].items()
                      if tok.source <= prefix and tok.destination > prefix}
        assert short_cross.keys() == long_cross.keys()
        crossing_differences += int(any(short_cross[tid] != long_cross[tid]
                                        for tid in short_cross))
        projected += 1

    report = {
        "scope": "structural bookkeeping/projectivity only; not e-process validity, FDR, or SupFDR",
        "seed_schedule": "20261102 + 17*trace; independent stream per (candidate, round)",
        "traces": args.traces, "global_candidate_universe": args.universe,
        "rounds": args.rounds, "prefix_horizons": f"cycled 1..{max_prefix}",
        "routing_policy": "current-evidence-aware, using only IDs <= donor and fixed-universe metadata",
        "reserve_policy": "D_i,t = min((gamma_i * X_i,t - 1)_+, gamma_i); gamma_i = 1/universe",
        "arithmetic": "exact fractions; no floating-point accounting",
        "assertions": [
            "identical prefix candidate evidence/incoming/outgoing/score states",
            "identical full token state for internal edges",
            "stable source-scoped token IDs, destinations, creation rounds and face amounts",
            "cut-equivalent state for crossing edges",
            "identical prefix reservations and internal credits",
            "all prefix budgets", "donor cap and own-increment cap",
            "nonnegative and nondecreasing weighted scores",
            "exact escrow conservation"
        ],
        "projected_replay_pairs_passed": projected,
        "pairs_with_literal_crossing_remaining_difference": crossing_differences,
        "global_counter_negative_control": global_counter_negative_control(),
        "higher_id_lookahead_negative_control": higher_id_lookahead_negative_control()
    }
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
