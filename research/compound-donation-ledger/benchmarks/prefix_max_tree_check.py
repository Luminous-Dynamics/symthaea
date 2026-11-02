#!/usr/bin/env python3
"""Exact-Fraction differential screen for suffix-add/max-prefix indexing.

Research-only data-structure check. This does not test Rust performance,
statistical validity, e-processes, FDR or SupFDR.
"""
from __future__ import annotations
import argparse
import json
import random
from fractions import Fraction

class LazyMaxTree:
    def __init__(self, candidate_count: int):
        self.last = candidate_count
        self.size = 1
        while self.size < candidate_count + 1:
            self.size *= 2
        neg_inf = Fraction(-10**100)
        self.maximum = [neg_inf] * (2 * self.size)
        self.lazy = [Fraction(0)] * (2 * self.size)
        for k in range(candidate_count + 1):
            self.maximum[self.size + k] = Fraction(0)
        for node in range(self.size - 1, 0, -1):
            self.maximum[node] = max(self.maximum[2*node],
                                      self.maximum[2*node+1])
        self.visited = 0
        self.update_count = 0
        self.max_visits = 0

    def _apply(self, node: int, delta: Fraction) -> None:
        self.maximum[node] += delta
        self.lazy[node] += delta

    def _push(self, node: int) -> None:
        tag = self.lazy[node]
        if tag:
            self._apply(2*node, tag)
            self._apply(2*node+1, tag)
            self.lazy[node] = Fraction(0)

    def _add(self, node: int, left: int, right: int,
             ql: int, qr: int, delta: Fraction) -> None:
        self.visited += 1
        if ql <= left and right <= qr:
            self._apply(node, delta)
            return
        self._push(node)
        mid = (left + right) // 2
        if ql <= mid:
            self._add(2*node, left, mid, ql, qr, delta)
        if qr > mid:
            self._add(2*node+1, mid+1, right, ql, qr, delta)
        self.maximum[node] = max(self.maximum[2*node],
                                 self.maximum[2*node+1])

    def suffix_add(self, first_prefix: int, delta: Fraction) -> None:
        before = self.visited
        self._add(1, 0, self.size-1, first_prefix, self.last, delta)
        visits = self.visited - before
        self.update_count += 1
        self.max_visits = max(self.max_visits, visits)

    @property
    def root_max(self) -> Fraction:
        return self.maximum[1]


def run_trace(seed: int, n: int, rounds: int) -> dict:
    weights = [Fraction(1, (i+1)*(i+2)) for i in range(n)]
    evidence = [Fraction(1) for _ in range(n)]
    incoming = [Fraction(0) for _ in range(n)]
    outgoing = [Fraction(0) for _ in range(n)]
    previous_scores = [weights[i] * evidence[i] for i in range(n)]
    tokens = {}
    local_ord = [0] * n
    total_in = Fraction(0)
    total_out = Fraction(0)
    prefix = [Fraction(0) for _ in range(n+1)]
    tree = LazyMaxTree(n)
    checked_events = 0

    def check_all() -> None:
        assert max(prefix) == 0
        assert tree.root_max == max(prefix) == 0
        running_in = Fraction(0)
        running_out = Fraction(0)
        for k in range(1, n+1):
            running_in += incoming[k-1]
            running_out += outgoing[k-1]
            assert prefix[k] == running_in - running_out
        assert sum(tok["remaining"] for tok in tokens.values()) == (
            total_out-total_in
        )

    for t in range(1, rounds+1):
        for idx in range(n):
            donor = idx + 1
            rng = random.Random(seed + donor*1_000_003 + t*97_409)
            dx = Fraction(rng.randint(0, 2_000_000), 1_000_000)
            old_x = evidence[idx]
            old_score = weights[idx]*old_x + incoming[idx] - outgoing[idx]
            evidence[idx] += dx
            dest = donor + 1 if donor < n else None
            target = (Fraction(0) if dest is None else
                      min(max(weights[idx]*evidence[idx]-1, Fraction(0)),
                          weights[idx]))
            reserve = target - outgoing[idx]
            assert reserve >= 0 and reserve <= weights[idx]*dx
            assert target <= weights[idx]*min(evidence[idx], Fraction(1))
            if reserve:
                local_ord[idx] += 1
                tid = (donor, local_ord[idx])
                tokens[tid] = {"source": donor, "destination": dest,
                               "amount": reserve, "remaining": reserve}
                outgoing[idx] += reserve
                total_out += reserve
                for k in range(donor, n+1):
                    prefix[k] -= reserve
                tree.suffix_add(donor, -reserve)
                checked_events += 1
            new_score = weights[idx]*evidence[idx] + incoming[idx] - outgoing[idx]
            assert new_score >= old_score >= 0
            check_all()

        for tid in sorted(tokens):
            tok = tokens[tid]
            if not tok["remaining"]:
                continue
            rng = random.Random(seed + t*71_111 + tid[0]*313 + tid[1]*17)
            if rng.randrange(3) == 0:
                continue
            amount = (tok["remaining"] if rng.randrange(2)
                      else tok["remaining"] / 2)
            recipient = tok["destination"]
            idx = recipient - 1
            old_score = weights[idx]*evidence[idx] + incoming[idx] - outgoing[idx]
            assert amount > 0 and amount <= tok["remaining"]
            tok["remaining"] -= amount
            incoming[idx] += amount
            total_in += amount
            for k in range(recipient, n+1):
                prefix[k] += amount
            tree.suffix_add(recipient, amount)
            new_score = weights[idx]*evidence[idx] + incoming[idx] - outgoing[idx]
            assert new_score >= old_score >= 0
            checked_events += 1
            check_all()

        scores = [weights[i]*evidence[i] + incoming[i] - outgoing[i]
                  for i in range(n)]
        assert all(scores[i] >= previous_scores[i] for i in range(n))
        previous_scores = scores

    return {"events": checked_events, "tree_updates": tree.update_count,
            "tree_nodes_visited": tree.visited,
            "max_nodes_visited_per_update": tree.max_visits,
            "final_root_max": str(tree.root_max), "tokens": len(tokens)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=10)
    parser.add_argument("--universe", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    rows = [run_trace(20261102 + 31*i, args.universe, args.rounds)
            for i in range(args.traces)]
    report = {
        "scope": "exact structural differential screen only; not Rust timing or statistical validity",
        "arithmetic": "fractions; exact, no float",
        "traces": args.traces, "universe": args.universe,
        "rounds": args.rounds,
        "checks": [
            "range-add/max tree equals brute-force prefix vector after each event",
            "root maximum remains exactly zero", "all prefix budgets",
            "donor caps and own-increment caps", "score nonnegative/monotone",
            "escrow conservation", "source-local token ordinals"
        ],
        "total_events": sum(row["events"] for row in rows),
        "total_tree_updates": sum(row["tree_updates"] for row in rows),
        "total_tree_nodes_visited": sum(row["tree_nodes_visited"] for row in rows),
        "max_nodes_visited_per_update": max(
            row["max_nodes_visited_per_update"] for row in rows
        ),
        "all_final_root_maxima_zero": all(
            row["final_root_max"] == "0" for row in rows
        ),
        "note": "Node visits validate the suffix-update structure; they are not a Rust latency benchmark."
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
