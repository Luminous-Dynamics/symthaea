#!/usr/bin/env python3
"""Exact-Fraction reserve/assign/credit accounting and independent replay."""
from __future__ import annotations
import copy
import json
import random
from fractions import Fraction

POLICY = "sha256:two-phase-source-escrow-v1"


def reserve_target(gamma, x):
    return min(max(gamma * x - 1, Fraction(0)), gamma)


class Writer:
    def __init__(self, weights):
        self.w = weights
        self.n = len(weights)
        self.x = [Fraction(0) for _ in weights]
        self.incoming = [Fraction(0) for _ in weights]
        self.outgoing = [Fraction(0) for _ in weights]
        self.active = [False] * self.n
        self.tokens = {}
        self.token_ord = [0] * self.n
        self.last_round = -1
        self.events = []
        self.last_scores = [Fraction(0)] * self.n

    def _time(self, t):
        if t < self.last_round:
            raise ValueError("round rollback")
        self.last_round = t

    def _scores(self):
        return [
            self.w[i] * self.x[i] + self.incoming[i] - self.outgoing[i]
            for i in range(self.n)
        ]

    def _check(self):
        z = self._scores()
        if any(v < 0 for v in z) or any(a < b for a, b in zip(z, self.last_scores)):
            raise ValueError("compound score negative or decreasing")
        balance = Fraction(0)
        for i in range(self.n):
            if self.outgoing[i] > self.w[i] * min(self.x[i], Fraction(1)):
                raise ValueError("donor cap")
            balance += self.incoming[i] - self.outgoing[i]
            if balance > 0:
                raise ValueError("prefix budget")
        total_out = sum(self.outgoing, Fraction(0))
        total_in = sum(self.incoming, Fraction(0))
        escrow = Fraction(0)
        for token in self.tokens.values():
            escrow += token["unassigned"]
            assigned_face = sum(
                (a["original"] for a in token["assignments"].values()),
                Fraction(0),
            )
            if token["original"] != token["unassigned"] + assigned_face:
                raise ValueError("source token conservation")
            for assignment in token["assignments"].values():
                if assignment["original"] != assignment["remaining"] + assignment["credited"]:
                    raise ValueError("assignment conservation")
                escrow += assignment["remaining"]
        if total_out != total_in + escrow:
            raise ValueError("global escrow conservation")
        self.last_scores = z

    def activate(self, t, i):
        self._time(t)
        if self.active[i]:
            raise ValueError("duplicate activation")
        self.active[i] = True
        self.events.append({"kind": "activate", "round": t, "candidate": i})
        self._check()

    def update(self, t, i, new_x):
        self._time(t)
        if not self.active[i] or new_x < self.x[i]:
            raise ValueError("invalid evidence")
        old_x = self.x[i]
        amount = reserve_target(self.w[i], new_x) - reserve_target(self.w[i], old_x)
        if amount < 0 or amount > self.w[i] * (new_x - old_x):
            raise ValueError("own-increment cap")
        self.x[i] = new_x
        self.outgoing[i] += amount
        token_id = None
        if amount:
            self.token_ord[i] += 1
            token_id = (i, self.token_ord[i])
            self.tokens[token_id] = {
                "source": i,
                "created": t,
                "original": amount,
                "unassigned": amount,
                "assignments": {},
                "next_assignment": 1,
            }
        self.events.append({
            "kind": "evidence",
            "round": t,
            "candidate": i,
            "new_x": new_x,
            "reserve": amount,
            "token_id": token_id,
        })
        self._check()

    def assign(self, t, token_id, destination, amount):
        self._time(t)
        token = self.tokens[token_id]
        if t <= token["created"] or destination <= token["source"] or destination >= self.n:
            raise ValueError("assignment timing/direction")
        if amount <= 0 or amount > token["unassigned"]:
            raise ValueError("assignment overdraw")
        ordinal = token["next_assignment"]
        token["next_assignment"] += 1
        assignment_id = (token_id[0], token_id[1], ordinal)
        token["unassigned"] -= amount
        token["assignments"][assignment_id] = {
            "destination": destination,
            "original": amount,
            "remaining": amount,
            "credited": Fraction(0),
            "round": t,
        }
        self.events.append({
            "kind": "assign",
            "round": t,
            "token_id": token_id,
            "assignment_id": assignment_id,
            "destination": destination,
            "amount": amount,
            "policy_digest": POLICY,
        })
        self._check()
        return assignment_id

    def credit(self, t, token_id, assignment_id, amount):
        self._time(t)
        token = self.tokens[token_id]
        assignment = token["assignments"][assignment_id]
        destination = assignment["destination"]
        if t < assignment["round"] or not self.active[destination]:
            raise ValueError("credit before assignment/activation")
        if amount <= 0 or amount > assignment["remaining"]:
            raise ValueError("credit overdraw")
        assignment["remaining"] -= amount
        assignment["credited"] += amount
        self.incoming[destination] += amount
        self.events.append({
            "kind": "credit",
            "round": t,
            "token_id": token_id,
            "assignment_id": assignment_id,
            "destination": destination,
            "amount": amount,
        })
        self._check()


def independent_replay(weights, events):
    """Independent dictionary replay; does not invoke Writer transitions."""
    n = len(weights)
    x = [Fraction(0)] * n
    incoming = [Fraction(0)] * n
    outgoing = [Fraction(0)] * n
    active = [False] * n
    tokens = {}
    ordinals = [0] * n
    last_round = -1
    previous_scores = [Fraction(0)] * n

    for event in events:
        t = event["round"]
        if t < last_round:
            raise ValueError("event time rollback")
        last_round = t
        kind = event["kind"]
        if kind == "activate":
            i = event["candidate"]
            if active[i]:
                raise ValueError("duplicate activation")
            active[i] = True
        elif kind == "evidence":
            i = event["candidate"]
            new_x = event["new_x"]
            if not active[i] or new_x < x[i]:
                raise ValueError("bad evidence update")
            amount = reserve_target(weights[i], new_x) - reserve_target(weights[i], x[i])
            if amount != event["reserve"] or amount > weights[i] * (new_x - x[i]):
                raise ValueError("reserve mismatch")
            x[i] = new_x
            outgoing[i] += amount
            if amount:
                ordinals[i] += 1
                token_id = (i, ordinals[i])
                if token_id != event["token_id"]:
                    raise ValueError("source-local token ID mismatch")
                tokens[token_id] = {
                    "source": i,
                    "created": t,
                    "original": amount,
                    "unassigned": amount,
                    "assignments": {},
                    "next": 1,
                }
            elif event["token_id"] is not None:
                raise ValueError("unexpected token ID")
        elif kind == "assign":
            token_id = event["token_id"]
            token = tokens[token_id]
            destination = event["destination"]
            amount = event["amount"]
            assignment_id = event["assignment_id"]
            if event["policy_digest"] != POLICY:
                raise ValueError("policy digest mismatch")
            if t <= token["created"] or destination <= token["source"] or destination >= n:
                raise ValueError("bad assignment route")
            if amount <= 0 or amount > token["unassigned"]:
                raise ValueError("assignment overdraw")
            if assignment_id in token["assignments"] or assignment_id != (
                token_id[0], token_id[1], token["next"]
            ):
                raise ValueError("assignment ID replay")
            token["next"] += 1
            token["unassigned"] -= amount
            token["assignments"][assignment_id] = {
                "destination": destination,
                "original": amount,
                "remaining": amount,
                "credited": Fraction(0),
                "round": t,
            }
        elif kind == "credit":
            token = tokens[event["token_id"]]
            assignment = token["assignments"][event["assignment_id"]]
            destination = assignment["destination"]
            amount = event["amount"]
            if destination != event["destination"] or t < assignment["round"] or not active[destination]:
                raise ValueError("bad credit destination/time")
            if amount <= 0 or amount > assignment["remaining"]:
                raise ValueError("credit overdraw")
            assignment["remaining"] -= amount
            assignment["credited"] += amount
            incoming[destination] += amount
        else:
            raise ValueError("unknown event")

        # A deliberately independent full scan after every state transition.
        prefix = Fraction(0)
        for i in range(n):
            if outgoing[i] > weights[i] * min(x[i], Fraction(1)):
                raise ValueError("donor cap")
            prefix += incoming[i] - outgoing[i]
            if prefix > 0:
                raise ValueError("prefix budget")
        escrow = Fraction(0)
        for token in tokens.values():
            escrow += token["unassigned"]
            assigned_face = sum(
                (a["original"] for a in token["assignments"].values()),
                Fraction(0),
            )
            if token["original"] != token["unassigned"] + assigned_face:
                raise ValueError("token split conservation")
            for assignment in token["assignments"].values():
                if assignment["original"] != assignment["remaining"] + assignment["credited"]:
                    raise ValueError("subtoken credit conservation")
                escrow += assignment["remaining"]
        if sum(outgoing, Fraction(0)) != sum(incoming, Fraction(0)) + escrow:
            raise ValueError("global escrow conservation")
        scores = [
            weights[i] * x[i] + incoming[i] - outgoing[i]
            for i in range(n)
        ]
        if any(v < 0 for v in scores) or any(a < b for a, b in zip(scores, previous_scores)):
            raise ValueError("score negative/decreasing")
        previous_scores = scores
    return incoming, outgoing, tokens


def randomized_trace(seed, n=16, rounds=40):
    rng = random.Random(seed)
    weights = [Fraction(1, (i + 1) * (i + 2)) for i in range(n)]
    arrival = [min((i // 3) * 4, rounds - 1) for i in range(n)]
    writer = Writer(weights)
    assigned_once = set()
    for t in range(rounds):
        for i in range(n):
            if arrival[i] == t:
                writer.activate(t, i)
        # Route checkpoint: only past ledger state and precommitted policy metadata.
        for token_id, token in sorted(writer.tokens.items()):
            if token_id in assigned_once or token["unassigned"] <= 0 or rng.random() >= 0.35:
                continue
            destination = rng.randrange(token_id[0] + 1, n)
            amount = token["unassigned"] * Fraction(rng.randint(1, 4), 5)
            assignment_id = writer.assign(t, token_id, destination, amount)
            assigned_once.add(token_id)
            if writer.active[destination] and rng.random() < 0.65:
                writer.credit(t, token_id, assignment_id, amount)
        # Previously assigned sub-tokens can be credited after their recipient activates.
        for token_id, token in sorted(writer.tokens.items()):
            for assignment_id, assignment in sorted(token["assignments"].items()):
                if assignment["remaining"] > 0 and writer.active[assignment["destination"]] and rng.random() < 0.15:
                    amount = assignment["remaining"] * Fraction(rng.randint(1, 4), 5)
                    writer.credit(t, token_id, assignment_id, amount)
        # New evidence arrives only after route decisions have been frozen.
        for i in range(n):
            if writer.active[i]:
                writer.update(t, i, writer.x[i] + Fraction(rng.randint(0, 1_500_000), 1_000_000))
        writer._check()
    replay_in, replay_out, _ = independent_replay(weights, writer.events)
    assert replay_in == writer.incoming and replay_out == writer.outgoing
    return writer


def negative_controls():
    writer = Writer([Fraction(1, 2), Fraction(1, 6), Fraction(1, 12)])
    for i in range(3):
        writer.activate(0, i)
    writer.update(1, 0, Fraction(8))
    token_id = (0, 1)
    assignment_id = writer.assign(2, token_id, 1, Fraction(1, 10))
    writer.credit(2, token_id, assignment_id, Fraction(1, 10))
    base = writer.events
    rejected = 0

    def must_reject(mutator):
        nonlocal rejected
        events = copy.deepcopy(base)
        mutator(events)
        try:
            independent_replay(writer.w, events)
        except (ValueError, KeyError, IndexError):
            rejected += 1
            return
        raise AssertionError("tampered receipt was accepted")

    must_reject(lambda events: events[-2].update(destination=0))
    must_reject(lambda events: events[-2].update(amount=Fraction(9, 10)))
    must_reject(lambda events: events[-1].update(amount=Fraction(9, 10)))
    must_reject(lambda events: events[-1].update(destination=2))
    must_reject(lambda events: events.insert(-1, copy.deepcopy(events[-2])))
    must_reject(lambda events: events[-2].update(policy_digest="sha256:tampered"))
    return rejected


def main():
    traces = 50
    total_events = 0
    for seed in range(traces):
        total_events += len(randomized_trace(10075 + seed).events)
    rejected = negative_controls()
    return {
        "scope": "exact Fraction accounting and independent event replay only; no statistical validity or power claim",
        "traces": traces,
        "events_replayed": total_events,
        "negative_controls_rejected": rejected,
        "result": "all randomized traces and negative controls passed",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
