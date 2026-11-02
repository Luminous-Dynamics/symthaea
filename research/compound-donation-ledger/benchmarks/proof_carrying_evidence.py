#!/usr/bin/env python3
"""Proof-carrying Bernoulli e-process receipt prototype.

Verifies exact LR arithmetic, running maxima, A2 adjustment, atomic global
round batches, roster completeness relative to a precommitted schedule, and
SHA-256 hash chaining. Does not prove the real conditional law or source
attestation authenticity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
from fractions import Fraction as F


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def make_spec(n=12):
    candidates = [
        {
            "id": i,
            "p0": [1, 2],
            "p1": [7, 10],
            "arrival_round": (i // 3) * 2,
        }
        for i in range(n)
    ]
    spec = {
        "version": 1,
        "filtration_id": "global-round-batch-v1",
        "adjuster_id": "A2_SQRT_RUNNING_MAX_V1",
        "adjuster_scale": 1_000_000,
        "arrival_policy_digest": sha256({"kind": "fixed-block-arrivals", "block": 3, "delay": 2}),
        "null_assumption_digest": sha256({"claim": "conditional Bernoulli p0=1/2 given full global past"}),
        "source_completeness_attestation_digest": sha256({"attestation": "simulation generator declares full active roster"}),
        "candidates": candidates,
    }
    spec["spec_digest"] = sha256(spec)
    return spec


def validate_spec(spec):
    if spec["version"] != 1 or spec["adjuster_id"] != "A2_SQRT_RUNNING_MAX_V1":
        raise ValueError("unsupported spec")
    if spec["adjuster_scale"] <= 0:
        raise ValueError("invalid adjuster scale")
    ids = [c["id"] for c in spec["candidates"]]
    if ids != sorted(set(ids)):
        raise ValueError("candidate IDs must be strictly increasing")
    for c in spec["candidates"]:
        p0, p1 = F(*c["p0"]), F(*c["p1"])
        if not (0 < p0 < 1 and 0 < p1 < 1):
            raise ValueError("Bernoulli probabilities must lie in (0,1)")
    body = {k: v for k, v in spec.items() if k != "spec_digest"}
    if sha256(body) != spec["spec_digest"]:
        raise ValueError("spec digest mismatch")


def floor_a2(maximum: F, scale: int) -> int:
    root = math.isqrt((maximum.numerator * scale * scale) // maximum.denominator)
    return max(0, root - scale)


class EvidenceWriter:
    def __init__(self, spec):
        validate_spec(spec)
        self.spec = spec
        self.state = {
            c["id"]: {"lr": F(1), "sup": F(1), "adjusted": 0}
            for c in spec["candidates"]
        }
        self.previous_digest = "0" * 64
        self.previous_round = -1

    def append_round(self, round_no, observations):
        if round_no != self.previous_round + 1:
            raise ValueError("round sequence gap")
        active = [
            c["id"] for c in self.spec["candidates"]
            if c["arrival_round"] <= round_no
        ]
        if list(observations) != active:
            raise ValueError("observations must match sorted active roster")

        claims = []
        staged = {}
        for candidate in self.spec["candidates"]:
            i = candidate["id"]
            if i not in observations:
                continue
            p0, p1 = F(*candidate["p0"]), F(*candidate["p1"])
            factor = p1 / p0 if observations[i] == 1 else (1 - p1) / (1 - p0)
            lr = self.state[i]["lr"] * factor
            sup = max(self.state[i]["sup"], lr)
            adjusted = floor_a2(sup, self.spec["adjuster_scale"])
            staged[i] = {"lr": lr, "sup": sup, "adjusted": adjusted}
            claims.append({
                "candidate_id": i,
                "factor": [factor.numerator, factor.denominator],
                "lr": [lr.numerator, lr.denominator],
                "running_max": [sup.numerator, sup.denominator],
                "adjusted_integer": adjusted,
            })

        payload = {
            "version": 1,
            "filtration_id": self.spec["filtration_id"],
            "spec_digest": self.spec["spec_digest"],
            "round": round_no,
            "sequence": round_no + 1,
            "previous_batch_digest": self.previous_digest,
            "arrival_policy_digest": self.spec["arrival_policy_digest"],
            "source_completeness_attestation_digest": self.spec["source_completeness_attestation_digest"],
            "active_candidate_ids": active,
            "observations": [
                {"candidate_id": i, "value": observations[i]} for i in active
            ],
            "claims": claims,
        }
        receipt = dict(payload)
        receipt["batch_digest"] = sha256(payload)
        self.state.update(staged)
        self.previous_digest = receipt["batch_digest"]
        self.previous_round = round_no
        return receipt


def independent_verify(spec, receipts):
    """Independent replay; no call into EvidenceWriter or floor_a2."""
    validate_spec(spec)
    candidates = {c["id"]: c for c in spec["candidates"]}
    state = {
        i: {"product": F(1), "supremum": F(1)}
        for i in candidates
    }
    prior_digest = "0" * 64
    prior_round = -1
    updates = 0

    for batch in receipts:
        if batch["version"] != 1 or batch["filtration_id"] != spec["filtration_id"]:
            raise ValueError("version/filtration mismatch")
        if batch["spec_digest"] != spec["spec_digest"]:
            raise ValueError("spec precommitment mismatch")
        if batch["round"] != prior_round + 1 or batch["sequence"] != batch["round"] + 1:
            raise ValueError("round sequence rollback or gap")
        if batch["previous_batch_digest"] != prior_digest:
            raise ValueError("batch chain mismatch")
        if batch["arrival_policy_digest"] != spec["arrival_policy_digest"]:
            raise ValueError("arrival policy mismatch")
        if batch["source_completeness_attestation_digest"] != spec["source_completeness_attestation_digest"]:
            raise ValueError("source attestation reference mismatch")

        t = batch["round"]
        active = [i for i, c in candidates.items() if c["arrival_round"] <= t]
        if batch["active_candidate_ids"] != active:
            raise ValueError("active roster mismatch")
        observations = batch["observations"]
        if [row["candidate_id"] for row in observations] != active:
            raise ValueError("partial, duplicated or unordered round batch")

        expected_claims = []
        staged = {}
        for row in observations:
            i, y = row["candidate_id"], row["value"]
            if y not in (0, 1):
                raise ValueError("nonbinary observation")
            c = candidates[i]
            p0, p1 = F(*c["p0"]), F(*c["p1"])
            # Derive the likelihood ratio independently from the model contract.
            if y:
                e_factor = F(p1.numerator * p0.denominator, p1.denominator * p0.numerator)
            else:
                # (1-p1)/(1-p0), expanded from the registered integer pairs.
                e_factor = F(
                    (p1.denominator - p1.numerator) * p0.denominator,
                    p1.denominator * (p0.denominator - p0.numerator),
                )
            product = state[i]["product"] * e_factor
            supremum = max(state[i]["supremum"], product)
            scale = spec["adjuster_scale"]
            root = math.isqrt((supremum.numerator * scale * scale) // supremum.denominator)
            adjusted = max(0, root - scale)
            staged[i] = {"product": product, "supremum": supremum}
            expected_claims.append({
                "candidate_id": i,
                "factor": [e_factor.numerator, e_factor.denominator],
                "lr": [product.numerator, product.denominator],
                "running_max": [supremum.numerator, supremum.denominator],
                "adjusted_integer": adjusted,
            })
            updates += 1

        if batch["claims"] != expected_claims:
            raise ValueError("factor/product/supremum/adjuster mismatch")
        payload = {k: v for k, v in batch.items() if k != "batch_digest"}
        expected_hash = hashlib.sha256(canonical(payload)).hexdigest()
        if batch["batch_digest"] != expected_hash:
            raise ValueError("batch digest mismatch")

        state.update(staged)
        prior_digest = expected_hash
        prior_round = t

    return {
        "verified": True,
        "rounds": len(receipts),
        "evidence_updates": updates,
        "final_batch_digest": prior_digest,
        "trust_boundary": (
            "Arithmetic and envelope consistency verified. The conditional-null "
            "law, real filtration completeness and source-attestation authenticity "
            "remain external assumptions."
        ),
    }


def generate_round(spec, round_no, dependence, rng):
    active = [
        c["id"] for c in spec["candidates"]
        if c["arrival_round"] <= round_no
    ]
    if dependence == "independent":
        return {i: rng.randrange(2) for i in active}
    if dependence == "shared_uniform":
        u = rng.random()
        return {i: int(u < 0.5) for i in active}
    if dependence == "balanced_negative":
        count = len(active)
        successes = count // 2 + (rng.randrange(2) if count % 2 else 0)
        shuffled = list(active)
        rng.shuffle(shuffled)
        selected = set(shuffled[:successes])
        return {i: int(i in selected) for i in active}
    raise ValueError("unknown dependence")


def build_trace(spec, rounds, dependence, seed):
    rng = random.Random(seed)
    writer = EvidenceWriter(spec)
    receipts = []
    for t in range(rounds):
        receipts.append(writer.append_round(
            t, generate_round(spec, t, dependence, rng)
        ))
    return receipts


def negative_controls(spec, receipts):
    results = {}
    def rejected(mutator):
        altered = copy.deepcopy(receipts)
        mutator(altered)
        try:
            independent_verify(spec, altered)
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            return True
        return False

    results["observation_bit_stale_digest"] = rejected(
        lambda r: r[0]["observations"][0].update(
            value=1-r[0]["observations"][0]["value"]
        )
    )
    results["adjusted_value_tamper"] = rejected(
        lambda r: r[0]["claims"][0].update(
            adjusted_integer=r[0]["claims"][0]["adjusted_integer"]+1
        )
    )
    results["previous_hash_tamper"] = rejected(
        lambda r: r[1].update(previous_batch_digest="f"*64)
    )
    results["roster_omission"] = rejected(
        lambda r: r[1]["active_candidate_ids"].pop()
    )
    results["round_rollback"] = rejected(
        lambda r: r[2].update(round=1, sequence=2)
    )
    results["observation_order"] = rejected(
        lambda r: r[1]["observations"].reverse()
    )
    results["spec_digest_tamper"] = rejected(
        lambda r: r[0].update(spec_digest="0"*64)
    )
    return results


if __name__ == "__main__":
    traces, candidates, rounds = 100, 12, 30
    spec = make_spec(candidates)
    timings = []
    total_updates = 0
    for trace in range(traces):
        dep = ("independent", "shared_uniform", "balanced_negative")[trace % 3]
        receipts = build_trace(spec, rounds, dep, 10146+trace)
        start = __import__("time").perf_counter()
        verified = independent_verify(spec, receipts)
        timings.append(__import__("time").perf_counter()-start)
        total_updates += verified["evidence_updates"]
    first = build_trace(spec, rounds, "independent", 10146)
    controls = negative_controls(spec, first)
    if not all(controls.values()):
        raise AssertionError("tamper control was accepted")
    print(json.dumps({
        "traces": traces,
        "candidates": candidates,
        "rounds_per_trace": rounds,
        "updates_replayed": total_updates,
        "verified_traces": traces,
        "median_verifier_seconds": statistics.median(timings),
        "tamper_controls": controls,
        "all_tamper_controls_rejected": all(controls.values()),
        "claim_boundary": "Arithmetic and declared-contract verification only; not proof of real conditional law or source authenticity.",
    }, indent=2, sort_keys=True))
