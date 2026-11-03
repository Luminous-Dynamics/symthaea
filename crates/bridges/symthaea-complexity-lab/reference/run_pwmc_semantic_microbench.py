#!/usr/bin/env python3
"""Exhaustive exact PWMC micro-oracle. This is deliberately not a scalable solver."""
from __future__ import annotations
import hashlib, itertools, json, pathlib
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
INPUT = HERE / "pwmc_semantic_microbench_v1.json"
OUTPUT = HERE / "pwmc_semantic_microbench_results.v1.json"

def q(text: str) -> Fraction:
    return Fraction(text)

def sat(clauses, bits):
    return all(any((bits[abs(lit)-1] == 1) == (lit > 0) for lit in clause) for clause in clauses)

def project(case):
    n = len(case["variables"])
    visible_ids = [case["variables"].index(v) for v in case["visible"]]
    witnesses, full_models = {}, 0
    for bits in itertools.product((0, 1), repeat=n):
        if not sat(case["clauses"], bits):
            continue
        full_models += 1
        key = "".join(str(bits[i]) for i in visible_ids)
        witnesses[key] = witnesses.get(key, 0) + 1
    return full_models, witnesses

def weighted_support(support, visible, weights):
    total = Fraction(0, 1)
    for key in support:
        term = Fraction(1, 1)
        for offset, variable in enumerate(visible):
            term *= q(weights[variable]["true" if key[offset] == "1" else "false"])
        total += term
    return total

def evaluate(case):
    full_models, witnesses = project(case)
    support = sorted(witnesses)
    total = weighted_support(support, case["visible"], case["weights"])
    return {"full_model_count":str(full_models),"projected_support":support,
            "witness_counts":{k:str(witnesses[k]) for k in support},
            "support_cardinality":str(len(support)),
            "pwmc":f"{total.numerator}/{total.denominator}"}

def main():
    raw = INPUT.read_bytes()
    corpus = json.loads(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    results = []
    for case in corpus["cases"]:
        actual, expected = evaluate(case), case["expected"]
        for key in ("full_model_count","projected_support","witness_counts","pwmc"):
            if actual[key] != expected[key]:
                raise SystemExit(f"FAIL {case['id']} {key}: expected {expected[key]!r}, got {actual[key]!r}")
        results.append({"case_id":case["id"],**actual,"status":"exact_oracle_match"})
        print(f"PASS {case['id']}: models={actual['full_model_count']} support={actual['support_cardinality']} PWMC={actual['pwmc']}")
    by_id={r["case_id"]:r for r in results}
    small,large=by_id["single-hidden-witness-scope"],by_id["twelve-hidden-free-witnesses"]
    assert small["projected_support"] == large["projected_support"]
    assert small["pwmc"] == large["pwmc"]
    assert int(large["full_model_count"]) == 2048 * int(small["full_model_count"])
    assert by_id["nonempty-support-exact-cancellation"]["support_cardinality"] == "2"
    assert by_id["nonempty-support-exact-cancellation"]["pwmc"] == "0/1"
    reuse=corpus["reusable_projection"]
    _,witnesses=project(reuse)
    support=sorted(witnesses)
    assert support == reuse["expected_projected_support"]
    assert {k:str(v) for k,v in sorted(witnesses.items())} == reuse["expected_witness_counts"]
    reuse_results=[]
    for weight_map in reuse["weight_maps"]:
        value=weighted_support(support,reuse["visible"],weight_map["weights"])
        actual=f"{value.numerator}/{value.denominator}"
        assert actual == weight_map["expected_pwmc"], (weight_map["id"],actual,weight_map["expected_pwmc"])
        _,replay_witnesses=project(reuse)
        replay_support=sorted(replay_witnesses)
        replay_value=weighted_support(replay_support,reuse["visible"],weight_map["weights"])
        assert replay_support == support and replay_value == value
        reuse_results.append({"weight_map_id":weight_map["id"],"pwmc":actual,"independent_projection_replay":"matched"})
    print(f"PASS reusable support: one retained support, {len(reuse_results)} exact weight maps, all independent replays matched")
    output={"format":"symthaea.pwmc-semantic-microbench-results","version":1,
      "input_sha256":digest,"method":"exhaustive full-assignment enumeration and exact Fraction arithmetic",
      "claim_ceiling":"finite_reference_oracle_not_solver_performance","case_count":len(results),
      "hidden_witness_invariance":"passed","exact_zero_nonempty_support":"passed",
      "reusable_projection":{"support_compilations_in_reuse_path":1,"weight_map_evaluations":len(reuse_results),
        "independent_exhaustive_replays":len(reuse_results),"retained_support":support,"results":reuse_results},
      "results":results}
    OUTPUT.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PASS all {len(results)} PWMC semantic microbench cases; input={digest}")

if __name__ == "__main__":
    main()
