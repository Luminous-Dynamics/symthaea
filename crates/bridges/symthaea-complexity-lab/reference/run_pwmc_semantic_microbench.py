#!/usr/bin/env python3
"""PWMC semantic microbench: existential-support producer vs full-model replay."""
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

def project_full_assignments(case):
    """Independent verifier path: enumerate the entire universe, then quotient by V."""
    visible_ids = [case["variables"].index(v) for v in case["visible"]]
    witnesses, full_models = {}, 0
    for bits in itertools.product((0, 1), repeat=len(case["variables"])):
        if not sat(case["clauses"], bits):
            continue
        full_models += 1
        key = "".join(str(bits[i]) for i in visible_ids)
        witnesses[key] = witnesses.get(key, 0) + 1
    return full_models, witnesses

def compile_existential_support(case):
    """Producer path: enumerate visible assignments and stop at the first hidden witness."""
    visible, hidden = case["visible"], case["hidden"]
    visible_ids = [case["variables"].index(v) for v in visible]
    hidden_ids = [case["variables"].index(v) for v in hidden]
    clauses, support = case["clauses"], []
    for vbits in itertools.product((0, 1), repeat=len(visible)):
        assignment = {idx: bit for idx, bit in zip(visible_ids, vbits)}
        found = False
        for hbits in itertools.product((0, 1), repeat=len(hidden)):
            assignment.update(zip(hidden_ids, hbits))
            if all(any((assignment[abs(lit)-1] == 1) == (lit > 0) for lit in clause) for clause in clauses):
                found = True
                break
        if found:
            support.append("".join(str(bit) for bit in vbits))
    return support

def weighted_support(support, visible, weights):
    total = Fraction(0, 1)
    for key in support:
        term = Fraction(1, 1)
        for offset, variable in enumerate(visible):
            term *= q(weights[variable]["true" if key[offset] == "1" else "false"])
        total += term
    return total

def evaluate(case):
    full_models, witnesses = project_full_assignments(case)
    support = sorted(witnesses)
    total = weighted_support(support, case["visible"], case["weights"])
    return {"full_model_count":str(full_models),"projected_support":support,
            "witness_counts":{k:str(witnesses[k]) for k in support},
            "support_cardinality":str(len(support)),
            "pwmc":f"{total.numerator}/{total.denominator}"}

def run_metamorphic_tests(corpus):
    cases = {case["id"]: case for case in corpus["cases"]}
    reuse = corpus["reusable_projection"]
    output = []
    for test in corpus["metamorphic_tests"]:
        op = test["operation"]
        source = json.loads(json.dumps(reuse if test["source"] == "reusable_projection" else cases[test["source"]]))
        if test["source"] == "reusable_projection":
            source["weights"] = reuse["weight_maps"][0]["weights"]
        base = evaluate(source)
        transformed = json.loads(json.dumps(source))
        if op == "add_free_hidden":
            transformed["variables"].append("h_added")
            transformed["hidden"].append("h_added")
        elif op == "permute_variable_order":
            new_order = ["h", "u", "d"]
            old = transformed["variables"]
            new_index = {name: i + 1 for i, name in enumerate(new_order)}
            transformed["clauses"] = [[(1 if lit > 0 else -1) * new_index[old[abs(lit)-1]] for lit in clause] for clause in transformed["clauses"]]
            transformed["variables"] = new_order
        elif op == "duplicate_clause_and_literal":
            clause = transformed["clauses"][0]
            transformed["clauses"] = [clause + [clause[0]], list(reversed(clause))]
        elif op == "add_subsumed_clause":
            transformed["clauses"].append([1, 2])
        elif op == "add_free_visible":
            transformed["variables"].append("x")
            transformed["visible"].append("x")
            transformed["weights"]["x"] = {"false":"2/1", "true":"3/1"}
        elif op == "move_hidden_to_visible":
            variable = transformed["hidden"].pop()
            transformed["visible"].append(variable)
            transformed["weights"][variable] = {"false":"1/1", "true":"1/1"}
        else:
            raise AssertionError(f"unknown metamorphic operation: {op}")
        actual = evaluate(transformed)
        expected = test["expected"]
        if op == "add_free_hidden":
            assert actual["projected_support"] == base["projected_support"]
            assert actual["pwmc"] == base["pwmc"]
            assert int(actual["full_model_count"]) == int(base["full_model_count"]) * int(expected["full_model_multiplier"])
        elif op in {"permute_variable_order", "duplicate_clause_and_literal", "add_subsumed_clause"}:
            for field in ("full_model_count", "projected_support", "pwmc"):
                assert actual[field] == base[field], (test["id"], field, actual[field], base[field])
        elif op == "add_free_visible":
            assert int(actual["full_model_count"]) == int(base["full_model_count"]) * int(expected["full_model_multiplier"])
            assert actual["support_cardinality"] == expected["support_cardinality"]
            assert actual["pwmc"] == expected["pwmc"]
        elif op == "move_hidden_to_visible":
            assert actual["support_cardinality"] == expected["support_cardinality"]
            assert actual["pwmc"] == expected["pwmc"]
        output.append({"test_id":test["id"],"operation":op,"status":"exact_metamorphic_relation_passed",
                       "base_full_model_count":base["full_model_count"],"transformed_full_model_count":actual["full_model_count"],
                       "base_support_cardinality":base["support_cardinality"],"transformed_support_cardinality":actual["support_cardinality"],
                       "base_pwmc":base["pwmc"],"transformed_pwmc":actual["pwmc"]})
        print(f"PASS metamorphic {test['id']}: {base['pwmc']} -> {actual['pwmc']}")
    return output

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
    small,large=by_id["single-hidden-witness-scope"],by_id["twenty-hidden-free-witnesses"]
    assert small["projected_support"] == large["projected_support"]
    assert small["pwmc"] == large["pwmc"]
    assert int(large["full_model_count"]) == 524288 * int(small["full_model_count"])
    assert by_id["nonempty-support-exact-cancellation"]["support_cardinality"] == "2"
    assert by_id["nonempty-support-exact-cancellation"]["pwmc"] == "0/1"

    reuse=corpus["reusable_projection"]
    retained_support=compile_existential_support(reuse)
    assert retained_support == reuse["expected_projected_support"]
    replay_results=[]
    for weight_map in reuse["weight_maps"]:
        cached_value=weighted_support(retained_support,reuse["visible"],weight_map["weights"])
        cached_text=f"{cached_value.numerator}/{cached_value.denominator}"
        assert cached_text == weight_map["expected_pwmc"], (weight_map["id"],cached_text,weight_map["expected_pwmc"])
        # Verifier uses a separate full-assignment traversal and derives support afresh.
        _, replay_witnesses=project_full_assignments(reuse)
        replay_support=sorted(replay_witnesses)
        replay_value=weighted_support(replay_support,reuse["visible"],weight_map["weights"])
        replay_text=f"{replay_value.numerator}/{replay_value.denominator}"
        assert replay_support == retained_support and replay_text == cached_text
        replay_results.append({"weight_map_id":weight_map["id"],"pwmc":cached_text,"independent_projection_replay":"matched"})
    print("PASS independent projection: visible-first existential producer vs full-assignment verifier")
    print(f"PASS reusable support: one retained support, {len(replay_results)} exact weight maps, all independent replays matched")

    metamorphic_results = run_metamorphic_tests(corpus)
    output={"format":"symthaea.pwmc-semantic-microbench-results","version":1,
      "input_sha256":digest,"method":"visible-first existential support compilation; independent full-assignment exact replay; Fraction arithmetic",
      "claim_ceiling":"finite_reference_oracle_not_solver_performance","case_count":len(results),
      "hidden_witness_invariance":"passed","exact_zero_nonempty_support":"passed",
      "projection_algorithms":{"producer":"visible_assignment_then_existential_hidden_search_with_early_exit",
        "verifier":"full_universe_assignment_enumeration_then_visible_projection","agreement":"exact"},
      "reusable_projection":{"support_compilations_in_reuse_path":1,"weight_map_evaluations":len(replay_results),
        "independent_exhaustive_replays":len(replay_results),"retained_support":retained_support,"results":replay_results},
      "results":results,"metamorphic_test_count":len(metamorphic_results),"metamorphic_tests":metamorphic_results}
    OUTPUT.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PASS all {len(results)} PWMC semantic microbench cases; input={digest}")

if __name__ == "__main__":
    main()
