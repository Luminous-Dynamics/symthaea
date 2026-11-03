"""Finite reference experiments for nested projected-support circuits.

This is an exhaustive test harness, not a production knowledge compiler or proof checker.
"""
from fractions import Fraction as F
from itertools import product
from dataclasses import dataclass
from typing import Any
import random
import json
import platform
import time
import collections

# Bottom-up exact weighted evaluation of a small d-DNNF-like circuit with explicit scope completion.
def z(domain):
    if domain == "q":
        return F(0)
    if domain == "qi":
        return (F(0), F(0))
    raise ValueError(f"unsupported coefficient domain: {domain}")

def one(domain):
    if domain == "q":
        return F(1)
    if domain == "qi":
        return (F(1), F(0))
    raise ValueError(f"unsupported coefficient domain: {domain}")
def addv(domain, a, b):
    if domain == "q": return a+b
    return (a[0]+b[0], a[1]+b[1])
def mulv(domain, a, b):
    if domain == "q": return a*b
    return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
def sum_weights(domain, pair):
    return addv(domain, pair[0], pair[1])

def fmtv(domain, a):
    if domain == "q":
        return str(a)
    if a[1] == 0:
        return str(a[0])
    return f"{a[0]}{'+' if a[1] >= 0 else ''}{a[1]}i"

class InvalidCircuit(ValueError):
    pass

@dataclass
class Circuit:
    variables: tuple[str, ...]
    nodes: list[dict[str, Any]]
    root: str

    def _maps_scopes(self):
        by = {}
        scopes = {}
        for node in self.nodes:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                raise InvalidCircuit("node/id")
            node_id = node["id"]
            if node_id in by:
                raise InvalidCircuit("duplicate node id")
            kind = node.get("kind")
            if kind == "lit":
                if set(node) != {"id", "kind", "var", "value"}:
                    raise InvalidCircuit("literal shape")
                if node["var"] not in self.variables or type(node["value"]) is not bool:
                    raise InvalidCircuit("literal fields")
                scopes[node_id] = {node["var"]}
            elif kind == "const":
                if set(node) != {"id", "kind", "value"} or type(node["value"]) is not bool:
                    raise InvalidCircuit("constant shape")
                scopes[node_id] = set()
            elif kind in {"and", "or"}:
                if set(node) != {"id", "kind", "children"}:
                    raise InvalidCircuit(f"{kind} shape")
                children = node["children"]
                if not isinstance(children, list) or not children:
                    raise InvalidCircuit(f"{kind} children")
                if any(not isinstance(c, str) or c not in scopes for c in children):
                    raise InvalidCircuit(f"{kind} forward/unknown edge")
                child_scopes = [scopes[c] for c in children]
                if kind == "and":
                    flat = [v for s in child_scopes for v in s]
                    if len(flat) != len(set(flat)):
                        raise InvalidCircuit("nondecomposable product")
                scopes[node_id] = set().union(*child_scopes)
            else:
                raise InvalidCircuit("unknown node kind")
            by[node_id] = node
        if self.root not in by:
            raise InvalidCircuit("unknown root")
        if not isinstance(self.variables, tuple) or len(set(self.variables)) != len(self.variables):
            raise InvalidCircuit("variable universe")
        if not scopes[self.root] <= set(self.variables):
            raise InvalidCircuit("root scope")
        return by, scopes

    def eval_bool(self, node_id, assignment, by=None, memo=None):
        if by is None:
            by, _ = self._maps_scopes()
        if memo is None:
            memo = {}
        if node_id in memo:
            return memo[node_id]
        node = by[node_id]
        k = node["kind"]
        if k == "lit":
            out = assignment[node["var"]] is node["value"]
        elif k == "const":
            out = node["value"]
        elif k == "and":
            out = all(self.eval_bool(c, assignment, by, memo) for c in node["children"])
        else:
            out = any(self.eval_bool(c, assignment, by, memo) for c in node["children"])
        memo[node_id] = out
        return out

    def validate(self, check_determinism=True):
        if not isinstance(self.variables, tuple) or len(set(self.variables)) != len(self.variables):
            raise InvalidCircuit("variable universe")
        by, scopes = self._maps_scopes()
        if check_determinism:
            for node in self.nodes:
                if node["kind"] != "or":
                    continue
                children = node["children"]
                union = sorted(scopes[node["id"]])
                for bits in product((False, True), repeat=len(union)):
                    assignment = dict(zip(union, bits))
                    vals = [self.eval_bool(c, assignment, by, {}) for c in children]
                    if sum(vals) > 1:
                        raise InvalidCircuit("nondeterministic sum")
        return scopes

    def weighted(self, weights, domain="q"):
        if domain not in {"q", "qi"}:
            raise ValueError(f"unsupported coefficient domain: {domain}")
        scopes = self.validate(check_determinism=True)
        weights = weights or {}
        for var, pair in weights.items():
            if var not in self.variables or not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise InvalidCircuit("invalid literal-weight entry")
            if domain == "q":
                if any(not isinstance(c, F) for c in pair):
                    raise InvalidCircuit("Q weights must be exact Fractions")
            else:
                if any(not isinstance(part, (tuple, list)) or len(part) != 2 for part in pair):
                    raise InvalidCircuit("Q(i) weights must be pairs of exact Fractions")
                coefficients = [c for part in pair for c in part]
                if any(not isinstance(c, F) for c in coefficients):
                    raise InvalidCircuit("Q(i) weights must be pairs of exact Fractions")
        values = {}
        for node in self.nodes:
            node_id, kind = node["id"], node["kind"]
            if kind == "const":
                values[node_id] = one(domain) if node["value"] else z(domain)
            elif kind == "lit":
                pair = weights.get(node["var"], (one(domain), one(domain)))
                values[node_id] = pair[1 if node["value"] else 0]
            elif kind == "and":
                acc = one(domain)
                for child in node["children"]:
                    acc = mulv(domain, acc, values[child])
                values[node_id] = acc
            else:
                parent_scope = scopes[node_id]
                acc = z(domain)
                for child in node["children"]:
                    child_scope = scopes[child]
                    completion = one(domain)
                    for var in parent_scope - child_scope:
                        pair = weights.get(var, (one(domain), one(domain)))
                        completion = mulv(domain, completion, sum_weights(domain, pair))
                    contribution = mulv(domain, values[child], completion)
                    acc = addv(domain, acc, contribution)
                values[node_id] = acc
        root_scope = scopes[self.root]
        completion = one(domain)
        for var in set(self.variables) - root_scope:
            pair = weights.get(var, (one(domain), one(domain)))
            completion = mulv(domain, completion, sum_weights(domain, pair))
        result = mulv(domain, values[self.root], completion)
        return result, values, scopes

def compile_truth_table(variables, table):
    """Compile a truth table to a shared decision-DNNF via Shannon decomposition."""
    variables = tuple(variables)
    nodes = []
    cache = {}
    counter = 0

    def emit(obj):
        nonlocal counter
        node = {"id": f"n{counter}", **obj}
        counter += 1
        nodes.append(node)
        return node["id"]

    def rec(vs, subtable):
        nonlocal cache
        keys = list(product((False, True), repeat=len(vs)))
        signature = (vs, tuple(subtable[k] for k in keys))
        if signature in cache:
            return cache[signature]
        vals = [subtable[k] for k in keys]
        if all(v == vals[0] for v in vals):
            result = emit({"kind": "const", "value": bool(vals[0])})
            cache[signature] = result
            return result
        if not vs:
            raise AssertionError("empty nonconstant table")
        var, rest = vs[0], vs[1:]
        t0 = {row[1:]: subtable[row] for row in keys if row[0] is False}
        t1 = {row[1:]: subtable[row] for row in keys if row[0] is True}
        c0 = rec(rest, t0)
        c1 = rec(rest, t1)
        if c0 == c1:
            cache[signature] = c0
            return c0
        l0 = emit({"kind": "lit", "var": var, "value": False})
        p0 = emit({"kind": "and", "children": [l0, c0]})
        l1 = emit({"kind": "lit", "var": var, "value": True})
        p1 = emit({"kind": "and", "children": [l1, c1]})
        out = emit({"kind": "or", "children": [p0, p1]})
        cache[signature] = out
        return out

    root = rec(variables, table)
    return Circuit(variables, nodes, root)

def eval_oracle(variables, table, weights, domain):
    total = z(domain)
    support = 0
    for bits in product((False, True), repeat=len(variables)):
        if not table[bits]:
            continue
        support += 1
        term = one(domain)
        for var, state in zip(variables, bits):
            pair = weights.get(var, (one(domain), one(domain)))
            term = mulv(domain, term, pair[1 if state else 0])
        total = addv(domain, total, term)
    return total, support

def rand_q(rng):
    return F(rng.randint(-3, 3), rng.randint(1, 5))
def rand_qi(rng):
    return (rand_q(rng), rand_q(rng))

def random_table(rng, variables):
    return {bits: bool(rng.getrandbits(1)) for bits in product((False, True), repeat=len(variables))}

def count_structure(c):
    parents = collections.Counter()
    for n in c.nodes:
        if n["kind"] in ("and", "or"):
            parents.update(n["children"])
    scopes = c.validate()
    nonsmooth = sum(
        1 for n in c.nodes if n["kind"] == "or"
        if any(scopes[ch] != scopes[n["id"]] for ch in n["children"])
    )
    shared = sum(1 for node_id, count in parents.items() if count > 1)
    return nonsmooth, shared, len(c.nodes)

def run_nested_suite(seed=0xE7027, cases_per_contract=1000):
    rng = random.Random(seed)
    contracts = ["#sat", "wmc", "pmc", "pwmc", "amc-complex"]
    passed = {m: 0 for m in contracts}
    failures = []
    structural = {"non_smooth_or_nodes": 0, "shared_nodes_referenced_by_multiple_parents": 0, "circuit_nodes": 0}
    exact_zero = {m: 0 for m in contracts}
    zero_with_nonempty_support = {m: 0 for m in contracts}
    start = time.perf_counter()
    max_vars = 0
    for mode in contracts:
        for case_idx in range(cases_per_contract):
            if mode in ("pmc", "pwmc"):
                nv = rng.randint(1, 4)
                nh = rng.randint(1, 3)
                visible = tuple(f"d{i}" for i in range(nv))
                hidden = tuple(f"h{i}" for i in range(nh))
                full_vars = visible + hidden
                full_table = random_table(rng, full_vars)
                visible_table = {}
                for dbits in product((False, True), repeat=nv):
                    visible_table[dbits] = any(
                        full_table[dbits + hbits]
                        for hbits in product((False, True), repeat=nh)
                    )
                c = compile_truth_table(visible, visible_table)
                domain = "q"
                if mode == "pmc":
                    weights = {}
                else:
                    weights = {v: (rand_q(rng), rand_q(rng)) for v in visible}
                expected, support = eval_oracle(visible, visible_table, weights, domain)
            else:
                n = rng.randint(1, 6)
                visible = tuple(f"x{i}" for i in range(n))
                full_vars = visible
                full_table = random_table(rng, full_vars)
                c = compile_truth_table(full_vars, full_table)
                domain = "qi" if mode == "amc-complex" else "q"
                weights = {}
                if mode == "wmc":
                    weights = {v: (rand_q(rng), rand_q(rng)) for v in full_vars}
                elif mode == "amc-complex":
                    weights = {v: (rand_qi(rng), rand_qi(rng)) for v in full_vars}
                expected, support = eval_oracle(full_vars, full_table, weights, domain)
            actual, vals, scopes = c.weighted(weights, domain)
            max_vars = max(max_vars, len(full_vars))
            if actual == expected:
                passed[mode] += 1
            else:
                failures.append({"contract": mode, "case": case_idx, "expected": fmtv(domain, expected), "actual": fmtv(domain, actual)})
            if expected == z(domain):
                exact_zero[mode] += 1
                if support > 0:
                    zero_with_nonempty_support[mode] += 1
            ns, sh, nn = count_structure(c)
            structural["non_smooth_or_nodes"] += ns
            structural["shared_nodes_referenced_by_multiple_parents"] += sh
            structural["circuit_nodes"] += nn
    # Hidden-witness invariance on existential support: adding an unconstrained hidden variable.
    witness_checks = 0
    for case_idx in range(500):
        nv = rng.randint(1, 4)
        nh = rng.randint(1, 3)
        dvars = tuple(f"d{i}" for i in range(nv))
        hvars = tuple(f"h{i}" for i in range(nh))
        fv = dvars + hvars
        t = random_table(rng, fv)
        proj = {}
        for db in product((False, True), repeat=nv):
            proj[db] = any(t[db + hb] for hb in product((False, True), repeat=nh))
        # Add a fresh hidden variable z while leaving the original relation independent of it.
        fv2 = fv + ("hz",)
        t2 = {}
        for allbits in product((False, True), repeat=len(fv2)):
            t2[allbits] = t[allbits[:-1]]
        proj2 = {}
        for db in product((False, True), repeat=nv):
            proj2[db] = any(t2[db + hb] for hb in product((False, True), repeat=nh+1))
        if proj == proj2:
            witness_checks += 1
        else:
            failures.append({"case": case_idx, "error": "hidden witness altered projected support"})
    elapsed = time.perf_counter() - start
    return {
        "suite": "ETH-027 nested projected-support / bottom-up exact evaluation prototype",
        "seed": hex(seed),
        "cases_per_contract": cases_per_contract,
        "contracts": contracts,
        "cases": cases_per_contract * len(contracts),
        "passed_by_contract": passed,
        "passed": sum(passed.values()),
        "failed": len(failures),
        "failure_examples": failures[:10],
        "hidden_witness_invariance_cases": witness_checks,
        "exact_zero_by_contract": exact_zero,
        "zero_with_nonempty_support_by_contract": zero_with_nonempty_support,
        "structural_coverage": structural,
        "maximum_input_variables": max_vars,
        "elapsed_seconds": round(elapsed, 6),
        "claim_ceiling": "Locally executed research prototype: Shannon-compiled nested deterministic/decomposable DAGs, explicit non-smooth scope completion, exact Q/Q(i) bottom-up evaluation, and direct truth-table oracle. Not SCPOG proof verification, source CNF compiler, Rust implementation, or solver differential."
    }

def run_controls():
    results={}
    def weighted_without_completion(circuit, weights):
        by, scopes = circuit._maps_scopes()
        vals = {}
        for node in circuit.nodes:
            node_id, kind = node["id"], node["kind"]
            if kind == "const":
                vals[node_id] = one("q") if node["value"] else z("q")
            elif kind == "lit":
                pair = weights.get(node["var"], (one("q"), one("q")))
                vals[node_id] = pair[1 if node["value"] else 0]
            elif kind == "and":
                acc = one("q")
                for child in node["children"]:
                    acc = mulv("q", acc, vals[child])
                vals[node_id] = acc
            else:
                acc = z("q")
                for child in node["children"]:
                    acc = addv("q", acc, vals[child])
                vals[node_id] = acc
        return vals[circuit.root]

    # Non-smooth OR: x OR y. The x=true branch omits y from its scope.
    variables=("x","y")
    table={(x,y):(x or y) for x,y in product((False,True),repeat=2)}
    circuit=compile_truth_table(variables,table)
    weights={"x":(F(2,3),F(5,7)),"y":(F(-1,2),F(3,4))}
    actual,_,scopes=circuit.weighted(weights,"q")
    expected,support=eval_oracle(variables,table,weights,"q")
    results["non_smooth_scope_completion"]={
        "exact_output":str(actual),"oracle":str(expected),"support":support,
        "has_non_smooth_sum":any(n["kind"]=="or" and any(scopes[ch]!=scopes[n["id"]] for ch in n["children"]) for n in circuit.nodes)
    }
    assert actual==expected and results["non_smooth_scope_completion"]["has_non_smooth_sum"]

    nonsmooth_mutant = weighted_without_completion(circuit, weights)
    results["non_smooth_completion_mutant_detected"] = {
        "correct": str(actual),
        "mutant_without_scope_completion": str(nonsmooth_mutant),
        "detected": nonsmooth_mutant != expected
    }
    assert nonsmooth_mutant != expected

    # Root omits free variables; their literal-weight sums must be included.
    variables=("x","free1","free2")
    table={(x,y,z):x for x,y,z in product((False,True),repeat=3)}
    circuit=compile_truth_table(variables,table)
    weights={"x":(F(2),F(3)),"free1":(F(5),F(7)),"free2":(F(11),F(13))}
    actual,_,scopes=circuit.weighted(weights,"q")
    expected,support=eval_oracle(variables,table,weights,"q")
    results["free_variable_completion"]={
        "root_scope":sorted(scopes[circuit.root]),
        "exact_output":str(actual),"oracle":str(expected),"support":support
    }
    assert actual==expected and set(scopes[circuit.root])=={"x"}

    # Exact signed and Gaussian-rational cancellation with nonempty support.
    v=("x",)
    t={(False,):True,(True,):True}
    c=compile_truth_table(v,t)
    q_weights={"x":(F(1),F(-1))}
    q_value,_,_=c.weighted(q_weights,"q")
    _,q_support=eval_oracle(v,t,q_weights,"q")
    qi_weights={"x":((F(1),F(2)),(F(-1),F(-2)))}
    qi_value,_,_=c.weighted(qi_weights,"qi")
    _,qi_support=eval_oracle(v,t,qi_weights,"qi")
    results["signed_q_cancellation"]={"exact_output":str(q_value),"support":q_support}
    results["complex_qi_cancellation"]={"exact_output":fmtv("qi",qi_value),"support":qi_support}
    assert q_value==0 and qi_value==z("qi") and q_support==2 and qi_support==2


    # Mutation control: omitting root free-variable completion must be detected.
    omitted_completion = weighted_without_completion(circuit, weights)
    results["free_variable_completion_mutant_detected"] = {
        "correct": str(actual), "mutant_without_completion": str(omitted_completion),
        "detected": omitted_completion != expected
    }
    assert omitted_completion != expected

    # Mutation control: adding over hidden witnesses is not existential projection.
    full_vars = ("d", "h")
    full_table = {
        (False, False): False, (False, True): False,
        (True, False): True, (True, True): True
    }
    projected_table = {(False,): False, (True,): True}
    projected_circuit = compile_truth_table(("d",), projected_table)
    visible_weights = {"d": (F(1, 3), F(2, 3))}
    correct_pw, _, _ = projected_circuit.weighted(visible_weights, "q")
    naive_hidden_sum, _ = eval_oracle(
        full_vars, full_table,
        {"d": (F(1, 3), F(2, 3)), "h": (F(1), F(1))}, "q"
    )
    results["hidden_witness_multiplier_mutant_detected"] = {
        "correct_pwmc": str(correct_pw),
        "wrong_hidden_witness_sum": str(naive_hidden_sum),
        "detected": correct_pw != naive_hidden_sum
    }
    assert correct_pw == F(2, 3) and naive_hidden_sum == F(4, 3)

    # Malformed product with overlapping scope.
    bad_product=Circuit(("x",),[
        {"id":"a","kind":"lit","var":"x","value":True},
        {"id":"b","kind":"lit","var":"x","value":False},
        {"id":"p","kind":"and","children":["a","b"]}], "p")
    try: bad_product.validate()
    except InvalidCircuit: results["overlapping_product_rejected"]=True
    else: raise AssertionError("overlapping product accepted")

    # Reject binary floating-point weights in an exact Q evaluation.
    try:
        c.weighted({"x": (0.5, 0.5)}, "q")
    except InvalidCircuit:
        results["inexact_float_weights_rejected"] = True
    else:
        raise AssertionError("inexact floating-point weights accepted")

    # False determinism: both OR branches denote x.
    bad_or=Circuit(("x",),[
        {"id":"a","kind":"lit","var":"x","value":True},
        {"id":"b","kind":"lit","var":"x","value":True},
        {"id":"o","kind":"or","children":["a","b"]}], "o")
    try: bad_or.validate()
    except InvalidCircuit: results["false_determinism_rejected"]=True
    else: raise AssertionError("nondeterministic OR accepted")
    return results


def main():
    result = run_nested_suite()
    result["controls"] = run_controls()
    result["all_passed"] = (
        result["failed"] == 0
        and result["passed"] == result["cases"]
        and result["hidden_witness_invariance_cases"] == 500
        and result["controls"]["non_smooth_scope_completion"]["exact_output"]
            == result["controls"]["non_smooth_scope_completion"]["oracle"]
        and result["controls"]["overlapping_product_rejected"]
        and result["controls"]["false_determinism_rejected"]
        and result["controls"]["inexact_float_weights_rejected"]
        and result["controls"]["signed_q_cancellation"]["support"] == 2
        and result["controls"]["complex_qi_cancellation"]["support"] == 2
    )
    result["python_version"] = platform.python_version()
    result["claim_ceiling"] = (
        "Small finite research prototype only. Shannon compilation and extensional "
        "determinism validation are exhaustive; arithmetic evaluation is bottom-up "
        "with explicit non-smooth scope completion. Not a scalable proof checker, "
        "SCPOG implementation, CNF compiler, Rust implementation, or solver differential."
    )
    return result

if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
