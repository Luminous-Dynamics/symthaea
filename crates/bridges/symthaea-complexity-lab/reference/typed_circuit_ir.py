#!/usr/bin/env python3
"""Small exact typed pseudo-Boolean circuit IR reference.

This is a finite semantic/schema prototype. It is not a CPOG or VeriPB
emitter, a formally verified checker, or a production scalable evaluator.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import json

SCHEMA = "symthaea.typed-tractable-circuit/v1"


class CircuitError(ValueError):
    pass


class QI:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re, self.im = Fraction(re), Fraction(im)

    def __add__(self, other):
        return QI(self.re + other.re, self.im + other.im)

    def __mul__(self, other):
        return QI(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def __eq__(self, other):
        return isinstance(other, QI) and (self.re, self.im) == (other.re, other.im)

    def __str__(self):
        if self.im == 0:
            return str(self.re)
        return f"{self.re}{'+' if self.im >= 0 else ''}{self.im}i"


def rational(value):
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
        raise CircuitError("rational coefficient must be [numerator, denominator]")
    n, d = value
    if d <= 0:
        raise CircuitError("rational denominator must be positive")
    result = Fraction(n, d)
    if [result.numerator, result.denominator] != value:
        raise CircuitError("rational coefficient must be reduced/canonical")
    return result


def decode(domain, value):
    if domain == "bool":
        if type(value) is not bool:
            raise CircuitError("Boolean coefficients must be bool")
        return value
    if domain == "q":
        return rational(value)
    if domain == "qi":
        if not isinstance(value, dict) or set(value) != {"re", "im"}:
            raise CircuitError("Q(i) coefficient must contain re and im")
        return QI(rational(value["re"]), rational(value["im"]))
    raise CircuitError(f"unsupported domain: {domain}")


def zero(domain):
    return False if domain == "bool" else Fraction(0) if domain == "q" else QI()


def one(domain):
    return True if domain == "bool" else Fraction(1) if domain == "q" else QI(1, 0)


def plus(domain, a, b):
    return (a or b) if domain == "bool" else a + b


def times(domain, a, b):
    return (a and b) if domain == "bool" else a * b


def evaluate(circuit):
    required = {"schema", "contract", "variables", "visible", "domain", "nodes", "root"}
    optional = {"visible_weights"}
    if not isinstance(circuit, dict) or not required <= set(circuit) or set(circuit) - required - optional:
        raise CircuitError("invalid circuit header")
    if circuit["schema"] != SCHEMA:
        raise CircuitError("unsupported schema")

    contract, domain = circuit["contract"], circuit["domain"]
    domains = {"#sat": "bool", "pmc": "bool", "pwmc": "bool", "wmc": "q", "amc-complex": "qi"}
    if contract not in domains or domains[contract] != domain:
        raise CircuitError("output contract / coefficient domain mismatch")

    variables, visible = circuit["variables"], circuit["visible"]
    if not isinstance(variables, list) or any(not isinstance(v, str) or not v for v in variables):
        raise CircuitError("variable universe must be a string list")
    universe = set(variables)
    if len(universe) != len(variables):
        raise CircuitError("duplicate variable")
    if not isinstance(visible, list) or len(set(visible)) != len(visible) or not set(visible) <= universe:
        raise CircuitError("invalid visible-variable set")
    if contract not in {"pmc", "pwmc"} and visible:
        raise CircuitError("visible variables only apply to projected contracts")
    if contract != "pwmc" and "visible_weights" in circuit:
        raise CircuitError("visible_weights is reserved for PWMC")

    nodes = circuit["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise CircuitError("nodes must be nonempty")
    by_id, scopes = {}, {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in by_id:
            raise CircuitError("invalid or duplicate node ID")
        by_id[node["id"]] = node

    # Topological IDs make cycles and forward references impossible.
    for node in nodes:
        nid, kind = node["id"], node.get("kind")
        if kind == "indicator":
            if set(node) != {"id", "kind", "variable", "value"}:
                raise CircuitError("indicator shape")
            if node["variable"] not in universe or type(node["value"]) is not bool:
                raise CircuitError("invalid indicator")
            scopes[nid] = {node["variable"]}
        elif kind == "constant":
            if set(node) != {"id", "kind", "value"}:
                raise CircuitError("constant shape")
            decode(domain, node["value"])
            scopes[nid] = set()
        elif kind == "product":
            if set(node) != {"id", "kind", "children"} or not isinstance(node["children"], list) or not node["children"]:
                raise CircuitError("product shape")
            children = node["children"]
            if any(ch not in scopes for ch in children):
                raise CircuitError("product child must precede parent")
            flat = [v for ch in children for v in scopes[ch]]
            if len(flat) != len(set(flat)):
                raise CircuitError("product is not decomposable: child scopes overlap")
            scopes[nid] = set(flat)
        elif kind == "sum":
            if set(node) - {"id", "kind", "terms", "deterministic"} or not {"id", "kind", "terms"} <= set(node):
                raise CircuitError("sum shape")
            if not isinstance(node["terms"], list) or not node["terms"]:
                raise CircuitError("sum terms must be nonempty")
            if "deterministic" in node and type(node["deterministic"]) is not bool:
                raise CircuitError("deterministic flag must be Boolean")
            children = []
            for term in node["terms"]:
                if not isinstance(term, dict) or set(term) != {"child", "weight"}:
                    raise CircuitError("sum term shape")
                child = term["child"]
                if child not in scopes:
                    raise CircuitError("sum child must precede parent")
                decode(domain, term["weight"])
                children.append(child)
            scopes[nid] = set().union(*(scopes[ch] for ch in children))
        else:
            raise CircuitError(f"unknown node kind: {kind}")

    root = circuit["root"]
    if root not in by_id or not scopes[root] <= universe:
        raise CircuitError("invalid root or root scope")

    def value(node_id, assignment):
        node = by_id[node_id]
        if node["kind"] == "indicator":
            return one(domain) if assignment[node["variable"]] == node["value"] else zero(domain)
        if node["kind"] == "constant":
            return decode(domain, node["value"])
        if node["kind"] == "product":
            result = one(domain)
            for child in node["children"]:
                result = times(domain, result, value(child, assignment))
            return result
        result = zero(domain)
        for term in node["terms"]:
            term_value = times(domain, decode(domain, term["weight"]), value(term["child"], assignment))
            result = plus(domain, result, term_value)
        return result

    # Verify a claimed deterministic sum extensionally over its finite scope.
    checked_deterministic = []
    for node in nodes:
        if node["kind"] == "sum" and node.get("deterministic", False):
            scope = sorted(scopes[node["id"]])
            for bits in product((False, True), repeat=len(scope)):
                assignment = dict(zip(scope, bits))
                terms = [
                    times(domain, decode(domain, t["weight"]), value(t["child"], assignment))
                    for t in node["terms"]
                ]
                for i in range(len(terms)):
                    for j in range(i + 1, len(terms)):
                        if times(domain, terms[i], terms[j]) != zero(domain):
                            raise CircuitError(f"sum {node['id']} falsely claims determinism")
            checked_deterministic.append(node["id"])

    nonzero_assignments = 0
    satisfying_assignments = 0
    projections = set()
    total = zero("qi" if domain == "qi" else "q")
    for bits in product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        result = value(root, assignment)
        supported = bool(result) if domain == "bool" else result != zero(domain)
        if not supported:
            continue
        nonzero_assignments += 1
        if contract in {"#sat", "pmc", "pwmc"}:
            satisfying_assignments += 1
            projections.add(tuple(assignment[v] for v in visible))
        if contract == "#sat":
            total += 1
        elif contract in {"wmc", "amc-complex"}:
            total = plus(domain, total, result)

    if contract == "pmc":
        total = Fraction(len(projections))
    elif contract == "pwmc":
        weights = circuit.get("visible_weights", {})
        if not isinstance(weights, dict) or not set(weights) <= set(visible):
            raise CircuitError("PWMC weights must refer only to visible variables")
        for key in projections:
            assignment = dict(zip(visible, key))
            contribution = Fraction(1)
            for variable in visible:
                pair = weights.get(variable, [[1, 1], [1, 1]])
                if not isinstance(pair, list) or len(pair) != 2:
                    raise CircuitError("visible weight pair shape")
                contribution *= rational(pair[1 if assignment[variable] else 0])
            total += contribution

    return {
        "contract": contract,
        "domain": domain,
        "root_scope": sorted(scopes[root]),
        "checked_deterministic_nodes": checked_deterministic,
        "satisfying_full_assignments": satisfying_assignments if contract in {"#sat", "pmc", "pwmc"} else None,
        "nonzero_function_assignments": nonzero_assignments,
        "distinct_projections": len(projections) if contract in {"pmc", "pwmc"} else None,
        "exact_output": str(total),
    }


def hidden_witness_circuit(contract):
    # Root predicate x=true; y is free and hidden. Thus #SAT=2, PMC=1.
    c = {
        "schema": SCHEMA, "contract": contract,
        "variables": ["x", "y"],
        "visible": ["x"] if contract in {"pmc", "pwmc"} else [],
        "domain": "bool",
        "nodes": [{"id": "x_true", "kind": "indicator", "variable": "x", "value": True}],
        "root": "x_true",
    }
    if contract == "pwmc":
        c["visible_weights"] = {"x": [[1, 3], [2, 3]]}
    return c


def cancellation_circuit(domain, contract):
    positive, negative = ([1, 1], [-1, 1]) if domain == "q" else (
        {"re": [1, 1], "im": [2, 1]}, {"re": [-1, 1], "im": [-2, 1]}
    )
    return {
        "schema": SCHEMA, "contract": contract, "variables": ["x"], "visible": [], "domain": domain,
        "nodes": [
            {"id": "x_false", "kind": "indicator", "variable": "x", "value": False},
            {"id": "x_true", "kind": "indicator", "variable": "x", "value": True},
            {"id": "root", "kind": "sum", "deterministic": True, "terms": [
                {"child": "x_false", "weight": negative},
                {"child": "x_true", "weight": positive},
            ]},
        ],
        "root": "root",
    }


def fixture_tests():
    tests = []
    for contract, expected in (("#sat", "2"), ("pmc", "1"), ("pwmc", "2/3")):
        result = evaluate(hidden_witness_circuit(contract))
        assert result["exact_output"] == expected and result["satisfying_full_assignments"] == 2
        if contract in {"pmc", "pwmc"}:
            assert result["distinct_projections"] == 1
        tests.append({"name": f"{contract} hidden-witness behavior", "result": result})

    for domain, contract in (("q", "wmc"), ("qi", "amc-complex")):
        result = evaluate(cancellation_circuit(domain, contract))
        assert result["exact_output"] == "0" and result["checked_deterministic_nodes"] == ["root"]
        tests.append({"name": f"{contract} exact cancellation", "result": result})

    bad_product = {
        "schema": SCHEMA, "contract": "#sat", "variables": ["x"], "visible": [], "domain": "bool",
        "nodes": [
            {"id": "x0", "kind": "indicator", "variable": "x", "value": False},
            {"id": "p", "kind": "product", "children": ["x0", "x0"]},
        ], "root": "p",
    }
    try:
        evaluate(bad_product)
    except CircuitError as exc:
        tests.append({"name": "overlapping product scope rejected", "error": str(exc)})
    else:
        raise AssertionError("overlapping product gate accepted")

    bad_sum = {
        "schema": SCHEMA, "contract": "#sat", "variables": ["x"], "visible": [], "domain": "bool",
        "nodes": [
            {"id": "x1a", "kind": "indicator", "variable": "x", "value": True},
            {"id": "x1b", "kind": "indicator", "variable": "x", "value": True},
            {"id": "s", "kind": "sum", "deterministic": True, "terms": [
                {"child": "x1a", "weight": True}, {"child": "x1b", "weight": True},
            ]},
        ], "root": "s",
    }
    try:
        evaluate(bad_sum)
    except CircuitError as exc:
        tests.append({"name": "false determinism claim rejected", "error": str(exc)})
    else:
        raise AssertionError("overlapping deterministic sum accepted")

    try:
        evaluate(hidden_witness_circuit("amc-complex"))
    except CircuitError as exc:
        tests.append({"name": "cross-domain label rejected", "error": str(exc)})
    else:
        raise AssertionError("Boolean support circuit accepted as complex AMC")

    return {
        "suite": "ETH-026 typed tractable circuit IR semantic fixtures",
        "checks": len(tests),
        "passed": len(tests),
        "all_passed": True,
        "fixtures": tests,
    }


if __name__ == "__main__":
    print(json.dumps(fixture_tests(), indent=2, sort_keys=True))
