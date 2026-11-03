#!/usr/bin/env python3
"""Independent finite semantic and structural checker for ETH-018 certificates."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd
import hashlib
import json
import sys

FMT = "symthaea.algebraic-rewrite-certificate/v1"
KINDS = {"bool", "nat", "maxcount", "pareto", "ipoly", "qpoly"}
RING = {"ipoly", "qpoly"}


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def parse_value(kind, value):
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("boolean must be JSON boolean")
        return value
    if kind == "nat":
        if not _int(value) or value < 0:
            raise ValueError("natural must be a nonnegative integer")
        return value
    if kind == "maxcount":
        if value is None:
            return None
        if (not isinstance(value, list) or len(value) != 2
                or not _int(value[0]) or not _int(value[1]) or value[1] <= 0):
            raise ValueError("invalid MaxCount pair")
        return tuple(value)
    if kind == "pareto":
        if not isinstance(value, list):
            raise ValueError("Pareto value must be a row list")
        out, previous = {}, None
        for row in value:
            if (not isinstance(row, list) or len(row) != 3
                    or not all(_int(v) for v in row) or row[2] <= 0):
                raise ValueError("invalid Pareto row")
            point = (row[0], row[1])
            if previous is not None and point <= previous:
                raise ValueError("Pareto rows are not canonical and unique")
            out[point] = row[2]
            previous = point
        if any(q != p and all(a >= b for a, b in zip(q, p))
               and any(a > b for a, b in zip(q, p)) for p in out for q in out):
            raise ValueError("dominated Pareto point")
        return out
    if kind in RING:
        if not isinstance(value, list):
            raise ValueError("polynomial must be a sparse row list")
        out, previous = {}, None
        width = 3 if kind == "ipoly" else 4
        for row in value:
            if (not isinstance(row, list) or len(row) != width
                    or not _int(row[0]) or not _int(row[1])
                    or not (0 <= row[0] < 4 and 0 <= row[1] < 4)):
                raise ValueError("invalid polynomial exponent row")
            exponent = (row[0], row[1])
            if previous is not None and exponent <= previous:
                raise ValueError("polynomial rows are not canonical and unique")
            if kind == "ipoly":
                coefficient = row[2]
                if not _int(coefficient) or coefficient == 0:
                    raise ValueError("integer coefficient must be nonzero integer")
            else:
                numerator, denominator = row[2], row[3]
                if (not _int(numerator) or not _int(denominator) or numerator == 0
                        or denominator <= 0 or gcd(abs(numerator), denominator) != 1):
                    raise ValueError("rational coefficient must be reduced n/d, d>0")
                coefficient = Fraction(numerator, denominator)
            out[exponent] = coefficient
            previous = exponent
        return out
    raise ValueError("unknown algebra")


def zero(kind):
    return {"bool": False, "nat": 0, "maxcount": None,
            "pareto": {}, "ipoly": {}, "qpoly": {}}[kind]


def add(kind, left, right):
    if kind == "bool":
        return left or right
    if kind == "nat":
        return left + right
    if kind == "maxcount":
        if left is None:
            return right
        if right is None:
            return left
        if left[0] > right[0]:
            return left
        if right[0] > left[0]:
            return right
        return (left[0], left[1] + right[1])
    out = dict(left)
    for point, count in right.items():
        out[point] = out.get(point, 0) + count
    if kind == "pareto":
        return {p: c for p, c in out.items() if c and not any(
            q != p and all(a >= b for a, b in zip(q, p))
            and any(a > b for a, b in zip(q, p)) for q in out)}
    return {p: c for p, c in out.items() if c}


def mul(kind, left, right):
    if kind == "bool":
        return left and right
    if kind == "nat":
        return left * right
    if kind == "maxcount":
        if left is None or right is None:
            return None
        return (left[0] + right[0], left[1] * right[1])
    out = {}
    for p, a in left.items():
        for q, b in right.items():
            exponent = tuple(x + y for x, y in zip(p, q))
            if kind not in RING or all(x < 4 for x in exponent):
                out[exponent] = out.get(exponent, 0) + a * b
    if kind == "pareto":
        return {p: c for p, c in out.items() if c and not any(
            q != p and all(a >= b for a, b in zip(q, p))
            and any(a > b for a, b in zip(q, p)) for q in out)}
    return {p: c for p, c in out.items() if c}


def free_vars(term):
    op = term["op"]
    if op == "factor":
        return set(term["scope"])
    if op in ("add", "mul"):
        return free_vars(term["left"]) | free_vars(term["right"])
    if op == "sum":
        return free_vars(term["body"]) - {term["variable"]}
    if op == "neg":
        return free_vars(term["body"])
    if op == "zero":
        return set()
    raise ValueError("unknown operation")


def validate_term(term, kind, domains):
    if not isinstance(term, dict) or not isinstance(term.get("op"), str):
        raise ValueError("term node must be object")
    op = term["op"]
    shapes = {
        "factor": {"op", "name", "scope", "values"},
        "add": {"op", "left", "right"},
        "mul": {"op", "left", "right"},
        "sum": {"op", "variable", "body"},
        "neg": {"op", "body"},
        "zero": {"op"},
    }
    if op not in shapes or set(term) != shapes[op]:
        raise ValueError("incorrect node shape")
    if op == "factor":
        if not isinstance(term["name"], str) or not term["name"]:
            raise ValueError("invalid factor name")
        scope = term["scope"]
        if (not isinstance(scope, list) or any(not isinstance(v, str) for v in scope)
                or scope != sorted(set(scope)) or any(v not in domains for v in scope)):
            raise ValueError("factor scope is not canonical")
        expected = 1
        for variable in scope:
            expected *= domains[variable]
        if not isinstance(term["values"], list) or len(term["values"]) != expected:
            raise ValueError("factor table has incorrect size")
        for value in term["values"]:
            parse_value(kind, value)
    elif op in ("add", "mul"):
        validate_term(term["left"], kind, domains)
        validate_term(term["right"], kind, domains)
    elif op in ("sum", "neg"):
        if op == "sum":
            variable = term["variable"]
            if not isinstance(variable, str) or variable not in domains:
                raise ValueError("unbound finite-sum variable")
        elif kind not in RING:
            raise ValueError("negation is ring-only")
        validate_term(term["body"], kind, domains)


def apply_rule(rule, source, kind):
    op = source["op"]
    if rule == "factor-independent-summand":
        if op != "sum" or source["body"]["op"] != "mul":
            raise ValueError("rule pattern mismatch")
        variable = source["variable"]
        left, right = source["body"]["left"], source["body"]["right"]
        if variable not in free_vars(left) or variable in free_vars(right):
            raise ValueError("factor independence guard failed")
        return {"op": "mul",
                "left": {"op": "sum", "variable": variable, "body": left},
                "right": right}
    if rule == "swap-finite-sums":
        if op != "sum" or source["body"]["op"] != "sum":
            raise ValueError("rule pattern mismatch")
        outer, inner = source["variable"], source["body"]["variable"]
        if outer == inner:
            raise ValueError("bound variable alias")
        return {"op": "sum", "variable": inner,
                "body": {"op": "sum", "variable": outer, "body": source["body"]["body"]}}
    if rule == "split-independent-components":
        if (op != "sum" or source["body"]["op"] != "sum"
                or source["body"]["body"]["op"] != "mul"):
            raise ValueError("rule pattern mismatch")
        x, y = source["variable"], source["body"]["variable"]
        left, right = source["body"]["body"]["left"], source["body"]["body"]["right"]
        if x == y or x in free_vars(right) or y in free_vars(left):
            raise ValueError("component independence guard failed")
        return {"op": "mul",
                "left": {"op": "sum", "variable": x, "body": left},
                "right": {"op": "sum", "variable": y, "body": right}}
    if rule == "swap-multiplicands":
        if op != "mul" or kind not in KINDS:
            raise ValueError("commutative multiplication not qualified")
        return {"op": "mul", "left": source["right"], "right": source["left"]}
    if rule == "cancel-additive-inverse":
        if op != "add" or kind not in RING:
            raise ValueError("additive inverse is ring-only")
        left, right = source["left"], source["right"]
        if ((left.get("op") == "neg" and left["body"] == right)
                or (right.get("op") == "neg" and right["body"] == left)):
            return {"op": "zero"}
        raise ValueError("not an inverse pair")
    raise ValueError("unknown rewrite identifier")


def eval_term(term, kind, domains, env):
    op = term["op"]
    if op == "zero":
        return zero(kind)
    if op == "factor":
        index = 0
        for variable in term["scope"]:
            index = index * domains[variable] + env[variable]
        return parse_value(kind, term["values"][index])
    if op == "sum":
        variable, total = term["variable"], zero(kind)
        had_old, old = variable in env, env.get(variable)
        for value in range(domains[variable]):
            env[variable] = value
            total = add(kind, total, eval_term(term["body"], kind, domains, env))
        if had_old:
            env[variable] = old
        else:
            env.pop(variable, None)
        return total
    if op == "neg":
        return {p: -c for p, c in eval_term(term["body"], kind, domains, env).items()}
    if op in ("add", "mul"):
        left = eval_term(term["left"], kind, domains, env)
        right = eval_term(term["right"], kind, domains, env)
        return add(kind, left, right) if op == "add" else mul(kind, left, right)
    raise ValueError("unknown operation")


def check(certificate):
    try:
        required = {"format", "algebra", "domains", "rule_id", "source", "target",
                    "source_sha256", "target_sha256"}
        if not isinstance(certificate, dict) or set(certificate) != required:
            return False
        if certificate["format"] != FMT or certificate["algebra"] not in KINDS:
            return False
        domains = certificate["domains"]
        if (not isinstance(domains, dict) or not domains
                or any(not isinstance(v, str) or not v or not _int(n) or not 1 <= n <= 8
                       for v, n in domains.items())):
            return False
        kind, source, target = certificate["algebra"], certificate["source"], certificate["target"]
        validate_term(source, kind, domains)
        validate_term(target, kind, domains)
        if sha(source) != certificate["source_sha256"] or sha(target) != certificate["target_sha256"]:
            return False
        if apply_rule(certificate["rule_id"], source, kind) != target:
            return False
        variables = sorted(free_vars(source) | free_vars(target))
        for assignment in product(*(range(domains[v]) for v in variables)):
            env = dict(zip(variables, assignment))
            if eval_term(source, kind, domains, env) != eval_term(target, kind, domains, env):
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return False


if __name__ == "__main__":
    certificates = json.load(sys.stdin)
    accepted = [check(item) for item in certificates]
    print(json.dumps({"accepted": accepted, "accepted_count": sum(accepted),
                      "rejected_count": len(accepted) - sum(accepted)}))
