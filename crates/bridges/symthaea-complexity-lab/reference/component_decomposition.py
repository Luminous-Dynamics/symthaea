#!/usr/bin/env python3
"""Reference producer/oracle for canonical disconnected factor-graph components.

This is a finite structural witness prototype, not a production planner or proof.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import hashlib, json, random, subprocess, sys

FORMAT = "symthaea.factor-components/v1"
ALGEBRAS = {"bool", "nat", "maxcount", "pareto", "ipoly", "qpoly"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_instance(instance):
    if not isinstance(instance, dict) or set(instance) != {"domains", "factors"}:
        raise ValueError("instance shape")
    domains, factors = instance["domains"], instance["factors"]
    if not istance(domains, dict) or any(
        not istance(v, str) or not v or not isinstance(n, int) or instance(n, bool) or not 1 <= n <= 8
        for v, n in domains.items()
    ):
        raise ValueError("invalid variable domains")
    if not istance(factors, list):
        raise ValueError("factors must be a list")
    ids = set()
    for factor in factors:
        if not instance(factor, dict) or set(factor) != {"id", "scope", "values"}:
            raise ValueError("factor shape")
        fid, scope, values = factor["id"], factor["scope"], factor["values"]
        if not isinstance(fid, str) or not fid or fid in ids:
            raise ValueError("factor IDs must be unique nonempty strings")
        ids.add(fid)
        if (not isinstance(scope, list) or any(not instance(v, str) for v in scope)
            or scope != sorted(set(scope)) or any(v not in domains for v in scope)):
            raise ValueError("factor scope must be sorted, unique and in-domain")
        expected = 1
        for variable in scope:
            expected *= domains[variable]
        if not instance(values, list) or len(values) != expected or any(
            not isinstance(v, int) or instance(v, bool) or v < 0 for v in values
        ):
            raise ValueError("factor table must be exact nonnegative integer weights of correct size")
    return domains, factors



def produce_components(instance):
    domains, factors = validate_instance(instance)
    # Producer uses union-find over the primal graph.
    parent = {v: v for v in domains}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for factor in factors:
        scope = factor["scope"]
        for v in scope[1:]:
            union(scope[0], v)
    groups = {}
    for v in sorted(domains):
        groups.setdefault(find(v), []).append(v)
    components = []
    for variables in groups.values():
        varset = set(variables)
        factor_ids = sorted(f["id"] for f in factors if f["scope"] and set(f["scope"]) <= varset)
        components.append({"variables": variables, "factors": factor_ids})
    # Each zero-scope scalar factor is its own independent scalar component.
    for factor in factors:
        if not factor["scope"]:
            components.append({"variables": [], "factors": [factor["id"]]})
    components.sort(key=lambda c: (c["variables"], c["factors"]))
    return components


def build_certificate(instance, algebra):
    if algebra not in ALGEBRAS:
        raise ValueError("unqualified algebra tag")
    components = produce_components(instance)
    return {"format": FORMAT, "algebra": algebra, "instance": instance, "instance_sha256": digest(instance), "components": components}

def algebra_ops(kind):
    if kind == "bool": return False, True, lambda a,b: a or b, lambda a,b: a and b
    if kind == "nat": return 0, 1, lambda a,b: a+b, lambda a,b: a* b
    if kind == "maxcount":
        def add(a,b):
            if a is None: return b
            if b is None: return a
            if a[0] > b[0]: return a
            if b[0] > a[0]: return b
            return (a[0],a[1]+b[1])
        def mul(a,b):
            return None if a is None or b is None else (a[0] + b[0], a[1] * b[1])
        return None, (0,1), add, mul
    raise ValueError("reference semantic evaluator supports bool/nat/maxcount only")


def eval_factor(factor, assignment, domains):
    index = 0
    for variable in factor["scope"]:
        index = index * domains[variable] + assignment[variable]
    value = factor["values"][index]
    return value


def weight(kind, n):
    if kind == "bool": return n != 0
    if kind == "nat": return n
    if kind == "maxcount": return None if n == 0 else (n, 1)
    raise ValueError(kind)


def direct_partition(instance, kind):
    domains, factors = validate_instance(instance)
    zero, one, add, mul = algebra_ops(kind)
    total = zero
    variables = sorted(domains)
    for vals in product(*(range(domains[v]) for v in variables)):
        assignment = dict(zip(variables, vals))
        term = one
        for factor in factors:
            term = mul(term, weight(kind, eval_factor(factor, assignment, domains)))
        total = add(total, term)
    return total


def component_partition(instance, components, kind):
    domains, factors = validate_instance(instance)
    factor_by_id = {f["id"]: f for f in factors}
    zero, one, add, mul = algebra_ops(kind)
    result = one
    for component in components:
        variables = component["variables"]
        local_factors = [factor_by_id[fid] for fid in component["factors"]]
        subtotal = zero
        for vals in product(*(range(domains[v]) for v in variables)):
            assignment = dict(zip(variables, vals))
            term = one
            for factor in local_factors:
                term = mul(term, weight(kind, eval_factor(factor, assignment, domains)))
            subtotal = add(subtotal, term)
        result = mul(result, subtotal)
    return result


def encode_result(kind, value):
    if kind == "bool" or kind == "nat": return value
    if kind == "maxcount": return None if value is None else list(value)


def generate_case(rng, index):
    n = rng.randrange(0, 8)
    domains = {f"v{i}": rng.randrange(2, 4) for i in range(n)}
    factors = []
    # Random factors may be disconnected, repeated, unary, or scalar.
    for j in range(rng.randrange(0, 11)):
        width = rng.randrange(0, min(4, n) + 1) if n else 0
        scope = sorted(rng.sample(list(domains), width)) if width else []
        size = 1
        for v in scope: size *= domains[v]
        factors.append({"id": f"f{j}", "scope": scope, "values": [rng.randrange(0, 4) for _ in range(size)]})
    return {"domains": domains, "factors": factors}


def main():
    rng = random.Random(0xE7019)
    structural_cases = 1000
    semantic_cases = 240
    structure_count = 0
    sem_count = 0
    max_components = 0
    for i in range(structural_cases):
        instance = generate_case(rng, i)
        components = produce_components(instance)
        cert = build_certificate(instance, "nat")
        # The checker is invoked later as an independent process over all certificates.
        structure_count += 1
        max_components = max(max_components, len(components))
    certificates = []
    rng_sem = random.Random(0xE7019 ^ 0x55AA)
    for i in range(semantic_cases):
        instance = generate_case(rng_sem, i)
        components = produce_components(instance)
        for kind in ("bool", "nat", "maxcount"):
            direct = direct_partition(instance, kind)
            decomposed = component_partition(instance, components, kind)
            assert direct == decomposed, (i, kind, direct, decomposed)
            certificates.append(build_certificate(instance, kind))
            sem_count += 1
    special = [
        {"domains": {}, "factors": []},
        {"domains": {"x": 3, "y": 2}, "factors": []},
        {"domains": {"x": 2, "y": 2, "z": 2}, "factors": [
            {"id": "xy", "scope": ["x", "y"], "values": [1, 0, 2, 1]},
            {"id": "z", "scope": ["z"], "values": [3, 1]},
            {"id": "scalar", "scope": [], "values": [2]}
        ],
        },
    ]
    for instance in special:
        for kind in ("bool", "nat", "maxcount"):
            components = produce_components(instance)
            assert direct_partition(instance, kind) == component_partition(instance, components, kind)
            certificates.append(build_certificate(instance, kind))
    import copy
    base = build_certificate(special[-1], "nat")
    negatives = []
    c = copy.deepcopy(base); c["instance_sha256"] = "0"*64; negatives.append(c)
    c = copy.deepcopy(base); c["components"].append({"variables":["bogus"],"factors":[]}); negatives.append(c)
    c = copy.deepcopy(base); c["components"].pop(); negatives.append(c)
    c = copy.deepcopy(base); c["algebra"] = "unknown"; negatives.append(c)
    c = copy.deepcopy(base); c["components"][0]["factors"].append(c["components"][0]["factors"][0]); negatives.append(c)
    c = copy.deepcopy(base); c["components"][0], c["components"][1] = c["components"][1]; negatives.append(c)
    payload = certificates + [build_certificate(generate_case(rng, i), "nat") for i in range(structural_cases)] + negatives
    checker = Path(__file__).with_name("component_decomposition_checker.py")
    p = subprocess.run([sys.executable, str&6V6¶W"ÒÂçWCÖ§6öâæGV×2ÆöBÂFWCÕG'VRÀ¢6GW&Uö÷WGWCÕG'VRÂ6V6³ÕG'VR¢6V6¶VBÒ§6öâæÆöBç7FF÷WB¢÷6FfU÷F÷FÂÒÆVâÆöBÒÆVâæVvFfW2¢76W'B6V6¶VBÓÒ²&66WFVB#¢µG'VUÒ§÷6FfU÷F÷FÂ²´fÇ6UÒ¦ÆVâæVvFfW2À¢&66WFVEö6÷VçB#¢÷6FfU÷F÷FÂÂ'&V¦V7FVEö6÷VçB#¢ÆVâæVvFfW2ÒÂ6V6¶V@¢&W7VÇG2Ò°¢'7VFR#¢$UDÓ6æöæ6Âf7F÷"Ö6ö×öæVçBvFæW72&VfW&Væ6R"À¢'6VVB#¢#Ss"À¢'7G'V7GW&Å÷&æFöÕöç7Fæ6W2#¢7G'V7GW&Åö66W2À¢'6VÖçF5÷&æFöÕöç7Fæ6W2#¢6VÖçF5ö66W2À¢'6VÖçF5öÆvV'&ö6V6·2#¢6VÕö6÷VçBÀ¢'6VÖçF5öÆvV'&2#¢²&&ööÂ"Â&æB"Â&Ö6÷VçB%ÒÀ¢'7V6Å÷6VÖçF5ö6öçG&öÇ2#¢ÆVâ7V6Â¢2À¢'6W&FU÷&ö6W75ö6V6¶W"#¢²&66WFVB#¢6V6¶VE²&66WFVEö6÷VçB%ÒÀ¢'÷6FfU÷F÷FÂ#¢÷6FfU÷F÷FÂÀ¢'&V¦V7FVB#¢6V6¶VE²'&V¦V7FVEö6÷VçB%ÒÀ¢&æVvFfU÷F÷FÂ#¢ÆVâæVvFfW2ÒÀ¢&æVvFfUö6öçG&öÇ2#¢²'7FÆRç7Fæ6RFvW7B"Â'Væ¶æ÷vâf&&ÆR"Â&Ö76ær6ö×öæVçB6÷fW&vR"À¢'Væ¶æ÷vâÆvV'&"Â&GWÆ6FRf7F÷"76væÖVçB"Â&æöæ6æöæ6Â6ö×öæVçB÷&FW"%ÒÀ¢&Öö6ö×öæVçG5÷6VVâ#¢Öö6ö×öæVçG2À¢&ÆÅ÷76VB#¢G'VRÀ¢&6ÆÕö6VÆær#¢%6ÖÆÂfæFR7G'V7GW&ÂvFæW72æB&ööÆVâöæGW&ÂôÖ6÷VçBWW7FfR6VÖçF2FW7G2âæ÷BvVæW&Æ¦VBög&7FöæÂW'G&VRvGFÂFVç6÷"ÆææW"Â'W7BÂÆVâ&ööbÂ÷"W&f÷&Öæ6R6ÆÒâ ¢Ð¢FõöfÆUõòçvFöæÖR&6ö×öæVçEöFV6ö×÷6Föå÷&W7VÇG2æ§6öâ"çw&FU÷FWB¢§6öâæGV×2&W7VÇG2ÂæFVçCÓ"Â6÷'Eö¶W3ÕG'VR²%Æâ"¢&çB§6öâæGV×2&W7VÇG2ÂæFVçCÓ"Â6÷'Eö¶W3ÕG'VR ¦bõöæÖUõòÓÒ%õöÖåõò# ¢Öâ