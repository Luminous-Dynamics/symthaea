#!/usr/bin/env python3
"""Independent BFS checker for ETH-019 factor-component witnesses."""
from __future__ import annotations

import hashlib
import json
import sys

FORMAT = "symthaea.factor-components/v1"
ALGEBRAS = {"bool", "nat", "maxcount", "pareto", "ipoly", "qpoly"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def check(certificate):
    try:
        required = {"format", "algebra", "instance", "instance_sha256", "components"}
        if not isinstance(certificate, dict) or set(certificate) != required:
            return False
        if (certificate["format"] != FORMAT or
                certificate["algebra"] not in ALGEBRAS):
            return False

        instance = certificate["instance"]
        if not isinstance(instance, dict) or set(instance) != {"domains", "factors"}:
            return False
        domains, factors = instance["domains"], instance["factors"]
        if not isinstance(domains, dict) or not isinstance(factors, list):
            return False
        for variable, cardinality in domains.items():
            if (not isinstance(variable, str) or not variable or
                    not isinstance(cardinality, int) or isinstance(cardinality, bool) or
                    not 1 <= cardinality <= 8):
                return False
        if digest(instance) != certificate["instance_sha256"]:
            return False

        factor_by_id = {}
        for factor in factors:
            if not isinstance(factor, dict) or set(factor) != {"id", "scope", "values"}:
                return False
            fid, scope, values = factor["id"], factor["scope"], factor["values"]
            if not isinstance(fid, str) or not fid or fid in factor_by_id:
                return False
            if (not isinstance(scope, list) or any(not isinstance(v, str) for v in scope) or
                    scope != sorted(set(scope)) or any(v not in domains for v in scope)):
                return False
            size = 1
            for variable in scope:
                size *= domains[variable]
            if (not isinstance(values, list) or len(values) != size or
                    any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                        for x in values)):
                return False
            factor_by_id[fid] = factor

        # Independent BFS over primal adjacency; producer uses union-find.
        adjacency = {variable: set() for variable in domains}
        for factor in factors:
            scope = factor["scope"]
            for variable in scope:
                adjacency[variable].update(
                    other for other in scope if other != variable)

        visited = set()
        expected = []
        for start in sorted(domains):
            if start in visited:
                continue
            queue = [start]
            visited.add(start)
            component = []
            while queue:
                variable = queue.pop(0)
                component.append(variable)
                for neighbor in sorted(adjacency[variable]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            variables = sorted(component)
            variable_set = set(variables)
            factor_ids = sorted(
                fid for fid, factor in factor_by_id.items()
                if factor["scope"] and set(factor["scope"]) <= variable_set)
            expected.append({"variables": variables, "factors": factor_ids})

        for fid, factor in factor_by_id.items():
            if not factor["scope"]:
                expected.append({"variables": [], "factors": [fid]})
        expected.sort(key=lambda item: (item["variables"], item["factors"]))

        components = certificate["components"]
        if not isinstance(components, list):
            return False
        seen_variables, seen_factors = [], []
        for component in components:
            if not isinstance(component, dict) or set(component) != {"variables", "factors"}:
                return False
            if (not isinstance(component["variables"], list) or
                    not isinstance(component["factors"], list)):
                return False
            seen_variables.extend(component["variables"])
            seen_factors.extend(component["factors"])
        if (sorted(seen_variables) != sorted(domains) or
                len(seen_variables) != len(set(seen_variables))):
            return False
        if (sorted(seen_factors) != sorted(factor_by_id) or
                len(seen_factors) != len(set(seen_factors))):
            return False
        return components == expected
    except (KeyError, TypeError, ValueError, IndexError):
        return False


if __name__ == "__main__":
    certificates = json.load(sys.stdin)
    accepted = [check(certificate) for certificate in certificates]
    print(json.dumps({
        "accepted": accepted,
        "accepted_count": sum(accepted),
        "rejected_count": len(accepted) - sum(accepted),
    }))
