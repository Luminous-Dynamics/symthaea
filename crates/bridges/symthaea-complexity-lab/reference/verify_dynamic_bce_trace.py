#!/usr/bin/env python3
"""Reference checker for a context-indexed two-branch projected-BCE trace.

Finite exhaustive fixture only. Not D4 dynamic BCE, not a formal proof checker.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, pathlib, sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
CERT = HERE / "dynamic_bce_trace_certificate.v1.json"
OUT = HERE / "dynamic_bce_trace_results.v1.json"


def parts(lit):
    if lit.startswith("!") and len(lit) > 1:
        return lit[1:], False
    return lit, True


def lit_value(lit, assignment):
    var, positive = parts(lit)
    return bool(assignment[var]) if positive else not bool(assignment[var])


def canonical_formula(variables, clauses):
    return {
        "variables": sorted(variables),
        "clauses": [
            {"id": c["id"], "literals": sorted(set(c["literals"]),
              key=lambda z: (parts(z)[0], parts(z)[1]))}
            for c in sorted(clauses, key=lambda c: c["id"])
        ],
    }


def formula_digest(variables, clauses):
    raw = json.dumps(canonical_formula(variables, clauses),
                     sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def context_digest(parent_digest, branch_id, assignment):
    obj = {"parent_formula_digest": parent_digest, "context_id": branch_id,
           "assignment": [[k, assignment[k]] for k in sorted(assignment)]}
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def condition_formula(variables, clauses, assignment):
    residual = [v for v in variables if v not in assignment]
    output = []
    for clause in clauses:
        if any(parts(lit)[0] in assignment and lit_value(lit, assignment)
               for lit in clause["literals"]):
            continue
        output.append({"id": clause["id"],
                       "literals": [lit for lit in clause["literals"]
                                    if parts(lit)[0] not in assignment]})
    return residual, output


def tautology_variable(literals):
    values = set(literals)
    for lit in sorted(values, key=lambda z: (parts(z)[0], parts(z)[1])):
        var, positive = parts(lit)
        complement = "!" + var if positive else var
        if complement in values:
            return var
    return None


def check_bce(variables, hidden, source, operation):
    errors = []
    by_id = {c["id"]: c for c in source}
    if len(by_id) != len(source):
        return ["duplicate clause IDs in conditioned formula"]
    removed_id = operation["removed_clause_id"]
    if removed_id not in by_id:
        return ["removed clause missing in this branch context"]
    removed = by_id[removed_id]
    blocker = operation["blocking_literal"]
    blocker_var, _ = parts(blocker)
    if blocker not in removed["literals"]:
        errors.append("blocking literal absent from removed clause")
    if blocker_var not in hidden:
        errors.append("blocking variable is not hidden in this context")
    if tautology_variable(removed["literals"]) is not None:
        errors.append("removed clause is tautological")
    complement = "!" + blocker_var if not blocker.startswith("!") else blocker_var
    opposing = {c["id"]: c for c in source
                if c["id"] != removed_id and complement in c["literals"]}
    listed = operation["opposing_clause_checks"]
    supplied = {x["clause_id"]: x for x in listed}
    if len(supplied) != len(listed) or set(supplied) != set(opposing):
        errors.append("opposing-clause coverage incomplete, duplicated or extraneous")
    for clause_id, other in opposing.items():
        item = supplied.get(clause_id)
        if item is None:
            continue
        resolvent = [x for x in removed["literals"] if x != blocker]
        resolvent += [x for x in other["literals"] if x != complement]
        witness = tautology_variable(resolvent)
        canonical = sorted(set(resolvent), key=lambda z: (parts(z)[0], parts(z)[1]))
        if witness is None:
            errors.append("resolvent with " + clause_id + " is not tautological")
        if item.get("resolvent") != canonical:
            errors.append("recorded resolvent mismatch for " + clause_id)
        if item.get("tautology_variable") != witness:
            errors.append("tautology witness mismatch for " + clause_id)
    target = [c for c in source if c["id"] != removed_id]
    if operation.get("target_clauses") != target:
        errors.append("target is not exactly source minus declared clause")
    if operation.get("target_formula_digest") != formula_digest(variables, target):
        errors.append("target formula digest mismatch")
    return errors


def sat(clauses, assignment):
    return all(any(lit_value(lit, assignment) for lit in c["literals"])
               for c in clauses)


def projected_oracle(variables, visible, clauses):
    support, full_models = set(), 0
    for bits in itertools.product((0, 1), repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        if sat(clauses, assignment):
            full_models += 1
            support.add(tuple(assignment[v] for v in visible))
    return full_models, sorted(support)


def weighted_support(support, visible, weights):
    total = Fraction(0, 1)
    for row in support:
        term = Fraction(1, 1)
        for variable, bit in zip(visible, row):
            term *= Fraction(weights[variable]["true" if bit else "false"])
        total += term
    return f"{total.numerator}/{total.denominator}"


def check_trace(cert):
    errors = []
    if cert.get("format") != "symthaea.dynamic-projected-bce-trace" or cert.get("version") != 1:
        return ["unsupported format/version"], {}
    root = cert["root"]
    variables, visible, hidden = root["variables"], root["visible"], root["hidden"]
    if (len(variables) != len(set(variables)) or len(visible) != len(set(visible))
            or len(hidden) != len(set(hidden)) or set(visible) & set(hidden)
            or set(visible) | set(hidden) != set(variables)):
        errors.append("invalid root variable partition")
    if "x" not in visible:
        errors.append("branch variable x is not visible at root")
    for clause in root["clauses"]:
        if any(parts(lit)[0] not in variables for lit in clause["literals"]):
            errors.append("root clause contains undeclared variable")
    root_digest = formula_digest(variables, root["clauses"])
    if root_digest != root["formula_digest"]:
        errors.append("root formula digest mismatch")
    branches = cert["branches"]
    branch_ids = [b["branch_id"] for b in branches]
    branch_values = [b["assignment"].get("x") for b in branches]
    if len(branch_ids) != len(set(branch_ids)) or sorted(branch_values) != [0, 1]:
        errors.append("branch split incomplete, duplicated or not x=0/x=1")
    branch_results, union_before, union_after = [], set(), set()
    source_models_total = target_models_total = 0
    for branch in branches:
        branch_id, assignment = branch["branch_id"], branch["assignment"]
        if set(assignment) != {"x"} or assignment["x"] not in (0, 1):
            errors.append(branch_id + ": unsupported branch assignment")
            continue
        expected_context = context_digest(root_digest, branch_id, assignment)
        if branch["context_digest"] != expected_context:
            errors.append(branch_id + ": context digest mismatch")
        residual, conditioned = condition_formula(variables, root["clauses"], assignment)
        declared = branch["conditioned"]
        if residual != declared["variables"] or conditioned != declared["clauses"]:
            errors.append(branch_id + ": conditioned formula does not derive from parent/context")
        cond_digest = formula_digest(residual, conditioned)
        if cond_digest != declared["formula_digest"]:
            errors.append(branch_id + ": conditioned formula digest mismatch")
        branch_visible = [v for v in visible if v not in assignment]
        branch_hidden = [v for v in hidden if v not in assignment]
        if declared["visible"] != branch_visible or declared["hidden"] != branch_hidden:
            errors.append(branch_id + ": conditioned quantifier partition mismatch")
        current, current_digest = conditioned, cond_digest
        for operation in branch["operations"]:
            if operation.get("type") != "remove_blocked_clause":
                errors.append(branch_id + ": unsupported trace operation")
                continue
            if (operation.get("context_digest") != expected_context
                    or operation.get("source_formula_digest") != current_digest):
                errors.append(branch_id + ": operation bound to wrong context/formula")
            errors.extend(branch_id + ": " + e for e in
                          check_bce(residual, branch_hidden, current, operation))
            current = operation["target_clauses"]
            current_digest = operation["target_formula_digest"]
        backtrack = branch["backtrack"]
        if backtrack.get("restored_context_id") != "root":
            errors.append(branch_id + ": wrong backtrack parent context")
        if backtrack.get("restored_formula_digest") != root_digest:
            errors.append(branch_id + ": backtrack failed to restore root formula")
        if backtrack.get("child_final_formula_digest") != current_digest:
            errors.append(branch_id + ": backtrack child digest mismatch")
        before_models, before_support = projected_oracle(residual, branch_visible, conditioned)
        after_models, after_support = projected_oracle(residual, branch_visible, current)
        if before_support != after_support:
            errors.append(branch_id + ": projected support changed")
        if branch_id == "branch-x0" and before_models == after_models:
            errors.append("fixture no longer demonstrates permitted full-model change")
        if branch_id == "branch-x1" and before_models != after_models:
            errors.append("no-op sibling branch unexpectedly changes models")
        union_before.update((assignment["x"],) + row for row in before_support)
        union_after.update((assignment["x"],) + row for row in after_support)
        source_models_total += before_models
        target_models_total += after_models
        branch_results.append({
            "branch_id": branch_id,
            "context_digest": expected_context,
            "conditioned_formula_digest": cond_digest,
            "final_formula_digest": current_digest,
            "source_full_models": before_models,
            "target_full_models": after_models,
            "source_projected_support": ["".join(map(str, x)) for x in before_support],
            "target_projected_support": ["".join(map(str, x)) for x in after_support],
            "backtrack_restored_root": backtrack.get("restored_formula_digest") == root_digest,
        })
    root_models, root_support = projected_oracle(variables, visible, root["clauses"])
    if union_before != set(root_support) or union_after != set(root_support):
        errors.append("branch support union does not equal root projected support")
    if source_models_total != root_models:
        errors.append("branch source model accounting does not equal root")
    summary = {
        "root_full_models": root_models,
        "processed_branch_full_models": target_models_total,
        "root_projected_support": ["".join(map(str, x)) for x in root_support],
        "processed_projected_support": ["".join(map(str, x)) for x in sorted(union_after)],
        "root_pmc": len(root_support),
        "processed_pmc": len(union_after),
        "root_pwmc": weighted_support(root_support, visible, root["weights"]),
        "processed_pwmc": weighted_support(sorted(union_after), visible, root["weights"]),
        "branches": branch_results,
    }
    return errors, summary


def main():
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    errors, summary = check_trace(cert)
    if errors:
        raise SystemExit("FAIL valid trace: " + "; ".join(errors))
    if summary["root_pwmc"] != summary["processed_pwmc"]:
        raise AssertionError("root PWMC changed")
    print("PASS context-indexed branch trace: conditioning, hidden BCE and backtrack")
    print(f"PASS support/PMC/PWMC preserved; full models {summary['root_full_models']} -> {summary['processed_branch_full_models']}")
    mutations = []
    def mutate(name, fn):
        candidate = copy.deepcopy(cert)
        fn(candidate)
        mutations.append((name, candidate))
    mutate("missing x=1 sibling branch", lambda c: c["branches"].pop())
    mutate("duplicate branch assignment", lambda c: c["branches"][1].update(assignment={"x": 0}))
    mutate("stale context digest", lambda c: c["branches"][0].update(context_digest="sha256:" + "0" * 64))
    mutate("conditioned formula tampering", lambda c: c["branches"][0]["conditioned"]["clauses"].pop())
    mutate("visible blocker", lambda c: c["branches"][0]["conditioned"].update(hidden=[]))
    mutate("missing opposing clause check", lambda c: c["branches"][0]["operations"][0].update(opposing_clause_checks=[]))
    mutate("cross-branch context reuse", lambda c: c["branches"][0]["operations"][0].update(context_digest=c["branches"][1]["context_digest"]))
    mutate("backtrack restores child instead of parent", lambda c: c["branches"][0]["backtrack"].update(restored_formula_digest=c["branches"][0]["operations"][0]["target_formula_digest"]))
    mutate("wrong branch ancestry", lambda c: c["branches"][1]["backtrack"].update(restored_context_id="branch-x0"))
    mutate("stale root digest", lambda c: c["root"].update(formula_digest="sha256:" + "f" * 64))
    for name, candidate in mutations:
        mutant_errors, _ = check_trace(candidate)
        if not mutant_errors:
            raise AssertionError("false accept: " + name)
        print("REJECT", name, "-", mutant_errors[0])
    result = {
        "format": "symthaea.dynamic-projected-bce-trace-results",
        "version": 1,
        "claim_ceiling": "two_branch_context_trace_reference_checker_plus_exhaustive_oracle_not_formal_proof_or_D4_trace_replay",
        "certificate_sha256": "sha256:" + hashlib.sha256(CERT.read_bytes()).hexdigest(),
        "checker_sha256": "sha256:" + hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "certificate_check": "passed",
        "summary": summary,
        "negative_controls": len(mutations),
        "negative_controls_rejected": len(mutations),
        "negative_control_labels": [name for name, _ in mutations],
        "runtime": {"python": sys.version.split()[0]},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS {len(mutations)}/{len(mutations)} context/backtrack negative controls")


if __name__ == "__main__":
    main()
