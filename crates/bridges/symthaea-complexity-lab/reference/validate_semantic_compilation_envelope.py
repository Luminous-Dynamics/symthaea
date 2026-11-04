#!/usr/bin/env python3
"""Finite reference validator for ETH-027 semantic compilation envelopes.

Requires Python 3.10+ and jsonschema>=4.18. Research tooling, not a proof checker.
"""
from __future__ import annotations
import copy, json, math, pathlib, sys
from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "semantic_compilation_envelope.v1.schema.json"
FIXTURE = pathlib.Path(__file__).with_name("semantic_compilation_envelope.pwmc.fixture.json")


def load(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON key: {k}")
            out[k] = v
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def reduced(q):
    try:
        n, d = int(q["numerator"]), int(q["denominator"])
        return d > 0 and math.gcd(abs(n), d) == 1
    except (KeyError, TypeError, ValueError):
        return False


def semantic_errors(x):
    errors = []
    p, v = x["problem"], x["problem"]["variables"]
    universe, visible, hidden = set(v["universe"]), set(v["visible"]), set(v["hidden"])
    if visible & hidden: errors.append("visible/hidden sets overlap")
    if visible | hidden != universe: errors.append("visible/hidden do not partition universe")
    contracts = {
        "#sat": ("all_assignments", "N", None, ("none", "count")),
        "wmc": ("all_assignments", "Q", "universe", ("none", "weighted_sum")),
        "pmc": ("exists_hidden_per_visible_assignment", "N", None, ("exists", "count")),
        "pwmc": ("exists_hidden_per_visible_assignment", "Q", "visible", ("exists", "weighted_sum")),
        "amc-complex": ("all_assignments", "Q(i)", "universe", ("none", "weighted_sum")),
    }
    q, domain, scope, aggregation = contracts[p["contract"]]
    if (p["quantifier_semantics"], p["coefficient_domain"]) != (q, domain):
        errors.append("contract/quantifier/domain mismatch")
    plan = p["aggregation_plan"]
    if (plan["projection"], plan["final"]) != aggregation:
        errors.append("contract/aggregation-plan mismatch")
    if q == "all_assignments" and (hidden or visible != universe):
        errors.append("full-assignment contract must expose the entire universe")
    wm = p.get("weight_map")
    if scope is None and wm is not None: errors.append("unweighted contract carries a weight map")
    if scope is not None:
        expected = universe if scope == "universe" else visible
        entries = [] if wm is None else wm["entries"]
        names = [e["variable"] for e in entries]
        if wm is None or wm["domain"] != domain or len(names) != len(set(names)) or set(names) != expected:
            errors.append("weight-map scope/domain mismatch")
        for e in entries:
            for z in (e["false_weight"], e["true_weight"]):
                qs = [z] if domain == "Q" else [z.get("real", {}), z.get("imaginary", {})]
                if not all(reduced(a) for a in qs): errors.append("noncanonical or wrong-domain literal weight")
                if domain == "Q(i)" and set(z) != {"real", "imaginary"}: errors.append("malformed Gaussian rational")
    stages = x["stages"]
    ids = [s["stage_id"] for s in stages]
    if len(ids) != len(set(ids)): errors.append("duplicate stage IDs")
    if stages[0]["input"]["digest"] != p["source"]["bytes_digest"]: errors.append("first stage is not bound to source")
    for a, b in zip(stages, stages[1:]):
        if a["output"]["digest"] != b["input"]["digest"]: errors.append("stage digest chain discontinuity")
    if stages[-1]["output"]["digest"] != x["result"]["artifact_digest"]: errors.append("result does not bind final stage")
    required = {
        "satisfiability_equivalence": "satisfiability_equivalence",
        "full_model_equivalence": "full_model_equivalence",
        "projected_support_equivalence": "projected_support_equivalence",
        "model_bijection": "model_bijection",
        "weighted_value_preservation": "weighted_value_preservation",
    }
    for s in stages:
        by_kind = {}
        for o in s["obligations"]:
            for evidence_name in ("proof", "tests"):
                evidence = o.get(evidence_name)
                if evidence is not None and (
                    evidence.get("input_representation_digest") != s["input"]["digest"]
                    or evidence.get("output_representation_digest") != s["output"]["digest"]
                ):
                    errors.append(f"{evidence_name} evidence is not bound to stage {s['stage_id']} input/output")
            by_kind.setdefault(o["kind"], []).append(o)
        k = required.get(s["relation"])
        if k and (k not in by_kind or any(o["status"] in {"unsupported", "failed"} for o in by_kind[k])):
            errors.append(f"stage {s['stage_id']} lacks an established {k} obligation")
        if s["relation"] == "support_to_scalar_evaluation" and not ({"weighted_value_preservation", "exact_arithmetic_replay"} & set(by_kind)):
            errors.append("scalar stage lacks arithmetic obligation")
        if s["relation"] == "satisfiability_equivalence" and not ({"full_model_equivalence", "projected_support_equivalence", "model_bijection", "weighted_value_preservation"} & set(by_kind)):
            errors.append("SAT-only relation cannot certify a counting contract")
    if p["contract"] in {"pmc", "pwmc"} and not any(o["kind"] == "projected_support_equivalence" for s in stages for o in s["obligations"]):
        errors.append("projected contract lacks projected-support obligation")
    result = x["result"]
    if result["arithmetic"]["domain"] != domain: errors.append("result arithmetic domain mismatch")
    if result["arithmetic"]["status"] == "computed_exact":
        value = result["arithmetic"]["value"]
        if domain == "N":
            if not isinstance(value, str) or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
                errors.append("noncanonical natural result")
        elif domain == "Q":
            if not reduced(value): errors.append("noncanonical rational result")
        elif not all(reduced(value.get(k, {})) for k in ("real", "imaginary")):
            errors.append("noncanonical Gaussian-rational result")
    replay = x["independent_replay"]
    if replay["input_representation_digest"] != stages[-1]["input"]["digest"]: errors.append("replay input digest mismatch")
    if replay["status"] == "matched" and replay["replayed_result_digest"] != result["artifact_digest"]:
        errors.append("replay result digest mismatch")
    return sorted(set(errors))


def errors(x, schema):
    validator = Draft202012Validator(schema)
    out = ["schema: " + e.message for e in validator.iter_errors(x)]
    return out if out else ["semantic: " + e for e in semantic_errors(x)]


def reject(label, x, schema):
    found = errors(x, schema)
    if not found: raise AssertionError("accepted invalid case: " + label)
    print("REJECTED", label, "—", found[0])


def main():
    schema, fixture = load(SCHEMA), load(FIXTURE)
    Draft202012Validator.check_schema(schema)
    if errors(fixture, schema): raise SystemExit("fixture rejected: " + "; ".join(errors(fixture, schema)))
    print("ACCEPTED valid projected-WMC fixture")
    zero = copy.deepcopy(fixture)
    zero["result"]["arithmetic"]["value"] = {"numerator": "0", "denominator": "1"}
    assert not errors(zero, schema)
    print("ACCEPTED exact-zero arithmetic with nonempty support")
    cases = []
    a=copy.deepcopy(fixture); a["problem"]["variables"]["hidden"]=["d","h"]; cases.append(("overlapping partition",a))
    a=copy.deepcopy(fixture); a["problem"]["variables"]["hidden"]=[]; cases.append(("incomplete partition",a))
    a=copy.deepcopy(fixture); a["problem"]["quantifier_semantics"]="all_assignments"; cases.append(("wrong PWMC quantifier",a))
    a=copy.deepcopy(fixture); a["problem"]["weight_map"]["entries"].append({"variable":"h","false_weight":{"numerator":"1","denominator":"1"},"true_weight":{"numerator":"1","denominator":"1"}}); cases.append(("hidden variable weighted in PWMC",a))
    a=copy.deepcopy(fixture); a["stages"][0]["output"]["digest"]="sha256:"+"0"*64; cases.append(("broken transformation digest chain",a))
    a=copy.deepcopy(fixture); a["stages"][0]["obligations"][0].update(status="proof_checked"); cases.append(("proof claim without proof artifact",a))
    a=copy.deepcopy(fixture); a["problem"]["weight_map"]["entries"][0]["true_weight"]={"numerator":"4","denominator":"6"}; cases.append(("unreduced rational",a))
    a=copy.deepcopy(fixture); a["problem"]["contract"]="wmc"; a["problem"]["quantifier_semantics"]="all_assignments"; a["problem"]["weight_map"]["entries"].append({"variable":"h","false_weight":{"numerator":"1","denominator":"1"},"true_weight":{"numerator":"1","denominator":"1"}}); cases.append(("hidden variable in full WMC universe",a))
    a=copy.deepcopy(fixture); a["independent_replay"]["input_representation_digest"]="sha256:"+"f"*64; cases.append(("replay bound to wrong input",a))
    a=copy.deepcopy(fixture); a["stages"][0]["relation"]="satisfiability_equivalence"; a["stages"][0]["obligations"]=[{"obligation_id":"sat-only","kind":"satisfiability_equivalence","status":"tested_only","tests":{"suite_id":"negative-control","artifact_digest":fixture["stages"][0]["obligations"][0]["tests"]["artifact_digest"],"case_count":1,"claim_ceiling":"finite_test_evidence","input_representation_digest":a["stages"][0]["input"]["digest"],"output_representation_digest":a["stages"][0]["output"]["digest"]}}]; cases.append(("SAT-only proof used for projected count",a))
    a=copy.deepcopy(fixture); a["stages"][0]["obligations"][0]["tests"]["output_representation_digest"]="sha256:"+"e"*64; cases.append(("evidence bound to wrong stage output",a))
    a=copy.deepcopy(fixture); a["problem"]["aggregation_plan"]["final"]="count"; cases.append(("PWMC uses wrong final aggregation operator",a))
    for name, case in cases: reject(name,case,schema)
    print(f"PASS: 14 semantic-contract checks (2 positive cases, {len(cases)} negative controls)")
    return 0

if __name__ == "__main__": sys.exit(main())
