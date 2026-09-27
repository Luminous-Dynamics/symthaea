#!/usr/bin/env python3
"""Independent stdlib oracle for MAT-ENG-TIM-001A."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

SOURCE_HEAD = "e2cff3b2b16e4424b53a2a1838c27bed4f508a64"
SOURCE_PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
DOC_BLOB = "a330661fb8e0462a2a1bab5ca6c0d03ed30ef1d7"
CORPUS_BLOB = "dd677417337cfb1372e556cc90423b7b02e74573"
CORPUS_SHA256 = "b975e03078278879e2907d72c92b3f39d8e59e92ae970076aa9f8347726e046f"
SCHEMA = "mat-eng-tim-001a-reference-v1"
DOC_PATH = Path("docs/engineering/MAT_ENG_TIM_001A.md")
CORPUS_PATH = Path("docs/engineering/data/mat_eng_tim_001a_reference_v1.json")
QUALIFIER_FILES = {
    "scripts/validate_mat_eng_tim_001a.py",
    ".github/workflows/mat-eng-tim-001a1-qualifier.yml",
}
TOL = 1e-12


def fail(msg: str) -> None:
    raise AssertionError(msg)


def require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def close(a, b, tol=TOL):
    return finite_number(a) and finite_number(b) and abs(float(a) - float(b)) <= tol


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def validate_path(p: dict) -> None:
    req = {"R_c1", "t", "k", "A", "R_c2", "R_sink"}
    require(req <= set(p), f"path missing {sorted(req-set(p))}")
    for key in req:
        require(finite_number(p[key]), f"{key} nonfinite/non-numeric")
    require(p["k"] > 0, "k must be >0")
    require(p["A"] > 0, "A must be >0")
    require(p["t"] >= 0, "t must be >=0")
    for key in ("R_c1", "R_c2", "R_sink"):
        require(p[key] >= 0, f"{key} must be >=0")
    if "Qdot" in p:
        require(finite_number(p["Qdot"]) and p["Qdot"] >= 0, "Qdot invalid")
    if "T_sink" in p:
        require(finite_number(p["T_sink"]), "T_sink invalid")


def calc_path(p: dict) -> dict:
    validate_path(p)
    r_tim = p["t"] / (p["k"] * p["A"])
    r_total = p["R_c1"] + r_tim + p["R_c2"] + p["R_sink"]
    out = {"R_tim": r_tim, "R_total": r_total}
    if "Qdot" in p:
        dt = p["Qdot"] * r_total
        out["delta_t"] = dt
        if "T_sink" in p:
            out["T_hot"] = p["T_sink"] + dt
    return out


def group_from_terms(terms: dict) -> dict:
    for key in ("R_c1", "R_tim", "R_c2", "R_sink"):
        require(key in terms and finite_number(terms[key]) and terms[key] >= 0, f"invalid term {key}")
    return {
        "bulk": terms["R_tim"],
        "contact": terms["R_c1"] + terms["R_c2"],
        "sink": terms["R_sink"],
    }


def dominant_from_groups(groups: dict, semantics: dict) -> str:
    require(set(groups) == {"bulk", "contact", "sink"}, "group identity mismatch")
    total = 0.0
    vals = []
    for key, value in groups.items():
        require(finite_number(value) and value >= 0, f"invalid group {key}")
        total += value
    require(total > 0, "group total must be >0")
    for key, value in groups.items():
        vals.append((value / total, key))
    vals.sort(reverse=True)
    top_fraction, top = vals[0]
    second_fraction = vals[1][0]
    rule = semantics["dominance_rule"]
    if top_fraction > rule["dominant_fraction_gt"] and (top_fraction - second_fraction) > rule["separation_margin_gt"]:
        return {"bulk": "BulkTIMDominant", "contact": "ContactDominant", "sink": "SinkDominant"}[top]
    return "Unresolved"


def dominant_from_terms(terms: dict, semantics: dict) -> str:
    return dominant_from_groups(group_from_terms(terms), semantics)


def resolve_ref(name: str, semantics: dict) -> dict:
    ref = semantics["nominal_reference"]
    require(name == ref["id"], f"unknown reference {name}")
    return {k: v for k, v in ref.items() if k != "id"}


def subset_compare(got, expected, path="$"):
    require(isinstance(expected, dict) and isinstance(got, dict), f"{path} expected mapping")
    for key, value in expected.items():
        require(key in got, f"{path}.{key} missing")
        actual = got[key]
        if isinstance(value, float):
            require(close(actual, value), f"{path}.{key}: {actual} != {value}")
        else:
            require(actual == value, f"{path}.{key}: {actual!r} != {value!r}")


def derive(case: dict, data: dict) -> dict:
    kind = case["kind"]
    raw = copy.deepcopy(case["raw"])
    s = data["semantics"]
    out = {}

    if kind == "arithmetic":
        out.update(calc_path(resolve_ref(raw["reference"], s)))

    elif kind == "arithmetic_intervention":
        base = resolve_ref(raw["baseline_ref"], s)
        base_calc = calc_path(base)
        candidate = copy.deepcopy(base)
        candidate.update(raw["overrides"])
        candidate_calc = calc_path(candidate)
        out.update(candidate_calc)
        out["fractional_improvement"] = (base_calc["R_total"] - candidate_calc["R_total"]) / base_calc["R_total"]

    elif kind == "validation":
        base = resolve_ref(raw["baseline_ref"], s)
        base.update(raw["overrides"])
        try:
            calc_path(base)
        except AssertionError:
            out["arithmetic"] = s["validation_rule"]["invalid_input_disposition"]
        else:
            out["arithmetic"] = "Valid"

    elif kind == "bottleneck":
        dom = dominant_from_terms(raw["terms"], s)
        out["bottleneck"] = dom
        out["research"] = s["dominance_research_map"][dom]

    elif kind == "bottleneck_uncertain":
        means = raw["group_means"]
        sigmas = raw["group_sigmas"]
        require(set(means) == {"bulk", "contact", "sink"} and set(sigmas) == set(means), "uncertain groups mismatch")
        nominal = dominant_from_groups(means, s)
        out["nominal_bottleneck"] = nominal
        ranked = sorted(((means[k], k) for k in means), reverse=True)
        top = ranked[0][1]
        second = ranked[1][1]
        z = s["uncertainty_rule"]["z"]
        top_interval = (means[top] - z * sigmas[top], means[top] + z * sigmas[top])
        second_interval = (means[second] - z * sigmas[second], means[second] + z * sigmas[second])
        overlap = max(top_interval[0], second_interval[0]) <= min(top_interval[1], second_interval[1])
        final = "Unresolved" if overlap else nominal
        out["bottleneck"] = final
        out["research"] = s["dominance_research_map"][final]

    elif kind == "migration":
        before = dominant_from_terms(raw["before"], s)
        after = dominant_from_terms(raw["after"], s)
        out["before_bottleneck"] = before
        out["after_bottleneck"] = after
        out["bottleneck"] = s["migration_rule"].get(f"{before}_to_{after}", "NoDeclaredMigration")

    elif kind == "headroom":
        if "base" in raw:
            base = raw["base"]
            r_base = calc_path(base)["R_total"]
            by_t = copy.deepcopy(base)
            by_t["t"] = raw["feasible_t"]
            by_k = copy.deepcopy(base)
            by_k["k"] = raw["feasible_k"]
            gain_t = (r_base - calc_path(by_t)["R_total"]) / r_base
            gain_k = (r_base - calc_path(by_k)["R_total"]) / r_base
            out["research"] = s["headroom_rule"]["thickness_gain_gt_conductivity_gain_disposition"] if gain_t > gain_k else "NoProcessHeadroomAdvantage"
        elif "R_tim_before" in raw:
            gain = (raw["R_tim_before"] - raw["R_tim_after"]) / raw["baseline_R"]
            out["max_system_gain_fraction"] = gain
            out["research"] = s["diminishing_return_rule"]["disposition"] if gain <= s["diminishing_return_rule"]["disposition_if_max_system_gain_lte"] else "BulkKHeadroomMaterial"
        else:
            base = resolve_ref(raw["baseline_ref"], s)
            r_base = calc_path(base)["R_total"]
            candidate = copy.deepcopy(base)
            candidate.update(raw["overrides"])
            r_candidate = calc_path(candidate)["R_total"]
            out["R_total"] = r_candidate
            out["fractional_improvement"] = (r_base - r_candidate) / r_base
            if close(r_candidate, r_base, s["nominal_gain_erased_rule"]["compare_R_total_to_nominal_reference_with_abs_tolerance"]):
                out["research"] = s["nominal_gain_erased_rule"]["equal_disposition"]

    elif kind == "candidate":
        path = resolve_ref(raw["path_ref"], s)
        candidate = raw["candidate"]
        constraints = raw["constraints"]
        if "k" in candidate:
            path["k"] = candidate["k"]
        thermal = calc_path(path)
        out["candidate_R_total"] = thermal["R_total"]
        if "T_hot" in thermal:
            out["candidate_T_hot"] = thermal["T_hot"]
        violated = False
        if constraints.get("isolation_required") and candidate.get("electrical_isolation") is not True:
            violated = True
        if "max_process_temp_c" in constraints and candidate.get("process_temp_c", float("-inf")) > constraints["max_process_temp_c"]:
            violated = True
        if "max_T_hot" in constraints and thermal.get("T_hot", float("inf")) > constraints["max_T_hot"]:
            violated = True
        if violated:
            out["hard_constraint"] = s["candidate_rule"]["hard_constraint_violation_disposition"]
            out["candidate"] = s["candidate_rule"]["blocked_disposition"]
        elif candidate.get("form_ok") is False or candidate.get("lifecycle_pass") is False:
            out["candidate"] = s["candidate_rule"]["blocked_disposition"]
        else:
            out["candidate"] = s["candidate_rule"]["eligible_disposition"]

    elif kind == "robustness":
        lower_tail = raw["mean_k"] - raw["lower_tail_sigma"] * raw["sigma_k"]
        out["lower_tail_k"] = lower_tail
        out["robustness"] = s["robustness_rule"]["pass_disposition"] if lower_tail >= raw["thermal_pass_requires_k_min"] else s["robustness_rule"]["fail_disposition"]

    elif kind == "process":
        if raw["required_form"] not in raw["available_forms"]:
            out["process"] = s["process_form_rule"]["blocked_process_disposition"]
            out["candidate"] = s["process_form_rule"]["blocked_candidate_disposition"]
        else:
            out["process"] = "Eligible"

    elif kind == "evidence":
        if "source" in raw:
            key = f'{raw["source"]}_to_{raw["target"]}'
        else:
            key = f'{raw["source_quantity"]}_to_{raw["target_quantity"]}'
        require(key in s["evidence_rules"], f"unknown evidence mapping {key}")
        out["evidence"] = s["evidence_rules"][key]

    elif kind == "applicability":
        fields = s["applicability_rule"]["coupon_article_fields"]
        if any(raw["coupon"].get(field) != raw["article"].get(field) for field in fields):
            out["candidate"] = s["applicability_rule"]["any_declared_field_mismatch"]
            out["currentness"] = s["applicability_rule"]["mismatch_currentness"]
        else:
            out["candidate"] = "Applicable"

    elif kind == "lifecycle":
        if raw["initial_pass"] and not raw["end_of_life_pass"]:
            out["lifecycle"] = s["lifecycle_rule"]["initial_pass_and_end_of_life_fail"]
            out["candidate"] = s["lifecycle_rule"]["blocked_candidate_disposition"]
        else:
            out["lifecycle"] = "NotBlocked"

    elif kind == "currentness":
        if raw["new_evidence"] and raw["ranking_changed"]:
            out["currentness"] = s["currentness_rule"]["new_evidence_changes_ranking"]
            out["history"] = s["currentness_rule"]["history_rule"]

    elif kind == "diagnosis":
        if raw["observable_terms"] == ["R_total"] and raw["bulk_and_contact_separable"] is False:
            out["bottleneck"] = s["diagnosis_rule"]["R_total_only_and_bulk_contact_not_separable"]
            out["research"] = s["diagnosis_rule"]["unresolved_research"]

    elif kind == "authority":
        if raw["synthetic_all_pass"] and raw["target"] in s["authority_rule"]["synthetic_result_cannot_promote_to"]:
            out["authority"] = s["authority_rule"]["promotion_disposition"]

    else:
        fail(f"unknown case kind {kind}")

    return out


def validate_semantics(data: dict) -> None:
    s = data["semantics"]
    require(s["attribution_groups"] == {"bulk": ["R_tim"], "contact": ["R_c1", "R_c2"], "sink": ["R_sink"]}, "attribution groups drift")
    require(s["dominance_rule"]["separation_margin_definition"] == "largest_group_fraction_minus_second_largest_group_fraction", "separation semantics drift")
    require(s["fractional_improvement_rule"]["baseline_ref"] == "NOMINAL_TIM_PATH_V1", "baseline ref drift")
    require(s["validation_rule"]["invalid_input_disposition"] == "RejectInvalidInput", "validation disposition drift")
    require(s["migration_rule"]["BulkTIMDominant_to_SinkDominant"] == "MigratedToSink", "migration mapping drift")
    require(s["headroom_rule"]["thickness_gain_gt_conductivity_gain_disposition"] == "ProcessInnovationJustified", "headroom disposition drift")
    require(s["candidate_rule"]["hard_constraint_violation_disposition"] == "Violated", "hard-constraint disposition drift")
    require(s["robustness_rule"]["fail_disposition"] == "Blocked", "robustness disposition drift")
    require(s["process_form_rule"]["blocked_process_disposition"] == "Blocked", "process disposition drift")
    require(s["lifecycle_rule"]["blocked_candidate_disposition"] == "Blocked", "lifecycle disposition drift")


def run_mutations(data: dict) -> None:
    s = data["semantics"]
    nominal = resolve_ref("NOMINAL_TIM_PATH_V1", s)
    count = 0

    def must_fail_path(**updates):
        nonlocal count
        path = copy.deepcopy(nominal)
        path.update(updates)
        try:
            calc_path(path)
        except AssertionError:
            count += 1
            return
        fail(f"mutation unexpectedly valid {updates}")

    must_fail_path(k=0)
    must_fail_path(k=-1)
    must_fail_path(A=0)
    must_fail_path(A=-1)
    must_fail_path(t=-1)
    must_fail_path(R_c1=-.1)
    must_fail_path(R_c2=-.1)
    must_fail_path(R_sink=-.1)
    must_fail_path(Qdot=-1)
    must_fail_path(k=float("nan"))
    must_fail_path(A=float("inf"))

    require(dominant_from_terms({"R_c1": .3, "R_tim": .1, "R_c2": .3, "R_sink": .1}, s) == "ContactDominant", "contact grouping mutation")
    count += 1
    require(dominant_from_terms({"R_c1": .1, "R_tim": .5, "R_c2": .1, "R_sink": .3}, s) == "Unresolved", "strict dominance mutation")
    count += 1
    require(dominant_from_terms({"R_c1": .185, "R_tim": .51, "R_c2": .185, "R_sink": .12}, s) == "Unresolved", "separation-margin mutation")
    count += 1

    case = {"kind": "bottleneck_uncertain", "raw": {"group_means": {"bulk": .25, "contact": .45, "sink": .1}, "group_sigmas": {"bulk": .12, "contact": .12, "sink": .04}}}
    require(derive(case, data)["bottleneck"] == "Unresolved", "uncertainty-overlap mutation")
    count += 1
    case["raw"]["group_sigmas"] = {"bulk": .01, "contact": .01, "sink": .01}
    require(derive(case, data)["bottleneck"] == "ContactDominant", "uncertainty non-overlap mutation")
    count += 1

    case = {"kind": "arithmetic_intervention", "raw": {"baseline_ref": "NOMINAL_TIM_PATH_V1", "overrides": {"k": 10.0}}}
    require(close(derive(case, data)["fractional_improvement"], .1), "explicit baseline mutation")
    count += 1
    case = {"kind": "migration", "raw": {"before": {"R_c1": .1, "R_tim": .6, "R_c2": .1, "R_sink": .2}, "after": {"R_c1": .05, "R_tim": .1, "R_c2": .05, "R_sink": .3}}}
    require(derive(case, data)["bottleneck"] == "MigratedToSink", "migration mutation")
    count += 1

    case = {"kind": "headroom", "raw": {"base": {"R_c1": .05, "t": .001, "k": 5.0, "A": .001, "R_c2": .05, "R_sink": .1}, "feasible_t": .0002, "feasible_k": 6.0}}
    require(derive(case, data)["research"] == "ProcessInnovationJustified", "headroom mutation")
    count += 1
    case = {"kind": "headroom", "raw": {"baseline_ref": "NOMINAL_TIM_PATH_V1", "overrides": {"k": 10.0, "t": .001}}}
    require(derive(case, data)["research"] == "NominalGainErased", "gain-erased mutation")
    count += 1

    case = {"kind": "candidate", "raw": {"path_ref": "NOMINAL_TIM_PATH_V1", "candidate": {"k": 10.0, "electrical_isolation": True, "process_temp_c": 120.0, "form_ok": True, "lifecycle_pass": True}, "constraints": {"max_T_hot": 70.0, "isolation_required": True, "max_process_temp_c": 150.0}}}
    result = derive(case, data)
    require(close(result["candidate_R_total"], .45) and result["candidate"] == "EligibleBounded", "candidate k override mutation")
    count += 1
    case["raw"]["candidate"]["electrical_isolation"] = False
    require(derive(case, data)["hard_constraint"] == "Violated", "isolation mutation")
    count += 1
    case["raw"]["candidate"]["electrical_isolation"] = True
    case["raw"]["candidate"]["process_temp_c"] = 220.0
    require(derive(case, data)["candidate"] == "Blocked", "process-temperature mutation")
    count += 1

    case = {"kind": "robustness", "raw": {"mean_k": 8.0, "sigma_k": 2.5, "lower_tail_sigma": 2.0, "thermal_pass_requires_k_min": 4.0}}
    result = derive(case, data)
    require(close(result["lower_tail_k"], 3.0) and result["robustness"] == "Blocked", "robustness mutation")
    count += 1

    case = {"kind": "process", "raw": {"properties_pass": True, "required_form": "100um_bondline", "available_forms": ["bulk_plate"]}}
    require(derive(case, data)["candidate"] == "Blocked", "form mutation")
    count += 1
    case = {"kind": "evidence", "raw": {"source": "HandbookBulkK", "target": "ProcessConditionedInterfaceProperty"}}
    require(derive(case, data)["evidence"] == "EvidenceLimited", "evidence-class mutation")
    count += 1
    case = {"kind": "evidence", "raw": {"source_quantity": "BulkThermalConductivity", "target_quantity": "ContactThermalResistance"}}
    require(derive(case, data)["evidence"] == "QuantityMismatch", "quantity mutation")
    count += 1
    case = {"kind": "applicability", "raw": {"coupon": {"pressure_kpa": 100.0, "thickness_um": 100.0, "process": "P1"}, "article": {"pressure_kpa": 300.0, "thickness_um": 200.0, "process": "P2"}}}
    require(derive(case, data)["candidate"] == "ApplicabilityMismatch", "applicability mutation")
    count += 1
    case = {"kind": "lifecycle", "raw": {"initial_pass": True, "end_of_life_pass": False}}
    require(derive(case, data)["lifecycle"] == "Blocked", "lifecycle mutation")
    count += 1
    case = {"kind": "currentness", "raw": {"old_generation": "G1", "new_evidence": "contact_R_updated", "ranking_changed": True}}
    require(derive(case, data)["history"] == "PreservePriorRanking", "history mutation")
    count += 1
    case = {"kind": "diagnosis", "raw": {"observable_terms": ["R_total"], "bulk_and_contact_separable": False}}
    require(derive(case, data)["bottleneck"] == "Unresolved", "observability mutation")
    count += 1
    case = {"kind": "authority", "raw": {"synthetic_all_pass": True, "target": "ProductQualified"}}
    require(derive(case, data)["authority"] == "RejectPromotion", "authority mutation")
    count += 1

    require(count >= 30, f"mutation suite too small: {count}")


def verify_git_shape() -> None:
    require(git("rev-parse", "HEAD^") == SOURCE_HEAD, "qualifier parent is not exact source head")
    require(git("rev-parse", "HEAD^^") == SOURCE_PARENT, "source parent changed")
    require(git("rev-parse", f"HEAD^:{DOC_PATH.as_posix()}") == DOC_BLOB, "contract blob mismatch")
    require(git("rev-parse", f"HEAD^:{CORPUS_PATH.as_posix()}") == CORPUS_BLOB, "corpus blob mismatch")
    changed = set(filter(None, git("diff", "--name-only", "HEAD^", "HEAD").splitlines()))
    require(changed == QUALIFIER_FILES, f"qualifier scope mismatch {sorted(changed)}")


def main() -> None:
    verify_git_shape()
    raw = CORPUS_PATH.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CORPUS_SHA256, "corpus digest drift")
    data = json.loads(raw.decode())
    require(raw == canonical(data), "corpus is not exact compact canonical JSON")
    require(data.get("schema") == SCHEMA, "schema drift")
    require(data.get("issue") == 6191, "issue binding drift")
    require(data.get("source_kind") == "synthetic-reference", "source kind drift")
    require(data.get("claim_ceiling") == "software semantics and synthetic analytical known answers only; no real TIM/package/product authority", "claim ceiling drift")
    validate_semantics(data)
    cases = data.get("cases")
    require(isinstance(cases, list) and len(cases) == 28, "case count != 28")
    ids = [case.get("id") for case in cases]
    require(ids == [f"T{i:02d}" for i in range(1, 29)], "case order drift")
    require(len(set(ids)) == 28, "duplicate case ID")
    for case in cases:
        require(set(case) == {"id", "title", "kind", "raw", "expected"}, f"{case.get('id')}: case shape drift")
        subset_compare(derive(case, data), case["expected"], f"${case['id']}")
    run_mutations(data)
    print(f"PASS_MAT_ENG_TIM_001A oracle_cases=28 mutations>=30 sha256={CORPUS_SHA256}")


if __name__ == "__main__":
    main()
