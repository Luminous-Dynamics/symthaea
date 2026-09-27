#!/usr/bin/env python3
"""Independent SE-FOSS-003A cognition→engineering proposal oracle.

Python stdlib only. This qualifier intentionally imports no Symthaea code.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys

SOURCE_HEAD = "541cdecd268718d5164b12c7291f26558f7b1dae"
SOURCE_PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
CONTRACT_PATH = "docs/engineering/SE_FOSS_003A.md"
CORPUS_PATH = "docs/engineering/data/se_foss_003a_reference_v1.json"
CONTRACT_BLOB = "8278e612ac54e8bb0f47a6ae605136c284fce3e1"
CORPUS_BLOB = "fb5484df52cf30fc09ab21f3b67ff497264b5697"
CORPUS_SHA256 = "ee1a66b8fa4fa3ccb312e14252363eb6c05c960f0d195de1623a3c770f6f59b3"
SCHEMA = "se-foss-003a-reference-v1"
CASE_IDS = [f"C{i:02d}" for i in range(1, 25)]
CLAIM_CEILING = (
    "software proposal-routing semantics only; no cognitive correctness, engineering truth, "
    "causal truth, requirement acceptance, proof validity, experiment/fabrication/procurement/actuation authority"
)

TOP_KEYS = {"schema", "issue", "source_kind", "claim_ceiling", "vocabularies", "cases"}
CASE_KEYS = {"id", "producer", "output", "raw", "expected"}
VOCABULARIES = {
    "proposal": {
        "DraftOnly", "HypothesisOnly", "RouteToMatEng", "RetrievalOnly", "Blocked",
        "RequestOnly", "PriorityOnly", "ModelRelativeOnly", "CandidateOnly",
        "RouteToSEModel", "SoftwareSemanticsOnly",
    },
    "target_currentness": {"ReviewRebindRequired"},
    "evidence_promotion": {
        "Rejected", "NoIndependenceGain", "RetainUnderlyingClass", "SimulationLineageRetained",
        "NotObservation", "NotQualifiedDesign", "NotCheckedProof", "NoRealEngineeringAuthority",
    },
    "causal_admission": {"Rejected", "BoundedModelAdmission"},
    "learning_use": {"AllowedBounded", "RejectedContamination"},
    "prospective_prediction": {"ValidCommitment", "RejectedRetrospective", "DiscrepancyPreserved"},
    "execution_authority": {"None", "NoNewAuthority", "AuthorityUnchanged"},
}
PRODUCERS = {
    "Broca", "HDC", "EngineeringMemory", "LTC_CfC", "FEP", "CausalReasoning",
    "WorldModel", "GlobalWorkspace", "OperationsResearch", "FormalProofSuggestion", "SyntheticCorpus",
}
OUTPUTS = {
    "RequirementDraft", "AssumptionDraft", "RelationHypothesis", "MaterialOrProcessCandidate",
    "RetrievalSuggestion", "DegradationHypothesis", "MeasurementRequest", "ExperimentRequest",
    "PrioritySuggestion", "CausalHypothesis", "ScenarioRequest", "DesignAlternative",
    "FormalProofRequest", "ModelRevisionRequest",
}


def fail(message):
    raise AssertionError(message)


def run(*args):
    return subprocess.check_output(args).decode("utf-8").strip()


def git_bytes(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def resolve_source(ref):
    resolved = run("git", "rev-parse", ref)
    if resolved != SOURCE_HEAD:
        fail(f"source ref resolves to {resolved}, expected {SOURCE_HEAD}")
    parent = run("git", "rev-parse", f"{resolved}^")
    if parent != SOURCE_PARENT:
        fail(f"source parent {parent}, expected {SOURCE_PARENT}")
    contract_blob = run("git", "rev-parse", f"{resolved}:{CONTRACT_PATH}")
    corpus_blob = run("git", "rev-parse", f"{resolved}:{CORPUS_PATH}")
    if contract_blob != CONTRACT_BLOB:
        fail(f"contract blob {contract_blob}, expected {CONTRACT_BLOB}")
    if corpus_blob != CORPUS_BLOB:
        fail(f"corpus blob {corpus_blob}, expected {CORPUS_BLOB}")
    corpus_bytes = git_bytes(resolved, CORPUS_PATH)
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    if digest != CORPUS_SHA256:
        fail(f"corpus SHA-256 {digest}, expected {CORPUS_SHA256}")
    return json.loads(corpus_bytes.decode("utf-8"))


def exact_keys(obj, required, context):
    if set(obj) != set(required):
        fail(f"{context}: keys {sorted(obj)} != {sorted(required)}")


def raw_keys(case, required, optional=()):
    keys = set(case["raw"])
    req, opt = set(required), set(optional)
    if not req <= keys or not keys <= req | opt:
        fail(f"{case['id']}: raw keys {sorted(keys)} invalid for {case['producer']}/{case['output']}")


def validate_case_shape(case):
    p, o, r = case["producer"], case["output"], case["raw"]
    exact_keys(case, CASE_KEYS, case.get("id", "case"))
    if p not in PRODUCERS or o not in OUTPUTS:
        fail(f"{case['id']}: unknown producer/output {p}/{o}")
    if not isinstance(r, dict) or not isinstance(case["expected"], dict):
        fail(f"{case['id']}: raw/expected must be objects")

    if p == "Broca" and o == "RequirementDraft":
        raw_keys(case, {"content", "confidence"})
        if not isinstance(r["content"], str) or r["confidence"] not in {"low", "medium", "high"}:
            fail(f"{case['id']}: malformed requirement draft")
    elif p == "Broca" and o == "AssumptionDraft":
        raw_keys(case, {"content", "tone"})
        if not isinstance(r["content"], str) or not isinstance(r["tone"], str):
            fail(f"{case['id']}: malformed assumption draft")
    elif p == "HDC" and o == "RelationHypothesis":
        if "target_generation" in r or "current_generation" in r:
            raw_keys(case, {"target_generation", "current_generation"})
            if r["target_generation"] == r["current_generation"]:
                fail(f"{case['id']}: currentness fixture requires changed generation")
        else:
            raw_keys(case, {"similarity", "relation"})
            if not 0.0 <= float(r["similarity"]) <= 1.0 or r["relation"] != "structural_analogue":
                fail(f"{case['id']}: HDC relation is not bounded structural analogy")
    elif p == "HDC" and o == "MaterialOrProcessCandidate":
        raw_keys(case, {"candidate", "basis"})
        if r["basis"] != "vector_similarity":
            fail(f"{case['id']}: unsupported HDC material-candidate basis")
    elif p == "HDC" and o == "RetrievalSuggestion":
        raw_keys(case, {"hits", "independence_group"})
        if not isinstance(r["hits"], int) or r["hits"] < 1 or r["independence_group"] != "same-source":
            fail(f"{case['id']}: retrieval independence fixture malformed")
    elif p == "EngineeringMemory" and o == "RetrievalSuggestion":
        if "partition" in r:
            raw_keys(case, {"partition", "use_policy"})
            if r != {"partition": "HiddenEvaluatorOracle", "use_policy": "strict"}:
                fail(f"{case['id']}: hidden-evaluator firewall malformed")
        else:
            raw_keys(case, {"source_class", "use_policy"})
            if r["source_class"] not in {"PhysicalExperimentObservation", "ExternalSimulation"}:
                fail(f"{case['id']}: unsupported memory source class")
            if r["use_policy"] != "proposal_allowed":
                fail(f"{case['id']}: unsupported memory use policy")
    elif p == "LTC_CfC" and o == "DegradationHypothesis":
        if "prediction" in r or "observation" in r:
            raw_keys(case, {"committed_before_outcome", "prediction", "observation"})
            if r["committed_before_outcome"] is not True:
                fail(f"{case['id']}: discrepancy case must be prospectively committed")
        else:
            raw_keys(case, {"prediction_time", "outcome_time", "committed_before_outcome"})
            if r["committed_before_outcome"] not in {True, False}:
                fail(f"{case['id']}: commitment flag must be boolean")
    elif p == "FEP" and o in {"MeasurementRequest", "ExperimentRequest"}:
        raw_keys(case, {"epistemic_utility", "requested"})
        if not 0.0 <= float(r["epistemic_utility"]) <= 1.0:
            fail(f"{case['id']}: epistemic utility outside fixture range")
    elif p == "FEP" and o == "PrioritySuggestion":
        raw_keys(case, {"authorized_analysis", "ranked"})
        if sorted(r["authorized_analysis"]) != sorted(r["ranked"]):
            fail(f"{case['id']}: FEP priority changed authorized set")
    elif p == "CausalReasoning" and o == "CausalHypothesis":
        if r.get("basis") == "correlation":
            raw_keys(case, {"basis", "model_admission"})
            if r["model_admission"] is not False:
                fail(f"{case['id']}: correlation cannot self-admit")
        elif r.get("basis") == "evidence_bound_claim":
            raw_keys(case, {"basis", "model_profile", "model_admission"})
            if not r["model_profile"] or r["model_admission"] is not True:
                fail(f"{case['id']}: bounded causal admission malformed")
        else:
            fail(f"{case['id']}: unknown causal basis")
    elif p == "WorldModel" and o == "ScenarioRequest":
        if "result_kind" in r:
            raw_keys(case, {"result_kind", "observed"})
            if r != {"result_kind": "counterfactual", "observed": False}:
                fail(f"{case['id']}: counterfactual must remain unobserved")
        else:
            raw_keys(case, {"executed", "physical_observation"})
            if r != {"executed": False, "physical_observation": False}:
                fail(f"{case['id']}: unexecuted scenario cannot be physical observation")
    elif p == "GlobalWorkspace" and o == "PrioritySuggestion":
        raw_keys(case, {"broadcast_count", "original_authority"})
        if not isinstance(r["broadcast_count"], int) or r["broadcast_count"] < 1 or r["original_authority"] != "proposal":
            fail(f"{case['id']}: workspace rebroadcast fixture malformed")
    elif p == "OperationsResearch" and o == "DesignAlternative":
        raw_keys(case, {"pareto_optimal", "validated"})
        if r != {"pareto_optimal": True, "validated": False}:
            fail(f"{case['id']}: Pareto candidate must remain unvalidated")
    elif p == "FormalProofSuggestion" and o == "FormalProofRequest":
        raw_keys(case, {"statement", "proof_receipt"})
        if not isinstance(r["statement"], str) or r["proof_receipt"] is not False:
            fail(f"{case['id']}: proof suggestion cannot contain proof receipt")
    elif p == "CausalReasoning" and o == "ModelRevisionRequest":
        raw_keys(case, {"residual_count", "review_complete"})
        if r["residual_count"] < 1 or r["review_complete"] is not False:
            fail(f"{case['id']}: model revision request bypasses review")
    elif p == "SyntheticCorpus" and o == "PrioritySuggestion":
        raw_keys(case, {"synthetic_all_pass"})
        if r["synthetic_all_pass"] is not True:
            fail(f"{case['id']}: synthetic ceiling fixture malformed")
    else:
        fail(f"{case['id']}: unsupported producer/output shape {p}/{o}")


def validate_document(doc):
    exact_keys(doc, TOP_KEYS, "document")
    if doc["schema"] != SCHEMA or doc["issue"] != 6195 or doc["source_kind"] != "synthetic-reference":
        fail("frozen document header mismatch")
    if doc["claim_ceiling"] != CLAIM_CEILING:
        fail("claim ceiling drift")
    if set(doc["vocabularies"]) != set(VOCABULARIES):
        fail("vocabulary dimensions drift")
    for key, values in doc["vocabularies"].items():
        if set(values) != VOCABULARIES[key] or len(values) != len(set(values)):
            fail(f"vocabulary drift in {key}")
    ids = [case.get("id") for case in doc["cases"]]
    if ids != CASE_IDS:
        fail(f"case order drift: {ids}")
    for case in doc["cases"]:
        validate_case_shape(case)
        for dimension, value in case["expected"].items():
            if dimension not in VOCABULARIES or value not in VOCABULARIES[dimension]:
                fail(f"{case['id']}: invalid expected disposition {dimension}={value}")


def derive(case):
    p, o, r = case["producer"], case["output"], case["raw"]
    validate_case_shape(case)
    if p == "Broca":
        result = {"proposal": "DraftOnly", "evidence_promotion": "Rejected"}
        if o == "RequirementDraft":
            result["execution_authority"] = "None"
        return result
    if p == "HDC" and o == "RelationHypothesis":
        result = {"proposal": "HypothesisOnly"}
        if "target_generation" in r:
            result["target_currentness"] = "ReviewRebindRequired"
        else:
            result["evidence_promotion"] = "Rejected"
        return result
    if p == "HDC" and o == "MaterialOrProcessCandidate":
        return {"proposal": "RouteToMatEng", "evidence_promotion": "Rejected"}
    if p == "HDC" and o == "RetrievalSuggestion":
        return {"proposal": "RetrievalOnly", "evidence_promotion": "NoIndependenceGain"}
    if p == "EngineeringMemory":
        if "partition" in r:
            return {"learning_use": "RejectedContamination", "proposal": "Blocked"}
        evidence = "RetainUnderlyingClass" if r["source_class"] == "PhysicalExperimentObservation" else "SimulationLineageRetained"
        return {"proposal": "RetrievalOnly", "learning_use": "AllowedBounded", "evidence_promotion": evidence}
    if p == "LTC_CfC":
        if "prediction" in r:
            disposition = "DiscrepancyPreserved" if r["prediction"] != r["observation"] else "ValidCommitment"
        else:
            disposition = "ValidCommitment" if r["committed_before_outcome"] else "RejectedRetrospective"
        return {"prospective_prediction": disposition, "proposal": "HypothesisOnly"}
    if p == "FEP" and o in {"MeasurementRequest", "ExperimentRequest"}:
        return {"proposal": "RequestOnly", "execution_authority": "None"}
    if p == "FEP" and o == "PrioritySuggestion":
        return {"proposal": "PriorityOnly", "execution_authority": "NoNewAuthority"}
    if p == "CausalReasoning" and o == "CausalHypothesis":
        admission = "BoundedModelAdmission" if r["basis"] == "evidence_bound_claim" else "Rejected"
        return {"proposal": "HypothesisOnly", "causal_admission": admission}
    if p == "WorldModel" and o == "ScenarioRequest":
        proposal = "ModelRelativeOnly" if "result_kind" in r else "RequestOnly"
        return {"proposal": proposal, "evidence_promotion": "NotObservation"}
    if p == "GlobalWorkspace":
        return {"proposal": "PriorityOnly", "execution_authority": "AuthorityUnchanged"}
    if p == "OperationsResearch":
        return {"proposal": "CandidateOnly", "evidence_promotion": "NotQualifiedDesign"}
    if p == "FormalProofSuggestion":
        return {"proposal": "RequestOnly", "evidence_promotion": "NotCheckedProof"}
    if p == "CausalReasoning" and o == "ModelRevisionRequest":
        return {"proposal": "RouteToSEModel", "evidence_promotion": "Rejected"}
    if p == "SyntheticCorpus":
        return {"proposal": "SoftwareSemanticsOnly", "execution_authority": "None", "evidence_promotion": "NoRealEngineeringAuthority"}
    fail(f"{case['id']}: no derivation rule")


def derive_all(doc):
    return {case["id"]: derive(case) for case in doc["cases"]}


def check_expected(doc):
    derived = derive_all(doc)
    for case in doc["cases"]:
        if derived[case["id"]] != case["expected"]:
            fail(f"{case['id']}: derived {derived[case['id']]} != expected {case['expected']}")
    return derived


def case(doc, cid):
    return next(item for item in doc["cases"] if item["id"] == cid)


def mutation_suite(doc):
    baseline = derive_all(doc)
    mutations = [
        ("schema drift", lambda d: d.__setitem__("schema", "bad")),
        ("missing case", lambda d: d["cases"].pop()),
        ("reordered cases", lambda d: d["cases"].__setitem__(slice(0, 2), list(reversed(d["cases"][:2])))),
        ("unknown top field", lambda d: d.__setitem__("authority", "qualified")),
        ("Broca accepted promotion", lambda d: case(d,"C01")["expected"].__setitem__("proposal", "Accepted")),
        ("HDC physical equivalence", lambda d: case(d,"C03")["raw"].__setitem__("relation", "physical_equivalence")),
        ("HDC material evidence", lambda d: case(d,"C04")["expected"].__setitem__("evidence_promotion", "RetainUnderlyingClass")),
        ("independence inflation", lambda d: case(d,"C05")["raw"].__setitem__("independence_group", "independent")),
        ("physical source erased", lambda d: case(d,"C06")["raw"].__setitem__("source_class", "UnknownLegacy")),
        ("simulation promoted physical", lambda d: case(d,"C07")["raw"].__setitem__("source_class", "PhysicalExperimentObservation")),
        ("hidden evaluator visible", lambda d: case(d,"C08")["raw"].__setitem__("use_policy", "proposal_allowed")),
        ("prospective moved late", lambda d: case(d,"C09")["raw"].__setitem__("committed_before_outcome", False)),
        ("contradiction rewritten", lambda d: case(d,"C11")["raw"].__setitem__("observation", "increase")),
        ("measurement becomes observation", lambda d: case(d,"C12").__setitem__("output", "ScenarioRequest")),
        ("experiment authority promotion", lambda d: case(d,"C13")["expected"].__setitem__("execution_authority", "AuthorityUnchanged")),
        ("priority changes authorization", lambda d: case(d,"C14")["raw"].__setitem__("ranked", ["B", "C"])),
        ("correlation self admission", lambda d: case(d,"C15")["raw"].__setitem__("model_admission", True)),
        ("bounded causal universal promotion", lambda d: case(d,"C16")["expected"].__setitem__("causal_admission", "Rejected")),
        ("counterfactual becomes observed", lambda d: case(d,"C17")["raw"].__setitem__("observed", True)),
        ("workspace authority promotion", lambda d: case(d,"C18")["raw"].__setitem__("original_authority", "qualified")),
        ("Pareto becomes validated", lambda d: case(d,"C19")["raw"].__setitem__("validated", True)),
        ("proof request gets receipt", lambda d: case(d,"C20")["raw"].__setitem__("proof_receipt", True)),
        ("scenario becomes physical observation", lambda d: case(d,"C21")["raw"].__setitem__("physical_observation", True)),
        ("model review bypass", lambda d: case(d,"C22")["raw"].__setitem__("review_complete", True)),
        ("stale generation made current", lambda d: case(d,"C23")["raw"].__setitem__("current_generation", "G1")),
        ("synthetic pass gets real authority", lambda d: case(d,"C24")["expected"].__setitem__("evidence_promotion", "RetainUnderlyingClass")),
    ]
    passed = []
    for label, mutate in mutations:
        candidate = copy.deepcopy(doc)
        mutate(candidate)
        rejected = False
        changed = False
        try:
            validate_document(candidate)
            changed = derive_all(candidate) != baseline
            check_expected(candidate)
        except (AssertionError, KeyError, TypeError, ValueError):
            rejected = True
        if not rejected and not changed:
            fail(f"mutation neither rejected nor changed derived semantics: {label}")
        passed.append(label)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", default=SOURCE_HEAD)
    args = parser.parse_args()
    doc = resolve_source(args.source_ref)
    validate_document(doc)
    check_expected(doc)
    mutations = mutation_suite(doc)
    print(json.dumps({
        "schema": SCHEMA,
        "source_head": SOURCE_HEAD,
        "case_count": len(CASE_IDS),
        "mutation_count": len(mutations),
        "status": "PASS",
        "claim_ceiling": CLAIM_CEILING,
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        sys.exit(1)
