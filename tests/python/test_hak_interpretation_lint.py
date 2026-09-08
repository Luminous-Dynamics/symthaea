import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hak_evidence_lint as evidence

spec = importlib.util.spec_from_file_location(
    "hak_interpretation_lint", SCRIPTS / "hak_interpretation_lint.py"
)
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

PLAN_PATH = ROOT / "docs/architecture/hak/plans/hak007-evidence-bookkeeping-e5-v1.plan.json"
PLAN_REPO_PATH = "docs/architecture/hak/plans/hak007-evidence-bookkeeping-e5-v1.plan.json"
SUBJECT = "2" * 40
PLAN_COMMIT = "1" * 40
PLAN_REF = f"git:Luminous-Dynamics/symthaea@{PLAN_COMMIT}:{PLAN_REPO_PATH}"


def load_plan():
    import json
    return json.loads(PLAN_PATH.read_text())


def receipt(plan=None, conclusion="success"):
    plan = plan or load_plan()
    doc = {
        "schema_version": "hak.qualification-receipt.v1",
        "receipt_id": "receipt-1",
        "subject": {
            "repository": "Luminous-Dynamics/symthaea",
            "commit_sha": SUBJECT,
            "base_commit_sha": "0" * 40,
            "branch": "synthetic",
        },
        "qualification_plan": {
            "plan_kind": "SelfDeclaredPlan",
            "plan_ref": PLAN_REF,
            "plan_digest": hak.compute_plan_digest(plan),
            "precommit_status": "KnownPrecommitted",
        },
        "execution": {
            "provider": "github-actions",
            "run_id": 1001,
            "run_attempt": 1,
            "workflow_id": 1002,
            "workflow_name": "HAK Evidence",
            "workflow_path": ".github/workflows/hak-evidence.yml",
            "event": "pull_request",
            "provider_record_ref": "github-actions:Luminous-Dynamics/symthaea:run/1001:attempt/1",
        },
        "terminal": {
            "status": "completed",
            "conclusion": conclusion,
            "provider_started_at": "2026-09-08T10:00:00Z",
            "provider_completed_at": "2026-09-08T10:01:00Z",
        },
        "job_receipts": [{
            "job_id": 1003,
            "name": "HAK Evidence Linter",
            "status": "completed",
            "conclusion": conclusion,
        }],
    }
    doc["receipt_digest"] = evidence.compute_receipt_digest(doc)
    return doc


def conformance(plan=None, rec=None, status="Satisfied"):
    plan = plan or load_plan()
    rec = rec or receipt(plan)
    doc = {
        "schema_version": "hak.plan-conformance.v1",
        "conformance_id": "conformance-1",
        "subject": deepcopy(rec["subject"]),
        "plan": {
            "plan_ref": rec["qualification_plan"]["plan_ref"],
            "plan_digest": hak.compute_plan_digest(plan),
        },
        "receipt": {
            "receipt_id": rec["receipt_id"],
            "receipt_digest": rec["receipt_digest"],
        },
        "evaluator": {
            "kind": "DeterministicPolicy",
            "identity": "test:deterministic-conformance",
            "policy_ref": "hak:test-policy:v1",
            "model_assisted": False,
            "model_ref": None,
        },
        "status": status,
        "checks": [
            {"check_id": item["check_id"], "status": "Passed", "evidence_refs": ["synthetic:test"]}
            for item in plan["required_checks"]
        ],
        "negative_cases": [
            {"case": item, "status": "Passed", "evidence_refs": ["synthetic:test"]}
            for item in plan["required_negative_cases"]
        ],
        "limitations": [],
    }
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    return doc


def interpretation(plan=None, rec=None, conf=None, status="Qualified", tier="E5"):
    plan = plan or load_plan()
    rec = rec or receipt(plan)
    conf = conf or conformance(plan, rec)
    claims = []
    for item in plan["claims"]:
        claim = {
            "claim_id": item["claim_id"],
            "status": status,
            "supported_tier": tier if status == "Qualified" else None,
            "supporting_conformance_ids": [conf["conformance_id"]],
            "supporting_receipt_ids": [rec["receipt_id"]],
            "limitations": [],
            "blockers": [],
        }
        if status == "InsufficientEvidence":
            claim["limitations"] = ["synthetic missing evidence"]
        if status == "BlockedBy":
            claim["blockers"] = ["synthetic blocker"]
        if status == "Revoked":
            claim["revocation_ref"] = "synthetic:revocation"
        claims.append(claim)
    doc = {
        "schema_version": "hak.evidence-interpretation.v1",
        "record_id": "interpretation-1",
        "plan_ref": rec["qualification_plan"]["plan_ref"],
        "plan_digest": hak.compute_plan_digest(plan),
        "subject": {"repository": "Luminous-Dynamics/symthaea", "commit_sha": SUBJECT},
        "receipt_bindings": [{
            "receipt_id": rec["receipt_id"],
            "receipt_digest": rec["receipt_digest"],
        }],
        "conformance_bindings": [{
            "conformance_id": conf["conformance_id"],
            "conformance_digest": conf["conformance_digest"],
        }],
        "interpreter": {
            "kind": "DeterministicPolicy",
            "identity": "test:interpretation",
            "policy_ref": "hak:test-interpretation:v1",
            "model_assisted": False,
            "model_ref": None,
        },
        "claims": claims,
    }
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    return doc


def validate_interp(plan, rec, conf, doc):
    hak.validate_interpretation(
        plan,
        {rec["receipt_id"]: rec},
        {conf["conformance_id"]: conf},
        doc,
        plan_repo_path=PLAN_REPO_PATH,
    )


def test_satisfied_conformance_is_valid():
    plan = load_plan()
    rec = receipt(plan)
    hak.validate_plan_conformance(plan, rec, conformance(plan, rec), plan_repo_path=PLAN_REPO_PATH)


def test_satisfied_conformance_rejects_missing_required_check():
    plan = load_plan()
    rec = receipt(plan)
    doc = conformance(plan, rec)
    doc["checks"].pop()
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_satisfied_conformance_requires_success_receipt():
    plan = load_plan()
    rec = receipt(plan, "failure")
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, conformance(plan, rec), plan_repo_path=PLAN_REPO_PATH)


def test_failed_receipt_is_valid_negative_evidence():
    plan = load_plan()
    rec = receipt(plan, "failure")
    doc = conformance(plan, rec, "NotSatisfied")
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_wrong_plan_digest_rejected_before_conformance():
    plan = load_plan()
    rec = receipt(plan)
    rec["qualification_plan"]["plan_digest"] = "sha256:" + "0" * 64
    rec["receipt_digest"] = evidence.compute_receipt_digest(rec)
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, conformance(plan, rec), plan_repo_path=PLAN_REPO_PATH)


def test_forged_satisfied_conformance_is_revalidated():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    conf["checks"].pop()
    conf["conformance_digest"] = hak.compute_conformance_digest(conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, interpretation(plan, rec, conf))


def test_qualified_interpretation_is_valid_at_plan_ceiling():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    validate_interp(plan, rec, conf, interpretation(plan, rec, conf))


def test_qualified_interpretation_cannot_exceed_claim_ceiling():
    plan = load_plan()
    plan["claims"][0]["target_tier"] = "E4"
    rec = receipt(plan)
    conf = conformance(plan, rec)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, interpretation(plan, rec, conf, tier="E5"))


def test_interpretation_subject_must_match_receipt_subject():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["subject"]["commit_sha"] = "3" * 40
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_interpretation_plan_ref_must_match_receipt():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["plan_ref"] = f"git:Luminous-Dynamics/symthaea@{'4' * 40}:{PLAN_REPO_PATH}"
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_interpretation_plan_digest_must_match_loaded_plan():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["plan_digest"] = "sha256:" + "0" * 64
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_receipt_digest_binding_cannot_be_repointed():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["receipt_bindings"][0]["receipt_digest"] = "sha256:" + "0" * 64
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_conformance_digest_binding_cannot_be_repointed():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["conformance_bindings"][0]["conformance_digest"] = "sha256:" + "0" * 64
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_nonqualified_claim_cannot_claim_tier():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, "InsufficientEvidence", None)
    doc["claims"][0]["supported_tier"] = "E5"
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_insufficient_evidence_requires_limitation():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, "InsufficientEvidence", None)
    for claim in doc["claims"]:
        claim["limitations"] = []
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_interpretation_must_cover_exact_plan_claims():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["claims"].pop()
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_interpretation_digest_detects_tampering():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["interpreter"]["identity"] = "changed-after-materialization"
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_model_assistance_requires_model_ref_and_cannot_raise_ceiling():
    plan = load_plan()
    plan["claims"][0]["target_tier"] = "E4"
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, tier="E5")
    doc["interpreter"]["kind"] = "HumanReviewer"
    doc["interpreter"]["model_assisted"] = True
    doc["interpreter"]["model_ref"] = "model:test-assistant"
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)


def test_model_assistance_without_model_ref_rejected():
    plan = load_plan()
    rec = receipt(plan)
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["interpreter"]["kind"] = "HumanReviewer"
    doc["interpreter"]["model_assisted"] = True
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, rec, conf, doc)
