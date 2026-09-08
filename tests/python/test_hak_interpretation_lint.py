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


def load_plan():
    import json
    return json.loads(PLAN_PATH.read_text())


def receipt(conclusion="success"):
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
            "plan_ref": f"git:Luminous-Dynamics/symthaea@{PLAN_COMMIT}:{PLAN_REPO_PATH}",
            "plan_digest": None,
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
    rec = rec or receipt()
    checks = [
        {"check_id": item["check_id"], "status": "Passed", "evidence_refs": ["synthetic:test"]}
        for item in plan["required_checks"]
    ]
    negatives = [
        {"case": item, "status": "Passed", "evidence_refs": ["synthetic:test"]}
        for item in plan["required_negative_cases"]
    ]
    doc = {
        "schema_version": "hak.plan-conformance.v1",
        "conformance_id": "conformance-1",
        "subject": deepcopy(rec["subject"]),
        "plan": {"plan_ref": rec["qualification_plan"]["plan_ref"]},
        "receipt": {
            "receipt_id": rec["receipt_id"],
            "receipt_digest": rec["receipt_digest"],
        },
        "evaluator": {
            "identity": "test:deterministic-conformance",
            "policy_ref": "hak:test-policy:v1",
            "model_assisted": False,
        },
        "status": status,
        "checks": checks,
        "negative_cases": negatives,
        "limitations": [],
    }
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    return doc


def interpretation(plan=None, rec=None, conf=None, status="Qualified", tier="E5"):
    plan = plan or load_plan()
    rec = rec or receipt()
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
        "subject": {"repository": "Luminous-Dynamics/symthaea", "commit_sha": SUBJECT},
        "interpreter": {
            "kind": "DeterministicPolicy",
            "identity": "test:interpretation",
            "policy_ref": "hak:test-interpretation:v1",
            "model_assisted": False,
        },
        "claims": claims,
    }
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    return doc


def maps(rec, conf):
    return {rec["receipt_id"]: rec}, {conf["conformance_id"]: conf}


def validate_interp(plan, receipts, conformances, doc):
    hak.validate_interpretation(
        plan, receipts, conformances, doc, plan_repo_path=PLAN_REPO_PATH
    )


def test_satisfied_conformance_is_valid():
    plan, rec = load_plan(), receipt()
    hak.validate_plan_conformance(plan, rec, conformance(plan, rec), plan_repo_path=PLAN_REPO_PATH)


def test_satisfied_conformance_rejects_missing_required_check():
    plan, rec = load_plan(), receipt()
    doc = conformance(plan, rec)
    doc["checks"].pop()
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_satisfied_conformance_requires_success_receipt():
    plan, rec = load_plan(), receipt("failure")
    doc = conformance(plan, rec)
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_failed_receipt_can_support_not_satisfied_conformance():
    plan, rec = load_plan(), receipt("failure")
    doc = conformance(plan, rec, status="NotSatisfied")
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_indeterminate_requires_limitation():
    plan, rec = load_plan(), receipt()
    doc = conformance(plan, rec, status="Indeterminate")
    doc["checks"].pop()
    doc["conformance_digest"] = hak.compute_conformance_digest(doc)
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_conformance_digest_detects_tampering():
    plan, rec = load_plan(), receipt()
    doc = conformance(plan, rec)
    doc["evaluator"]["identity"] = "changed-after-materialization"
    with pytest.raises(hak.InterpretationLintError):
        hak.validate_plan_conformance(plan, rec, doc, plan_repo_path=PLAN_REPO_PATH)


def test_qualified_interpretation_is_valid_at_plan_ceiling():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    receipts, conformances = maps(rec, conf)
    validate_interp(plan, receipts, conformances, doc)


def test_qualified_interpretation_cannot_exceed_claim_ceiling():
    plan, rec = load_plan(), receipt()
    plan["claims"][0]["target_tier"] = "E4"
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, tier="E5")
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_provider_success_without_satisfied_conformance_cannot_qualify():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec, status="Indeterminate")
    conf["checks"].pop()
    conf["limitations"] = ["missing required check"]
    conf["conformance_digest"] = hak.compute_conformance_digest(conf)
    doc = interpretation(plan, rec, conf)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_forged_satisfied_conformance_is_revalidated():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    conf["checks"].pop()
    conf["conformance_digest"] = hak.compute_conformance_digest(conf)
    doc = interpretation(plan, rec, conf)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_nonqualified_claim_cannot_claim_tier():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, status="InsufficientEvidence", tier=None)
    doc["claims"][0]["supported_tier"] = "E5"
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_insufficient_evidence_requires_limitation():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, status="InsufficientEvidence", tier=None)
    for claim in doc["claims"]:
        claim["limitations"] = []
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_interpretation_must_cover_exact_plan_claims():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["claims"].pop()
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_interpretation_digest_detects_tampering():
    plan, rec = load_plan(), receipt()
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf)
    doc["interpreter"]["identity"] = "changed-after-materialization"
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)


def test_model_assistance_does_not_change_evidence_ceiling():
    plan, rec = load_plan(), receipt()
    plan["claims"][0]["target_tier"] = "E4"
    conf = conformance(plan, rec)
    doc = interpretation(plan, rec, conf, tier="E5")
    doc["interpreter"]["kind"] = "ModelAssistedReview"
    doc["interpreter"]["model_assisted"] = True
    doc["interpretation_digest"] = hak.compute_interpretation_digest(doc)
    receipts, conformances = maps(rec, conf)
    with pytest.raises(hak.InterpretationLintError):
        validate_interp(plan, receipts, conformances, doc)
