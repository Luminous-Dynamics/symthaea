import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hak_evidence_lint as evidence
import hak_interpretation_lint as interpretation

spec = importlib.util.spec_from_file_location("hak_real_evidence_lint", SCRIPTS / "hak_real_evidence_lint.py")
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

PLAN_PATH = ROOT / "docs/architecture/hak/plans/hak007-evidence-bookkeeping-e5-v1.plan.json"
EVID_DIR = ROOT / "docs/architecture/hak/evidence/real"
RECEIPT_PATH = EVID_DIR / "hak007-run-34225891059.receipt.json"
CONFORMANCE_PATH = EVID_DIR / "hak007-run-34225891059.conformance.json"
INTERPRETATION_PATH = EVID_DIR / "hak007-run-34225891059.interpretation.json"
CAPSULE_PATH = EVID_DIR / "hak007-run-34225891059.capsule.json"


def load(path):
    return json.loads(path.read_text())


def documents():
    return tuple(load(path) for path in (PLAN_PATH, RECEIPT_PATH, CONFORMANCE_PATH, INTERPRETATION_PATH, CAPSULE_PATH))


def validate(plan, receipt, conformance, interp, capsule):
    hak.validate_real_provider_capsule(plan, receipt, conformance, interp, capsule, plan_repo_path="docs/architecture/hak/plans/hak007-evidence-bookkeeping-e5-v1.plan.json")


def redigest_receipt(doc):
    doc["receipt_digest"] = evidence.compute_receipt_digest(doc)


def redigest_conformance(doc):
    doc["conformance_digest"] = interpretation.compute_conformance_digest(doc)


def redigest_interpretation(doc):
    doc["interpretation_digest"] = interpretation.compute_interpretation_digest(doc)


def redigest_capsule(doc):
    doc["capsule_digest"] = hak.compute_capsule_digest(doc)


def test_real_cancelled_run_capsule_is_valid():
    validate(*documents())


def test_real_receipt_subject_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    rec["subject"]["commit_sha"] = "9" * 40
    redigest_receipt(rec)
    cap["bindings"]["receipt"]["receipt_digest"] = rec["receipt_digest"]
    redigest_capsule(cap)
    with pytest.raises(Exception):
        validate(plan, rec, conf, interp, cap)


def test_real_receipt_run_attempt_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    rec["execution"]["run_attempt"] = 2
    redigest_receipt(rec)
    cap["bindings"]["receipt"]["receipt_digest"] = rec["receipt_digest"]
    redigest_capsule(cap)
    with pytest.raises(Exception):
        validate(plan, rec, conf, interp, cap)


def test_real_terminal_job_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    cap["provider_snapshot"]["job"]["job_id"] = 999999
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_cancelled_no_step_run_cannot_be_satisfied():
    plan, rec, conf, interp, cap = documents()
    conf["status"] = "Satisfied"
    redigest_conformance(conf)
    cap["bindings"]["conformance"]["conformance_digest"] = conf["conformance_digest"]
    redigest_capsule(cap)
    with pytest.raises(Exception):
        validate(plan, rec, conf, interp, cap)


def test_cancelled_no_step_run_cannot_contain_passed_obligation():
    plan, rec, conf, interp, cap = documents()
    conf["checks"][0]["status"] = "Passed"
    conf["checks"][0]["evidence_refs"] = ["fabricated:evidence"]
    redigest_conformance(conf)
    cap["bindings"]["conformance"]["conformance_digest"] = conf["conformance_digest"]
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_cancelled_no_step_run_cannot_fabricate_provider_check_evidence():
    plan, rec, conf, interp, cap = documents()
    cap["provider_check_evidence"] = ["hak-check-evidence:fabricated:sha256:" + "0" * 64]
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_cancelled_no_step_claim_cannot_be_qualified():
    plan, rec, conf, interp, cap = documents()
    interp["claims"][0]["status"] = "Qualified"
    interp["claims"][0]["supported_tier"] = "E5"
    redigest_interpretation(interp)
    cap["bindings"]["interpretation"]["interpretation_digest"] = interp["interpretation_digest"]
    redigest_capsule(cap)
    with pytest.raises(Exception):
        validate(plan, rec, conf, interp, cap)


def test_cancelled_no_step_claim_is_not_semantically_disproven():
    plan, rec, conf, interp, cap = documents()
    for claim in interp["claims"]:
        claim["status"] = "NotSatisfied"
        claim["limitations"] = []
    redigest_interpretation(interp)
    cap["bindings"]["interpretation"]["interpretation_digest"] = interp["interpretation_digest"]
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_capsule_receipt_digest_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    cap["bindings"]["receipt"]["receipt_digest"] = "sha256:" + "0" * 64
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_capsule_conformance_digest_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    cap["bindings"]["conformance"]["conformance_digest"] = "sha256:" + "0" * 64
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_capsule_interpretation_digest_mismatch_rejected():
    plan, rec, conf, interp, cap = documents()
    cap["bindings"]["interpretation"]["interpretation_digest"] = "sha256:" + "0" * 64
    redigest_capsule(cap)
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)


def test_capsule_digest_tampering_rejected():
    plan, rec, conf, interp, cap = documents()
    cap["provider_snapshot"]["provider_updated_at"] = "2026-09-08T12:26:00Z"
    with pytest.raises(hak.RealEvidenceLintError):
        validate(plan, rec, conf, interp, cap)
