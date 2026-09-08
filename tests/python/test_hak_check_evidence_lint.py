import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "hak_check_evidence_lint", SCRIPTS / "hak_check_evidence_lint.py"
)
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

fixture_spec = importlib.util.spec_from_file_location(
    "hak_interpretation_fixtures", ROOT / "tests/python/test_hak_interpretation_lint.py"
)
assert fixture_spec and fixture_spec.loader
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)


def record(plan, receipt, *, kind="RequiredCheck", obligation_id=None, step_conclusion=None, suffix="1"):
    if obligation_id is None:
        obligation_id = plan["required_checks"][0]["check_id"]
    source_job = receipt["job_receipts"][0]
    if step_conclusion is None:
        step_conclusion = source_job["conclusion"]
    doc = {
        "schema_version": "hak.provider-bound-check-evidence.v1",
        "evidence_id": f"check-evidence-{suffix}",
        "assurance_class": "ProviderBound",
        "subject": {
            "repository": receipt["subject"]["repository"],
            "commit_sha": receipt["subject"]["commit_sha"],
        },
        "plan": {
            "plan_ref": receipt["qualification_plan"]["plan_ref"],
            "plan_digest": receipt["qualification_plan"]["plan_digest"],
        },
        "qualification_receipt": {
            "receipt_id": receipt["receipt_id"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "execution": {
            "provider": receipt["execution"]["provider"],
            "run_id": receipt["execution"]["run_id"],
            "run_attempt": receipt["execution"]["run_attempt"],
            "workflow_id": receipt["execution"]["workflow_id"],
            "workflow_path": receipt["execution"]["workflow_path"],
        },
        "obligation": {"kind": kind, "id": obligation_id},
        "provider_binding": {
            "job_id": source_job["job_id"],
            "job_name": source_job["name"],
            "job_status": source_job["status"],
            "job_conclusion": source_job["conclusion"],
            "step_number": int(suffix),
            "step_name": "synthetic obligation step",
            "step_status": "completed",
            "step_conclusion": step_conclusion,
            "provider_job_ref": (
                f"github-actions:{receipt['subject']['repository']}:job/{source_job['job_id']}"
            ),
        },
        "collected_by": {
            "identity": "test:provider-collector",
            "method": "synthetic-regression-fixture",
        },
        "observed_at": "2026-09-08T12:00:00Z",
    }
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    return doc


def validate(plan, receipt, doc):
    hak.validate_provider_bound_check_evidence(
        plan, receipt, doc, plan_repo_path=fixtures.PLAN_REPO_PATH
    )


def test_provider_bound_record_is_valid():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    validate(plan, receipt, record(plan, receipt))


def test_subject_mismatch_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["subject"]["commit_sha"] = "9" * 40
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_run_attempt_mismatch_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["execution"]["run_attempt"] = 2
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_unknown_obligation_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt, obligation_id="not-in-plan")
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_plan_digest_mismatch_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["plan"]["plan_digest"] = "sha256:" + "0" * 64
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_receipt_digest_mismatch_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["qualification_receipt"]["receipt_digest"] = "sha256:" + "0" * 64
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_provider_job_ref_mismatch_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["provider_binding"]["provider_job_ref"] = "github-actions:Luminous-Dynamics/symthaea:job/9999"
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_provider_job_must_exist_in_terminal_receipt():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["provider_binding"]["job_id"] = 9999
    doc["provider_binding"]["provider_job_ref"] = (
        "github-actions:Luminous-Dynamics/symthaea:job/9999"
    )
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_provider_job_metadata_must_match_terminal_receipt():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["provider_binding"]["job_name"] = "different job"
    doc["evidence_digest"] = hak.compute_check_evidence_digest(doc)
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_digest_tampering_rejected():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    doc = record(plan, receipt)
    doc["provider_binding"]["step_name"] = "tampered-after-materialization"
    with pytest.raises(hak.CheckEvidenceLintError):
        validate(plan, receipt, doc)


def test_failed_provider_step_is_still_valid_negative_evidence_record():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan, conclusion="failure")
    validate(plan, receipt, record(plan, receipt, step_conclusion="failure"))


def evidence_for_all_passed_obligations(plan, receipt, conformance):
    records = []
    counter = 1
    by_check = {item["check_id"]: item for item in conformance["checks"]}
    for item in plan["required_checks"]:
        ident = item["check_id"]
        rec = record(plan, receipt, kind="RequiredCheck", obligation_id=ident, suffix=str(counter))
        counter += 1
        by_check[ident]["evidence_refs"] = [hak.evidence_ref(rec)]
        records.append(rec)

    by_negative = {item["case"]: item for item in conformance["negative_cases"]}
    for ident in plan["required_negative_cases"]:
        rec = record(plan, receipt, kind="NegativeCase", obligation_id=ident, suffix=str(counter))
        counter += 1
        by_negative[ident]["evidence_refs"] = [hak.evidence_ref(rec)]
        records.append(rec)

    import hak_interpretation_lint as interpretation
    conformance["conformance_digest"] = interpretation.compute_conformance_digest(conformance)
    return records


def test_strict_conformance_requires_provider_evidence_for_passed_obligations():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    conformance = fixtures.conformance(plan, receipt)
    with pytest.raises(hak.CheckEvidenceLintError):
        hak.validate_conformance_with_provider_evidence(
            plan, receipt, conformance, [], plan_repo_path=fixtures.PLAN_REPO_PATH
        )


def test_strict_conformance_accepts_exact_provider_evidence_for_all_passed_obligations():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    conformance = fixtures.conformance(plan, receipt)
    records = evidence_for_all_passed_obligations(plan, receipt, conformance)
    hak.validate_conformance_with_provider_evidence(
        plan, receipt, conformance, records, plan_repo_path=fixtures.PLAN_REPO_PATH
    )


def test_strict_conformance_rejects_detached_evidence_ref():
    plan = fixtures.load_plan()
    receipt = fixtures.receipt(plan)
    conformance = fixtures.conformance(plan, receipt)
    records = evidence_for_all_passed_obligations(plan, receipt, conformance)
    conformance["checks"][0]["evidence_refs"] = ["hak-check-evidence:wrong:sha256:" + "0" * 64]
    import hak_interpretation_lint as interpretation
    conformance["conformance_digest"] = interpretation.compute_conformance_digest(conformance)
    with pytest.raises(hak.CheckEvidenceLintError):
        hak.validate_conformance_with_provider_evidence(
            plan, receipt, conformance, records, plan_repo_path=fixtures.PLAN_REPO_PATH
        )
