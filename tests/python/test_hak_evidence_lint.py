import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "hak_evidence_lint.py"
OBSERVATION = (
    ROOT
    / "docs"
    / "release"
    / "evidence"
    / "hak"
    / "executions"
    / "hak006-github-34212685766-attempt1.observation.json"
)

spec = importlib.util.spec_from_file_location("hak_evidence_lint", SCRIPT)
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

SUBJECT = "98883dfb03594f777390abe557e307af99ff4b5d"


def load_observation():
    return json.loads(OBSERVATION.read_text())


def terminal_receipt():
    observation = load_observation()
    return {
        "schema_version": "hak.qualification-receipt.v1",
        "receipt_id": "synthetic-receipt-for-regression-only",
        "subject": deepcopy(observation["subject"]),
        "qualification_plan": deepcopy(observation["qualification_plan"]),
        "execution": deepcopy(observation["execution"]),
        "terminal": {
            "status": "completed",
            "conclusion": "success",
            "provider_started_at": "2026-09-08T09:55:38Z",
            "provider_completed_at": "2026-09-08T10:00:00Z",
        },
        "job_receipts": [
            {
                "job_id": 102017126847,
                "name": "HAK Conformance Linter",
                "status": "completed",
                "conclusion": "success",
            }
        ],
        "receipt_digest": None,
    }


def test_real_queued_observation_is_valid():
    hak.validate_observation(load_observation(), expected_subject=SUBJECT)


def test_observation_cannot_have_terminal_conclusion():
    doc = load_observation()
    doc["observation"]["conclusion"] = "success"
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(doc)


def test_observation_cannot_use_completed_status():
    doc = load_observation()
    doc["observation"]["status"] = "completed"
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(doc)


def test_observation_cannot_claim_e5():
    doc = load_observation()
    doc["claims"]["e5_qualified"] = True
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(doc)


def test_observation_cannot_claim_terminal_receipt():
    doc = load_observation()
    doc["claims"]["terminal_receipt_exists"] = True
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(doc)


def test_expected_subject_mismatch_rejected():
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(load_observation(), expected_subject="0" * 40)


def test_synthetic_terminal_receipt_is_valid_bookkeeping():
    hak.validate_receipt(terminal_receipt(), expected_subject=SUBJECT)


def test_receipt_requires_completed_status():
    doc = terminal_receipt()
    doc["terminal"]["status"] = "in_progress"
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_receipt(doc)


def test_receipt_requires_terminal_conclusion():
    doc = terminal_receipt()
    doc["terminal"]["conclusion"] = None
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_receipt(doc)


def test_receipt_cannot_smuggle_evidence_tier():
    doc = terminal_receipt()
    doc["evidence_tier"] = "E5"
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_receipt(doc)


def test_receipt_cannot_smuggle_e5_interpretation():
    doc = terminal_receipt()
    doc["e5_qualified"] = True
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_receipt(doc)


def test_receipt_job_must_be_terminal():
    doc = terminal_receipt()
    doc["job_receipts"][0]["status"] = "queued"
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_receipt(doc)


def test_run_attempt_must_be_positive():
    doc = load_observation()
    doc["execution"]["run_attempt"] = 0
    with pytest.raises(hak.EvidenceLintError):
        hak.validate_observation(doc)


def test_cli_json_output(tmp_path, capsys):
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(load_observation()))
    rc = hak.main([str(path), "--expected-subject", SUBJECT, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["results"][0]["kind"] == "observation"
