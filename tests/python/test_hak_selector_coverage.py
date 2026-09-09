import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec14 = importlib.util.spec_from_file_location("hak_normalization_execute", SCRIPTS / "hak_normalization_execute.py")
assert spec14 and spec14.loader
hak14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(hak14)

spec16 = importlib.util.spec_from_file_location("hak_selector_coverage", SCRIPTS / "hak_selector_coverage.py")
assert spec16 and spec16.loader
hak16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(hak16)

POLICY = json.loads((ROOT / "docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json").read_text())
POLICY_REF = "git:Luminous-Dynamics/symthaea@877ae00faed562ebdc4a18834cd6c30c9649fa96:docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json"
INTERPRETER_REF = "git:Luminous-Dynamics/symthaea@1111111111111111111111111111111111111111:scripts/hak_normalization_execute.py"
INTERPRETER_BYTES = hak14.current_interpreter_bytes()
SOURCE_REF = "github-actions:test-coverage-resource"
KIND = "WorkflowJobStepsObservation"


def run_steps(steps):
    raw = json.dumps({"steps": steps}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    receipt, code = hak14.execute_normalization(
        raw,
        deepcopy(POLICY),
        resource_kind=KIND,
        raw_source_ref=SOURCE_REF,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        interpreter_bytes=INTERPRETER_BYTES,
    )
    hak14.validate_receipt_against_inputs(
        receipt,
        raw,
        POLICY,
        INTERPRETER_BYTES,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        raw_source_ref=SOURCE_REF,
        resource_kind=KIND,
    )
    record = hak16.derive_coverage_record(receipt)
    hak16.validate_coverage_record_against_inputs(
        record,
        receipt,
        raw,
        POLICY,
        INTERPRETER_BYTES,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        raw_source_ref=SOURCE_REF,
        resource_kind=KIND,
    )
    return raw, receipt, record, code


def by_path(record, path):
    return next(item for item in record["optional_results"] if item["path"] == path)


def full_step(number=1, conclusion="success"):
    return {
        "name": f"step-{number}",
        "status": "completed",
        "conclusion": conclusion,
        "number": number,
        "started_at": "s",
        "completed_at": "e",
    }


def test_present_means_every_applicable_context_present():
    _, receipt, record, code = run_steps([full_step(1), full_step(2)])
    assert code == 0 and receipt["execution_status"] == "Succeeded"
    result = by_path(record, "steps[*].conclusion")
    assert result == {
        "path": "steps[*].conclusion",
        "source_status": "Present",
        "coverage_state": "Present",
        "applicable": 2,
        "matches": 2,
        "missing": 0,
    }


def test_partial_presence_is_not_summarized_as_full_presence():
    second = full_step(2)
    del second["conclusion"]
    _, receipt, record, code = run_steps([full_step(1), second])
    assert code == 0 and receipt["execution_status"] == "Succeeded"
    source = next(item for item in receipt["optional_selector_results"] if item["path"] == "steps[*].conclusion")
    assert source["status"] == "Present" and source["matches"] == 1 and source["missing"] == 1
    result = by_path(record, "steps[*].conclusion")
    assert result["source_status"] == "Present"
    assert result["coverage_state"] == "PartiallyPresent"
    assert result["applicable"] == 2 and result["matches"] == 1 and result["missing"] == 1


def test_absent_means_applicable_contexts_exist_but_none_match():
    a, b = full_step(1), full_step(2)
    del a["conclusion"]
    del b["conclusion"]
    _, _, record, _ = run_steps([a, b])
    result = by_path(record, "steps[*].conclusion")
    assert result["coverage_state"] == "Absent"
    assert result["applicable"] == 2 and result["matches"] == 0 and result["missing"] == 2


def test_empty_wildcard_is_not_applicable_not_absent():
    _, receipt, record, code = run_steps([])
    assert code == 0 and receipt["execution_status"] == "Succeeded"
    source = by_path({"optional_results": receipt["optional_selector_results"]}, "steps[*].name")
    assert source["status"] == "Absent" and source["matches"] == 0 and source["missing"] == 0
    result = by_path(record, "steps[*].name")
    assert result["source_status"] == "Absent"
    assert result["coverage_state"] == "NotApplicable"
    assert result["applicable"] == 0 and result["matches"] == 0 and result["missing"] == 0


def test_wrong_wildcard_container_type_is_failed_with_unknown_applicability():
    raw = b'{"steps":{}}'
    receipt, _ = hak14.execute_normalization(
        raw,
        deepcopy(POLICY),
        resource_kind=KIND,
        raw_source_ref=SOURCE_REF,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        interpreter_bytes=INTERPRETER_BYTES,
    )
    hak14.validate_receipt_against_inputs(
        receipt, raw, POLICY, INTERPRETER_BYTES,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        raw_source_ref=SOURCE_REF,
        resource_kind=KIND,
    )
    record = hak16.derive_coverage_record(receipt)
    result = by_path(record, "steps[*].name")
    assert result["source_status"] == "Failed"
    assert result["coverage_state"] == "Failed"
    assert result["applicable"] is None


def test_partial_record_cannot_be_redigested_as_present():
    second = full_step(2)
    del second["conclusion"]
    _, receipt, record, _ = run_steps([full_step(1), second])
    target = by_path(record, "steps[*].conclusion")
    target["coverage_state"] = "Present"
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_not_applicable_cannot_be_redigested_as_absent():
    _, receipt, record, _ = run_steps([])
    target = by_path(record, "steps[*].name")
    target["coverage_state"] = "Absent"
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_absent_cannot_be_redigested_as_not_applicable():
    step = full_step(1)
    del step["conclusion"]
    _, receipt, record, _ = run_steps([step])
    target = by_path(record, "steps[*].conclusion")
    target["coverage_state"] = "NotApplicable"
    target["applicable"] = 0
    target["missing"] = 0
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_applicability_conservation_is_enforced():
    _, receipt, record, _ = run_steps([full_step(1), full_step(2)])
    target = by_path(record, "steps[*].name")
    target["applicable"] = 3
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_failed_result_cannot_claim_numeric_applicability():
    raw = b'{"steps":{}}'
    receipt, _ = hak14.execute_normalization(
        raw, deepcopy(POLICY), resource_kind=KIND, raw_source_ref=SOURCE_REF,
        policy_artifact_ref=POLICY_REF, interpreter_ref=INTERPRETER_REF,
        interpreter_bytes=INTERPRETER_BYTES,
    )
    record = hak16.derive_coverage_record(receipt)
    target = by_path(record, "steps[*].name")
    target["applicable"] = 0
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_source_status_is_preserved_not_rewritten():
    second = full_step(2)
    del second["conclusion"]
    _, receipt, record, _ = run_steps([full_step(1), second])
    source = next(item for item in receipt["optional_selector_results"] if item["path"] == "steps[*].conclusion")
    derived = by_path(record, "steps[*].conclusion")
    assert source["status"] == derived["source_status"] == "Present"
    assert derived["coverage_state"] == "PartiallyPresent"


def test_source_receipt_digest_mismatch_rejected_even_if_coverage_redigested():
    _, receipt, record, _ = run_steps([])
    record["source_receipt"]["receipt_digest"] = "sha256:" + "0" * 64
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


def test_raw_input_substitution_rejected_by_strong_join():
    raw, receipt, record, _ = run_steps([])
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record_against_inputs(
            record, receipt, raw + b" ", POLICY, INTERPRETER_BYTES,
            policy_artifact_ref=POLICY_REF,
            interpreter_ref=INTERPRETER_REF,
            raw_source_ref=SOURCE_REF,
            resource_kind=KIND,
        )


def test_coverage_digest_uses_hak015_profile_and_domain():
    _, _, record, _ = run_steps([])
    assert record["canonicalization_profile"] == "hak.canonical-json.v1"
    assert record["digest_domain"] == "hak.selector-coverage.v1"
    assert record["coverage_digest"] == hak16.compute_coverage_digest(record)
