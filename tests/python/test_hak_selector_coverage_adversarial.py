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
SOURCE_REF = "github-actions:test-coverage-adversarial"
KIND = "WorkflowJobStepsObservation"


def _step(number: int, *, conclusion="success"):
    return {
        "name": f"step-{number}",
        "status": "completed",
        "conclusion": conclusion,
        "number": number,
        "started_at": "s",
        "completed_at": "e",
    }


def _record_for(steps):
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
    assert code == 0
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
    hak16.validate_coverage_record(record, receipt)
    return receipt, record


def _by_path(record, path):
    return next(item for item in record["optional_results"] if item["path"] == path)


def test_historical_source_status_rewrite_rejected_after_redigest():
    second = _step(2)
    del second["conclusion"]
    receipt, record = _record_for([_step(1), second])
    target = _by_path(record, "steps[*].conclusion")
    assert target["source_status"] == "Present"
    target["source_status"] = "Absent"
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("canonicalization_profile", "hak.canonical-json.v999"),
        ("digest_domain", "hak.selector-coverage.other"),
    ],
)
def test_canonical_profile_or_digest_domain_substitution_rejected_after_redigest(field, replacement):
    receipt, record = _record_for([])
    record[field] = replacement
    # Recompute using the implementation's real digest function to model a
    # coherent local redigest attack, not a stale-digest failure.
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    with pytest.raises(hak16.SelectorCoverageError):
        hak16.validate_coverage_record(record, receipt)
