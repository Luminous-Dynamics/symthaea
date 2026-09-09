import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec16 = importlib.util.spec_from_file_location("hak_selector_coverage", SCRIPTS / "hak_selector_coverage.py")
assert spec16 and spec16.loader
hak16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(hak16)

spec_binding = importlib.util.spec_from_file_location(
    "hak_selector_coverage_profile_binding",
    SCRIPTS / "hak_selector_coverage_profile_binding.py",
)
assert spec_binding and spec_binding.loader
binding = importlib.util.module_from_spec(spec_binding)
spec_binding.loader.exec_module(binding)

SCHEMA = json.loads(
    (ROOT / "docs/architecture/hak/selector-coverage-profile-binding-v1.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)


def coverage_record():
    record = {
        "schema_version": "hak.selector-coverage.v1",
        "canonicalization_profile": "hak.canonical-json.v1",
        "digest_domain": "hak.selector-coverage.v1",
        "source_receipt": {
            "schema_version": "hak.normalization-execution-receipt.v1",
            "receipt_digest": "sha256:" + "1" * 64,
            "execution_status": "Succeeded",
        },
        "resource_kind": "WorkflowJobStepsObservation",
        "policy": {
            "policy_id": "github-actions-source-observation-normalization-v1",
            "policy_digest": "sha256:" + "2" * 64,
        },
        "coverage_semantics": {
            "model": "HAKSelectorPathTerminalContextV1",
            "nonfailed_conservation": "applicable == matches + missing",
            "empty_wildcard_semantics": "zero applicable contexts -> NotApplicable",
            "historical_receipt_status_preserved": True,
        },
        "optional_results": [],
    }
    record["coverage_digest"] = hak16.compute_coverage_digest(record)
    return record


def test_binding_identifies_exact_hak015_profile_artifact_and_raw_bytes():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    binding.validate_profile_binding(result, record)
    VALIDATOR.validate(result)
    assert result["canonicalization_profile"] == {
        "profile_id": "hak.canonical-json.v1",
        "artifact_ref": binding.PROFILE_ARTIFACT_REF,
        "raw_sha256": binding.EXPECTED_PROFILE_RAW_SHA256,
    }
    assert binding.sha256_bytes(binding.current_profile_bytes()) == binding.EXPECTED_PROFILE_RAW_SHA256


def test_profile_artifact_ref_substitution_rejected_after_redigest():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    result["canonicalization_profile"]["artifact_ref"] = (
        "git:Luminous-Dynamics/symthaea@" + "0" * 40 + ":docs/architecture/hak/canonical-json-v1.profile.json"
    )
    result["binding_digest"] = binding.compute_binding_digest(result)
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_binding(result, record)


def test_raw_profile_digest_substitution_rejected_after_redigest():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    result["canonicalization_profile"]["raw_sha256"] = "sha256:" + "0" * 64
    result["binding_digest"] = binding.compute_binding_digest(result)
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_binding(result, record)


def test_profile_id_substitution_rejected_after_redigest():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    result["canonicalization_profile"]["profile_id"] = "hak.canonical-json.v999"
    result["binding_digest"] = binding.compute_binding_digest(result)
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_binding(result, record)


def test_inherited_profile_byte_drift_rejected():
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_bytes(b"{}\n")


def test_coverage_digest_substitution_rejected_after_binding_redigest():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    result["coverage"]["coverage_digest"] = "sha256:" + "0" * 64
    result["binding_digest"] = binding.compute_binding_digest(result)
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_binding(result, record)


def test_binding_digest_tampering_rejected():
    record = coverage_record()
    result = binding.derive_profile_binding(record)
    result["binding_digest"] = "sha256:" + "0" * 64
    with pytest.raises(binding.SelectorCoverageProfileBindingError):
        binding.validate_profile_binding(result, record)


def test_schema_rejects_unknown_binding_field():
    result = binding.derive_profile_binding(coverage_record())
    result["unexpected"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(result)
