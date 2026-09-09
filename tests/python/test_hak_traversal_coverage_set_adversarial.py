import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/python"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SCRIPTS))

import hak_normalization_execute as hak14
import hak_selector_coverage as hak16
import hak_selector_coverage_profile_binding as hak16_profile
import hak_traversal_coverage_set as hak18b
import test_hak_traversal_coverage_set as fixtures


def test_cross_page_policy_artifact_ref_mismatch_is_rejected():
    declaration, traversal, witnesses = fixtures.build_world()
    altered_receipt = copy.deepcopy(witnesses[1]["receipt"])
    altered_receipt["policy"]["artifact_ref"] = (
        "git:Luminous-Dynamics/symthaea@2222222222222222222222222222222222222222:"
        "docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json"
    )
    altered_receipt["receipt_digest"] = hak14.compute_receipt_digest(altered_receipt)
    hak14.validate_receipt(altered_receipt)
    altered_coverage = hak16.derive_coverage_record(altered_receipt)
    hak16.validate_coverage_record(altered_coverage, altered_receipt)
    altered_binding = hak16_profile.derive_profile_binding(altered_coverage)
    witnesses[1] = {
        "page_id": "page-2",
        "disposition": "Assessed",
        "receipt": altered_receipt,
        "coverage": altered_coverage,
        "profile_binding": altered_binding,
    }
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=fixtures.SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_policy_content_verification_cannot_be_promoted_after_redigest():
    declaration, traversal, witnesses, record = fixtures.derive()
    record["normalization_policy_content_verification"] = "Verified"
    record["coverage_set_digest"] = hak18b.compute_coverage_set_digest(record)
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.validate_coverage_set(
            record,
            selector_path=fixtures.SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )
