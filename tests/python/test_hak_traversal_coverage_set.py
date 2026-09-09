import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hak_collection_completeness as hak17
import hak_normalization_execute as hak14
import hak_selector_coverage as hak16
import hak_selector_coverage_profile_binding as hak16_profile
import hak_traversal_coverage_set as hak18b
import hak_verified_traversal as hak18a

POLICY = json.loads(
    (ROOT / "docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json").read_text()
)
POLICY_REF = (
    "git:Luminous-Dynamics/symthaea@877ae00faed562ebdc4a18834cd6c30c9649fa96:"
    "docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json"
)
INTERPRETER_REF = "git:Luminous-Dynamics/symthaea@1111111111111111111111111111111111111111:scripts/hak_normalization_execute.py"
INTERPRETER_BYTES = hak14.current_interpreter_bytes()
KIND = "WorkflowJobStepsObservation"
SELECTOR = "steps[*].conclusion"


def full_step(number: int, *, conclusion: str | None = "success"):
    result = {
        "name": f"step-{number}",
        "status": "completed",
        "number": number,
        "started_at": "s",
        "completed_at": "e",
    }
    if conclusion is not None:
        result["conclusion"] = conclusion
    return result


def page_bundle(source_ref: str, steps, *, kind: str = KIND):
    raw = json.dumps({"steps": steps}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    receipt, code = hak14.execute_normalization(
        raw,
        copy.deepcopy(POLICY),
        resource_kind=kind,
        raw_source_ref=source_ref,
        policy_artifact_ref=POLICY_REF,
        interpreter_ref=INTERPRETER_REF,
        interpreter_bytes=INTERPRETER_BYTES,
    )
    assert code == 0
    hak14.validate_receipt(receipt)
    coverage = hak16.derive_coverage_record(receipt)
    hak16.validate_coverage_record(coverage, receipt)
    binding = hak16_profile.derive_profile_binding(coverage)
    hak16_profile.validate_profile_binding(binding, coverage)
    return {"raw": raw, "receipt": receipt, "coverage": coverage, "profile_binding": binding}


def build_world(
    page1_steps=None,
    page2_steps=None,
    *,
    traversal_resource_kind: str = KIND,
):
    page1_steps = [full_step(1), full_step(2)] if page1_steps is None else page1_steps
    page2_steps = [full_step(3), full_step(4)] if page2_steps is None else page2_steps
    p1 = page_bundle("response:page-1", page1_steps)
    p2 = page_bundle("response:page-2", page2_steps)

    ids1 = [f"entity:p1:{i}" for i in range(len(page1_steps))]
    ids2 = [f"entity:p2:{i}" for i in range(len(page2_steps))]
    total = len(ids1) + len(ids2)
    declaration = hak17.make_record(
        provider="github-actions",
        resource_kind=traversal_resource_kind,
        collection_ref="github-actions:test:steps:all-pages",
        query_scope_ref="query:steps:all-pages",
        population_extent="FullProviderPopulation",
        scope_ref="scope:steps:all-pages",
        filtered=False,
        sampled=False,
        time_windowed=False,
        permission_limited=False,
        pagination_model="LinkHeader",
        pagination_contract_ref="provider-contract:github-rest-pagination:v1",
        pages_observed=2,
        page_size=100,
        exhaustion_kind="NoContinuationByProviderContract",
        exhaustion_evidence_refs=["provider-header:link:no-next"],
        retained_raw_count=total,
        retained_unique_count=total,
        duplicate_count=0,
        dedup_identity_ref=hak18a.DEDUP_IDENTITY_REF,
        dedup_rule_ref=hak18a.DEDUP_RULE_REF,
        provider_reported_total_count=total,
        provider_total_scope_ref="scope:steps:all-pages",
        collection_status="DeclaredComplete",
        reasons=[],
    )
    transcript = {
        "provider": "github-actions",
        "resource_kind": traversal_resource_kind,
        "collection_ref": "github-actions:test:steps:all-pages",
        "query_scope_ref": "query:steps:all-pages",
        "scope_ref": "scope:steps:all-pages",
        "pagination_model": "LinkHeader",
        "pagination_contract_ref": "provider-contract:github-rest-pagination:v1",
        "temporal": {"declared_semantics": "BestEffortLiveTraversal", "evidence_refs": []},
        "pages": [
            {
                "ordinal": 1,
                "page_id": "page-1",
                "request_ref": "request:page-1",
                "continuation_in": None,
                "response_ref": "response:page-1",
                "raw_response_digest": p1["receipt"]["raw_source"]["raw_response_digest"],
                "continuation_out": "next:page-2",
                "observed_after_or_at": "2026-09-09T18:00:00Z",
                "observed_before_or_at": "2026-09-09T18:00:01Z",
                "entity_ids": ids1,
            },
            {
                "ordinal": 2,
                "page_id": "page-2",
                "request_ref": "request:page-2",
                "continuation_in": "next:page-2",
                "response_ref": "response:page-2",
                "raw_response_digest": p2["receipt"]["raw_source"]["raw_response_digest"],
                "continuation_out": None,
                "observed_after_or_at": "2026-09-09T18:00:01Z",
                "observed_before_or_at": "2026-09-09T18:00:02Z",
                "entity_ids": ids2,
            },
        ],
    }
    traversal = hak18a.make_verification_record(transcript, declaration)
    witnesses = [
        {
            "page_id": "page-1",
            "disposition": "Assessed",
            "receipt": p1["receipt"],
            "coverage": p1["coverage"],
            "profile_binding": p1["profile_binding"],
        },
        {
            "page_id": "page-2",
            "disposition": "Assessed",
            "receipt": p2["receipt"],
            "coverage": p2["coverage"],
            "profile_binding": p2["profile_binding"],
        },
    ]
    return declaration, traversal, witnesses


def derive(world=None, *, selector=SELECTOR):
    declaration, traversal, witnesses = world or build_world()
    record = hak18b.derive_coverage_set(
        selector_path=selector,
        traversal_record=traversal,
        hak017_record=declaration,
        page_witnesses=witnesses,
    )
    hak18b.validate_coverage_set(
        record,
        selector_path=selector,
        traversal_record=traversal,
        hak017_record=declaration,
        page_witnesses=witnesses,
    )
    return declaration, traversal, witnesses, record


def redigest(record):
    record["coverage_set_digest"] = hak18b.compute_coverage_set_digest(record)
    return record


def test_all_assessed_present_aggregates_present():
    _, _, _, record = derive()
    assert record["aggregate"] == {
        "coverage_state": "Present",
        "assessed_pages": 2,
        "unassessed_pages": 0,
        "failed_pages": 0,
        "observed_counts": {"applicable": 4, "matches": 4, "missing": 0},
    }
    assert record["population_qualification"] == "NotEstablished"
    assert record["normalization_input_replay"] == "NotEstablished"


def test_partial_presence_aggregates_counts_across_pages():
    page2 = [full_step(3), full_step(4, conclusion=None)]
    _, _, _, record = derive(build_world(page2_steps=page2))
    assert record["aggregate"]["coverage_state"] == "PartiallyPresent"
    assert record["aggregate"]["observed_counts"] == {"applicable": 4, "matches": 3, "missing": 1}


def test_absence_across_all_assessed_pages_is_absent():
    p1 = [full_step(1, conclusion=None), full_step(2, conclusion=None)]
    p2 = [full_step(3, conclusion=None), full_step(4, conclusion=None)]
    _, _, _, record = derive(build_world(p1, p2))
    assert record["aggregate"]["coverage_state"] == "Absent"
    assert record["aggregate"]["observed_counts"] == {"applicable": 4, "matches": 0, "missing": 4}


def test_zero_applicable_contexts_are_not_applicable():
    _, _, _, record = derive(build_world([], []))
    assert record["aggregate"]["coverage_state"] == "NotApplicable"
    assert record["aggregate"]["observed_counts"] == {"applicable": 0, "matches": 0, "missing": 0}


def test_explicit_unassessed_page_forces_indeterminate_without_disappearing():
    declaration, traversal, witnesses = build_world()
    witnesses[1] = {"page_id": "page-2", "disposition": "Unassessed", "reason": "coverage not executed"}
    _, _, _, record = derive((declaration, traversal, witnesses))
    assert record["aggregate"]["coverage_state"] == "Indeterminate"
    assert record["aggregate"]["assessed_pages"] == 1
    assert record["aggregate"]["unassessed_pages"] == 1
    assert record["pages"][1]["reason"] == "coverage not executed"


def test_missing_page_witness_is_rejected():
    declaration, traversal, witnesses = build_world()
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses[:1],
        )


def test_duplicate_page_witness_is_rejected():
    declaration, traversal, witnesses = build_world()
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=[witnesses[0], copy.deepcopy(witnesses[0]), witnesses[1]],
        )


def test_unassessed_witness_cannot_smuggle_assessed_artifacts():
    declaration, traversal, witnesses = build_world()
    witnesses[1] = {
        "page_id": "page-2",
        "disposition": "Unassessed",
        "reason": "not assessed",
        "coverage": witnesses[1]["coverage"],
    }
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_valid_receipt_from_wrong_source_ref_is_rejected():
    declaration, traversal, witnesses = build_world()
    wrong = page_bundle("response:other", [full_step(3), full_step(4)])
    witnesses[1] = {
        "page_id": "page-2",
        "disposition": "Assessed",
        "receipt": wrong["receipt"],
        "coverage": wrong["coverage"],
        "profile_binding": wrong["profile_binding"],
    }
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_valid_receipt_with_wrong_raw_digest_is_rejected():
    declaration, traversal, witnesses = build_world()
    wrong = page_bundle("response:page-2", [full_step(30)])
    witnesses[1] = {
        "page_id": "page-2",
        "disposition": "Assessed",
        "receipt": wrong["receipt"],
        "coverage": wrong["coverage"],
        "profile_binding": wrong["profile_binding"],
    }
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_receipt_resource_kind_must_match_traversal_domain():
    declaration, traversal, witnesses = build_world(traversal_resource_kind="WorkflowJobsCollection")
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_profile_binding_substitution_is_rejected():
    declaration, traversal, witnesses = build_world()
    witnesses[1]["profile_binding"] = copy.deepcopy(witnesses[0]["profile_binding"])
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.derive_coverage_set(
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_selector_must_exist_exactly_once_in_each_assessed_page():
    with pytest.raises(hak18b.TraversalCoverageSetError):
        derive(selector="steps[*].does_not_exist")


def test_indeterminate_cannot_be_redigested_as_present():
    declaration, traversal, witnesses = build_world()
    witnesses[1] = {"page_id": "page-2", "disposition": "Unassessed", "reason": "not assessed"}
    _, _, _, record = derive((declaration, traversal, witnesses))
    record["aggregate"]["coverage_state"] = "Present"
    redigest(record)
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.validate_coverage_set(
            record,
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_population_qualification_cannot_be_promoted_after_redigest():
    declaration, traversal, witnesses, record = derive()
    record["population_qualification"] = "Verified"
    redigest(record)
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.validate_coverage_set(
            record,
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_normalization_input_replay_cannot_be_promoted_after_redigest():
    declaration, traversal, witnesses, record = derive()
    record["normalization_input_replay"] = "Verified"
    redigest(record)
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.validate_coverage_set(
            record,
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )


def test_source_assurance_cannot_be_promoted_after_redigest():
    declaration, traversal, witnesses, record = derive()
    record["source_assurance"]["provider_authentication"] = "Verified"
    redigest(record)
    with pytest.raises(hak18b.TraversalCoverageSetError):
        hak18b.validate_coverage_set(
            record,
            selector_path=SELECTOR,
            traversal_record=traversal,
            hak017_record=declaration,
            page_witnesses=witnesses,
        )
