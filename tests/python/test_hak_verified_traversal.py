import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec17 = importlib.util.spec_from_file_location(
    "hak_collection_completeness", SCRIPTS / "hak_collection_completeness.py"
)
assert spec17 and spec17.loader
hak17 = importlib.util.module_from_spec(spec17)
sys.modules["hak_collection_completeness"] = hak17
spec17.loader.exec_module(hak17)

spec18 = importlib.util.spec_from_file_location(
    "hak_verified_traversal", SCRIPTS / "hak_verified_traversal.py"
)
assert spec18 and spec18.loader
hak = importlib.util.module_from_spec(spec18)
spec18.loader.exec_module(hak)


def base_hak017(*, page_size=100):
    return hak17.make_record(
        provider="github-actions",
        resource_kind="WorkflowJobsCollection",
        collection_ref="github-actions:test:jobs:all-pages",
        query_scope_ref="query:jobs:latest-attempt",
        population_extent="FullProviderPopulation",
        scope_ref="scope:workflow-run-jobs:latest-attempt",
        filtered=False,
        sampled=False,
        time_windowed=False,
        permission_limited=False,
        pagination_model="LinkHeader",
        pagination_contract_ref="provider-contract:github-rest-pagination:v1",
        pages_observed=2,
        page_size=page_size,
        exhaustion_kind="NoContinuationByProviderContract",
        exhaustion_evidence_refs=["provider-header:link:no-next"],
        retained_raw_count=4,
        retained_unique_count=3,
        duplicate_count=1,
        dedup_identity_ref=hak.DEDUP_IDENTITY_REF,
        dedup_rule_ref=hak.DEDUP_RULE_REF,
        provider_reported_total_count=3,
        provider_total_scope_ref="scope:workflow-run-jobs:latest-attempt",
        collection_status="DeclaredComplete",
        reasons=[],
    )


def base_transcript():
    return {
        "provider": "github-actions",
        "resource_kind": "WorkflowJobsCollection",
        "collection_ref": "github-actions:test:jobs:all-pages",
        "query_scope_ref": "query:jobs:latest-attempt",
        "scope_ref": "scope:workflow-run-jobs:latest-attempt",
        "pagination_model": "LinkHeader",
        "pagination_contract_ref": "provider-contract:github-rest-pagination:v1",
        "temporal": {
            "declared_semantics": "BestEffortLiveTraversal",
            "evidence_refs": [],
        },
        "pages": [
            {
                "ordinal": 1,
                "page_id": "page-1",
                "request_ref": "request:page-1",
                "continuation_in": None,
                "response_ref": "response:page-1",
                "raw_response_digest": "sha256:" + "1" * 64,
                "continuation_out": "next:page-2",
                "observed_after_or_at": "2026-09-09T18:00:00Z",
                "observed_before_or_at": "2026-09-09T18:00:01Z",
                "entity_ids": ["job-a", "job-b"],
            },
            {
                "ordinal": 2,
                "page_id": "page-2",
                "request_ref": "request:page-2",
                "continuation_in": "next:page-2",
                "response_ref": "response:page-2",
                "raw_response_digest": "sha256:" + "2" * 64,
                "continuation_out": None,
                "observed_after_or_at": "2026-09-09T18:00:01Z",
                "observed_before_or_at": "2026-09-09T18:00:02Z",
                "entity_ids": ["job-b", "job-c"],
            },
        ],
    }


def valid_record():
    return hak.make_verification_record(base_transcript(), base_hak017())


def redigest(record):
    record["transcript_digest"] = hak.compute_transcript_digest(record["transcript"])
    record["traversal_digest"] = hak.compute_traversal_digest(record)
    return record


def coherent_redigest_dedup(record):
    payload = {k: v for k, v in record["dedup_execution"].items() if k != "receipt_digest"}
    record["dedup_execution"]["receipt_digest"] = hak.canonical.hak_sha256(
        hak.DEDUP_DOMAIN, payload
    )
    return redigest(record)


def test_valid_retained_transcript_verifies_and_replays_dedup():
    record = valid_record()
    hak.validate_verification_record(record, hak017_record=base_hak017())
    assert record["verification_status"] == "VerifiedAgainstRetainedTranscript"
    assert record["provider_authentication"] == "NotEstablished"
    assert record["continuation_source_verification"] == "NotEstablished"
    assert record["temporal_snapshot_verification"] == "NotEstablished"
    assert record["raw_response_content_verification"] == "NotEstablished"
    assert record["entity_projection_verification"] == "NotEstablished"
    assert record["dedup_execution"]["retained_unique_ids"] == ["job-a", "job-b", "job-c"]
    assert record["dedup_execution"]["duplicate_count"] == 1


def test_skipped_continuation_edge_is_rejected():
    transcript = base_transcript()
    transcript["pages"][1]["continuation_in"] = "next:page-3"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_reused_continuation_token_cycle_is_rejected():
    transcript = base_transcript()
    transcript["pages"].insert(1, {
        "ordinal": 2,
        "page_id": "page-middle",
        "request_ref": "request:page-middle",
        "continuation_in": "next:page-2",
        "response_ref": "response:page-middle",
        "raw_response_digest": "sha256:" + "3" * 64,
        "continuation_out": "next:page-2",
        "observed_after_or_at": "2026-09-09T18:00:01Z",
        "observed_before_or_at": "2026-09-09T18:00:01Z",
        "entity_ids": [],
    })
    transcript["pages"][2]["ordinal"] = 3
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_duplicate_page_id_is_rejected():
    transcript = base_transcript()
    transcript["pages"][1]["page_id"] = "page-1"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_final_page_cannot_retain_continuation():
    transcript = base_transcript()
    transcript["pages"][-1]["continuation_out"] = "next:page-3"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_nonpaginated_model_cannot_accept_multiple_pages():
    transcript = base_transcript()
    transcript["pagination_model"] = "NonPaginated"
    transcript["pagination_contract_ref"] = "provider-contract:single-response:v1"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_page_observation_window_inversion_is_rejected():
    transcript = base_transcript()
    transcript["pages"][0]["observed_after_or_at"] = "2026-09-09T18:00:02Z"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_later_page_cannot_begin_before_prior_response_bound():
    transcript = base_transcript()
    transcript["pages"][1]["observed_after_or_at"] = "2026-09-09T18:00:00Z"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_dedup_output_changed_after_coherent_redigest_is_rejected():
    record = valid_record()
    record["dedup_execution"]["retained_unique_ids"] = ["job-a", "job-c", "job-b"]
    coherent_redigest_dedup(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_hak017_retained_counts_must_match_transcript_replay():
    declaration = base_hak017()
    transcript = base_transcript()
    transcript["pages"][1]["entity_ids"].append("job-d")
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, declaration)


def test_hak017_source_scope_must_match_transcript():
    transcript = base_transcript()
    transcript["scope_ref"] = "scope:different"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_hak017_pagination_contract_must_match_transcript():
    transcript = base_transcript()
    transcript["pagination_contract_ref"] = "provider-contract:different:v1"
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(transcript, base_hak017())


def test_different_hak017_declaration_with_same_human_refs_is_rejected():
    record = valid_record()
    different = base_hak017(page_size=50)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=different)


def test_provider_authentication_cannot_be_promoted_by_redigest():
    record = valid_record()
    record["provider_authentication"] = "Authenticated"
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_temporal_verification_cannot_be_promoted_by_redigest():
    record = valid_record()
    record["temporal_snapshot_verification"] = "ProviderSnapshotBound"
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_continuation_source_verification_cannot_be_promoted_by_redigest():
    record = valid_record()
    record["continuation_source_verification"] = "ProviderResponseVerified"
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_raw_response_content_verification_cannot_be_promoted_by_redigest():
    record = valid_record()
    record["raw_response_content_verification"] = "RawBytesReplayed"
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_entity_projection_verification_cannot_be_promoted_by_redigest():
    record = valid_record()
    record["entity_projection_verification"] = "ProviderResponseVerified"
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_verification_requires_exact_hak017_witness():
    record = valid_record()
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record)


def test_non_declared_complete_hak017_witness_is_rejected():
    declaration = base_hak017()
    declaration["collection_status"] = "Partial"
    declaration["reasons"] = ["collection declaration is partial"]
    declaration["collection_digest"] = hak17.compute_collection_digest(declaration)
    hak17.validate_collection_record(declaration)
    with pytest.raises(hak.TraversalVerificationError):
        hak.make_verification_record(base_transcript(), declaration)


def test_profile_identity_substitution_rejected_after_redigest():
    record = valid_record()
    record["canonicalization_profile"]["artifact_ref"] = (
        "git:Luminous-Dynamics/symthaea@" + "0" * 40
        + ":docs/architecture/hak/canonical-json-v1.profile.json"
    )
    redigest(record)
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())


def test_traversal_digest_tampering_is_rejected():
    record = valid_record()
    record["traversal_digest"] = "sha256:" + "0" * 64
    with pytest.raises(hak.TraversalVerificationError):
        hak.validate_verification_record(record, hak017_record=base_hak017())
