import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("hak_collection_completeness", SCRIPTS / "hak_collection_completeness.py")
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

SCOPE_REF = "scope:workflow-run-jobs:latest-attempt"


def complete_record(*, total=2, raw=2, unique=2, duplicates=0,
                    extent="FullProviderPopulation", filtered=False,
                    sampled=False, time_windowed=False, permission_limited=False):
    return hak.make_record(
        provider="github-actions",
        resource_kind="WorkflowJobsCollection",
        collection_ref="github-actions:test:jobs:all-pages",
        query_scope_ref="query:jobs:latest-attempt",
        population_extent=extent,
        scope_ref=SCOPE_REF,
        filtered=filtered,
        sampled=sampled,
        time_windowed=time_windowed,
        permission_limited=permission_limited,
        pagination_model="LinkHeader",
        pagination_contract_ref="provider-contract:github-rest-pagination:v1",
        pages_observed=1,
        page_size=100,
        exhaustion_kind="NoContinuationByProviderContract",
        exhaustion_evidence_refs=["provider-header:link:no-next"],
        retained_raw_count=raw,
        retained_unique_count=unique,
        duplicate_count=duplicates,
        dedup_identity_ref="provider-field:job.id",
        dedup_rule_ref="hak:dedup:first-occurrence-by-identity:v1",
        provider_reported_total_count=total,
        provider_total_scope_ref=SCOPE_REF if total is not None else None,
        collection_status="DeclaredComplete",
        reasons=[],
    )


def redigest(record):
    record["collection_digest"] = hak.compute_collection_digest(record)
    return record


def test_declared_complete_requires_exhaustion_not_only_count_equality():
    record = complete_record()
    record["pagination"]["exhaustion"] = {"kind": "Unknown", "evidence_refs": []}
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_first_page_only_cannot_claim_declared_complete_without_exhaustion():
    record = complete_record()
    record["counts"]["provider_reported_total"]["value"] = 100
    record["pagination"]["exhaustion"] = {"kind": "Unknown", "evidence_refs": []}
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_defined_filtered_subset_can_be_declared_complete_without_becoming_provider_population_complete():
    record = complete_record(extent="DefinedSubset", filtered=True)
    hak.validate_collection_record(record)
    assert hak.provider_population_declared_complete(record) is False


def test_unknown_population_extent_cannot_claim_declared_complete():
    record = complete_record()
    record["scope"]["population_extent"] = "Unknown"
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_full_provider_population_cannot_carry_narrowing_constraints():
    record = complete_record()
    record["scope"]["constraints"]["filtered"] = True
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_full_provider_population_declared_complete_requires_unconstrained_record():
    record = complete_record()
    assert hak.provider_population_declared_complete(record) is True


def test_unknown_pagination_cannot_claim_declared_complete():
    record = complete_record()
    record["pagination"]["model"] = "Unknown"
    record["pagination"]["contract_ref"] = None
    record["pagination"]["exhaustion"] = {"kind": "Unknown", "evidence_refs": []}
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_unknown_pagination_cannot_keep_contract_ref():
    record = complete_record()
    record["collection_status"] = "Unknown"
    record["reasons"] = ["pagination model unavailable"]
    record["pagination"]["model"] = "Unknown"
    record["pagination"]["exhaustion"] = {"kind": "Unknown", "evidence_refs": []}
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_cursor_model_rejects_link_header_exhaustion_kind():
    record = complete_record()
    record["pagination"]["model"] = "Cursor"
    record["pagination"]["contract_ref"] = "provider-contract:cursor:v1"
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_link_header_rejects_cursor_exhaustion_kind():
    record = complete_record()
    record["pagination"]["exhaustion"] = {
        "kind": "CursorExhausted",
        "evidence_refs": ["provider-cursor:none"],
    }
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_provider_total_less_than_retained_unique_is_rejected():
    record = complete_record()
    record["counts"]["provider_reported_total"]["value"] = 1
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_provider_total_scope_must_match_declared_scope():
    record = complete_record()
    record["counts"]["provider_reported_total"]["scope_ref"] = "scope:some-other-query"
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_count_conservation_rejects_duplicate_inflation():
    record = complete_record()
    record["counts"]["duplicate_count"] = 1
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_unique_count_requires_named_dedup_identity_and_rule():
    record = complete_record()
    record["counts"]["deduplication"]["identity_ref"] = ""
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_declared_complete_with_provider_total_requires_exact_unique_count():
    record = complete_record()
    record["counts"]["provider_reported_total"]["value"] = 3
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_partial_requires_reason_and_can_preserve_exhaustion_unknown():
    record = complete_record()
    record["collection_status"] = "Partial"
    record["reasons"] = ["collection interrupted after first page"]
    record["pagination"]["exhaustion"] = {"kind": "Unknown", "evidence_refs": []}
    redigest(record)
    hak.validate_collection_record(record)


def test_unknown_requires_reason():
    record = complete_record()
    record["collection_status"] = "Unknown"
    record["reasons"] = []
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_not_applicable_requires_reason():
    record = complete_record()
    record["collection_status"] = "NotApplicable"
    record["reasons"] = []
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_nonpaginated_declared_complete_requires_matching_exhaustion_kind():
    record = complete_record()
    record["pagination"]["model"] = "NonPaginated"
    record["pagination"]["contract_ref"] = "provider-contract:nonpaginated:v1"
    record["pagination"]["exhaustion"] = {
        "kind": "NoContinuationByProviderContract",
        "evidence_refs": ["provider-response:no-pagination"],
    }
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_nonpaginated_endpoint_can_be_declared_complete_with_explicit_reference():
    record = complete_record()
    record["pagination"]["model"] = "NonPaginated"
    record["pagination"]["contract_ref"] = "provider-contract:nonpaginated:v1"
    record["pagination"]["exhaustion"] = {
        "kind": "NonPaginatedEndpoint",
        "evidence_refs": ["provider-contract:endpoint-nonpaginated"],
    }
    redigest(record)
    hak.validate_collection_record(record)


def test_unknown_exhaustion_cannot_claim_positive_evidence_refs():
    record = complete_record()
    record["collection_status"] = "Unknown"
    record["reasons"] = ["no verified exhaustion evidence"]
    record["pagination"]["exhaustion"] = {
        "kind": "Unknown",
        "evidence_refs": ["convenient-but-unverified:ref"],
    }
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_profile_identity_substitution_rejected_after_redigest():
    record = complete_record()
    record["canonicalization_profile"]["artifact_ref"] = (
        "git:Luminous-Dynamics/symthaea@" + "0" * 40 + ":docs/architecture/hak/canonical-json-v1.profile.json"
    )
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_profile_raw_digest_substitution_rejected_after_redigest():
    record = complete_record()
    record["canonicalization_profile"]["raw_sha256"] = "sha256:" + "0" * 64
    redigest(record)
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)


def test_collection_digest_tampering_rejected():
    record = complete_record()
    record["collection_digest"] = "sha256:" + "0" * 64
    with pytest.raises(hak.CollectionCompletenessError):
        hak.validate_collection_record(record)
