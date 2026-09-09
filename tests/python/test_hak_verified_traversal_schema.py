import importlib.util
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

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

SCHEMA = json.loads(
    (ROOT / "docs/architecture/hak/collection-traversal-verification-v1.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)


def declaration():
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
        page_size=100,
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


def transcript():
    return {
        "provider": "github-actions",
        "resource_kind": "WorkflowJobsCollection",
        "collection_ref": "github-actions:test:jobs:all-pages",
        "query_scope_ref": "query:jobs:latest-attempt",
        "scope_ref": "scope:workflow-run-jobs:latest-attempt",
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


def record():
    return hak.make_verification_record(transcript(), declaration())


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


def test_schema_accepts_generated_record():
    VALIDATOR.validate(record())


def test_schema_rejects_unknown_top_level_field():
    value = record()
    value["unexpected"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_provider_authentication_promotion():
    value = record()
    value["provider_authentication"] = "Authenticated"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_temporal_verification_promotion():
    value = record()
    value["temporal_snapshot_verification"] = "ProviderSnapshotBound"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_continuation_source_verification_promotion():
    value = record()
    value["continuation_source_verification"] = "ProviderResponseVerified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_raw_response_content_verification_promotion():
    value = record()
    value["raw_response_content_verification"] = "RawBytesReplayed"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_entity_projection_verification_promotion():
    value = record()
    value["entity_projection_verification"] = "ProviderResponseVerified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)


def test_schema_rejects_unknown_page_field():
    value = record()
    value["transcript"]["pages"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(value)
