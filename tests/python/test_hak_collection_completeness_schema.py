import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs/architecture/hak/collection-completeness-v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def base_record():
    scope_ref = "scope:all-jobs"
    return {
        "schema_version": "hak.collection-completeness.v1",
        "canonicalization_profile": {
            "profile_id": "hak.canonical-json.v1",
            "artifact_ref": "git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:docs/architecture/hak/canonical-json-v1.profile.json",
            "raw_sha256": "sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830",
        },
        "digest_domain": "hak.collection-completeness.v1",
        "source_collection": {
            "provider": "github-actions",
            "resource_kind": "WorkflowJobsCollection",
            "collection_ref": "github-actions:test:jobs",
            "query_scope_ref": "query:latest-attempt",
        },
        "scope": {
            "population_extent": "FullProviderPopulation",
            "scope_ref": scope_ref,
            "constraints": {
                "filtered": False,
                "sampled": False,
                "time_windowed": False,
                "permission_limited": False,
            },
        },
        "pagination": {
            "model": "LinkHeader",
            "contract_ref": "provider-contract:github-rest-pagination:v1",
            "pages_observed": 1,
            "page_size": 100,
            "exhaustion": {
                "kind": "NoContinuationByProviderContract",
                "evidence_refs": ["provider-header:link:no-next"],
            },
        },
        "counts": {
            "retained_raw_count": 2,
            "retained_unique_count": 2,
            "duplicate_count": 0,
            "deduplication": {
                "identity_ref": "provider-field:job.id",
                "rule_ref": "hak:dedup:first-occurrence-by-identity:v1",
            },
            "provider_reported_total": {
                "value": 2,
                "scope_ref": scope_ref,
            },
        },
        "collection_status": "DeclaredComplete",
        "reasons": [],
        "collection_digest": "sha256:" + "1" * 64,
    }


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


def test_schema_accepts_structurally_valid_record():
    VALIDATOR.validate(base_record())


def test_schema_accepts_defined_filtered_subset_shape():
    record = base_record()
    record["scope"]["population_extent"] = "DefinedSubset"
    record["scope"]["constraints"]["filtered"] = True
    VALIDATOR.validate(record)


def test_schema_accepts_missing_provider_total_as_null():
    record = base_record()
    record["counts"]["provider_reported_total"] = None
    VALIDATOR.validate(record)


def test_schema_rejects_old_complete_label():
    record = base_record()
    record["collection_status"] = "Complete"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_unknown_top_level_field():
    record = base_record()
    record["unexpected"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_unknown_pagination_field():
    record = base_record()
    record["pagination"]["unexpected"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_unknown_population_extent():
    record = base_record()
    record["scope"]["population_extent"] = "EverythingProbably"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_unknown_scope_constraint():
    record = base_record()
    record["scope"]["constraints"]["unexpected"] = False
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_unknown_dedup_field():
    record = base_record()
    record["counts"]["deduplication"]["unexpected"] = "x"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_schema_rejects_out_of_range_count():
    record = base_record()
    record["counts"]["retained_raw_count"] = 9007199254740992
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)
