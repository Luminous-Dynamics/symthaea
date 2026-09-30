//! End-to-end research gate: generated operator evidence -> verified artifact -> trajectory qualification.
//!
//! This test intentionally crosses the producer/contract/trajectory boundary.
//! It proves the dependency chain without making any task-quality or performance claim.

#![cfg(feature = "simd")]

use symthaea_core::hdc::{
    operator_evidence::{
        extended_resolution_evidence_json, generate_extended_resolution_evidence,
    },
    operator_evidence_contract::{qualify_records, sha256_hex, verify_artifact_bytes},
    resource_evidence::{ResourceBudget, ResourceEvidenceRecord, ResourceWorkload, RESOURCE_EVIDENCE_SCHEMA_VERSION, RESOURCE_QUALIFIED_STATUS},
    trajectory_evidence::{qualify_transition, OperatorEvidenceDependency, RequiredOperator, ResourceEvidenceDependency},
};


fn resource_record() -> ResourceEvidenceRecord {
    let workload = ResourceWorkload {
        resolution: 131_072,
        representation: "continuous_f32".to_owned(),
        element_size_bytes: 4,
        resident_vectors: 4,
    };
    ResourceEvidenceRecord {
        schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION,
        vector_bytes: workload.vector_bytes().unwrap(),
        resident_bytes: workload.resident_bytes().unwrap(),
        peak_temporary_bytes: Some(64 * 1024),
        conversion_bytes: Some(0),
        provenance_id: "e2e-resource-v1".to_owned(),
        qualification_status: RESOURCE_QUALIFIED_STATUS.to_owned(),
        budget: Some(ResourceBudget::new(512 * 1024, 2 * 1024 * 1024, Some(64 * 1024))),
        workload,
    }
}

#[test]
fn generated_operator_evidence_unlocks_a_declared_trajectory_transition() {
    let json = extended_resolution_evidence_json();
    let digest = sha256_hex(json.as_bytes());

    let artifact =
        verify_artifact_bytes(json.as_bytes(), &digest).expect("generated evidence must verify");
    let summary = qualify_records(&artifact.records);
    assert!(summary.qualified);

    let resource = resource_record();
    let resource_dependency = ResourceEvidenceDependency {
        schema_version: resource.schema_version,
        resolution: resource.workload.resolution,
        representation: resource.workload.representation.clone(),
        provenance_id: resource.provenance_id.clone(),
    };

    let dependency = OperatorEvidenceDependency {
        schema_version: artifact.schema_version,
        representation: artifact.representation.clone(),
        artifact_sha256: digest,
        required_operators: vec![
            RequiredOperator {
                resolution: 65_536,
                operation: "dot".to_owned(),
            },
            RequiredOperator {
                resolution: 131_072,
                operation: "bind".to_owned(),
            },
            RequiredOperator {
                resolution: 262_144,
                operation: "similarity".to_owned(),
            },
        ],
    };

    qualify_transition(&artifact, &dependency, &resource, &resource_dependency)
        .expect("qualified operator and resource evidence must unlock transition");
}

#[test]
fn generated_evidence_is_rejected_when_digest_is_tampered() {
    let mut json = extended_resolution_evidence_json().into_bytes();
    let digest = sha256_hex(&json);
    json.push(b' ');

    assert!(verify_artifact_bytes(&json, &digest).is_err());
}

#[test]
fn generated_evidence_has_the_declared_complete_matrix() {
    let records = generate_extended_resolution_evidence();
    assert_eq!(records.len(), 25);
    assert!(qualify_records(&records).qualified);
}

#[test]
fn trajectory_chain_rejects_resource_budget_excess() {
    let json = extended_resolution_evidence_json();
    let digest = sha256_hex(json.as_bytes());
    let artifact = verify_artifact_bytes(json.as_bytes(), &digest).expect("evidence verifies");

    let resource = resource_record();
    let mut resource = resource;
    resource.budget.as_mut().unwrap().max_resident_bytes -= 1;
    let resource_dependency = ResourceEvidenceDependency {
        schema_version: resource.schema_version,
        resolution: resource.workload.resolution,
        representation: resource.workload.representation.clone(),
        provenance_id: resource.provenance_id.clone(),
    };
    let dependency = OperatorEvidenceDependency {
        schema_version: artifact.schema_version,
        representation: artifact.representation.clone(),
        artifact_sha256: digest,
        required_operators: vec![RequiredOperator {
            resolution: 65_536,
            operation: "dot".to_owned(),
        }],
    };

    assert!(qualify_transition(&artifact, &dependency, &resource, &resource_dependency).is_err());
}
