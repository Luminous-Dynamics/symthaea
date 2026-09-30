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
    trajectory_evidence::{qualify_transition, OperatorEvidenceDependency, RequiredOperator},
};

#[test]
fn generated_operator_evidence_unlocks_a_declared_trajectory_transition() {
    let json = extended_resolution_evidence_json();
    let digest = sha256_hex(json.as_bytes());

    let artifact =
        verify_artifact_bytes(json.as_bytes(), &digest).expect("generated evidence must verify");
    let summary = qualify_records(&artifact.records);
    assert!(summary.qualified);

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

    qualify_transition(&artifact, &dependency)
        .expect("qualified operator evidence must unlock transition");
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
