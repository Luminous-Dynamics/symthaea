// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::{Value, json};
use symthaea_continuity::{
    CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1, CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1,
    CapabilityActivationAssumptionsV1, CapabilityActivationProvenanceV1,
    CapabilityAnalysisProvenanceError, CapabilityCounterfactualConfigV1,
    CapabilityCounterfactualFrontierV1, CapabilityCounterfactualProvenanceV1,
    CapabilityCounterfactualQueryV1, CapabilityDefinitionV1, CapabilityGraphSnapshotV1,
    CapabilityRequirementV1, capability_activation_algorithm_id_v1,
    capability_counterfactual_algorithm_id_v1, derive_capability_activation_closure,
    derive_capability_counterfactual_frontier,
};

fn leaf(name: &str) -> CapabilityDefinitionV1 {
    CapabilityDefinitionV1::new("org.example", name, None).unwrap()
}

fn activation_fixture(prefix: &str) -> symthaea_continuity::CapabilityActivationClosureV1 {
    let dependency = leaf(&format!("{prefix}-dependency"));
    let target = CapabilityDefinitionV1::new(
        "org.example",
        &format!("{prefix}-target"),
        Some(CapabilityRequirementV1::leaf(dependency.id())),
    )
    .unwrap();
    let target_id = target.id();
    let graph = CapabilityGraphSnapshotV1::new(vec![dependency.clone(), target])
        .unwrap()
        .validate()
        .unwrap();
    let assumptions =
        CapabilityActivationAssumptionsV1::new(&graph, vec![dependency.id()], vec![target_id])
            .unwrap()
            .validate(&graph)
            .unwrap();
    derive_capability_activation_closure(&graph, &assumptions).unwrap()
}

fn counterfactual_fixture(prefix: &str) -> CapabilityCounterfactualFrontierV1 {
    let dependency = leaf(&format!("{prefix}-dependency"));
    let target = CapabilityDefinitionV1::new(
        "org.example",
        &format!("{prefix}-target"),
        Some(CapabilityRequirementV1::leaf(dependency.id())),
    )
    .unwrap();
    let target_id = target.id();
    let graph = CapabilityGraphSnapshotV1::new(vec![dependency, target])
        .unwrap()
        .validate()
        .unwrap();
    let assumptions = CapabilityActivationAssumptionsV1::new(&graph, vec![], vec![target_id])
        .unwrap()
        .validate(&graph)
        .unwrap();
    let query = CapabilityCounterfactualQueryV1::new(&graph, &assumptions, vec![target_id])
        .unwrap()
        .validate(&graph, &assumptions)
        .unwrap();
    let config = CapabilityCounterfactualConfigV1::new(8, 2, 16, 100)
        .unwrap()
        .validate()
        .unwrap();
    derive_capability_counterfactual_frontier(&graph, &assumptions, &query, &config).unwrap()
}

fn mutate_32_byte_id(value: &mut Value, field: &str, byte: u8) {
    value[field] = Value::Array((0..32).map(|_| json!(byte)).collect());
}

#[test]
fn activation_transport_round_trip_preserves_exact_provenance() {
    let closure = activation_fixture("activation-roundtrip");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let encoded = serde_json::to_vec(&provenance).unwrap();
    let transported: CapabilityActivationProvenanceV1 = serde_json::from_slice(&encoded).unwrap();

    assert_eq!(transported, provenance);
    assert_eq!(
        transported.algorithm_semantics(),
        CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1
    );
    transported.validate_against(&closure).unwrap();
}

#[test]
fn activation_transport_schema_substitution_fails_closed() {
    let closure = activation_fixture("activation-schema");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["schema_version"] = json!("future-activation-provenance-v2");

    let error =
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported activation provenance schema")
    );
}

#[test]
fn activation_transport_semantics_substitution_fails_closed() {
    let closure = activation_fixture("activation-semantics");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["algorithm_semantics"] = json!("future-activation-semantics-v2");

    let error =
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported activation algorithm semantics")
    );
}

#[test]
fn activation_transport_algorithm_id_substitution_fails_closed() {
    let closure = activation_fixture("activation-algorithm-id");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let mut transported = serde_json::to_value(provenance).unwrap();
    mutate_32_byte_id(&mut transported, "algorithm_id", 9);

    let error =
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).unwrap_err();
    assert!(error.to_string().contains("activation algorithm identity"));
}

#[test]
fn activation_transport_result_substitution_fails_closed() {
    let left = activation_fixture("activation-left");
    let right = activation_fixture("activation-right");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&left);

    assert_eq!(
        provenance.validate_against(&right),
        Err(CapabilityAnalysisProvenanceError::ActivationResultReceiptMismatch)
    );
}

#[test]
fn activation_transport_provenance_id_substitution_fails_closed() {
    let closure = activation_fixture("activation-provenance-id");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let mut transported = serde_json::to_value(provenance).unwrap();
    mutate_32_byte_id(&mut transported, "provenance_id", 11);

    let error =
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).unwrap_err();
    assert!(error.to_string().contains("activation provenance identity"));
}

#[test]
fn counterfactual_transport_round_trip_preserves_exact_provenance() {
    let frontier = counterfactual_fixture("counterfactual-roundtrip");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let encoded = serde_json::to_vec(&provenance).unwrap();
    let transported: CapabilityCounterfactualProvenanceV1 =
        serde_json::from_slice(&encoded).unwrap();

    assert_eq!(transported, provenance);
    assert_eq!(
        transported.algorithm_semantics(),
        CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1
    );
    transported.validate_against(&frontier).unwrap();
}

#[test]
fn counterfactual_transport_schema_substitution_fails_closed() {
    let frontier = counterfactual_fixture("counterfactual-schema");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["schema_version"] = json!("future-counterfactual-provenance-v2");

    let error =
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported counterfactual provenance schema")
    );
}

#[test]
fn counterfactual_transport_semantics_substitution_fails_closed() {
    let frontier = counterfactual_fixture("counterfactual-semantics");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["algorithm_semantics"] = json!("future-counterfactual-semantics-v2");

    let error =
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported counterfactual algorithm semantics")
    );
}

#[test]
fn counterfactual_transport_algorithm_id_substitution_fails_closed() {
    let frontier = counterfactual_fixture("counterfactual-algorithm-id");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let mut transported = serde_json::to_value(provenance).unwrap();
    mutate_32_byte_id(&mut transported, "algorithm_id", 13);

    let error =
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("counterfactual algorithm identity")
    );
}

#[test]
fn counterfactual_transport_result_substitution_fails_closed() {
    let left = counterfactual_fixture("counterfactual-left");
    let right = counterfactual_fixture("counterfactual-right");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&left);

    assert_eq!(
        provenance.validate_against(&right),
        Err(CapabilityAnalysisProvenanceError::CounterfactualResultReceiptMismatch)
    );
}

#[test]
fn counterfactual_transport_provenance_id_substitution_fails_closed() {
    let frontier = counterfactual_fixture("counterfactual-provenance-id");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let mut transported = serde_json::to_value(provenance).unwrap();
    mutate_32_byte_id(&mut transported, "provenance_id", 15);

    let error =
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("counterfactual provenance identity")
    );
}

#[test]
fn activation_and_counterfactual_algorithm_domains_are_distinct() {
    assert_ne!(
        capability_activation_algorithm_id_v1().as_bytes(),
        capability_counterfactual_algorithm_id_v1().as_bytes(),
    );
}

#[test]
fn activation_transport_unknown_field_is_rejected() {
    let closure = activation_fixture("activation-unknown-field");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["authority"] = json!("operator-approved");

    assert!(
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).is_err(),
        "V1 activation provenance must reject uncommitted transport fields"
    );
}

#[test]
fn counterfactual_transport_unknown_field_is_rejected() {
    let frontier = counterfactual_fixture("counterfactual-unknown-field");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let mut transported = serde_json::to_value(provenance).unwrap();
    transported["authority"] = json!("operator-approved");

    assert!(
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).is_err(),
        "V1 counterfactual provenance must reject uncommitted transport fields"
    );
}

#[test]
fn activation_payload_cannot_decode_as_counterfactual_provenance() {
    let closure = activation_fixture("activation-cross-family");
    let provenance = CapabilityActivationProvenanceV1::from_closure(&closure);
    let transported = serde_json::to_value(provenance).unwrap();

    let error =
        serde_json::from_value::<CapabilityCounterfactualProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported counterfactual provenance schema")
    );
}

#[test]
fn counterfactual_payload_cannot_decode_as_activation_provenance() {
    let frontier = counterfactual_fixture("counterfactual-cross-family");
    let provenance = CapabilityCounterfactualProvenanceV1::from_frontier(&frontier);
    let transported = serde_json::to_value(provenance).unwrap();

    let error =
        serde_json::from_value::<CapabilityActivationProvenanceV1>(transported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported activation provenance schema")
    );
}

#[test]
fn decoded_activation_provenance_binding_is_runtime_specific() {
    let left = activation_fixture("activation-bound-left");
    let right = activation_fixture("activation-bound-right");
    let encoded =
        serde_json::to_vec(&CapabilityActivationProvenanceV1::from_closure(&left)).unwrap();
    let decoded: CapabilityActivationProvenanceV1 = serde_json::from_slice(&encoded).unwrap();

    let bound = decoded.bind_to(&left).unwrap();
    assert_eq!(bound.provenance_id(), decoded.id());
    assert_eq!(bound.result_receipt_id(), decoded.result_receipt_id());
    assert!(std::ptr::eq(bound.provenance(), &decoded));
    assert!(std::ptr::eq(bound.runtime_result(), &left));
    assert_eq!(
        decoded.bind_to(&right).unwrap_err(),
        CapabilityAnalysisProvenanceError::ActivationResultReceiptMismatch
    );
}

#[test]
fn decoded_counterfactual_provenance_binding_is_runtime_specific() {
    let left = counterfactual_fixture("counterfactual-bound-left");
    let right = counterfactual_fixture("counterfactual-bound-right");
    let encoded =
        serde_json::to_vec(&CapabilityCounterfactualProvenanceV1::from_frontier(&left)).unwrap();
    let decoded: CapabilityCounterfactualProvenanceV1 = serde_json::from_slice(&encoded).unwrap();

    let bound = decoded.bind_to(&left).unwrap();
    assert_eq!(bound.provenance_id(), decoded.id());
    assert_eq!(bound.result_receipt_id(), decoded.result_receipt_id());
    assert!(std::ptr::eq(bound.provenance(), &decoded));
    assert!(std::ptr::eq(bound.runtime_result(), &left));
    assert_eq!(
        decoded.bind_to(&right).unwrap_err(),
        CapabilityAnalysisProvenanceError::CounterfactualResultReceiptMismatch
    );
}
