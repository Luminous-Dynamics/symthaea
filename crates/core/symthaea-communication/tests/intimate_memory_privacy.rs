// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimate_memory_privacy.rs"]
mod intimate_memory_privacy;

use intimate_memory_privacy::*;
use std::collections::BTreeSet;

fn root(id: &str, retention: IntimateRetentionClassV1) -> IntimateMemoryArtifactV1 {
    IntimateMemoryArtifactV1::root(id, IntimateArtifactKindV1::RawTranscript, retention).unwrap()
}

#[test]
fn reconstructive_chain_is_removed_from_retrieval_after_source_retraction() {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    graph.insert(root("source", IntimateRetentionClassV1::EphemeralSession)).unwrap();
    graph.insert(IntimateMemoryArtifactV1 {
        artifact_id: "summary".into(),
        kind: IntimateArtifactKindV1::SessionSummary,
        retention: IntimateRetentionClassV1::EphemeralSession,
        dependency_state: IntimateDependencyStateV1::Complete,
        derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
        reconstructive: true,
        source_ids: BTreeSet::from(["source".into()]),
        durability_authorization_ref: None,
        export_receipt_ref: None,
    }).unwrap();

    let receipt = graph.retract_source("source").unwrap();
    assert_eq!(receipt.status, IntimateRetractionStatusV1::Applied);
    assert!(!graph.is_retrievable("source"));
    assert!(!graph.is_retrievable("summary"));
}

#[test]
fn external_export_never_turns_into_a_false_deletion_claim() {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    graph.insert(root("source", IntimateRetentionClassV1::DurableOptIn)).unwrap();
    graph.insert(IntimateMemoryArtifactV1 {
        artifact_id: "export".into(),
        kind: IntimateArtifactKindV1::ExternalExport,
        retention: IntimateRetentionClassV1::ExternalExport,
        dependency_state: IntimateDependencyStateV1::Complete,
        derivation_policy: IntimateDerivationPolicyV1::ExternalExportRequiresNotice,
        reconstructive: false,
        source_ids: BTreeSet::from(["source".into()]),
        durability_authorization_ref: None,
        export_receipt_ref: Some("export:receipt".into()),
    }).unwrap();

    let receipt = graph.retract_source("source").unwrap();
    assert!(receipt.external_deletion_unproven);
}

#[test]
fn ephemeral_to_durable_promotion_is_explicitly_authorized() {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    graph.insert(root("ephemeral", IntimateRetentionClassV1::EphemeralSession)).unwrap();
    let mut durable = IntimateMemoryArtifactV1 {
        artifact_id: "durable".into(),
        kind: IntimateArtifactKindV1::ReflectiveMemory,
        retention: IntimateRetentionClassV1::DurableOptIn,
        dependency_state: IntimateDependencyStateV1::Complete,
        derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
        reconstructive: false,
        source_ids: BTreeSet::from(["ephemeral".into()]),
        durability_authorization_ref: None,
        export_receipt_ref: None,
    };
    assert_eq!(
        graph.insert(durable.clone()),
        Err(IntimateMemoryPrivacyErrorV1::EphemeralToDurableRequiresAuthorization)
    );
    durable.durability_authorization_ref = Some("user-opt-in:receipt".into());
    assert!(graph.insert(durable).is_ok());
}
