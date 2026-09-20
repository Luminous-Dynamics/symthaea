// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimate_memory_privacy.rs"]
mod intimate_memory_privacy;
#[path = "../src/reflective_intimacy_memory.rs"]
mod reflective_intimacy_memory;

use intimate_memory_privacy::{
    IntimateArtifactKindV1, IntimateDependencyStateV1, IntimateDerivationPolicyV1,
    IntimateMemoryArtifactV1, IntimateMemoryPrivacyGraphV1, IntimateRetentionClassV1,
};
use reflective_intimacy_memory::*;
use std::collections::BTreeSet;

#[test]
fn durable_reflective_memory_is_retrievable_only_while_exact_privacy_lineage_is_live() {
    let mut privacy = IntimateMemoryPrivacyGraphV1::default();
    privacy
        .insert(
            IntimateMemoryArtifactV1::root(
                "explicit-source",
                IntimateArtifactKindV1::PsychologyEvidence,
                IntimateRetentionClassV1::DurableOptIn,
            )
            .unwrap(),
        )
        .unwrap();
    privacy
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: "preference-memory-privacy".into(),
            kind: IntimateArtifactKindV1::ExplicitPreference,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
            reconstructive: false,
            source_ids: BTreeSet::from(["explicit-source".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();

    let item = ReflectiveIntimacyMemoryItemV1 {
        memory_id: "memory-1".into(),
        privacy_artifact_id: "preference-memory-privacy".into(),
        content_ref: "local-vault:memory-1".into(),
        content_commitment: reflective_content_commitment_v1(b"opaque-test-content"),
        namespace: ReflectiveMemoryNamespaceV1::ExplicitPreference,
        reality: ReflectiveRealityNamespaceV1::RealWorld,
        perspective: ReflectiveMemoryPerspectiveV1::Participant,
        provenance: ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
        sensitivity: ReflectiveMemorySensitivityV1::Intimate,
        confidence: 1.0,
        retention: IntimateRetentionClassV1::DurableOptIn,
        source_artifact_ids: BTreeSet::from(["explicit-source".into()]),
        created_at_ns: 100,
        updated_at_ns: 100,
        supersedes: None,
    };

    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index.admit(item, &privacy).unwrap();
    assert!(index.current("memory-1", &privacy).is_some());

    privacy.retract_source("explicit-source").unwrap();
    assert!(index.current("memory-1", &privacy).is_none());
}
