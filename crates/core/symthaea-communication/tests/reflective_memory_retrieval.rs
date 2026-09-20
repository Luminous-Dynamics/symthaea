// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimate_memory_privacy.rs"]
mod intimate_memory_privacy;
#[path = "../src/reflective_intimacy_memory.rs"]
mod reflective_intimacy_memory;
#[path = "../src/reflective_memory_retrieval.rs"]
mod reflective_memory_retrieval;

use intimate_memory_privacy::{
    IntimateArtifactKindV1, IntimateDependencyStateV1, IntimateDerivationPolicyV1,
    IntimateMemoryArtifactV1, IntimateMemoryPrivacyGraphV1, IntimateRetentionClassV1,
};
use reflective_intimacy_memory::*;
use reflective_memory_retrieval::*;
use std::collections::BTreeSet;

fn base_privacy() -> IntimateMemoryPrivacyGraphV1 {
    let mut privacy = IntimateMemoryPrivacyGraphV1::default();
    privacy
        .insert(
            IntimateMemoryArtifactV1::root(
                "source",
                IntimateArtifactKindV1::PsychologyEvidence,
                IntimateRetentionClassV1::DurableOptIn,
            )
            .unwrap(),
        )
        .unwrap();
    privacy
}

fn add_privacy_artifact(
    privacy: &mut IntimateMemoryPrivacyGraphV1,
    artifact_id: &str,
    kind: IntimateArtifactKindV1,
) {
    privacy
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: artifact_id.into(),
            kind,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
            reconstructive: false,
            source_ids: BTreeSet::from(["source".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();
}

fn memory(
    memory_id: &str,
    privacy_artifact_id: &str,
    namespace: ReflectiveMemoryNamespaceV1,
    reality: ReflectiveRealityNamespaceV1,
    perspective: ReflectiveMemoryPerspectiveV1,
    provenance: ReflectiveMemoryProvenanceV1,
    sensitivity: ReflectiveMemorySensitivityV1,
    supersedes: Option<&str>,
) -> ReflectiveIntimacyMemoryItemV1 {
    ReflectiveIntimacyMemoryItemV1 {
        memory_id: memory_id.into(),
        privacy_artifact_id: privacy_artifact_id.into(),
        content_ref: format!("local-vault:{memory_id}"),
        content_commitment: reflective_content_commitment_v1(memory_id.as_bytes()),
        namespace,
        reality,
        perspective,
        provenance,
        sensitivity,
        confidence: 0.9,
        retention: IntimateRetentionClassV1::DurableOptIn,
        source_artifact_ids: BTreeSet::from(["source".into()]),
        created_at_ns: 100,
        updated_at_ns: if supersedes.is_some() { 200 } else { 100 },
        supersedes: supersedes.map(str::to_owned),
    }
}

fn real_preference_scope(max_sensitivity: ReflectiveMemorySensitivityV1) -> ReflectiveMemoryRetrievalScopeV1 {
    ReflectiveMemoryRetrievalScopeV1 {
        reality: ReflectiveRealityNamespaceV1::RealWorld,
        allowed_namespaces: vec![ReflectiveMemoryNamespaceV1::ExplicitPreference],
        allowed_perspectives: vec![ReflectiveMemoryPerspectiveV1::Participant],
        max_sensitivity,
        minimum_confidence: None,
        allowed_retentions: vec![IntimateRetentionClassV1::DurableOptIn],
        query_context_id: "query-1".into(),
        retrieval_policy_id: "retrieval-policy-v1".into(),
    }
}

#[test]
fn cross_reality_candidate_is_excluded_before_ranking() {
    let mut privacy = base_privacy();
    add_privacy_artifact(
        &mut privacy,
        "real-privacy",
        IntimateArtifactKindV1::ExplicitPreference,
    );
    add_privacy_artifact(
        &mut privacy,
        "fantasy-privacy",
        IntimateArtifactKindV1::FantasyHistory,
    );

    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index
        .admit(
            memory(
                "real-memory",
                "real-privacy",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();
    index
        .admit(
            memory(
                "fantasy-memory",
                "fantasy-privacy",
                ReflectiveMemoryNamespaceV1::FantasyWorld,
                ReflectiveRealityNamespaceV1::Fantasy {
                    world_id: "world-a".into(),
                },
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();

    let scope = real_preference_scope(ReflectiveMemorySensitivityV1::Intimate);
    let candidates = vec!["fantasy-memory".into(), "real-memory".into()];
    let partition = build_pre_rank_partition_v1(&scope, &candidates, &index, &privacy).unwrap();
    assert_eq!(partition.eligible_memory_ids(), &["real-memory".to_string()]);
    assert_eq!(
        admit_ranked_ids_v1(
            &scope,
            &partition,
            &["fantasy-memory".into()],
            &index,
            &privacy,
        ),
        Err(ReflectiveMemoryRetrievalErrorV1::RankedCandidateOutsidePartition)
    );
}

#[test]
fn sensitivity_and_perspective_are_eligibility_not_ranking_features() {
    let mut privacy = base_privacy();
    add_privacy_artifact(
        &mut privacy,
        "intimate-privacy",
        IntimateArtifactKindV1::ExplicitPreference,
    );
    add_privacy_artifact(
        &mut privacy,
        "symthaea-privacy",
        IntimateArtifactKindV1::ReflectiveMemory,
    );

    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index
        .admit(
            memory(
                "intimate-memory",
                "intimate-privacy",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Intimate,
                None,
            ),
            &privacy,
        )
        .unwrap();
    index
        .admit(
            memory(
                "symthaea-memory",
                "symthaea-privacy",
                ReflectiveMemoryNamespaceV1::SymthaeaPersona,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Symthaea,
                ReflectiveMemoryProvenanceV1::SymthaeaAuthored,
                ReflectiveMemorySensitivityV1::Ordinary,
                None,
            ),
            &privacy,
        )
        .unwrap();

    let personal_scope = real_preference_scope(ReflectiveMemorySensitivityV1::Personal);
    let intimate_partition = build_pre_rank_partition_v1(
        &personal_scope,
        &["intimate-memory".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert!(intimate_partition.is_empty());

    let participant_scope = ReflectiveMemoryRetrievalScopeV1 {
        allowed_namespaces: vec![ReflectiveMemoryNamespaceV1::SymthaeaPersona],
        ..real_preference_scope(ReflectiveMemorySensitivityV1::Intimate)
    };
    let perspective_partition = build_pre_rank_partition_v1(
        &participant_scope,
        &["symthaea-memory".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert!(perspective_partition.is_empty());
}

#[test]
fn stale_partition_cannot_resurrect_memory_after_privacy_retraction() {
    let mut privacy = base_privacy();
    add_privacy_artifact(
        &mut privacy,
        "memory-privacy",
        IntimateArtifactKindV1::ExplicitPreference,
    );
    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index
        .admit(
            memory(
                "memory-1",
                "memory-privacy",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();

    let scope = real_preference_scope(ReflectiveMemorySensitivityV1::Intimate);
    let partition = build_pre_rank_partition_v1(
        &scope,
        &["memory-1".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert_eq!(partition.eligible_memory_ids(), &["memory-1".to_string()]);

    privacy.retract_source("source").unwrap();
    assert_eq!(
        admit_ranked_ids_v1(
            &scope,
            &partition,
            &["memory-1".into()],
            &index,
            &privacy,
        ),
        Err(ReflectiveMemoryRetrievalErrorV1::StalePartition)
    );
}

#[test]
fn superseded_memory_is_not_a_current_rankable_candidate() {
    let mut privacy = base_privacy();
    add_privacy_artifact(
        &mut privacy,
        "memory-privacy",
        IntimateArtifactKindV1::ExplicitPreference,
    );
    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index
        .admit(
            memory(
                "old-memory",
                "memory-privacy",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();
    index
        .admit(
            memory(
                "new-memory",
                "memory-privacy",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                Some("old-memory"),
            ),
            &privacy,
        )
        .unwrap();

    let scope = real_preference_scope(ReflectiveMemorySensitivityV1::Intimate);
    let partition = build_pre_rank_partition_v1(
        &scope,
        &["old-memory".into(), "new-memory".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert_eq!(partition.eligible_memory_ids(), &["new-memory".to_string()]);
}

#[test]
fn empty_partition_can_issue_empty_receipt_without_widening_scope() {
    let privacy = base_privacy();
    let index = ReflectiveIntimacyMemoryIndexV1::default();
    let scope = real_preference_scope(ReflectiveMemorySensitivityV1::Personal);
    let partition = build_pre_rank_partition_v1(
        &scope,
        &["stale-backend-id".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert!(partition.is_empty());
    let receipt = admit_ranked_ids_v1(&scope, &partition, &[], &index, &privacy).unwrap();
    assert!(receipt.admitted_memory_ids.is_empty());
}

#[test]
fn candidate_and_scope_list_order_do_not_change_eligibility_commitments() {
    let mut privacy = base_privacy();
    add_privacy_artifact(
        &mut privacy,
        "privacy-a",
        IntimateArtifactKindV1::ExplicitPreference,
    );
    add_privacy_artifact(
        &mut privacy,
        "privacy-b",
        IntimateArtifactKindV1::ReflectiveMemory,
    );
    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index
        .admit(
            memory(
                "memory-a",
                "privacy-a",
                ReflectiveMemoryNamespaceV1::ExplicitPreference,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();
    index
        .admit(
            memory(
                "memory-b",
                "privacy-b",
                ReflectiveMemoryNamespaceV1::ParticipantFact,
                ReflectiveRealityNamespaceV1::RealWorld,
                ReflectiveMemoryPerspectiveV1::Participant,
                ReflectiveMemoryProvenanceV1::ValidatedSelfReport,
                ReflectiveMemorySensitivityV1::Personal,
                None,
            ),
            &privacy,
        )
        .unwrap();

    let mut scope_a = real_preference_scope(ReflectiveMemorySensitivityV1::Intimate);
    scope_a.allowed_namespaces = vec![
        ReflectiveMemoryNamespaceV1::ExplicitPreference,
        ReflectiveMemoryNamespaceV1::ParticipantFact,
    ];
    let mut scope_b = scope_a.clone();
    scope_b.allowed_namespaces.reverse();
    assert_eq!(
        scope_a.scope_commitment_v1().unwrap(),
        scope_b.scope_commitment_v1().unwrap()
    );

    let partition_a = build_pre_rank_partition_v1(
        &scope_a,
        &["memory-b".into(), "memory-a".into()],
        &index,
        &privacy,
    )
    .unwrap();
    let partition_b = build_pre_rank_partition_v1(
        &scope_b,
        &["memory-a".into(), "memory-b".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert_eq!(
        partition_a.partition_commitment(),
        partition_b.partition_commitment()
    );
    assert_eq!(
        partition_a.eligible_memory_ids(),
        partition_b.eligible_memory_ids()
    );
}
