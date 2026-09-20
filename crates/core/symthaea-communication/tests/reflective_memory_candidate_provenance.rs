// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimate_memory_privacy.rs"]
mod intimate_memory_privacy;
#[path = "../src/reflective_intimacy_memory.rs"]
mod reflective_intimacy_memory;
#[path = "../src/reflective_memory_retrieval.rs"]
mod reflective_memory_retrieval;
#[path = "../src/reflective_memory_candidate_provenance.rs"]
mod reflective_memory_candidate_provenance;

use intimate_memory_privacy::*;
use reflective_intimacy_memory::*;
use reflective_memory_candidate_provenance::*;
use reflective_memory_retrieval::*;
use std::collections::BTreeSet;

fn digest(label: &str) -> String {
    format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
}

fn scope() -> ReflectiveMemoryRetrievalScopeV1 {
    ReflectiveMemoryRetrievalScopeV1 {
        reality: ReflectiveRealityNamespaceV1::RealWorld,
        allowed_namespaces: vec![ReflectiveMemoryNamespaceV1::ExplicitPreference],
        allowed_perspectives: vec![ReflectiveMemoryPerspectiveV1::Participant],
        max_sensitivity: ReflectiveMemorySensitivityV1::Intimate,
        minimum_confidence: Some(0.5),
        allowed_retentions: vec![IntimateRetentionClassV1::DurableOptIn],
        query_context_id: "query-context-1".into(),
        retrieval_policy_id: "retrieval-policy-1".into(),
    }
}

fn graph_index_and_memory() -> (
    IntimateMemoryPrivacyGraphV1,
    ReflectiveIntimacyMemoryIndexV1,
) {
    let mut privacy = IntimateMemoryPrivacyGraphV1::default();
    privacy
        .insert(
            IntimateMemoryArtifactV1::root(
                "source-1",
                IntimateArtifactKindV1::PsychologyEvidence,
                IntimateRetentionClassV1::DurableOptIn,
            )
            .unwrap(),
        )
        .unwrap();
    privacy
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: "memory-privacy-1".into(),
            kind: IntimateArtifactKindV1::ExplicitPreference,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
            reconstructive: false,
            source_ids: BTreeSet::from(["source-1".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();
    privacy
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: "index-1".into(),
            kind: IntimateArtifactKindV1::EmbeddingOrIndex,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
            reconstructive: true,
            source_ids: BTreeSet::from(["memory-privacy-1".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();

    let item = ReflectiveIntimacyMemoryItemV1 {
        memory_id: "memory-1".into(),
        privacy_artifact_id: "memory-privacy-1".into(),
        content_ref: "local-vault:memory-1".into(),
        content_commitment: reflective_content_commitment_v1(b"opaque"),
        namespace: ReflectiveMemoryNamespaceV1::ExplicitPreference,
        reality: ReflectiveRealityNamespaceV1::RealWorld,
        perspective: ReflectiveMemoryPerspectiveV1::Participant,
        provenance: ReflectiveMemoryProvenanceV1::ExplicitParticipantStatement,
        sensitivity: ReflectiveMemorySensitivityV1::Intimate,
        confidence: 1.0,
        retention: IntimateRetentionClassV1::DurableOptIn,
        source_artifact_ids: BTreeSet::from(["source-1".into()]),
        created_at_ns: 1,
        updated_at_ns: 1,
        supersedes: None,
    };
    let mut index = ReflectiveIntimacyMemoryIndexV1::default();
    index.admit(item, &privacy).unwrap();
    (privacy, index)
}

fn candidate_source(
    privacy: &IntimateMemoryPrivacyGraphV1,
    retrieval_scope: &ReflectiveMemoryRetrievalScopeV1,
) -> ReflectiveMemoryCandidateSourceReceiptV1 {
    ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-1",
        retrieval_scope,
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend"),
        digest("snapshot"),
        "receipt:index-build-1",
        CandidateSetCompletenessV1::BoundedSearch {
            limit: 8,
            search_profile_ref: "profile:topk-8".into(),
        },
        vec![CandidateSearchParameterV1 {
            parameter_id: "top_k".into(),
            value: CandidateSearchParameterValueV1::Unsigned(8),
        }],
        vec!["memory-1".into()],
        privacy,
    )
    .unwrap()
}

#[test]
fn end_to_end_candidate_source_partition_and_retrieval_are_chained() {
    let (privacy, index) = graph_index_and_memory();
    let retrieval_scope = scope();
    let source = candidate_source(&privacy, &retrieval_scope);
    let partition =
        build_provenanced_partition_v1(&source, &retrieval_scope, &index, &privacy).unwrap();
    assert_eq!(partition.inner_partition.eligible_memory_ids(), &["memory-1"]);

    let receipt = admit_ranked_ids_with_candidate_source_v1(
        &source,
        &retrieval_scope,
        &partition,
        &["memory-1".into()],
        &index,
        &privacy,
    )
    .unwrap();
    assert_eq!(receipt.inner_retrieval_receipt.admitted_memory_ids, vec!["memory-1"]);
    assert!(receipt
        .receipt_commitment
        .starts_with("reflective-memory-candidate-retrieval:"));
}

#[test]
fn deleted_privacy_index_invalidates_candidate_source_before_ranking() {
    let (mut privacy, index) = graph_index_and_memory();
    let retrieval_scope = scope();
    let source = candidate_source(&privacy, &retrieval_scope);
    let partition =
        build_provenanced_partition_v1(&source, &retrieval_scope, &index, &privacy).unwrap();

    privacy.retract_source("source-1").unwrap();
    assert_eq!(
        admit_ranked_ids_with_candidate_source_v1(
            &source,
            &retrieval_scope,
            &partition,
            &["memory-1".into()],
            &index,
            &privacy,
        ),
        Err(CandidateProvenanceErrorV1::IndexArtifactUnavailable)
    );
}

#[test]
fn narrowed_index_lineage_invalidates_old_candidate_receipt() {
    let mut privacy = IntimateMemoryPrivacyGraphV1::default();
    for id in ["source-a", "source-b"] {
        privacy
            .insert(
                IntimateMemoryArtifactV1::root(
                    id,
                    IntimateArtifactKindV1::PsychologyEvidence,
                    IntimateRetentionClassV1::DurableOptIn,
                )
                .unwrap(),
            )
            .unwrap();
    }
    privacy
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: "index-1".into(),
            kind: IntimateArtifactKindV1::EmbeddingOrIndex,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MayRetainWithIndependentBasis,
            reconstructive: false,
            source_ids: BTreeSet::from(["source-a".into(), "source-b".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();

    let retrieval_scope = scope();
    let source = ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-1",
        &retrieval_scope,
        "index-1",
        "schema-v1",
        "backend-v1",
        digest("backend"),
        digest("snapshot"),
        "receipt:index-build",
        CandidateSetCompletenessV1::UnknownCompleteness {
            search_profile_ref: "profile:unknown".into(),
        },
        vec![],
        vec![],
        &privacy,
    )
    .unwrap();

    privacy.retract_source("source-a").unwrap();
    assert_eq!(
        source.validate_live(&retrieval_scope, &privacy),
        Err(CandidateProvenanceErrorV1::IndexSourceLineageMismatch)
    );
}

#[test]
fn query_or_policy_substitution_is_rejected() {
    let (privacy, _) = graph_index_and_memory();
    let retrieval_scope = scope();
    let source = candidate_source(&privacy, &retrieval_scope);
    let mut other = retrieval_scope.clone();
    other.query_context_id = "different-query".into();
    assert_eq!(
        source.validate_live(&other, &privacy),
        Err(CandidateProvenanceErrorV1::ScopeIdentityMismatch)
    );
}

#[test]
fn exact_enumeration_cannot_lie_about_candidate_count() {
    let (privacy, _) = graph_index_and_memory();
    let retrieval_scope = scope();
    let result = ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-exact",
        &retrieval_scope,
        "index-1",
        "schema-v1",
        "backend-v1",
        digest("backend"),
        digest("snapshot"),
        "receipt:index-build",
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref: "partition:real-explicit-preference".into(),
            expected_count: 2,
        },
        vec![],
        vec!["memory-1".into()],
        &privacy,
    );
    assert_eq!(
        result,
        Err(CandidateProvenanceErrorV1::ExactEnumerationCountMismatch)
    );
}

#[test]
fn bounded_search_remains_non_exhaustive() {
    let (privacy, _) = graph_index_and_memory();
    let retrieval_scope = scope();
    let source = candidate_source(&privacy, &retrieval_scope);
    assert!(matches!(
        source.completeness,
        CandidateSetCompletenessV1::BoundedSearch { limit: 8, .. }
    ));
}

#[test]
fn candidate_and_parameter_order_are_canonicalized() {
    let (privacy, _) = graph_index_and_memory();
    let retrieval_scope = scope();
    let make = |parameters: Vec<CandidateSearchParameterV1>, ids: Vec<String>| {
        ReflectiveMemoryCandidateSourceReceiptV1::new(
            "candidate-source-order",
            &retrieval_scope,
            "index-1",
            "schema-v1",
            "backend-v1",
            digest("backend"),
            digest("snapshot"),
            "receipt:index-build",
            CandidateSetCompletenessV1::BoundedSearch {
                limit: 8,
                search_profile_ref: "profile:topk".into(),
            },
            parameters,
            ids,
            &privacy,
        )
        .unwrap()
    };
    let a = make(
        vec![
            CandidateSearchParameterV1 {
                parameter_id: "top_k".into(),
                value: CandidateSearchParameterValueV1::Unsigned(8),
            },
            CandidateSearchParameterV1 {
                parameter_id: "normalized".into(),
                value: CandidateSearchParameterValueV1::Bool(true),
            },
        ],
        vec!["memory-z".into(), "memory-a".into()],
    );
    let b = make(
        vec![
            CandidateSearchParameterV1 {
                parameter_id: "normalized".into(),
                value: CandidateSearchParameterValueV1::Bool(true),
            },
            CandidateSearchParameterV1 {
                parameter_id: "top_k".into(),
                value: CandidateSearchParameterValueV1::Unsigned(8),
            },
        ],
        vec!["memory-a".into(), "memory-z".into()],
    );
    assert_eq!(a.receipt_commitment, b.receipt_commitment);
}

#[test]
fn mutating_search_parameters_invalidates_receipt_commitment() {
    let (privacy, _) = graph_index_and_memory();
    let retrieval_scope = scope();
    let mut source = candidate_source(&privacy, &retrieval_scope);
    source.search_parameters[0].value = CandidateSearchParameterValueV1::Unsigned(7);
    assert_eq!(
        source.validate_live(&retrieval_scope, &privacy),
        Err(CandidateProvenanceErrorV1::ReceiptCommitmentMismatch)
    );
}
