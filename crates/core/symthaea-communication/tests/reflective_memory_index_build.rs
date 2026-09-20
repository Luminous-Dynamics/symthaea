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
#[path = "../src/reflective_memory_index_build.rs"]
mod reflective_memory_index_build;

use intimate_memory_privacy::*;
use reflective_memory_candidate_provenance::*;
use reflective_memory_index_build::*;
use reflective_memory_retrieval::*;
use std::collections::{BTreeMap, BTreeSet};

fn digest(label: &str) -> String {
    format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
}

fn scope() -> ReflectiveMemoryRetrievalScopeV1 {
    ReflectiveMemoryRetrievalScopeV1 {
        reality: reflective_intimacy_memory::ReflectiveRealityNamespaceV1::RealWorld,
        allowed_namespaces: vec![reflective_intimacy_memory::ReflectiveMemoryNamespaceV1::ExplicitPreference],
        allowed_perspectives: vec![reflective_intimacy_memory::ReflectiveMemoryPerspectiveV1::Participant],
        max_sensitivity: reflective_intimacy_memory::ReflectiveMemorySensitivityV1::Intimate,
        minimum_confidence: Some(0.5),
        allowed_retentions: vec![IntimateRetentionClassV1::DurableOptIn],
        query_context_id: "query-1".into(),
        retrieval_policy_id: "policy-1".into(),
    }
}

fn simple_privacy_graph() -> IntimateMemoryPrivacyGraphV1 {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    graph
        .insert(
            IntimateMemoryArtifactV1::root(
                "source-1",
                IntimateArtifactKindV1::ExplicitPreference,
                IntimateRetentionClassV1::DurableOptIn,
            )
            .unwrap(),
        )
        .unwrap();
    graph
        .insert(IntimateMemoryArtifactV1 {
            artifact_id: "index-1".into(),
            kind: IntimateArtifactKindV1::EmbeddingOrIndex,
            retention: IntimateRetentionClassV1::DurableOptIn,
            dependency_state: IntimateDependencyStateV1::Complete,
            derivation_policy: IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction,
            reconstructive: true,
            source_ids: BTreeSet::from(["source-1".into()]),
            durability_authorization_ref: None,
            export_receipt_ref: None,
        })
        .unwrap();
    graph
}

fn state_commitments(ids: &[&str]) -> BTreeMap<String, String> {
    ids.iter()
        .map(|id| ((*id).to_owned(), digest(&format!("state:{id}"))))
        .collect()
}

fn build_receipt(
    graph: &IntimateMemoryPrivacyGraphV1,
    build_id: &str,
    receipt_ref: &str,
    snapshot_label: &str,
    source_ids: &[&str],
    enumeration: Vec<IndexEnumerationEvidenceV1>,
) -> ReflectiveMemoryIndexBuildReceiptV1 {
    ReflectiveMemoryIndexBuildReceiptV1::new(
        build_id,
        receipt_ref,
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend-v1"),
        "embedding-model-v1",
        digest("embedding-model-v1"),
        "tokenizer-v1",
        digest("tokenizer-v1"),
        "nix:index-build-env",
        digest("index-build-env"),
        "monotonic-ns-v1",
        100,
        200,
        state_commitments(source_ids),
        vec![IndexBuildParameterV1 {
            parameter_id: "dimensions".into(),
            value: IndexBuildParameterValueV1::Unsigned(1024),
        }],
        enumeration,
        format!("artifact:{snapshot_label}"),
        digest(snapshot_label),
        graph,
    )
    .unwrap()
}

fn candidate_source(
    graph: &IntimateMemoryPrivacyGraphV1,
    snapshot: String,
    build_ref: &str,
    completeness: CandidateSetCompletenessV1,
) -> ReflectiveMemoryCandidateSourceReceiptV1 {
    ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-1",
        &scope(),
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend-v1"),
        snapshot,
        build_ref,
        completeness,
        vec![CandidateSearchParameterV1 {
            parameter_id: "top_k".into(),
            value: CandidateSearchParameterValueV1::Unsigned(8),
        }],
        vec![],
        graph,
    )
    .unwrap()
}

#[test]
fn deterministic_build_receipt_and_current_candidate_binding() {
    let graph = simple_privacy_graph();
    let a = build_receipt(&graph, "build-1", "receipt:build-1", "snapshot-1", &["source-1"], vec![]);
    let b = build_receipt(&graph, "build-1", "receipt:build-1", "snapshot-1", &["source-1"], vec![]);
    assert_eq!(a.receipt_commitment, b.receipt_commitment);

    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(a.clone(), &graph).unwrap();
    let source = candidate_source(
        &graph,
        a.snapshot_commitment.clone(),
        &a.build_receipt_ref,
        CandidateSetCompletenessV1::BoundedSearch {
            limit: 8,
            search_profile_ref: "profile:topk-8".into(),
        },
    );
    let binding = validate_candidate_source_against_current_index_v1(
        &source,
        &scope(),
        &registry,
        &graph,
    )
    .unwrap();
    assert!(binding.starts_with("reflective-index-current-candidate:"));
}

#[test]
fn snapshot_substitution_is_rejected() {
    let graph = simple_privacy_graph();
    let build = build_receipt(&graph, "build-1", "receipt:build-1", "snapshot-1", &["source-1"], vec![]);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build.clone(), &graph).unwrap();
    let source = candidate_source(
        &graph,
        digest("other-snapshot"),
        &build.build_receipt_ref,
        CandidateSetCompletenessV1::BoundedSearch {
            limit: 8,
            search_profile_ref: "profile:topk-8".into(),
        },
    );
    assert_eq!(
        validate_candidate_source_against_current_index_v1(&source, &scope(), &registry, &graph),
        Err(ReflectiveMemoryIndexBuildErrorV1::CandidateDoesNotMatchCurrentSnapshot)
    );
}

#[test]
fn privacy_retraction_immediately_stales_old_snapshot() {
    let mut graph = simple_privacy_graph();
    let build = build_receipt(&graph, "build-1", "receipt:build-1", "snapshot-1", &["source-1"], vec![]);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build, &graph).unwrap();
    graph.retract_source("source-1").unwrap();
    assert_eq!(
        registry.current_validated("index-1", &graph),
        Err(ReflectiveMemoryIndexBuildErrorV1::IndexArtifactUnavailable)
    );
}

#[test]
fn rebuild_after_lineage_narrowing_requires_fresh_snapshot() {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    for id in ["source-a", "source-b"] {
        graph
            .insert(
                IntimateMemoryArtifactV1::root(
                    id,
                    IntimateArtifactKindV1::ExplicitPreference,
                    IntimateRetentionClassV1::DurableOptIn,
                )
                .unwrap(),
            )
            .unwrap();
    }
    graph
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

    let old = build_receipt(
        &graph,
        "build-old",
        "receipt:build-old",
        "snapshot-old",
        &["source-a", "source-b"],
        vec![],
    );
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(old.clone(), &graph).unwrap();
    graph.retract_source("source-a").unwrap();
    assert_eq!(
        old.validate_live(&graph),
        Err(ReflectiveMemoryIndexBuildErrorV1::IndexSourceLineageMismatch)
    );

    let rebuilt = build_receipt(
        &graph,
        "build-new",
        "receipt:build-new",
        "snapshot-new",
        &["source-b"],
        vec![],
    );
    registry.advance(rebuilt.clone(), &graph).unwrap();
    assert_eq!(
        registry.current_validated("index-1", &graph).unwrap().snapshot_commitment,
        rebuilt.snapshot_commitment
    );
}

#[test]
fn old_candidate_receipt_cannot_authorize_after_registry_advances() {
    let mut graph = IntimateMemoryPrivacyGraphV1::default();
    for id in ["source-a", "source-b"] {
        graph
            .insert(
                IntimateMemoryArtifactV1::root(
                    id,
                    IntimateArtifactKindV1::ExplicitPreference,
                    IntimateRetentionClassV1::DurableOptIn,
                )
                .unwrap(),
            )
            .unwrap();
    }
    graph
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
    let old = build_receipt(
        &graph,
        "build-old",
        "receipt:build-old",
        "snapshot-old",
        &["source-a", "source-b"],
        vec![],
    );
    let old_source = candidate_source(
        &graph,
        old.snapshot_commitment.clone(),
        &old.build_receipt_ref,
        CandidateSetCompletenessV1::BoundedSearch {
            limit: 8,
            search_profile_ref: "profile:topk-8".into(),
        },
    );
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(old, &graph).unwrap();
    graph.retract_source("source-a").unwrap();
    let rebuilt = build_receipt(
        &graph,
        "build-new",
        "receipt:build-new",
        "snapshot-new",
        &["source-b"],
        vec![],
    );
    registry.advance(rebuilt, &graph).unwrap();
    assert!(validate_candidate_source_against_current_index_v1(
        &old_source,
        &scope(),
        &registry,
        &graph,
    )
    .is_err());
}

#[test]
fn exact_enumeration_requires_index_layer_evidence() {
    let graph = simple_privacy_graph();
    let build = build_receipt(&graph, "build-1", "receipt:build-1", "snapshot-1", &["source-1"], vec![]);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build.clone(), &graph).unwrap();
    let source = candidate_source(
        &graph,
        build.snapshot_commitment.clone(),
        &build.build_receipt_ref,
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref: "partition:explicit-preferences".into(),
            expected_count: 0,
        },
    );
    assert_eq!(
        validate_candidate_source_against_current_index_v1(&source, &scope(), &registry, &graph),
        Err(ReflectiveMemoryIndexBuildErrorV1::MissingExactEnumerationEvidence)
    );
}

#[test]
fn exact_enumeration_is_admitted_when_matching_evidence_exists() {
    let graph = simple_privacy_graph();
    let build = build_receipt(
        &graph,
        "build-1",
        "receipt:build-1",
        "snapshot-1",
        &["source-1"],
        vec![IndexEnumerationEvidenceV1 {
            partition_descriptor_ref: "partition:explicit-preferences".into(),
            enumeration_protocol_id: "metadata-partition-enumeration-v1".into(),
            enumeration_result_commitment: digest("enumeration-result"),
            enumerated_count: 0,
        }],
    );
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build.clone(), &graph).unwrap();
    let source = candidate_source(
        &graph,
        build.snapshot_commitment.clone(),
        &build.build_receipt_ref,
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref: "partition:explicit-preferences".into(),
            expected_count: 0,
        },
    );
    assert!(validate_candidate_source_against_current_index_v1(
        &source,
        &scope(),
        &registry,
        &graph,
    )
    .is_ok());
}
