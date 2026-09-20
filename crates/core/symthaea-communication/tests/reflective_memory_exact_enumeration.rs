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
#[path = "../src/reflective_memory_exact_enumeration.rs"]
mod reflective_memory_exact_enumeration;

use intimate_memory_privacy::*;
use reflective_memory_candidate_provenance::*;
use reflective_memory_exact_enumeration::*;
use reflective_memory_index_build::*;
use reflective_memory_retrieval::*;
use std::collections::{BTreeMap, BTreeSet};

fn digest(label: &str) -> String {
    format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
}

fn scope() -> ReflectiveMemoryRetrievalScopeV1 {
    ReflectiveMemoryRetrievalScopeV1 {
        reality: reflective_intimacy_memory::ReflectiveRealityNamespaceV1::RealWorld,
        allowed_namespaces: vec![
            reflective_intimacy_memory::ReflectiveMemoryNamespaceV1::ExplicitPreference,
        ],
        allowed_perspectives: vec![
            reflective_intimacy_memory::ReflectiveMemoryPerspectiveV1::Participant,
        ],
        max_sensitivity: reflective_intimacy_memory::ReflectiveMemorySensitivityV1::Intimate,
        minimum_confidence: Some(0.5),
        allowed_retentions: vec![IntimateRetentionClassV1::DurableOptIn],
        query_context_id: "query-1".into(),
        retrieval_policy_id: "retrieval-policy-1".into(),
    }
}

fn graph() -> IntimateMemoryPrivacyGraphV1 {
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

fn build(
    graph: &IntimateMemoryPrivacyGraphV1,
    enumeration_evidence: Vec<IndexEnumerationEvidenceV1>,
) -> ReflectiveMemoryIndexBuildReceiptV1 {
    ReflectiveMemoryIndexBuildReceiptV1::new(
        "build-1",
        "receipt:build-1",
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend-v1"),
        "embedding-model-v1",
        digest("embedding-model-v1"),
        "tokenizer-v1",
        digest("tokenizer-v1"),
        "nix:index-build-env",
        digest("build-env"),
        "monotonic-ns-v1",
        100,
        200,
        BTreeMap::from([("source-1".into(), digest("source-1-state"))]),
        vec![],
        enumeration_evidence,
        "artifact:snapshot-1",
        digest("snapshot-1"),
        graph,
    )
    .unwrap()
}

fn exact_source(
    graph: &IntimateMemoryPrivacyGraphV1,
    build: &ReflectiveMemoryIndexBuildReceiptV1,
    partition: &str,
    ids: Vec<String>,
) -> ReflectiveMemoryCandidateSourceReceiptV1 {
    ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-1",
        &scope(),
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend-v1"),
        build.snapshot_commitment.clone(),
        build.build_receipt_ref.clone(),
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref: partition.into(),
            expected_count: ids.len() as u64,
        },
        vec![],
        ids,
        graph,
    )
    .unwrap()
}

fn bounded_source(
    graph: &IntimateMemoryPrivacyGraphV1,
    build: &ReflectiveMemoryIndexBuildReceiptV1,
) -> ReflectiveMemoryCandidateSourceReceiptV1 {
    ReflectiveMemoryCandidateSourceReceiptV1::new(
        "candidate-source-bounded",
        &scope(),
        "index-1",
        "index-schema-v1",
        "local-vector-backend-v1",
        digest("backend-v1"),
        build.snapshot_commitment.clone(),
        build.build_receipt_ref.clone(),
        CandidateSetCompletenessV1::BoundedSearch {
            limit: 8,
            search_profile_ref: "profile:topk-8".into(),
        },
        vec![],
        vec!["memory-a".into()],
        graph,
    )
    .unwrap()
}

#[test]
fn exact_candidate_set_is_admitted_when_enumeration_commitment_matches() {
    let graph = graph();
    let partition = "partition:explicit-preferences";
    let ids = vec!["memory-a".into(), "memory-b".into()];
    let enumeration_result = exact_enumeration_result_commitment_v1(partition, &ids).unwrap();
    let build = build(
        &graph,
        vec![IndexEnumerationEvidenceV1 {
            partition_descriptor_ref: partition.into(),
            enumeration_protocol_id: "metadata-enumeration-v1".into(),
            enumeration_result_commitment: enumeration_result,
            enumerated_count: ids.len() as u64,
        }],
    );
    let source = exact_source(&graph, &build, partition, ids);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build, &graph).unwrap();
    let binding = validate_candidate_source_against_current_index_strict_v1(
        &source,
        &scope(),
        &registry,
        &graph,
    )
    .unwrap();
    assert!(binding.starts_with("reflective-memory-current-candidate-strict:"));
}

#[test]
fn same_count_different_candidate_set_is_rejected() {
    let graph = graph();
    let partition = "partition:explicit-preferences";
    let enumerated_ids = vec!["memory-a".into()];
    let enumeration_result =
        exact_enumeration_result_commitment_v1(partition, &enumerated_ids).unwrap();
    let build = build(
        &graph,
        vec![IndexEnumerationEvidenceV1 {
            partition_descriptor_ref: partition.into(),
            enumeration_protocol_id: "metadata-enumeration-v1".into(),
            enumeration_result_commitment: enumeration_result,
            enumerated_count: 1,
        }],
    );
    let source = exact_source(&graph, &build, partition, vec!["memory-b".into()]);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build, &graph).unwrap();
    assert_eq!(
        validate_candidate_source_against_current_index_strict_v1(
            &source,
            &scope(),
            &registry,
            &graph,
        ),
        Err(ReflectiveMemoryExactEnumerationErrorV1::EnumerationResultCommitmentMismatch)
    );
}

#[test]
fn exact_enumeration_commitment_is_order_canonical() {
    let a = exact_enumeration_result_commitment_v1(
        "partition:p",
        &["memory-b".into(), "memory-a".into()],
    )
    .unwrap();
    let b = exact_enumeration_result_commitment_v1(
        "partition:p",
        &["memory-a".into(), "memory-b".into()],
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn duplicate_candidate_ids_fail_closed() {
    assert_eq!(
        exact_enumeration_result_commitment_v1(
            "partition:p",
            &["memory-a".into(), "memory-a".into()],
        ),
        Err(ReflectiveMemoryExactEnumerationErrorV1::CandidateIdsNotUnique)
    );
}

#[test]
fn partition_descriptor_is_commitment_significant() {
    let ids = vec!["memory-a".into()];
    let a = exact_enumeration_result_commitment_v1("partition:a", &ids).unwrap();
    let b = exact_enumeration_result_commitment_v1("partition:b", &ids).unwrap();
    assert_ne!(a, b);
}

#[test]
fn bounded_search_stays_non_exact_and_needs_no_enumeration_receipt() {
    let graph = graph();
    let build = build(&graph, vec![]);
    let source = bounded_source(&graph, &build);
    let mut registry = CurrentReflectiveMemoryIndexRegistryV1::default();
    registry.register_initial(build, &graph).unwrap();
    assert!(validate_candidate_source_against_current_index_strict_v1(
        &source,
        &scope(),
        &registry,
        &graph,
    )
    .is_ok());
}
