// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reality-, privacy-, perspective-, and sensitivity-scoped retrieval firewall.
//!
//! Semantic similarity is never retrieval authority. Candidate IDs are first
//! resolved through the live reflective-memory/privacy graph and partitioned by
//! explicit metadata eligibility. Ranking is allowed only inside that partition.

use crate::intimate_memory_privacy::{
    IntimateMemoryPrivacyGraphV1, IntimateRetentionClassV1,
};
use crate::reflective_intimacy_memory::{
    ReflectiveIntimacyMemoryIndexV1, ReflectiveIntimacyMemoryItemV1,
    ReflectiveMemoryNamespaceV1, ReflectiveMemoryPerspectiveV1,
    ReflectiveMemorySensitivityV1, ReflectiveRealityNamespaceV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REFLECTIVE_MEMORY_RETRIEVAL_SCHEMA_V1: &str =
    "symthaea.communication.reflective-memory-retrieval.v1";
const SCOPE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea:reflective-memory-retrieval-scope:v1\0";
const PARTITION_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea:reflective-memory-retrieval-partition:v1\0";
const RECEIPT_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea:reflective-memory-retrieval-receipt:v1\0";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReflectiveMemoryRetrievalScopeV1 {
    pub reality: ReflectiveRealityNamespaceV1,
    pub allowed_namespaces: Vec<ReflectiveMemoryNamespaceV1>,
    pub allowed_perspectives: Vec<ReflectiveMemoryPerspectiveV1>,
    pub max_sensitivity: ReflectiveMemorySensitivityV1,
    pub minimum_confidence: Option<f32>,
    pub allowed_retentions: Vec<IntimateRetentionClassV1>,
    pub query_context_id: String,
    pub retrieval_policy_id: String,
}

impl ReflectiveMemoryRetrievalScopeV1 {
    pub fn validate(&self) -> Result<(), ReflectiveMemoryRetrievalErrorV1> {
        validate_id(
            &self.query_context_id,
            ReflectiveMemoryRetrievalErrorV1::InvalidQueryContextId,
        )?;
        validate_id(
            &self.retrieval_policy_id,
            ReflectiveMemoryRetrievalErrorV1::InvalidRetrievalPolicyId,
        )?;
        if let ReflectiveRealityNamespaceV1::Fantasy { world_id } = &self.reality {
            validate_id(
                world_id,
                ReflectiveMemoryRetrievalErrorV1::InvalidFantasyWorldId,
            )?;
        }
        if self.allowed_namespaces.is_empty() {
            return Err(ReflectiveMemoryRetrievalErrorV1::EmptyNamespaceSet);
        }
        if self.allowed_perspectives.is_empty() {
            return Err(ReflectiveMemoryRetrievalErrorV1::EmptyPerspectiveSet);
        }
        if self.allowed_retentions.is_empty() {
            return Err(ReflectiveMemoryRetrievalErrorV1::EmptyRetentionSet);
        }
        ensure_unique_tags(
            self.allowed_namespaces.iter().copied().map(namespace_tag),
            ReflectiveMemoryRetrievalErrorV1::DuplicateNamespace,
        )?;
        ensure_unique_tags(
            self.allowed_perspectives.iter().copied().map(perspective_tag),
            ReflectiveMemoryRetrievalErrorV1::DuplicatePerspective,
        )?;
        ensure_unique_tags(
            self.allowed_retentions.iter().copied().map(retention_tag),
            ReflectiveMemoryRetrievalErrorV1::DuplicateRetention,
        )?;
        if self
            .allowed_retentions
            .contains(&IntimateRetentionClassV1::ExternalExport)
        {
            return Err(ReflectiveMemoryRetrievalErrorV1::ExternalExportRetentionForbidden);
        }
        if let Some(confidence) = self.minimum_confidence {
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(ReflectiveMemoryRetrievalErrorV1::InvalidMinimumConfidence);
            }
        }
        Ok(())
    }

    pub fn scope_commitment_v1(&self) -> Result<String, ReflectiveMemoryRetrievalErrorV1> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SCOPE_COMMITMENT_DOMAIN_V1);
        hash_str(&mut hasher, REFLECTIVE_MEMORY_RETRIEVAL_SCHEMA_V1);
        commit_reality(&mut hasher, &self.reality);
        commit_sorted_tags(
            &mut hasher,
            self.allowed_namespaces.iter().copied().map(namespace_tag),
        );
        commit_sorted_tags(
            &mut hasher,
            self.allowed_perspectives.iter().copied().map(perspective_tag),
        );
        hasher.update(&[sensitivity_tag(self.max_sensitivity)]);
        match self.minimum_confidence {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        commit_sorted_tags(
            &mut hasher,
            self.allowed_retentions.iter().copied().map(retention_tag),
        );
        hash_str(&mut hasher, &self.query_context_id);
        hash_str(&mut hasher, &self.retrieval_policy_id);
        Ok(format!(
            "reflective-memory-retrieval-scope:{}",
            hasher.finalize().to_hex()
        ))
    }

    pub fn metadata_eligible(&self, item: &ReflectiveIntimacyMemoryItemV1) -> bool {
        if self.reality != item.reality
            || !self.allowed_namespaces.contains(&item.namespace)
            || !self.allowed_perspectives.contains(&item.perspective)
            || sensitivity_tag(item.sensitivity) > sensitivity_tag(self.max_sensitivity)
            || !self.allowed_retentions.contains(&item.retention)
        {
            return false;
        }
        self.minimum_confidence
            .map(|minimum| item.confidence >= minimum)
            .unwrap_or(true)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectiveMemoryRetrievalPartitionV1 {
    scope_commitment: String,
    partition_commitment: String,
    source_candidate_ids: Vec<String>,
    eligible_memory_ids: Vec<String>,
}

impl ReflectiveMemoryRetrievalPartitionV1 {
    pub fn scope_commitment(&self) -> &str {
        &self.scope_commitment
    }

    pub fn partition_commitment(&self) -> &str {
        &self.partition_commitment
    }

    /// IDs that may be passed to a semantic/vector ranker.
    pub fn eligible_memory_ids(&self) -> &[String] {
        &self.eligible_memory_ids
    }

    pub fn is_empty(&self) -> bool {
        self.eligible_memory_ids.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectiveMemoryRetrievalReceiptV1 {
    pub query_context_id: String,
    pub retrieval_policy_id: String,
    pub scope_commitment: String,
    pub partition_commitment: String,
    pub admitted_memory_ids: Vec<String>,
    pub receipt_commitment: String,
}

pub fn build_pre_rank_partition_v1(
    scope: &ReflectiveMemoryRetrievalScopeV1,
    candidate_memory_ids: &[String],
    index: &ReflectiveIntimacyMemoryIndexV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<ReflectiveMemoryRetrievalPartitionV1, ReflectiveMemoryRetrievalErrorV1> {
    scope.validate()?;
    let scope_commitment = scope.scope_commitment_v1()?;

    let mut source_candidate_ids = candidate_memory_ids.to_vec();
    for memory_id in &source_candidate_ids {
        validate_id(
            memory_id,
            ReflectiveMemoryRetrievalErrorV1::InvalidCandidateMemoryId,
        )?;
    }
    source_candidate_ids.sort();
    if source_candidate_ids
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err(ReflectiveMemoryRetrievalErrorV1::DuplicateCandidateMemoryId);
    }

    let mut eligible_memory_ids = Vec::new();
    for memory_id in &source_candidate_ids {
        let Some(item) = index.current(memory_id, privacy) else {
            continue;
        };
        if scope.metadata_eligible(item) {
            eligible_memory_ids.push(memory_id.clone());
        }
    }

    let partition_commitment = partition_commitment_v1(
        &scope_commitment,
        &source_candidate_ids,
        &eligible_memory_ids,
    );
    Ok(ReflectiveMemoryRetrievalPartitionV1 {
        scope_commitment,
        partition_commitment,
        source_candidate_ids,
        eligible_memory_ids,
    })
}

/// Recompute live eligibility before admitting any ranked IDs.
///
/// A vector/semantic backend may only rank `partition.eligible_memory_ids()`.
/// If privacy, supersession, source lineage, or scope eligibility changes after
/// partition construction, this method returns `StalePartition` rather than
/// allowing the stale ranker output to resurrect a memory.
pub fn admit_ranked_ids_v1(
    scope: &ReflectiveMemoryRetrievalScopeV1,
    partition: &ReflectiveMemoryRetrievalPartitionV1,
    ranked_memory_ids: &[String],
    index: &ReflectiveIntimacyMemoryIndexV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<ReflectiveMemoryRetrievalReceiptV1, ReflectiveMemoryRetrievalErrorV1> {
    scope.validate()?;
    let expected_scope = scope.scope_commitment_v1()?;
    if partition.scope_commitment != expected_scope {
        return Err(ReflectiveMemoryRetrievalErrorV1::ScopeCommitmentMismatch);
    }

    let fresh = build_pre_rank_partition_v1(
        scope,
        &partition.source_candidate_ids,
        index,
        privacy,
    )?;
    if fresh.partition_commitment != partition.partition_commitment
        || fresh.eligible_memory_ids != partition.eligible_memory_ids
    {
        return Err(ReflectiveMemoryRetrievalErrorV1::StalePartition);
    }

    let eligible: BTreeSet<&str> = fresh
        .eligible_memory_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut seen = BTreeSet::new();
    for memory_id in ranked_memory_ids {
        validate_id(
            memory_id,
            ReflectiveMemoryRetrievalErrorV1::InvalidRankedMemoryId,
        )?;
        if !seen.insert(memory_id.as_str()) {
            return Err(ReflectiveMemoryRetrievalErrorV1::DuplicateRankedMemoryId);
        }
        if !eligible.contains(memory_id.as_str()) {
            return Err(ReflectiveMemoryRetrievalErrorV1::RankedCandidateOutsidePartition);
        }
    }

    let receipt_commitment = retrieval_receipt_commitment_v1(
        &expected_scope,
        &fresh.partition_commitment,
        ranked_memory_ids,
    );
    Ok(ReflectiveMemoryRetrievalReceiptV1 {
        query_context_id: scope.query_context_id.clone(),
        retrieval_policy_id: scope.retrieval_policy_id.clone(),
        scope_commitment: expected_scope,
        partition_commitment: fresh.partition_commitment,
        admitted_memory_ids: ranked_memory_ids.to_vec(),
        receipt_commitment,
    })
}

fn partition_commitment_v1(
    scope_commitment: &str,
    source_candidate_ids: &[String],
    eligible_memory_ids: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PARTITION_COMMITMENT_DOMAIN_V1);
    hash_str(&mut hasher, scope_commitment);
    commit_string_slice(&mut hasher, source_candidate_ids);
    commit_string_slice(&mut hasher, eligible_memory_ids);
    format!(
        "reflective-memory-retrieval-partition:{}",
        hasher.finalize().to_hex()
    )
}

fn retrieval_receipt_commitment_v1(
    scope_commitment: &str,
    partition_commitment: &str,
    admitted_memory_ids: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_COMMITMENT_DOMAIN_V1);
    hash_str(&mut hasher, scope_commitment);
    hash_str(&mut hasher, partition_commitment);
    // Ranked/admitted order is intentionally receipt-significant.
    commit_string_slice(&mut hasher, admitted_memory_ids);
    format!(
        "reflective-memory-retrieval-receipt:{}",
        hasher.finalize().to_hex()
    )
}

fn commit_reality(hasher: &mut blake3::Hasher, reality: &ReflectiveRealityNamespaceV1) {
    match reality {
        ReflectiveRealityNamespaceV1::RealWorld => {
            hasher.update(&[0]);
        }
        ReflectiveRealityNamespaceV1::Fantasy { world_id } => {
            hasher.update(&[1]);
            hash_str(hasher, world_id);
        }
    }
}

fn commit_sorted_tags(hasher: &mut blake3::Hasher, tags: impl Iterator<Item = u8>) {
    let mut tags: Vec<u8> = tags.collect();
    tags.sort_unstable();
    hasher.update(&(tags.len() as u32).to_le_bytes());
    hasher.update(&tags);
}

fn commit_string_slice(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u32).to_le_bytes());
    for value in values {
        hash_str(hasher, value);
    }
}

fn namespace_tag(value: ReflectiveMemoryNamespaceV1) -> u8 {
    match value {
        ReflectiveMemoryNamespaceV1::ParticipantFact => 0,
        ReflectiveMemoryNamespaceV1::SymthaeaPersona => 1,
        ReflectiveMemoryNamespaceV1::SharedRelationship => 2,
        ReflectiveMemoryNamespaceV1::FantasyWorld => 3,
        ReflectiveMemoryNamespaceV1::ExplicitPreference => 4,
        ReflectiveMemoryNamespaceV1::InferredPreference => 5,
        ReflectiveMemoryNamespaceV1::Boundary => 6,
        ReflectiveMemoryNamespaceV1::Reflection => 7,
    }
}

fn perspective_tag(value: ReflectiveMemoryPerspectiveV1) -> u8 {
    match value {
        ReflectiveMemoryPerspectiveV1::Participant => 0,
        ReflectiveMemoryPerspectiveV1::Symthaea => 1,
        ReflectiveMemoryPerspectiveV1::Shared => 2,
    }
}

fn sensitivity_tag(value: ReflectiveMemorySensitivityV1) -> u8 {
    match value {
        ReflectiveMemorySensitivityV1::Ordinary => 0,
        ReflectiveMemorySensitivityV1::Personal => 1,
        ReflectiveMemorySensitivityV1::Intimate => 2,
    }
}

fn retention_tag(value: IntimateRetentionClassV1) -> u8 {
    match value {
        IntimateRetentionClassV1::EphemeralSession => 0,
        IntimateRetentionClassV1::DurableOptIn => 1,
        IntimateRetentionClassV1::ExternalExport => 2,
    }
}

fn ensure_unique_tags(
    tags: impl Iterator<Item = u8>,
    error: ReflectiveMemoryRetrievalErrorV1,
) -> Result<(), ReflectiveMemoryRetrievalErrorV1> {
    let mut seen = BTreeSet::new();
    for tag in tags {
        if !seen.insert(tag) {
            return Err(error);
        }
    }
    Ok(())
}

fn validate_id(
    value: &str,
    error: ReflectiveMemoryRetrievalErrorV1,
) -> Result<(), ReflectiveMemoryRetrievalErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(error)
    } else {
        Ok(())
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectiveMemoryRetrievalErrorV1 {
    InvalidQueryContextId,
    InvalidRetrievalPolicyId,
    InvalidFantasyWorldId,
    EmptyNamespaceSet,
    EmptyPerspectiveSet,
    EmptyRetentionSet,
    DuplicateNamespace,
    DuplicatePerspective,
    DuplicateRetention,
    ExternalExportRetentionForbidden,
    InvalidMinimumConfidence,
    InvalidCandidateMemoryId,
    DuplicateCandidateMemoryId,
    ScopeCommitmentMismatch,
    StalePartition,
    InvalidRankedMemoryId,
    DuplicateRankedMemoryId,
    RankedCandidateOutsidePartition,
}
