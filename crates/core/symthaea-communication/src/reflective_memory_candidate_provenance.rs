// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance for reflective-memory candidate generation.
//!
//! Candidate generation is evidence, not retrieval authority. This module binds
//! the candidate universe to a live KAMA-PRIV index artifact and then chains it
//! into KAMA-DIALOGUE-001N2's pre-rank eligibility firewall.

use crate::intimate_memory_privacy::{
    IntimateArtifactKindV1, IntimateDependencyStateV1, IntimateMemoryArtifactV1,
    IntimateMemoryPrivacyGraphV1, IntimateRetentionClassV1,
};
use crate::reflective_intimacy_memory::ReflectiveIntimacyMemoryIndexV1;
use crate::reflective_memory_retrieval::{
    ReflectiveMemoryRetrievalPartitionV1, ReflectiveMemoryRetrievalReceiptV1,
    ReflectiveMemoryRetrievalScopeV1, admit_ranked_ids_v1, build_pre_rank_partition_v1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REFLECTIVE_MEMORY_CANDIDATE_PROVENANCE_SCHEMA_V1: &str =
    "symthaea.communication.reflective-memory-candidate-provenance.v1";
const SOURCE_DOMAIN_V1: &[u8] = b"symthaea:reflective-memory-candidate-source:v1\0";
const PARTITION_CHAIN_DOMAIN_V1: &[u8] = b"symthaea:reflective-memory-candidate-partition:v1\0";
const RETRIEVAL_CHAIN_DOMAIN_V1: &[u8] = b"symthaea:reflective-memory-candidate-retrieval:v1\0";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateSetCompletenessV1 {
    ExactEnumeratedPartition {
        partition_descriptor_ref: String,
        expected_count: u64,
    },
    BoundedSearch {
        limit: u32,
        search_profile_ref: String,
    },
    UnknownCompleteness {
        search_profile_ref: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateSearchParameterValueV1 {
    Unsigned(u64),
    Bool(bool),
    /// Exact IEEE-754 representation. Non-finite values are rejected.
    FiniteF64Bits(u64),
    Token(String),
    Reference(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSearchParameterV1 {
    pub parameter_id: String,
    pub value: CandidateSearchParameterValueV1,
}

impl CandidateSearchParameterV1 {
    fn validate(&self) -> Result<(), CandidateProvenanceErrorV1> {
        validate_id(&self.parameter_id)?;
        match &self.value {
            CandidateSearchParameterValueV1::Unsigned(_)
            | CandidateSearchParameterValueV1::Bool(_) => Ok(()),
            CandidateSearchParameterValueV1::FiniteF64Bits(bits) => {
                if f64::from_bits(*bits).is_finite() {
                    Ok(())
                } else {
                    Err(CandidateProvenanceErrorV1::NonFiniteSearchParameter)
                }
            }
            CandidateSearchParameterValueV1::Token(value) => validate_token(value),
            CandidateSearchParameterValueV1::Reference(value) => validate_ref(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectiveMemoryCandidateSourceReceiptV1 {
    pub candidate_source_id: String,
    pub query_context_id: String,
    pub retrieval_policy_id: String,
    pub privacy_index_artifact_id: String,
    pub index_schema_id: String,
    pub backend_implementation_id: String,
    pub backend_implementation_commitment: String,
    pub index_snapshot_commitment: String,
    pub index_build_receipt_ref: String,
    pub recorded_source_artifact_ids: BTreeSet<String>,
    pub completeness: CandidateSetCompletenessV1,
    /// Canonical order by parameter ID.
    pub search_parameters: Vec<CandidateSearchParameterV1>,
    /// Canonical sorted set. Backend rank is intentionally discarded here.
    pub candidate_memory_ids: Vec<String>,
    pub receipt_commitment: String,
}

impl ReflectiveMemoryCandidateSourceReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        candidate_source_id: impl Into<String>,
        scope: &ReflectiveMemoryRetrievalScopeV1,
        privacy_index_artifact_id: impl Into<String>,
        index_schema_id: impl Into<String>,
        backend_implementation_id: impl Into<String>,
        backend_implementation_commitment: impl Into<String>,
        index_snapshot_commitment: impl Into<String>,
        index_build_receipt_ref: impl Into<String>,
        completeness: CandidateSetCompletenessV1,
        mut search_parameters: Vec<CandidateSearchParameterV1>,
        mut candidate_memory_ids: Vec<String>,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<Self, CandidateProvenanceErrorV1> {
        scope
            .validate()
            .map_err(|_| CandidateProvenanceErrorV1::InvalidRetrievalScope)?;
        search_parameters.sort_by(|a, b| a.parameter_id.cmp(&b.parameter_id));
        candidate_memory_ids.sort();
        let privacy_index_artifact_id = privacy_index_artifact_id.into();
        let index_artifact = live_index_artifact(privacy, &privacy_index_artifact_id)?;
        let mut receipt = Self {
            candidate_source_id: candidate_source_id.into(),
            query_context_id: scope.query_context_id.clone(),
            retrieval_policy_id: scope.retrieval_policy_id.clone(),
            privacy_index_artifact_id,
            index_schema_id: index_schema_id.into(),
            backend_implementation_id: backend_implementation_id.into(),
            backend_implementation_commitment: backend_implementation_commitment.into(),
            index_snapshot_commitment: index_snapshot_commitment.into(),
            index_build_receipt_ref: index_build_receipt_ref.into(),
            recorded_source_artifact_ids: index_artifact.source_ids.clone(),
            completeness,
            search_parameters,
            candidate_memory_ids,
            receipt_commitment: String::new(),
        };
        receipt.validate_shape()?;
        receipt.receipt_commitment = receipt.compute_commitment_v1();
        receipt.validate_live(scope, privacy)?;
        Ok(receipt)
    }

    pub fn validate_live(
        &self,
        scope: &ReflectiveMemoryRetrievalScopeV1,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<(), CandidateProvenanceErrorV1> {
        self.validate_shape()?;
        scope
            .validate()
            .map_err(|_| CandidateProvenanceErrorV1::InvalidRetrievalScope)?;
        if self.query_context_id != scope.query_context_id
            || self.retrieval_policy_id != scope.retrieval_policy_id
        {
            return Err(CandidateProvenanceErrorV1::ScopeIdentityMismatch);
        }
        let index_artifact = live_index_artifact(privacy, &self.privacy_index_artifact_id)?;
        if index_artifact.source_ids != self.recorded_source_artifact_ids {
            return Err(CandidateProvenanceErrorV1::IndexSourceLineageMismatch);
        }
        if self.receipt_commitment != self.compute_commitment_v1() {
            return Err(CandidateProvenanceErrorV1::ReceiptCommitmentMismatch);
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), CandidateProvenanceErrorV1> {
        for value in [
            &self.candidate_source_id,
            &self.query_context_id,
            &self.retrieval_policy_id,
            &self.privacy_index_artifact_id,
            &self.index_schema_id,
            &self.backend_implementation_id,
        ] {
            validate_id(value)?;
        }
        validate_commitment(&self.backend_implementation_commitment)?;
        validate_commitment(&self.index_snapshot_commitment)?;
        validate_ref(&self.index_build_receipt_ref)?;

        if self.recorded_source_artifact_ids.is_empty() {
            return Err(CandidateProvenanceErrorV1::MissingIndexSourceLineage);
        }
        for source_id in &self.recorded_source_artifact_ids {
            validate_id(source_id)?;
        }

        if self.search_parameters.len() > 64 {
            return Err(CandidateProvenanceErrorV1::TooManySearchParameters);
        }
        let mut previous_parameter: Option<&str> = None;
        for parameter in &self.search_parameters {
            parameter.validate()?;
            if previous_parameter.is_some_and(|previous| previous >= parameter.parameter_id.as_str()) {
                return Err(CandidateProvenanceErrorV1::SearchParametersNotCanonicalUnique);
            }
            previous_parameter = Some(parameter.parameter_id.as_str());
        }

        if self.candidate_memory_ids.len() > 65_536 {
            return Err(CandidateProvenanceErrorV1::TooManyCandidates);
        }
        let mut previous_candidate: Option<&str> = None;
        for memory_id in &self.candidate_memory_ids {
            validate_id(memory_id)?;
            if previous_candidate.is_some_and(|previous| previous >= memory_id.as_str()) {
                return Err(CandidateProvenanceErrorV1::CandidateIdsNotCanonicalUnique);
            }
            previous_candidate = Some(memory_id.as_str());
        }

        match &self.completeness {
            CandidateSetCompletenessV1::ExactEnumeratedPartition {
                partition_descriptor_ref,
                expected_count,
            } => {
                validate_ref(partition_descriptor_ref)?;
                if *expected_count != self.candidate_memory_ids.len() as u64 {
                    return Err(CandidateProvenanceErrorV1::ExactEnumerationCountMismatch);
                }
            }
            CandidateSetCompletenessV1::BoundedSearch {
                limit,
                search_profile_ref,
            } => {
                validate_ref(search_profile_ref)?;
                if *limit == 0 || self.candidate_memory_ids.len() > *limit as usize {
                    return Err(CandidateProvenanceErrorV1::InvalidBoundedSearchLimit);
                }
            }
            CandidateSetCompletenessV1::UnknownCompleteness { search_profile_ref } => {
                validate_ref(search_profile_ref)?;
            }
        }
        Ok(())
    }

    fn compute_commitment_v1(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SOURCE_DOMAIN_V1);
        hash_str(&mut hasher, REFLECTIVE_MEMORY_CANDIDATE_PROVENANCE_SCHEMA_V1);
        for value in [
            &self.candidate_source_id,
            &self.query_context_id,
            &self.retrieval_policy_id,
            &self.privacy_index_artifact_id,
            &self.index_schema_id,
            &self.backend_implementation_id,
            &self.backend_implementation_commitment,
            &self.index_snapshot_commitment,
            &self.index_build_receipt_ref,
        ] {
            hash_str(&mut hasher, value);
        }
        hasher.update(&(self.recorded_source_artifact_ids.len() as u32).to_le_bytes());
        for source_id in &self.recorded_source_artifact_ids {
            hash_str(&mut hasher, source_id);
        }
        commit_completeness(&mut hasher, &self.completeness);
        hasher.update(&(self.search_parameters.len() as u32).to_le_bytes());
        for parameter in &self.search_parameters {
            commit_parameter(&mut hasher, parameter);
        }
        hasher.update(&(self.candidate_memory_ids.len() as u32).to_le_bytes());
        for memory_id in &self.candidate_memory_ids {
            hash_str(&mut hasher, memory_id);
        }
        format!(
            "reflective-memory-candidate-source:{}",
            hasher.finalize().to_hex()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenancedReflectiveMemoryPartitionV1 {
    pub candidate_source_receipt_commitment: String,
    pub inner_partition: ReflectiveMemoryRetrievalPartitionV1,
    pub chain_commitment: String,
}

pub fn build_provenanced_partition_v1(
    source: &ReflectiveMemoryCandidateSourceReceiptV1,
    scope: &ReflectiveMemoryRetrievalScopeV1,
    index: &ReflectiveIntimacyMemoryIndexV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<ProvenancedReflectiveMemoryPartitionV1, CandidateProvenanceErrorV1> {
    source.validate_live(scope, privacy)?;
    let inner_partition = build_pre_rank_partition_v1(
        scope,
        &source.candidate_memory_ids,
        index,
        privacy,
    )
    .map_err(|_| CandidateProvenanceErrorV1::RetrievalFirewallRejected)?;
    let chain_commitment = partition_chain_commitment_v1(
        &source.receipt_commitment,
        inner_partition.scope_commitment(),
        inner_partition.partition_commitment(),
    );
    Ok(ProvenancedReflectiveMemoryPartitionV1 {
        candidate_source_receipt_commitment: source.receipt_commitment.clone(),
        inner_partition,
        chain_commitment,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenancedReflectiveMemoryRetrievalReceiptV1 {
    pub candidate_source_receipt_commitment: String,
    pub partition_chain_commitment: String,
    pub inner_retrieval_receipt: ReflectiveMemoryRetrievalReceiptV1,
    pub receipt_commitment: String,
}

pub fn admit_ranked_ids_with_candidate_source_v1(
    source: &ReflectiveMemoryCandidateSourceReceiptV1,
    scope: &ReflectiveMemoryRetrievalScopeV1,
    partition: &ProvenancedReflectiveMemoryPartitionV1,
    ranked_memory_ids: &[String],
    index: &ReflectiveIntimacyMemoryIndexV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<ProvenancedReflectiveMemoryRetrievalReceiptV1, CandidateProvenanceErrorV1> {
    source.validate_live(scope, privacy)?;
    if partition.candidate_source_receipt_commitment != source.receipt_commitment {
        return Err(CandidateProvenanceErrorV1::CandidateSourceReceiptMismatch);
    }
    let fresh = build_provenanced_partition_v1(source, scope, index, privacy)?;
    if fresh != *partition {
        return Err(CandidateProvenanceErrorV1::StaleProvenancedPartition);
    }
    let inner_retrieval_receipt = admit_ranked_ids_v1(
        scope,
        &partition.inner_partition,
        ranked_memory_ids,
        index,
        privacy,
    )
    .map_err(|_| CandidateProvenanceErrorV1::RetrievalFirewallRejected)?;
    let receipt_commitment = retrieval_chain_commitment_v1(
        &source.receipt_commitment,
        &partition.chain_commitment,
        &inner_retrieval_receipt.receipt_commitment,
    );
    Ok(ProvenancedReflectiveMemoryRetrievalReceiptV1 {
        candidate_source_receipt_commitment: source.receipt_commitment.clone(),
        partition_chain_commitment: partition.chain_commitment.clone(),
        inner_retrieval_receipt,
        receipt_commitment,
    })
}

fn live_index_artifact<'a>(
    privacy: &'a IntimateMemoryPrivacyGraphV1,
    artifact_id: &str,
) -> Result<&'a IntimateMemoryArtifactV1, CandidateProvenanceErrorV1> {
    validate_id(artifact_id)?;
    let artifact = privacy
        .active_artifact(artifact_id)
        .ok_or(CandidateProvenanceErrorV1::IndexArtifactUnavailable)?;
    if artifact.kind != IntimateArtifactKindV1::EmbeddingOrIndex {
        return Err(CandidateProvenanceErrorV1::WrongIndexArtifactKind);
    }
    if artifact.dependency_state != IntimateDependencyStateV1::Complete
        || !privacy.is_retrievable(artifact_id)
    {
        return Err(CandidateProvenanceErrorV1::IndexArtifactNotRetrievable);
    }
    if artifact.retention == IntimateRetentionClassV1::ExternalExport {
        return Err(CandidateProvenanceErrorV1::ExternalExportCannotBeRetrievalIndex);
    }
    if artifact.source_ids.is_empty() {
        return Err(CandidateProvenanceErrorV1::MissingIndexSourceLineage);
    }
    Ok(artifact)
}

fn partition_chain_commitment_v1(source: &str, scope: &str, partition: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PARTITION_CHAIN_DOMAIN_V1);
    hash_str(&mut hasher, source);
    hash_str(&mut hasher, scope);
    hash_str(&mut hasher, partition);
    format!(
        "reflective-memory-candidate-partition:{}",
        hasher.finalize().to_hex()
    )
}

fn retrieval_chain_commitment_v1(source: &str, partition: &str, retrieval: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RETRIEVAL_CHAIN_DOMAIN_V1);
    hash_str(&mut hasher, source);
    hash_str(&mut hasher, partition);
    hash_str(&mut hasher, retrieval);
    format!(
        "reflective-memory-candidate-retrieval:{}",
        hasher.finalize().to_hex()
    )
}

fn commit_completeness(hasher: &mut blake3::Hasher, value: &CandidateSetCompletenessV1) {
    match value {
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref,
            expected_count,
        } => {
            hasher.update(&[0]);
            hash_str(hasher, partition_descriptor_ref);
            hasher.update(&expected_count.to_le_bytes());
        }
        CandidateSetCompletenessV1::BoundedSearch {
            limit,
            search_profile_ref,
        } => {
            hasher.update(&[1]);
            hasher.update(&limit.to_le_bytes());
            hash_str(hasher, search_profile_ref);
        }
        CandidateSetCompletenessV1::UnknownCompleteness { search_profile_ref } => {
            hasher.update(&[2]);
            hash_str(hasher, search_profile_ref);
        }
    }
}

fn commit_parameter(hasher: &mut blake3::Hasher, parameter: &CandidateSearchParameterV1) {
    hash_str(hasher, &parameter.parameter_id);
    match &parameter.value {
        CandidateSearchParameterValueV1::Unsigned(value) => {
            hasher.update(&[0]);
            hasher.update(&value.to_le_bytes());
        }
        CandidateSearchParameterValueV1::Bool(value) => {
            hasher.update(&[1, u8::from(*value)]);
        }
        CandidateSearchParameterValueV1::FiniteF64Bits(bits) => {
            hasher.update(&[2]);
            hasher.update(&bits.to_le_bytes());
        }
        CandidateSearchParameterValueV1::Token(value) => {
            hasher.update(&[3]);
            hash_str(hasher, value);
        }
        CandidateSearchParameterValueV1::Reference(value) => {
            hasher.update(&[4]);
            hash_str(hasher, value);
        }
    }
}

fn validate_id(value: &str) -> Result<(), CandidateProvenanceErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(CandidateProvenanceErrorV1::InvalidIdentity)
    } else {
        Ok(())
    }
}

fn validate_token(value: &str) -> Result<(), CandidateProvenanceErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(CandidateProvenanceErrorV1::InvalidToken)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), CandidateProvenanceErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 {
        Err(CandidateProvenanceErrorV1::InvalidReference)
    } else {
        Ok(())
    }
}

fn validate_commitment(value: &str) -> Result<(), CandidateProvenanceErrorV1> {
    let Some((prefix, hex)) = value.rsplit_once(':') else {
        return Err(CandidateProvenanceErrorV1::InvalidCommitment);
    };
    if prefix.trim().is_empty()
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CandidateProvenanceErrorV1::InvalidCommitment);
    }
    Ok(())
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateProvenanceErrorV1 {
    InvalidRetrievalScope,
    InvalidIdentity,
    InvalidToken,
    InvalidReference,
    InvalidCommitment,
    NonFiniteSearchParameter,
    IndexArtifactUnavailable,
    WrongIndexArtifactKind,
    IndexArtifactNotRetrievable,
    ExternalExportCannotBeRetrievalIndex,
    MissingIndexSourceLineage,
    IndexSourceLineageMismatch,
    ScopeIdentityMismatch,
    TooManySearchParameters,
    SearchParametersNotCanonicalUnique,
    TooManyCandidates,
    CandidateIdsNotCanonicalUnique,
    ExactEnumerationCountMismatch,
    InvalidBoundedSearchLimit,
    ReceiptCommitmentMismatch,
    RetrievalFirewallRejected,
    CandidateSourceReceiptMismatch,
    StaleProvenancedPartition,
}
