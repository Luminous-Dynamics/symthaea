// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Privacy-bound reflective-memory index build and snapshot receipts.
//!
//! This layer makes the current index snapshot evidence-bearing. It does not
//! store raw intimate text or embedding vectors.

use crate::intimate_memory_privacy::{
    IntimateArtifactKindV1, IntimateDependencyStateV1, IntimateMemoryArtifactV1,
    IntimateMemoryPrivacyGraphV1, IntimateRetentionClassV1,
};
use crate::reflective_memory_candidate_provenance::{
    CandidateSetCompletenessV1, ReflectiveMemoryCandidateSourceReceiptV1,
};
use crate::reflective_memory_retrieval::ReflectiveMemoryRetrievalScopeV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REFLECTIVE_MEMORY_INDEX_BUILD_SCHEMA_V1: &str =
    "symthaea.communication.reflective-memory-index-build.v1";
const SOURCE_LINEAGE_DOMAIN_V1: &[u8] = b"symthaea:reflective-index-source-lineage:v1\0";
const BUILD_RECEIPT_DOMAIN_V1: &[u8] = b"symthaea:reflective-index-build-receipt:v1\0";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexBuildParameterValueV1 {
    Unsigned(u64),
    Bool(bool),
    FiniteF64Bits(u64),
    Token(String),
    Reference(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexBuildParameterV1 {
    pub parameter_id: String,
    pub value: IndexBuildParameterValueV1,
}

impl IndexBuildParameterV1 {
    fn validate(&self) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        validate_id(&self.parameter_id)?;
        match &self.value {
            IndexBuildParameterValueV1::Unsigned(_) | IndexBuildParameterValueV1::Bool(_) => Ok(()),
            IndexBuildParameterValueV1::FiniteF64Bits(bits) => {
                if f64::from_bits(*bits).is_finite() {
                    Ok(())
                } else {
                    Err(ReflectiveMemoryIndexBuildErrorV1::NonFiniteBuildParameter)
                }
            }
            IndexBuildParameterValueV1::Token(value) => validate_token(value),
            IndexBuildParameterValueV1::Reference(value) => validate_ref(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEnumerationEvidenceV1 {
    pub partition_descriptor_ref: String,
    pub enumeration_protocol_id: String,
    pub enumeration_result_commitment: String,
    pub enumerated_count: u64,
}

impl IndexEnumerationEvidenceV1 {
    fn validate(&self) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        validate_ref(&self.partition_descriptor_ref)?;
        validate_id(&self.enumeration_protocol_id)?;
        validate_commitment(&self.enumeration_result_commitment)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectiveMemoryIndexBuildReceiptV1 {
    pub build_id: String,
    pub build_receipt_ref: String,
    pub privacy_index_artifact_id: String,
    pub index_schema_id: String,
    pub backend_implementation_id: String,
    pub backend_implementation_commitment: String,
    pub embedding_model_id: String,
    pub embedding_model_commitment: String,
    pub tokenizer_or_encoder_id: String,
    pub tokenizer_or_encoder_commitment: String,
    pub build_environment_ref: String,
    pub build_environment_commitment: String,
    pub clock_domain_id: String,
    pub started_at_ns: u64,
    pub ended_at_ns: u64,
    pub recorded_source_artifact_ids: BTreeSet<String>,
    pub source_state_commitments: BTreeMap<String, String>,
    pub source_lineage_commitment: String,
    pub build_parameters: Vec<IndexBuildParameterV1>,
    pub enumeration_evidence: Vec<IndexEnumerationEvidenceV1>,
    pub snapshot_artifact_ref: String,
    pub snapshot_commitment: String,
    pub receipt_commitment: String,
}

impl ReflectiveMemoryIndexBuildReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        build_id: impl Into<String>,
        build_receipt_ref: impl Into<String>,
        privacy_index_artifact_id: impl Into<String>,
        index_schema_id: impl Into<String>,
        backend_implementation_id: impl Into<String>,
        backend_implementation_commitment: impl Into<String>,
        embedding_model_id: impl Into<String>,
        embedding_model_commitment: impl Into<String>,
        tokenizer_or_encoder_id: impl Into<String>,
        tokenizer_or_encoder_commitment: impl Into<String>,
        build_environment_ref: impl Into<String>,
        build_environment_commitment: impl Into<String>,
        clock_domain_id: impl Into<String>,
        started_at_ns: u64,
        ended_at_ns: u64,
        source_state_commitments: BTreeMap<String, String>,
        mut build_parameters: Vec<IndexBuildParameterV1>,
        mut enumeration_evidence: Vec<IndexEnumerationEvidenceV1>,
        snapshot_artifact_ref: impl Into<String>,
        snapshot_commitment: impl Into<String>,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<Self, ReflectiveMemoryIndexBuildErrorV1> {
        let privacy_index_artifact_id = privacy_index_artifact_id.into();
        let index_artifact = live_index_artifact(privacy, &privacy_index_artifact_id)?;
        build_parameters.sort_by(|a, b| a.parameter_id.cmp(&b.parameter_id));
        enumeration_evidence.sort_by(|a, b| {
            a.partition_descriptor_ref
                .cmp(&b.partition_descriptor_ref)
                .then(a.enumeration_protocol_id.cmp(&b.enumeration_protocol_id))
        });
        let recorded_source_artifact_ids = index_artifact.source_ids.clone();
        let source_lineage_commitment = source_lineage_commitment_v1(
            &privacy_index_artifact_id,
            &recorded_source_artifact_ids,
            &source_state_commitments,
        )?;
        let mut receipt = Self {
            build_id: build_id.into(),
            build_receipt_ref: build_receipt_ref.into(),
            privacy_index_artifact_id,
            index_schema_id: index_schema_id.into(),
            backend_implementation_id: backend_implementation_id.into(),
            backend_implementation_commitment: backend_implementation_commitment.into(),
            embedding_model_id: embedding_model_id.into(),
            embedding_model_commitment: embedding_model_commitment.into(),
            tokenizer_or_encoder_id: tokenizer_or_encoder_id.into(),
            tokenizer_or_encoder_commitment: tokenizer_or_encoder_commitment.into(),
            build_environment_ref: build_environment_ref.into(),
            build_environment_commitment: build_environment_commitment.into(),
            clock_domain_id: clock_domain_id.into(),
            started_at_ns,
            ended_at_ns,
            recorded_source_artifact_ids,
            source_state_commitments,
            source_lineage_commitment,
            build_parameters,
            enumeration_evidence,
            snapshot_artifact_ref: snapshot_artifact_ref.into(),
            snapshot_commitment: snapshot_commitment.into(),
            receipt_commitment: String::new(),
        };
        receipt.validate_shape()?;
        receipt.receipt_commitment = receipt.compute_commitment_v1();
        receipt.validate_live(privacy)?;
        Ok(receipt)
    }

    pub fn validate_live(
        &self,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        self.validate_shape()?;
        let artifact = live_index_artifact(privacy, &self.privacy_index_artifact_id)?;
        if artifact.source_ids != self.recorded_source_artifact_ids {
            return Err(ReflectiveMemoryIndexBuildErrorV1::IndexSourceLineageMismatch);
        }
        let expected_lineage = source_lineage_commitment_v1(
            &self.privacy_index_artifact_id,
            &artifact.source_ids,
            &self.source_state_commitments,
        )?;
        if self.source_lineage_commitment != expected_lineage {
            return Err(ReflectiveMemoryIndexBuildErrorV1::SourceLineageCommitmentMismatch);
        }
        if self.receipt_commitment != self.compute_commitment_v1() {
            return Err(ReflectiveMemoryIndexBuildErrorV1::ReceiptCommitmentMismatch);
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        for value in [
            &self.build_id,
            &self.privacy_index_artifact_id,
            &self.index_schema_id,
            &self.backend_implementation_id,
            &self.embedding_model_id,
            &self.tokenizer_or_encoder_id,
            &self.clock_domain_id,
        ] {
            validate_id(value)?;
        }
        for value in [
            &self.build_receipt_ref,
            &self.build_environment_ref,
            &self.snapshot_artifact_ref,
        ] {
            validate_ref(value)?;
        }
        for value in [
            &self.backend_implementation_commitment,
            &self.embedding_model_commitment,
            &self.tokenizer_or_encoder_commitment,
            &self.build_environment_commitment,
            &self.source_lineage_commitment,
            &self.snapshot_commitment,
        ] {
            validate_commitment(value)?;
        }
        if self.ended_at_ns < self.started_at_ns {
            return Err(ReflectiveMemoryIndexBuildErrorV1::InvalidTimestamps);
        }
        if self.recorded_source_artifact_ids.is_empty() {
            return Err(ReflectiveMemoryIndexBuildErrorV1::MissingIndexSourceLineage);
        }
        if self.source_state_commitments.len() != self.recorded_source_artifact_ids.len()
            || self
                .recorded_source_artifact_ids
                .iter()
                .any(|id| !self.source_state_commitments.contains_key(id))
        {
            return Err(ReflectiveMemoryIndexBuildErrorV1::SourceStateCommitmentSetMismatch);
        }
        for (source_id, commitment) in &self.source_state_commitments {
            validate_id(source_id)?;
            validate_commitment(commitment)?;
        }
        if self.build_parameters.len() > 128 {
            return Err(ReflectiveMemoryIndexBuildErrorV1::TooManyBuildParameters);
        }
        let mut prior_parameter: Option<&str> = None;
        for parameter in &self.build_parameters {
            parameter.validate()?;
            if prior_parameter.is_some_and(|prior| prior >= parameter.parameter_id.as_str()) {
                return Err(ReflectiveMemoryIndexBuildErrorV1::BuildParametersNotCanonicalUnique);
            }
            prior_parameter = Some(parameter.parameter_id.as_str());
        }
        if self.enumeration_evidence.len() > 256 {
            return Err(ReflectiveMemoryIndexBuildErrorV1::TooManyEnumerationEvidenceItems);
        }
        let mut prior_partition: Option<&str> = None;
        for evidence in &self.enumeration_evidence {
            evidence.validate()?;
            if prior_partition.is_some_and(|prior| prior >= evidence.partition_descriptor_ref.as_str()) {
                return Err(ReflectiveMemoryIndexBuildErrorV1::EnumerationEvidenceNotCanonicalUnique);
            }
            prior_partition = Some(evidence.partition_descriptor_ref.as_str());
        }
        Ok(())
    }

    fn compute_commitment_v1(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(BUILD_RECEIPT_DOMAIN_V1);
        hash_str(&mut hasher, REFLECTIVE_MEMORY_INDEX_BUILD_SCHEMA_V1);
        for value in [
            &self.build_id,
            &self.build_receipt_ref,
            &self.privacy_index_artifact_id,
            &self.index_schema_id,
            &self.backend_implementation_id,
            &self.backend_implementation_commitment,
            &self.embedding_model_id,
            &self.embedding_model_commitment,
            &self.tokenizer_or_encoder_id,
            &self.tokenizer_or_encoder_commitment,
            &self.build_environment_ref,
            &self.build_environment_commitment,
            &self.clock_domain_id,
        ] {
            hash_str(&mut hasher, value);
        }
        hasher.update(&self.started_at_ns.to_le_bytes());
        hasher.update(&self.ended_at_ns.to_le_bytes());
        hasher.update(&(self.recorded_source_artifact_ids.len() as u32).to_le_bytes());
        for source_id in &self.recorded_source_artifact_ids {
            hash_str(&mut hasher, source_id);
            hash_str(
                &mut hasher,
                self.source_state_commitments
                    .get(source_id)
                    .expect("validated source state commitment exists"),
            );
        }
        hash_str(&mut hasher, &self.source_lineage_commitment);
        hasher.update(&(self.build_parameters.len() as u32).to_le_bytes());
        for parameter in &self.build_parameters {
            commit_build_parameter(&mut hasher, parameter);
        }
        hasher.update(&(self.enumeration_evidence.len() as u32).to_le_bytes());
        for evidence in &self.enumeration_evidence {
            hash_str(&mut hasher, &evidence.partition_descriptor_ref);
            hash_str(&mut hasher, &evidence.enumeration_protocol_id);
            hash_str(&mut hasher, &evidence.enumeration_result_commitment);
            hasher.update(&evidence.enumerated_count.to_le_bytes());
        }
        hash_str(&mut hasher, &self.snapshot_artifact_ref);
        hash_str(&mut hasher, &self.snapshot_commitment);
        format!("reflective-index-build:{}", hasher.finalize().to_hex())
    }

    fn enumeration_evidence_for(&self, partition_descriptor_ref: &str) -> Option<&IndexEnumerationEvidenceV1> {
        self.enumeration_evidence
            .iter()
            .find(|item| item.partition_descriptor_ref == partition_descriptor_ref)
    }
}

#[derive(Debug, Default)]
pub struct CurrentReflectiveMemoryIndexRegistryV1 {
    current: BTreeMap<String, ReflectiveMemoryIndexBuildReceiptV1>,
    known_snapshot_commitments: BTreeSet<String>,
}

impl CurrentReflectiveMemoryIndexRegistryV1 {
    pub fn register_initial(
        &mut self,
        receipt: ReflectiveMemoryIndexBuildReceiptV1,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        receipt.validate_live(privacy)?;
        if self.current.contains_key(&receipt.privacy_index_artifact_id) {
            return Err(ReflectiveMemoryIndexBuildErrorV1::CurrentSnapshotAlreadyRegistered);
        }
        if !self
            .known_snapshot_commitments
            .insert(receipt.snapshot_commitment.clone())
        {
            return Err(ReflectiveMemoryIndexBuildErrorV1::SnapshotCommitmentReused);
        }
        self.current
            .insert(receipt.privacy_index_artifact_id.clone(), receipt);
        Ok(())
    }

    pub fn advance(
        &mut self,
        receipt: ReflectiveMemoryIndexBuildReceiptV1,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
        receipt.validate_live(privacy)?;
        let previous = self
            .current
            .get(&receipt.privacy_index_artifact_id)
            .ok_or(ReflectiveMemoryIndexBuildErrorV1::NoCurrentSnapshot)?;
        if previous.snapshot_commitment == receipt.snapshot_commitment {
            return Err(ReflectiveMemoryIndexBuildErrorV1::SnapshotDidNotAdvance);
        }
        if !self
            .known_snapshot_commitments
            .insert(receipt.snapshot_commitment.clone())
        {
            return Err(ReflectiveMemoryIndexBuildErrorV1::SnapshotCommitmentReused);
        }
        self.current
            .insert(receipt.privacy_index_artifact_id.clone(), receipt);
        Ok(())
    }

    pub fn current_validated<'a>(
        &'a self,
        privacy_index_artifact_id: &str,
        privacy: &IntimateMemoryPrivacyGraphV1,
    ) -> Result<&'a ReflectiveMemoryIndexBuildReceiptV1, ReflectiveMemoryIndexBuildErrorV1> {
        let receipt = self
            .current
            .get(privacy_index_artifact_id)
            .ok_or(ReflectiveMemoryIndexBuildErrorV1::NoCurrentSnapshot)?;
        receipt.validate_live(privacy)?;
        Ok(receipt)
    }
}

pub fn validate_candidate_source_against_current_index_v1(
    source: &ReflectiveMemoryCandidateSourceReceiptV1,
    scope: &ReflectiveMemoryRetrievalScopeV1,
    registry: &CurrentReflectiveMemoryIndexRegistryV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<String, ReflectiveMemoryIndexBuildErrorV1> {
    source
        .validate_live(scope, privacy)
        .map_err(|_| ReflectiveMemoryIndexBuildErrorV1::CandidateSourceInvalid)?;
    let build = registry.current_validated(&source.privacy_index_artifact_id, privacy)?;
    if source.index_schema_id != build.index_schema_id
        || source.backend_implementation_id != build.backend_implementation_id
        || source.backend_implementation_commitment != build.backend_implementation_commitment
        || source.index_snapshot_commitment != build.snapshot_commitment
        || source.index_build_receipt_ref != build.build_receipt_ref
        || source.recorded_source_artifact_ids != build.recorded_source_artifact_ids
    {
        return Err(ReflectiveMemoryIndexBuildErrorV1::CandidateDoesNotMatchCurrentSnapshot);
    }

    if let CandidateSetCompletenessV1::ExactEnumeratedPartition {
        partition_descriptor_ref,
        expected_count,
    } = &source.completeness
    {
        let evidence = build
            .enumeration_evidence_for(partition_descriptor_ref)
            .ok_or(ReflectiveMemoryIndexBuildErrorV1::MissingExactEnumerationEvidence)?;
        if evidence.enumerated_count != *expected_count {
            return Err(ReflectiveMemoryIndexBuildErrorV1::ExactEnumerationEvidenceMismatch);
        }
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea:reflective-index-current-candidate:v1\0");
    hash_str(&mut hasher, &build.receipt_commitment);
    hash_str(&mut hasher, &source.receipt_commitment);
    Ok(format!(
        "reflective-index-current-candidate:{}",
        hasher.finalize().to_hex()
    ))
}

fn live_index_artifact<'a>(
    privacy: &'a IntimateMemoryPrivacyGraphV1,
    artifact_id: &str,
) -> Result<&'a IntimateMemoryArtifactV1, ReflectiveMemoryIndexBuildErrorV1> {
    validate_id(artifact_id)?;
    let artifact = privacy
        .active_artifact(artifact_id)
        .ok_or(ReflectiveMemoryIndexBuildErrorV1::IndexArtifactUnavailable)?;
    if artifact.kind != IntimateArtifactKindV1::EmbeddingOrIndex {
        return Err(ReflectiveMemoryIndexBuildErrorV1::WrongIndexArtifactKind);
    }
    if artifact.dependency_state != IntimateDependencyStateV1::Complete
        || !privacy.is_retrievable(artifact_id)
    {
        return Err(ReflectiveMemoryIndexBuildErrorV1::IndexArtifactNotRetrievable);
    }
    if artifact.retention == IntimateRetentionClassV1::ExternalExport {
        return Err(ReflectiveMemoryIndexBuildErrorV1::ExternalExportCannotBeCurrentIndex);
    }
    if artifact.source_ids.is_empty() {
        return Err(ReflectiveMemoryIndexBuildErrorV1::MissingIndexSourceLineage);
    }
    Ok(artifact)
}

fn source_lineage_commitment_v1(
    privacy_index_artifact_id: &str,
    source_ids: &BTreeSet<String>,
    source_state_commitments: &BTreeMap<String, String>,
) -> Result<String, ReflectiveMemoryIndexBuildErrorV1> {
    if source_ids.len() != source_state_commitments.len()
        || source_ids
            .iter()
            .any(|id| !source_state_commitments.contains_key(id))
    {
        return Err(ReflectiveMemoryIndexBuildErrorV1::SourceStateCommitmentSetMismatch);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(SOURCE_LINEAGE_DOMAIN_V1);
    hash_str(&mut hasher, privacy_index_artifact_id);
    hasher.update(&(source_ids.len() as u32).to_le_bytes());
    for source_id in source_ids {
        let commitment = source_state_commitments
            .get(source_id)
            .ok_or(ReflectiveMemoryIndexBuildErrorV1::SourceStateCommitmentSetMismatch)?;
        validate_id(source_id)?;
        validate_commitment(commitment)?;
        hash_str(&mut hasher, source_id);
        hash_str(&mut hasher, commitment);
    }
    Ok(format!(
        "reflective-index-source-lineage:{}",
        hasher.finalize().to_hex()
    ))
}

fn commit_build_parameter(hasher: &mut blake3::Hasher, parameter: &IndexBuildParameterV1) {
    hash_str(hasher, &parameter.parameter_id);
    match &parameter.value {
        IndexBuildParameterValueV1::Unsigned(value) => {
            hasher.update(&[0]);
            hasher.update(&value.to_le_bytes());
        }
        IndexBuildParameterValueV1::Bool(value) => {
            hasher.update(&[1, u8::from(*value)]);
        }
        IndexBuildParameterValueV1::FiniteF64Bits(bits) => {
            hasher.update(&[2]);
            hasher.update(&bits.to_le_bytes());
        }
        IndexBuildParameterValueV1::Token(value) => {
            hasher.update(&[3]);
            hash_str(hasher, value);
        }
        IndexBuildParameterValueV1::Reference(value) => {
            hasher.update(&[4]);
            hash_str(hasher, value);
        }
    }
}

fn validate_id(value: &str) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(ReflectiveMemoryIndexBuildErrorV1::InvalidIdentity)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 {
        Err(ReflectiveMemoryIndexBuildErrorV1::InvalidReference)
    } else {
        Ok(())
    }
}

fn validate_token(value: &str) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 192 {
        Err(ReflectiveMemoryIndexBuildErrorV1::InvalidToken)
    } else {
        Ok(())
    }
}

fn validate_commitment(value: &str) -> Result<(), ReflectiveMemoryIndexBuildErrorV1> {
    let Some((prefix, hex)) = value.rsplit_once(':') else {
        return Err(ReflectiveMemoryIndexBuildErrorV1::InvalidCommitment);
    };
    if prefix.trim().is_empty()
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ReflectiveMemoryIndexBuildErrorV1::InvalidCommitment);
    }
    Ok(())
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectiveMemoryIndexBuildErrorV1 {
    InvalidIdentity,
    InvalidReference,
    InvalidToken,
    InvalidCommitment,
    NonFiniteBuildParameter,
    InvalidTimestamps,
    IndexArtifactUnavailable,
    WrongIndexArtifactKind,
    IndexArtifactNotRetrievable,
    ExternalExportCannotBeCurrentIndex,
    MissingIndexSourceLineage,
    IndexSourceLineageMismatch,
    SourceStateCommitmentSetMismatch,
    SourceLineageCommitmentMismatch,
    TooManyBuildParameters,
    BuildParametersNotCanonicalUnique,
    TooManyEnumerationEvidenceItems,
    EnumerationEvidenceNotCanonicalUnique,
    ReceiptCommitmentMismatch,
    CurrentSnapshotAlreadyRegistered,
    NoCurrentSnapshot,
    SnapshotDidNotAdvance,
    SnapshotCommitmentReused,
    CandidateSourceInvalid,
    CandidateDoesNotMatchCurrentSnapshot,
    MissingExactEnumerationEvidence,
    ExactEnumerationEvidenceMismatch,
}
