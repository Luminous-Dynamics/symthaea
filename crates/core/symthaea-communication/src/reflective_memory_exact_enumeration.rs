// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-enumeration set binding for reflective-memory candidate provenance.

use crate::intimate_memory_privacy::IntimateMemoryPrivacyGraphV1;
use crate::reflective_memory_candidate_provenance::{
    CandidateSetCompletenessV1, ReflectiveMemoryCandidateSourceReceiptV1,
};
use crate::reflective_memory_index_build::{
    CurrentReflectiveMemoryIndexRegistryV1, ReflectiveMemoryIndexBuildErrorV1,
    validate_candidate_source_against_current_index_v1,
};
use crate::reflective_memory_retrieval::ReflectiveMemoryRetrievalScopeV1;

pub const REFLECTIVE_MEMORY_EXACT_ENUMERATION_SCHEMA_V1: &str =
    "symthaea.communication.reflective-memory-exact-enumeration.v1";
const EXACT_ENUMERATION_DOMAIN_V1: &[u8] =
    b"symthaea:reflective-memory-exact-enumeration-result:v1\0";
const STRICT_CURRENT_BINDING_DOMAIN_V1: &[u8] =
    b"symthaea:reflective-memory-current-candidate-strict:v1\0";

pub fn exact_enumeration_result_commitment_v1(
    partition_descriptor_ref: &str,
    candidate_memory_ids: &[String],
) -> Result<String, ReflectiveMemoryExactEnumerationErrorV1> {
    validate_ref(partition_descriptor_ref)?;
    let mut ids = candidate_memory_ids.to_vec();
    ids.sort();
    if ids.len() > 65_536 {
        return Err(ReflectiveMemoryExactEnumerationErrorV1::TooManyCandidates);
    }
    let mut previous: Option<&str> = None;
    for id in &ids {
        validate_id(id)?;
        if previous.is_some_and(|prior| prior >= id.as_str()) {
            return Err(ReflectiveMemoryExactEnumerationErrorV1::CandidateIdsNotUnique);
        }
        previous = Some(id.as_str());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(EXACT_ENUMERATION_DOMAIN_V1);
    hash_str(&mut hasher, REFLECTIVE_MEMORY_EXACT_ENUMERATION_SCHEMA_V1);
    hash_str(&mut hasher, partition_descriptor_ref);
    hasher.update(&(ids.len() as u64).to_le_bytes());
    for id in &ids {
        hash_str(&mut hasher, id);
    }
    Ok(format!(
        "reflective-memory-exact-enumeration:{}",
        hasher.finalize().to_hex()
    ))
}

pub fn validate_candidate_source_against_current_index_strict_v1(
    source: &ReflectiveMemoryCandidateSourceReceiptV1,
    scope: &ReflectiveMemoryRetrievalScopeV1,
    registry: &CurrentReflectiveMemoryIndexRegistryV1,
    privacy: &IntimateMemoryPrivacyGraphV1,
) -> Result<String, ReflectiveMemoryExactEnumerationErrorV1> {
    let current_binding = validate_candidate_source_against_current_index_v1(
        source,
        scope,
        registry,
        privacy,
    )
    .map_err(ReflectiveMemoryExactEnumerationErrorV1::CurrentIndexValidationFailed)?;

    let exact_enumeration_binding = match &source.completeness {
        CandidateSetCompletenessV1::ExactEnumeratedPartition {
            partition_descriptor_ref,
            expected_count,
        } => {
            if *expected_count != source.candidate_memory_ids.len() as u64 {
                return Err(ReflectiveMemoryExactEnumerationErrorV1::CandidateCountMismatch);
            }
            let build = registry
                .current_validated(&source.privacy_index_artifact_id, privacy)
                .map_err(ReflectiveMemoryExactEnumerationErrorV1::CurrentIndexValidationFailed)?;
            let evidence = build
                .enumeration_evidence
                .iter()
                .find(|item| {
                    item.partition_descriptor_ref.as_str() == partition_descriptor_ref.as_str()
                })
                .ok_or(ReflectiveMemoryExactEnumerationErrorV1::MissingEnumerationEvidence)?;
            if evidence.enumerated_count != *expected_count {
                return Err(ReflectiveMemoryExactEnumerationErrorV1::EnumerationCountMismatch);
            }
            let expected_result = exact_enumeration_result_commitment_v1(
                partition_descriptor_ref,
                &source.candidate_memory_ids,
            )?;
            if evidence.enumeration_result_commitment != expected_result {
                return Err(
                    ReflectiveMemoryExactEnumerationErrorV1::EnumerationResultCommitmentMismatch,
                );
            }
            Some(expected_result)
        }
        CandidateSetCompletenessV1::BoundedSearch { .. }
        | CandidateSetCompletenessV1::UnknownCompleteness { .. } => None,
    };

    let mut hasher = blake3::Hasher::new();
    hasher.update(STRICT_CURRENT_BINDING_DOMAIN_V1);
    hash_str(&mut hasher, REFLECTIVE_MEMORY_EXACT_ENUMERATION_SCHEMA_V1);
    hash_str(&mut hasher, &current_binding);
    match exact_enumeration_binding {
        Some(commitment) => {
            hasher.update(&[1]);
            hash_str(&mut hasher, &commitment);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Ok(format!(
        "reflective-memory-current-candidate-strict:{}",
        hasher.finalize().to_hex()
    ))
}

fn validate_id(value: &str) -> Result<(), ReflectiveMemoryExactEnumerationErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 256 {
        Err(ReflectiveMemoryExactEnumerationErrorV1::InvalidIdentity)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), ReflectiveMemoryExactEnumerationErrorV1> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 {
        Err(ReflectiveMemoryExactEnumerationErrorV1::InvalidReference)
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
pub enum ReflectiveMemoryExactEnumerationErrorV1 {
    InvalidIdentity,
    InvalidReference,
    TooManyCandidates,
    CandidateIdsNotUnique,
    CandidateCountMismatch,
    MissingEnumerationEvidence,
    EnumerationCountMismatch,
    EnumerationResultCommitmentMismatch,
    CurrentIndexValidationFailed(ReflectiveMemoryIndexBuildErrorV1),
}
