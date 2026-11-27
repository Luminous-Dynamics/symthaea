// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical payload binding for AS-Bench Lab API v1 design batches.
//!
//! This module does not implement the AS-Bench client, catalog parser, job lifecycle or verifier.
//! Those remain owned by #10947/#10921. It provides a deterministic SHA-256 identity for the
//! exact ordered experiment request so a scientific DecisionProposal and later EvaluatorResolution
//! can bind to the same batch.
//!
//! serde_json serialization of this fixed-field struct is the canonical byte contract for schema
//! version 1. Changing field order, names, representation or semantics requires a schema-version
//! increment; design ID order is preserved exactly as submitted.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const ASB_BATCH_PAYLOAD_SCHEMA_VERSION: u16 = 1;
pub const ASB_LAB_API_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsBenchBatchPayloadV1 {
    pub schema_version: u16,
    pub lab_api_version: u32,
    pub compatibility_manifest_sha256: String,
    pub campaign_id: String,
    pub stage: u32,
    pub design_id_field: String,
    /// Exact ordered IDs sent to POST /v1/experiments. Never sort before hashing.
    pub design_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsBenchPayloadError {
    UnsupportedSchemaVersion,
    UnsupportedLabApiVersion,
    InvalidManifestDigest,
    EmptyCampaignId,
    InvalidStage,
    EmptyDesignIdField,
    EmptyBatch,
    BatchSizeMismatch { expected: usize, actual: usize },
    EmptyDesignId { index: usize },
    DuplicateDesignId(String),
    DesignNotInPermittedCatalog(String),
    Serialization(String),
}

impl std::fmt::Display for AsBenchPayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AsBenchPayloadError {}

impl AsBenchBatchPayloadV1 {
    pub fn validate(
        &self,
        expected_batch_size: usize,
        permitted_design_ids: &[String],
    ) -> Result<(), AsBenchPayloadError> {
        if self.schema_version != ASB_BATCH_PAYLOAD_SCHEMA_VERSION {
            return Err(AsBenchPayloadError::UnsupportedSchemaVersion);
        }
        if self.lab_api_version != ASB_LAB_API_VERSION {
            return Err(AsBenchPayloadError::UnsupportedLabApiVersion);
        }
        if !is_sha256(&self.compatibility_manifest_sha256) {
            return Err(AsBenchPayloadError::InvalidManifestDigest);
        }
        if self.campaign_id.trim().is_empty() {
            return Err(AsBenchPayloadError::EmptyCampaignId);
        }
        if self.stage == 0 {
            return Err(AsBenchPayloadError::InvalidStage);
        }
        if self.design_id_field.trim().is_empty() {
            return Err(AsBenchPayloadError::EmptyDesignIdField);
        }
        if self.design_ids.is_empty() {
            return Err(AsBenchPayloadError::EmptyBatch);
        }
        if self.design_ids.len() != expected_batch_size {
            return Err(AsBenchPayloadError::BatchSizeMismatch {
                expected: expected_batch_size,
                actual: self.design_ids.len(),
            });
        }

        let permitted: HashSet<&str> = permitted_design_ids.iter().map(String::as_str).collect();
        let mut seen = HashSet::with_capacity(self.design_ids.len());
        for (index, design_id) in self.design_ids.iter().enumerate() {
            if design_id.trim().is_empty() {
                return Err(AsBenchPayloadError::EmptyDesignId { index });
            }
            if !seen.insert(design_id.as_str()) {
                return Err(AsBenchPayloadError::DuplicateDesignId(design_id.clone()));
            }
            if !permitted.contains(design_id.as_str()) {
                return Err(AsBenchPayloadError::DesignNotInPermittedCatalog(
                    design_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// SHA-256 over the exact versioned JSON request identity.
    ///
    /// The struct uses no maps, so field ordering is deterministic. The design_ids vector
    /// retains request order. Call validate first with the exact stage batch size/catalog.
    pub fn payload_sha256(&self) -> Result<String, AsBenchPayloadError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| AsBenchPayloadError::Serialization(error.to_string()))?;
        let digest = Sha256::digest(bytes);
        Ok(encode_hex(&digest))
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(ids: &[&str]) -> AsBenchBatchPayloadV1 {
        AsBenchBatchPayloadV1 {
            schema_version: ASB_BATCH_PAYLOAD_SCHEMA_VERSION,
            lab_api_version: ASB_LAB_API_VERSION,
            compatibility_manifest_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            campaign_id: "deepmind/nuclease-active-learning".into(),
            stage: 2,
            design_id_field: "variant_id".into(),
            design_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
        }
    }

    fn permitted() -> Vec<String> {
        ["variant-a", "variant-b", "variant-c"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn computes_stable_sha256_for_exact_payload() {
        let first = batch(&["variant-a", "variant-b"]);
        let second = batch(&["variant-a", "variant-b"]);
        first.validate(2, &permitted()).unwrap();
        second.validate(2, &permitted()).unwrap();
        assert_eq!(first.payload_sha256().unwrap(), second.payload_sha256().unwrap());
        assert_eq!(first.payload_sha256().unwrap().len(), 64);
    }

    #[test]
    fn submitted_design_order_is_part_of_payload_identity() {
        let forward = batch(&["variant-a", "variant-b"]);
        let reverse = batch(&["variant-b", "variant-a"]);
        forward.validate(2, &permitted()).unwrap();
        reverse.validate(2, &permitted()).unwrap();
        assert_ne!(forward.payload_sha256().unwrap(), reverse.payload_sha256().unwrap());
    }

    #[test]
    fn campaign_stage_and_design_field_are_bound() {
        let base = batch(&["variant-a", "variant-b"]);
        let mut other = base.clone();
        other.campaign_id = "propylene-active-learning".into();
        assert_ne!(base.payload_sha256().unwrap(), other.payload_sha256().unwrap());
        other = base.clone();
        other.stage = 3;
        assert_ne!(base.payload_sha256().unwrap(), other.payload_sha256().unwrap());
        other = base.clone();
        other.design_id_field = "design_id".into();
        assert_ne!(base.payload_sha256().unwrap(), other.payload_sha256().unwrap());
    }

    #[test]
    fn rejects_duplicate_or_non_catalog_designs() {
        let duplicated = batch(&["variant-a", "variant-a"]);
        assert_eq!(
            duplicated.validate(2, &permitted()).unwrap_err(),
            AsBenchPayloadError::DuplicateDesignId("variant-a".into())
        );
        let unknown = batch(&["variant-a", "not-in-catalog"]);
        assert_eq!(
            unknown.validate(2, &permitted()).unwrap_err(),
            AsBenchPayloadError::DesignNotInPermittedCatalog("not-in-catalog".into())
        );
    }

    #[test]
    fn rejects_batch_size_and_manifest_digest_mismatch() {
        let value = batch(&["variant-a", "variant-b"]);
        assert_eq!(
            value.validate(96, &permitted()).unwrap_err(),
            AsBenchPayloadError::BatchSizeMismatch {
                expected: 96,
                actual: 2
            }
        );
        let mut invalid_digest = value;
        invalid_digest.compatibility_manifest_sha256 = "sha256:abc".into();
        assert_eq!(
            invalid_digest.validate(2, &permitted()).unwrap_err(),
            AsBenchPayloadError::InvalidManifestDigest
        );
    }
}
