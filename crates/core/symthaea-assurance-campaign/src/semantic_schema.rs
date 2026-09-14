// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-002A: self-describing semantic-definition commitments.
//!
//! This module is intentionally staged beside the ASSURE-002 kernel before
//! integration. It closes one specific ambiguity:
//!
//! ```text
//! semantic id + definition SHA-256
//!     != self-describing semantic commitment
//! ```
//!
//! The definition schema is identity-bearing. This module does not establish
//! that definition bytes are available, semantically adequate, or trusted.

use sha2::{Digest, Sha256};
use symthaea_assurance_core::{DigestSha256, StableId};
use thiserror::Error;

pub const SEMANTIC_COMMITMENT_SCHEMA: &str = "symthaea.assurance.semantic-commitment.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SemanticCommitmentError {
    #[error("duplicate semantic identifier: {0}")]
    DuplicateSemanticId(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SelfDescribingSemanticCommitmentV1 {
    semantic_id: StableId,
    definition_schema: StableId,
    definition_digest: DigestSha256,
}

impl SelfDescribingSemanticCommitmentV1 {
    pub fn new(
        semantic_id: StableId,
        definition_schema: StableId,
        definition_digest: DigestSha256,
    ) -> Self {
        Self {
            semantic_id,
            definition_schema,
            definition_digest,
        }
    }

    pub fn semantic_id(&self) -> &StableId {
        &self.semantic_id
    }

    pub fn definition_schema(&self) -> &StableId {
        &self.definition_schema
    }

    pub fn definition_digest(&self) -> &DigestSha256 {
        &self.definition_digest
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-semantic-commitment-v1\n");
        field(&mut out, "schema", SEMANTIC_COMMITMENT_SCHEMA);
        field(&mut out, "semantic-id", self.semantic_id.as_str());
        field(
            &mut out,
            "definition-schema",
            self.definition_schema.as_str(),
        );
        field(
            &mut out,
            "definition-digest",
            self.definition_digest.as_str(),
        );
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

/// Canonicalizes a semantic set by semantic identifier and rejects duplicate
/// identifiers even when callers supply different schemas or definition
/// digests. A semantic identifier names one slot in one declared set.
pub fn canonical_semantic_set(
    mut values: Vec<SelfDescribingSemanticCommitmentV1>,
) -> Result<Vec<SelfDescribingSemanticCommitmentV1>, SemanticCommitmentError> {
    values.sort_by(|left, right| left.semantic_id.cmp(&right.semantic_id));
    for pair in values.windows(2) {
        if pair[0].semantic_id == pair[1].semantic_id {
            return Err(SemanticCommitmentError::DuplicateSemanticId(
                pair[0].semantic_id.as_str().to_owned(),
            ));
        }
    }
    Ok(values)
}

/// Identity required before two externally ordered events may be compared.
/// Sequence numbers remain event data; source, validation semantics, and epoch
/// define the lineage in which those sequence numbers have meaning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderingLineageIdentityV1 {
    source: StableId,
    validation_profile: SelfDescribingSemanticCommitmentV1,
    epoch: u64,
}

impl OrderingLineageIdentityV1 {
    pub fn new(
        source: StableId,
        validation_profile: SelfDescribingSemanticCommitmentV1,
        epoch: u64,
    ) -> Self {
        Self {
            source,
            validation_profile,
            epoch,
        }
    }

    pub fn source(&self) -> &StableId {
        &self.source
    }

    pub fn validation_profile(&self) -> &SelfDescribingSemanticCommitmentV1 {
        &self.validation_profile
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn is_comparable_with(&self, other: &Self) -> bool {
        self == other
    }
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}

fn digest_canonical(bytes: &[u8]) -> DigestSha256 {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    DigestSha256::new(encoded).expect("SHA-256 encoding is always valid")
}
