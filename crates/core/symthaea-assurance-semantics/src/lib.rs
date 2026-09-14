// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable self-describing semantic-definition commitments for Symthaea Assurance.
//!
//! Governing theorem:
//!
//! ```text
//! semantic id + definition SHA-256
//!     != self-describing semantic commitment
//!
//! schema label
//!     != immutable schema specification
//! ```
//!
//! Both the definition schema and its exact specification commitment are
//! identity-bearing. This crate does not establish that definition/schema bytes
//! are available, semantically adequate, or trusted.

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
pub struct DefinitionSchemaV1 {
    schema_id: StableId,
    specification_digest: DigestSha256,
}

impl DefinitionSchemaV1 {
    pub fn new(schema_id: StableId, specification_digest: DigestSha256) -> Self {
        Self {
            schema_id,
            specification_digest,
        }
    }

    pub fn schema_id(&self) -> &StableId {
        &self.schema_id
    }

    pub fn specification_digest(&self) -> &DigestSha256 {
        &self.specification_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticCommitmentV1 {
    semantic_id: StableId,
    definition_schema: DefinitionSchemaV1,
    definition_digest: DigestSha256,
}

impl SemanticCommitmentV1 {
    pub fn new(
        semantic_id: StableId,
        definition_schema: DefinitionSchemaV1,
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

    pub fn definition_schema(&self) -> &DefinitionSchemaV1 {
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
            "definition-schema-id",
            self.definition_schema.schema_id.as_str(),
        );
        field(
            &mut out,
            "definition-schema-specification-digest",
            self.definition_schema.specification_digest.as_str(),
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
    mut values: Vec<SemanticCommitmentV1>,
) -> Result<Vec<SemanticCommitmentV1>, SemanticCommitmentError> {
    values.sort_by_cached_key(|value| value.semantic_id.clone());
    for pair in values.windows(2) {
        if pair[0].semantic_id == pair[1].semantic_id {
            return Err(SemanticCommitmentError::DuplicateSemanticId(
                pair[0].semantic_id.as_str().to_owned(),
            ));
        }
    }
    Ok(values)
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
