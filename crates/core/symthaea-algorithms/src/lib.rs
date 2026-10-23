// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-first algorithm registry contracts for Symthaea.
//!
//! This crate is deliberately descriptive. It records problems, algorithm families,
//! implementations, and derivation lineage, but it does not execute implementations or grant
//! promotion/runtime authority.
//!
//! The central boundary is:
//!
//! ```text
//! Problem != Algorithm != Implementation != Evaluation != Evidence != Promotion != Authority
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use thiserror::Error;

const CONTENT_ID_PREFIX: &str = "sha256:";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RegistryError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("content id must use the sha256:<64 lowercase hex> form")]
    InvalidContentId,
    #[error("an implementation must reference the algorithm's exact problem identity")]
    ProblemMismatch,
    #[error("lineage candidate must match the implementation being validated")]
    CandidateMismatch,
    #[error("lineage contains its candidate as a parent")]
    SelfParent,
    #[error("duplicate lineage parent")]
    DuplicateParent,
    #[error("duplicate transformation identity")]
    DuplicateTransformation,
}

fn validate_text(field: &'static str, value: &str) -> Result<(), RegistryError> {
    if value.trim().is_empty() {
        return Err(RegistryError::Empty { field });
    }
    if value.chars().any(char::is_control) {
        return Err(RegistryError::ControlCharacters { field });
    }
    Ok(())
}

fn encode_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn digest_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Deterministic SHA-256 content identity.
///
/// Identity construction is domain-separated and length-prefixed so concatenation ambiguity is
/// impossible. The textual representation is always `sha256:<64 lowercase hex>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentId(String);

impl ContentId {
    pub fn derive<'a>(domain: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut hasher = Sha256::new();
        encode_part(&mut hasher, domain.as_bytes());
        for part in parts {
            encode_part(&mut hasher, part);
        }
        let digest = hasher.finalize();
        Self(format!("{CONTENT_ID_PREFIX}{}", digest_hex(&digest)))
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, RegistryError> {
        let value = value.into();
        let Some(hex) = value.strip_prefix(CONTENT_ID_PREFIX) else {
            return Err(RegistryError::InvalidContentId);
        };
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(RegistryError::InvalidContentId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub ContentId);

        impl $name {
            pub fn as_content_id(&self) -> &ContentId {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

typed_id!(ProblemId);
typed_id!(AlgorithmId);
typed_id!(ImplementationId);
typed_id!(TransformationId);
typed_id!(LineageId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeterminismRequirement {
    Required,
    Seeded,
    NotRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SemanticGuarantee {
    Exact,
    Approximate,
    Probabilistic,
}

/// Coarse discovery-risk class. This is not a safety certification; it only controls what the
/// discovery subsystem is allowed to experiment with automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiscoveryRisk {
    Ordinary,
    SecuritySensitive,
    SafetyCritical,
}

/// Semantic identity of a computational problem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemSpec {
    pub id: ProblemId,
    pub name: String,
    pub semantic_contract: String,
    pub guarantee: SemanticGuarantee,
    pub determinism: DeterminismRequirement,
    pub invariants: Vec<String>,
    pub risk: DiscoveryRisk,
}

impl ProblemSpec {
    pub fn new(
        name: impl Into<String>,
        semantic_contract: impl Into<String>,
        guarantee: SemanticGuarantee,
        determinism: DeterminismRequirement,
        invariants: Vec<String>,
        risk: DiscoveryRisk,
    ) -> Result<Self, RegistryError> {
        let name = name.into();
        let semantic_contract = semantic_contract.into();
        validate_text("problem name", &name)?;
        validate_text("semantic contract", &semantic_contract)?;
        for invariant in &invariants {
            validate_text("problem invariant", invariant)?;
        }

        let guarantee_tag = format!("{guarantee:?}");
        let determinism_tag = format!("{determinism:?}");
        let risk_tag = format!("{risk:?}");
        let mut parts: Vec<&[u8]> = vec![
            name.as_bytes(),
            semantic_contract.as_bytes(),
            guarantee_tag.as_bytes(),
            determinism_tag.as_bytes(),
            risk_tag.as_bytes(),
        ];
        for invariant in &invariants {
            parts.push(invariant.as_bytes());
        }
        let id = ProblemId(ContentId::derive("symthaea.problem.v1", parts));

        Ok(Self {
            id,
            name,
            semantic_contract,
            guarantee,
            determinism,
            invariants,
            risk,
        })
    }

    pub fn validate(&self) -> Result<(), RegistryError> {
        let rebuilt = Self::new(
            self.name.clone(),
            self.semantic_contract.clone(),
            self.guarantee,
            self.determinism,
            self.invariants.clone(),
            self.risk,
        )?;
        if rebuilt.id != self.id {
            return Err(RegistryError::InvalidContentId);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlgorithmProvenance {
    HumanAuthored,
    ImportedReference,
    Synthesized,
    Evolved,
    Rewritten,
    Hybrid,
}

/// One algorithm family/strategy. It has no executable handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlgorithmRecord {
    pub id: AlgorithmId,
    pub problem_id: ProblemId,
    pub name: String,
    pub description: String,
    pub provenance: AlgorithmProvenance,
}

impl AlgorithmRecord {
    pub fn new(
        problem_id: ProblemId,
        name: impl Into<String>,
        description: impl Into<String>,
        provenance: AlgorithmProvenance,
    ) -> Result<Self, RegistryError> {
        let name = name.into();
        let description = description.into();
        validate_text("algorithm name", &name)?;
        validate_text("algorithm description", &description)?;
        let provenance_tag = format!("{provenance:?}");
        let id = AlgorithmId(ContentId::derive(
            "symthaea.algorithm.v1",
            [
                problem_id.0.as_str().as_bytes(),
                name.as_bytes(),
                description.as_bytes(),
                provenance_tag.as_bytes(),
            ],
        ));
        Ok(Self {
            id,
            problem_id,
            name,
            description,
            provenance,
        })
    }
}

/// One concrete implementation of an algorithm family.
///
/// `source_ref` identifies the source/artifact location; `artifact_id` commits the exact bytes
/// or package supplied by the caller. Neither field is executable authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImplementationRecord {
    pub id: ImplementationId,
    pub problem_id: ProblemId,
    pub algorithm_id: AlgorithmId,
    pub source_ref: String,
    pub artifact_id: ContentId,
    pub target_profile: Option<String>,
}

impl ImplementationRecord {
    pub fn new(
        problem_id: ProblemId,
        algorithm_id: AlgorithmId,
        source_ref: impl Into<String>,
        artifact_id: ContentId,
        target_profile: Option<String>,
    ) -> Result<Self, RegistryError> {
        let source_ref = source_ref.into();
        validate_text("implementation source_ref", &source_ref)?;
        if let Some(profile) = &target_profile {
            validate_text("implementation target_profile", profile)?;
        }
        let profile = target_profile.as_deref().unwrap_or("");
        let id = ImplementationId(ContentId::derive(
            "symthaea.implementation.v1",
            [
                problem_id.0.as_str().as_bytes(),
                algorithm_id.0.as_str().as_bytes(),
                source_ref.as_bytes(),
                artifact_id.as_str().as_bytes(),
                profile.as_bytes(),
            ],
        ));
        Ok(Self {
            id,
            problem_id,
            algorithm_id,
            source_ref,
            artifact_id,
            target_profile,
        })
    }

    pub fn validate_for(&self, algorithm: &AlgorithmRecord) -> Result<(), RegistryError> {
        if self.problem_id != algorithm.problem_id || self.algorithm_id != algorithm.id {
            return Err(RegistryError::ProblemMismatch);
        }
        Ok(())
    }
}

/// Named transformation used to derive one candidate from another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationRecord {
    pub id: TransformationId,
    pub name: String,
    pub description: String,
}

impl TransformationRecord {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Self, RegistryError> {
        let name = name.into();
        let description = description.into();
        validate_text("transformation name", &name)?;
        validate_text("transformation description", &description)?;
        let id = TransformationId(ContentId::derive(
            "symthaea.transformation.v1",
            [name.as_bytes(), description.as_bytes()],
        ));
        Ok(Self {
            id,
            name,
            description,
        })
    }
}

/// Exact derivation lineage for a candidate implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlgorithmLineage {
    pub id: LineageId,
    pub candidate_id: ImplementationId,
    pub parent_ids: Vec<ImplementationId>,
    pub transformation_ids: Vec<TransformationId>,
}

impl AlgorithmLineage {
    pub fn new(
        candidate_id: ImplementationId,
        mut parent_ids: Vec<ImplementationId>,
        mut transformation_ids: Vec<TransformationId>,
    ) -> Result<Self, RegistryError> {
        if parent_ids.iter().any(|parent| parent == &candidate_id) {
            return Err(RegistryError::SelfParent);
        }

        parent_ids.sort();
        transformation_ids.sort();

        let parent_set: BTreeSet<_> = parent_ids.iter().collect();
        if parent_set.len() != parent_ids.len() {
            return Err(RegistryError::DuplicateParent);
        }
        let transform_set: BTreeSet<_> = transformation_ids.iter().collect();
        if transform_set.len() != transformation_ids.len() {
            return Err(RegistryError::DuplicateTransformation);
        }

        let mut owned_parts = Vec::with_capacity(1 + parent_ids.len() + transformation_ids.len());
        owned_parts.push(candidate_id.0.as_str().as_bytes().to_vec());
        owned_parts.extend(
            parent_ids
                .iter()
                .map(|id| id.0.as_str().as_bytes().to_vec()),
        );
        owned_parts.extend(
            transformation_ids
                .iter()
                .map(|id| id.0.as_str().as_bytes().to_vec()),
        );
        let id = LineageId(ContentId::derive(
            "symthaea.algorithm-lineage.v1",
            owned_parts.iter().map(Vec::as_slice),
        ));

        Ok(Self {
            id,
            candidate_id,
            parent_ids,
            transformation_ids,
        })
    }

    pub fn validate_for(&self, implementation: &ImplementationRecord) -> Result<(), RegistryError> {
        if self.candidate_id != implementation.id {
            return Err(RegistryError::CandidateMismatch);
        }
        let rebuilt = Self::new(
            self.candidate_id.clone(),
            self.parent_ids.clone(),
            self.transformation_ids.clone(),
        )?;
        if rebuilt.id != self.id {
            return Err(RegistryError::InvalidContentId);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> ProblemSpec {
        ProblemSpec::new(
            "binary-hdc-hamming-distance",
            "Return the exact Hamming distance between two equal-width binary hypervectors.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["result <= vector width".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap()
    }

    #[test]
    fn content_ids_are_domain_separated_and_deterministic() {
        let a = ContentId::derive("a", [b"same".as_slice()]);
        let a_again = ContentId::derive("a", [b"same".as_slice()]);
        let b = ContentId::derive("b", [b"same".as_slice()]);
        assert_eq!(a, a_again);
        assert_ne!(a, b);
        assert!(ContentId::parse(a.to_string()).is_ok());
    }

    #[test]
    fn problem_identity_changes_with_semantics() {
        let first = problem();
        let second = ProblemSpec::new(
            first.name.clone(),
            "Return an approximate Hamming distance.",
            SemanticGuarantee::Approximate,
            first.determinism,
            first.invariants.clone(),
            first.risk,
        )
        .unwrap();
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn implementation_is_bound_to_exact_problem_and_algorithm() {
        let problem = problem();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "scalar-popcount",
            "Reference scalar popcount reduction.",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let implementation = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id.clone(),
            "crates/core/symthaea-core/src/hdc/binary_hv.rs",
            ContentId::derive("test-artifact", [b"reference".as_slice()]),
            None,
        )
        .unwrap();
        assert!(implementation.validate_for(&algorithm).is_ok());
    }

    #[test]
    fn lineage_is_order_independent_but_rejects_duplicates() {
        let candidate = ImplementationId(ContentId::derive("impl", [b"candidate".as_slice()]));
        let p1 = ImplementationId(ContentId::derive("impl", [b"p1".as_slice()]));
        let p2 = ImplementationId(ContentId::derive("impl", [b"p2".as_slice()]));
        let t1 = TransformationId(ContentId::derive("transform", [b"t1".as_slice()]));
        let t2 = TransformationId(ContentId::derive("transform", [b"t2".as_slice()]));

        let a = AlgorithmLineage::new(
            candidate.clone(),
            vec![p1.clone(), p2.clone()],
            vec![t1.clone(), t2.clone()],
        )
        .unwrap();
        let b = AlgorithmLineage::new(
            candidate.clone(),
            vec![p2, p1.clone()],
            vec![t2, t1],
        )
        .unwrap();
        assert_eq!(a.id, b.id);

        let duplicate = AlgorithmLineage::new(candidate, vec![p1.clone(), p1], vec![]);
        assert_eq!(duplicate.unwrap_err(), RegistryError::DuplicateParent);
    }

    #[test]
    fn serde_cannot_turn_presence_into_authority() {
        let problem = problem();
        let json = serde_json::to_string(&problem).unwrap();
        let decoded: ProblemSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, problem);
        assert!(decoded.validate().is_ok());
        // There is intentionally no execute(), promote(), activate(), or function-pointer field.
    }
}
