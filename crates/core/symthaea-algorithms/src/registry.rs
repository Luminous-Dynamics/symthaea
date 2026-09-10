use serde::{Deserialize, Deserializer, Serialize};
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
    #[error("{kind} identity does not match its canonical fields")]
    IdentityMismatch { kind: &'static str },
    #[error("implementation does not match the exact algorithm/problem pair")]
    ProblemMismatch,
    #[error("lineage candidate does not match the implementation")]
    CandidateMismatch,
    #[error("lineage cannot contain its candidate as a parent")]
    SelfParent,
    #[error("duplicate lineage parent")]
    DuplicateParent,
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

/// Deterministic, domain-separated SHA-256 content identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ContentId(String);

impl ContentId {
    pub fn derive<'a>(domain: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut hasher = Sha256::new();
        encode_part(&mut hasher, domain.as_bytes());
        for part in parts {
            encode_part(&mut hasher, part);
        }
        Self(format!("{CONTENT_ID_PREFIX}{}", digest_hex(&hasher.finalize())))
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

impl<'de> Deserialize<'de> for ContentId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
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

/// Discovery policy class only; this is not a safety certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiscoveryRisk {
    Ordinary,
    SecuritySensitive,
    SafetyCritical,
}

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
        parts.extend(invariants.iter().map(|s| s.as_bytes()));
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
        if rebuilt.id == self.id {
            Ok(())
        } else {
            Err(RegistryError::IdentityMismatch { kind: "problem" })
        }
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

    pub fn validate(&self) -> Result<(), RegistryError> {
        let rebuilt = Self::new(
            self.problem_id.clone(),
            self.name.clone(),
            self.description.clone(),
            self.provenance,
        )?;
        if rebuilt.id == self.id {
            Ok(())
        } else {
            Err(RegistryError::IdentityMismatch { kind: "algorithm" })
        }
    }
}

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

    pub fn validate(&self) -> Result<(), RegistryError> {
        let rebuilt = Self::new(
            self.problem_id.clone(),
            self.algorithm_id.clone(),
            self.source_ref.clone(),
            self.artifact_id.clone(),
            self.target_profile.clone(),
        )?;
        if rebuilt.id == self.id {
            Ok(())
        } else {
            Err(RegistryError::IdentityMismatch {
                kind: "implementation",
            })
        }
    }

    pub fn validate_for(&self, algorithm: &AlgorithmRecord) -> Result<(), RegistryError> {
        self.validate()?;
        algorithm.validate()?;
        if self.problem_id == algorithm.problem_id && self.algorithm_id == algorithm.id {
            Ok(())
        } else {
            Err(RegistryError::ProblemMismatch)
        }
    }
}

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

    pub fn validate(&self) -> Result<(), RegistryError> {
        let rebuilt = Self::new(self.name.clone(), self.description.clone())?;
        if rebuilt.id == self.id {
            Ok(())
        } else {
            Err(RegistryError::IdentityMismatch {
                kind: "transformation",
            })
        }
    }
}

/// Exact derivation lineage for one implementation candidate.
///
/// Parent identities are a canonical unordered ancestry set. Transformation identities are an
/// ordered derivation trace: order is semantically meaningful and the same transformation may
/// appear multiple times. This distinction is essential for rewrite/e-graph search, where
/// `T1 -> T2` need not produce the same candidate as `T2 -> T1` and a rule may fire repeatedly.
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
        transformation_ids: Vec<TransformationId>,
    ) -> Result<Self, RegistryError> {
        if parent_ids.iter().any(|parent| parent == &candidate_id) {
            return Err(RegistryError::SelfParent);
        }
        parent_ids.sort();
        if parent_ids.iter().collect::<BTreeSet<_>>().len() != parent_ids.len() {
            return Err(RegistryError::DuplicateParent);
        }

        let parent_count = (parent_ids.len() as u64).to_be_bytes();
        let transformation_count = (transformation_ids.len() as u64).to_be_bytes();
        let mut owned = Vec::with_capacity(5 + parent_ids.len() + transformation_ids.len());
        owned.push(candidate_id.0.as_str().as_bytes().to_vec());
        owned.push(b"parents".to_vec());
        owned.push(parent_count.to_vec());
        owned.extend(parent_ids.iter().map(|id| id.0.as_str().as_bytes().to_vec()));
        owned.push(b"transformations".to_vec());
        owned.push(transformation_count.to_vec());
        owned.extend(
            transformation_ids
                .iter()
                .map(|id| id.0.as_str().as_bytes().to_vec()),
        );
        let id = LineageId(ContentId::derive(
            "symthaea.algorithm-lineage.v2",
            owned.iter().map(Vec::as_slice),
        ));
        Ok(Self {
            id,
            candidate_id,
            parent_ids,
            transformation_ids,
        })
    }

    pub fn validate(&self) -> Result<(), RegistryError> {
        let rebuilt = Self::new(
            self.candidate_id.clone(),
            self.parent_ids.clone(),
            self.transformation_ids.clone(),
        )?;
        if rebuilt.id == self.id
            && rebuilt.parent_ids == self.parent_ids
            && rebuilt.transformation_ids == self.transformation_ids
        {
            Ok(())
        } else {
            Err(RegistryError::IdentityMismatch { kind: "lineage" })
        }
    }

    pub fn validate_for(&self, implementation: &ImplementationRecord) -> Result<(), RegistryError> {
        self.validate()?;
        implementation.validate()?;
        if self.candidate_id == implementation.id {
            Ok(())
        } else {
            Err(RegistryError::CandidateMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> ProblemSpec {
        ProblemSpec::new(
            "binary-hdc-hamming-distance",
            "Return the exact Hamming distance between equal-width binary hypervectors.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["result <= vector width".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap()
    }

    #[test]
    fn identities_are_domain_separated_and_deterministic() {
        let a = ContentId::derive("a", [b"same".as_slice()]);
        assert_eq!(a, ContentId::derive("a", [b"same".as_slice()]));
        assert_ne!(a, ContentId::derive("b", [b"same".as_slice()]));
        assert!(ContentId::parse(a.to_string()).is_ok());
    }

    #[test]
    fn serde_rejects_malformed_content_ids() {
        let json = r#""not-a-content-id""#;
        assert!(serde_json::from_str::<ContentId>(json).is_err());
    }

    #[test]
    fn semantic_change_changes_problem_identity() {
        let exact = problem();
        let approximate = ProblemSpec::new(
            exact.name.clone(),
            "Return an approximate Hamming distance.",
            SemanticGuarantee::Approximate,
            exact.determinism,
            exact.invariants.clone(),
            exact.risk,
        )
        .unwrap();
        assert_ne!(exact.id, approximate.id);
    }

    #[test]
    fn implementation_validation_detects_redigested_field_substitution() {
        let problem = problem();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "scalar-popcount",
            "Reference scalar popcount reduction.",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let mut implementation = ImplementationRecord::new(
            problem.id,
            algorithm.id.clone(),
            "crates/core/symthaea-core/src/hdc/binary_hv.rs",
            ContentId::derive("test-artifact", [b"reference".as_slice()]),
            None,
        )
        .unwrap();
        assert!(implementation.validate_for(&algorithm).is_ok());
        implementation.source_ref = "candidate://substituted".into();
        assert!(matches!(
            implementation.validate(),
            Err(RegistryError::IdentityMismatch {
                kind: "implementation"
            })
        ));
    }

    #[test]
    fn parent_order_is_canonical_but_transformation_order_is_not() {
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
        let same_parent_set = AlgorithmLineage::new(
            candidate.clone(),
            vec![p2, p1.clone()],
            vec![t1.clone(), t2.clone()],
        )
        .unwrap();
        let reversed_transformations = AlgorithmLineage::new(
            candidate.clone(),
            a.parent_ids.clone(),
            vec![t2, t1.clone()],
        )
        .unwrap();

        assert_eq!(a.id, same_parent_set.id);
        assert_eq!(a.parent_ids, same_parent_set.parent_ids);
        assert_ne!(a.id, reversed_transformations.id);
        assert_ne!(a.transformation_ids, reversed_transformations.transformation_ids);
        assert_eq!(
            AlgorithmLineage::new(candidate, vec![p1.clone(), p1], vec![]).unwrap_err(),
            RegistryError::DuplicateParent
        );
    }

    #[test]
    fn repeated_transformation_steps_are_preserved() {
        let candidate = ImplementationId(ContentId::derive("impl", [b"candidate".as_slice()]));
        let t = TransformationId(ContentId::derive("transform", [b"repeatable".as_slice()]));
        let once = AlgorithmLineage::new(candidate.clone(), vec![], vec![t.clone()]).unwrap();
        let twice = AlgorithmLineage::new(candidate, vec![], vec![t.clone(), t]).unwrap();
        assert_ne!(once.id, twice.id);
        assert_eq!(twice.transformation_ids.len(), 2);
        assert!(twice.validate().is_ok());
    }

    #[test]
    fn noncanonical_parent_order_fails_self_validation() {
        let candidate = ImplementationId(ContentId::derive("impl", [b"candidate".as_slice()]));
        let p1 = ImplementationId(ContentId::derive("impl", [b"p1".as_slice()]));
        let p2 = ImplementationId(ContentId::derive("impl", [b"p2".as_slice()]));
        let mut lineage = AlgorithmLineage::new(candidate, vec![p1, p2], vec![]).unwrap();
        lineage.parent_ids.reverse();
        assert!(matches!(
            lineage.validate(),
            Err(RegistryError::IdentityMismatch { kind: "lineage" })
        ));
    }
}
