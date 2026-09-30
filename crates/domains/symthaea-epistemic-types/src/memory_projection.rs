use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MEMORY_PROJECTION_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind { Working, Episodic, Semantic, Procedural, KnowledgeGraph, Vector, Hdc }

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryProjectionRef {
    pub schema_version: u16,
    pub canonical_identity: String,
    pub memory_kind: MemoryKind,
    pub projection_profile: String,
    pub representation_digest: String,
    pub source_frontier: Option<String>,
}

impl MemoryProjectionRef {
    pub fn new(canonical_identity: impl Into<String>, memory_kind: MemoryKind, projection_profile: impl Into<String>, representation_bytes: &[u8], source_frontier: Option<String>) -> Self {
        Self {
            schema_version: MEMORY_PROJECTION_SCHEMA_VERSION,
            canonical_identity: canonical_identity.into(),
            memory_kind,
            projection_profile: projection_profile.into(),
            representation_digest: sha256_hex(representation_bytes),
            source_frontier,
        }
    }

    pub fn canonical_identity_digest(&self) -> String {
        sha256_hex(self.canonical_identity.as_bytes())
    }

    pub fn projection_identity_digest(&self) -> String {
        let projection = (self.schema_version, &self.canonical_identity, self.memory_kind, &self.projection_profile, &self.source_frontier);
        let bytes = serde_json::to_vec(&projection).expect("projection fields are serializable");
        sha256_hex(&bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryProvenance {
    /// Stable identity of the canonical semantic object when one is known.
    /// This is deliberately optional: local cognitive memory may exist before admission to the canonical fabric.
    pub canonical_identity: Option<String>,
    /// Stable identity of this memory record across persistence/retrieval projections.
    pub memory_id: String,
    pub memory_kind: MemoryKind,
    pub created_at: String,
    pub source_event: Option<String>,
    pub canonical_artifact_ref: Option<String>,
    pub statement_ref: Option<String>,
    pub provenance_family: Option<String>,
    pub epistemic_state: Option<String>,
    pub claim_ceiling: Option<String>,
    pub frontier_ref: Option<String>,
    pub derivation_ref: Option<String>,
    pub model_ref: Option<String>,
    pub retrieval_index_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProvenanceRelationKind {
    DerivedFrom,
    RevisedFrom,
    Supersedes,
    Contradicts,
    Corroborates,
    RepresentationOf,
}

impl ProvenanceRelationKind {
    /// Stable wire/digest label. Do not derive snapshot ordering from Rust enum discriminants.
    pub const fn stable_code(self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::RevisedFrom => "revised_from",
            Self::Supersedes => "supersedes",
            Self::Contradicts => "contradicts",
            Self::Corroborates => "corroborates",
            Self::RepresentationOf => "representation_of",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProvenanceRelation {
    pub source_memory_id: String,
    pub target_memory_id: String,
    pub kind: ProvenanceRelationKind,
    pub created_at: String,
}

impl ProvenanceRelation {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.source_memory_id.trim().is_empty() || self.target_memory_id.trim().is_empty() {
            return Err("provenance relation memory identities must be non-empty");
        }
        if self.source_memory_id == self.target_memory_id {
            return Err("provenance relation cannot self-reference");
        }
        if self.created_at.trim().is_empty() {
            return Err("provenance relation created_at must be non-empty");
        }
        Ok(())
    }
}

/// Version of the local structural provenance validator. Bump when validation
/// semantics change; this is intentionally independent of epistemic truth assessment.
pub const PROVENANCE_VALIDATOR_VERSION: &str = "melothaea-provenance-structural-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceValidationViolation {
    pub code: String,
    pub source_memory_id: Option<String>,
    pub target_memory_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceValidationReport {
    pub validator_version: String,
    pub snapshot_digest: String,
    pub relation_count: usize,
    pub conforms: bool,
    pub violations: Vec<ProvenanceValidationViolation>,
}

impl ProvenanceValidationReport {
    pub fn from_relations(relations: &[ProvenanceRelation]) -> Self {
        let mut canonical = relations.to_vec();
        canonical.sort_by(|a, b| {
            (&a.source_memory_id, &a.target_memory_id, a.kind.stable_code(), &a.created_at)
                .cmp(&(&b.source_memory_id, &b.target_memory_id, b.kind.stable_code(), &b.created_at))
        });

        let bytes = serde_json::to_vec(&canonical).expect("provenance relations are serializable");
        Self {
            validator_version: PROVENANCE_VALIDATOR_VERSION.to_owned(),
            snapshot_digest: sha256_hex(&bytes),
            relation_count: relations.len(),
            conforms: true,
            violations: Vec::new(),
        }
    }

    pub fn with_violations(mut self, violations: Vec<ProvenanceValidationViolation>) -> Self {
        self.conforms = violations.is_empty();
        self.violations = violations;
        self
    }
}

impl MemoryProvenance {
    pub fn provenance_identity(&self) -> Option<&str> { self.provenance_family.as_deref() }

    /// Structural validity only. This does not assert truth, reliability, or admission.
    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.memory_id.trim().is_empty() {
            return Err("memory_id must be non-empty");
        }
        if self.created_at.trim().is_empty() {
            return Err("created_at must be non-empty");
        }
        if let Some(family) = &self.provenance_family {
            if family.trim().is_empty() {
                return Err("provenance_family must be non-empty when present");
            }
        }
        if let Some(canonical) = &self.canonical_identity {
            if canonical.trim().is_empty() {
                return Err("canonical_identity must be non-empty when present");
            }
        }
        Ok(())
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representations_share_canonical_identity_but_not_projection_identity() {
        let a = MemoryProjectionRef::new("claim:abc", MemoryKind::Semantic, "semantic-v1", b"summary-a", None);
        let b = MemoryProjectionRef::new("claim:abc", MemoryKind::Vector, "embedding-v1", b"embedding-b", None);
        assert_eq!(a.canonical_identity_digest(), b.canonical_identity_digest());
        assert_ne!(a.representation_digest, b.representation_digest);
        assert_ne!(a.projection_identity_digest(), b.projection_identity_digest());
    }

    #[test]
    fn schema_version_changes_projection_not_canonical_identity() {
        let a = MemoryProjectionRef::new("claim:abc", MemoryKind::Semantic, "semantic-v1", b"same", None);
        let mut b = a.clone();
        b.schema_version += 1;
        assert_eq!(a.canonical_identity_digest(), b.canonical_identity_digest());
        assert_ne!(a.projection_identity_digest(), b.projection_identity_digest());
    }

    #[test]
    fn provenance_family_is_stable_across_retrieval_indexes() {
        let a = MemoryProvenance {
            canonical_identity: Some("claim:abc".into()),
            memory_id: "mem-a".into(), memory_kind: MemoryKind::Vector,
            created_at: "2026-09-29T00:00:00Z".into(),
            source_event: Some("event-1".into()), canonical_artifact_ref: Some("artifact-1".into()),
            statement_ref: Some("statement-1".into()), provenance_family: Some("family-1".into()),
            epistemic_state: Some("Observed".into()), claim_ceiling: Some("source-scoped".into()),
            frontier_ref: Some("frontier-1".into()), derivation_ref: None, model_ref: None,
            retrieval_index_ref: Some("index-a".into()),
        };
        let mut b = a.clone();
        b.memory_id = "mem-b".into();
        b.retrieval_index_ref = Some("index-b".into());
        assert_eq!(a.provenance_identity(), b.provenance_identity());
    }

    #[test]
    fn structural_validation_rejects_missing_identity() {
        let mut p = MemoryProvenance {
            canonical_identity: None,
            memory_id: "mem-1".into(),
            memory_kind: MemoryKind::KnowledgeGraph,
            created_at: "cycle:1".into(),
            source_event: None,
            canonical_artifact_ref: None,
            statement_ref: None,
            provenance_family: Some("family-1".into()),
            epistemic_state: None,
            claim_ceiling: None,
            frontier_ref: None,
            derivation_ref: None,
            model_ref: None,
            retrieval_index_ref: None,
        };
        assert!(p.validate_structure().is_ok());

        p.memory_id.clear();
        assert_eq!(p.validate_structure(), Err("memory_id must be non-empty"));
    }

    #[test]
    fn provenance_relations_are_typed_and_non_self_referential() {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        assert!(relation.validate().is_ok());

        let mut self_relation = relation.clone();
        self_relation.target_memory_id = self_relation.source_memory_id.clone();
        assert_eq!(self_relation.validate(), Err("provenance relation cannot self-reference"));
    }

    #[test]
    fn validation_report_binds_to_order_independent_snapshot() {
        let a = ProvenanceRelation {
            source_memory_id: "derived".into(), target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom, created_at: "cycle:2".into(),
        };
        let b = ProvenanceRelation {
            source_memory_id: "revision".into(), target_memory_id: "derived".into(),
            kind: ProvenanceRelationKind::RevisedFrom, created_at: "cycle:3".into(),
        };
        let first = ProvenanceValidationReport::from_relations(&[a.clone(), b.clone()]);
        let second = ProvenanceValidationReport::from_relations(&[b, a]);
        assert_eq!(first.snapshot_digest, second.snapshot_digest);
        assert_eq!(first.relation_count, 2);
        assert!(first.conforms);
        assert!(first.violations.is_empty());
        assert_eq!(first.validator_version, PROVENANCE_VALIDATOR_VERSION);
    }

    #[test]
    fn provenance_relation_codes_are_explicit_and_unique() {
        let codes = [
            ProvenanceRelationKind::DerivedFrom.stable_code(),
            ProvenanceRelationKind::RevisedFrom.stable_code(),
            ProvenanceRelationKind::Supersedes.stable_code(),
            ProvenanceRelationKind::Contradicts.stable_code(),
            ProvenanceRelationKind::Corroborates.stable_code(),
            ProvenanceRelationKind::RepresentationOf.stable_code(),
        ];
        let unique: std::collections::HashSet<_> = codes.into_iter().collect();
        assert_eq!(unique.len(), 6);
        assert_eq!(ProvenanceRelationKind::DerivedFrom.stable_code(), "derived_from");
        assert_eq!(ProvenanceRelationKind::RevisedFrom.stable_code(), "revised_from");
    }

    #[test]
    fn revision_is_a_distinct_typed_relation() {
        let relation = ProvenanceRelation {
            source_memory_id: "revision".into(),
            target_memory_id: "original".into(),
            kind: ProvenanceRelationKind::RevisedFrom,
            created_at: "cycle:3".into(),
        };
        assert_eq!(relation.kind, ProvenanceRelationKind::RevisedFrom);
        assert!(relation.validate().is_ok());
    }

    #[test]
    fn sha256_is_deterministic() {
        assert_eq!(sha256_hex(b"symthaea-memory"), sha256_hex(b"symthaea-memory"));
    }
}
