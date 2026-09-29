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
        crate::canonical_identity::canonical_identity_digest(&self.canonical_identity)
            .expect("canonical memory identity is bounded by the canonical encoder")
    }

    pub fn projection_identity_digest(&self) -> String {
        crate::canonical_identity::projection_identity_digest(
            self.schema_version,
            &self.canonical_identity,
            self.memory_kind,
            &self.projection_profile,
            self.source_frontier.as_deref(),
        ).expect("projection fields are bounded by the canonical encoder")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryProvenance {
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

impl MemoryProvenance {
    pub fn provenance_identity(&self) -> Option<&str> { self.provenance_family.as_deref() }
    pub fn canonical_artifact(&self) -> Option<Result<CanonicalArtifactRef, RefValidationError>> { self.canonical_artifact_ref.as_deref().map(CanonicalArtifactRef::try_from) }
    pub fn statement(&self) -> Option<Result<StatementRef, RefValidationError>> { self.statement_ref.as_deref().map(StatementRef::try_from) }
    pub fn provenance_family_ref(&self) -> Option<Result<ProvenanceFamilyRef, RefValidationError>> { self.provenance_family.as_deref().map(ProvenanceFamilyRef::try_from) }
    pub fn frontier(&self) -> Option<Result<FrontierRef, RefValidationError>> { self.frontier_ref.as_deref().map(FrontierRef::try_from) }
    pub fn derivation(&self) -> Option<Result<DerivationRef, RefValidationError>> { self.derivation_ref.as_deref().map(DerivationRef::try_from) }
    pub fn model(&self) -> Option<Result<ModelRef, RefValidationError>> { self.model_ref.as_deref().map(ModelRef::try_from) }
    pub fn retrieval_index(&self) -> Option<Result<RetrievalIndexRef, RefValidationError>> { self.retrieval_index_ref.as_deref().map(RetrievalIndexRef::try_from) }
    pub fn source_event_ref(&self) -> Option<Result<SourceEventRef, RefValidationError>> { self.source_event.as_deref().map(SourceEventRef::try_from) }
    pub fn epistemic_state_ref(&self) -> Option<Result<EpistemicStateRef, RefValidationError>> { self.epistemic_state.as_deref().map(EpistemicStateRef::try_from) }
    pub fn claim_ceiling_ref(&self) -> Option<Result<ClaimCeilingRef, RefValidationError>> { self.claim_ceiling.as_deref().map(ClaimCeilingRef::try_from) }
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
    fn sha256_is_deterministic() {
        assert_eq!(sha256_hex(b"symthaea-memory"), sha256_hex(b"symthaea-memory"));
    }
}
