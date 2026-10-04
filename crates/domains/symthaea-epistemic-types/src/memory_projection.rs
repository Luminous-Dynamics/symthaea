use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub const MEMORY_PROJECTION_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    Working,
    Episodic,
    Semantic,
    Procedural,
    KnowledgeGraph,
    Vector,
    Hdc,
}

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
    pub fn new(
        canonical_identity: impl Into<String>,
        memory_kind: MemoryKind,
        projection_profile: impl Into<String>,
        representation_bytes: &[u8],
        source_frontier: Option<String>,
    ) -> Self {
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
        let projection = (
            self.schema_version,
            &self.canonical_identity,
            self.memory_kind,
            &self.projection_profile,
            &self.source_frontier,
        );
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

/// Software-level receipt binding an explicit canonical admission to the
/// provenance snapshot and frontier from which the admission was made.
/// This is not a claim that the admitted object is true.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalAdmissionReceipt {
    pub admission_event: String,
    pub frontier_ref: Option<String>,
    /// Deterministic binding to the canonical subject and provenance family admitted.
    pub admitted_subject_digest: String,
    pub provenance_snapshot_digest: String,
    pub validator_version: String,
    pub snapshot_schema_version: u16,
}

impl CanonicalAdmissionReceipt {
    pub fn new(
        admission_event: impl Into<String>,
        frontier_ref: Option<String>,
        canonical_identity: impl Into<String>,
        provenance_family: Option<String>,
        provenance_snapshot_digest: impl Into<String>,
        validator_version: impl Into<String>,
        snapshot_schema_version: u16,
    ) -> Result<Self, &'static str> {
        let admission_event = admission_event.into();
        let canonical_identity = canonical_identity.into();
        let provenance_snapshot_digest = provenance_snapshot_digest.into();
        let validator_version = validator_version.into();
        if admission_event.trim().is_empty() {

            return Err("admission event must be non-empty");
        }
        if frontier_ref.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err("frontier reference must be non-empty when present");
        }
        if canonical_identity.trim().is_empty() {
            return Err("canonical identity must be non-empty");
        }
        if provenance_family.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err("provenance family must be non-empty when present");
        }
        if !is_hex_digest(&provenance_snapshot_digest) {
            return Err("provenance snapshot digest must be a 64-character hexadecimal digest");
        }
        if validator_version != PROVENANCE_VALIDATOR_VERSION {
            return Err("admission receipt validator version mismatch");
        }
        if snapshot_schema_version != PROVENANCE_SNAPSHOT_SCHEMA_VERSION {
            return Err("admission receipt snapshot schema version mismatch");
        }
        Ok(Self {
            admission_event,
            frontier_ref,
            admitted_subject_digest: Self::subject_digest_for(
                &canonical_identity,
                provenance_family.as_deref(),
            ),
            provenance_snapshot_digest,
            validator_version,
            snapshot_schema_version,
        })
    }

    fn subject_digest_for(canonical_identity: &str, provenance_family: Option<&str>) -> String {
        let subject = (
            "symthaea:canonical-admission-subject:v1",
            canonical_identity,
            provenance_family,
        );
        let bytes = serde_json::to_vec(&subject)
            .expect("canonical admission subject is serializable");
        sha256_hex(&bytes)
    }

    /// Returns true when the receipt is bound to the exact canonical subject envelope.
    pub fn binds_subject(
        &self,
        canonical_identity: &str,
        provenance_family: Option<&str>,
    ) -> bool {
        !canonical_identity.trim().is_empty()
            && provenance_family.map_or(true, |family| !family.trim().is_empty())
            && self.admitted_subject_digest
                == Self::subject_digest_for(canonical_identity, provenance_family)
    }

    /// Validate the receipt fields before they are used as an admission binding.
    ///
    /// This is intentionally structural: it checks the receipt's own envelope but does
    /// not assert that the admission event was authentically authored.
    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.admission_event.trim().is_empty() {
            return Err("admission event must be non-empty");
        }
        if self.frontier_ref.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err("frontier reference must be non-empty when present");
        }
        if !is_hex_digest(&self.admitted_subject_digest) {
            return Err("admitted subject digest must be a 64-character hexadecimal digest");
        }
        if !is_hex_digest(&self.provenance_snapshot_digest) {
            return Err("provenance snapshot digest must be a 64-character hexadecimal digest");
        }
        if self.validator_version != PROVENANCE_VALIDATOR_VERSION {
            return Err("admission receipt validator version mismatch");
        }
        if self.snapshot_schema_version != PROVENANCE_SNAPSHOT_SCHEMA_VERSION {
            return Err("admission receipt snapshot schema version mismatch");
        }
        Ok(())
    }

    /// Returns true only when this receipt identifies exactly the supplied
    /// provenance validation snapshot. This binds admission to a concrete,
    /// schema-versioned structural state without assigning epistemic weight.
    pub fn binds_validation(&self, validation: &ProvenanceValidationReport) -> bool {
        self.validate_structure().is_ok()
            && validation.validate_metadata().is_ok()
            && validation.conforms
            && self.provenance_snapshot_digest == validation.snapshot_digest
            && self.validator_version == validation.validator_version
            && self.snapshot_schema_version == validation.snapshot_schema_version
    }
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
pub const PROVENANCE_VALIDATOR_VERSION: &str = "symthaea-provenance-structural-v1";
/// Version of the canonical provenance snapshot encoding used by the digest.
pub const PROVENANCE_SNAPSHOT_SCHEMA_VERSION: u16 = 1;

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
    pub snapshot_schema_version: u16,
    pub snapshot_digest: String,
    pub relation_count: usize,
    pub conforms: bool,
    pub violations: Vec<ProvenanceValidationViolation>,
}

/// Read-only provenance boundary for downstream evidence/federation adapters.
///
/// This view contains lineage metadata and the structural validation report only.
/// It intentionally exposes no mutable cognitive state and assigns no evidential weight.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceView {
    pub validator_version: String,
    pub snapshot_schema_version: u16,
    pub snapshot_digest: String,
    pub relations: Vec<ProvenanceRelation>,
    pub validation: ProvenanceValidationReport,
}

impl ProvenanceView {
    pub fn from_relations(
        relations: &[ProvenanceRelation],
        validation: ProvenanceValidationReport,
    ) -> Result<Self, &'static str> {
        validation.validate_against_relations(relations)?;
        Ok(Self {
            validator_version: validation.validator_version.clone(),
            snapshot_schema_version: validation.snapshot_schema_version,
            snapshot_digest: validation.snapshot_digest.clone(),
            relations: relations.to_vec(),
            validation,
        })
    }

    pub fn relations_from(&self, memory_id: &str) -> Vec<&ProvenanceRelation> {
        self.relations
            .iter()
            .filter(|relation| relation.source_memory_id == memory_id)
            .collect()
    }

    pub fn relations_to(&self, memory_id: &str) -> Vec<&ProvenanceRelation> {
        self.relations
            .iter()
            .filter(|relation| relation.target_memory_id == memory_id)
            .collect()
    }

    /// Validate a view even when it originated from deserialization rather than
    /// from the constructor. This keeps the read-only boundary fail-closed.
    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.validator_version != self.validation.validator_version {
            return Err("provenance view validator version must match validation report");
        }
        if self.snapshot_schema_version != self.validation.snapshot_schema_version {
            return Err("provenance view schema version must match validation report");
        }
        if self.snapshot_digest != self.validation.snapshot_digest {
            return Err("provenance view digest must match validation report");
        }
        self.validation.validate_against_relations(&self.relations)
    }

    pub fn is_structurally_conforming(&self) -> bool {
        self.validate_structure().is_ok() && self.validation.conforms
    }
}

impl ProvenanceValidationReport {
    fn snapshot_digest_for(
        relations: &[ProvenanceRelation],
        schema_version: u16,
    ) -> String {
        let mut canonical = relations.to_vec();
        canonical.sort_by(|a, b| {
            (&a.source_memory_id, &a.target_memory_id, a.kind.stable_code(), &a.created_at)
                .cmp(&(&b.source_memory_id, &b.target_memory_id, b.kind.stable_code(), &b.created_at))
        });

        // Hash an explicitly tagged snapshot envelope. The schema version is part of
        // the digest so a future encoding change cannot silently reuse a prior digest.
        // Relation tuples use stable wire codes rather than Rust enum discriminants/encoding.
        let canonical_fields: Vec<(&str, &str, &str, &str)> = canonical
            .iter()
            .map(|relation| (
                relation.source_memory_id.as_str(),
                relation.target_memory_id.as_str(),
                relation.kind.stable_code(),
                relation.created_at.as_str(),
            ))
            .collect();
        let canonical_snapshot = (schema_version, canonical_fields);
        let bytes = serde_json::to_vec(&canonical_snapshot)
            .expect("canonical provenance snapshot is serializable");
        sha256_hex(&bytes)
    }

    pub fn from_relations(relations: &[ProvenanceRelation]) -> Self {
        let snapshot_digest =
            Self::snapshot_digest_for(relations, PROVENANCE_SNAPSHOT_SCHEMA_VERSION);
        Self {
            snapshot_schema_version: PROVENANCE_SNAPSHOT_SCHEMA_VERSION,
            validator_version: PROVENANCE_VALIDATOR_VERSION.to_owned(),
            snapshot_digest,
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

    /// Validate the report's self-contained metadata without requiring the relation slice.
    ///
    /// This closes the deserialized-report boundary for callers that only possess the
    /// report, such as an admission receipt, while leaving relation binding to
    /// validate_against_relations.
    pub fn validate_metadata(&self) -> Result<(), &'static str> {
        if self.validator_version != PROVENANCE_VALIDATOR_VERSION {
            return Err("provenance validator version mismatch");
        }
        if self.snapshot_schema_version != PROVENANCE_SNAPSHOT_SCHEMA_VERSION {
            return Err("provenance validation schema version mismatch");
        }
        if !is_hex_digest(&self.snapshot_digest) {
            return Err("provenance snapshot digest must be a 64-character hexadecimal digest");
        }
        if self.conforms != self.violations.is_empty() {
            return Err("provenance validation outcome does not match violations");
        }
        Ok(())
    }

    /// Defensive structural validation for a report that may have come from
    /// deserialization rather than from from_relations.
    ///
    /// This binds report metadata to the concrete relation slice presented to a
    /// ProvenanceView, preventing a mismatched digest/count/schema/version from
    /// crossing the read-only provenance boundary.
    pub fn validate_against_relations(
        &self,
        relations: &[ProvenanceRelation],
    ) -> Result<(), &'static str> {
        self.validate_metadata()?
        if self.relation_count != relations.len() {
            return Err("provenance validation relation count mismatch");
        }
        let expected_digest = Self::snapshot_digest_for(relations, self.snapshot_schema_version);
        if expected_digest != self.snapshot_digest {
            return Err("provenance view validation digest mismatch");
        }

        let mut seen = std::collections::HashSet::with_capacity(relations.len());
        for relation in relations {
            relation.validate()?;
            let key = (
                relation.source_memory_id.as_str(),
                relation.target_memory_id.as_str(),
                relation.kind.stable_code(),
                relation.created_at.as_str(),
            );
            if !seen.insert(key) {
                return Err("provenance relations must be unique");
            }
        }
        Ok(())
    }
}

impl MemoryProvenance {
    pub fn provenance_identity(&self) -> Option<&str> {
        self.provenance_family.as_deref()
    }

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
        for reference in [
            self.source_event.as_deref(),
            self.canonical_artifact_ref.as_deref(),
            self.statement_ref.as_deref(),
            self.frontier_ref.as_deref(),
            self.derivation_ref.as_deref(),
            self.model_ref.as_deref(),
            self.retrieval_index_ref.as_deref(),
        ] {
            if reference.is_some_and(|value| value.trim().is_empty()) {
                return Err("provenance reference must be non-empty when present");
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
        let a = MemoryProjectionRef::new(
            "claim:abc",
            MemoryKind::Semantic,
            "semantic-v1",
            b"summary-a",
            None,
        );
        let b = MemoryProjectionRef::new(
            "claim:abc",
            MemoryKind::Vector,
            "embedding-v1",
            b"embedding-b",
            None,
        );
        assert_eq!(a.canonical_identity_digest(), b.canonical_identity_digest());
        assert_ne!(a.representation_digest, b.representation_digest);
        assert_ne!(
            a.projection_identity_digest(),
            b.projection_identity_digest()
        );
    }

    #[test]
    fn schema_version_changes_projection_not_canonical_identity() {
        let a = MemoryProjectionRef::new(
            "claim:abc",
            MemoryKind::Semantic,
            "semantic-v1",
            b"same",
            None,
        );
        let mut b = a.clone();
        b.schema_version += 1;
        assert_eq!(a.canonical_identity_digest(), b.canonical_identity_digest());
        assert_ne!(
            a.projection_identity_digest(),
            b.projection_identity_digest()
        );
    }

    #[test]
    fn provenance_family_is_stable_across_retrieval_indexes() {
        let a = MemoryProvenance {
            canonical_identity: Some("claim:abc".into()),
            memory_id: "mem-a".into(),
            memory_kind: MemoryKind::Vector,
            created_at: "2026-09-29T00:00:00Z".into(),
            source_event: Some("event-1".into()),
            canonical_artifact_ref: Some("artifact-1".into()),
            statement_ref: Some("statement-1".into()),
            provenance_family: Some("family-1".into()),
            epistemic_state: Some("Observed".into()),
            claim_ceiling: Some("source-scoped".into()),
            frontier_ref: Some("frontier-1".into()),
            derivation_ref: None,
            model_ref: None,
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

        p.memory_id = "mem-1".into();
        macro_rules! assert_blank_optional_rejected {
            ($field:ident) => {
                p.$field = Some("   ".into());
                assert_eq!(
                    p.validate_structure(),
                    Err("provenance reference must be non-empty when present")
                );
                p.$field = None;
            };
        }
        assert_blank_optional_rejected!(source_event);
        assert_blank_optional_rejected!(canonical_artifact_ref);
        assert_blank_optional_rejected!(statement_ref);
        assert_blank_optional_rejected!(frontier_ref);
        assert_blank_optional_rejected!(derivation_ref);
        assert_blank_optional_rejected!(model_ref);
        assert_blank_optional_rejected!(retrieval_index_ref);
    }

    #[test]
    fn admission_receipt_binds_exact_validation_snapshot() {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation =
            ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        assert!(validation.conforms);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        assert!(receipt.binds_validation(&validation));

        let mut nonconforming = validation.clone();
        nonconforming.conforms = false;
        assert!(!receipt.binds_validation(&nonconforming));

        let mut inconsistent = validation.clone();
        inconsistent.violations.push(ProvenanceValidationViolation {
            code: "tampered".into(),
            source_memory_id: None,
            target_memory_id: None,
            message: "tampered report".into(),
        });
        assert!(!receipt.binds_validation(&inconsistent));

        let mut malformed_digest = validation.clone();
        malformed_digest.snapshot_digest = "not-a-digest".into();
        assert!(!receipt.binds_validation(&malformed_digest));

        let mut malformed_receipt = receipt.clone();
        malformed_receipt.admission_event = "   ".into();
        assert!(!malformed_receipt.binds_validation(&validation));

        let mut changed = validation.clone();
        changed.snapshot_schema_version += 1;
        assert!(!receipt.binds_validation(&changed));

        let mut forged_validator = validation.clone();
        forged_validator.validator_version = "attacker-defined-v999".into();
        assert_eq!(
            forged_validator.validate_metadata(),
            Err("provenance validator version mismatch")
        );
        assert!(!receipt.binds_validation(&forged_validator));

        assert!(CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:1".into()),
            validation.snapshot_digest.clone(),
            "attacker-defined-v999",
            PROVENANCE_SNAPSHOT_SCHEMA_VERSION,
        ).is_err());

        assert!(CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:1".into()),
            validation.snapshot_digest.clone(),
            PROVENANCE_VALIDATOR_VERSION,
            PROVENANCE_SNAPSHOT_SCHEMA_VERSION + 1,
        ).is_err());

        let mut forged_receipt = receipt.clone();
        forged_receipt.validator_version = "attacker-defined-v999".into();
        assert_eq!(
            forged_receipt.validate_structure(),
            Err("admission receipt validator version mismatch")
        );
        assert!(!forged_receipt.binds_validation(&validation));

        let mut subject = receipt.clone();
        subject.admitted_subject_digest = "0".repeat(64);
        assert_eq!(
            subject.binds_subject("canonical:1", Some("family:1")),
            false
        );
        assert!(!subject.binds_validation(&validation));
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
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let b = ProvenanceRelation {
            source_memory_id: "revision".into(),
            target_memory_id: "derived".into(),
            kind: ProvenanceRelationKind::RevisedFrom,
            created_at: "cycle:3".into(),
        };
        let first = ProvenanceValidationReport::from_relations(&[a.clone(), b.clone()]);
        let second = ProvenanceValidationReport::from_relations(&[b, a]);
        assert_eq!(first.snapshot_digest, second.snapshot_digest);
        assert_eq!(first.relation_count, 2);
        assert!(first.conforms);
        assert!(first.violations.is_empty());
        assert_eq!(first.validator_version, PROVENANCE_VALIDATOR_VERSION);
        assert_eq!(first.snapshot_schema_version, PROVENANCE_SNAPSHOT_SCHEMA_VERSION);

        // The schema tag is part of the digest input, not merely report metadata.
        let empty = ProvenanceValidationReport::from_relations(&[]);
        let singleton = ProvenanceValidationReport::from_relations(&[a, b]);
        assert_ne!(empty.snapshot_digest, singleton.snapshot_digest);
        assert_ne!(
            singleton.snapshot_digest,
            ProvenanceValidationReport::snapshot_digest_for(
                &[a, b],
                PROVENANCE_SNAPSHOT_SCHEMA_VERSION + 1,
            )
        );
    }

    #[test]
    fn validation_report_rejects_deserialized_metadata_drift() {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let valid = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));

        let mut wrong_digest = valid.clone();
        wrong_digest.snapshot_digest = "not-a-digest".into();
        assert_eq!(
            ProvenanceView::from_relations(std::slice::from_ref(&relation), wrong_digest).unwrap_err(),
            "provenance snapshot digest must be a 64-character hexadecimal digest"
        );

        let mut wrong_count = valid.clone();
        wrong_count.relation_count = 0;
        assert_eq!(
            ProvenanceView::from_relations(std::slice::from_ref(&relation), wrong_count).unwrap_err(),
            "provenance validation relation count mismatch"
        );

        let mut blank_validator = valid.clone();
        blank_validator.validator_version.clear();
        assert_eq!(
            ProvenanceView::from_relations(std::slice::from_ref(&relation), blank_validator).unwrap_err(),
            "provenance validator version must be non-empty"
        );

        let mut mismatched_outcome = valid.clone();
        mismatched_outcome.conforms = true;
        mismatched_outcome.violations.push(ProvenanceValidationViolation {
            code: "test".into(),
            source_memory_id: None,
            target_memory_id: None,
            message: "invalid".into(),
        });
        assert_eq!(
            ProvenanceView::from_relations(std::slice::from_ref(&relation), mismatched_outcome).unwrap_err(),
            "provenance validation outcome does not match violations"
        );
        assert!(valid.validate_metadata().is_ok());
    }

    #[test]
    fn provenance_view_is_read_only_snapshot_with_bound_validation() {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        assert_eq!(view.relations, vec![relation]);
        assert_eq!(view.relations_from("derived").len(), 1);
        assert_eq!(view.relations_to("source").len(), 1);
        assert!(view.is_structurally_conforming());
        assert_eq!(
            view.snapshot_digest,
            view.validation.snapshot_digest
        );
        assert_eq!(
            view.snapshot_schema_version,
            view.validation.snapshot_schema_version
        );
    }

    #[test]
    fn deserialized_provenance_view_rejects_envelope_drift() {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let mut view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();

        view.snapshot_digest = "f".repeat(64);
        assert_eq!(
            view.validate_structure(),
            Err("provenance view digest must match validation report")
        );
        assert!(!view.is_structurally_conforming());

        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let mut view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        view.validator_version = "attacker-defined-v999".into();
        assert_eq!(
            view.validate_structure(),
            Err("provenance view validator version must match validation report")
        );
        assert!(!view.is_structurally_conforming());

        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let mut view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        view.relations[0].target_memory_id = "tampered".into();
        assert_eq!(
            view.validate_structure(),
            Err("provenance view validation digest mismatch")
        );
        assert!(!view.is_structurally_conforming());

        let mut view = ProvenanceView::from_relations(
            std::slice::from_ref(&relation),
            ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation)),
        ).unwrap();
        view.snapshot_schema_version += 1;
        assert_eq!(
            view.validate_structure(),
            Err("provenance view schema version must match validation report")
        );
        assert!(!view.is_structurally_conforming());
    }

    #[test]
    fn provenance_view_does_not_assign_evidence_weight() {
        let relation = ProvenanceRelation {
            source_memory_id: "a".into(),
            target_memory_id: "b".into(),
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(&[relation]);
        let view = ProvenanceView::from_relations(&[relation], validation).unwrap();
        assert!(view.is_structurally_conforming());
        // The view exposes relation structure only; no confidence/evidence field exists.
        assert_eq!(view.validation.relation_count, 1);
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
        assert_eq!(
            sha256_hex(b"symthaea-memory"),
            sha256_hex(b"symthaea-memory")
        );
    }
}
