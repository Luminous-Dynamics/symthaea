//! Substrate-neutral contract for exporting a locally admitted epistemic claim
//! to a federated provenance fabric.
//!
//! This module intentionally contains no Holochain, database, network, or truth
//! semantics. It describes what a federation adapter must carry and structurally
//! validate when transporting an explicitly admitted claim.

use crate::{CanonicalAdmissionReceipt, ProvenanceRelation, ProvenanceRelationKind};

pub const FEDERATED_CLAIM_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FederatedClaim {
    pub schema_version: u16,
    pub claim_identity: String,
    pub canonical_identity: String,
    pub provenance_family: String,
    pub author: String,
    pub statement_ref: String,
    pub source_event: Option<String>,
    pub frontier_ref: Option<String>,
    pub provenance_snapshot_digest: String,
    pub admission_receipt: CanonicalAdmissionReceipt,
    pub epistemic_state: Option<String>,
    pub claim_ceiling: Option<String>,
    pub model_ref: Option<String>,
    pub derivation_refs: Vec<String>,
    pub relations: Vec<ProvenanceRelation>,
}

impl FederatedClaim {
    pub fn new(
        claim_identity: impl Into<String>,
        canonical_identity: impl Into<String>,
        provenance_family: impl Into<String>,
        author: impl Into<String>,
        statement_ref: impl Into<String>,
        provenance_snapshot_digest: impl Into<String>,
        admission_receipt: CanonicalAdmissionReceipt,
    ) -> Result<Self, &'static str> {
        let claim = Self {
            schema_version: FEDERATED_CLAIM_SCHEMA_VERSION,
            claim_identity: claim_identity.into(),
            canonical_identity: canonical_identity.into(),
            provenance_family: provenance_family.into(),
            author: author.into(),
            statement_ref: statement_ref.into(),
            source_event: None,
            frontier_ref: admission_receipt.frontier_ref.clone(),
            provenance_snapshot_digest: provenance_snapshot_digest.into(),
            admission_receipt,
            epistemic_state: None,
            claim_ceiling: None,
            model_ref: None,
            derivation_refs: Vec::new(),
            relations: Vec::new(),
        };
        claim.validate_structure()?;
        Ok(claim)
    }

    /// Structural contract only. It does not establish truth, reliability, authorship
    /// cryptography, or scientific validity; those belong to the federation adapter
    /// and/or higher-level epistemic assessment.
    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.schema_version != FEDERATED_CLAIM_SCHEMA_VERSION {
            return Err("unsupported federated claim schema version");
        }
        for (name, value) in [
            ("claim_identity", self.claim_identity.as_str()),
            ("canonical_identity", self.canonical_identity.as_str()),
            ("provenance_family", self.provenance_family.as_str()),
            ("author", self.author.as_str()),
            ("statement_ref", self.statement_ref.as_str()),
            ("provenance_snapshot_digest", self.provenance_snapshot_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(match name {
                    "claim_identity" => "claim identity must be non-empty",
                    "canonical_identity" => "canonical identity must be non-empty",
                    "provenance_family" => "provenance family must be non-empty",
                    "author" => "author must be non-empty",
                    "statement_ref" => "statement reference must be non-empty",
                    _ => "provenance snapshot digest must be non-empty",
                });
            }
        }

        if self.admission_receipt.provenance_snapshot_digest != self.provenance_snapshot_digest {
            return Err("claim snapshot digest must match admission receipt");
        }
        if self.admission_receipt.frontier_ref != self.frontier_ref {
            return Err("claim frontier must match admission receipt");
        }
        for relation in &self.relations {
            relation.validate()?;
            if relation.kind == ProvenanceRelationKind::RepresentationOf
                && relation.source_memory_id == relation.target_memory_id
            {
                return Err("representation relation cannot self-reference");
            }
        }
        Ok(())
    }

    pub fn is_admission_bound(
        &self,
        validation: &crate::ProvenanceValidationReport,
    ) -> bool {
        self.admission_receipt.binds_validation(validation)
            && self.provenance_snapshot_digest == validation.snapshot_digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProvenanceRelation, ProvenanceRelationKind, ProvenanceValidationReport};

    fn receipt() -> CanonicalAdmissionReceipt {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(&[relation]);
        CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            validation.snapshot_digest,
            validation.validator_version,
            validation.snapshot_schema_version,
        ).unwrap()
    }

    #[test]
    fn federated_claim_preserves_admission_boundary() {
        let r = receipt();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            r.provenance_snapshot_digest.clone(),
            r,
        ).unwrap();
        assert_eq!(claim.schema_version, FEDERATED_CLAIM_SCHEMA_VERSION);
        assert_eq!(claim.frontier_ref.as_deref(), Some("frontier:1"));
        assert!(claim.validate_structure().is_ok());
    }

    #[test]
    fn federated_claim_rejects_snapshot_mismatch() {
        let r = receipt();
        assert_eq!(
            FederatedClaim::new(
                "claim:1", "canonical:1", "family:1", "author:1", "statement:1",
                "wrong-digest", r,
            ).unwrap_err(),
            "claim snapshot digest must match admission receipt"
        );
    }

    #[test]
    fn federated_claim_rejects_frontier_mismatch() {
        let mut r = receipt();
        r.frontier_ref = Some("frontier:other".into());
        assert_eq!(
            FederatedClaim::new(
                "claim:1", "canonical:1", "family:1", "author:1", "statement:1",
                r.provenance_snapshot_digest.clone(), r,
            ).unwrap_err(),
            "claim frontier must match admission receipt"
        );
    }
}
