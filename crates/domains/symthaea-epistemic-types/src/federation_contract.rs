//! Substrate-neutral contract for exporting a locally admitted epistemic claim
//! to a federated provenance fabric.
//!
//! This module intentionally contains no Holochain, database, network, or truth
//! semantics. It describes what a federation adapter must carry and structurally
//! validate when transporting an explicitly admitted claim.

use crate::{sha256_hex, CanonicalAdmissionReceipt, ProvenanceRelation, ProvenanceRelationKind, ProvenanceValidationReport, ProvenanceView};

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
    pub provenance_validation: ProvenanceValidationReport,
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
        provenance_view: ProvenanceView,
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
            provenance_snapshot_digest: provenance_view.snapshot_digest.clone(),
            provenance_validation: provenance_view.validation.clone(),
            admission_receipt,
            epistemic_state: None,
            claim_ceiling: None,
            model_ref: None,
            derivation_refs: Vec::new(),
            relations: provenance_view.relations.clone(),
        };
        if !claim.admission_receipt.binds_validation(&provenance_view.validation) {
            return Err("admission receipt must bind provenance view validation");
        }
        claim.validate_structure()?;
        Ok(claim)
    }

    /// Structural contract only. It does not establish truth, reliability, authorship
    /// cryptography, or scientific validity; those belong to the federation adapter
    /// and/or higher-level epistemic assessment.
    /// Deterministic digest of the complete federated envelope. This is an identity
    /// for the exported representation, not a truth or authority signal.
    pub fn canonical_digest(&self) -> String {
        fn field(bytes: &mut Vec<u8>, label: &str, value: Option<&str>) {
            bytes.extend_from_slice(&(label.len() as u64).to_be_bytes());
            bytes.extend_from_slice(label.as_bytes());
            match value {
                Some(value) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
                    bytes.extend_from_slice(value.as_bytes());
                }
                None => bytes.push(0),
            }
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:federated-claim:v1\0");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        field(&mut bytes, "claim_identity", Some(&self.claim_identity));
        field(&mut bytes, "canonical_identity", Some(&self.canonical_identity));
        field(&mut bytes, "provenance_family", Some(&self.provenance_family));
        field(&mut bytes, "author", Some(&self.author));
        field(&mut bytes, "statement_ref", Some(&self.statement_ref));
        field(&mut bytes, "source_event", self.source_event.as_deref());
        field(&mut bytes, "frontier_ref", self.frontier_ref.as_deref());
        field(&mut bytes, "provenance_snapshot_digest", Some(&self.provenance_snapshot_digest));
        field(&mut bytes, "validator_version", Some(&self.provenance_validation.validator_version));
        bytes.extend_from_slice(&self.provenance_validation.snapshot_schema_version.to_be_bytes());
        bytes.extend_from_slice(&(self.provenance_validation.relation_count as u64).to_be_bytes());
        bytes.push(self.provenance_validation.conforms as u8);
        field(&mut bytes, "admission_event", Some(&self.admission_receipt.admission_event));
        field(&mut bytes, "receipt_validator_version", Some(&self.admission_receipt.validator_version));
        bytes.extend_from_slice(&self.admission_receipt.snapshot_schema_version.to_be_bytes());
        field(&mut bytes, "epistemic_state", self.epistemic_state.as_deref());
        field(&mut bytes, "claim_ceiling", self.claim_ceiling.as_deref());
        field(&mut bytes, "model_ref", self.model_ref.as_deref());

        let mut derivation_refs = self.derivation_refs.clone();
        derivation_refs.sort();
        bytes.extend_from_slice(&(derivation_refs.len() as u64).to_be_bytes());
        for reference in derivation_refs {
            field(&mut bytes, "derivation_ref", Some(&reference));
        }

        let mut relations = self.relations.clone();
        relations.sort_by(|a, b| {
            (&a.source_memory_id, &a.target_memory_id, a.kind.stable_code(), &a.created_at)
                .cmp(&(&b.source_memory_id, &b.target_memory_id, b.kind.stable_code(), &b.created_at))
        });
        bytes.extend_from_slice(&(relations.len() as u64).to_be_bytes());
        for relation in relations {
            field(&mut bytes, "relation_source", Some(&relation.source_memory_id));
            field(&mut bytes, "relation_target", Some(&relation.target_memory_id));
            field(&mut bytes, "relation_kind", Some(relation.kind.stable_code()));
            field(&mut bytes, "relation_created_at", Some(&relation.created_at));
        }

        sha256_hex(&bytes)
    }

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

        if !self.provenance_validation.conforms {
            return Err("federated claim provenance validation does not conform");
        }
        if !self.provenance_validation.violations.is_empty() {
            return Err("conforming federated claim cannot contain validation violations");
        }
        if self.provenance_validation.relation_count != self.relations.len() {
            return Err("claim relation count must match validation report");
        }
        if self.provenance_validation.snapshot_digest != self.provenance_snapshot_digest {
            return Err("claim snapshot digest must match validation report");
        }
        if ProvenanceView::from_relations(&self.relations, self.provenance_validation.clone()).is_err() {
            return Err("claim relations must match validation snapshot");
        }
        if !self.admission_receipt.binds_validation(&self.provenance_validation) {
            return Err("admission receipt must bind claim validation report");
        }
        if self.admission_receipt.provenance_snapshot_digest != self.provenance_snapshot_digest {
            return Err("claim snapshot digest must match validation report");
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
            && self.provenance_validation == *validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProvenanceRelation, ProvenanceRelationKind, ProvenanceValidationReport};

    fn receipt() -> (CanonicalAdmissionReceipt, ProvenanceValidationReport) {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(&[relation]);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        (receipt, validation)
    }

    #[test]
    fn federated_claim_preserves_admission_boundary() {
        let (r, validation) = receipt();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&[ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            }], validation).unwrap(),
            r,
        ).unwrap();
        assert_eq!(claim.schema_version, FEDERATED_CLAIM_SCHEMA_VERSION);
        assert_eq!(claim.frontier_ref.as_deref(), Some("frontier:1"));
        assert!(claim.validate_structure().is_ok());
        assert!(!claim.canonical_digest().is_empty());
    }

    #[test]
    fn federated_claim_rejects_snapshot_mismatch() {
        let (r, _valid) = receipt();
        let validation = ProvenanceValidationReport::from_relations(&[]);
        assert_eq!(
            FederatedClaim::new(
                "claim:1", "canonical:1", "family:1", "author:1", "statement:1",
                ProvenanceView::from_relations(&[], validation).unwrap(), r,
            ).unwrap_err(),
            "claim snapshot digest must match validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_frontier_mismatch() {
        let (r, validation) = receipt();
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&[ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            }], validation).unwrap(),
            r,
        ).unwrap();
        claim.frontier_ref = Some("frontier:other".into());
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "claim frontier must match admission receipt"
        );
    }
}


#[cfg(test)]
mod digest_tests {
    use super::*;

    fn base_claim() -> FederatedClaim {
        let relation_a = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let relation_b = ProvenanceRelation {
            source_memory_id: "claim".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:3".into(),
        };
        let relations = vec![relation_b, relation_a];
        let validation = ProvenanceValidationReport::from_relations(&relations);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&relations, validation).unwrap(),
            receipt,
        ).unwrap();
        claim.derivation_refs = vec!["derivation:b".into(), "derivation:a".into()];
        claim.relations = relations;
        claim
    }

    #[test]
    fn canonical_digest_is_order_independent_for_sets() {
        let mut a = base_claim();
        let mut b = a.clone();
        b.derivation_refs.reverse();
        b.relations.reverse();
        assert_eq!(a.canonical_digest(), b.canonical_digest());
        assert!(a.validate_structure().is_ok());
    }

    #[test]
    fn canonical_digest_changes_when_claim_content_changes() {
        let a = base_claim();
        let mut b = a.clone();
        b.author = "author:2".into();
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn federated_claim_json_round_trip_preserves_digest_and_admission_binding() {
        let claim = base_claim();
        let digest = claim.canonical_digest();
        let encoded = serde_json::to_string(&claim).expect("claim should serialize");
        let decoded: FederatedClaim =
            serde_json::from_str(&encoded).expect("claim should deserialize");
        assert_eq!(decoded, claim);
        assert_eq!(decoded.canonical_digest(), digest);
        assert!(decoded.validate_structure().is_ok());
        assert!(decoded.is_admission_bound(&decoded.provenance_validation));
    }

    #[test]
    fn federated_claim_rejects_mutated_admission_event() {
        let mut claim = base_claim();
        let original = claim.canonical_digest();
        claim.admission_receipt.admission_event = "admission:event-tampered".into();
        assert_ne!(claim.canonical_digest(), original);
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt must bind claim validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_validation_version() {
        let mut claim = base_claim();
        claim.provenance_validation.validator_version = "tampered-validator".into();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt must bind claim validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_schema_version() {
        let mut claim = base_claim();
        claim.provenance_validation.snapshot_schema_version += 1;
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt must bind claim validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_relation_content() {
        let mut claim = base_claim();
        claim.relations[0].created_at = "cycle:999".into();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "claim snapshot digest must match validation report"
        );
    }

    #[test]
    fn canonical_digest_changes_when_admission_event_changes() {
        let a = base_claim();
        let mut b = a.clone();
        b.admission_receipt.admission_event = "admission:event-2".into();
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn validation_rejects_relation_snapshot_drift() {
        let a = base_claim();
        let mut b = a.clone();
        b.relations.clear();
        assert_eq!(
            b.validate_structure().unwrap_err(),
            "claim relation count must match validation report"
        );
    }
}
