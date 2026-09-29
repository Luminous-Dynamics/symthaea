//! Qualified derivation context for exact federated output lineage.
//!
//! Inspired by the qualification pattern in W3C PROV: a derivation can carry
//! additional context about the activity, plan, roles, and responsible agent.
//! This module records such context without authenticating it.
//!
//! A context receipt is provenance metadata, not evidence, truth, execution
//! attestation, or authority transfer.

use super::federated_projection_output_lineage::FederatedProjectionOutputLineageReceipt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:federated-qualified-derivation:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedDerivationParentRole {
    pub parent_projection_digest: String,
    pub parent_output_record_id: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedQualifiedDerivationReceipt {
    pub receipt_version: String,
    pub output_lineage_digest: String,
    pub activity_id: String,
    pub activity_record_digest: String,
    pub plan_ref: Option<String>,
    pub responsible_agent_id: Option<String>,
    pub parent_roles: Vec<FederatedDerivationParentRole>,
    pub context_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualifiedDerivationError {
    EmptyField(&'static str),
    InvalidDigest,
    LineageDigestMismatch,
    ParentRoleMismatch,
    DuplicateParentRole,
    NonCanonicalParentRoleOrder,
    ContextDigestMismatch,
}

impl FederatedQualifiedDerivationReceipt {
    pub fn new(
        lineage: &FederatedProjectionOutputLineageReceipt,
        activity_id: impl Into<String>,
        activity_record_digest: impl Into<String>,
        plan_ref: Option<String>,
        responsible_agent_id: Option<String>,
        parent_roles: Vec<FederatedDerivationParentRole>,
    ) -> Result<Self, QualifiedDerivationError> {
        let activity_id = activity_id.into();
        let activity_record_digest = activity_record_digest.into();

        require_nonempty(&activity_id, "activity_id")?;
        require_nonempty(&activity_record_digest, "activity_record_digest")?;
        if !valid_digest(&activity_record_digest) || !valid_digest(&lineage.lineage_digest) {
            return Err(QualifiedDerivationError::InvalidDigest);
        }
        if let Some(value) = &plan_ref {
            require_nonempty(value, "plan_ref")?;
        }
        if let Some(value) = &responsible_agent_id {
            require_nonempty(value, "responsible_agent_id")?;
        }

        let mut roles = parent_roles;
        roles.sort_by(|a, b| {
            a.parent_projection_digest
                .cmp(&b.parent_projection_digest)
                .then_with(|| a.parent_output_record_id.cmp(&b.parent_output_record_id))
                .then_with(|| a.role.cmp(&b.role))
        });

        if roles.windows(2).any(|w| w[0] == w[1]) {
            return Err(QualifiedDerivationError::DuplicateParentRole);
        }

        for role in &roles {
            require_nonempty(&role.parent_projection_digest, "parent_projection_digest")?;
            require_nonempty(&role.parent_output_record_id, "parent_output_record_id")?;
            require_nonempty(&role.role, "role")?;
            if !lineage.parent_outputs.iter().any(|parent| {
                parent.projection_digest == role.parent_projection_digest
                    && parent.output_record_id == role.parent_output_record_id
            }) {
                return Err(QualifiedDerivationError::ParentRoleMismatch);
            }
        }

        if roles.len() != lineage.parent_outputs.len() {
            return Err(QualifiedDerivationError::ParentRoleMismatch);
        }

        let mut receipt = Self {
            receipt_version: VERSION.into(),
            output_lineage_digest: lineage.lineage_digest.clone(),
            activity_id,
            activity_record_digest,
            plan_ref,
            responsible_agent_id,
            parent_roles: roles,
            context_digest: String::new(),
        };
        receipt.context_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify(
        &self,
        lineage: &FederatedProjectionOutputLineageReceipt,
    ) -> Result<(), QualifiedDerivationError> {
        if self.receipt_version != VERSION
            || self.output_lineage_digest != lineage.lineage_digest
        {
            return Err(QualifiedDerivationError::LineageDigestMismatch);
        }
        if !valid_digest(&self.output_lineage_digest)
            || !valid_digest(&self.activity_record_digest)
        {
            return Err(QualifiedDerivationError::InvalidDigest);
        }
        require_nonempty(&self.activity_id, "activity_id")?;
        if let Some(value) = &self.plan_ref {
            require_nonempty(value, "plan_ref")?;
        }
        if let Some(value) = &self.responsible_agent_id {
            require_nonempty(value, "responsible_agent_id")?;
        }

        if self.parent_roles.windows(2).any(|w| {
            (
                w[0].parent_projection_digest.as_str(),
                w[0].parent_output_record_id.as_str(),
                w[0].role.as_str(),
            ) >= (
                w[1].parent_projection_digest.as_str(),
                w[1].parent_output_record_id.as_str(),
                w[1].role.as_str(),
            )
        }) {
            return Err(QualifiedDerivationError::NonCanonicalParentRoleOrder);
        }

        if self.parent_roles.len() != lineage.parent_outputs.len() {
            return Err(QualifiedDerivationError::ParentRoleMismatch);
        }

        for role in &self.parent_roles {
            if !lineage.parent_outputs.iter().any(|parent| {
                parent.projection_digest == role.parent_projection_digest
                    && parent.output_record_id == role.parent_output_record_id
            }) {
                return Err(QualifiedDerivationError::ParentRoleMismatch);
            }
        }

        if self.context_digest != self.compute_digest() {
            return Err(QualifiedDerivationError::ContextDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            self.receipt_version.as_str(),
            self.output_lineage_digest.as_str(),
            self.activity_id.as_str(),
            self.activity_record_digest.as_str(),
        ] {
            put(&mut h, value);
        }
        put(&mut h, self.plan_ref.as_deref().unwrap_or(""));
        put(&mut h, self.responsible_agent_id.as_deref().unwrap_or(""));
        for role in &self.parent_roles {
            put(&mut h, &role.parent_projection_digest);
            put(&mut h, &role.parent_output_record_id);
            put(&mut h, &role.role);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), QualifiedDerivationError> {
    if value.trim().is_empty() {
        Err(QualifiedDerivationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn valid_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..].iter().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_dkg_adapter::{
        AdapterCapability, AdapterDisposition, CrossDkgAdapterDeclaration, CrossDkgAdapterReceipt,
    };
    use crate::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgRelation};
    use crate::cross_dkg_link_collection::CrossDkgLinkCollection;
    use crate::federated_projection_output_lineage::FederatedProjectionParentOutput;

    fn lineage() -> FederatedProjectionOutputLineageReceipt {
        let declaration = CrossDkgAdapterDeclaration {
            adapter_id: "mycelix-to-symthaea".into(),
            adapter_version: "1.0".into(),
            source_dkg_id: "mycelix".into(),
            target_dkg_id: "symthaea".into(),
            allowed_relations: vec![CrossDkgRelation::References],
            capability: AdapterCapability::ReferenceOnly,
            policy_ref: "policy:research-reference".into(),
        };
        let link = CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:1", "sha256:r1",
            CrossDkgRelation::References, "symthaea", "sha256:g2",
            "observation:1", "sha256:t1",
        ).unwrap();
        let adapter = CrossDkgAdapterReceipt::derive_reference(
            &declaration, &link, AdapterDisposition::ReferenceOnly,
            "local:1", "sha256:o1",
        ).unwrap();
        let collection = CrossDkgLinkCollection::new(vec![link]).unwrap();
        let projection = super::super::cross_dkg_projection::FederatedProjectionReceipt::new(
            "projection:1", "1", &collection, &[adapter], &[],
        ).unwrap();
        let output = projection.outputs[0].clone();
        FederatedProjectionOutputLineageReceipt::new(
            &projection, &output, &[], &[projection],
        ).unwrap()
    }

    #[test]
    fn qualified_context_verifies() {
        let lineage = lineage();
        let receipt = FederatedQualifiedDerivationReceipt::new(
            &lineage,
            "activity:1",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            Some("plan:1".into()),
            Some("agent:1".into()),
            vec![],
        ).unwrap();
        assert!(receipt.verify(&lineage).is_ok());
    }

    #[test]
    fn activity_tampering_is_detected() {
        let lineage = lineage();
        let mut receipt = FederatedQualifiedDerivationReceipt::new(
            &lineage,
            "activity:1",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            None,
            None,
            vec![],
        ).unwrap();
        receipt.activity_id = "activity:tampered".into();
        assert_eq!(
            receipt.verify(&lineage),
            Err(QualifiedDerivationError::ContextDigestMismatch)
        );
    }

    #[test]
    fn parent_roles_must_match_lineage() {
        let lineage = lineage();
        assert_eq!(
            FederatedQualifiedDerivationReceipt::new(
                &lineage,
                "activity:1",
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                None,
                None,
                vec![FederatedDerivationParentRole {
                    parent_projection_digest:
                        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                    parent_output_record_id: "missing".into(),
                    role: "input".into(),
                }],
            ),
            Err(QualifiedDerivationError::ParentRoleMismatch)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let lineage = lineage();
        let receipt = FederatedQualifiedDerivationReceipt::new(
            &lineage, "activity:1",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            Some("plan:1".into()), Some("agent:1".into()), vec![],
        ).unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        assert_eq!(
            serde_json::from_slice::<FederatedQualifiedDerivationReceipt>(&bytes).unwrap(),
            receipt
        );
    }
}
