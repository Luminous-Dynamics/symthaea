//! Authority-bounded receipts for cross-DKG projection adapters.
//!
//! Receipts bind a declared adapter to one exact federation link and output
//! reference. They do not execute transformations or confer semantic authority.

use super::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgRelation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:cross-dkg-adapter-receipt:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterCapability {
    ReferenceOnly,
    ContextAttachment,
    DerivedMetadata,
}

impl AdapterCapability {
    fn permits(self, disposition: AdapterDisposition) -> bool {
        match self {
            Self::ReferenceOnly => disposition == AdapterDisposition::ReferenceOnly,
            Self::ContextAttachment => disposition != AdapterDisposition::DerivedMetadata,
            Self::DerivedMetadata => true,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceOnly => "reference_only",
            Self::ContextAttachment => "context_attachment",
            Self::DerivedMetadata => "derived_metadata",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterDisposition {
    ReferenceOnly,
    ContextOnly,
    DerivedMetadata,
}
impl AdapterDisposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceOnly => "reference_only",
            Self::ContextOnly => "context_only",
            Self::DerivedMetadata => "derived_metadata",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossDkgAdapterDeclaration {
    pub adapter_id: String,
    pub adapter_version: String,
    pub source_dkg_id: String,
    pub target_dkg_id: String,
    pub allowed_relations: Vec<CrossDkgRelation>,
    pub capability: AdapterCapability,
    /// Human-readable policy reference; not itself an authenticated policy.
    pub policy_ref: String,
}

impl CrossDkgAdapterDeclaration {
    pub fn validate(&self) -> Result<(), AdapterReceiptError> {
        if [&self.adapter_id, &self.adapter_version, &self.source_dkg_id,
            &self.target_dkg_id, &self.policy_ref].iter().any(|s| s.trim().is_empty()) {
            return Err(AdapterReceiptError::MissingDeclarationField);
        }
        if self.allowed_relations.is_empty() {
            return Err(AdapterReceiptError::NoAllowedRelations);
        }
        for (i, relation) in self.allowed_relations.iter().enumerate() {
            if self.allowed_relations[..i].contains(relation) {
                return Err(AdapterReceiptError::DuplicateAllowedRelation);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossDkgAdapterReceipt {
    pub receipt_version: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub policy_ref: String,
    pub source_dkg_id: String,
    pub target_dkg_id: String,
    pub link_digest: String,
    pub source_record_id: String,
    pub source_record_digest: String,
    pub target_record_id: String,
    pub target_record_digest: String,
    pub relation: CrossDkgRelation,
    pub capability: AdapterCapability,
    pub disposition: AdapterDisposition,
    pub output_record_id: String,
    pub output_record_digest: String,
    pub receipt_digest: String,
}

impl CrossDkgAdapterReceipt {
    pub fn derive_reference(
        declaration: &CrossDkgAdapterDeclaration,
        link: &CrossDkgFederationLink,
        disposition: AdapterDisposition,
        output_record_id: impl Into<String>,
        output_record_digest: impl Into<String>,
    ) -> Result<Self, AdapterReceiptError> {
        declaration.validate()?;
        link.verify_integrity().map_err(|_| AdapterReceiptError::InvalidLink)?;
        if declaration.source_dkg_id != link.source_dkg_id {
            return Err(AdapterReceiptError::SourceDkgMismatch);
        }
        if declaration.target_dkg_id != link.target_dkg_id {
            return Err(AdapterReceiptError::TargetDkgMismatch);
        }
        if !declaration.allowed_relations.contains(&link.relation) {
            return Err(AdapterReceiptError::RelationNotAllowed);
        }
        if !declaration.capability.permits(disposition) {
            return Err(AdapterReceiptError::CapabilityExceeded);
        }
        let output_record_id = output_record_id.into();
        let output_record_digest = output_record_digest.into();
        if output_record_id.trim().is_empty() || output_record_digest.trim().is_empty() {
            return Err(AdapterReceiptError::MissingOutput);
        }
        let mut receipt = Self {
            receipt_version: VERSION.into(),
            adapter_id: declaration.adapter_id.clone(),
            adapter_version: declaration.adapter_version.clone(),
            policy_ref: declaration.policy_ref.clone(),
            source_dkg_id: link.source_dkg_id.clone(),
            target_dkg_id: link.target_dkg_id.clone(),
            link_digest: link.link_digest.clone(),
            source_record_id: link.source_record_id.clone(),
            source_record_digest: link.source_record_digest.clone(),
            target_record_id: link.target_record_id.clone(),
            target_record_digest: link.target_record_digest.clone(),
            relation: link.relation,
            capability: declaration.capability,
            disposition,
            output_record_id,
            output_record_digest,
            receipt_digest: String::new(),
        };
        receipt.receipt_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify_against(
        &self,
        declaration: &CrossDkgAdapterDeclaration,
        link: &CrossDkgFederationLink,
    ) -> Result<(), AdapterReceiptError> {
        declaration.validate()?;
        link.verify_integrity().map_err(|_| AdapterReceiptError::InvalidLink)?;
        if self.receipt_version != VERSION { return Err(AdapterReceiptError::VersionMismatch); }
        if self.adapter_id != declaration.adapter_id || self.adapter_version != declaration.adapter_version
            || self.policy_ref != declaration.policy_ref || self.capability != declaration.capability {
            return Err(AdapterReceiptError::DeclarationMismatch);
        }
        if self.source_dkg_id != declaration.source_dkg_id || self.source_dkg_id != link.source_dkg_id {
            return Err(AdapterReceiptError::SourceDkgMismatch);
        }
        if self.target_dkg_id != declaration.target_dkg_id || self.target_dkg_id != link.target_dkg_id {
            return Err(AdapterReceiptError::TargetDkgMismatch);
        }
        if !declaration.allowed_relations.contains(&link.relation) || self.relation != link.relation {
            return Err(AdapterReceiptError::RelationNotAllowed);
        }
        if self.link_digest != link.link_digest || self.source_record_id != link.source_record_id
            || self.source_record_digest != link.source_record_digest
            || self.target_record_id != link.target_record_id
            || self.target_record_digest != link.target_record_digest {
            return Err(AdapterReceiptError::LinkageMismatch);
        }
        if !declaration.capability.permits(self.disposition) {
            return Err(AdapterReceiptError::CapabilityExceeded);
        }
        if self.output_record_id.trim().is_empty() || self.output_record_digest.trim().is_empty() {
            return Err(AdapterReceiptError::MissingOutput);
        }
        if self.receipt_digest != self.compute_digest() {
            return Err(AdapterReceiptError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for field in [
            self.receipt_version.as_str(), self.adapter_id.as_str(),
            self.adapter_version.as_str(), self.policy_ref.as_str(),
            self.source_dkg_id.as_str(), self.target_dkg_id.as_str(),
            self.link_digest.as_str(), self.source_record_id.as_str(),
            self.source_record_digest.as_str(), self.target_record_id.as_str(),
            self.target_record_digest.as_str(), self.relation.as_str(),
            self.capability.as_str(), self.disposition.as_str(),
            self.output_record_id.as_str(), self.output_record_digest.as_str(),
        ] {
            h.update((field.len() as u64).to_be_bytes());
            h.update(field.as_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }
}

impl CrossDkgRelation {
    fn as_str(self) -> &'static str {
        super::cross_dkg_federation::CrossDkgRelation::as_str(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterReceiptError {
    MissingDeclarationField, NoAllowedRelations, DuplicateAllowedRelation,
    InvalidLink, SourceDkgMismatch, TargetDkgMismatch, RelationNotAllowed,
    CapabilityExceeded, MissingOutput, VersionMismatch, DeclarationMismatch,
    LinkageMismatch, ReceiptDigestMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declaration(capability: AdapterCapability) -> CrossDkgAdapterDeclaration {
        CrossDkgAdapterDeclaration {
            adapter_id: "mycelix-to-symthaea".into(), adapter_version: "1.0".into(),
            source_dkg_id: "mycelix".into(), target_dkg_id: "symthaea".into(),
            allowed_relations: vec![CrossDkgRelation::References, CrossDkgRelation::ContextFor],
            capability, policy_ref: "policy:research-reference".into(),
        }
    }
    fn link() -> CrossDkgFederationLink {
        CrossDkgFederationLink::new("mycelix","sha256:g1","claim:1","sha256:r1",
            CrossDkgRelation::References,"symthaea","sha256:g2","observation:2","sha256:r2").unwrap()
    }
    #[test]
    fn reference_receipt_verifies() {
        let d=declaration(AdapterCapability::ReferenceOnly); let l=link();
        let r=CrossDkgAdapterReceipt::derive_reference(&d,&l,AdapterDisposition::ReferenceOnly,
            "local:ref1","sha256:out").unwrap();
        assert!(r.verify_against(&d,&l).is_ok());
    }
    #[test]
    fn capability_cannot_be_exceeded() {
        assert_eq!(CrossDkgAdapterReceipt::derive_reference(&declaration(AdapterCapability::ReferenceOnly),
            &link(),AdapterDisposition::DerivedMetadata,"local:1","sha256:o"),
            Err(AdapterReceiptError::CapabilityExceeded));
    }
    #[test]
    fn unlisted_relation_is_rejected() {
        let mut l=link(); l.relation=CrossDkgRelation::Supersedes;
        assert_eq!(CrossDkgAdapterReceipt::derive_reference(&declaration(AdapterCapability::DerivedMetadata),
            &l,AdapterDisposition::DerivedMetadata,"local:1","sha256:o"),Err(AdapterReceiptError::InvalidLink));
    }
    #[test]
    fn tampered_output_is_rejected() {
        let d=declaration(AdapterCapability::ContextAttachment); let l=link();
        let mut r=CrossDkgAdapterReceipt::derive_reference(&d,&l,AdapterDisposition::ContextOnly,
            "local:ref1","sha256:out").unwrap();
        r.output_record_id="local:tampered".into();
        assert_eq!(r.verify_against(&d,&l),Err(AdapterReceiptError::ReceiptDigestMismatch));
    }
    #[test]
    fn serde_roundtrip() {
        let d=declaration(AdapterCapability::ReferenceOnly); let l=link();
        let r=CrossDkgAdapterReceipt::derive_reference(&d,&l,AdapterDisposition::ReferenceOnly,
            "local:ref1","sha256:out").unwrap();
        let bytes=serde_json::to_vec(&r).unwrap();
        let decoded:CrossDkgAdapterReceipt=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(r,decoded); assert!(decoded.verify_against(&d,&l).is_ok());
    }
}
