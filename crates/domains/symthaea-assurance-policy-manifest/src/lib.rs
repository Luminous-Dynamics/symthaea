// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned provenance for the policies that define safety-evidence readiness.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_evidence_deployment_scope::DeploymentScopedSafetyReceipt;

const MANIFEST_DIGEST_SCHEMA: &[u8] = b"symthaea-assurance-policy-manifest-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AssurancePolicyKind {
    EvidenceLifecycle,
    DeploymentScope,
    EvidenceQuarantine,
    TrustedTime,
    Requalification,
    VerifierDiversity,
    AtomicCoverage,
    EvidenceDependency,
}

impl AssurancePolicyKind {
    pub const REQUIRED: [Self; 8] = [
        Self::EvidenceLifecycle,
        Self::DeploymentScope,
        Self::EvidenceQuarantine,
        Self::TrustedTime,
        Self::Requalification,
        Self::VerifierDiversity,
        Self::AtomicCoverage,
        Self::EvidenceDependency,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::EvidenceLifecycle => "evidence-lifecycle",
            Self::DeploymentScope => "deployment-scope",
            Self::EvidenceQuarantine => "evidence-quarantine",
            Self::TrustedTime => "trusted-time",
            Self::Requalification => "requalification",
            Self::VerifierDiversity => "verifier-diversity",
            Self::AtomicCoverage => "atomic-coverage",
            Self::EvidenceDependency => "evidence-dependency",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssurancePolicyBinding {
    pub kind: AssurancePolicyKind,
    pub policy_id: String,
    pub schema_version: String,
    pub content_digest: String,
    pub evidence_refs: Vec<String>,
}

impl AssurancePolicyBinding {
    pub fn validate(&self) -> bool {
        !self.policy_id.trim().is_empty()
            && !self.schema_version.trim().is_empty()
            && valid_digest(&self.content_digest)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssurancePolicyManifest {
    pub schema_version: String,
    pub manifest_id: String,
    pub revision: u64,
    pub safety_contract_digest: String,
    pub deployment_id: String,
    pub configuration_digest: String,
    pub model_manifest_digest: Option<String>,
    pub calibration_manifest_digest: Option<String>,
    /// Previous manifest digest for revisions after revision 1.
    pub predecessor_manifest_digest: Option<String>,
    /// Durable reviewed change record. Required after revision 1.
    pub change_ref: Option<String>,
    pub policies: Vec<AssurancePolicyBinding>,
    pub evidence_refs: Vec<String>,
}

impl AssurancePolicyManifest {
    /// Structural validity plus exact coverage of every currently required policy kind.
    pub fn validate_complete(&self) -> bool {
        if self.schema_version.trim().is_empty()
            || self.manifest_id.trim().is_empty()
            || self.revision == 0
            || !valid_digest(&self.safety_contract_digest)
            || self.deployment_id.trim().is_empty()
            || !valid_digest(&self.configuration_digest)
            || self
                .model_manifest_digest
                .as_ref()
                .is_some_and(|value| !valid_digest(value))
            || self
                .calibration_manifest_digest
                .as_ref()
                .is_some_and(|value| !valid_digest(value))
            || self.evidence_refs.is_empty()
            || self.evidence_refs.iter().any(|value| value.trim().is_empty())
        {
            return false;
        }

        match self.revision {
            1 => {
                if self.predecessor_manifest_digest.is_some() || self.change_ref.is_some() {
                    return false;
                }
            }
            _ => {
                if self
                    .predecessor_manifest_digest
                    .as_ref()
                    .is_none_or(|value| !valid_digest(value))
                    || self
                        .change_ref
                        .as_ref()
                        .is_none_or(|value| value.trim().is_empty())
                {
                    return false;
                }
            }
        }

        let mut seen = BTreeSet::new();
        for policy in &self.policies {
            if !policy.validate() || !seen.insert(policy.kind) {
                return false;
            }
        }

        let required = AssurancePolicyKind::REQUIRED.into_iter().collect::<BTreeSet<_>>();
        seen == required
    }

    /// Deterministic digest of the exact reviewed assurance rules and scope.
    pub fn manifest_digest(&self) -> String {
        let mut policies = self.policies.clone();
        policies.sort_by(|a, b| {
            a.kind
                .cmp(&b.kind)
                .then_with(|| a.policy_id.cmp(&b.policy_id))
        });

        let mut evidence_refs = self.evidence_refs.clone();
        evidence_refs.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(MANIFEST_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.manifest_id);
        push_field(&mut hasher, &self.revision.to_string());
        push_field(&mut hasher, &self.safety_contract_digest);
        push_field(&mut hasher, &self.deployment_id);
        push_field(&mut hasher, &self.configuration_digest);
        push_optional(&mut hasher, self.model_manifest_digest.as_deref());
        push_optional(&mut hasher, self.calibration_manifest_digest.as_deref());
        push_optional(&mut hasher, self.predecessor_manifest_digest.as_deref());
        push_optional(&mut hasher, self.change_ref.as_deref());

        for policy in policies {
            push_field(&mut hasher, policy.kind.code());
            push_field(&mut hasher, &policy.policy_id);
            push_field(&mut hasher, &policy.schema_version);
            push_field(&mut hasher, &policy.content_digest);
            let mut refs = policy.evidence_refs;
            refs.sort();
            for reference in refs {
                push_field(&mut hasher, &reference);
            }
            hasher.update(b"policy-end\0");
        }

        for reference in evidence_refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Durable record that an external cryptographic verification process checked a
/// signature over the exact manifest digest.
///
/// This crate validates binding/provenance only; it does not perform signature
/// verification itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSignatureVerificationReceipt {
    pub receipt_id: String,
    pub manifest_digest: String,
    pub signer_ref: String,
    pub key_ref: String,
    pub signature_algorithm: String,
    pub signature_ref: String,
    pub verified_by_ref: String,
    pub verification_ref: String,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl ManifestSignatureVerificationReceipt {
    pub fn validate_for(&self, manifest: &AssurancePolicyManifest) -> bool {
        manifest.validate_complete()
            && !self.receipt_id.trim().is_empty()
            && self.manifest_digest == manifest.manifest_digest()
            && !self.signer_ref.trim().is_empty()
            && !self.key_ref.trim().is_empty()
            && !self.signature_algorithm.trim().is_empty()
            && !self.signature_ref.trim().is_empty()
            && !self.verified_by_ref.trim().is_empty()
            && !self.verification_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyScopedSafetyReceipt {
    pub deployment_receipt: DeploymentScopedSafetyReceipt,
    pub assurance_manifest_digest: String,
    pub manifest_signature_receipt_id: String,
    pub policy_scope_ref: String,
}

impl PolicyScopedSafetyReceipt {
    pub fn validate(&self) -> bool {
        self.deployment_receipt.validate()
            && valid_digest(&self.assurance_manifest_digest)
            && !self.manifest_signature_receipt_id.trim().is_empty()
            && !self.policy_scope_ref.trim().is_empty()
    }

    pub fn matches_manifest(&self, manifest: &AssurancePolicyManifest) -> bool {
        self.validate()
            && manifest.validate_complete()
            && self.assurance_manifest_digest == manifest.manifest_digest()
            && self.deployment_receipt.scoped_receipt.contract_digest
                == manifest.safety_contract_digest
            && self.deployment_receipt.deployment_id == manifest.deployment_id
            && self.deployment_receipt.configuration_digest == manifest.configuration_digest
            && self.deployment_receipt.model_manifest_digest == manifest.model_manifest_digest
            && self.deployment_receipt.calibration_manifest_digest
                == manifest.calibration_manifest_digest
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyScopeError {
    InvalidManifest,
    InvalidSignatureVerification,
    DeploymentScopeMismatch,
    InvalidPolicyScopeReference,
}

pub fn bind_receipt_to_assurance_manifest(
    deployment_receipt: DeploymentScopedSafetyReceipt,
    manifest: &AssurancePolicyManifest,
    signature_receipt: &ManifestSignatureVerificationReceipt,
    policy_scope_ref: impl Into<String>,
) -> Result<PolicyScopedSafetyReceipt, PolicyScopeError> {
    if !manifest.validate_complete() {
        return Err(PolicyScopeError::InvalidManifest);
    }
    if !signature_receipt.validate_for(manifest) {
        return Err(PolicyScopeError::InvalidSignatureVerification);
    }
    if deployment_receipt.scoped_receipt.contract_digest != manifest.safety_contract_digest
        || deployment_receipt.deployment_id != manifest.deployment_id
        || deployment_receipt.configuration_digest != manifest.configuration_digest
        || deployment_receipt.model_manifest_digest != manifest.model_manifest_digest
        || deployment_receipt.calibration_manifest_digest != manifest.calibration_manifest_digest
    {
        return Err(PolicyScopeError::DeploymentScopeMismatch);
    }
    let policy_scope_ref = policy_scope_ref.into();
    if policy_scope_ref.trim().is_empty() {
        return Err(PolicyScopeError::InvalidPolicyScopeReference);
    }

    Ok(PolicyScopedSafetyReceipt {
        deployment_receipt,
        assurance_manifest_digest: manifest.manifest_digest(),
        manifest_signature_receipt_id: signature_receipt.receipt_id.clone(),
        policy_scope_ref,
    })
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(value.trim().as_bytes());
    hasher.update(b"\0");
}

fn push_optional(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(b"some\0");
            push_field(hasher, value);
        }
        None => hasher.update(b"none\0"),
    }
}

fn valid_digest(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed
        .split_once(':')
        .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_deployment_scope::DeploymentScopedSafetyReceipt;
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
    use symthaea_formal_safety::{EvidenceKind, SafetyEvidenceReceipt};

    fn binding(kind: AssurancePolicyKind, suffix: &str) -> AssurancePolicyBinding {
        AssurancePolicyBinding {
            kind,
            policy_id: format!("policy:{suffix}"),
            schema_version: "1".into(),
            content_digest: format!("blake3:{suffix}"),
            evidence_refs: vec![format!("review:{suffix}")],
        }
    }

    fn manifest() -> AssurancePolicyManifest {
        AssurancePolicyManifest {
            schema_version: "1".into(),
            manifest_id: "assurance:harbor-1".into(),
            revision: 1,
            safety_contract_digest: "blake3:contract-v1".into(),
            deployment_id: "harbor-node-1".into(),
            configuration_digest: "blake3:config-v1".into(),
            model_manifest_digest: Some("blake3:model-v1".into()),
            calibration_manifest_digest: Some("blake3:cal-v1".into()),
            predecessor_manifest_digest: None,
            change_ref: None,
            policies: vec![
                binding(AssurancePolicyKind::EvidenceLifecycle, "lifecycle"),
                binding(AssurancePolicyKind::DeploymentScope, "scope"),
                binding(AssurancePolicyKind::EvidenceQuarantine, "quarantine"),
                binding(AssurancePolicyKind::TrustedTime, "time"),
                binding(AssurancePolicyKind::Requalification, "requalification"),
                binding(AssurancePolicyKind::VerifierDiversity, "diversity"),
                binding(AssurancePolicyKind::AtomicCoverage, "atomic"),
                binding(AssurancePolicyKind::EvidenceDependency, "dependency"),
            ],
            evidence_refs: vec!["review:assurance-manifest".into()],
        }
    }

    fn signature(manifest: &AssurancePolicyManifest) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: "sig-receipt-1".into(),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: "signer:safety-board".into(),
            key_ref: "key:safety-board:v1".into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: "signature:artifact:1".into(),
            verified_by_ref: "verifier:signature-service".into(),
            verification_ref: "verification:signature-run-1".into(),
            verified_at_ms: 1_000,
            evidence_refs: vec!["audit:signature-1".into()],
        }
    }

    fn deployment_receipt(manifest: &AssurancePolicyManifest) -> DeploymentScopedSafetyReceipt {
        DeploymentScopedSafetyReceipt {
            scoped_receipt: ScopedSafetyEvidenceReceipt {
                receipt: SafetyEvidenceReceipt {
                    receipt_id: "r1".into(),
                    obligation_key: "blake3:obligation".into(),
                    evidence_kind: EvidenceKind::Test,
                    evidence_ref: "artifact:r1".into(),
                    evidence_digest: "blake3:r1".into(),
                    verifier_ref: "verifier:a".into(),
                    verified_at_ms: 900,
                },
                contract_digest: manifest.safety_contract_digest.clone(),
                valid_from_ms: 900,
                valid_until_ms: 5_000,
                applicability_refs: vec!["deployment:harbor-node-1".into()],
            },
            deployment_id: manifest.deployment_id.clone(),
            configuration_digest: manifest.configuration_digest.clone(),
            model_manifest_digest: manifest.model_manifest_digest.clone(),
            calibration_manifest_digest: manifest.calibration_manifest_digest.clone(),
            scope_binding_ref: "scope:r1".into(),
        }
    }

    #[test]
    fn complete_manifest_is_valid_and_order_independent() {
        let a = manifest();
        let mut b = a.clone();
        b.policies.reverse();
        b.evidence_refs.reverse();
        assert!(a.validate_complete());
        assert!(b.validate_complete());
        assert_eq!(a.manifest_digest(), b.manifest_digest());
        assert!(!a.grants_physical_authority());
    }

    #[test]
    fn policy_change_changes_manifest_digest() {
        let a = manifest();
        let mut b = a.clone();
        b.policies[0].content_digest = "blake3:lifecycle-v2".into();
        assert_ne!(a.manifest_digest(), b.manifest_digest());
    }

    #[test]
    fn incomplete_policy_set_is_rejected() {
        let mut value = manifest();
        value
            .policies
            .retain(|policy| policy.kind != AssurancePolicyKind::TrustedTime);
        assert!(!value.validate_complete());
    }

    #[test]
    fn revision_requires_predecessor_and_change_record() {
        let mut value = manifest();
        value.revision = 2;
        assert!(!value.validate_complete());
        value.predecessor_manifest_digest = Some("blake3:previous".into());
        value.change_ref = Some("change:review-2".into());
        assert!(value.validate_complete());
    }

    #[test]
    fn signature_receipt_must_bind_exact_manifest_digest() {
        let value = manifest();
        let mut receipt = signature(&value);
        assert!(receipt.validate_for(&value));
        receipt.manifest_digest = "blake3:other".into();
        assert!(!receipt.validate_for(&value));
        assert!(!signature(&value).grants_physical_authority());
    }

    #[test]
    fn wrong_safety_contract_cannot_be_policy_scoped() {
        let value = manifest();
        let mut receipt = deployment_receipt(&value);
        receipt.scoped_receipt.contract_digest = "blake3:other-contract".into();
        assert_eq!(
            bind_receipt_to_assurance_manifest(
                receipt,
                &value,
                &signature(&value),
                "policy-scope:r1",
            ),
            Err(PolicyScopeError::DeploymentScopeMismatch)
        );
    }

    #[test]
    fn scoped_receipt_stops_matching_after_policy_drift() {
        let original = manifest();
        let scoped = bind_receipt_to_assurance_manifest(
            deployment_receipt(&original),
            &original,
            &signature(&original),
            "policy-scope:r1",
        )
        .unwrap();
        assert!(scoped.matches_manifest(&original));

        let mut changed = original.clone();
        changed.revision = 2;
        changed.predecessor_manifest_digest = Some(original.manifest_digest());
        changed.change_ref = Some("change:policy-review-2".into());
        changed.policies[5].content_digest = "blake3:diversity-v2".into();
        assert!(changed.validate_complete());
        assert!(!scoped.matches_manifest(&changed));
        assert!(!scoped.grants_physical_authority());
    }
}
