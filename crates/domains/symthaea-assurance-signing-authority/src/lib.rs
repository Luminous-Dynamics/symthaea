// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent signer/key authority governance for assurance-policy lineages.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_lineage::{
    AssurancePolicyLineageStatus, SignedAssurancePolicyRevision, assess_signed_policy_lineage,
};

const GOVERNANCE_DIGEST_SCHEMA: &[u8] = b"symthaea-assurance-signing-authority-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSigningAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub manifest_id: String,
    pub deployment_id: String,
    pub initial_signer_ref: String,
    pub initial_key_ref: String,
    pub allowed_signature_algorithms: Vec<String>,
    /// Durable external trust-root / governance reference. This policy is not
    /// self-authorized by the assurance manifest it governs.
    pub trust_root_ref: String,
    pub evidence_refs: Vec<String>,
}

impl ManifestSigningAuthorityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.manifest_id.trim().is_empty()
            || self.deployment_id.trim().is_empty()
            || self.initial_signer_ref.trim().is_empty()
            || self.initial_key_ref.trim().is_empty()
            || self.trust_root_ref.trim().is_empty()
            || self.allowed_signature_algorithms.is_empty()
            || self.evidence_refs.is_empty()
            || self.evidence_refs.iter().any(|value| value.trim().is_empty())
            || self
                .allowed_signature_algorithms
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return false;
        }
        self.allowed_signature_algorithms
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            == self.allowed_signature_algorithms.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningAuthorityTransition {
    pub transition_id: String,
    /// Revision signed by the new signer/key pair. Revision 1 is governed by the
    /// policy's initial signer/key and cannot be a transition revision.
    pub effective_revision: u64,
    pub from_signer_ref: String,
    pub from_key_ref: String,
    pub to_signer_ref: String,
    pub to_key_ref: String,
    /// Reviewed governance authorization for the rotation/replacement.
    pub authorization_ref: String,
    /// Independent verification of the authorization artifact/process.
    pub independent_verification_ref: String,
    pub evidence_refs: Vec<String>,
}

impl SigningAuthorityTransition {
    pub fn validate(&self) -> bool {
        !self.transition_id.trim().is_empty()
            && self.effective_revision > 1
            && !self.from_signer_ref.trim().is_empty()
            && !self.from_key_ref.trim().is_empty()
            && !self.to_signer_ref.trim().is_empty()
            && !self.to_key_ref.trim().is_empty()
            && (self.from_signer_ref != self.to_signer_ref || self.from_key_ref != self.to_key_ref)
            && !self.authorization_ref.trim().is_empty()
            && !self.independent_verification_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningAuthorityGovernance {
    pub policy: ManifestSigningAuthorityPolicy,
    pub transitions: Vec<SigningAuthorityTransition>,
}

impl SigningAuthorityGovernance {
    pub fn validate(&self) -> bool {
        if !self.policy.validate() {
            return false;
        }
        let mut ids = BTreeSet::new();
        let mut revisions = BTreeSet::new();
        self.transitions.iter().all(|transition| {
            transition.validate()
                && ids.insert(transition.transition_id.clone())
                && revisions.insert(transition.effective_revision)
        })
    }

    pub fn governance_digest(&self) -> String {
        let mut algorithms = self.policy.allowed_signature_algorithms.clone();
        algorithms.sort();
        let mut policy_refs = self.policy.evidence_refs.clone();
        policy_refs.sort();
        let mut transitions = self.transitions.clone();
        transitions.sort_by(|a, b| {
            a.effective_revision
                .cmp(&b.effective_revision)
                .then_with(|| a.transition_id.cmp(&b.transition_id))
        });

        let mut hasher = blake3::Hasher::new();
        hasher.update(GOVERNANCE_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.policy.schema_version);
        push_field(&mut hasher, &self.policy.policy_id);
        push_field(&mut hasher, &self.policy.manifest_id);
        push_field(&mut hasher, &self.policy.deployment_id);
        push_field(&mut hasher, &self.policy.initial_signer_ref);
        push_field(&mut hasher, &self.policy.initial_key_ref);
        push_field(&mut hasher, &self.policy.trust_root_ref);
        for algorithm in algorithms {
            push_field(&mut hasher, &algorithm);
        }
        for reference in policy_refs {
            push_field(&mut hasher, &reference);
        }
        hasher.update(b"policy-end\0");

        for transition in transitions {
            push_field(&mut hasher, &transition.transition_id);
            push_field(&mut hasher, &transition.effective_revision.to_string());
            push_field(&mut hasher, &transition.from_signer_ref);
            push_field(&mut hasher, &transition.from_key_ref);
            push_field(&mut hasher, &transition.to_signer_ref);
            push_field(&mut hasher, &transition.to_key_ref);
            push_field(&mut hasher, &transition.authorization_ref);
            push_field(&mut hasher, &transition.independent_verification_ref);
            let mut refs = transition.evidence_refs;
            refs.sort();
            for reference in refs {
                push_field(&mut hasher, &reference);
            }
            hasher.update(b"transition-end\0");
        }

        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SigningAuthorityStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SigningAuthorityIssue {
    InvalidGovernance,
    UnderlyingPolicyLineageInvalid,
    ManifestIdMismatch,
    DeploymentIdMismatch,
    TransitionForUnknownRevision(u64),
    TransitionFromUnexpectedAuthority {
        revision: u64,
        expected_signer_ref: String,
        expected_key_ref: String,
        observed_signer_ref: String,
        observed_key_ref: String,
    },
    SignerOrKeyNotAuthorized {
        revision: u64,
        expected_signer_ref: String,
        expected_key_ref: String,
        observed_signer_ref: String,
        observed_key_ref: String,
    },
    UnsupportedSignatureAlgorithm {
        revision: u64,
        algorithm: String,
    },
    SignerAlsoVerifiedOwnManifest {
        revision: u64,
        signer_ref: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningAuthorityReport {
    pub status: SigningAuthorityStatus,
    pub governance_digest: String,
    pub lineage_tip_revision: Option<u64>,
    pub final_signer_ref: Option<String>,
    pub final_key_ref: Option<String>,
    pub issues: Vec<SigningAuthorityIssue>,
}

impl SigningAuthorityReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Validate every signed assurance-policy revision against an independent
/// signer/key governance policy and its explicitly reviewed transitions.
pub fn assess_signing_authority(
    lineage: &[SignedAssurancePolicyRevision],
    governance: &SigningAuthorityGovernance,
) -> SigningAuthorityReport {
    let lineage_report = assess_signed_policy_lineage(lineage);
    let governance_digest = governance.governance_digest();
    let mut issues = Vec::new();

    if !governance.validate() {
        issues.push(SigningAuthorityIssue::InvalidGovernance);
    }
    if lineage_report.status != AssurancePolicyLineageStatus::Valid {
        issues.push(SigningAuthorityIssue::UnderlyingPolicyLineageInvalid);
    }

    let mut ordered = lineage.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|revision| revision.manifest.revision);

    if let Some(first) = ordered.first() {
        if first.manifest.manifest_id != governance.policy.manifest_id {
            issues.push(SigningAuthorityIssue::ManifestIdMismatch);
        }
        if first.manifest.deployment_id != governance.policy.deployment_id {
            issues.push(SigningAuthorityIssue::DeploymentIdMismatch);
        }
    }

    let known_revisions = ordered
        .iter()
        .map(|revision| revision.manifest.revision)
        .collect::<BTreeSet<_>>();
    let transitions = governance
        .transitions
        .iter()
        .map(|transition| (transition.effective_revision, transition))
        .collect::<BTreeMap<_, _>>();

    for transition in &governance.transitions {
        if !known_revisions.contains(&transition.effective_revision) {
            issues.push(SigningAuthorityIssue::TransitionForUnknownRevision(
                transition.effective_revision,
            ));
        }
    }

    let mut expected_signer = governance.policy.initial_signer_ref.clone();
    let mut expected_key = governance.policy.initial_key_ref.clone();

    for revision in &ordered {
        let number = revision.manifest.revision;
        if let Some(transition) = transitions.get(&number) {
            if transition.from_signer_ref != expected_signer
                || transition.from_key_ref != expected_key
            {
                issues.push(SigningAuthorityIssue::TransitionFromUnexpectedAuthority {
                    revision: number,
                    expected_signer_ref: expected_signer.clone(),
                    expected_key_ref: expected_key.clone(),
                    observed_signer_ref: transition.from_signer_ref.clone(),
                    observed_key_ref: transition.from_key_ref.clone(),
                });
            } else {
                expected_signer = transition.to_signer_ref.clone();
                expected_key = transition.to_key_ref.clone();
            }
        }

        let signature = &revision.signature_receipt;
        if signature.signer_ref != expected_signer || signature.key_ref != expected_key {
            issues.push(SigningAuthorityIssue::SignerOrKeyNotAuthorized {
                revision: number,
                expected_signer_ref: expected_signer.clone(),
                expected_key_ref: expected_key.clone(),
                observed_signer_ref: signature.signer_ref.clone(),
                observed_key_ref: signature.key_ref.clone(),
            });
        }
        if !governance
            .policy
            .allowed_signature_algorithms
            .iter()
            .any(|algorithm| algorithm == &signature.signature_algorithm)
        {
            issues.push(SigningAuthorityIssue::UnsupportedSignatureAlgorithm {
                revision: number,
                algorithm: signature.signature_algorithm.clone(),
            });
        }
        if signature.verified_by_ref == signature.signer_ref {
            issues.push(SigningAuthorityIssue::SignerAlsoVerifiedOwnManifest {
                revision: number,
                signer_ref: signature.signer_ref.clone(),
            });
        }
    }

    let status = if issues.is_empty() {
        SigningAuthorityStatus::Valid
    } else {
        SigningAuthorityStatus::Invalid
    };

    SigningAuthorityReport {
        status,
        governance_digest,
        lineage_tip_revision: lineage_report.tip_revision,
        final_signer_ref: ordered.last().map(|_| expected_signer),
        final_key_ref: ordered.last().map(|_| expected_key),
        issues,
    }
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(value.trim().as_bytes());
    hasher.update(b"\0");
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind, AssurancePolicyManifest,
        ManifestSignatureVerificationReceipt,
    };

    fn binding(kind: AssurancePolicyKind, revision: u64) -> AssurancePolicyBinding {
        AssurancePolicyBinding {
            kind,
            policy_id: format!("policy:{}", kind.code()),
            schema_version: "1".into(),
            content_digest: format!("blake3:{}-r{revision}", kind.code()),
            evidence_refs: vec![format!("review:{}:r{revision}", kind.code())],
        }
    }

    fn manifest(revision: u64, predecessor: Option<String>) -> AssurancePolicyManifest {
        AssurancePolicyManifest {
            schema_version: "1".into(),
            manifest_id: "assurance:node-1".into(),
            revision,
            safety_contract_digest: "blake3:contract".into(),
            deployment_id: "node-1".into(),
            configuration_digest: format!("blake3:config-r{revision}"),
            model_manifest_digest: Some(format!("blake3:model-r{revision}")),
            calibration_manifest_digest: Some(format!("blake3:cal-r{revision}")),
            predecessor_manifest_digest: predecessor,
            change_ref: (revision > 1).then(|| format!("change:r{revision}")),
            policies: AssurancePolicyKind::REQUIRED
                .into_iter()
                .map(|kind| binding(kind, revision))
                .collect(),
            evidence_refs: vec![format!("review:manifest-r{revision}")],
        }
    }

    fn signature(
        manifest: &AssurancePolicyManifest,
        signer: &str,
        key: &str,
    ) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: format!("sig:r{}", manifest.revision),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: signer.into(),
            key_ref: key.into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: format!("signature:r{}", manifest.revision),
            verified_by_ref: "verifier:independent".into(),
            verification_ref: format!("verification:r{}", manifest.revision),
            verified_at_ms: manifest.revision * 100,
            evidence_refs: vec![format!("audit:r{}", manifest.revision)],
        }
    }

    fn signed(
        manifest: AssurancePolicyManifest,
        signer: &str,
        key: &str,
    ) -> SignedAssurancePolicyRevision {
        let signature_receipt = signature(&manifest, signer, key);
        SignedAssurancePolicyRevision {
            manifest,
            signature_receipt,
        }
    }

    fn lineage_with_rotation() -> Vec<SignedAssurancePolicyRevision> {
        let r1 = manifest(1, None);
        let r2 = manifest(2, Some(r1.manifest_digest()));
        let r3 = manifest(3, Some(r2.manifest_digest()));
        vec![
            signed(r1, "signer:a", "key:a1"),
            signed(r2, "signer:a", "key:a1"),
            signed(r3, "signer:b", "key:b1"),
        ]
    }

    fn policy() -> ManifestSigningAuthorityPolicy {
        ManifestSigningAuthorityPolicy {
            schema_version: "1".into(),
            policy_id: "signing-authority-v1".into(),
            manifest_id: "assurance:node-1".into(),
            deployment_id: "node-1".into(),
            initial_signer_ref: "signer:a".into(),
            initial_key_ref: "key:a1".into(),
            allowed_signature_algorithms: vec!["ed25519".into()],
            trust_root_ref: "trust-root:safety-board".into(),
            evidence_refs: vec!["review:signing-authority".into()],
        }
    }

    fn rotation() -> SigningAuthorityTransition {
        SigningAuthorityTransition {
            transition_id: "rotation:r3".into(),
            effective_revision: 3,
            from_signer_ref: "signer:a".into(),
            from_key_ref: "key:a1".into(),
            to_signer_ref: "signer:b".into(),
            to_key_ref: "key:b1".into(),
            authorization_ref: "authorization:safety-board:r3".into(),
            independent_verification_ref: "verification:rotation:r3".into(),
            evidence_refs: vec!["audit:rotation:r3".into()],
        }
    }

    #[test]
    fn reviewed_signer_rotation_is_valid() {
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![rotation()],
        };
        let report = assess_signing_authority(&lineage_with_rotation(), &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Valid);
        assert_eq!(report.final_signer_ref.as_deref(), Some("signer:b"));
        assert_eq!(report.final_key_ref.as_deref(), Some("key:b1"));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn signer_replacement_without_transition_is_rejected() {
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![],
        };
        let report = assess_signing_authority(&lineage_with_rotation(), &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            SigningAuthorityIssue::SignerOrKeyNotAuthorized { revision: 3, .. }
        )));
    }

    #[test]
    fn transition_from_wrong_previous_key_is_rejected() {
        let mut transition = rotation();
        transition.from_key_ref = "key:wrong".into();
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![transition],
        };
        let report = assess_signing_authority(&lineage_with_rotation(), &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            SigningAuthorityIssue::TransitionFromUnexpectedAuthority { revision: 3, .. }
        )));
    }

    #[test]
    fn unsupported_signature_algorithm_is_rejected() {
        let mut lineage = lineage_with_rotation();
        lineage[1].signature_receipt.signature_algorithm = "unknown-algorithm".into();
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![rotation()],
        };
        let report = assess_signing_authority(&lineage, &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            SigningAuthorityIssue::UnsupportedSignatureAlgorithm { revision: 2, .. }
        )));
    }

    #[test]
    fn signer_cannot_be_its_own_signature_verifier() {
        let mut lineage = lineage_with_rotation();
        lineage[0].signature_receipt.verified_by_ref = "signer:a".into();
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![rotation()],
        };
        let report = assess_signing_authority(&lineage, &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            SigningAuthorityIssue::SignerAlsoVerifiedOwnManifest { revision: 1, .. }
        )));
    }

    #[test]
    fn transition_for_unknown_revision_is_rejected() {
        let mut transition = rotation();
        transition.effective_revision = 4;
        let governance = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![transition],
        };
        let report = assess_signing_authority(&lineage_with_rotation(), &governance);
        assert_eq!(report.status, SigningAuthorityStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            SigningAuthorityIssue::TransitionForUnknownRevision(4)
        )));
    }

    #[test]
    fn governance_digest_changes_when_rotation_authority_changes() {
        let a = SigningAuthorityGovernance {
            policy: policy(),
            transitions: vec![rotation()],
        };
        let mut b = a.clone();
        b.transitions[0].authorization_ref = "authorization:other".into();
        assert_ne!(a.governance_digest(), b.governance_digest());
        assert!(!a.grants_physical_authority());
    }
}
