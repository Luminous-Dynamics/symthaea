// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Contiguous externally verified lineage for assurance-policy manifests.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedAssurancePolicyRevision {
    pub manifest: AssurancePolicyManifest,
    pub signature_receipt: ManifestSignatureVerificationReceipt,
}

impl SignedAssurancePolicyRevision {
    pub fn validate(&self) -> bool {
        self.manifest.validate_complete()
            && self.signature_receipt.validate_for(&self.manifest)
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssurancePolicyLineageStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssurancePolicyLineageIssue {
    EmptyLineage,
    InvalidRevision(u64),
    DuplicateRevision(u64),
    DuplicateManifestDigest(String),
    FirstRevisionIsNotOne(u64),
    RevisionGap {
        previous: u64,
        next: u64,
    },
    ManifestIdChanged {
        revision: u64,
    },
    DeploymentIdChanged {
        revision: u64,
    },
    PredecessorDigestMismatch {
        revision: u64,
        expected: String,
        observed: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssurancePolicyLineageReport {
    pub status: AssurancePolicyLineageStatus,
    pub manifest_id: Option<String>,
    pub deployment_id: Option<String>,
    pub first_revision: Option<u64>,
    pub tip_revision: Option<u64>,
    pub tip_manifest_digest: Option<String>,
    pub revision_count: usize,
    pub issues: Vec<AssurancePolicyLineageIssue>,
}

impl AssurancePolicyLineageReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    pub fn contains_tip(&self, manifest: &AssurancePolicyManifest) -> bool {
        let digest = manifest.manifest_digest();
        self.status == AssurancePolicyLineageStatus::Valid
            && self.tip_revision == Some(manifest.revision)
            && self.tip_manifest_digest.as_deref() == Some(digest.as_str())
    }
}

/// Validate a complete signed manifest lineage.
///
/// Revisions are sorted by revision number before assessment. A valid lineage
/// starts at revision 1, is contiguous, preserves manifest/deployment identity,
/// and binds every revision N>1 to the exact digest of revision N-1.
pub fn assess_signed_policy_lineage(
    revisions: &[SignedAssurancePolicyRevision],
) -> AssurancePolicyLineageReport {
    if revisions.is_empty() {
        return AssurancePolicyLineageReport {
            status: AssurancePolicyLineageStatus::Invalid,
            manifest_id: None,
            deployment_id: None,
            first_revision: None,
            tip_revision: None,
            tip_manifest_digest: None,
            revision_count: 0,
            issues: vec![AssurancePolicyLineageIssue::EmptyLineage],
        };
    }

    let mut ordered = revisions.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|revision| revision.manifest.revision);

    let manifest_id = ordered[0].manifest.manifest_id.clone();
    let deployment_id = ordered[0].manifest.deployment_id.clone();
    let mut issues = Vec::new();
    let mut seen_revisions = BTreeSet::new();
    let mut seen_digests = BTreeSet::new();

    for revision in &ordered {
        let number = revision.manifest.revision;
        if !revision.validate() {
            issues.push(AssurancePolicyLineageIssue::InvalidRevision(number));
        }
        if !seen_revisions.insert(number) {
            issues.push(AssurancePolicyLineageIssue::DuplicateRevision(number));
        }
        let digest = revision.manifest.manifest_digest();
        if !seen_digests.insert(digest.clone()) {
            issues.push(AssurancePolicyLineageIssue::DuplicateManifestDigest(digest));
        }
        if revision.manifest.manifest_id != manifest_id {
            issues.push(AssurancePolicyLineageIssue::ManifestIdChanged { revision: number });
        }
        if revision.manifest.deployment_id != deployment_id {
            issues.push(AssurancePolicyLineageIssue::DeploymentIdChanged { revision: number });
        }
    }

    if ordered[0].manifest.revision != 1 {
        issues.push(AssurancePolicyLineageIssue::FirstRevisionIsNotOne(
            ordered[0].manifest.revision,
        ));
    }

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        let previous_number = previous.manifest.revision;
        let current_number = current.manifest.revision;
        if current_number != previous_number.saturating_add(1) {
            issues.push(AssurancePolicyLineageIssue::RevisionGap {
                previous: previous_number,
                next: current_number,
            });
        }
        let expected = previous.manifest.manifest_digest();
        if current.manifest.predecessor_manifest_digest.as_deref() != Some(expected.as_str()) {
            issues.push(AssurancePolicyLineageIssue::PredecessorDigestMismatch {
                revision: current_number,
                expected,
                observed: current.manifest.predecessor_manifest_digest.clone(),
            });
        }
    }

    let tip = ordered.last().expect("non-empty lineage");
    let status = if issues.is_empty() {
        AssurancePolicyLineageStatus::Valid
    } else {
        AssurancePolicyLineageStatus::Invalid
    };

    AssurancePolicyLineageReport {
        status,
        manifest_id: Some(manifest_id),
        deployment_id: Some(deployment_id),
        first_revision: Some(ordered[0].manifest.revision),
        tip_revision: Some(tip.manifest.revision),
        tip_manifest_digest: Some(tip.manifest.manifest_digest()),
        revision_count: ordered.len(),
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind,
    };

    fn binding(kind: AssurancePolicyKind, revision: u64) -> AssurancePolicyBinding {
        AssurancePolicyBinding {
            kind,
            policy_id: format!("policy:{}", kind.code()),
            schema_version: "1".into(),
            content_digest: format!("blake3:{}-r{revision}", kind.code()),
            evidence_refs: vec![format!("review:{}:{revision}", kind.code())],
        }
    }

    fn manifest(revision: u64, predecessor: Option<String>) -> AssurancePolicyManifest {
        AssurancePolicyManifest {
            schema_version: "1".into(),
            manifest_id: "assurance:node-1".into(),
            revision,
            safety_contract_digest: format!("blake3:contract-r{revision}"),
            deployment_id: "node-1".into(),
            configuration_digest: format!("blake3:config-r{revision}"),
            model_manifest_digest: Some(format!("blake3:model-r{revision}")),
            calibration_manifest_digest: Some(format!("blake3:cal-r{revision}")),
            predecessor_manifest_digest: predecessor,
            change_ref: (revision > 1).then(|| format!("change:review-r{revision}")),
            policies: AssurancePolicyKind::REQUIRED
                .into_iter()
                .map(|kind| binding(kind, revision))
                .collect(),
            evidence_refs: vec![format!("review:manifest-r{revision}")],
        }
    }

    fn signed(manifest: AssurancePolicyManifest) -> SignedAssurancePolicyRevision {
        let digest = manifest.manifest_digest();
        SignedAssurancePolicyRevision {
            signature_receipt: ManifestSignatureVerificationReceipt {
                receipt_id: format!("sig:{}", manifest.revision),
                manifest_digest: digest,
                signer_ref: "signer:safety-board".into(),
                key_ref: "key:safety-board:v1".into(),
                signature_algorithm: "ed25519".into(),
                signature_ref: format!("signature:r{}", manifest.revision),
                verified_by_ref: "verifier:signature-service".into(),
                verification_ref: format!("verification:r{}", manifest.revision),
                verified_at_ms: manifest.revision * 1_000,
                evidence_refs: vec![format!("audit:sig-r{}", manifest.revision)],
            },
            manifest,
        }
    }

    fn valid_lineage() -> Vec<SignedAssurancePolicyRevision> {
        let r1 = manifest(1, None);
        let r2 = manifest(2, Some(r1.manifest_digest()));
        let r3 = manifest(3, Some(r2.manifest_digest()));
        vec![signed(r1), signed(r2), signed(r3)]
    }

    #[test]
    fn contiguous_signed_lineage_is_valid() {
        let lineage = valid_lineage();
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Valid);
        assert_eq!(report.first_revision, Some(1));
        assert_eq!(report.tip_revision, Some(3));
        assert!(report.contains_tip(&lineage[2].manifest));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn missing_middle_revision_fails_closed() {
        let mut lineage = valid_lineage();
        lineage.remove(1);
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AssurancePolicyLineageIssue::RevisionGap { previous: 1, next: 3 }
        )));
    }

    #[test]
    fn forged_predecessor_digest_is_rejected() {
        let mut lineage = valid_lineage();
        lineage[1].manifest.predecessor_manifest_digest = Some("blake3:forged".into());
        lineage[1].signature_receipt.manifest_digest = lineage[1].manifest.manifest_digest();
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AssurancePolicyLineageIssue::PredecessorDigestMismatch { revision: 2, .. }
        )));
    }

    #[test]
    fn signature_mismatch_invalidates_revision() {
        let mut lineage = valid_lineage();
        lineage[1].signature_receipt.manifest_digest = "blake3:wrong".into();
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AssurancePolicyLineageIssue::InvalidRevision(2)
        )));
    }

    #[test]
    fn manifest_identity_change_is_rejected() {
        let mut lineage = valid_lineage();
        lineage[2].manifest.manifest_id = "assurance:other".into();
        lineage[2].signature_receipt.manifest_digest = lineage[2].manifest.manifest_digest();
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AssurancePolicyLineageIssue::ManifestIdChanged { revision: 3 }
        )));
    }

    #[test]
    fn duplicate_revision_is_rejected() {
        let mut lineage = valid_lineage();
        lineage.push(lineage[1].clone());
        let report = assess_signed_policy_lineage(&lineage);
        assert_eq!(report.status, AssurancePolicyLineageStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AssurancePolicyLineageIssue::DuplicateRevision(2)
        )));
    }
}
