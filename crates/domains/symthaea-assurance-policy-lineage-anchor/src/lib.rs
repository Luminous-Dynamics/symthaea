// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Externally provisioned monotonic checkpoints for assurance-policy lineages.

#![deny(unsafe_code)]

use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_lineage::{
    AssurancePolicyLineageReport, AssurancePolicyLineageStatus,
};

const ANCHOR_DIGEST_SCHEMA: &[u8] = b"symthaea-policy-lineage-anchor-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyLineageAnchor {
    pub schema_version: String,
    pub anchor_id: String,
    /// Monotonic revision of the anchor/checkpoint object itself.
    pub anchor_revision: u64,
    pub manifest_id: String,
    pub deployment_id: String,
    /// Latest accepted assurance-policy manifest revision.
    pub tip_revision: u64,
    pub tip_manifest_digest: String,
    /// Exact previous anchor digest after anchor revision 1.
    pub predecessor_anchor_digest: Option<String>,
    /// Trusted recording time supplied by the external checkpoint store.
    pub recorded_at_ms: u64,
    /// Durable identity/reference for the external monotonic trust store.
    pub trust_store_ref: String,
    /// Independent verification / attestation reference for this checkpoint.
    pub independent_verification_ref: String,
    pub evidence_refs: Vec<String>,
}

impl PolicyLineageAnchor {
    pub fn validate(&self) -> bool {
        if self.schema_version.trim().is_empty()
            || self.anchor_id.trim().is_empty()
            || self.anchor_revision == 0
            || self.manifest_id.trim().is_empty()
            || self.deployment_id.trim().is_empty()
            || self.tip_revision == 0
            || !valid_digest(&self.tip_manifest_digest)
            || self.trust_store_ref.trim().is_empty()
            || self.independent_verification_ref.trim().is_empty()
            || self.evidence_refs.is_empty()
            || self.evidence_refs.iter().any(|value| value.trim().is_empty())
        {
            return false;
        }
        match self.anchor_revision {
            1 => self.predecessor_anchor_digest.is_none(),
            _ => self
                .predecessor_anchor_digest
                .as_ref()
                .is_some_and(|value| valid_digest(value)),
        }
    }

    pub fn anchor_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(ANCHOR_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.anchor_id);
        push_field(&mut hasher, &self.anchor_revision.to_string());
        push_field(&mut hasher, &self.manifest_id);
        push_field(&mut hasher, &self.deployment_id);
        push_field(&mut hasher, &self.tip_revision.to_string());
        push_field(&mut hasher, &self.tip_manifest_digest);
        push_optional(&mut hasher, self.predecessor_anchor_digest.as_deref());
        push_field(&mut hasher, &self.recorded_at_ms.to_string());
        push_field(&mut hasher, &self.trust_store_ref);
        push_field(&mut hasher, &self.independent_verification_ref);
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub fn matches_lineage_tip(&self, lineage: &AssurancePolicyLineageReport) -> bool {
        lineage.status == AssurancePolicyLineageStatus::Valid
            && lineage.manifest_id.as_deref() == Some(self.manifest_id.as_str())
            && lineage.deployment_id.as_deref() == Some(self.deployment_id.as_str())
            && lineage.tip_revision == Some(self.tip_revision)
            && lineage.tip_manifest_digest.as_deref() == Some(self.tip_manifest_digest.as_str())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyLineageAnchorStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyLineageAnchorIssue {
    EmptyAnchorChain,
    InvalidAnchor(u64),
    DuplicateAnchorRevision(u64),
    DuplicateAnchorDigest(String),
    FirstAnchorRevisionIsNotOne(u64),
    AnchorRevisionGap {
        previous: u64,
        next: u64,
    },
    AnchorIdChanged {
        anchor_revision: u64,
    },
    ManifestIdChanged {
        anchor_revision: u64,
    },
    DeploymentIdChanged {
        anchor_revision: u64,
    },
    TrustStoreChanged {
        anchor_revision: u64,
    },
    PredecessorAnchorDigestMismatch {
        anchor_revision: u64,
        expected: String,
        observed: Option<String>,
    },
    AnchoredTipRevisionDecreased {
        previous_tip_revision: u64,
        next_tip_revision: u64,
    },
    SameTipRevisionDigestChanged {
        tip_revision: u64,
    },
    AnchorRecordedTimeRegressed {
        previous_recorded_at_ms: u64,
        next_recorded_at_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyLineageAnchorReport {
    pub status: PolicyLineageAnchorStatus,
    pub anchor_id: Option<String>,
    pub manifest_id: Option<String>,
    pub deployment_id: Option<String>,
    pub trust_store_ref: Option<String>,
    pub first_anchor_revision: Option<u64>,
    pub tip_anchor_revision: Option<u64>,
    pub tip_anchor_digest: Option<String>,
    pub anchored_policy_tip_revision: Option<u64>,
    pub anchored_policy_tip_digest: Option<String>,
    pub anchor_count: usize,
    pub issues: Vec<PolicyLineageAnchorIssue>,
}

impl PolicyLineageAnchorReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    pub fn contains_tip_anchor(&self, anchor: &PolicyLineageAnchor) -> bool {
        let digest = anchor.anchor_digest();
        self.status == PolicyLineageAnchorStatus::Valid
            && self.tip_anchor_revision == Some(anchor.anchor_revision)
            && self.tip_anchor_digest.as_deref() == Some(digest.as_str())
    }
}

/// Validate a complete monotonic chain of externally recorded policy-lineage
/// checkpoints.
pub fn assess_policy_lineage_anchor_chain(
    anchors: &[PolicyLineageAnchor],
) -> PolicyLineageAnchorReport {
    if anchors.is_empty() {
        return PolicyLineageAnchorReport {
            status: PolicyLineageAnchorStatus::Invalid,
            anchor_id: None,
            manifest_id: None,
            deployment_id: None,
            trust_store_ref: None,
            first_anchor_revision: None,
            tip_anchor_revision: None,
            tip_anchor_digest: None,
            anchored_policy_tip_revision: None,
            anchored_policy_tip_digest: None,
            anchor_count: 0,
            issues: vec![PolicyLineageAnchorIssue::EmptyAnchorChain],
        };
    }

    let mut ordered = anchors.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|anchor| anchor.anchor_revision);
    let first = ordered[0];
    let anchor_id = first.anchor_id.clone();
    let manifest_id = first.manifest_id.clone();
    let deployment_id = first.deployment_id.clone();
    let trust_store_ref = first.trust_store_ref.clone();
    let mut issues = Vec::new();
    let mut seen_revisions = BTreeSet::new();
    let mut seen_digests = HashSet::new();

    for anchor in &ordered {
        if !anchor.validate() {
            issues.push(PolicyLineageAnchorIssue::InvalidAnchor(anchor.anchor_revision));
        }
        if !seen_revisions.insert(anchor.anchor_revision) {
            issues.push(PolicyLineageAnchorIssue::DuplicateAnchorRevision(
                anchor.anchor_revision,
            ));
        }
        let digest = anchor.anchor_digest();
        if !seen_digests.insert(digest.clone()) {
            issues.push(PolicyLineageAnchorIssue::DuplicateAnchorDigest(digest));
        }
        if anchor.anchor_id != anchor_id {
            issues.push(PolicyLineageAnchorIssue::AnchorIdChanged {
                anchor_revision: anchor.anchor_revision,
            });
        }
        if anchor.manifest_id != manifest_id {
            issues.push(PolicyLineageAnchorIssue::ManifestIdChanged {
                anchor_revision: anchor.anchor_revision,
            });
        }
        if anchor.deployment_id != deployment_id {
            issues.push(PolicyLineageAnchorIssue::DeploymentIdChanged {
                anchor_revision: anchor.anchor_revision,
            });
        }
        if anchor.trust_store_ref != trust_store_ref {
            issues.push(PolicyLineageAnchorIssue::TrustStoreChanged {
                anchor_revision: anchor.anchor_revision,
            });
        }
    }

    if first.anchor_revision != 1 {
        issues.push(PolicyLineageAnchorIssue::FirstAnchorRevisionIsNotOne(
            first.anchor_revision,
        ));
    }

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.anchor_revision != previous.anchor_revision.saturating_add(1) {
            issues.push(PolicyLineageAnchorIssue::AnchorRevisionGap {
                previous: previous.anchor_revision,
                next: current.anchor_revision,
            });
        }
        let expected = previous.anchor_digest();
        if current.predecessor_anchor_digest.as_deref() != Some(expected.as_str()) {
            issues.push(PolicyLineageAnchorIssue::PredecessorAnchorDigestMismatch {
                anchor_revision: current.anchor_revision,
                expected,
                observed: current.predecessor_anchor_digest.clone(),
            });
        }
        if current.tip_revision < previous.tip_revision {
            issues.push(PolicyLineageAnchorIssue::AnchoredTipRevisionDecreased {
                previous_tip_revision: previous.tip_revision,
                next_tip_revision: current.tip_revision,
            });
        }
        if current.tip_revision == previous.tip_revision
            && current.tip_manifest_digest != previous.tip_manifest_digest
        {
            issues.push(PolicyLineageAnchorIssue::SameTipRevisionDigestChanged {
                tip_revision: current.tip_revision,
            });
        }
        if current.recorded_at_ms < previous.recorded_at_ms {
            issues.push(PolicyLineageAnchorIssue::AnchorRecordedTimeRegressed {
                previous_recorded_at_ms: previous.recorded_at_ms,
                next_recorded_at_ms: current.recorded_at_ms,
            });
        }
    }

    let tip = ordered.last().expect("non-empty anchor chain");
    let status = if issues.is_empty() {
        PolicyLineageAnchorStatus::Valid
    } else {
        PolicyLineageAnchorStatus::Invalid
    };
    PolicyLineageAnchorReport {
        status,
        anchor_id: Some(anchor_id),
        manifest_id: Some(manifest_id),
        deployment_id: Some(deployment_id),
        trust_store_ref: Some(trust_store_ref),
        first_anchor_revision: Some(first.anchor_revision),
        tip_anchor_revision: Some(tip.anchor_revision),
        tip_anchor_digest: Some(tip.anchor_digest()),
        anchored_policy_tip_revision: Some(tip.tip_revision),
        anchored_policy_tip_digest: Some(tip.tip_manifest_digest.clone()),
        anchor_count: ordered.len(),
        issues,
    }
}

fn valid_digest(value: &str) -> bool {
    value
        .trim()
        .split_once(':')
        .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(
        anchor_revision: u64,
        tip_revision: u64,
        tip_digest: &str,
        predecessor: Option<String>,
        recorded_at_ms: u64,
    ) -> PolicyLineageAnchor {
        PolicyLineageAnchor {
            schema_version: "1".into(),
            anchor_id: "anchor:node-1".into(),
            anchor_revision,
            manifest_id: "assurance:node-1".into(),
            deployment_id: "node-1".into(),
            tip_revision,
            tip_manifest_digest: tip_digest.into(),
            predecessor_anchor_digest: predecessor,
            recorded_at_ms,
            trust_store_ref: "trust-store:tpm-nv:index-1".into(),
            independent_verification_ref: format!("verification:anchor:{anchor_revision}"),
            evidence_refs: vec![format!("audit:anchor:{anchor_revision}")],
        }
    }

    fn valid_chain() -> Vec<PolicyLineageAnchor> {
        let a1 = anchor(1, 1, "blake3:manifest-r1", None, 100);
        let a2 = anchor(
            2,
            2,
            "blake3:manifest-r2",
            Some(a1.anchor_digest()),
            200,
        );
        let a3 = anchor(
            3,
            3,
            "blake3:manifest-r3",
            Some(a2.anchor_digest()),
            300,
        );
        vec![a1, a2, a3]
    }

    #[test]
    fn monotonic_anchor_chain_is_valid() {
        let chain = valid_chain();
        let report = assess_policy_lineage_anchor_chain(&chain);
        assert_eq!(report.status, PolicyLineageAnchorStatus::Valid);
        assert_eq!(report.tip_anchor_revision, Some(3));
        assert_eq!(report.anchored_policy_tip_revision, Some(3));
        assert!(report.contains_tip_anchor(&chain[2]));
        assert!(!report.grants_physical_authority());
        assert!(!chain[2].grants_physical_authority());
    }

    #[test]
    fn truncated_anchor_history_is_invalid() {
        let mut chain = valid_chain();
        chain.remove(1);
        let report = assess_policy_lineage_anchor_chain(&chain);
        assert_eq!(report.status, PolicyLineageAnchorStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageAnchorIssue::AnchorRevisionGap { previous: 1, next: 3 }
        )));
    }

    #[test]
    fn anchored_policy_revision_cannot_decrease() {
        let mut chain = valid_chain();
        chain[2].tip_revision = 1;
        chain[2].tip_manifest_digest = "blake3:manifest-r1".into();
        chain[2].predecessor_anchor_digest = Some(chain[1].anchor_digest());
        let report = assess_policy_lineage_anchor_chain(&chain);
        assert_eq!(report.status, PolicyLineageAnchorStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageAnchorIssue::AnchoredTipRevisionDecreased { .. }
        )));
    }

    #[test]
    fn same_revision_cannot_change_manifest_digest() {
        let mut chain = valid_chain();
        chain[2].tip_revision = 2;
        chain[2].tip_manifest_digest = "blake3:forged-r2".into();
        chain[2].predecessor_anchor_digest = Some(chain[1].anchor_digest());
        let report = assess_policy_lineage_anchor_chain(&chain);
        assert_eq!(report.status, PolicyLineageAnchorStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageAnchorIssue::SameTipRevisionDigestChanged { tip_revision: 2 }
        )));
    }

    #[test]
    fn anchor_recording_time_cannot_go_backwards() {
        let mut chain = valid_chain();
        chain[2].recorded_at_ms = 150;
        chain[2].predecessor_anchor_digest = Some(chain[1].anchor_digest());
        let report = assess_policy_lineage_anchor_chain(&chain);
        assert_eq!(report.status, PolicyLineageAnchorStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageAnchorIssue::AnchorRecordedTimeRegressed { .. }
        )));
    }

    #[test]
    fn anchor_digest_changes_with_tip() {
        let a = anchor(1, 1, "blake3:manifest-r1", None, 100);
        let mut b = a.clone();
        b.tip_manifest_digest = "blake3:other".into();
        assert_ne!(a.anchor_digest(), b.anchor_digest());
    }
}
