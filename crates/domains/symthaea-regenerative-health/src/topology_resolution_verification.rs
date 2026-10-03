// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verifier-result binding for topology lifecycle resolutions.
//!
//! The verifier result is kept separate from the authority statement. This
//! mirrors the RATS distinction between an asserted statement and a verifier
//! produced Attestation Result. The gate binds the result to the exact
//! resolution and checks its validity window, but does not establish the
//! verifier's trust anchor.

use serde::{Deserialize, Serialize};

use crate::topology_resolution::TopologyLifecycleResolution;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionVerificationState {
    Verified,
    InsufficientEvidence,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyResolutionVerificationIssue {
    EmptyIdentity,
    WrongVerifier,
    ResolutionMismatch,
    StatementMismatch,
    VerificationReferenceMismatch,
    InvalidVerificationWindow,
    FutureVerification,
    StaleVerification,
    MissingFreshness,
    WrongFreshnessScheme,
    WrongFreshnessSource,
    FreshnessRollback,
    EmptyFreshnessMarker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionFreshness {
    /// Transport-neutral freshness scheme identifier (for example a signed
    /// epoch marker, monotonic counter, or other protocol freshness handle).
    pub scheme: String,
    /// Authority/source that issued the freshness marker.
    pub source_id: String,
    /// Receiver-visible freshness epoch/counter.
    pub epoch: u64,
    /// Digest identifying the exact marker used.
    pub marker_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationResult {
    pub schema_version: String,
    pub verifier_id: String,
    pub resolution_id: String,
    pub authority_statement_digest: String,
    pub verification_reference: String,
    /// Fingerprint of the complete verifier report that produced this result.
    pub verification_report_digest: String,
    /// Fingerprint of the exact verifier policy inputs.
    pub policy_fingerprint: String,
    /// Fingerprint of the verifier execution environment identity.
    pub environment_fingerprint: String,
    pub freshness: Option<TopologyResolutionFreshness>,
    pub verified_at_ms: u64,
    pub valid_until_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_verifier_id: String,
    pub required_freshness_scheme: Option<String>,
    pub required_freshness_source_id: Option<String>,
    pub minimum_freshness_epoch: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyResolutionVerificationDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TopologyResolutionVerificationState,
    pub verifier_id: Option<String>,
    pub verification_reference: Option<String>,
    pub verification_report_digest: Option<String>,
    pub policy_fingerprint: Option<String>,
    pub environment_fingerprint: Option<String>,
    pub freshness_scheme: Option<String>,
    pub freshness_source_id: Option<String>,
    pub freshness_epoch: Option<u64>,
    pub freshness_marker_digest: Option<String>,
    pub issues: Vec<TopologyResolutionVerificationIssue>,
}

#[derive(Debug, Clone)]
pub struct TopologyResolutionVerificationGate {
    policy: TopologyResolutionVerificationPolicy,
}

impl TopologyResolutionVerificationGate {
    fn decision(
            schema_version: String,
            policy_id: String,
            verification: &TopologyResolutionVerificationResult,
            issues: Vec<TopologyResolutionVerificationIssue>,
        ) -> TopologyResolutionVerificationDecision {
            TopologyResolutionVerificationDecision {
                schema_version,
                policy_id,
                state: if issues.is_empty() {
                    TopologyResolutionVerificationState::Verified
                } else {
                    TopologyResolutionVerificationState::Quarantined
                },
                verifier_id: Some(verification.verifier_id.clone()),
                verification_reference: Some(verification.verification_reference.clone()),
                verification_report_digest: Some(verification.verification_report_digest.clone()),
                policy_fingerprint: Some(verification.policy_fingerprint.clone()),
                environment_fingerprint: Some(verification.environment_fingerprint.clone()),
                freshness_scheme: verification.freshness.as_ref().map(|item| item.scheme.clone()),
                freshness_source_id: verification.freshness.as_ref().map(|item| item.source_id.clone()),
                freshness_epoch: verification.freshness.as_ref().map(|item| item.epoch),
                freshness_marker_digest: verification
                    .freshness
                    .as_ref()
                    .map(|item| item.marker_digest.clone()),
                issues,
            }
        }

    pub fn new(
        policy: TopologyResolutionVerificationPolicy,
    ) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.expected_verifier_id.trim().is_empty()
        {
            return Err("invalid topology resolution verification policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        resolution: &TopologyLifecycleResolution,
        verification: Option<&TopologyResolutionVerificationResult>,
        now_ms: u64,
    ) -> TopologyResolutionVerificationDecision {
        let mut issues = Vec::new();
        let Some(verification) = verification else {
            return TopologyResolutionVerificationDecision {
                schema_version: self.policy.schema_version.clone(),
                policy_id: self.policy.policy_id.clone(),
                state: TopologyResolutionVerificationState::InsufficientEvidence,
                verifier_id: None,
                verification_reference: None,
                verification_report_digest: None,
                policy_fingerprint: None,
                environment_fingerprint: None,
                freshness_scheme: None,
                freshness_source_id: None,
                freshness_epoch: None,
                freshness_marker_digest: None,
                issues: vec![],
            };
        };

        if verification.schema_version.trim().is_empty()
            || verification.verifier_id.trim().is_empty()
            || verification.resolution_id.trim().is_empty()
            || verification.authority_statement_digest.trim().is_empty()
            || verification.verification_reference.trim().is_empty()
            || verification.verification_report_digest.trim().is_empty()
            || verification.policy_fingerprint.trim().is_empty()
            || verification.environment_fingerprint.trim().is_empty()
        {

            issues.push(TopologyResolutionVerificationIssue::EmptyIdentity);
        }

        if verification.verifier_id != self.policy.expected_verifier_id {
            issues.push(TopologyResolutionVerificationIssue::WrongVerifier);
        }
        if verification.resolution_id != resolution.resolution_id {
            issues.push(TopologyResolutionVerificationIssue::ResolutionMismatch);
        }
        if verification.authority_statement_digest != resolution.authority_statement_digest {
            issues.push(TopologyResolutionVerificationIssue::StatementMismatch);
        }
        if verification.verification_reference != resolution.verification_reference {
            issues.push(TopologyResolutionVerificationIssue::VerificationReferenceMismatch);
        }
        if verification.verified_at_ms == 0 || verification.verified_at_ms > verification.valid_until_ms {
            issues.push(TopologyResolutionVerificationIssue::InvalidVerificationWindow);
        } else if verification.verified_at_ms > now_ms {
            issues.push(TopologyResolutionVerificationIssue::FutureVerification);
        } else if now_ms > verification.valid_until_ms {
            issues.push(TopologyResolutionVerificationIssue::StaleVerification);
        }
        if let Some(expected_source) = self.policy.required_freshness_source_id.as_deref() {
            let Some(freshness) = verification.freshness.as_ref() else {
                issues.push(TopologyResolutionVerificationIssue::MissingFreshness);
                return Self::decision(self.policy.schema_version.clone(), self.policy.policy_id.clone(), verification, issues);
            };
            if freshness.scheme.trim().is_empty()
                || freshness.source_id.trim().is_empty()
                || freshness.marker_digest.trim().is_empty()
                || freshness.epoch == 0
            {
                issues.push(TopologyResolutionVerificationIssue::EmptyFreshnessMarker);
            }
            if let Some(expected_scheme) = self.policy.required_freshness_scheme.as_deref() {
                if freshness.scheme != expected_scheme {
                    issues.push(TopologyResolutionVerificationIssue::WrongFreshnessScheme);
                }
            }
            if freshness.source_id != expected_source {
                issues.push(TopologyResolutionVerificationIssue::WrongFreshnessSource);
            }
            if let Some(minimum_epoch) = self.policy.minimum_freshness_epoch {
                if freshness.epoch < minimum_epoch {
                    issues.push(TopologyResolutionVerificationIssue::FreshnessRollback);
                }
            }
        } else if let Some(freshness) = verification.freshness.as_ref() {
            if freshness.scheme.trim().is_empty()
                || freshness.source_id.trim().is_empty()
                || freshness.marker_digest.trim().is_empty()
                || freshness.epoch == 0
            {
                issues.push(TopologyResolutionVerificationIssue::EmptyFreshnessMarker);
            }
        }

        Self::decision(
            self.policy.schema_version.clone(),
            self.policy.policy_id.clone(),
            verification,
            issues,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_resolution::TopologyBranchReference;

    fn branch() -> TopologyBranchReference {
        TopologyBranchReference {
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            successor_epoch: 2,
            successor_topology_digest: "topology-v2a".into(),
            successor_effective_from_ms: 1_500,
        }
    }

    fn resolution() -> TopologyLifecycleResolution {
        TopologyLifecycleResolution {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            predecessor_epoch: 1,
            predecessor_topology_digest: "topology-v1".into(),
            selected_successor_epoch: 2,
            selected_successor_topology_digest: "topology-v2a".into(),
            selected_successor_effective_from_ms: 1_500,
            resolution_epoch: 1,
            predecessor_resolution_id: None,
            predecessor_resolution_digest: None,
            observed_successors: vec![branch()],
            observed_successors_digest: crate::topology_resolution::branch_set_digest(&[branch()]),
            resolution_id: "resolution-2".into(),
            authority_id: "mycelix-topology-authority".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            resolved_at_ms: 2_000,
            configuration_digest: "cfg-1".into(),
        }
    }

    fn result() -> TopologyResolutionVerificationResult {
        TopologyResolutionVerificationResult {
            schema_version: "0.1".into(),
            verifier_id: "mycelix-topology-verifier".into(),
            resolution_id: "resolution-2".into(),
            authority_statement_digest: "resolution-digest-2".into(),
            verification_reference: "resolution-verify-2".into(),
            verification_report_digest: "report-digest-2".into(),
            policy_fingerprint: "policy-fingerprint-2".into(),
            environment_fingerprint: "environment-fingerprint-2".into(),
            freshness: Some(TopologyResolutionFreshness {
                scheme: "epoch-marker-v1".into(),
                source_id: "topology-epoch-bell".into(),
                epoch: 7,
                marker_digest: "epoch-marker-7".into(),
            }),
            verified_at_ms: 2_100,
            valid_until_ms: 4_000,
        }
    }

    fn gate() -> TopologyResolutionVerificationGate {
        TopologyResolutionVerificationGate::new(
            TopologyResolutionVerificationPolicy {
                schema_version: "0.1".into(),
                policy_id: "topology-resolution-verification-v1".into(),
                expected_verifier_id: "mycelix-topology-verifier".into(),
                required_freshness_scheme: Some("epoch-marker-v1".into()),
                required_freshness_source_id: Some("topology-epoch-bell".into()),
                minimum_freshness_epoch: Some(7),
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_verifier_result_is_admitted() {
        let d = gate().assess(&resolution(), Some(&result()), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Verified);
    }

    #[test]
    #[test]
    fn missing_required_freshness_is_quarantined() {
        let mut r = result();
        r.freshness = None;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::MissingFreshness));
    }

    #[test]
    fn older_freshness_epoch_is_quarantined() {
        let mut r = result();
        r.freshness.as_mut().unwrap().epoch = 6;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::FreshnessRollback));
    }

    #[test]
    fn freshness_scheme_must_match_policy() {
        let mut r = result();
        r.freshness.as_mut().unwrap().scheme = "unexpected-scheme".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d
            .issues
            .contains(&TopologyResolutionVerificationIssue::WrongFreshnessScheme));
    }

    #[test]
    fn freshness_source_must_match_policy() {
        let mut r = result();
        r.freshness.as_mut().unwrap().source_id = "unexpected-bell".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::WrongFreshnessSource));
    }

    fn missing_verifier_result_is_insufficient_evidence() {
        let d = gate().assess(&resolution(), None, 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::InsufficientEvidence);
    }

    #[test]
    fn wrong_verifier_is_quarantined() {
        let mut r = result();
        r.verifier_id = "unexpected".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::WrongVerifier));
    }

    #[test]
    fn verification_must_bind_exact_resolution_identity() {
        let mut r = result();
        r.resolution_id = "resolution-old".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::ResolutionMismatch));
    }

    #[test]
    fn verification_must_bind_exact_statement_digest() {
        let mut r = result();
        r.authority_statement_digest = "wrong-digest".into();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::StatementMismatch));
    }

    #[test]
    fn missing_report_provenance_is_quarantined() {
        let mut r = result();
        r.verification_report_digest.clear();
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::EmptyIdentity));
    }

    #[test]
    fn stale_verification_cannot_support_current_authority() {
        let mut r = result();
        r.valid_until_ms = 2_999;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::StaleVerification));
    }

    #[test]
    fn future_verification_cannot_support_current_authority() {
        let mut r = result();
        r.verified_at_ms = 3_001;
        let d = gate().assess(&resolution(), Some(&r), 3_000);
        assert_eq!(d.state, TopologyResolutionVerificationState::Quarantined);
        assert!(d.issues.contains(&TopologyResolutionVerificationIssue::FutureVerification));
    }
}
