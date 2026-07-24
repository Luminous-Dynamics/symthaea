// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persisted publication-continuity bundle models.

use serde::{Deserialize, Serialize};

use crate::evidence_calibration::{
    CalibrationPublicationCatalogHeadBundle,
    CalibrationPublicationGossipConflictProof,
    CalibrationPublicationGossipLedger,
    CalibrationPublicationWitnessPolicyLedger,
};

pub const CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_VERSION: &str =
    "score-evidence-calibration-publication-continuity-bundle-v1";
pub const CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_AUDIT_VERSION: &str =
    "score-evidence-calibration-publication-continuity-bundle-audit-v1";

pub(crate) const CONTINUITY_BUNDLE_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-continuity-bundle.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationContinuityLimitation {
    ExternalVerifiersDefineAuthentication,
    WitnessIndependenceNotEstablished,
    GossipCoverageMayBePartial,
    ConflictAbsenceIsNotGlobal,
    PolicyHistoryIsLinear,
    RotationQuorumsMayOverlap,
    ExplicitLineageIsNotCompact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationContinuityBundle {
    pub bundle_version: String,
    pub head_bundle: CalibrationPublicationCatalogHeadBundle,
    pub witness_policy_ledger: CalibrationPublicationWitnessPolicyLedger,
    pub gossip_ledger: Option<CalibrationPublicationGossipLedger>,
    pub conflict_proofs: Vec<CalibrationPublicationGossipConflictProof>,
    pub limitations: Vec<CalibrationPublicationContinuityLimitation>,
    pub bundle_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationContinuityIssueCode {
    BundleVersionMismatch,
    HeadBundleInvalid,
    HeadWitnessAuthenticationFailed,
    PolicyLedgerInvalid,
    PolicyRotationAuthenticationFailed,
    CatalogIdentityMismatch,
    ActivePolicyMissing,
    ActivePolicyMismatch,
    GossipLedgerInvalid,
    GossipAuthenticationFailed,
    GossipIdentityMismatch,
    GossipHeadNotObserved,
    GossipPolicyEpochMissing,
    GossipPolicyEpochMismatch,
    ConflictProofInvalid,
    MissingConflictProof,
    UnexpectedConflictProof,
    DuplicateConflictProof,
    DuplicateLimitation,
    MissingLimitation,
    BundleSha256Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationContinuityIssue {
    pub code: CalibrationPublicationContinuityIssueCode,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationContinuityAuditReport {
    pub audit_version: String,
    pub structurally_valid: bool,
    pub head_authenticated: bool,
    pub policy_rotations_authenticated: bool,
    pub gossip_authenticated: bool,
    pub conflict_detected: bool,
    pub issues: Vec<CalibrationPublicationContinuityIssue>,
}

impl CalibrationPublicationContinuityAuditReport {
    pub fn valid(&self) -> bool {
        self.structurally_valid && self.issues.is_empty()
    }

    pub fn authenticated(&self) -> bool {
        self.valid()
            && self.head_authenticated
            && self.policy_rotations_authenticated
            && self.gossip_authenticated
    }

    pub fn accepted(&self) -> bool {
        self.authenticated() && !self.conflict_detected
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "details", rename_all = "snake_case")]
pub enum CalibrationPublicationContinuityError {
    InvalidBundle { issues: usize },
}

impl std::fmt::Display for CalibrationPublicationContinuityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBundle { issues } => {
                write!(formatter, "publication continuity audit failed with {issues} issues")
            }
        }
    }
}

impl std::error::Error for CalibrationPublicationContinuityError {}
