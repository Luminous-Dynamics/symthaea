// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trust-root-backed deep readiness for assurance policy governance.

#![deny(unsafe_code)]

use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
use symthaea_assurance_policy_lineage_anchor::PolicyLineageAnchor;
use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
};
use symthaea_assurance_signing_authority::SigningAuthorityGovernance;
use symthaea_assurance_trust_store::{TrustStoreCheckpoint, TrustStoreProfile};
use symthaea_assurance_trust_store_attestation::CheckpointAttestationVerificationReceipt;
use symthaea_assurance_trust_store_current_state::{
    CurrentTrustStoreReport, CurrentTrustStoreStatus, assess_current_trust_store_state,
};
use symthaea_assurance_trust_store_recovery_ledger::RecoveryAcceptanceRecord;
use symthaea_evidence_anchored_governed_readiness::{
    AnchoredGovernedReadinessReport, assess_anchored_governed_readiness,
};
use symthaea_evidence_atomic_coverage::{AtomicCoveragePolicy, FacetEvidenceBinding};
use symthaea_evidence_deployment_scope::DeploymentEvidenceContext;
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineResolution,
};
use symthaea_evidence_time_assurance::TrustedEvidenceTime;
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierFaultDomainProfile,
};
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRootReadinessIssue {
    LowerReadinessInvalid,
    CurrentTrustStoreInvalid,
    MissingCurrentAnchor,
    MissingCurrentCheckpoint,
    CheckpointDoesNotBindCurrentAnchor,
    InvalidCheckpointAttestationVerification,
    CheckpointRecordedAfterTrustedInterval {
        recorded_at_ms: u64,
        earliest_trusted_ms: u64,
    },
    CheckpointAttestationVerifiedAfterTrustedInterval {
        verified_at_ms: u64,
        earliest_trusted_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustRootReadinessReport {
    pub status: StrictSafetyCaseStatus,
    pub anchored_readiness: AnchoredGovernedReadinessReport,
    pub current_trust_store: CurrentTrustStoreReport,
    pub current_checkpoint_digest: Option<String>,
    pub attestation_verification_receipt_id: String,
    pub issues: Vec<TrustRootReadinessIssue>,
}

impl TrustRootReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Recompute anchored signer-governed readiness and require the current policy
/// anchor to be represented by the current independently verified trust-store
/// checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn assess_trust_root_readiness(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    current_manifest: &AssurancePolicyManifest,
    current_signature_receipt: &ManifestSignatureVerificationReceipt,
    signed_lineage: &[SignedAssurancePolicyRevision],
    policy_receipts: &[PolicyScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    quarantine_directives: &[EvidenceQuarantineDirective],
    quarantine_resolutions: &[EvidenceQuarantineResolution],
    trusted_time: &TrustedEvidenceTime,
    verifier_profiles: &[VerifierFaultDomainProfile],
    verifier_policy: &VerifierDiversityPolicy,
    atomic_policy: &AtomicCoveragePolicy,
    facet_bindings: &[FacetEvidenceBinding],
    signing_governance: &SigningAuthorityGovernance,
    expected_signing_governance_digest: &str,
    anchor_chain: &[PolicyLineageAnchor],
    expected_anchor_digest: &str,
    trust_store_profile: &TrustStoreProfile,
    current_checkpoint_segment: &[TrustStoreCheckpoint],
    recovery_ledger: &[RecoveryAcceptanceRecord],
    checkpoint_attestation_receipt: &CheckpointAttestationVerificationReceipt,
) -> TrustRootReadinessReport {
    let anchored_readiness = assess_anchored_governed_readiness(
        safety_case,
        context,
        current_manifest,
        current_signature_receipt,
        signed_lineage,
        policy_receipts,
        lifecycle_events,
        quarantine_directives,
        quarantine_resolutions,
        trusted_time,
        verifier_profiles,
        verifier_policy,
        atomic_policy,
        facet_bindings,
        signing_governance,
        expected_signing_governance_digest,
        anchor_chain,
        expected_anchor_digest,
    );
    let current_trust_store = assess_current_trust_store_state(
        trust_store_profile,
        current_checkpoint_segment,
        recovery_ledger,
    );

    let mut issues = Vec::new();
    if anchored_readiness.status == StrictSafetyCaseStatus::Invalid {
        issues.push(TrustRootReadinessIssue::LowerReadinessInvalid);
    }
    if current_trust_store.status != CurrentTrustStoreStatus::Valid {
        issues.push(TrustRootReadinessIssue::CurrentTrustStoreInvalid);
    }

    let current_anchor = anchor_chain.iter().max_by_key(|anchor| anchor.anchor_revision);
    let current_checkpoint = current_checkpoint_segment
        .iter()
        .max_by_key(|checkpoint| checkpoint.store_revision);

    if current_anchor.is_none() {
        issues.push(TrustRootReadinessIssue::MissingCurrentAnchor);
    }
    if current_checkpoint.is_none() {
        issues.push(TrustRootReadinessIssue::MissingCurrentCheckpoint);
    }

    if let (Some(anchor), Some(checkpoint)) = (current_anchor, current_checkpoint) {
        if !checkpoint.binds_anchor(trust_store_profile, anchor) {
            issues.push(TrustRootReadinessIssue::CheckpointDoesNotBindCurrentAnchor);
        }
        if !checkpoint_attestation_receipt.validate_for(trust_store_profile, checkpoint) {
            issues.push(TrustRootReadinessIssue::InvalidCheckpointAttestationVerification);
        }
        if checkpoint.recorded_at_ms > trusted_time.earliest_ms() {
            issues.push(TrustRootReadinessIssue::CheckpointRecordedAfterTrustedInterval {
                recorded_at_ms: checkpoint.recorded_at_ms,
                earliest_trusted_ms: trusted_time.earliest_ms(),
            });
        }
        if checkpoint_attestation_receipt.verified_at_ms > trusted_time.earliest_ms() {
            issues.push(
                TrustRootReadinessIssue::CheckpointAttestationVerifiedAfterTrustedInterval {
                    verified_at_ms: checkpoint_attestation_receipt.verified_at_ms,
                    earliest_trusted_ms: trusted_time.earliest_ms(),
                },
            );
        }
    }

    let structural_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            TrustRootReadinessIssue::LowerReadinessInvalid
                | TrustRootReadinessIssue::CurrentTrustStoreInvalid
                | TrustRootReadinessIssue::MissingCurrentAnchor
                | TrustRootReadinessIssue::MissingCurrentCheckpoint
                | TrustRootReadinessIssue::CheckpointDoesNotBindCurrentAnchor
                | TrustRootReadinessIssue::InvalidCheckpointAttestationVerification
        )
    });
    let temporal_block = issues.iter().any(|issue| {
        matches!(
            issue,
            TrustRootReadinessIssue::CheckpointRecordedAfterTrustedInterval { .. }
                | TrustRootReadinessIssue::CheckpointAttestationVerifiedAfterTrustedInterval { .. }
        )
    });

    let status = if structural_invalid {
        StrictSafetyCaseStatus::Invalid
    } else if anchored_readiness.status != StrictSafetyCaseStatus::Ready || temporal_block {
        StrictSafetyCaseStatus::Blocked
    } else {
        StrictSafetyCaseStatus::Ready
    };

    TrustRootReadinessReport {
        status,
        anchored_readiness,
        current_trust_store,
        current_checkpoint_digest: current_checkpoint.map(TrustStoreCheckpoint::checkpoint_digest),
        attestation_verification_receipt_id: checkpoint_attestation_receipt.receipt_id.clone(),
        issues,
    }
}
