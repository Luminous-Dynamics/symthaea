// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only eligibility receipt for later isolated support-store hydration review.
//!
//! EKM-064 can reproduce persisted support mutations through the real EKM-026
//! firewall while intentionally using private ID-cursor placeholders for revision
//! receipts that never sourced mutations. This module binds that replay result to
//! the exact EKM-053 read-only quarantine and EKM-054 protected trust-context
//! checkpoint without constructing writable restart state.
//!
//! Support-store reproducibility and complete revision-history reproducibility are
//! reported separately. A successful support replay must not silently imply that
//! rejected/unapplied revision receipts were historically reconstructed.

use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use super::epistemic_restart_historical_firewall_replay::{
    HistoricalFirewallReplayError, HistoricalFirewallReplayReportV1,
};
use super::epistemic_restart_historical_projection::HistoricalEvidenceProjectionV1;
use super::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use super::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use super::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use super::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use super::epistemic_restart_quarantine_facade::{
    ReadOnlyEpistemicRestartQuarantineV2, RestartQuarantineFacadeError,
};
use super::epistemic_restart_trust_checkpoint::{
    RestartTrustContextCheckpointError, VerifiedRestartTrustContextCheckpointV1,
};
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportHydrationEligibilityVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SupportHydrationEligibilityDigestV1([u8; 32]);

impl SupportHydrationEligibilityDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// Immutable composition receipt only. No writable support store or revision
/// history is constructed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportHydrationEligibilityReceiptV1 {
    version: SupportHydrationEligibilityVersion,
    restart_capture_cycle: u64,
    reviewed_at_cycle: u64,
    restart_outer_checksum: [u8; 32],
    restart_v2_digest: [u8; 32],
    quarantine_digest: [u8; 32],
    trust_context_digest: [u8; 32],
    trust_checkpoint_digest: [u8; 32],
    replay_report_digest: [u8; 32],
    mutation_count: usize,
    non_mutation_revision_placeholder_count: usize,
    historical_support_mutations_reproduced: bool,
    final_support_state_equivalent: bool,
    consumed_authorization_count_equivalent: bool,
    quarantine_state_equivalent: bool,
    trust_context_checkpoint_verified: bool,
    support_store_hydration_review_eligible: bool,
    full_revision_history_hydration_review_eligible: bool,
    complete_epistemic_hydration_review_eligible: bool,
    immutable_revision_restore_path_required: bool,
    writable_hydration_authorized: bool,
    writable_state_export_authorized: bool,
    activation_authorized: bool,
    receipt_digest: SupportHydrationEligibilityDigestV1,
}

impl SupportHydrationEligibilityReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        reviewed_at_cycle: u64,
    ) -> Result<Self, SupportHydrationEligibilityError> {
        replay_report
            .verify_against(
                restart,
                seals,
                restart_receipt,
                mutation_checkpoint,
                admission,
                currentness,
                eligibility,
                projection,
                replay_report.replayed_at_cycle(),
            )
            .map_err(SupportHydrationEligibilityError::ReplayRejected)?;
        quarantine
            .verify()
            .map_err(SupportHydrationEligibilityError::QuarantineRejected)?;
        trust_checkpoint
            .verify_internal()
            .map_err(SupportHydrationEligibilityError::TrustCheckpointRejected)?;

        if reviewed_at_cycle < replay_report.replayed_at_cycle() {
            return Err(SupportHydrationEligibilityError::ReviewPredatesReplay {
                reviewed_at_cycle,
                replayed_at_cycle: replay_report.replayed_at_cycle(),
            });
        }
        if reviewed_at_cycle < trust_checkpoint.verified_at_cycle() {
            return Err(SupportHydrationEligibilityError::ReviewPredatesTrustCheckpoint {
                reviewed_at_cycle,
                checkpoint_verified_at_cycle: trust_checkpoint.verified_at_cycle(),
            });
        }
        if reviewed_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(SupportHydrationEligibilityError::CurrentnessExpired {
                reviewed_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }
        if reviewed_at_cycle >= trust_checkpoint.statement().expires_at_cycle() {
            return Err(SupportHydrationEligibilityError::TrustCheckpointExpired {
                reviewed_at_cycle,
                expires_at_cycle: trust_checkpoint.statement().expires_at_cycle(),
            });
        }

        if restart.outer_checksum != quarantine.source_outer_checksum()
            || restart.claimed_v2_digest != quarantine.source_v2_digest()
            || restart.base.captured_at_cycle != quarantine.captured_at_cycle()
        {
            return Err(SupportHydrationEligibilityError::RestartQuarantineMismatch);
        }
        if trust_checkpoint.statement().context_digest() != quarantine.trust_context_digest() {
            return Err(SupportHydrationEligibilityError::TrustContextMismatch);
        }
        if quarantine.writable_hydration_authorized()
            || quarantine.activation_authorized()
            || trust_checkpoint.trusted_state_mutated()
            || trust_checkpoint.quarantine_construction_authorized()
            || trust_checkpoint.writable_hydration_authorized()
            || trust_checkpoint.activation_authorized()
        {
            return Err(SupportHydrationEligibilityError::UnexpectedTrustAuthority);
        }
        if !replay_report.persisted_mutation_receipts_exactly_reproduced()
            || !replay_report.final_support_state_equivalent()
            || !replay_report.consumed_authorization_count_equivalent()
            || replay_report.writable_state_export_authorized()
            || replay_report.writable_hydration_authorized()
            || replay_report.activation_authorized()
        {
            return Err(SupportHydrationEligibilityError::UnexpectedReplayClaims);
        }

        let placeholder_count = replay_report.non_mutation_revision_placeholder_count();
        let full_revision_history_hydration_review_eligible = placeholder_count == 0;
        let immutable_revision_restore_path_required = placeholder_count != 0;

        let mut receipt = Self {
            version: SupportHydrationEligibilityVersion::V1,
            restart_capture_cycle: restart.base.captured_at_cycle,
            reviewed_at_cycle,
            restart_outer_checksum: restart.outer_checksum,
            restart_v2_digest: restart.claimed_v2_digest,
            quarantine_digest: quarantine.quarantine_digest().as_bytes(),
            trust_context_digest: quarantine.trust_context_digest().as_bytes(),
            trust_checkpoint_digest: trust_checkpoint.statement_digest().as_bytes(),
            replay_report_digest: replay_report.report_digest().as_bytes(),
            mutation_count: replay_report.mutation_count(),
            non_mutation_revision_placeholder_count: placeholder_count,
            historical_support_mutations_reproduced: true,
            final_support_state_equivalent: true,
            consumed_authorization_count_equivalent: true,
            quarantine_state_equivalent: true,
            trust_context_checkpoint_verified: true,
            support_store_hydration_review_eligible: true,
            full_revision_history_hydration_review_eligible,
            complete_epistemic_hydration_review_eligible:
                full_revision_history_hydration_review_eligible,
            immutable_revision_restore_path_required,
            writable_hydration_authorized: false,
            writable_state_export_authorized: false,
            activation_authorized: false,
            receipt_digest: SupportHydrationEligibilityDigestV1([0; 32]),
        };
        receipt.receipt_digest = digest_receipt(&receipt)?;
        Ok(receipt)
    }

    pub fn version(&self) -> SupportHydrationEligibilityVersion {
        self.version
    }

    pub fn reviewed_at_cycle(&self) -> u64 {
        self.reviewed_at_cycle
    }

    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    pub fn non_mutation_revision_placeholder_count(&self) -> usize {
        self.non_mutation_revision_placeholder_count
    }

    pub fn historical_support_mutations_reproduced(&self) -> bool {
        self.historical_support_mutations_reproduced
    }

    pub fn support_store_hydration_review_eligible(&self) -> bool {
        self.support_store_hydration_review_eligible
    }

    pub fn full_revision_history_hydration_review_eligible(&self) -> bool {
        self.full_revision_history_hydration_review_eligible
    }

    pub fn complete_epistemic_hydration_review_eligible(&self) -> bool {
        self.complete_epistemic_hydration_review_eligible
    }

    pub fn immutable_revision_restore_path_required(&self) -> bool {
        self.immutable_revision_restore_path_required
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn writable_state_export_authorized(&self) -> bool {
        self.writable_state_export_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn receipt_digest(&self) -> SupportHydrationEligibilityDigestV1 {
        self.receipt_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        mutation_checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replay_report: &HistoricalFirewallReplayReportV1,
        quarantine: &ReadOnlyEpistemicRestartQuarantineV2,
        trust_checkpoint: &VerifiedRestartTrustContextCheckpointV1,
        reviewed_at_cycle: u64,
    ) -> Result<(), SupportHydrationEligibilityError> {
        let live = Self::evaluate(
            restart,
            seals,
            restart_receipt,
            mutation_checkpoint,
            admission,
            currentness,
            eligibility,
            projection,
            replay_report,
            quarantine,
            trust_checkpoint,
            reviewed_at_cycle,
        )?;
        if &live != self {
            return Err(SupportHydrationEligibilityError::ReceiptMismatch);
        }
        if digest_receipt(self)? != self.receipt_digest {
            return Err(SupportHydrationEligibilityError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

fn digest_receipt(
    receipt: &SupportHydrationEligibilityReceiptV1,
) -> Result<SupportHydrationEligibilityDigestV1, SupportHydrationEligibilityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-support-hydration-eligibility-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.restart_capture_cycle.to_le_bytes());
    hasher.update(&receipt.reviewed_at_cycle.to_le_bytes());
    hasher.update(&receipt.restart_outer_checksum);
    hasher.update(&receipt.restart_v2_digest);
    hasher.update(&receipt.quarantine_digest);
    hasher.update(&receipt.trust_context_digest);
    hasher.update(&receipt.trust_checkpoint_digest);
    hasher.update(&receipt.replay_report_digest);
    let mutation_count = u64::try_from(receipt.mutation_count)
        .map_err(|_| SupportHydrationEligibilityError::LengthOverflow)?;
    let placeholder_count = u64::try_from(receipt.non_mutation_revision_placeholder_count)
        .map_err(|_| SupportHydrationEligibilityError::LengthOverflow)?;
    hasher.update(&mutation_count.to_le_bytes());
    hasher.update(&placeholder_count.to_le_bytes());
    hasher.update(&[u8::from(receipt.historical_support_mutations_reproduced)]);
    hasher.update(&[u8::from(receipt.final_support_state_equivalent)]);
    hasher.update(&[u8::from(
        receipt.consumed_authorization_count_equivalent,
    )]);
    hasher.update(&[u8::from(receipt.quarantine_state_equivalent)]);
    hasher.update(&[u8::from(receipt.trust_context_checkpoint_verified)]);
    hasher.update(&[u8::from(
        receipt.support_store_hydration_review_eligible,
    )]);
    hasher.update(&[u8::from(
        receipt.full_revision_history_hydration_review_eligible,
    )]);
    hasher.update(&[u8::from(
        receipt.complete_epistemic_hydration_review_eligible,
    )]);
    hasher.update(&[u8::from(
        receipt.immutable_revision_restore_path_required,
    )]);
    hasher.update(&[u8::from(receipt.writable_hydration_authorized)]);
    hasher.update(&[u8::from(receipt.writable_state_export_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    Ok(SupportHydrationEligibilityDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug)]
pub enum SupportHydrationEligibilityError {
    ReplayRejected(HistoricalFirewallReplayError),
    QuarantineRejected(RestartQuarantineFacadeError),
    TrustCheckpointRejected(RestartTrustContextCheckpointError),
    ReviewPredatesReplay {
        reviewed_at_cycle: u64,
        replayed_at_cycle: u64,
    },
    ReviewPredatesTrustCheckpoint {
        reviewed_at_cycle: u64,
        checkpoint_verified_at_cycle: u64,
    },
    CurrentnessExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    TrustCheckpointExpired {
        reviewed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    RestartQuarantineMismatch,
    TrustContextMismatch,
    UnexpectedTrustAuthority,
    UnexpectedReplayClaims,
    ReceiptMismatch,
    ReceiptDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for SupportHydrationEligibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "support hydration eligibility rejected: {self:?}")
    }
}

impl Error for SupportHydrationEligibilityError {}
