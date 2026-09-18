// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable restoration of the complete EKM-025 revision audit record set.
//!
//! EKM-064 historically re-executes only revision receipts that actually sourced
//! persisted belief mutations. Rejected and unapplied receipts are represented by
//! private ID-cursor placeholders so mutation IDs can be reproduced without
//! inventing historical ledger chronology. EKM-065 therefore keeps complete
//! epistemic hydration review ineligible when placeholders exist.
//!
//! This module restores the *typed immutable audit semantics* of every persisted
//! revision receipt directly from the already validated EKM-034/EKM-040 restart
//! wires. It does not create `BeliefRevisionHistory`, does not re-execute rejected
//! decisions, and grants no mutation or hydration authority.

use super::belief_mutation_firewall::BeliefMutationReceiptId;
use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use super::belief_revision_receipt::BeliefRevisionReceiptId;
use super::belief_revision_snapshot::{
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionPolicySchemaV1,
};
use super::claim_evidence::{ClaimId, EvidenceId};
use super::epistemic_restart_historical_firewall_replay::HistoricalFirewallReplayReportV1;
use super::epistemic_restart_historical_projection::HistoricalEvidenceProjectionV1;
use super::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use super::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use super::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use super::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use super::epistemic_restart_quarantine_facade::ReadOnlyEpistemicRestartQuarantineV2;
use super::epistemic_restart_support_hydration_eligibility::{
    SupportHydrationEligibilityError, SupportHydrationEligibilityReceiptV1,
};
use super::epistemic_restart_trust_checkpoint::VerifiedRestartTrustContextCheckpointV1;
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire::{WireRevisionBasisV1, WireUncertaintyAssessmentV1};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

const MAX_REVISION_AUDIT_RECORDS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmutableRevisionAuditRestorationVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImmutableRevisionAuditRestorationDigestV1([u8; 32]);

impl ImmutableRevisionAuditRestorationDigestV1 {
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

/// Typed immutable audit view of one EKM-025 receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct ImmutableRevisionAuditRecordV1 {
    receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    rationale: String,
    basis: Vec<WireRevisionBasisV1>,
    duplicate_basis_evidence_ids: Vec<EvidenceId>,
    policy_schema: BeliefRevisionPolicySchemaV1,
    calibration: Option<(u64, f64)>,
    uncertainty: Option<WireUncertaintyAssessmentV1>,
    decision_snapshot: BeliefRevisionDecisionSnapshotV1,
    evaluated_at_cycle: u64,
    mutation_id: Option<BeliefMutationReceiptId>,
    historically_reexecuted: bool,
    restored_from_typed_persistence: bool,
    mutation_authority: bool,
}

impl ImmutableRevisionAuditRecordV1 {
    pub fn receipt_id(&self) -> BeliefRevisionReceiptId {
        self.receipt_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn proposed_delta(&self) -> f32 {
        self.proposed_delta
    }

    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    pub fn basis(&self) -> &[WireRevisionBasisV1] {
        &self.basis
    }

    pub fn duplicate_basis_evidence_ids(&self) -> &[EvidenceId] {
        &self.duplicate_basis_evidence_ids
    }

    pub fn policy_schema(&self) -> &BeliefRevisionPolicySchemaV1 {
        &self.policy_schema
    }

    pub fn calibration(&self) -> Option<(u64, f64)> {
        self.calibration
    }

    pub fn uncertainty(&self) -> Option<&WireUncertaintyAssessmentV1> {
        self.uncertainty.as_ref()
    }

    pub fn decision_snapshot(&self) -> &BeliefRevisionDecisionSnapshotV1 {
        &self.decision_snapshot
    }

    pub fn evaluated_at_cycle(&self) -> u64 {
        self.evaluated_at_cycle
    }

    pub fn mutation_id(&self) -> Option<BeliefMutationReceiptId> {
        self.mutation_id
    }

    pub fn historically_reexecuted(&self) -> bool {
        self.historically_reexecuted
    }

    pub fn restored_from_typed_persistence(&self) -> bool {
        self.restored_from_typed_persistence
    }

    pub fn mutation_authority(&self) -> bool {
        self.mutation_authority
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImmutableRevisionAuditRestorationV1 {
    version: ImmutableRevisionAuditRestorationVersion,
    restart_capture_cycle: u64,
    restored_at_cycle: u64,
    source_outer_checksum: [u8; 32],
    source_base_wire_checksum: [u8; 32],
    source_schema_wire_checksum: [u8; 32],
    support_hydration_eligibility_digest: [u8; 32],
    records: Vec<ImmutableRevisionAuditRecordV1>,
    eligible_count: usize,
    rejected_count: usize,
    mutation_source_count: usize,
    historically_reexecuted_count: usize,
    restored_without_reexecution_count: usize,
    complete_immutable_revision_audit_restored: bool,
    operational_revision_history_constructed: bool,
    nonmutation_decisions_historically_reexecuted: bool,
    mutation_authority: bool,
    writable_hydration_authorized: bool,
    writable_state_export_authorized: bool,
    activation_authorized: bool,
    restoration_digest: ImmutableRevisionAuditRestorationDigestV1,
}

impl ImmutableRevisionAuditRestorationV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
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
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        restored_at_cycle: u64,
    ) -> Result<Self, ImmutableRevisionAuditRestorationError> {
        support_eligibility
            .verify_against(
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
                support_eligibility.reviewed_at_cycle(),
            )
            .map_err(ImmutableRevisionAuditRestorationError::SupportEligibilityRejected)?;

        if restored_at_cycle < support_eligibility.reviewed_at_cycle() {
            return Err(ImmutableRevisionAuditRestorationError::RestorationPredatesReview {
                restored_at_cycle,
                reviewed_at_cycle: support_eligibility.reviewed_at_cycle(),
            });
        }
        if restored_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(ImmutableRevisionAuditRestorationError::CurrentnessExpired {
                restored_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }
        if support_eligibility.writable_hydration_authorized()
            || support_eligibility.writable_state_export_authorized()
            || support_eligibility.activation_authorized()
        {
            return Err(ImmutableRevisionAuditRestorationError::UnexpectedEligibilityAuthority);
        }
        if restart.base.revisions.len() > MAX_REVISION_AUDIT_RECORDS {
            return Err(ImmutableRevisionAuditRestorationError::TooManyRevisionRecords {
                actual: restart.base.revisions.len(),
                maximum: MAX_REVISION_AUDIT_RECORDS,
            });
        }
        if restart.base.revisions.len() != restart.revision_schemas.records.len() {
            return Err(ImmutableRevisionAuditRestorationError::RevisionSchemaCountMismatch);
        }

        let replayed_revision_ids = replay_report
            .mutation_replays()
            .iter()
            .map(|record| record.source_revision_receipt_id())
            .collect::<HashSet<_>>();
        if replayed_revision_ids.len() != replay_report.mutation_replays().len() {
            return Err(ImmutableRevisionAuditRestorationError::DuplicateReplayedSourceRevision);
        }
        let mutation_by_revision = restart
            .base
            .mutations
            .iter()
            .map(|mutation| (mutation.source_revision_receipt_id.0, mutation.id))
            .collect::<HashMap<_, _>>();
        if mutation_by_revision.len() != restart.base.mutations.len() {
            return Err(ImmutableRevisionAuditRestorationError::DuplicateMutationSourceRevision);
        }
        if mutation_by_revision.len() != replayed_revision_ids.len()
            || mutation_by_revision.len() != replay_report.mutation_count()
        {
            return Err(ImmutableRevisionAuditRestorationError::ReplayMutationCountMismatch {
                persisted: mutation_by_revision.len(),
                replayed: replayed_revision_ids.len(),
                report: replay_report.mutation_count(),
            });
        }

        let mut records = Vec::with_capacity(restart.base.revisions.len());
        let mut eligible_count = 0usize;
        let mut rejected_count = 0usize;
        let mut mutation_source_count = 0usize;
        let mut historically_reexecuted_count = 0usize;

        for (wire, schema) in restart
            .base
            .revisions
            .iter()
            .zip(&restart.revision_schemas.records)
        {
            let declared_roots = usize::try_from(wire.declared_provenance_root_count)
                .map_err(|_| ImmutableRevisionAuditRestorationError::LengthOverflow)?;
            if wire.id != schema.receipt_id
                || wire.claim_id != schema.claim_id
                || wire.proposed_delta.to_bits() != schema.proposed_delta.to_bits()
                || wire.evaluated_at_cycle != schema.evaluated_at_cycle
                || wire.decision_eligible != schema.decision_snapshot.eligible
                || declared_roots != schema.decision_snapshot.declared_provenance_root_count
            {
                return Err(ImmutableRevisionAuditRestorationError::RevisionSchemaMismatch(
                    wire.id.0,
                ));
            }

            if schema.decision_snapshot.eligible {
                eligible_count = eligible_count
                    .checked_add(1)
                    .ok_or(ImmutableRevisionAuditRestorationError::LengthOverflow)?;
            } else {
                rejected_count = rejected_count
                    .checked_add(1)
                    .ok_or(ImmutableRevisionAuditRestorationError::LengthOverflow)?;
            }

            let mutation_id = mutation_by_revision.get(&wire.id.0).copied();
            let historically_reexecuted = replayed_revision_ids.contains(&wire.id.0);
            if mutation_id.is_some() {
                mutation_source_count = mutation_source_count
                    .checked_add(1)
                    .ok_or(ImmutableRevisionAuditRestorationError::LengthOverflow)?;
                if !historically_reexecuted {
                    return Err(
                        ImmutableRevisionAuditRestorationError::MutationSourceNotHistoricallyReexecuted(
                            wire.id.0,
                        ),
                    );
                }
            } else if historically_reexecuted {
                return Err(
                    ImmutableRevisionAuditRestorationError::UnexpectedHistoricalReexecution(
                        wire.id.0,
                    ),
                );
            }
            if historically_reexecuted {
                historically_reexecuted_count = historically_reexecuted_count
                    .checked_add(1)
                    .ok_or(ImmutableRevisionAuditRestorationError::LengthOverflow)?;
            }

            records.push(ImmutableRevisionAuditRecordV1 {
                receipt_id: wire.id,
                claim_id: wire.claim_id,
                proposed_delta: wire.proposed_delta,
                rationale: wire.rationale.clone(),
                basis: wire.basis.clone(),
                duplicate_basis_evidence_ids: wire.duplicate_basis_evidence_ids.clone(),
                policy_schema: schema.policy_schema.clone(),
                calibration: wire.calibration,
                uncertainty: wire.uncertainty.clone(),
                decision_snapshot: schema.decision_snapshot.clone(),
                evaluated_at_cycle: wire.evaluated_at_cycle,
                mutation_id,
                historically_reexecuted,
                restored_from_typed_persistence: true,
                mutation_authority: false,
            });
        }

        if mutation_source_count != mutation_by_revision.len()
            || historically_reexecuted_count != replayed_revision_ids.len()
        {
            return Err(ImmutableRevisionAuditRestorationError::ReplayMutationCountMismatch {
                persisted: mutation_source_count,
                replayed: historically_reexecuted_count,
                report: replay_report.mutation_count(),
            });
        }

        let restored_without_reexecution_count = records
            .len()
            .checked_sub(historically_reexecuted_count)
            .ok_or(ImmutableRevisionAuditRestorationError::LengthOverflow)?;
        let expected_placeholder_count = restored_without_reexecution_count;
        if expected_placeholder_count
            != support_eligibility.non_mutation_revision_placeholder_count()
        {
            return Err(ImmutableRevisionAuditRestorationError::PlaceholderCountMismatch {
                restored_without_reexecution: expected_placeholder_count,
                eligibility_placeholders: support_eligibility
                    .non_mutation_revision_placeholder_count(),
            });
        }

        let mut out = Self {
            version: ImmutableRevisionAuditRestorationVersion::V1,
            restart_capture_cycle: restart.base.captured_at_cycle,
            restored_at_cycle,
            source_outer_checksum: restart.outer_checksum,
            source_base_wire_checksum: restart.base.wire_checksum,
            source_schema_wire_checksum: restart.revision_schemas.wire_checksum,
            support_hydration_eligibility_digest: support_eligibility.receipt_digest().as_bytes(),
            records,
            eligible_count,
            rejected_count,
            mutation_source_count,
            historically_reexecuted_count,
            restored_without_reexecution_count,
            complete_immutable_revision_audit_restored: true,
            operational_revision_history_constructed: false,
            nonmutation_decisions_historically_reexecuted: false,
            mutation_authority: false,
            writable_hydration_authorized: false,
            writable_state_export_authorized: false,
            activation_authorized: false,
            restoration_digest: ImmutableRevisionAuditRestorationDigestV1([0; 32]),
        };
        out.restoration_digest = digest_restoration(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> ImmutableRevisionAuditRestorationVersion {
        self.version
    }

    pub fn restored_at_cycle(&self) -> u64 {
        self.restored_at_cycle
    }

    pub fn records(&self) -> &[ImmutableRevisionAuditRecordV1] {
        &self.records
    }

    pub fn eligible_count(&self) -> usize {
        self.eligible_count
    }

    pub fn rejected_count(&self) -> usize {
        self.rejected_count
    }

    pub fn mutation_source_count(&self) -> usize {
        self.mutation_source_count
    }

    pub fn historically_reexecuted_count(&self) -> usize {
        self.historically_reexecuted_count
    }

    pub fn restored_without_reexecution_count(&self) -> usize {
        self.restored_without_reexecution_count
    }

    pub fn complete_immutable_revision_audit_restored(&self) -> bool {
        self.complete_immutable_revision_audit_restored
    }

    pub fn operational_revision_history_constructed(&self) -> bool {
        self.operational_revision_history_constructed
    }

    pub fn nonmutation_decisions_historically_reexecuted(&self) -> bool {
        self.nonmutation_decisions_historically_reexecuted
    }

    pub fn mutation_authority(&self) -> bool {
        self.mutation_authority
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

    pub fn restoration_digest(&self) -> ImmutableRevisionAuditRestorationDigestV1 {
        self.restoration_digest
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
        support_eligibility: &SupportHydrationEligibilityReceiptV1,
        restored_at_cycle: u64,
    ) -> Result<(), ImmutableRevisionAuditRestorationError> {
        let live = Self::restore(
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
            support_eligibility,
            restored_at_cycle,
        )?;
        if &live != self {
            return Err(ImmutableRevisionAuditRestorationError::RestorationMismatch);
        }
        if digest_restoration(self)? != self.restoration_digest {
            return Err(ImmutableRevisionAuditRestorationError::RestorationDigestMismatch);
        }
        Ok(())
    }
}

fn digest_restoration(
    restoration: &ImmutableRevisionAuditRestorationV1,
) -> Result<ImmutableRevisionAuditRestorationDigestV1, ImmutableRevisionAuditRestorationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-immutable-revision-audit-restoration-v1");
    hasher.update(&[1]);
    hasher.update(&restoration.restart_capture_cycle.to_le_bytes());
    hasher.update(&restoration.restored_at_cycle.to_le_bytes());
    hasher.update(&restoration.source_outer_checksum);
    hasher.update(&restoration.source_base_wire_checksum);
    hasher.update(&restoration.source_schema_wire_checksum);
    hasher.update(&restoration.support_hydration_eligibility_digest);
    hash_usize(&mut hasher, restoration.records.len())?;
    hash_usize(&mut hasher, restoration.eligible_count)?;
    hash_usize(&mut hasher, restoration.rejected_count)?;
    hash_usize(&mut hasher, restoration.mutation_source_count)?;
    hash_usize(&mut hasher, restoration.historically_reexecuted_count)?;
    hash_usize(&mut hasher, restoration.restored_without_reexecution_count)?;
    for record in &restoration.records {
        hasher.update(&record.receipt_id.0.to_le_bytes());
        hasher.update(&record.claim_id.0.to_le_bytes());
        hasher.update(&record.proposed_delta.to_bits().to_le_bytes());
        match record.mutation_id {
            Some(id) => {
                hasher.update(&[1]);
                hasher.update(&id.0.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
        hasher.update(&[u8::from(record.decision_snapshot.eligible)]);
        hasher.update(&record.evaluated_at_cycle.to_le_bytes());
        hasher.update(&[u8::from(record.historically_reexecuted)]);
        hasher.update(&[u8::from(record.restored_from_typed_persistence)]);
        hasher.update(&[u8::from(record.mutation_authority)]);
    }
    hasher.update(&[u8::from(
        restoration.complete_immutable_revision_audit_restored,
    )]);
    hasher.update(&[u8::from(
        restoration.operational_revision_history_constructed,
    )]);
    hasher.update(&[u8::from(
        restoration.nonmutation_decisions_historically_reexecuted,
    )]);
    hasher.update(&[u8::from(restoration.mutation_authority)]);
    hasher.update(&[u8::from(restoration.writable_hydration_authorized)]);
    hasher.update(&[u8::from(restoration.writable_state_export_authorized)]);
    hasher.update(&[u8::from(restoration.activation_authorized)]);
    Ok(ImmutableRevisionAuditRestorationDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), ImmutableRevisionAuditRestorationError> {
    let value = u64::try_from(value)
        .map_err(|_| ImmutableRevisionAuditRestorationError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

#[derive(Debug)]
pub enum ImmutableRevisionAuditRestorationError {
    SupportEligibilityRejected(SupportHydrationEligibilityError),
    RestorationPredatesReview {
        restored_at_cycle: u64,
        reviewed_at_cycle: u64,
    },
    CurrentnessExpired {
        restored_at_cycle: u64,
        expires_at_cycle: u64,
    },
    UnexpectedEligibilityAuthority,
    TooManyRevisionRecords {
        actual: usize,
        maximum: usize,
    },
    RevisionSchemaCountMismatch,
    DuplicateReplayedSourceRevision,
    DuplicateMutationSourceRevision,
    ReplayMutationCountMismatch {
        persisted: usize,
        replayed: usize,
        report: usize,
    },
    RevisionSchemaMismatch(u64),
    MutationSourceNotHistoricallyReexecuted(u64),
    UnexpectedHistoricalReexecution(u64),
    PlaceholderCountMismatch {
        restored_without_reexecution: usize,
        eligibility_placeholders: usize,
    },
    RestorationMismatch,
    RestorationDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for ImmutableRevisionAuditRestorationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "immutable revision audit restoration rejected: {self:?}")
    }
}

impl Error for ImmutableRevisionAuditRestorationError {}
