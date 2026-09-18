// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed split-state hydration sandbox for epistemic restart.
//!
//! EKM-064 proves persisted support mutations can be reproduced through the real
//! EKM-026 firewall. EKM-066 restores the complete typed revision audit surface
//! without constructing an operational `BeliefRevisionHistory`. This module keeps
//! those concerns split deliberately:
//!
//! - a real writable `EpistemicSupportStore` is reconstructed independently inside
//!   a sealed sandbox;
//! - the complete revision history remains an immutable EKM-066 audit overlay;
//! - the temporary replay ledger/history/firewall/authorizations are discarded;
//! - no mutable handle, operational revision history, activation handle, or trusted
//!   checkpoint mutation is exposed.
//! 
//! The independent replay is intentionally separate from EKM-064's implementation.
//! EKM-067 first verifies the EKM-064 report, then performs a second firewall replay
//! and requires the retained support store to re-export exactly to persistence.

use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationAuthorizationDecision, BeliefMutationFirewall,
    BeliefMutationOutcome, BeliefMutationReceiptId, EpistemicSupportStore,
};
use super::belief_mutation_persistence::BeliefMutationPersistenceCapsuleV1;
use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use super::belief_revision_gate::{
    BeliefRevisionPolicy, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{BeliefRevisionHistory, BeliefRevisionReceipt};
use super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1;
use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceId};
use super::epistemic_restart_historical_firewall_replay::{
    HistoricalFirewallReplayError, HistoricalFirewallReplayReportV1,
};
use super::epistemic_restart_historical_projection::{
    HistoricalEvidenceProjectionRecordV1, HistoricalEvidenceProjectionV1,
};
use super::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use super::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use super::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use super::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use super::epistemic_restart_quarantine_facade::ReadOnlyEpistemicRestartQuarantineV2;
use super::epistemic_restart_revision_audit_restoration::{
    ImmutableRevisionAuditRecordV1, ImmutableRevisionAuditRestorationError,
    ImmutableRevisionAuditRestorationV1,
};
use super::epistemic_restart_support_hydration_eligibility::{
    SupportHydrationEligibilityError, SupportHydrationEligibilityReceiptV1,
};
use super::epistemic_restart_trust_checkpoint::VerifiedRestartTrustContextCheckpointV1;
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire::{
    EpistemicRestartWireSnapshotV1, WireMutationV1, WireRevisionReceiptV1, WireSupportStateV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_vector::{ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

const MAX_SPLIT_STATE_MUTATIONS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStateHydrationSandboxVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplitStateHydrationSandboxDigestV1([u8; 32]);

impl SplitStateHydrationSandboxDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitStateSupportSummaryV1 {
    pub claim_id: ClaimId,
    pub support: f32,
    pub revision: u64,
    pub initialized_at_cycle: u64,
    pub last_updated_cycle: u64,
    pub last_mutation_id: Option<BeliefMutationReceiptId>,
}

/// A sealed split-state sandbox.
///
/// The support store is genuinely writable internally, but no public API exposes a
/// mutable or shared reference to it. The revision audit overlay is immutable and
/// cannot authorize belief mutation.
#[derive(Debug)]
pub struct SealedSplitStateHydrationSandboxV1 {
    version: SplitStateHydrationSandboxVersion,
    sandboxed_at_cycle: u64,
    restart_capture_cycle: u64,
    source_outer_checksum: [u8; 32],
    source_v2_digest: [u8; 32],
    replay_report_digest: [u8; 32],
    support_eligibility_digest: [u8; 32],
    audit_restoration_digest: [u8; 32],
    source_base: EpistemicRestartWireSnapshotV1,
    support_store: EpistemicSupportStore,
    audit: ImmutableRevisionAuditRestorationV1,
    support_store_constructed: bool,
    support_store_independently_replayed: bool,
    operational_revision_history_constructed: bool,
    immutable_revision_audit_attached: bool,
    trusted_state_mutated: bool,
    writable_state_export_authorized: bool,
    activation_authorized: bool,
    sandbox_digest: SplitStateHydrationSandboxDigestV1,
}

impl SealedSplitStateHydrationSandboxV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn hydrate_sealed(
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
        audit: ImmutableRevisionAuditRestorationV1,
        sandboxed_at_cycle: u64,
    ) -> Result<Self, SplitStateHydrationSandboxError> {
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
            .map_err(SplitStateHydrationSandboxError::SupportEligibilityRejected)?;
        audit
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
                support_eligibility,
                audit.restored_at_cycle(),
            )
            .map_err(SplitStateHydrationSandboxError::AuditRestorationRejected)?;
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
            .map_err(SplitStateHydrationSandboxError::ReplayReportRejected)?;

        if sandboxed_at_cycle < audit.restored_at_cycle() {
            return Err(SplitStateHydrationSandboxError::SandboxPredatesAuditRestoration {
                sandboxed_at_cycle,
                audit_restored_at_cycle: audit.restored_at_cycle(),
            });
        }
        if sandboxed_at_cycle < support_eligibility.reviewed_at_cycle() {
            return Err(SplitStateHydrationSandboxError::SandboxPredatesSupportReview {
                sandboxed_at_cycle,
                support_reviewed_at_cycle: support_eligibility.reviewed_at_cycle(),
            });
        }
        if sandboxed_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(SplitStateHydrationSandboxError::CurrentnessExpired {
                sandboxed_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }
        if sandboxed_at_cycle >= trust_checkpoint.statement().expires_at_cycle() {
            return Err(SplitStateHydrationSandboxError::TrustCheckpointExpired {
                sandboxed_at_cycle,
                expires_at_cycle: trust_checkpoint.statement().expires_at_cycle(),
            });
        }
        if !support_eligibility.support_store_hydration_review_eligible()
            || support_eligibility.writable_hydration_authorized()
            || support_eligibility.writable_state_export_authorized()
            || support_eligibility.activation_authorized()
        {
            return Err(SplitStateHydrationSandboxError::UnexpectedSupportEligibilityClaims);
        }
        if !audit.complete_immutable_revision_audit_restored()
            || audit.operational_revision_history_constructed()
            || audit.nonmutation_decisions_historically_reexecuted()
            || audit.mutation_authority()
            || audit.writable_hydration_authorized()
            || audit.writable_state_export_authorized()
            || audit.activation_authorized()
        {
            return Err(SplitStateHydrationSandboxError::UnexpectedAuditAuthority);
        }
        if restart.base.mutations.len() > MAX_SPLIT_STATE_MUTATIONS {
            return Err(SplitStateHydrationSandboxError::TooManyMutations {
                actual: restart.base.mutations.len(),
                maximum: MAX_SPLIT_STATE_MUTATIONS,
            });
        }

        let support_store = independently_replay_support_store(restart, projection)?;
        verify_store_against_base(&support_store, &restart.base)?;
        verify_audit_against_snapshot(&audit, restart)?;

        if replay_report.mutation_count() != support_store.history().len()
            || !replay_report.persisted_mutation_receipts_exactly_reproduced()
            || !replay_report.final_support_state_equivalent()
            || !replay_report.consumed_authorization_count_equivalent()
            || replay_report.writable_state_export_authorized()
            || replay_report.writable_hydration_authorized()
            || replay_report.activation_authorized()
        {
            return Err(SplitStateHydrationSandboxError::ReplayReportStateMismatch);
        }

        let mut out = Self {
            version: SplitStateHydrationSandboxVersion::V1,
            sandboxed_at_cycle,
            restart_capture_cycle: restart.base.captured_at_cycle,
            source_outer_checksum: restart.outer_checksum,
            source_v2_digest: restart.claimed_v2_digest,
            replay_report_digest: replay_report.report_digest().as_bytes(),
            support_eligibility_digest: support_eligibility.receipt_digest().as_bytes(),
            audit_restoration_digest: audit.restoration_digest().as_bytes(),
            source_base: restart.base.clone(),
            support_store,
            audit,
            support_store_constructed: true,
            support_store_independently_replayed: true,
            operational_revision_history_constructed: false,
            immutable_revision_audit_attached: true,
            trusted_state_mutated: false,
            writable_state_export_authorized: false,
            activation_authorized: false,
            sandbox_digest: SplitStateHydrationSandboxDigestV1([0; 32]),
        };
        out.sandbox_digest = digest_sandbox(&out)?;
        out.verify()?;
        Ok(out)
    }

    pub fn version(&self) -> SplitStateHydrationSandboxVersion {
        self.version
    }

    pub fn sandboxed_at_cycle(&self) -> u64 {
        self.sandboxed_at_cycle
    }

    pub fn restart_capture_cycle(&self) -> u64 {
        self.restart_capture_cycle
    }

    pub fn support_state_count(&self) -> usize {
        self.support_store.len()
    }

    pub fn mutation_count(&self) -> usize {
        self.support_store.history().len()
    }

    pub fn revision_audit_record_count(&self) -> usize {
        self.audit.records().len()
    }

    pub fn support_summary(&self, claim_id: ClaimId) -> Option<SplitStateSupportSummaryV1> {
        let state = self.support_store.state(claim_id)?;
        Some(SplitStateSupportSummaryV1 {
            claim_id,
            support: state.support().get(),
            revision: state.revision(),
            initialized_at_cycle: state.initialized_at_cycle(),
            last_updated_cycle: state.last_updated_cycle(),
            last_mutation_id: state.last_mutation_id(),
        })
    }

    pub fn revision_audit_record(
        &self,
        receipt_id: u64,
    ) -> Option<&ImmutableRevisionAuditRecordV1> {
        self.audit
            .records()
            .iter()
            .find(|record| record.receipt_id().0 == receipt_id)
    }

    pub fn support_store_constructed(&self) -> bool {
        self.support_store_constructed
    }

    pub fn support_store_independently_replayed(&self) -> bool {
        self.support_store_independently_replayed
    }

    pub fn operational_revision_history_constructed(&self) -> bool {
        self.operational_revision_history_constructed
    }

    pub fn immutable_revision_audit_attached(&self) -> bool {
        self.immutable_revision_audit_attached
    }

    pub fn writable_state_export_authorized(&self) -> bool {
        self.writable_state_export_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn sandbox_digest(&self) -> SplitStateHydrationSandboxDigestV1 {
        self.sandbox_digest
    }

    /// Self-check retained state without exposing its writable objects.
    pub fn verify(&self) -> Result<(), SplitStateHydrationSandboxError> {
        if !self.support_store_constructed
            || !self.support_store_independently_replayed
            || self.operational_revision_history_constructed
            || !self.immutable_revision_audit_attached
            || self.trusted_state_mutated
            || self.writable_state_export_authorized
            || self.activation_authorized
        {
            return Err(SplitStateHydrationSandboxError::UnexpectedSandboxAuthority);
        }
        verify_store_against_base(&self.support_store, &self.source_base)?;
        verify_audit_records_against_base(&self.audit, &self.source_base)?;
        if digest_sandbox(self)? != self.sandbox_digest {
            return Err(SplitStateHydrationSandboxError::SandboxDigestMismatch);
        }
        Ok(())
    }
}

fn independently_replay_support_store(
    restart: &EpistemicRestartWireSnapshotV2,
    projection: &HistoricalEvidenceProjectionV1,
) -> Result<EpistemicSupportStore, SplitStateHydrationSandboxError> {
    let full_ledger = rebuild_ledger_prefix(&restart.base, None, None, None)?;
    let source_mutations = mutation_by_source_revision(&restart.base.mutations)?;
    let projection_by_mutation = projection_record_by_mutation(projection)?;
    if projection_by_mutation.len() != restart.base.mutations.len() {
        return Err(SplitStateHydrationSandboxError::ProjectionMutationCountMismatch);
    }

    let mut revision_history = BeliefRevisionHistory::new();
    let placeholder_policy = BeliefRevisionPolicy::new(1.0, 0, false, 0, 1.0)
        .map_err(|_| SplitStateHydrationSandboxError::PlaceholderPolicyRejected)?;

    for wire in &restart.base.revisions {
        if let Some(mutation) = source_mutations.get(&wire.id.0) {
            let projected = projection_by_mutation
                .get(&mutation.id.0)
                .copied()
                .ok_or(SplitStateHydrationSandboxError::ProjectionMissingMutation(
                    mutation.id.0,
                ))?;
            let historical_ledger = rebuild_historical_ledger(&restart.base, projected)?;
            record_actual_revision(
                &mut revision_history,
                &historical_ledger,
                restart,
                wire,
            )?;
        } else {
            record_placeholder_revision(
                &mut revision_history,
                &full_ledger,
                wire,
                &placeholder_policy,
            )?;
        }
    }
    if revision_history.len() != restart.base.revisions.len() {
        return Err(SplitStateHydrationSandboxError::RevisionIdentitySpaceMismatch);
    }

    let mut store = EpistemicSupportStore::new();
    for state in &restart.base.support_states {
        store
            .register_claim(
                &full_ledger,
                state.claim_id,
                state.baseline_support,
                state.initialized_at_cycle,
            )
            .map_err(|_| SplitStateHydrationSandboxError::SupportBaselineRejected(state.claim_id))?;
    }

    let mut firewall = BeliefMutationFirewall::new();
    for mutation in &restart.base.mutations {
        let projected = projection_by_mutation
            .get(&mutation.id.0)
            .copied()
            .ok_or(SplitStateHydrationSandboxError::ProjectionMissingMutation(
                mutation.id.0,
            ))?;
        if projected.source_revision_receipt_id() != mutation.source_revision_receipt_id
            || projected.claim_id() != mutation.claim_id
        {
            return Err(SplitStateHydrationSandboxError::ProjectionMutationBindingMismatch(
                mutation.id.0,
            ));
        }
        let historical_ledger = rebuild_historical_ledger(&restart.base, projected)?;
        let revision = revision_history
            .get(mutation.source_revision_receipt_id)
            .ok_or(SplitStateHydrationSandboxError::SourceRevisionMissing(
                mutation.source_revision_receipt_id.0,
            ))?;
        let state = store
            .state(mutation.claim_id)
            .ok_or(SplitStateHydrationSandboxError::SupportStateMissing(mutation.claim_id))?;
        if state.revision() != mutation.state_revision_before
            || state.support() != mutation.support_before
        {
            return Err(SplitStateHydrationSandboxError::PersistedPreStateMismatch(
                mutation.id.0,
            ));
        }
        let authorization = BeliefMutationAuthorization::new(
            mutation.authorization_id.clone(),
            mutation.authority_label.clone(),
            BeliefMutationAuthorizationDecision::Approved,
            mutation.authorized_at_cycle,
            revision,
            state,
        )
        .map_err(|_| SplitStateHydrationSandboxError::AuthorizationRebuildRejected(
            mutation.id.0,
        ))?;
        let outcome = firewall
            .apply(
                &historical_ledger,
                &mut store,
                revision,
                &authorization,
                mutation.applied_at_cycle,
            )
            .map_err(|error| SplitStateHydrationSandboxError::FirewallReplayRejected {
                mutation_id: mutation.id.0,
                detail: format!("{error:?}"),
            })?;
        if !matches!(outcome, BeliefMutationOutcome::Applied(_)) {
            return Err(SplitStateHydrationSandboxError::FirewallReplayWasNotNew(
                mutation.id.0,
            ));
        }
        compare_mutation(outcome.receipt(), mutation)?;
    }

    verify_store_against_base(&store, &restart.base)?;
    Ok(store)
}

fn mutation_by_source_revision(
    mutations: &[WireMutationV1],
) -> Result<HashMap<u64, &WireMutationV1>, SplitStateHydrationSandboxError> {
    let mut out = HashMap::with_capacity(mutations.len());
    for mutation in mutations {
        if out
            .insert(mutation.source_revision_receipt_id.0, mutation)
            .is_some()
        {
            return Err(SplitStateHydrationSandboxError::DuplicateSourceRevision(
                mutation.source_revision_receipt_id.0,
            ));
        }
    }
    Ok(out)
}

fn projection_record_by_mutation(
    projection: &HistoricalEvidenceProjectionV1,
) -> Result<HashMap<u64, &HistoricalEvidenceProjectionRecordV1>, SplitStateHydrationSandboxError> {
    let mut out = HashMap::with_capacity(projection.records().len());
    for record in projection.records() {
        if out.insert(record.mutation_id().0, record).is_some() {
            return Err(SplitStateHydrationSandboxError::DuplicateProjectionMutation(
                record.mutation_id().0,
            ));
        }
    }
    Ok(out)
}

fn rebuild_historical_ledger(
    base: &EpistemicRestartWireSnapshotV1,
    projection: &HistoricalEvidenceProjectionRecordV1,
) -> Result<EpistemicLedger, SplitStateHydrationSandboxError> {
    let max_evidence_id = projection
        .evidence()
        .iter()
        .map(|record| record.evidence_id.0)
        .max();
    let evidence_prefix = max_evidence_id
        .map(|maximum| {
            base.evidence
                .iter()
                .take_while(move |record| record.id.0 <= maximum)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let max_claim_id = evidence_prefix
        .iter()
        .map(|record| record.claim_id.0)
        .chain(std::iter::once(projection.claim_id().0))
        .max();
    let max_provenance_id = evidence_prefix
        .iter()
        .map(|record| record.provenance_id.0)
        .max();
    let ledger = rebuild_ledger_prefix(
        base,
        max_provenance_id,
        max_claim_id,
        max_evidence_id,
    )?;

    let mut live_ids = ledger
        .evidence_for_claim(projection.claim_id())
        .into_iter()
        .map(|record| record.id)
        .collect::<Vec<_>>();
    live_ids.sort_unstable();
    let mut protected_ids = projection
        .evidence()
        .iter()
        .map(|record| record.evidence_id)
        .collect::<Vec<_>>();
    protected_ids.sort_unstable();
    if live_ids != protected_ids {
        return Err(SplitStateHydrationSandboxError::HistoricalClaimCensusMismatch(
            projection.mutation_id().0,
        ));
    }
    for expected in projection.evidence() {
        let live = ledger
            .evidence(expected.evidence_id)
            .ok_or(SplitStateHydrationSandboxError::HistoricalEvidenceMissing(
                expected.evidence_id,
            ))?;
        if live.id != expected.evidence_id
            || live.claim_id != expected.claim_id
            || live.kind != expected.kind
            || live.polarity != expected.polarity
            || live.provenance_id != expected.provenance_id
            || live.observed_at_cycle != expected.observed_at_cycle
            || live.context != expected.context
            || live.method != expected.method
        {
            return Err(SplitStateHydrationSandboxError::HistoricalEvidenceMismatch(
                expected.evidence_id,
            ));
        }
    }
    Ok(ledger)
}

fn rebuild_ledger_prefix(
    base: &EpistemicRestartWireSnapshotV1,
    max_provenance_id: Option<u64>,
    max_claim_id: Option<u64>,
    max_evidence_id: Option<u64>,
) -> Result<EpistemicLedger, SplitStateHydrationSandboxError> {
    let mut ledger = EpistemicLedger::new();
    for record in &base.provenance {
        if max_provenance_id.is_some_and(|maximum| record.id.0 > maximum) {
            break;
        }
        let id = ledger
            .add_provenance(
                record.source_label.clone(),
                record.source_uri.clone(),
                record.content_hash.clone(),
                record.recorded_at_cycle,
                record.parent_ids.clone(),
            )
            .map_err(|_| SplitStateHydrationSandboxError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(SplitStateHydrationSandboxError::LedgerIdMismatch);
        }
    }
    for record in &base.claims {
        if max_claim_id.is_some_and(|maximum| record.id.0 > maximum) {
            break;
        }
        let id = ledger.add_claim(
            record.statement.clone(),
            record.kind,
            record.domain.clone(),
            record.scope.clone(),
            record.created_at_cycle,
        );
        if id != record.id {
            return Err(SplitStateHydrationSandboxError::LedgerIdMismatch);
        }
    }
    for record in &base.evidence {
        if max_evidence_id.is_some_and(|maximum| record.id.0 > maximum) {
            break;
        }
        let id = ledger
            .add_evidence(
                record.claim_id,
                record.kind,
                record.polarity,
                record.provenance_id,
                record.observed_at_cycle,
                record.context.clone(),
                record.method.clone(),
            )
            .map_err(|_| SplitStateHydrationSandboxError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(SplitStateHydrationSandboxError::LedgerIdMismatch);
        }
    }
    Ok(ledger)
}

fn record_actual_revision(
    history: &mut BeliefRevisionHistory,
    ledger: &EpistemicLedger,
    restart: &EpistemicRestartWireSnapshotV2,
    wire: &WireRevisionReceiptV1,
) -> Result<(), SplitStateHydrationSandboxError> {
    let schema = restart
        .revision_schemas
        .records
        .iter()
        .find(|record| record.receipt_id == wire.id)
        .ok_or(SplitStateHydrationSandboxError::RevisionSchemaMissing(wire.id.0))?;
    let policy = schema
        .policy_schema
        .build_policy()
        .map_err(|_| SplitStateHydrationSandboxError::PolicyRebuildRejected(wire.id.0))?;
    let proposal = rebuild_proposal(wire)?;
    let calibration = rebuild_calibration(wire)?;
    let uncertainty = rebuild_uncertainty(ledger, wire)?;
    let id = history
        .evaluate_and_record(
            ledger,
            &proposal,
            &policy,
            calibration,
            uncertainty.as_ref(),
            wire.evaluated_at_cycle,
        )
        .map_err(|_| SplitStateHydrationSandboxError::RevisionReplayRejected(wire.id.0))?;
    if id != wire.id {
        return Err(SplitStateHydrationSandboxError::RevisionIdMismatch {
            expected: wire.id.0,
            actual: id.0,
        });
    }
    let receipt = history
        .get(id)
        .ok_or(SplitStateHydrationSandboxError::SourceRevisionMissing(id.0))?;
    compare_revision(receipt, wire, &schema.decision_snapshot)
}

fn record_placeholder_revision(
    history: &mut BeliefRevisionHistory,
    ledger: &EpistemicLedger,
    wire: &WireRevisionReceiptV1,
    policy: &BeliefRevisionPolicy,
) -> Result<(), SplitStateHydrationSandboxError> {
    let proposal = EpistemicRevisionProposal::new(
        wire.claim_id,
        0.0,
        Vec::new(),
        "split-state-replay-id-placeholder",
    )
    .map_err(|_| SplitStateHydrationSandboxError::PlaceholderProposalRejected(wire.id.0))?;
    let id = history
        .evaluate_and_record(
            ledger,
            &proposal,
            policy,
            None,
            None,
            wire.evaluated_at_cycle,
        )
        .map_err(|_| SplitStateHydrationSandboxError::PlaceholderRevisionRejected(wire.id.0))?;
    if id != wire.id {
        return Err(SplitStateHydrationSandboxError::RevisionIdMismatch {
            expected: wire.id.0,
            actual: id.0,
        });
    }
    Ok(())
}

fn rebuild_proposal(
    wire: &WireRevisionReceiptV1,
) -> Result<EpistemicRevisionProposal, SplitStateHydrationSandboxError> {
    let mut basis_ids = wire
        .basis
        .iter()
        .map(|basis| basis.requested_id)
        .collect::<Vec<_>>();
    for duplicate in &wire.duplicate_basis_evidence_ids {
        if !basis_ids.contains(duplicate) {
            return Err(SplitStateHydrationSandboxError::DuplicateBasisDiagnosticOrphan {
                receipt_id: wire.id.0,
                evidence_id: duplicate.0,
            });
        }
        basis_ids.push(*duplicate);
    }
    EpistemicRevisionProposal::new(
        wire.claim_id,
        wire.proposed_delta,
        basis_ids,
        wire.rationale.clone(),
    )
    .map_err(|_| SplitStateHydrationSandboxError::ProposalRebuildRejected(wire.id.0))
}

fn rebuild_calibration(
    wire: &WireRevisionReceiptV1,
) -> Result<Option<CalibrationSnapshot>, SplitStateHydrationSandboxError> {
    wire.calibration
        .map(|(sample_count, ece)| {
            CalibrationSnapshot::new(sample_count, ece)
                .map_err(|_| SplitStateHydrationSandboxError::CalibrationRebuildRejected(wire.id.0))
        })
        .transpose()
}

fn rebuild_uncertainty(
    ledger: &EpistemicLedger,
    wire: &WireRevisionReceiptV1,
) -> Result<Option<ClaimUncertaintyAssessment>, SplitStateHydrationSandboxError> {
    let Some(source) = &wire.uncertainty else {
        return Ok(None);
    };
    let mut vector = EpistemicVector::new();
    for (dimension, value) in [
        (UncertaintyDimension::Epistemic, source.epistemic),
        (UncertaintyDimension::Aleatoric, source.aleatoric),
        (UncertaintyDimension::Ontological, source.ontological),
        (
            UncertaintyDimension::DistributionShift,
            source.distribution_shift,
        ),
    ] {
        if let Some(value) = value {
            vector
                .set(dimension, value)
                .map_err(|_| SplitStateHydrationSandboxError::UncertaintyRebuildRejected(wire.id.0))?;
        }
    }
    ClaimUncertaintyAssessment::new(
        ledger,
        source.claim_id,
        vector,
        source.basis_evidence_ids.clone(),
        source.assessed_at_cycle,
    )
    .map(Some)
    .map_err(|_| SplitStateHydrationSandboxError::UncertaintyRebuildRejected(wire.id.0))
}

fn compare_revision(
    rebuilt: &BeliefRevisionReceipt,
    wire: &WireRevisionReceiptV1,
    expected_decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), SplitStateHydrationSandboxError> {
    if rebuilt.id() != wire.id
        || rebuilt.claim_id() != wire.claim_id
        || rebuilt.proposed_delta().to_bits() != wire.proposed_delta.to_bits()
        || rebuilt.rationale() != wire.rationale
        || rebuilt.evaluated_at_cycle() != wire.evaluated_at_cycle
        || rebuilt.duplicate_basis_evidence_ids() != wire.duplicate_basis_evidence_ids
        || rebuilt.basis().len() != wire.basis.len()
        || BeliefRevisionDecisionSnapshotV1::capture(rebuilt.decision()) != *expected_decision
    {
        return Err(SplitStateHydrationSandboxError::RevisionSemanticMismatch(wire.id.0));
    }
    for (left, right) in rebuilt.basis().iter().zip(&wire.basis) {
        if left.requested_id != right.requested_id || left.snapshot != right.snapshot {
            return Err(SplitStateHydrationSandboxError::RevisionSemanticMismatch(wire.id.0));
        }
    }
    if !calibration_matches(rebuilt.calibration(), wire.calibration)
        || !uncertainty_matches(rebuilt.uncertainty(), wire.uncertainty.as_ref())
    {
        return Err(SplitStateHydrationSandboxError::RevisionSemanticMismatch(wire.id.0));
    }
    Ok(())
}

fn calibration_matches(left: Option<CalibrationSnapshot>, right: Option<(u64, f64)>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some((sample_count, ece))) => {
            left.sample_count == sample_count && left.ece.to_bits() == ece.to_bits()
        }
        _ => false,
    }
}

fn uncertainty_matches(
    left: Option<&ClaimUncertaintyAssessment>,
    right: Option<&super::epistemic_restart_wire::WireUncertaintyAssessmentV1>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.claim_id == right.claim_id
                && left.basis_evidence_ids == right.basis_evidence_ids
                && left.assessed_at_cycle == right.assessed_at_cycle
                && uncertainty_value_bits(left.vector.epistemic.map(|v| v.get()))
                    == uncertainty_value_bits(right.epistemic)
                && uncertainty_value_bits(left.vector.aleatoric.map(|v| v.get()))
                    == uncertainty_value_bits(right.aleatoric)
                && uncertainty_value_bits(left.vector.ontological.map(|v| v.get()))
                    == uncertainty_value_bits(right.ontological)
                && uncertainty_value_bits(left.vector.distribution_shift.map(|v| v.get()))
                    == uncertainty_value_bits(right.distribution_shift)
        }
        _ => false,
    }
}

fn uncertainty_value_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn compare_mutation(
    receipt: &super::belief_mutation_firewall::BeliefMutationReceipt,
    wire: &WireMutationV1,
) -> Result<(), SplitStateHydrationSandboxError> {
    if receipt.id() != wire.id
        || receipt.source_revision_receipt_id() != wire.source_revision_receipt_id
        || receipt.claim_id() != wire.claim_id
        || receipt.proposed_delta().to_bits() != wire.proposed_delta.to_bits()
        || receipt.support_before() != wire.support_before
        || receipt.support_after() != wire.support_after
        || receipt.state_revision_before() != wire.state_revision_before
        || receipt.state_revision_after() != wire.state_revision_after
        || receipt.authorization_id() != wire.authorization_id
        || receipt.authority_label() != wire.authority_label
        || receipt.authorized_at_cycle() != wire.authorized_at_cycle
        || receipt.applied_at_cycle() != wire.applied_at_cycle
    {
        return Err(SplitStateHydrationSandboxError::MutationSemanticMismatch(wire.id.0));
    }
    Ok(())
}

fn verify_store_against_base(
    store: &EpistemicSupportStore,
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<(), SplitStateHydrationSandboxError> {
    if store.len() != base.support_states.len() || store.history().len() != base.mutations.len() {
        return Err(SplitStateHydrationSandboxError::SupportStateCountMismatch);
    }
    for wire in &base.support_states {
        let state = store
            .state(wire.claim_id)
            .ok_or(SplitStateHydrationSandboxError::SupportStateMissing(wire.claim_id))?;
        if state.support() != wire.current_support
            || state.revision() != wire.revision
            || state.initialized_at_cycle() != wire.initialized_at_cycle
            || state.last_updated_cycle() != wire.last_updated_cycle
            || state.last_mutation_id() != wire.last_mutation_id
        {
            return Err(SplitStateHydrationSandboxError::SupportStateMismatch(wire.claim_id));
        }
    }
    for (receipt, wire) in store.history().iter().zip(&base.mutations) {
        compare_mutation(receipt, wire)?;
    }
    let consumed = u64::try_from(store.consumed_authorization_count())
        .map_err(|_| SplitStateHydrationSandboxError::LengthOverflow)?;
    if consumed != base.consumed_authorization_count {
        return Err(SplitStateHydrationSandboxError::ConsumedAuthorizationCountMismatch {
            replayed: store.consumed_authorization_count(),
            persisted: base.consumed_authorization_count,
        });
    }

    let claim_ids = base
        .support_states
        .iter()
        .map(|state| state.claim_id)
        .collect::<Vec<_>>();
    let capsule = BeliefMutationPersistenceCapsuleV1::capture(
        store,
        &claim_ids,
        base.captured_at_cycle,
    )
    .map_err(|error| SplitStateHydrationSandboxError::SupportRecaptureRejected {
        detail: format!("{error:?}"),
    })?;
    if capsule.states().len() != base.support_states.len()
        || capsule.mutations().len() != base.mutations.len()
        || u64::try_from(capsule.consumed_authorization_count())
            .map_err(|_| SplitStateHydrationSandboxError::LengthOverflow)?
            != base.consumed_authorization_count
    {
        return Err(SplitStateHydrationSandboxError::SupportRecaptureMismatch);
    }
    for (left, right) in capsule.states().iter().zip(&base.support_states) {
        if left.claim_id != right.claim_id
            || left.baseline_support != right.baseline_support
            || left.current_support != right.current_support
            || left.revision != right.revision
            || left.initialized_at_cycle != right.initialized_at_cycle
            || left.last_updated_cycle != right.last_updated_cycle
            || left.last_mutation_id != right.last_mutation_id
        {
            return Err(SplitStateHydrationSandboxError::SupportRecaptureMismatch);
        }
    }
    for (left, right) in capsule.mutations().iter().zip(&base.mutations) {
        if left.id != right.id
            || left.source_revision_receipt_id != right.source_revision_receipt_id
            || left.claim_id != right.claim_id
            || left.proposed_delta.to_bits() != right.proposed_delta.to_bits()
            || left.support_before != right.support_before
            || left.support_after != right.support_after
            || left.state_revision_before != right.state_revision_before
            || left.state_revision_after != right.state_revision_after
            || left.authorization_id != right.authorization_id
            || left.authority_label != right.authority_label
            || left.authorized_at_cycle != right.authorized_at_cycle
            || left.applied_at_cycle != right.applied_at_cycle
        {
            return Err(SplitStateHydrationSandboxError::SupportRecaptureMismatch);
        }
    }
    Ok(())
}

fn verify_audit_against_snapshot(
    audit: &ImmutableRevisionAuditRestorationV1,
    restart: &EpistemicRestartWireSnapshotV2,
) -> Result<(), SplitStateHydrationSandboxError> {
    verify_audit_records_against_base(audit, &restart.base)?;
    if audit.records().len() != restart.revision_schemas.records.len() {
        return Err(SplitStateHydrationSandboxError::AuditRecordCountMismatch);
    }
    for (record, schema) in audit.records().iter().zip(&restart.revision_schemas.records) {
        if record.receipt_id() != schema.receipt_id
            || record.claim_id() != schema.claim_id
            || record.proposed_delta().to_bits() != schema.proposed_delta.to_bits()
            || record.evaluated_at_cycle() != schema.evaluated_at_cycle
            || record.policy_schema() != &schema.policy_schema
            || record.decision_snapshot() != &schema.decision_snapshot
            || record.mutation_authority()
        {
            return Err(SplitStateHydrationSandboxError::AuditSchemaMismatch(
                record.receipt_id().0,
            ));
        }
    }
    Ok(())
}

fn verify_audit_records_against_base(
    audit: &ImmutableRevisionAuditRestorationV1,
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<(), SplitStateHydrationSandboxError> {
    if audit.records().len() != base.revisions.len() {
        return Err(SplitStateHydrationSandboxError::AuditRecordCountMismatch);
    }
    let mutation_by_revision = base
        .mutations
        .iter()
        .map(|mutation| (mutation.source_revision_receipt_id.0, mutation.id))
        .collect::<HashMap<_, _>>();
    if mutation_by_revision.len() != base.mutations.len() {
        return Err(SplitStateHydrationSandboxError::DuplicateSourceRevision(0));
    }
    for (record, wire) in audit.records().iter().zip(&base.revisions) {
        let expected_mutation = mutation_by_revision.get(&wire.id.0).copied();
        if record.receipt_id() != wire.id
            || record.claim_id() != wire.claim_id
            || record.proposed_delta().to_bits() != wire.proposed_delta.to_bits()
            || record.rationale() != wire.rationale
            || record.basis() != wire.basis
            || record.duplicate_basis_evidence_ids() != wire.duplicate_basis_evidence_ids
            || record.calibration() != wire.calibration
            || record.uncertainty() != wire.uncertainty.as_ref()
            || record.evaluated_at_cycle() != wire.evaluated_at_cycle
            || record.mutation_id() != expected_mutation
            || !record.restored_from_typed_persistence()
            || record.mutation_authority()
        {
            return Err(SplitStateHydrationSandboxError::AuditRecordMismatch(wire.id.0));
        }
    }
    Ok(())
}

fn digest_sandbox(
    sandbox: &SealedSplitStateHydrationSandboxV1,
) -> Result<SplitStateHydrationSandboxDigestV1, SplitStateHydrationSandboxError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-sealed-split-state-hydration-sandbox-v1");
    hasher.update(&[1]);
    hasher.update(&sandbox.sandboxed_at_cycle.to_le_bytes());
    hasher.update(&sandbox.restart_capture_cycle.to_le_bytes());
    hasher.update(&sandbox.source_outer_checksum);
    hasher.update(&sandbox.source_v2_digest);
    hasher.update(&sandbox.replay_report_digest);
    hasher.update(&sandbox.support_eligibility_digest);
    hasher.update(&sandbox.audit_restoration_digest);
    hash_usize(&mut hasher, sandbox.support_store.len())?;
    hash_usize(&mut hasher, sandbox.support_store.history().len())?;
    hash_usize(&mut hasher, sandbox.audit.records().len())?;
    for state in &sandbox.source_base.support_states {
        let live = sandbox
            .support_store
            .state(state.claim_id)
            .ok_or(SplitStateHydrationSandboxError::SupportStateMissing(state.claim_id))?;
        hasher.update(&state.claim_id.0.to_le_bytes());
        hasher.update(&live.support().get().to_bits().to_le_bytes());
        hasher.update(&live.revision().to_le_bytes());
        hasher.update(&live.initialized_at_cycle().to_le_bytes());
        hasher.update(&live.last_updated_cycle().to_le_bytes());
        match live.last_mutation_id() {
            Some(id) => {
                hasher.update(&[1]);
                hasher.update(&id.0.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
    }
    hasher.update(&[u8::from(sandbox.support_store_constructed)]);
    hasher.update(&[u8::from(sandbox.support_store_independently_replayed)]);
    hasher.update(&[u8::from(sandbox.operational_revision_history_constructed)]);
    hasher.update(&[u8::from(sandbox.immutable_revision_audit_attached)]);
    hasher.update(&[u8::from(sandbox.trusted_state_mutated)]);
    hasher.update(&[u8::from(sandbox.writable_state_export_authorized)]);
    hasher.update(&[u8::from(sandbox.activation_authorized)]);
    Ok(SplitStateHydrationSandboxDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), SplitStateHydrationSandboxError> {
    let value = u64::try_from(value).map_err(|_| SplitStateHydrationSandboxError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug)]
pub enum SplitStateHydrationSandboxError {
    SupportEligibilityRejected(SupportHydrationEligibilityError),
    AuditRestorationRejected(ImmutableRevisionAuditRestorationError),
    ReplayReportRejected(HistoricalFirewallReplayError),
    SandboxPredatesAuditRestoration {
        sandboxed_at_cycle: u64,
        audit_restored_at_cycle: u64,
    },
    SandboxPredatesSupportReview {
        sandboxed_at_cycle: u64,
        support_reviewed_at_cycle: u64,
    },
    CurrentnessExpired {
        sandboxed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    TrustCheckpointExpired {
        sandboxed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    UnexpectedSupportEligibilityClaims,
    UnexpectedAuditAuthority,
    TooManyMutations { actual: usize, maximum: usize },
    DuplicateSourceRevision(u64),
    DuplicateProjectionMutation(u64),
    ProjectionMutationCountMismatch,
    ProjectionMissingMutation(u64),
    ProjectionMutationBindingMismatch(u64),
    PlaceholderPolicyRejected,
    PlaceholderProposalRejected(u64),
    PlaceholderRevisionRejected(u64),
    RevisionIdentitySpaceMismatch,
    RevisionSchemaMissing(u64),
    PolicyRebuildRejected(u64),
    DuplicateBasisDiagnosticOrphan { receipt_id: u64, evidence_id: u64 },
    ProposalRebuildRejected(u64),
    CalibrationRebuildRejected(u64),
    UncertaintyRebuildRejected(u64),
    RevisionReplayRejected(u64),
    RevisionIdMismatch { expected: u64, actual: u64 },
    RevisionSemanticMismatch(u64),
    LedgerRebuildRejected,
    LedgerIdMismatch,
    HistoricalClaimCensusMismatch(u64),
    HistoricalEvidenceMissing(EvidenceId),
    HistoricalEvidenceMismatch(EvidenceId),
    SupportBaselineRejected(ClaimId),
    SourceRevisionMissing(u64),
    SupportStateMissing(ClaimId),
    PersistedPreStateMismatch(u64),
    AuthorizationRebuildRejected(u64),
    FirewallReplayRejected { mutation_id: u64, detail: String },
    FirewallReplayWasNotNew(u64),
    MutationSemanticMismatch(u64),
    SupportStateCountMismatch,
    SupportStateMismatch(ClaimId),
    ConsumedAuthorizationCountMismatch { replayed: usize, persisted: u64 },
    SupportRecaptureRejected { detail: String },
    SupportRecaptureMismatch,
    AuditRecordCountMismatch,
    AuditRecordMismatch(u64),
    AuditSchemaMismatch(u64),
    ReplayReportStateMismatch,
    UnexpectedSandboxAuthority,
    SandboxDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for SplitStateHydrationSandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sealed split-state hydration sandbox rejected: {self:?}")
    }
}

impl Error for SplitStateHydrationSandboxError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_sandbox_surface_does_not_expose_mutable_support_store() {
        fn type_check(_: Option<&SealedSplitStateHydrationSandboxV1>) {}
        type_check(None);
    }
}
