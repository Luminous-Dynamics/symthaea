// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isolated EKM-026 historical firewall replay verification.
//!
//! EKM-063 proves which evidence records belonged to each successful mutation-time
//! claim census. This module uses those protected projections to execute the real
//! EKM-026 belief-mutation firewall inside disposable private state and compare the
//! resulting mutation receipts/support state with persistence.
//!
//! No replayed ledger, support store, revision history, authorization, or mutation
//! outcome is exposed. Non-mutating revision receipts are represented only by
//! private ID-cursor placeholders so the source receipt IDs of persisted mutations
//! can be reproduced without pretending the complete rejected-decision history was
//! historically replayed.

use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationAuthorizationDecision, BeliefMutationFirewall,
    BeliefMutationOutcome, EpistemicSupportStore,
};
use super::belief_revision_gate::{
    BeliefRevisionPolicy, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_receipt::{BeliefRevisionHistory, BeliefRevisionReceipt};
use super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1;
use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceId};
use super::epistemic_restart_historical_projection::{
    HistoricalEvidenceProjectionError, HistoricalEvidenceProjectionRecordV1,
    HistoricalEvidenceProjectionV1,
};
use super::epistemic_restart_historical_replay_eligibility::HistoricalReplayEligibilityReceiptV1;
use super::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use super::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use super::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire::{
    EpistemicRestartWireSnapshotV1, WireMutationV1, WireRevisionReceiptV1, WireSupportStateV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_vector::{ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension};
use super::belief_mutation_seal_wire::BeliefMutationSealWireSnapshotV1;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

const MAX_ISOLATED_REPLAY_MUTATIONS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalFirewallReplayVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoricalFirewallReplayDigestV1([u8; 32]);

impl HistoricalFirewallReplayDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalFirewallMutationReplayV1 {
    mutation_id: u64,
    source_revision_receipt_id: u64,
    claim_id: ClaimId,
    projection_record_digest: [u8; 32],
    historical_evidence_count: usize,
    excluded_final_evidence_count: usize,
    excluded_backdated_final_evidence_count: usize,
    support_before_bits: u32,
    support_after_bits: u32,
    state_revision_before: u64,
    state_revision_after: u64,
    exact_receipt_reproduced: bool,
    digest: HistoricalFirewallReplayDigestV1,
}

impl HistoricalFirewallMutationReplayV1 {
    pub fn mutation_id(&self) -> u64 {
        self.mutation_id
    }

    pub fn source_revision_receipt_id(&self) -> u64 {
        self.source_revision_receipt_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn historical_evidence_count(&self) -> usize {
        self.historical_evidence_count
    }

    pub fn excluded_final_evidence_count(&self) -> usize {
        self.excluded_final_evidence_count
    }

    pub fn excluded_backdated_final_evidence_count(&self) -> usize {
        self.excluded_backdated_final_evidence_count
    }

    pub fn exact_receipt_reproduced(&self) -> bool {
        self.exact_receipt_reproduced
    }

    pub fn digest(&self) -> HistoricalFirewallReplayDigestV1 {
        self.digest
    }
}

/// Verification-only replay report. All mutable replay objects are dropped before
/// this value is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalFirewallReplayReportV1 {
    version: HistoricalFirewallReplayVersion,
    restart_capture_cycle: u64,
    replayed_at_cycle: u64,
    projection_digest: [u8; 32],
    eligibility_digest: [u8; 32],
    mutation_count: usize,
    firewall_invocation_count: usize,
    non_mutation_revision_placeholder_count: usize,
    mutation_replays: Vec<HistoricalFirewallMutationReplayV1>,
    persisted_mutation_receipts_exactly_reproduced: bool,
    final_support_state_equivalent: bool,
    consumed_authorization_count_equivalent: bool,
    full_nonmutation_revision_history_replayed: bool,
    isolated_firewall_replay_performed: bool,
    writable_state_export_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    report_digest: HistoricalFirewallReplayDigestV1,
}

impl HistoricalFirewallReplayReportV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn verify_isolated(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replayed_at_cycle: u64,
    ) -> Result<Self, HistoricalFirewallReplayError> {
        projection
            .verify_against(
                restart,
                seals,
                restart_receipt,
                checkpoint,
                admission,
                currentness,
                eligibility,
                projection.projected_at_cycle(),
            )
            .map_err(HistoricalFirewallReplayError::ProjectionRejected)?;

        if !projection.membership_derived_from_protected_census()
            || projection.observation_cycle_used_as_membership()
            || projection.ledger_constructed()
            || projection.historical_replay_authorized()
            || projection.writable_hydration_authorized()
            || projection.activation_authorized()
        {
            return Err(HistoricalFirewallReplayError::UnexpectedProjectionAuthority);
        }
        if replayed_at_cycle < projection.projected_at_cycle() {
            return Err(HistoricalFirewallReplayError::ReplayPredatesProjection {
                replayed_at_cycle,
                projected_at_cycle: projection.projected_at_cycle(),
            });
        }
        if replayed_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(HistoricalFirewallReplayError::CurrentnessExpired {
                replayed_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }
        if restart.base.mutations.len() > MAX_ISOLATED_REPLAY_MUTATIONS {
            return Err(HistoricalFirewallReplayError::TooManyMutations {
                actual: restart.base.mutations.len(),
                maximum: MAX_ISOLATED_REPLAY_MUTATIONS,
            });
        }
        if projection.records().len() != restart.base.mutations.len() {
            return Err(HistoricalFirewallReplayError::ProjectionMutationCountMismatch);
        }

        let full_ledger = rebuild_full_ledger(&restart.base)?;
        let source_mutations = mutation_by_source_revision(&restart.base.mutations)?;
        let projection_by_mutation = projection_record_by_mutation(projection)?;
        let mut revision_history = BeliefRevisionHistory::new();
        let placeholder_policy = BeliefRevisionPolicy::new(1.0, 0, false, 0, 1.0)
            .map_err(|_| HistoricalFirewallReplayError::PlaceholderPolicyRejected)?;
        let mut placeholder_count = 0usize;

        // Rebuild receipt identity space in exact source order. Only receipts that
        // actually authorized persisted mutations are semantically replayed. Other
        // receipts are private placeholders used solely to preserve ID allocation.
        for wire in &restart.base.revisions {
            if let Some(mutation) = source_mutations.get(&wire.id.0) {
                let projection_record = projection_by_mutation
                    .get(&mutation.id.0)
                    .copied()
                    .ok_or(HistoricalFirewallReplayError::ProjectionMissingMutation(
                        mutation.id.0,
                    ))?;
                let historical_ledger = rebuild_historical_ledger(&restart.base, projection_record)?;
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
                placeholder_count = placeholder_count
                    .checked_add(1)
                    .ok_or(HistoricalFirewallReplayError::LengthOverflow)?;
            }
        }

        if revision_history.len() != restart.base.revisions.len() {
            return Err(HistoricalFirewallReplayError::RevisionIdentitySpaceMismatch);
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
                .map_err(|_| HistoricalFirewallReplayError::SupportBaselineRejected(state.claim_id))?;
        }

        let mut firewall = BeliefMutationFirewall::new();
        let mut mutation_replays = Vec::with_capacity(restart.base.mutations.len());
        for mutation in &restart.base.mutations {
            let projection_record = projection_by_mutation
                .get(&mutation.id.0)
                .copied()
                .ok_or(HistoricalFirewallReplayError::ProjectionMissingMutation(
                    mutation.id.0,
                ))?;
            if projection_record.source_revision_receipt_id() != mutation.source_revision_receipt_id
                || projection_record.claim_id() != mutation.claim_id
            {
                return Err(HistoricalFirewallReplayError::ProjectionMutationBindingMismatch(
                    mutation.id.0,
                ));
            }

            let historical_ledger = rebuild_historical_ledger(&restart.base, projection_record)?;
            let revision = revision_history
                .get(mutation.source_revision_receipt_id)
                .ok_or(HistoricalFirewallReplayError::SourceRevisionMissing(
                    mutation.source_revision_receipt_id.0,
                ))?;
            let state = store
                .state(mutation.claim_id)
                .ok_or(HistoricalFirewallReplayError::SupportStateMissing(mutation.claim_id))?;
            if state.revision() != mutation.state_revision_before
                || state.support() != mutation.support_before
            {
                return Err(HistoricalFirewallReplayError::PersistedPreStateMismatch(
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
            .map_err(|_| HistoricalFirewallReplayError::AuthorizationRebuildRejected(
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
                .map_err(|error| HistoricalFirewallReplayError::FirewallReplayRejected {
                    mutation_id: mutation.id.0,
                    detail: format!("{error:?}"),
                })?;
            if !matches!(outcome, BeliefMutationOutcome::Applied(_)) {
                return Err(HistoricalFirewallReplayError::FirewallReplayWasNotNew(
                    mutation.id.0,
                ));
            }
            compare_mutation(outcome.receipt(), mutation)?;

            let mut replay = HistoricalFirewallMutationReplayV1 {
                mutation_id: mutation.id.0,
                source_revision_receipt_id: mutation.source_revision_receipt_id.0,
                claim_id: mutation.claim_id,
                projection_record_digest: projection_record.record_digest().as_bytes(),
                historical_evidence_count: projection_record.evidence().len(),
                excluded_final_evidence_count: projection_record.excluded_final_evidence_ids().len(),
                excluded_backdated_final_evidence_count: projection_record
                    .excluded_with_observation_not_after_seal_count(),
                support_before_bits: mutation.support_before.get().to_bits(),
                support_after_bits: mutation.support_after.get().to_bits(),
                state_revision_before: mutation.state_revision_before,
                state_revision_after: mutation.state_revision_after,
                exact_receipt_reproduced: true,
                digest: HistoricalFirewallReplayDigestV1([0; 32]),
            };
            replay.digest = digest_mutation_replay(&replay)?;
            mutation_replays.push(replay);
        }

        compare_final_support_states(&store, &restart.base.support_states)?;
        let consumed_equivalent = store.consumed_authorization_count() as u64
            == restart.base.consumed_authorization_count;
        if !consumed_equivalent {
            return Err(HistoricalFirewallReplayError::ConsumedAuthorizationCountMismatch {
                replayed: store.consumed_authorization_count(),
                persisted: restart.base.consumed_authorization_count,
            });
        }

        let mut report = Self {
            version: HistoricalFirewallReplayVersion::V1,
            restart_capture_cycle: restart.base.captured_at_cycle,
            replayed_at_cycle,
            projection_digest: projection.projection_digest().as_bytes(),
            eligibility_digest: eligibility.receipt_digest().as_bytes(),
            mutation_count: restart.base.mutations.len(),
            firewall_invocation_count: restart.base.mutations.len(),
            non_mutation_revision_placeholder_count: placeholder_count,
            mutation_replays,
            persisted_mutation_receipts_exactly_reproduced: true,
            final_support_state_equivalent: true,
            consumed_authorization_count_equivalent: true,
            full_nonmutation_revision_history_replayed: false,
            isolated_firewall_replay_performed: !restart.base.mutations.is_empty(),
            writable_state_export_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            report_digest: HistoricalFirewallReplayDigestV1([0; 32]),
        };
        report.report_digest = digest_report(&report)?;
        Ok(report)
    }

    pub fn version(&self) -> HistoricalFirewallReplayVersion {
        self.version
    }

    pub fn replayed_at_cycle(&self) -> u64 {
        self.replayed_at_cycle
    }

    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    pub fn firewall_invocation_count(&self) -> usize {
        self.firewall_invocation_count
    }

    pub fn non_mutation_revision_placeholder_count(&self) -> usize {
        self.non_mutation_revision_placeholder_count
    }

    pub fn mutation_replays(&self) -> &[HistoricalFirewallMutationReplayV1] {
        &self.mutation_replays
    }

    pub fn persisted_mutation_receipts_exactly_reproduced(&self) -> bool {
        self.persisted_mutation_receipts_exactly_reproduced
    }

    pub fn final_support_state_equivalent(&self) -> bool {
        self.final_support_state_equivalent
    }

    pub fn consumed_authorization_count_equivalent(&self) -> bool {
        self.consumed_authorization_count_equivalent
    }

    pub fn full_nonmutation_revision_history_replayed(&self) -> bool {
        self.full_nonmutation_revision_history_replayed
    }

    pub fn isolated_firewall_replay_performed(&self) -> bool {
        self.isolated_firewall_replay_performed
    }

    pub fn writable_state_export_authorized(&self) -> bool {
        self.writable_state_export_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn report_digest(&self) -> HistoricalFirewallReplayDigestV1 {
        self.report_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projection: &HistoricalEvidenceProjectionV1,
        replayed_at_cycle: u64,
    ) -> Result<(), HistoricalFirewallReplayError> {
        let live = Self::verify_isolated(
            restart,
            seals,
            restart_receipt,
            checkpoint,
            admission,
            currentness,
            eligibility,
            projection,
            replayed_at_cycle,
        )?;
        if &live != self {
            return Err(HistoricalFirewallReplayError::ReportMismatch);
        }
        if digest_report(self)? != self.report_digest {
            return Err(HistoricalFirewallReplayError::ReportDigestMismatch);
        }
        Ok(())
    }
}

fn mutation_by_source_revision(
    mutations: &[WireMutationV1],
) -> Result<HashMap<u64, &WireMutationV1>, HistoricalFirewallReplayError> {
    let mut out = HashMap::with_capacity(mutations.len());
    for mutation in mutations {
        if out
            .insert(mutation.source_revision_receipt_id.0, mutation)
            .is_some()
        {
            return Err(HistoricalFirewallReplayError::DuplicateSourceRevision(
                mutation.source_revision_receipt_id.0,
            ));
        }
    }
    Ok(out)
}

fn projection_record_by_mutation(
    projection: &HistoricalEvidenceProjectionV1,
) -> Result<HashMap<u64, &HistoricalEvidenceProjectionRecordV1>, HistoricalFirewallReplayError> {
    let mut out = HashMap::with_capacity(projection.records().len());
    for record in projection.records() {
        if out.insert(record.mutation_id().0, record).is_some() {
            return Err(HistoricalFirewallReplayError::DuplicateProjectionMutation(
                record.mutation_id().0,
            ));
        }
    }
    Ok(out)
}

fn rebuild_full_ledger(
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<EpistemicLedger, HistoricalFirewallReplayError> {
    rebuild_ledger_prefix(base, None, None, None)
}

fn rebuild_historical_ledger(
    base: &EpistemicRestartWireSnapshotV1,
    projection: &HistoricalEvidenceProjectionRecordV1,
) -> Result<EpistemicLedger, HistoricalFirewallReplayError> {
    let max_evidence_id = projection
        .evidence()
        .iter()
        .map(|record| record.evidence_id.0)
        .max();
    let max_claim_id = Some(projection.claim_id().0);
    let max_provenance_id = match max_evidence_id {
        Some(max_id) => base
            .evidence
            .iter()
            .take_while(|record| record.id.0 <= max_id)
            .map(|record| record.provenance_id.0)
            .max(),
        None => None,
    };
    let mut ledger = rebuild_ledger_prefix(
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
        return Err(HistoricalFirewallReplayError::HistoricalClaimCensusMismatch(
            projection.mutation_id().0,
        ));
    }
    for expected in projection.evidence() {
        let live = ledger
            .evidence(expected.evidence_id)
            .ok_or(HistoricalFirewallReplayError::HistoricalEvidenceMissing(
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
            return Err(HistoricalFirewallReplayError::HistoricalEvidenceMismatch(
                expected.evidence_id,
            ));
        }
    }

    // Keep the local variable mutable only during construction. No handle escapes.
    let _ = &mut ledger;
    Ok(ledger)
}

fn rebuild_ledger_prefix(
    base: &EpistemicRestartWireSnapshotV1,
    max_provenance_id: Option<u64>,
    max_claim_id: Option<u64>,
    max_evidence_id: Option<u64>,
) -> Result<EpistemicLedger, HistoricalFirewallReplayError> {
    let mut ledger = EpistemicLedger::new();
    for record in &base.provenance {
        if let Some(maximum) = max_provenance_id {
            if record.id.0 > maximum {
                break;
            }
        }
        let id = ledger
            .add_provenance(
                record.source_label.clone(),
                record.source_uri.clone(),
                record.content_hash.clone(),
                record.recorded_at_cycle,
                record.parent_ids.clone(),
            )
            .map_err(|_| HistoricalFirewallReplayError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(HistoricalFirewallReplayError::LedgerIdMismatch);
        }
    }
    for record in &base.claims {
        if let Some(maximum) = max_claim_id {
            if record.id.0 > maximum {
                break;
            }
        }
        let id = ledger.add_claim(
            record.statement.clone(),
            record.kind,
            record.domain.clone(),
            record.scope.clone(),
            record.created_at_cycle,
        );
        if id != record.id {
            return Err(HistoricalFirewallReplayError::LedgerIdMismatch);
        }
    }
    for record in &base.evidence {
        if let Some(maximum) = max_evidence_id {
            if record.id.0 > maximum {
                break;
            }
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
            .map_err(|_| HistoricalFirewallReplayError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(HistoricalFirewallReplayError::LedgerIdMismatch);
        }
    }
    Ok(ledger)
}

fn record_actual_revision(
    history: &mut BeliefRevisionHistory,
    ledger: &EpistemicLedger,
    restart: &EpistemicRestartWireSnapshotV2,
    wire: &WireRevisionReceiptV1,
) -> Result<(), HistoricalFirewallReplayError> {
    let schema = restart
        .revision_schemas
        .records
        .iter()
        .find(|record| record.receipt_id == wire.id)
        .ok_or(HistoricalFirewallReplayError::RevisionSchemaMissing(wire.id.0))?;
    let policy = schema
        .policy_schema
        .build_policy()
        .map_err(|_| HistoricalFirewallReplayError::PolicyRebuildRejected(wire.id.0))?;
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
        .map_err(|_| HistoricalFirewallReplayError::RevisionReplayRejected(wire.id.0))?;
    if id != wire.id {
        return Err(HistoricalFirewallReplayError::RevisionIdMismatch {
            expected: wire.id.0,
            actual: id.0,
        });
    }
    let receipt = history
        .get(id)
        .ok_or(HistoricalFirewallReplayError::SourceRevisionMissing(id.0))?;
    compare_revision(receipt, wire, &schema.decision_snapshot)
}

fn record_placeholder_revision(
    history: &mut BeliefRevisionHistory,
    ledger: &EpistemicLedger,
    wire: &WireRevisionReceiptV1,
    policy: &BeliefRevisionPolicy,
) -> Result<(), HistoricalFirewallReplayError> {
    let proposal = EpistemicRevisionProposal::new(
        wire.claim_id,
        0.0,
        Vec::new(),
        "restart-verification-id-placeholder",
    )
    .map_err(|_| HistoricalFirewallReplayError::PlaceholderProposalRejected(wire.id.0))?;
    let id = history
        .evaluate_and_record(
            ledger,
            &proposal,
            policy,
            None,
            None,
            wire.evaluated_at_cycle,
        )
        .map_err(|_| HistoricalFirewallReplayError::PlaceholderRevisionRejected(wire.id.0))?;
    if id != wire.id {
        return Err(HistoricalFirewallReplayError::RevisionIdMismatch {
            expected: wire.id.0,
            actual: id.0,
        });
    }
    Ok(())
}

fn rebuild_proposal(
    wire: &WireRevisionReceiptV1,
) -> Result<EpistemicRevisionProposal, HistoricalFirewallReplayError> {
    let mut basis_ids = wire
        .basis
        .iter()
        .map(|basis| basis.requested_id)
        .collect::<Vec<_>>();
    for duplicate in &wire.duplicate_basis_evidence_ids {
        if !basis_ids.contains(duplicate) {
            return Err(HistoricalFirewallReplayError::DuplicateBasisDiagnosticOrphan {
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
    .map_err(|_| HistoricalFirewallReplayError::ProposalRebuildRejected(wire.id.0))
}

fn rebuild_calibration(
    wire: &WireRevisionReceiptV1,
) -> Result<Option<CalibrationSnapshot>, HistoricalFirewallReplayError> {
    wire.calibration
        .map(|(sample_count, ece)| {
            CalibrationSnapshot::new(sample_count, ece)
                .map_err(|_| HistoricalFirewallReplayError::CalibrationRebuildRejected(wire.id.0))
        })
        .transpose()
}

fn rebuild_uncertainty(
    ledger: &EpistemicLedger,
    wire: &WireRevisionReceiptV1,
) -> Result<Option<ClaimUncertaintyAssessment>, HistoricalFirewallReplayError> {
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
                .map_err(|_| HistoricalFirewallReplayError::UncertaintyRebuildRejected(wire.id.0))?;
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
    .map_err(|_| HistoricalFirewallReplayError::UncertaintyRebuildRejected(wire.id.0))
}

fn compare_revision(
    rebuilt: &BeliefRevisionReceipt,
    wire: &WireRevisionReceiptV1,
    expected_decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), HistoricalFirewallReplayError> {
    if rebuilt.id() != wire.id
        || rebuilt.claim_id() != wire.claim_id
        || rebuilt.proposed_delta().to_bits() != wire.proposed_delta.to_bits()
        || rebuilt.rationale() != wire.rationale
        || rebuilt.evaluated_at_cycle() != wire.evaluated_at_cycle
        || rebuilt.duplicate_basis_evidence_ids() != wire.duplicate_basis_evidence_ids
        || rebuilt.basis().len() != wire.basis.len()
        || rebuilt.eligible() != wire.decision_eligible
        || rebuilt.decision().declared_provenance_root_count() as u64
            != wire.declared_provenance_root_count
        || format!("{:?}", rebuilt.policy()) != wire.policy_debug
        || format!("{:?}", rebuilt.decision()) != wire.decision_debug
        || BeliefRevisionDecisionSnapshotV1::capture(rebuilt.decision()) != *expected_decision
    {
        return Err(HistoricalFirewallReplayError::RevisionSemanticMismatch(wire.id.0));
    }
    for (left, right) in rebuilt.basis().iter().zip(&wire.basis) {
        if left.requested_id != right.requested_id || left.snapshot != right.snapshot {
            return Err(HistoricalFirewallReplayError::RevisionSemanticMismatch(wire.id.0));
        }
    }
    Ok(())
}

fn compare_mutation(
    receipt: &super::belief_mutation_firewall::BeliefMutationReceipt,
    wire: &WireMutationV1,
) -> Result<(), HistoricalFirewallReplayError> {
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
        return Err(HistoricalFirewallReplayError::MutationSemanticMismatch(wire.id.0));
    }
    Ok(())
}

fn compare_final_support_states(
    store: &EpistemicSupportStore,
    expected: &[WireSupportStateV1],
) -> Result<(), HistoricalFirewallReplayError> {
    if store.len() != expected.len() {
        return Err(HistoricalFirewallReplayError::SupportStateCountMismatch);
    }
    for wire in expected {
        let state = store
            .state(wire.claim_id)
            .ok_or(HistoricalFirewallReplayError::SupportStateMissing(wire.claim_id))?;
        if state.support() != wire.current_support
            || state.revision() != wire.revision
            || state.initialized_at_cycle() != wire.initialized_at_cycle
            || state.last_updated_cycle() != wire.last_updated_cycle
            || state.last_mutation_id() != wire.last_mutation_id
        {
            return Err(HistoricalFirewallReplayError::SupportStateMismatch(wire.claim_id));
        }
    }
    Ok(())
}

fn digest_mutation_replay(
    replay: &HistoricalFirewallMutationReplayV1,
) -> Result<HistoricalFirewallReplayDigestV1, HistoricalFirewallReplayError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-historical-firewall-mutation-replay-v1");
    hasher.update(&replay.mutation_id.to_le_bytes());
    hasher.update(&replay.source_revision_receipt_id.to_le_bytes());
    hasher.update(&replay.claim_id.0.to_le_bytes());
    hasher.update(&replay.projection_record_digest);
    hash_usize(&mut hasher, replay.historical_evidence_count)?;
    hash_usize(&mut hasher, replay.excluded_final_evidence_count)?;
    hash_usize(
        &mut hasher,
        replay.excluded_backdated_final_evidence_count,
    )?;
    hasher.update(&replay.support_before_bits.to_le_bytes());
    hasher.update(&replay.support_after_bits.to_le_bytes());
    hasher.update(&replay.state_revision_before.to_le_bytes());
    hasher.update(&replay.state_revision_after.to_le_bytes());
    hasher.update(&[u8::from(replay.exact_receipt_reproduced)]);
    Ok(HistoricalFirewallReplayDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_report(
    report: &HistoricalFirewallReplayReportV1,
) -> Result<HistoricalFirewallReplayDigestV1, HistoricalFirewallReplayError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-historical-firewall-replay-v1");
    hasher.update(&[1]);
    hasher.update(&report.restart_capture_cycle.to_le_bytes());
    hasher.update(&report.replayed_at_cycle.to_le_bytes());
    hasher.update(&report.projection_digest);
    hasher.update(&report.eligibility_digest);
    hash_usize(&mut hasher, report.mutation_count)?;
    hash_usize(&mut hasher, report.firewall_invocation_count)?;
    hash_usize(
        &mut hasher,
        report.non_mutation_revision_placeholder_count,
    )?;
    hash_usize(&mut hasher, report.mutation_replays.len())?;
    for mutation in &report.mutation_replays {
        hasher.update(&mutation.digest.as_bytes());
    }
    hasher.update(&[u8::from(
        report.persisted_mutation_receipts_exactly_reproduced,
    )]);
    hasher.update(&[u8::from(report.final_support_state_equivalent)]);
    hasher.update(&[u8::from(
        report.consumed_authorization_count_equivalent,
    )]);
    hasher.update(&[u8::from(
        report.full_nonmutation_revision_history_replayed,
    )]);
    hasher.update(&[u8::from(report.isolated_firewall_replay_performed)]);
    hasher.update(&[u8::from(report.writable_state_export_authorized)]);
    hasher.update(&[u8::from(report.writable_hydration_authorized)]);
    hasher.update(&[u8::from(report.activation_authorized)]);
    Ok(HistoricalFirewallReplayDigestV1(*hasher.finalize().as_bytes()))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), HistoricalFirewallReplayError> {
    let value = u64::try_from(value).map_err(|_| HistoricalFirewallReplayError::LengthOverflow)?;
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
pub enum HistoricalFirewallReplayError {
    ProjectionRejected(HistoricalEvidenceProjectionError),
    UnexpectedProjectionAuthority,
    ReplayPredatesProjection { replayed_at_cycle: u64, projected_at_cycle: u64 },
    CurrentnessExpired { replayed_at_cycle: u64, expires_at_cycle: u64 },
    TooManyMutations { actual: usize, maximum: usize },
    ProjectionMutationCountMismatch,
    DuplicateSourceRevision(u64),
    DuplicateProjectionMutation(u64),
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
    ReportMismatch,
    ReportDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for HistoricalFirewallReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "historical firewall replay verification rejected: {self:?}")
    }
}

impl Error for HistoricalFirewallReplayError {}
