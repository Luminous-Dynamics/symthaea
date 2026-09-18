// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isolated writable hydration for admitted epistemic restart state.
//!
//! This module is intentionally not an activation path. It consumes the sealed
//! read-only quarantine handle, rebuilds a real belief-revision history and a real
//! epistemic-support store inside a private sandbox, then immediately re-captures
//! both persistence capsules and checks them against the source wire state.
//!
//! V1 fails closed when the retained restart payload is insufficient to reproduce
//! an original mutation through the existing firewall. In particular, the payload
//! records evidence observation cycles but not ledger insertion cycles, so a claim
//! with evidence observed after an applied revision cannot be historically replayed
//! without additional chronology evidence.

use super::belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationAuthorizationDecision, BeliefMutationFirewall,
    BeliefMutationOutcome, EpistemicSupportStore,
};
use super::belief_mutation_persistence::BeliefMutationPersistenceCapsuleV1;
use super::belief_revision_gate::{
    BeliefRevisionGate, CalibrationSnapshot, EpistemicRevisionProposal,
};
use super::belief_revision_persistence::BeliefRevisionHistoryCapsuleV1;
use super::belief_revision_receipt::BeliefRevisionHistory;
use super::belief_revision_snapshot::BeliefRevisionDecisionSnapshotV1;
use super::claim_evidence::{ClaimId, EpistemicLedger};
use super::epistemic_restart_quarantine_facade::{
    ReadOnlyEpistemicRestartQuarantineV2, ReadOnlyRestartQuarantineDigestV1,
    RestartTrustContextDigestV1,
};
use super::epistemic_restart_trust_checkpoint::{
    RestartTrustContextCheckpointDigestV1, VerifiedRestartTrustContextCheckpointV1,
};
use super::epistemic_restart_wire::{
    EpistemicRestartWireSnapshotV1, WireMutationV1, WireRevisionReceiptV1, WireSupportStateV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_restart_wire_v2_validation::EpistemicRestartWireV2Validator;
use super::epistemic_vector::{ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolatedRestartHydrationVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsolatedRestartHydrationDigestV1([u8; 32]);

impl IsolatedRestartHydrationDigestV1 {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydratedSupportSummaryV1 {
    pub claim_id: ClaimId,
    pub support: f32,
    pub revision: u64,
    pub initialized_at_cycle: u64,
    pub last_updated_cycle: u64,
}

/// Sealed sandbox containing real writable EKM state that cannot be exported or
/// activated through this API.
#[derive(Debug)]
pub struct IsolatedEpistemicRestartHydrationV1 {
    version: IsolatedRestartHydrationVersion,
    quarantine: ReadOnlyEpistemicRestartQuarantineV2,
    source_snapshot: EpistemicRestartWireSnapshotV2,
    ledger: EpistemicLedger,
    support_store: EpistemicSupportStore,
    revision_history: BeliefRevisionHistory,
    protected_checkpoint: VerifiedRestartTrustContextCheckpointV1,
    hydrated_at_cycle: u64,
    hydration_digest: IsolatedRestartHydrationDigestV1,
    trusted_state_mutated: bool,
    writable_state_export_authorized: bool,
    activation_authorized: bool,
}

impl IsolatedEpistemicRestartHydrationV1 {
    /// Consume a read-only quarantine and hydrate writable EKM state inside this
    /// sealed object. No writable handle is returned to the caller.
    pub fn hydrate(
        snapshot: &EpistemicRestartWireSnapshotV2,
        quarantine: ReadOnlyEpistemicRestartQuarantineV2,
        protected_checkpoint: VerifiedRestartTrustContextCheckpointV1,
        hydrated_at_cycle: u64,
    ) -> Result<Self, IsolatedRestartHydrationError> {
        quarantine
            .verify()
            .map_err(|error| IsolatedRestartHydrationError::QuarantineRejected {
                detail: format!("{error:?}"),
            })?;
        protected_checkpoint
            .verify_internal()
            .map_err(|error| IsolatedRestartHydrationError::CheckpointRejected {
                detail: format!("{error:?}"),
            })?;

        EpistemicRestartWireV2Validator::validate(snapshot)
            .map_err(|error| IsolatedRestartHydrationError::SnapshotRejected {
                detail: format!("{error:?}"),
            })?;

        if snapshot.outer_checksum != quarantine.source_outer_checksum()
            || snapshot.claimed_v2_digest != quarantine.source_v2_digest()
            || snapshot.base.captured_at_cycle != quarantine.captured_at_cycle()
        {
            return Err(IsolatedRestartHydrationError::SnapshotQuarantineMismatch);
        }
        if protected_checkpoint.statement().context_digest() != quarantine.trust_context_digest() {
            return Err(IsolatedRestartHydrationError::CheckpointContextMismatch);
        }
        if hydrated_at_cycle < protected_checkpoint.verified_at_cycle() {
            return Err(IsolatedRestartHydrationError::HydrationPredatesCheckpointVerification {
                hydrated_at_cycle,
                verified_at_cycle: protected_checkpoint.verified_at_cycle(),
            });
        }
        if hydrated_at_cycle >= protected_checkpoint.statement().expires_at_cycle() {
            return Err(IsolatedRestartHydrationError::CheckpointExpiredAtHydration {
                hydrated_at_cycle,
                expires_at_cycle: protected_checkpoint.statement().expires_at_cycle(),
            });
        }
        if protected_checkpoint.trusted_state_mutated()
            || protected_checkpoint.quarantine_construction_authorized()
            || protected_checkpoint.writable_hydration_authorized()
            || protected_checkpoint.activation_authorized()
        {
            return Err(IsolatedRestartHydrationError::UnexpectedCheckpointAuthority);
        }

        let ledger = rebuild_ledger(&snapshot.base)?;
        let revision_history = rebuild_revision_history(&ledger, snapshot)?;
        let support_store = rebuild_support_store(&ledger, &revision_history, &snapshot.base)?;
        verify_reexport(&support_store, &revision_history, &snapshot.base)?;

        let mut out = Self {
            version: IsolatedRestartHydrationVersion::V1,
            quarantine,
            source_snapshot: snapshot.clone(),
            ledger,
            support_store,
            revision_history,
            protected_checkpoint,
            hydrated_at_cycle,
            hydration_digest: IsolatedRestartHydrationDigestV1([0; 32]),
            trusted_state_mutated: false,
            writable_state_export_authorized: false,
            activation_authorized: false,
        };
        out.hydration_digest = digest_hydration(&out);
        out.verify()?;
        Ok(out)
    }

    pub fn version(&self) -> IsolatedRestartHydrationVersion {
        self.version
    }

    pub fn hydrated_at_cycle(&self) -> u64 {
        self.hydrated_at_cycle
    }

    pub fn claim_count(&self) -> usize {
        self.ledger.claim_count()
    }

    pub fn support_state_count(&self) -> usize {
        self.support_store.len()
    }

    pub fn mutation_count(&self) -> usize {
        self.support_store.history().len()
    }

    pub fn revision_count(&self) -> usize {
        self.revision_history.len()
    }

    pub fn support_summary(&self, claim_id: ClaimId) -> Option<HydratedSupportSummaryV1> {
        let state = self.support_store.state(claim_id)?;
        Some(HydratedSupportSummaryV1 {
            claim_id,
            support: state.support().get(),
            revision: state.revision(),
            initialized_at_cycle: state.initialized_at_cycle(),
            last_updated_cycle: state.last_updated_cycle(),
        })
    }

    pub fn quarantine_digest(&self) -> ReadOnlyRestartQuarantineDigestV1 {
        self.quarantine.quarantine_digest()
    }

    pub fn trust_context_digest(&self) -> RestartTrustContextDigestV1 {
        self.quarantine.trust_context_digest()
    }

    pub fn checkpoint_statement_digest(&self) -> RestartTrustContextCheckpointDigestV1 {
        self.protected_checkpoint.statement_digest()
    }

    pub fn hydration_digest(&self) -> IsolatedRestartHydrationDigestV1 {
        self.hydration_digest
    }

    pub fn isolated_hydration_performed(&self) -> bool {
        true
    }

    pub fn writable_state_export_authorized(&self) -> bool {
        self.writable_state_export_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    /// Re-capture the hydrated support/history state and compare it with the exact
    /// source wire snapshot. This never activates or exports the writable objects.
    pub fn verify(&self) -> Result<(), IsolatedRestartHydrationError> {
        self.quarantine
            .verify()
            .map_err(|error| IsolatedRestartHydrationError::QuarantineRejected {
                detail: format!("{error:?}"),
            })?;
        self.protected_checkpoint
            .verify_internal()
            .map_err(|error| IsolatedRestartHydrationError::CheckpointRejected {
                detail: format!("{error:?}"),
            })?;
        if self.protected_checkpoint.statement().context_digest()
            != self.quarantine.trust_context_digest()
        {
            return Err(IsolatedRestartHydrationError::CheckpointContextMismatch);
        }
        if self.trusted_state_mutated
            || self.writable_state_export_authorized
            || self.activation_authorized
        {
            return Err(IsolatedRestartHydrationError::UnexpectedHydrationAuthority);
        }
        verify_reexport(
            &self.support_store,
            &self.revision_history,
            &self.source_snapshot.base,
        )?;
        if digest_hydration(self) != self.hydration_digest {
            return Err(IsolatedRestartHydrationError::HydrationDigestMismatch);
        }
        Ok(())
    }
}

fn rebuild_ledger(
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<EpistemicLedger, IsolatedRestartHydrationError> {
    let mut ledger = EpistemicLedger::new();
    for record in &base.provenance {
        let id = ledger
            .add_provenance(
                record.source_label.clone(),
                record.source_uri.clone(),
                record.content_hash.clone(),
                record.recorded_at_cycle,
                record.parent_ids.clone(),
            )
            .map_err(|_| IsolatedRestartHydrationError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(IsolatedRestartHydrationError::LedgerIdMismatch);
        }
    }
    for record in &base.claims {
        let id = ledger.add_claim(
            record.statement.clone(),
            record.kind,
            record.domain.clone(),
            record.scope.clone(),
            record.created_at_cycle,
        );
        if id != record.id {
            return Err(IsolatedRestartHydrationError::LedgerIdMismatch);
        }
    }
    for record in &base.evidence {
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
            .map_err(|_| IsolatedRestartHydrationError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(IsolatedRestartHydrationError::LedgerIdMismatch);
        }
    }
    Ok(ledger)
}

fn rebuild_revision_history(
    ledger: &EpistemicLedger,
    snapshot: &EpistemicRestartWireSnapshotV2,
) -> Result<BeliefRevisionHistory, IsolatedRestartHydrationError> {
    let mut history = BeliefRevisionHistory::new();

    for (wire, schema) in snapshot
        .base
        .revisions
        .iter()
        .zip(&snapshot.revision_schemas.records)
    {
        if wire.id != schema.receipt_id {
            return Err(IsolatedRestartHydrationError::RevisionSchemaMismatch(wire.id.0));
        }
        let policy = schema
            .policy_schema
            .build_policy()
            .map_err(|_| IsolatedRestartHydrationError::PolicyRebuildRejected(wire.id.0))?;

        let mut basis_ids = wire
            .basis
            .iter()
            .map(|basis| basis.requested_id)
            .collect::<Vec<_>>();
        for duplicate in &wire.duplicate_basis_evidence_ids {
            if !basis_ids.contains(duplicate) {
                return Err(IsolatedRestartHydrationError::DuplicateBasisDiagnosticOrphan {
                    receipt_id: wire.id.0,
                    evidence_id: duplicate.0,
                });
            }
            basis_ids.push(*duplicate);
        }

        let proposal = EpistemicRevisionProposal::new(
            wire.claim_id,
            wire.proposed_delta,
            basis_ids,
            wire.rationale.clone(),
        )
        .map_err(|_| IsolatedRestartHydrationError::ProposalRebuildRejected(wire.id.0))?;
        let calibration = rebuild_calibration(wire)?;
        let uncertainty = rebuild_uncertainty(ledger, wire)?;

        let independent_decision = BeliefRevisionGate::evaluate(
            ledger,
            &proposal,
            &policy,
            calibration,
            uncertainty.as_ref(),
        );
        if BeliefRevisionDecisionSnapshotV1::capture(&independent_decision)
            != schema.decision_snapshot
        {
            return Err(IsolatedRestartHydrationError::DecisionReplayMismatch(wire.id.0));
        }

        let id = history
            .evaluate_and_record(
                ledger,
                &proposal,
                &policy,
                calibration,
                uncertainty.as_ref(),
                wire.evaluated_at_cycle,
            )
            .map_err(|_| IsolatedRestartHydrationError::RevisionHistoryRebuildRejected(wire.id.0))?;
        if id != wire.id {
            return Err(IsolatedRestartHydrationError::RevisionIdMismatch {
                expected: wire.id.0,
                actual: id.0,
            });
        }
        let rebuilt = history
            .get(id)
            .ok_or(IsolatedRestartHydrationError::RevisionMissingAfterRebuild(id.0))?;
        compare_revision(rebuilt, wire, &schema.decision_snapshot)?;
    }

    if snapshot.base.revision_next_receipt_id.0 != history.len() as u64 + 1 {
        return Err(IsolatedRestartHydrationError::NextRevisionIdMismatch);
    }
    Ok(history)
}

fn rebuild_calibration(
    wire: &WireRevisionReceiptV1,
) -> Result<Option<CalibrationSnapshot>, IsolatedRestartHydrationError> {
    wire.calibration
        .map(|(sample_count, ece)| {
            CalibrationSnapshot::new(sample_count, ece)
                .map_err(|_| IsolatedRestartHydrationError::CalibrationRebuildRejected(wire.id.0))
        })
        .transpose()
}

fn rebuild_uncertainty(
    ledger: &EpistemicLedger,
    wire: &WireRevisionReceiptV1,
) -> Result<Option<ClaimUncertaintyAssessment>, IsolatedRestartHydrationError> {
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
                .map_err(|_| IsolatedRestartHydrationError::UncertaintyRebuildRejected(wire.id.0))?;
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
    .map_err(|_| IsolatedRestartHydrationError::UncertaintyRebuildRejected(wire.id.0))
}

fn compare_revision(
    rebuilt: &super::belief_revision_receipt::BeliefRevisionReceipt,
    wire: &WireRevisionReceiptV1,
    expected_decision: &BeliefRevisionDecisionSnapshotV1,
) -> Result<(), IsolatedRestartHydrationError> {
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
        return Err(IsolatedRestartHydrationError::RevisionSemanticMismatch(wire.id.0));
    }
    for (rebuilt_basis, wire_basis) in rebuilt.basis().iter().zip(&wire.basis) {
        if rebuilt_basis.requested_id != wire_basis.requested_id
            || rebuilt_basis.snapshot != wire_basis.snapshot
        {
            return Err(IsolatedRestartHydrationError::RevisionSemanticMismatch(wire.id.0));
        }
    }
    Ok(())
}

fn rebuild_support_store(
    ledger: &EpistemicLedger,
    history: &BeliefRevisionHistory,
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<EpistemicSupportStore, IsolatedRestartHydrationError> {
    let mut store = EpistemicSupportStore::new();
    for state in &base.support_states {
        store
            .register_claim(
                ledger,
                state.claim_id,
                state.baseline_support,
                state.initialized_at_cycle,
            )
            .map_err(|_| IsolatedRestartHydrationError::SupportBaselineRejected(state.claim_id))?;
    }

    let mut firewall = BeliefMutationFirewall::new();
    for mutation in &base.mutations {
        let revision = history
            .get(mutation.source_revision_receipt_id)
            .ok_or(IsolatedRestartHydrationError::MutationRevisionMissing(mutation.id.0))?;

        // The live firewall rejects a mutation when any evidence currently linked
        // to its claim postdates the frozen revision decision. V1 cannot prove when
        // such later evidence was inserted, so exact replay fails closed here.
        if ledger
            .evidence_for_claim(mutation.claim_id)
            .into_iter()
            .any(|record| record.observed_at_cycle > revision.evaluated_at_cycle())
        {
            return Err(IsolatedRestartHydrationError::HistoricalLedgerStateInsufficient {
                mutation_id: mutation.id.0,
                claim_id: mutation.claim_id,
            });
        }

        let state = store
            .state(mutation.claim_id)
            .ok_or(IsolatedRestartHydrationError::SupportStateMissing(mutation.claim_id))?;
        let authorization = BeliefMutationAuthorization::new(
            mutation.authorization_id.clone(),
            mutation.authority_label.clone(),
            BeliefMutationAuthorizationDecision::Approved,
            mutation.authorized_at_cycle,
            revision,
            state,
        )
        .map_err(|_| IsolatedRestartHydrationError::AuthorizationRebuildRejected(mutation.id.0))?;

        let outcome = firewall
            .apply(
                ledger,
                &mut store,
                revision,
                &authorization,
                mutation.applied_at_cycle,
            )
            .map_err(|_| IsolatedRestartHydrationError::MutationReplayRejected(mutation.id.0))?;
        if !matches!(outcome, BeliefMutationOutcome::Applied(_)) {
            return Err(IsolatedRestartHydrationError::MutationReplayWasNotNew(mutation.id.0));
        }
        compare_mutation(outcome.receipt(), mutation)?;
    }

    compare_final_support_states(&store, &base.support_states)?;
    Ok(store)
}

fn compare_mutation(
    receipt: &super::belief_mutation_firewall::BeliefMutationReceipt,
    wire: &WireMutationV1,
) -> Result<(), IsolatedRestartHydrationError> {
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
        return Err(IsolatedRestartHydrationError::MutationSemanticMismatch(wire.id.0));
    }
    Ok(())
}

fn compare_final_support_states(
    store: &EpistemicSupportStore,
    expected: &[WireSupportStateV1],
) -> Result<(), IsolatedRestartHydrationError> {
    if store.len() != expected.len() {
        return Err(IsolatedRestartHydrationError::SupportStateCountMismatch);
    }
    for wire in expected {
        let state = store
            .state(wire.claim_id)
            .ok_or(IsolatedRestartHydrationError::SupportStateMissing(wire.claim_id))?;
        if state.support() != wire.current_support
            || state.revision() != wire.revision
            || state.initialized_at_cycle() != wire.initialized_at_cycle
            || state.last_updated_cycle() != wire.last_updated_cycle
            || state.last_mutation_id() != wire.last_mutation_id
        {
            return Err(IsolatedRestartHydrationError::SupportStateMismatch(wire.claim_id));
        }
    }
    Ok(())
}

fn verify_reexport(
    store: &EpistemicSupportStore,
    history: &BeliefRevisionHistory,
    base: &EpistemicRestartWireSnapshotV1,
) -> Result<(), IsolatedRestartHydrationError> {
    let claim_ids = base
        .support_states
        .iter()
        .map(|state| state.claim_id)
        .collect::<Vec<_>>();
    let mutations = BeliefMutationPersistenceCapsuleV1::capture(
        store,
        &claim_ids,
        base.captured_at_cycle,
    )
    .map_err(|error| IsolatedRestartHydrationError::MutationRecaptureRejected {
        detail: format!("{error:?}"),
    })?;
    if mutations.states().len() != base.support_states.len()
        || mutations.mutations().len() != base.mutations.len()
        || mutations.consumed_authorization_count() as u64 != base.consumed_authorization_count
    {
        return Err(IsolatedRestartHydrationError::MutationRecaptureMismatch);
    }
    for (left, right) in mutations.states().iter().zip(&base.support_states) {
        if left.claim_id != right.claim_id
            || left.baseline_support != right.baseline_support
            || left.current_support != right.current_support
            || left.revision != right.revision
            || left.initialized_at_cycle != right.initialized_at_cycle
            || left.last_updated_cycle != right.last_updated_cycle
            || left.last_mutation_id != right.last_mutation_id
        {
            return Err(IsolatedRestartHydrationError::MutationRecaptureMismatch);
        }
    }
    for (left, right) in mutations.mutations().iter().zip(&base.mutations) {
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
            return Err(IsolatedRestartHydrationError::MutationRecaptureMismatch);
        }
    }

    let revisions = BeliefRevisionHistoryCapsuleV1::capture(
        history,
        &mutations,
        base.captured_at_cycle,
    )
    .map_err(|error| IsolatedRestartHydrationError::RevisionRecaptureRejected {
        detail: format!("{error:?}"),
    })?;
    if revisions.next_receipt_id() != base.revision_next_receipt_id
        || revisions.receipts().len() != base.revisions.len()
    {
        return Err(IsolatedRestartHydrationError::RevisionRecaptureMismatch);
    }
    for (receipt, wire) in revisions.receipts().iter().zip(&base.revisions) {
        let schema_decision = BeliefRevisionDecisionSnapshotV1::capture(receipt.decision());
        compare_revision(receipt, wire, &schema_decision)?;
    }
    Ok(())
}

fn digest_hydration(hydration: &IsolatedEpistemicRestartHydrationV1) -> IsolatedRestartHydrationDigestV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-isolated-restart-hydration-v1");
    hasher.update(&[1]);
    hasher.update(&hydration.quarantine.quarantine_digest().as_bytes());
    hasher.update(&hydration.quarantine.trust_context_digest().as_bytes());
    hasher.update(&hydration.protected_checkpoint.statement_digest().as_bytes());
    hasher.update(&hydration.source_snapshot.outer_checksum);
    hasher.update(&hydration.source_snapshot.claimed_v2_digest);
    hasher.update(&hydration.hydrated_at_cycle.to_le_bytes());
    hasher.update(&(hydration.support_store.len() as u64).to_le_bytes());
    hasher.update(&(hydration.support_store.history().len() as u64).to_le_bytes());
    hasher.update(&(hydration.revision_history.len() as u64).to_le_bytes());
    hasher.update(&[u8::from(hydration.trusted_state_mutated)]);
    hasher.update(&[u8::from(hydration.writable_state_export_authorized)]);
    hasher.update(&[u8::from(hydration.activation_authorized)]);
    IsolatedRestartHydrationDigestV1(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
pub enum IsolatedRestartHydrationError {
    QuarantineRejected { detail: String },
    CheckpointRejected { detail: String },
    SnapshotRejected { detail: String },
    SnapshotQuarantineMismatch,
    CheckpointContextMismatch,
    HydrationPredatesCheckpointVerification { hydrated_at_cycle: u64, verified_at_cycle: u64 },
    CheckpointExpiredAtHydration { hydrated_at_cycle: u64, expires_at_cycle: u64 },
    UnexpectedCheckpointAuthority,
    LedgerRebuildRejected,
    LedgerIdMismatch,
    RevisionSchemaMismatch(u64),
    PolicyRebuildRejected(u64),
    DuplicateBasisDiagnosticOrphan { receipt_id: u64, evidence_id: u64 },
    ProposalRebuildRejected(u64),
    CalibrationRebuildRejected(u64),
    UncertaintyRebuildRejected(u64),
    DecisionReplayMismatch(u64),
    RevisionHistoryRebuildRejected(u64),
    RevisionIdMismatch { expected: u64, actual: u64 },
    RevisionMissingAfterRebuild(u64),
    RevisionSemanticMismatch(u64),
    NextRevisionIdMismatch,
    SupportBaselineRejected(ClaimId),
    MutationRevisionMissing(u64),
    HistoricalLedgerStateInsufficient { mutation_id: u64, claim_id: ClaimId },
    SupportStateMissing(ClaimId),
    AuthorizationRebuildRejected(u64),
    MutationReplayRejected(u64),
    MutationReplayWasNotNew(u64),
    MutationSemanticMismatch(u64),
    SupportStateCountMismatch,
    SupportStateMismatch(ClaimId),
    MutationRecaptureRejected { detail: String },
    MutationRecaptureMismatch,
    RevisionRecaptureRejected { detail: String },
    RevisionRecaptureMismatch,
    UnexpectedHydrationAuthority,
    HydrationDigestMismatch,
}

impl fmt::Display for IsolatedRestartHydrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "isolated restart hydration rejected: {self:?}")
    }
}

impl Error for IsolatedRestartHydrationError {}
