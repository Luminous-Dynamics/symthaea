// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only historical evidence projections for protected restart mutation seals.
//!
//! EKM-062 establishes that one exact protected mutation-time evidence census is
//! eligible for isolated historical projection review. This module materializes
//! only immutable claim/evidence projection records from that protected census.
//!
//! Historical membership comes from the protected EKM-028 census, never from
//! `observed_at_cycle`. Final-ledger records absent from the protected census are
//! classified as later ledger growth even when they carry an older observation
//! timestamp. No `EpistemicLedger`, mutation firewall, support store, writable
//! history, activation handle, or trusted-state mutation is constructed here.

use super::belief_mutation_firewall::BeliefMutationReceiptId;
use super::belief_mutation_seal_wire::{
    BeliefMutationSealWireSnapshotV1, WireBeliefMutationEvidenceSealV1,
};
use super::belief_mutation_transaction::SealedClaimSnapshot;
use super::belief_revision_receipt::{BeliefRevisionReceiptId, RevisionEvidenceSnapshot};
use super::claim_evidence::{ClaimId, ClaimKind, EvidenceId, EvidenceKind, EvidencePolarity};
use super::epistemic_restart_historical_replay_eligibility::{
    HistoricalReplayEligibilityError, HistoricalReplayEligibilityReceiptV1,
};
use super::epistemic_restart_mutation_seal_admission::ProtectedMutationSealAdmissionV1;
use super::epistemic_restart_mutation_seal_checkpoint::VerifiedRestartMutationSealCheckpointV1;
use super::epistemic_restart_mutation_seal_currentness::VerifiedRestartMutationSealCurrentnessV1;
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_PROJECTION_RECORDS: usize = 1_000_000;
const MAX_PROJECTED_EVIDENCE: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalEvidenceProjectionVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoricalEvidenceProjectionDigestV1([u8; 32]);

impl HistoricalEvidenceProjectionDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Immutable view of the exact claim/evidence census sealed for one successful
/// belief mutation. `excluded_final_evidence_ids` are records present on the final
/// append-only claim but absent from the protected historical census.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalEvidenceProjectionRecordV1 {
    mutation_id: BeliefMutationReceiptId,
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    sealed_at_cycle: u64,
    claim: SealedClaimSnapshot,
    evidence: Vec<RevisionEvidenceSnapshot>,
    final_claim_evidence_count: usize,
    excluded_final_evidence_ids: Vec<EvidenceId>,
    excluded_with_observation_not_after_seal_count: usize,
    record_digest: HistoricalEvidenceProjectionDigestV1,
}

impl HistoricalEvidenceProjectionRecordV1 {
    pub fn mutation_id(&self) -> BeliefMutationReceiptId {
        self.mutation_id
    }

    pub fn source_revision_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.source_revision_receipt_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn sealed_at_cycle(&self) -> u64 {
        self.sealed_at_cycle
    }

    pub fn claim(&self) -> &SealedClaimSnapshot {
        &self.claim
    }

    pub fn evidence(&self) -> &[RevisionEvidenceSnapshot] {
        &self.evidence
    }

    pub fn final_claim_evidence_count(&self) -> usize {
        self.final_claim_evidence_count
    }

    pub fn excluded_final_evidence_ids(&self) -> &[EvidenceId] {
        &self.excluded_final_evidence_ids
    }

    pub fn excluded_with_observation_not_after_seal_count(&self) -> usize {
        self.excluded_with_observation_not_after_seal_count
    }

    pub fn record_digest(&self) -> HistoricalEvidenceProjectionDigestV1 {
        self.record_digest
    }
}

/// Read-only projection report over every retained mutation-time census.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalEvidenceProjectionV1 {
    version: HistoricalEvidenceProjectionVersion,
    restart_capture_cycle: u64,
    projected_at_cycle: u64,
    eligibility_receipt_digest: [u8; 32],
    protected_seal_capsule_digest: [u8; 32],
    restart_outer_checksum: [u8; 32],
    seal_wire_checksum: [u8; 32],
    records: Vec<HistoricalEvidenceProjectionRecordV1>,
    total_projected_evidence: usize,
    total_excluded_final_evidence: usize,
    total_excluded_with_observation_not_after_seal: usize,
    membership_derived_from_protected_census: bool,
    observation_cycle_used_as_membership: bool,
    ledger_constructed: bool,
    historical_replay_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    projection_digest: HistoricalEvidenceProjectionDigestV1,
}

impl HistoricalEvidenceProjectionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        restart: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
        restart_receipt: &EpistemicRestartValidationReceiptV1,
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        admission: &ProtectedMutationSealAdmissionV1,
        currentness: &VerifiedRestartMutationSealCurrentnessV1,
        eligibility: &HistoricalReplayEligibilityReceiptV1,
        projected_at_cycle: u64,
    ) -> Result<Self, HistoricalEvidenceProjectionError> {
        eligibility
            .verify_against(
                restart,
                seals,
                restart_receipt,
                checkpoint,
                admission,
                currentness,
                eligibility.reviewed_at_cycle(),
            )
            .map_err(HistoricalEvidenceProjectionError::EligibilityRejected)?;

        if !eligibility.protected_source_equivalence_verified()
            || !eligibility.checkpoint_currentness_verified()
            || !eligibility.isolated_historical_projection_review_eligible()
            || eligibility.historical_replay_authorized()
            || eligibility.capsule_construction_authorized()
            || eligibility.writable_hydration_authorized()
            || eligibility.activation_authorized()
        {
            return Err(HistoricalEvidenceProjectionError::UnexpectedEligibilityClaims);
        }
        if projected_at_cycle < eligibility.reviewed_at_cycle() {
            return Err(HistoricalEvidenceProjectionError::ProjectionPredatesEligibility {
                projected_at_cycle,
                eligibility_reviewed_at_cycle: eligibility.reviewed_at_cycle(),
            });
        }
        if projected_at_cycle >= currentness.statement().expires_at_cycle() {
            return Err(HistoricalEvidenceProjectionError::CurrentnessExpired {
                projected_at_cycle,
                expires_at_cycle: currentness.statement().expires_at_cycle(),
            });
        }
        if seals.records.len() > MAX_PROJECTION_RECORDS {
            return Err(HistoricalEvidenceProjectionError::TooManyProjectionRecords {
                actual: seals.records.len(),
                maximum: MAX_PROJECTION_RECORDS,
            });
        }

        let mut records = Vec::with_capacity(seals.records.len());
        let mut total_projected_evidence = 0usize;
        let mut total_excluded_final_evidence = 0usize;
        let mut total_excluded_with_observation_not_after_seal = 0usize;

        for seal in &seals.records {
            let record = project_record(restart, seal)?;
            total_projected_evidence = total_projected_evidence
                .checked_add(record.evidence.len())
                .ok_or(HistoricalEvidenceProjectionError::LengthOverflow)?;
            if total_projected_evidence > MAX_PROJECTED_EVIDENCE {
                return Err(HistoricalEvidenceProjectionError::TooManyProjectedEvidenceRecords {
                    actual: total_projected_evidence,
                    maximum: MAX_PROJECTED_EVIDENCE,
                });
            }
            total_excluded_final_evidence = total_excluded_final_evidence
                .checked_add(record.excluded_final_evidence_ids.len())
                .ok_or(HistoricalEvidenceProjectionError::LengthOverflow)?;
            total_excluded_with_observation_not_after_seal =
                total_excluded_with_observation_not_after_seal
                    .checked_add(record.excluded_with_observation_not_after_seal_count)
                    .ok_or(HistoricalEvidenceProjectionError::LengthOverflow)?;
            records.push(record);
        }

        let mut out = Self {
            version: HistoricalEvidenceProjectionVersion::V1,
            restart_capture_cycle: restart.base.captured_at_cycle,
            projected_at_cycle,
            eligibility_receipt_digest: eligibility.receipt_digest().as_bytes(),
            protected_seal_capsule_digest: eligibility.protected_seal_capsule_digest(),
            restart_outer_checksum: restart.outer_checksum,
            seal_wire_checksum: seals.wire_checksum,
            records,
            total_projected_evidence,
            total_excluded_final_evidence,
            total_excluded_with_observation_not_after_seal,
            membership_derived_from_protected_census: true,
            observation_cycle_used_as_membership: false,
            ledger_constructed: false,
            historical_replay_authorized: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            projection_digest: HistoricalEvidenceProjectionDigestV1([0; 32]),
        };
        out.projection_digest = digest_projection(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> HistoricalEvidenceProjectionVersion {
        self.version
    }

    pub fn restart_capture_cycle(&self) -> u64 {
        self.restart_capture_cycle
    }

    pub fn projected_at_cycle(&self) -> u64 {
        self.projected_at_cycle
    }

    pub fn records(&self) -> &[HistoricalEvidenceProjectionRecordV1] {
        &self.records
    }

    pub fn total_projected_evidence(&self) -> usize {
        self.total_projected_evidence
    }

    pub fn total_excluded_final_evidence(&self) -> usize {
        self.total_excluded_final_evidence
    }

    pub fn total_excluded_with_observation_not_after_seal(&self) -> usize {
        self.total_excluded_with_observation_not_after_seal
    }

    pub fn membership_derived_from_protected_census(&self) -> bool {
        self.membership_derived_from_protected_census
    }

    pub fn observation_cycle_used_as_membership(&self) -> bool {
        self.observation_cycle_used_as_membership
    }

    pub fn ledger_constructed(&self) -> bool {
        self.ledger_constructed
    }

    pub fn historical_replay_authorized(&self) -> bool {
        self.historical_replay_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn projection_digest(&self) -> HistoricalEvidenceProjectionDigestV1 {
        self.projection_digest
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
        projected_at_cycle: u64,
    ) -> Result<(), HistoricalEvidenceProjectionError> {
        let live = Self::project(
            restart,
            seals,
            restart_receipt,
            checkpoint,
            admission,
            currentness,
            eligibility,
            projected_at_cycle,
        )?;
        if &live != self {
            return Err(HistoricalEvidenceProjectionError::ProjectionMismatch);
        }
        if digest_projection(self)? != self.projection_digest {
            return Err(HistoricalEvidenceProjectionError::ProjectionDigestMismatch);
        }
        Ok(())
    }
}

fn project_record(
    restart: &EpistemicRestartWireSnapshotV2,
    seal: &WireBeliefMutationEvidenceSealV1,
) -> Result<HistoricalEvidenceProjectionRecordV1, HistoricalEvidenceProjectionError> {
    let final_claim = restart
        .base
        .claims
        .iter()
        .find(|claim| claim.id == seal.claim_id)
        .ok_or(HistoricalEvidenceProjectionError::FinalClaimMissing(seal.claim_id))?;

    if final_claim.id != seal.claim.claim_id
        || final_claim.statement != seal.claim.statement
        || final_claim.kind != seal.claim.kind
        || final_claim.domain != seal.claim.domain
        || final_claim.scope != seal.claim.scope
        || final_claim.created_at_cycle != seal.claim.created_at_cycle
    {
        return Err(HistoricalEvidenceProjectionError::FinalClaimChanged(seal.claim_id));
    }

    let sealed_ids = seal
        .evidence
        .iter()
        .map(|evidence| evidence.evidence_id)
        .collect::<HashSet<_>>();
    if sealed_ids.len() != seal.evidence.len() {
        return Err(HistoricalEvidenceProjectionError::DuplicateSealedEvidence(
            seal.mutation_id,
        ));
    }

    let final_ids = final_claim.evidence_ids.iter().copied().collect::<HashSet<_>>();
    if final_ids.len() != final_claim.evidence_ids.len() {
        return Err(HistoricalEvidenceProjectionError::DuplicateFinalClaimEvidence(
            seal.claim_id,
        ));
    }
    if !sealed_ids.is_subset(&final_ids) {
        return Err(HistoricalEvidenceProjectionError::ProtectedCensusNotSubsetOfFinalClaim(
            seal.mutation_id,
        ));
    }

    let mut excluded_final_evidence_ids = final_ids
        .difference(&sealed_ids)
        .copied()
        .collect::<Vec<_>>();
    excluded_final_evidence_ids.sort_unstable();

    let mut excluded_with_observation_not_after_seal_count = 0usize;
    for evidence_id in &excluded_final_evidence_ids {
        let evidence = restart
            .base
            .evidence
            .iter()
            .find(|candidate| candidate.id == *evidence_id)
            .ok_or(HistoricalEvidenceProjectionError::FinalEvidenceMissing(*evidence_id))?;
        if evidence.claim_id != seal.claim_id {
            return Err(HistoricalEvidenceProjectionError::FinalEvidenceForDifferentClaim {
                evidence_id: *evidence_id,
                expected_claim: seal.claim_id,
                actual_claim: evidence.claim_id,
            });
        }
        if evidence.observed_at_cycle <= seal.sealed_at_cycle {
            excluded_with_observation_not_after_seal_count =
                excluded_with_observation_not_after_seal_count
                    .checked_add(1)
                    .ok_or(HistoricalEvidenceProjectionError::LengthOverflow)?;
        }
    }

    let mut record = HistoricalEvidenceProjectionRecordV1 {
        mutation_id: seal.mutation_id,
        source_revision_receipt_id: seal.source_revision_receipt_id,
        claim_id: seal.claim_id,
        sealed_at_cycle: seal.sealed_at_cycle,
        claim: seal.claim.clone(),
        evidence: seal.evidence.clone(),
        final_claim_evidence_count: final_claim.evidence_ids.len(),
        excluded_final_evidence_ids,
        excluded_with_observation_not_after_seal_count,
        record_digest: HistoricalEvidenceProjectionDigestV1([0; 32]),
    };
    record.record_digest = digest_record(&record)?;
    Ok(record)
}

fn digest_record(
    record: &HistoricalEvidenceProjectionRecordV1,
) -> Result<HistoricalEvidenceProjectionDigestV1, HistoricalEvidenceProjectionError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-historical-evidence-projection-record-v1");
    hasher.update(&record.mutation_id.0.to_le_bytes());
    hasher.update(&record.source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&record.claim_id.0.to_le_bytes());
    hasher.update(&record.sealed_at_cycle.to_le_bytes());
    hash_claim(&mut hasher, &record.claim)?;
    hash_usize(&mut hasher, record.evidence.len())?;
    for evidence in &record.evidence {
        hash_evidence(&mut hasher, evidence)?;
    }
    hash_usize(&mut hasher, record.final_claim_evidence_count)?;
    hash_usize(&mut hasher, record.excluded_final_evidence_ids.len())?;
    for evidence_id in &record.excluded_final_evidence_ids {
        hasher.update(&evidence_id.0.to_le_bytes());
    }
    hash_usize(
        &mut hasher,
        record.excluded_with_observation_not_after_seal_count,
    )?;
    Ok(HistoricalEvidenceProjectionDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn digest_projection(
    projection: &HistoricalEvidenceProjectionV1,
) -> Result<HistoricalEvidenceProjectionDigestV1, HistoricalEvidenceProjectionError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-historical-evidence-projection-v1");
    hasher.update(&[1]);
    hasher.update(&projection.restart_capture_cycle.to_le_bytes());
    hasher.update(&projection.projected_at_cycle.to_le_bytes());
    hasher.update(&projection.eligibility_receipt_digest);
    hasher.update(&projection.protected_seal_capsule_digest);
    hasher.update(&projection.restart_outer_checksum);
    hasher.update(&projection.seal_wire_checksum);
    hash_usize(&mut hasher, projection.records.len())?;
    for record in &projection.records {
        hasher.update(&record.record_digest.as_bytes());
    }
    hash_usize(&mut hasher, projection.total_projected_evidence)?;
    hash_usize(&mut hasher, projection.total_excluded_final_evidence)?;
    hash_usize(
        &mut hasher,
        projection.total_excluded_with_observation_not_after_seal,
    )?;
    hasher.update(&[u8::from(
        projection.membership_derived_from_protected_census,
    )]);
    hasher.update(&[u8::from(projection.observation_cycle_used_as_membership)]);
    hasher.update(&[u8::from(projection.ledger_constructed)]);
    hasher.update(&[u8::from(projection.historical_replay_authorized)]);
    hasher.update(&[u8::from(projection.writable_hydration_authorized)]);
    hasher.update(&[u8::from(projection.activation_authorized)]);
    Ok(HistoricalEvidenceProjectionDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_claim(
    hasher: &mut blake3::Hasher,
    claim: &SealedClaimSnapshot,
) -> Result<(), HistoricalEvidenceProjectionError> {
    hasher.update(&claim.claim_id.0.to_le_bytes());
    hash_bytes(hasher, claim.statement.as_bytes())?;
    hasher.update(&[claim_kind_tag(claim.kind)]);
    hash_optional_string(hasher, claim.domain.as_deref())?;
    hash_optional_string(hasher, claim.scope.as_deref())?;
    hasher.update(&claim.created_at_cycle.to_le_bytes());
    Ok(())
}

fn hash_evidence(
    hasher: &mut blake3::Hasher,
    evidence: &RevisionEvidenceSnapshot,
) -> Result<(), HistoricalEvidenceProjectionError> {
    hasher.update(&evidence.evidence_id.0.to_le_bytes());
    hasher.update(&evidence.claim_id.0.to_le_bytes());
    hasher.update(&[evidence_kind_tag(evidence.kind)]);
    hasher.update(&[evidence_polarity_tag(evidence.polarity)]);
    hasher.update(&evidence.provenance_id.0.to_le_bytes());
    hasher.update(&evidence.observed_at_cycle.to_le_bytes());
    hash_optional_string(hasher, evidence.context.as_deref())?;
    hash_optional_string(hasher, evidence.method.as_deref())?;
    Ok(())
}

fn hash_optional_string(
    hasher: &mut blake3::Hasher,
    value: Option<&str>,
) -> Result<(), HistoricalEvidenceProjectionError> {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_bytes(hasher, value.as_bytes())?;
        }
        None => hasher.update(&[0]),
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    value: &[u8],
) -> Result<(), HistoricalEvidenceProjectionError> {
    let len = u64::try_from(value.len())
        .map_err(|_| HistoricalEvidenceProjectionError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(value);
    Ok(())
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), HistoricalEvidenceProjectionError> {
    let value = u64::try_from(value)
        .map_err(|_| HistoricalEvidenceProjectionError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn claim_kind_tag(kind: ClaimKind) -> u8 {
    match kind {
        ClaimKind::Descriptive => 1,
        ClaimKind::Predictive => 2,
        ClaimKind::Causal => 3,
        ClaimKind::Counterfactual => 4,
        ClaimKind::Procedural => 5,
        ClaimKind::Normative => 6,
    }
}

fn evidence_kind_tag(kind: EvidenceKind) -> u8 {
    match kind {
        EvidenceKind::Report => 1,
        EvidenceKind::Observation => 2,
        EvidenceKind::Measurement => 3,
        EvidenceKind::Intervention => 4,
        EvidenceKind::Replication => 5,
        EvidenceKind::Simulation => 6,
        EvidenceKind::Deduction => 7,
        EvidenceKind::ToolResult => 8,
    }
}

fn evidence_polarity_tag(polarity: EvidencePolarity) -> u8 {
    match polarity {
        EvidencePolarity::Supports => 1,
        EvidencePolarity::Contradicts => 2,
        EvidencePolarity::Contextualizes => 3,
    }
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
pub enum HistoricalEvidenceProjectionError {
    EligibilityRejected(HistoricalReplayEligibilityError),
    UnexpectedEligibilityClaims,
    ProjectionPredatesEligibility {
        projected_at_cycle: u64,
        eligibility_reviewed_at_cycle: u64,
    },
    CurrentnessExpired {
        projected_at_cycle: u64,
        expires_at_cycle: u64,
    },
    TooManyProjectionRecords {
        actual: usize,
        maximum: usize,
    },
    TooManyProjectedEvidenceRecords {
        actual: usize,
        maximum: usize,
    },
    FinalClaimMissing(ClaimId),
    FinalClaimChanged(ClaimId),
    DuplicateSealedEvidence(BeliefMutationReceiptId),
    DuplicateFinalClaimEvidence(ClaimId),
    ProtectedCensusNotSubsetOfFinalClaim(BeliefMutationReceiptId),
    FinalEvidenceMissing(EvidenceId),
    FinalEvidenceForDifferentClaim {
        evidence_id: EvidenceId,
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    ProjectionMismatch,
    ProjectionDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for HistoricalEvidenceProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "historical evidence projection rejected: {self:?}")
    }
}

impl Error for HistoricalEvidenceProjectionError {}
