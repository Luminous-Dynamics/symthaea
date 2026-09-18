// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistence sidecar for EKM-028 mutation-time evidence seals.
//!
//! EKM-055 discovered that a final restart image cannot always replay an older
//! belief mutation because the final ledger may contain evidence added after the
//! mutation decision. EKM-028 already captured the exact claim/evidence census at
//! decision time; the missing property is retention of that seal alongside the
//! successful mutation.
//!
//! This module does not create a new chronology vocabulary and does not mutate
//! epistemic state. It binds the existing `BeliefRevisionEvidenceSeal` to the
//! exact mutation receipt and requires complete one-to-one coverage of the
//! persisted mutation history.

use super::belief_mutation_authority::PreparedBeliefMutation;
use super::belief_mutation_firewall::{
    BeliefMutationOutcome, BeliefMutationReceipt, BeliefMutationReceiptId,
};
use super::belief_mutation_persistence::{
    BeliefMutationPersistenceCapsuleV1, PersistedBeliefMutationV1,
};
use super::belief_mutation_transaction::{BeliefRevisionEvidenceSeal, SealedClaimSnapshot};
use super::belief_revision_receipt::{BeliefRevisionReceiptId, RevisionEvidenceSnapshot};
use super::claim_evidence::{
    ClaimId, ClaimKind, EpistemicLedger, EvidenceKind, EvidencePolarity,
};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationSealPersistenceVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BeliefMutationSealRecordDigestV1([u8; 32]);

impl BeliefMutationSealRecordDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BeliefMutationSealCapsuleDigestV1([u8; 32]);

impl BeliefMutationSealCapsuleDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Immutable retention record binding one EKM-028 evidence seal to the exact
/// mutation receipt produced from that prepared decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedBeliefMutationEvidenceSealV1 {
    mutation_id: BeliefMutationReceiptId,
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    sealed_at_cycle: u64,
    applied_at_cycle: u64,
    mutation_binding_digest: [u8; 32],
    seal: BeliefRevisionEvidenceSeal,
    record_digest: BeliefMutationSealRecordDigestV1,
}

impl PersistedBeliefMutationEvidenceSealV1 {
    /// Capture the already-created EKM-028 seal after a successful or idempotent
    /// EKM-029 authority application. No mutation is performed here.
    pub fn capture(
        prepared: &PreparedBeliefMutation,
        outcome: &BeliefMutationOutcome,
    ) -> Result<Self, BeliefMutationSealPersistenceError> {
        let mutation = outcome.receipt();
        let seal = prepared.seal();
        validate_pair(prepared, mutation, seal)?;
        let mutation_binding_digest = digest_live_mutation(mutation)?;
        let mut record = Self {
            mutation_id: mutation.id(),
            source_revision_receipt_id: mutation.source_revision_receipt_id(),
            claim_id: mutation.claim_id(),
            sealed_at_cycle: seal.sealed_at_cycle(),
            applied_at_cycle: mutation.applied_at_cycle(),
            mutation_binding_digest,
            seal: seal.clone(),
            record_digest: BeliefMutationSealRecordDigestV1([0; 32]),
        };
        validate_seal_shape(&record.seal)?;
        record.record_digest = digest_record(&record)?;
        Ok(record)
    }

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

    pub fn applied_at_cycle(&self) -> u64 {
        self.applied_at_cycle
    }

    pub fn seal(&self) -> &BeliefRevisionEvidenceSeal {
        &self.seal
    }

    pub fn record_digest(&self) -> BeliefMutationSealRecordDigestV1 {
        self.record_digest
    }

    pub fn verify_internal(&self) -> Result<(), BeliefMutationSealPersistenceError> {
        if self.source_revision_receipt_id != self.seal.source_revision_receipt_id()
            || self.claim_id != self.seal.claim().claim_id
            || self.sealed_at_cycle != self.seal.sealed_at_cycle()
            || self.sealed_at_cycle > self.applied_at_cycle
        {
            return Err(BeliefMutationSealPersistenceError::RecordSealBindingMismatch(
                self.mutation_id,
            ));
        }
        validate_seal_shape(&self.seal)?;
        if digest_record(self)? != self.record_digest {
            return Err(BeliefMutationSealPersistenceError::RecordDigestMismatch(
                self.mutation_id,
            ));
        }
        Ok(())
    }
}

/// Complete mutation-time evidence-seal inventory for one persisted support-store
/// snapshot. A capsule is valid only when every persisted mutation has exactly one
/// bound EKM-028 seal record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefMutationEvidenceSealCapsuleV1 {
    version: BeliefMutationSealPersistenceVersion,
    captured_at_cycle: u64,
    linked_mutation_capture_cycle: u64,
    records: Vec<PersistedBeliefMutationEvidenceSealV1>,
    capsule_digest: BeliefMutationSealCapsuleDigestV1,
}

impl BeliefMutationEvidenceSealCapsuleV1 {
    pub fn capture(
        records: &[PersistedBeliefMutationEvidenceSealV1],
        mutations: &BeliefMutationPersistenceCapsuleV1,
        ledger: &EpistemicLedger,
        captured_at_cycle: u64,
    ) -> Result<Self, BeliefMutationSealPersistenceError> {
        if captured_at_cycle < mutations.captured_at_cycle() {
            return Err(BeliefMutationSealPersistenceError::CapturePredatesMutationCapsule {
                captured_at_cycle,
                mutation_capture_cycle: mutations.captured_at_cycle(),
            });
        }
        if records.len() != mutations.mutations().len() {
            return Err(BeliefMutationSealPersistenceError::MutationSealCountMismatch {
                mutations: mutations.mutations().len(),
                seals: records.len(),
            });
        }

        let mut seen_mutations = HashSet::new();
        let mut seen_revisions = HashSet::new();
        let mut persisted = records.to_vec();
        persisted.sort_by_key(|record| record.mutation_id);

        for record in &persisted {
            record.verify_internal()?;
            if !seen_mutations.insert(record.mutation_id) {
                return Err(BeliefMutationSealPersistenceError::DuplicateMutationSeal(
                    record.mutation_id,
                ));
            }
            if !seen_revisions.insert(record.source_revision_receipt_id) {
                return Err(BeliefMutationSealPersistenceError::DuplicateRevisionSeal(
                    record.source_revision_receipt_id,
                ));
            }
            if record.applied_at_cycle > captured_at_cycle {
                return Err(BeliefMutationSealPersistenceError::SealRecordPostdatesCapture {
                    mutation_id: record.mutation_id,
                    applied_at_cycle: record.applied_at_cycle,
                    captured_at_cycle,
                });
            }
            validate_seal_against_final_ledger(&record.seal, ledger)?;
        }

        for mutation in mutations.mutations() {
            let record = persisted
                .iter()
                .find(|record| record.mutation_id == mutation.id)
                .ok_or(BeliefMutationSealPersistenceError::MissingMutationSeal(
                    mutation.id,
                ))?;
            validate_record_against_persisted_mutation(record, mutation)?;
        }

        let mut capsule = Self {
            version: BeliefMutationSealPersistenceVersion::V1,
            captured_at_cycle,
            linked_mutation_capture_cycle: mutations.captured_at_cycle(),
            records: persisted,
            capsule_digest: BeliefMutationSealCapsuleDigestV1([0; 32]),
        };
        capsule.capsule_digest = digest_capsule(&capsule)?;
        capsule.verify(mutations, ledger, captured_at_cycle)?;
        Ok(capsule)
    }

    pub fn version(&self) -> BeliefMutationSealPersistenceVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn linked_mutation_capture_cycle(&self) -> u64 {
        self.linked_mutation_capture_cycle
    }

    pub fn records(&self) -> &[PersistedBeliefMutationEvidenceSealV1] {
        &self.records
    }

    pub fn record_for_mutation(
        &self,
        mutation_id: BeliefMutationReceiptId,
    ) -> Option<&PersistedBeliefMutationEvidenceSealV1> {
        self.records
            .binary_search_by_key(&mutation_id, |record| record.mutation_id)
            .ok()
            .map(|index| &self.records[index])
    }

    pub fn capsule_digest(&self) -> BeliefMutationSealCapsuleDigestV1 {
        self.capsule_digest
    }

    pub fn verify(
        &self,
        mutations: &BeliefMutationPersistenceCapsuleV1,
        ledger: &EpistemicLedger,
        observed_at_cycle: u64,
    ) -> Result<(), BeliefMutationSealPersistenceError> {
        if observed_at_cycle < self.captured_at_cycle {
            return Err(BeliefMutationSealPersistenceError::ObservationPredatesCapture {
                observed_at_cycle,
                captured_at_cycle: self.captured_at_cycle,
            });
        }
        if self.linked_mutation_capture_cycle != mutations.captured_at_cycle() {
            return Err(BeliefMutationSealPersistenceError::MutationCaptureCycleMismatch);
        }
        if self.records.len() != mutations.mutations().len() {
            return Err(BeliefMutationSealPersistenceError::MutationSealCountMismatch {
                mutations: mutations.mutations().len(),
                seals: self.records.len(),
            });
        }
        for window in self.records.windows(2) {
            if window[0].mutation_id >= window[1].mutation_id {
                return Err(BeliefMutationSealPersistenceError::MutationSealsNotOrdered);
            }
        }
        for mutation in mutations.mutations() {
            let record = self
                .record_for_mutation(mutation.id)
                .ok_or(BeliefMutationSealPersistenceError::MissingMutationSeal(
                    mutation.id,
                ))?;
            record.verify_internal()?;
            validate_record_against_persisted_mutation(record, mutation)?;
            validate_seal_against_final_ledger(&record.seal, ledger)?;
        }
        if digest_capsule(self)? != self.capsule_digest {
            return Err(BeliefMutationSealPersistenceError::CapsuleDigestMismatch);
        }
        Ok(())
    }
}

fn validate_pair(
    prepared: &PreparedBeliefMutation,
    mutation: &BeliefMutationReceipt,
    seal: &BeliefRevisionEvidenceSeal,
) -> Result<(), BeliefMutationSealPersistenceError> {
    if mutation.source_revision_receipt_id() != prepared.receipt_id()
        || mutation.claim_id() != prepared.claim_id()
        || prepared.receipt_id() != seal.source_revision_receipt_id()
        || prepared.claim_id() != seal.claim().claim_id
        || mutation.proposed_delta().to_bits() != prepared.receipt().proposed_delta().to_bits()
    {
        return Err(BeliefMutationSealPersistenceError::AppliedMutationPreparedPairMismatch);
    }
    if seal.sealed_at_cycle() > mutation.authorized_at_cycle()
        || mutation.authorized_at_cycle() > mutation.applied_at_cycle()
    {
        return Err(BeliefMutationSealPersistenceError::MutationSealTemporalMismatch(
            mutation.id(),
        ));
    }
    Ok(())
}

fn validate_record_against_persisted_mutation(
    record: &PersistedBeliefMutationEvidenceSealV1,
    mutation: &PersistedBeliefMutationV1,
) -> Result<(), BeliefMutationSealPersistenceError> {
    if record.mutation_id != mutation.id
        || record.source_revision_receipt_id != mutation.source_revision_receipt_id
        || record.claim_id != mutation.claim_id
        || record.applied_at_cycle != mutation.applied_at_cycle
        || record.sealed_at_cycle > mutation.authorized_at_cycle
        || mutation.authorized_at_cycle > mutation.applied_at_cycle
    {
        return Err(BeliefMutationSealPersistenceError::MutationRecordBindingMismatch(
            mutation.id,
        ));
    }
    if digest_persisted_mutation(mutation)? != record.mutation_binding_digest {
        return Err(BeliefMutationSealPersistenceError::MutationBindingDigestMismatch(
            mutation.id,
        ));
    }
    Ok(())
}

fn validate_seal_shape(
    seal: &BeliefRevisionEvidenceSeal,
) -> Result<(), BeliefMutationSealPersistenceError> {
    let claim = seal.claim();
    if claim.claim_id.0 == 0 || claim.created_at_cycle > seal.sealed_at_cycle() {
        return Err(BeliefMutationSealPersistenceError::InvalidSealedClaim(
            claim.claim_id,
        ));
    }
    let mut previous = None;
    for evidence in seal.evidence() {
        if evidence.claim_id != claim.claim_id
            || evidence.observed_at_cycle > seal.sealed_at_cycle()
        {
            return Err(BeliefMutationSealPersistenceError::InvalidSealedEvidence(
                evidence.evidence_id,
            ));
        }
        if let Some(previous_id) = previous {
            if evidence.evidence_id.0 <= previous_id {
                return Err(BeliefMutationSealPersistenceError::SealedEvidenceNotOrdered);
            }
        }
        previous = Some(evidence.evidence_id.0);
    }
    Ok(())
}

fn validate_seal_against_final_ledger(
    seal: &BeliefRevisionEvidenceSeal,
    ledger: &EpistemicLedger,
) -> Result<(), BeliefMutationSealPersistenceError> {
    let expected_claim = seal.claim();
    let claim = ledger
        .claim(expected_claim.claim_id)
        .ok_or(BeliefMutationSealPersistenceError::SealedClaimMissing(
            expected_claim.claim_id,
        ))?;
    let live_claim = SealedClaimSnapshot {
        claim_id: claim.id,
        statement: claim.statement.clone(),
        kind: claim.kind,
        domain: claim.domain.clone(),
        scope: claim.scope.clone(),
        created_at_cycle: claim.created_at_cycle,
    };
    if &live_claim != expected_claim {
        return Err(BeliefMutationSealPersistenceError::SealedClaimChanged(
            expected_claim.claim_id,
        ));
    }
    for expected in seal.evidence() {
        let live = ledger
            .evidence(expected.evidence_id)
            .ok_or(BeliefMutationSealPersistenceError::SealedEvidenceMissing(
                expected.evidence_id,
            ))?;
        if !evidence_snapshot_matches(expected, live) {
            return Err(BeliefMutationSealPersistenceError::SealedEvidenceChanged(
                expected.evidence_id,
            ));
        }
    }
    Ok(())
}

fn evidence_snapshot_matches(
    snapshot: &RevisionEvidenceSnapshot,
    live: &super::claim_evidence::EvidenceRecord,
) -> bool {
    snapshot.evidence_id == live.id
        && snapshot.claim_id == live.claim_id
        && snapshot.kind == live.kind
        && snapshot.polarity == live.polarity
        && snapshot.provenance_id == live.provenance_id
        && snapshot.observed_at_cycle == live.observed_at_cycle
        && snapshot.context == live.context
        && snapshot.method == live.method
}

fn digest_live_mutation(
    mutation: &BeliefMutationReceipt,
) -> Result<[u8; 32], BeliefMutationSealPersistenceError> {
    digest_mutation_parts(
        mutation.id(),
        mutation.source_revision_receipt_id(),
        mutation.claim_id(),
        mutation.proposed_delta(),
        mutation.support_before().get(),
        mutation.support_after().get(),
        mutation.state_revision_before(),
        mutation.state_revision_after(),
        mutation.authorization_id(),
        mutation.authority_label(),
        mutation.authorized_at_cycle(),
        mutation.applied_at_cycle(),
    )
}

fn digest_persisted_mutation(
    mutation: &PersistedBeliefMutationV1,
) -> Result<[u8; 32], BeliefMutationSealPersistenceError> {
    digest_mutation_parts(
        mutation.id,
        mutation.source_revision_receipt_id,
        mutation.claim_id,
        mutation.proposed_delta,
        mutation.support_before.get(),
        mutation.support_after.get(),
        mutation.state_revision_before,
        mutation.state_revision_after,
        &mutation.authorization_id,
        &mutation.authority_label,
        mutation.authorized_at_cycle,
        mutation.applied_at_cycle,
    )
}

#[allow(clippy::too_many_arguments)]
fn digest_mutation_parts(
    id: BeliefMutationReceiptId,
    source_revision_receipt_id: BeliefRevisionReceiptId,
    claim_id: ClaimId,
    proposed_delta: f32,
    support_before: f32,
    support_after: f32,
    state_revision_before: u64,
    state_revision_after: u64,
    authorization_id: &str,
    authority_label: &str,
    authorized_at_cycle: u64,
    applied_at_cycle: u64,
) -> Result<[u8; 32], BeliefMutationSealPersistenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-binding-v1");
    hasher.update(&id.0.to_le_bytes());
    hasher.update(&source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&claim_id.0.to_le_bytes());
    hasher.update(&proposed_delta.to_bits().to_le_bytes());
    hasher.update(&support_before.to_bits().to_le_bytes());
    hasher.update(&support_after.to_bits().to_le_bytes());
    hasher.update(&state_revision_before.to_le_bytes());
    hasher.update(&state_revision_after.to_le_bytes());
    hash_bytes(&mut hasher, authorization_id.as_bytes())?;
    hash_bytes(&mut hasher, authority_label.as_bytes())?;
    hasher.update(&authorized_at_cycle.to_le_bytes());
    hasher.update(&applied_at_cycle.to_le_bytes());
    Ok(*hasher.finalize().as_bytes())
}

fn digest_record(
    record: &PersistedBeliefMutationEvidenceSealV1,
) -> Result<BeliefMutationSealRecordDigestV1, BeliefMutationSealPersistenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-evidence-seal-record-v1");
    hasher.update(&record.mutation_id.0.to_le_bytes());
    hasher.update(&record.source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&record.claim_id.0.to_le_bytes());
    hasher.update(&record.sealed_at_cycle.to_le_bytes());
    hasher.update(&record.applied_at_cycle.to_le_bytes());
    hasher.update(&record.mutation_binding_digest);
    hash_seal(&mut hasher, &record.seal)?;
    Ok(BeliefMutationSealRecordDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_capsule(
    capsule: &BeliefMutationEvidenceSealCapsuleV1,
) -> Result<BeliefMutationSealCapsuleDigestV1, BeliefMutationSealPersistenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-evidence-seal-capsule-v1");
    hasher.update(&[1]);
    hasher.update(&capsule.captured_at_cycle.to_le_bytes());
    hasher.update(&capsule.linked_mutation_capture_cycle.to_le_bytes());
    let count = u64::try_from(capsule.records.len())
        .map_err(|_| BeliefMutationSealPersistenceError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for record in &capsule.records {
        hasher.update(&record.record_digest.as_bytes());
    }
    Ok(BeliefMutationSealCapsuleDigestV1(*hasher.finalize().as_bytes()))
}

fn hash_seal(
    hasher: &mut blake3::Hasher,
    seal: &BeliefRevisionEvidenceSeal,
) -> Result<(), BeliefMutationSealPersistenceError> {
    hasher.update(&seal.source_revision_receipt_id().0.to_le_bytes());
    hasher.update(&seal.sealed_at_cycle().to_le_bytes());
    hash_claim(hasher, seal.claim())?;
    let count = u64::try_from(seal.evidence().len())
        .map_err(|_| BeliefMutationSealPersistenceError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for evidence in seal.evidence() {
        hasher.update(&evidence.evidence_id.0.to_le_bytes());
        hasher.update(&evidence.claim_id.0.to_le_bytes());
        hasher.update(&[evidence_kind_tag(evidence.kind)]);
        hasher.update(&[evidence_polarity_tag(evidence.polarity)]);
        hasher.update(&evidence.provenance_id.0.to_le_bytes());
        hasher.update(&evidence.observed_at_cycle.to_le_bytes());
        hash_optional_string(hasher, evidence.context.as_deref())?;
        hash_optional_string(hasher, evidence.method.as_deref())?;
    }
    Ok(())
}

fn hash_claim(
    hasher: &mut blake3::Hasher,
    claim: &SealedClaimSnapshot,
) -> Result<(), BeliefMutationSealPersistenceError> {
    hasher.update(&claim.claim_id.0.to_le_bytes());
    hash_bytes(hasher, claim.statement.as_bytes())?;
    hasher.update(&[claim_kind_tag(claim.kind)]);
    hash_optional_string(hasher, claim.domain.as_deref())?;
    hash_optional_string(hasher, claim.scope.as_deref())?;
    hasher.update(&claim.created_at_cycle.to_le_bytes());
    Ok(())
}

fn hash_optional_string(
    hasher: &mut blake3::Hasher,
    value: Option<&str>,
) -> Result<(), BeliefMutationSealPersistenceError> {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_bytes(hasher, value.as_bytes())?;
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), BeliefMutationSealPersistenceError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| BeliefMutationSealPersistenceError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefMutationSealPersistenceError {
    AppliedMutationPreparedPairMismatch,
    MutationSealTemporalMismatch(BeliefMutationReceiptId),
    RecordSealBindingMismatch(BeliefMutationReceiptId),
    InvalidSealedClaim(ClaimId),
    InvalidSealedEvidence(super::claim_evidence::EvidenceId),
    SealedEvidenceNotOrdered,
    RecordDigestMismatch(BeliefMutationReceiptId),
    CapturePredatesMutationCapsule {
        captured_at_cycle: u64,
        mutation_capture_cycle: u64,
    },
    MutationSealCountMismatch { mutations: usize, seals: usize },
    DuplicateMutationSeal(BeliefMutationReceiptId),
    DuplicateRevisionSeal(BeliefRevisionReceiptId),
    SealRecordPostdatesCapture {
        mutation_id: BeliefMutationReceiptId,
        applied_at_cycle: u64,
        captured_at_cycle: u64,
    },
    MissingMutationSeal(BeliefMutationReceiptId),
    MutationRecordBindingMismatch(BeliefMutationReceiptId),
    MutationBindingDigestMismatch(BeliefMutationReceiptId),
    SealedClaimMissing(ClaimId),
    SealedClaimChanged(ClaimId),
    SealedEvidenceMissing(super::claim_evidence::EvidenceId),
    SealedEvidenceChanged(super::claim_evidence::EvidenceId),
    MutationCaptureCycleMismatch,
    ObservationPredatesCapture {
        observed_at_cycle: u64,
        captured_at_cycle: u64,
    },
    MutationSealsNotOrdered,
    CapsuleDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for BeliefMutationSealPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief mutation seal persistence rejected: {self:?}")
    }
}

impl Error for BeliefMutationSealPersistenceError {}
