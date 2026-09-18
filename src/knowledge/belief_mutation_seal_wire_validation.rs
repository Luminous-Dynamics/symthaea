// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent cross-component validation for EKM-057 mutation evidence seals.
//!
//! The seal sidecar is not self-authenticating. This module recomputes each
//! EKM-056 mutation-binding and record digest from the independent base restart
//! mutation history plus the sealed census, then re-derives the capsule digest.
//!
//! Important boundary: this proves that the supplied base restart and supplied
//! seal sidecar are mutually consistent. It does **not** independently prove that
//! an untrusted sidecar contains the complete historical non-basis evidence census.
//! That stronger claim requires the original EKM-056 capsule digest to be bound by
//! a trusted restart manifest/checkpoint lineage.
//!
//! No EKM-056 capsule, quarantine, hydration object, or activation capability is
//! constructed here.

use super::belief_mutation_seal_wire::{
    BeliefMutationSealWireEncoding, BeliefMutationSealWireSnapshotV1,
    BeliefMutationSealWireVersion, WireBeliefMutationEvidenceSealV1,
};
use super::belief_revision_receipt::RevisionEvidenceSnapshot;
use super::claim_evidence::{ClaimKind, EvidenceKind, EvidencePolarity};
use super::epistemic_restart_wire::{WireEvidenceV1, WireMutationV1, WireRevisionReceiptV1};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_restart_wire_v2_validation::{
    EpistemicRestartWireV2ValidationError, EpistemicRestartWireV2Validator,
};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_VALIDATED_SEAL_RECORDS: usize = 1_000_000;
const MAX_VALIDATED_SEALED_EVIDENCE: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefMutationSealValidationVersion {
    V1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefMutationSealValidationReportV1 {
    version: BeliefMutationSealValidationVersion,
    mutation_count: usize,
    sealed_evidence_count: usize,
    base_capture_cycle: u64,
    seal_capture_cycle: u64,
    recomputed_capsule_digest: [u8; 32],
    cross_component_consistent: bool,
    historical_census_completeness_independently_proven: bool,
    capsule_construction_authorized: bool,
    hydration_authorized: bool,
    activation_authorized: bool,
}

impl BeliefMutationSealValidationReportV1 {
    pub fn version(&self) -> BeliefMutationSealValidationVersion {
        self.version
    }

    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    pub fn sealed_evidence_count(&self) -> usize {
        self.sealed_evidence_count
    }

    pub fn base_capture_cycle(&self) -> u64 {
        self.base_capture_cycle
    }

    pub fn seal_capture_cycle(&self) -> u64 {
        self.seal_capture_cycle
    }

    pub fn recomputed_capsule_digest(&self) -> [u8; 32] {
        self.recomputed_capsule_digest
    }

    /// True when the supplied base restart history and supplied seal sidecar agree
    /// under all independently reproducible EKM-056 bindings checked here.
    pub fn cross_component_consistent(&self) -> bool {
        self.cross_component_consistent
    }

    /// Deliberately false in EKM-058.
    ///
    /// An unsigned/untrusted sidecar can omit historical non-basis evidence and
    /// recompute its own digest. Absence of omitted historical evidence becomes a
    /// trustworthy claim only after the original EKM-056 capsule digest is bound
    /// into protected restart lineage.
    pub fn historical_census_completeness_independently_proven(&self) -> bool {
        self.historical_census_completeness_independently_proven
    }

    pub fn capsule_construction_authorized(&self) -> bool {
        self.capsule_construction_authorized
    }

    pub fn hydration_authorized(&self) -> bool {
        self.hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub struct BeliefMutationSealWireValidator;

impl BeliefMutationSealWireValidator {
    pub fn validate(
        base: &EpistemicRestartWireSnapshotV2,
        seals: &BeliefMutationSealWireSnapshotV1,
    ) -> Result<BeliefMutationSealValidationReportV1, BeliefMutationSealValidationError> {
        EpistemicRestartWireV2Validator::validate(base)
            .map_err(BeliefMutationSealValidationError::BaseRestartRejected)?;

        if seals.version != BeliefMutationSealWireVersion::V1
            || seals.encoding != BeliefMutationSealWireEncoding::ExplicitMutationSealFieldsV1
        {
            return Err(BeliefMutationSealValidationError::UnsupportedSealSchema);
        }
        if seals.records.len() > MAX_VALIDATED_SEAL_RECORDS {
            return Err(BeliefMutationSealValidationError::SealRecordCountTooLarge {
                actual: seals.records.len(),
                maximum: MAX_VALIDATED_SEAL_RECORDS,
            });
        }
        if seals.linked_mutation_capture_cycle != base.base.captured_at_cycle {
            return Err(BeliefMutationSealValidationError::MutationCaptureCycleMismatch {
                base: base.base.captured_at_cycle,
                seal: seals.linked_mutation_capture_cycle,
            });
        }
        if seals.captured_at_cycle < seals.linked_mutation_capture_cycle {
            return Err(BeliefMutationSealValidationError::SealCapturePredatesMutationCapture {
                seal_capture_cycle: seals.captured_at_cycle,
                mutation_capture_cycle: seals.linked_mutation_capture_cycle,
            });
        }
        if seals.records.len() != base.base.mutations.len() {
            return Err(BeliefMutationSealValidationError::MutationSealCountMismatch {
                mutations: base.base.mutations.len(),
                seals: seals.records.len(),
            });
        }

        let mut seen_mutations = HashSet::new();
        let mut seen_revisions = HashSet::new();
        let mut previous_mutation_id = None;
        let mut sealed_evidence_count = 0usize;
        let mut expected_record_digests = Vec::with_capacity(seals.records.len());

        for record in &seals.records {
            if !seen_mutations.insert(record.mutation_id) {
                return Err(BeliefMutationSealValidationError::DuplicateMutationSeal(
                    record.mutation_id.0,
                ));
            }
            if !seen_revisions.insert(record.source_revision_receipt_id) {
                return Err(BeliefMutationSealValidationError::DuplicateRevisionSeal(
                    record.source_revision_receipt_id.0,
                ));
            }
            if let Some(previous) = previous_mutation_id {
                if record.mutation_id.0 <= previous {
                    return Err(BeliefMutationSealValidationError::MutationSealsNotOrdered);
                }
            }
            previous_mutation_id = Some(record.mutation_id.0);

            sealed_evidence_count = sealed_evidence_count
                .checked_add(record.evidence.len())
                .ok_or(BeliefMutationSealValidationError::LengthOverflow)?;
            if sealed_evidence_count > MAX_VALIDATED_SEALED_EVIDENCE {
                return Err(BeliefMutationSealValidationError::SealedEvidenceCountTooLarge {
                    actual: sealed_evidence_count,
                    maximum: MAX_VALIDATED_SEALED_EVIDENCE,
                });
            }

            let mutation = base
                .base
                .mutations
                .iter()
                .find(|mutation| mutation.id == record.mutation_id)
                .ok_or(BeliefMutationSealValidationError::MutationMissing(
                    record.mutation_id.0,
                ))?;
            let revision = base
                .base
                .revisions
                .iter()
                .find(|revision| revision.id == record.source_revision_receipt_id)
                .ok_or(BeliefMutationSealValidationError::RevisionMissing(
                    record.source_revision_receipt_id.0,
                ))?;

            validate_record_identity(record, mutation, revision, seals.captured_at_cycle)?;
            validate_sealed_claim(record, base)?;
            validate_sealed_evidence(record, base)?;
            validate_revision_basis_subset(record, revision)?;

            let mutation_binding = digest_mutation(mutation)?;
            let expected_record_digest = digest_record(record, mutation_binding)?;
            if expected_record_digest != record.record_digest {
                return Err(BeliefMutationSealValidationError::RecordDigestMismatch(
                    record.mutation_id.0,
                ));
            }
            expected_record_digests.push(expected_record_digest);
        }

        let expected_capsule_digest = digest_capsule(
            seals.captured_at_cycle,
            seals.linked_mutation_capture_cycle,
            &expected_record_digests,
        )?;
        if expected_capsule_digest != seals.claimed_capsule_digest {
            return Err(BeliefMutationSealValidationError::CapsuleDigestMismatch);
        }

        Ok(BeliefMutationSealValidationReportV1 {
            version: BeliefMutationSealValidationVersion::V1,
            mutation_count: seals.records.len(),
            sealed_evidence_count,
            base_capture_cycle: base.base.captured_at_cycle,
            seal_capture_cycle: seals.captured_at_cycle,
            recomputed_capsule_digest: expected_capsule_digest,
            cross_component_consistent: true,
            historical_census_completeness_independently_proven: false,
            capsule_construction_authorized: false,
            hydration_authorized: false,
            activation_authorized: false,
        })
    }
}

fn validate_record_identity(
    record: &WireBeliefMutationEvidenceSealV1,
    mutation: &WireMutationV1,
    revision: &WireRevisionReceiptV1,
    seal_capture_cycle: u64,
) -> Result<(), BeliefMutationSealValidationError> {
    if record.source_revision_receipt_id != mutation.source_revision_receipt_id
        || record.claim_id != mutation.claim_id
        || record.applied_at_cycle != mutation.applied_at_cycle
        || record.source_revision_receipt_id != revision.id
        || record.claim_id != revision.claim_id
    {
        return Err(BeliefMutationSealValidationError::RecordMutationBindingMismatch(
            record.mutation_id.0,
        ));
    }
    if !revision.decision_eligible {
        return Err(BeliefMutationSealValidationError::MutationReferencesRejectedRevision(
            revision.id.0,
        ));
    }
    if revision.evaluated_at_cycle != record.sealed_at_cycle
        || record.sealed_at_cycle > mutation.authorized_at_cycle
        || mutation.authorized_at_cycle > mutation.applied_at_cycle
        || mutation.applied_at_cycle > seal_capture_cycle
    {
        return Err(BeliefMutationSealValidationError::RecordTemporalMismatch(
            record.mutation_id.0,
        ));
    }
    Ok(())
}

fn validate_sealed_claim(
    record: &WireBeliefMutationEvidenceSealV1,
    base: &EpistemicRestartWireSnapshotV2,
) -> Result<(), BeliefMutationSealValidationError> {
    if record.claim.claim_id != record.claim_id {
        return Err(BeliefMutationSealValidationError::SealedClaimBindingMismatch(
            record.mutation_id.0,
        ));
    }
    let live = base
        .base
        .claims
        .iter()
        .find(|claim| claim.id == record.claim_id)
        .ok_or(BeliefMutationSealValidationError::SealedClaimMissing(
            record.claim_id.0,
        ))?;
    if live.statement != record.claim.statement
        || live.kind != record.claim.kind
        || live.domain != record.claim.domain
        || live.scope != record.claim.scope
        || live.created_at_cycle != record.claim.created_at_cycle
        || record.claim.created_at_cycle > record.sealed_at_cycle
    {
        return Err(BeliefMutationSealValidationError::SealedClaimChanged(
            record.claim_id.0,
        ));
    }
    Ok(())
}

fn validate_sealed_evidence(
    record: &WireBeliefMutationEvidenceSealV1,
    base: &EpistemicRestartWireSnapshotV2,
) -> Result<(), BeliefMutationSealValidationError> {
    let mut previous = None;
    for evidence in &record.evidence {
        if evidence.claim_id != record.claim_id
            || evidence.observed_at_cycle > record.sealed_at_cycle
        {
            return Err(BeliefMutationSealValidationError::InvalidSealedEvidence(
                evidence.evidence_id.0,
            ));
        }
        if let Some(previous_id) = previous {
            if evidence.evidence_id.0 <= previous_id {
                return Err(BeliefMutationSealValidationError::SealedEvidenceNotOrdered(
                    record.mutation_id.0,
                ));
            }
        }
        previous = Some(evidence.evidence_id.0);
        let live = base
            .base
            .evidence
            .iter()
            .find(|candidate| candidate.id == evidence.evidence_id)
            .ok_or(BeliefMutationSealValidationError::SealedEvidenceMissing(
                evidence.evidence_id.0,
            ))?;
        if !evidence_matches(evidence, live) {
            return Err(BeliefMutationSealValidationError::SealedEvidenceChanged(
                evidence.evidence_id.0,
            ));
        }
    }
    Ok(())
}

fn validate_revision_basis_subset(
    record: &WireBeliefMutationEvidenceSealV1,
    revision: &WireRevisionReceiptV1,
) -> Result<(), BeliefMutationSealValidationError> {
    for basis in &revision.basis {
        let expected = basis.snapshot.as_ref().ok_or(
            BeliefMutationSealValidationError::EligibleRevisionBasisMissingSnapshot {
                revision_id: revision.id.0,
                evidence_id: basis.requested_id.0,
            },
        )?;
        let sealed = record
            .evidence
            .iter()
            .find(|evidence| evidence.evidence_id == basis.requested_id)
            .ok_or(BeliefMutationSealValidationError::RevisionBasisMissingFromSeal {
                revision_id: revision.id.0,
                evidence_id: basis.requested_id.0,
            })?;
        if sealed != expected {
            return Err(BeliefMutationSealValidationError::RevisionBasisSealMismatch {
                revision_id: revision.id.0,
                evidence_id: basis.requested_id.0,
            });
        }
    }
    Ok(())
}

fn evidence_matches(snapshot: &RevisionEvidenceSnapshot, live: &WireEvidenceV1) -> bool {
    snapshot.evidence_id == live.id
        && snapshot.claim_id == live.claim_id
        && snapshot.kind == live.kind
        && snapshot.polarity == live.polarity
        && snapshot.provenance_id == live.provenance_id
        && snapshot.observed_at_cycle == live.observed_at_cycle
        && snapshot.context == live.context
        && snapshot.method == live.method
}

fn digest_mutation(
    mutation: &WireMutationV1,
) -> Result<[u8; 32], BeliefMutationSealValidationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-binding-v1");
    hasher.update(&mutation.id.0.to_le_bytes());
    hasher.update(&mutation.source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&mutation.claim_id.0.to_le_bytes());
    hasher.update(&mutation.proposed_delta.to_bits().to_le_bytes());
    hasher.update(&mutation.support_before.get().to_bits().to_le_bytes());
    hasher.update(&mutation.support_after.get().to_bits().to_le_bytes());
    hasher.update(&mutation.state_revision_before.to_le_bytes());
    hasher.update(&mutation.state_revision_after.to_le_bytes());
    hash_bytes(&mut hasher, mutation.authorization_id.as_bytes())?;
    hash_bytes(&mut hasher, mutation.authority_label.as_bytes())?;
    hasher.update(&mutation.authorized_at_cycle.to_le_bytes());
    hasher.update(&mutation.applied_at_cycle.to_le_bytes());
    Ok(*hasher.finalize().as_bytes())
}

fn digest_record(
    record: &WireBeliefMutationEvidenceSealV1,
    mutation_binding: [u8; 32],
) -> Result<[u8; 32], BeliefMutationSealValidationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-evidence-seal-record-v1");
    hasher.update(&record.mutation_id.0.to_le_bytes());
    hasher.update(&record.source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&record.claim_id.0.to_le_bytes());
    hasher.update(&record.sealed_at_cycle.to_le_bytes());
    hasher.update(&record.applied_at_cycle.to_le_bytes());
    hasher.update(&mutation_binding);
    hash_seal(&mut hasher, record)?;
    Ok(*hasher.finalize().as_bytes())
}

fn hash_seal(
    hasher: &mut blake3::Hasher,
    record: &WireBeliefMutationEvidenceSealV1,
) -> Result<(), BeliefMutationSealValidationError> {
    // Mirrors EKM-056 `hash_seal`: the seal itself repeats the source receipt ID
    // and seal cycle after the outer persisted-record header.
    hasher.update(&record.source_revision_receipt_id.0.to_le_bytes());
    hasher.update(&record.sealed_at_cycle.to_le_bytes());
    hasher.update(&record.claim.claim_id.0.to_le_bytes());
    hash_bytes(hasher, record.claim.statement.as_bytes())?;
    hasher.update(&[claim_kind_tag(record.claim.kind)]);
    hash_optional_string(hasher, record.claim.domain.as_deref())?;
    hash_optional_string(hasher, record.claim.scope.as_deref())?;
    hasher.update(&record.claim.created_at_cycle.to_le_bytes());
    let count = u64::try_from(record.evidence.len())
        .map_err(|_| BeliefMutationSealValidationError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for evidence in &record.evidence {
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

fn digest_capsule(
    captured_at_cycle: u64,
    linked_mutation_capture_cycle: u64,
    record_digests: &[[u8; 32]],
) -> Result<[u8; 32], BeliefMutationSealValidationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-belief-mutation-evidence-seal-capsule-v1");
    hasher.update(&[1]);
    hasher.update(&captured_at_cycle.to_le_bytes());
    hasher.update(&linked_mutation_capture_cycle.to_le_bytes());
    let count = u64::try_from(record_digests.len())
        .map_err(|_| BeliefMutationSealValidationError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for digest in record_digests {
        hasher.update(digest);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_optional_string(
    hasher: &mut blake3::Hasher,
    value: Option<&str>,
) -> Result<(), BeliefMutationSealValidationError> {
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
) -> Result<(), BeliefMutationSealValidationError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| BeliefMutationSealValidationError::LengthOverflow)?;
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

#[derive(Debug)]
pub enum BeliefMutationSealValidationError {
    BaseRestartRejected(EpistemicRestartWireV2ValidationError),
    UnsupportedSealSchema,
    SealRecordCountTooLarge { actual: usize, maximum: usize },
    SealedEvidenceCountTooLarge { actual: usize, maximum: usize },
    MutationCaptureCycleMismatch { base: u64, seal: u64 },
    SealCapturePredatesMutationCapture {
        seal_capture_cycle: u64,
        mutation_capture_cycle: u64,
    },
    MutationSealCountMismatch { mutations: usize, seals: usize },
    DuplicateMutationSeal(u64),
    DuplicateRevisionSeal(u64),
    MutationSealsNotOrdered,
    MutationMissing(u64),
    RevisionMissing(u64),
    RecordMutationBindingMismatch(u64),
    MutationReferencesRejectedRevision(u64),
    RecordTemporalMismatch(u64),
    SealedClaimBindingMismatch(u64),
    SealedClaimMissing(u64),
    SealedClaimChanged(u64),
    InvalidSealedEvidence(u64),
    SealedEvidenceNotOrdered(u64),
    SealedEvidenceMissing(u64),
    SealedEvidenceChanged(u64),
    EligibleRevisionBasisMissingSnapshot { revision_id: u64, evidence_id: u64 },
    RevisionBasisMissingFromSeal { revision_id: u64, evidence_id: u64 },
    RevisionBasisSealMismatch { revision_id: u64, evidence_id: u64 },
    RecordDigestMismatch(u64),
    CapsuleDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for BeliefMutationSealValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "belief mutation seal validation rejected: {self:?}")
    }
}

impl Error for BeliefMutationSealValidationError {}
