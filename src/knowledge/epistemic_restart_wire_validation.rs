// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent semantic validation for decoded EKM restart wire snapshots.
//!
//! EKM-034 proves framing/checksum/parser bounds. This module treats the decoded
//! DTO as still untrusted and checks cross-record epistemic invariants without
//! constructing any operational ledger, capsule, support store, history, or
//! activation handle.

use super::belief_mutation_firewall::BeliefMutationReceiptId;
use super::belief_revision_receipt::BeliefRevisionReceiptId;
use super::claim_evidence::{ClaimId, EvidenceId, ProvenanceId};
use super::epistemic_restart_wire::{
    EpistemicRestartWireEncoding, EpistemicRestartWireSnapshotV1, EpistemicRestartWireVersion,
    WireEvidenceV1, WireRevisionReceiptV1,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicRestartWireValidationReport {
    pub captured_at_cycle: u64,
    pub provenance_count: usize,
    pub claim_count: usize,
    pub evidence_count: usize,
    pub support_state_count: usize,
    pub mutation_count: usize,
    pub revision_count: usize,
    pub rejected_revision_count: usize,
}

pub struct EpistemicRestartWireValidator;

impl EpistemicRestartWireValidator {
    pub fn validate(
        snapshot: &EpistemicRestartWireSnapshotV1,
    ) -> Result<EpistemicRestartWireValidationReport, EpistemicRestartWireValidationError> {
        if snapshot.version != EpistemicRestartWireVersion::V1 {
            return Err(EpistemicRestartWireValidationError::UnsupportedVersion);
        }
        if snapshot.encoding
            != EpistemicRestartWireEncoding::ExplicitFieldsWithOpaqueRevisionPolicyDecisionV1
        {
            return Err(EpistemicRestartWireValidationError::UnsupportedEncoding);
        }
        if snapshot.manifest.captured_at_cycle != snapshot.captured_at_cycle {
            return Err(EpistemicRestartWireValidationError::CaptureCycleMismatch {
                envelope: snapshot.captured_at_cycle,
                manifest: snapshot.manifest.captured_at_cycle,
            });
        }

        validate_ledger(snapshot)?;
        validate_support_and_mutations(snapshot)?;
        validate_revisions(snapshot)?;
        validate_mutation_revision_links(snapshot)?;

        Ok(EpistemicRestartWireValidationReport {
            captured_at_cycle: snapshot.captured_at_cycle,
            provenance_count: snapshot.provenance.len(),
            claim_count: snapshot.claims.len(),
            evidence_count: snapshot.evidence.len(),
            support_state_count: snapshot.support_states.len(),
            mutation_count: snapshot.mutations.len(),
            revision_count: snapshot.revisions.len(),
            rejected_revision_count: snapshot
                .revisions
                .iter()
                .filter(|receipt| !receipt.decision_eligible)
                .count(),
        })
    }
}

fn validate_ledger(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    if snapshot.manifest.provenance_count != snapshot.provenance.len() as u64
        || snapshot.manifest.claim_count != snapshot.claims.len() as u64
        || snapshot.manifest.evidence_count != snapshot.evidence.len() as u64
    {
        return Err(EpistemicRestartWireValidationError::ManifestCountMismatch);
    }

    validate_contiguous_provenance_ids(snapshot)?;
    validate_contiguous_claim_ids(snapshot)?;
    validate_contiguous_evidence_ids(snapshot)?;

    if snapshot.manifest.next_provenance_id.0 != snapshot.provenance.len() as u64 + 1 {
        return Err(EpistemicRestartWireValidationError::NextProvenanceIdMismatch);
    }
    if snapshot.manifest.next_claim_id.0 != snapshot.claims.len() as u64 + 1 {
        return Err(EpistemicRestartWireValidationError::NextClaimIdMismatch);
    }
    if snapshot.manifest.next_evidence_id.0 != snapshot.evidence.len() as u64 + 1 {
        return Err(EpistemicRestartWireValidationError::NextEvidenceIdMismatch);
    }

    let provenance_ids = snapshot
        .provenance
        .iter()
        .map(|record| record.id)
        .collect::<BTreeSet<_>>();
    let claim_ids = snapshot
        .claims
        .iter()
        .map(|record| record.id)
        .collect::<BTreeSet<_>>();
    let evidence_ids = snapshot
        .evidence
        .iter()
        .map(|record| record.id)
        .collect::<BTreeSet<_>>();

    for provenance in &snapshot.provenance {
        if provenance.recorded_at_cycle > snapshot.captured_at_cycle {
            return Err(EpistemicRestartWireValidationError::ProvenancePostdatesCapture(
                provenance.id,
            ));
        }
        for parent in &provenance.parent_ids {
            if !provenance_ids.contains(parent) {
                return Err(EpistemicRestartWireValidationError::UnknownProvenanceParent {
                    provenance_id: provenance.id,
                    parent_id: *parent,
                });
            }
            // The live ledger only accepts parents that already exist. Stable IDs
            // therefore make a parent >= child impossible for a genuine capture.
            if parent.0 >= provenance.id.0 {
                return Err(EpistemicRestartWireValidationError::ProvenanceParentNotPrior {
                    provenance_id: provenance.id,
                    parent_id: *parent,
                });
            }
        }
    }

    let mut actual_by_claim: BTreeMap<ClaimId, Vec<EvidenceId>> = BTreeMap::new();
    for evidence in &snapshot.evidence {
        validate_evidence_record(evidence, snapshot.captured_at_cycle)?;
        if !claim_ids.contains(&evidence.claim_id) {
            return Err(EpistemicRestartWireValidationError::EvidenceUnknownClaim {
                evidence_id: evidence.id,
                claim_id: evidence.claim_id,
            });
        }
        if !provenance_ids.contains(&evidence.provenance_id) {
            return Err(EpistemicRestartWireValidationError::EvidenceUnknownProvenance {
                evidence_id: evidence.id,
                provenance_id: evidence.provenance_id,
            });
        }
        actual_by_claim
            .entry(evidence.claim_id)
            .or_default()
            .push(evidence.id);
    }

    for claim in &snapshot.claims {
        if claim.created_at_cycle > snapshot.captured_at_cycle {
            return Err(EpistemicRestartWireValidationError::ClaimPostdatesCapture(
                claim.id,
            ));
        }
        let mut seen = HashSet::new();
        for evidence_id in &claim.evidence_ids {
            if !seen.insert(*evidence_id) {
                return Err(EpistemicRestartWireValidationError::DuplicateClaimEvidenceLink {
                    claim_id: claim.id,
                    evidence_id: *evidence_id,
                });
            }
            if !evidence_ids.contains(evidence_id) {
                return Err(EpistemicRestartWireValidationError::ClaimLinksUnknownEvidence {
                    claim_id: claim.id,
                    evidence_id: *evidence_id,
                });
            }
        }
        let actual = actual_by_claim.remove(&claim.id).unwrap_or_default();
        if claim.evidence_ids != actual {
            return Err(EpistemicRestartWireValidationError::ClaimEvidenceCensusMismatch(
                claim.id,
            ));
        }
    }
    if let Some((claim_id, _)) = actual_by_claim.into_iter().next() {
        return Err(EpistemicRestartWireValidationError::EvidenceTargetsMissingClaim(
            claim_id,
        ));
    }

    Ok(())
}

fn validate_evidence_record(
    evidence: &WireEvidenceV1,
    captured_at_cycle: u64,
) -> Result<(), EpistemicRestartWireValidationError> {
    if evidence.observed_at_cycle > captured_at_cycle {
        return Err(EpistemicRestartWireValidationError::EvidencePostdatesCapture(
            evidence.id,
        ));
    }
    Ok(())
}

fn validate_contiguous_provenance_ids(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    for (index, record) in snapshot.provenance.iter().enumerate() {
        let expected = index as u64 + 1;
        if record.id.0 != expected {
            return Err(EpistemicRestartWireValidationError::ProvenanceIdLineageGap {
                expected: ProvenanceId(expected),
                actual: record.id,
            });
        }
    }
    Ok(())
}

fn validate_contiguous_claim_ids(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    for (index, record) in snapshot.claims.iter().enumerate() {
        let expected = index as u64 + 1;
        if record.id.0 != expected {
            return Err(EpistemicRestartWireValidationError::ClaimIdLineageGap {
                expected: ClaimId(expected),
                actual: record.id,
            });
        }
    }
    Ok(())
}

fn validate_contiguous_evidence_ids(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    for (index, record) in snapshot.evidence.iter().enumerate() {
        let expected = index as u64 + 1;
        if record.id.0 != expected {
            return Err(EpistemicRestartWireValidationError::EvidenceIdLineageGap {
                expected: EvidenceId(expected),
                actual: record.id,
            });
        }
    }
    Ok(())
}

fn validate_support_and_mutations(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    let claim_ids = snapshot
        .claims
        .iter()
        .map(|record| record.id)
        .collect::<HashSet<_>>();
    let mut support_by_claim = HashMap::new();
    let mut previous_support_claim = None;
    for state in &snapshot.support_states {
        if !claim_ids.contains(&state.claim_id) {
            return Err(EpistemicRestartWireValidationError::SupportStateUnknownClaim(
                state.claim_id,
            ));
        }
        if let Some(previous) = previous_support_claim {
            if state.claim_id.0 <= previous {
                return Err(EpistemicRestartWireValidationError::SupportStatesNotOrdered);
            }
        }
        previous_support_claim = Some(state.claim_id.0);
        if state.initialized_at_cycle > state.last_updated_cycle
            || state.last_updated_cycle > snapshot.captured_at_cycle
        {
            return Err(EpistemicRestartWireValidationError::SupportStateTemporalMismatch(
                state.claim_id,
            ));
        }
        if support_by_claim.insert(state.claim_id, state).is_some() {
            return Err(EpistemicRestartWireValidationError::DuplicateSupportState(
                state.claim_id,
            ));
        }
    }

    if snapshot.consumed_authorization_count != snapshot.mutations.len() as u64 {
        return Err(
            EpistemicRestartWireValidationError::AuthorizationMutationCountMismatch {
                consumed: snapshot.consumed_authorization_count,
                mutations: snapshot.mutations.len(),
            },
        );
    }

    let mut mutation_ids = HashSet::new();
    let mut authorization_ids = HashSet::new();
    let mut source_revision_ids = HashSet::new();
    let mut previous_mutation_id = None;
    let mut by_claim: HashMap<ClaimId, Vec<_>> = HashMap::new();

    for mutation in &snapshot.mutations {
        if !mutation.proposed_delta.is_finite() || !(-1.0..=1.0).contains(&mutation.proposed_delta)
        {
            return Err(EpistemicRestartWireValidationError::InvalidMutationDelta {
                mutation_id: mutation.id,
                delta: mutation.proposed_delta,
            });
        }
        if !mutation_ids.insert(mutation.id) {
            return Err(EpistemicRestartWireValidationError::DuplicateMutationId(
                mutation.id,
            ));
        }
        if !authorization_ids.insert(mutation.authorization_id.clone()) {
            return Err(EpistemicRestartWireValidationError::DuplicateAuthorizationId(
                mutation.authorization_id.clone(),
            ));
        }
        if !source_revision_ids.insert(mutation.source_revision_receipt_id) {
            return Err(
                EpistemicRestartWireValidationError::DuplicateSourceRevisionReceipt(
                    mutation.source_revision_receipt_id,
                ),
            );
        }
        if let Some(previous) = previous_mutation_id {
            if mutation.id.0 <= previous {
                return Err(EpistemicRestartWireValidationError::MutationIdsNotMonotonic);
            }
        }
        previous_mutation_id = Some(mutation.id.0);
        if mutation.authorized_at_cycle > mutation.applied_at_cycle
            || mutation.applied_at_cycle > snapshot.captured_at_cycle
        {
            return Err(EpistemicRestartWireValidationError::MutationTemporalMismatch(
                mutation.id,
            ));
        }
        let expected_after_revision = mutation
            .state_revision_before
            .checked_add(1)
            .ok_or(EpistemicRestartWireValidationError::MutationRevisionOverflow(
                mutation.id,
            ))?;
        if mutation.state_revision_after != expected_after_revision {
            return Err(EpistemicRestartWireValidationError::MutationRevisionStepMismatch(
                mutation.id,
            ));
        }
        let expected_after = mutation.support_before.get() + mutation.proposed_delta;
        if (mutation.support_after.get() - expected_after).abs() > 1e-6 {
            return Err(EpistemicRestartWireValidationError::MutationDeltaMismatch(
                mutation.id,
            ));
        }
        if !support_by_claim.contains_key(&mutation.claim_id) {
            return Err(EpistemicRestartWireValidationError::MutationWithoutSupportState(
                mutation.claim_id,
            ));
        }
        by_claim.entry(mutation.claim_id).or_default().push(mutation);
    }

    for (claim_id, state) in support_by_claim {
        let mut chain = by_claim.remove(&claim_id).unwrap_or_default();
        chain.sort_by_key(|mutation| mutation.state_revision_after);
        if chain.is_empty() {
            if state.revision != 0
                || state.last_mutation_id.is_some()
                || state.baseline_support != state.current_support
                || state.initialized_at_cycle != state.last_updated_cycle
            {
                return Err(EpistemicRestartWireValidationError::UnmutatedSupportStateInvalid(
                    claim_id,
                ));
            }
            continue;
        }

        if chain[0].state_revision_before != 0
            || chain[0].support_before != state.baseline_support
        {
            return Err(EpistemicRestartWireValidationError::MutationChainBaselineMismatch(
                claim_id,
            ));
        }
        let mut expected_revision = 0u64;
        let mut expected_support = state.baseline_support;
        for mutation in &chain {
            let next_revision = expected_revision
                .checked_add(1)
                .ok_or(EpistemicRestartWireValidationError::SupportRevisionOverflow(
                    claim_id,
                ))?;
            if mutation.state_revision_before != expected_revision
                || mutation.state_revision_after != next_revision
                || mutation.support_before != expected_support
            {
                return Err(EpistemicRestartWireValidationError::MutationChainBroken(
                    claim_id,
                ));
            }
            expected_revision = mutation.state_revision_after;
            expected_support = mutation.support_after;
        }
        let latest = chain.last().expect("non-empty chain");
        if state.revision != latest.state_revision_after
            || state.current_support != latest.support_after
            || state.last_mutation_id != Some(latest.id)
            || state.last_updated_cycle != latest.applied_at_cycle
        {
            return Err(EpistemicRestartWireValidationError::SupportStateFinalMismatch(
                claim_id,
            ));
        }
    }

    if let Some((claim_id, _)) = by_claim.into_iter().next() {
        return Err(EpistemicRestartWireValidationError::MutationWithoutSupportState(
            claim_id,
        ));
    }

    Ok(())
}

fn validate_revisions(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    let evidence_by_id = snapshot
        .evidence
        .iter()
        .map(|record| (record.id, record))
        .collect::<HashMap<_, _>>();
    let provenance_by_id = snapshot
        .provenance
        .iter()
        .map(|record| (record.id, record))
        .collect::<HashMap<_, _>>();

    let mut previous_cycle = None;
    for (index, receipt) in snapshot.revisions.iter().enumerate() {
        let expected_id = BeliefRevisionReceiptId(index as u64 + 1);
        if receipt.id != expected_id {
            return Err(EpistemicRestartWireValidationError::RevisionIdLineageGap {
                expected: expected_id,
                actual: receipt.id,
            });
        }
        if receipt.evaluated_at_cycle > snapshot.captured_at_cycle {
            return Err(EpistemicRestartWireValidationError::RevisionPostdatesCapture(
                receipt.id,
            ));
        }
        if let Some(previous) = previous_cycle {
            if receipt.evaluated_at_cycle < previous {
                return Err(EpistemicRestartWireValidationError::RevisionCyclesRegressed);
            }
        }
        previous_cycle = Some(receipt.evaluated_at_cycle);
        if receipt.policy_debug.is_empty() || receipt.decision_debug.is_empty() {
            return Err(EpistemicRestartWireValidationError::OpaqueRevisionFieldEmpty(
                receipt.id,
            ));
        }

        let mut requested = HashSet::new();
        for basis in &receipt.basis {
            if !requested.insert(basis.requested_id) {
                return Err(EpistemicRestartWireValidationError::DuplicateRevisionBasisId {
                    receipt_id: receipt.id,
                    evidence_id: basis.requested_id,
                });
            }
            if let Some(record) = &basis.snapshot {
                if record.evidence_id != basis.requested_id {
                    return Err(EpistemicRestartWireValidationError::RevisionBasisIdMismatch {
                        receipt_id: receipt.id,
                        requested: basis.requested_id,
                        snapshotted: record.evidence_id,
                    });
                }
                if record.observed_at_cycle > receipt.evaluated_at_cycle {
                    return Err(EpistemicRestartWireValidationError::RevisionPredatesBasisEvidence {
                        receipt_id: receipt.id,
                        evidence_id: record.evidence_id,
                    });
                }
                let Some(live) = evidence_by_id.get(&record.evidence_id) else {
                    return Err(EpistemicRestartWireValidationError::RevisionBasisEvidenceMissing {
                        receipt_id: receipt.id,
                        evidence_id: record.evidence_id,
                    });
                };
                if !revision_snapshot_matches_wire(record, live) {
                    return Err(
                        EpistemicRestartWireValidationError::RevisionBasisSemanticMismatch {
                            receipt_id: receipt.id,
                            evidence_id: record.evidence_id,
                        },
                    );
                }
            }
        }
        let mut duplicate_attempts = HashSet::new();
        for evidence_id in &receipt.duplicate_basis_evidence_ids {
            if !duplicate_attempts.insert(*evidence_id) {
                return Err(
                    EpistemicRestartWireValidationError::DuplicateDuplicateBasisDiagnostic {
                        receipt_id: receipt.id,
                        evidence_id: *evidence_id,
                    },
                );
            }
        }
        if let Some(uncertainty) = &receipt.uncertainty {
            if uncertainty.assessed_at_cycle > receipt.evaluated_at_cycle {
                return Err(
                    EpistemicRestartWireValidationError::RevisionPredatesUncertaintyAssessment(
                        receipt.id,
                    ),
                );
            }
        }

        let roots = declared_roots_for_receipt(receipt, &provenance_by_id);
        if roots.len() as u64 != receipt.declared_provenance_root_count {
            return Err(EpistemicRestartWireValidationError::RevisionRootCountMismatch {
                receipt_id: receipt.id,
                declared: receipt.declared_provenance_root_count,
                actual: roots.len(),
            });
        }
    }

    let expected_next = snapshot.revisions.len() as u64 + 1;
    if snapshot.revision_next_receipt_id.0 != expected_next {
        return Err(EpistemicRestartWireValidationError::NextRevisionReceiptIdMismatch {
            expected: BeliefRevisionReceiptId(expected_next),
            actual: snapshot.revision_next_receipt_id,
        });
    }
    Ok(())
}

fn revision_snapshot_matches_wire(
    snapshot: &super::belief_revision_receipt::RevisionEvidenceSnapshot,
    live: &WireEvidenceV1,
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

fn declared_roots_for_receipt(
    receipt: &WireRevisionReceiptV1,
    provenance: &HashMap<ProvenanceId, &super::epistemic_restart_wire::WireProvenanceV1>,
) -> BTreeSet<ProvenanceId> {
    let mut roots = BTreeSet::new();
    let mut visited = HashSet::new();
    let mut stack = Vec::new();

    for basis in &receipt.basis {
        let Some(snapshot) = &basis.snapshot else {
            continue;
        };
        if snapshot.claim_id == receipt.claim_id {
            stack.push(snapshot.provenance_id);
        }
    }

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(record) = provenance.get(&current) else {
            continue;
        };
        if record.parent_ids.is_empty() {
            roots.insert(current);
        } else {
            stack.extend(record.parent_ids.iter().copied());
        }
    }
    roots
}

fn validate_mutation_revision_links(
    snapshot: &EpistemicRestartWireSnapshotV1,
) -> Result<(), EpistemicRestartWireValidationError> {
    let revisions = snapshot
        .revisions
        .iter()
        .map(|receipt| (receipt.id, receipt))
        .collect::<HashMap<_, _>>();
    for mutation in &snapshot.mutations {
        let receipt = revisions
            .get(&mutation.source_revision_receipt_id)
            .ok_or(EpistemicRestartWireValidationError::MutationDecisionMissing {
                mutation_id: mutation.id,
                receipt_id: mutation.source_revision_receipt_id,
            })?;
        if !receipt.decision_eligible {
            return Err(EpistemicRestartWireValidationError::MutationReferencesRejectedDecision {
                mutation_id: mutation.id,
                receipt_id: receipt.id,
            });
        }
        if receipt.claim_id != mutation.claim_id {
            return Err(EpistemicRestartWireValidationError::MutationDecisionClaimMismatch {
                mutation_id: mutation.id,
                decision_claim: receipt.claim_id,
                mutation_claim: mutation.claim_id,
            });
        }
        if receipt.proposed_delta.to_bits() != mutation.proposed_delta.to_bits() {
            return Err(EpistemicRestartWireValidationError::MutationDecisionDeltaMismatch(
                mutation.id,
            ));
        }
        if mutation.authorized_at_cycle < receipt.evaluated_at_cycle {
            return Err(EpistemicRestartWireValidationError::AuthorizationPredatesDecision(
                mutation.id,
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartWireValidationError {
    UnsupportedVersion,
    UnsupportedEncoding,
    CaptureCycleMismatch {
        envelope: u64,
        manifest: u64,
    },
    ManifestCountMismatch,
    ProvenanceIdLineageGap {
        expected: ProvenanceId,
        actual: ProvenanceId,
    },
    ClaimIdLineageGap {
        expected: ClaimId,
        actual: ClaimId,
    },
    EvidenceIdLineageGap {
        expected: EvidenceId,
        actual: EvidenceId,
    },
    NextProvenanceIdMismatch,
    NextClaimIdMismatch,
    NextEvidenceIdMismatch,
    ProvenancePostdatesCapture(ProvenanceId),
    UnknownProvenanceParent {
        provenance_id: ProvenanceId,
        parent_id: ProvenanceId,
    },
    ProvenanceParentNotPrior {
        provenance_id: ProvenanceId,
        parent_id: ProvenanceId,
    },
    ClaimPostdatesCapture(ClaimId),
    EvidencePostdatesCapture(EvidenceId),
    EvidenceUnknownClaim {
        evidence_id: EvidenceId,
        claim_id: ClaimId,
    },
    EvidenceUnknownProvenance {
        evidence_id: EvidenceId,
        provenance_id: ProvenanceId,
    },
    DuplicateClaimEvidenceLink {
        claim_id: ClaimId,
        evidence_id: EvidenceId,
    },
    ClaimLinksUnknownEvidence {
        claim_id: ClaimId,
        evidence_id: EvidenceId,
    },
    ClaimEvidenceCensusMismatch(ClaimId),
    EvidenceTargetsMissingClaim(ClaimId),
    SupportStateUnknownClaim(ClaimId),
    SupportStatesNotOrdered,
    DuplicateSupportState(ClaimId),
    SupportStateTemporalMismatch(ClaimId),
    AuthorizationMutationCountMismatch {
        consumed: u64,
        mutations: usize,
    },
    InvalidMutationDelta {
        mutation_id: BeliefMutationReceiptId,
        delta: f32,
    },
    DuplicateMutationId(BeliefMutationReceiptId),
    DuplicateAuthorizationId(String),
    DuplicateSourceRevisionReceipt(BeliefRevisionReceiptId),
    MutationIdsNotMonotonic,
    MutationTemporalMismatch(BeliefMutationReceiptId),
    MutationRevisionOverflow(BeliefMutationReceiptId),
    MutationRevisionStepMismatch(BeliefMutationReceiptId),
    MutationDeltaMismatch(BeliefMutationReceiptId),
    MutationWithoutSupportState(ClaimId),
    UnmutatedSupportStateInvalid(ClaimId),
    MutationChainBaselineMismatch(ClaimId),
    SupportRevisionOverflow(ClaimId),
    MutationChainBroken(ClaimId),
    SupportStateFinalMismatch(ClaimId),
    RevisionIdLineageGap {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    RevisionPostdatesCapture(BeliefRevisionReceiptId),
    RevisionCyclesRegressed,
    OpaqueRevisionFieldEmpty(BeliefRevisionReceiptId),
    DuplicateRevisionBasisId {
        receipt_id: BeliefRevisionReceiptId,
        evidence_id: EvidenceId,
    },
    RevisionBasisIdMismatch {
        receipt_id: BeliefRevisionReceiptId,
        requested: EvidenceId,
        snapshotted: EvidenceId,
    },
    RevisionPredatesBasisEvidence {
        receipt_id: BeliefRevisionReceiptId,
        evidence_id: EvidenceId,
    },
    RevisionBasisEvidenceMissing {
        receipt_id: BeliefRevisionReceiptId,
        evidence_id: EvidenceId,
    },
    RevisionBasisSemanticMismatch {
        receipt_id: BeliefRevisionReceiptId,
        evidence_id: EvidenceId,
    },
    DuplicateDuplicateBasisDiagnostic {
        receipt_id: BeliefRevisionReceiptId,
        evidence_id: EvidenceId,
    },
    RevisionPredatesUncertaintyAssessment(BeliefRevisionReceiptId),
    RevisionRootCountMismatch {
        receipt_id: BeliefRevisionReceiptId,
        declared: u64,
        actual: usize,
    },
    NextRevisionReceiptIdMismatch {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    MutationDecisionMissing {
        mutation_id: BeliefMutationReceiptId,
        receipt_id: BeliefRevisionReceiptId,
    },
    MutationReferencesRejectedDecision {
        mutation_id: BeliefMutationReceiptId,
        receipt_id: BeliefRevisionReceiptId,
    },
    MutationDecisionClaimMismatch {
        mutation_id: BeliefMutationReceiptId,
        decision_claim: ClaimId,
        mutation_claim: ClaimId,
    },
    MutationDecisionDeltaMismatch(BeliefMutationReceiptId),
    AuthorizationPredatesDecision(BeliefMutationReceiptId),
}

impl fmt::Display for EpistemicRestartWireValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart wire semantic validation failed: {self:?}")
    }
}

impl Error for EpistemicRestartWireValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BoundedWeight, ClaimKind, EpistemicRestartWireEncoding, EpistemicRestartWireSnapshotV1,
        EpistemicRestartWireVersion, EvidenceKind, EvidencePolarity, RevisionEvidenceSnapshot,
        WireClaimV1, WireEvidenceV1, WireManifestSummaryV1, WireMutationV1, WireProvenanceV1,
        WireRevisionBasisV1, WireRevisionReceiptV1, WireSupportStateV1,
    };

    fn snapshot() -> EpistemicRestartWireSnapshotV1 {
        let evidence_snapshot = RevisionEvidenceSnapshot {
            evidence_id: EvidenceId(1),
            claim_id: ClaimId(1),
            kind: EvidenceKind::Measurement,
            polarity: EvidencePolarity::Supports,
            provenance_id: ProvenanceId(1),
            observed_at_cycle: 2,
            context: Some("fixture".into()),
            method: Some("protocol-v1".into()),
        };
        EpistemicRestartWireSnapshotV1 {
            version: EpistemicRestartWireVersion::V1,
            encoding:
                EpistemicRestartWireEncoding::ExplicitFieldsWithOpaqueRevisionPolicyDecisionV1,
            captured_at_cycle: 6,
            manifest: WireManifestSummaryV1 {
                captured_at_cycle: 6,
                claim_count: 1,
                evidence_count: 1,
                provenance_count: 1,
                next_claim_id: ClaimId(2),
                next_evidence_id: EvidenceId(2),
                next_provenance_id: ProvenanceId(2),
                ledger_digest: [1; 32],
                mutation_capsule_digest: [2; 32],
                revision_capsule_digest: [3; 32],
                manifest_digest: [4; 32],
            },
            provenance: vec![WireProvenanceV1 {
                id: ProvenanceId(1),
                source_label: "lab".into(),
                source_uri: None,
                content_hash: None,
                recorded_at_cycle: 1,
                parent_ids: vec![],
            }],
            claims: vec![WireClaimV1 {
                id: ClaimId(1),
                statement: "X predicts Y".into(),
                kind: ClaimKind::Predictive,
                domain: None,
                scope: None,
                created_at_cycle: 1,
                evidence_ids: vec![EvidenceId(1)],
            }],
            evidence: vec![WireEvidenceV1 {
                id: EvidenceId(1),
                claim_id: ClaimId(1),
                kind: EvidenceKind::Measurement,
                polarity: EvidencePolarity::Supports,
                provenance_id: ProvenanceId(1),
                observed_at_cycle: 2,
                context: Some("fixture".into()),
                method: Some("protocol-v1".into()),
            }],
            support_states: vec![WireSupportStateV1 {
                claim_id: ClaimId(1),
                baseline_support: BoundedWeight::new(0.5).unwrap(),
                current_support: BoundedWeight::new(0.6).unwrap(),
                revision: 1,
                initialized_at_cycle: 3,
                last_updated_cycle: 5,
                last_mutation_id: Some(BeliefMutationReceiptId(1)),
            }],
            mutations: vec![WireMutationV1 {
                id: BeliefMutationReceiptId(1),
                source_revision_receipt_id: BeliefRevisionReceiptId(1),
                claim_id: ClaimId(1),
                proposed_delta: 0.1,
                support_before: BoundedWeight::new(0.5).unwrap(),
                support_after: BoundedWeight::new(0.6).unwrap(),
                state_revision_before: 0,
                state_revision_after: 1,
                authorization_id: "auth-1".into(),
                authority_label: "test".into(),
                authorized_at_cycle: 4,
                applied_at_cycle: 5,
            }],
            consumed_authorization_count: 1,
            revision_next_receipt_id: BeliefRevisionReceiptId(2),
            revisions: vec![WireRevisionReceiptV1 {
                id: BeliefRevisionReceiptId(1),
                claim_id: ClaimId(1),
                proposed_delta: 0.1,
                rationale: "support".into(),
                basis: vec![WireRevisionBasisV1 {
                    requested_id: EvidenceId(1),
                    snapshot: Some(evidence_snapshot),
                }],
                duplicate_basis_evidence_ids: vec![],
                policy_debug: "policy".into(),
                calibration: None,
                uncertainty: None,
                decision_eligible: true,
                declared_provenance_root_count: 1,
                decision_debug: "decision".into(),
                evaluated_at_cycle: 3,
            }],
            wire_checksum: [9; 32],
        }
    }

    #[test]
    fn valid_snapshot_passes_independent_semantic_validation() {
        let snapshot = snapshot();
        let report = EpistemicRestartWireValidator::validate(&snapshot).unwrap();
        assert_eq!(report.claim_count, 1);
        assert_eq!(report.mutation_count, 1);
        assert_eq!(report.revision_count, 1);
        assert_eq!(report.rejected_revision_count, 0);
    }

    #[test]
    fn claim_evidence_census_tamper_is_rejected() {
        let mut snapshot = snapshot();
        snapshot.claims[0].evidence_ids.clear();
        assert_eq!(
            EpistemicRestartWireValidator::validate(&snapshot).unwrap_err(),
            EpistemicRestartWireValidationError::ClaimEvidenceCensusMismatch(ClaimId(1))
        );
    }

    #[test]
    fn mutation_cannot_reference_rejected_decision() {
        let mut snapshot = snapshot();
        snapshot.revisions[0].decision_eligible = false;
        assert!(matches!(
            EpistemicRestartWireValidator::validate(&snapshot),
            Err(EpistemicRestartWireValidationError::MutationReferencesRejectedDecision { .. })
        ));
    }

    #[test]
    fn provenance_root_count_is_independently_recomputed() {
        let mut snapshot = snapshot();
        snapshot.revisions[0].declared_provenance_root_count = 2;
        assert!(matches!(
            EpistemicRestartWireValidator::validate(&snapshot),
            Err(EpistemicRestartWireValidationError::RevisionRootCountMismatch { .. })
        ));
    }

    #[test]
    fn authorization_cannot_predate_decision() {
        let mut snapshot = snapshot();
        snapshot.mutations[0].authorized_at_cycle = 2;
        assert_eq!(
            EpistemicRestartWireValidator::validate(&snapshot).unwrap_err(),
            EpistemicRestartWireValidationError::AuthorizationPredatesDecision(
                BeliefMutationReceiptId(1)
            )
        );
    }

    #[test]
    fn max_revision_is_rejected_without_overflow() {
        let mut snapshot = snapshot();
        snapshot.mutations[0].state_revision_before = u64::MAX;
        snapshot.mutations[0].state_revision_after = 0;
        assert_eq!(
            EpistemicRestartWireValidator::validate(&snapshot).unwrap_err(),
            EpistemicRestartWireValidationError::MutationRevisionOverflow(
                BeliefMutationReceiptId(1)
            )
        );
    }
}
