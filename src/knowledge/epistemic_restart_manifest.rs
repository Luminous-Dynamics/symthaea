// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atomic read-only restart manifest for the EKM epistemic state.
//!
//! EKM-030 and EKM-031 define validated support/mutation and revision-decision
//! capsules. This module binds those capsules to the exact claim/evidence/
//! provenance ledger lineage at one logical capture cycle and produces a BLAKE3
//! integrity digest.
//!
//! The manifest is still export/validation only: it cannot hydrate a ledger,
//! revision history, or support store and performs no file/database I/O.

use super::belief_mutation_persistence::{
    BeliefMutationPersistenceCapsuleV1, BeliefMutationPersistenceVersion,
};
use super::belief_revision_persistence::{
    BeliefRevisionHistoryCapsuleV1, BeliefRevisionPersistenceVersion,
};
use super::claim_evidence::{
    ClaimId, ClaimKind, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartManifestVersion {
    V1,
}

/// V1 hashes explicit ledger/mutation fields and a version-pinned Rust `Debug`
/// representation of each immutable revision receipt.
///
/// The receipt representation is intentionally named here rather than silently
/// pretending to be a cross-version wire format. A future serialized restore
/// format should replace this with an explicitly specified byte schema and bump
/// the manifest version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartEncoding {
    ExplicitFieldsPlusReceiptDebugV1,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicRestartDigest([u8; 32]);

impl EpistemicRestartDigest {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

impl fmt::Debug for EpistemicRestartDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EpistemicRestartDigest({})", self.to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicLedgerInventoryV1 {
    claim_ids: Vec<ClaimId>,
    evidence_ids: Vec<EvidenceId>,
    provenance_ids: Vec<ProvenanceId>,
}

impl EpistemicLedgerInventoryV1 {
    pub fn new(
        mut claim_ids: Vec<ClaimId>,
        mut evidence_ids: Vec<EvidenceId>,
        mut provenance_ids: Vec<ProvenanceId>,
    ) -> Result<Self, EpistemicRestartManifestError> {
        claim_ids.sort_unstable();
        evidence_ids.sort_unstable();
        provenance_ids.sort_unstable();
        ensure_unique_claims(&claim_ids)?;
        ensure_unique_evidence(&evidence_ids)?;
        ensure_unique_provenance(&provenance_ids)?;
        Ok(Self {
            claim_ids,
            evidence_ids,
            provenance_ids,
        })
    }

    pub fn claim_ids(&self) -> &[ClaimId] {
        &self.claim_ids
    }

    pub fn evidence_ids(&self) -> &[EvidenceId] {
        &self.evidence_ids
    }

    pub fn provenance_ids(&self) -> &[ProvenanceId] {
        &self.provenance_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicLedgerLineageV1 {
    pub claim_count: usize,
    pub evidence_count: usize,
    pub provenance_count: usize,
    pub next_claim_id: ClaimId,
    pub next_evidence_id: EvidenceId,
    pub next_provenance_id: ProvenanceId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRestartManifestV1 {
    version: EpistemicRestartManifestVersion,
    encoding: EpistemicRestartEncoding,
    captured_at_cycle: u64,
    ledger_lineage: EpistemicLedgerLineageV1,
    ledger_digest: EpistemicRestartDigest,
    mutation_capsule_digest: EpistemicRestartDigest,
    revision_capsule_digest: EpistemicRestartDigest,
    manifest_digest: EpistemicRestartDigest,
}

impl EpistemicRestartManifestV1 {
    pub fn capture(
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        mutation_capsule: &BeliefMutationPersistenceCapsuleV1,
        revision_capsule: &BeliefRevisionHistoryCapsuleV1,
        captured_at_cycle: u64,
    ) -> Result<Self, EpistemicRestartManifestError> {
        require_atomic_epoch(mutation_capsule, revision_capsule, captured_at_cycle)?;
        validate_revision_mutation_links(mutation_capsule, revision_capsule)?;

        let ledger_lineage = validate_ledger(ledger, inventory, captured_at_cycle)?;
        validate_support_claims_exist(ledger, mutation_capsule)?;

        let ledger_digest = digest_ledger(ledger, inventory, &ledger_lineage);
        let mutation_capsule_digest = digest_mutation_capsule(mutation_capsule);
        let revision_capsule_digest = digest_revision_capsule(revision_capsule);
        let manifest_digest = digest_manifest(
            captured_at_cycle,
            &ledger_lineage,
            ledger_digest,
            mutation_capsule_digest,
            revision_capsule_digest,
        );

        Ok(Self {
            version: EpistemicRestartManifestVersion::V1,
            encoding: EpistemicRestartEncoding::ExplicitFieldsPlusReceiptDebugV1,
            captured_at_cycle,
            ledger_lineage,
            ledger_digest,
            mutation_capsule_digest,
            revision_capsule_digest,
            manifest_digest,
        })
    }

    pub fn version(&self) -> EpistemicRestartManifestVersion {
        self.version
    }

    pub fn encoding(&self) -> EpistemicRestartEncoding {
        self.encoding
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn ledger_lineage(&self) -> &EpistemicLedgerLineageV1 {
        &self.ledger_lineage
    }

    pub fn ledger_digest(&self) -> EpistemicRestartDigest {
        self.ledger_digest
    }

    pub fn mutation_capsule_digest(&self) -> EpistemicRestartDigest {
        self.mutation_capsule_digest
    }

    pub fn revision_capsule_digest(&self) -> EpistemicRestartDigest {
        self.revision_capsule_digest
    }

    pub fn manifest_digest(&self) -> EpistemicRestartDigest {
        self.manifest_digest
    }

    /// Recompute the complete manifest from supplied immutable components.
    ///
    /// This is a read-only equivalence check, not restore or repair authority.
    pub fn verify_components(
        &self,
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        mutation_capsule: &BeliefMutationPersistenceCapsuleV1,
        revision_capsule: &BeliefRevisionHistoryCapsuleV1,
    ) -> Result<(), EpistemicRestartManifestError> {
        let live = Self::capture(
            ledger,
            inventory,
            mutation_capsule,
            revision_capsule,
            self.captured_at_cycle,
        )?;
        if live.ledger_lineage != self.ledger_lineage {
            return Err(EpistemicRestartManifestError::LedgerLineageMismatch);
        }
        if live.ledger_digest != self.ledger_digest {
            return Err(EpistemicRestartManifestError::LedgerDigestMismatch);
        }
        if live.mutation_capsule_digest != self.mutation_capsule_digest {
            return Err(EpistemicRestartManifestError::MutationCapsuleDigestMismatch);
        }
        if live.revision_capsule_digest != self.revision_capsule_digest {
            return Err(EpistemicRestartManifestError::RevisionCapsuleDigestMismatch);
        }
        if live.manifest_digest != self.manifest_digest {
            return Err(EpistemicRestartManifestError::ManifestDigestMismatch);
        }
        Ok(())
    }
}

fn require_atomic_epoch(
    mutation: &BeliefMutationPersistenceCapsuleV1,
    revision: &BeliefRevisionHistoryCapsuleV1,
    captured_at_cycle: u64,
) -> Result<(), EpistemicRestartManifestError> {
    if mutation.version() != BeliefMutationPersistenceVersion::V1 {
        return Err(EpistemicRestartManifestError::UnsupportedMutationVersion);
    }
    if revision.version() != BeliefRevisionPersistenceVersion::V1 {
        return Err(EpistemicRestartManifestError::UnsupportedRevisionVersion);
    }
    if mutation.captured_at_cycle() != captured_at_cycle
        || revision.captured_at_cycle() != captured_at_cycle
        || revision.linked_mutation_capture_cycle() != captured_at_cycle
    {
        return Err(EpistemicRestartManifestError::NonAtomicCaptureEpoch {
            manifest_cycle: captured_at_cycle,
            mutation_cycle: mutation.captured_at_cycle(),
            revision_cycle: revision.captured_at_cycle(),
            linked_mutation_cycle: revision.linked_mutation_capture_cycle(),
        });
    }
    if revision.linked_mutation_count() != mutation.mutations().len() {
        return Err(EpistemicRestartManifestError::LinkedMutationCountMismatch {
            revision_linked: revision.linked_mutation_count(),
            mutation_actual: mutation.mutations().len(),
        });
    }
    Ok(())
}

fn validate_revision_mutation_links(
    mutation: &BeliefMutationPersistenceCapsuleV1,
    revision: &BeliefRevisionHistoryCapsuleV1,
) -> Result<(), EpistemicRestartManifestError> {
    for applied in mutation.mutations() {
        let source = revision
            .receipts()
            .iter()
            .find(|receipt| receipt.id() == applied.source_revision_receipt_id)
            .ok_or(EpistemicRestartManifestError::MutationDecisionMissing {
                mutation_id: applied.id.0,
                receipt_id: applied.source_revision_receipt_id.0,
            })?;
        if !source.eligible() {
            return Err(EpistemicRestartManifestError::MutationDecisionRejected {
                mutation_id: applied.id.0,
                receipt_id: source.id().0,
            });
        }
        if source.claim_id() != applied.claim_id {
            return Err(EpistemicRestartManifestError::MutationDecisionClaimMismatch {
                mutation_id: applied.id.0,
                decision_claim: source.claim_id(),
                mutation_claim: applied.claim_id,
            });
        }
        if source.proposed_delta().to_bits() != applied.proposed_delta.to_bits() {
            return Err(EpistemicRestartManifestError::MutationDecisionDeltaMismatch {
                mutation_id: applied.id.0,
            });
        }
        if applied.authorized_at_cycle < source.evaluated_at_cycle() {
            return Err(EpistemicRestartManifestError::MutationAuthorizationPredatesDecision {
                mutation_id: applied.id.0,
                decision_cycle: source.evaluated_at_cycle(),
                authorization_cycle: applied.authorized_at_cycle,
            });
        }
    }
    Ok(())
}

fn validate_support_claims_exist(
    ledger: &EpistemicLedger,
    mutation: &BeliefMutationPersistenceCapsuleV1,
) -> Result<(), EpistemicRestartManifestError> {
    for state in mutation.states() {
        if ledger.claim(state.claim_id).is_none() {
            return Err(EpistemicRestartManifestError::SupportStateUnknownClaim(
                state.claim_id,
            ));
        }
    }
    Ok(())
}

fn validate_ledger(
    ledger: &EpistemicLedger,
    inventory: &EpistemicLedgerInventoryV1,
    captured_at_cycle: u64,
) -> Result<EpistemicLedgerLineageV1, EpistemicRestartManifestError> {
    if inventory.claim_ids.len() != ledger.claim_count()
        || inventory.evidence_ids.len() != ledger.evidence_count()
        || inventory.provenance_ids.len() != ledger.provenance_count()
    {
        return Err(EpistemicRestartManifestError::LedgerInventoryCountMismatch {
            declared_claims: inventory.claim_ids.len(),
            actual_claims: ledger.claim_count(),
            declared_evidence: inventory.evidence_ids.len(),
            actual_evidence: ledger.evidence_count(),
            declared_provenance: inventory.provenance_ids.len(),
            actual_provenance: ledger.provenance_count(),
        });
    }

    let next_claim_id = contiguous_next_claim_id(&inventory.claim_ids)?;
    let next_evidence_id = contiguous_next_evidence_id(&inventory.evidence_ids)?;
    let next_provenance_id = contiguous_next_provenance_id(&inventory.provenance_ids)?;

    let evidence_set = inventory.evidence_ids.iter().copied().collect::<BTreeSet<_>>();
    let provenance_set = inventory
        .provenance_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let claim_set = inventory.claim_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut evidence_by_claim: BTreeMap<ClaimId, Vec<EvidenceId>> = BTreeMap::new();

    for provenance_id in &inventory.provenance_ids {
        let record = ledger
            .provenance(*provenance_id)
            .ok_or(EpistemicRestartManifestError::MissingProvenance(*provenance_id))?;
        if record.recorded_at_cycle > captured_at_cycle {
            return Err(EpistemicRestartManifestError::ProvenancePostdatesCapture {
                provenance_id: *provenance_id,
                record_cycle: record.recorded_at_cycle,
                capture_cycle: captured_at_cycle,
            });
        }
        for parent in &record.parent_ids {
            if !provenance_set.contains(parent) || ledger.provenance(*parent).is_none() {
                return Err(EpistemicRestartManifestError::MissingProvenanceParent {
                    provenance_id: *provenance_id,
                    parent_id: *parent,
                });
            }
        }
    }

    for evidence_id in &inventory.evidence_ids {
        let record = ledger
            .evidence(*evidence_id)
            .ok_or(EpistemicRestartManifestError::MissingEvidence(*evidence_id))?;
        if record.observed_at_cycle > captured_at_cycle {
            return Err(EpistemicRestartManifestError::EvidencePostdatesCapture {
                evidence_id: *evidence_id,
                record_cycle: record.observed_at_cycle,
                capture_cycle: captured_at_cycle,
            });
        }
        if !claim_set.contains(&record.claim_id) || ledger.claim(record.claim_id).is_none() {
            return Err(EpistemicRestartManifestError::EvidenceUnknownClaim {
                evidence_id: *evidence_id,
                claim_id: record.claim_id,
            });
        }
        if !provenance_set.contains(&record.provenance_id)
            || ledger.provenance(record.provenance_id).is_none()
        {
            return Err(EpistemicRestartManifestError::EvidenceUnknownProvenance {
                evidence_id: *evidence_id,
                provenance_id: record.provenance_id,
            });
        }
        evidence_by_claim
            .entry(record.claim_id)
            .or_default()
            .push(*evidence_id);
    }

    for claim_id in &inventory.claim_ids {
        let claim = ledger
            .claim(*claim_id)
            .ok_or(EpistemicRestartManifestError::MissingClaim(*claim_id))?;
        if claim.created_at_cycle > captured_at_cycle {
            return Err(EpistemicRestartManifestError::ClaimPostdatesCapture {
                claim_id: *claim_id,
                record_cycle: claim.created_at_cycle,
                capture_cycle: captured_at_cycle,
            });
        }
        let mut declared = claim.evidence_ids.clone();
        declared.sort_unstable();
        if declared.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(EpistemicRestartManifestError::DuplicateClaimEvidenceLink(
                *claim_id,
            ));
        }
        if declared.iter().any(|id| !evidence_set.contains(id)) {
            return Err(EpistemicRestartManifestError::ClaimLinksUnknownEvidence(
                *claim_id,
            ));
        }
        let mut actual = evidence_by_claim.remove(claim_id).unwrap_or_default();
        actual.sort_unstable();
        if declared != actual {
            return Err(EpistemicRestartManifestError::ClaimEvidenceCensusMismatch(
                *claim_id,
            ));
        }
    }

    Ok(EpistemicLedgerLineageV1 {
        claim_count: ledger.claim_count(),
        evidence_count: ledger.evidence_count(),
        provenance_count: ledger.provenance_count(),
        next_claim_id,
        next_evidence_id,
        next_provenance_id,
    })
}

fn ensure_unique_claims(ids: &[ClaimId]) -> Result<(), EpistemicRestartManifestError> {
    if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(EpistemicRestartManifestError::DuplicateClaimId(pair[0]));
    }
    Ok(())
}

fn ensure_unique_evidence(ids: &[EvidenceId]) -> Result<(), EpistemicRestartManifestError> {
    if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(EpistemicRestartManifestError::DuplicateEvidenceId(pair[0]));
    }
    Ok(())
}

fn ensure_unique_provenance(ids: &[ProvenanceId]) -> Result<(), EpistemicRestartManifestError> {
    if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(EpistemicRestartManifestError::DuplicateProvenanceId(pair[0]));
    }
    Ok(())
}

fn contiguous_next_claim_id(ids: &[ClaimId]) -> Result<ClaimId, EpistemicRestartManifestError> {
    for (index, id) in ids.iter().enumerate() {
        let expected = index as u64 + 1;
        if id.0 != expected {
            return Err(EpistemicRestartManifestError::ClaimIdLineageGap {
                expected: ClaimId(expected),
                actual: *id,
            });
        }
    }
    Ok(ClaimId(ids.len() as u64 + 1))
}

fn contiguous_next_evidence_id(
    ids: &[EvidenceId],
) -> Result<EvidenceId, EpistemicRestartManifestError> {
    for (index, id) in ids.iter().enumerate() {
        let expected = index as u64 + 1;
        if id.0 != expected {
            return Err(EpistemicRestartManifestError::EvidenceIdLineageGap {
                expected: EvidenceId(expected),
                actual: *id,
            });
        }
    }
    Ok(EvidenceId(ids.len() as u64 + 1))
}

fn contiguous_next_provenance_id(
    ids: &[ProvenanceId],
) -> Result<ProvenanceId, EpistemicRestartManifestError> {
    for (index, id) in ids.iter().enumerate() {
        let expected = index as u64 + 1;
        if id.0 != expected {
            return Err(EpistemicRestartManifestError::ProvenanceIdLineageGap {
                expected: ProvenanceId(expected),
                actual: *id,
            });
        }
    }
    Ok(ProvenanceId(ids.len() as u64 + 1))
}

struct CanonicalHasher(blake3::Hasher);

impl CanonicalHasher {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    fn u8(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u64(&mut self, value: u64) {
        self.0.update(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn f32(&mut self, value: f32) {
        self.u64(value.to_bits() as u64);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.0.update(bytes);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn optional_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.string(value);
            }
            None => self.u8(0),
        }
    }

    fn digest(self) -> EpistemicRestartDigest {
        EpistemicRestartDigest(*self.0.finalize().as_bytes())
    }
}

fn digest_ledger(
    ledger: &EpistemicLedger,
    inventory: &EpistemicLedgerInventoryV1,
    lineage: &EpistemicLedgerLineageV1,
) -> EpistemicRestartDigest {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-ledger-v1");
    h.usize(lineage.claim_count);
    h.usize(lineage.evidence_count);
    h.usize(lineage.provenance_count);
    h.u64(lineage.next_claim_id.0);
    h.u64(lineage.next_evidence_id.0);
    h.u64(lineage.next_provenance_id.0);

    for provenance_id in &inventory.provenance_ids {
        let record = ledger
            .provenance(*provenance_id)
            .expect("validated inventory must resolve provenance");
        h.u64(record.id.0);
        h.string(&record.source_label);
        h.optional_string(record.source_uri.as_deref());
        h.optional_string(record.content_hash.as_deref());
        h.u64(record.recorded_at_cycle);
        h.usize(record.parent_ids.len());
        for parent in &record.parent_ids {
            h.u64(parent.0);
        }
    }

    for claim_id in &inventory.claim_ids {
        let claim = ledger
            .claim(*claim_id)
            .expect("validated inventory must resolve claim");
        h.u64(claim.id.0);
        h.string(&claim.statement);
        h.u8(claim_kind_tag(claim.kind));
        h.optional_string(claim.domain.as_deref());
        h.optional_string(claim.scope.as_deref());
        h.u64(claim.created_at_cycle);
        h.usize(claim.evidence_ids.len());
        for evidence_id in &claim.evidence_ids {
            h.u64(evidence_id.0);
        }
    }

    for evidence_id in &inventory.evidence_ids {
        let record = ledger
            .evidence(*evidence_id)
            .expect("validated inventory must resolve evidence");
        h.u64(record.id.0);
        h.u64(record.claim_id.0);
        h.u8(evidence_kind_tag(record.kind));
        h.u8(evidence_polarity_tag(record.polarity));
        h.u64(record.provenance_id.0);
        h.u64(record.observed_at_cycle);
        h.optional_string(record.context.as_deref());
        h.optional_string(record.method.as_deref());
    }
    h.digest()
}

fn digest_mutation_capsule(
    capsule: &BeliefMutationPersistenceCapsuleV1,
) -> EpistemicRestartDigest {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-belief-mutation-capsule-v1");
    h.u8(1);
    h.u64(capsule.captured_at_cycle());
    h.usize(capsule.consumed_authorization_count());
    h.usize(capsule.states().len());
    for state in capsule.states() {
        h.u64(state.claim_id.0);
        h.f32(state.baseline_support.get());
        h.f32(state.current_support.get());
        h.u64(state.revision);
        h.u64(state.initialized_at_cycle);
        h.u64(state.last_updated_cycle);
        match state.last_mutation_id {
            Some(id) => {
                h.u8(1);
                h.u64(id.0);
            }
            None => h.u8(0),
        }
    }
    h.usize(capsule.mutations().len());
    for mutation in capsule.mutations() {
        h.u64(mutation.id.0);
        h.u64(mutation.source_revision_receipt_id.0);
        h.u64(mutation.claim_id.0);
        h.f32(mutation.proposed_delta);
        h.f32(mutation.support_before.get());
        h.f32(mutation.support_after.get());
        h.u64(mutation.state_revision_before);
        h.u64(mutation.state_revision_after);
        h.string(&mutation.authorization_id);
        h.string(&mutation.authority_label);
        h.u64(mutation.authorized_at_cycle);
        h.u64(mutation.applied_at_cycle);
    }
    h.digest()
}

fn digest_revision_capsule(
    capsule: &BeliefRevisionHistoryCapsuleV1,
) -> EpistemicRestartDigest {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-belief-revision-capsule-v1");
    h.u8(1);
    h.u64(capsule.captured_at_cycle());
    h.u64(capsule.next_receipt_id().0);
    h.u64(capsule.linked_mutation_capture_cycle());
    h.usize(capsule.linked_mutation_count());
    h.usize(capsule.receipts().len());
    for receipt in capsule.receipts() {
        // V1 intentionally pins the complete derived-Debug representation of the
        // immutable receipt so private policy fields remain integrity-bound
        // without widening their public authority surface. This is not claimed
        // as a cross-version wire encoding; the manifest encoding enum makes
        // that limitation explicit.
        h.u64(receipt.id().0);
        h.string(&format!("{receipt:?}"));
    }
    h.digest()
}

fn digest_manifest(
    captured_at_cycle: u64,
    lineage: &EpistemicLedgerLineageV1,
    ledger: EpistemicRestartDigest,
    mutation: EpistemicRestartDigest,
    revision: EpistemicRestartDigest,
) -> EpistemicRestartDigest {
    let mut h = CanonicalHasher::new(b"symthaea-ekm-atomic-restart-manifest-v1");
    h.u8(1);
    h.u8(1);
    h.u64(captured_at_cycle);
    h.usize(lineage.claim_count);
    h.usize(lineage.evidence_count);
    h.usize(lineage.provenance_count);
    h.u64(lineage.next_claim_id.0);
    h.u64(lineage.next_evidence_id.0);
    h.u64(lineage.next_provenance_id.0);
    h.bytes(&ledger.0);
    h.bytes(&mutation.0);
    h.bytes(&revision.0);
    h.digest()
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

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartManifestError {
    UnsupportedMutationVersion,
    UnsupportedRevisionVersion,
    NonAtomicCaptureEpoch {
        manifest_cycle: u64,
        mutation_cycle: u64,
        revision_cycle: u64,
        linked_mutation_cycle: u64,
    },
    LinkedMutationCountMismatch {
        revision_linked: usize,
        mutation_actual: usize,
    },
    MutationDecisionMissing {
        mutation_id: u64,
        receipt_id: u64,
    },
    MutationDecisionRejected {
        mutation_id: u64,
        receipt_id: u64,
    },
    MutationDecisionClaimMismatch {
        mutation_id: u64,
        decision_claim: ClaimId,
        mutation_claim: ClaimId,
    },
    MutationDecisionDeltaMismatch {
        mutation_id: u64,
    },
    MutationAuthorizationPredatesDecision {
        mutation_id: u64,
        decision_cycle: u64,
        authorization_cycle: u64,
    },
    SupportStateUnknownClaim(ClaimId),
    DuplicateClaimId(ClaimId),
    DuplicateEvidenceId(EvidenceId),
    DuplicateProvenanceId(ProvenanceId),
    LedgerInventoryCountMismatch {
        declared_claims: usize,
        actual_claims: usize,
        declared_evidence: usize,
        actual_evidence: usize,
        declared_provenance: usize,
        actual_provenance: usize,
    },
    ClaimIdLineageGap {
        expected: ClaimId,
        actual: ClaimId,
    },
    EvidenceIdLineageGap {
        expected: EvidenceId,
        actual: EvidenceId,
    },
    ProvenanceIdLineageGap {
        expected: ProvenanceId,
        actual: ProvenanceId,
    },
    MissingClaim(ClaimId),
    MissingEvidence(EvidenceId),
    MissingProvenance(ProvenanceId),
    MissingProvenanceParent {
        provenance_id: ProvenanceId,
        parent_id: ProvenanceId,
    },
    ClaimPostdatesCapture {
        claim_id: ClaimId,
        record_cycle: u64,
        capture_cycle: u64,
    },
    EvidencePostdatesCapture {
        evidence_id: EvidenceId,
        record_cycle: u64,
        capture_cycle: u64,
    },
    ProvenancePostdatesCapture {
        provenance_id: ProvenanceId,
        record_cycle: u64,
        capture_cycle: u64,
    },
    EvidenceUnknownClaim {
        evidence_id: EvidenceId,
        claim_id: ClaimId,
    },
    EvidenceUnknownProvenance {
        evidence_id: EvidenceId,
        provenance_id: ProvenanceId,
    },
    DuplicateClaimEvidenceLink(ClaimId),
    ClaimLinksUnknownEvidence(ClaimId),
    ClaimEvidenceCensusMismatch(ClaimId),
    LedgerLineageMismatch,
    LedgerDigestMismatch,
    MutationCapsuleDigestMismatch,
    RevisionCapsuleDigestMismatch,
    ManifestDigestMismatch,
}

impl fmt::Display for EpistemicRestartManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart manifest invalid: {self:?}")
    }
}

impl Error for EpistemicRestartManifestError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthority, BeliefMutationAuthorization,
        BeliefMutationAuthorizationDecision, BeliefRevisionHistory, BeliefRevisionPolicy,
        BoundedWeight, EpistemicRevisionProposal, EpistemicSupportStore,
    };

    struct Fixture {
        ledger: EpistemicLedger,
        inventory: EpistemicLedgerInventoryV1,
        mutations: BeliefMutationPersistenceCapsuleV1,
        revisions: BeliefRevisionHistoryCapsuleV1,
    }

    fn fixture(capture_cycle: u64) -> Fixture {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("lab", Some("lab://a".into()), Some("abc123".into()), 1, vec![])
            .unwrap();
        let claim = ledger.add_claim(
            "X predicts Y",
            ClaimKind::Predictive,
            Some("test".into()),
            Some("fixture".into()),
            1,
        );
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                Some("fixture".into()),
                Some("protocol-v1".into()),
            )
            .unwrap();

        let policy = BeliefRevisionPolicy::new(0.20, 1, false, 0, 1.0).unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &proposal, &policy, None, None, 3)
            .unwrap();
        let mut store = EpistemicSupportStore::new();
        store
            .register_claim(&ledger, claim, BoundedWeight::new(0.50).unwrap(), 3)
            .unwrap();
        let authorization = BeliefMutationAuthorization::new(
            "auth-1",
            "test-authority",
            BeliefMutationAuthorizationDecision::Approved,
            4,
            prepared.receipt(),
            store.state(claim).unwrap(),
        )
        .unwrap();
        authority
            .apply(&ledger, &mut store, &prepared, &authorization, 5)
            .unwrap();

        let mutations = BeliefMutationPersistenceCapsuleV1::capture(
            &store,
            &[claim],
            capture_cycle,
        )
        .unwrap();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(
            &history,
            &mutations,
            capture_cycle,
        )
        .unwrap();
        let inventory =
            EpistemicLedgerInventoryV1::new(vec![claim], vec![evidence], vec![provenance]).unwrap();
        Fixture {
            ledger,
            inventory,
            mutations,
            revisions,
        }
    }

    #[test]
    fn atomic_manifest_binds_ledger_mutations_and_decisions() {
        let fixture = fixture(6);
        let manifest = EpistemicRestartManifestV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        assert_eq!(manifest.version(), EpistemicRestartManifestVersion::V1);
        assert_eq!(
            manifest.encoding(),
            EpistemicRestartEncoding::ExplicitFieldsPlusReceiptDebugV1
        );
        assert_eq!(manifest.ledger_lineage().next_claim_id, ClaimId(2));
        assert_eq!(manifest.ledger_lineage().next_evidence_id, EvidenceId(2));
        assert_eq!(manifest.ledger_lineage().next_provenance_id, ProvenanceId(2));
        assert_eq!(manifest.manifest_digest().to_hex().len(), 64);
        manifest
            .verify_components(
                &fixture.ledger,
                &fixture.inventory,
                &fixture.mutations,
                &fixture.revisions,
            )
            .unwrap();
    }

    #[test]
    fn mixed_capture_epochs_are_not_atomic() {
        let fixture = fixture(6);
        assert!(matches!(
            EpistemicRestartManifestV1::capture(
                &fixture.ledger,
                &fixture.inventory,
                &fixture.mutations,
                &fixture.revisions,
                7,
            ),
            Err(EpistemicRestartManifestError::NonAtomicCaptureEpoch { .. })
        ));
    }

    #[test]
    fn inventory_order_does_not_change_digest() {
        let mut ledger = EpistemicLedger::new();
        let p1 = ledger.add_provenance("p1", None, None, 1, vec![]).unwrap();
        let p2 = ledger.add_provenance("p2", None, None, 1, vec![]).unwrap();
        let c1 = ledger.add_claim("c1", ClaimKind::Descriptive, None, None, 1);
        let c2 = ledger.add_claim("c2", ClaimKind::Descriptive, None, None, 1);
        let e1 = ledger
            .add_evidence(c1, EvidenceKind::Report, EvidencePolarity::Supports, p1, 2, None, None)
            .unwrap();
        let e2 = ledger
            .add_evidence(c2, EvidenceKind::Observation, EvidencePolarity::Supports, p2, 2, None, None)
            .unwrap();
        let a = EpistemicLedgerInventoryV1::new(vec![c1, c2], vec![e1, e2], vec![p1, p2]).unwrap();
        let b = EpistemicLedgerInventoryV1::new(vec![c2, c1], vec![e2, e1], vec![p2, p1]).unwrap();
        let la = validate_ledger(&ledger, &a, 3).unwrap();
        let lb = validate_ledger(&ledger, &b, 3).unwrap();
        assert_eq!(digest_ledger(&ledger, &a, &la), digest_ledger(&ledger, &b, &lb));
    }

    #[test]
    fn ledger_semantic_change_changes_digest() {
        let mut left = EpistemicLedger::new();
        let lp = left.add_provenance("source", None, None, 1, vec![]).unwrap();
        let lc = left.add_claim("alpha", ClaimKind::Descriptive, None, None, 1);
        let le = left
            .add_evidence(lc, EvidenceKind::Report, EvidencePolarity::Supports, lp, 2, None, None)
            .unwrap();
        let li = EpistemicLedgerInventoryV1::new(vec![lc], vec![le], vec![lp]).unwrap();

        let mut right = EpistemicLedger::new();
        let rp = right.add_provenance("source", None, None, 1, vec![]).unwrap();
        let rc = right.add_claim("beta", ClaimKind::Descriptive, None, None, 1);
        let re = right
            .add_evidence(rc, EvidenceKind::Report, EvidencePolarity::Supports, rp, 2, None, None)
            .unwrap();
        let ri = EpistemicLedgerInventoryV1::new(vec![rc], vec![re], vec![rp]).unwrap();

        let ld = digest_ledger(&left, &li, &validate_ledger(&left, &li, 3).unwrap());
        let rd = digest_ledger(&right, &ri, &validate_ledger(&right, &ri, 3).unwrap());
        assert_ne!(ld, rd);
    }

    #[test]
    fn incomplete_inventory_fails_closed() {
        let fixture = fixture(6);
        let incomplete = EpistemicLedgerInventoryV1::new(
            fixture.inventory.claim_ids().to_vec(),
            Vec::new(),
            fixture.inventory.provenance_ids().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            EpistemicRestartManifestV1::capture(
                &fixture.ledger,
                &incomplete,
                &fixture.mutations,
                &fixture.revisions,
                6,
            ),
            Err(EpistemicRestartManifestError::LedgerInventoryCountMismatch { .. })
        ));
    }

    #[test]
    fn future_dated_ledger_record_is_rejected() {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger.add_provenance("future", None, None, 9, vec![]).unwrap();
        let inventory = EpistemicLedgerInventoryV1::new(vec![], vec![], vec![provenance]).unwrap();
        assert!(matches!(
            validate_ledger(&ledger, &inventory, 5),
            Err(EpistemicRestartManifestError::ProvenancePostdatesCapture { .. })
        ));
    }
}
