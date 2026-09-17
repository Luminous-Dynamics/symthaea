// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Complete, read-only EKM restart payload with quarantine reconstruction.
//!
//! EKM-032 proves that a live ledger, mutation capsule, and complete revision
//! history describe one atomic epistemic epoch, but a manifest digest alone is
//! not enough to reconstruct a cold process. This module closes that payload gap
//! by carrying the immutable claim/evidence/provenance records themselves beside
//! the already-validated EKM-030 and EKM-031 capsules and EKM-032 manifest.
//!
//! The capsule can reconstruct a fresh [`EpistemicLedger`] in quarantine and
//! require the reconstructed components to reproduce the source manifest exactly.
//! It deliberately has no activation, mutable-store hydration, file/database I/O,
//! or `into_live` escape hatch.

use super::belief_mutation_firewall::EpistemicSupportStore;
use super::belief_mutation_persistence::{
    BeliefMutationPersistenceCapsuleV1, BeliefMutationPersistenceError,
    PersistedBeliefMutationV1, PersistedEpistemicSupportStateV1,
};
use super::belief_revision_persistence::{
    BeliefRevisionHistoryCapsuleV1, BeliefRevisionPersistenceError,
};
use super::belief_revision_receipt::{BeliefRevisionHistory, BeliefRevisionReceipt};
use super::claim_evidence::{
    ClaimId, EpistemicLedger, EvidenceId, EvidenceRecord, KnowledgeClaim, ProvenanceId,
    ProvenanceRecord,
};
use super::epistemic_restart_manifest::{
    EpistemicLedgerInventoryV1, EpistemicLedgerLineageV1, EpistemicRestartDigest,
    EpistemicRestartManifestError, EpistemicRestartManifestV1,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartCapsuleVersion {
    V1,
}

/// Complete immutable ledger payload needed for a cold reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedEpistemicLedgerV1 {
    captured_at_cycle: u64,
    lineage: EpistemicLedgerLineageV1,
    provenance: Vec<ProvenanceRecord>,
    claims: Vec<KnowledgeClaim>,
    evidence: Vec<EvidenceRecord>,
}

impl PersistedEpistemicLedgerV1 {
    fn capture(
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        lineage: &EpistemicLedgerLineageV1,
        captured_at_cycle: u64,
    ) -> Result<Self, EpistemicRestartCapsuleError> {
        let provenance = inventory
            .provenance_ids()
            .iter()
            .map(|id| {
                ledger
                    .provenance(*id)
                    .cloned()
                    .ok_or(EpistemicRestartCapsuleError::MissingSourceProvenance(*id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let claims = inventory
            .claim_ids()
            .iter()
            .map(|id| {
                ledger
                    .claim(*id)
                    .cloned()
                    .ok_or(EpistemicRestartCapsuleError::MissingSourceClaim(*id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let evidence = inventory
            .evidence_ids()
            .iter()
            .map(|id| {
                ledger
                    .evidence(*id)
                    .cloned()
                    .ok_or(EpistemicRestartCapsuleError::MissingSourceEvidence(*id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            captured_at_cycle,
            lineage: lineage.clone(),
            provenance,
            claims,
            evidence,
        })
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn lineage(&self) -> &EpistemicLedgerLineageV1 {
        &self.lineage
    }

    pub fn provenance(&self) -> &[ProvenanceRecord] {
        &self.provenance
    }

    pub fn claims(&self) -> &[KnowledgeClaim] {
        &self.claims
    }

    pub fn evidence(&self) -> &[EvidenceRecord] {
        &self.evidence
    }

    fn inventory(&self) -> Result<EpistemicLedgerInventoryV1, EpistemicRestartCapsuleError> {
        EpistemicLedgerInventoryV1::new(
            self.claims.iter().map(|record| record.id).collect(),
            self.evidence.iter().map(|record| record.id).collect(),
            self.provenance.iter().map(|record| record.id).collect(),
        )
        .map_err(EpistemicRestartCapsuleError::Manifest)
    }

    /// Reconstruct a fresh ledger using only the ordinary append APIs.
    ///
    /// This deliberately does not write private ledger fields directly. Stable IDs
    /// must re-emerge from the same append order, and every reconstructed record is
    /// compared byte-for-byte at the Rust value level with the persisted record.
    fn reconstruct_quarantined(&self) -> Result<EpistemicLedger, EpistemicRestartCapsuleError> {
        let mut ledger = EpistemicLedger::new();

        for record in &self.provenance {
            let rebuilt_id = ledger
                .add_provenance(
                    record.source_label.clone(),
                    record.source_uri.clone(),
                    record.content_hash.clone(),
                    record.recorded_at_cycle,
                    record.parent_ids.clone(),
                )
                .map_err(|_| EpistemicRestartCapsuleError::ProvenanceRebuildRejected(record.id))?;
            if rebuilt_id != record.id {
                return Err(EpistemicRestartCapsuleError::ProvenanceIdRebuildMismatch {
                    expected: record.id,
                    actual: rebuilt_id,
                });
            }
        }

        for record in &self.claims {
            let rebuilt_id = ledger.add_claim(
                record.statement.clone(),
                record.kind,
                record.domain.clone(),
                record.scope.clone(),
                record.created_at_cycle,
            );
            if rebuilt_id != record.id {
                return Err(EpistemicRestartCapsuleError::ClaimIdRebuildMismatch {
                    expected: record.id,
                    actual: rebuilt_id,
                });
            }
        }

        for record in &self.evidence {
            let rebuilt_id = ledger
                .add_evidence(
                    record.claim_id,
                    record.kind,
                    record.polarity,
                    record.provenance_id,
                    record.observed_at_cycle,
                    record.context.clone(),
                    record.method.clone(),
                )
                .map_err(|_| EpistemicRestartCapsuleError::EvidenceRebuildRejected(record.id))?;
            if rebuilt_id != record.id {
                return Err(EpistemicRestartCapsuleError::EvidenceIdRebuildMismatch {
                    expected: record.id,
                    actual: rebuilt_id,
                });
            }
        }

        for record in &self.provenance {
            if ledger.provenance(record.id) != Some(record) {
                return Err(EpistemicRestartCapsuleError::ProvenanceSemanticRebuildMismatch(
                    record.id,
                ));
            }
        }
        for record in &self.claims {
            if ledger.claim(record.id) != Some(record) {
                return Err(EpistemicRestartCapsuleError::ClaimSemanticRebuildMismatch(
                    record.id,
                ));
            }
        }
        for record in &self.evidence {
            if ledger.evidence(record.id) != Some(record) {
                return Err(EpistemicRestartCapsuleError::EvidenceSemanticRebuildMismatch(
                    record.id,
                ));
            }
        }

        if ledger.claim_count() != self.lineage.claim_count
            || ledger.evidence_count() != self.lineage.evidence_count
            || ledger.provenance_count() != self.lineage.provenance_count
        {
            return Err(EpistemicRestartCapsuleError::RebuiltLedgerCountMismatch);
        }

        Ok(ledger)
    }
}

/// Complete restart payload. All fields are private so construction must pass the
/// live-state capture and internal-equivalence checks in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicRestartCapsuleV1 {
    version: EpistemicRestartCapsuleVersion,
    captured_at_cycle: u64,
    ledger: PersistedEpistemicLedgerV1,
    mutations: BeliefMutationPersistenceCapsuleV1,
    revisions: BeliefRevisionHistoryCapsuleV1,
    manifest: EpistemicRestartManifestV1,
}

impl EpistemicRestartCapsuleV1 {
    pub fn capture(
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        mutations: &BeliefMutationPersistenceCapsuleV1,
        revisions: &BeliefRevisionHistoryCapsuleV1,
        captured_at_cycle: u64,
    ) -> Result<Self, EpistemicRestartCapsuleError> {
        let manifest = EpistemicRestartManifestV1::capture(
            ledger,
            inventory,
            mutations,
            revisions,
            captured_at_cycle,
        )
        .map_err(EpistemicRestartCapsuleError::Manifest)?;
        let persisted_ledger = PersistedEpistemicLedgerV1::capture(
            ledger,
            inventory,
            manifest.ledger_lineage(),
            captured_at_cycle,
        )?;

        let capsule = Self {
            version: EpistemicRestartCapsuleVersion::V1,
            captured_at_cycle,
            ledger: persisted_ledger,
            mutations: mutations.clone(),
            revisions: revisions.clone(),
            manifest,
        };
        capsule.validate_internal()?;
        Ok(capsule)
    }

    pub fn version(&self) -> EpistemicRestartCapsuleVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn ledger_payload(&self) -> &PersistedEpistemicLedgerV1 {
        &self.ledger
    }

    pub fn mutation_capsule(&self) -> &BeliefMutationPersistenceCapsuleV1 {
        &self.mutations
    }

    pub fn revision_capsule(&self) -> &BeliefRevisionHistoryCapsuleV1 {
        &self.revisions
    }

    pub fn manifest(&self) -> &EpistemicRestartManifestV1 {
        &self.manifest
    }

    pub fn manifest_digest(&self) -> EpistemicRestartDigest {
        self.manifest.manifest_digest()
    }

    /// Reconstruct and verify a cold, read-only quarantine image.
    ///
    /// No live support store or revision-history writer is created. The returned
    /// object exposes only immutable views and therefore cannot be activated by
    /// this API.
    pub fn quarantine(&self) -> Result<QuarantinedEpistemicRestartV1, EpistemicRestartCapsuleError> {
        self.validate_internal()?;
        let ledger = self.ledger.reconstruct_quarantined()?;
        let inventory = self.ledger.inventory()?;
        self.manifest
            .verify_components(&ledger, &inventory, &self.mutations, &self.revisions)
            .map_err(EpistemicRestartCapsuleError::Manifest)?;

        let reconstructed = EpistemicRestartManifestV1::capture(
            &ledger,
            &inventory,
            &self.mutations,
            &self.revisions,
            self.captured_at_cycle,
        )
        .map_err(EpistemicRestartCapsuleError::Manifest)?;
        if reconstructed.manifest_digest() != self.manifest.manifest_digest() {
            return Err(EpistemicRestartCapsuleError::QuarantineManifestDigestMismatch {
                expected: self.manifest.manifest_digest(),
                actual: reconstructed.manifest_digest(),
            });
        }

        Ok(QuarantinedEpistemicRestartV1 {
            ledger,
            inventory,
            mutations: self.mutations.clone(),
            revisions: self.revisions.clone(),
            manifest: reconstructed,
        })
    }

    /// Validate that the original live objects have not diverged from this capsule.
    pub fn verify_source_live(
        &self,
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        store: &EpistemicSupportStore,
        history: &BeliefRevisionHistory,
        observed_at_cycle: u64,
    ) -> Result<(), EpistemicRestartCapsuleError> {
        if observed_at_cycle < self.captured_at_cycle {
            return Err(EpistemicRestartCapsuleError::ObservationPredatesCapsule {
                observed_at_cycle,
                captured_at_cycle: self.captured_at_cycle,
            });
        }
        self.mutations
            .validate_live(store, observed_at_cycle)
            .map_err(EpistemicRestartCapsuleError::MutationPersistence)?;
        self.revisions
            .validate_live(history, &self.mutations, observed_at_cycle)
            .map_err(EpistemicRestartCapsuleError::RevisionPersistence)?;
        self.manifest
            .verify_components(ledger, inventory, &self.mutations, &self.revisions)
            .map_err(EpistemicRestartCapsuleError::Manifest)?;

        let source_payload = PersistedEpistemicLedgerV1::capture(
            ledger,
            inventory,
            self.manifest.ledger_lineage(),
            self.captured_at_cycle,
        )?;
        if source_payload != self.ledger {
            return Err(EpistemicRestartCapsuleError::LiveLedgerPayloadMismatch);
        }
        Ok(())
    }

    fn validate_internal(&self) -> Result<(), EpistemicRestartCapsuleError> {
        if self.ledger.captured_at_cycle != self.captured_at_cycle
            || self.mutations.captured_at_cycle() != self.captured_at_cycle
            || self.revisions.captured_at_cycle() != self.captured_at_cycle
            || self.manifest.captured_at_cycle() != self.captured_at_cycle
        {
            return Err(EpistemicRestartCapsuleError::CapsuleEpochMismatch {
                capsule_cycle: self.captured_at_cycle,
                ledger_cycle: self.ledger.captured_at_cycle,
                mutation_cycle: self.mutations.captured_at_cycle(),
                revision_cycle: self.revisions.captured_at_cycle(),
                manifest_cycle: self.manifest.captured_at_cycle(),
            });
        }
        if self.ledger.lineage != *self.manifest.ledger_lineage() {
            return Err(EpistemicRestartCapsuleError::LedgerLineageMismatch);
        }

        let ledger = self.ledger.reconstruct_quarantined()?;
        let inventory = self.ledger.inventory()?;
        self.manifest
            .verify_components(&ledger, &inventory, &self.mutations, &self.revisions)
            .map_err(EpistemicRestartCapsuleError::Manifest)?;
        validate_revision_basis_snapshots(&ledger, &self.revisions)?;
        Ok(())
    }
}

/// Verified cold-reconstruction image. The operational mutation/history types are
/// intentionally not hydrated here, and no mutable accessor or activation method
/// is provided.
#[derive(Debug, Clone)]
pub struct QuarantinedEpistemicRestartV1 {
    ledger: EpistemicLedger,
    inventory: EpistemicLedgerInventoryV1,
    mutations: BeliefMutationPersistenceCapsuleV1,
    revisions: BeliefRevisionHistoryCapsuleV1,
    manifest: EpistemicRestartManifestV1,
}

impl QuarantinedEpistemicRestartV1 {
    pub fn ledger(&self) -> &EpistemicLedger {
        &self.ledger
    }

    pub fn inventory(&self) -> &EpistemicLedgerInventoryV1 {
        &self.inventory
    }

    pub fn support_states(&self) -> &[PersistedEpistemicSupportStateV1] {
        self.mutations.states()
    }

    pub fn mutations(&self) -> &[PersistedBeliefMutationV1] {
        self.mutations.mutations()
    }

    pub fn revision_receipts(&self) -> &[BeliefRevisionReceipt] {
        self.revisions.receipts()
    }

    pub fn manifest(&self) -> &EpistemicRestartManifestV1 {
        &self.manifest
    }

    pub fn manifest_digest(&self) -> EpistemicRestartDigest {
        self.manifest.manifest_digest()
    }

    pub fn verify(&self) -> Result<(), EpistemicRestartCapsuleError> {
        self.manifest
            .verify_components(
                &self.ledger,
                &self.inventory,
                &self.mutations,
                &self.revisions,
            )
            .map_err(EpistemicRestartCapsuleError::Manifest)?;
        validate_revision_basis_snapshots(&self.ledger, &self.revisions)
    }
}

fn validate_revision_basis_snapshots(
    ledger: &EpistemicLedger,
    revisions: &BeliefRevisionHistoryCapsuleV1,
) -> Result<(), EpistemicRestartCapsuleError> {
    for receipt in revisions.receipts() {
        for basis in receipt.basis() {
            let Some(snapshot) = &basis.snapshot else {
                // Unknown evidence is valid audit history for a rejected decision.
                continue;
            };
            let live = ledger
                .evidence(snapshot.evidence_id)
                .ok_or(EpistemicRestartCapsuleError::RevisionSnapshotEvidenceMissing {
                    receipt_id: receipt.id().0,
                    evidence_id: snapshot.evidence_id,
                })?;
            if live.id != snapshot.evidence_id
                || live.claim_id != snapshot.claim_id
                || live.kind != snapshot.kind
                || live.polarity != snapshot.polarity
                || live.provenance_id != snapshot.provenance_id
                || live.observed_at_cycle != snapshot.observed_at_cycle
                || live.context != snapshot.context
                || live.method != snapshot.method
            {
                return Err(EpistemicRestartCapsuleError::RevisionSnapshotSemanticMismatch {
                    receipt_id: receipt.id().0,
                    evidence_id: snapshot.evidence_id,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartCapsuleError {
    Manifest(EpistemicRestartManifestError),
    MutationPersistence(BeliefMutationPersistenceError),
    RevisionPersistence(BeliefRevisionPersistenceError),
    MissingSourceClaim(ClaimId),
    MissingSourceEvidence(EvidenceId),
    MissingSourceProvenance(ProvenanceId),
    ProvenanceRebuildRejected(ProvenanceId),
    ClaimIdRebuildMismatch {
        expected: ClaimId,
        actual: ClaimId,
    },
    EvidenceRebuildRejected(EvidenceId),
    ProvenanceIdRebuildMismatch {
        expected: ProvenanceId,
        actual: ProvenanceId,
    },
    EvidenceIdRebuildMismatch {
        expected: EvidenceId,
        actual: EvidenceId,
    },
    ProvenanceSemanticRebuildMismatch(ProvenanceId),
    ClaimSemanticRebuildMismatch(ClaimId),
    EvidenceSemanticRebuildMismatch(EvidenceId),
    RebuiltLedgerCountMismatch,
    CapsuleEpochMismatch {
        capsule_cycle: u64,
        ledger_cycle: u64,
        mutation_cycle: u64,
        revision_cycle: u64,
        manifest_cycle: u64,
    },
    LedgerLineageMismatch,
    QuarantineManifestDigestMismatch {
        expected: EpistemicRestartDigest,
        actual: EpistemicRestartDigest,
    },
    ObservationPredatesCapsule {
        observed_at_cycle: u64,
        captured_at_cycle: u64,
    },
    LiveLedgerPayloadMismatch,
    RevisionSnapshotEvidenceMissing {
        receipt_id: u64,
        evidence_id: EvidenceId,
    },
    RevisionSnapshotSemanticMismatch {
        receipt_id: u64,
        evidence_id: EvidenceId,
    },
}

impl fmt::Display for EpistemicRestartCapsuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart capsule invalid: {self:?}")
    }
}

impl Error for EpistemicRestartCapsuleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationAuthority, BeliefMutationAuthorization,
        BeliefMutationAuthorizationDecision, BeliefRevisionPolicy, BoundedWeight, ClaimKind,
        EpistemicRevisionProposal, EvidenceKind, EvidencePolarity,
    };

    struct Fixture {
        ledger: EpistemicLedger,
        inventory: EpistemicLedgerInventoryV1,
        store: EpistemicSupportStore,
        history: BeliefRevisionHistory,
        mutations: BeliefMutationPersistenceCapsuleV1,
        revisions: BeliefRevisionHistoryCapsuleV1,
        claim: ClaimId,
    }

    fn fixture(capture_cycle: u64) -> Fixture {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance(
                "lab",
                Some("lab://a".into()),
                Some("abc123".into()),
                1,
                vec![],
            )
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
        let eligible = EpistemicRevisionProposal::new(claim, 0.10, vec![evidence], "support")
            .unwrap();
        let rejected = EpistemicRevisionProposal::new(
            claim,
            -0.10,
            vec![evidence],
            "wrong direction",
        )
        .unwrap();
        let mut history = BeliefRevisionHistory::new();
        let mut authority = BeliefMutationAuthority::new();
        let prepared = authority
            .prepare(&ledger, &mut history, &eligible, &policy, None, None, 3)
            .unwrap();
        history
            .evaluate_and_record(&ledger, &rejected, &policy, None, None, 4)
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

        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[claim], capture_cycle).unwrap();
        let revisions =
            BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, capture_cycle).unwrap();
        let inventory = EpistemicLedgerInventoryV1::new(
            vec![claim],
            vec![evidence],
            vec![provenance],
        )
        .unwrap();

        Fixture {
            ledger,
            inventory,
            store,
            history,
            mutations,
            revisions,
            claim,
        }
    }

    #[test]
    fn complete_capsule_cold_reconstructs_in_quarantine() {
        let fixture = fixture(6);
        let capsule = EpistemicRestartCapsuleV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        let quarantine = capsule.quarantine().unwrap();

        assert_eq!(quarantine.ledger().claim_count(), 1);
        assert_eq!(quarantine.ledger().evidence_count(), 1);
        assert_eq!(quarantine.ledger().provenance_count(), 1);
        assert_eq!(quarantine.support_states().len(), 1);
        assert_eq!(quarantine.mutations().len(), 1);
        assert_eq!(quarantine.revision_receipts().len(), 2);
        assert!(quarantine.revision_receipts()[0].eligible());
        assert!(!quarantine.revision_receipts()[1].eligible());
        assert_eq!(
            quarantine.ledger().claim(fixture.claim).unwrap().statement,
            "X predicts Y"
        );
        assert_eq!(quarantine.manifest_digest(), capsule.manifest_digest());
        quarantine.verify().unwrap();
    }

    #[test]
    fn live_source_equivalence_is_checked_without_mutation() {
        let fixture = fixture(6);
        let capsule = EpistemicRestartCapsuleV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        capsule
            .verify_source_live(
                &fixture.ledger,
                &fixture.inventory,
                &fixture.store,
                &fixture.history,
                7,
            )
            .unwrap();
    }

    #[test]
    fn semantic_ledger_tamper_breaks_manifest_equivalence() {
        let fixture = fixture(6);
        let mut capsule = EpistemicRestartCapsuleV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        capsule.ledger.claims[0].statement = "tampered".into();
        assert!(capsule.quarantine().is_err());
    }

    #[test]
    fn claim_evidence_census_tamper_fails_before_activation_exists() {
        let fixture = fixture(6);
        let mut capsule = EpistemicRestartCapsuleV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        capsule.ledger.claims[0].evidence_ids.clear();
        assert!(matches!(
            capsule.quarantine(),
            Err(EpistemicRestartCapsuleError::ClaimSemanticRebuildMismatch(_))
                | Err(EpistemicRestartCapsuleError::Manifest(_))
        ));
    }

    #[test]
    fn rejected_decision_history_is_part_of_restart_payload() {
        let fixture = fixture(6);
        let capsule = EpistemicRestartCapsuleV1::capture(
            &fixture.ledger,
            &fixture.inventory,
            &fixture.mutations,
            &fixture.revisions,
            6,
        )
        .unwrap();
        assert_eq!(capsule.revision_capsule().receipts().len(), 2);
        assert!(!capsule.revision_capsule().receipts()[1].eligible());
    }
}
