// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated checkpoint gossip and portable conflict proofs.
//!
//! Gossip records what identified observers signed.  It can preserve compact
//! evidence of rollback, same-height equivocation, or forks, but it does not
//! establish that an observer is independent or globally representative.

mod integrity;
mod model;

pub use integrity::*;
pub use model::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_calibration::{
        CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION,
        CalibrationPublicationCatalogCheckpoint, CalibrationSignerIdentity,
        calibration_publication_catalog_checkpoint_sha256,
    };

    struct AcceptVerifier;

    impl CalibrationPublicationGossipVerifier for AcceptVerifier {
        type Error = &'static str;

        fn verify(
            &self,
            _payload: &[u8],
            _signer: &CalibrationSignerIdentity,
            _signature: &[u8],
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    fn checkpoint(
        event_count: u64,
        catalog_marker: u64,
        previous: Option<String>,
    ) -> CalibrationPublicationCatalogCheckpoint {
        let mut checkpoint = CalibrationPublicationCatalogCheckpoint {
            checkpoint_version: CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION.into(),
            catalog_id: "catalog".into(),
            authority_id: "authority".into(),
            catalog_version: "catalog-v1".into(),
            catalog_sha256: format!("{:064x}", catalog_marker),
            record_count: event_count,
            event_count,
            head_event_sha256: Some(format!("{:064x}", catalog_marker + 1)),
            previous_checkpoint_sha256: previous,
            issued_epoch: event_count,
            checkpoint_sha256: String::new(),
        };
        checkpoint.checkpoint_sha256 = calibration_publication_catalog_checkpoint_sha256(&checkpoint);
        checkpoint
    }

    fn statement(
        observer: &str,
        checkpoint: CalibrationPublicationCatalogCheckpoint,
        previous_observed: Option<String>,
        epoch: u64,
    ) -> CalibrationSignedPublicationGossip {
        let payload = build_calibration_publication_gossip_payload(
            observer,
            checkpoint,
            previous_observed,
            "11".repeat(32),
            epoch,
        );
        build_calibration_signed_publication_gossip(
            payload,
            CalibrationSignerIdentity {
                key_id: observer.into(),
                algorithm: "test".into(),
                issuer: None,
            },
            &[1],
        )
    }

    #[test]
    fn authenticated_gossip_append_is_transactional() {
        let first_checkpoint = checkpoint(1, 10, None);
        let first = statement("mirror-a", first_checkpoint, None, 1);
        let mut ledger = build_calibration_publication_gossip_ledger("catalog", "authority");
        record_calibration_publication_gossip_statement(&mut ledger, first, &AcceptVerifier)
            .expect("record");
        let report = verify_calibration_publication_gossip_ledger(&ledger, &AcceptVerifier);
        assert!(report.accepted(), "{:?}", report.issues);
    }

    #[test]
    fn same_height_conflict_produces_portable_proof() {
        let first = statement("mirror-a", checkpoint(2, 20, None), None, 2);
        let second = statement("mirror-b", checkpoint(2, 21, None), None, 2);
        let proof = build_calibration_publication_gossip_conflict_proof(
            CalibrationPublicationGossipConflictKind::AuthorityEquivocation,
            first,
            second,
        )
        .expect("proof");
        assert!(audit_calibration_publication_gossip_conflict_proof(&proof).valid());
    }

    #[test]
    fn ledger_preserves_conflict_evidence_without_becoming_malformed() {
        let first = statement("mirror-a", checkpoint(2, 20, None), None, 2);
        let second = statement("mirror-b", checkpoint(2, 21, None), None, 2);
        let mut ledger = build_calibration_publication_gossip_ledger("catalog", "authority");
        record_calibration_publication_gossip_statement(&mut ledger, first, &AcceptVerifier)
            .expect("first");
        record_calibration_publication_gossip_statement(&mut ledger, second, &AcceptVerifier)
            .expect("second");
        let report = verify_calibration_publication_gossip_ledger(&ledger, &AcceptVerifier);
        assert!(report.integrity_valid());
        assert!(!report.accepted());
        assert!(report.authority_equivocation_detected);
        assert!(!extract_calibration_publication_gossip_conflict_proofs(&ledger).is_empty());
    }

    #[test]
    fn observer_rollback_is_detected() {
        let high = checkpoint(4, 40, None);
        let high_sha = high.checkpoint_sha256.clone();
        let low = checkpoint(3, 30, None);
        let first = statement("mirror-a", high, None, 4);
        let second = statement("mirror-a", low, Some(high_sha), 5);
        let mut ledger = build_calibration_publication_gossip_ledger("catalog", "authority");
        record_calibration_publication_gossip_statement(&mut ledger, first, &AcceptVerifier)
            .expect("first");
        record_calibration_publication_gossip_statement(&mut ledger, second, &AcceptVerifier)
            .expect("second");
        let report = audit_calibration_publication_gossip_ledger(&ledger);
        assert!(report.rollback_detected);
        assert!(report.integrity_valid());
    }

    #[test]
    fn tampered_statement_is_rejected_before_append() {
        let mut signed = statement("mirror-a", checkpoint(1, 10, None), None, 1);
        signed.payload.observed_epoch = 99;
        let mut ledger = build_calibration_publication_gossip_ledger("catalog", "authority");
        assert!(record_calibration_publication_gossip_statement(
            &mut ledger,
            signed,
            &AcceptVerifier,
        )
        .is_err());
        assert!(ledger.statements.is_empty());
    }
}
