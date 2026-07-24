// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable witness-policy, head, and authenticated-gossip continuity bundles.

mod integrity;
mod model;

pub use integrity::*;
pub use model::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_calibration::*;

    struct AcceptVerifier;

    impl CalibrationPublicationCheckpointWitnessVerifier for AcceptVerifier {
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
    impl CalibrationPublicationWitnessPolicyRotationVerifier for AcceptVerifier {
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

    fn signer(key_id: &str) -> CalibrationSignerIdentity {
        CalibrationSignerIdentity {
            key_id: key_id.into(),
            algorithm: "test".into(),
            issuer: None,
        }
    }

    fn fixture() -> (
        CalibrationPublicationCatalogHeadBundle,
        CalibrationPublicationWitnessPolicyLedger,
        CalibrationPublicationGossipLedger,
    ) {
        let catalog = build_calibration_publication_catalog("catalog", "authority");
        let checkpoint = build_calibration_publication_catalog_checkpoint(&catalog, None, 1)
            .expect("checkpoint");
        let policy = build_calibration_publication_checkpoint_witness_policy(1, vec!["w1".into()]);
        let payload = build_calibration_publication_checkpoint_witness_payload(&checkpoint, 1);
        let statement = build_calibration_signed_publication_checkpoint_witness(
            payload,
            signer("w1"),
            &[1],
        );
        let witness_set = build_calibration_publication_checkpoint_witness_set(
            &checkpoint,
            policy.clone(),
            vec![statement],
        );
        let head = build_calibration_publication_catalog_head_bundle(
            catalog,
            checkpoint.clone(),
            None,
            witness_set,
            None,
            Vec::new(),
            &AcceptVerifier,
        )
        .expect("head");
        let policy_ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint.clone(),
            policy,
            1,
        )
        .expect("policy ledger");
        let gossip_payload = build_calibration_publication_gossip_payload(
            "mirror-a",
            checkpoint,
            None,
            policy_ledger.epochs[0].epoch_sha256.clone(),
            1,
        );
        let gossip_statement = build_calibration_signed_publication_gossip(
            gossip_payload,
            signer("mirror-a"),
            &[2],
        );
        let mut gossip = build_calibration_publication_gossip_ledger("catalog", "authority");
        record_calibration_publication_gossip_statement(
            &mut gossip,
            gossip_statement,
            &AcceptVerifier,
        )
        .expect("gossip");
        (head, policy_ledger, gossip)
    }

    #[test]
    fn authenticated_continuity_bundle_is_accepted() {
        let (head, policy, gossip) = fixture();
        let bundle = build_calibration_publication_continuity_bundle(
            head,
            policy,
            Some(gossip),
            Vec::new(),
            &AcceptVerifier,
            &AcceptVerifier,
            &AcceptVerifier,
        )
        .expect("bundle");
        let report = verify_calibration_publication_continuity_bundle(
            &bundle,
            &AcceptVerifier,
            &AcceptVerifier,
            &AcceptVerifier,
        );
        assert!(report.accepted(), "{:?}", report.issues);
    }

    #[test]
    fn wrong_active_policy_is_rejected() {
        let (mut head, policy, gossip) = fixture();
        head.witness_set.policy = build_calibration_publication_checkpoint_witness_policy(
            1,
            vec!["other".into()],
        );
        head.witness_set.set_sha256 = calibration_publication_checkpoint_witness_set_sha256(
            &head.witness_set,
        );
        head.bundle_sha256 = calibration_publication_catalog_head_bundle_sha256(&head);
        let mut bundle = CalibrationPublicationContinuityBundle {
            bundle_version: CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_VERSION.into(),
            head_bundle: head,
            witness_policy_ledger: policy,
            gossip_ledger: Some(gossip),
            conflict_proofs: Vec::new(),
            limitations: calibration_publication_continuity_required_limitations(),
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = calibration_publication_continuity_bundle_sha256(&bundle);
        let report = verify_calibration_publication_continuity_bundle(
            &bundle,
            &AcceptVerifier,
            &AcceptVerifier,
            &AcceptVerifier,
        );
        assert!(!report.accepted());
        assert!(report.issues.iter().any(|issue| {
            issue.code == CalibrationPublicationContinuityIssueCode::ActivePolicyMismatch
        }));
    }

    #[test]
    fn gossip_policy_epoch_mismatch_is_rejected() {
        let (head, policy, mut gossip) = fixture();
        gossip.statements[0].payload.witness_policy_epoch_sha256 = "22".repeat(32);
        gossip.statements[0].payload.payload_sha256 =
            calibration_publication_gossip_payload_sha256(&gossip.statements[0].payload);
        gossip.statements[0].envelope_sha256 =
            calibration_signed_publication_gossip_sha256(&gossip.statements[0]);
        gossip.ledger_sha256 = calibration_publication_gossip_ledger_sha256(&gossip);
        let mut bundle = CalibrationPublicationContinuityBundle {
            bundle_version: CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_VERSION.into(),
            head_bundle: head,
            witness_policy_ledger: policy,
            gossip_ledger: Some(gossip),
            conflict_proofs: Vec::new(),
            limitations: calibration_publication_continuity_required_limitations(),
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = calibration_publication_continuity_bundle_sha256(&bundle);
        let report = verify_calibration_publication_continuity_bundle(
            &bundle,
            &AcceptVerifier,
            &AcceptVerifier,
            &AcceptVerifier,
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CalibrationPublicationContinuityIssueCode::GossipPolicyEpochMismatch
        }));
    }
}
