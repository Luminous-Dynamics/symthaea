mod fixture {
    include!("../../symthaea-iot-actuation-guard-interlock/tests/post_semantic.rs");

    use fips204::{
        ml_dsa_65,
        traits::{KeyGen, SerDes, Signer as MlDsaSigner},
    };
    use symthaea_iot_composed_actuation_evidence::{
        ComposedActuationEvidenceError, compose_actuation_evidence,
    };
    use symthaea_iot_transport_current_revalidation::{
        CurrentXeniaTransportRevalidator, RevalidatedXeniaTransport,
    };
    use symthaea_iot_transport_exact_evidence::bind_exact_xenia_transport_evidence;
    use symthaea_iot_transport_receipt::VerifiedTransportEnvelope;
    use symthaea_iot_xenia_hybrid_verifier::verify_xenia_physical_effect_receipt;

    fn real_transport_pair(
        now_ms: u64,
        physical_envelope: PhysicalEffectEnvelopeV1,
    ) -> (VerifiedTransportEnvelope, RevalidatedXeniaTransport) {
        let raw_payload = bincode::serialize(&physical_envelope).unwrap();
        let peer = [0x77; 32];
        let ed25519 = SigningKey::from_bytes(&[0x66; 32]);
        let (ml_dsa_public, ml_dsa_private) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
        let snapshot = TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: now_ms.saturating_sub(2_000),
            expires_at_unix_ms: now_ms + 30_000,
            previous_snapshot_digest: None,
            keys: vec![TransportAttestorKeyV1 {
                attestor_id: "attestor:composed-evidence".into(),
                key_id: "transport-key-1".into(),
                ed25519_public_key: ed25519.verifying_key().to_bytes(),
                ml_dsa_public_key: ml_dsa_public.into_bytes().to_vec(),
                status: TransportAttestorStatus::Active,
                not_before_unix_ms: now_ms.saturating_sub(5_000),
                not_after_unix_ms: now_ms + 20_000,
                max_receipt_lifetime_ms: 5_000,
                required_peer_role: XeniaReceiptPeerRoleV1::Host,
                allowed_peer_fingerprints: BTreeSet::from([peer]),
                require_input_control: true,
            }],
        };
        let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
            schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
            attestor_id: "attestor:composed-evidence".into(),
            key_id: "transport-key-1".into(),
            signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
            session_evidence_digest: [0x61; 32],
            peer_role: XeniaReceiptPeerRoleV1::Host,
            peer_identity_fingerprint: peer,
            transcript_hash: [0x62; 32],
            session_context_hash: [0x63; 32],
            telemetry_enabled: true,
            input_control_enabled: true,
            payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
            payload_len: u32::try_from(raw_payload.len()).unwrap(),
            payload_digest: *blake3::hash(&raw_payload).as_bytes(),
            sealed_envelope_digest: [0x64; 32],
            opened_at_unix_ms: now_ms,
            expires_at_unix_ms: now_ms + 4_000,
        };
        let digest = body.signing_digest().unwrap();
        let ml_dsa_signature = ml_dsa_private
            .try_sign_with_seed(&[0x88; 32], &digest, &[])
            .expect("deterministic ML-DSA-65 signature");
        let receipt = XeniaAuthenticatedPayloadReceiptV1 {
            body,
            ed25519_signature: ed25519.sign(&digest).to_bytes(),
            ml_dsa_signature,
        };
        let raw_receipt = bincode::serialize(&receipt).unwrap();

        // Verify the same exact portable evidence independently for each affine branch. No opaque
        // proof is cloned or forked.
        let semantic_registry = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
        let semantic_transport = verify_xenia_physical_effect_receipt(
            &semantic_registry,
            &raw_receipt,
            &raw_payload,
            now_ms,
        )
        .unwrap();

        let exact_registry = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
        let exact_verified = verify_xenia_physical_effect_receipt(
            &exact_registry,
            &raw_receipt,
            &raw_payload,
            now_ms,
        )
        .unwrap();
        let exact = bind_exact_xenia_transport_evidence(exact_verified, &raw_receipt, &raw_payload)
            .unwrap();

        let current_registry = TransportTrustRegistry::genesis(snapshot).unwrap();
        let current_head = current_registry.head();
        let current_transport = CurrentXeniaTransportRevalidator::new(current_registry, current_head)
            .unwrap()
            .revalidate(exact)
            .unwrap();

        (semantic_transport, current_transport)
    }

    fn semantic_acceptance_from_transport(
        admission_root: &PathBuf,
        semantic_root: &PathBuf,
        transport: VerifiedTransportEnvelope,
    ) -> PersistedSemanticAcceptance {
        let admission_store =
            DurableAdmissionReservationStore::open(admission_root, config()).unwrap();
        let reservation = admission_store.reserve_verified_transport(transport).unwrap();
        let challenge =
            AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
        let device_key = SigningKey::from_bytes(&[0x61; 32]);
        let reality_state = reality_state(
            &device_key,
            reservation.persisted_at_unix_ms().saturating_sub(1_000),
            challenge.expires_at_unix_ms().saturating_add(5_000),
        );
        let frame = signed_device_response(&device_key, &challenge);
        let decoded = decode_admission_device_reality_response(&frame, &challenge).unwrap();
        let reality = reality_state
            .verify_admission_evidence(decoded, &challenge)
            .unwrap();

        let cfg = config();
        let genesis_head = DeviceSemanticCheckpointV1::genesis(&cfg)
            .unwrap()
            .head()
            .unwrap();
        DurableSemanticAcceptanceStore::open(semantic_root, cfg, genesis_head)
            .unwrap()
            .persist_semantic_acceptance(reservation, reality)
            .unwrap()
    }

    fn post_semantic_interlock(
        semantic: &PersistedSemanticAcceptance,
    ) -> symthaea_iot_actuation_guard_interlock::VerifiedPostSemanticPhysicalInterlock {
        let challenge =
            PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(semantic)
                .unwrap();
        let controller_key = SigningKey::from_bytes(&[0x71; 32]);
        let state = interlock_state(
            &controller_key,
            &challenge,
            InterlockControllerKeyStatus::Active,
            exact_interlocks(),
        );
        let frame = controller_frame(&controller_key, &challenge, exact_interlocks());
        let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
        state
            .verify_post_semantic_controller(decoded, challenge)
            .unwrap()
    }

    #[test]
    fn independently_verified_transport_and_durable_chain_compose_without_authority() {
        let admission_root = temp_root("compose-admission");
        let semantic_root = temp_root("compose-semantic");
        let now = wall_ms();
        let (semantic_transport, current_transport) = real_transport_pair(now, envelope(now));
        let semantic =
            semantic_acceptance_from_transport(&admission_root, &semantic_root, semantic_transport);
        let interlock = post_semantic_interlock(&semantic);

        let expected_envelope = current_transport.envelope_digest();
        let expected_receipt = current_transport.receipt_digest();
        let expected_admission_head = semantic.admission_reservation().head();
        let expected_semantic_head = semantic.device_head();
        let expected_statement = interlock.statement_digest();

        let composed = compose_actuation_evidence(current_transport, semantic, interlock).unwrap();

        assert_eq!(composed.transport().envelope_digest(), expected_envelope);
        assert_eq!(composed.transport().receipt_digest(), expected_receipt);
        assert_eq!(
            composed
                .semantic_acceptance()
                .admission_reservation()
                .head(),
            expected_admission_head
        );
        assert_eq!(
            composed.semantic_acceptance().device_head(),
            expected_semantic_head
        );
        assert_eq!(
            composed.post_semantic_interlock().statement_digest(),
            expected_statement
        );
        assert_ne!(composed.composition_digest(), Digest32([0; 32]));

        std::fs::remove_dir_all(admission_root).unwrap();
        std::fs::remove_dir_all(semantic_root).unwrap();
    }

    #[test]
    fn independently_valid_but_different_transport_lineage_cannot_compose() {
        let admission_root = temp_root("compose-mismatch-admission");
        let semantic_root = temp_root("compose-mismatch-semantic");
        let now = wall_ms();
        let (semantic_transport, _matching_current) = real_transport_pair(now, envelope(now));
        let semantic =
            semantic_acceptance_from_transport(&admission_root, &semantic_root, semantic_transport);
        let interlock = post_semantic_interlock(&semantic);

        let mut different_envelope = envelope(now);
        different_envelope.proposal_digest = d(0xE1);
        let (_unused_semantic, different_current) = real_transport_pair(now, different_envelope);

        assert!(matches!(
            compose_actuation_evidence(different_current, semantic, interlock),
            Err(ComposedActuationEvidenceError::PhysicalEnvelopeMismatch)
        ));

        std::fs::remove_dir_all(admission_root).unwrap();
        std::fs::remove_dir_all(semantic_root).unwrap();
    }
}
