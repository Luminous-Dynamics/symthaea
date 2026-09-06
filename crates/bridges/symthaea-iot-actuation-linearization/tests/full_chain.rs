mod fixture {
    include!("../../symthaea-iot-actuation-guard-interlock/tests/post_semantic.rs");

    use std::error::Error as StdError;
    use std::fmt;

    use fips204::{
        ml_dsa_65,
        traits::{KeyGen, SerDes, Signer as MlDsaSigner},
    };
    use symthaea_iot_actuation_effect_dispatch::{
        AdapterAttemptAcknowledgement, AuthorizedPhysicalEffectRequest,
        PhysicalEffectDispatchError, PrivilegedPhysicalEffectPort, dispatch_current_attempt,
    };
    use symthaea_iot_actuation_guard_device_reality::CurrentAdmissionDeviceRealityGuard;
    use symthaea_iot_actuation_guard_interlock::{
        CurrentPostSemanticInterlockGuard, VerifiedPostSemanticPhysicalInterlock,
    };
    use symthaea_iot_actuation_linearization::ActuationLinearizer;
    use symthaea_iot_actuation_trust_publication::{
        ActuationPolicyAnchorV1, ActuationTrustRootsV1, DurableActuationTrustPublicationStore,
    };
    use symthaea_iot_composed_actuation_evidence::{
        ComposedActuationEvidence, compose_actuation_evidence,
    };
    use symthaea_iot_transport_current_fence::CurrentXeniaTransportFenceGuard;
    use symthaea_iot_transport_current_revalidation::{
        CurrentXeniaTransportRevalidator, RevalidatedXeniaTransport,
    };
    use symthaea_iot_transport_exact_evidence::bind_exact_xenia_transport_evidence;
    use symthaea_iot_transport_receipt::VerifiedTransportEnvelope;
    use symthaea_iot_xenia_hybrid_verifier::verify_xenia_physical_effect_receipt;

    fn real_transport_bundle(
        now_ms: u64,
        physical_envelope: PhysicalEffectEnvelopeV1,
    ) -> (
        VerifiedTransportEnvelope,
        RevalidatedXeniaTransport,
        CurrentXeniaTransportFenceGuard,
    ) {
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
                attestor_id: "attestor:linearized-evidence".into(),
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
            attestor_id: "attestor:linearized-evidence".into(),
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

        let revalidation_registry = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
        let current_head = revalidation_registry.head();
        let current_transport =
            CurrentXeniaTransportRevalidator::new(revalidation_registry, current_head)
                .unwrap()
                .revalidate(exact)
                .unwrap();

        let guard_registry = TransportTrustRegistry::genesis(snapshot).unwrap();
        let current_guard =
            CurrentXeniaTransportFenceGuard::new(guard_registry, current_head).unwrap();

        (semantic_transport, current_transport, current_guard)
    }

    fn device_trust_snapshot(
        signing_key: &SigningKey,
        issued_at_unix_ms: u64,
        expires_at_unix_ms: u64,
    ) -> DeviceRealityTrustSnapshotV1 {
        DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms,
            expires_at_unix_ms,
            previous_snapshot_digest: None,
            keys: vec![DeviceRealityVerifierKeyV1 {
                verifier_id: "verifier:fleet-a".into(),
                key_id: "device-key-1".into(),
                algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
                public_key: signing_key.verifying_key().to_bytes(),
                status: DeviceRealityVerifierKeyStatus::Active,
                not_before_unix_ms: issued_at_unix_ms,
                not_after_unix_ms: expires_at_unix_ms,
                max_result_lifetime_ms: 2_000,
            }],
        }
    }

    fn semantic_bundle(
        admission_root: &PathBuf,
        semantic_root: &PathBuf,
        transport: VerifiedTransportEnvelope,
    ) -> (PersistedSemanticAcceptance, CurrentAdmissionDeviceRealityGuard) {
        let admission_store =
            DurableAdmissionReservationStore::open(admission_root, config()).unwrap();
        let reservation = admission_store.reserve_verified_transport(transport).unwrap();
        let challenge =
            AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
        let device_key = SigningKey::from_bytes(&[0x61; 32]);
        let trust_issued_at = reservation.persisted_at_unix_ms().saturating_sub(1_000);
        let trust_expires_at = challenge.expires_at_unix_ms().saturating_add(5_000);
        let policy_digest = reality_policy().digest().unwrap();

        let verification_registry = DeviceRealityTrustRegistry::genesis(device_trust_snapshot(
            &device_key,
            trust_issued_at,
            trust_expires_at,
        ))
        .unwrap();
        let trust_head = verification_registry.head();
        let verification_state = GuardAdmissionDeviceRealityState::new(
            reality_policy(),
            policy_digest,
            verification_registry,
            trust_head,
        )
        .unwrap();

        let frame = signed_device_response(&device_key, &challenge);
        let decoded = decode_admission_device_reality_response(&frame, &challenge).unwrap();
        let reality = verification_state
            .verify_admission_evidence(decoded, &challenge)
            .unwrap();

        let current_registry = DeviceRealityTrustRegistry::genesis(device_trust_snapshot(
            &device_key,
            trust_issued_at,
            trust_expires_at,
        ))
        .unwrap();
        let current_guard = CurrentAdmissionDeviceRealityGuard::new(
            reality_policy(),
            policy_digest,
            current_registry,
            trust_head,
        )
        .unwrap();

        let cfg = config();
        let genesis_head = DeviceSemanticCheckpointV1::genesis(&cfg)
            .unwrap()
            .head()
            .unwrap();
        let semantic = DurableSemanticAcceptanceStore::open(semantic_root, cfg, genesis_head)
            .unwrap()
            .persist_semantic_acceptance(reservation, reality)
            .unwrap();

        (semantic, current_guard)
    }

    fn interlock_trust_snapshot(
        controller_key: &SigningKey,
        challenge: &PostSemanticControllerChallengeV1,
    ) -> InterlockTrustSnapshotV1 {
        let issued_at = challenge.issued_at_unix_ms().saturating_sub(1_000);
        let expires_at = challenge.expires_at_unix_ms().saturating_add(5_000);
        InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: issued_at,
            expires_at_unix_ms: expires_at,
            previous_snapshot_digest: None,
            keys: vec![InterlockControllerKeyV1 {
                controller_id: "controller:valve-72".into(),
                key_id: "controller-key-1".into(),
                algorithm: INTERLOCK_ED25519_ALGORITHM.into(),
                public_key: controller_key.verifying_key().to_bytes().to_vec(),
                status: InterlockControllerKeyStatus::Active,
                not_before_unix_ms: issued_at,
                not_after_unix_ms: expires_at,
            }],
        }
    }

    fn interlock_bundle(
        semantic: &PersistedSemanticAcceptance,
    ) -> (
        VerifiedPostSemanticPhysicalInterlock,
        CurrentPostSemanticInterlockGuard,
    ) {
        let challenge =
            PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(semantic)
                .unwrap();
        let controller_key = SigningKey::from_bytes(&[0x71; 32]);
        let policy_digest = physical_policy(exact_interlocks()).digest().unwrap();

        let verification_registry =
            InterlockTrustRegistry::genesis(interlock_trust_snapshot(&controller_key, &challenge))
                .unwrap();
        let trust_head = verification_registry.head();
        let verification_state = GuardInterlockState::new(
            physical_policy(exact_interlocks()),
            policy_digest,
            verification_registry,
            trust_head,
        )
        .unwrap();
        let frame = controller_frame(&controller_key, &challenge, exact_interlocks());
        let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
        let proof = verification_state
            .verify_post_semantic_controller(decoded, challenge.clone())
            .unwrap();

        let current_registry =
            InterlockTrustRegistry::genesis(interlock_trust_snapshot(&controller_key, &challenge))
                .unwrap();
        let current_guard = CurrentPostSemanticInterlockGuard::new(
            physical_policy(exact_interlocks()),
            policy_digest,
            current_registry,
            trust_head,
        )
        .unwrap();

        (proof, current_guard)
    }

    fn complete_composed_fixture(
        admission_root: &PathBuf,
        semantic_root: &PathBuf,
    ) -> (
        ComposedActuationEvidence,
        CurrentXeniaTransportFenceGuard,
        CurrentAdmissionDeviceRealityGuard,
        CurrentPostSemanticInterlockGuard,
    ) {
        let now = wall_ms();
        let (semantic_transport, current_transport, transport_guard) =
            real_transport_bundle(now, envelope(now));
        let (semantic, device_reality_guard) =
            semantic_bundle(admission_root, semantic_root, semantic_transport);
        let (interlock, interlock_guard) = interlock_bundle(&semantic);
        let composed = compose_actuation_evidence(current_transport, semantic, interlock).unwrap();
        (
            composed,
            transport_guard,
            device_reality_guard,
            interlock_guard,
        )
    }

    fn published_roots(
        composed: &ComposedActuationEvidence,
        transport_guard: &CurrentXeniaTransportFenceGuard,
        device_reality_guard: &CurrentAdmissionDeviceRealityGuard,
        interlock_guard: &CurrentPostSemanticInterlockGuard,
    ) -> ActuationTrustRootsV1 {
        ActuationTrustRootsV1 {
            device: composed
                .semantic_acceptance()
                .admission_reservation()
                .envelope()
                .command
                .device
                .clone(),
            transport_trust_head: transport_guard.anchored_current_head(),
            device_reality_trust_head: device_reality_guard.anchored_trust_head(),
            device_reality_policy: ActuationPolicyAnchorV1 {
                generation: 1,
                digest: device_reality_guard.anchored_policy_digest(),
            },
            interlock_trust_head: interlock_guard.anchored_trust_head(),
            interlock_policy: ActuationPolicyAnchorV1 {
                generation: 1,
                digest: interlock_guard.anchored_policy_digest(),
            },
        }
    }

    #[derive(Debug)]
    struct RecordingPortError;

    impl fmt::Display for RecordingPortError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("recording port failure")
        }
    }

    impl StdError for RecordingPortError {}

    struct RecordingPort {
        adapter_id: String,
        device: ResourceRef,
        operation: Operation,
        executor: PrincipalId,
        calls: usize,
        seen_command_digest: Option<Digest32>,
        seen_envelope_digest: Option<Digest32>,
        seen_composition_digest: Option<Digest32>,
    }

    impl RecordingPort {
        fn from_command(command: &DeviceCommand) -> Self {
            Self {
                adapter_id: "mock-hal:valve-72".into(),
                device: command.device.clone(),
                operation: command.operation.clone(),
                executor: command.executor.clone(),
                calls: 0,
                seen_command_digest: None,
                seen_envelope_digest: None,
                seen_composition_digest: None,
            }
        }
    }

    impl PrivilegedPhysicalEffectPort for RecordingPort {
        type Error = RecordingPortError;

        fn adapter_id(&self) -> &str {
            &self.adapter_id
        }

        fn device(&self) -> &ResourceRef {
            &self.device
        }

        fn operation(&self) -> &Operation {
            &self.operation
        }

        fn executor(&self) -> &PrincipalId {
            &self.executor
        }

        fn attempt_effect(
            &mut self,
            request: AuthorizedPhysicalEffectRequest<'_>,
        ) -> Result<AdapterAttemptAcknowledgement, Self::Error> {
            self.calls += 1;
            self.seen_command_digest = Some(request.command_digest());
            self.seen_envelope_digest = Some(request.envelope_digest());
            self.seen_composition_digest = Some(request.composition_digest());
            assert_eq!(request.command().digest(), request.command_digest());
            AdapterAttemptAcknowledgement::new(d(0xE7)).map_err(|_| RecordingPortError)
        }
    }

    #[test]
    fn real_two_branch_chain_linearizes_under_all_held_roots_without_hal_io() {
        let admission_root = temp_root("linearize-admission");
        let semantic_root = temp_root("linearize-semantic");
        let trust_root = temp_root("linearize-trust");
        let (composed, transport_guard, device_reality_guard, interlock_guard) =
            complete_composed_fixture(&admission_root, &semantic_root);

        let expected_digest = composed.composition_digest();
        let expected_device = composed
            .semantic_acceptance()
            .admission_reservation()
            .envelope()
            .command
            .device
            .clone();
        let semantic_head = composed.semantic_acceptance().device_head();

        let roots = published_roots(
            &composed,
            &transport_guard,
            &device_reality_guard,
            &interlock_guard,
        );
        let publication =
            DurableActuationTrustPublicationStore::initialize(&trust_root, roots).unwrap();

        let observed_digest = {
            let trust_store =
                DurableActuationTrustPublicationStore::open(&trust_root, publication.head())
                    .unwrap();
            let admission_store =
                DurableAdmissionReservationStore::open(&admission_root, config()).unwrap();
            let semantic_store =
                DurableSemanticAcceptanceStore::open(&semantic_root, config(), semantic_head)
                    .unwrap();
            let linearizer = ActuationLinearizer::new(
                &trust_store,
                &admission_store,
                &semantic_store,
                &transport_guard,
                &device_reality_guard,
                &interlock_guard,
            );

            linearizer
                .with_current_attempt(composed, |attempt| {
                    assert_eq!(attempt.device(), &expected_device);
                    assert_eq!(attempt.composition_digest(), expected_digest);
                    assert!(attempt.wall_valid_until_unix_ms() > attempt.common_fenced_at_unix_ms());
                    attempt.validate_dispatch_window_now().unwrap();
                    attempt.composition_digest()
                })
                .unwrap()
        };

        assert_eq!(observed_digest, expected_digest);
        std::fs::remove_dir_all(admission_root).unwrap();
        std::fs::remove_dir_all(semantic_root).unwrap();
        std::fs::remove_dir_all(trust_root).unwrap();
    }

    #[test]
    fn real_two_branch_chain_reaches_matching_privileged_port_exactly_once() {
        let admission_root = temp_root("dispatch-admission");
        let semantic_root = temp_root("dispatch-semantic");
        let trust_root = temp_root("dispatch-trust");
        let (composed, transport_guard, device_reality_guard, interlock_guard) =
            complete_composed_fixture(&admission_root, &semantic_root);

        let expected_command = composed
            .semantic_acceptance()
            .admission_reservation()
            .envelope()
            .command
            .clone();
        let expected_command_digest = expected_command.digest();
        let expected_envelope_digest = composed.transport().envelope_digest();
        let expected_composition_digest = composed.composition_digest();
        let semantic_head = composed.semantic_acceptance().device_head();
        let roots = published_roots(
            &composed,
            &transport_guard,
            &device_reality_guard,
            &interlock_guard,
        );
        let publication =
            DurableActuationTrustPublicationStore::initialize(&trust_root, roots).unwrap();
        let mut port = RecordingPort::from_command(&expected_command);

        let record = {
            let trust_store =
                DurableActuationTrustPublicationStore::open(&trust_root, publication.head())
                    .unwrap();
            let admission_store =
                DurableAdmissionReservationStore::open(&admission_root, config()).unwrap();
            let semantic_store =
                DurableSemanticAcceptanceStore::open(&semantic_root, config(), semantic_head)
                    .unwrap();
            let linearizer = ActuationLinearizer::new(
                &trust_store,
                &admission_store,
                &semantic_store,
                &transport_guard,
                &device_reality_guard,
                &interlock_guard,
            );

            linearizer
                .with_current_attempt(composed, |attempt| {
                    dispatch_current_attempt(attempt, &mut port)
                })
                .unwrap()
                .unwrap()
        };

        assert_eq!(port.calls, 1);
        assert_eq!(port.seen_command_digest, Some(expected_command_digest));
        assert_eq!(port.seen_envelope_digest, Some(expected_envelope_digest));
        assert_eq!(port.seen_composition_digest, Some(expected_composition_digest));
        assert_eq!(record.correlation().command_digest(), expected_command_digest);
        assert_eq!(record.correlation().envelope_digest(), expected_envelope_digest);
        assert_eq!(
            record.correlation().composition_digest(),
            expected_composition_digest
        );
        assert_eq!(record.correlation().sequence(), expected_command.sequence);
        assert_eq!(record.correlation().adapter_id(), "mock-hal:valve-72");
        assert_eq!(record.adapter_evidence_digest(), d(0xE7));

        std::fs::remove_dir_all(admission_root).unwrap();
        std::fs::remove_dir_all(semantic_root).unwrap();
        std::fs::remove_dir_all(trust_root).unwrap();
    }

    #[test]
    fn wrong_privileged_port_binding_never_invokes_effect_method() {
        let admission_root = temp_root("dispatch-wrong-port-admission");
        let semantic_root = temp_root("dispatch-wrong-port-semantic");
        let trust_root = temp_root("dispatch-wrong-port-trust");
        let (composed, transport_guard, device_reality_guard, interlock_guard) =
            complete_composed_fixture(&admission_root, &semantic_root);

        let expected_command = composed
            .semantic_acceptance()
            .admission_reservation()
            .envelope()
            .command
            .clone();
        let semantic_head = composed.semantic_acceptance().device_head();
        let roots = published_roots(
            &composed,
            &transport_guard,
            &device_reality_guard,
            &interlock_guard,
        );
        let publication =
            DurableActuationTrustPublicationStore::initialize(&trust_root, roots).unwrap();
        let mut port = RecordingPort::from_command(&expected_command);
        port.device = ResourceRef("iot:valve:attacker".into());

        let result = {
            let trust_store =
                DurableActuationTrustPublicationStore::open(&trust_root, publication.head())
                    .unwrap();
            let admission_store =
                DurableAdmissionReservationStore::open(&admission_root, config()).unwrap();
            let semantic_store =
                DurableSemanticAcceptanceStore::open(&semantic_root, config(), semantic_head)
                    .unwrap();
            let linearizer = ActuationLinearizer::new(
                &trust_store,
                &admission_store,
                &semantic_store,
                &transport_guard,
                &device_reality_guard,
                &interlock_guard,
            );

            linearizer
                .with_current_attempt(composed, |attempt| {
                    dispatch_current_attempt(attempt, &mut port)
                })
                .unwrap()
        };

        assert!(matches!(
            result,
            Err(PhysicalEffectDispatchError::PortDeviceMismatch)
        ));
        assert_eq!(port.calls, 0);

        std::fs::remove_dir_all(admission_root).unwrap();
        std::fs::remove_dir_all(semantic_root).unwrap();
        std::fs::remove_dir_all(trust_root).unwrap();
    }
}
