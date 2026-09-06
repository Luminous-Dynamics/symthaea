include!("post_semantic.rs");

use symthaea_iot_actuation_guard_interlock::{
    CurrentPostSemanticInterlockError, CurrentPostSemanticInterlockGuard,
};

fn current_interlock_registry(
    controller_key: &SigningKey,
    challenge: &PostSemanticControllerChallengeV1,
) -> (InterlockTrustSnapshotV1, InterlockTrustRegistry) {
    let issued_at = challenge.issued_at_unix_ms().saturating_sub(1_000);
    let expires_at = challenge.expires_at_unix_ms().saturating_add(5_000);
    let snapshot = InterlockTrustSnapshotV1 {
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
    };
    let registry = InterlockTrustRegistry::genesis(snapshot.clone()).unwrap();
    (snapshot, registry)
}

#[test]
fn current_post_semantic_interlock_reverifies_fixed_key_and_rejects_generation_advance() {
    let admission_root = temp_root("admission-current-fence");
    let semantic_root = temp_root("semantic-current-fence");
    let semantic = semantic_acceptance(&admission_root, &semantic_root);
    let challenge = PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(
        &semantic,
    )
    .unwrap();
    let controller_key = SigningKey::from_bytes(&[0x71; 32]);

    let historical = interlock_state(
        &controller_key,
        &challenge,
        InterlockControllerKeyStatus::Active,
        exact_interlocks(),
    );
    let frame = controller_frame(&controller_key, &challenge, exact_interlocks());
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    let report_expires_at = decoded.report().statement.expires_at_unix_ms;
    let proof = historical
        .verify_post_semantic_controller(decoded, challenge)
        .unwrap();

    // Reconstruct the exact same independently anchored generation as current state. The
    // historical proof retained only commitments, not ownership of this registry value.
    let policy = physical_policy(exact_interlocks());
    let policy_digest = policy.digest().unwrap();
    let (snapshot, current_registry) =
        current_interlock_registry(&controller_key, proof.challenge());
    let current_head = current_registry.head();
    assert_eq!(current_head, proof.interlock_trust_head());
    let current = CurrentPostSemanticInterlockGuard::new(
        policy.clone(),
        policy_digest,
        current_registry,
        current_head,
    )
    .unwrap();

    let fence = current.fence_current(&proof).unwrap();
    assert_eq!(fence.proof().statement_digest(), proof.statement_digest());
    assert_eq!(
        fence.controller_report_expires_at_unix_ms(),
        report_expires_at
    );
    assert_eq!(
        fence.controller_key_not_after_unix_ms(),
        snapshot.keys[0].not_after_unix_ms
    );
    assert_eq!(
        fence.trust_snapshot_expires_at_unix_ms(),
        snapshot.expires_at_unix_ms
    );
    assert_eq!(fence.valid_until_unix_ms(), report_expires_at);

    // A successor trust generation kills the outstanding proof even when the exact same key
    // remains active. Currentness is generation-specific and cannot be retroactively inherited.
    let base = InterlockTrustRegistry::genesis(snapshot.clone()).unwrap();
    let successor = InterlockTrustSnapshotV1 {
        schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 2,
        issued_at_unix_ms: snapshot.issued_at_unix_ms,
        expires_at_unix_ms: snapshot.expires_at_unix_ms,
        previous_snapshot_digest: Some(base.head().digest),
        keys: snapshot.keys.clone(),
    };
    let advanced = base.successor(successor).unwrap();
    let advanced_head = advanced.head();
    let advanced_guard = CurrentPostSemanticInterlockGuard::new(
        policy,
        policy_digest,
        advanced,
        advanced_head,
    )
    .unwrap();
    assert!(matches!(
        advanced_guard.fence_current(&proof),
        Err(CurrentPostSemanticInterlockError::ProofTrustHeadMismatch)
    ));

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}
