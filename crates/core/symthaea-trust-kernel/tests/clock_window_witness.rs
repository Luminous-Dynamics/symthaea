// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockObservation, ClockObservationVerifier, ClockQuorumPolicy,
    ClockSignerV1, ClockWindowWitnessError, DetachedSignature, KeyLifecycleStatus, KeyTrustRecord,
    KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot,
    digest_clock_window_evaluation_witness, verify_clock_quorum, verify_clock_quorum_with_witness,
    verify_clock_window_evaluation_witness,
};

const WINDOW_E42: &str =
    "5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229";
const TRUST_DIGEST: &str =
    "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const OBS_A_E42: &str =
    "2825223c11e45e51555244013478b5d83feb601d559ba5fcbb6e103b2e220b0c";
const OBS_B_E42: &str =
    "a5f898ec3f8e095aea0ac7081eda2b249d94dcc39954556a0e0b1da72a9cd1a8";
const WITNESS_E42: &str =
    "e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2";

struct CountingVerifier {
    calls: Cell<usize>,
}

impl CountingVerifier {
    fn new() -> Self {
        Self { calls: Cell::new(0) }
    }
}

impl ClockObservationVerifier for CountingVerifier {
    fn verify_clock_observation(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        self.calls.set(self.calls.get() + 1);
        Ok(true)
    }
}

fn clock_key(algorithm: SignatureAlgorithm, key_id: &str) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.to_string(),
        not_before_unix_s: 900,
        not_after_unix_s: Some(3_000),
        status: KeyLifecycleStatus::Active,
        usages: BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity]),
    }
}

fn snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            clock_key(SignatureAlgorithm::MlDsa65, "clock-b"),
            clock_key(SignatureAlgorithm::Ed25519, "clock-a"),
        ],
    )
    .unwrap()
}

fn observation(
    source_id: &str,
    observed_unix_ms: u64,
    uncertainty_ms: u64,
    algorithm: SignatureAlgorithm,
    key_id: &str,
    signature_byte: u8,
) -> ClockObservation {
    ClockObservation {
        schema_version: CLOCK_OBSERVATION_SCHEMA.to_string(),
        source_id: source_id.to_string(),
        observed_unix_ms,
        uncertainty_ms,
        epoch: 42,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature: vec![signature_byte],
        },
    }
}

fn fixture() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            1_500_040,
            120,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
            0xb2,
        ),
        observation(
            "source-a",
            1_500_000,
            100,
            SignatureAlgorithm::Ed25519,
            "clock-a",
            0xa1,
        ),
    ]
}

#[test]
fn companion_witness_preserves_frozen_v1_window_exactly() {
    let observations = fixture();
    let snapshot = snapshot();
    let policy = ClockQuorumPolicy::default();
    let old_verifier = CountingVerifier::new();
    let new_verifier = CountingVerifier::new();

    let old_window = verify_clock_quorum(
        &observations,
        &policy,
        &snapshot,
        1_500,
        &old_verifier,
    )
    .unwrap();
    let (window, witness) = verify_clock_quorum_with_witness(
        &observations,
        &policy,
        &snapshot,
        1_500,
        &new_verifier,
    )
    .unwrap();

    assert_eq!(old_window, window);
    assert_eq!(old_verifier.calls.get(), 2);
    assert_eq!(new_verifier.calls.get(), 2);
    assert_eq!(window.evidence_digest.to_hex(), WINDOW_E42);
    assert_eq!(witness.window_evidence_digest.to_hex(), WINDOW_E42);
    assert_eq!(witness.trust_snapshot_digest.to_hex(), TRUST_DIGEST);
    assert_eq!(witness.epoch, 42);
    assert_eq!(
        witness
            .accepted_observation_digests
            .iter()
            .map(|digest| digest.to_hex())
            .collect::<Vec<_>>(),
        vec![OBS_A_E42, OBS_B_E42]
    );
    assert_eq!(
        witness.signers,
        vec![
            ClockSignerV1 {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "clock-a".to_string(),
            },
            ClockSignerV1 {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "clock-b".to_string(),
            },
        ]
    );
    assert_eq!(witness.observations[0].source_id, "source-a");
    assert_eq!(witness.observations[0].observed_unix_ms, 1_500_000);
    assert_eq!(witness.observations[0].uncertainty_ms, 100);
    assert_eq!(witness.observations[1].source_id, "source-b");
    assert_eq!(witness.observations[1].observed_unix_ms, 1_500_040);
    assert_eq!(witness.observations[1].uncertainty_ms, 120);
    assert_eq!(witness.witness_digest.to_hex(), WITNESS_E42);
    assert_eq!(
        digest_clock_window_evaluation_witness(&witness)
            .unwrap()
            .to_hex(),
        WITNESS_E42
    );
    assert_eq!(
        verify_clock_window_evaluation_witness(&window, &witness)
            .unwrap()
            .to_hex(),
        WITNESS_E42
    );
}

#[test]
fn companion_witness_is_input_order_invariant() {
    let snapshot = snapshot();
    let policy = ClockQuorumPolicy::default();
    let verifier = CountingVerifier::new();
    let observations = fixture();
    let (first_window, first_witness) = verify_clock_quorum_with_witness(
        &observations,
        &policy,
        &snapshot,
        1_500,
        &verifier,
    )
    .unwrap();

    let mut reordered = observations;
    reordered.reverse();
    let (second_window, second_witness) = verify_clock_quorum_with_witness(
        &reordered,
        &policy,
        &snapshot,
        1_500,
        &verifier,
    )
    .unwrap();

    assert_eq!(first_window, second_window);
    assert_eq!(first_witness, second_witness);
    assert_eq!(verifier.calls.get(), 4);
}

#[test]
fn witness_tampering_fails_closed() {
    let snapshot = snapshot();
    let verifier = CountingVerifier::new();
    let (window, witness) = verify_clock_quorum_with_witness(
        &fixture(),
        &ClockQuorumPolicy::default(),
        &snapshot,
        1_500,
        &verifier,
    )
    .unwrap();

    let mut bad_signer = witness.clone();
    bad_signer.signers[0].key_id = "clock-x".to_string();
    assert_eq!(
        verify_clock_window_evaluation_witness(&window, &bad_signer).unwrap_err(),
        ClockWindowWitnessError::SignerSetMismatch
    );

    let mut bad_observation = witness.clone();
    bad_observation.observations[0].uncertainty_ms += 1;
    assert!(matches!(
        verify_clock_window_evaluation_witness(&window, &bad_observation),
        Err(ClockWindowWitnessError::ObservationDigestMismatch(source)) if source == "source-a"
    ));

    let mut bad_set = witness.clone();
    bad_set.accepted_observation_digests[0] = Sha256Digest([1; 32]);
    assert_eq!(
        verify_clock_window_evaluation_witness(&window, &bad_set).unwrap_err(),
        ClockWindowWitnessError::ObservationSetMismatch
    );

    let mut bad_witness_digest = witness.clone();
    bad_witness_digest.witness_digest = Sha256Digest([7; 32]);
    assert_eq!(
        verify_clock_window_evaluation_witness(&window, &bad_witness_digest).unwrap_err(),
        ClockWindowWitnessError::WitnessDigestMismatch
    );

    let mut wrong_window = window;
    wrong_window.evidence_digest = Sha256Digest([9; 32]);
    assert_eq!(
        verify_clock_window_evaluation_witness(&wrong_window, &witness).unwrap_err(),
        ClockWindowWitnessError::WindowBindingMismatch
    );
}
