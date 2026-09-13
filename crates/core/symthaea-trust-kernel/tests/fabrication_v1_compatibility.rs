// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact compatibility with #2283's frozen fabrication trust/clock protocol.

use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockContinuityPolicy, ClockEpochTracker, ClockObservation,
    ClockObservationVerifier, ClockQuorumPolicy, ClockTrackingError, DetachedSignature,
    KeyEligibility, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, SignatureAlgorithm,
    TrustSnapshot, TrustSnapshotTracker, TrustSnapshotTrackingError,
    canonical_trust_snapshot_bytes, digest_clock_continuity, digest_clock_observation,
    digest_trust_snapshot, verify_clock_continuity, verify_clock_quorum,
};

const TRUST_DIGEST: &str =
    "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const OBS_A_E42: &str =
    "2825223c11e45e51555244013478b5d83feb601d559ba5fcbb6e103b2e220b0c";
const OBS_B_E42: &str =
    "a5f898ec3f8e095aea0ac7081eda2b249d94dcc39954556a0e0b1da72a9cd1a8";
const WINDOW_E42: &str =
    "5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229";
const WINDOW_E43: &str =
    "d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b";
const CONTINUITY_42_43: &str =
    "ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15";

const CANONICAL_TRUST_JSON: &str = concat!(
    "{\"schema_version\":\"symthaea.fabrication.trust-snapshot.v1\",",
    "\"sequence\":7,\"issued_at_unix_s\":1000,\"expires_at_unix_s\":2000,",
    "\"keys\":[",
    "{\"algorithm\":\"Ed25519\",\"key_id\":\"clock-a\",",
    "\"not_before_unix_s\":900,\"not_after_unix_s\":3000,",
    "\"status\":\"Active\",\"usages\":[\"ClockAuthority\",\"ClockContinuity\"]},",
    "{\"algorithm\":\"MlDsa65\",\"key_id\":\"clock-b\",",
    "\"not_before_unix_s\":900,\"not_after_unix_s\":3000,",
    "\"status\":\"Active\",\"usages\":[\"ClockAuthority\",\"ClockContinuity\"]}",
    "]}"
);

struct AcceptAllSignatures;

impl ClockObservationVerifier for AcceptAllSignatures {
    fn verify_clock_observation(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
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
    epoch: u64,
    algorithm: SignatureAlgorithm,
    key_id: &str,
    signature_byte: u8,
) -> ClockObservation {
    ClockObservation {
        schema_version: CLOCK_OBSERVATION_SCHEMA.to_string(),
        source_id: source_id.to_string(),
        observed_unix_ms,
        uncertainty_ms,
        epoch,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature: vec![signature_byte],
        },
    }
}

fn epoch_42() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            1_500_040,
            120,
            42,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
            0xb2,
        ),
        observation(
            "source-a",
            1_500_000,
            100,
            42,
            SignatureAlgorithm::Ed25519,
            "clock-a",
            0xa1,
        ),
    ]
}

fn epoch_43() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-a",
            1_500_500,
            100,
            43,
            SignatureAlgorithm::Ed25519,
            "clock-a",
            0xa3,
        ),
        observation(
            "source-b",
            1_500_540,
            120,
            43,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
            0xb4,
        ),
    ]
}

#[test]
fn generic_snapshot_is_byte_identical_to_fabrication_v1() {
    let snapshot = snapshot();
    let bytes = canonical_trust_snapshot_bytes(&snapshot).unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), CANONICAL_TRUST_JSON);
    assert_eq!(digest_trust_snapshot(&snapshot).unwrap().to_hex(), TRUST_DIGEST);

    assert_eq!(
        snapshot.key_eligibility(
            &SignatureAlgorithm::Ed25519,
            "clock-a",
            KeyUsage::ClockAuthority,
            1_500,
        ),
        KeyEligibility::Eligible
    );
    assert_eq!(
        snapshot.key_eligibility(
            &SignatureAlgorithm::Ed25519,
            "clock-a",
            KeyUsage::PolicyMigration,
            1_500,
        ),
        KeyEligibility::UsageNotAllowed
    );
    assert_eq!(
        snapshot.key_eligibility(
            &SignatureAlgorithm::Ed25519,
            "clock-a",
            KeyUsage::ClockAuthority,
            899,
        ),
        KeyEligibility::NotYetValid
    );
    assert_eq!(
        snapshot.key_eligibility(
            &SignatureAlgorithm::Ed25519,
            "clock-a",
            KeyUsage::ClockAuthority,
            3_000,
        ),
        KeyEligibility::Expired
    );
}

#[test]
fn generic_snapshot_tracker_preserves_fabrication_v1_semantics() {
    let baseline = snapshot();
    let mut tracker = TrustSnapshotTracker::default();
    assert_eq!(tracker.accept(&baseline).unwrap().to_hex(), TRUST_DIGEST);
    assert_eq!(tracker.accept(&baseline).unwrap().to_hex(), TRUST_DIGEST);

    let rollback = TrustSnapshot::new(
        6,
        1_100,
        2_100,
        vec![clock_key(SignatureAlgorithm::Ed25519, "clock-a")],
    )
    .unwrap();
    assert_eq!(
        tracker.accept(&rollback).unwrap_err(),
        TrustSnapshotTrackingError::SequenceRollback {
            latest: 7,
            proposed: 6,
        }
    );

    let collision = TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![clock_key(SignatureAlgorithm::Ed25519, "different-clock")],
    )
    .unwrap();
    assert_eq!(
        tracker.accept(&collision).unwrap_err(),
        TrustSnapshotTrackingError::SequenceCollision { sequence: 7 }
    );
}

#[test]
fn generic_quorum_and_continuity_are_identity_compatible() {
    let snapshot = snapshot();
    let policy = ClockQuorumPolicy::default();
    let verifier = AcceptAllSignatures;
    let observations_42 = epoch_42();

    let obs_a = observations_42
        .iter()
        .find(|observation| observation.source_id == "source-a")
        .unwrap();
    let obs_b = observations_42
        .iter()
        .find(|observation| observation.source_id == "source-b")
        .unwrap();
    assert_eq!(digest_clock_observation(obs_a).unwrap().to_hex(), OBS_A_E42);
    assert_eq!(digest_clock_observation(obs_b).unwrap().to_hex(), OBS_B_E42);

    let first = verify_clock_quorum(&observations_42, &policy, &snapshot, 1_500, &verifier)
        .unwrap();
    assert_eq!(first.lower_unix_ms, 1_499_920);
    assert_eq!(first.upper_unix_ms, 1_500_100);
    assert_eq!(first.consensus_unix_ms, 1_500_010);
    assert_eq!(first.epoch, 42);
    assert_eq!(first.source_ids, vec!["source-a", "source-b"]);
    assert_eq!(
        first.algorithms,
        vec![SignatureAlgorithm::Ed25519, SignatureAlgorithm::MlDsa65]
    );
    assert_eq!(first.trust_snapshot_digest.to_hex(), TRUST_DIGEST);
    assert_eq!(first.evidence_digest.to_hex(), WINDOW_E42);

    let mut reordered_input = observations_42.clone();
    reordered_input.reverse();
    let reordered =
        verify_clock_quorum(&reordered_input, &policy, &snapshot, 1_500, &verifier).unwrap();
    assert_eq!(reordered, first);

    let second = verify_clock_quorum(&epoch_43(), &policy, &snapshot, 1_500, &verifier).unwrap();
    assert_eq!(second.lower_unix_ms, 1_500_420);
    assert_eq!(second.upper_unix_ms, 1_500_600);
    assert_eq!(second.consensus_unix_ms, 1_500_510);
    assert_eq!(second.epoch, 43);
    assert_eq!(second.evidence_digest.to_hex(), WINDOW_E43);

    let continuity = verify_clock_continuity(&first, &second, &ClockContinuityPolicy::default())
        .unwrap();
    assert_eq!(continuity.forward_gap_ms, 320);
    assert_eq!(continuity.consensus_jump_ms, 500);
    assert_eq!(continuity.shared_sources, vec!["source-a", "source-b"]);
    assert_eq!(
        continuity.shared_algorithms,
        vec![SignatureAlgorithm::Ed25519, SignatureAlgorithm::MlDsa65]
    );
    assert_eq!(
        digest_clock_continuity(&continuity).unwrap().to_hex(),
        CONTINUITY_42_43
    );

    let mut tracker = ClockEpochTracker::default();
    tracker.accept(&first).unwrap();
    tracker.accept(&second).unwrap();
    assert_eq!(tracker.latest_epoch(), Some(43));
    assert_eq!(tracker.latest_consensus_unix_ms(), Some(1_500_510));
    assert_eq!(
        tracker.accept(&first).unwrap_err(),
        ClockTrackingError::EpochRollback {
            latest: 43,
            proposed: 42,
        }
    );

    let mut collision = second.clone();
    collision.evidence_digest = first.evidence_digest;
    assert_eq!(
        tracker.accept(&collision).unwrap_err(),
        ClockTrackingError::EpochCollision { epoch: 43 }
    );
}
