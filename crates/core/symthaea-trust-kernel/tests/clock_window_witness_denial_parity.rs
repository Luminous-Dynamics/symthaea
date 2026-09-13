// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Denial parity between the frozen V1 verifier and the additive witness verifier.

use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockObservation, ClockObservationVerifier, ClockQuorumPolicy,
    DetachedSignature, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, SignatureAlgorithm,
    TrustSnapshot, verify_clock_quorum, verify_clock_quorum_with_witness,
};

struct Verdict(bool);
impl ClockObservationVerifier for Verdict {
    fn verify_clock_observation(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        Ok(self.0)
    }
}

fn key(algorithm: SignatureAlgorithm, key_id: &str) -> KeyTrustRecord {
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
            key(SignatureAlgorithm::Ed25519, "clock-a"),
            key(SignatureAlgorithm::MlDsa65, "clock-b"),
        ],
    )
    .unwrap()
}

fn observation(
    source: &str,
    time: u64,
    uncertainty: u64,
    algorithm: SignatureAlgorithm,
    key_id: &str,
) -> ClockObservation {
    ClockObservation {
        schema_version: CLOCK_OBSERVATION_SCHEMA.to_string(),
        source_id: source.to_string(),
        observed_unix_ms: time,
        uncertainty_ms: uncertainty,
        epoch: 42,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature: vec![1],
        },
    }
}

fn fixture() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-a",
            1_500_000,
            100,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
        observation(
            "source-b",
            1_500_040,
            120,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ]
}

fn assert_same_denial(
    observations: &[ClockObservation],
    policy: &ClockQuorumPolicy,
    snapshot: &TrustSnapshot,
    evaluation_time_unix_s: u64,
    verifier: &dyn ClockObservationVerifier,
) {
    let old = verify_clock_quorum(
        observations,
        policy,
        snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .unwrap_err();
    let new = verify_clock_quorum_with_witness(
        observations,
        policy,
        snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .unwrap_err();
    assert_eq!(new, old);
}

#[test]
fn denial_sequences_match_frozen_v1() {
    let trust = snapshot();
    let accept = Verdict(true);

    let mut invalid_policy = ClockQuorumPolicy::default();
    invalid_policy.minimum_distinct_sources = 0;
    assert_same_denial(&fixture(), &invalid_policy, &trust, 1_500, &accept);

    let mut excessive_uncertainty = fixture();
    excessive_uncertainty[0].uncertainty_ms = 5_001;
    assert_same_denial(
        &excessive_uncertainty,
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &accept,
    );

    let mut duplicate_source = fixture();
    duplicate_source[1].source_id = "source-a".to_string();
    assert_same_denial(
        &duplicate_source,
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &accept,
    );

    assert_same_denial(
        &fixture(),
        &ClockQuorumPolicy::default(),
        &trust,
        2_000,
        &accept,
    );

    assert_same_denial(
        &fixture(),
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &Verdict(false),
    );
}
