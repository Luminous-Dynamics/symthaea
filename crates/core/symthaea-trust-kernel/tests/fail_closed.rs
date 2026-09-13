// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockContinuityError, ClockContinuityPolicy, ClockObservation,
    ClockObservationVerifier, ClockQuorumPolicy, ClockViolation, DetachedSignature,
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, SignatureAlgorithm, TrustSnapshot,
    verify_clock_continuity, verify_clock_quorum,
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

fn key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    usages: BTreeSet<KeyUsage>,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.to_string(),
        not_before_unix_s: 900,
        not_after_unix_s: Some(3_000),
        status: KeyLifecycleStatus::Active,
        usages,
    }
}

fn snapshot(usages: BTreeSet<KeyUsage>) -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            key(SignatureAlgorithm::Ed25519, "clock-a", usages.clone()),
            key(SignatureAlgorithm::MlDsa65, "clock-b", usages),
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
            signature: vec![1],
        },
    }
}

fn observations(epoch: u64, offset_ms: u64) -> Vec<ClockObservation> {
    vec![
        observation(
            "source-a",
            1_500_000 + offset_ms,
            100,
            epoch,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
        observation(
            "source-b",
            1_500_040 + offset_ms,
            120,
            epoch,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ]
}

fn clock_usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity])
}

#[test]
fn wrong_key_usage_cannot_become_clock_authority() {
    let trust = snapshot(BTreeSet::from([KeyUsage::ClockContinuity]));
    let errors = verify_clock_quorum(
        &observations(42, 0),
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &Verdict(true),
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| matches!(error, ClockViolation::SignerIneligible(_))));
}

#[test]
fn stale_snapshot_cannot_authorize_clock_window() {
    let trust = snapshot(clock_usages());
    let errors = verify_clock_quorum(
        &observations(42, 0),
        &ClockQuorumPolicy::default(),
        &trust,
        2_000,
        &Verdict(true),
    )
    .unwrap_err();
    assert!(errors.contains(&ClockViolation::SnapshotStale));
}

#[test]
fn invalid_signature_provider_result_fails_closed() {
    let trust = snapshot(clock_usages());
    let errors = verify_clock_quorum(
        &observations(42, 0),
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &Verdict(false),
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| matches!(error, ClockViolation::SignatureInvalid(_))));
}

#[test]
fn uncertainty_above_policy_cannot_enter_quorum() {
    let trust = snapshot(clock_usages());
    let mut samples = observations(42, 0);
    samples[0].uncertainty_ms = 5_001;
    let errors = verify_clock_quorum(
        &samples,
        &ClockQuorumPolicy::default(),
        &trust,
        1_500,
        &Verdict(true),
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| matches!(error, ClockViolation::UncertaintyTooLarge(_))));
}

#[test]
fn continuity_rejects_consensus_time_regression() {
    let trust = snapshot(clock_usages());
    let policy = ClockQuorumPolicy::default();
    let first = verify_clock_quorum(&observations(42, 500), &policy, &trust, 1_500, &Verdict(true))
        .unwrap();
    let second = verify_clock_quorum(&observations(43, 0), &policy, &trust, 1_500, &Verdict(true))
        .unwrap();

    assert_eq!(
        verify_clock_continuity(&first, &second, &ClockContinuityPolicy::default()).unwrap_err(),
        ClockContinuityError::TimeRegression
    );
}
