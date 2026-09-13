// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockContinuityPolicyRevisionV1, ClockEvaluationPolicyV4, ClockGovernanceTimeError,
    ClockObservation, ClockObservationVerifier, ClockQuorumPolicyRevisionV1, DetachedSignature,
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot,
    accept_bootstrap_clock_basis_v5, bind_bootstrap_operational_clock_basis_v1,
    derive_bootstrap_clock_evaluation_permit_v4, derive_clock_governance_evaluation_envelope_v1,
    digest_trust_snapshot, verify_clock_bootstrap_authority, CLOCK_OBSERVATION_SCHEMA,
};

const ENVELOPE_ID: &str =
    "afbbee8cf9de3d39600c3d3a51901e3563467b13be664a681e4b5cd84c90082d";
const OPERATIONAL_ROOT_ID: &str =
    "f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).expect("canonical digest")
}

fn repeated_hex(ch: char) -> Sha256Digest {
    digest(&std::iter::repeat_n(ch, 64).collect::<String>())
}

fn usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity])
}

fn snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "clock-a".into(),
                not_before_unix_s: 900,
                not_after_unix_s: Some(3_000),
                status: KeyLifecycleStatus::Active,
                usages: usages(),
            },
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "clock-b".into(),
                not_before_unix_s: 900,
                not_after_unix_s: Some(3_000),
                status: KeyLifecycleStatus::Active,
                usages: usages(),
            },
        ],
    )
    .expect("fixture snapshot")
}

#[derive(Default)]
struct BootstrapVerifier;

impl ClockBootstrapAuthorityVerifier for BootstrapVerifier {
    fn provider_id(&self) -> &str {
        "platform-root-01"
    }

    fn authority_policy_digest(&self) -> Sha256Digest {
        repeated_hex('b')
    }

    fn verify_clock_bootstrap_authority(
        &self,
        _canonical_claim_bytes: &[u8],
        external_evidence_digest: Sha256Digest,
    ) -> Result<bool, String> {
        Ok(external_evidence_digest == repeated_hex('c'))
    }
}

struct ObservationVerifier {
    calls: Cell<usize>,
}

impl ClockObservationVerifier for ObservationVerifier {
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

fn observation(
    source_id: &str,
    observed_unix_ms: u64,
    uncertainty_ms: u64,
    algorithm: SignatureAlgorithm,
    key_id: &str,
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
            signature: vec![1, 2, 3],
        },
    }
}

fn operational_root() -> symthaea_trust_kernel::OperationalClockBasisV1 {
    let snapshot = snapshot();
    let quorum = ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true).unwrap();
    let continuity = ClockContinuityPolicyRevisionV1::new(1, 10_000, 60_000, 1, true).unwrap();
    let evaluation = ClockEvaluationPolicyV4::new(
        repeated_hex('b'),
        &quorum,
        &continuity,
        2_000,
        2,
        true,
    )
    .unwrap();
    let snapshot_digest = digest_trust_snapshot(&snapshot).unwrap();
    let claim = ClockBootstrapClaimV2::new(
        snapshot_digest,
        evaluation.id().as_digest(),
        1_499_000,
        1_499_500,
    )
    .unwrap();
    let evidence = ClockBootstrapAuthorityEvidenceV2::new(
        &claim,
        "platform-root-01",
        repeated_hex('b'),
        repeated_hex('c'),
    )
    .unwrap();
    let authority =
        verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
    let permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &continuity,
        &snapshot,
    )
    .unwrap();
    let observations = vec![
        observation(
            "source-b",
            1_500_040,
            120,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            1_500_000,
            100,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ];
    let verifier = ObservationVerifier {
        calls: Cell::new(0),
    };
    let basis = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier)
        .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    bind_bootstrap_operational_clock_basis_v1(&basis, &evaluation, &snapshot).unwrap()
}

#[test]
fn frozen_governance_envelope_matches_independent_reference() {
    let root = operational_root();
    assert_eq!(root.id().to_hex(), OPERATIONAL_ROOT_ID);
    let envelope = derive_clock_governance_evaluation_envelope_v1(&root).unwrap();
    assert_eq!(envelope.id().to_hex(), ENVELOPE_ID);
    assert_eq!(envelope.operational_basis_id(), root.id());
    assert_eq!(envelope.epoch(), 42);
    assert_eq!(envelope.lower_unix_ms(), 1_499_920);
    assert_eq!(envelope.upper_unix_ms(), 1_500_100);
    assert_eq!(envelope.consensus_unix_ms(), 1_500_010);
}

#[test]
fn validity_must_cover_the_entire_trusted_interval() {
    let envelope = derive_clock_governance_evaluation_envelope_v1(&operational_root()).unwrap();

    envelope
        .require_valid_across_seconds_window(1_499, 1_501)
        .unwrap();
    assert_eq!(
        envelope.require_valid_across_seconds_window(1_499, 1_500),
        Err(ClockGovernanceTimeError::NotValidAcrossEnvelope)
    );
    assert_eq!(
        envelope.require_valid_across_seconds_window(1_500, 3_000),
        Err(ClockGovernanceTimeError::NotValidAcrossEnvelope)
    );
    assert_eq!(
        envelope.require_valid_across_seconds_window(1_500, 1_500),
        Err(ClockGovernanceTimeError::InvalidValidityWindow)
    );
}

#[test]
fn activation_uses_upper_for_not_past_and_lower_for_maximum_delay() {
    let envelope = derive_clock_governance_evaluation_envelope_v1(&operational_root()).unwrap();

    envelope
        .require_activation_within_delay_seconds(1_501, 10)
        .unwrap();
    assert_eq!(
        envelope.require_activation_within_delay_seconds(1_500, 10),
        Err(ClockGovernanceTimeError::ActivationMayBeInPast)
    );
    assert_eq!(
        envelope.require_activation_within_delay_seconds(1_510, 10),
        Err(ClockGovernanceTimeError::ActivationMayBeTooLate)
    );
}

#[test]
fn time_scaling_fails_closed_on_overflow() {
    let envelope = derive_clock_governance_evaluation_envelope_v1(&operational_root()).unwrap();
    assert_eq!(
        envelope.require_activation_within_delay_seconds(u64::MAX, 1),
        Err(ClockGovernanceTimeError::TimeScaleOverflow)
    );
    assert_eq!(
        envelope.require_activation_within_delay_seconds(1_501, u64::MAX),
        Err(ClockGovernanceTimeError::TimeScaleOverflow)
    );
}

#[test]
fn governance_time_surface_has_no_caller_current_time_or_foreign_authority_inputs() {
    let source = include_str!("../src/clock_governance_time.rs");
    let start = source
        .find("pub fn derive_clock_governance_evaluation_envelope_v1")
        .expect("derive function");
    let rest = &source[start..];
    let end = rest.find(") -> Result").expect("derive signature") + 1;
    let signature = &rest[..end];

    assert!(signature.contains("OperationalClockBasisV1"));
    assert!(!signature.contains("unix_s"));
    assert!(!signature.contains("unix_ms"));
    assert!(!source.contains("active: bool"));
    assert!(!source.contains("usage_allowed: bool"));
    assert!(!source.contains("InactiveAuthority"));
    assert!(!source.contains("UsageNotAllowed"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernanceEvaluationEnvelopeV1"
    ));
}
