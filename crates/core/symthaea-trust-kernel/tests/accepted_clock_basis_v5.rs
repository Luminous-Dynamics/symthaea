// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    accept_bootstrap_clock_basis_v5, derive_bootstrap_clock_evaluation_permit_v4,
    digest_trust_snapshot, verify_clock_bootstrap_authority, AcceptedClockBasisError,
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockContinuityPolicyRevisionV1, ClockEvaluationPolicyV4, ClockObservation,
    ClockObservationVerifier, ClockQuorumPolicyRevisionV1, ClockViolation, DetachedSignature,
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot,
    CLOCK_OBSERVATION_SCHEMA,
};

const QUORUM_ID: &str = "4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304";
const CONTINUITY_ID: &str = "904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06";
const EVAL_ID: &str = "4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234";
const PERMIT_ID: &str = "1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f";
const WINDOW_ID: &str = "5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229";
const WITNESS_ID: &str = "e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2";
const BASIS_ID: &str = "241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).expect("canonical digest")
}

fn repeated_hex(ch: char) -> Sha256Digest {
    digest(&std::iter::repeat_n(ch, 64).collect::<String>())
}

fn usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity])
}

fn standard_snapshot() -> TrustSnapshot {
    snapshot_with_late_key(false)
}

fn snapshot_with_late_key(include_late: bool) -> TrustSnapshot {
    let mut keys = vec![
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
    ];
    if include_late {
        keys.push(KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "clock-c".into(),
            not_before_unix_s: 1_500,
            not_after_unix_s: Some(3_000),
            status: KeyLifecycleStatus::Active,
            usages: usages(),
        });
    }
    TrustSnapshot::new(7, 1_000, 2_000, keys).expect("trust snapshot")
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
    accept: bool,
}

impl ObservationVerifier {
    fn accepting() -> Self {
        Self { calls: Cell::new(0), accept: true }
    }

    fn rejecting() -> Self {
        Self { calls: Cell::new(0), accept: false }
    }
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
        Ok(self.accept)
    }
}

fn quorum_policy() -> ClockQuorumPolicyRevisionV1 {
    ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true).unwrap()
}

fn continuity_policy() -> ClockContinuityPolicyRevisionV1 {
    ClockContinuityPolicyRevisionV1::new(1, 10_000, 60_000, 1, true).unwrap()
}

fn evaluation_policy(
    quorum: &ClockQuorumPolicyRevisionV1,
    continuity: &ClockContinuityPolicyRevisionV1,
) -> ClockEvaluationPolicyV4 {
    ClockEvaluationPolicyV4::new(repeated_hex('b'), quorum, continuity, 2_000, 2, true).unwrap()
}

fn permit_for(snapshot: &TrustSnapshot) -> symthaea_trust_kernel::ClockEvaluationPermitV4 {
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let snapshot_digest = digest_trust_snapshot(snapshot).unwrap();
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
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
    derive_bootstrap_clock_evaluation_permit_v4(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &continuity,
        snapshot,
    )
    .unwrap()
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

fn standard_observations() -> Vec<ClockObservation> {
    vec![
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
    ]
}

#[test]
fn frozen_v5_basis_matches_independent_reference_and_retains_authority() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    assert_eq!(permit.clock_quorum_policy_id().to_hex(), QUORUM_ID);
    assert_eq!(permit.clock_continuity_policy_id().to_hex(), CONTINUITY_ID);
    assert_eq!(permit.evaluation_policy_id().to_hex(), EVAL_ID);
    assert_eq!(permit.id().to_hex(), PERMIT_ID);

    let verifier = ObservationVerifier::accepting();
    let basis = accept_bootstrap_clock_basis_v5(
        &permit,
        &standard_observations(),
        &snapshot,
        &verifier,
    )
    .unwrap();

    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(basis.id().to_hex(), BASIS_ID);
    assert_eq!(basis.permit_id().to_hex(), PERMIT_ID);
    assert_eq!(basis.clock_evaluation_policy_id().to_hex(), EVAL_ID);
    assert_eq!(basis.clock_quorum_policy_id().to_hex(), QUORUM_ID);
    assert_eq!(basis.clock_continuity_policy_id().to_hex(), CONTINUITY_ID);
    assert_eq!(basis.clock_window_evidence_digest().to_hex(), WINDOW_ID);
    assert_eq!(basis.clock_window_witness_digest().to_hex(), WITNESS_ID);
    assert_eq!(basis.epoch(), 42);
    assert_eq!(basis.lower_unix_ms(), 1_499_920);
    assert_eq!(basis.upper_unix_ms(), 1_500_100);
    assert_eq!(basis.consensus_unix_ms(), 1_500_010);

    assert_eq!(basis.originating_permit().id(), permit.id());
    assert_eq!(
        basis.originating_permit().clock_continuity_policy_id(),
        permit.clock_continuity_policy_id()
    );
    assert_eq!(
        basis.originating_permit().clock_quorum_policy_id(),
        permit.clock_quorum_policy_id()
    );
}

#[test]
fn candidate_threshold_comes_from_bound_quorum_policy() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    let one = vec![observation(
        "source-a",
        1_500_000,
        100,
        SignatureAlgorithm::Ed25519,
        "clock-a",
    )];
    let verifier = ObservationVerifier::accepting();
    let error = accept_bootstrap_clock_basis_v5(&permit, &one, &snapshot, &verifier).unwrap_err();
    match error {
        AcceptedClockBasisError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                ClockViolation::InsufficientSources { actual: 1, required: 2 }
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn quorum_uncertainty_limit_cannot_be_weakened_at_acceptance_time() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    let observations = vec![
        observation(
            "source-a",
            1_500_000,
            6_000,
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
    ];
    let verifier = ObservationVerifier::accepting();
    let error = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier)
        .unwrap_err();
    match error {
        AcceptedClockBasisError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                ClockViolation::UncertaintyTooLarge(source) if source == "source-a"
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn key_valid_at_legacy_instant_but_not_whole_envelope_is_rejected_by_permit() {
    let snapshot = snapshot_with_late_key(true);
    let permit = permit_for(&snapshot);
    assert!(permit
        .eligible_clock_keys()
        .iter()
        .all(|key| key.key_id() != "clock-c"));

    let observations = vec![
        observation(
            "source-c",
            1_500_000,
            100,
            SignatureAlgorithm::Ed25519,
            "clock-c",
        ),
        observation(
            "source-b",
            1_500_040,
            120,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ];
    let verifier = ObservationVerifier::accepting();
    let error = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier)
        .unwrap_err();
    assert_eq!(
        error,
        AcceptedClockBasisError::UnpermittedSigner("clock-c".to_string())
    );
    assert_eq!(verifier.calls.get(), 2);
}

#[test]
fn individual_observation_interval_must_fit_permit_even_if_window_intersection_does() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    let observations = vec![
        observation(
            "source-a",
            1_500_000,
            1_600,
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
    ];
    let verifier = ObservationVerifier::accepting();
    let error = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier)
        .unwrap_err();
    assert_eq!(
        error,
        AcceptedClockBasisError::ObservationIntervalOutsidePermit("source-a".to_string())
    );
}

#[test]
fn trust_snapshot_substitution_fails_before_signature_verification() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    let mut substituted = standard_snapshot();
    substituted.sequence = 8;
    let verifier = ObservationVerifier::accepting();

    let error = accept_bootstrap_clock_basis_v5(
        &permit,
        &standard_observations(),
        &substituted,
        &verifier,
    )
    .unwrap_err();
    assert_eq!(error, AcceptedClockBasisError::TrustSnapshotMismatch);
    assert_eq!(verifier.calls.get(), 0);
}

#[test]
fn signature_rejection_cannot_mint_basis() {
    let snapshot = standard_snapshot();
    let permit = permit_for(&snapshot);
    let verifier = ObservationVerifier::rejecting();
    let error = accept_bootstrap_clock_basis_v5(
        &permit,
        &standard_observations(),
        &snapshot,
        &verifier,
    )
    .unwrap_err();
    match error {
        AcceptedClockBasisError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                ClockViolation::SignatureInvalid(_)
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn acceptance_surface_has_no_caller_policy_time_window_or_witness_parameter() {
    let source = include_str!("../src/accepted_clock_basis.rs");
    let start = source
        .find("pub fn accept_bootstrap_clock_basis_v5")
        .expect("public acceptance function");
    let rest = &source[start..];
    let signature_end = rest.find(") -> Result").expect("function signature end") + 1;
    let signature = &rest[..signature_end];

    assert!(!signature.contains("ClockQuorumPolicy"));
    assert!(!signature.contains("ClockContinuityPolicy"));
    assert!(!signature.contains("evaluation_time_unix_s"));
    assert!(!signature.contains("VerifiedClockWindow"));
    assert!(!signature.contains("ClockWindowEvaluationWitnessV1"));
    assert!(source.contains("permit.runtime_quorum_policy()"));
    assert!(source.contains("permit.evaluation_upper_unix_ms() / 1_000"));
    assert!(source.contains("originating_permit: permit.clone()"));
    assert!(source.contains("pub struct AcceptedClockBasisV5"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct AcceptedClockBasisV5"));
    assert!(!source.contains("AcceptedClockBasisV4"));
}
