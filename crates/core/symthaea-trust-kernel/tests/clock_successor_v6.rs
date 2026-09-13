// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    AcceptedClockBasisError, ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier,
    ClockBootstrapClaimV2, ClockContinuityError, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPolicyV4, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicyRevisionV1, ClockSuccessorError, ClockViolation, DetachedSignature,
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest, SignatureAlgorithm,
    TrustSnapshot, accept_bootstrap_clock_basis_v5, accept_successor_clock_basis_v6,
    bind_clock_successor_authority_context_v1, derive_bootstrap_clock_evaluation_permit_v4,
    derive_clock_successor_evaluation_permit_v1, digest_trust_snapshot,
    verify_clock_bootstrap_authority, CLOCK_OBSERVATION_SCHEMA,
};

const TRUST_ID: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const QUORUM_ID: &str = "4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304";
const CONTINUITY_POLICY_ID: &str = "904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06";
const EVALUATION_ID: &str = "4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234";
const BOOTSTRAP_PERMIT_ID: &str = "1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f";
const PRIOR_BASIS_ID: &str = "241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8";
const SUCCESSOR_PERMIT_ID: &str = "eec874620319419ac4ef0663997ee2a6700db268fad6367ee28b684816c5d5d8";
const SUCCESSOR_WINDOW_ID: &str = "d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b";
const SUCCESSOR_WITNESS_ID: &str = "6d330cbf15d9dde7bf11aada0fb5b53feed204573308c6bcafd5569f07b4e3a3";
const CONTINUITY_ID: &str = "ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15";
const SUCCESSOR_BASIS_ID: &str = "6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).expect("canonical digest")
}

fn repeated_hex(ch: char) -> Sha256Digest {
    digest(&std::iter::repeat_n(ch, 64).collect::<String>())
}

fn usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity])
}

fn snapshot(expires_at_unix_s: u64) -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        1_000,
        expires_at_unix_s,
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
            signature: vec![1, 2, 3],
        },
    }
}

fn epoch_42_observations() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            1_500_040,
            120,
            42,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            1_500_000,
            100,
            42,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn epoch_43_observations() -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            1_500_540,
            120,
            43,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            1_500_500,
            100,
            43,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn prior_basis(
    snapshot: &TrustSnapshot,
) -> (
    symthaea_trust_kernel::AcceptedClockBasisV5,
    ClockEvaluationPolicyV4,
) {
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
    let permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &continuity,
        snapshot,
    )
    .unwrap();
    let verifier = ObservationVerifier::accepting();
    let basis = accept_bootstrap_clock_basis_v5(
        &permit,
        &epoch_42_observations(),
        snapshot,
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    (basis, evaluation)
}

#[test]
fn frozen_successor_v6_chain_matches_independent_reference() {
    let snapshot = snapshot(2_000);
    assert_eq!(digest_trust_snapshot(&snapshot).unwrap().to_hex(), TRUST_ID);
    let (prior, evaluation) = prior_basis(&snapshot);
    assert_eq!(prior.clock_quorum_policy_id().to_hex(), QUORUM_ID);
    assert_eq!(prior.clock_continuity_policy_id().to_hex(), CONTINUITY_POLICY_ID);
    assert_eq!(prior.clock_evaluation_policy_id().to_hex(), EVALUATION_ID);
    assert_eq!(prior.permit_id().to_hex(), BOOTSTRAP_PERMIT_ID);
    assert_eq!(prior.id().to_hex(), PRIOR_BASIS_ID);

    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    assert_eq!(permit.id().to_hex(), SUCCESSOR_PERMIT_ID);
    assert_eq!(permit.evaluation_lower_unix_ms(), 1_499_920);
    assert_eq!(permit.evaluation_upper_unix_ms(), 1_502_100);
    assert_eq!(permit.eligible_clock_keys().len(), 2);

    let verifier = ObservationVerifier::accepting();
    let successor = accept_successor_clock_basis_v6(
        &permit,
        &epoch_43_observations(),
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(successor.clock_window_evidence_digest().to_hex(), SUCCESSOR_WINDOW_ID);
    assert_eq!(successor.clock_window_witness_digest().to_hex(), SUCCESSOR_WITNESS_ID);
    assert_eq!(successor.clock_continuity_digest().to_hex(), CONTINUITY_ID);
    assert_eq!(successor.id().to_hex(), SUCCESSOR_BASIS_ID);
    assert_eq!(successor.prior_basis_id(), prior.id());
    assert_eq!(successor.clock_evaluation_policy_id(), prior.clock_evaluation_policy_id());
    assert_eq!(successor.clock_quorum_policy_id(), prior.clock_quorum_policy_id());
    assert_eq!(successor.clock_continuity_policy_id(), prior.clock_continuity_policy_id());
    assert_eq!(successor.epoch(), 43);
    assert_eq!(successor.lower_unix_ms(), 1_500_420);
    assert_eq!(successor.upper_unix_ms(), 1_500_600);
    assert_eq!(successor.consensus_unix_ms(), 1_500_510);
}

#[test]
fn witness_records_may_be_supplied_but_cannot_substitute_authority() {
    let snapshot = snapshot(2_000);
    let (prior, evaluation) = prior_basis(&snapshot);

    let weaker_continuity = ClockContinuityPolicyRevisionV1::new(1, 20_000, 60_000, 1, true).unwrap();
    let weaker_evaluation = evaluation_policy(&quorum_policy(), &weaker_continuity);
    let error = bind_clock_successor_authority_context_v1(
        &prior,
        &weaker_evaluation,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockSuccessorError::EvaluationPolicyMismatch);

    let mut substituted = snapshot.clone();
    substituted.sequence = 8;
    let error = bind_clock_successor_authority_context_v1(
        &prior,
        &evaluation,
        &substituted,
    )
    .unwrap_err();
    assert_eq!(error, ClockSuccessorError::TrustSnapshotMismatch);
}

#[test]
fn one_source_successor_is_denied_by_retained_quorum_policy() {
    let snapshot = snapshot(2_000);
    let (prior, evaluation) = prior_basis(&snapshot);
    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    let one = vec![observation(
        "source-a",
        1_500_500,
        100,
        43,
        SignatureAlgorithm::Ed25519,
        "clock-a",
    )];
    let verifier = ObservationVerifier::accepting();
    let error = accept_successor_clock_basis_v6(&permit, &one, &verifier).unwrap_err();
    match error {
        ClockSuccessorError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                ClockViolation::InsufficientSources { actual: 1, required: 2 }
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn successor_window_must_fit_pre_candidate_permit() {
    let snapshot = snapshot(2_000);
    let (prior, evaluation) = prior_basis(&snapshot);
    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    let far = vec![
        observation(
            "source-a",
            1_511_000,
            100,
            43,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
        observation(
            "source-b",
            1_511_040,
            120,
            43,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ];
    let verifier = ObservationVerifier::accepting();
    let error = accept_successor_clock_basis_v6(&permit, &far, &verifier).unwrap_err();
    assert_eq!(error, ClockSuccessorError::WindowOutsidePermit);
}

#[test]
fn epoch_jump_is_denied_by_retained_continuity_policy() {
    let snapshot = snapshot(2_000);
    let (prior, evaluation) = prior_basis(&snapshot);
    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    let skipped = vec![
        observation(
            "source-a",
            1_500_500,
            100,
            44,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
        observation(
            "source-b",
            1_500_540,
            120,
            44,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ];
    let verifier = ObservationVerifier::accepting();
    let error = accept_successor_clock_basis_v6(&permit, &skipped, &verifier).unwrap_err();
    assert_eq!(
        error,
        ClockSuccessorError::ContinuityInvalid(ClockContinuityError::EpochStepTooLarge)
    );
}

#[test]
fn snapshot_valid_for_bootstrap_can_expire_before_successor_envelope() {
    let snapshot = snapshot(1_502);
    let (prior, evaluation) = prior_basis(&snapshot);
    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let error = derive_clock_successor_evaluation_permit_v1(&context).unwrap_err();
    assert_eq!(error, ClockSuccessorError::SnapshotNotValidForEnvelope);
}

#[test]
fn signature_rejection_cannot_mint_successor_basis() {
    let snapshot = snapshot(2_000);
    let (prior, evaluation) = prior_basis(&snapshot);
    let context = bind_clock_successor_authority_context_v1(&prior, &evaluation, &snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    let verifier = ObservationVerifier::rejecting();
    let error = accept_successor_clock_basis_v6(
        &permit,
        &epoch_43_observations(),
        &verifier,
    )
    .unwrap_err();
    match error {
        ClockSuccessorError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                ClockViolation::SignatureInvalid(_)
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn normal_successor_api_has_no_policy_snapshot_or_candidate_time_authority() {
    let source = include_str!("../src/clock_successor.rs");

    let derive_start = source
        .find("pub fn derive_clock_successor_evaluation_permit_v1")
        .expect("derive function");
    let derive_rest = &source[derive_start..];
    let derive_end = derive_rest.find(") -> Result").expect("derive signature") + 1;
    let derive_signature = &derive_rest[..derive_end];
    assert!(!derive_signature.contains("ClockEvaluationPolicyV4"));
    assert!(!derive_signature.contains("ClockQuorumPolicyRevisionV1"));
    assert!(!derive_signature.contains("ClockContinuityPolicyRevisionV1"));
    assert!(!derive_signature.contains("TrustSnapshot"));
    assert!(!derive_signature.contains("ClockObservation"));

    let accept_start = source
        .find("pub fn accept_successor_clock_basis_v6")
        .expect("accept function");
    let accept_rest = &source[accept_start..];
    let accept_end = accept_rest.find(") -> Result").expect("accept signature") + 1;
    let accept_signature = &accept_rest[..accept_end];
    assert!(!accept_signature.contains("ClockEvaluationPolicyV4"));
    assert!(!accept_signature.contains("ClockQuorumPolicy"));
    assert!(!accept_signature.contains("ClockContinuityPolicy"));
    assert!(!accept_signature.contains("TrustSnapshot"));
    assert!(!accept_signature.contains("evaluation_time_unix_s"));
    assert!(!accept_signature.contains("VerifiedClockWindow"));
    assert!(!accept_signature.contains("ClockWindowEvaluationWitnessV1"));

    assert!(source.contains("pub struct ClockSuccessorAuthorityContextV1"));
    assert!(source.contains("pub struct ClockSuccessorEvaluationPermitV1"));
    assert!(source.contains("pub struct AcceptedClockBasisV6"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ClockSuccessorAuthorityContextV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ClockSuccessorEvaluationPermitV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct AcceptedClockBasisV6"));
}

#[test]
fn accepted_bootstrap_error_type_remains_available_as_separate_boundary() {
    // A compile-surface assertion: successor errors do not collapse bootstrap
    // acceptance into the same authority type.
    let _ = std::mem::size_of::<AcceptedClockBasisError>();
}
