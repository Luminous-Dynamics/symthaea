// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockContinuityError, ClockContinuityPolicyRevisionV1, ClockEvaluationPolicyV4,
    ClockObservation, ClockObservationVerifier, ClockQuorumPolicyRevisionV1,
    DetachedSignature, KeyLifecycleStatus, KeyTrustRecord, KeyUsage,
    OperationalClockBasisKindV1, OperationalClockError, Sha256Digest, SignatureAlgorithm,
    TrustSnapshot, accept_bootstrap_clock_basis_v5, accept_operational_clock_successor_v1,
    bind_bootstrap_operational_clock_basis_v1, derive_bootstrap_clock_evaluation_permit_v4,
    derive_operational_clock_successor_permit_v2, digest_trust_snapshot,
    verify_clock_bootstrap_authority, CLOCK_OBSERVATION_SCHEMA,
};

const TRUST_ID: &str =
    "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const QUORUM_ID: &str =
    "4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304";
const CONTINUITY_POLICY_ID: &str =
    "904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06";
const EVALUATION_ID: &str =
    "4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234";
const BOOTSTRAP_V5_ID: &str =
    "241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8";
const ROOT42_ID: &str =
    "f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b";
const PERMIT43_ID: &str =
    "750c044e48a5395ee04a457ba7b89cb2397c2bf2cea76e96e71e50dcd338b73b";
const WINDOW43_ID: &str =
    "d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b";
const WITNESS43_ID: &str =
    "6d330cbf15d9dde7bf11aada0fb5b53feed204573308c6bcafd5569f07b4e3a3";
const CONTINUITY43_ID: &str =
    "ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15";
const ROOT43_ID: &str =
    "0e36e238e300ae1d626846220bbba84f5e87a50be7ffa2ac3b8e7822f4d47810";
const PERMIT44_ID: &str =
    "68b6dd92d42f3b42093f33d310d69eb8ac1bb259b02bd698256f67ea872b8a3d";
const WINDOW44_ID: &str =
    "2304236bcb292476137284a25e943fe54bd0addac1e48ab917ba353bd25d1ffa";
const WITNESS44_ID: &str =
    "0bd33fe50684886e8cd5fea899ac73c57add275855778bde69f5e396e0f3db89";
const CONTINUITY44_ID: &str =
    "194ce5eb8c491f02fe2d596801100abf3bd587513d1b6b569c692ae6e2cbfbff";
const ROOT44_ID: &str =
    "9373fac49b1b488859de56cd6cd588b8e5868cf9fb455c8b3546b46974d8ffce";

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
        Self {
            calls: Cell::new(0),
            accept: true,
        }
    }

    fn rejecting() -> Self {
        Self {
            calls: Cell::new(0),
            accept: false,
        }
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
    ClockEvaluationPolicyV4::new(
        repeated_hex('b'),
        quorum,
        continuity,
        2_000,
        2,
        true,
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
            signature: vec![1, 2, 3],
        },
    }
}

fn observations(epoch: u64, base_ms: u64) -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            base_ms + 40,
            120,
            epoch,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            base_ms,
            100,
            epoch,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn bootstrap_v5(
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
    let authority =
        verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
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
        &observations(42, 1_500_000),
        snapshot,
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    (basis, evaluation)
}

#[test]
fn recursive_operational_basis_matches_independent_reference_for_two_successors() {
    let snapshot = snapshot(2_000);
    assert_eq!(digest_trust_snapshot(&snapshot).unwrap().to_hex(), TRUST_ID);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    assert_eq!(v5.id().to_hex(), BOOTSTRAP_V5_ID);
    assert_eq!(v5.clock_quorum_policy_id().to_hex(), QUORUM_ID);
    assert_eq!(v5.clock_continuity_policy_id().to_hex(), CONTINUITY_POLICY_ID);
    assert_eq!(v5.clock_evaluation_policy_id().to_hex(), EVALUATION_ID);

    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    assert_eq!(root42.kind(), OperationalClockBasisKindV1::Bootstrap);
    assert_eq!(root42.id().to_hex(), ROOT42_ID);

    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();
    assert_eq!(permit43.id().to_hex(), PERMIT43_ID);
    assert_eq!(permit43.prior_operational_basis_id(), root42.id());

    let verifier43 = ObservationVerifier::accepting();
    let root43 = accept_operational_clock_successor_v1(
        &permit43,
        &observations(43, 1_500_500),
        &verifier43,
    )
    .unwrap();
    assert_eq!(verifier43.calls.get(), 2);
    assert_eq!(root43.kind(), OperationalClockBasisKindV1::Continuous);
    assert_eq!(root43.id().to_hex(), ROOT43_ID);
    assert_eq!(root43.predecessor_operational_basis_id(), Some(root42.id()));
    assert_eq!(root43.clock_window_evidence_digest().to_hex(), WINDOW43_ID);
    assert_eq!(root43.clock_window_witness_digest().to_hex(), WITNESS43_ID);
    assert_eq!(
        root43.verified_continuity().unwrap().continuity_digest.to_hex(),
        CONTINUITY43_ID
    );

    let permit44 = derive_operational_clock_successor_permit_v2(&root43).unwrap();
    assert_eq!(permit44.id().to_hex(), PERMIT44_ID);
    assert_eq!(permit44.prior_operational_basis_id(), root43.id());

    let verifier44 = ObservationVerifier::accepting();
    let root44 = accept_operational_clock_successor_v1(
        &permit44,
        &observations(44, 1_501_000),
        &verifier44,
    )
    .unwrap();
    assert_eq!(verifier44.calls.get(), 2);
    assert_eq!(root44.kind(), OperationalClockBasisKindV1::Continuous);
    assert_eq!(root44.id().to_hex(), ROOT44_ID);
    assert_eq!(root44.predecessor_operational_basis_id(), Some(root43.id()));
    assert_eq!(root44.clock_window_evidence_digest().to_hex(), WINDOW44_ID);
    assert_eq!(root44.clock_window_witness_digest().to_hex(), WITNESS44_ID);
    assert_eq!(
        root44.verified_continuity().unwrap().continuity_digest.to_hex(),
        CONTINUITY44_ID
    );
    assert_ne!(root43.id(), root44.id());
}

#[test]
fn bootstrap_witness_records_cannot_substitute_authority() {
    let snapshot = snapshot(2_000);
    let (v5, evaluation) = bootstrap_v5(&snapshot);

    let weaker_continuity =
        ClockContinuityPolicyRevisionV1::new(1, 20_000, 60_000, 1, true).unwrap();
    let weaker_evaluation = evaluation_policy(&quorum_policy(), &weaker_continuity);
    assert_eq!(
        bind_bootstrap_operational_clock_basis_v1(
            &v5,
            &weaker_evaluation,
            &snapshot,
        )
        .unwrap_err(),
        OperationalClockError::EvaluationPolicyMismatch
    );

    let mut substituted = snapshot.clone();
    substituted.sequence = 8;
    assert_eq!(
        bind_bootstrap_operational_clock_basis_v1(
            &v5,
            &evaluation,
            &substituted,
        )
        .unwrap_err(),
        OperationalClockError::TrustSnapshotMismatch
    );
}

#[test]
fn retained_quorum_policy_denies_one_source_on_later_generation() {
    let snapshot = snapshot(2_000);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();
    let verifier43 = ObservationVerifier::accepting();
    let root43 = accept_operational_clock_successor_v1(
        &permit43,
        &observations(43, 1_500_500),
        &verifier43,
    )
    .unwrap();
    let permit44 = derive_operational_clock_successor_permit_v2(&root43).unwrap();

    let one = vec![observation(
        "source-a",
        1_501_000,
        100,
        44,
        SignatureAlgorithm::Ed25519,
        "clock-a",
    )];
    let verifier = ObservationVerifier::accepting();
    let error =
        accept_operational_clock_successor_v1(&permit44, &one, &verifier).unwrap_err();
    match error {
        OperationalClockError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                symthaea_trust_kernel::ClockViolation::InsufficientSources {
                    actual: 1,
                    required: 2
                }
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn successor_window_must_fit_pre_candidate_recursive_permit() {
    let snapshot = snapshot(2_000);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();
    let verifier43 = ObservationVerifier::accepting();
    let root43 = accept_operational_clock_successor_v1(
        &permit43,
        &observations(43, 1_500_500),
        &verifier43,
    )
    .unwrap();
    let permit44 = derive_operational_clock_successor_permit_v2(&root43).unwrap();

    let far = observations(44, 1_511_000);
    let verifier = ObservationVerifier::accepting();
    assert_eq!(
        accept_operational_clock_successor_v1(&permit44, &far, &verifier).unwrap_err(),
        OperationalClockError::WindowOutsidePermit
    );
}

#[test]
fn epoch_jump_is_denied_by_retained_continuity_policy() {
    let snapshot = snapshot(2_000);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();

    let skipped = observations(44, 1_500_500);
    let verifier = ObservationVerifier::accepting();
    assert_eq!(
        accept_operational_clock_successor_v1(&permit43, &skipped, &verifier).unwrap_err(),
        OperationalClockError::ContinuityInvalid(ClockContinuityError::EpochStepTooLarge)
    );
}

#[test]
fn snapshot_that_expires_before_next_envelope_fails_at_permit_derivation() {
    let snapshot = snapshot(1_502);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();
    let verifier43 = ObservationVerifier::accepting();
    let root43 = accept_operational_clock_successor_v1(
        &permit43,
        &observations(43, 1_500_500),
        &verifier43,
    )
    .unwrap();

    assert_eq!(
        derive_operational_clock_successor_permit_v2(&root43).unwrap_err(),
        OperationalClockError::SnapshotNotValidForEnvelope
    );
}

#[test]
fn signature_rejection_cannot_mint_operational_successor() {
    let snapshot = snapshot(2_000);
    let (v5, evaluation) = bootstrap_v5(&snapshot);
    let root42 =
        bind_bootstrap_operational_clock_basis_v1(&v5, &evaluation, &snapshot).unwrap();
    let permit43 = derive_operational_clock_successor_permit_v2(&root42).unwrap();
    let verifier = ObservationVerifier::rejecting();
    let error = accept_operational_clock_successor_v1(
        &permit43,
        &observations(43, 1_500_500),
        &verifier,
    )
    .unwrap_err();
    match error {
        OperationalClockError::QuorumVerificationFailed(violations) => {
            assert!(violations.iter().any(|violation| matches!(
                violation,
                symthaea_trust_kernel::ClockViolation::SignatureInvalid(_)
            )));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn recursive_operational_surface_has_no_caller_policy_snapshot_or_time_argument() {
    let source = include_str!("../src/clock_operational.rs");

    let derive_start = source
        .find("pub fn derive_operational_clock_successor_permit_v2")
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
        .find("pub fn accept_operational_clock_successor_v1")
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

    assert!(source.contains("pub struct OperationalClockBasisV1"));
    assert!(source.contains("pub struct ClockSuccessorEvaluationPermitV2"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct OperationalClockBasisV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ClockSuccessorEvaluationPermitV2"));
}
