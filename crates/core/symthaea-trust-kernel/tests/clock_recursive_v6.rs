// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockContinuityPolicyRevisionV1, ClockEvaluationPolicyV4, ClockObservation,
    ClockObservationVerifier, ClockQuorumPolicyRevisionV1, ClockSuccessorError,
    ContinuousClockError, DetachedSignature, KeyLifecycleStatus, KeyTrustRecord, KeyUsage,
    Sha256Digest, SignatureAlgorithm, TrustSnapshot, accept_bootstrap_clock_basis_v5,
    accept_successor_clock_basis_v6, advance_continuous_clock_basis_v1,
    bind_clock_successor_authority_context_v1, bind_continuous_clock_basis_v1,
    derive_bootstrap_clock_evaluation_permit_v4, derive_clock_successor_evaluation_permit_v1,
    derive_continuous_clock_successor_permit_v1, digest_trust_snapshot,
    verify_clock_bootstrap_authority, CLOCK_OBSERVATION_SCHEMA,
};

const BASIS_43: &str = "6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e";
const PERMIT_44: &str = "d154b7bc136371d3d0bd24389dcc139769a20fa36dcf34dd4d0b2162658345e7";
const WINDOW_44: &str = "2304236bcb292476137284a25e943fe54bd0addac1e48ab917ba353bd25d1ffa";
const WITNESS_44: &str = "0bd33fe50684886e8cd5fea899ac73c57add275855778bde69f5e396e0f3db89";
const CONTINUITY_44: &str = "194ce5eb8c491f02fe2d596801100abf3bd587513d1b6b569c692ae6e2cbfbff";
const BASIS_44: &str = "e1d9035a7b938521e4339836fd48f74d54ce685795b8acdefe42e18b02c86825";
const PERMIT_45: &str = "6b8bf16525b315c9d20c0b5cad0a5c3f61c241cddc60ecc1fde7b5708d090b91";
const WINDOW_45: &str = "2322df198ef985211b8c2af47e317b729926786be770b85c4777213e5dc4441e";
const WITNESS_45: &str = "8d4b6e26fd9f7a5f659938a6ab4fc617a32b9e45913d170c1e91356fd5837da0";
const CONTINUITY_45: &str = "e2fa905fb5b1791bdd3dbbd240e468ab9352c6bdebc9f5977158110aebd0e4ec";
const BASIS_45: &str = "825721c953d67cf60e7882a6c0547e00166906f330ed012f8ef4a5a8372a1686";

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
    .unwrap()
}

#[derive(Default)]
struct BootstrapVerifier;

impl ClockBootstrapAuthorityVerifier for BootstrapVerifier {
    fn provider_id(&self) -> &str { "platform-root-01" }
    fn authority_policy_digest(&self) -> Sha256Digest { repeated_hex('b') }
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

impl ObservationVerifier {
    fn accepting() -> Self { Self { calls: Cell::new(0) } }
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

fn epoch_observations(epoch: u64, source_a_ms: u64) -> Vec<ClockObservation> {
    vec![
        observation(
            "source-b",
            source_a_ms + 40,
            120,
            epoch,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            source_a_ms,
            100,
            epoch,
            SignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn accepted_v6_epoch_43(
    snapshot: &TrustSnapshot,
) -> (
    symthaea_trust_kernel::AcceptedClockBasisV6,
    ClockEvaluationPolicyV4,
    ClockQuorumPolicyRevisionV1,
    ClockContinuityPolicyRevisionV1,
) {
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let claim = ClockBootstrapClaimV2::new(
        digest_trust_snapshot(snapshot).unwrap(),
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
    let bootstrap_permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &continuity,
        snapshot,
    )
    .unwrap();
    let bootstrap_verifier = ObservationVerifier::accepting();
    let basis_v5 = accept_bootstrap_clock_basis_v5(
        &bootstrap_permit,
        &epoch_observations(42, 1_500_000),
        snapshot,
        &bootstrap_verifier,
    )
    .unwrap();
    assert_eq!(bootstrap_verifier.calls.get(), 2);

    let context = bind_clock_successor_authority_context_v1(&basis_v5, &evaluation, snapshot).unwrap();
    let permit = derive_clock_successor_evaluation_permit_v1(&context).unwrap();
    let verifier = ObservationVerifier::accepting();
    let basis_v6 = accept_successor_clock_basis_v6(
        &permit,
        &epoch_observations(43, 1_500_500),
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(basis_v6.id().to_hex(), BASIS_43);
    (basis_v6, evaluation, quorum, continuity)
}

#[test]
fn accepted_clock_basis_v6_recurses_without_v7_protocol() {
    let snapshot = snapshot(2_000);
    let (basis_43, evaluation, quorum, continuity) = accepted_v6_epoch_43(&snapshot);

    let continuous_43 = bind_continuous_clock_basis_v1(
        &basis_43,
        &evaluation,
        &quorum,
        &continuity,
        &snapshot,
    )
    .unwrap();
    assert_eq!(continuous_43.wire_v6_digest().to_hex(), BASIS_43);

    let permit_44 = derive_continuous_clock_successor_permit_v1(&continuous_43).unwrap();
    assert_eq!(permit_44.id().to_hex(), PERMIT_44);
    let verifier_44 = ObservationVerifier::accepting();
    let basis_44 = advance_continuous_clock_basis_v1(
        &permit_44,
        &epoch_observations(44, 1_501_000),
        &verifier_44,
    )
    .unwrap();
    assert_eq!(verifier_44.calls.get(), 2);
    assert_eq!(basis_44.wire_v6_digest().to_hex(), BASIS_44);
    assert_eq!(basis_44.verified_window().evidence_digest.to_hex(), WINDOW_44);
    assert_eq!(basis_44.evaluation_witness().witness_digest.to_hex(), WITNESS_44);
    assert_eq!(basis_44.verified_continuity().continuity_digest.to_hex(), CONTINUITY_44);

    let permit_45 = derive_continuous_clock_successor_permit_v1(&basis_44).unwrap();
    assert_eq!(permit_45.id().to_hex(), PERMIT_45);
    let verifier_45 = ObservationVerifier::accepting();
    let basis_45 = advance_continuous_clock_basis_v1(
        &permit_45,
        &epoch_observations(45, 1_501_500),
        &verifier_45,
    )
    .unwrap();
    assert_eq!(verifier_45.calls.get(), 2);
    assert_eq!(basis_45.wire_v6_digest().to_hex(), BASIS_45);
    assert_eq!(basis_45.verified_window().evidence_digest.to_hex(), WINDOW_45);
    assert_eq!(basis_45.evaluation_witness().witness_digest.to_hex(), WITNESS_45);
    assert_eq!(basis_45.verified_continuity().continuity_digest.to_hex(), CONTINUITY_45);
    assert_eq!(basis_45.epoch(), 45);

    assert_eq!(basis_45.clock_evaluation_policy_id(), evaluation.id());
    assert_eq!(basis_45.clock_quorum_policy_id(), quorum.id());
    assert_eq!(basis_45.clock_continuity_policy_id(), continuity.id());
}

#[test]
fn initial_runtime_binding_cannot_substitute_policy_or_snapshot() {
    let snapshot = snapshot(2_000);
    let (basis_43, evaluation, quorum, continuity) = accepted_v6_epoch_43(&snapshot);

    let weaker_continuity = ClockContinuityPolicyRevisionV1::new(1, 20_000, 60_000, 1, true).unwrap();
    let weaker_evaluation = evaluation_policy(&quorum, &weaker_continuity);
    let error = bind_continuous_clock_basis_v1(
        &basis_43,
        &weaker_evaluation,
        &quorum,
        &weaker_continuity,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ContinuousClockError::EvaluationPolicyMismatch);

    let mut substituted = snapshot.clone();
    substituted.sequence = 8;
    let error = bind_continuous_clock_basis_v1(
        &basis_43,
        &evaluation,
        &quorum,
        &continuity,
        &substituted,
    )
    .unwrap_err();
    assert_eq!(error, ContinuousClockError::TrustSnapshotMismatch);
}

#[test]
fn retained_snapshot_must_remain_valid_for_each_new_envelope() {
    let snapshot = snapshot(1_503);
    let (basis_43, evaluation, quorum, continuity) = accepted_v6_epoch_43(&snapshot);
    let continuous_43 = bind_continuous_clock_basis_v1(
        &basis_43,
        &evaluation,
        &quorum,
        &continuity,
        &snapshot,
    )
    .unwrap();

    let permit_44 = derive_continuous_clock_successor_permit_v1(&continuous_43).unwrap();
    let basis_44 = advance_continuous_clock_basis_v1(
        &permit_44,
        &epoch_observations(44, 1_501_000),
        &ObservationVerifier::accepting(),
    )
    .unwrap();

    let error = derive_continuous_clock_successor_permit_v1(&basis_44).unwrap_err();
    assert_eq!(error, ContinuousClockError::SnapshotNotValidForEnvelope);
}

#[test]
fn recursive_normal_path_has_no_policy_snapshot_or_version_bump_argument() {
    let source = include_str!("../src/clock_continuous.rs");

    let derive_start = source
        .find("pub fn derive_continuous_clock_successor_permit_v1")
        .expect("derive function");
    let derive_rest = &source[derive_start..];
    let derive_end = derive_rest.find(") -> Result").expect("derive signature") + 1;
    let derive_signature = &derive_rest[..derive_end];
    assert!(!derive_signature.contains("ClockEvaluationPolicy"));
    assert!(!derive_signature.contains("ClockQuorumPolicy"));
    assert!(!derive_signature.contains("ClockContinuityPolicy"));
    assert!(!derive_signature.contains("TrustSnapshot"));
    assert!(!derive_signature.contains("ClockObservation"));

    let advance_start = source
        .find("pub fn advance_continuous_clock_basis_v1")
        .expect("advance function");
    let advance_rest = &source[advance_start..];
    let advance_end = advance_rest.find(") -> Result").expect("advance signature") + 1;
    let advance_signature = &advance_rest[..advance_end];
    assert!(!advance_signature.contains("ClockEvaluationPolicy"));
    assert!(!advance_signature.contains("ClockQuorumPolicy"));
    assert!(!advance_signature.contains("ClockContinuityPolicy"));
    assert!(!advance_signature.contains("TrustSnapshot"));
    assert!(!advance_signature.contains("evaluation_time_unix_s"));
    assert!(!advance_signature.contains("VerifiedClockWindow"));
    assert!(!advance_signature.contains("ClockWindowEvaluationWitnessV1"));

    assert!(!source.contains("accepted-clock-basis.v7"));
    assert!(!source.contains("AcceptedClockBasisV7"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ContinuousClockBasisV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ContinuousClockSuccessorPermitV1"));
}

#[test]
fn legacy_successor_error_type_remains_a_separate_boundary() {
    let _ = std::mem::size_of::<ClockSuccessorError>();
}
