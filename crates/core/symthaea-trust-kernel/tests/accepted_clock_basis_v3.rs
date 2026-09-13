// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    ACCEPTED_CLOCK_BASIS_SCHEMA, CLOCK_OBSERVATION_SCHEMA, AcceptedClockBasisError,
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockEvaluationPolicyV2, ClockObservation, ClockObservationVerifier, ClockQuorumPolicy,
    ClockViolation, DetachedSignature, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest,
    SignatureAlgorithm, TrustSnapshot, accept_bootstrap_clock_basis_v3,
    derive_bootstrap_clock_evaluation_permit_v2, digest_trust_snapshot,
    verify_clock_bootstrap_authority,
};

const TRUST: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const WINDOW: &str = "5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229";
const WITNESS: &str = "e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2";
const PERMIT: &str = "12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9";
const BASIS: &str = "1442964fea0e569d5ef59e2df593b5399eb2a7cc568417e1a4c34c796a20d72e";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).unwrap()
}

fn key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    not_before_unix_s: u64,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.to_string(),
        not_before_unix_s,
        not_after_unix_s: Some(3_000),
        status: KeyLifecycleStatus::Active,
        usages: BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity]),
    }
}

fn base_keys() -> Vec<KeyTrustRecord> {
    vec![
        key(SignatureAlgorithm::Ed25519, "clock-a", 900),
        key(SignatureAlgorithm::MlDsa65, "clock-b", 900),
    ]
}

fn snapshot(keys: Vec<KeyTrustRecord>) -> TrustSnapshot {
    TrustSnapshot::new(7, 1_000, 2_000, keys).unwrap()
}

fn policy() -> ClockEvaluationPolicyV2 {
    ClockEvaluationPolicyV2::new(digest(&"b".repeat(64)), 2_000, 2, true).unwrap()
}

struct BootstrapVerifier;
impl ClockBootstrapAuthorityVerifier for BootstrapVerifier {
    fn provider_id(&self) -> &str {
        "platform-root-01"
    }

    fn authority_policy_digest(&self) -> Sha256Digest {
        digest(&"b".repeat(64))
    }

    fn verify_clock_bootstrap_authority(
        &self,
        _canonical_claim_bytes: &[u8],
        external_evidence_digest: Sha256Digest,
    ) -> Result<bool, String> {
        assert_eq!(external_evidence_digest, digest(&"c".repeat(64)));
        Ok(true)
    }
}

fn permit_for(trust: &TrustSnapshot) -> symthaea_trust_kernel::ClockEvaluationPermitV2 {
    let policy = policy();
    let claim = ClockBootstrapClaimV2::new(
        digest_trust_snapshot(trust).unwrap(),
        policy.id().as_digest(),
        1_499_000,
        1_499_500,
    )
    .unwrap();
    let evidence = ClockBootstrapAuthorityEvidenceV2::new(
        &claim,
        "platform-root-01",
        digest(&"b".repeat(64)),
        digest(&"c".repeat(64)),
    )
    .unwrap();
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
    derive_bootstrap_clock_evaluation_permit_v2(&authority, &claim, &policy, trust).unwrap()
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
            signature: vec![1],
        },
    }
}

fn fixture_observations() -> Vec<ClockObservation> {
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

struct Verdict {
    accept: bool,
    calls: Cell<usize>,
}

impl Verdict {
    fn accepting() -> Self {
        Self {
            accept: true,
            calls: Cell::new(0),
        }
    }
}

impl ClockObservationVerifier for Verdict {
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

#[test]
fn production_basis_matches_independent_v3_reference() {
    assert_eq!(ACCEPTED_CLOCK_BASIS_SCHEMA, "symthaea.trust.accepted-clock-basis.v3");
    let trust = snapshot(base_keys());
    assert_eq!(digest_trust_snapshot(&trust).unwrap().to_hex(), TRUST);
    let permit = permit_for(&trust);
    assert_eq!(permit.id().to_hex(), PERMIT);

    let verifier = Verdict::accepting();
    let basis = accept_bootstrap_clock_basis_v3(
        &permit,
        &fixture_observations(),
        &ClockQuorumPolicy::default(),
        &trust,
        &verifier,
    )
    .unwrap();

    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(basis.id().to_hex(), BASIS);
    assert_eq!(basis.permit_id(), permit.id());
    assert_eq!(basis.trust_snapshot_digest().to_hex(), TRUST);
    assert_eq!(basis.clock_window_evidence_digest().to_hex(), WINDOW);
    assert_eq!(basis.clock_window_witness_digest().to_hex(), WITNESS);
    assert_eq!(basis.epoch(), 42);
    assert_eq!(basis.lower_unix_ms(), 1_499_920);
    assert_eq!(basis.upper_unix_ms(), 1_500_100);
    assert_eq!(basis.consensus_unix_ms(), 1_500_010);
}

#[test]
fn signature_rejection_cannot_mint_basis() {
    let trust = snapshot(base_keys());
    let permit = permit_for(&trust);
    let verifier = Verdict {
        accept: false,
        calls: Cell::new(0),
    };
    let error = accept_bootstrap_clock_basis_v3(
        &permit,
        &fixture_observations(),
        &ClockQuorumPolicy::default(),
        &trust,
        &verifier,
    )
    .unwrap_err();
    match error {
        AcceptedClockBasisError::ClockVerification(violations) => {
            assert!(violations.iter().any(|violation| {
                matches!(violation, ClockViolation::SignatureInvalid(_))
            }));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(verifier.calls.get(), 2);
}

#[test]
fn point_in_time_eligible_but_not_whole_envelope_key_is_denied() {
    let mut keys = base_keys();
    // Not valid at the permit lower bound (1499s) but valid at the permit-derived
    // verifier instant (1501s). The legacy verifier can accept it; the permit may not.
    keys.push(key(SignatureAlgorithm::Ed25519, "clock-x", 1_500));
    let trust = snapshot(keys);
    let permit = permit_for(&trust);
    assert!(permit
        .eligible_clock_keys()
        .iter()
        .all(|key| key.key_id() != "clock-x"));

    let observations = vec![
        observation(
            "source-x",
            1_500_000,
            100,
            SignatureAlgorithm::Ed25519,
            "clock-x",
        ),
        observation(
            "source-b",
            1_500_040,
            120,
            SignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
    ];
    let verifier = Verdict::accepting();
    assert_eq!(
        accept_bootstrap_clock_basis_v3(
            &permit,
            &observations,
            &ClockQuorumPolicy::default(),
            &trust,
            &verifier,
        )
        .unwrap_err(),
        AcceptedClockBasisError::UnpermittedSigner("clock-x".to_string())
    );
    assert_eq!(verifier.calls.get(), 2);
}

#[test]
fn permit_minimum_signer_count_is_independent_of_weaker_quorum_policy() {
    let trust = snapshot(base_keys());
    let permit = permit_for(&trust);
    let observations = vec![observation(
        "source-a",
        1_500_000,
        100,
        SignatureAlgorithm::Ed25519,
        "clock-a",
    )];
    let mut weak = ClockQuorumPolicy::default();
    weak.minimum_distinct_sources = 1;
    weak.require_algorithm_diversity = false;
    let verifier = Verdict::accepting();
    assert_eq!(
        accept_bootstrap_clock_basis_v3(&permit, &observations, &weak, &trust, &verifier)
            .unwrap_err(),
        AcceptedClockBasisError::InsufficientSigners {
            actual: 1,
            required: 2,
        }
    );
}

#[test]
fn individual_observation_interval_cannot_hide_outside_permit() {
    let trust = snapshot(base_keys());
    let permit = permit_for(&trust);
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
    let verifier = Verdict::accepting();
    assert_eq!(
        accept_bootstrap_clock_basis_v3(
            &permit,
            &observations,
            &ClockQuorumPolicy::default(),
            &trust,
            &verifier,
        )
        .unwrap_err(),
        AcceptedClockBasisError::ObservationIntervalOutsidePermit("source-a".to_string())
    );
}

#[test]
fn snapshot_mismatch_fails_before_signature_verification() {
    let trust = snapshot(base_keys());
    let permit = permit_for(&trust);
    let other = TrustSnapshot::new(8, 1_000, 2_000, base_keys()).unwrap();
    let verifier = Verdict::accepting();
    assert_eq!(
        accept_bootstrap_clock_basis_v3(
            &permit,
            &fixture_observations(),
            &ClockQuorumPolicy::default(),
            &other,
            &verifier,
        )
        .unwrap_err(),
        AcceptedClockBasisError::PermitSnapshotMismatch
    );
    assert_eq!(verifier.calls.get(), 0);
}

#[test]
fn public_acceptance_surface_has_no_precomputed_window_witness_or_time_parameter() {
    let source = include_str!("../src/accepted_clock_basis.rs");
    let start = source
        .find("pub fn accept_bootstrap_clock_basis_v3")
        .unwrap();
    let tail = &source[start..];
    let end = tail.find(") -> Result").unwrap();
    let signature = &tail[..end];
    assert!(!signature.contains("VerifiedClockWindow"));
    assert!(!signature.contains("ClockWindowEvaluationWitness"));
    assert!(!signature.contains("evaluation_time"));

    let basis_at = source.find("pub struct AcceptedClockBasisV3").unwrap();
    let derive_at = source[..basis_at].rfind("#[derive").unwrap();
    let basis_derive = &source[derive_at..basis_at];
    assert!(!basis_derive.contains("Serialize"));
    assert!(!basis_derive.contains("Deserialize"));
    assert!(!source.contains("pub fn unchecked"));
    assert!(!source.contains("pub fn from_digest"));
    assert!(!source.contains("pub fn from_verified"));
}
