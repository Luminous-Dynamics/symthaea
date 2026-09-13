// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockEvaluationPermitError, ClockEvaluationPolicyV2, KeyLifecycleStatus, KeyTrustRecord,
    KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot,
    derive_bootstrap_clock_evaluation_permit_v2, digest_trust_snapshot,
    verify_clock_bootstrap_authority,
};

const TRUST: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const POLICY: &str = "7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318";
const CLAIM: &str = "15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877";
const AUTHORITY: &str = "b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3";
const ANCHOR: &str = "fdfb5cfe737fc326792ed57dfe6aa93b1a2005fef021de6ddcb60e6a56144282";
const PERMIT: &str = "12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).unwrap()
}

fn key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    status: KeyLifecycleStatus,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.to_string(),
        not_before_unix_s: 900,
        not_after_unix_s: Some(3_000),
        status,
        usages: BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity]),
    }
}

fn snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "clock-a",
                KeyLifecycleStatus::Active,
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "clock-b",
                KeyLifecycleStatus::Active,
            ),
        ],
    )
    .unwrap()
}

fn policy(step_ms: u64, diversity: bool) -> ClockEvaluationPolicyV2 {
    ClockEvaluationPolicyV2::new(digest(&"b".repeat(64)), step_ms, 2, diversity).unwrap()
}

struct AcceptBootstrap;
impl ClockBootstrapAuthorityVerifier for AcceptBootstrap {
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

fn verified_bootstrap(
    policy: &ClockEvaluationPolicyV2,
    trust: &TrustSnapshot,
) -> (
    ClockBootstrapClaimV2,
    symthaea_trust_kernel::VerifiedClockBootstrapAuthorityV2,
) {
    let trust_digest = digest_trust_snapshot(trust).unwrap();
    let claim = ClockBootstrapClaimV2::new(
        trust_digest,
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
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &AcceptBootstrap).unwrap();
    (claim, authority)
}

#[test]
fn production_permit_matches_independent_reference_vectors() {
    let trust = snapshot();
    assert_eq!(digest_trust_snapshot(&trust).unwrap().to_hex(), TRUST);

    let policy = policy(2_000, true);
    assert_eq!(policy.id().to_hex(), POLICY);

    let (claim, authority) = verified_bootstrap(&policy, &trust);
    assert_eq!(claim.id().to_hex(), CLAIM);
    assert_eq!(authority.id().to_hex(), AUTHORITY);

    let permit =
        derive_bootstrap_clock_evaluation_permit_v2(&authority, &claim, &policy, &trust).unwrap();
    assert_eq!(permit.basis_id().to_hex(), ANCHOR);
    assert_eq!(permit.id().to_hex(), PERMIT);
    assert_eq!(permit.policy_id(), policy.id());
    assert_eq!(permit.trust_snapshot_digest().to_hex(), TRUST);
    assert_eq!(permit.evaluation_lower_unix_ms(), 1_499_000);
    assert_eq!(permit.evaluation_upper_unix_ms(), 1_501_500);
    assert_eq!(permit.minimum_clock_authority_keys(), 2);
    assert!(permit.require_algorithm_diversity());
    assert_eq!(permit.eligible_clock_keys().len(), 2);
    assert_eq!(
        permit.eligible_clock_keys()[0].algorithm(),
        &SignatureAlgorithm::Ed25519
    );
    assert_eq!(permit.eligible_clock_keys()[0].key_id(), "clock-a");
    assert_eq!(
        permit.eligible_clock_keys()[1].algorithm(),
        &SignatureAlgorithm::MlDsa65
    );
    assert_eq!(permit.eligible_clock_keys()[1].key_id(), "clock-b");
}

#[test]
fn policy_or_snapshot_substitution_fails_closed() {
    let trust = snapshot();
    let original_policy = policy(2_000, true);
    let (claim, authority) = verified_bootstrap(&original_policy, &trust);

    let other_policy = policy(1_999, true);
    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(
            &authority,
            &claim,
            &other_policy,
            &trust,
        )
        .unwrap_err(),
        ClockEvaluationPermitError::BootstrapPolicyMismatch
    );

    let other_snapshot = TrustSnapshot::new(8, 1_000, 2_000, trust.keys.clone()).unwrap();
    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(
            &authority,
            &claim,
            &original_policy,
            &other_snapshot,
        )
        .unwrap_err(),
        ClockEvaluationPermitError::BootstrapSnapshotMismatch
    );
}

#[test]
fn whole_envelope_snapshot_expiry_is_denied() {
    let trust = TrustSnapshot::new(
        7,
        1_000,
        1_501,
        snapshot().keys,
    )
    .unwrap();
    let policy = policy(2_000, true);
    let (claim, authority) = verified_bootstrap(&policy, &trust);

    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(&authority, &claim, &policy, &trust)
            .unwrap_err(),
        ClockEvaluationPermitError::SnapshotNotValidForEnvelope
    );
}

#[test]
fn whole_envelope_key_count_and_diversity_are_enforced() {
    let one_active = TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "clock-a",
                KeyLifecycleStatus::Active,
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "clock-b",
                KeyLifecycleStatus::Retired,
            ),
        ],
    )
    .unwrap();
    let policy = policy(2_000, true);
    let (claim, authority) = verified_bootstrap(&policy, &one_active);
    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(
            &authority,
            &claim,
            &policy,
            &one_active,
        )
        .unwrap_err(),
        ClockEvaluationPermitError::InsufficientClockAuthorityKeys {
            actual: 1,
            required: 2,
        }
    );

    let one_algorithm = TrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "clock-a",
                KeyLifecycleStatus::Active,
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "clock-b",
                KeyLifecycleStatus::Active,
            ),
        ],
    )
    .unwrap();
    let (claim, authority) = verified_bootstrap(&policy, &one_algorithm);
    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(
            &authority,
            &claim,
            &policy,
            &one_algorithm,
        )
        .unwrap_err(),
        ClockEvaluationPermitError::EligibleAlgorithmDiversityMissing
    );
}

#[test]
fn transition_arithmetic_fails_closed_on_overflow() {
    let trust = snapshot();
    let policy = policy(u64::MAX, true);
    let (claim, authority) = verified_bootstrap(&policy, &trust);
    assert_eq!(
        derive_bootstrap_clock_evaluation_permit_v2(&authority, &claim, &policy, &trust)
            .unwrap_err(),
        ClockEvaluationPermitError::TransitionEnvelopeOverflow
    );
}

#[test]
fn permit_source_cannot_admit_clock_evidence_or_deserialize_authority() {
    let source = include_str!("../src/clock_evaluation_permit.rs");
    assert!(!source.contains("ClockObservation"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("ClockWindowEvaluationWitness"));
    assert!(!source.contains("verify_clock_quorum"));
    assert!(!source.contains("pub fn unchecked"));
    assert!(!source.contains("pub fn from_digest"));
    assert!(!source.contains("pub fn from_verified"));

    let permit_at = source.find("pub struct ClockEvaluationPermitV2").unwrap();
    let derive_at = source[..permit_at].rfind("#[derive").unwrap();
    let permit_derive = &source[derive_at..permit_at];
    assert!(!permit_derive.contains("Serialize"));
    assert!(!permit_derive.contains("Deserialize"));
}
