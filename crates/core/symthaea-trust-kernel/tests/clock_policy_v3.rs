// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockEvaluationPermitError, ClockEvaluationPolicyV3, ClockQuorumPolicyRevisionV1,
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest, SignatureAlgorithm, TrustSnapshot,
    derive_bootstrap_clock_evaluation_permit_v3, digest_trust_snapshot,
    verify_clock_bootstrap_authority,
};

const TRUST_ID: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const QUORUM_ID: &str = "4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304";
const EVAL_ID: &str = "7a782b8adbf4e66a352c4a26ae614a458971d78c33c136cb8eeee6c6c28433b4";
const CLAIM_ID: &str = "840d076db077b2df95277923b4be3b38f1e2a7027222298876041c036456d212";
const EVIDENCE_ID: &str = "0f3169321314c398a6ca68be67447b14ffd3d4409d228662d1effaa3d60f55a6";
const AUTHORITY_ID: &str = "a0286a04ec3f4c0c6f8f2cc4bf97c7638bed43f913c09456e0931ea9c49e5f7c";
const ANCHOR_ID: &str = "61fe43c253ca8dde9c152b641208f4eddc2ddbbfc263124d9b2a712e1e23d9e5";
const PERMIT_ID: &str = "716c65376a721574bb53d2415e6b58f83d9c7483666b3799ca8fc4fe214f2cce";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).expect("frozen digest must be canonical")
}

fn repeated_hex(ch: char) -> Sha256Digest {
    digest(&std::iter::repeat_n(ch, 64).collect::<String>())
}

fn usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([KeyUsage::ClockAuthority, KeyUsage::ClockContinuity])
}

fn trust_snapshot() -> TrustSnapshot {
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
    .expect("fixture trust snapshot")
}

#[derive(Default)]
struct AcceptingBootstrapVerifier {
    calls: Cell<usize>,
}

impl ClockBootstrapAuthorityVerifier for AcceptingBootstrapVerifier {
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
        self.calls.set(self.calls.get() + 1);
        Ok(external_evidence_digest == repeated_hex('c'))
    }
}

fn quorum_policy() -> ClockQuorumPolicyRevisionV1 {
    ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true)
        .expect("fixture quorum policy")
}

fn evaluation_policy(quorum: &ClockQuorumPolicyRevisionV1) -> ClockEvaluationPolicyV3 {
    ClockEvaluationPolicyV3::new(repeated_hex('b'), quorum, 2_000, 2, true)
        .expect("fixture evaluation policy")
}

fn verified_authority(
    snapshot: &TrustSnapshot,
    evaluation_policy: &ClockEvaluationPolicyV3,
) -> (
    ClockBootstrapClaimV2,
    symthaea_trust_kernel::VerifiedClockBootstrapAuthorityV2,
) {
    let snapshot_digest = digest_trust_snapshot(snapshot).expect("snapshot digest");
    let claim = ClockBootstrapClaimV2::new(
        snapshot_digest,
        evaluation_policy.id().as_digest(),
        1_499_000,
        1_499_500,
    )
    .expect("bootstrap claim");
    let evidence = ClockBootstrapAuthorityEvidenceV2::new(
        &claim,
        "platform-root-01",
        repeated_hex('b'),
        repeated_hex('c'),
    )
    .expect("bootstrap authority evidence");
    let verifier = AcceptingBootstrapVerifier::default();
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &verifier)
        .expect("verified bootstrap authority");
    assert_eq!(verifier.calls.get(), 1);
    (claim, authority)
}

#[test]
fn frozen_policy_v3_chain_matches_independent_reference() {
    let snapshot = trust_snapshot();
    assert_eq!(digest_trust_snapshot(&snapshot).unwrap().to_hex(), TRUST_ID);

    let quorum = quorum_policy();
    assert_eq!(quorum.id().to_hex(), QUORUM_ID);

    let evaluation = evaluation_policy(&quorum);
    assert_eq!(evaluation.id().to_hex(), EVAL_ID);
    assert_eq!(evaluation.clock_quorum_policy_id(), quorum.id());

    let snapshot_digest = digest_trust_snapshot(&snapshot).unwrap();
    let claim = ClockBootstrapClaimV2::new(
        snapshot_digest,
        evaluation.id().as_digest(),
        1_499_000,
        1_499_500,
    )
    .unwrap();
    assert_eq!(claim.id().to_hex(), CLAIM_ID);

    let evidence = ClockBootstrapAuthorityEvidenceV2::new(
        &claim,
        "platform-root-01",
        repeated_hex('b'),
        repeated_hex('c'),
    )
    .unwrap();
    assert_eq!(evidence.id().to_hex(), EVIDENCE_ID);

    let verifier = AcceptingBootstrapVerifier::default();
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &verifier).unwrap();
    assert_eq!(authority.id().to_hex(), AUTHORITY_ID);

    let permit = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &snapshot,
    )
    .unwrap();

    assert_eq!(permit.basis_id().to_hex(), ANCHOR_ID);
    assert_eq!(permit.id().to_hex(), PERMIT_ID);
    assert_eq!(permit.evaluation_policy_id(), evaluation.id());
    assert_eq!(permit.clock_quorum_policy_id(), quorum.id());
    assert_eq!(permit.evaluation_lower_unix_ms(), 1_499_000);
    assert_eq!(permit.evaluation_upper_unix_ms(), 1_501_500);
    assert_eq!(permit.eligible_clock_keys().len(), 2);

    let runtime = permit.runtime_quorum_policy().unwrap();
    assert_eq!(runtime.minimum_distinct_sources, 2);
    assert_eq!(runtime.maximum_observations, 8);
    assert_eq!(runtime.maximum_uncertainty_ms, 5_000);
    assert_eq!(runtime.maximum_consensus_width_ms, 10_000);
    assert!(runtime.require_algorithm_diversity);
}

#[test]
fn weaker_caller_quorum_cannot_substitute_for_authorized_revision() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let evaluation = evaluation_policy(&quorum);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);

    let weaker = ClockQuorumPolicyRevisionV1::new(2, 8, 10_000, 10_000, true).unwrap();
    assert_ne!(weaker.id(), quorum.id());

    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &weaker,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::QuorumPolicyMismatch);
}

#[test]
fn new_evaluation_policy_for_weaker_quorum_is_not_authorized_by_old_claim() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let evaluation = evaluation_policy(&quorum);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);

    let weaker = ClockQuorumPolicyRevisionV1::new(2, 8, 10_000, 10_000, true).unwrap();
    let weaker_evaluation = evaluation_policy(&weaker);
    assert_ne!(weaker_evaluation.id(), evaluation.id());

    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &weaker_evaluation,
        &weaker,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::BootstrapPolicyMismatch);
}

#[test]
fn permit_retains_authorized_quorum_semantics_internally() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let evaluation = evaluation_policy(&quorum);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);
    let permit = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &snapshot,
    )
    .unwrap();

    let unrelated = ClockQuorumPolicyRevisionV1::new(2, 8, 20_000, 20_000, false).unwrap();
    assert_ne!(unrelated.id(), permit.clock_quorum_policy_id());

    let runtime = permit.runtime_quorum_policy().unwrap();
    assert_eq!(runtime.maximum_uncertainty_ms, 5_000);
    assert_eq!(runtime.maximum_consensus_width_ms, 10_000);
    assert!(runtime.require_algorithm_diversity);
}

#[test]
fn snapshot_must_cover_entire_transition_envelope() {
    let mut snapshot = trust_snapshot();
    snapshot.expires_at_unix_s = 1_501;

    let quorum = quorum_policy();
    let evaluation = evaluation_policy(&quorum);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);

    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::SnapshotNotValidForEnvelope);
}

#[test]
fn whole_envelope_key_pool_still_requires_count_and_diversity() {
    let mut one_key = trust_snapshot();
    one_key.keys[1].status = KeyLifecycleStatus::Retired;
    let quorum = quorum_policy();
    let evaluation = evaluation_policy(&quorum);
    let (claim, authority) = verified_authority(&one_key, &evaluation);
    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &one_key,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ClockEvaluationPermitError::InsufficientClockAuthorityKeys { actual: 1, required: 2 }
    ));

    let mut same_algorithm = trust_snapshot();
    same_algorithm.keys[1].algorithm = SignatureAlgorithm::Ed25519;
    let (claim, authority) = verified_authority(&same_algorithm, &evaluation);
    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &evaluation,
        &quorum,
        &same_algorithm,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ClockEvaluationPermitError::EligibleAlgorithmDiversityMissing
    );
}

#[test]
fn transition_envelope_overflow_fails_closed() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let huge = ClockEvaluationPolicyV3::new(repeated_hex('b'), &quorum, u64::MAX, 2, true)
        .unwrap();
    let (claim, authority) = verified_authority(&snapshot, &huge);

    let error = derive_bootstrap_clock_evaluation_permit_v3(
        &authority,
        &claim,
        &huge,
        &quorum,
        &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::TransitionEnvelopeOverflow);
}

#[test]
fn source_surface_has_no_v2_permit_or_evaluation_policy() {
    let source = include_str!("../src/clock_evaluation_permit.rs");
    assert!(!source.contains("ClockEvaluationPolicyV2"));
    assert!(!source.contains("ClockEvaluationPermitV2"));
    assert!(!source.contains("derive_bootstrap_clock_evaluation_permit_v2"));
    assert!(source.contains("ClockQuorumPolicyRevisionV1"));
    assert!(source.contains("ClockEvaluationPolicyV3"));
    assert!(source.contains("ClockEvaluationPermitV3"));
}
