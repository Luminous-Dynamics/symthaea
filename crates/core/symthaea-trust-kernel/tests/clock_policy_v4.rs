// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_trust_kernel::{
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    ClockContinuityPolicyRevisionV1, ClockEvaluationPermitError, ClockEvaluationPolicyV4,
    ClockQuorumPolicyRevisionV1, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, Sha256Digest,
    SignatureAlgorithm, TrustSnapshot, derive_bootstrap_clock_evaluation_permit_v4,
    digest_trust_snapshot, verify_clock_bootstrap_authority,
};

const TRUST_ID: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const QUORUM_ID: &str = "4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304";
const CONTINUITY_ID: &str = "904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06";
const EVAL_ID: &str = "4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234";
const CLAIM_ID: &str = "c913a1829fe0bde63bbd2f0a4fda9344ed4de09fb9dece23ebf3189cb39e9fbd";
const EVIDENCE_ID: &str = "42d1b984349521b7d9dfc0d7ef149094a15231bd557c698acc7e276efb8a901d";
const AUTHORITY_ID: &str = "aecff7b0ad2a40e2ab4a0dd7fc6dca4684046cceda5df0b36071e6bf623f8582";
const ANCHOR_ID: &str = "c9fc46d4bbdccef92c2bc86c3682f2319d6837f753cf954fff30dcd451b5b588";
const PERMIT_ID: &str = "1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f";

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
    fn provider_id(&self) -> &str { "platform-root-01" }
    fn authority_policy_digest(&self) -> Sha256Digest { repeated_hex('b') }
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
    ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true).expect("fixture quorum policy")
}

fn continuity_policy() -> ClockContinuityPolicyRevisionV1 {
    ClockContinuityPolicyRevisionV1::new(1, 10_000, 60_000, 1, true)
        .expect("fixture continuity policy")
}

fn evaluation_policy(
    quorum: &ClockQuorumPolicyRevisionV1,
    continuity: &ClockContinuityPolicyRevisionV1,
) -> ClockEvaluationPolicyV4 {
    ClockEvaluationPolicyV4::new(repeated_hex('b'), quorum, continuity, 2_000, 2, true)
        .expect("fixture evaluation policy")
}

fn verified_authority(
    snapshot: &TrustSnapshot,
    evaluation_policy: &ClockEvaluationPolicyV4,
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
fn frozen_policy_v4_chain_matches_independent_reference() {
    let snapshot = trust_snapshot();
    assert_eq!(digest_trust_snapshot(&snapshot).unwrap().to_hex(), TRUST_ID);

    let quorum = quorum_policy();
    let continuity = continuity_policy();
    assert_eq!(quorum.id().to_hex(), QUORUM_ID);
    assert_eq!(continuity.id().to_hex(), CONTINUITY_ID);

    let evaluation = evaluation_policy(&quorum, &continuity);
    assert_eq!(evaluation.id().to_hex(), EVAL_ID);
    assert_eq!(evaluation.clock_quorum_policy_id(), quorum.id());
    assert_eq!(evaluation.clock_continuity_policy_id(), continuity.id());

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

    let permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &snapshot,
    )
    .unwrap();

    assert_eq!(permit.basis_id().to_hex(), ANCHOR_ID);
    assert_eq!(permit.id().to_hex(), PERMIT_ID);
    assert_eq!(permit.evaluation_policy_id(), evaluation.id());
    assert_eq!(permit.clock_quorum_policy_id(), quorum.id());
    assert_eq!(permit.clock_continuity_policy_id(), continuity.id());
    assert_eq!(permit.evaluation_lower_unix_ms(), 1_499_000);
    assert_eq!(permit.evaluation_upper_unix_ms(), 1_501_500);
    assert_eq!(permit.eligible_clock_keys().len(), 2);

    let runtime_quorum = permit.runtime_quorum_policy().unwrap();
    assert_eq!(runtime_quorum.minimum_distinct_sources, 2);
    assert_eq!(runtime_quorum.maximum_uncertainty_ms, 5_000);
    assert_eq!(runtime_quorum.maximum_consensus_width_ms, 10_000);
    assert!(runtime_quorum.require_algorithm_diversity);

    let runtime_continuity = permit.runtime_continuity_policy().unwrap();
    assert_eq!(runtime_continuity.maximum_epoch_step, 1);
    assert_eq!(runtime_continuity.maximum_forward_gap_ms, 10_000);
    assert_eq!(runtime_continuity.maximum_consensus_jump_ms, 60_000);
    assert_eq!(runtime_continuity.minimum_shared_sources, 1);
    assert!(runtime_continuity.require_shared_algorithm);
}

#[test]
fn weaker_continuity_policy_cannot_substitute_for_authenticated_revision() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);
    let weaker = ClockContinuityPolicyRevisionV1::new(1, 20_000, 60_000, 1, true).unwrap();
    assert_ne!(weaker.id(), continuity.id());

    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &weaker, &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::ContinuityPolicyMismatch);
}

#[test]
fn new_evaluation_policy_for_weaker_continuity_is_not_authorized_by_old_claim() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);
    let weaker = ClockContinuityPolicyRevisionV1::new(1, 20_000, 60_000, 1, true).unwrap();
    let weaker_evaluation = evaluation_policy(&quorum, &weaker);
    assert_ne!(weaker_evaluation.id(), evaluation.id());

    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &weaker_evaluation, &quorum, &weaker, &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::BootstrapPolicyMismatch);
}

#[test]
fn weaker_quorum_policy_cannot_substitute_for_authenticated_revision() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);
    let weaker = ClockQuorumPolicyRevisionV1::new(2, 8, 10_000, 10_000, true).unwrap();

    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &weaker, &continuity, &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::QuorumPolicyMismatch);
}

#[test]
fn permit_retains_both_authorized_policy_records() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);
    let permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &snapshot,
    )
    .unwrap();

    let unrelated_quorum = ClockQuorumPolicyRevisionV1::new(2, 8, 20_000, 20_000, false).unwrap();
    let unrelated_continuity = ClockContinuityPolicyRevisionV1::new(2, 30_000, 120_000, 1, false).unwrap();
    assert_ne!(unrelated_quorum.id(), permit.clock_quorum_policy_id());
    assert_ne!(unrelated_continuity.id(), permit.clock_continuity_policy_id());
    assert_eq!(permit.runtime_quorum_policy().unwrap().maximum_uncertainty_ms, 5_000);
    assert_eq!(permit.runtime_continuity_policy().unwrap().maximum_forward_gap_ms, 10_000);
}

#[test]
fn snapshot_must_cover_entire_transition_envelope() {
    let mut snapshot = trust_snapshot();
    snapshot.expires_at_unix_s = 1_501;
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);
    let (claim, authority) = verified_authority(&snapshot, &evaluation);

    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::SnapshotNotValidForEnvelope);
}

#[test]
fn whole_envelope_key_pool_still_requires_count_and_diversity() {
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let evaluation = evaluation_policy(&quorum, &continuity);

    let mut one_key = trust_snapshot();
    one_key.keys[1].status = KeyLifecycleStatus::Retired;
    let (claim, authority) = verified_authority(&one_key, &evaluation);
    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &one_key,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ClockEvaluationPermitError::InsufficientClockAuthorityKeys { actual: 1, required: 2 }
    ));

    let mut same_algorithm = trust_snapshot();
    same_algorithm.keys[1].algorithm = SignatureAlgorithm::Ed25519;
    let (claim, authority) = verified_authority(&same_algorithm, &evaluation);
    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &same_algorithm,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::EligibleAlgorithmDiversityMissing);
}

#[test]
fn transition_envelope_overflow_fails_closed() {
    let snapshot = trust_snapshot();
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let huge = ClockEvaluationPolicyV4::new(
        repeated_hex('b'), &quorum, &continuity, u64::MAX, 2, true,
    )
    .unwrap();
    let (claim, authority) = verified_authority(&snapshot, &huge);

    let error = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &huge, &quorum, &continuity, &snapshot,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::TransitionEnvelopeOverflow);
}

#[test]
fn zero_policy_record_digest_is_rejected() {
    let quorum = quorum_policy();
    let continuity = continuity_policy();
    let error = ClockEvaluationPolicyV4::new(
        repeated_hex('0'), &quorum, &continuity, 2_000, 2, true,
    )
    .unwrap_err();
    assert_eq!(error, ClockEvaluationPermitError::InvalidEvaluationPolicy);
}

#[test]
fn source_surface_has_no_provisional_v3_or_v2_permit_path() {
    let source = include_str!("../src/clock_evaluation_permit.rs");
    assert!(!source.contains("ClockEvaluationPolicyV3"));
    assert!(!source.contains("ClockEvaluationPermitV3"));
    assert!(!source.contains("derive_bootstrap_clock_evaluation_permit_v3"));
    assert!(!source.contains("ClockEvaluationPolicyV2"));
    assert!(!source.contains("ClockEvaluationPermitV2"));
    assert!(source.contains("ClockQuorumPolicyRevisionV1"));
    assert!(source.contains("ClockContinuityPolicyRevisionV1"));
    assert!(source.contains("ClockEvaluationPolicyV4"));
    assert!(source.contains("ClockEvaluationPermitV4"));
}
