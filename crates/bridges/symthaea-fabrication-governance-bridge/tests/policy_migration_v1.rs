use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_governance_bridge::{
    CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE, ClockGovernedPolicyMigrationError,
    PreparedClockGovernedPolicyMigrationV1, authorize_clock_governed_policy_migration_v1,
    prepare_clock_governed_policy_migration_v1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::{Sha256Digest, sha256};
use symthaea_fabrication_kernel::policy_migration::{
    PolicyBinding, PolicyInvariantBinding, PolicyInvariantDisposition, PolicyInvariantMigration,
    PolicyMigrationPlan, PolicyMigrationPolicy, digest_policy_migration_plan,
};
use symthaea_fabrication_kernel::threshold::{
    SignedThresholdApproval, ThresholdApprovalSigner, ThresholdApprovalVerifier,
    ThresholdCeremonyPolicy, sign_threshold_approval, verify_threshold_ceremony,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyV1, qualify_clock_governed_threshold_ceremony_v1,
};
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockBootstrapAuthorityEvidenceV2,
    ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPolicyV4, ClockGovernanceEvaluationEnvelopeV1, ClockGovernanceTimeError,
    ClockObservation, ClockObservationVerifier, ClockQuorumPolicyRevisionV1, DetachedSignature,
    KeyLifecycleStatus as ClockKeyLifecycleStatus, KeyTrustRecord as ClockKeyTrustRecord,
    KeyUsage as ClockKeyUsage, Sha256Digest as ClockSha256Digest,
    SignatureAlgorithm as ClockSignatureAlgorithm, TrustSnapshot as ClockTrustSnapshot,
    accept_bootstrap_clock_basis_v5, bind_bootstrap_operational_clock_basis_v1,
    derive_bootstrap_clock_evaluation_permit_v4, derive_clock_governance_evaluation_envelope_v1,
    digest_trust_snapshot as digest_clock_trust_snapshot, verify_clock_bootstrap_authority,
};

struct Provider {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

impl ThresholdApprovalSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }

    fn key_id(&self) -> &str {
        self.key_id
    }

    fn sign_threshold_approval(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}

impl ThresholdApprovalVerifier for Provider {
    fn verify_threshold_approval(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

fn fabrication_trust() -> TrustSnapshot {
    TrustSnapshot::new(
        9,
        1_000,
        2_000,
        vec![
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "a".into(),
                not_before_unix_s: 1_000,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::PolicyMigration]),
            },
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "b".into(),
                not_before_unix_s: 1_000,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::PolicyMigration]),
            },
        ],
    )
    .unwrap()
}

fn migration_policy() -> PolicyMigrationPolicy {
    PolicyMigrationPolicy {
        maximum_activation_delay_s: 10,
        maximum_rollback_window_s: 600,
        maximum_waiver_lifetime_s: 60,
        allow_waivers: false,
    }
}

fn threshold_policy() -> ThresholdCeremonyPolicy {
    ThresholdCeremonyPolicy {
        key_usage: KeyUsage::PolicyMigration,
        ..ThresholdCeremonyPolicy::default()
    }
}

fn migration_plan(activates_at_unix_s: u64) -> PolicyMigrationPlan {
    let predecessor = PolicyBinding::new(
        "upgrade-authority",
        "1",
        sha256(b"policy-v1"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"fail-closed-v1"),
        }],
    )
    .unwrap();
    let successor = PolicyBinding::new(
        "upgrade-authority",
        "2",
        sha256(b"policy-v2"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"fail-closed-v2"),
        }],
    )
    .unwrap();
    PolicyMigrationPlan {
        schema_version: "symthaea.fabrication.policy-migration.v1".into(),
        predecessor,
        successor,
        activates_at_unix_s,
        rollback_deadline_unix_s: activates_at_unix_s + 300,
        rationale: "strengthen fail-closed upgrade authority".into(),
        migrations: vec![PolicyInvariantMigration {
            name: "fail-closed".into(),
            predecessor_digest: sha256(b"fail-closed-v1"),
            successor_digest: Some(sha256(b"fail-closed-v2")),
            disposition: PolicyInvariantDisposition::Strengthened,
        }],
    }
}

fn containment_state() -> FabricationContainmentState {
    FabricationContainmentState::genesis(1, sha256(b"resilience-v1")).unwrap()
}

fn clock_digest(hex: &str) -> ClockSha256Digest {
    ClockSha256Digest::from_hex(hex).expect("canonical digest")
}

fn repeated_clock_hex(ch: char) -> ClockSha256Digest {
    clock_digest(&std::iter::repeat(ch).take(64).collect::<String>())
}

fn clock_usages() -> BTreeSet<ClockKeyUsage> {
    BTreeSet::from([ClockKeyUsage::ClockAuthority, ClockKeyUsage::ClockContinuity])
}

fn clock_snapshot() -> ClockTrustSnapshot {
    ClockTrustSnapshot::new(
        7,
        1_000,
        2_000,
        vec![
            ClockKeyTrustRecord {
                algorithm: ClockSignatureAlgorithm::Ed25519,
                key_id: "clock-a".into(),
                not_before_unix_s: 900,
                not_after_unix_s: Some(3_000),
                status: ClockKeyLifecycleStatus::Active,
                usages: clock_usages(),
            },
            ClockKeyTrustRecord {
                algorithm: ClockSignatureAlgorithm::MlDsa65,
                key_id: "clock-b".into(),
                not_before_unix_s: 900,
                not_after_unix_s: Some(3_000),
                status: ClockKeyLifecycleStatus::Active,
                usages: clock_usages(),
            },
        ],
    )
    .unwrap()
}

#[derive(Default)]
struct BootstrapVerifier;

impl ClockBootstrapAuthorityVerifier for BootstrapVerifier {
    fn provider_id(&self) -> &str {
        "platform-root-01"
    }

    fn authority_policy_digest(&self) -> ClockSha256Digest {
        repeated_clock_hex('b')
    }

    fn verify_clock_bootstrap_authority(
        &self,
        _canonical_claim_bytes: &[u8],
        external_evidence_digest: ClockSha256Digest,
    ) -> Result<bool, String> {
        Ok(external_evidence_digest == repeated_clock_hex('c'))
    }
}

struct ObservationVerifier {
    calls: Cell<usize>,
}

impl ClockObservationVerifier for ObservationVerifier {
    fn verify_clock_observation(
        &self,
        _algorithm: &ClockSignatureAlgorithm,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        self.calls.set(self.calls.get() + 1);
        Ok(true)
    }
}

fn clock_observation(
    source_id: &str,
    observed_unix_ms: u64,
    uncertainty_ms: u64,
    algorithm: ClockSignatureAlgorithm,
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

fn clock_envelope() -> ClockGovernanceEvaluationEnvelopeV1 {
    let snapshot = clock_snapshot();
    let quorum = ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true).unwrap();
    let continuity = ClockContinuityPolicyRevisionV1::new(1, 10_000, 60_000, 1, true).unwrap();
    let evaluation = ClockEvaluationPolicyV4::new(
        repeated_clock_hex('b'),
        &quorum,
        &continuity,
        2_000,
        2,
        true,
    )
    .unwrap();
    let snapshot_digest = digest_clock_trust_snapshot(&snapshot).unwrap();
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
        repeated_clock_hex('b'),
        repeated_clock_hex('c'),
    )
    .unwrap();
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
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
        clock_observation(
            "source-b",
            1_500_040,
            120,
            ClockSignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        clock_observation(
            "source-a",
            1_500_000,
            100,
            ClockSignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ];
    let verifier = ObservationVerifier {
        calls: Cell::new(0),
    };
    let basis =
        accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier).unwrap();
    assert_eq!(verifier.calls.get(), 2);
    let operational =
        bind_bootstrap_operational_clock_basis_v1(&basis, &evaluation, &snapshot).unwrap();
    let envelope = derive_clock_governance_evaluation_envelope_v1(&operational).unwrap();
    assert_eq!(envelope.lower_unix_ms(), 1_499_920);
    assert_eq!(envelope.upper_unix_ms(), 1_500_100);
    envelope
}

fn approvals(payload: Sha256Digest) -> Vec<SignedThresholdApproval> {
    let a = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "a",
    };
    let b = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "b",
    };
    vec![
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
            payload,
            1_499,
            1_502,
            &a,
        )
        .unwrap(),
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
            payload,
            1_499,
            1_502,
            &b,
        )
        .unwrap(),
    ]
}

fn qualify_threshold(
    prepared: &PreparedClockGovernedPolicyMigrationV1,
    policy: &ThresholdCeremonyPolicy,
    trust: &TrustSnapshot,
    containment: &FabricationContainmentState,
    envelope: &ClockGovernanceEvaluationEnvelopeV1,
) -> ClockGovernedThresholdCeremonyV1 {
    let signed = approvals(prepared.signing_payload_digest());
    let verifier = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "a",
    };
    let legacy = verify_threshold_ceremony(
        CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
        prepared.signing_payload_digest(),
        &signed,
        policy,
        trust,
        1_500,
        &verifier,
    )
    .unwrap();
    qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &signed,
        policy,
        trust,
        &containment.signer_compromise_tracker,
        envelope,
    )
    .unwrap()
}

#[test]
fn full_context_quorum_mints_opaque_interval_safe_migration_authority() {
    let plan = migration_plan(1_501);
    let migration_policy = migration_policy();
    let threshold_policy = threshold_policy();
    let trust = fabrication_trust();
    let containment = containment_state();
    let envelope = clock_envelope();

    let prepared = prepare_clock_governed_policy_migration_v1(
        plan.clone(),
        &migration_policy,
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();

    // Preserve legacy plan-digest parity, but do not use the bare plan digest as
    // the new quorum payload. The prepared ID additionally commits policy,
    // trust, containment and clock context.
    let legacy_plan_digest = digest_policy_migration_plan(&plan, &migration_policy, 1_500).unwrap();
    assert_eq!(prepared.plan_digest(), legacy_plan_digest);
    assert_ne!(prepared.signing_payload_digest(), legacy_plan_digest);

    let threshold = qualify_threshold(
        &prepared,
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    );
    let authorized = authorize_clock_governed_policy_migration_v1(prepared, &threshold).unwrap();

    assert_eq!(authorized.plan().activates_at_unix_s, 1_501);
    assert_eq!(authorized.clock_envelope_id(), envelope.id());
    assert_eq!(authorized.threshold_ceremony_id(), threshold.id());
    assert_eq!(authorized.containment_generation(), 1);
}

#[test]
fn activation_at_scalar_second_1500_is_rejected_when_upper_bound_is_1500_100() {
    let error = prepare_clock_governed_policy_migration_v1(
        migration_plan(1_500),
        &migration_policy(),
        &threshold_policy(),
        &fabrication_trust(),
        &containment_state(),
        &clock_envelope(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyMigrationError::Clock(
            ClockGovernanceTimeError::ActivationMayBeInPast
        )
    );
}

#[test]
fn activation_at_1510_is_too_late_from_interval_lower_bound_under_ten_second_limit() {
    let error = prepare_clock_governed_policy_migration_v1(
        migration_plan(1_510),
        &migration_policy(),
        &threshold_policy(),
        &fabrication_trust(),
        &containment_state(),
        &clock_envelope(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyMigrationError::Clock(
            ClockGovernanceTimeError::ActivationMayBeTooLate
        )
    );
}

#[test]
fn migration_policy_is_part_of_the_quorum_signing_payload() {
    let plan = migration_plan(1_501);
    let threshold_policy = threshold_policy();
    let trust = fabrication_trust();
    let containment = containment_state();
    let envelope = clock_envelope();
    let strict = migration_policy();
    let mut broader = migration_policy();
    broader.maximum_rollback_window_s += 1;

    let strict_prepared = prepare_clock_governed_policy_migration_v1(
        plan.clone(),
        &strict,
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();
    let broader_prepared = prepare_clock_governed_policy_migration_v1(
        plan,
        &broader,
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();

    assert_eq!(strict_prepared.plan_digest(), broader_prepared.plan_digest());
    assert_ne!(strict_prepared.migration_policy_digest(), broader_prepared.migration_policy_digest());
    assert_ne!(strict_prepared.id(), broader_prepared.id());
}

#[test]
fn threshold_policy_cannot_be_substituted_after_signing() {
    let plan = migration_plan(1_501);
    let migration_policy = migration_policy();
    let expected_policy = threshold_policy();
    let trust = fabrication_trust();
    let containment = containment_state();
    let envelope = clock_envelope();
    let prepared = prepare_clock_governed_policy_migration_v1(
        plan,
        &migration_policy,
        &expected_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();

    let substituted_policy = ThresholdCeremonyPolicy {
        minimum_distinct_signers: 1,
        require_algorithm_diversity: false,
        key_usage: KeyUsage::PolicyMigration,
        ..ThresholdCeremonyPolicy::default()
    };
    let threshold = qualify_threshold(
        &prepared,
        &substituted_policy,
        &trust,
        &containment,
        &envelope,
    );
    let error = authorize_clock_governed_policy_migration_v1(prepared, &threshold).unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyMigrationError::ThresholdPolicyDigestMismatch
    );
}

#[test]
fn containment_head_substitution_changes_the_signing_payload_even_with_same_tracker() {
    let plan = migration_plan(1_501);
    let migration_policy = migration_policy();
    let threshold_policy = threshold_policy();
    let trust = fabrication_trust();
    let envelope = clock_envelope();
    let first = containment_state();
    let successor = first.successor(2, sha256(b"resilience-v2")).unwrap();

    let first_prepared = prepare_clock_governed_policy_migration_v1(
        plan.clone(),
        &migration_policy,
        &threshold_policy,
        &trust,
        &first,
        &envelope,
    )
    .unwrap();
    let successor_prepared = prepare_clock_governed_policy_migration_v1(
        plan,
        &migration_policy,
        &threshold_policy,
        &trust,
        &successor,
        &envelope,
    )
    .unwrap();

    assert_eq!(
        first_prepared.compromise_tracker_digest(),
        successor_prepared.compromise_tracker_digest()
    );
    assert_ne!(
        first_prepared.containment_state_digest(),
        successor_prepared.containment_state_digest()
    );
    assert_ne!(first_prepared.id(), successor_prepared.id());
}

#[test]
fn policy_migration_requires_real_policy_migration_key_usage() {
    let wrong_threshold_policy = ThresholdCeremonyPolicy::default();
    let error = prepare_clock_governed_policy_migration_v1(
        migration_plan(1_501),
        &migration_policy(),
        &wrong_threshold_policy,
        &fabrication_trust(),
        &containment_state(),
        &clock_envelope(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyMigrationError::ThresholdPolicyUsageMismatch
    );
}

#[test]
fn live_authority_surface_has_no_caller_scalar_time_or_deserialization_path() {
    let source = include_str!("../src/lib.rs");
    for function in [
        "pub fn prepare_clock_governed_policy_migration_v1",
        "pub fn authorize_clock_governed_policy_migration_v1",
    ] {
        let start = source.find(function).expect("authority function");
        let rest = &source[start..];
        let end = rest.find(") -> Result").expect("authority signature") + 1;
        let signature = &rest[..end];
        assert!(!signature.contains("now_unix_s"));
        assert!(!signature.contains("proposed_at_unix_s"));
        assert!(!signature.contains("evaluation_time_unix_s"));
    }

    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct PreparedClockGovernedPolicyMigrationV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedPolicyMigrationV1"
    ));
}
