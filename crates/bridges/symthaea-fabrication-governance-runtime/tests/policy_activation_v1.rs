use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_governance_bridge::{
    CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE, ClockGovernedPolicyMigrationV1,
    authorize_clock_governed_policy_migration_v1, prepare_clock_governed_policy_migration_v1,
};
use symthaea_fabrication_governance_runtime::{
    ClockGovernedPolicyActivationError, derive_clock_governed_policy_activation_permit_v1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::sha256;
use symthaea_fabrication_kernel::policy_migration::{
    PolicyBinding, PolicyInvariantBinding, PolicyInvariantDisposition, PolicyInvariantMigration,
    PolicyMigrationPlan, PolicyMigrationPolicy,
};
use symthaea_fabrication_kernel::threshold::{
    ThresholdApprovalSigner, ThresholdApprovalVerifier, ThresholdCeremonyPolicy,
    sign_threshold_approval, verify_threshold_ceremony,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_trust_bridge::qualify_clock_governed_threshold_ceremony_v1;
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockBootstrapAuthorityEvidenceV2,
    ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPolicyV4, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicyRevisionV1, DetachedSignature,
    KeyLifecycleStatus as ClockKeyLifecycleStatus, KeyTrustRecord as ClockKeyTrustRecord,
    KeyUsage as ClockKeyUsage, OperationalClockBasisV1, Sha256Digest as ClockSha256Digest,
    SignatureAlgorithm as ClockSignatureAlgorithm, TrustSnapshot as ClockTrustSnapshot,
    accept_bootstrap_clock_basis_v5, accept_operational_clock_successor_v1,
    bind_bootstrap_operational_clock_basis_v1, derive_bootstrap_clock_evaluation_permit_v4,
    derive_clock_governance_evaluation_envelope_v1,
    derive_operational_clock_successor_permit_v2,
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

fn migration_plan() -> PolicyMigrationPlan {
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
        activates_at_unix_s: 1_501,
        rollback_deadline_unix_s: 1_801,
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
        3_000,
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
    epoch: u64,
    algorithm: ClockSignatureAlgorithm,
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
        clock_observation(
            "source-b",
            base_ms + 40,
            120,
            epoch,
            ClockSignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        clock_observation(
            "source-a",
            base_ms,
            100,
            epoch,
            ClockSignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn authorization_basis() -> OperationalClockBasisV1 {
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
    let verifier = ObservationVerifier {
        calls: Cell::new(0),
    };
    let accepted = accept_bootstrap_clock_basis_v5(
        &permit,
        &observations(42, 1_500_000),
        &snapshot,
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    bind_bootstrap_operational_clock_basis_v1(&accepted, &evaluation, &snapshot).unwrap()
}

fn successor(prior: &OperationalClockBasisV1, base_ms: u64) -> OperationalClockBasisV1 {
    let permit = derive_operational_clock_successor_permit_v2(prior).unwrap();
    let verifier = ObservationVerifier {
        calls: Cell::new(0),
    };
    let next = accept_operational_clock_successor_v1(
        &permit,
        &observations(prior.epoch() + 1, base_ms),
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(next.predecessor_operational_basis_id(), Some(prior.id()));
    next
}

fn authorized_migration(basis: &OperationalClockBasisV1) -> ClockGovernedPolicyMigrationV1 {
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    assert_eq!(envelope.lower_unix_ms(), 1_499_920);
    assert_eq!(envelope.upper_unix_ms(), 1_500_100);
    let migration_policy = migration_policy();
    let threshold_policy = threshold_policy();
    let trust = fabrication_trust();
    let containment = containment_state();
    let prepared = prepare_clock_governed_policy_migration_v1(
        migration_plan(),
        &migration_policy,
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();

    let a = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "a",
    };
    let b = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "b",
    };
    let signed = vec![
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
            prepared.signing_payload_digest(),
            1_499,
            1_502,
            &a,
        )
        .unwrap(),
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
            prepared.signing_payload_digest(),
            1_499,
            1_502,
            &b,
        )
        .unwrap(),
    ];
    let legacy = verify_threshold_ceremony(
        CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
        prepared.signing_payload_digest(),
        &signed,
        &threshold_policy,
        &trust,
        1_500,
        &a,
    )
    .unwrap();
    let threshold = qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &signed,
        &threshold_policy,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap();
    authorize_clock_governed_policy_migration_v1(prepared, &threshold).unwrap()
}

#[test]
fn activation_requires_unbroken_descendant_clock_lineage_and_definite_time() {
    let root42 = authorization_basis();
    let migration = authorized_migration(&root42);
    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);

    // Root 44 straddles second 1501: lower is 1500.920, so activation is
    // possible but not yet certain for every true time in the interval.
    let not_yet = derive_clock_governed_policy_activation_permit_v1(
        &migration,
        &root42,
        &[root43.clone(), root44.clone()],
    )
    .unwrap_err();
    assert!(matches!(
        not_yet,
        ClockGovernedPolicyActivationError::ActivationNotYetCertain { .. }
    ));

    let root45 = successor(&root44, 1_501_500);
    let permit = derive_clock_governed_policy_activation_permit_v1(
        &migration,
        &root42,
        &[root43, root44, root45.clone()],
    )
    .unwrap();

    assert_eq!(permit.migration_id(), migration.id());
    assert_eq!(permit.authorization_operational_basis_id(), root42.id());
    assert_eq!(permit.activation_operational_basis_id(), root45.id());
    assert_eq!(permit.clock_lineage_hops(), 3);
    assert_eq!(permit.activates_at_unix_s(), 1_501);
}

#[test]
fn valid_descendant_cannot_skip_the_explicit_ancestry_proof() {
    let root42 = authorization_basis();
    let migration = authorized_migration(&root42);
    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);

    let error = derive_clock_governed_policy_activation_permit_v1(
        &migration,
        &root42,
        &[root44, root45],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ClockGovernedPolicyActivationError::BrokenClockLineage { hop: 1, .. }
    ));
}

#[test]
fn different_authorization_basis_cannot_be_substituted() {
    let root42 = authorization_basis();
    let migration = authorized_migration(&root42);
    let root43 = successor(&root42, 1_500_500);

    let error = derive_clock_governed_policy_activation_permit_v1(
        &migration,
        &root43,
        &[],
    )
    .unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyActivationError::AuthorizationClockMismatch
    );
}

#[test]
fn runtime_authority_surface_has_no_scalar_now_or_deserialization_path() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub fn derive_clock_governed_policy_activation_permit_v1")
        .expect("activation permit function");
    let rest = &source[start..];
    let end = rest.find(") -> Result").expect("activation signature") + 1;
    let signature = &rest[..end];

    assert!(!signature.contains("now_unix_s"));
    assert!(!signature.contains("evaluation_time_unix_s"));
    assert!(signature.contains("OperationalClockBasisV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedPolicyActivationPermitV1"
    ));
}
