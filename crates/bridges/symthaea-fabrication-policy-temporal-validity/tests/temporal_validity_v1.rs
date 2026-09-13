use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_governance_bridge::{
    CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE, authorize_clock_governed_policy_migration_v1,
    prepare_clock_governed_policy_migration_v1,
};
use symthaea_fabrication_governance_runtime::derive_clock_governed_policy_activation_permit_v1;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::sha256;
use symthaea_fabrication_kernel::policy_migration::{
    POLICY_MIGRATION_SCHEMA, PolicyBinding, PolicyInvariantBinding, PolicyInvariantDisposition,
    PolicyInvariantMigration, PolicyMigrationPlan, PolicyMigrationPolicy,
};
use symthaea_fabrication_kernel::threshold::{
    ThresholdApprovalSigner, ThresholdApprovalVerifier, ThresholdCeremonyPolicy,
    sign_threshold_approval, verify_threshold_ceremony,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_policy_lineage::{
    CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE, ClockGovernedPolicyLineageV1,
    advance_clock_governed_policy_lineage_v1,
    authorize_clock_governed_policy_lineage_genesis_v1,
    prepare_clock_governed_policy_lineage_genesis_v1,
};
use symthaea_fabrication_policy_temporal_validity::{
    ClockGovernedPolicyTemporalValidityError,
    derive_clock_governed_policy_temporal_validity_permit_v1,
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
        3_000,
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

fn threshold_policy() -> ThresholdCeremonyPolicy {
    ThresholdCeremonyPolicy {
        key_usage: KeyUsage::PolicyMigration,
        ..ThresholdCeremonyPolicy::default()
    }
}

fn containment_state() -> FabricationContainmentState {
    FabricationContainmentState::genesis(1, sha256(b"resilience-v1")).unwrap()
}

fn clock_digest(hex: &str) -> ClockSha256Digest {
    ClockSha256Digest::from_hex(hex).unwrap()
}

fn repeated_clock_hex(ch: char) -> ClockSha256Digest {
    clock_digest(&std::iter::repeat_n(ch, 64).collect::<String>())
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

fn observation(
    source_id: &str,
    observed_unix_ms: u64,
    uncertainty_ms: u64,
    epoch: u64,
    algorithm: ClockSignatureAlgorithm,
    key_id: &str,
) -> ClockObservation {
    ClockObservation {
        schema_version: CLOCK_OBSERVATION_SCHEMA.into(),
        source_id: source_id.into(),
        observed_unix_ms,
        uncertainty_ms,
        epoch,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.into(),
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
            ClockSignatureAlgorithm::MlDsa65,
            "clock-b",
        ),
        observation(
            "source-a",
            base_ms,
            100,
            epoch,
            ClockSignatureAlgorithm::Ed25519,
            "clock-a",
        ),
    ]
}

fn root42() -> OperationalClockBasisV1 {
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
    bind_bootstrap_operational_clock_basis_v1(&accepted, &evaluation, &snapshot).unwrap()
}

fn successor(prior: &OperationalClockBasisV1, base_ms: u64) -> OperationalClockBasisV1 {
    let permit = derive_operational_clock_successor_permit_v2(prior).unwrap();
    accept_operational_clock_successor_v1(
        &permit,
        &observations(prior.epoch() + 1, base_ms),
        &ObservationVerifier {
            calls: Cell::new(0),
        },
    )
    .unwrap()
}

fn policy(version: &str, audit: bool) -> PolicyBinding {
    let mut invariants = vec![PolicyInvariantBinding {
        name: "fail-closed".into(),
        digest: sha256(if version == "1" { b"fail-v1" } else { b"fail-v2" }),
    }];
    if audit {
        invariants.push(PolicyInvariantBinding {
            name: "audit-anchor".into(),
            digest: sha256(b"audit-v1"),
        });
    }
    PolicyBinding::new(
        "upgrade-authority",
        version,
        sha256(format!("policy-{version}").as_bytes()),
        invariants,
    )
    .unwrap()
}

fn waiver_plan() -> PolicyMigrationPlan {
    PolicyMigrationPlan {
        schema_version: POLICY_MIGRATION_SCHEMA.into(),
        predecessor: policy("1", true),
        successor: policy("2", false),
        activates_at_unix_s: 1_501,
        rollback_deadline_unix_s: 1_801,
        rationale: "temporary audited incident waiver".into(),
        migrations: vec![
            PolicyInvariantMigration {
                name: "audit-anchor".into(),
                predecessor_digest: sha256(b"audit-v1"),
                successor_digest: None,
                disposition: PolicyInvariantDisposition::Waived {
                    incident_digest: sha256(b"incident-audit"),
                    expires_at_unix_s: 1_504,
                },
            },
            PolicyInvariantMigration {
                name: "fail-closed".into(),
                predecessor_digest: sha256(b"fail-v1"),
                successor_digest: Some(sha256(b"fail-v2")),
                disposition: PolicyInvariantDisposition::Strengthened,
            },
        ],
    }
}

fn governed_ceremony(
    purpose: &str,
    payload: symthaea_fabrication_kernel::crypto_digest::Sha256Digest,
    basis: &OperationalClockBasisV1,
) -> symthaea_fabrication_trust_bridge::ClockGovernedThresholdCeremonyV1 {
    let trust = fabrication_trust();
    let containment = containment_state();
    let threshold = threshold_policy();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    let a = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "a",
    };
    let b = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "b",
    };
    let approvals = vec![
        sign_threshold_approval(purpose, payload, 1_499, 1_505, &a).unwrap(),
        sign_threshold_approval(purpose, payload, 1_499, 1_505, &b).unwrap(),
    ];
    let scalar = verify_threshold_ceremony(
        purpose,
        payload,
        &approvals,
        &threshold,
        &trust,
        1_500,
        &a,
    )
    .unwrap();
    qualify_clock_governed_threshold_ceremony_v1(
        &scalar,
        &approvals,
        &threshold,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap()
}

fn waived_lineage(
    root: &OperationalClockBasisV1,
) -> (ClockGovernedPolicyLineageV1, Vec<OperationalClockBasisV1>) {
    let trust = fabrication_trust();
    let containment = containment_state();
    let threshold = threshold_policy();
    let genesis_prepared = prepare_clock_governed_policy_lineage_genesis_v1(
        policy("1", true),
        &threshold,
        &trust,
        &containment,
        root,
    )
    .unwrap();
    let genesis_ceremony = governed_ceremony(
        CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        genesis_prepared.signing_payload_digest(),
        root,
    );
    let genesis = authorize_clock_governed_policy_lineage_genesis_v1(
        genesis_prepared,
        &genesis_ceremony,
    )
    .unwrap();

    let envelope = derive_clock_governance_evaluation_envelope_v1(root).unwrap();
    let migration_prepared = prepare_clock_governed_policy_migration_v1(
        waiver_plan(),
        &PolicyMigrationPolicy {
            maximum_activation_delay_s: 10,
            maximum_rollback_window_s: 600,
            maximum_waiver_lifetime_s: 10,
            allow_waivers: true,
        },
        &threshold,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();
    let migration_ceremony = governed_ceremony(
        CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
        migration_prepared.signing_payload_digest(),
        root,
    );
    let migration = authorize_clock_governed_policy_migration_v1(
        migration_prepared,
        &migration_ceremony,
    )
    .unwrap();

    let root43 = successor(root, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);
    let activation = derive_clock_governed_policy_activation_permit_v1(
        &migration,
        root,
        &[root43.clone(), root44.clone(), root45.clone()],
    )
    .unwrap();
    let lineage = advance_clock_governed_policy_lineage_v1(
        &genesis,
        &migration,
        &activation,
        &[root43.clone(), root44.clone()],
        &root45,
    )
    .unwrap();
    (lineage, vec![root43, root44, root45])
}

#[test]
fn unresolved_waiver_is_temporally_valid_before_expiry() {
    let root = root42();
    let (lineage, chain) = waived_lineage(&root);
    let root45 = chain.last().unwrap();
    let permit = derive_clock_governed_policy_temporal_validity_permit_v1(
        &lineage,
        &[],
        root45,
    )
    .unwrap();

    assert_eq!(permit.lineage_id(), lineage.id());
    assert_eq!(permit.lineage_sequence(), lineage.sequence());
    assert_eq!(permit.active_waiver_count(), 1);
    assert_eq!(permit.current_operational_basis_id(), root45.id());
}

#[test]
fn unresolved_waiver_fails_closed_once_fresh_upper_bound_reaches_expiry() {
    let root = root42();
    let (lineage, chain) = waived_lineage(&root);
    let root45 = chain.last().unwrap().clone();
    let root46 = successor(&root45, 1_502_000);
    let root47 = successor(&root46, 1_502_500);
    let root48 = successor(&root47, 1_503_000);
    let root49 = successor(&root48, 1_503_500);
    let root50 = successor(&root49, 1_504_000);

    let error = derive_clock_governed_policy_temporal_validity_permit_v1(
        &lineage,
        &[root46, root47, root48, root49],
        &root50,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ClockGovernedPolicyTemporalValidityError::WaiverMayBeExpired {
            invariant,
            ..
        } if invariant == "audit-anchor"
    ));
}

#[test]
fn fresh_time_cannot_skip_the_lineage_states_latest_clock_basis() {
    let root = root42();
    let (lineage, chain) = waived_lineage(&root);
    let root45 = chain.last().unwrap().clone();
    let root46 = successor(&root45, 1_502_000);
    let root47 = successor(&root46, 1_502_500);

    let error = derive_clock_governed_policy_temporal_validity_permit_v1(
        &lineage,
        &[],
        &root47,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ClockGovernedPolicyTemporalValidityError::BrokenClockLineage { .. }
    ));
}

#[test]
fn temporal_validity_surface_has_no_scalar_now_or_deserialization_path() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub fn derive_clock_governed_policy_temporal_validity_permit_v1")
        .unwrap();
    let rest = &source[start..];
    let end = rest.find(") -> Result").unwrap() + 1;
    let signature = &rest[..end];
    assert!(!signature.contains("now_unix_s"));
    assert!(!signature.contains("evaluation_time_unix_s"));
    assert!(signature.contains("OperationalClockBasisV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedPolicyTemporalValidityPermitV1"
    ));
}
