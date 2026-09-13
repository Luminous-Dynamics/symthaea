use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_governance_bridge::{
    CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE, ClockGovernedPolicyMigrationV1,
    authorize_clock_governed_policy_migration_v1, prepare_clock_governed_policy_migration_v1,
};
use symthaea_fabrication_governance_runtime::{
    ClockGovernedPolicyActivationPermitV1, derive_clock_governed_policy_activation_permit_v1,
};
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
    CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE, ClockGovernedPolicyLineageError,
    ClockGovernedPolicyLineageV1, advance_clock_governed_policy_lineage_v1,
    authorize_clock_governed_policy_lineage_genesis_v1,
    prepare_clock_governed_policy_lineage_genesis_v1,
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

fn providers() -> (Provider, Provider) {
    (
        Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        },
        Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "b",
        },
    )
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

fn migration_policy(allow_waivers: bool) -> PolicyMigrationPolicy {
    PolicyMigrationPolicy {
        maximum_activation_delay_s: 10,
        maximum_rollback_window_s: 600,
        maximum_waiver_lifetime_s: 10,
        allow_waivers,
    }
}

fn clock_digest(hex: &str) -> ClockSha256Digest {
    ClockSha256Digest::from_hex(hex).expect("canonical digest")
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
    next
}

fn binding(version: &str, fail_closed: &[u8], audit: Option<&[u8]>) -> PolicyBinding {
    let mut invariants = vec![PolicyInvariantBinding {
        name: "fail-closed".into(),
        digest: sha256(fail_closed),
    }];
    if let Some(audit) = audit {
        invariants.push(PolicyInvariantBinding {
            name: "audit-anchor".into(),
            digest: sha256(audit),
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

fn strengthened_plan(
    predecessor: PolicyBinding,
    successor: PolicyBinding,
    activation: u64,
) -> PolicyMigrationPlan {
    PolicyMigrationPlan {
        schema_version: POLICY_MIGRATION_SCHEMA.into(),
        predecessor,
        successor,
        activates_at_unix_s: activation,
        rollback_deadline_unix_s: activation + 300,
        rationale: "strengthen fail-closed policy authority".into(),
        migrations: vec![PolicyInvariantMigration {
            name: "fail-closed".into(),
            predecessor_digest: sha256(b"fail-v1"),
            successor_digest: Some(sha256(b"fail-v2")),
            disposition: PolicyInvariantDisposition::Strengthened,
        }],
    }
}

fn first_waiver_plan() -> PolicyMigrationPlan {
    let predecessor = binding("1", b"fail-v1", Some(b"audit-v1"));
    let successor = binding("2", b"fail-v2", None);
    PolicyMigrationPlan {
        schema_version: POLICY_MIGRATION_SCHEMA.into(),
        predecessor,
        successor,
        activates_at_unix_s: 1_501,
        rollback_deadline_unix_s: 1_801,
        rationale: "temporarily waive audit while strengthening fail-closed".into(),
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

fn restoration_plan(restored_audit: &[u8]) -> PolicyMigrationPlan {
    let predecessor = binding("2", b"fail-v2", None);
    let successor = binding("3", b"fail-v2", Some(restored_audit));
    PolicyMigrationPlan {
        schema_version: POLICY_MIGRATION_SCHEMA.into(),
        predecessor,
        successor,
        activates_at_unix_s: 1_502,
        rollback_deadline_unix_s: 1_802,
        rationale: "restore audit anchor after temporary waiver".into(),
        migrations: vec![PolicyInvariantMigration {
            name: "fail-closed".into(),
            predecessor_digest: sha256(b"fail-v2"),
            successor_digest: Some(sha256(b"fail-v2")),
            disposition: PolicyInvariantDisposition::Retained,
        }],
    }
}

fn scalar_then_interval_ceremony(
    purpose: &str,
    payload: symthaea_fabrication_kernel::crypto_digest::Sha256Digest,
    basis: &OperationalClockBasisV1,
) -> symthaea_fabrication_trust_bridge::ClockGovernedThresholdCeremonyV1 {
    let trust = fabrication_trust();
    let threshold_policy = threshold_policy();
    let containment = containment_state();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    let (a, b) = providers();
    let approvals = vec![
        sign_threshold_approval(purpose, payload, 1_499, 1_505, &a).unwrap(),
        sign_threshold_approval(purpose, payload, 1_499, 1_505, &b).unwrap(),
    ];
    let legacy = verify_threshold_ceremony(
        purpose,
        payload,
        &approvals,
        &threshold_policy,
        &trust,
        1_500,
        &a,
    )
    .unwrap();
    qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &approvals,
        &threshold_policy,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap()
}

fn genesis(
    initial_policy: PolicyBinding,
    basis: &OperationalClockBasisV1,
) -> ClockGovernedPolicyLineageV1 {
    let trust = fabrication_trust();
    let threshold_policy = threshold_policy();
    let containment = containment_state();
    let prepared = prepare_clock_governed_policy_lineage_genesis_v1(
        initial_policy,
        &threshold_policy,
        &trust,
        &containment,
        basis,
    )
    .unwrap();
    let ceremony = scalar_then_interval_ceremony(
        CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        basis,
    );
    authorize_clock_governed_policy_lineage_genesis_v1(prepared, &ceremony).unwrap()
}

fn authorized_migration(
    plan: PolicyMigrationPlan,
    allow_waivers: bool,
    basis: &OperationalClockBasisV1,
) -> ClockGovernedPolicyMigrationV1 {
    let trust = fabrication_trust();
    let threshold_policy = threshold_policy();
    let containment = containment_state();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    let prepared = prepare_clock_governed_policy_migration_v1(
        plan,
        &migration_policy(allow_waivers),
        &threshold_policy,
        &trust,
        &containment,
        &envelope,
    )
    .unwrap();
    let ceremony = scalar_then_interval_ceremony(
        CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
        prepared.signing_payload_digest(),
        basis,
    );
    authorize_clock_governed_policy_migration_v1(prepared, &ceremony).unwrap()
}

fn permit(
    migration: &ClockGovernedPolicyMigrationV1,
    root: &OperationalClockBasisV1,
    successors: &[OperationalClockBasisV1],
) -> ClockGovernedPolicyActivationPermitV1 {
    derive_clock_governed_policy_activation_permit_v1(migration, root, successors).unwrap()
}

#[test]
fn threshold_genesis_and_activation_advance_exact_current_policy() {
    let root42 = authorization_basis();
    let p1 = binding("1", b"fail-v1", None);
    let p2 = binding("2", b"fail-v2", None);
    let lineage = genesis(p1.clone(), &root42);
    assert_eq!(lineage.sequence(), 1);
    assert_eq!(lineage.current_policy(), &p1);
    assert!(lineage.active_waivers().is_empty());
    assert_eq!(lineage.latest_operational_basis_id(), root42.id());

    let migration = authorized_migration(strengthened_plan(p1, p2.clone(), 1_501), false, &root42);
    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);
    let activation = permit(
        &migration,
        &root42,
        &[root43.clone(), root44.clone(), root45.clone()],
    );
    let next = advance_clock_governed_policy_lineage_v1(
        &lineage,
        &migration,
        &activation,
        &[root43, root44],
        &root45,
    )
    .unwrap();

    assert_eq!(next.sequence(), 2);
    assert_eq!(next.previous_lineage_id(), Some(lineage.id()));
    assert_eq!(next.current_policy(), &p2);
    assert!(next.active_waivers().is_empty());
    assert_eq!(next.last_migration_id(), Some(migration.id()));
    assert_eq!(next.last_activation_permit_id(), Some(activation.id()));
    assert_eq!(next.latest_operational_basis_id(), root45.id());
}

#[test]
fn waiver_state_is_current_not_historical_and_exact_restoration_resolves_it() {
    let root42 = authorization_basis();
    let initial = binding("1", b"fail-v1", Some(b"audit-v1"));
    let lineage1 = genesis(initial, &root42);
    let migration1 = authorized_migration(first_waiver_plan(), true, &root42);

    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);
    let activation1 = permit(
        &migration1,
        &root42,
        &[root43.clone(), root44.clone(), root45.clone()],
    );
    let lineage2 = advance_clock_governed_policy_lineage_v1(
        &lineage1,
        &migration1,
        &activation1,
        &[root43.clone(), root44.clone()],
        &root45,
    )
    .unwrap();
    assert_eq!(lineage2.active_waivers().len(), 1);
    let waiver = &lineage2.active_waivers()[0];
    assert_eq!(waiver.invariant(), "audit-anchor");
    assert_eq!(waiver.predecessor_digest(), sha256(b"audit-v1"));
    assert_eq!(waiver.originating_migration_id(), migration1.id());

    let migration2 = authorized_migration(restoration_plan(b"audit-v1"), false, &root42);
    let root46 = successor(&root45, 1_502_000);
    let root47 = successor(&root46, 1_502_500);
    let activation2 = permit(
        &migration2,
        &root42,
        &[
            root43,
            root44,
            root45.clone(),
            root46.clone(),
            root47.clone(),
        ],
    );
    let lineage3 = advance_clock_governed_policy_lineage_v1(
        &lineage2,
        &migration2,
        &activation2,
        &[root46],
        &root47,
    )
    .unwrap();

    assert_eq!(lineage3.sequence(), 3);
    assert!(lineage3.active_waivers().is_empty());
    assert!(
        lineage3
            .current_policy()
            .invariants
            .iter()
            .any(|invariant| invariant.name == "audit-anchor" && invariant.digest == sha256(b"audit-v1"))
    );
}

#[test]
fn differently_reintroduced_invariant_is_not_silently_called_waiver_restoration() {
    let root42 = authorization_basis();
    let lineage1 = genesis(binding("1", b"fail-v1", Some(b"audit-v1")), &root42);
    let migration1 = authorized_migration(first_waiver_plan(), true, &root42);
    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);
    let activation1 = permit(
        &migration1,
        &root42,
        &[root43.clone(), root44.clone(), root45.clone()],
    );
    let lineage2 = advance_clock_governed_policy_lineage_v1(
        &lineage1,
        &migration1,
        &activation1,
        &[root43.clone(), root44.clone()],
        &root45,
    )
    .unwrap();

    let migration2 = authorized_migration(restoration_plan(b"audit-v2"), false, &root42);
    let root46 = successor(&root45, 1_502_000);
    let root47 = successor(&root46, 1_502_500);
    let activation2 = permit(
        &migration2,
        &root42,
        &[root43, root44, root45.clone(), root46.clone(), root47.clone()],
    );
    let error = advance_clock_governed_policy_lineage_v1(
        &lineage2,
        &migration2,
        &activation2,
        &[root46],
        &root47,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ClockGovernedPolicyLineageError::WaiverRestorationDigestMismatch(
            "audit-anchor".into()
        )
    );
}

#[test]
fn currentness_clock_cannot_skip_a_generation_even_with_valid_activation_permit() {
    let root42 = authorization_basis();
    let p1 = binding("1", b"fail-v1", None);
    let p2 = binding("2", b"fail-v2", None);
    let lineage1 = genesis(p1.clone(), &root42);
    let migration1 = authorized_migration(strengthened_plan(p1, p2, 1_501), false, &root42);
    let root43 = successor(&root42, 1_500_500);
    let root44 = successor(&root43, 1_501_000);
    let root45 = successor(&root44, 1_501_500);
    let activation1 = permit(
        &migration1,
        &root42,
        &[root43, root44, root45.clone()],
    );

    let error = advance_clock_governed_policy_lineage_v1(
        &lineage1,
        &migration1,
        &activation1,
        &[],
        &root45,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ClockGovernedPolicyLineageError::BrokenCurrentClockLineage { .. }
    ));
}

#[test]
fn live_lineage_surface_excludes_legacy_tracker_scalar_time_and_deserialization() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("AuthorizedPolicyMigration"));
    assert!(!source.contains("PolicyMigrationTracker"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedPolicyLineageV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct PreparedPolicyLineageGenesisV1"
    ));
}
