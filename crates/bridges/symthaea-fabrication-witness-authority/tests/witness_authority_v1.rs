use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::sha256;
use symthaea_fabrication_kernel::policy_migration::{PolicyBinding, PolicyInvariantBinding};
use symthaea_fabrication_kernel::threshold::{
    ThresholdApprovalSigner, ThresholdApprovalVerifier, ThresholdCeremonyPolicy,
    sign_threshold_approval, verify_threshold_ceremony,
};
use symthaea_fabrication_kernel::transparency::TransparencyLog;
use symthaea_fabrication_kernel::transparency_checkpoint::{
    TransparencyCheckpointSigner, TransparencyCheckpointVerifier, sign_transparency_checkpoint,
    verify_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    SignedTransparencyWitness, TransparencyWitnessPolicy, TransparencyWitnessSigner,
    TransparencyWitnessVerifier, sign_transparency_witness, verify_transparency_witness_quorum,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_policy_head_observation::{
    QuorumObservedPolicyHeadV1, build_policy_lineage_head_publication_v1,
    digest_policy_lineage_head_publication_v1, policy_lineage_head_log_kind,
    qualify_quorum_observed_policy_head_v1,
};
use symthaea_fabrication_policy_lineage::{
    CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE, ClockGovernedPolicyLineageV1,
    authorize_clock_governed_policy_lineage_genesis_v1,
    prepare_clock_governed_policy_lineage_genesis_v1,
};
use symthaea_fabrication_policy_temporal_validity::{
    ClockGovernedPolicyTemporalValidityPermitV1,
    derive_clock_governed_policy_temporal_validity_permit_v1,
};
use symthaea_fabrication_trust_bridge::qualify_clock_governed_threshold_ceremony_v1;
use symthaea_fabrication_witness_authority::{
    WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE, WitnessAuthorityError,
    WitnessAuthorityProfileV1, WitnessAuthorityRegistryV1,
    authorize_witness_authority_registry_genesis_v1,
    bind_quorum_observed_policy_head_to_witness_registry_v1,
    prepare_witness_authority_registry_genesis_v1,
};
use symthaea_trust_kernel::{
    CLOCK_OBSERVATION_SCHEMA, ClockBootstrapAuthorityEvidenceV2,
    ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPolicyV4, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicyRevisionV1, DetachedSignature,
    KeyLifecycleStatus as ClockKeyLifecycleStatus, KeyTrustRecord as ClockKeyTrustRecord,
    KeyUsage as ClockKeyUsage, OperationalClockBasisV1, Sha256Digest as ClockSha256Digest,
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
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
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
impl TransparencyCheckpointSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
    fn sign_transparency_checkpoint(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}
impl TransparencyCheckpointVerifier for Provider {
    fn verify_transparency_checkpoint(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}
impl TransparencyWitnessSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
    fn sign_transparency_witness(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}
impl TransparencyWitnessVerifier for Provider {
    fn verify_transparency_witness(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

fn key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    usages: BTreeSet<KeyUsage>,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.into(),
        not_before_unix_s: 1_000,
        not_after_unix_s: None,
        status: KeyLifecycleStatus::Active,
        usages,
    }
}

fn trust() -> TrustSnapshot {
    TrustSnapshot::new(
        11,
        1_000,
        2_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "admin-a",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::ThresholdCeremony]),
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "admin-b",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::ThresholdCeremony]),
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "log",
                BTreeSet::from([KeyUsage::TransparencyLog]),
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "witness-a",
                BTreeSet::from([KeyUsage::TransparencyWitness]),
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "witness-b",
                BTreeSet::from([KeyUsage::TransparencyWitness]),
            ),
        ],
    )
    .unwrap()
}

fn policy_threshold() -> ThresholdCeremonyPolicy {
    ThresholdCeremonyPolicy {
        key_usage: KeyUsage::PolicyMigration,
        allowed_key_ids: Some(BTreeSet::from(["admin-a".into(), "admin-b".into()])),
        ..ThresholdCeremonyPolicy::default()
    }
}

fn registry_threshold() -> ThresholdCeremonyPolicy {
    ThresholdCeremonyPolicy {
        key_usage: KeyUsage::ThresholdCeremony,
        allowed_key_ids: Some(BTreeSet::from(["admin-a".into(), "admin-b".into()])),
        ..ThresholdCeremonyPolicy::default()
    }
}

fn containment() -> FabricationContainmentState {
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
    fn provider_id(&self) -> &str { "platform-root-01" }
    fn authority_policy_digest(&self) -> ClockSha256Digest { repeated_clock_hex('b') }
    fn verify_clock_bootstrap_authority(
        &self,
        _canonical_claim_bytes: &[u8],
        external_evidence_digest: ClockSha256Digest,
    ) -> Result<bool, String> {
        Ok(external_evidence_digest == repeated_clock_hex('c'))
    }
}

struct ObservationVerifier { calls: Cell<usize> }
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
        schema_version: CLOCK_OBSERVATION_SCHEMA.into(),
        source_id: source_id.into(),
        observed_unix_ms,
        uncertainty_ms,
        epoch: 42,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.into(),
            signature: vec![1, 2, 3],
        },
    }
}

fn basis() -> OperationalClockBasisV1 {
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
    let verifier = ObservationVerifier { calls: Cell::new(0) };
    let accepted = accept_bootstrap_clock_basis_v5(
        &permit,
        &observations,
        &snapshot,
        &verifier,
    )
    .unwrap();
    assert_eq!(verifier.calls.get(), 2);
    bind_bootstrap_operational_clock_basis_v1(&accepted, &evaluation, &snapshot).unwrap()
}

fn admin_providers() -> (Provider, Provider) {
    (
        Provider { algorithm: SignatureAlgorithm::Ed25519, key_id: "admin-a" },
        Provider { algorithm: SignatureAlgorithm::MlDsa65, key_id: "admin-b" },
    )
}

fn qualify_threshold(
    purpose: &str,
    payload: symthaea_fabrication_kernel::crypto_digest::Sha256Digest,
    threshold_policy: &ThresholdCeremonyPolicy,
    basis: &OperationalClockBasisV1,
) -> symthaea_fabrication_trust_bridge::ClockGovernedThresholdCeremonyV1 {
    let trust = trust();
    let containment = containment();
    let (a, b) = admin_providers();
    let approvals = vec![
        sign_threshold_approval(purpose, payload, 1_499, 1_501, &a).unwrap(),
        sign_threshold_approval(purpose, payload, 1_499, 1_501, &b).unwrap(),
    ];
    let legacy = verify_threshold_ceremony(
        purpose,
        payload,
        &approvals,
        threshold_policy,
        &trust,
        1_500,
        &a,
    )
    .unwrap();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &approvals,
        threshold_policy,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap()
}

fn lineage_and_temporal(
    basis: &OperationalClockBasisV1,
) -> (ClockGovernedPolicyLineageV1, ClockGovernedPolicyTemporalValidityPermitV1) {
    let prepared = prepare_clock_governed_policy_lineage_genesis_v1(
        PolicyBinding::new(
            "upgrade-authority",
            "1",
            sha256(b"policy-v1"),
            vec![PolicyInvariantBinding {
                name: "fail-closed".into(),
                digest: sha256(b"fail-closed-v1"),
            }],
        )
        .unwrap(),
        &policy_threshold(),
        &trust(),
        &containment(),
        basis,
    )
    .unwrap();
    let threshold = qualify_threshold(
        CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        &policy_threshold(),
        basis,
    );
    let lineage = authorize_clock_governed_policy_lineage_genesis_v1(prepared, &threshold).unwrap();
    let temporal = derive_clock_governed_policy_temporal_validity_permit_v1(
        &lineage,
        &[],
        basis,
    )
    .unwrap();
    (lineage, temporal)
}

fn registry(basis: &OperationalClockBasisV1) -> WitnessAuthorityRegistryV1 {
    let profiles = vec![
        WitnessAuthorityProfileV1 {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "witness-a".into(),
            organization: "org-a".into(),
            failure_domain: "region-a".into(),
        },
        WitnessAuthorityProfileV1 {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "witness-b".into(),
            organization: "org-b".into(),
            failure_domain: "region-b".into(),
        },
    ];
    let prepared = prepare_witness_authority_registry_genesis_v1(
        profiles,
        &registry_threshold(),
        &trust(),
        &containment(),
        basis,
    )
    .unwrap();
    let threshold = qualify_threshold(
        WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        &registry_threshold(),
        basis,
    );
    authorize_witness_authority_registry_genesis_v1(prepared, &threshold).unwrap()
}

fn observed_head(
    basis: &OperationalClockBasisV1,
    witness_a_organization: &str,
) -> (QuorumObservedPolicyHeadV1, Vec<SignedTransparencyWitness>) {
    let (lineage, temporal) = lineage_and_temporal(basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let mut log = TransparencyLog::default();
    log.append(
        1_499,
        policy_lineage_head_log_kind(lineage.domain()).unwrap(),
        digest_policy_lineage_head_publication_v1(&publication).unwrap(),
    )
    .unwrap();
    let log_provider = Provider { algorithm: SignatureAlgorithm::Ed25519, key_id: "log" };
    let witness_a = Provider { algorithm: SignatureAlgorithm::Ed25519, key_id: "witness-a" };
    let witness_b = Provider { algorithm: SignatureAlgorithm::MlDsa65, key_id: "witness-b" };
    let signed_checkpoint = sign_transparency_checkpoint(&log, None, 1_499, 1_501, &log_provider).unwrap();
    let verified_checkpoint = verify_transparency_checkpoint(
        &signed_checkpoint,
        &log,
        &trust(),
        1_500,
        &log_provider,
    )
    .unwrap();
    let witnesses = vec![
        sign_transparency_witness(
            &verified_checkpoint,
            witness_a_organization,
            "region-a",
            1_499,
            &witness_a,
        )
        .unwrap(),
        sign_transparency_witness(
            &verified_checkpoint,
            "org-b",
            "region-b",
            1_499,
            &witness_b,
        )
        .unwrap(),
    ];
    let witness_policy = TransparencyWitnessPolicy::default();
    let verified_witnesses = verify_transparency_witness_quorum(
        &verified_checkpoint,
        &witnesses,
        &witness_policy,
        &trust(),
        1_500,
        &witness_a,
    )
    .unwrap();
    let observed = qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        basis,
        &publication,
        &log,
        &signed_checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &trust(),
        &containment(),
    )
    .unwrap();
    (observed, witnesses)
}

#[test]
fn governed_registry_upgrades_witness_declared_diversity_to_bound_identity() {
    let basis = basis();
    let registry = registry(&basis);
    let (observed, witnesses) = observed_head(&basis, "org-a");
    let bound = bind_quorum_observed_policy_head_to_witness_registry_v1(
        &observed,
        &witnesses,
        &registry,
    )
    .unwrap();

    assert_eq!(bound.observed_head_id(), observed.id());
    assert_eq!(bound.registry_id(), registry.id());
    assert_eq!(bound.witness_count(), 2);
    assert_eq!(bound.organization_count(), 2);
    assert_eq!(bound.failure_domain_count(), 2);
}

#[test]
fn self_signed_organization_claim_can_pass_base_quorum_but_not_governed_registry() {
    let basis = basis();
    let registry = registry(&basis);

    // The base witness theorem accepts this because organization is part of the witness-signed
    // statement. The stronger layer must resolve the signing key through the governed registry.
    let (observed, witnesses) = observed_head(&basis, "claimed-other-org");
    let errors = bind_quorum_observed_policy_head_to_witness_registry_v1(
        &observed,
        &witnesses,
        &registry,
    )
    .unwrap_err();

    assert!(errors.iter().any(|error| matches!(
        error,
        WitnessAuthorityError::WitnessOrganizationMismatch(key_id) if key_id == "witness-a"
    )));
}

#[test]
fn unregistered_witness_is_rejected_even_when_signed_witness_set_is_exact() {
    let basis = basis();
    let registry = registry(&basis);
    let (observed, mut witnesses) = observed_head(&basis, "org-a");
    witnesses[0].signature.key_id = "unknown".into();

    let errors = bind_quorum_observed_policy_head_to_witness_registry_v1(
        &observed,
        &witnesses,
        &registry,
    )
    .unwrap_err();
    // Exact signed evidence no longer matches #2829 before registry lookup can confer authority.
    assert!(errors.iter().any(|error| matches!(
        error,
        WitnessAuthorityError::WitnessSetMismatch
    )));
}

#[test]
fn registry_and_bound_head_have_no_deserialization_authority_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct WitnessAuthorityRegistryV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct RegistryBoundPolicyHeadV1"
    ));
    let start = source
        .find("pub fn prepare_witness_authority_registry_genesis_v1")
        .unwrap();
    let signature = &source[start..source[start..].find(") -> Result").unwrap() + start + 1];
    assert!(!signature.contains("now_unix_s"));
    assert!(signature.contains("OperationalClockBasisV1"));
}
