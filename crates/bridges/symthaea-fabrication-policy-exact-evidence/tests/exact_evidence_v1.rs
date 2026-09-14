use std::cell::Cell;
use std::collections::BTreeSet;

use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::{Sha256Digest, sha256};
use symthaea_fabrication_kernel::policy_migration::{PolicyBinding, PolicyInvariantBinding};
use symthaea_fabrication_kernel::threshold::{
    ThresholdApprovalSigner, ThresholdApprovalVerifier, ThresholdCeremonyPolicy,
    sign_threshold_approval, verify_threshold_ceremony,
};
use symthaea_fabrication_kernel::transparency::TransparencyLog;
use symthaea_fabrication_kernel::transparency_checkpoint::{
    SignedTransparencyCheckpoint, TransparencyCheckpointSigner, TransparencyCheckpointVerifier,
    sign_transparency_checkpoint, verify_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    SignedTransparencyWitness, TransparencyWitnessPolicy, TransparencyWitnessSigner,
    TransparencyWitnessVerifier, sign_transparency_witness, verify_transparency_witness_quorum,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_policy_exact_evidence::{
    ExactEvidenceBindingError, ExactEvidenceVerificationPolicyV1,
    ExactPolicyHeadEvidenceVerifierV1, bind_exact_policy_head_signature_evidence_v1,
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
    RegistryBoundPolicyHeadV1, WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE,
    WitnessAuthorityProfileV1, authorize_witness_authority_registry_genesis_v1,
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

struct SignerProvider {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

impl ThresholdApprovalSigner for SignerProvider {
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
    fn sign_threshold_approval(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}
impl ThresholdApprovalVerifier for SignerProvider {
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
impl TransparencyCheckpointSigner for SignerProvider {
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
    fn sign_transparency_checkpoint(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}
impl TransparencyCheckpointVerifier for SignerProvider {
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
impl TransparencyWitnessSigner for SignerProvider {
    fn algorithm(&self) -> SignatureAlgorithm { self.algorithm.clone() }
    fn key_id(&self) -> &str { self.key_id }
    fn sign_transparency_witness(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}
impl TransparencyWitnessVerifier for SignerProvider {
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

struct ExactVerifier {
    provider_id: &'static str,
}
impl ExactPolicyHeadEvidenceVerifierV1 for ExactVerifier {
    fn provider_id(&self) -> &str { self.provider_id }
    fn verification_policy_digest(&self) -> Sha256Digest {
        sha256(format!("exact-verifier-policy:{}", self.provider_id).as_bytes())
    }
    fn verify_checkpoint_signature(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
    fn verify_witness_signature(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

fn fabrication_key(
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

fn fabrication_trust() -> TrustSnapshot {
    TrustSnapshot::new(
        12,
        1_000,
        2_000,
        vec![
            fabrication_key(
                SignatureAlgorithm::Ed25519,
                "admin-a",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::ThresholdCeremony]),
            ),
            fabrication_key(
                SignatureAlgorithm::MlDsa65,
                "admin-b",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::ThresholdCeremony]),
            ),
            fabrication_key(
                SignatureAlgorithm::Ed25519,
                "log",
                BTreeSet::from([KeyUsage::TransparencyLog]),
            ),
            fabrication_key(
                SignatureAlgorithm::Ed25519,
                "witness-a",
                BTreeSet::from([KeyUsage::TransparencyWitness]),
            ),
            fabrication_key(
                SignatureAlgorithm::MlDsa65,
                "witness-b",
                BTreeSet::from([KeyUsage::TransparencyWitness]),
            ),
        ],
    )
    .unwrap()
}

fn containment() -> FabricationContainmentState {
    FabricationContainmentState::genesis(1, sha256(b"resilience-v1")).unwrap()
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

fn admin_providers() -> (SignerProvider, SignerProvider) {
    (
        SignerProvider { algorithm: SignatureAlgorithm::Ed25519, key_id: "admin-a" },
        SignerProvider { algorithm: SignatureAlgorithm::MlDsa65, key_id: "admin-b" },
    )
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
fn operational_basis() -> OperationalClockBasisV1 {
    let snapshot = clock_snapshot();
    let quorum = ClockQuorumPolicyRevisionV1::new(2, 8, 5_000, 10_000, true).unwrap();
    let continuity = ClockContinuityPolicyRevisionV1::new(1, 10_000, 60_000, 1, true).unwrap();
    let evaluation = ClockEvaluationPolicyV4::new(
        repeated_clock_hex('b'), &quorum, &continuity, 2_000, 2, true,
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
        &claim, "platform-root-01", repeated_clock_hex('b'), repeated_clock_hex('c'),
    )
    .unwrap();
    let authority = verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
    let permit = derive_bootstrap_clock_evaluation_permit_v4(
        &authority, &claim, &evaluation, &quorum, &continuity, &snapshot,
    )
    .unwrap();
    let observations = vec![
        clock_observation("source-b", 1_500_040, 120, ClockSignatureAlgorithm::MlDsa65, "clock-b"),
        clock_observation("source-a", 1_500_000, 100, ClockSignatureAlgorithm::Ed25519, "clock-a"),
    ];
    let verifier = ObservationVerifier { calls: Cell::new(0) };
    let accepted = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier).unwrap();
    assert_eq!(verifier.calls.get(), 2);
    bind_bootstrap_operational_clock_basis_v1(&accepted, &evaluation, &snapshot).unwrap()
}

fn qualify_threshold(
    purpose: &str,
    payload: Sha256Digest,
    policy: &ThresholdCeremonyPolicy,
    basis: &OperationalClockBasisV1,
) -> symthaea_fabrication_trust_bridge::ClockGovernedThresholdCeremonyV1 {
    let trust = fabrication_trust();
    let containment = containment();
    let (a, b) = admin_providers();
    let approvals = vec![
        sign_threshold_approval(purpose, payload, 1_499, 1_501, &a).unwrap(),
        sign_threshold_approval(purpose, payload, 1_499, 1_501, &b).unwrap(),
    ];
    let legacy = verify_threshold_ceremony(
        purpose, payload, &approvals, policy, &trust, 1_500, &a,
    )
    .unwrap();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &approvals,
        policy,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap()
}

fn lineage_and_temporal(
    basis: &OperationalClockBasisV1,
) -> (ClockGovernedPolicyLineageV1, ClockGovernedPolicyTemporalValidityPermitV1) {
    let policy = PolicyBinding::new(
        "upgrade-authority",
        "1",
        sha256(b"policy-v1"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"fail-closed-v1"),
        }],
    )
    .unwrap();
    let threshold_policy = policy_threshold();
    let prepared = prepare_clock_governed_policy_lineage_genesis_v1(
        policy, &threshold_policy, &fabrication_trust(), &containment(), basis,
    )
    .unwrap();
    let threshold = qualify_threshold(
        CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        &threshold_policy,
        basis,
    );
    let lineage = authorize_clock_governed_policy_lineage_genesis_v1(prepared, &threshold).unwrap();
    let temporal = derive_clock_governed_policy_temporal_validity_permit_v1(&lineage, &[], basis).unwrap();
    (lineage, temporal)
}

fn witness_registry(
    basis: &OperationalClockBasisV1,
) -> symthaea_fabrication_witness_authority::WitnessAuthorityRegistryV1 {
    let threshold_policy = registry_threshold();
    let prepared = prepare_witness_authority_registry_genesis_v1(
        vec![
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
        ],
        &threshold_policy,
        &fabrication_trust(),
        &containment(),
        basis,
    )
    .unwrap();
    let threshold = qualify_threshold(
        WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        &threshold_policy,
        basis,
    );
    authorize_witness_authority_registry_genesis_v1(prepared, &threshold).unwrap()
}

struct HeadFixture {
    observed: QuorumObservedPolicyHeadV1,
    registry_bound: RegistryBoundPolicyHeadV1,
    checkpoint: SignedTransparencyCheckpoint,
    witnesses: Vec<SignedTransparencyWitness>,
}

fn head_fixture(mutate_checkpoint_signature: bool, mutate_witness_signature: bool) -> HeadFixture {
    let basis = operational_basis();
    let (lineage, temporal) = lineage_and_temporal(&basis);
    let registry = witness_registry(&basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let mut log = TransparencyLog::default();
    log.append(
        1_499,
        policy_lineage_head_log_kind(lineage.domain()).unwrap(),
        digest_policy_lineage_head_publication_v1(&publication).unwrap(),
    )
    .unwrap();

    let log_signer = SignerProvider { algorithm: SignatureAlgorithm::Ed25519, key_id: "log" };
    let witness_a = SignerProvider { algorithm: SignatureAlgorithm::Ed25519, key_id: "witness-a" };
    let witness_b = SignerProvider { algorithm: SignatureAlgorithm::MlDsa65, key_id: "witness-b" };
    let correct_checkpoint = sign_transparency_checkpoint(&log, None, 1_499, 1_501, &log_signer).unwrap();
    let verified_checkpoint = verify_transparency_checkpoint(
        &correct_checkpoint, &log, &fabrication_trust(), 1_500, &log_signer,
    )
    .unwrap();
    let correct_witnesses = vec![
        sign_transparency_witness(&verified_checkpoint, "org-a", "region-a", 1_499, &witness_a).unwrap(),
        sign_transparency_witness(&verified_checkpoint, "org-b", "region-b", 1_499, &witness_b).unwrap(),
    ];
    let witness_policy = TransparencyWitnessPolicy::default();
    let verified_witnesses = verify_transparency_witness_quorum(
        &verified_checkpoint,
        &correct_witnesses,
        &witness_policy,
        &fabrication_trust(),
        1_500,
        &witness_a,
    )
    .unwrap();

    // The upstream opaque capabilities prove the logical checkpoint/statements. We intentionally
    // substitute raw signature bytes only after those opaque proofs were produced.
    let mut checkpoint = correct_checkpoint;
    if mutate_checkpoint_signature {
        checkpoint.signature.signature = vec![9; 32];
    }
    let mut witnesses = correct_witnesses;
    if mutate_witness_signature {
        witnesses[0].signature.signature = vec![8; 32];
    }

    let observed = qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        &basis,
        &publication,
        &log,
        &checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &fabrication_trust(),
        &containment(),
    )
    .unwrap();
    let registry_bound = bind_quorum_observed_policy_head_to_witness_registry_v1(
        &observed,
        &witnesses,
        &registry,
    )
    .unwrap();

    HeadFixture { observed, registry_bound, checkpoint, witnesses }
}

#[test]
fn exact_raw_signatures_are_reverified_by_two_committed_providers() {
    let fixture = head_fixture(false, false);
    let provider_a = ExactVerifier { provider_id: "verifier-a" };
    let provider_b = ExactVerifier { provider_id: "verifier-b" };
    let providers: [&dyn ExactPolicyHeadEvidenceVerifierV1; 2] = [&provider_a, &provider_b];
    let bound = bind_exact_policy_head_signature_evidence_v1(
        &fixture.registry_bound,
        &fixture.observed,
        &fixture.checkpoint,
        &fixture.witnesses,
        &ExactEvidenceVerificationPolicyV1::default(),
        &providers,
    )
    .unwrap();

    assert_eq!(bound.observed_head_id(), fixture.observed.id());
    assert_eq!(bound.registry_bound_head_id(), fixture.registry_bound.id());
    assert_eq!(bound.verifier_count(), 2);
    assert_eq!(bound.witness_count(), 2);
}

#[test]
fn upstream_logical_witness_proof_cannot_launder_different_raw_signature_bytes() {
    let fixture = head_fixture(false, true);
    let provider_a = ExactVerifier { provider_id: "verifier-a" };
    let provider_b = ExactVerifier { provider_id: "verifier-b" };
    let providers: [&dyn ExactPolicyHeadEvidenceVerifierV1; 2] = [&provider_a, &provider_b];
    let errors = bind_exact_policy_head_signature_evidence_v1(
        &fixture.registry_bound,
        &fixture.observed,
        &fixture.checkpoint,
        &fixture.witnesses,
        &ExactEvidenceVerificationPolicyV1::default(),
        &providers,
    )
    .unwrap_err();

    assert!(errors.iter().any(|error| matches!(
        error,
        ExactEvidenceBindingError::WitnessSignatureRejected { key_id, .. }
            if key_id == "witness-a"
    )));
}

#[test]
fn upstream_logical_checkpoint_proof_cannot_launder_different_raw_signature_bytes() {
    let fixture = head_fixture(true, false);
    let provider_a = ExactVerifier { provider_id: "verifier-a" };
    let provider_b = ExactVerifier { provider_id: "verifier-b" };
    let providers: [&dyn ExactPolicyHeadEvidenceVerifierV1; 2] = [&provider_a, &provider_b];
    let errors = bind_exact_policy_head_signature_evidence_v1(
        &fixture.registry_bound,
        &fixture.observed,
        &fixture.checkpoint,
        &fixture.witnesses,
        &ExactEvidenceVerificationPolicyV1::default(),
        &providers,
    )
    .unwrap_err();

    assert!(errors.iter().any(|error| matches!(
        error,
        ExactEvidenceBindingError::CheckpointSignatureRejected(provider)
            if provider == "verifier-a" || provider == "verifier-b"
    )));
}

#[test]
fn default_policy_requires_two_distinct_exact_evidence_verifiers() {
    let fixture = head_fixture(false, false);
    let provider_a = ExactVerifier { provider_id: "verifier-a" };
    let providers: [&dyn ExactPolicyHeadEvidenceVerifierV1; 1] = [&provider_a];
    let errors = bind_exact_policy_head_signature_evidence_v1(
        &fixture.registry_bound,
        &fixture.observed,
        &fixture.checkpoint,
        &fixture.witnesses,
        &ExactEvidenceVerificationPolicyV1::default(),
        &providers,
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| matches!(
        error,
        ExactEvidenceBindingError::InsufficientProviders { actual: 1, required: 2 }
    )));
}

#[test]
fn exact_evidence_capability_is_not_deserializable() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ExactEvidenceBoundPolicyHeadV1"
    ));
    assert!(source.contains("minimum_distinct_providers: 2"));
}
