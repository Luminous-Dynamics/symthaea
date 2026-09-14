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
    TransparencyCheckpointSigner, TransparencyCheckpointVerifier, VerifiedTransparencyCheckpoint,
    sign_transparency_checkpoint, verify_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    SignedTransparencyWitness, TransparencyWitnessPolicy, TransparencyWitnessSigner,
    TransparencyWitnessVerifier, VerifiedTransparencyWitnessQuorum, sign_transparency_witness,
    verify_transparency_witness_quorum,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_policy_head_observation::{
    PolicyHeadObservationError, build_policy_lineage_head_publication_v1,
    digest_policy_lineage_head_publication_v1, policy_lineage_head_log_kind,
    qualify_quorum_observed_policy_head_v1,
};
use symthaea_fabrication_policy_lineage::{
    CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
    authorize_clock_governed_policy_lineage_genesis_v1,
    prepare_clock_governed_policy_lineage_genesis_v1,
};
use symthaea_fabrication_policy_temporal_validity::{
    ClockGovernedPolicyTemporalValidityPermitV1,
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

impl TransparencyCheckpointSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
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
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
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
        9,
        1_000,
        2_000,
        vec![
            fabrication_key(
                SignatureAlgorithm::Ed25519,
                "log",
                BTreeSet::from([KeyUsage::TransparencyLog]),
            ),
            fabrication_key(
                SignatureAlgorithm::Ed25519,
                "a",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::TransparencyWitness]),
            ),
            fabrication_key(
                SignatureAlgorithm::MlDsa65,
                "b",
                BTreeSet::from([KeyUsage::PolicyMigration, KeyUsage::TransparencyWitness]),
            ),
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

fn initial_policy() -> PolicyBinding {
    PolicyBinding::new(
        "upgrade-authority",
        "1",
        sha256(b"policy-v1"),
        vec![PolicyInvariantBinding {
            name: "fail-closed".into(),
            digest: sha256(b"fail-closed-v1"),
        }],
    )
    .unwrap()
}

fn containment_state() -> FabricationContainmentState {
    FabricationContainmentState::genesis(1, sha256(b"resilience-v1")).unwrap()
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

fn operational_basis() -> OperationalClockBasisV1 {
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
    let accepted =
        accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier).unwrap();
    assert_eq!(verifier.calls.get(), 2);
    bind_bootstrap_operational_clock_basis_v1(&accepted, &evaluation, &snapshot).unwrap()
}

fn lineage_and_temporal(
    basis: &OperationalClockBasisV1,
) -> (
    symthaea_fabrication_policy_lineage::ClockGovernedPolicyLineageV1,
    ClockGovernedPolicyTemporalValidityPermitV1,
) {
    let trust = fabrication_trust();
    let containment = containment_state();
    let threshold_policy = threshold_policy();
    let prepared = prepare_clock_governed_policy_lineage_genesis_v1(
        initial_policy(),
        &threshold_policy,
        &trust,
        &containment,
        basis,
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
    let approvals = vec![
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
            prepared.signing_payload_digest(),
            1_499,
            1_501,
            &a,
        )
        .unwrap(),
        sign_threshold_approval(
            CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
            prepared.signing_payload_digest(),
            1_499,
            1_501,
            &b,
        )
        .unwrap(),
    ];
    let legacy = verify_threshold_ceremony(
        CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        prepared.signing_payload_digest(),
        &approvals,
        &threshold_policy,
        &trust,
        1_500,
        &a,
    )
    .unwrap();
    let envelope = derive_clock_governance_evaluation_envelope_v1(basis).unwrap();
    let threshold = qualify_clock_governed_threshold_ceremony_v1(
        &legacy,
        &approvals,
        &threshold_policy,
        &trust,
        &containment.signer_compromise_tracker,
        &envelope,
    )
    .unwrap();
    let lineage = authorize_clock_governed_policy_lineage_genesis_v1(prepared, &threshold).unwrap();
    let temporal = derive_clock_governed_policy_temporal_validity_permit_v1(
        &lineage,
        &[],
        basis,
    )
    .unwrap();
    (lineage, temporal)
}

fn witnessed_checkpoint(
    log: &TransparencyLog,
    trust: &TrustSnapshot,
    checkpoint_issued: u64,
    checkpoint_expires: u64,
    scalar_verify_time: u64,
) -> (
    symthaea_fabrication_kernel::transparency_checkpoint::SignedTransparencyCheckpoint,
    VerifiedTransparencyCheckpoint,
    Vec<SignedTransparencyWitness>,
    VerifiedTransparencyWitnessQuorum,
    TransparencyWitnessPolicy,
) {
    let log_provider = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "log",
    };
    let a = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "a",
    };
    let b = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "b",
    };
    let signed_checkpoint = sign_transparency_checkpoint(
        log,
        None,
        checkpoint_issued,
        checkpoint_expires,
        &log_provider,
    )
    .unwrap();
    let verified_checkpoint = verify_transparency_checkpoint(
        &signed_checkpoint,
        log,
        trust,
        scalar_verify_time,
        &log_provider,
    )
    .unwrap();
    let witnesses = vec![
        sign_transparency_witness(
            &verified_checkpoint,
            "org-a",
            "region-a",
            1_499,
            &a,
        )
        .unwrap(),
        sign_transparency_witness(
            &verified_checkpoint,
            "org-b",
            "region-b",
            1_499,
            &b,
        )
        .unwrap(),
    ];
    let policy = TransparencyWitnessPolicy::default();
    let verified_witnesses = verify_transparency_witness_quorum(
        &verified_checkpoint,
        &witnesses,
        &policy,
        trust,
        scalar_verify_time,
        &a,
    )
    .unwrap();
    (
        signed_checkpoint,
        verified_checkpoint,
        witnesses,
        verified_witnesses,
        policy,
    )
}

#[test]
fn exact_lineage_head_is_bound_to_latest_witnessed_checkpoint_view() {
    let basis = operational_basis();
    let envelope = derive_clock_governance_evaluation_envelope_v1(&basis).unwrap();
    assert_eq!(envelope.lower_unix_ms(), 1_499_920);
    assert_eq!(envelope.upper_unix_ms(), 1_500_100);
    let (lineage, temporal) = lineage_and_temporal(&basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let publication_digest = digest_policy_lineage_head_publication_v1(&publication).unwrap();
    let mut log = TransparencyLog::default();
    log.append(
        1_499,
        policy_lineage_head_log_kind(lineage.domain()).unwrap(),
        publication_digest,
    )
    .unwrap();
    let trust = fabrication_trust();
    let containment = containment_state();
    let (signed_checkpoint, verified_checkpoint, witnesses, verified_witnesses, witness_policy) =
        witnessed_checkpoint(&log, &trust, 1_499, 1_501, 1_500);

    let observed = qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        &basis,
        &publication,
        &log,
        &signed_checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &trust,
        &containment,
    )
    .unwrap();

    assert_eq!(observed.lineage_id(), lineage.id());
    assert_eq!(observed.lineage_sequence(), 1);
    assert_eq!(observed.transparency_log_size(), 1);
    assert_eq!(observed.publication_entry_sequence(), 1);
}

#[test]
fn later_same_domain_head_in_same_checkpoint_view_supersedes_old_head() {
    let basis = operational_basis();
    let (lineage, temporal) = lineage_and_temporal(&basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let mut log = TransparencyLog::default();
    let kind = policy_lineage_head_log_kind(lineage.domain()).unwrap();
    log.append(
        1_499,
        kind.clone(),
        digest_policy_lineage_head_publication_v1(&publication).unwrap(),
    )
    .unwrap();
    log.append(1_499, kind, sha256(b"newer-policy-head")).unwrap();
    let trust = fabrication_trust();
    let containment = containment_state();
    let (signed_checkpoint, verified_checkpoint, witnesses, verified_witnesses, witness_policy) =
        witnessed_checkpoint(&log, &trust, 1_499, 1_501, 1_500);

    let errors = qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        &basis,
        &publication,
        &log,
        &signed_checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &trust,
        &containment,
    )
    .unwrap_err();

    assert!(errors.iter().any(|error| matches!(
        error,
        PolicyHeadObservationError::PublicationNotLatestInCheckpointView
    )));
}

#[test]
fn scalar_valid_checkpoint_expiring_inside_clock_uncertainty_is_rejected() {
    let basis = operational_basis();
    let (lineage, temporal) = lineage_and_temporal(&basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let mut log = TransparencyLog::default();
    log.append(
        1_499,
        policy_lineage_head_log_kind(lineage.domain()).unwrap(),
        digest_policy_lineage_head_publication_v1(&publication).unwrap(),
    )
    .unwrap();
    let trust = fabrication_trust();
    let containment = containment_state();
    let (signed_checkpoint, verified_checkpoint, witnesses, verified_witnesses, witness_policy) =
        witnessed_checkpoint(&log, &trust, 1_498, 1_500, 1_499);

    let errors = qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        &basis,
        &publication,
        &log,
        &signed_checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &trust,
        &containment,
    )
    .unwrap_err();

    assert!(errors.iter().any(|error| matches!(
        error,
        PolicyHeadObservationError::CheckpointInvalid(_)
    )));
}

#[test]
fn unrelated_domain_publication_does_not_supersede_this_domain() {
    let basis = operational_basis();
    let (lineage, temporal) = lineage_and_temporal(&basis);
    let publication = build_policy_lineage_head_publication_v1(&lineage, &temporal).unwrap();
    let mut log = TransparencyLog::default();
    log.append(
        1_499,
        policy_lineage_head_log_kind(lineage.domain()).unwrap(),
        digest_policy_lineage_head_publication_v1(&publication).unwrap(),
    )
    .unwrap();
    log.append(
        1_499,
        policy_lineage_head_log_kind("other-domain").unwrap(),
        sha256(b"other-head"),
    )
    .unwrap();
    let trust = fabrication_trust();
    let containment = containment_state();
    let (signed_checkpoint, verified_checkpoint, witnesses, verified_witnesses, witness_policy) =
        witnessed_checkpoint(&log, &trust, 1_499, 1_501, 1_500);

    assert!(qualify_quorum_observed_policy_head_v1(
        &lineage,
        &temporal,
        &basis,
        &publication,
        &log,
        &signed_checkpoint,
        &verified_checkpoint,
        &witnesses,
        &verified_witnesses,
        &witness_policy,
        &trust,
        &containment,
    )
    .is_ok());
}

#[test]
fn authority_surface_has_no_scalar_now_or_deserialization_path() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub fn qualify_quorum_observed_policy_head_v1")
        .expect("qualifier");
    let rest = &source[start..];
    let end = rest.find(") -> Result").expect("qualifier signature") + 1;
    let signature = &rest[..end];

    assert!(!signature.contains("now_unix_s"));
    assert!(!signature.contains("evaluation_time_unix_s"));
    assert!(signature.contains("OperationalClockBasisV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QuorumObservedPolicyHeadV1"
    ));
    assert!(source.contains("This crate deliberately proves a bounded claim"));
}
