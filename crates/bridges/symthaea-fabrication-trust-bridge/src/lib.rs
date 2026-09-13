// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe authority bridge from `symthaea-trust-kernel` into fabrication governance.
//!
//! A scalar-time [`VerifiedThresholdCeremony`] remains useful cryptographic and quorum evidence,
//! but it is not sufficient temporal authority when the trusted clock proves an interval rather
//! than one exact instant. This crate re-qualifies that opaque ceremony against the complete
//! trusted interval, the actual fabrication trust records, the exact threshold policy, and the
//! persistent signer-compromise tracker.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::{
    SignerCompromiseTracker, digest_signer_compromise_tracker,
};
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, SIGNED_THRESHOLD_APPROVAL_SCHEMA,
    SignedThresholdApproval, ThresholdCeremonyPolicy, VerifiedThresholdCeremony,
    digest_threshold_approval,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError,
};

pub const CLOCK_GOVERNED_THRESHOLD_CEREMONY_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-threshold-ceremony.v1";
const CLOCK_GOVERNED_THRESHOLD_CEREMONY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-ceremony.v1\0";
const CLOCK_GOVERNED_THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const VERIFIED_THRESHOLD_CEREMONY_DOMAIN: &[u8] =
    b"symthaea.fabrication.verified-threshold-ceremony.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedThresholdCeremonyIdV1(Sha256Digest);

impl ClockGovernedThresholdCeremonyIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that an existing threshold ceremony remains authoritative for
/// every possible true time in one trusted operational-clock envelope.
///
/// This type is intentionally not serializable and exposes no public fields or
/// public constructor. Scalar-time ceremonies cannot be substituted for it.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedThresholdCeremonyV1 {
    id: ClockGovernedThresholdCeremonyIdV1,
    ceremony_digest: Sha256Digest,
    payload_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    purpose: String,
    signers: Vec<(SignatureAlgorithm, String)>,
}

impl ClockGovernedThresholdCeremonyV1 {
    pub fn id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.id
    }

    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }

    pub fn payload_digest(&self) -> Sha256Digest {
        self.payload_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }

    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }

    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }

    pub fn purpose(&self) -> &str {
        &self.purpose
    }

    pub fn signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.signers
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedThresholdError {
    InvalidPolicy,
    TooManyApprovals {
        actual: usize,
        maximum: usize,
    },
    InvalidApproval(String),
    ApprovalDigestMismatch(String),
    ApprovalPurposeMismatch(String),
    ApprovalPayloadMismatch(String),
    DuplicateApprovalSigner(String),
    ApprovalSignerSetMismatch,
    ApprovalNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    CeremonyDigestMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotDigestMismatch,
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    KeyNotAllowed(String),
    InsufficientDistinctSigners {
        actual: usize,
        required: usize,
    },
    MissingAlgorithm(SignatureAlgorithm),
    MissingAlgorithmDiversity,
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    CompromiseTrackerInvalid(String),
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    Encoding(String),
}

/// Re-qualify an already cryptographically verified threshold ceremony under
/// interval-safe clock authority.
///
/// `VerifiedThresholdCeremony` is treated only as the cryptographic/quorum
/// substrate. Every temporal property is re-evaluated against `clock`, and
/// lifecycle/usage/containment authority is derived from the actual governed
/// fabrication records. There is deliberately no caller-supplied `now`.
pub fn qualify_clock_governed_threshold_ceremony_v1(
    ceremony: &VerifiedThresholdCeremony,
    approvals: &[SignedThresholdApproval],
    policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    compromise_tracker: &SignerCompromiseTracker,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
) -> Result<ClockGovernedThresholdCeremonyV1, Vec<ClockGovernedThresholdError>> {
    let mut violations = Vec::new();

    if !threshold_policy_is_valid(policy) {
        violations.push(ClockGovernedThresholdError::InvalidPolicy);
        return Err(violations);
    }
    if approvals.len() > policy.maximum_approvals {
        violations.push(ClockGovernedThresholdError::TooManyApprovals {
            actual: approvals.len(),
            maximum: policy.maximum_approvals,
        });
    }

    if let Err(error) = trust_snapshot.validate() {
        violations.push(ClockGovernedThresholdError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(digest) => Some(digest),
        Err(error) => {
            violations.push(ClockGovernedThresholdError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            None
        }
    };
    if trust_snapshot_digest.is_some_and(|digest| digest != ceremony.trust_snapshot_digest()) {
        violations.push(ClockGovernedThresholdError::TrustSnapshotDigestMismatch);
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(ClockGovernedThresholdError::TrustSnapshotNotValidAcrossEnvelope(
            reason,
        ));
    }

    if let Err(error) = compromise_tracker.validate() {
        violations.push(ClockGovernedThresholdError::CompromiseTrackerInvalid(format!(
            "{error:?}"
        )));
    }
    let compromise_tracker_digest = match digest_signer_compromise_tracker(compromise_tracker) {
        Ok(digest) => Some(digest),
        Err(error) => {
            violations.push(ClockGovernedThresholdError::CompromiseTrackerInvalid(format!(
                "{error:?}"
            )));
            None
        }
    };

    let mut approval_signers = Vec::with_capacity(approvals.len());
    let mut approval_digests = Vec::with_capacity(approvals.len());
    let mut seen_approval_signers = BTreeSet::new();
    for signed in approvals.iter().take(policy.maximum_approvals) {
        let key_id = signed.signature.key_id.clone();
        let algorithm = signed.signature.algorithm.clone();
        if signed.schema_version != SIGNED_THRESHOLD_APPROVAL_SCHEMA
            || !algorithm.is_canonical()
            || key_id.trim().is_empty()
            || key_id != key_id.trim()
            || key_id.len() > MAX_THRESHOLD_KEY_ID_BYTES
        {
            violations.push(ClockGovernedThresholdError::InvalidApproval(key_id));
            continue;
        }
        if let Err(error) = signed.approval.validate() {
            violations.push(ClockGovernedThresholdError::InvalidApproval(format!(
                "{key_id}: {error:?}"
            )));
            continue;
        }
        let expected_digest = match digest_threshold_approval(&signed.approval) {
            Ok(digest) => digest,
            Err(error) => {
                violations.push(ClockGovernedThresholdError::InvalidApproval(format!(
                    "{key_id}: {error:?}"
                )));
                continue;
            }
        };
        if expected_digest != signed.approval_digest {
            violations.push(ClockGovernedThresholdError::ApprovalDigestMismatch(
                key_id.clone(),
            ));
            continue;
        }
        if signed.approval.purpose != ceremony.purpose() {
            violations.push(ClockGovernedThresholdError::ApprovalPurposeMismatch(
                key_id.clone(),
            ));
            continue;
        }
        if signed.approval.payload_digest != ceremony.payload_digest() {
            violations.push(ClockGovernedThresholdError::ApprovalPayloadMismatch(
                key_id.clone(),
            ));
            continue;
        }
        if let Err(reason) = clock.require_valid_across_seconds_window(
            signed.approval.issued_at_unix_s,
            signed.approval.expires_at_unix_s,
        ) {
            violations.push(ClockGovernedThresholdError::ApprovalNotValidAcrossEnvelope {
                key_id: key_id.clone(),
                reason,
            });
        }
        if !seen_approval_signers.insert((algorithm.clone(), key_id.clone())) {
            violations.push(ClockGovernedThresholdError::DuplicateApprovalSigner(
                key_id.clone(),
            ));
            continue;
        }
        approval_signers.push((algorithm, key_id));
        approval_digests.push(signed.approval_digest);
    }
    approval_signers.sort();
    approval_digests.sort();

    if approval_signers.as_slice() != ceremony.signers() {
        violations.push(ClockGovernedThresholdError::ApprovalSignerSetMismatch);
    }
    match reconstruct_verified_ceremony_digest(
        ceremony.purpose(),
        ceremony.payload_digest(),
        ceremony.trust_snapshot_digest(),
        &approval_signers,
        &approval_digests,
    ) {
        Ok(digest) if digest == ceremony.ceremony_digest() => {}
        Ok(_) => violations.push(ClockGovernedThresholdError::CeremonyDigestMismatch),
        Err(error) => violations.push(error),
    }

    let signers = ceremony.signers();
    if signers.len() < policy.minimum_distinct_signers {
        violations.push(ClockGovernedThresholdError::InsufficientDistinctSigners {
            actual: signers.len(),
            required: policy.minimum_distinct_signers,
        });
    }
    let algorithms = signers
        .iter()
        .map(|(algorithm, _)| algorithm.clone())
        .collect::<BTreeSet<_>>();
    for required in &policy.required_algorithms {
        if !algorithms.contains(required) {
            violations.push(ClockGovernedThresholdError::MissingAlgorithm(required.clone()));
        }
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ClockGovernedThresholdError::MissingAlgorithmDiversity);
    }

    for (algorithm, key_id) in signers {
        if policy
            .allowed_key_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(key_id))
        {
            violations.push(ClockGovernedThresholdError::KeyNotAllowed(key_id.clone()));
        }
        let Some(record) = trust_snapshot
            .keys
            .iter()
            .find(|record| &record.algorithm == algorithm && record.key_id == *key_id)
        else {
            violations.push(ClockGovernedThresholdError::SignerUnknown(key_id.clone()));
            continue;
        };
        if record.status != KeyLifecycleStatus::Active {
            violations.push(ClockGovernedThresholdError::SignerNotActive(key_id.clone()));
        }
        if !record.usages.contains(&policy.key_usage) {
            violations.push(ClockGovernedThresholdError::SignerUsageNotAllowed(
                key_id.clone(),
            ));
        }
        if let Err(reason) = clock.require_valid_across_optional_seconds_window(
            record.not_before_unix_s,
            record.not_after_unix_s,
        ) {
            violations.push(ClockGovernedThresholdError::SignerNotValidAcrossEnvelope {
                key_id: key_id.clone(),
                reason,
            });
        }

        for containment in compromise_tracker.records().iter().filter(|containment| {
            &containment.signer.algorithm == algorithm
                && containment.signer.key_id == *key_id
                && containment.affected_usages.contains(&policy.key_usage)
        }) {
            match clock.require_effective_time_after_envelope_seconds(containment.effective_at_unix_s)
            {
                Ok(()) => {}
                Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                    ClockGovernedThresholdError::SignerCompromisedAcrossEnvelope(key_id.clone()),
                ),
                Err(reason) => violations.push(ClockGovernedThresholdError::CompromiseTimeInvalid {
                    key_id: key_id.clone(),
                    reason,
                }),
            }
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let policy_digest = digest_threshold_policy(policy).map_err(|error| vec![error])?;
    let compromise_tracker_digest = compromise_tracker_digest
        .ok_or_else(|| vec![ClockGovernedThresholdError::CompromiseTrackerInvalid(
            "missing validated tracker digest".into(),
        )])?;
    let id = ClockGovernedThresholdCeremonyIdV1(
        digest_clock_governed_threshold_capability(
            ceremony.ceremony_digest(),
            policy_digest,
            compromise_tracker_digest,
            clock.id(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedThresholdCeremonyV1 {
        id,
        ceremony_digest: ceremony.ceremony_digest(),
        payload_digest: ceremony.payload_digest(),
        trust_snapshot_digest: ceremony.trust_snapshot_digest(),
        policy_digest,
        compromise_tracker_digest,
        clock_envelope_id: clock.id(),
        purpose: ceremony.purpose().to_string(),
        signers: ceremony.signers().to_vec(),
    })
}

#[derive(Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}

fn threshold_policy_is_valid(policy: &ThresholdCeremonyPolicy) -> bool {
    policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy
            .required_algorithms
            .iter()
            .all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !id.trim().is_empty()
                    && id == id.trim()
                    && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
                    && !id.chars().any(char::is_control)
            })
        })
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, ClockGovernedThresholdError> {
    let commitment = ThresholdPolicyCommitment {
        minimum_distinct_signers: policy.minimum_distinct_signers,
        maximum_approvals: policy.maximum_approvals,
        require_algorithm_diversity: policy.require_algorithm_diversity,
        required_algorithms: &policy.required_algorithms,
        allowed_key_ids: &policy.allowed_key_ids,
        key_usage: policy.key_usage,
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|error| ClockGovernedThresholdError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(CLOCK_GOVERNED_THRESHOLD_POLICY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn reconstruct_verified_ceremony_digest(
    purpose: &str,
    payload_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    signers: &[(SignatureAlgorithm, String)],
    approval_digests: &[Sha256Digest],
) -> Result<Sha256Digest, ClockGovernedThresholdError> {
    let bytes = serde_json::to_vec(&(
        purpose,
        payload_digest,
        trust_snapshot_digest,
        signers,
        approval_digests,
    ))
    .map_err(|error| ClockGovernedThresholdError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(VERIFIED_THRESHOLD_CEREMONY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct ClockGovernedThresholdCapabilityCommitment {
    schema: &'static str,
    ceremony_digest: String,
    policy_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
}

fn digest_clock_governed_threshold_capability(
    ceremony_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
) -> Result<Sha256Digest, ClockGovernedThresholdError> {
    let commitment = ClockGovernedThresholdCapabilityCommitment {
        schema: CLOCK_GOVERNED_THRESHOLD_CEREMONY_SCHEMA,
        ceremony_digest: ceremony_digest.to_hex(),
        policy_digest: policy_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        clock_envelope_id: clock_envelope_id.to_hex(),
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|error| ClockGovernedThresholdError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(CLOCK_GOVERNED_THRESHOLD_CEREMONY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use symthaea_fabrication_kernel::crypto_digest::sha256;
    use symthaea_fabrication_kernel::threshold::{
        ThresholdApprovalSigner, ThresholdApprovalVerifier, sign_threshold_approval,
        verify_threshold_ceremony,
    };
    use symthaea_fabrication_kernel::trust::KeyTrustRecord;
    use symthaea_trust_kernel::{
        CLOCK_OBSERVATION_SCHEMA, ClockBootstrapAuthorityEvidenceV2,
        ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
        ClockContinuityPolicyRevisionV1, ClockEvaluationPolicyV4, ClockObservation,
        ClockObservationVerifier, ClockQuorumPolicyRevisionV1, DetachedSignature,
        KeyLifecycleStatus as ClockKeyLifecycleStatus, KeyTrustRecord as ClockKeyTrustRecord,
        KeyUsage as ClockKeyUsage, Sha256Digest as ClockSha256Digest,
        SignatureAlgorithm as ClockSignatureAlgorithm, TrustSnapshot as ClockTrustSnapshot,
        accept_bootstrap_clock_basis_v5, bind_bootstrap_operational_clock_basis_v1,
        derive_bootstrap_clock_evaluation_permit_v4,
        derive_clock_governance_evaluation_envelope_v1,
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

    fn fabrication_key(
        algorithm: SignatureAlgorithm,
        key_id: &str,
        not_before_unix_s: u64,
    ) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm,
            key_id: key_id.into(),
            not_before_unix_s,
            not_after_unix_s: None,
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::ThresholdCeremony]),
        }
    }

    fn fabrication_trust(two_signers: bool, first_not_before: u64) -> TrustSnapshot {
        let mut keys = vec![fabrication_key(
            SignatureAlgorithm::Ed25519,
            "a",
            first_not_before,
        )];
        if two_signers {
            keys.push(fabrication_key(
                SignatureAlgorithm::MlDsa65,
                "b",
                1_000,
            ));
        }
        TrustSnapshot::new(1, 1_000, 2_000, keys).unwrap()
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
        let authority =
            verify_clock_bootstrap_authority(&claim, &evidence, &BootstrapVerifier).unwrap();
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
        let basis = accept_bootstrap_clock_basis_v5(&permit, &observations, &snapshot, &verifier)
            .unwrap();
        assert_eq!(verifier.calls.get(), 2);
        let operational =
            bind_bootstrap_operational_clock_basis_v1(&basis, &evaluation, &snapshot).unwrap();
        derive_clock_governance_evaluation_envelope_v1(&operational).unwrap()
    }

    #[test]
    fn scalar_verified_ceremony_is_upgraded_only_after_whole_interval_requalification() {
        let payload = sha256(b"governance-payload");
        let a = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        };
        let b = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "b",
        };
        let approvals = vec![
            sign_threshold_approval("policy-migration", payload, 1_499, 1_501, &a).unwrap(),
            sign_threshold_approval("policy-migration", payload, 1_499, 1_501, &b).unwrap(),
        ];
        let trust = fabrication_trust(true, 1_000);
        let policy = ThresholdCeremonyPolicy::default();
        let ceremony = verify_threshold_ceremony(
            "policy-migration",
            payload,
            &approvals,
            &policy,
            &trust,
            1_500,
            &a,
        )
        .unwrap();
        let envelope = clock_envelope();
        let governed = qualify_clock_governed_threshold_ceremony_v1(
            &ceremony,
            &approvals,
            &policy,
            &trust,
            &SignerCompromiseTracker::default(),
            &envelope,
        )
        .unwrap();

        assert_eq!(governed.clock_envelope_id(), envelope.id());
        assert_eq!(governed.ceremony_digest(), ceremony.ceremony_digest());
        assert_eq!(governed.signers().len(), 2);
    }

    #[test]
    fn scalar_valid_approval_that_expires_inside_clock_uncertainty_is_rejected() {
        let payload = sha256(b"governance-payload");
        let a = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        };
        let b = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "b",
        };
        let approvals = vec![
            sign_threshold_approval("policy-migration", payload, 1_498, 1_500, &a).unwrap(),
            sign_threshold_approval("policy-migration", payload, 1_498, 1_500, &b).unwrap(),
        ];
        let trust = fabrication_trust(true, 1_000);
        let policy = ThresholdCeremonyPolicy::default();
        let ceremony = verify_threshold_ceremony(
            "policy-migration",
            payload,
            &approvals,
            &policy,
            &trust,
            1_499,
            &a,
        )
        .unwrap();
        let errors = qualify_clock_governed_threshold_ceremony_v1(
            &ceremony,
            &approvals,
            &policy,
            &trust,
            &SignerCompromiseTracker::default(),
            &clock_envelope(),
        )
        .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            ClockGovernedThresholdError::ApprovalNotValidAcrossEnvelope { .. }
        )));
    }

    #[test]
    fn scalar_valid_signer_that_starts_inside_clock_uncertainty_is_rejected() {
        let payload = sha256(b"governance-payload");
        let a = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        };
        let approvals = vec![
            sign_threshold_approval("policy-migration", payload, 1_499, 1_501, &a).unwrap(),
        ];
        let trust = fabrication_trust(false, 1_500);
        let policy = ThresholdCeremonyPolicy {
            minimum_distinct_signers: 1,
            require_algorithm_diversity: false,
            ..ThresholdCeremonyPolicy::default()
        };
        let ceremony = verify_threshold_ceremony(
            "policy-migration",
            payload,
            &approvals,
            &policy,
            &trust,
            1_500,
            &a,
        )
        .unwrap();
        let errors = qualify_clock_governed_threshold_ceremony_v1(
            &ceremony,
            &approvals,
            &policy,
            &trust,
            &SignerCompromiseTracker::default(),
            &clock_envelope(),
        )
        .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            ClockGovernedThresholdError::SignerNotValidAcrossEnvelope { .. }
        )));
    }

    #[test]
    fn capability_surface_has_no_scalar_now_or_deserialization_path() {
        let source = include_str!("lib.rs");
        let start = source
            .find("pub fn qualify_clock_governed_threshold_ceremony_v1")
            .expect("qualifier");
        let rest = &source[start..];
        let end = rest.find(") -> Result").expect("qualifier signature") + 1;
        let signature = &rest[..end];

        assert!(!signature.contains("now_unix_s"));
        assert!(!signature.contains("evaluation_time_unix_s"));
        assert!(signature.contains("ClockGovernanceEvaluationEnvelopeV1"));
        assert!(!source.contains(
            "Serialize, Deserialize)]\npub struct ClockGovernedThresholdCeremonyV1"
        ));
    }
}
