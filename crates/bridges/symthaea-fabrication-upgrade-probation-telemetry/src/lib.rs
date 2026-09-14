// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact machine-telemetry evidence binding for clock-governed upgrade probation.
//!
//! This bridge closes the deliberate evidence gap left by the probation authority layer.
//! It first qualifies an exact telemetry bundle independently of any probation observation, so the
//! resulting bundle ID can be written into `UpgradeProbationObservation.telemetry_evidence_digest`
//! before that observation is signed. It then binds the exact signed observation set approved by
//! the probation clearance to those exact telemetry bundles.
//!
//! The theorem remains deliberately bounded: telemetry proves exact signed machine/session/job
//! frames and attempted-job coverage. The success/failure/uncertain outcome split remains an
//! authenticated probation-observer assertion because the machine telemetry schema does not encode
//! job outcome classes.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::telemetry::{
    MAX_TELEMETRY_ID_BYTES, MAX_TELEMETRY_SIGNATURE_BYTES, SignedMachineTelemetry,
    VerifiedMachineTelemetry, canonical_machine_telemetry_bytes, digest_machine_telemetry,
};
use symthaea_fabrication_kernel::telemetry_tracker::MachineTelemetryTracker;
use symthaea_fabrication_kernel::trust::{
    KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_kernel::upgrade_probation::digest_upgrade_probation_observation;
use symthaea_fabrication_upgrade_probation_authority::{
    ClockGovernedUpgradeProbationClearanceIdV1, ClockGovernedUpgradeProbationClearanceV1,
    SignedUpgradeProbationObservationV1,
};

pub const EXACT_PROBATION_TELEMETRY_BUNDLE_SCHEMA: &str =
    "symthaea.fabrication.exact-probation-telemetry-bundle.v1";
pub const TELEMETRY_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.telemetry-bound-upgrade-probation-clearance.v1";
pub const MAX_PROBATION_TELEMETRY_FRAMES_PER_BUNDLE: usize = 65_536;
pub const MAX_PROBATION_TELEMETRY_VERIFIERS: usize = 8;

const SIGNED_TELEMETRY_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.probation-signed-telemetry-evidence.v1\0";
const TELEMETRY_FRAME_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.probation-telemetry-frame-set.v1\0";
const TELEMETRY_TRUST_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.probation-telemetry-trust-set.v1\0";
const TELEMETRY_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.probation-telemetry-verifier-set.v1\0";
const TELEMETRY_BUNDLE_DOMAIN: &[u8] =
    b"symthaea.fabrication.exact-probation-telemetry-bundle.v1\0";
const UPSTREAM_SIGNED_OBSERVATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-evidence.v1\0";
const UPSTREAM_SIGNED_OBSERVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-set.v1\0";
const CLEARANCE_TELEMETRY_BINDING_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-telemetry-binding-set.v1\0";
const TELEMETRY_BOUND_CLEARANCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.telemetry-bound-upgrade-probation-clearance.v1\0";

/// Raw + opaque verified telemetry pair plus the exact trust snapshot used by the opaque verifier.
pub struct ProbationTelemetryFrameInputV1<'a> {
    pub signed: &'a SignedMachineTelemetry,
    pub verified: &'a VerifiedMachineTelemetry,
    pub trust_snapshot: &'a TrustSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactProbationTelemetryVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactProbationTelemetryVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: MAX_PROBATION_TELEMETRY_VERIFIERS,
        }
    }
}

/// Runtime verifier for the exact raw telemetry signature bytes committed by a bundle.
pub trait ExactProbationTelemetryVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;
    fn verify_telemetry_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExactProbationTelemetryBundleIdV1(Sha256Digest);

impl ExactProbationTelemetryBundleIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof for one exact set of authenticated telemetry frames.
///
/// Its ID is intentionally suitable for use as the portable probation observation's
/// `telemetry_evidence_digest` before that observation is signed.
#[derive(Debug, Clone)]
#[must_use]
pub struct ExactProbationTelemetryBundleV1 {
    id: ExactProbationTelemetryBundleIdV1,
    machine_id: String,
    frame_set_digest: Sha256Digest,
    replay_tracker_digest: Sha256Digest,
    trust_snapshot_set_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    frame_count: usize,
    distinct_job_count: usize,
    observed_started_at_unix_ms: u64,
    observed_ended_at_unix_ms: u64,
}

impl ExactProbationTelemetryBundleV1 {
    pub fn id(&self) -> ExactProbationTelemetryBundleIdV1 {
        self.id
    }
    pub fn evidence_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn machine_id(&self) -> &str {
        &self.machine_id
    }
    pub fn frame_set_digest(&self) -> Sha256Digest {
        self.frame_set_digest
    }
    pub fn replay_tracker_digest(&self) -> Sha256Digest {
        self.replay_tracker_digest
    }
    pub fn trust_snapshot_set_digest(&self) -> Sha256Digest {
        self.trust_snapshot_set_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
    pub fn distinct_job_count(&self) -> usize {
        self.distinct_job_count
    }
    pub fn observed_started_at_unix_ms(&self) -> u64 {
        self.observed_started_at_unix_ms
    }
    pub fn observed_ended_at_unix_ms(&self) -> u64 {
        self.observed_ended_at_unix_ms
    }
}

/// Exact signed observation paired with the telemetry bundle named by its telemetry digest.
pub struct ProbationTelemetryBindingInputV1<'a> {
    pub signed_observation: &'a SignedUpgradeProbationObservationV1,
    pub telemetry_bundle: &'a ExactProbationTelemetryBundleV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TelemetryBoundUpgradeProbationClearanceIdV1(Sha256Digest);

impl TelemetryBoundUpgradeProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that every exact signed observation approved by one probation clearance names and
/// is semantically covered by one exact authenticated telemetry bundle.
#[derive(Debug, Clone)]
#[must_use]
pub struct TelemetryBoundUpgradeProbationClearanceV1 {
    id: TelemetryBoundUpgradeProbationClearanceIdV1,
    clearance_id: ClockGovernedUpgradeProbationClearanceIdV1,
    observation_set_digest: Sha256Digest,
    telemetry_binding_set_digest: Sha256Digest,
    observation_count: usize,
    total_frame_count: usize,
    total_attempted_job_count: u64,
}

impl TelemetryBoundUpgradeProbationClearanceV1 {
    pub fn id(&self) -> TelemetryBoundUpgradeProbationClearanceIdV1 {
        self.id
    }
    pub fn clearance_id(&self) -> ClockGovernedUpgradeProbationClearanceIdV1 {
        self.clearance_id
    }
    pub fn observation_set_digest(&self) -> Sha256Digest {
        self.observation_set_digest
    }
    pub fn telemetry_binding_set_digest(&self) -> Sha256Digest {
        self.telemetry_binding_set_digest
    }
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }
    pub fn total_frame_count(&self) -> usize {
        self.total_frame_count
    }
    pub fn total_attempted_job_count(&self) -> u64 {
        self.total_attempted_job_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbationTelemetryBindingError {
    InvalidVerificationPolicy,
    InsufficientVerificationProviders { actual: usize, required: usize },
    TooManyVerificationProviders { actual: usize, maximum: usize },
    InvalidProvider(String),
    DuplicateProvider(String),
    EmptyFrames,
    TooManyFrames { actual: usize, maximum: usize },
    InvalidSignedTelemetry(String),
    VerifiedTelemetryMismatch(String),
    DuplicateTelemetryFrame,
    MultipleMachines,
    TrustSnapshotInvalid(String),
    TrustSnapshotMismatch(String),
    TrustSnapshotNotFreshForFrame(String),
    TelemetryKeyIneligible { key_id: String, eligibility: KeyEligibility },
    TelemetrySignerCompromised(String),
    TelemetrySignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    ReplayTracker(String),
    EmptyBindings,
    BindingCountMismatch { actual: usize, expected: usize },
    ObservationDigestMismatch(String),
    ObservationSetMismatch,
    DuplicateObservation,
    DuplicateTelemetryBundle,
    TelemetryReferenceMismatch(String),
    TelemetryMachineMismatch(String),
    TelemetryOutsideObservationWindow(String),
    AttemptedJobCoverageMismatch { key_id: String, attempted: u64, covered: usize },
    CountOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct VerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryFrameCommitment {
    telemetry_digest: String,
    signed_evidence_digest: String,
    trust_snapshot_digest: String,
    session_digest: String,
    session_sequence: u64,
    printer_job_id: String,
    frame_sequence: u64,
    observed_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryBundleCommitment {
    schema: &'static str,
    machine_id: String,
    frame_set_digest: String,
    replay_tracker_digest: String,
    trust_snapshot_set_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    verifier_set_digest: String,
    frame_count: usize,
    distinct_job_count: usize,
    observed_started_at_unix_ms: u64,
    observed_ended_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ObservationTelemetryBindingCommitment {
    observation_digest: String,
    telemetry_bundle_id: String,
    frame_count: usize,
    attempted_job_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryBoundClearanceCommitment {
    schema: &'static str,
    clearance_id: String,
    observation_set_digest: String,
    telemetry_binding_set_digest: String,
    observation_count: usize,
    total_frame_count: usize,
    total_attempted_job_count: u64,
}

/// Qualify exact raw telemetry evidence before the probation observation is signed.
///
/// There is no caller-selected current time. Historical key validity and compromise state are
/// evaluated at each frame's own signed observation timestamp. Later compromise does not rewrite
/// history unless its effective time is at or before the frame.
pub fn qualify_exact_probation_telemetry_bundle_v1(
    frames: &[ProbationTelemetryFrameInputV1<'_>],
    containment_state: &FabricationContainmentState,
    verification_policy: &ExactProbationTelemetryVerificationPolicyV1,
    verification_providers: &[&dyn ExactProbationTelemetryVerifierV1],
) -> Result<ExactProbationTelemetryBundleV1, Vec<ProbationTelemetryBindingError>> {
    let mut violations = Vec::new();

    if !valid_verification_policy(verification_policy) {
        violations.push(ProbationTelemetryBindingError::InvalidVerificationPolicy);
    }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(ProbationTelemetryBindingError::InsufficientVerificationProviders {
            actual: verification_providers.len(),
            required: verification_policy.minimum_distinct_providers,
        });
    }
    if verification_providers.len() > verification_policy.maximum_providers {
        violations.push(ProbationTelemetryBindingError::TooManyVerificationProviders {
            actual: verification_providers.len(),
            maximum: verification_policy.maximum_providers,
        });
    }
    if frames.is_empty() {
        violations.push(ProbationTelemetryBindingError::EmptyFrames);
    }
    if frames.len() > MAX_PROBATION_TELEMETRY_FRAMES_PER_BUNDLE {
        violations.push(ProbationTelemetryBindingError::TooManyFrames {
            actual: frames.len(),
            maximum: MAX_PROBATION_TELEMETRY_FRAMES_PER_BUNDLE,
        });
    }
    if let Err(error) = containment_state.validate() {
        violations.push(ProbationTelemetryBindingError::ReplayTracker(format!(
            "invalid containment state: {error:?}"
        )));
    }

    let verifier_commitments = match validate_verification_providers(verification_providers) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            Vec::new()
        }
    };

    let mut machine_id: Option<String> = None;
    let mut frame_commitments = Vec::with_capacity(frames.len());
    let mut seen_telemetry = BTreeSet::new();
    let mut trust_digests = BTreeSet::new();
    let mut job_ids = BTreeSet::new();
    let mut started_at = u64::MAX;
    let mut ended_at = 0u64;

    for frame in frames {
        let signed = frame.signed;
        let payload = &signed.payload;
        let key_id = signed.key_id.clone();

        if payload.validate().is_err()
            || !signed.algorithm.is_canonical()
            || invalid_identifier(&signed.key_id)
            || signed.signature.is_empty()
            || signed.signature.len() > MAX_TELEMETRY_SIGNATURE_BYTES
        {
            violations.push(ProbationTelemetryBindingError::InvalidSignedTelemetry(key_id));
            continue;
        }

        let telemetry_digest = match digest_machine_telemetry(payload) {
            Ok(value) => value,
            Err(error) => {
                violations.push(ProbationTelemetryBindingError::InvalidSignedTelemetry(format!(
                    "{}: {error:?}", signed.key_id
                )));
                continue;
            }
        };
        let (verified_algorithm, verified_key_id) = frame.verified.signer();
        if frame.verified.payload() != payload
            || verified_algorithm != &signed.algorithm
            || verified_key_id != signed.key_id.as_str()
            || frame.verified.telemetry_digest() != telemetry_digest
        {
            violations.push(ProbationTelemetryBindingError::VerifiedTelemetryMismatch(
                signed.key_id.clone(),
            ));
        }
        if !seen_telemetry.insert(telemetry_digest) {
            violations.push(ProbationTelemetryBindingError::DuplicateTelemetryFrame);
        }

        match &machine_id {
            Some(expected) if expected != &payload.machine_id => {
                violations.push(ProbationTelemetryBindingError::MultipleMachines)
            }
            None => machine_id = Some(payload.machine_id.clone()),
            _ => {}
        }

        let trust_digest = match digest_trust_snapshot(frame.trust_snapshot) {
            Ok(value) => value,
            Err(error) => {
                violations.push(ProbationTelemetryBindingError::TrustSnapshotInvalid(format!(
                    "{}: {error:?}", signed.key_id
                )));
                continue;
            }
        };
        if trust_digest != frame.verified.trust_snapshot_digest() {
            violations.push(ProbationTelemetryBindingError::TrustSnapshotMismatch(
                signed.key_id.clone(),
            ));
        }
        let observed_at_unix_s = payload.observed_at_unix_ms / 1_000;
        if !frame.trust_snapshot.is_fresh_at(observed_at_unix_s) {
            violations.push(ProbationTelemetryBindingError::TrustSnapshotNotFreshForFrame(
                signed.key_id.clone(),
            ));
        }
        let eligibility = frame.trust_snapshot.key_eligibility(
            &signed.algorithm,
            &signed.key_id,
            KeyUsage::MachineTelemetry,
            observed_at_unix_s,
        );
        if eligibility != KeyEligibility::Eligible {
            violations.push(ProbationTelemetryBindingError::TelemetryKeyIneligible {
                key_id: signed.key_id.clone(),
                eligibility,
            });
        }
        if containment_state.signer_compromise_tracker.is_compromised_at(
            &signed.algorithm,
            &signed.key_id,
            KeyUsage::MachineTelemetry,
            observed_at_unix_s,
        ) {
            violations.push(ProbationTelemetryBindingError::TelemetrySignerCompromised(
                signed.key_id.clone(),
            ));
        }

        let message = match canonical_machine_telemetry_bytes(payload) {
            Ok(value) => value,
            Err(error) => {
                violations.push(ProbationTelemetryBindingError::InvalidSignedTelemetry(format!(
                    "{}: {error:?}", signed.key_id
                )));
                continue;
            }
        };
        for provider in verification_providers {
            match provider.verify_telemetry_signature(
                &signed.algorithm,
                &signed.key_id,
                &message,
                &signed.signature,
            ) {
                Ok(true) => {}
                Ok(false) => violations.push(
                    ProbationTelemetryBindingError::TelemetrySignatureRejected {
                        provider: provider.provider_id().to_string(),
                        key_id: signed.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    ProbationTelemetryBindingError::VerificationProviderError {
                        provider: provider.provider_id().to_string(),
                        reason,
                    },
                ),
            }
        }

        let signed_evidence_digest = match hash_serializable(SIGNED_TELEMETRY_EVIDENCE_DOMAIN, signed)
        {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        trust_digests.insert(trust_digest);
        job_ids.insert((payload.session_digest, payload.printer_job_id.clone()));
        started_at = started_at.min(payload.observed_at_unix_ms);
        ended_at = ended_at.max(payload.observed_at_unix_ms);
        frame_commitments.push(TelemetryFrameCommitment {
            telemetry_digest: telemetry_digest.to_hex(),
            signed_evidence_digest: signed_evidence_digest.to_hex(),
            trust_snapshot_digest: trust_digest.to_hex(),
            session_digest: payload.session_digest.to_hex(),
            session_sequence: payload.session_sequence,
            printer_job_id: payload.printer_job_id.clone(),
            frame_sequence: payload.frame_sequence,
            observed_at_unix_ms: payload.observed_at_unix_ms,
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let machine_id = machine_id.ok_or_else(|| vec![ProbationTelemetryBindingError::EmptyFrames])?;
    frame_commitments.sort_by(|left, right| {
        (
            left.session_digest.as_str(),
            left.printer_job_id.as_str(),
            left.frame_sequence,
            left.telemetry_digest.as_str(),
        )
            .cmp(&(
                right.session_digest.as_str(),
                right.printer_job_id.as_str(),
                right.frame_sequence,
                right.telemetry_digest.as_str(),
            ))
    });
    let frame_set_digest =
        hash_serializable(TELEMETRY_FRAME_SET_DOMAIN, &frame_commitments).map_err(|error| vec![error])?;

    let mut replay_order = (0..frames.len()).collect::<Vec<_>>();
    replay_order.sort_by(|left_index, right_index| {
        let left = &frames[*left_index].signed.payload;
        let right = &frames[*right_index].signed.payload;
        (
            left.machine_id.as_str(),
            left.session_digest,
            left.session_sequence,
            left.printer_job_id.as_str(),
            left.frame_sequence,
            left.observed_at_unix_ms,
        )
            .cmp(&(
                right.machine_id.as_str(),
                right.session_digest,
                right.session_sequence,
                right.printer_job_id.as_str(),
                right.frame_sequence,
                right.observed_at_unix_ms,
            ))
    });
    let mut tracker = MachineTelemetryTracker::default();
    for index in replay_order {
        tracker.accept(frames[index].verified).map_err(|error| {
            vec![ProbationTelemetryBindingError::ReplayTracker(format!("{error:?}"))]
        })?;
    }
    let replay_tracker_digest = tracker
        .digest()
        .map_err(|error| vec![ProbationTelemetryBindingError::ReplayTracker(format!("{error:?}"))])?;

    let trust_snapshot_set = trust_digests
        .into_iter()
        .map(|digest| digest.to_hex())
        .collect::<Vec<_>>();
    let trust_snapshot_set_digest =
        hash_serializable(TELEMETRY_TRUST_SET_DOMAIN, &trust_snapshot_set).map_err(|error| vec![error])?;
    let verifier_set_digest =
        hash_serializable(TELEMETRY_VERIFIER_SET_DOMAIN, &verifier_commitments)
            .map_err(|error| vec![error])?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ProbationTelemetryBindingError::ReplayTracker(format!(
            "containment digest: {error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ProbationTelemetryBindingError::ReplayTracker(format!(
                    "compromise digest: {error:?}"
                ))]
            },
        )?;

    let commitment = TelemetryBundleCommitment {
        schema: EXACT_PROBATION_TELEMETRY_BUNDLE_SCHEMA,
        machine_id: machine_id.clone(),
        frame_set_digest: frame_set_digest.to_hex(),
        replay_tracker_digest: replay_tracker_digest.to_hex(),
        trust_snapshot_set_digest: trust_snapshot_set_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        verifier_set_digest: verifier_set_digest.to_hex(),
        frame_count: frames.len(),
        distinct_job_count: job_ids.len(),
        observed_started_at_unix_ms: started_at,
        observed_ended_at_unix_ms: ended_at,
    };
    let id = ExactProbationTelemetryBundleIdV1(
        hash_serializable(TELEMETRY_BUNDLE_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ExactProbationTelemetryBundleV1 {
        id,
        machine_id,
        frame_set_digest,
        replay_tracker_digest,
        trust_snapshot_set_digest,
        containment_state_digest,
        compromise_tracker_digest,
        verifier_set_digest,
        frame_count: frames.len(),
        distinct_job_count: job_ids.len(),
        observed_started_at_unix_ms: started_at,
        observed_ended_at_unix_ms: ended_at,
    })
}

/// Bind the exact observation set already authorized by the probation clearance to exact telemetry.
pub fn bind_probation_clearance_to_exact_telemetry_v1(
    clearance: &ClockGovernedUpgradeProbationClearanceV1,
    bindings: &[ProbationTelemetryBindingInputV1<'_>],
) -> Result<TelemetryBoundUpgradeProbationClearanceV1, Vec<ProbationTelemetryBindingError>> {
    let mut violations = Vec::new();
    if bindings.is_empty() {
        violations.push(ProbationTelemetryBindingError::EmptyBindings);
    }
    if bindings.len() != clearance.observation_count() {
        violations.push(ProbationTelemetryBindingError::BindingCountMismatch {
            actual: bindings.len(),
            expected: clearance.observation_count(),
        });
    }

    let mut observation_evidence_digests = Vec::with_capacity(bindings.len());
    let mut seen_observations = BTreeSet::new();
    let mut seen_bundles = BTreeSet::new();
    let mut binding_commitments = Vec::with_capacity(bindings.len());
    let mut total_frame_count = 0usize;
    let mut total_attempted_job_count = 0u64;

    for binding in bindings {
        let signed = binding.signed_observation;
        let observation = &signed.observation;
        let expected_observation_digest = match digest_upgrade_probation_observation(observation) {
            Ok(value) => value,
            Err(error) => {
                violations.push(ProbationTelemetryBindingError::ObservationDigestMismatch(format!(
                    "{}: {error:?}", signed.key_id
                )));
                continue;
            }
        };
        if expected_observation_digest != signed.observation_digest {
            violations.push(ProbationTelemetryBindingError::ObservationDigestMismatch(
                signed.key_id.clone(),
            ));
        }
        if !seen_observations.insert(signed.observation_digest) {
            violations.push(ProbationTelemetryBindingError::DuplicateObservation);
        }
        if !seen_bundles.insert(binding.telemetry_bundle.id().as_digest()) {
            violations.push(ProbationTelemetryBindingError::DuplicateTelemetryBundle);
        }
        if observation.telemetry_evidence_digest != binding.telemetry_bundle.evidence_digest() {
            violations.push(ProbationTelemetryBindingError::TelemetryReferenceMismatch(
                signed.key_id.clone(),
            ));
        }
        if observation.machine_id != binding.telemetry_bundle.machine_id() {
            violations.push(ProbationTelemetryBindingError::TelemetryMachineMismatch(
                signed.key_id.clone(),
            ));
        }
        if binding.telemetry_bundle.observed_started_at_unix_ms()
            < observation.started_at_unix_ms
            || binding.telemetry_bundle.observed_ended_at_unix_ms()
                > observation.ended_at_unix_ms
        {
            violations.push(ProbationTelemetryBindingError::TelemetryOutsideObservationWindow(
                signed.key_id.clone(),
            ));
        }
        if u64::try_from(binding.telemetry_bundle.distinct_job_count()).ok()
            != Some(observation.attempted_jobs)
        {
            violations.push(ProbationTelemetryBindingError::AttemptedJobCoverageMismatch {
                key_id: signed.key_id.clone(),
                attempted: observation.attempted_jobs,
                covered: binding.telemetry_bundle.distinct_job_count(),
            });
        }

        match hash_serializable(UPSTREAM_SIGNED_OBSERVATION_EVIDENCE_DOMAIN, signed) {
            Ok(value) => observation_evidence_digests.push(value),
            Err(error) => violations.push(error),
        }
        total_frame_count = match total_frame_count.checked_add(binding.telemetry_bundle.frame_count()) {
            Some(value) => value,
            None => {
                violations.push(ProbationTelemetryBindingError::CountOverflow);
                total_frame_count
            }
        };
        total_attempted_job_count = match total_attempted_job_count.checked_add(observation.attempted_jobs) {
            Some(value) => value,
            None => {
                violations.push(ProbationTelemetryBindingError::CountOverflow);
                total_attempted_job_count
            }
        };
        binding_commitments.push(ObservationTelemetryBindingCommitment {
            observation_digest: signed.observation_digest.to_hex(),
            telemetry_bundle_id: binding.telemetry_bundle.id().to_hex(),
            frame_count: binding.telemetry_bundle.frame_count(),
            attempted_job_count: observation.attempted_jobs,
        });
    }

    observation_evidence_digests.sort();
    let observation_set_digest = match hash_serializable(
        UPSTREAM_SIGNED_OBSERVATION_SET_DOMAIN,
        &observation_evidence_digests,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    if observation_set_digest != clearance.observation_set_digest() {
        violations.push(ProbationTelemetryBindingError::ObservationSetMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    binding_commitments.sort_by(|left, right| left.observation_digest.cmp(&right.observation_digest));
    let telemetry_binding_set_digest =
        hash_serializable(CLEARANCE_TELEMETRY_BINDING_SET_DOMAIN, &binding_commitments)
            .map_err(|error| vec![error])?;
    let commitment = TelemetryBoundClearanceCommitment {
        schema: TELEMETRY_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
        clearance_id: clearance.id().to_hex(),
        observation_set_digest: observation_set_digest.to_hex(),
        telemetry_binding_set_digest: telemetry_binding_set_digest.to_hex(),
        observation_count: bindings.len(),
        total_frame_count,
        total_attempted_job_count,
    };
    let id = TelemetryBoundUpgradeProbationClearanceIdV1(
        hash_serializable(TELEMETRY_BOUND_CLEARANCE_DOMAIN, &commitment)
            .map_err(|error| vec![error])?,
    );

    Ok(TelemetryBoundUpgradeProbationClearanceV1 {
        id,
        clearance_id: clearance.id(),
        observation_set_digest,
        telemetry_binding_set_digest,
        observation_count: bindings.len(),
        total_frame_count,
        total_attempted_job_count,
    })
}

fn valid_verification_policy(policy: &ExactProbationTelemetryVerificationPolicyV1) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_PROBATION_TELEMETRY_VERIFIERS
}

fn validate_verification_providers(
    providers: &[&dyn ExactProbationTelemetryVerifierV1],
) -> Result<Vec<VerifierCommitment>, Vec<ProbationTelemetryBindingError>> {
    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    let mut commitments = Vec::with_capacity(providers.len());
    for provider in providers {
        let provider_id = provider.provider_id().to_string();
        if invalid_identifier(&provider_id) {
            violations.push(ProbationTelemetryBindingError::InvalidProvider(provider_id));
            continue;
        }
        if !seen.insert(provider_id.clone()) {
            violations.push(ProbationTelemetryBindingError::DuplicateProvider(provider_id));
            continue;
        }
        let digest = provider.verification_policy_digest();
        if digest == Sha256Digest([0; 32]) {
            violations.push(ProbationTelemetryBindingError::InvalidProvider(provider_id));
            continue;
        }
        commitments.push(VerifierCommitment {
            provider_id,
            verification_policy_digest: digest.to_hex(),
        });
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    Ok(commitments)
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TELEMETRY_ID_BYTES
        || value.chars().any(char::is_control)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ProbationTelemetryBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ProbationTelemetryBindingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_policy_requires_real_diversity_capacity() {
        assert!(valid_verification_policy(
            &ExactProbationTelemetryVerificationPolicyV1::default()
        ));
        assert!(!valid_verification_policy(
            &ExactProbationTelemetryVerificationPolicyV1 {
                minimum_distinct_providers: 0,
                maximum_providers: 2,
            }
        ));
    }

    #[test]
    fn bound_clearance_identity_changes_with_telemetry_binding_set() {
        let first = TelemetryBoundClearanceCommitment {
            schema: TELEMETRY_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
            clearance_id: "11".repeat(32),
            observation_set_digest: "22".repeat(32),
            telemetry_binding_set_digest: "33".repeat(32),
            observation_count: 2,
            total_frame_count: 20,
            total_attempted_job_count: 4,
        };
        let mut second = first.clone();
        second.telemetry_binding_set_digest = "44".repeat(32);
        assert_ne!(
            hash_serializable(TELEMETRY_BOUND_CLEARANCE_DOMAIN, &first).unwrap(),
            hash_serializable(TELEMETRY_BOUND_CLEARANCE_DOMAIN, &second).unwrap()
        );
    }
}
