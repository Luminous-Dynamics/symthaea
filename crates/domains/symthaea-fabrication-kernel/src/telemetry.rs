// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signed, lifecycle-governed machine telemetry.
//!
//! Raw sensor values are descriptive input only. A [`VerifiedMachineTelemetry`]
//! is the capability-bearing form whose canonical payload, signature, trust
//! lifecycle, freshness window, manifest, machine, session, and printer job
//! identity have all been checked.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::execution_guard::ExecutionTelemetry;
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};

pub const MACHINE_TELEMETRY_SCHEMA: &str = "symthaea.fabrication.machine-telemetry.v1";
pub const MAX_TELEMETRY_ID_BYTES: usize = 256;
pub const MAX_TELEMETRY_SIGNATURE_BYTES: usize = 64 * 1024;
pub const PROGRESS_PARTS_PER_MILLION: u32 = 1_000_000;
pub const ABSOLUTE_ZERO_MILLI_C: i32 = -273_150;
pub const MAX_SENSOR_MILLI_C: i32 = 1_000_000;

/// Integer-valued telemetry avoids NaN and representation ambiguity at the
/// signed evidence boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineTelemetryPayload {
    pub schema_version: String,
    pub manifest_digest: Sha256Digest,
    pub machine_id: String,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
    pub printer_job_id: String,
    pub frame_sequence: u64,
    pub observed_at_unix_ms: u64,
    pub elapsed_ms: u64,
    pub heartbeat_sequence: u64,
    pub progress_ppm: u32,
    pub nozzle_actual_milli_c: i32,
    pub nozzle_target_milli_c: i32,
    pub bed_actual_milli_c: i32,
    pub bed_target_milli_c: i32,
}

impl MachineTelemetryPayload {
    pub fn validate(&self) -> Result<(), TelemetryViolation> {
        if self.schema_version != MACHINE_TELEMETRY_SCHEMA {
            return Err(TelemetryViolation::UnsupportedSchema);
        }
        for (field, value) in [
            ("machine_id", self.machine_id.as_str()),
            ("printer_job_id", self.printer_job_id.as_str()),
        ] {
            if !canonical_identifier(value) {
                return Err(TelemetryViolation::InvalidIdentifier(field));
            }
        }
        if self.session_sequence == 0 {
            return Err(TelemetryViolation::ZeroSequence("session_sequence"));
        }
        if self.frame_sequence == 0 {
            return Err(TelemetryViolation::ZeroSequence("frame_sequence"));
        }
        if self.progress_ppm > PROGRESS_PARTS_PER_MILLION {
            return Err(TelemetryViolation::ProgressOutOfRange(self.progress_ppm));
        }
        for (field, value) in [
            ("nozzle_actual_milli_c", self.nozzle_actual_milli_c),
            ("nozzle_target_milli_c", self.nozzle_target_milli_c),
            ("bed_actual_milli_c", self.bed_actual_milli_c),
            ("bed_target_milli_c", self.bed_target_milli_c),
        ] {
            if !(ABSOLUTE_ZERO_MILLI_C..=MAX_SENSOR_MILLI_C).contains(&value) {
                return Err(TelemetryViolation::TemperatureOutOfRange { field, value });
            }
        }
        Ok(())
    }

    pub fn to_execution_telemetry(&self) -> ExecutionTelemetry {
        ExecutionTelemetry {
            elapsed_s: self.elapsed_ms as f64 / 1_000.0,
            heartbeat_sequence: self.heartbeat_sequence,
            progress: self.progress_ppm as f32 / PROGRESS_PARTS_PER_MILLION as f32,
            nozzle_actual_c: self.nozzle_actual_milli_c as f32 / 1_000.0,
            nozzle_target_c: self.nozzle_target_milli_c as f32 / 1_000.0,
            bed_actual_c: self.bed_actual_milli_c as f32 / 1_000.0,
            bed_target_c: self.bed_target_milli_c as f32 / 1_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMachineTelemetry {
    pub payload: MachineTelemetryPayload,
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

pub trait MachineTelemetrySigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_telemetry(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait MachineTelemetryVerifier {
    fn verify_telemetry(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineTelemetryPolicy {
    pub maximum_age_ms: u64,
    pub maximum_future_skew_ms: u64,
    pub maximum_signature_bytes: usize,
}

impl Default for MachineTelemetryPolicy {
    fn default() -> Self {
        Self {
            maximum_age_ms: 10_000,
            maximum_future_skew_ms: 1_000,
            maximum_signature_bytes: MAX_TELEMETRY_SIGNATURE_BYTES,
        }
    }
}

impl MachineTelemetryPolicy {
    pub fn validate(&self) -> Result<(), TelemetryViolation> {
        if self.maximum_age_ms == 0 {
            return Err(TelemetryViolation::InvalidPolicy("maximum_age_ms"));
        }
        if self.maximum_signature_bytes == 0
            || self.maximum_signature_bytes > MAX_TELEMETRY_SIGNATURE_BYTES
        {
            return Err(TelemetryViolation::InvalidPolicy("maximum_signature_bytes"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryViolation {
    UnsupportedSchema,
    InvalidPolicy(&'static str),
    InvalidAlgorithm,
    InvalidKeyId,
    InvalidIdentifier(&'static str),
    ZeroSequence(&'static str),
    ProgressOutOfRange(u32),
    TemperatureOutOfRange { field: &'static str, value: i32 },
    SignatureTooLarge { actual: usize, maximum: usize },
    Encoding(String),
    Signing(String),
    Verification(String),
    SignatureInvalid,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
    KeyIneligible(KeyEligibility),
    ManifestMismatch,
    MachineMismatch,
    SessionDigestMismatch,
    SessionSequenceMismatch,
    PrinterJobMismatch,
    ObservationFromFuture,
    ObservationStale,
}

#[derive(Debug, Clone)]
pub struct TelemetryExpectation<'a> {
    pub manifest_digest: Sha256Digest,
    pub machine_id: &'a str,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
    pub printer_job_id: &'a str,
}

/// Telemetry authority for one exact canonical payload.
#[derive(Debug, Clone)]
pub struct VerifiedMachineTelemetry {
    signed: SignedMachineTelemetry,
    telemetry_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    verified_at_unix_ms: u64,
}

impl VerifiedMachineTelemetry {
    pub fn payload(&self) -> &MachineTelemetryPayload {
        &self.signed.payload
    }

    pub fn telemetry_digest(&self) -> Sha256Digest {
        self.telemetry_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }

    pub fn signer(&self) -> (&SignatureAlgorithm, &str) {
        (&self.signed.algorithm, &self.signed.key_id)
    }

    pub fn execution_telemetry(&self) -> ExecutionTelemetry {
        self.signed.payload.to_execution_telemetry()
    }
}

pub fn canonical_machine_telemetry_bytes(
    payload: &MachineTelemetryPayload,
) -> Result<Vec<u8>, TelemetryViolation> {
    payload.validate()?;
    serde_json::to_vec(payload).map_err(|error| TelemetryViolation::Encoding(error.to_string()))
}

pub fn digest_machine_telemetry(
    payload: &MachineTelemetryPayload,
) -> Result<Sha256Digest, TelemetryViolation> {
    let bytes = canonical_machine_telemetry_bytes(payload)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.machine-telemetry-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_machine_telemetry(
    payload: MachineTelemetryPayload,
    signer: &dyn MachineTelemetrySigner,
) -> Result<SignedMachineTelemetry, TelemetryViolation> {
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(TelemetryViolation::InvalidAlgorithm);
    }
    if !canonical_identifier(signer.key_id()) {
        return Err(TelemetryViolation::InvalidKeyId);
    }
    let bytes = canonical_machine_telemetry_bytes(&payload)?;
    let signature = signer
        .sign_telemetry(&bytes)
        .map_err(TelemetryViolation::Signing)?;
    if signature.is_empty() || signature.len() > MAX_TELEMETRY_SIGNATURE_BYTES {
        return Err(TelemetryViolation::SignatureTooLarge {
            actual: signature.len(),
            maximum: MAX_TELEMETRY_SIGNATURE_BYTES,
        });
    }
    Ok(SignedMachineTelemetry {
        payload,
        algorithm,
        key_id: signer.key_id().to_string(),
        signature,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_machine_telemetry(
    signed: SignedMachineTelemetry,
    policy: &MachineTelemetryPolicy,
    expectation: TelemetryExpectation<'_>,
    trust_snapshot: &TrustSnapshot,
    evaluation_time_unix_ms: u64,
    verifier: &dyn MachineTelemetryVerifier,
) -> Result<VerifiedMachineTelemetry, TelemetryViolation> {
    policy.validate()?;
    signed.payload.validate()?;
    if !signed.algorithm.is_canonical() {
        return Err(TelemetryViolation::InvalidAlgorithm);
    }
    if !canonical_identifier(&signed.key_id) {
        return Err(TelemetryViolation::InvalidKeyId);
    }
    if signed.signature.is_empty() || signed.signature.len() > policy.maximum_signature_bytes {
        return Err(TelemetryViolation::SignatureTooLarge {
            actual: signed.signature.len(),
            maximum: policy.maximum_signature_bytes,
        });
    }
    if signed.payload.manifest_digest != expectation.manifest_digest {
        return Err(TelemetryViolation::ManifestMismatch);
    }
    if signed.payload.machine_id != expectation.machine_id {
        return Err(TelemetryViolation::MachineMismatch);
    }
    if signed.payload.session_digest != expectation.session_digest {
        return Err(TelemetryViolation::SessionDigestMismatch);
    }
    if signed.payload.session_sequence != expectation.session_sequence {
        return Err(TelemetryViolation::SessionSequenceMismatch);
    }
    if signed.payload.printer_job_id != expectation.printer_job_id {
        return Err(TelemetryViolation::PrinterJobMismatch);
    }
    if signed.payload.observed_at_unix_ms
        > evaluation_time_unix_ms.saturating_add(policy.maximum_future_skew_ms)
    {
        return Err(TelemetryViolation::ObservationFromFuture);
    }
    if evaluation_time_unix_ms.saturating_sub(signed.payload.observed_at_unix_ms)
        > policy.maximum_age_ms
    {
        return Err(TelemetryViolation::ObservationStale);
    }
    trust_snapshot
        .validate()
        .map_err(|error| TelemetryViolation::TrustSnapshotInvalid(format!("{error:?}")))?;
    let evaluation_time_unix_s = evaluation_time_unix_ms / 1_000;
    if !trust_snapshot.is_fresh_at(evaluation_time_unix_s) {
        return Err(TelemetryViolation::TrustSnapshotStale);
    }
    let eligibility = trust_snapshot.key_eligibility(
        &signed.algorithm,
        &signed.key_id,
        KeyUsage::MachineTelemetry,
        evaluation_time_unix_s,
    );
    if eligibility != KeyEligibility::Eligible {
        return Err(TelemetryViolation::KeyIneligible(eligibility));
    }
    let bytes = canonical_machine_telemetry_bytes(&signed.payload)?;
    let valid = verifier
        .verify_telemetry(&signed.algorithm, &signed.key_id, &bytes, &signed.signature)
        .map_err(TelemetryViolation::Verification)?;
    if !valid {
        return Err(TelemetryViolation::SignatureInvalid);
    }
    let telemetry_digest = digest_machine_telemetry(&signed.payload)?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| TelemetryViolation::TrustSnapshotInvalid(format!("{error:?}")))?;
    Ok(VerifiedMachineTelemetry {
        signed,
        telemetry_digest,
        trust_snapshot_digest,
        verified_at_unix_ms: evaluation_time_unix_ms,
    })
}

fn canonical_identifier(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TELEMETRY_ID_BYTES
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};
    use std::collections::BTreeSet;

    struct TestProvider;

    impl MachineTelemetrySigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test-telemetry".into())
        }

        fn key_id(&self) -> &str {
            "machine-telemetry-key"
        }

        fn sign_telemetry(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl MachineTelemetryVerifier for TestProvider {
        fn verify_telemetry(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(
                algorithm == &SignatureAlgorithm::Other("test-telemetry".into())
                    && key_id == "machine-telemetry-key"
                    && signature == sha256(message).0.as_slice(),
            )
        }
    }

    fn payload() -> MachineTelemetryPayload {
        MachineTelemetryPayload {
            schema_version: MACHINE_TELEMETRY_SCHEMA.into(),
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1".into(),
            session_digest: sha256(b"session"),
            session_sequence: 7,
            printer_job_id: "job-1".into(),
            frame_sequence: 1,
            observed_at_unix_ms: 500_000,
            elapsed_ms: 1_250,
            heartbeat_sequence: 9,
            progress_ppm: 250_000,
            nozzle_actual_milli_c: 200_000,
            nozzle_target_milli_c: 205_000,
            bed_actual_milli_c: 60_000,
            bed_target_milli_c: 60_000,
        }
    }

    fn trust() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            400,
            800,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test-telemetry".into()),
                key_id: "machine-telemetry-key".into(),
                not_before_unix_s: 400,
                not_after_unix_s: Some(700),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::MachineTelemetry]),
            }],
        )
        .unwrap()
    }

    fn expectation() -> TelemetryExpectation<'static> {
        TelemetryExpectation {
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1",
            session_digest: sha256(b"session"),
            session_sequence: 7,
            printer_job_id: "job-1",
        }
    }

    #[test]
    fn verified_payload_converts_to_finite_guard_telemetry() {
        let provider = TestProvider;
        let signed = sign_machine_telemetry(payload(), &provider).unwrap();
        let verified = verify_machine_telemetry(
            signed,
            &MachineTelemetryPolicy::default(),
            expectation(),
            &trust(),
            500_100,
            &provider,
        )
        .unwrap();
        let telemetry = verified.execution_telemetry();
        assert_eq!(telemetry.progress, 0.25);
        assert_eq!(telemetry.elapsed_s, 1.25);
        assert_eq!(telemetry.nozzle_actual_c, 200.0);
    }

    #[test]
    fn manifest_session_and_job_identity_are_authority_boundaries() {
        let provider = TestProvider;
        let signed = sign_machine_telemetry(payload(), &provider).unwrap();
        let mut wrong = expectation();
        wrong.printer_job_id = "job-2";
        assert!(matches!(
            verify_machine_telemetry(
                signed,
                &MachineTelemetryPolicy::default(),
                wrong,
                &trust(),
                500_100,
                &provider,
            ),
            Err(TelemetryViolation::PrinterJobMismatch)
        ));
    }

    #[test]
    fn stale_and_future_frames_fail_closed() {
        let provider = TestProvider;
        let signed = sign_machine_telemetry(payload(), &provider).unwrap();
        assert!(matches!(
            verify_machine_telemetry(
                signed.clone(),
                &MachineTelemetryPolicy::default(),
                expectation(),
                &trust(),
                520_001,
                &provider,
            ),
            Err(TelemetryViolation::ObservationStale)
        ));
        assert!(matches!(
            verify_machine_telemetry(
                signed,
                &MachineTelemetryPolicy::default(),
                expectation(),
                &trust(),
                498_000,
                &provider,
            ),
            Err(TelemetryViolation::ObservationFromFuture)
        ));
    }
}
