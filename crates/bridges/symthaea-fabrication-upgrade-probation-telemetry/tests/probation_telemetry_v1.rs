// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::FabricationContainmentState;
use symthaea_fabrication_kernel::crypto_digest::{Sha256Digest, sha256};
use symthaea_fabrication_kernel::telemetry::{
    MACHINE_TELEMETRY_SCHEMA, MachineTelemetryPayload, MachineTelemetryPolicy,
    MachineTelemetrySigner, MachineTelemetryVerifier, TelemetryExpectation,
    sign_machine_telemetry, verify_machine_telemetry,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
};
use symthaea_fabrication_upgrade_probation_telemetry::{
    ExactProbationTelemetryVerificationPolicyV1, ExactProbationTelemetryVerifierV1,
    ProbationTelemetryBindingError, ProbationTelemetryFrameInputV1,
    qualify_exact_probation_telemetry_bundle_v1,
};

struct KernelProvider;

impl MachineTelemetrySigner for KernelProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Other("probation-telemetry-test".into())
    }
    fn key_id(&self) -> &str {
        "machine-telemetry-key"
    }
    fn sign_telemetry(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}

impl MachineTelemetryVerifier for KernelProvider {
    fn verify_telemetry(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(
            algorithm == &SignatureAlgorithm::Other("probation-telemetry-test".into())
                && key_id == "machine-telemetry-key"
                && signature == sha256(message).0.as_slice(),
        )
    }
}

struct ExactProvider {
    id: &'static str,
    policy_byte: u8,
}

impl ExactProbationTelemetryVerifierV1 for ExactProvider {
    fn provider_id(&self) -> &str {
        self.id
    }
    fn verification_policy_digest(&self) -> Sha256Digest {
        Sha256Digest([self.policy_byte; 32])
    }
    fn verify_telemetry_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(
            algorithm == &SignatureAlgorithm::Other("probation-telemetry-test".into())
                && key_id == "machine-telemetry-key"
                && signature == sha256(message).0.as_slice(),
        )
    }
}

fn trust() -> TrustSnapshot {
    TrustSnapshot::new(
        7,
        400,
        800,
        vec![KeyTrustRecord {
            algorithm: SignatureAlgorithm::Other("probation-telemetry-test".into()),
            key_id: "machine-telemetry-key".into(),
            not_before_unix_s: 400,
            not_after_unix_s: Some(700),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::MachineTelemetry]),
        }],
    )
    .unwrap()
}

fn signed_and_verified() -> (
    symthaea_fabrication_kernel::telemetry::SignedMachineTelemetry,
    symthaea_fabrication_kernel::telemetry::VerifiedMachineTelemetry,
    TrustSnapshot,
) {
    let payload = MachineTelemetryPayload {
        schema_version: MACHINE_TELEMETRY_SCHEMA.into(),
        manifest_digest: sha256(b"manifest"),
        machine_id: "machine-1".into(),
        session_digest: sha256(b"session-job-1"),
        session_sequence: 1,
        printer_job_id: "job-1".into(),
        frame_sequence: 1,
        observed_at_unix_ms: 500_000,
        elapsed_ms: 1_000,
        heartbeat_sequence: 1,
        progress_ppm: 100_000,
        nozzle_actual_milli_c: 200_000,
        nozzle_target_milli_c: 200_000,
        bed_actual_milli_c: 60_000,
        bed_target_milli_c: 60_000,
    };
    let kernel = KernelProvider;
    let signed = sign_machine_telemetry(payload, &kernel).unwrap();
    let trust = trust();
    let verified = verify_machine_telemetry(
        signed.clone(),
        &MachineTelemetryPolicy::default(),
        TelemetryExpectation {
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1",
            session_digest: sha256(b"session-job-1"),
            session_sequence: 1,
            printer_job_id: "job-1",
        },
        &trust,
        500_001,
        &kernel,
    )
    .unwrap();
    (signed, verified, trust)
}

#[test]
fn exact_raw_verified_telemetry_bundle_is_qualified() {
    let (signed, verified, trust) = signed_and_verified();
    let containment = FabricationContainmentState::genesis(1, sha256(b"resilience")).unwrap();
    let provider_a = ExactProvider { id: "exact-a", policy_byte: 1 };
    let provider_b = ExactProvider { id: "exact-b", policy_byte: 2 };
    let frame = ProbationTelemetryFrameInputV1 {
        signed: &signed,
        verified: &verified,
        trust_snapshot: &trust,
    };
    let bundle = qualify_exact_probation_telemetry_bundle_v1(
        &[frame],
        &containment,
        &ExactProbationTelemetryVerificationPolicyV1::default(),
        &[&provider_a, &provider_b],
    )
    .unwrap();
    assert_eq!(bundle.machine_id(), "machine-1");
    assert_eq!(bundle.frame_count(), 1);
    assert_eq!(bundle.distinct_job_count(), 1);
    assert_eq!(bundle.observed_started_at_unix_ms(), 500_000);
    assert_eq!(bundle.observed_ended_at_unix_ms(), 500_000);
}

#[test]
fn substituted_raw_signature_is_rejected_even_when_opaque_verified_frame_is_retained() {
    let (signed, verified, trust) = signed_and_verified();
    let mut substituted = signed.clone();
    substituted.signature = vec![0xA5; 32];
    let containment = FabricationContainmentState::genesis(1, sha256(b"resilience")).unwrap();
    let provider_a = ExactProvider { id: "exact-a", policy_byte: 1 };
    let provider_b = ExactProvider { id: "exact-b", policy_byte: 2 };
    let frame = ProbationTelemetryFrameInputV1 {
        signed: &substituted,
        verified: &verified,
        trust_snapshot: &trust,
    };
    let errors = qualify_exact_probation_telemetry_bundle_v1(
        &[frame],
        &containment,
        &ExactProbationTelemetryVerificationPolicyV1::default(),
        &[&provider_a, &provider_b],
    )
    .unwrap_err();
    assert!(errors.iter().any(|error| matches!(
        error,
        ProbationTelemetryBindingError::TelemetrySignatureRejected { .. }
    )));
}

#[test]
fn source_ratchets_keep_telemetry_binding_non_scalar_and_non_deserializable() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_ms:"));
    assert!(source.contains("KeyUsage::MachineTelemetry"));
    assert!(source.contains("MachineTelemetryTracker::default()"));
    assert!(source.contains("UPSTREAM_SIGNED_OBSERVATION_SET_DOMAIN"));
    assert!(source.contains("AttemptedJobCoverageMismatch"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct ExactProbationTelemetryBundleV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct TelemetryBoundUpgradeProbationClearanceV1"));
}
