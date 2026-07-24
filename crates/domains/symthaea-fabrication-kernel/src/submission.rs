// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Single-use print authorization bound to attestation and machine session identity.

use crate::attestation::VerifiedAttestation;
use crate::crypto_digest::Sha256Digest;
use crate::machine::{NegotiatedMachine, ValidatedGCode};
use crate::printer_control::{PrinterApi, PrinterError};
use crate::provenance::{fingerprint_gcode_program, fingerprint_machine_profile};
use crate::release::ReleaseAuthority;
use crate::session::MachineSessionLease;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrintAuthorizationError {
    ValidatedProfileMismatch,
    ManifestMachineProfileMismatch,
    ManifestGCodeMismatch,
}

/// Single-use authority to submit one exact attested program in one machine session.
///
/// The type is intentionally not `Clone`. Submission consumes it, making replay
/// an explicit re-authorization decision rather than an accidental API call.
#[derive(Debug)]
pub struct AuthorizedPrintJob {
    validated: ValidatedGCode,
    attestation: VerifiedAttestation,
    machine: NegotiatedMachine,
}

impl AuthorizedPrintJob {
    pub fn manifest_digest(&self) -> Sha256Digest {
        self.attestation.manifest_digest()
    }

    pub fn machine_id(&self) -> &str {
        self.machine.machine_id()
    }

    pub fn session_nonce(&self) -> &str {
        self.machine.session_nonce()
    }

    pub fn validated(&self) -> &ValidatedGCode {
        &self.validated
    }
}

pub fn authorize_print_job(
    validated: ValidatedGCode,
    attestation: VerifiedAttestation,
    machine: NegotiatedMachine,
) -> Result<AuthorizedPrintJob, PrintAuthorizationError> {
    if validated.profile() != machine.profile() {
        return Err(PrintAuthorizationError::ValidatedProfileMismatch);
    }
    if attestation.manifest().machine_profile != fingerprint_machine_profile(machine.profile()) {
        return Err(PrintAuthorizationError::ManifestMachineProfileMismatch);
    }
    if attestation.manifest().gcode_program != fingerprint_gcode_program(validated.program()) {
        return Err(PrintAuthorizationError::ManifestGCodeMismatch);
    }
    Ok(AuthorizedPrintJob {
        validated,
        attestation,
        machine,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GovernedPrintAuthorizationError {
    Base(PrintAuthorizationError),
    ManifestReleaseMismatch,
    TrustSnapshotMismatch,
    FutureReleaseEvaluation,
    SessionMachineMismatch,
    SessionNonceMismatch,
    SessionDigestMismatch,
    SessionExpired {
        now_unix_s: u64,
        expires_at_unix_s: u64,
    },
}

/// Single-use print authority bound to release quorum and a persisted timed-session lease.
#[derive(Debug)]
pub struct GovernedAuthorizedPrintJob {
    inner: AuthorizedPrintJob,
    release_policy_digest: Sha256Digest,
    delegation_digest: Option<Sha256Digest>,
    trust_snapshot_digest: Sha256Digest,
    session_digest: Sha256Digest,
    session_sequence: u64,
    session_expires_at_unix_s: u64,
}

impl GovernedAuthorizedPrintJob {
    pub fn manifest_digest(&self) -> Sha256Digest {
        self.inner.manifest_digest()
    }
    pub fn machine_id(&self) -> &str {
        self.inner.machine_id()
    }
    pub fn session_nonce(&self) -> &str {
        self.inner.session_nonce()
    }
    pub fn release_policy_digest(&self) -> Sha256Digest {
        self.release_policy_digest
    }
    pub fn delegation_digest(&self) -> Option<Sha256Digest> {
        self.delegation_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn session_digest(&self) -> Sha256Digest {
        self.session_digest
    }
    pub fn session_sequence(&self) -> u64 {
        self.session_sequence
    }
    pub fn session_expires_at_unix_s(&self) -> u64 {
        self.session_expires_at_unix_s
    }
}

pub fn authorize_governed_print_job(
    validated: ValidatedGCode,
    attestation: VerifiedAttestation,
    release: &dyn ReleaseAuthority,
    machine: NegotiatedMachine,
    lease: MachineSessionLease,
    now_unix_s: u64,
) -> Result<GovernedAuthorizedPrintJob, GovernedPrintAuthorizationError> {
    if release.manifest_digest() != attestation.manifest_digest() {
        return Err(GovernedPrintAuthorizationError::ManifestReleaseMismatch);
    }
    if attestation.trust_snapshot_digest() != Some(release.trust_snapshot_digest()) {
        return Err(GovernedPrintAuthorizationError::TrustSnapshotMismatch);
    }
    if release.evaluation_time_unix_s() > now_unix_s {
        return Err(GovernedPrintAuthorizationError::FutureReleaseEvaluation);
    }
    if lease.machine_id() != machine.machine_id() {
        return Err(GovernedPrintAuthorizationError::SessionMachineMismatch);
    }
    if lease.session_nonce() != machine.session_nonce() {
        return Err(GovernedPrintAuthorizationError::SessionNonceMismatch);
    }
    if machine
        .session_window()
        .is_none_or(|window| window.digest != lease.session_digest())
    {
        return Err(GovernedPrintAuthorizationError::SessionDigestMismatch);
    }
    if now_unix_s >= lease.expires_at_unix_s() {
        return Err(GovernedPrintAuthorizationError::SessionExpired {
            now_unix_s,
            expires_at_unix_s: lease.expires_at_unix_s(),
        });
    }
    let inner = authorize_print_job(validated, attestation, machine)
        .map_err(GovernedPrintAuthorizationError::Base)?;
    Ok(GovernedAuthorizedPrintJob {
        inner,
        release_policy_digest: release.policy_digest(),
        delegation_digest: release.delegation_digest(),
        trust_snapshot_digest: release.trust_snapshot_digest(),
        session_digest: lease.session_digest(),
        session_sequence: lease.session_sequence(),
        session_expires_at_unix_s: lease.expires_at_unix_s(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GovernedSubmittedJobReceipt {
    pub submission: SubmittedJobReceipt,
    pub release_policy_digest: Sha256Digest,
    pub delegation_digest: Option<Sha256Digest>,
    pub trust_snapshot_digest: Sha256Digest,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
}

pub fn submit_governed_authorized_job(
    printer: &mut dyn PrinterApi,
    job: GovernedAuthorizedPrintJob,
    active_machine_id: &str,
    active_session_nonce: &str,
    now_unix_s: u64,
) -> Result<GovernedSubmittedJobReceipt, SubmissionError> {
    if now_unix_s >= job.session_expires_at_unix_s {
        return Err(SubmissionError::SessionExpired {
            authorized: format!(
                "{} (expired at {})",
                job.session_nonce(),
                job.session_expires_at_unix_s
            ),
            active: format!("{} (time {})", active_session_nonce, now_unix_s),
        });
    }
    let release_policy_digest = job.release_policy_digest;
    let delegation_digest = job.delegation_digest;
    let trust_snapshot_digest = job.trust_snapshot_digest;
    let session_digest = job.session_digest;
    let session_sequence = job.session_sequence;
    let submission =
        submit_authorized_job(printer, job.inner, active_machine_id, active_session_nonce)?;
    Ok(GovernedSubmittedJobReceipt {
        submission,
        release_policy_digest,
        delegation_digest,
        trust_snapshot_digest,
        session_digest,
        session_sequence,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionError {
    MachineIdentityChanged { authorized: String, active: String },
    SessionExpired { authorized: String, active: String },
    Printer(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmittedJobReceipt {
    pub printer_job_id: String,
    pub manifest_digest: Sha256Digest,
    pub machine_id: String,
    pub session_nonce: String,
}

/// Consume one print authority after checking the active machine session.
pub fn submit_authorized_job(
    printer: &mut dyn PrinterApi,
    job: AuthorizedPrintJob,
    active_machine_id: &str,
    active_session_nonce: &str,
) -> Result<SubmittedJobReceipt, SubmissionError> {
    if job.machine_id() != active_machine_id {
        return Err(SubmissionError::MachineIdentityChanged {
            authorized: job.machine_id().to_string(),
            active: active_machine_id.to_string(),
        });
    }
    if job.session_nonce() != active_session_nonce {
        return Err(SubmissionError::SessionExpired {
            authorized: job.session_nonce().to_string(),
            active: active_session_nonce.to_string(),
        });
    }
    let manifest_digest = job.manifest_digest();
    let machine_id = job.machine_id().to_string();
    let session_nonce = job.session_nonce().to_string();
    let printer_job_id = printer
        .submit_gcode(&job.validated.program().to_gcode_string())
        .map_err(|error| SubmissionError::Printer(error.to_string()))?;
    Ok(SubmittedJobReceipt {
        printer_job_id,
        manifest_digest,
        machine_id,
        session_nonce,
    })
}

impl From<PrinterError> for SubmissionError {
    fn from(error: PrinterError) -> Self {
        Self::Printer(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{
        AttestationPolicy, ManifestSignatureVerifier, ManifestSigner, SignatureAlgorithm,
        attest_fabrication_manifest, verify_attestation_authority,
    };
    use crate::crypto_digest::sha256;
    use crate::machine::{
        MachineCapabilities, MachineProfile, MachineSession, negotiate_machine_profile,
    };
    use crate::printer_control::{MockPrinter, PrinterApi};
    use crate::provenance::{FabricationManifest, StableFingerprint};
    use crate::toolpath::{GCodeCommand, GCodeProgram};

    struct TestProvider;

    impl ManifestSigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test".into())
        }
        fn key_id(&self) -> &str {
            "test-key"
        }
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl ManifestSignatureVerifier for TestProvider {
        fn verify(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(algorithm == &SignatureAlgorithm::Other("test".into())
                && key_id == "test-key"
                && signature == sha256(message).0.as_slice())
        }
    }

    fn program() -> GCodeProgram {
        GCodeProgram {
            commands: vec![
                GCodeCommand::G28,
                GCodeCommand::G1 {
                    x: Some(1.0),
                    y: Some(1.0),
                    z: Some(0.2),
                    e: Some(1.0),
                    f: Some(1000.0),
                },
            ],
            total_extrusion_mm: 1.0,
        }
    }

    fn authorized_job(session_nonce: &str) -> AuthorizedPrintJob {
        let provider = TestProvider;
        let profile = MachineProfile::default();
        let validated = ValidatedGCode::try_new(program(), &profile).unwrap();
        let fallback = StableFingerprint([1, 2, 3, 4]);
        let manifest = FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fallback,
            process_policy: fallback,
            process_evidence: fallback,
            minimum_feature_policy: fallback,
            minimum_feature_evidence: fallback,
            slice_config: fallback,
            slice_layers: fallback,
            toolpath_config: fallback,
            machine_profile: fingerprint_machine_profile(&profile),
            gcode_program: fingerprint_gcode_program(validated.program()),
            pipeline: fallback,
            layer_count: 1,
            command_count: validated.program().commands.len(),
            total_extrusion_mm: validated.program().total_extrusion_mm,
        };
        let attested = attest_fabrication_manifest(manifest, &[&provider]).unwrap();
        let verified =
            verify_attestation_authority(attested, &AttestationPolicy::default(), &provider)
                .unwrap();
        let negotiated = negotiate_machine_profile(
            &profile,
            MachineSession {
                session_nonce: session_nonce.into(),
                capabilities: MachineCapabilities::from_profile("mock-printer", &profile),
            },
        )
        .unwrap();
        authorize_print_job(validated, verified, negotiated).unwrap()
    }

    #[test]
    fn exact_session_can_submit_once() {
        let mut printer = MockPrinter::new();
        printer.connect().unwrap();
        let receipt = submit_authorized_job(
            &mut printer,
            authorized_job("nonce-1"),
            "mock-printer",
            "nonce-1",
        )
        .unwrap();
        assert!(receipt.printer_job_id.starts_with("mock-job-"));
        assert_eq!(receipt.machine_id, "mock-printer");
    }

    #[test]
    fn stale_session_is_rejected_before_submission() {
        let mut printer = MockPrinter::new();
        printer.connect().unwrap();
        let result = submit_authorized_job(
            &mut printer,
            authorized_job("nonce-1"),
            "mock-printer",
            "nonce-2",
        );
        assert!(matches!(
            result,
            Err(SubmissionError::SessionExpired { .. })
        ));
    }
}
