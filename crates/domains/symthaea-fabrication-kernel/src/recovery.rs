// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed re-authorization after an interrupted machine session.
//!
//! This module does not pretend to resume arbitrary G-code mid-stream. It only
//! authorizes an explicit restart from the beginning after binding interruption
//! evidence, guard state, attestation, program, machine identity, and a fresh
//! session nonce.

use crate::attestation::VerifiedAttestation;
use crate::crypto_digest::Sha256Digest;
use crate::execution_guard::{
    ContainmentAction, ExecutionCheckpointError, ExecutionGuardCheckpoint,
    digest_execution_checkpoint,
};
use crate::machine::{NegotiatedMachine, ValidatedGCode};
use crate::submission::{
    AuthorizedPrintJob, PrintAuthorizationError, SubmittedJobReceipt, authorize_print_job,
};

#[derive(Debug, Clone, PartialEq)]
pub struct InterruptedPrintEvidence {
    pub receipt: SubmittedJobReceipt,
    pub checkpoint: ExecutionGuardCheckpoint,
    pub checkpoint_digest: Sha256Digest,
    pub interrupted_at_unix_s: u64,
}

impl InterruptedPrintEvidence {
    pub fn new(
        receipt: SubmittedJobReceipt,
        checkpoint: ExecutionGuardCheckpoint,
        interrupted_at_unix_s: u64,
    ) -> Result<Self, ExecutionCheckpointError> {
        let checkpoint_digest = digest_execution_checkpoint(&checkpoint)?;
        Ok(Self {
            receipt,
            checkpoint,
            checkpoint_digest,
            interrupted_at_unix_s,
        })
    }

    pub fn validate_checkpoint(&self) -> Result<(), RecoveryAuthorizationError> {
        let actual = digest_execution_checkpoint(&self.checkpoint)
            .map_err(RecoveryAuthorizationError::InvalidCheckpoint)?;
        if actual != self.checkpoint_digest {
            return Err(RecoveryAuthorizationError::CheckpointDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryPolicy {
    pub maximum_interruption_age_s: u64,
    /// Permit a clean disconnect checkpoint that had not yet latched Pause.
    pub allow_continue_checkpoint: bool,
    /// Always require a new session nonce to prevent replay into the dead session.
    pub require_fresh_session: bool,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            maximum_interruption_age_s: 15 * 60,
            allow_continue_checkpoint: false,
            require_fresh_session: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAuthorizationError {
    InvalidPolicy,
    InvalidCheckpoint(ExecutionCheckpointError),
    CheckpointDigestMismatch,
    ManifestDigestMismatch,
    MachineIdentityChanged { previous: String, current: String },
    SessionNonceReused,
    InterruptionFromFuture,
    InterruptionExpired { age_s: u64, maximum_s: u64 },
    MissingProgress,
    NoWorkCompleted,
    JobAlreadyComplete,
    TerminalContainmentAction(ContainmentAction),
    RestartAuthorization(PrintAuthorizationError),
}

/// Re-authorize a complete restart in a fresh session.
///
/// The returned [`AuthorizedPrintJob`] remains single-use. No mid-program resume
/// is inferred from progress because doing so would require printer-specific
/// modal-state reconstruction and physical position evidence.
pub fn reauthorize_print_restart(
    validated: ValidatedGCode,
    attestation: VerifiedAttestation,
    machine: NegotiatedMachine,
    evidence: &InterruptedPrintEvidence,
    now_unix_s: u64,
    policy: RecoveryPolicy,
) -> Result<AuthorizedPrintJob, RecoveryAuthorizationError> {
    if policy.maximum_interruption_age_s == 0 {
        return Err(RecoveryAuthorizationError::InvalidPolicy);
    }
    evidence.validate_checkpoint()?;
    if evidence.receipt.manifest_digest != attestation.manifest_digest() {
        return Err(RecoveryAuthorizationError::ManifestDigestMismatch);
    }
    if evidence.receipt.machine_id != machine.machine_id() {
        return Err(RecoveryAuthorizationError::MachineIdentityChanged {
            previous: evidence.receipt.machine_id.clone(),
            current: machine.machine_id().to_string(),
        });
    }
    if policy.require_fresh_session && evidence.receipt.session_nonce == machine.session_nonce() {
        return Err(RecoveryAuthorizationError::SessionNonceReused);
    }
    let age_s = now_unix_s
        .checked_sub(evidence.interrupted_at_unix_s)
        .ok_or(RecoveryAuthorizationError::InterruptionFromFuture)?;
    if age_s > policy.maximum_interruption_age_s {
        return Err(RecoveryAuthorizationError::InterruptionExpired {
            age_s,
            maximum_s: policy.maximum_interruption_age_s,
        });
    }
    let progress = evidence
        .checkpoint
        .progress()
        .ok_or(RecoveryAuthorizationError::MissingProgress)?;
    if progress <= 0.0 {
        return Err(RecoveryAuthorizationError::NoWorkCompleted);
    }
    if progress >= 1.0 {
        return Err(RecoveryAuthorizationError::JobAlreadyComplete);
    }
    match evidence.checkpoint.latched_action {
        ContainmentAction::Pause => {}
        ContainmentAction::Continue if policy.allow_continue_checkpoint => {}
        action => {
            return Err(RecoveryAuthorizationError::TerminalContainmentAction(
                action,
            ));
        }
    }
    authorize_print_job(validated, attestation, machine)
        .map_err(RecoveryAuthorizationError::RestartAuthorization)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{
        AttestationPolicy, ManifestSignatureVerifier, ManifestSigner, SignatureAlgorithm,
        attest_fabrication_manifest, verify_attestation_authority,
    };
    use crate::crypto_digest::sha256;
    use crate::execution_guard::{ExecutionGuard, ExecutionGuardPolicy, ExecutionTelemetry};
    use crate::machine::{
        MachineCapabilities, MachineProfile, MachineSession, negotiate_machine_profile,
    };
    use crate::provenance::{
        FabricationManifest, StableFingerprint, fingerprint_gcode_program,
        fingerprint_machine_profile,
    };
    use crate::toolpath::{GCodeCommand, GCodeProgram};

    struct TestProvider;

    impl ManifestSigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test".into())
        }
        fn key_id(&self) -> &str {
            "recovery-test-key"
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
                && key_id == "recovery-test-key"
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
                    f: Some(1_000.0),
                },
            ],
            total_extrusion_mm: 1.0,
        }
    }

    fn inputs(session_nonce: &str) -> (ValidatedGCode, VerifiedAttestation, NegotiatedMachine) {
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
        let machine = negotiate_machine_profile(
            &profile,
            MachineSession {
                session_nonce: session_nonce.into(),
                capabilities: MachineCapabilities::from_profile("machine-1", &profile),
            },
        )
        .unwrap();
        (validated, verified, machine)
    }

    fn paused_checkpoint() -> ExecutionGuardCheckpoint {
        let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
        guard.observe(ExecutionTelemetry {
            elapsed_s: 0.0,
            heartbeat_sequence: 1,
            progress: 0.2,
            nozzle_actual_c: 200.0,
            nozzle_target_c: 200.0,
            bed_actual_c: 60.0,
            bed_target_c: 60.0,
        });
        let mut checkpoint = guard.checkpoint();
        checkpoint.latched_action = ContainmentAction::Pause;
        checkpoint
    }

    #[test]
    fn paused_job_can_be_reauthorized_only_in_a_fresh_session() {
        let (validated, verified, new_machine) = inputs("session-2");
        let evidence = InterruptedPrintEvidence::new(
            SubmittedJobReceipt {
                printer_job_id: "job-1".into(),
                manifest_digest: verified.manifest_digest(),
                machine_id: "machine-1".into(),
                session_nonce: "session-1".into(),
            },
            paused_checkpoint(),
            100,
        )
        .unwrap();
        let restarted = reauthorize_print_restart(
            validated,
            verified,
            new_machine,
            &evidence,
            120,
            RecoveryPolicy::default(),
        )
        .unwrap();
        assert_eq!(restarted.session_nonce(), "session-2");
    }

    #[test]
    fn cancelled_job_cannot_be_restarted_through_recovery_lane() {
        let (validated, verified, new_machine) = inputs("session-2");
        let mut checkpoint = paused_checkpoint();
        checkpoint.latched_action = ContainmentAction::Cancel;
        let evidence = InterruptedPrintEvidence::new(
            SubmittedJobReceipt {
                printer_job_id: "job-1".into(),
                manifest_digest: verified.manifest_digest(),
                machine_id: "machine-1".into(),
                session_nonce: "session-1".into(),
            },
            checkpoint,
            100,
        )
        .unwrap();
        assert!(matches!(
            reauthorize_print_restart(
                validated,
                verified,
                new_machine,
                &evidence,
                120,
                RecoveryPolicy::default(),
            ),
            Err(RecoveryAuthorizationError::TerminalContainmentAction(
                ContainmentAction::Cancel
            ))
        ));
    }
}
